import os, json, requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

# -------- Models --------
class SearchBody(BaseModel):
    query: str
    top_k: Optional[int] = 6
    namespace: Optional[str] = None

# -------- Helpers --------
def require_secret(x_action_secret: Optional[str]):
    expected = os.getenv("ACTION_SHARED_SECRET", "")
    if not expected or x_action_secret != expected:
        raise HTTPException(status_code=401, detail="bad secret")

def openai_headers_json() -> dict:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not (key.startswith("sk-") and len(key) > 20):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return {
        "Authorization": f"Bearer {key}",
        "OpenAI-Beta": "assistants=v2",
        "Content-Type": "application/json",
    }

def openai_headers_noctype() -> dict:
    # same as above but without Content-Type so we can send multipart
    h = openai_headers_json().copy()
    h.pop("Content-Type", None)
    return h

def get_vs() -> str:
    vs = os.getenv("VECTOR_STORE_ID", "").strip()
    if not vs:
        raise HTTPException(status_code=500, detail="VECTOR_STORE_ID not configured")
    return vs

# -------- Health / Info / Diag / Auth --------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/info")
def info():
    return {"vector_store_id": os.getenv("VECTOR_STORE_ID", ""), "vector_store_name": "bullz-vector-store"}

@app.get("/diag")
def diag():
    return {
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "vector_store_id": os.getenv("VECTOR_STORE_ID", ""),
        "beta_header_expected": "assistants=v2",
    }

@app.post("/authcheck")
def authcheck(x_action_secret: str = Header(None, alias="X-Action-Secret")):
    require_secret(x_action_secret)
    return {"ok": True}

# -------- Upload (two-step: /files multipart, then attach via JSON) --------
@app.post("/upload")
def upload(
    x_action_secret: str = Header(None, alias="X-Action-Secret"),
    file: UploadFile = File(...),
    namespace: Optional[str] = Form(None),
):
    require_secret(x_action_secret)
    vs = get_vs()

    # 1) Upload binary to /v1/files (multipart/form-data)
    try:
        r1 = requests.post(
            "https://api.openai.com/v1/files",
            headers=openai_headers_noctype(),
            files={"file": (file.filename, file.file)},
            data={"purpose": "assistants"},
            timeout=120
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")
    if r1.status_code != 200:
        try:
            b1 = r1.json()
        except Exception:
            b1 = {"text": r1.text}
        return {"stage": "upload", "upstream_status": r1.status_code, "error": b1}, r1.status_code
    file_json = r1.json()
    file_id = file_json.get("id")
    if not file_id:
        raise HTTPException(status_code=502, detail="upstream error: missing file id")

    # 2) Attach file_id to vector store with JSON
    try:
        r2 = requests.post(
            f"https://api.openai.com/v1/vector_stores/{vs}/files",
            headers=openai_headers_json(),
            json={"file_id": file_id},
            timeout=60
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")
    if r2.status_code != 200:
        try:
            b2 = r2.json()
        except Exception:
            b2 = {"text": r2.text}
        return {"stage": "attach", "upstream_status": r2.status_code, "error": b2}, r2.status_code

    return {"ok": True, "file_id": file_id, "vector_store_id": vs, "namespace": namespace, "attach": r2.json()}

# -------- Search (Responses API; robust fallback) --------
@app.post("/search")
def search(
    body: SearchBody,
    x_action_secret: str = Header(None, alias="X-Action-Secret"),
):
    require_secret(x_action_secret)
    vs = get_vs()
    headers = openai_headers_json()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # Primary payload (v2, tool_resources at top-level)
    payload = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": body.query}]}
        ],
        "tools": [{"type": "file_search"}],
        "tool_choice": "auto",
        "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
        "max_output_tokens": 800,
        "temperature": 0.2,
        "tool_resources": {"file_search": {"vector_store_ids": [vs]}},
    }

    try:
        r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")

    # If 'tool_resources' rejected or unsupported, fallback by embedding vector_store_ids into tools[0].file_search
    if r.status_code == 400:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        msg = json.dumps(err)
        if ("Unknown parameter: 'tool_resources'" in msg) or ("tools[0].vector_store_ids" in msg):
            fallback = {
                "model": model,
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": body.query}]}
                ],
                "tools": [{"type": "file_search", "file_search": {"vector_store_ids": [vs]}}],
                "tool_choice": "auto",
                "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
                "max_output_tokens": 800,
                "temperature": 0.2
            }
            try:
                r2 = requests.post("https://api.openai.com/v1/responses", headers=headers, json=fallback, timeout=120)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")
            if r2.status_code != 200:
                try:
                    b2 = r2.json()
                except Exception:
                    b2 = {"text": r2.text}
                return [{"upstream_status": r.status_code, "error": err},
                        {"upstream_status": r2.status_code, "error": b2}], 502
            return {"ok": True, "response": r2.json()}

    if r.status_code != 200:
        try:
            b = r.json()
        except Exception:
            b = {"text": r.text}
        return {"upstream_status": r.status_code, "error": b}, 502

    return {"ok": True, "response": r.json()}

# redeploy-marker: FINAL-FIX
