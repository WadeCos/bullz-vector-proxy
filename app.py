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

    user_msg = {"role": "user", "content": [{"type": "input_text", "text": body.query}]}

    # 1) Preferred v2: top-level tool_resources
    p1 = {
        "model": model,
        "input": [user_msg],
        "tools": [{"type": "file_search"}],
        "tool_choice": "auto",
        "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
        "max_output_tokens": 800,
        "temperature": 0.2,
        "tool_resources": {"file_search": {"vector_store_ids": [vs]}},
    }

    # 2) Nested: tools[0].file_search.vector_store_ids
    p2 = {
        "model": model,
        "input": [user_msg],
        "tools": [{"type": "file_search", "file_search": {"vector_store_ids": [vs]}}],
        "tool_choice": "auto",
        "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
        "max_output_tokens": 800,
        "temperature": 0.2,
    }

    # 3) Flat (legacy): tools[0].vector_store_ids
    p3 = {
        "model": model,
        "input": [user_msg],
        "tools": [{"type": "file_search", "vector_store_ids": [vs]}],
        "tool_choice": "auto",
        "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
        "max_output_tokens": 800,
        "temperature": 0.2,
    }

    import requests, json

    attempts = [
        ("tool_resources", p1),
        ("nested_file_search", p2),
        ("flat_vector_store_ids", p3),
    ]

    results = []
    for variant, payload in attempts:
        try:
            r = requests.post("https://api.openai.com/v1/responses",
                              headers=headers, json=payload, timeout=120)
        except Exception as e:
            results.append({"variant": variant, "exception": str(e)})
            continue

        if r.status_code == 200:
            return {"ok": True, "variant": variant, "response": r.json()}

        # collect structured error and try next variant
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        results.append({"variant": variant, "status": r.status_code, "error": err})

        # Only retry for shape-related 400s; otherwise stop early
        if r.status_code != 400:
            break

    # Nothing worked: return all attempt diagnostics
    return {"attempts": results}, 502

    if r.status_code != 200:
        try:
            b = r.json()
        except Exception:
            b = {"text": r.text}
        return {"upstream_status": r.status_code, "error": b}, 502

    return {"ok": True, "response": r.json()}

# O62134004
