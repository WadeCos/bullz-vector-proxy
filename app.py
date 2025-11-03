import os, json, requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ---- Models ----
class SearchBody(BaseModel):
    query: str
    top_k: Optional[int] = 6
    namespace: Optional[str] = None

# ---- Helpers ----
def require_secret(x_action_secret: Optional[str]):
    expected = os.getenv("ACTION_SHARED_SECRET", "")
    if not expected or x_action_secret != expected:
        raise HTTPException(status_code=401, detail="bad secret")

def openai_headers():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not (key.startswith("sk-") and len(key) > 20):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return {
        "Authorization": f"Bearer {key}",
        "OpenAI-Beta": "assistants=v2",
    }

def get_vs():
    vs = os.getenv("VECTOR_STORE_ID", "").strip()
    if not vs:
        raise HTTPException(status_code=500, detail="VECTOR_STORE_ID not configured")
    return vs

# ---- Health & Info ----
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/info")
def info():
    return {"vector_store_id": os.getenv("VECTOR_STORE_ID", ""), "vector_store_name": "bullz-vector-store"}

@app.post("/authcheck")
def authcheck(x_action_secret: str = Header(None, alias="X-Action-Secret")):
    require_secret(x_action_secret)
    return {"ok": True}

# ---- Upload ----
@app.post("/upload")
def upload(
    x_action_secret: str = Header(None, alias="X-Action-Secret"),
    file: UploadFile = File(...),
    namespace: Optional[str] = Form(None),
):
    require_secret(x_action_secret)
    vs = get_vs()
    headers = openai_headers()

    # POST /v1/vector_stores/{id}/files (Assistants v2)
    url = f"https://api.openai.com/v1/vector_stores/{vs}/files"
    files = {"file": (file.filename, file.file)}
    data = {}
    # (We can choose to encode namespace via metadata later if needed)
    try:
        r = requests.post(url, headers=headers, files=files, data=data, timeout=90)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")
    if r.status_code != 200:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        return {"upstream_status": r.status_code, "error": body}, r.status_code
    out = r.json()
    return {
        "ok": True,
        "file_id": out.get("id") or out.get("data", [{}])[0].get("id"),
        "vector_store_id": vs,
        "namespace": namespace,
        "attach": out,
    }

# ---- Search ----
@app.post("/search")
def search(
    body: SearchBody,
    x_action_secret: str = Header(None, alias="X-Action-Secret"),
):
    require_secret(x_action_secret)
    vs = get_vs()
    headers = openai_headers()
    headers["Content-Type"] = "application/json"

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": body.query}
                ],
            }
        ],
        "tools": [{"type": "file_search"}],
        "tool_choice": "auto",
        "metadata": {"source": "bullz-vector-proxy", "namespace": body.namespace},
        "max_output_tokens": 800,
        "temperature": 0.2,
        "tool_resources": {
            "file_search": {"vector_store_ids": [vs]}
        },
    }

    try:
        r = requests.post("https://api.openai.com/v1/responses", headers=headers, data=json.dumps(payload), timeout=90)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {str(e)}")

    if r.status_code != 200:
        try:
            body_json = r.json()
        except Exception:
            body_json = {"text": r.text}
        # return cleanly without tuple formatting
        return {"upstream_status": r.status_code, "error": body_json}, 502

    return {"ok": True, "response": r.json()}

# redeploy-marker: REWRITE
