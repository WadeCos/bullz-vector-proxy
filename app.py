# --- Bullz Vector Proxy: clean, fully-defined app.py (no module-scope 'body' refs) ---
import os, json, typing
from typing import Optional
import json
import os
import requests
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
try:
    from pydantic import BaseModel
except Exception:
    # Minimal fallback to avoid NameError if pydantic import fails
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

app = FastAPI(title="Bullz Vector Proxy", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchBody(BaseModel):
    query: str
    top_k: typing.Optional[int] = 6
    namespace: typing.Optional[str] = None

def _require_secret(x_action_secret: typing.Optional[str]) -> None:
    expected = os.getenv("ACTION_SHARED_SECRET", "")
    if not expected or x_action_secret != expected:
        raise HTTPException(status_code=401, detail="bad secret")

def _env(name: str, required: bool = True) -> str:
    val = os.getenv(name, "")
    if required and not val:
        raise HTTPException(status_code=500, detail=f"missing {name}")
    return val

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/info")
def info():
    return {
        "vector_store_id": os.getenv("VECTOR_STORE_ID", ""),
        "vector_store_name": os.getenv("VECTOR_STORE_NAME", "bullz-vector-store"),
    }

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    namespace: typing.Optional[str] = None,
    x_action_secret: str = Header(None, alias="X-Action-Secret"),
):
    _require_secret(x_action_secret)
    api_key = _env("OPENAI_API_KEY")
    vector_store_id = _env("VECTOR_STORE_ID")

    # 1) Upload the raw file to OpenAI Files
    content = await file.read()
    files = {"file": (file.filename or "upload.bin", content)}
    data = {"purpose": "assistants"}  # purpose suitable for vector stores/assistants
    r1 = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=300,
    )
    if r1.status_code >= 400:
        try:
            return {"stage": "files.upload", "status": r1.status_code, "error": r1.json()}
        except Exception:
            return {"stage": "files.upload", "status": r1.status_code, "error_text": r1.text}
    file_id = r1.json().get("id")

    # 2) Attach file to Vector Store
    payload = {"file_id": file_id}
    # Some accounts support metadata; if present, include namespace
    if namespace:
        payload["metadata"] = {"namespace": namespace}
    r2 = requests.post(
        f"https://api.openai.com/v1/vector_stores/{vector_store_id}/files",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    if r2.status_code >= 400:
        try:
            return {"stage": "vector_stores.attach", "status": r2.status_code, "error": r2.json()}
        except Exception:
            return {"stage": "vector_stores.attach", "status": r2.status_code, "error_text": r2.text}

    return {
        "ok": True,
        "file_id": file_id,
        "vector_store_id": vector_store_id,
        "namespace": namespace,
        "attach": r2.json(),
    }

@app.post("/search")
def search(body: SearchBody, x_action_secret: str = Header(None, alias="X-Action-Secret")):
    _require_secret(x_action_secret)
    api_key = _env("OPENAI_API_KEY")
    vector_store_id = _env("VECTOR_STORE_ID")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # Build Responses API payload with file_search tool
    payload: dict = {
        "model": model,
        "input": body.query,
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [vector_store_id]}},
    }
    # Namespace passed as metadata hint (if you use it during ingestion)
    if body.namespace:
        payload["metadata"] = {"namespace": body.namespace}
    _ = body.top_k  # kept for compatibility; not directly used by Responses API

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key, 'OpenAI-Beta': 'assistants=v2'}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    if resp.status_code >= 400:
        try:
            return {"upstream_status": resp.status_code, "error": resp.json()}
        except Exception:
            return {"upstream_status": resp.status_code, "error_text": resp.text}

    try:
        return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="invalid JSON from upstream")

# redeploy-marker: 1762130704
