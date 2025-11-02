from typing import Optional, List, Dict, Any
import os, sys, json
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
import httpx

try:
    from openai import OpenAI
except Exception as _e:
    OpenAI = None  # type: ignore

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[/search] OPENAI_API_KEY missing", file=sys.stderr)

# SDK client if available
client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI else None

VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "bullz-vector-store")
ACTION_SECRET_EXPECTED = os.getenv("ACTION_SHARED_SECRET", "")
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "gpt-4.1-mini")

class SearchBody(BaseModel):
    query: str
    top_k: Optional[int] = 6
    namespace: Optional[str] = None  # accepted but currently unused (stable path)

def _get_or_create_vector_store_id() -> str:
    """
    Use SDK if available. If not, attempt REST creation/list using /v1/vector_stores later (TODO).
    For now, we require SDK for vector store ops; the app’s startup likely created the store already.
    """
    if not client:
        # Best effort: ask app's /info via local import would be messy.
        # Fail clearly so the logs instruct upgrading SDK.
        raise RuntimeError("OpenAI SDK unavailable; cannot resolve vector store id")
    try:
        vs = client.beta.vector_stores.create(name=VECTOR_STORE_NAME)
        return vs.id
    except Exception:
        stores = list(client.beta.vector_stores.list(limit=100))
        for s in stores:
            if getattr(s, "name", None) == VECTOR_STORE_NAME:
                return s.id
        # Try to create once more
        vs = client.beta.vector_stores.create(name=VECTOR_STORE_NAME)
        return vs.id

def _extract_text(resp: Any) -> str:
    # Prefer convenience property if present
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    # Fallback: walk content
    out: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for ct in getattr(item, "content", []) or []:
            if getattr(ct, "type", "") in ("output_text", "text") and getattr(ct, "text", None):
                out.append(ct.text)
    return "\n".join(out).strip() or "(no text output)"

def _responses_via_sdk(vs_id: str, query: str, top_k: int) -> Dict[str, Any]:
    resp = client.responses.create(
        model=SEARCH_MODEL,
        input=[{"role":"user","content":[{"type":"input_text","text": query}]}],
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
        metadata={"origin":"bullz-vector-proxy","top_k": top_k},
    )
    text = _extract_text(resp)
    cites = []
    for item in getattr(resp, "output", []) or []:
        for ct in getattr(item, "content", []) or []:
            if getattr(ct, "type", "") == "file_citation" and getattr(ct, "file_id", None):
                cites.append({"file_id": ct.file_id, "quote": getattr(ct, "quote", None)})
    return {"ok": True, "text": text, "citations": cites}

def _responses_via_rest(vs_id: str, query: str, top_k: int) -> Dict[str, Any]:
    """
    Fallback path: call HTTPS /v1/responses directly.
    Works even if the Python SDK is older and lacks .responses.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SEARCH_MODEL,
        "input": [
            {
                "role": "user",
                "content": [{"type":"input_text","text": query}],
            }
        ],
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [vs_id]}},
        "metadata": {"origin":"bullz-vector-proxy","top_k": top_k},
    }
    with httpx.Client(timeout=60) as http:
        r = http.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail={"error": f"REST responses failed: {r.status_code}", "body": r.text})
        data = r.json()
    # Try to normalize output text
    text = ""
    if isinstance(data, dict):
        # new schema has top-level "output_text" convenience in some SDKs; REST returns "output" list
        text = data.get("output_text") or ""
        if not text:
            items = data.get("output") or []
            parts: List[str] = []
            for item in items:
                for ct in (item.get("content") or []):
                    if ct.get("type") in ("output_text","text") and ct.get("text"):
                        parts.append(ct["text"])
            text = "\n".join(parts).strip() or "(no text output)"
    return {"ok": True, "text": text, "citations": []}

@router.post("/search")
def search(body: SearchBody, X_Action_Secret: str = Header(..., alias="X-Action-Secret")):
    # Auth
    if not ACTION_SECRET_EXPECTED or X_Action_Secret != ACTION_SECRET_EXPECTED:
        raise HTTPException(status_code=401, detail="bad secret")
    # Input
    if not body.query or not isinstance(body.query, str):
        raise HTTPException(status_code=400, detail="query required")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")

    # Resolve vector store id using SDK (app startup likely created one already)
    try:
        vs_id = _get_or_create_vector_store_id()
    except Exception as e:
        # If SDK is too old to do vector_store ops, try to read from your own /info route if available
        # but to keep this generic, just surface a clear message:
        raise HTTPException(status_code=500, detail={"error":"SDK too old to manage vector stores; upgrade openai to >=1.52.0", "detail": str(e)})

    top_k = int(body.top_k or 6)

    # Prefer SDK if it has .responses; else REST fallback
    try:
        if client and hasattr(client, "responses") and hasattr(client.responses, "create"):
            return _responses_via_sdk(vs_id, body.query, top_k)
        else:
            return _responses_via_rest(vs_id, body.query, top_k)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[/search] ERROR: {e}", file=sys.stderr)
        try: print("DETAIL:", json.dumps(getattr(e, "args", [])[:1]), file=sys.stderr)
        except Exception: pass
        raise HTTPException(status_code=500, detail={"error": str(e) or repr(e)})
