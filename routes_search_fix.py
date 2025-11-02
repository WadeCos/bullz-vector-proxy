from typing import Optional, List
import os, sys, json
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

# Required env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[/search] OPENAI_API_KEY missing", file=sys.stderr)
client = OpenAI(api_key=OPENAI_API_KEY)

VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "bullz-vector-store")
ACTION_SECRET_EXPECTED = os.getenv("ACTION_SHARED_SECRET", "")

class SearchBody(BaseModel):
    query: str
    top_k: Optional[int] = 6
    namespace: Optional[str] = None  # accepted but optional

def _get_or_create_vector_store_id() -> str:
    # Try create; if conflicts, list and pick by name
    try:
        vs = client.beta.vector_stores.create(name=VECTOR_STORE_NAME)
        return vs.id
    except Exception:
        stores = list(client.beta.vector_stores.list(limit=100))
        for s in stores:
            if getattr(s, "name", None) == VECTOR_STORE_NAME:
                return s.id
        # last-ditch: create again with unique name
        vs = client.beta.vector_stores.create(name=VECTOR_STORE_NAME)
        return vs.id

def _extract_text(resp) -> str:
    # Prefer convenience property if present
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Fallback: walk items/content
    out: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for ct in getattr(item, "content", []) or []:
            if getattr(ct, "type", "") in ("output_text", "text") and getattr(ct, "text", None):
                out.append(ct.text)
    return "\n".join(out).strip() or "(no text output)"

@router.post("/search")
def search(body: SearchBody, X_Action_Secret: str = Header(..., alias="X-Action-Secret")):
    if not ACTION_SECRET_EXPECTED or X_Action_Secret != ACTION_SECRET_EXPECTED:
        raise HTTPException(status_code=401, detail="bad secret")
    if not body.query or not isinstance(body.query, str):
        raise HTTPException(status_code=400, detail="query required")

    try:
        vs_id = _get_or_create_vector_store_id()

        # Minimal, stable Responses call; avoids brittle fields
        resp = client.responses.create(
            model=os.getenv("SEARCH_MODEL", "gpt-4.1-mini"),
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": body.query}]
            }],
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
            metadata={"origin": "bullz-vector-proxy", "top_k": int(body.top_k or 6)},
        )

        text = _extract_text(resp)

        # Try best-effort citations if present
        cites = []
        try:
            for item in getattr(resp, "output", []) or []:
                for ct in getattr(item, "content", []) or []:
                    if getattr(ct, "type", "") == "file_citation" and getattr(ct, "file_id", None):
                        cites.append({"file_id": ct.file_id, "quote": getattr(ct, "quote", None)})
        except Exception:
            pass

        return {"ok": True, "text": text, "citations": cites}

    except HTTPException:
        raise
    except Exception as e:
        # Log server-side for Render logs, return compact error to client
        print(f"[/search] ERROR: {e}", file=sys.stderr)
        try:
            print("DETAIL:", json.dumps(getattr(e, "args", [])[:1]), file=sys.stderr)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail={"error": str(e) or repr(e)})
