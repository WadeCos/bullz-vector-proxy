import os, secrets
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- Env config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it in Render → Settings → Environment.")

ACTION_SHARED_SECRET = os.getenv("ACTION_SHARED_SECRET")
if not ACTION_SHARED_SECRET:
    ACTION_SHARED_SECRET = "bullz_" + secrets.token_hex(24)
    print(f"[startup] ACTION_SHARED_SECRET was missing. Generated one-time secret: {ACTION_SHARED_SECRET}", flush=True)

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "https://chat.openai.com")
VECTOR_STORE_ID = (os.getenv("VECTOR_STORE_ID") or "").strip()
VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "bullz-vector-store")

client = OpenAI(api_key=OPENAI_API_KEY)

def ensure_vector_store(vs_id: str, vs_name: str) -> str:
    """
    Ensure a vector store exists. Try env ID, then find by name, else create.
    Uses the beta.vector_stores API.
    """
    try:
        if vs_id:
            print(f"[startup] Using VECTOR_STORE_ID from env: {vs_id}", flush=True)
            return vs_id

        # find by name
        stores = client.beta.vector_stores.list()
        for s in getattr(stores, "data", []):
            if (getattr(s, "name", "") or "") == vs_name:
                print(f"[startup] Using existing vector store: {s.id} ({vs_name})", flush=True)
                return s.id

        # create new
        vs = client.beta.vector_stores.create(name=vs_name)
        print(f"[startup] Created vector store: {vs.id} ({vs_name})", flush=True)
        return vs.id

    except Exception as e:
        print(f"[startup] ERROR ensuring vector store: {e}", flush=True)
        raise

VECTOR_STORE_ID = ensure_vector_store_safe(VECTOR_STORE_ID, VECTOR_STORE_NAME)

app = FastAPI(title="Bullz Vector Proxy", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def check_secret(secret: str | None):
    if not secret or secret != ACTION_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/info")
def info():
    return {"vector_store_id": VECTOR_STORE_ID, "vector_store_name": VECTOR_STORE_NAME}

@app.post("/upload")
async def upload(file: UploadFile = File(...), namespace: str | None = None,
                 x_action_secret: str | None = Header(None)):
    check_secret(x_action_secret)

    # 1) Upload file to OpenAI Files
    data = await file.read()
    ofile = client.files.create(
        file=(file.filename, data, file.content_type or "application/octet-stream"),
        purpose="assistants"
    )

    # 2) Attach to Vector Store (beta API)
    client.beta.vector_stores.files.create(
        vector_store_id=VECTOR_STORE_ID,
        file_id=ofile.id
    )

    # 3) Optional: tag with namespace (metadata)
    if namespace:
        try:
            client.files.update(ofile.id, metadata={"namespace": namespace})
        except Exception as e:
            # If metadata update isn't supported in your region/version, just log and continue
            print(f"[upload] metadata update skipped: {e}", flush=True)

    return {"ok": True, "file_id": ofile.id, "namespace": namespace}

class SearchReq(BaseModel):
    query: str
    top_k: int | None = 6
    namespace: str | None = None

@app.post("/search")
def search(body: SearchReq, x_action_secret: str | None = Header(None)):
    check_secret(x_action_secret)

    # Responses API with file_search tool and your Vector Store attached
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role":"user","content": body.query}],
        tools=[{"type":"file_search"}],
        tool_choice="file_search",
        file_search={"vector_stores":[{"vector_store_id": VECTOR_STORE_ID}]}
    )

    # Return structured output (includes citations)
    return resp.output

from routes_search_fix import router as search_fix_router
try:
    app.include_router(search_fix_router)
except Exception:
    pass

# --- Version-agnostic vector store resolver (SDK then REST fallback) ---
def ensure_vector_store_safe(vs_id: str | None, vs_name: str) -> str:
    """
    Works across old/new OpenAI SDKs.
    1) Try SDK: client.beta.vector_stores.list/create if available.
    2) Fallback to REST: GET/POST https://api.openai.com/v1/vector_stores
    Returns a vector_store_id.
    """
    import os, sys, json
    import httpx
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        OpenAI = None  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    # If caller already provided a store id, accept it
    if vs_id:
        return vs_id

    # 1) Try SDK first (if installed and has beta.vector_stores)
    client = None
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            beta = getattr(client, "beta", None)
            if beta and hasattr(beta, "vector_stores"):
                # List to find by name
                try:
                    stores = list(beta.vector_stores.list(limit=100))
                    for s in stores:
                        if getattr(s, "name", None) == vs_name:
                            return s.id
                except Exception:
                    pass
                # Create if not found
                vs = beta.vector_stores.create(name=vs_name)
                return vs.id
        except Exception as e:
            print(f"[ensure_vector_store_safe] SDK path failed: {e}", file=sys.stderr)

    # 2) REST fallback (works regardless of SDK version)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as http:
        # list and find by name
        r = http.get("https://api.openai.com/v1/vector_stores", headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"vector_stores list failed: {r.status_code} {r.text}")
        data = r.json()
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list):
            for s in items:
                if s.get("name") == vs_name and s.get("id"):
                    return s["id"]
        # create new
        r = http.post("https://api.openai.com/v1/vector_stores", headers=headers, json={"name": vs_name})
        if r.status_code >= 400:
            raise RuntimeError(f"vector_stores create failed: {r.status_code} {r.text}")
        created = r.json()
        vsid = created.get("id")
        if not vsid:
            raise RuntimeError("vector_stores create returned no id")
        return vsid
# --- end safe resolver ---
