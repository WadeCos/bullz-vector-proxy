import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- Config via env ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]           # required
ACTION_SHARED_SECRET = os.environ["ACTION_SHARED_SECRET"]  # required
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "https://chat.openai.com")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "").strip()
VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "bullz-vector-store")

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Ensure a Vector Store exists (create if missing) ---
def ensure_vector_store(vs_id: str, vs_name: str) -> str:
    if vs_id:
        return vs_id
    # Try to find by name
    try:
        stores = client.vector_stores.list()
        for s in stores.data:
            if (getattr(s, "name", None) or "") == vs_name:
                return s.id
    except Exception:
        pass
    # Create new
    vs = client.vector_stores.create(name=vs_name)
    print(f"[startup] Created vector store: {vs.id} ({vs_name})", flush=True)
    return vs.id

VECTOR_STORE_ID = ensure_vector_store(VECTOR_STORE_ID, VECTOR_STORE_NAME)

# --- FastAPI app ---
app = FastAPI(title="Bullz Vector Proxy", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    data = await file.read()
    ofile = client.files.create(
        file=(file.filename, data, file.content_type or "application/octet-stream"),
        purpose="assistants"
    )
    client.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=ofile.id)
    if namespace:
        client.files.update(ofile.id, metadata={"namespace": namespace})
    return {"ok": True, "file_id": ofile.id, "namespace": namespace}

class SearchReq(BaseModel):
    query: str
    top_k: int | None = 6
    namespace: str | None = None

@app.post("/search")
def search(body: SearchReq, x_action_secret: str | None = Header(None)):
    check_secret(x_action_secret)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role":"user","content": body.query}],
        tools=[{"type":"file_search"}],
        tool_choice="file_search",
        file_search={"vector_stores":[{"vector_store_id": VECTOR_STORE_ID}]}
    )
    return resp.output
