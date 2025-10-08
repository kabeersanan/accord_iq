# src/main.py
import os
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# load env
load_dotenv()

# utils (your modules)
from utils.extractor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.embedder import Embedder
from utils.vectorstore_faiss import FaissVectorStore
from utils.generator_gemini import GeminiGenerator

# configs
INDEX_DIR = os.path.join(os.path.dirname(__file__), "indexes")
METADATA_PATH = os.path.join(INDEX_DIR, "id_map.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMBED_DIM = 384  # update if you use different model

app = FastAPI(title="RAG PDF Chat")

# allow local testing from a frontend - remove or lock down in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (singletons)
embedder = Embedder()               # loads sentence-transformers (local) OR your Gemini embedder
generator = GeminiGenerator()       # uses GEMINI_API_KEY from .env
vector_store = FaissVectorStore(dim=EMBED_DIM)

# load persisted index/metadata if exists
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR, exist_ok=True)

# load id_map if present
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            vector_store.id_map = json.load(f)
        # optionally load the faiss binary index if present
        if os.path.exists(FAISS_INDEX_PATH):
            import faiss
            vector_store.index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ Loaded FAISS index and metadata from disk")
    except Exception as e:
        print("⚠️ Failed to load persisted index:", e)


def persist_index_and_map():
    """Save FAISS index and id_map to disk (call after upserts)."""
    try:
        import faiss
        faiss.write_index(vector_store.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(vector_store.id_map, f, ensure_ascii=False, indent=2)
        print("✅ Persisted FAISS index and metadata")
    except Exception as e:
        print("❌ Error persisting index:", e)


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, chunk, embed, and add to FAISS.
    Returns: file_id and summary info.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # generate a unique file id
    file_id = str(uuid.uuid4())

    # save uploaded file temporarily
    tmp_path = os.path.join(INDEX_DIR, f"{file_id}.pdf")
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # STEP: Extract text per page
    pages_text = extract_text_from_pdf(tmp_path)  # expected: list of strings (per page)
    # convert to list of page dicts for chunk_pages if you need page-level metadata
    pages = [{"page": i+1, "text": p} for i, p in enumerate(pages_text)]

    # STEP: Chunk the entire document (simple approach: join and chunk)
    full_text = " ".join(pages_text)
    chunks_texts = chunk_text(full_text)

    # build chunk objects with metadata
    chunks = []
    for i, t in enumerate(chunks_texts):
        cid = f"{file_id}_c{i}"
        chunks.append({
            "id": cid,
            "text": t,
            "metadata": {"file_id": file_id, "chunk_index": i}
        })

    # STEP: create embeddings
    chunks = embedder.create_embeddings(chunks)

    # STEP: add to FAISS
    vector_store.add_embeddings(chunks)

    # persist index and metadata
    persist_index_and_map()

    return JSONResponse(
        {
            "file_id": file_id,
            "num_pages": len(pages_text),
            "num_chunks": len(chunks_texts)
        }
    )


@app.post("/ask")
async def ask(question: str, file_id: Optional[str] = None, top_k: int = 3):
    """
    Ask a question. Optionally pass file_id to restrict search to that document.
    Returns a response with answer and citations.
    """
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # embed the question
    q_vec = embedder.embed_query(question)

    # perform search
    results = vector_store.search(q_vec, top_k=top_k)

    # optional: filter by file_id (simple filter)
    if file_id:
        results = [r for r in results if r.get("metadata", {}).get("file_id") == file_id]
        if not results:
            return JSONResponse({"answer": None, "retrieved": [], "note": "no chunks for that file_id"}, status_code=200)

    # call Gemini to generate an answer using retrieved context
    answer_text = generator.generate_answer(question, results)

    # return answer + provenance snippets
    return {
        "answer": answer_text,
        "retrieved": results
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG PDF Chat backend is running"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
