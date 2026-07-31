
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.vector_db import get_vector_store
from src.rag_chain import run_retrieval

app = FastAPI(title="RAG Chatbot API")
PERSIST_DIR = "data/vector_store"

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_bot(req: QueryRequest):
    try:
        vs = get_vector_store(PERSIST_DIR)
        answer, docs = run_retrieval(vs, req.query)
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
