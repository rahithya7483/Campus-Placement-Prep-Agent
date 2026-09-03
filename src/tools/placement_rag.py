import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "chroma_db")
COLLECTION_NAME = "company_notes"

_model = None
_collection = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def search_placement_data(query: str, k: int = 5, field: str | None = None) -> list[dict]:
    """
    Returns [{"text": str, "field": str, "company": str}, ...] for the k
    most relevant chunks. Pass field= to restrict to one field
    (interview_process | role_type | cgpa_cutoff | technical_questions | hr_questions).
    """
    model = _get_model()
    collection = _get_collection()
    q_emb = model.encode([query]).tolist()
    where = {"field": field} if field else None
    results = collection.query(query_embeddings=q_emb, n_results=k, where=where)

    out = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        out.append({"text": doc, "field": meta["field"], "company": meta["company"]})
    return out

if __name__ == "__main__":
    for r in search_placement_data("What is the interview process for TCS?"):
        print(f"[{r['field']}] {r['company']}: {r['text']}")