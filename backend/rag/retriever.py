"""Retrieval logic with cosine similarity computation."""
import numpy as np
from .chroma_client import collection
from .embeddings import create_embedding


def retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.3
) -> list[dict]:
    """Retrieve relevant documents based on cosine similarity.
    
    Fetches more candidates than requested (top_k + 10, max 60) to compensate
    for ANN approximation, then computes exact cosine similarity and re-ranks.
    This improves recall by ensuring the true top-k results are included in the
    candidate set, while maintaining full control over ranking and thresholds.
    
    Similarity calculation: Compute cosine similarity directly from embeddings
    using dot product. OpenAI embeddings are normalized, so cosine similarity
    is bounded [0, 1] and provides true semantic similarity measure.
    
    Args:
        query: The search query text.
        top_k: Number of results to return (max 50 as per API contract).
        similarity_threshold: Minimum cosine similarity required (0.0-1.0).
                              Lower values (0.2-0.3) = more permissive,
                              higher values (0.5+) = stricter relevance.
    
    Returns:
        List of result dictionaries, each containing:
        - id: Document ID
        - similarity: Cosine similarity score (0-1)
        - text: Document text content
        Results are sorted by similarity descending, limited to top_k.
    """
    # Create query embedding (query-time embedding for semantic search)
    query_embedding = create_embedding(query)
    query_embedding = np.array(query_embedding)
    
    # Fetch more candidates than requested to compensate for ANN approximation
    # Request top_k + 10, capped at 60 (since user can request up to 50)
    fetch_k = min(top_k + 10, 60)
    
    # Query Chroma to get candidate documents
    # Chroma uses approximate nearest neighbor search, so fetching more candidates
    # increases the chance that the true top-k results are in the candidate set
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=fetch_k,
        include=["documents", "embeddings"]
    )
    
    # Extract results
    # Handle case where no results are returned
    if not results["ids"] or len(results["ids"][0]) == 0:
        return []
    
    doc_ids = results["ids"][0]
    doc_texts = results["documents"][0]
    
    # Chroma returns embeddings as a list of lists (one per query)
    # Since we have one query, get the first element
    doc_embeddings = results.get("embeddings")
    if doc_embeddings and len(doc_embeddings) > 0:
        doc_embeddings = doc_embeddings[0]
    else:
        # Fallback: fetch embeddings explicitly
        doc_data = collection.get(ids=doc_ids, include=["embeddings"])
        if "embeddings" in doc_data and doc_data["embeddings"]:
            doc_embeddings = doc_data["embeddings"]
        else:
            raise ValueError("Could not retrieve document embeddings from Chroma")
    
    # Compute exact cosine similarity for all candidates
    # This ensures true similarity ordering, not ANN approximations
    candidates = []
    for doc_id, doc_text, doc_emb in zip(doc_ids, doc_texts, doc_embeddings):
        doc_emb = np.array(doc_emb)
        # Cosine similarity: dot product (since OpenAI embeddings are normalized)
        cosine_sim = float(np.dot(query_embedding, doc_emb))
        candidates.append({
            "id": doc_id,
            "similarity": cosine_sim,
            "text": doc_text
        })
    
    # Re-rank by exact similarity (descending order)
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Filter by threshold and return top_k results
    results_list = []
    for candidate in candidates:
        if candidate["similarity"] >= similarity_threshold and len(results_list) < top_k:
            candidate["similarity"] = round(candidate["similarity"], 4)
            results_list.append(candidate)
    
    return results_list

