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
    
    Similarity calculation: Compute cosine similarity directly from embeddings
    using dot product and L2 norms. OpenAI embeddings are normalized, so cosine
    similarity is bounded [0, 1] and provides true semantic similarity measure.
    
    Args:
        query: The search query text.
        top_k: Number of candidate documents to retrieve from Chroma.
        similarity_threshold: Minimum cosine similarity required (0.0-1.0).
                              Lower values (0.2-0.3) = more permissive,
                              higher values (0.5+) = stricter relevance.
    
    Returns:
        List of result dictionaries, each containing:
        - id: Document ID
        - similarity: Cosine similarity score (0-1)
        - text: Document text content
        Results are sorted by similarity descending.
    """
    # Create query embedding (query-time embedding for semantic search)
    query_embedding = create_embedding(query)
    query_embedding = np.array(query_embedding)
    
    # Query Chroma to get candidate documents
    # Note: Chroma doesn't support threshold natively, so we retrieve top_k
    # and filter post-retrieval based on cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
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
    
    # Compute cosine similarity for each document
    similarities = []
    for doc_emb in doc_embeddings:
        doc_emb = np.array(doc_emb)
        # Cosine similarity: dot product / (norm(query) * norm(doc))
        # Since OpenAI embeddings are normalized, this simplifies to dot product
        cosine_sim = np.dot(query_embedding, doc_emb)
        similarities.append(float(cosine_sim))
    
    # Create results with similarity scores
    results_list = []
    for doc_id, doc_text, similarity in zip(doc_ids, doc_texts, similarities):
        if similarity >= similarity_threshold:
            results_list.append({
                "id": doc_id,
                "similarity": round(similarity, 4),
                "text": doc_text
            })
    
    # Sort by similarity descending
    results_list.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results_list

