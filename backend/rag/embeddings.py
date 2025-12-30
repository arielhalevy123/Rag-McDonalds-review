"""OpenAI embedding helper for creating query and document embeddings.

Query-time embedding rationale: Query-time embedding creation is necessary for 
semantic search - the query must be embedded using the same model as documents 
to enable vector similarity search. This is distinct from ingestion-time 
embeddings (documents) and is a standard RAG pattern.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "Missing OPENAI_API_KEY in environment. Please set it in .env file."
    )

_client = OpenAI(api_key=_api_key)


def create_embedding(text: str) -> list[float]:
    """Create an embedding for the given text using OpenAI.
    
    Args:
        text: The text to embed.
        
    Returns:
        A list of floats representing the embedding vector.
        
    Raises:
        ValueError: If API key is missing.
        Exception: If OpenAI API call fails.
    """
    response = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

