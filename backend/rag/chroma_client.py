"""Chroma client initialization and collection management.

Design rationale: Local persistent Chroma chosen for simplicity, no external 
dependencies/credentials, suitable for assignment scope, and allows offline 
operation after initial ingestion.
"""
import chromadb
from pathlib import Path

# Detect project root robustly, regardless of working directory
# This file is at backend/rag/chroma_client.py, so project root is 3 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "mcd_reviews"

# Initialize persistent Chroma client
_chroma_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

# Get or create collection
collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)

