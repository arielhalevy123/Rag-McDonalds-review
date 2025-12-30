"""Ingestion script for populating Chroma database with document embeddings.

Chunking strategy: Each review is treated as a single chunk (no further splitting).
Rationale: Reviews are naturally short (typically <500 words), self-contained 
semantic units, and splitting would fragment sentiment/context. This is a 
deliberate design choice for this dataset.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

# Detect project root robustly, regardless of working directory
# This file is at scripts/ingest.py, so project root is 2 levels up
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_PATH = _PROJECT_ROOT / "data" / "documents.jsonl"
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "mcd_reviews"
EMBED_MODEL = "text-embedding-3-small"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

chroma = chromadb.PersistentClient(path=str(_CHROMA_DIR))

# Delete existing collection if it exists (distance metric cannot be changed after creation)
try:
    chroma.delete_collection(name=COLLECTION_NAME)
except Exception:
    pass  # Collection doesn't exist, which is fine

# Create collection with cosine distance metric
col = chroma.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

docs, ids = [], []
with open(_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj["id"])
        docs.append(obj["text"])

# Avoid duplicates
existing = set()
try:
    existing = set(col.get(ids=ids).get("ids", []))
except Exception:
    pass

new_docs, new_ids = [], []
for d, i in zip(docs, ids):
    if i not in existing:
        new_docs.append(d)
        new_ids.append(i)

if not new_docs:
    print("Nothing new to ingest.")
    raise SystemExit(0)

emb = client.embeddings.create(model=EMBED_MODEL, input=new_docs)
vectors = [e.embedding for e in emb.data]

col.add(ids=new_ids, documents=new_docs, embeddings=vectors)
print(f"Ingested {len(new_docs)} docs into {_CHROMA_DIR} / {COLLECTION_NAME}")

