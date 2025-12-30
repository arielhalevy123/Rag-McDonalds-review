# Retrieval-only RAG System

A production-quality retrieval-only RAG (Retrieval-Augmented Generation) system that performs semantic search over McDonald's store reviews. This system implements **only the retrieval component** - no LLM generation, no chatbot responses, no summarization. The UI displays retrieved context chunks with similarity scores.

## Overview

This project demonstrates a complete retrieval-only RAG pipeline:
- **Ingestion**: Documents are embedded using OpenAI's `text-embedding-3-small` model and stored in a local Chroma vector database
- **Retrieval**: User queries are embedded and matched against document embeddings using cosine similarity
- **Display**: Retrieved chunks are displayed with similarity scores, filtered by a configurable threshold

## Dataset

**Source**: McDonald's Store Reviews from Kaggle (original dataset is large)

**Sampling**: This project uses ~100 reviews sampled from the original dataset to:
- Stay under cost constraints (embedding costs)
- Focus on retrieval quality demonstration
- Provide a manageable dataset for assignment purposes

Each review is treated as a single document chunk (see Design Decisions for rationale).

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Ingestion

Before running the search system, you need to ingest documents into Chroma:

```bash
python scripts/ingest.py
```

This script:
- Reads documents from `data/documents.jsonl`
- Creates embeddings using OpenAI's `text-embedding-3-small` model
- Stores embeddings in the local Chroma database at `chroma_db/`
- Skips documents that are already ingested (idempotent)

**Note**: The first run will create embeddings for all documents. Subsequent runs will only process new documents.

## Running the Application

1. **Start the FastAPI server**:
   ```bash
   uvicorn backend.app:app --reload
   ```

2. **Open the UI**:
   Navigate to `http://localhost:8000` in your browser

3. **Search**:
   - Enter a query in the search box
   - Optionally adjust `top_k` (number of candidates) and `similarity_threshold`
   - Click "Search" to retrieve relevant document chunks

## Design Decisions

### Chunking Strategy

**Decision**: Each review is treated as a single chunk (no further splitting).

**Rationale**:
- Reviews are naturally short (typically <500 words)
- Reviews are self-contained semantic units
- Splitting would fragment sentiment and context
- This is a deliberate design choice for this dataset

### Vector Database

**Decision**: Chroma (local persistent database)

**Rationale**:
- **Simplicity**: No external service setup required
- **No credentials**: Works offline after initial ingestion
- **Assignment scope**: Suitable for demonstration purposes
- **Offline operation**: Once embeddings are created, queries work without API calls (except for query embedding)

**Trade-off**: Local-only vs. cloud scalability. For production at scale, consider cloud vector databases (Pinecone, Weaviate, etc.).

### Similarity Scoring

**Decision**: Cosine similarity computed directly from embeddings

**Implementation**: 
```python
cosine_similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
```

**Rationale**:
- Provides true semantic similarity measure (not a distance-based heuristic)
- Bounded [0, 1] for normalized embeddings (OpenAI embeddings are normalized)
- More interpretable than distance-based measures
- Directly interpretable: 1.0 = identical, 0.0 = orthogonal

### Threshold Semantics

**Decision**: `similarity_threshold` filters results where cosine similarity >= threshold

**Interpretation**:
- **Lower threshold (0.2-0.3)**: More permissive, returns more results (higher recall)
- **Higher threshold (0.5+)**: Stricter relevance, returns fewer results (higher precision)
- **Default 0.3**: Balances recall and precision

**Cosine similarity values**:
- `1.0`: Identical semantic meaning
- `0.7-0.9`: Very similar
- `0.5-0.7`: Moderately similar
- `0.3-0.5`: Somewhat related
- `0.0-0.3`: Weakly related or orthogonal

### Query-time Embeddings

**Decision**: Query embeddings are created at runtime using the same model as documents

**Rationale**:
- **Necessary for semantic search**: The query must be embedded using the same model as documents to enable vector similarity search
- **Standard RAG pattern**: This is how retrieval works in production RAG systems
- **Distinct from ingestion**: Ingestion-time embeddings (documents) are created once and stored; query-time embeddings are created on-demand

### Top K Parameter

**Decision**: `top_k` specifies the number of candidate documents retrieved before threshold filtering

**Behavior**:
- Higher `top_k`: More candidates retrieved, but may include less relevant results
- Lower `top_k`: Fewer candidates, but only the most similar are considered
- Threshold filtering is applied after retrieval, so final results may be fewer than `top_k`

## Evaluation Criteria

When assessing retrieval quality, consider:

1. **Relevance**: Retrieved chunks should be semantically related to the query (not just keyword matches)
2. **Ranking**: Most similar results should appear first (sorted by similarity descending)
3. **Threshold behavior**: Results above threshold should be meaningfully related; below threshold should be less relevant
4. **Coverage**: For diverse queries, the system should retrieve relevant reviews across different topics/sentiments
5. **No false positives**: Avoid returning completely unrelated reviews

## Example Queries

Here are 5 example queries to test the system:

### 1. "Poor service and slow"
**Expected**: Reviews mentioning bad service, slow service, or customer service issues
**Evaluation**: Should return reviews with complaints about service quality, wait times, or staff behavior

### 2. "Good food quality"
**Expected**: Positive reviews about food taste, quality, or freshness
**Evaluation**: Should return reviews praising food, with similarity scores typically >0.4

### 3. "Clean restaurant"
**Expected**: Reviews mentioning cleanliness, hygiene, or restaurant condition
**Evaluation**: Should return reviews discussing cleanliness (both positive and negative)

### 4. "Drive-thru experience"
**Expected**: Reviews specifically about drive-thru service
**Evaluation**: Should return reviews mentioning drive-thru, with context about speed, accuracy, or convenience

### 5. "Price and value"
**Expected**: Reviews discussing pricing, value for money, or cost
**Evaluation**: Should return reviews mentioning prices, whether positive (good value) or negative (too expensive)

## API Endpoints

### POST `/search`

Search for relevant documents using semantic similarity.

**Request**:
```json
{
  "query": "poor service",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Response**:
```json
{
  "query": "poor service",
  "results": [
    {
      "id": "doc_007",
      "similarity": 0.7823,
      "text": "Poor service and constantly messing orders up!"
    },
    ...
  ]
}
```

**Error Responses**:
- `400`: Empty query
- `500`: Missing API key or other server errors

## Project Structure

```
SoluGenAi/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── rag/
│       ├── __init__.py
│       ├── chroma_client.py   # Chroma client initialization
│       ├── embeddings.py      # OpenAI embedding helper
│       └── retriever.py       # Retrieval logic
├── frontend/
│   ├── index.html             # Main UI
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend logic
├── data/
│   └── documents.jsonl        # Document dataset
├── scripts/
│   └── ingest.py              # Ingestion script
├── chroma_db/                 # Chroma database (gitignored)
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Important Notes

- **No LLM Generation**: This system only retrieves relevant chunks. It does not generate answers, summaries, or chatbot responses.
- **Cost Management**: Embedding costs are kept low by using a small dataset (~100 documents) and the efficient `text-embedding-3-small` model.
- **API Key Security**: Never commit your `.env` file with real API keys. Use `.env.example` as a template.

## Troubleshooting

**"Missing OPENAI_API_KEY" error**:
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY` is set correctly in `.env`

**No results returned**:
- Try lowering the `similarity_threshold` (e.g., 0.2)
- Increase `top_k` to retrieve more candidates
- Check that ingestion completed successfully

**Chroma collection not found**:
- Run `python scripts/ingest.py` to create the collection and ingest documents

## License

This project is for educational/assignment purposes.

