"""FastAPI application for retrieval-only RAG system."""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from .rag.retriever import retrieve

app = FastAPI(title="Retrieval-only RAG API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from frontend/static directory
_PROJECT_ROOT = Path(__file__).parent.parent
frontend_dir = _PROJECT_ROOT / "frontend"
static_dir = frontend_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of candidates to retrieve")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold (0.0-1.0)"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    similarity: float
    text: str


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: list[SearchResult]


@app.get("/")
async def root():
    """Serve the frontend HTML file."""
    return FileResponse(str(frontend_dir / "index.html"))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for relevant documents using semantic similarity.
    
    Args:
        request: Search request with query, top_k, and similarity_threshold.
    
    Returns:
        SearchResponse with query and list of results.
    
    Raises:
        HTTPException: 400 if query is empty, 500 if API key is missing.
    """
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Retrieve results
        results = retrieve(
            query=request.query.strip(),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(**result) for result in results
            ]
        )
    except ValueError as e:
        # Handle missing API key or other configuration errors
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key is missing. Please set OPENAI_API_KEY in .env file."
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during search: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

