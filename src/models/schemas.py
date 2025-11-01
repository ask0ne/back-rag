from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum


class SourceType(str, Enum):
    DOCUMENT = "document"
    WEB = "web"


class DocumentMetadata(BaseModel):
    title: str
    source: str
    page_number: Optional[int] = None
    file_type: str


class QueryRequest(BaseModel):
    query: str
    confidence_threshold: Optional[float] = 0.7


class SearchQueryRequest(BaseModel):
    """Request model for the unified /search/query endpoint."""
    query: str = Field(..., description="The search query", min_length=1)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold (0.0 to 1.0)")


class SearchResponse(BaseModel):
    """Response model for the unified /search/query endpoint."""
    source: SourceType
    answer: str
    confidence: float
    context_snippets: List[str]
    reasoning: str


class RAGResponse(BaseModel):
    source: SourceType
    answer: str
    confidence: float
    context_snippets: List[str]
    reasoning_trace: Optional[List[str]] = None


class DocumentUploadResponse(BaseModel):
    message: str
    documents_processed: int
    total_chunks: int


class AgentState(BaseModel):
    query: str
    rag_answer: Optional[str] = None
    rag_confidence: Optional[float] = None
    web_answer: Optional[str] = None
    context_snippets: List[str] = []
    reasoning_steps: List[str] = []
    final_source: Optional[SourceType] = None