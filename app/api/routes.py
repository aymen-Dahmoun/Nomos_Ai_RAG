from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.db.database import get_db
from app.services.rag import RAGService

router = APIRouter()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

class IngestRequest(BaseModel):
    content: str
    source: Optional[str] = "manual"

class RecommendRequest(BaseModel):
    text: str
    lawyers: List[dict]

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, explain: bool = False, db: Session = Depends(get_db)):
    rag_service = RAGService(db)
    return await rag_service.ask(request.question, explain=explain)

@router.post("/recommand")
async def recommand(request: RecommendRequest):
    from app.services.gemini import gemini_service
    answer = await gemini_service.recommend_lawyers(request.text, request.lawyers)
    return {"answer": answer}

@router.post("/precompute", response_model=AskResponse)
async def precompute(request: AskRequest, db: Session = Depends(get_db)):
    rag_service = RAGService(db)
    return await rag_service.precompute(request.question)

@router.post("/ingest")
async def ingest(request: IngestRequest, db: Session = Depends(get_db)):
    rag_service = RAGService(db)
    count = rag_service.ingest(request.content, request.source)
    return {"message": f"Successfully ingested {count} chunks"}
