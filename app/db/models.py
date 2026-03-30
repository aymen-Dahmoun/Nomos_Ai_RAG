from sqlalchemy import Column, Integer, Text, LargeBinary, DateTime, String
from sqlalchemy.sql import func
from app.db.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    source = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PrecomputedAnswer(Base):
    __tablename__ = "precomputed_answers"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, unique=True, index=True, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # Store as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
