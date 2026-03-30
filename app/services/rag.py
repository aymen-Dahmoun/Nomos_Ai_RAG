import json
import re

import numpy as np
from sqlalchemy.orm import Session

from app.core.cache import global_cache
from app.db.models import Document, PrecomputedAnswer
from app.services.embedding import embedding_service
from app.services.gemini import gemini_service
from app.services.retrieval import RetrievalService


class RAGService:
    def __init__(self, db: Session):
        self.db = db
        self.retrieval_service = RetrievalService(db)

    async def ask(self, question: str, explain: bool = False):
        cache_key = f"{question}_explain_{explain}"
        cached_answer = global_cache.get(cache_key)
        if cached_answer:
            return cached_answer
        if not explain:
            db_answer = (
                self.db.query(PrecomputedAnswer)
                .filter(PrecomputedAnswer.question == question)
                .first()
            )
            if db_answer:
                result = {
                    "answer": db_answer.answer,
                    "sources": json.loads(db_answer.sources),
                }
                global_cache.set(cache_key, result)
                return result
        relevant_docs = self.retrieval_service.get_top_k(question, k=10)
        context = "\n\n".join(
            [f"المصدر: {d['source']}\nالنص: {d['content']}" for d in relevant_docs]
        )
        print("\n=== FINAL CONTEXT SENT TO LLM ===")
        print(context[:500])

        # Generate answer
        answer = await gemini_service.generate_answer(
            question, context, explain=explain
        )

        result = {"answer": answer, "sources": relevant_docs}

        # Cache results
        global_cache.set(cache_key, result)

        return result

    async def precompute(self, question: str):
        # Run full RAG pipeline (non-explain by default)
        result = await self.ask(question, explain=False)

        # Store in DB if not already there
        existing = (
            self.db.query(PrecomputedAnswer)
            .filter(PrecomputedAnswer.question == question)
            .first()
        )
        if not existing:
            new_p = PrecomputedAnswer(
                question=question,
                answer=result["answer"],
                sources=json.dumps(result["sources"]),
            )
            self.db.add(new_p)
            self.db.commit()

        return result

    def ingest(self, content: str, source: str = None):
        def split_articles(text):
            # Split by "المادة" followed by one or more spaces and digits
            articles = re.split(r"(المادة\s+\d+)", text)

            chunks = []
            # re.split with groups returns [prefix, group1, suffix1, group2, suffix2, ...]
            # The first element is text before the first "المادة" (often empty or preamble)
            if articles[0].strip():
                chunks.append(articles[0].strip())

            for i in range(1, len(articles), 2):
                title = articles[i]
                body = articles[i + 1] if i + 1 < len(articles) else ""
                chunks.append((title + " " + body).strip())

            return chunks

        # Split into chunks based on "المادة"
        chunks = split_articles(content)

        for chunk in chunks:
            embedding = embedding_service.embed_text(chunk)
            # Store as BLOB (float32)
            doc = Document(
                content=chunk,
                source=source,
                embedding=embedding.astype(np.float32).tobytes(),
            )
            self.db.add(doc)

        self.db.commit()
        return len(chunks)
