import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.embedding import embedding_service


class RetrievalService:
    def __init__(self, db: Session):
        self.db = db

    def keyword_score(self, query: str, text: str):
        score = 0
        for word in query.split():
            if word in text:
                score += 1
        return score

    def get_top_k(self, query: str, k: int = 3):
        # 🔹 1. Embed & normalize query
        query_embedding = embedding_service.embed_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        # 🔹 2. Fetch documents
        documents = self.db.query(Document).all()
        if not documents:
            return []

        # 🔹 3. Decode embeddings from BLOB → numpy
        try:
            doc_embeddings = np.vstack(
                [np.frombuffer(doc.embedding, dtype=np.float32) for doc in documents]
            )
        except Exception as e:
            print("❌ Error decoding embeddings:", e)
            return []

        # 🔹 4. Normalize document embeddings
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        doc_embeddings = doc_embeddings / norms

        # 🔹 5. Compute semantic similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # 🔹 6. Combine with Keyword Signal (Hybrid Retrieval)
        HYBRID_SCORES = []
        for i, doc in enumerate(documents):
            k_score = self.keyword_score(query, doc.content)
            # Final score = semantic similarity + 0.1 * keyword_score
            final_score = float(similarities[i]) + (0.1 * k_score)
            HYBRID_SCORES.append(final_score)

        HYBRID_SCORES = np.array(HYBRID_SCORES)

        # 🔹 7. Get top-k indices (sorted high → low)
        top_k_indices = np.argsort(HYBRID_SCORES)[::-1][:k]

        # 🔹 8. Filter by similarity threshold
        SIMILARITY_THRESHOLD = 0.6

        results = []
        for idx in top_k_indices:
            score = float(HYBRID_SCORES[idx])

            if score < SIMILARITY_THRESHOLD:
                continue  # skip weak matches

            doc = documents[idx]

            results.append(
                {"content": doc.content, "source": doc.source, "score": score}
            )

        # 🔹 9. DEBUG (Print query and top RESULTS clearly)
        print("\n=== QUERY ===")
        print(query)

        print("\n=== TOP RETRIEVED RESULTS ===")
        for idx in top_k_indices[:3]:
            print(f"{HYBRID_SCORES[idx]:.4f} → {documents[idx].content[:150]}...")

        return results
