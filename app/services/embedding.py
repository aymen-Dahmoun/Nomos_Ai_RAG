import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            # Load model only once
            cls._model = SentenceTransformer(settings.MODEL_NAME)
        return cls._instance

    def embed_text(self, text: str) -> np.ndarray:
        embedding = self._model.encode(text)
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


embedding_service = EmbeddingService()
