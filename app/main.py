import os

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.db.database import Base, engine

# Create tables on startup for simplicity in prototype
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Algerian Law RAG API", version="1.0.0")

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Algerian Law RAG API is running"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
