from fastapi import FastAPI
from app.api.routes import router
from app.db.database import engine, Base
import uvicorn

# Create tables on startup for simplicity in prototype
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Algerian Law RAG API", version="1.0.0")

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Algerian Law RAG API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
