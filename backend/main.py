"""FastAPI Application Main Entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.config import settings
from utils.logger import logger

app = FastAPI(
    title="Small Language Model for Reliable Tool Calling API",
    version="1.0.0",
    description="Production-grade API serving 1-3B SLM with schema-constrained tool calling."
)

# CORS Middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info(f"SLM Tool Calling Backend starting in {settings.APP_ENV} mode...")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("SLM Tool Calling Backend shutting down cleanly.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)

