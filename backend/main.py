"""
backend/main.py
Entry point del server PALES-AI.
"""
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Load env
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# Importa il router dalla cartella api
from backend.api.chat import router as api_router

# ========================================
# APP
# ========================================
app = FastAPI(
    title="PALES-AI Backend",
    version="2.0.0",
    description="Agente Conversazionale per ClassyFarm"
)

# ========================================
# CORS (configurabile via env)
# ========================================
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ========================================
# RATE LIMITING (semplice, in-memory)
# ========================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter per IP: max N richieste per minuto."""

    def __init__(self, app, max_requests: int = 15, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting per health check
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Pulisci richieste scadute
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if now - t < self.window
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Troppe richieste. Riprova tra {self.window} secondi."
            )

        self.requests[client_ip].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware, max_requests=15, window_seconds=60)


# ========================================
# API KEY AUTH (opzionale, attivabile via env)
# ========================================
API_KEY = os.getenv("PALESAI_API_KEY")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Se API_KEY non e' configurata, skip auth (dev mode)
    if not API_KEY:
        return await call_next(request)

    # Skip auth per health check e docs
    if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)

    # Verifica API key
    provided_key = request.headers.get("X-API-Key")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key non valida o mancante.")

    return await call_next(request)


# ========================================
# ROUTES
# ========================================
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
def health_check():
    """Health check avanzato."""
    checks = {
        "status": "active",
        "api_key_configured": bool(API_KEY),
        "cors_origins": ALLOWED_ORIGINS,
    }

    # Verifica FAISS
    faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
    checks["faiss_index"] = "ok" if faiss_path.exists() else "missing"

    # Verifica DB
    db_path = BASE_DIR / "data" / "palesai.db"
    checks["database"] = "ok" if db_path.exists() else "missing"

    # Verifica OpenAI key
    checks["openai_key"] = "configured" if os.getenv("OPENAI_API_KEY") else "missing"

    return checks


# ========================================
# STARTUP
# ========================================
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
