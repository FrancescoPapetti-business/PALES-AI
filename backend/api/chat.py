"""
backend/api/chat.py
Router API principale — pipeline Layer 8 completa.
"""
import sys
import hashlib
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.layers.layer8_application import process_request
from core.database import (
    create_session, session_exists, get_session_history,
    save_message, log_audit, hash_query
)
from backend.models.chat_request import ChatRequest, ChatResponse
from backend.api.kpi import track_interaction, read_latest_kpi

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # ========================================
        # SESSIONE
        # ========================================
        session_id = request.session_id

        if session_id and session_exists(session_id):
            # Carica history dal DB (ultime 10 coppie)
            db_history = get_session_history(session_id, limit=10)
            # Merge: DB history + eventuale history dal client
            chat_history = db_history
        else:
            # Crea nuova sessione
            session_id = create_session(user_role="operatore")
            chat_history = request.chat_history or []

        # Salva messaggio utente
        save_message(session_id, "user", request.question)

        # ========================================
        # PIPELINE RAG
        # ========================================
        result = process_request(
            question=request.question,
            chat_history=chat_history
        )

        status = "blocked" if result.get("error") else "success"

        # Salva risposta assistente
        save_message(
            session_id, "assistant", result["answer"],
            safety_flags=result.get("safety_flags"),
            answer_mode=result.get("answer_mode"),
            latency=result.get("latency")
        )

        # ========================================
        # AUDIT LOG (DB) + KPI (legacy, background)
        # ========================================
        background_tasks.add_task(
            log_audit,
            endpoint="/chat",
            status=status,
            latency=result.get("latency", 0),
            query_hash=hash_query(request.question),
            answer_mode=result.get("answer_mode"),
            safety_flags=result.get("safety_flags", {}),
            docs_found=len(result.get("sources", [])),
            bm25_active=result.get("bm25_active", False)
        )

        background_tasks.add_task(
            track_interaction,
            endpoint="/chat",
            status=status,
            latency=result.get("latency", 0),
            user_role="utente",
            metadata={
                "user_query": request.question,
                "docs_found": len(result.get("sources", [])),
                "safety_flags": result.get("safety_flags", {}),
                "answer_mode": result.get("answer_mode", "unknown"),
            }
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            latency=result.get("latency", 0.0),
            safety_flags=result.get("safety_flags", {}),
            session_id=session_id
        )

    except Exception as e:
        background_tasks.add_task(
            track_interaction,
            "/chat", "crash", 0, "utente",
            metadata={"error": str(e), "user_query": request.question}
        )
        raise HTTPException(status_code=500, detail=f"Errore Interno Server: {str(e)}")


@router.get("/kpi", tags=["Monitoring"])
def get_kpi_stats():
    return {
        "status": "ok",
        "latest_metrics": read_latest_kpi(limit=50)
    }


@router.get("/stats", tags=["Monitoring"])
def get_stats():
    """Statistiche aggregate dal database audit log."""
    from core.database import get_audit_stats, get_chunk_count
    stats = get_audit_stats(days=7)
    stats["chunks_in_db"] = get_chunk_count()
    return stats
