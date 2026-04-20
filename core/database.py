"""
Database SQLite per PALES-AI
Gestisce: chunks, sessioni, messaggi, audit log, versioni dataset.
"""

import sqlite3
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Path del database
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "palesai.db"


@contextmanager
def get_connection():
    """Context manager per connessioni SQLite thread-safe."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Crea le tabelle se non esistono."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
            -- Chunks della knowledge base
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_doc_id TEXT NOT NULL,
                text_content TEXT NOT NULL,
                canonical_concepts TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Full-text search su chunks (sostituisce BM25/pickle)
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text_content,
                tokenize='unicode61'
            );

            -- Sessioni utente
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                user_role TEXT DEFAULT 'operatore',
                ip_hash TEXT,
                last_activity TEXT DEFAULT (datetime('now'))
            );

            -- Messaggi della conversazione
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT (datetime('now')),
                safety_flags TEXT DEFAULT '{}',
                answer_mode TEXT,
                latency REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Audit log per tracciabilita'
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'blocked', 'error')),
                latency REAL,
                query_hash TEXT,
                answer_mode TEXT,
                safety_flags TEXT DEFAULT '{}',
                docs_found INTEGER DEFAULT 0,
                bm25_active INTEGER DEFAULT 0
            );

            -- Versioni del dataset
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_tag TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                note TEXT,
                chunk_count INTEGER DEFAULT 0
            );

            -- Indici per performance
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity);
        """)

    print(f"✅ Database inizializzato: {DB_PATH}")


# ============================
# CHUNKS
# ============================

def insert_chunks(chunks: List[Dict[str, Any]]):
    """Inserisce chunk nel DB e nell'indice FTS5."""
    with get_connection() as conn:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text_content = chunk["text_content"]
            canonical = json.dumps(chunk.get("canonical_concepts", []), ensure_ascii=False)
            metadata = json.dumps(chunk.get("metadata", {}), ensure_ascii=False)

            conn.execute("""
                INSERT OR REPLACE INTO chunks (chunk_id, parent_doc_id, text_content, canonical_concepts, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, chunk["parent_doc_id"], text_content, canonical, metadata))

            # Aggiorna FTS5
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
            conn.execute("""
                INSERT INTO chunks_fts (chunk_id, text_content)
                VALUES (?, ?)
            """, (chunk_id, text_content))


def search_chunks_fts(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Ricerca full-text sui chunks (sostituisce BM25)."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT c.chunk_id, c.text_content, c.canonical_concepts, c.metadata,
                   rank
            FROM chunks_fts fts
            JOIN chunks c ON c.chunk_id = fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()

        return [
            {
                "chunk_id": row["chunk_id"],
                "text_content": row["text_content"],
                "canonical_concepts": json.loads(row["canonical_concepts"]),
                "metadata": json.loads(row["metadata"]),
                "fts_rank": row["rank"]
            }
            for row in rows
        ]


def get_all_chunks() -> List[Dict[str, Any]]:
    """Restituisce tutti i chunks (per ricostruzione indici)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "parent_doc_id": row["parent_doc_id"],
                "text_content": row["text_content"],
                "canonical_concepts": json.loads(row["canonical_concepts"]),
                "metadata": json.loads(row["metadata"])
            }
            for row in rows
        ]


def get_chunk_count() -> int:
    """Restituisce il numero totale di chunks."""
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ============================
# SESSIONS
# ============================

def create_session(user_role: str = "operatore", ip_hash: str = None) -> str:
    """Crea una nuova sessione e restituisce il session_id."""
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, user_role, ip_hash)
            VALUES (?, ?, ?)
        """, (session_id, user_role, ip_hash))
    return session_id


def get_session_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Recupera la cronologia messaggi di una sessione."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    # Inverti per ordine cronologico
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_message(session_id: str, role: str, content: str,
                 safety_flags: Dict = None, answer_mode: str = None, latency: float = None):
    """Salva un messaggio nella sessione."""
    with get_connection() as conn:
        # Aggiorna last_activity della sessione
        conn.execute("""
            UPDATE sessions SET last_activity = datetime('now')
            WHERE session_id = ?
        """, (session_id,))

        conn.execute("""
            INSERT INTO messages (session_id, role, content, safety_flags, answer_mode, latency)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id, role, content,
            json.dumps(safety_flags or {}, ensure_ascii=False),
            answer_mode, latency
        ))


def session_exists(session_id: str) -> bool:
    """Verifica se una sessione esiste."""
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row is not None


# ============================
# AUDIT LOG
# ============================

def log_audit(endpoint: str, status: str, latency: float,
              query_hash: str = None, answer_mode: str = None,
              safety_flags: Dict = None, docs_found: int = 0,
              bm25_active: bool = False):
    """Scrive una riga di audit log."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO audit_log (endpoint, status, latency, query_hash, answer_mode,
                                   safety_flags, docs_found, bm25_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            endpoint, status, latency, query_hash, answer_mode,
            json.dumps(safety_flags or {}, ensure_ascii=False),
            docs_found, int(bm25_active)
        ))


def get_audit_stats(days: int = 7) -> Dict[str, Any]:
    """Statistiche aggregate dall'audit log."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) as blocked,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                AVG(latency) as avg_latency,
                MAX(latency) as max_latency
            FROM audit_log
            WHERE timestamp >= datetime('now', ?)
        """, (f"-{days} days",)).fetchone()

        return {
            "total": row["total"] or 0,
            "success": row["success"] or 0,
            "blocked": row["blocked"] or 0,
            "errors": row["errors"] or 0,
            "avg_latency": round(row["avg_latency"] or 0, 3),
            "max_latency": round(row["max_latency"] or 0, 3)
        }


# ============================
# DATASET VERSIONS
# ============================

def save_dataset_version(version_tag: str, fingerprint: str,
                         note: str = None, chunk_count: int = 0):
    """Registra una versione del dataset."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO dataset_versions (version_tag, fingerprint, note, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (version_tag, fingerprint, note, chunk_count))


def get_latest_dataset_version() -> Optional[Dict[str, Any]]:
    """Restituisce l'ultima versione del dataset."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM dataset_versions
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()

        if row:
            return dict(row)
        return None


# ============================
# UTILS
# ============================

def hash_query(query: str) -> str:
    """Genera hash della query per audit trail."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


# Auto-init al primo import
init_database()
