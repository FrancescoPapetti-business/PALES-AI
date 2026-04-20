"""
backend/models/chat_request.py
Modelli Pydantic per la validazione.
Accetta il campo 'message' inviato dal widget frontend.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    # Campo inviato dal widget frontend
    message: Optional[str] = Field(default=None, description="Testo del messaggio (dal widget)")
    # Campo usato internamente dal Layer 8
    question: Optional[str] = Field(default=None, description="Testo della domanda (API interna)")

    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Sessione per persistenza conversazione
    session_id: Optional[str] = Field(default=None, description="ID sessione (generato dal server se assente)")

    @model_validator(mode="after")
    def resolve_question(self) -> "ChatRequest":
        """
        Normalizza: 'message' e 'question' sono alias dello stesso campo.
        Il Layer 8 lavora sempre con 'question'.
        """
        if not self.question and self.message:
            self.question = self.message
        elif not self.message and self.question:
            self.message = self.question
        if not self.question:
            raise ValueError("Richiesto 'message' oppure 'question'.")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Come registro un nuovo allevamento?",
                "chat_history": []
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    latency: float
    safety_flags: Dict[str, bool] = {}
    session_id: Optional[str] = None