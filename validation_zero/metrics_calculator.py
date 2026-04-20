"""
Calcolo metriche di validazione zero-knowledge
"""
import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .llm_judges import (
    get_claim_extraction_prompt,
    get_claim_verification_prompt,
    get_self_contained_prompt,
    get_chunk_relevance_prompt,
    get_answer_relevance_prompt,
    get_citation_extraction_prompt
)


class MetricsCalculator:
    """Calcola metriche di validazione senza ground truth"""
    
    def __init__(
        self, 
        openai_api_key: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.embedder = SentenceTransformer(embedding_model)
        
    def _call_llm_judge(self, prompt: str, max_tokens: int = 500) -> str:
        """Chiama LLM per giudizio"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Sei un valutatore oggettivo di risposte di chatbot. Segui esattamente le istruzioni fornite."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Errore chiamata LLM: {e}")
            return ""
    
    def calculate_groundedness(
        self, 
        response: str, 
        retrieved_chunks: List[str]
    ) -> Tuple[float, Dict]:
        """
        Calcola groundedness: % claim supportati dai documenti
        
        Returns:
            (score, details) dove details contiene claim e verifiche
        """
        # Step 1: Estrai claim
        claim_prompt = get_claim_extraction_prompt(response)
        claim_text = self._call_llm_judge(claim_prompt, max_tokens=1000)
        
        # Parse claim (formato: "1. claim\n2. claim\n...")
        claims = []
        for line in claim_text.split('\n'):
            line = line.strip()
            # Rimuovi numerazione (1., 2., -, *, etc)
            claim = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            if claim and len(claim) > 10:  # Ignora linee troppo corte
                claims.append(claim)
        
        if not claims:
            return 1.0, {"claims": [], "error": "Nessun claim estratto"}
        
        # Step 2: Verifica ogni claim
        context = "\n\n---\n\n".join(retrieved_chunks)
        claim_verifications = []
        
        for claim in claims:
            verify_prompt = get_claim_verification_prompt(claim, context)
            verification = self._call_llm_judge(verify_prompt, max_tokens=50)
            
            # Parse risposta (cerca SUPPORTATO/PARZIALMENTE/NON)
            if "SUPPORTATO" in verification and "NON" not in verification and "PARZIALMENTE" not in verification:
                score = 1.0
                status = "SUPPORTATO"
            elif "PARZIALMENTE" in verification:
                score = 0.5
                status = "PARZIALMENTE_SUPPORTATO"
            else:
                score = 0.0
                status = "NON_SUPPORTATO"
            
            claim_verifications.append({
                "claim": claim,
                "status": status,
                "score": score
            })
        
        # Step 3: Calcola score finale
        total_score = sum(v["score"] for v in claim_verifications)
        groundedness_score = total_score / len(claims)
        
        details = {
            "claims": claim_verifications,
            "num_claims": len(claims),
            "supported": sum(1 for v in claim_verifications if v["status"] == "SUPPORTATO"),
            "partial": sum(1 for v in claim_verifications if v["status"] == "PARZIALMENTE_SUPPORTATO"),
            "unsupported": sum(1 for v in claim_verifications if v["status"] == "NON_SUPPORTATO")
        }
        
        return groundedness_score, details
    
    def calculate_self_contained(
        self, 
        question: str, 
        response: str
    ) -> Tuple[float, str]:
        """
        Calcola self-contained score (1-5)
        
        Returns:
            (score, motivation)
        """
        prompt = get_self_contained_prompt(question, response)
        judgment = self._call_llm_judge(prompt, max_tokens=200)
        
        # Parse risposta (formato: "SCORE: X\nMOTIVAZIONE: ...")
        score_match = re.search(r'SCORE:\s*(\d)', judgment)
        motivation_match = re.search(r'MOTIVAZIONE:\s*(.+)', judgment, re.DOTALL)
        
        score = int(score_match.group(1)) if score_match else 3
        motivation = motivation_match.group(1).strip() if motivation_match else judgment
        
        # Normalizza su scala 0-1
        normalized_score = score / 5.0
        
        return normalized_score, motivation
    
    def calculate_relevance_embedding(
        self, 
        question: str, 
        response: str
    ) -> float:
        """
        Calcola relevance usando cosine similarity tra embeddings
        """
        emb_question = self.embedder.encode(question, convert_to_tensor=False)
        emb_response = self.embedder.encode(response, convert_to_tensor=False)
        
        # Cosine similarity
        similarity = np.dot(emb_question, emb_response) / (
            np.linalg.norm(emb_question) * np.linalg.norm(emb_response)
        )
        
        return float(similarity)
    
    def calculate_relevance_llm(
        self, 
        question: str, 
        response: str
    ) -> Tuple[float, str]:
        """
        Calcola relevance usando LLM-as-judge (alternativa a embeddings)
        """
        prompt = get_answer_relevance_prompt(question, response)
        judgment = self._call_llm_judge(prompt, max_tokens=50)
        
        if "RILEVANTE" in judgment and "NON" not in judgment and "PARZIALMENTE" not in judgment:
            score = 1.0
            status = "RILEVANTE"
        elif "PARZIALMENTE" in judgment:
            score = 0.5
            status = "PARZIALMENTE_RILEVANTE"
        else:
            score = 0.0
            status = "NON_RILEVANTE"
        
        return score, status
    
    def calculate_consistency(
        self, 
        responses: List[str]
    ) -> Tuple[float, Dict]:
        """
        Calcola consistency: similarità tra multiple risposte alla stessa domanda
        
        Args:
            responses: Lista di N risposte (tipicamente N=3)
        
        Returns:
            (consistency_score, details)
        """
        if len(responses) < 2:
            return 1.0, {"error": "Servono almeno 2 risposte"}
        
        # Encode tutte le risposte
        embeddings = self.embedder.encode(responses, convert_to_tensor=False)
        
        # Calcola pairwise similarity
        similarities = []
        pairs = []
        n = len(responses)
        
        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(float(sim))
                pairs.append((i+1, j+1))  # 1-indexed per leggibilità
        
        consistency_score = np.mean(similarities)
        
        details = {
            "num_responses": n,
            "num_pairs": len(similarities),
            "pairwise_similarities": [
                {"pair": f"R{p[0]}-R{p[1]}", "similarity": s} 
                for p, s in zip(pairs, similarities)
            ],
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities))
        }
        
        return consistency_score, details
    
    def calculate_context_precision(
        self, 
        question: str, 
        retrieved_chunks: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Calcola context precision: % chunk pertinenti
        
        Returns:
            (precision, chunk_details)
        """
        chunk_evaluations = []
        
        for idx, chunk in enumerate(retrieved_chunks):
            prompt = get_chunk_relevance_prompt(question, chunk)
            judgment = self._call_llm_judge(prompt, max_tokens=50)
            
            if "UTILE" in judgment and "PARZIALMENTE" not in judgment and "INUTILE" not in judgment:
                score = 1.0
                status = "UTILE"
            elif "PARZIALMENTE" in judgment:
                score = 0.5
                status = "PARZIALMENTE_UTILE"
            else:
                score = 0.0
                status = "INUTILE"
            
            chunk_evaluations.append({
                "chunk_index": idx,
                "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "status": status,
                "score": score
            })
        
        precision = sum(c["score"] for c in chunk_evaluations) / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        return precision, chunk_evaluations
    
    def calculate_citation_accuracy(
        self, 
        response: str, 
        chunks_metadata: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Calcola citation accuracy: % citazioni corrette
        
        Args:
            chunks_metadata: Lista di dict con metadata chunk (deve contenere 'source', 'page', etc)
        
        Returns:
            (accuracy, details)
        """
        # Estrai citazioni
        prompt = get_citation_extraction_prompt(response)
        citations_text = self._call_llm_judge(prompt, max_tokens=300)
        
        if "NESSUNA_CITAZIONE" in citations_text:
            return 1.0, {"citations": [], "message": "Nessuna citazione da verificare"}
        
        # Parse citazioni
        citations = [line.strip() for line in citations_text.split('\n') if line.strip()]
        
        if not citations:
            return 1.0, {"citations": [], "message": "Nessuna citazione trovata"}
        
        # Estrai metadata disponibili (pagine, sezioni, documenti)
        available_pages = set()
        available_sources = set()
        
        for meta in chunks_metadata:
            if 'page' in meta:
                available_pages.add(str(meta['page']))
            if 'source' in meta:
                available_sources.add(meta['source'])
        
        # Verifica citazioni
        citation_checks = []
        for citation in citations:
            found = False
            citation_lower = citation.lower()
            
            # Check pagine
            for page in available_pages:
                if page in citation_lower or f"pagina {page}" in citation_lower or f"pag. {page}" in citation_lower:
                    found = True
                    break
            
            # Check fonti
            if not found:
                for source in available_sources:
                    if source.lower() in citation_lower:
                        found = True
                        break
            
            citation_checks.append({
                "citation": citation,
                "found": found
            })
        
        correct = sum(1 for c in citation_checks if c["found"])
        accuracy = correct / len(citations) if citations else 1.0
        
        details = {
            "citations": citation_checks,
            "num_citations": len(citations),
            "correct": correct,
            "incorrect": len(citations) - correct
        }
        
        return accuracy, details