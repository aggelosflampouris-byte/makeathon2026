# src/rag/rag_engine.py
#
# Step 3 of the RAG Pipeline: Retriever + LLM Answer Generation
#
# Flow:
#   User Question
#       └─► embed question  (SentenceTransformer)
#       └─► query ChromaDB  (top-k semantic search)
#       └─► build strict grounding prompt
#       └─► call Gemini API (key × model matrix, same logic as qwentest.py)
#       └─► return RAGAnswer (answer text + source chunks)

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from vector_store import EmbeddingProvider, QueryResult, VectorStoreRepository


# ============================================================
# 1. RESPONSE CONTRACT
# ============================================================

class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation (role + message text)."""
    role: str          # "user" | "assistant"
    content: str


class RAGAnswer(BaseModel):
    """Structured response from the RAG engine."""
    question: str
    answer: str
    sources: List[QueryResult] = Field(default_factory=list)
    model_used: str = "unknown"
    grounded: bool = True   # False if LLM reported no relevant context found


# ============================================================
# 2. RAG ENGINE
# ============================================================

class RAGEngine:
    """
    Orchestrates the full RAG flow:
      1. Retrieve relevant chunks from the vector store.
      2. Build a strict, grounded prompt (no hallucination guardrail).
      3. Call Gemini with the key × model retry matrix.
      4. Return a structured RAGAnswer.
    """

    # Models in speed-priority order:
    # gemini-2.0-flash  → fastest with good quality (primary for chat)
    # gemini-2.5-flash  → higher reasoning but 2-3× slower (fallback for quota)
    # gemini-2.0-flash-lite → ultra-fast, lighter quality (last resort)
    CANDIDATE_MODELS = [
        "gemini-2.0-flash",       # Primary — best speed/quality balance for RAG chat
        "gemini-2.5-flash",       # Fallback 1 — high quality, slower
        "gemini-2.0-flash-lite",  # Fallback 2 — ultra-fast, quota safety net
    ]
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    STREAM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    MAX_RETRY_WAIT_SECONDS = 30

    def __init__(
        self,
        repo: VectorStoreRepository,
        gemini_api_keys: List[str],
        top_k: int = 5,  # 5 chunks: enough context, keeps prompt compact for low latency
    ):
        if not gemini_api_keys:
            raise ValueError("At least one Gemini API key is required for the RAG engine.")
        self._repo = repo
        self._keys = gemini_api_keys
        self._top_k = top_k

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def answer(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        conversation_history: Optional[List["ConversationTurn"]] = None,
    ) -> RAGAnswer:
        """
        Main entry point. Given a natural language question (and an optional
        prior conversation history), retrieves the most relevant chunks and
        generates a grounded, context-aware LLM answer.

        Args:
            question:             The user's question in any language.
            document_id:          Optional — restrict retrieval to one invoice.
            top_k:                Number of chunks to retrieve (overrides default).
            conversation_history: List of prior ConversationTurn objects (oldest first).
                                  These are injected into the prompt so that follow-up
                                  questions like "τι σημαίνει αυτό;" resolve correctly.
        """
        k = top_k or self._top_k
        history = conversation_history or []

        # Step 1: Semantic retrieval — embed the FULL conversational context for
        # better follow-up resolution (e.g. coreference across turns).
        enriched_query = self._enrich_query_with_history(question, history)
        sources = self._repo.query(
            query_text=enriched_query,
            n_results=k,
            filter_document_id=document_id,
        )

        if not sources:
            return RAGAnswer(
                question=question,
                answer="Δεν βρέθηκε κάποιο σχετικό έγγραφο στη βάση δεδομένων. "
                       "Παρακαλώ ανεβάστε πρώτα ένα τιμολόγιο.",
                sources=[],
                model_used="none",
                grounded=False,
            )

        # Step 2: Build strict grounded prompt (with conversation history)
        prompt = self._build_prompt(question, sources, history)

        # Step 3: Call Gemini
        answer_text, model_used = self._call_gemini(prompt)

        grounded = (
            "δεν βρέθηκε" not in answer_text.lower()
            and "not found" not in answer_text.lower()
        )

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            model_used=model_used,
            grounded=grounded,
        )

    # ----------------------------------------------------------
    # Internal: Prompt Engineering
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Internal: Query enrichment for better follow-up retrieval
    # ----------------------------------------------------------

    @staticmethod
    def _enrich_query_with_history(
        question: str,
        history: "List[ConversationTurn]",
        max_turns: int = 3,
    ) -> str:
        """
        Concatenates the last `max_turns` of conversation with the current
        question so the embedding step picks up on context from prior turns
        (e.g. coreferences like "αυτό" or "το προηγούμενο τιμολόγιο").
        """
        if not history:
            return question
        # Take the last max_turns turns (both user and assistant contribute)
        recent = history[-(max_turns * 2):]
        context_lines = " ".join(t.content for t in recent)
        return f"{context_lines} {question}"

    # ----------------------------------------------------------
    # Internal: Prompt Engineering
    # ----------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        sources: List[QueryResult],
        history: Optional["List[ConversationTurn]"] = None,
        max_history_turns: int = 6,
    ) -> str:
        """
        Builds the strict grounding prompt with an optional conversation history
        section. History is included so the LLM can resolve follow-up references
        without hallucinating data not present in the retrieved chunks.
        """
        # Format retrieved chunks as numbered context blocks
        context_blocks = "\n".join(
            f"[{i + 1}] (Τύπος: {r.chunk_type}, Απόσταση: {r.distance:.2f})\n{r.text}"
            for i, r in enumerate(sources)
        )

        # Build conversation history section (oldest first, capped)
        history_section = ""
        if history:
            recent_turns = history[-max_history_turns:]
            formatted_turns = []
            for turn in recent_turns:
                role_label = "Χρήστης" if turn.role == "user" else "Βοηθός"
                formatted_turns.append(f"{role_label}: {turn.content}")
            history_section = (
                "## Ιστορικό Συνομιλίας (χρησιμοποίησέ το μόνο για αποσαφήνιση αναφορών):\n"
                + "\n".join(formatted_turns)
                + "\n\n"
            )

        prompt = f"""Είσαι ένας αυστηρός οικονομικός βοηθός ανάλυσης τιμολογίων.

## Κανόνες (ΜΗΝ παραβείς αυτούς):
1. Απαντάς ΜΟΝΟ με βάση τα αποσπάσματα που σου δίνονται παρακάτω.
2. ΔΕΝ χρησιμοποιείς γνώσεις εκτός του παρακάτω κειμένου.
3. Αν η απάντηση δεν υπάρχει στα αποσπάσματα, απάντησε ακριβώς: "Δεν βρέθηκε".
4. Να είσαι συνοπτικός και άμεσος. Μην επαναλαμβάνεις την ερώτηση.
5. Αν υπάρχουν αριθμητικές τιμές, παρέχε τις ακριβώς όπως εμφανίζονται.
6. Μπορείς να χρησιμοποιήσεις το Ιστορικό Συνομιλίας για να αποσαφηνίσεις αναφορές
   (π.χ. «αυτό», «η προηγούμενη ερώτηση»), αλλά ΔΕΝ αντλείς νέες πληροφορίες από αυτό.
7. ΠΑΝΤΑ στο τέλος της απάντησής σου πρόσθεσε μία σύντομη, σχετική follow-up ερώτηση
   προς τον χρήστη, βασισμένη στα δεδομένα του εγγράφου. Χρησιμοποίησε το εικονίδιο 💬
   και πλάγια γραφή. Παράδειγμα: 💬 *Θέλετε να δείτε την ανάλυση ανά γραμμή;*
   Μην επαναλαμβάνεις ερωτήσεις που έχουν ήδη απαντηθεί στο Ιστορικό Συνομιλίας.
8. Αν απαριθμείς στοιχεία σε λίστα, ΠΑΝΤΑ αναφέρεις ΟΛΕΣ τις εγγραφές που βρίσκονται
   στα αποσπάσματα. ΔΕΝ σταματάς στη μέση λίστας. ΔΕΝ γράφεις "..." ή "κ.λπ." για
   να παραλείψεις στοιχεία — απαρίθμησέ τα όλα χωρίς εξαιρέσεις.

## Αποσπάσματα Εγγράφου:
{context_blocks}

{history_section}## Τρέχουσα Ερώτηση Χρήστη:
{question}

## Απάντηση:"""


        return prompt


    # ----------------------------------------------------------
    # Internal: Gemini API call (key × model matrix)
    # ----------------------------------------------------------

    def _call_gemini(self, prompt: str) -> tuple[str, str]:
        """
        Calls the Gemini text-only endpoint with the key × model matrix.
        Returns (answer_text, model_name_used).
        Falls back to a local error message if all combinations fail.
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                # 2048 tokens: enough for full multi-item lists + follow-up question.
                "maxOutputTokens": 2048,
            },
        }

        for key in self._keys:
            for model in self.CANDIDATE_MODELS:
                api_url = self.BASE_URL.format(model=model)
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": key,
                }
                try:
                    # 30s timeout: fast models answer in <5s; prevents hanging on slow models.
                    response = requests.post(api_url, json=payload, headers=headers, timeout=30)

                    if not response.ok:
                        if response.status_code == 429:
                            err_body = self._safe_json(response)
                            if self._is_daily_quota(err_body):
                                continue  # Daily quota exhausted — try next key/model
                            delay = self._parse_retry_delay(err_body)
                            if 0 < delay <= self.MAX_RETRY_WAIT_SECONDS:
                                time.sleep(delay + 1)
                                response = requests.post(api_url, json=payload, headers=headers, timeout=30)
                                if not response.ok:
                                    continue
                            else:
                                continue
                        else:
                            continue

                    response.raise_for_status()
                    data = response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return text, model

                except Exception:
                    continue

        # All combinations exhausted
        return (
            "⚠️ Δεν ήταν δυνατή η σύνδεση με το Gemini API. "
            "Ελέγξτε τα API keys ή δοκιμάστε αργότερα.",
            "none"
        )

    def stream_answer(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        conversation_history: Optional[List["ConversationTurn"]] = None,
    ):
        """
        Streaming variant of `answer()`. Yields (chunk_text, is_final, model_used)
        tuples so the caller can push tokens to the client as they arrive.

        Uses the Gemini :streamGenerateContent endpoint which returns NDJSON
        (newline-delimited JSON) chunks. Each chunk carries a partial text candidate.
        The caller should concatenate all chunk_texts to build the full answer.

        Yields:
            tuple[str, bool, str]  — (partial_text, is_final, model_used)
        """
        import json as _json

        k = top_k or self._top_k
        history = conversation_history or []

        # Retrieve relevant chunks
        enriched_query = self._enrich_query_with_history(question, history)
        sources = self._repo.query(
            query_text=enriched_query,
            n_results=k,
            filter_document_id=document_id,
        )

        if not sources:
            yield (
                "Δεν βρέθηκε κάποιο σχετικό έγγραφο στη βάση δεδομένων. "
                "Παρακαλώ ανεβάστε πρώτα ένα τιμολόγιο.",
                True,
                "none",
            )
            return

        prompt = self._build_prompt(question, sources, history)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
            },
        }

        for key in self._keys:
            for model in self.CANDIDATE_MODELS:
                api_url = self.STREAM_BASE_URL.format(model=model)
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": key,
                }
                try:
                    # stream=True: receive body incrementally as NDJSON chunks
                    with requests.post(
                        api_url, json=payload, headers=headers,
                        timeout=60, stream=True
                    ) as resp:
                        if not resp.ok:
                            if resp.status_code == 429:
                                err_body = self._safe_json(resp)
                                if self._is_daily_quota(err_body):
                                    continue
                            continue

                        # Gemini streaming returns an array of JSON chunks separated
                        # by newlines. We accumulate bytes until we can parse a full
                        # JSON object, then yield the text part.
                        buffer = ""
                        for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                            if not raw_chunk:
                                continue
                            buffer += raw_chunk
                            # Each Gemini streaming chunk is a JSON object followed by
                            # a newline (NDJSON). Try to parse as many complete objects
                            # as possible from the buffer.
                            while True:
                                try:
                                    # Strip leading '[', ',', whitespace (Gemini wraps in array)
                                    stripped = buffer.lstrip('[, \n\r')
                                    if not stripped or stripped == ']':
                                        buffer = ""
                                        break
                                    obj, idx = _json.JSONDecoder().raw_decode(stripped)
                                    buffer = stripped[idx:]
                                    try:
                                        part_text = (
                                            obj["candidates"][0]["content"]["parts"][0]["text"]
                                        )
                                        if part_text:
                                            yield part_text, False, model
                                    except (KeyError, IndexError):
                                        pass  # Safety/error candidates — skip
                                except ValueError:
                                    break  # Need more data in buffer

                        # Signal end of stream for this successful model
                        yield "", True, model
                        return  # Streaming complete — exit all loops

                except Exception:
                    continue  # Try next model/key

        # All combinations exhausted without streaming
        yield (
            "⚠️ Δεν ήταν δυνατή η σύνδεση με το Gemini API. "
            "Ελέγξτε τα API keys ή δοκιμάστε αργότερα.",
            True,
            "none",
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _safe_json(response: requests.Response) -> dict:
        try:
            return response.json()
        except Exception:
            return {}

    @staticmethod
    def _is_daily_quota(err_body: dict) -> bool:
        try:
            for detail in err_body.get("error", {}).get("details", []):
                for v in detail.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _parse_retry_delay(err_body: dict) -> float:
        try:
            for detail in err_body.get("error", {}).get("details", []):
                raw = detail.get("retryDelay", "")
                if raw:
                    m = re.search(r"(\d+(?:\.\d+)?)", raw)
                    if m:
                        return float(m.group(1))
        except Exception:
            pass
        return 0.0
