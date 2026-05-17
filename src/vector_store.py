# src/rag/vector_store.py
#
# Step 2 of the RAG Pipeline: Vector Store & Embeddings
#
# Architecture:
#   OCRResult (from qwentest.py)
#       └─► DocumentChunker        → List[InvoiceChunk]   (semantic splitting)
#       └─► EmbeddingProvider      → List[List[float]]    (abstract interface)
#              ├─ SentenceTransformerEmbedder  (HuggingFace, free, local)
#              └─ OpenAIEmbedder              (OpenAI API, optional)
#       └─► VectorStoreRepository  → ChromaDB             (persistent local store)
#
# Usage:
#   from vector_store import VectorStoreRepository, DocumentChunker, SentenceTransformerEmbedder
#   from qwentest import OCRResult  # or pass the dict directly
#
#   embedder = SentenceTransformerEmbedder()
#   repo = VectorStoreRepository(embedder=embedder)
#   chunks = DocumentChunker.chunk(ocr_result)
#   repo.upsert(chunks)
#   results = repo.query("total amount due", n_results=5)

from __future__ import annotations

import os
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 1. DATA CONTRACTS — Chunk Schema (Pydantic)
# ============================================================

class InvoiceChunk(BaseModel):
    """
    A single retrievable unit stored in the vector store.
    Carries both the raw text for embedding and structured
    metadata for post-retrieval filtering.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str                       # Parent OCRResult.document_id
    chunk_type: str                        # "full_text" | "supplier" | "line_item" | "total" | "header" | "footer"
    text: str                              # The text that will be embedded
    metadata: Dict[str, Any] = Field(default_factory=dict)  # bbox, confidence, page, etc.

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Chunk text must not be empty.")
        # Cap at 2000 chars to keep embeddings cost-effective
        return stripped[:2000]

    @field_validator("chunk_type")
    @classmethod
    def validate_chunk_type(cls, v: str) -> str:
        allowed = {"full_text", "supplier", "line_item", "total", "header", "footer"}
        if v not in allowed:
            raise ValueError(f"chunk_type must be one of {allowed}, got '{v}'")
        return v


class QueryResult(BaseModel):
    """Structured response from a similarity search."""
    chunk_id: str
    document_id: str
    chunk_type: str
    text: str
    metadata: Dict[str, Any]
    distance: float  # Lower = more similar (L2 distance)


# ============================================================
# 2. ABSTRACT EMBEDDING PROVIDER INTERFACE
# ============================================================

class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding backends.
    Decouples the vector store from any specific embedding API,
    enabling hot-swappable providers (HuggingFace ↔ OpenAI).
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of strings into a list of float vectors.
        Must return vectors of consistent dimensionality.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for the model in use."""
        ...

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        ...


# ============================================================
# 3a. PROVIDER IMPLEMENTATION — SentenceTransformers (Free / Local)
# ============================================================

class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    Uses HuggingFace sentence-transformers for local, offline embeddings.
    Default model: all-MiniLM-L6-v2
      - 384-dimensional vectors
      - Very fast (CPU-friendly)
      - Strong multilingual performance
      - Zero API cost / no internet required at inference time

    Install: pip install sentence-transformers
    """

    # A few recommended model options ranked by quality vs speed:
    # "all-MiniLM-L6-v2"          → 384-dim  | fastest, great for English
    # "paraphrase-multilingual-MiniLM-L12-v2" → 384-dim | multilingual (Greek, English, etc.)
    # "all-mpnet-base-v2"          → 768-dim  | best quality, slower

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            ) from exc

        print(f"[Embedder] Loading '{model_name}' — first run will download the model...")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        # Determine dimension by encoding a test string
        self._dim = len(self._model.encode(["test"])[0])
        print(f"[Embedder] Model ready. Vector dimension: {self._dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch-encode texts into float vectors."""
        if not texts:
            return []
        # show_progress_bar only when batch is large enough to be noticeable
        vectors = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        return vectors.tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dim


# ============================================================
# 3b. PROVIDER IMPLEMENTATION — OpenAI (Optional, paid)
# ============================================================

class OpenAIEmbedder(EmbeddingProvider):
    """
    Uses the OpenAI Embeddings API (text-embedding-3-small by default).
    Requires OPENAI_API_KEY environment variable — never hardcode keys.

    Install: pip install openai
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed.\n"
                "Run: pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Example (PowerShell): $env:OPENAI_API_KEY='sk-...'"
            )

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

        # Dimension map for well-known OpenAI embedding models
        _dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dim = _dim_map.get(model_name, 1536)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Calls the OpenAI Embeddings endpoint in a single batched request."""
        if not texts:
            return []
        # Replace newlines — they degrade embedding quality per OpenAI docs
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(input=cleaned, model=self._model_name)
        return [item.embedding for item in response.data]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dim


# ============================================================
# 4. DOCUMENT CHUNKER — OCRResult → List[InvoiceChunk]
# ============================================================

class DocumentChunker:
    """
    Converts an OCRResult (from qwentest.py) into a list of
    semantically meaningful InvoiceChunks ready for embedding.

    Chunking strategy:
      - Structured fields (supplier, total, line items) → one chunk each.
        These carry precise bounding-box and confidence metadata for
        post-retrieval grounding and citation.
      - full_text → split into overlapping 500-char windows so long
        documents don't lose context at chunk boundaries.
    """

    FULL_TEXT_CHUNK_SIZE: int = 500   # characters per window
    FULL_TEXT_OVERLAP: int = 80       # overlap to avoid cutting mid-sentence

    @classmethod
    def chunk(cls, ocr_result: Any) -> List[InvoiceChunk]:
        """
        Main entry point. Accepts either an OCRResult Pydantic object
        or a plain dict with the same schema.
        """
        # Support both Pydantic model and raw dict (for flexibility)
        if hasattr(ocr_result, "model_dump"):
            data = ocr_result.model_dump()
        elif isinstance(ocr_result, dict):
            data = ocr_result
        else:
            raise TypeError(f"Expected OCRResult or dict, got {type(ocr_result)}")

        doc_id = data["document_id"]
        fields = data.get("extracted_fields", {})
        chunks: List[InvoiceChunk] = []

        # --- Structured field chunks ---
        _field_map = {
            "supplier_name": "supplier",
            "header":        "header",
            "footer":        "footer",
            "total_amount":  "total",
        }
        for field_key, chunk_type in _field_map.items():
            field = fields.get(field_key)
            if field and field.get("value"):
                label = field.get("field_name", field_key)
                text = f"{label}: {field['value']}"
                try:
                    chunks.append(InvoiceChunk(
                        document_id=doc_id,
                        chunk_type=chunk_type,
                        text=text,
                        metadata={
                            "field_name":  field.get("field_name"),
                            "confidence":  field.get("confidence"),
                            "bounding_box": field.get("bbox") or field.get("bounding_box"),
                        }
                    ))
                except Exception as e:
                    print(f"[Chunker] Skipping field '{field_key}': {e}")

        # --- Line item chunks ---
        for i, item in enumerate(fields.get("line_items", [])):
            if not item.get("value"):
                continue
            label = item.get("field_name", f"Line item {i + 1}")
            text = f"Line item — {label}: {item['value']}"
            try:
                chunks.append(InvoiceChunk(
                    document_id=doc_id,
                    chunk_type="line_item",
                    text=text,
                    metadata={
                        "item_index":   i,
                        "field_name":   item.get("field_name"),
                        "confidence":   item.get("confidence"),
                        "bounding_box": item.get("bbox") or item.get("bounding_box"),
                    }
                ))
            except Exception as e:
                print(f"[Chunker] Skipping line item {i}: {e}")

        # --- Full-text sliding window chunks ---
        full_text = data.get("full_text", "").strip()
        if full_text:
            windows = cls._sliding_window(
                full_text,
                cls.FULL_TEXT_CHUNK_SIZE,
                cls.FULL_TEXT_OVERLAP
            )
            for i, window in enumerate(windows):
                try:
                    chunks.append(InvoiceChunk(
                        document_id=doc_id,
                        chunk_type="full_text",
                        text=window,
                        metadata={
                            "window_index": i,
                            "total_windows": len(windows),
                        }
                    ))
                except Exception as e:
                    print(f"[Chunker] Skipping text window {i}: {e}")

        print(f"[Chunker] Produced {len(chunks)} chunks from document '{doc_id}'")
        return chunks

    @staticmethod
    def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
        """
        Splits text into overlapping character windows.
        Tries to break at sentence boundaries ('. ', '? ', '! ')
        rather than mid-word for cleaner semantic chunks.
        """
        if len(text) <= size:
            return [text]

        windows: List[str] = []
        step = size - overlap
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            window = text[start:end]

            # Prefer splitting at a sentence boundary within the last 100 chars
            if end < len(text):
                boundary = _find_sentence_boundary(window, lookback=100)
                if boundary:
                    window = window[:boundary]
                    end = start + boundary

            windows.append(window.strip())
            start += step

        return [w for w in windows if w]


def _find_sentence_boundary(text: str, lookback: int = 100) -> Optional[int]:
    """
    Returns the index of the last sentence-ending punctuation in the
    final `lookback` characters of `text`, or None if not found.
    """
    tail = text[-lookback:]
    offset = len(text) - lookback
    # Search for sentence-end markers ('. ', '? ', '! ', newline)
    matches = list(re.finditer(r"(?<=[.?!\n])\s", tail))
    if matches:
        return offset + matches[-1].end()
    return None


# ============================================================
# 5. VECTOR STORE REPOSITORY — ChromaDB
# ============================================================

class VectorStoreRepository:
    """
    Persistent vector store backed by ChromaDB.
    Wraps all ChromaDB operations behind a clean Repository interface
    so the rest of the application never imports chromadb directly.

    Data is persisted to disk at `persist_directory` — safe to restart,
    data is retained across sessions.
    """

    DEFAULT_COLLECTION = "invoice_chunks"

    def __init__(
        self,
        embedder: EmbeddingProvider,
        persist_directory: str = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self._embedder = embedder
        self._collection_name = collection_name

        # Persistent client — writes to disk automatically
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),  # No telemetry phone-home
        )

        # Get or create the collection.
        # We store raw embeddings ourselves (embedding_function=None) so the
        # provider can be swapped without rebuilding the collection schema.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},  # L2 distance for similarity search
        )
        print(
            f"[VectorStore] Collection '{collection_name}' ready. "
            f"Documents: {self._collection.count()}"
        )

    # ----------------------------------------------------------
    # Write Operations
    # ----------------------------------------------------------

    def upsert(self, chunks: List[InvoiceChunk]) -> int:
        """
        Embeds and upserts a list of InvoiceChunks into the store.
        Uses 'upsert' (not 'add') so re-processing the same document
        is idempotent — existing chunks are updated, not duplicated.

        Returns the number of chunks successfully written.
        """
        if not chunks:
            print("[VectorStore] No chunks to upsert.")
            return 0

        texts = [c.text for c in chunks]
        print(f"[VectorStore] Embedding {len(texts)} chunks...")
        try:
            embeddings = self._embedder.embed(texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

        ids        = [c.chunk_id    for c in chunks]
        documents  = texts
        metadatas  = [
            {
                "document_id": c.document_id,
                "chunk_type":  c.chunk_type,
                # ChromaDB metadata values must be str | int | float | bool
                **{k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                   for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[VectorStore] Upserted {len(chunks)} chunks. Total in store: {self._collection.count()}")
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        """Removes all chunks associated with a given document_id."""
        self._collection.delete(where={"document_id": document_id})
        print(f"[VectorStore] Deleted all chunks for document '{document_id}'.")

    # ----------------------------------------------------------
    # Read Operations
    # ----------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_chunk_type: Optional[str] = None,
        filter_document_id: Optional[str] = None,
    ) -> List[QueryResult]:
        """
        Performs a semantic similarity search against the stored chunks.

        Args:
            query_text:          Natural language query.
            n_results:           Number of top results to return.
            filter_chunk_type:   Optional — restrict to "line_item", "total", etc.
            filter_document_id:  Optional — restrict to a specific invoice document.

        Returns:
            List of QueryResult objects sorted by similarity (closest first).
        """
        # Build ChromaDB 'where' filter
        where: Optional[Dict[str, Any]] = None
        filters = {}
        if filter_chunk_type:
            filters["chunk_type"] = filter_chunk_type
        if filter_document_id:
            filters["document_id"] = filter_document_id

        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{k: v} for k, v in filters.items()]}

        # Clamp n_results to available count to avoid ChromaDB errors
        available = self._collection.count()
        if available == 0:
            print("[VectorStore] Store is empty — no results.")
            return []
        n_results = min(n_results, available)

        query_embedding = self._embedder.embed([query_text])[0]

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: List[QueryResult] = []
        for i in range(len(raw["ids"][0])):
            meta = raw["metadatas"][0][i]
            results.append(QueryResult(
                chunk_id=raw["ids"][0][i],
                document_id=meta.get("document_id", ""),
                chunk_type=meta.get("chunk_type", ""),
                text=raw["documents"][0][i],
                metadata=meta,
                distance=raw["distances"][0][i],
            ))

        return results

    def count(self) -> int:
        """Returns the total number of chunks in the collection."""
        return self._collection.count()

    def list_documents(self) -> List[str]:
        """Returns a deduplicated list of document_ids stored in the collection."""
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return list({m.get("document_id", "") for m in all_meta if m.get("document_id")})


# ============================================================
# 6. PIPELINE ENTRY POINT — ties Step 1 → Step 2 together
# ============================================================

def ingest_ocr_result(
    ocr_result: Any,
    repo: VectorStoreRepository,
) -> List[InvoiceChunk]:
    """
    High-level convenience function:
    Takes an OCRResult from Step 1 and fully ingests it into the vector store.

    Returns the list of chunks that were created and stored.
    """
    chunks = DocumentChunker.chunk(ocr_result)
    repo.upsert(chunks)
    return chunks


# ============================================================
# 7. STANDALONE DEMO — run this file directly to test
# ============================================================

if __name__ == "__main__":
    import json

    # synthetic OCRResult used as the demo document
    SYNTHETIC_INVOICE = {
        "document_id": str(uuid.uuid4()),
        "full_text": (
            "INVOICE #INV-2024-001\n"
            "Supplier: Acme Solutions S.A.\n"
            "Date: 2024-03-15\n\n"
            "Item 1 — Cloud Services: 1,200.00 EUR\n"
            "Item 2 — Support Package: 350.00 EUR\n"
            "Item 3 — Training Session: 500.00 EUR\n\n"
            "Subtotal: 2,050.00 EUR\n"
            "VAT (24%): 492.00 EUR\n"
            "TOTAL DUE: 2,542.00 EUR\n"
        ),
        "extracted_fields": {
            "supplier_name": {
                "field_name": "Supplier",
                "value": "Acme Solutions S.A.",
                "confidence": 0.98,
                "bounding_box": [50, 10, 300, 35]
            },
            "line_items": [
                {"field_name": "Cloud Services",   "value": "1200.00", "confidence": 0.97, "bounding_box": [50, 150, 300, 170]},
                {"field_name": "Support Package",  "value": "350.00",  "confidence": 0.96, "bounding_box": [50, 175, 300, 195]},
                {"field_name": "Training Session", "value": "500.00",  "confidence": 0.95, "bounding_box": [50, 200, 300, 220]},
            ],
            "total_amount": {
                "field_name": "Total Due",
                "value": "2542.00",
                "confidence": 0.99,
                "bounding_box": [50, 300, 300, 320]
            },
        },
        "visual_context": [],
        "processing_metadata": {"layout_model": "demo"},
    }

    # Try to load sample_fallback.json only if it has the full OCRResult schema
    FALLBACK_PATH = "sample_fallback.json"
    raw_invoice = SYNTHETIC_INVOICE  # default
    try:
        with open(FALLBACK_PATH, "r") as f:
            candidate = json.load(f)
        if "document_id" in candidate and "full_text" in candidate:
            raw_invoice = candidate
            print(f"[Demo] Loaded OCRResult from '{FALLBACK_PATH}'")
        else:
            print(f"[Demo] '{FALLBACK_PATH}' is an InvoiceStructure-only file — using synthetic demo data.")
    except FileNotFoundError:
        print(f"[Demo] '{FALLBACK_PATH}' not found — using synthetic demo data.")

    # --- Initialize embedder and repository ---
    embedder = SentenceTransformerEmbedder()  # Downloads model on first run (~90MB)
    repo = VectorStoreRepository(
        embedder=embedder,
        persist_directory="./chroma_db",
    )

    # --- Ingest the invoice ---
    chunks = ingest_ocr_result(raw_invoice, repo)

    print(f"\n{'='*50}")
    print(f" Ingested {len(chunks)} chunks from document")
    print(f" Total chunks in store: {repo.count()}")
    print(f"{'='*50}\n")

    # --- Run semantic queries to verify ---
    demo_queries = [
        "What is the total amount?",
        "Who is the supplier or vendor?",
        "List all line items and their prices",
    ]

    for q in demo_queries:
        print(f"Query: '{q}'")
        results = repo.query(q, n_results=3)
        for r in results:
            print(f"  [{r.chunk_type:10s}] dist={r.distance:.4f} | {r.text[:80]}")
        print()
