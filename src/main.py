from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
from pathlib import Path
import tempfile
import uuid
import random
import string

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# Load environment variables from .env (if present).
# python-dotenv is already in requirements.txt; this is a no-op if .env is absent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars may already be set at OS level

# Import the shared OCR engine and data models
from qwentest import OCRExtractor, OCRResult, pytesseract

# Import Step 2 & 3: Vector Store and RAG Engine
from vector_store import SentenceTransformerEmbedder, VectorStoreRepository, DocumentChunker, ingest_ocr_result
from rag_engine import RAGEngine, RAGAnswer, ConversationTurn

# ==========================================
# 4. FASTAPI WEB SERVER INTERFACE
# ==========================================

app = FastAPI(title="Hybrid OCR Extractor API")
app.add_middleware(
    CORSMiddleware,
    # WARNING: This is insecure for production. Restrict this to your frontend's origin.
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic model for OTP request ---
class OTPRequest(BaseModel):
    email: str

# --- CONFIGURATION VIA ENVIRONMENT VARIABLES ---
# All sensitive values are read exclusively from the runtime environment.
# Set them in the .env file (never commit .env to version control).
# See .env.example for the required variable names and format.
TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD")

# Parse comma-separated Gemini API keys from a single env var.
# Example in .env:  GEMINI_API_KEYS=key1,key2,key3
_raw_keys = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

if not GEMINI_API_KEYS:
    # Hard-fail at startup rather than silently degrading to fallback mode
    # in a production server context.
    raise EnvironmentError(
        "[FATAL] GEMINI_API_KEYS environment variable is not set or is empty.\n"
        "Create a .env file in the project root (see .env.example) and add:\n"
        "  GEMINI_API_KEYS=your_key_1,your_key_2"
    )

# Create a single, shared instance of the extractor.
try:
    extractor = OCRExtractor(
        tesseract_cmd=TESSERACT_CMD_PATH,
        gemini_api_keys=GEMINI_API_KEYS,
    )
except (RuntimeError, pytesseract.TesseractNotFoundError) as e:
    print(f"[FATAL] OCRExtractor could not be initialized.\n{e}")
    extractor = None

# --- Step 2 & 3: Shared Vector Store + RAG Engine instances ---
# Loaded once at server startup; shared across all requests (thread-safe reads).
_embedder = SentenceTransformerEmbedder()
vector_repo = VectorStoreRepository(
    embedder=_embedder,
    persist_directory="./chroma_db",
    collection_name="invoice_chunks",
)
# top_k=5: balanced retrieval — enough context without bloating the prompt
rag_engine = RAGEngine(repo=vector_repo, gemini_api_keys=GEMINI_API_KEYS, top_k=5)
print(f"[RAG] Vector store ready. Chunks in store: {vector_repo.count()}")

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_root():
    """
    Serves the frontend HTML file when the user navigates to the root URL.
    """
    # Assumes front.html is in the same directory as main.py
    return "front.html"

@app.post("/api/send-otp")
async def send_otp(request: OTPRequest):
    """
    Simulates sending an OTP. In a real app, this would use an email service.
    For this demo, it generates a code and prints it to the console,
    as the frontend expects.
    """
    otp = "".join(random.choices(string.digits, k=6))
    print("=" * 40)
    print(f"OTP generated for {request.email}: {otp}")
    print("This is a simulation. The OTP is printed here instead of being emailed.")
    print("=" * 40)
    return {"success": True, "otp": otp}

@app.post("/api/extract", response_model=OCRResult)
async def extract_invoice_endpoint(file: UploadFile = File(...)):
    if not extractor:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: OCRExtractor could not be initialized. The OCR engine (Tesseract) may be missing or misconfigured."
        )

    file_ext = Path(file.filename).suffix
    temp_path = None
    try:
        # Use tempfile for robust, secure temporary file creation.
        # We create the file, get its path, and ensure it's cleaned up in the finally block.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        if file_ext.lower() == ".pdf":
            result = extractor.process_pdf(temp_path)
        else:
            result = extractor.process(temp_path)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure the temporary file is always cleaned up if it was created.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- Pydantic request models for new endpoints ---
class IngestRequest(BaseModel):
    document_id: str
    ocr_result: Dict[str, Any]   # Full OCRResult as a dict (serialized by the frontend)

class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None  # Optional: restrict search to one document
    # Ordered list of prior turns for multi-turn conversation support.
    # Each item: {"role": "user" | "assistant", "content": "<text>"}
    conversation_history: Optional[List[ConversationTurn]] = None


@app.post("/api/ingest")
async def ingest_endpoint(request: IngestRequest):
    """
    Receives a serialized OCRResult dict (sent by the frontend after /api/extract)
    and ingests it into the ChromaDB vector store.
    This is called automatically by front.html after each successful OCR extraction.
    """
    try:
        chunks = ingest_ocr_result(request.ocr_result, vector_repo)
        return {"status": "ok", "chunks_stored": len(chunks), "document_id": request.document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@app.post("/api/query", response_model=RAGAnswer)
async def query_endpoint(request: QueryRequest):
    """
    Accepts a natural language question and an optional document_id filter.
    Performs semantic retrieval from ChromaDB and generates a grounded answer
    using Gemini (with the strict anti-hallucination prompt from rag_engine.py).
    """
    if vector_repo.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="Vector store is empty. Please upload and process an invoice first."
        )
    try:
        result = rag_engine.answer(
            question=request.question,
            document_id=request.document_id,
            conversation_history=request.conversation_history,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")


@app.post("/api/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming variant of /api/query.
    Tokens are pushed to the client as they arrive from Gemini, so the user
    sees text appearing in real-time instead of waiting for the full answer.

    SSE frame format (one per token batch):
        data: {"token": "...", "done": false, "model": ""}
    Final frame:
        data: {"token": "", "done": true, "model": "gemini-2.0-flash"}
    """
    import json as _json

    if vector_repo.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="Vector store is empty. Please upload and process an invoice first."
        )

    def event_generator():
        try:
            for token, is_final, model in rag_engine.stream_answer(
                question=request.question,
                document_id=request.document_id,
                conversation_history=request.conversation_history,
            ):
                payload = _json.dumps(
                    {"token": token, "done": is_final, "model": model},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
        except Exception as exc:
            err = _json.dumps({"token": f"⚠️ {exc}", "done": True, "model": "error"})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Prevent nginx / reverse proxies from buffering the stream
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── CSV Reconciliation Helpers ───────────────────────────────────────────────

def _parse_csv_bytes(content: bytes) -> tuple:
    """Decode CSV bytes, auto-detect delimiter, strip BOM."""
    text = content.decode('utf-8-sig')  # utf-8-sig strips Excel BOM
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=',;\t|')
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = list(reader.fieldnames or [])
    rows = [dict(r) for r in reader]
    return headers, rows


def _try_parse_amount(value: str) -> Optional[float]:
    """Extract float from currency strings like €1.234,56 / $1,234.56 / (200)."""
    if not value:
        return None
    s = re.sub(r'[\$€£¥\s]', '', str(value))
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    if re.match(r'^-?\d{1,3}(\.\d{3})+(,\d{1,2})?$', s):
        s = s.replace('.', '').replace(',', '.')
    s = s.replace(',', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _detect_amount_cols(headers: list, rows: list) -> list:
    """Return column names that are likely monetary amounts."""
    kw = re.compile(r'amount|debit|credit|total|balance|value|price|payment|charge', re.I)
    # Exclude columns that are clearly IDs, dates, or numbers
    exclude_kw = re.compile(r'invoice|id|number|no|date|phone|zip|code', re.I)
    
    result = []
    for h in headers:
        if exclude_kw.search(h):
            continue
        if kw.search(h):
            result.append(h)
            continue
        sample = rows[:20]
        if sample:
            hits = sum(1 for r in sample if _try_parse_amount(r.get(h, '')) is not None)
            if hits / len(sample) >= 0.6:
                result.append(h)
    return result


def _flatten_invoice_json(obj: dict) -> dict:
    """
    Flatten a parsed invoice JSON object into a dict of display_name → value pairs.

    Rules:
    - Navigate into an "invoice" wrapper key if present.
    - Scalar fields are extracted as-is.
    - List fields (e.g. "items") produce:
        * "Total Amount"  — numeric sum of price sub-fields (for reconciliation)
        * "Line Items"    — human-readable preview of item descriptions
    - Nested dicts are flattened one level deep.
    - Column names are converted from snake_case to Title Case.
    """
    import re as _re

    def _col_name(raw: str) -> str:
        s = _re.sub(r'([a-z])([A-Z])', r'\1 \2', raw)
        return s.replace('_', ' ').strip().title()

    # Unwrap top-level "invoice" key if present
    if isinstance(obj, dict) and 'invoice' in obj:
        obj = obj['invoice']

    result: dict = {}
    numeric_kw = _re.compile(
        r'price|amount|total|cost|value|subtotal|tax|vat|charge|fee', _re.I
    )

    # Check if a top-level total already exists so we don't duplicate
    has_top_level_total = any(
        numeric_kw.search(k) and isinstance(v, (int, float, str))
        for k, v in obj.items()
        if not isinstance(v, (list, dict))
    )

    for key, val in obj.items():
        col = _col_name(key)

        if isinstance(val, (str, int, float, bool)) or val is None:
            result[col] = val if val is not None else ''

        elif isinstance(val, list):
            count = len(val)
            computed_total = 0.0
            descriptions = []

            for item in val:
                if not isinstance(item, dict):
                    continue
                # Collect description for display
                for dk in ('description', 'name', 'product', 'item', 'detail'):
                    desc = item.get(dk, '')
                    if desc:
                        descriptions.append(str(desc).replace('\n', ' ').strip()[:60])
                        break
                # Sum price fields
                for ik, iv in item.items():
                    if numeric_kw.search(ik):
                        try:
                            computed_total += float(str(iv).replace(',', '').strip())
                        except (ValueError, TypeError):
                            pass

            # Readable Line Items summary
            if descriptions:
                preview = '; '.join(descriptions[:3])
                if count > 3:
                    preview += f' … (+{count - 3} more)'
                result['Line Items'] = f'{count} × [{preview}]'
            else:
                result['Line Items'] = f'{count} items'

            # Standalone numeric total for reconciliation
            if computed_total and not has_top_level_total:
                result['Total Amount'] = round(computed_total, 2)

        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (str, int, float, bool)):
                    result[f'{col} {_col_name(sub_key)}'] = sub_val

    return result


def _expand_json_columns(headers: list, rows: list) -> tuple:
    """
    Detect columns whose name contains 'json' (case-insensitive), parse each
    cell as JSON, and dynamically flatten ALL discovered financial/invoice fields.

    The raw JSON columns are removed from the output. All scalar fields found
    across any row are surfaced as individual columns.

    Returns:
        (new_headers: list, new_rows: list[dict])
    """
    json_cols = [h for h in headers if 'json' in h.lower()]
    if not json_cols:
        return headers, rows  # Nothing to expand

    # Columns to keep = originals minus the raw json ones
    kept_headers = [h for h in headers if h not in json_cols]

    # First pass: parse every JSON cell and collect the union of all field names
    # to build a stable column order.
    parsed_cache: list = []
    all_field_names: list = []
    seen_fields: set = set()

    for row in rows:
        flattened: dict = {}
        for jcol in json_cols:
            raw = row.get(jcol, '')
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    flattened = _flatten_invoice_json(obj)
                break
            except (json.JSONDecodeError, TypeError):
                continue
        parsed_cache.append(flattened)
        for fname in flattened:
            if fname not in seen_fields:
                seen_fields.add(fname)
                all_field_names.append(fname)

    expanded_headers = kept_headers + all_field_names

    # Second pass: build the expanded rows
    expanded_rows: List[dict] = []
    for row, flattened in zip(rows, parsed_cache):
        new_row = {k: v for k, v in row.items() if k not in json_cols}
        for fname in all_field_names:
            new_row[fname] = flattened.get(fname, '')
        expanded_rows.append(new_row)

    return expanded_headers, expanded_rows


@app.post("/api/reconcile")
async def reconcile_endpoint(
    file: UploadFile = File(...),
    invoice_data: str = Form(default="[]"),
):
    """
    Accepts a bank-statement CSV and a JSON list of invoice totals.
    Returns per-invoice PAID/UNPAID status and annotated CSV rows.
    """
    try:
        content = await file.read()
        raw_headers, raw_rows = _parse_csv_bytes(content)
        if not raw_rows:
            raise HTTPException(status_code=422, detail="CSV is empty or could not be parsed.")

        # Expand any JSON columns into flat financial fields before further processing
        headers, rows = _expand_json_columns(raw_headers, raw_rows)

        try:
            invoices: list = json.loads(invoice_data)
        except json.JSONDecodeError:
            invoices = []

        amount_cols = _detect_amount_cols(headers, rows)

        # Enrich rows with parsed amounts
        enriched: List[dict] = []
        for i, row in enumerate(rows):
            amount = None
            for col in amount_cols:
                v = _try_parse_amount(row.get(col, ''))
                if v is not None:
                    amount = v
                    break
            enriched.append({**row, "_idx": i, "_amount": amount, "_match_id": None})

        total_debits  = sum(abs(r["_amount"]) for r in enriched if r["_amount"] is not None and r["_amount"] < 0)
        total_credits = sum(r["_amount"]       for r in enriched if r["_amount"] is not None and r["_amount"] >= 0)

        TOLERANCE = 0.01
        reconciliation: List[dict] = []

        for inv in invoices:
            doc_id    = inv.get("document_id", "")
            file_name = inv.get("file_name", "Unknown")
            inv_total = _try_parse_amount(str(inv.get("total_amount", "")))

            if inv_total is None:
                reconciliation.append({"document_id": doc_id, "file_name": file_name,
                                       "invoice_total": None, "status": "NO_TOTAL",
                                       "matched_transaction": None})
                continue

            # Find first unmatched row whose absolute amount equals the invoice total
            matched = next(
                (r for r in enriched
                 if r["_amount"] is not None
                 and abs(abs(r["_amount"]) - abs(inv_total)) <= TOLERANCE
                 and r["_match_id"] is None),
                None
            )

            if matched:
                matched["_match_id"] = doc_id
                tx = {k: v for k, v in matched.items() if not k.startswith("_")}
                reconciliation.append({"document_id": doc_id, "file_name": file_name,
                                       "invoice_total": inv_total, "status": "PAID",
                                       "matched_transaction": tx})
            else:
                reconciliation.append({"document_id": doc_id, "file_name": file_name,
                                       "invoice_total": inv_total, "status": "UNPAID",
                                       "matched_transaction": None})

        matched_count = sum(1 for r in enriched if r["_match_id"] is not None)
        clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in enriched]
        match_flags = [r["_match_id"] for r in enriched]

        return {
            "reconciliation": reconciliation,
            "csv_summary": {
                "total_rows": len(rows),
                "total_debits": round(total_debits, 2),
                "total_credits": round(total_credits, 2),
                "matched_count": matched_count,
                "headers": headers,
                "amount_columns": amount_cols,
            },
            "csv_rows": clean_rows,
            "match_flags": match_flags,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {e}")


if __name__ == "__main__":
    # Changed port to 8000 to avoid conflicts
    uvicorn.run(app, host="127.0.0.1", port=8000)