# src/ocr_extractor/engine.py

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
import base64
import uuid
import concurrent.futures
import sys

# Note: google-genai SDK is not used — all Gemini calls are made via direct
# REST requests (the `requests` library), which requires no extra SDK packages.
import json
import os

from PIL import Image, ImageFilter
import pytesseract
import requests
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo

import fitz  # PyMuPDF is used for PDF processing

# ==========================================
# 1. IMMUTABLE SCHEMAS & CONTRACTS (Pydantic)
# ==========================================

class InvoiceField(BaseModel):
    """
    Structure for identifying a field in an invoice.
    Immutable and type-safe.
    """
    field_name: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Tuple[int, int, int, int] = Field(..., alias="bounding_box")

    model_config = ConfigDict(
        frozen=True,  # Immutability by default
        arbitrary_types_allowed=False,
        populate_by_name=True
    )

    @field_validator("value")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped_v = v.strip()
        if not stripped_v:
            raise ValueError("Value must not be empty or contain only whitespace.")
        # Boundary Truncation for memory safety
        return stripped_v[:255]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: tuple, info: ValidationInfo) -> tuple:
        if not v or len(v) != 4:
            raise ValueError("Bounding box must be a tuple of 4 values.")

        # Check for and handle normalized coordinates (floats between 0.0 and 1.0)
        if all(isinstance(coord, float) and 0.0 <= coord <= 1.0 for coord in v):
            context = info.context
            if not context or "image_width" not in context or "image_height" not in context:
                raise ValueError("Cannot process normalized bounding box without image dimensions in context.")
            
            img_width = context["image_width"]
            img_height = context["image_height"]
            
            # Convert normalized to absolute pixel coordinates
            v = (
                int(v[0] * img_width),
                int(v[1] * img_height),
                int(v[2] * img_width),
                int(v[3] * img_height),
            )
        elif v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("Invalid bounding box coordinates: x_max > x_min and y_max > y_min")
        return v


class AmountField(InvoiceField):
    """
    A specialized InvoiceField that ensures the value is a valid number.
    """
    @field_validator("value")
    @classmethod
    def value_must_be_numeric(cls, v: str) -> str:
        # This validator runs *after* the `validate_non_empty` from the parent.
        # `v` is already stripped and truncated.

        # A simple cleaner for common currency/number formats.
        cleaned_v = "".join(c for c in v if c.isdigit() or c in ".,")
        cleaned_v = cleaned_v.replace(",", ".")

        try:
            # Check if the cleaned string can be converted to a float
            float(cleaned_v)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{v}' is not a valid numeric amount.")

        return v


class InvoiceStructure(BaseModel):
    """
    The complete structure of the invoice.
    """
    supplier_name: Optional[InvoiceField] = None
    header: Optional[InvoiceField] = None
    footer: Optional[InvoiceField] = None
    line_items: List[InvoiceField] = Field(default_factory=list)
    total_amount: Optional[AmountField] = None


class OCRResult(BaseModel):
    """
    The final extraction product for the RAG Pipeline.
    """
    document_id: str
    full_text: str
    extracted_fields: InvoiceStructure
    visual_context: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id")
    @classmethod
    def sanitize_id(cls, v: str) -> str:
        # Fail-Fast on invalid UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            return str(uuid.uuid4())


# ==========================================
# 2. DEFENSIVE IMAGE HANDLING
# ==========================================

# Reduce max dimensions to a more VLM-friendly size to improve performance.
# 2048 is a good balance between detail and processing speed.
MAX_IMAGE_WIDTH = 2048
MAX_IMAGE_HEIGHT = 2048
MIN_IMAGE_SIZE_KB = 1

def is_supported_format(img: Image.Image) -> bool:
    """Checks if the image format is supported."""
    return img.format in ["PNG", "JPEG", "TIFF", "BMP"]


def validate_image(img: Image.Image) -> Image.Image:
    """
    Validates image dimensions and size. Resizes if necessary.
    """
    if img.width > MAX_IMAGE_WIDTH or img.height > MAX_IMAGE_HEIGHT:
        print(f"Warning: Image too large ({img.width}x{img.height}), resizing.")
        img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    if len(buffered.getvalue()) < MIN_IMAGE_SIZE_KB * 1024:
        raise ValueError(f"Image size is too small (< {MIN_IMAGE_SIZE_KB} KB), may be corrupt or blank.")

    return img


# ==========================================
# 3. CORE OCR ENGINE
# ==========================================

class OCRExtractor:
    """
    Encapsulates the entire OCR and layout detection process.
    """
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        gemini_api_keys: Optional[List[str]] = None,
        vlm_fallback_path: str = "sample_fallback.json",
    ):
        # --- Tesseract Path Resolution ---
        # The class is responsible for finding the Tesseract executable.
        # Priority: 1. Explicit `tesseract_cmd` argument. 2. Default Windows path.
        cmd_path = tesseract_cmd

        if not cmd_path and sys.platform == "win32":
            # List of common installation paths on Windows to check automatically.
            default_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Path for the official installer
                os.path.join(os.path.expanduser("~"), r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe") # Path for the winget installer
            ]
            for path in default_paths:
                if os.path.exists(path):
                    print(f"Info: Tesseract path not specified. Found at common path: {path}")
                    cmd_path = path
                    break

        pytesseract.pytesseract.tesseract_cmd = cmd_path

        # --- TESSDATA_PREFIX Configuration ---
        # If TESSDATA_PREFIX is not set, try to deduce it from the tesseract command path.
        # This is crucial for preventing "Error opening data file" issues, especially if a
        # stale system-level TESSDATA_PREFIX environment variable points to an old installation.
        # We will override it for this session to ensure consistency with the found executable.
        if cmd_path:
            tessdata_dir = os.path.join(os.path.dirname(cmd_path), "tessdata")
            if os.path.isdir(tessdata_dir):
                print(f"Info: Ensuring TESSDATA_PREFIX is set to derived path: {tessdata_dir}")
                os.environ["TESSDATA_PREFIX"] = tessdata_dir

        # --- Fail-Fast Tesseract Validation ---
        # Verify that Tesseract is accessible upon initialization.
        try:
            if not cmd_path:
                raise pytesseract.TesseractNotFoundError()
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            msg = ("Tesseract is not installed, not in your PATH, or it cannot find its data files.\n\n"
                   "Please follow these steps:\n"
                   "1. Install Tesseract-OCR from: https://github.com/tesseract-ocr/tessdoc\n"
                   "   - The script will try to find it automatically in common locations.\n\n"
                   "2. If installed in a custom location, you may need to set environment variables:\n"
                   "   - 'TESSERACT_CMD': The full path to 'tesseract.exe'.\n"
                   "   - 'TESSDATA_PREFIX': The path to the 'tessdata' directory (e.g., 'C:\\Program Files\\Tesseract-OCR\\tessdata').\n\n"
                   "   Example (PowerShell):\n"
                   "   $env:TESSERACT_CMD = 'C:\\path\\to\\tesseract.exe'\n"
                   "   $env:TESSDATA_PREFIX = 'C:\\path\\to\\tessdata'\n\n"
                   f"The script tried to find Tesseract but failed. The resolved path was: '{cmd_path}'")
            tesseract_error = pytesseract.TesseractNotFoundError()
            tesseract_error.args = (msg,)
            raise tesseract_error from None

        # --- Gemini VLM Configuration ---
        self.vlm_fallback_path = vlm_fallback_path
        self.gemini_api_keys = gemini_api_keys or []

        # It is a major security risk to hardcode API keys. Always use environment variables.
        if not self.gemini_api_keys:
            # If no keys are provided, VLM processing will be disabled and fall back to local data.
            print("\n[!] Warning: 'GEMINI_API_KEYS' environment variable not set. VLM will use local fallback data.")
            print("    To enable Gemini, add the following to your .env file (see .env.example):")
            print("    GEMINI_API_KEYS=your_gemini_api_key_1,your_gemini_api_key_2")

    def extract_text(self, img: Image.Image) -> Tuple[str, float]:
        """
        Performs OCR using Tesseract to get the raw text and average confidence.
        """
        processed_img = img.convert("L").filter(ImageFilter.SHARPEN)
        data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)

        texts = filter(None, data['text'])
        confidences = [int(c) for c in data['conf'] if int(c) > -1]

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_confidence

    def _load_fallback_data(self) -> InvoiceStructure:
        """Helper method to load and parse the fallback JSON file."""
        print(f"\n[!] Falling back to data from '{self.vlm_fallback_path}'.")
        try:
            with open(self.vlm_fallback_path, 'r') as f:
                fallback_data = json.load(f)
            return InvoiceStructure.model_validate(fallback_data)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as fallback_e:
            print(f"[!!] CRITICAL: Could not load or parse fallback file '{self.vlm_fallback_path}': {fallback_e}")
            return InvoiceStructure()

    def detect_layout_vlm(self, img: Image.Image) -> InvoiceStructure:
        """
        Uses Google's Gemini Vision API via direct REST calls for invoice structure recognition.
        It will try the provided API keys in order until one succeeds.
        """
        if not self.gemini_api_keys:
            return self._load_fallback_data()

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = (
            "You are an expert invoice processing assistant. Your task is to analyze the provided invoice image and extract "
            "key financial data into a structured JSON object. The JSON object must strictly conform to the provided schema.\n\n"
            "## JSON Schema:\n"
            "```json\n"
            "{\n"
            "  \"supplier_name\": {\"field_name\": \"string\", \"value\": \"string\", \"confidence\": float, \"bounding_box\": [int, int, int, int]},\n"
            "  \"header\": {\"field_name\": \"string\", \"value\": \"string\", \"confidence\": float, \"bounding_box\": [int, int, int, int]},\n"
            "  \"footer\": {\"field_name\": \"string\", \"value\": \"string\", \"confidence\": float, \"bounding_box\": [int, int, int, int]},\n"
            "  \"line_items\": [{\"field_name\": \"string\", \"value\": \"string\", \"confidence\": float, \"bounding_box\": [int, int, int, int]}],\n"
            "  \"total_amount\": {\"field_name\": \"string\", \"value\": \"string\", \"confidence\": float, \"bounding_box\": [int, int, int, int]}\n"
            "}\n"
            "```\n\n"
            "## Example Output:\n"
            "```json\n"
            "{\n"
            "  \"supplier_name\": {\"field_name\": \"Supplier Name\", \"value\": \"Example Supplier Ltd.\", \"confidence\": 0.99, \"bounding_box\": [50, 10, 200, 40]},\n"
            "  \"line_items\": [\n"
            "    {\"field_name\": \"Product A\", \"value\": \"100.00\", \"confidence\": 0.98, \"bounding_box\": [150, 150, 180, 170]},\n"
            "    {\"field_name\": \"Service B\", \"value\": \"50.00\", \"confidence\": 0.96, \"bounding_box\": [150, 200, 180, 220]}\n"
            "  ],\n"
            "  \"total_amount\": {\"field_name\": \"Total Due\", \"value\": \"150.00\", \"confidence\": 0.99, \"bounding_box\": [150, 300, 180, 320]}\n"
            "}\n"
            "```\n\n"
            "## Instructions:\n"
            "1.  **Extract Supplier Name**: Identify the supplier's or company's name and create the `supplier_name` object.\n"
            "2.  **Extract Line Items**: Locate the main table of products/services. For each row, create a JSON object for the `line_items` array. The `field_name` should be the item's description, and the `value` should be its price.\n"
            "3.  **Extract Total Amount**: Find the final total amount of the invoice and create the `total_amount` object.\n"
            "4.  **Bounding Box**: For every extracted field, provide its absolute pixel coordinates `[x_min, y_min, x_max, y_max]`.\n"
            "5.  **Confidence**: Provide your estimated confidence (0.0 to 1.0) for each extracted `value`.\n"
            "6.  **Missing Fields**: If a field is not found, you MUST omit its key from the JSON. If no line items are found, provide an empty array `[]`.\n"
            "7.  **Response**: Provide ONLY the raw JSON object as a string. Do not include any other text or explanations."
        )

        import time
        import re

        # Use the v1beta endpoint — required for responseMimeType in generationConfig.
        # Model priority list: try best model first, then quota-independent fallbacks.
        # gemini-1.5-flash is retired on free-tier keys — use 2.x models instead.
        CANDIDATE_MODELS = [
            "gemini-2.0-flash",       # Primary: stable, fast, multimodal
            "gemini-2.0-flash-lite",  # Fallback 1: separate quota bucket, cheaper
            "gemini-2.5-flash",       # Fallback 2: next-gen, independent daily quota
        ]
        BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        # Maximum seconds to wait on an RPM-based 429 before retrying.
        # Daily-quota 429s (limit: 0) are skipped immediately — waiting is pointless.
        MAX_RETRY_WAIT_SECONDS = 30

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        def _is_daily_quota_exhausted(err_body: dict) -> bool:
            """Returns True when the 429 is a daily cap (limit: 0), not an RPM burst."""
            try:
                for detail in err_body.get("error", {}).get("details", []):
                    for violation in detail.get("violations", []):
                        if "PerDay" in violation.get("quotaId", ""):
                            return True
            except Exception:
                pass
            return False

        def _parse_retry_delay(err_body: dict) -> float:
            """Extracts the suggested retryDelay in seconds from a 429 body."""
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

        # Key × model matrix: exhaust all models per key before moving to next key.
        for key_idx, key in enumerate(self.gemini_api_keys):
            for model in CANDIDATE_MODELS:
                api_url = BASE_URL.format(model=model)
                print(f"Sending image to Gemini REST API (key #{key_idx + 1}, model: {model})...")
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": key,
                    }
                    response = requests.post(api_url, json=payload, headers=headers, timeout=120)

                    if not response.ok:
                        print(f"\n[!] API Error - HTTP {response.status_code}")
                        print(f"    Error Details: {response.text}\n")

                        if response.status_code == 429:
                            try:
                                err_body = response.json()
                            except Exception:
                                err_body = {}

                            if _is_daily_quota_exhausted(err_body):
                                # Daily cap hit — waiting won't help, skip this model
                                print(f"    Daily quota exhausted for '{model}' on key #{key_idx + 1}. Skipping...")
                                continue

                            delay = _parse_retry_delay(err_body)
                            if 0 < delay <= MAX_RETRY_WAIT_SECONDS:
                                # Short RPM burst limit — wait and retry once
                                print(f"    RPM rate-limit. Waiting {delay:.1f}s then retrying...")
                                time.sleep(delay + 1)
                                response = requests.post(api_url, json=payload, headers=headers, timeout=120)
                                if not response.ok:
                                    print(f"    Retry failed (HTTP {response.status_code}). Trying next combo...")
                                    continue
                            else:
                                # Unknown 429 with no actionable delay — skip
                                continue

                    response.raise_for_status()

                    response_data = response.json()
                    response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]

                    # Strip optional markdown fences the model may add despite the MIME hint
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]

                    parsed_json = json.loads(response_text.strip())
                    validation_context = {"image_width": img.width, "image_height": img.height}

                    print(f"Successfully parsed layout data (key #{key_idx + 1}, model: {model}).")
                    return InvoiceStructure.model_validate(parsed_json, context=validation_context)

                except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"\n[!] Gemini call failed (key #{key_idx + 1}, model: {model}): {e}")

            remaining = len(self.gemini_api_keys) - key_idx - 1
            if remaining > 0:
                print(f"    All models exhausted for key #{key_idx + 1}. Trying next key ({remaining} remaining)...")
            else:
                print("    All API keys and models exhausted.")

        # All (key, model) combinations failed — load from local fallback.
        return self._load_fallback_data()

    @staticmethod
    def generate_visual_evidence(img: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """Adds Bounding Box Evidence for immediate verification."""
        return [{"id": f"ev_{i}", "bbox": bbox, "content_preview": f"Crop_{i}_of_{img.width}x{img.height}", "type": "bounding_box"} for i, bbox in enumerate(bboxes)]

    def _process_image_core(self, img: Image.Image) -> OCRResult:
        """Core processing logic for a single validated PIL Image."""
        img = validate_image(img)
        document_id = str(uuid.uuid4())

        # Step 1: Perform OCR and Layout Detection sequentially for a single image.
        raw_text, text_conf = self.extract_text(img)
        layout = self.detect_layout_vlm(img)

        # Step 2: Aggregate visual evidence
        all_bboxes = [field.bbox for field in [layout.supplier_name, layout.header, layout.footer, layout.total_amount] if field]
        all_bboxes.extend(item.bbox for item in layout.line_items)
        visual_context = self.generate_visual_evidence(img, all_bboxes)

        # Step 3: Consolidate text and metadata
        metadata = {
            "tesseract_confidence": text_conf,
            "layout_model": "gemini-2.0-flash",
            "image_resolution": (img.width, img.height)
        }

        return OCRResult(
            document_id=document_id,
            full_text=raw_text,
            extracted_fields=layout,
            visual_context=visual_context,
            processing_metadata=metadata
        )

    def process(self, file_path: str) -> OCRResult:
        """Main Entry Point for image files."""
        try:
            img = Image.open(file_path)
            if not is_supported_format(img):
                raise ValueError(f"Unsupported image format: {img.format}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")
        return self._process_image_core(img)

    def process_pdf(self, pdf_path: str) -> OCRResult:
        """
        Master Orchestrator for PDF files.
        Translates and splits parsing of each page to work in parallel.
        """
        document_id = str(uuid.uuid4())
        
        with fitz.open(pdf_path) as doc:
            if not doc:
                raise ValueError("Cannot open or read PDF file.")

            page_images = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images.append(page_img)
            
            if not page_images:
                raise ValueError("PDF is empty or could not be rendered.")

            print(f"Orchestrating parallel processing for {len(page_images)} pages...")
            
            # Use ThreadPoolExecutor to run OCR and VLM calls for each page in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks: one for OCR, one for VLM, for each page
                future_to_task = {
                    executor.submit(self.extract_text, img): (i, 'ocr')
                    for i, img in enumerate(page_images)
                }
                future_to_task.update({
                    executor.submit(self.detect_layout_vlm, img): (i, 'vlm')
                    for i, img in enumerate(page_images)
                })
                
                # Initialize results list
                page_results = [{'ocr': None, 'vlm': None} for _ in page_images]

                for future in concurrent.futures.as_completed(future_to_task):
                    page_index, task_type = future_to_task[future]
                    try:
                        result = future.result()
                        page_results[page_index][task_type] = result
                    except Exception as exc:
                        print(f'Page {page_index} generated an exception during {task_type}: {exc}')
                        page_results[page_index][task_type] = ('', 0.0) if task_type == 'ocr' else InvoiceStructure()

        # --- AGGREGATION STEP ---
        print("Aggregating parallel results...")
        full_ocr_text_parts = []
        all_page_confidences = []
        aggregated_layout = InvoiceStructure()
        all_line_items = []
        all_bboxes = []
        
        # Extract native text from the whole PDF at once
        with fitz.open(pdf_path) as doc:
            native_text = "".join(page.get_text("text") for page in doc)

        for i, result in enumerate(page_results):
            # Aggregate OCR results
            ocr_result = result.get('ocr')
            if ocr_result:
                page_text, page_conf = ocr_result
                full_ocr_text_parts.append(page_text)
                if page_conf > 0:
                    all_page_confidences.append(page_conf)
            
            # Aggregate VLM layout results
            vlm_result = result.get('vlm')
            if vlm_result:
                # For supplier name and total, we usually only want one.
                # Prefer the one from the first page, or the first one found.
                if not aggregated_layout.supplier_name and vlm_result.supplier_name:
                    aggregated_layout.supplier_name = vlm_result.supplier_name
                if not aggregated_layout.total_amount and vlm_result.total_amount:
                    aggregated_layout.total_amount = vlm_result.total_amount
                
                all_line_items.extend(vlm_result.line_items)

        # Finalize aggregation
        aggregated_layout.line_items = all_line_items
        full_ocr_text = "\n\n--- Page Break ---\n\n".join(full_ocr_text_parts)
        avg_confidence = sum(all_page_confidences) / len(all_page_confidences) if all_page_confidences else 0.0
        
        # For visual context, we can only use the first page image as a reference
        visual_context = self.generate_visual_evidence(page_images[0], all_bboxes)

        full_text = f"{full_ocr_text}\n\n--- PDF Text ---\n{native_text}"
        metadata = {
            "tesseract_confidence": avg_confidence,
            "layout_model": "gemini-2.0-flash",
            "image_resolution": (page_images[0].width, page_images[0].height),
            "page_count": len(page_images),
            "pdf_text_length": len(native_text)
        }

        return OCRResult(
            document_id=document_id,
            full_text=full_text,
            extracted_fields=aggregated_layout,
            visual_context=visual_context,
            processing_metadata=metadata
        )

# ==========================================
# 4. USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    import sys
    
    # --- CONFIGURATION VIA ENVIRONMENT VARIABLES ---
    # All sensitive values MUST come from environment variables or a .env file.
    # Never hardcode API keys in source code.
    # See .env.example for the required format.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv optional — env vars may already be set at OS level

    TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD")

    # Parse comma-separated Gemini API keys: GEMINI_API_KEYS=key1,key2
    _raw_keys = os.getenv("GEMINI_API_KEYS", "")
    GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
    if not GEMINI_API_KEYS:
        print(
            "[!] Warning: GEMINI_API_KEYS not set. VLM will use local fallback.\n"
            "    Add GEMINI_API_KEYS=your_key to a .env file (see .env.example)."
        )

    try:
        # Create extractor instance
        extractor = OCRExtractor(
            tesseract_cmd=TESSERACT_CMD_PATH,
            gemini_api_keys=GEMINI_API_KEYS,
        )
    except (RuntimeError, pytesseract.TesseractNotFoundError) as e:
        # The custom error from OCRExtractor.__init__ provides detailed instructions.
        print(f"\n[!] Initialization Error:\n{e}")
        sys.exit(1)
    
    # Open file dialog to select image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Bring the dialog to the front
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select an Invoice Image or PDF",
        filetypes=[
            ("All supported files", "*.png *.jpg *.jpeg *.tiff *.bmp *.pdf"),
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
    )
    
    # Remove topmost after selection
    root.attributes('-topmost', False)
    
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    
    try:
        print(f"\nProcessing file: {file_path}")
        if file_path.lower().endswith(".pdf"):
            result = extractor.process_pdf(file_path)
        else:
            result = extractor.process(file_path)
        
        print(f"=== Extraction Successful ===")
        print(f"Document ID: {result.document_id}")
        print(f"Full Text Length: {len(result.full_text)} chars")
        print(f"Extracted {len(result.extracted_fields.line_items)} Line Items")
        print(f"Total Amount: {result.extracted_fields.total_amount.value if result.extracted_fields.total_amount else 'N/A'}")
        print(f"Visual Evidence Count: {len(result.visual_context)}")

    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except Exception as e:
        print(f"An unexpected processing error occurred: {e}")