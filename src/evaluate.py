# evaluate.py

import os
from PIL import Image
import json

# Import the tools from the existing project
from qwentest import OCRExtractor, InvoiceStructure
# Need to handle the case where Tesseract is not found
from pytesseract import TesseractNotFoundError

# Import the dataset loading library from your snippet
from datasets import load_dataset

# --- Configuration ---
# How many samples from the dataset to evaluate (set to a small number for a quick test)
NUM_SAMPLES_TO_EVALUATE = 10
# Which split of the dataset to use ('validation' or 'test' is best for evaluation)
DATASET_SPLIT = 'validation'

# --- Helper Functions ---

def parse_cord_ground_truth(ground_truth_str: str) -> dict:
    """
    Parses the CORD dataset's ground truth JSON and extracts key fields
    into a simplified dictionary that we can compare against.

    CORD format is complex. We will focus on extracting the total price
    and the supplier name for this simple evaluation.
    """
    try:
        truth_data = json.loads(ground_truth_str)
    except json.JSONDecodeError:
        return {"total_amount": None, "supplier_name": None}

    simplified_truth = {
        "total_amount": None,
        "supplier_name": None
    }

    for line in truth_data.get("valid_line", []):
        category = line.get("category")
        # The text is in the first "words" entry of a line
        text = line.get("words", [{}])[0].get("text", "")

        if category == "total.total_price":
            # Clean up the value to be numeric-like
            cleaned_value = "".join(c for c in text if c.isdigit() or c in ".,")
            simplified_truth["total_amount"] = cleaned_value
        elif category == "store.nm":  # CORD uses 'store.nm' for store name
            simplified_truth["supplier_name"] = text

    return simplified_truth

def compare_results(extracted: InvoiceStructure, ground_truth: dict) -> dict:
    """
    Compares the extracted data with the ground truth and returns a
    dictionary of comparison results (True for match, False for mismatch).
    """
    scores = {"total_amount_match": False, "supplier_name_match": False}

    # Compare total_amount
    extracted_total = extracted.total_amount.value if extracted.total_amount else None
    if extracted_total and ground_truth["total_amount"]:
        # Lenient check: see if the ground truth number is inside the extracted one
        cleaned_extracted = "".join(c for c in extracted_total if c.isdigit() or c in ".,")
        if ground_truth["total_amount"] in cleaned_extracted:
            scores["total_amount_match"] = True

    # Compare supplier_name
    extracted_supplier = extracted.supplier_name.value if extracted.supplier_name else None
    if extracted_supplier and ground_truth["supplier_name"]:
        # Lenient check: case-insensitive substring search
        if ground_truth["supplier_name"].lower() in extracted_supplier.lower():
            scores["supplier_name_match"] = True

    return scores

# --- Main Evaluation Logic ---

def main():
    """Main function to run the evaluation pipeline."""
    print("--- Starting Invoice Extraction Evaluation ---")

    try:
        print("Initializing OCRExtractor...")
        extractor = OCRExtractor()
    except (TesseractNotFoundError, RuntimeError) as e:
        print(f"\n[FATAL] Could not initialize OCRExtractor. Please ensure Tesseract is installed and configured.\n{e}")
        return

    print(f"Loading dataset 'naver-clova-ix/cord-v2' (split: {DATASET_SPLIT})...")
    try:
        # Use streaming=True to avoid downloading the whole dataset at once
        ds = load_dataset("naver-clova-ix/cord-v2", split=DATASET_SPLIT, streaming=True)
        dataset_subset = ds.take(NUM_SAMPLES_TO_EVALUATE)
    except Exception as e:
        print(f"\n[FATAL] Failed to load dataset from Hugging Face. Check your internet connection.\n{e}")
        return

    print(f"Evaluating on {NUM_SAMPLES_TO_EVALUATE} samples...")
    total_scores = {"total_amount_match": 0, "supplier_name_match": 0}
    processed_count = 0

    for i, example in enumerate(dataset_subset):
        processed_count += 1
        image: Image.Image = example['image']
        ground_truth_str: str = example['ground_truth']
        temp_image_path = f"temp_eval_image_{i}.png"
        image.save(temp_image_path)

        try:
            extraction_result = extractor.process(temp_image_path)
            ground_truth = parse_cord_ground_truth(ground_truth_str)
            comparison = compare_results(extraction_result.extracted_fields, ground_truth)

            if comparison["total_amount_match"]: total_scores["total_amount_match"] += 1
            if comparison["supplier_name_match"]: total_scores["supplier_name_match"] += 1
        except Exception as e:
            print(f"\n[!] Error processing sample {i}: {e}")
        finally:
            if os.path.exists(temp_image_path): os.remove(temp_image_path)

    print("\n--- Evaluation Complete ---")
    if processed_count == 0:
        print("No samples were processed.")
        return

    total_amount_accuracy = (total_scores["total_amount_match"] / processed_count) * 100
    supplier_name_accuracy = (total_scores["supplier_name_match"] / processed_count) * 100

    print(f"\nAccuracy for 'Total Amount': {total_amount_accuracy:.2f}%")
    print(f"Accuracy for 'Supplier Name': {supplier_name_accuracy:.2f}%")
    print("\nNote: 'Accuracy' here is a lenient measure, checking if the ground truth text was found within the extracted text.")

if __name__ == "__main__":
    try:
        import datasets
    except ImportError:
        print("Evaluation script requires the 'datasets' package.\nPlease install it by running: pip install datasets")
    else:
        main()