"""
Local test script — runs the full Dataset Identification Agent pipeline
directly from the command line, no server required.

Usage:
    python test_local.py                          # Tests all files in test_data/
    python test_local.py test_data/banking_variance_data.csv   # Test a single file
"""

import sys
import os
import json
import logging
import pandas as pd
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings
from app.services.database import init_db, get_next_dataset_id, insert_metadata, update_metadata_after_classification, get_metadata
from app.services.metadata_extractor import extract_metadata
from app.services.llm_service import generate_column_descriptions, classify_dataset
from app.datastore.registry import store_dataframe, get_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_local")


def load_file(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    ext = Path(filepath).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_file(filepath: str) -> dict:
    """
    Run the full agent pipeline on a single file:
      1. Load into DataFrame
      2. Extract metadata
      3. Generate column descriptions (LLM)
      4. Classify business domain (LLM)
      5. Store in SQLite + in-memory registry
    """
    filepath = str(Path(filepath).resolve())
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()

    print(f"\n{'='*70}")
    print(f"  Processing: {filename}")
    print(f"{'='*70}")

    # Step 1: Load
    print("\n[1/5] Loading dataset...")
    df = load_file(filepath)
    print(f"      ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 2: Extract metadata
    print("\n[2/5] Extracting metadata...")
    dataset_id = get_next_dataset_id()
    metadata = extract_metadata(df, filename, dataset_id, ext)
    insert_metadata(metadata)
    print(f"      ✓ Dataset ID: {dataset_id}")
    print(f"      ✓ File type: {metadata.file_type}")
    print(f"      ✓ Columns: {metadata.column_names}")

    # Step 3: Generate column descriptions via LLM
    print("\n[3/5] Generating column descriptions (Groq LLM)...")
    try:
        column_descriptions = generate_column_descriptions(metadata)
        metadata.column_descriptions = column_descriptions
        print(f"      ✓ Generated descriptions for {len(column_descriptions)} columns")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        column_descriptions = {col: f"Column '{col}'" for col in metadata.column_names}
        metadata.column_descriptions = column_descriptions

    # Step 4: Classify dataset via LLM
    print("\n[4/5] Classifying business domain (Groq LLM)...")
    try:
        classification = classify_dataset(metadata)
        metadata.business_domain = classification.business_domain
        metadata.dataset_summary = classification.dataset_summary
        processing_status = "Completed"
        print(f"      ✓ Domain: {classification.business_domain}")
        print(f"      ✓ Summary: {classification.dataset_summary}")
        if classification.confidence:
            print(f"      ✓ Confidence: {classification.confidence:.0%}")
        if classification.reason:
            print(f"      ✓ Reason: {classification.reason}")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        metadata.business_domain = "Other"
        metadata.dataset_summary = "Classification failed."
        processing_status = "Partial"

    # Step 5: Store results
    print("\n[5/5] Storing results...")
    metadata.processing_status = processing_status
    update_metadata_after_classification(
        dataset_id=dataset_id,
        business_domain=metadata.business_domain,
        dataset_summary=metadata.dataset_summary,
        column_descriptions=metadata.column_descriptions,
        processing_status=processing_status,
    )
    store_dataframe(dataset_id, df)
    print(f"      ✓ Metadata saved to SQLite ({settings.DATABASE_PATH})")
    print(f"      ✓ DataFrame stored in memory registry")

    # Serialize DataFrame records (full, for downstream handoff)
    dataframe_records = df.to_dict(orient="records")

    # Build result summary (mirrors UploadResponse)
    result = {
        "dataset_id": metadata.dataset_id,
        "dataset_name": metadata.dataset_name,
        "business_domain": metadata.business_domain,
        "dataset_summary": metadata.dataset_summary,
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "status": metadata.processing_status,
        "column_names": metadata.column_names,
        "column_data_types": metadata.column_data_types,
        "column_descriptions": metadata.column_descriptions,
        "dataframe_records": dataframe_records,  # full DataFrame for downstream
    }

    # Print final result (metadata only, not full records)
    printable_result = {k: v for k, v in result.items() if k != "dataframe_records"}
    print(f"\n{'─'*70}")
    print("  FINAL RESULT  (metadata + schema)")
    print(f"{'─'*70}")
    print(json.dumps(printable_result, indent=2))

    # Print DataFrame preview (first 5 rows)
    print(f"\n{'─'*70}")
    print(f"  DATAFRAME PREVIEW  (first 5 of {len(df)} rows)")
    print(f"{'─'*70}")
    print(df.head(5).to_string(index=False))
    print(f"{'─'*70}\n")

    return result


def main():
    # Initialize database
    init_db()

    # Determine which files to process
    if len(sys.argv) > 1:
        # Process specific file(s) passed as arguments
        files = sys.argv[1:]
    else:
        # Process all files in test_data/
        test_data_dir = Path(__file__).parent / "test_data"
        if not test_data_dir.exists():
            print("Error: test_data/ folder not found.")
            print("Usage: python test_local.py <file_path>")
            sys.exit(1)

        files = [
            str(f) for f in test_data_dir.iterdir()
            if f.suffix.lower() in (".csv", ".xlsx", ".xls")
        ]

        if not files:
            print("No CSV or Excel files found in test_data/")
            sys.exit(1)

    print(f"\n Dataset Identification Agent — Local Test")
    print(f"   Files to process: {len(files)}")
    print(f"   Groq Model: {settings.GROQ_MODEL}")
    print(f"   Database: {settings.DATABASE_PATH}")

    results = []
    for filepath in files:
        try:
            result = process_file(filepath)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            print(f"\n  ✗ ERROR processing {filepath}: {e}\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(results)}/{len(files)} files processed successfully")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['dataset_id']} | {r['dataset_name']:<45} | {r['business_domain']:<20} | {r['status']}")
    print()


if __name__ == "__main__":
    main()
