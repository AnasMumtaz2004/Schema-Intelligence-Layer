"""
API routes for dataset upload and retrieval.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import UploadResponse, DatasetMetadata, DatasetListItem
from app.services.validator import validate_upload_file, validate_dataframe
from app.services.loader import load_dataset
from app.services.metadata_extractor import extract_metadata
from app.services.llm_service import generate_column_descriptions, classify_dataset
from app.services.database import (
    get_next_dataset_id,
    insert_metadata,
    update_metadata_after_classification,
    get_metadata,
    list_all_metadata,
    get_metadata_by_name,
    update_metadata_after_append,
)
from app.datastore.registry import store_dataframe, get_dataframe
from app.services.persistence import save_to_dataset_db, read_from_dataset_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload-dataset",
    response_model=UploadResponse,
    summary="Upload a CSV or Excel dataset",
    description="Accepts a CSV or Excel file, validates it, persists/appends it in its dedicated SQLite database, and registers/classifies its metadata.",
    responses={
        400: {"description": "File unreadable or corrupted"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Empty dataset or invalid structure"},
        500: {"description": "Internal processing error"},
    },
)
async def upload_dataset(file: UploadFile = File(..., description="CSV or Excel file to upload")):
    """
    Main upload endpoint — processes the dataset through the full pipeline:
    1. Validate file type
    2. Load into DataFrame
    3. Validate DataFrame structure
    4. Check if file is already uploaded:
       - If yes: append rows to its dedicated DB, re-extract row counts, and update metadata catalog.
       - If no: save to dedicated DB, run LLM classification, and insert to metadata catalog.
    """

    # Step 1: Validate file type
    file_extension = validate_upload_file(file)
    logger.info(f"Received upload: {file.filename} (type: {file_extension})")

    # Step 2: Load dataset into DataFrame
    df = await load_dataset(file)
    logger.info(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Step 3: Validate DataFrame
    filename = file.filename or "unknown"
    validate_dataframe(df, filename)

    # Step 4: Check if file already exists
    existing = get_metadata_by_name(filename)

    if existing:
        logger.info(f"Dataset '{filename}' already exists (ID: {existing.dataset_id}). Appending rows.")
        # Append rows to target DB
        database_path = save_to_dataset_db(df, filename, mode="append")

        # Load combined dataframe to get updated counts and samples
        combined_df = read_from_dataset_db(database_path)
        new_metadata = extract_metadata(combined_df, filename, existing.dataset_id, file_extension)

        # Update metadata in catalog database
        update_metadata_after_append(
            dataset_id=existing.dataset_id,
            row_count=new_metadata.row_count,
            upload_timestamp=new_metadata.upload_timestamp,
            sample_data=new_metadata.sample_data,
            processing_status="Completed",
        )

        # Update cache registry
        store_dataframe(existing.dataset_id, combined_df)

        # Prepare response records from the combined df
        dataframe_records = combined_df.to_dict(orient="records")

        return UploadResponse(
            dataset_id=existing.dataset_id,
            dataset_name=existing.dataset_name,
            database_path=database_path,
            business_domain=existing.business_domain,
            sub_domain=existing.sub_domain,
            dataset_summary=existing.dataset_summary,
            row_count=new_metadata.row_count,
            column_count=existing.column_count,
            column_descriptions=existing.column_descriptions,
            status="Completed",
            dataframe_records=dataframe_records,
        )

    else:
        logger.info(f"Dataset '{filename}' is new. Creating metadata catalog and dedicated database.")
        # Generate new dataset ID
        dataset_id = get_next_dataset_id()

        # Save to dedicated SQLite database (mode="replace" to ensure clean table)
        database_path = save_to_dataset_db(df, filename, mode="replace")

        # Extract initial metadata
        metadata = extract_metadata(df, filename, dataset_id, file_extension)
        metadata.database_path = database_path

        # Insert initial metadata row (status = Processing)
        insert_metadata(metadata)

        # Generate column descriptions via LLM
        try:
            column_descriptions = generate_column_descriptions(metadata)
            metadata.column_descriptions = column_descriptions
            logger.info(f"Generated column descriptions for {dataset_id}")
        except Exception as e:
            logger.warning(f"Column description generation failed for {dataset_id}: {e}")
            column_descriptions = {col: f"Column '{col}'" for col in metadata.column_names}
            metadata.column_descriptions = column_descriptions

        # Classify dataset via LLM
        try:
            classification = classify_dataset(metadata)
            metadata.business_domain = classification.business_domain
            metadata.sub_domain = classification.sub_domain
            metadata.dataset_summary = classification.dataset_summary
            processing_status = "Completed"
            logger.info(
                f"Classified {dataset_id} as '{classification.business_domain}' "
                f"/ '{classification.sub_domain}'"
            )
        except Exception as e:
            logger.error(f"Dataset classification failed for {dataset_id}: {e}")
            metadata.business_domain = "Other"
            metadata.sub_domain = "General"
            metadata.dataset_summary = "Classification failed — domain could not be determined."
            processing_status = "Partial"

        # Update metadata in catalog database
        metadata.processing_status = processing_status
        update_metadata_after_classification(
            dataset_id=dataset_id,
            business_domain=metadata.business_domain,
            sub_domain=metadata.sub_domain,
            dataset_summary=metadata.dataset_summary,
            column_descriptions=metadata.column_descriptions,
            processing_status=processing_status,
        )

        # Store DataFrame in registry
        store_dataframe(dataset_id, df)

        # Serialize to records for response
        dataframe_records = df.to_dict(orient="records")

        return UploadResponse(
            dataset_id=metadata.dataset_id,
            dataset_name=metadata.dataset_name,
            database_path=database_path,
            business_domain=metadata.business_domain,
            sub_domain=metadata.sub_domain,
            dataset_summary=metadata.dataset_summary,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            column_descriptions=metadata.column_descriptions,
            status=metadata.processing_status,
            dataframe_records=dataframe_records,
        )


@router.get(
    "/datasets",
    response_model=list[DatasetListItem],
    summary="List all uploaded datasets",
    description="Returns a summary list of all datasets that have been uploaded and processed.",
)
async def list_datasets():
    """List all uploaded dataset metadata records."""
    all_metadata = list_all_metadata()
    return [
        DatasetListItem(
            dataset_id=m.dataset_id,
            dataset_name=m.dataset_name,
            database_path=m.database_path,
            business_domain=m.business_domain,
            sub_domain=m.sub_domain,
            row_count=m.row_count,
            column_count=m.column_count,
            upload_timestamp=m.upload_timestamp,
            processing_status=m.processing_status,
        )
        for m in all_metadata
    ]


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetMetadata,
    summary="Get dataset metadata",
    description="Returns the full metadata record for a specific dataset.",
    responses={404: {"description": "Dataset not found"}},
)
async def get_dataset_metadata(dataset_id: str):
    """Get full metadata for a specific dataset."""
    metadata = get_metadata(dataset_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return metadata


@router.get(
    "/datasets/{dataset_id}/dataframe",
    summary="Get dataset DataFrame as JSON",
    description="Returns the in-memory DataFrame for a dataset as JSON records. "
                "Intended for downstream agent consumption.",
    responses={404: {"description": "Dataset not found or DataFrame not in memory"}},
)
async def get_dataset_dataframe(dataset_id: str, limit: int = 100):
    """
    Get the stored DataFrame as JSON records.
    
    Args:
        dataset_id: The dataset identifier.
        limit: Maximum number of rows to return (default 100, use -1 for all).
    """
    df = get_dataframe(dataset_id)
    if df is None:
        metadata = get_metadata(dataset_id)
        if metadata and metadata.database_path:
            try:
                df = read_from_dataset_db(metadata.database_path)
                store_dataframe(dataset_id, df)
                logger.info(f"Loaded DataFrame for {dataset_id} from {metadata.database_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset from dedicated database {metadata.database_path}: {e}")

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"DataFrame for '{dataset_id}' not found. "
                   "The database file might be missing or corrupted."
        )

    if limit > 0:
        result_df = df.head(limit)
    else:
        result_df = df

    records = result_df.to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "total_rows": len(df),
        "returned_rows": len(records),
        "data": records,
    }
