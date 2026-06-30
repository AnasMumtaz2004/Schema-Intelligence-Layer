"""
SQLite database service.
Manages the dataset_catalog.db for storing dataset metadata.
"""

import json
import sqlite3
import logging
from typing import Optional

from app.config import settings
from app.models.schemas import DatasetMetadata

logger = logging.getLogger(__name__)

# SQL for creating the metadata table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dataset_metadata (
    dataset_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    upload_timestamp TEXT NOT NULL,
    business_domain TEXT DEFAULT 'Pending',
    dataset_summary TEXT DEFAULT '',
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    column_names TEXT NOT NULL,
    column_data_types TEXT NOT NULL,
    column_descriptions TEXT DEFAULT '{}',
    sample_data TEXT DEFAULT '[]',
    processing_status TEXT DEFAULT 'Processing'
);
"""

INSERT_METADATA_SQL = """
INSERT INTO dataset_metadata (
    dataset_id, dataset_name, file_type, upload_timestamp,
    business_domain, dataset_summary, row_count, column_count,
    column_names, column_data_types, column_descriptions,
    sample_data, processing_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

UPDATE_METADATA_SQL = """
UPDATE dataset_metadata
SET business_domain = ?,
    dataset_summary = ?,
    column_descriptions = ?,
    processing_status = ?
WHERE dataset_id = ?;
"""

SELECT_METADATA_SQL = """
SELECT * FROM dataset_metadata WHERE dataset_id = ?;
"""

SELECT_ALL_SQL = """
SELECT * FROM dataset_metadata ORDER BY upload_timestamp DESC;
"""

GET_NEXT_ID_SQL = """
SELECT COUNT(*) FROM dataset_metadata;
"""


def _get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    try:
        conn = _get_connection()
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {settings.DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_next_dataset_id() -> str:
    """
    Generate the next sequential dataset ID.

    Returns:
        A string like 'DS_001', 'DS_002', etc.
    """
    conn = _get_connection()
    cursor = conn.execute(GET_NEXT_ID_SQL)
    count = cursor.fetchone()[0]
    conn.close()
    return f"DS_{count + 1:03d}"


def insert_metadata(metadata: DatasetMetadata) -> None:
    """
    Insert a new dataset metadata record into the database.

    Args:
        metadata: The DatasetMetadata object to store.
    """
    conn = _get_connection()
    try:
        conn.execute(INSERT_METADATA_SQL, (
            metadata.dataset_id,
            metadata.dataset_name,
            metadata.file_type,
            metadata.upload_timestamp,
            metadata.business_domain,
            metadata.dataset_summary,
            metadata.row_count,
            metadata.column_count,
            json.dumps(metadata.column_names),
            json.dumps(metadata.column_data_types),
            json.dumps(metadata.column_descriptions),
            json.dumps(metadata.sample_data, default=str),
            metadata.processing_status,
        ))
        conn.commit()
        logger.info(f"Inserted metadata for dataset {metadata.dataset_id}")
    except Exception as e:
        logger.error(f"Failed to insert metadata for {metadata.dataset_id}: {e}")
        raise
    finally:
        conn.close()


def update_metadata_after_classification(
    dataset_id: str,
    business_domain: str,
    dataset_summary: str,
    column_descriptions: dict[str, str],
    processing_status: str = "Completed",
) -> None:
    """
    Update metadata after LLM classification completes.

    Args:
        dataset_id: The dataset identifier.
        business_domain: Classified business domain.
        dataset_summary: LLM-generated summary.
        column_descriptions: Generated column descriptions.
        processing_status: Processing status string.
    """
    conn = _get_connection()
    try:
        conn.execute(UPDATE_METADATA_SQL, (
            business_domain,
            dataset_summary,
            json.dumps(column_descriptions),
            processing_status,
            dataset_id,
        ))
        conn.commit()
        logger.info(f"Updated metadata for dataset {dataset_id}: domain={business_domain}")
    except Exception as e:
        logger.error(f"Failed to update metadata for {dataset_id}: {e}")
        raise
    finally:
        conn.close()


def _row_to_metadata(row: sqlite3.Row) -> DatasetMetadata:
    """Convert a database row to a DatasetMetadata object."""
    return DatasetMetadata(
        dataset_id=row["dataset_id"],
        dataset_name=row["dataset_name"],
        file_type=row["file_type"],
        upload_timestamp=row["upload_timestamp"],
        business_domain=row["business_domain"],
        dataset_summary=row["dataset_summary"],
        row_count=row["row_count"],
        column_count=row["column_count"],
        column_names=json.loads(row["column_names"]),
        column_data_types=json.loads(row["column_data_types"]),
        column_descriptions=json.loads(row["column_descriptions"]),
        sample_data=json.loads(row["sample_data"]),
        processing_status=row["processing_status"],
    )


def get_metadata(dataset_id: str) -> Optional[DatasetMetadata]:
    """
    Retrieve metadata for a specific dataset.

    Args:
        dataset_id: The dataset identifier.

    Returns:
        DatasetMetadata or None if not found.
    """
    conn = _get_connection()
    cursor = conn.execute(SELECT_METADATA_SQL, (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return _row_to_metadata(row)


def list_all_metadata() -> list[DatasetMetadata]:
    """
    List all dataset metadata records, ordered by upload time (newest first).
    """
    conn = _get_connection()
    cursor = conn.execute(SELECT_ALL_SQL)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_metadata(row) for row in rows]
