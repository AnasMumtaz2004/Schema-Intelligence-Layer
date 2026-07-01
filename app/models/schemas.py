"""
Pydantic schemas for request/response models and internal data structures.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ColumnInfo(BaseModel):
    """Schema for individual column metadata."""
    name: str
    data_type: str
    description: str = ""
    sample_values: list[str] = []


class LLMClassification(BaseModel):
    """Schema for LLM classification output."""
    business_domain: str = "Other"
    sub_domain: str = "General"
    dataset_summary: str = ""
    confidence: Optional[float] = None
    reason: Optional[str] = None


class DatasetMetadata(BaseModel):
    """Complete metadata record for a dataset."""
    dataset_id: str
    dataset_name: str
    file_type: str
    upload_timestamp: str
    database_path: str = ""
    business_domain: str = "Pending"
    sub_domain: str = "General"
    dataset_summary: str = ""
    row_count: int
    column_count: int
    column_names: list[str]
    column_data_types: list[str]
    column_descriptions: dict[str, str] = {}
    sample_data: list[dict] = []
    processing_status: str = "Processing"


class UploadResponse(BaseModel):
    """API response returned after successful dataset upload."""
    dataset_id: str
    dataset_name: str
    database_path: str
    business_domain: str
    sub_domain: str
    dataset_summary: str
    row_count: int
    column_count: int
    column_descriptions: dict[str, str]
    status: str
    dataframe_records: list[dict] = Field(
        default=[],
        description="Full dataset as a list of row records (dict per row). "
                    "Intended for downstream agent consumption."
    )


class DatasetListItem(BaseModel):
    """Summary item for dataset listing endpoint."""
    dataset_id: str
    dataset_name: str
    database_path: str
    business_domain: str
    sub_domain: str
    row_count: int
    column_count: int
    upload_timestamp: str
    processing_status: str


class ErrorResponse(BaseModel):
    """Structured error response."""
    detail: str
    error_code: Optional[str] = None
