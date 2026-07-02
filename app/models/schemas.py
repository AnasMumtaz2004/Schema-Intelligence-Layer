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


class MVASuitability(BaseModel):
    """Schema for LLM MVA suitability evaluation output."""
    mva_suitability_score: int = Field(..., description="Overall suitability score from 0 to 100")
    structural_consistency_score: int = Field(..., description="Structural consistency score from 0 to 100")
    structural_consistency_explanation: str = Field(..., description="Explanation of structural consistency")
    numerical_variable_density_score: int = Field(..., description="Numerical variable density score from 0 to 100")
    missing_data_risk: str = Field(..., description="Risk level (Low, Medium, High)")
    mva_techniques: list[str] = Field(default=[], description="Recommended MVA techniques")
    suitability_reasoning: str = Field(..., description="Detailed reasoning explaining the scores and recommendations")


class DatasetMetadata(BaseModel):
    """Complete metadata record for a dataset."""
    dataset_id: str
    dataset_name: str
    file_type: str
    upload_timestamp: str
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
    mva_suitability: Optional[MVASuitability] = None
    score: Optional[int] = None


class UploadResponse(BaseModel):
    """API response returned after successful dataset upload."""
    dataset_id: str
    dataset_name: str
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
    mva_suitability: Optional[MVASuitability] = None
    score: Optional[int] = None


class DatasetListItem(BaseModel):
    """Summary item for dataset listing endpoint."""
    dataset_id: str
    dataset_name: str
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
