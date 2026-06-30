"""
LLM service using Groq API.
Handles dataset classification and column description generation.
Only metadata (column names, types, sample values) is sent to the LLM — never the full dataset.
"""

import json
import logging
from groq import Groq

from app.config import settings
from app.models.schemas import DatasetMetadata, LLMClassification

logger = logging.getLogger(__name__)

# Valid business domains for classification
VALID_DOMAINS = [
    "Finance",
    "Sales",
    "Marketing",
    "Human Resources",
    "Operations",
    "Supply Chain",
    "Customer Support",
    "Healthcare",
    "Other",
]


def _get_groq_client() -> Groq:
    """Create and return a Groq client instance."""
    return Groq(api_key=settings.GROQ_API_KEY)


def _build_metadata_context(metadata: DatasetMetadata) -> str:
    """
    Build a text summary of dataset metadata for the LLM prompt.
    """
    lines = [
        f"Dataset Name: {metadata.dataset_name}",
        f"File Type: {metadata.file_type}",
        f"Number of Rows: {metadata.row_count}",
        f"Number of Columns: {metadata.column_count}",
        "",
        "Columns:",
    ]

    for i, col_name in enumerate(metadata.column_names):
        dtype = metadata.column_data_types[i] if i < len(metadata.column_data_types) else "unknown"
        lines.append(f"  - {col_name} (type: {dtype})")

    if metadata.sample_data:
        lines.append("")
        lines.append("Sample Rows (first few rows):")
        for idx, row in enumerate(metadata.sample_data[:5]):
            lines.append(f"  Row {idx + 1}: {json.dumps(row, default=str)}")

    return "\n".join(lines)


def generate_column_descriptions(metadata: DatasetMetadata) -> dict[str, str]:
    """
    Generate short descriptions for each column using the Groq LLM.

    """
    client = _get_groq_client()

    context = _build_metadata_context(metadata)

    prompt = f"""You are a data analyst. Given the following dataset metadata, generate a short, 
clear description (1 sentence) for each column. The description should explain what the column 
likely represents based on its name, data type, and sample values.

{context}

Return your response as a valid JSON object where keys are column names and values are descriptions.
Example format:
{{
    "Column1": "Description of what Column1 represents",
    "Column2": "Description of what Column2 represents"
}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis assistant. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response — handle possible markdown code fences
        if response_text.startswith("```"):
            # Strip markdown code fences
            response_text = response_text.strip("`").strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        descriptions = json.loads(response_text)

        # Ensure all columns are covered
        result = {}
        for col in metadata.column_names:
            result[col] = descriptions.get(col, f"Column '{col}' in the dataset.")

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM column descriptions as JSON: {e}")
        # Fallback: generate basic descriptions from column names
        return {col: f"Column '{col}' of type {metadata.column_data_types[i]}"
                for i, col in enumerate(metadata.column_names)}
    except Exception as e:
        logger.error(f"Groq API error during column description generation: {e}")
        return {col: f"Column '{col}' of type {metadata.column_data_types[i]}"
                for i, col in enumerate(metadata.column_names)}


def classify_dataset(metadata: DatasetMetadata) -> LLMClassification:
    """
    Classify the dataset into a business domain using the Groq LLM.
    """
    client = _get_groq_client()

    context = _build_metadata_context(metadata)

    # Include column descriptions if available
    desc_section = ""
    if metadata.column_descriptions:
        desc_lines = ["", "Column Descriptions:"]
        for col, desc in metadata.column_descriptions.items():
            desc_lines.append(f"  - {col}: {desc}")
        desc_section = "\n".join(desc_lines)

    valid_domains_str = ", ".join(VALID_DOMAINS)

    prompt = f"""You are a business data classification expert. Given the following dataset metadata, 
classify the dataset into the most appropriate business domain and provide a brief summary.

{context}
{desc_section}

Valid business domains: {valid_domains_str}

Return your response as a valid JSON object with the following fields:
- "business_domain": one of the valid domains listed above
- "dataset_summary": a 1-2 sentence summary of what this dataset contains
- "confidence": a number between 0.0 and 1.0 indicating classification confidence
- "reason": a brief explanation of why this domain was chosen

Example:
{{
    "business_domain": "Sales",
    "dataset_summary": "Contains customer orders, products, revenue, and regional sales information.",
    "confidence": 0.92,
    "reason": "The dataset contains columns related to revenue, customers, and products, which are typical of sales data."
}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a business data classification expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_tokens=500,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response — handle possible markdown code fences
        if response_text.startswith("```"):
            response_text = response_text.strip("`").strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        result = json.loads(response_text)

        # Validate the domain is in our list
        domain = result.get("business_domain", "Other")
        if domain not in VALID_DOMAINS:
            domain = "Other"

        return LLMClassification(
            business_domain=domain,
            dataset_summary=result.get("dataset_summary", ""),
            confidence=result.get("confidence"),
            reason=result.get("reason"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM classification response as JSON: {e}")
        return LLMClassification(
            business_domain="Other",
            dataset_summary="Unable to generate summary — LLM response parsing failed.",
            confidence=0.0,
            reason=str(e),
        )
    except Exception as e:
        logger.error(f"Groq API error during dataset classification: {e}")
        return LLMClassification(
            business_domain="Other",
            dataset_summary="Unable to generate summary — LLM service error.",
            confidence=0.0,
            reason=str(e),
        )
