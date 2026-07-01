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
from app.prompts.llm_service_prompt import (
    get_column_descriptions_prompt,
    get_classify_dataset_prompt,
)

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

# Sub-domain taxonomy — LLM must pick from these for the chosen domain
SUB_DOMAINS: dict[str, list[str]] = {
    "Finance": [
        "Payments",
        "Credit Cards",
        "Banking & Accounts",
        "Loans & Mortgages",
        "Investments & Trading",
        "Insurance",
        "Accounting & Ledger",
        "Fraud Detection",
        "Risk Management",
        "Other Finance",
    ],
    "Sales": [
        "E-commerce",
        "Retail",
        "B2B Sales",
        "CRM & Pipeline",
        "Revenue Analytics",
        "Pricing",
        "Other Sales",
    ],
    "Marketing": [
        "Digital Marketing",
        "Campaign Analytics",
        "SEO/SEM",
        "Social Media",
        "Email Marketing",
        "Customer Acquisition",
        "Other Marketing",
    ],
    "Human Resources": [
        "Payroll",
        "Recruitment",
        "Employee Performance",
        "Training & Development",
        "Benefits",
        "Workforce Planning",
        "Other HR",
    ],
    "Operations": [
        "Manufacturing",
        "Quality Control",
        "Project Management",
        "IT Operations",
        "Facilities",
        "Other Operations",
    ],
    "Supply Chain": [
        "Inventory",
        "Procurement",
        "Logistics & Shipping",
        "Warehouse",
        "Demand Forecasting",
        "Vendor Management",
        "Other Supply Chain",
    ],
    "Customer Support": [
        "Ticketing & Issues",
        "Customer Feedback",
        "SLA Management",
        "Chat & Interaction Logs",
        "Other Customer Support",
    ],
    "Healthcare": [
        "Patient Records",
        "Clinical Trials",
        "Medical Billing & Claims",
        "Pharmacy",
        "Lab Results",
        "Other Healthcare",
    ],
    "Other": [
        "General",
        "Unknown",
    ],
}


def _build_sub_domains_str() -> str:
    """Format SUB_DOMAINS taxonomy as a readable string for the LLM prompt."""
    lines = []
    for domain, subs in SUB_DOMAINS.items():
        lines.append(f"  {domain}: {', '.join(subs)}")
    return "\n".join(lines)


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

    # Pre-extract up to 3 distinct non-empty sample values per column from sample_data list
    col_samples = {col: [] for col in metadata.column_names}
    if metadata.sample_data:
        for row in metadata.sample_data:
            for col in metadata.column_names:
                val = row.get(col)
                if val is not None and val != "":
                    val_str = str(val)[:50]  # Limit length of a single sample value
                    if val_str not in col_samples[col] and len(col_samples[col]) < 3:
                        col_samples[col].append(val_str)

    for i, col_name in enumerate(metadata.column_names):
        dtype = metadata.column_data_types[i] if i < len(metadata.column_data_types) else "unknown"
        samples = col_samples.get(col_name, [])
        samples_str = f", samples: {samples}" if samples else ""
        lines.append(f"  - {col_name} (type: {dtype}{samples_str})")

    return "\n".join(lines)


def generate_column_descriptions(metadata: DatasetMetadata) -> dict[str, str]:
    """
    Generate short descriptions for each column using the Groq LLM.

    """
    client = _get_groq_client()

    context = _build_metadata_context(metadata)

    prompt = get_column_descriptions_prompt(context)

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

    prompt = get_classify_dataset_prompt(context, desc_section)

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

        # Retrieve classification from LLM response (fallback to defaults if empty/missing)
        domain = result.get("business_domain", "Other")
        if not isinstance(domain, str) or not domain.strip():
            domain = "Other"

        sub_domain = result.get("sub_domain", "General")
        if not isinstance(sub_domain, str) or not sub_domain.strip():
            sub_domain = "General"

        return LLMClassification(
            business_domain=domain.strip(),
            sub_domain=sub_domain.strip(),
            dataset_summary=result.get("dataset_summary", ""),
            confidence=result.get("confidence"),
            reason=result.get("reason"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM classification response as JSON: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM response parsing failed.",
            confidence=0.0,
            reason=str(e),
        )
    except Exception as e:
        logger.error(f"Groq API error during dataset classification: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM service error.",
            confidence=0.0,
            reason=str(e),
        )
