"""
Prompt templates for the LLM service.
Each function returns a fully-rendered prompt string ready to be sent to the LLM.
"""


def get_column_descriptions_prompt(context: str) -> str:
    """
    Prompt for generating short column descriptions.

    Args:
        context: Dataset metadata context built by _build_metadata_context().

    Returns:
        Rendered prompt string.
    """
    return f"""You are a data analyst. Given the following dataset metadata, generate a short, \
clear description (one sentence, max ~20 words) for each column. Base the description strictly on \
the column's name, data type, and sample values provided below — do not invent meaning that isn't \
supported by this evidence.

{context}

Rules:
- Include EVERY column listed in the metadata above, and ONLY those columns. Do not skip any, and \
do not add columns that are not present in the metadata.
- Each description must be a single sentence.
- If a column's purpose is unclear from the metadata, describe it based on its name and type and \
say so plainly (e.g. "Numeric identifier, likely a foreign key; purpose unclear from sample data.") \
rather than guessing a specific business meaning.

Return your response as a single valid JSON object, and nothing else — no markdown code fences, \
no ```json wrapper, no preamble, no trailing commentary. The response must start with {{ and end \
with }}.

Format:
{{
    "Column1": "Description of what Column1 represents",
    "Column2": "Description of what Column2 represents"
}}"""


def get_classify_dataset_prompt(
    context: str,
    desc_section: str,
) -> str:
    """
    Prompt for classifying a dataset into a business domain and sub-domain.

    Args:
        context: Dataset metadata context built by _build_metadata_context().
        desc_section: Optional block of generated column descriptions.

    Returns:
        Rendered prompt string.
    """
    return f"""You are a business data classification expert. Given the following dataset metadata, \
classify the dataset into the most appropriate business domain AND sub-domain, and provide a brief summary.

{context}
{desc_section}

Analyze the column names, types, descriptions, and sample values to determine:
1. The general business domain — prefer one of: Finance, Healthcare, E-commerce, Sales, Human \
Resources, Logistics, Marketing, Operations, Education, Real Estate, Insurance, Technology, Legal, \
Government, Manufacturing. Only use a different domain name if none of these reasonably fit.
2. The specific sub-domain — a short, specific phrase (2-4 words) describing the purpose of the \
data within that domain (e.g. "Payments", "Clinical Trials", "Inventory Management", "Payroll", \
"Digital Ad Campaigns").

Base your classification only on evidence in the metadata above. If the data is ambiguous or \
generic, choose the closest reasonable domain and lower the confidence score accordingly rather \
than forcing a specific-sounding answer.

Return your response as a single valid JSON object, and nothing else — no markdown code fences, \
no ```json wrapper, no preamble, no trailing commentary. The response must start with {{ and end \
with }}. Fields:
- "business_domain": string, the general business domain name
- "sub_domain": string, the specific sub-domain within that domain
- "dataset_summary": string, 1-2 sentences describing what this dataset contains
- "confidence": a number (not a string) between 0.0 and 1.0 indicating classification confidence
- "reason": string, a brief explanation of why this domain and sub-domain were chosen, referencing \
specific column names as evidence

Example:
{{
    "business_domain": "Finance",
    "sub_domain": "Payments",
    "dataset_summary": "Contains transaction records, payment methods, and settlement data for a payments platform.",
    "confidence": 0.95,
    "reason": "Columns such as transaction_id, amount, and payment_method are typical of payments data."
}}"""


def get_mva_suitability_prompt(
    context: str,
    first_20_str: str,
    last_20_str: str,
    col_truncated_msg: str,
) -> str:
    """
    Prompt for scoring a dataset's suitability for Multivariate Analysis (MVA).
    """
    return f"""You are a senior data scientist and statistician specializing in Multivariate Analysis (MVA).
Your goal is to evaluate the suitability of the following dataset for Multivariate Analysis (e.g. PCA, Factor Analysis, Multiple Regression, MANOVA, Cluster Analysis, etc.) based ONLY on its basic structure and a sample of its first 20 and last 20 rows.

### Dataset Metadata:
{context}

{col_truncated_msg}

### Sample - First 20 Rows (CSV Format):
```csv
{first_20_str}
```

### Sample - Last 20 Rows (CSV Format):
```csv
{last_20_str}
```

Please perform a thorough review and provide the response in valid JSON format with the following keys:
- "mva_suitability_score": int, overall score from 0 to 100 indicating how "good" or suitable this dataset is to send ahead for further Multivariate Analysis.
- "structural_consistency_score": int, score from 0 to 100 on how consistent the schema and data formats are between the first 20 and last 20 rows (look for schema drift, format shifts, or sudden data alignment problems).
- "structural_consistency_explanation": string, a brief explanation of structural consistency between the start and end of the dataset.
- "numerical_variable_density_score": int, score from 0 to 100 indicating the density and availability of numeric/continuous variables (essential for correlation, covariance, variance calculations in MVA).
- "missing_data_risk": string, risk level ("Low", "Medium", or "High") representing the risk posed by missing/null values in columns.
- "mva_techniques": list of strings, recommended multivariate techniques (e.g. ["PCA", "Multiple Regression", "Cluster Analysis"]) or why none are recommended.
- "suitability_reasoning": string, a detailed paragraph explaining the reasoning behind the scores and recommendations, listing what features make it suitable, any potential issues downstream agents will face (e.g. categorical variables needing encoding, scaling requirements, outliers, non-numeric formatting in numeric columns), and a clear final recommendation.

Return your response as a single valid JSON object, and nothing else — no markdown code fences, no ```json wrapper, no preamble, no trailing commentary. The response must start with {{ and end with }}."""