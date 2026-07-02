# Schema Intelligence Layer

A FastAPI-based dataset intelligence service that ingests CSV and Excel datasets, extracts core metadata structure, generates column descriptions, evaluates Multivariate Analysis (MVA) suitability, classifies business domains using a Groq-hosted LLM, and persists results in a centralized SQLite database.

---

## Features

- **Dataset Ingestion & Appending** — `POST /upload-dataset` accepts `.csv`, `.xlsx`, and `.xls` files. If a dataset already exists, it combines the new rows with the existing cache and recalculates stats.
- **Structural Metadata Extraction** — Automatically computes row and column counts, extracts column names, identifies data types, and records serialized sample rows.
- **LLM-Powered Column Descriptions** — Generates short, single-sentence descriptions for every column in the dataset based on name, type, and sample data.
- **Business Domain Classification** — Categorizes datasets into standard business domains (Finance, Sales, Marketing, HR, Operations, Supply Chain, Customer Support, Healthcare, or Other) and granular sub-domains.
- **Multivariate Analysis (MVA) Suitability Evaluation** — Evaluates structural consistency (checks start vs. end of dataset), continuous numerical density, missing data risks, and recommends statistical techniques (e.g., PCA, Factor Analysis, Multiple Regression) with a suitability score (0-100).
- **SQLite Metadata Catalog** — Persists all extraction details, classifications, and suitability ratings in `dataset_catalog.db`.
- **In-Memory DataFrame Registry** — Caches DataFrames in RAM for fast downstream consumption without repeated disk read/write overhead.

---

## How Data is Processed & Cached

1. **Upload Stream to Bytes**: When you upload a dataset, FastAPI receives it as a file stream. The loader ([loader.py](file:///c:/Users/ANAS%20MUMTAZ/Desktop/MVA/Schema_Intelligence_layer/app/services/loader.py)) reads the raw file bytes into an in-memory byte buffer (`io.BytesIO`).
2. **Immediate Conversion to Pandas DataFrame**: The raw bytes are instantly parsed into a **Pandas DataFrame** using `pd.read_csv` or `pd.read_excel`.
3. **RAM Registry Storage**: This active Pandas DataFrame object is cached in memory under a global `_dataframes` dictionary via the `store_dataframe` function in [registry.py](file:///c:/Users/ANAS%20MUMTAZ/Desktop/MVA/Schema_Intelligence_layer/app/datastore/registry.py).
4. **Metadata Extraction & SQLite Persistence**: Once the DataFrame is created, key statistics (dimensions, data types, sample rows) are extracted from it to store in the SQLite catalog (`dataset_catalog.db`). The raw file itself is never written to disk, and downstream agents query it directly from RAM.

---
## Architecture & Directory Structure

```
Schema_Intelligence_layer/
├── app/
│   ├── main.py                    # FastAPI application setup and database lifespan hook
│   ├── config.py                  # Pydantic-settings config (.env loader)
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models and schemas
│   ├── routes/
│   │   └── upload.py              # API routes (Upload, Catalog, DataFrame endpoints)
│   ├── services/
│   │   ├── validator.py           # File validation logic
│   │   ├── loader.py              # Tabular file reader (Pandas & Openpyxl)
│   │   ├── metadata_extractor.py  # Column/type extraction and sample preparation
│   │   ├── llm_service.py         # Groq LLM integration (Domain, Column, MVA Analysis)
│   │   └── database.py            # SQLite DB initializations and CRUD operations
│   ├── prompts/
│   │   └── llm_service_prompt.py  # Render-ready prompt templates for LLM tasks
│   └── datastore/
│       └── registry.py            # RAM-based DataFrame cache registry
├── test_data/                     # Directory containing sample datasets
├── test_local.py                  # Standalone CLI test pipeline script
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment configuration template
└── README.md                      # Project documentation
```

---

## Setup & Running

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your Groq API key:
```ini
GROQ_API_KEY=your_groq_api_key_here
DATABASE_PATH=./dataset_catalog.db
GROQ_MODEL=llama-3.1-8b-instant
```
*(Get a free API key at [console.groq.com](https://console.groq.com/keys))*

### 3. Run the Server
Start the FastAPI server via Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Running Offline CLI Tests
You can process local datasets and view metadata outputs directly in the terminal without launching the API server:
```bash
# Process all files in test_data/
python test_local.py

# Process a specific file
python test_local.py test_data/banking_variance_data.csv
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-dataset` | Ingests a dataset, executes metadata extraction, domain classification, MVA suitability scoring, and stores it in the database. |
| `GET` | `/datasets` | Returns a list of all cataloged datasets with high-level summaries. |
| `GET` | `/datasets/{dataset_id}` | Retrieves the full metadata record (including column descriptions and suitability statistics). |
| `GET` | `/datasets/{dataset_id}/dataframe` | Returns the raw cached dataset rows as JSON (accepts a `limit` query parameter). |
| `GET` | `/health` | API health check. |

---

## Example API Response (`POST /upload-dataset`)

```json
{
  "dataset_id": "DS_004",
  "dataset_name": "neutral_payments_variance_trxnid_1000.xlsx",
  "business_domain": "Finance",
  "sub_domain": "Transaction Analysis",
  "dataset_summary": "Contains transaction records, including payment methods, settlement data, and risk assessment metrics for a financial services platform.",
  "row_count": 1000,
  "column_count": 25,
  "status": "Completed",
  "score": 80,
  "mva_suitability": {
    "mva_suitability_score": 80,
    "structural_consistency_score": 95,
    "structural_consistency_explanation": "The dataset has a consistent schema and data formats between the first 20 and last 20 rows...",
    "numerical_variable_density_score": 60,
    "missing_data_risk": "Low",
    "mva_techniques": [
      "PCA",
      "Multiple Regression",
      "Cluster Analysis"
    ],
    "suitability_reasoning": "The dataset is suitable for multivariate analysis with a score of 80. The structural consistency score is high..."
  }
}
```

---

## Configuration Variables

| Variable | Default Value | Description |
|---|---|---|
| `GROQ_API_KEY` | *(Required)* | Groq API credentials. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | The LLM model used for classification, descriptions, and MVA scoring. |
| `DATABASE_PATH` | `./dataset_catalog.db` | File path where the SQLite metadata catalog is saved. |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum allowed dataset file upload size. |
| `SAMPLE_ROWS` | `5` | The number of row records stored directly inside the SQLite metadata row. |
