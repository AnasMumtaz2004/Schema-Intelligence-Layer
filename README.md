# Schema Intelligence Layer

A FastAPI-based agent that ingests CSV and Excel datasets, extracts metadata, classifies the business domain using a Groq-hosted LLM, and stores metadata in a centralized SQLite database.

## Features

- **Dataset Upload** — `POST /upload-dataset` accepts CSV, XLSX, and XLS files
- **Automatic Metadata Extraction** — Row/column counts, column names, data types, sample rows
- **LLM-Powered Column Descriptions** — Auto-generates descriptions for every column
- **Business Domain Classification** — Classifies datasets into Finance, Sales, Marketing, HR, Operations, Supply Chain, Customer Support, Healthcare, or Other
- **SQLite Metadata Catalog** — Persists all metadata in `dataset_catalog.db`
- **In-Memory DataFrame Registry** — Downstream agents can retrieve the loaded DataFrame without re-reading the file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys).

### 3. Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-dataset` | Upload a CSV or Excel dataset |
| `GET` | `/datasets` | List all uploaded datasets |
| `GET` | `/datasets/{dataset_id}` | Get full metadata for a dataset |
| `GET` | `/datasets/{dataset_id}/dataframe` | Get the in-memory DataFrame as JSON |
| `GET` | `/health` | Health check |

## API Documentation

Once the server is running, visit:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Example Usage

### Upload a CSV File

```bash
curl -X POST http://localhost:8000/upload-dataset \
  -F "file=@your_dataset.csv"
```

### Response

```json
{
  "dataset_id": "DS_001",
  "dataset_name": "your_dataset.csv",
  "business_domain": "Sales",
  "dataset_summary": "Contains customer orders, products, revenue, and regional sales information.",
  "row_count": 10500,
  "column_count": 12,
  "status": "Completed"
}
```

## Project Structure

```
Data_identification_agent/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (Groq API key, DB path)
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── validator.py           # File validation
│   │   ├── loader.py              # CSV/Excel → DataFrame loader
│   │   ├── metadata_extractor.py  # Metadata extraction
│   │   ├── llm_service.py         # Groq LLM integration
│   │   └── database.py            # SQLite operations
│   ├── routes/
│   │   └── upload.py              # API routes
│   └── datastore/
│       └── registry.py            # In-memory DataFrame registry
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `DATABASE_PATH` | `./dataset_catalog.db` | Path to SQLite database |
