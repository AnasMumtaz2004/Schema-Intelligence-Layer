# Schema Intelligence Layer

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

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-dataset` | Ingests a dataset, executes metadata extraction, domain classification, MVA suitability scoring, and stores it in the database. |
| `GET` | `/datasets` | Returns a list of all cataloged datasets with high-level summaries. |
| `GET` | `/datasets/{dataset_id}` | Retrieves the full metadata record (including column descriptions and suitability statistics). |
| `GET` | `/datasets/{dataset_id}/dataframe` | Returns the raw cached dataset rows as JSON (accepts a `limit` query parameter). |
| `GET` | `/health` | API health check. |

