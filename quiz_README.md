# PDF Quiz Generator API

FastAPI app that processes a PDF through a full NLP pipeline, then uses Gemini to generate MCQ questions.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # required by presidio-analyzer
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
uvicorn app:app --reload
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload-pdf` | Upload & process PDF |
| POST | `/generate-questions` | Generate MCQ questions |
| GET | `/chunks` | Preview processed text chunks |

### POST `/upload-pdf`

Query params:
- `source` – document name/label (optional)
- `section` – chapter/section name (optional)
- `redact_pii` – `true` to anonymize personal info (default: `false`)

### POST `/generate-questions`

Query params:
- `num_questions` – how many questions (1-30, default: 10)
- `chunk_index` – use specific chunk only (-1 = use all, default: -1)

## Processing Pipeline

1. **Structural clean** — fix encoding, strip HTML tags, normalize Unicode
2. **Noise removal** — deduplicate lines, remove lines < 10 chars
3. **PII redaction** (optional) — anonymize emails, names, phone numbers
4. **Text compression** — remove filler phrases, normalize punctuation
5. **Token-aware chunking** — split into ~600-token chunks with 80-token overlap
6. **Chunk enrichment** — add metadata header (source, section, chunk index)
