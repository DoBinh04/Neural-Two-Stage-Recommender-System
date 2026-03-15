# RecSys Project

A two-stage recommender system:
- **Retrieval** (Two-Tower + FAISS) to fetch candidates quickly.
- **Ranking** (Wide & Deep) to reorder candidates by relevance.
- **FastAPI** service for online inference.

## Project Architecture

### Folder overview
- `Pipeline/`: data ingestion, preprocessing, interaction building, and train/val/test split.
- `Retrieval/`: retrieval features, dataset builders, Two-Tower training, FAISS index creation, and embedding/index artifacts.
- `Ranking/`: Wide & Deep ranker definition and training artifacts.
- `Evaluation/`: offline evaluation scripts and saved metrics.
- `EDA/`: notebooks and exploratory analysis outputs.
- `api.py`: online inference API (startup loading + retrieval + ranking endpoints).
- `Dockerfile`: containerized API runtime.

## Training Phase (Offline Pipeline)

This phase runs offline to prepare everything needed for serving:

1. **Data preprocessing**
   - Clean/transform raw interaction data and build model-ready datasets.
2. **Model training**
   - Train retrieval model (Two-Tower).
   - Train ranking model (Wide & Deep).
3. **Embedding export**
   - Export user/item embeddings to artifact files.
4. **FAISS index building**
   - Build item vector index for fast ANN candidate retrieval.

## Inference Phase (Online Pipeline)

This phase runs per request through the API:

1. **API request**
   - Client sends request to `POST /recommend` (or tests via `GET /`).
2. **Retrieval**
   - Use user embedding + FAISS index to get top `candidate_k` items.
3. **Ranking**
   - Score retrieved candidates with Wide & Deep and return final top `top_k` item IDs.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API (Local)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Run with Dockerfile

### 1) Build image
```bash
docker build -t recsys-api .
```

### 2) Run container
```bash
docker run --rm -p 8000:8000 recsys-api
```

The API will be available at:
- `http://localhost:8000/`
- `http://localhost:8000/docs`

## API Endpoints

### `GET /`
Returns a simple HTML demo page to input `user_id` and test recommendations in the browser.

### `GET /health`
Checks service readiness.

**Example response:**

```json
{"status": "ok"}
```

If artifact/model loading fails at startup:

```json
{"status": "degraded", "detail": "<error message>"}
```

### `POST /recommend`
Generates top recommendations for a given user.

**Request body:**

```json
{
  "user_id": "123",
  "top_k": 10,
  "candidate_k": 100
}
```

- `user_id` (string, required): user ID to recommend for, user IDs are listed in [`Retrieval/data/user2idx.json`](./Retrieval/data/user2idx.json).
- `top_k` (int, optional, default `10`, range `1-100`): number of final results.
- `candidate_k` (int, optional, default `100`, range `1-500`): number of retrieved candidates before ranking.

**Success response:**

```json
{
  "user_id": "123",
  "top_k": 10,
  "recommendations": ["item_1", "item_2", "item_3"]
}
```

## Notes

- Ensure artifacts under `Retrieval/artifacts/` and `Ranking/artifacts/` exist before starting the API.
- Swagger docs are available at `/docs`.
