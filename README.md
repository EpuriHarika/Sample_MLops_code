# Sample_MLops_code

This project demonstrates an end-to-end machine learning workflow aligned with production MLOps practices.

## Components

1. Training Pipeline
- Model training using scikit-learn
- Logging and evaluation
- Model artifact generation

2. Deployment
- FastAPI-based inference service
- Containerized using Docker

3. CI/CD
- GitHub Actions pipeline for automated training

4. RAG Pipeline
- Vector search using FAISS
- Retrieval-based context generation

## How to Run

### Train model
python pipelines/train.py

### Run API
uvicorn app.main:app --reload

### Docker
docker build -t mlops-app .
docker run -p 8000:8000 mlops-app
