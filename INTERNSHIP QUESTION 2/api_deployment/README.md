# CIFAR-10 ResNet18 Inference API (FastAPI) + Docker

Wraps the trained model `cifar10_resnet18.pth` into a simple HTTP API.

## Endpoints
- `GET /health` -> health check
- `POST /predict` -> get top-k predictions for an image URL

## Run locally (Docker)

### 1) Build
From this directory:

```bash
cd "INTERNSHIP QUESTION 2/api_deployment"

docker build -t cifar10-api:latest .
```

### 2) Run
```bash
docker run --rm -p 8000:8000 cifar10-api:latest
```

### 3) Test
Health:
```bash
curl http://localhost:8000/health
```

Predict (example image URL):
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=256",
    "top_k": 3
  }'
```

## Example request
```json
{
  "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=256",
  "top_k": 3
}
```

## Example response
```json
{
  "device": "cpu",
  "model_path": "/app/cifar10_resnet18.pth",
  "predictions": [
    {"class_name": "cat", "probability": 0.5123},
    {"class_name": "dog", "probability": 0.2211},
    {"class_name": "frog", "probability": 0.1044}
  ]
}
```

## Notes
- The API expects an **image URL** for easy use inside Docker.
- Top probabilities are raw softmax probabilities (0 to 1). If you want percentages, multiply by 100.

