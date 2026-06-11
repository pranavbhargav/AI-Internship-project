from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

import io
import os
import urllib.request

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cifar10_resnet18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="CIFAR-10 ResNet18 Inference API", version="1.0")


class PredictRequest(BaseModel):
    # Use an image URL so the API works easily inside Docker.
    image_url: HttpUrl
    top_k: Optional[int] = 3


class Prediction(BaseModel):
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    device: str
    model_path: str
    predictions: List[Prediction]

    model_config = {"protected_namespaces": ()}



def build_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Make sure cifar10_resnet18.pth exists in INTERNSHIP QUESTION 2/"
        )

    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# Match inference.py preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_image_from_url(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        image_bytes = response.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# Loaded at startup
model: Optional[torch.nn.Module] = None


@app.on_event("startup")
def _startup():
    global model
    model = build_model(MODEL_PATH, DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if req.top_k is None or req.top_k <= 0:
        top_k = 3
    else:
        top_k = min(int(req.top_k), 10)

    try:
        img = load_image_from_url(str(req.image_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image_url: {e}")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    top_prob, top_classes = torch.topk(probabilities, top_k)
    predictions = [
        Prediction(
            class_name=CLASSES[idx.item()],
            probability=prob.item(),
        )
        for prob, idx in zip(top_prob, top_classes)
    ]

    return PredictResponse(
        device=str(DEVICE),
        model_path=os.path.abspath(MODEL_PATH),
        predictions=predictions,
    )

