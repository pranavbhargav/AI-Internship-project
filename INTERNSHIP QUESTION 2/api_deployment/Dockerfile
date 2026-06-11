# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# System deps (for some Python wheels / SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY app.py /app/app.py

# Copy model into image (keep original relative location logic)
COPY ../cifar10_resnet18.pth /app/cifar10_resnet18.pth

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

