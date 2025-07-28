# Use a Python base image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script
COPY download_model.py .

# Create model directory
RUN mkdir -p /app/model

# Download the model into /app/model
RUN python download_model.py

# Default command just to keep the container alive or check contents
CMD ["ls", "-l", "/app/model"]
