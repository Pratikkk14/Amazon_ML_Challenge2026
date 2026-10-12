# Multi-stage Dockerfile for complete Entity Resolution System

# Stage 1: Build the React Vite Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Nginx / Static Serving
FROM python:3.11-slim
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend & ML Assets
COPY backend_server.py lightgbm_model.pkl sample_20_entities.tsv ./
COPY checkpoints_india/ ./checkpoints_india/

# Copy built frontend from Stage 1 into Nginx web root
COPY --from=frontend-builder /app/frontend/dist /var/www/html

# Expose backend (5000) and frontend (80)
EXPOSE 80 5000

# Create simple startup script
RUN echo '#!/bin/sh\nservice nginx start\npython backend_server.py' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
