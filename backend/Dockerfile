FROM ghcr.io/ggml-org/llama.cpp:server@sha256:052888c603894ecfa08deb6714fdc2d7084e491e174d5ab41bfb36b49ee7151c AS llama-server

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir doclayout-yolo==0.0.4 --no-deps
COPY --from=llama-server /app /opt/llama.cpp
COPY backend/app ./app
ENV PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/llama.cpp
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]

