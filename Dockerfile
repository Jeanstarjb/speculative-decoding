# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY demo/ demo/
COPY backend/ backend/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
