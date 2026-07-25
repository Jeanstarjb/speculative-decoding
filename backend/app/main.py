from fastapi import FastAPI
import redis
import os

app = FastAPI()

# Initialize Redis connection
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    decode_responses=True
)

@app.get("/health")
async def health_check():
    return {
        "api_status": "healthy",
        "redis_status": "connected" if redis_conn.ping() else "disconnected"
    }