from pydantic import BaseModel

class HealthCheckResponse(BaseModel):
    api_status: str
    redis_status: str
    model_status: str
    vocab_status: str

class BenchmarkRequest(BaseModel):
    num_prompts: int = 10
    max_length: int = 128
    temperature: float = 0.7
    
class BenchmarkResult(BaseModel):
    strategy: str
    avg_tokens_sec: float
    total_time: float
    memory_usage: float

class MonitoringMetrics(BaseModel):
    throughput_tokens_per_sec: float
    average_latency_per_step_sec: float
    acceptance_rate: float