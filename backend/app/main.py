from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import time
from typing import List
from speculative_decoding import SpeculativeDecoder
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
import redis
import os

app = FastAPI()

# Initialize components
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    decode_responses=True
)

draft_model = DraftModel(vocab_size=32000)
target_model = TargetModel(vocab_size=32000)
spec_vocab = SpecVocab()
decoder = SpeculativeDecoder(draft_model, target_model, spec_vocab)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_text: str
    inference_time: float
    tokens_generated: int
    tokens_per_second: float

class ModelConfigResponse(BaseModel):
    draft_model: dict
    target_model: dict
    spec_vocab: dict

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        # Convert prompt to tensor (simplified tokenization)
        input_ids = torch.tensor([[ord(c) for c in request.prompt]], dtype=torch.long)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input processing error: {str(e)}")

    start_time = time.time()
    try:
        output_ids = decoder.generate(
            input_ids,
            max_length=request.max_length,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    inference_time = time.time() - start_time
    generated_text = ''.join([chr(i) for i in output_ids[0].tolist()])

    return GenerateResponse(
        generated_text=generated_text,
        inference_time=round(inference_time, 4),
        tokens_generated=len(output_ids[0]),
        tokens_per_second=round(len(output_ids[0]) / inference_time, 2)
    )

@app.get("/models/config", response_model=ModelConfigResponse)
async def get_model_configs():
    return ModelConfigResponse(
        draft_model={
            "d_model": draft_model.d_model,
            "nhead": draft_model.nhead,
            "max_seq_len": draft_model.max_seq_len
        },
        target_model={
            "d_model": target_model.d_model,
            "nhead": target_model.nhead,
            "num_layers": target_model.num_layers,
            "max_seq_len": target_model.max_seq_len
        },
        spec_vocab={
            "top_k": spec_vocab.top_k,
            "top_n": spec_vocab.top_n,
            "threshold": spec_vocab.threshold
        }
    )

@app.get("/vocab/status")
async def get_vocab_status():
    top_tokens = redis_conn.zrevrange("token_frequencies", 0, spec_vocab.top_k-1)
    return {
        "top_tokens": [int(token) for token in top_tokens],
        "cache_size": redis_conn.zcard("token_frequencies")
    }
