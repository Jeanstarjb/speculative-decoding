from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import time
from typing import List
from speculative_decoding import SpeculativeDecoder
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
from caching import RedisCache
import redis
import os

app = FastAPI()

# Initialize Redis-based components
redis_host = os.getenv('REDIS_HOST', 'localhost')
cache = RedisCache(host=redis_host)

# Model initialization
draft_model = DraftModel(vocab_size=32000)
target_model = TargetModel(vocab_size=32000)
spec_vocab = SpecVocab()
decoder = SpeculativeDecoder(
    draft_model,
    target_model,
    spec_vocab,
    cache=cache
)
