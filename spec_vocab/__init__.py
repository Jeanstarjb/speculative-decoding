import os
import redis
import torch
from typing import List

class SpecVocab:
    """Dynamic vocabulary management system for speculative decoding"""

    def __init__(self, top_k=50, top_n=5, threshold=0.1):
        self.redis_conn = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=6379,
            decode_responses=True
        )
        self.top_k = top_k
        self.top_n = top_n
        self.threshold = threshold

    def update_frequencies(self, token_ids):
        """Update token frequencies in Redis sorted set"""
        for token_id in token_ids:
            self.redis_conn.zincrby("token_frequencies", 1, str(token_id))

    def get_candidates(self, draft_top_n: List[int]) -> List[int]:
        """Get combined candidate tokens from frequency cache and draft model"""
        top_k_tokens = self.redis_conn.zrevrange("token_frequencies", 0, self.top_k-1)
        combined = list(set([int(t) for t in top_k_tokens] + draft_top_n))
        return combined[:self.top_k + self.top_n]