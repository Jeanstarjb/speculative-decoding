import os
import redis
import torch

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

    def generate_candidates(self):
        """Retrieve top-k most frequent tokens from Redis"""
        candidates = self.redis_conn.zrevrange("token_frequencies", 0, self.top_k-1)
        return [int(c) for c in candidates]

    def validate_candidates(self, candidates, target_logits):
        """Validate candidates against target model's predictions"""
        probs = torch.softmax(target_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_n)
        top_indices = top_indices.tolist()
        
        valid = []
        for candidate in candidates:
            if candidate in top_indices and probs[candidate].item() >= self.threshold:
                valid.append(candidate)
        
        if not valid:
            valid.append(top_indices[0])
        
        return valid