import redis
import torch
import hashlib
import base64
import io

class RedisCache:
    """Redis-based caching system for model outputs during speculative decoding"""

    def __init__(self, host='localhost', port=6379):
        self.redis_conn = redis.Redis(
            host=host,
            port=port,
            decode_responses=False,
            socket_connect_timeout=3,
            retry_on_timeout=True
        )

    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """Generate SHA256 hash from tensor data"""
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    def _serialize(self, tensor: torch.Tensor) -> str:
        """Serialize tensor to base64 string"""
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _deserialize(self, data: str, device: torch.device) -> torch.Tensor:
        """Deserialize base64 string to tensor"""
        buffer = io.BytesIO(base64.b64decode(data))
        return torch.load(buffer).to(device)

    def get_draft_logits(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Retrieve cached draft model outputs"""
        key = f'draft:{self._tensor_hash(input_ids)}'
        if cached := self.redis_conn.get(key):
            return self._deserialize(cached, input_ids.device)
        return None

    def set_draft_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> None:
        """Cache draft model outputs with 5 minute TTL"""
        key = f'draft:{self._tensor_hash(input_ids)}'
        self.redis_conn.setex(key, 300, self._serialize(logits))

    def get_target_logits(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Retrieve cached target model outputs"""
        key = f'target:{self._tensor_hash(input_ids)}'
        if cached := self.redis_conn.get(key):
            return self._deserialize(cached, input_ids.device)
        return None

    def set_target_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> None:
        """Cache target model outputs with 10 minute TTL"""
        key = f'target:{self._tensor_hash(input_ids)}'
        self.redis_conn.setex(key, 600, self._serialize(logits))