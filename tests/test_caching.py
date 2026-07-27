import pytest
from unittest.mock import Mock
import torch
import base64
import io
from backend.app.caching import RedisCache

@pytest.fixture
def mock_redis():
    return Mock(spec=['get', 'setex'])

def test_cache_roundtrip(mock_redis):
    cache = RedisCache(redis_conn=mock_redis)
    test_tensor = torch.randn(3, 256)
    device = torch.device('cpu')

    # Test serialization/deserialization
    mock_redis.get.return_value = base64.b64encode(
        torch.save(test_tensor, io.BytesIO()).getvalue()
    ).decode()
    
    deserialized = cache._deserialize(
        cache._serialize(test_tensor),
        device
    )
    assert torch.allclose(test_tensor, deserialized)

    # Test cache integration
    input_ids = torch.randint(0, 1000, (1, 10))
    cache.set_draft_logits(input_ids, test_tensor)
    assert mock_redis.setex.call_count == 1

    mock_redis.get.return_value = cache._serialize(test_tensor)
    cached = cache.get_draft_logits(input_ids)
    assert torch.allclose(cached, test_tensor)