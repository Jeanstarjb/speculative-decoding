import pytest
import torch
from unittest.mock import Mock
from spec_vocab import SpecVocab


def test_frequency_updates():
    mock_redis = Mock()
    spec_vocab = SpecVocab()
    spec_vocab.redis_conn = mock_redis
    
    test_ids = [101, 202, 303]
    spec_vocab.update_frequencies(test_ids)
    
    assert mock_redis.zincrby.call_count == 3
    calls = [call[0] for call in mock_redis.zincrby.call_args_list]
    assert all(c[0] == "token_frequencies" for c in calls)
    assert {int(c[2]) for c in calls} == set(test_ids)


def test_candidate_generation():
    mock_redis = Mock()
    mock_redis.zrevrange.return_value = ['789', '456', '123']
    spec_vocab = SpecVocab(top_k=3)
    spec_vocab.redis_conn = mock_redis
    
    candidates = spec_vocab.generate_candidates()
    
    assert candidates == [789, 456, 123]
    mock_redis.zrevrange.assert_called_once_with("token_frequencies", 0, 2)


def test_validation_logic():
    spec_vocab = SpecVocab(top_n=3, threshold=0.15)
    
    # Test with valid candidates
    logits = torch.tensor([0.1, 0.5, 0.3, 0.2], dtype=torch.float32)
    candidates = [1, 2]
    valid = spec_vocab.validate_candidates(candidates, logits)
    assert valid == [1, 2]
    
    # Test fallback to target model
    logits = torch.tensor([0.01, 0.01, 0.01, 0.97], dtype=torch.float32)
    candidates = [0, 1]
    valid = spec_vocab.validate_candidates(candidates, logits)
    assert valid == [3]