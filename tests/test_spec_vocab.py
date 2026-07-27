import pytest
from unittest.mock import Mock, call
import torch
from spec_vocab import SpecVocab

@pytest.fixture
def mock_redis():
    return Mock()

def test_low_frequency_fallback(mock_redis):
    spec_vocab = SpecVocab(top_k=3, top_n=5)
    spec_vocab.redis_conn = mock_redis
    
    # Simulate only 2 high-frequency tokens
    mock_redis.zrevrange.return_value = ['100', '200']
    mock_redis.zscore.side_effect = [0.09, 0.08]  # Below threshold
    
    candidates = spec_vocab.get_candidates(current_token=torch.tensor([42]))
    assert len(candidates) == 2  # Fallback to available tokens

def test_threshold_filtering(mock_redis):
    spec_vocab = SpecVocab(threshold=0.1)
    spec_vocab.redis_conn = mock_redis
    
    mock_redis.zrevrange.return_value = ['100', '200', '300']
    mock_redis.zscore.side_effect = [0.15, 0.09, 0.2]
    
    candidates = spec_vocab.get_candidates(current_token=torch.tensor([42]))
    assert set(candidates) == {100, 300}  # Filter out 200

def test_candidate_prioritization(mock_redis):
    spec_vocab = SpecVocab(top_k=5, top_n=3)
    spec_vocab.redis_conn = mock_redis
    
    mock_redis.zrevrange.return_value = ['100', '200', '300', '400', '500']
    mock_redis.zscore.side_effect = [0.2, 0.19, 0.18, 0.17, 0.16]
    
    candidates = spec_vocab.get_candidates(current_token=torch.tensor([42]))
    assert candidates == [100, 200, 300]

def test_empty_frequency_data(mock_redis):
    spec_vocab = SpecVocab()
    spec_vocab.redis_conn = mock_redis
    mock_redis.zrevrange.return_value = []
    
    candidates = spec_vocab.get_candidates(current_token=torch.tensor([42]))
    assert len(candidates) == 0

@pytest.mark.parametrize('batch_size', [1, 4, 8])
def test_batch_frequency_updates(batch_size, mock_redis):
    spec_vocab = SpecVocab()
    spec_vocab.redis_conn = mock_redis
    
    token_ids = torch.randint(100, 200, (batch_size, 10))
    spec_vocab.update_frequencies(token_ids.flatten().tolist())
    
    assert mock_redis.zincrby.call_count == batch_size * 10
    assert all(call_args[0][0] == 'token_frequencies' 
               for call_args in mock_redis.zincrby.call_args_list)