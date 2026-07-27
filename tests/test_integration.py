import pytest
from unittest.mock import Mock, patch
import torch
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
from backend.app.speculative_decoding import SpeculativeDecoder

@pytest.fixture
def pipeline_components():
    draft = DraftModel(vocab_size=1000)
    target = TargetModel(vocab_size=1000)
    spec_vocab = SpecVocab()
    decoder = SpeculativeDecoder(draft, target, spec_vocab, max_length=20)
    return draft, target, spec_vocab, decoder

@patch('backend.app.caching.RedisCache')
def test_end_to_end_generation(mock_cache, pipeline_components):
    _, _, _, decoder = pipeline_components
    input_ids = torch.tensor([[42, 24, 15]])
    
    # Mock cache to return empty
    mock_cache.get.return_value = None
    
    output = decoder.generate(input_ids)
    
    assert isinstance(output, list)
    assert len(output) <= 20
    assert all(isinstance(t, int) for t in output)

def test_cache_utilization(pipeline_components):
    draft, target, spec_vocab, decoder = pipeline_components
    decoder.cache = Mock()
    input_ids = torch.tensor([[1, 2, 3]])
    
    # First generation - should call set
    decoder.generate(input_ids)
    assert decoder.cache.set.called
    
    # Second generation - should call get
    decoder.generate(input_ids)
    assert decoder.cache.get.called

@patch('torch.nn.functional.softmax')
def test_acceptance_logic(mock_softmax, pipeline_components):
    draft, target, _, decoder = pipeline_components
    input_ids = torch.tensor([[10, 20]])
    
    # Force accept all predictions
    mock_softmax.return_value = torch.tensor([[0.9, 0.1]])
    
    output = decoder.generate(input_ids)
    assert len(output) > 2

def test_max_length_constraint(pipeline_components):
    _, _, _, decoder = pipeline_components
    decoder.max_length = 5
    input_ids = torch.tensor([[1, 2, 3]])
    
    output = decoder.generate(input_ids)
    assert len(output) == 5 - 3 + 1  # Initial tokens + generated

def test_temperature_adjustment(pipeline_components):
    _, _, _, decoder = pipeline_components
    decoder.temperature = 0.1  # Deterministic
    input_ids = torch.tensor([[50, 60]])
    
    output1 = decoder.generate(input_ids)
    output2 = decoder.generate(input_ids)
    
    assert output1 == output2