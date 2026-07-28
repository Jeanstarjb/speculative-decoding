import pytest
import torch
from unittest.mock import MagicMock, patch
from backend.app.speculative_decoding import SpeculativeDecoder

@patch('backend.app.speculative_decoding.TargetModel')
@patch('backend.app.speculative_decoding.DraftModel')
def test_speculative_decoding(mock_draft, mock_target):
    mock_draft.generate.return_value = (torch.tensor([[1,2,3]]), None)
    mock_target.verify.return_value = (torch.tensor([1,2]), True)
    
    decoder = SpeculativeDecoder(
        draft_model=mock_draft,
        target_model=mock_target,
        spec_vocab=MagicMock(),
        max_length=100
    )
    
    result = decoder.generate(torch.tensor([[0]]))
    assert len(result) > 0
    mock_draft.generate.assert_called_once()
    mock_target.verify.assert_called_once()

@patch('backend.app.speculative_decoding.RedisCache')
def test_caching_integration(mock_cache):
    decoder = SpeculativeDecoder(
        draft_model=MagicMock(),
        target_model=MagicMock(),
        spec_vocab=MagicMock(),
        cache=mock_cache
    )
    decoder.generate(torch.tensor([[0]]))
    assert mock_cache.get.called or mock_cache.set.called
