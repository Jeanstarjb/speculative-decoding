import pytest
import torch
from backend.app.spec_vocab import SpecVocab, SpecVocabConfig

@pytest.fixture
def sample_hidden_states():
    torch.manual_seed(42)
    return torch.randn(1, 128, 32000)  # (batch_size, seq_len, vocab_size)

@pytest.mark.parametrize('model_size,task_type,expected_range', [
    ('small', 'translation', (512, 2048)),
    ('large', 'summarization', (2048, 4096)),
    ('medium', 'code_generation', (1024, 3072))
])
def test_adaptive_vocab_selection(sample_hidden_states, model_size, task_type, expected_range):
    config = SpecVocabConfig(
        size_adaptation_strategy='linear',
        min_vocab_subset=512,
        max_vocab_subset=4096
    )
    spec_vocab = SpecVocab(config)
    
    vocab = spec_vocab.get_adaptive_vocab(
        sample_hidden_states,
        current_layer=6,
        model_size=model_size,
        task_type=task_type
    )
    
    assert expected_range[0] <= len(vocab) <= expected_range[1], \
        f"Vocab size {len(vocab)} outside expected range {expected_range}"

@pytest.mark.parametrize('layer,depth_factor', [
    (0, 1.0),
    (6, 1.5),
    (11, 1.9167)
])
def test_layer_wise_scaling(sample_hidden_states, layer, depth_factor):
    config = SpecVocabConfig(layer_wise_scaling=True)
    spec_vocab = SpecVocab(config)
    threshold = spec_vocab._calculate_dynamic_threshold(
        sample_hidden_states,
        layer=layer,
        scale_factor=1.0
    )
    
    base_threshold = spec_vocab.config.adaptive_threshold
    expected = base_threshold * depth_factor * (1 + 0.5)  # 0.5 mock entropy
    assert pytest.approx(threshold, rel=0.1) == expected

@pytest.mark.parametrize('strategy,model_size,expected', [
    ('linear', 'small', 614),
    ('exponential', 'large', 4096),
    (None, 'medium', 2048)
])
def test_size_adaptation_strategies(sample_hidden_states, strategy, model_size, expected):
    config = SpecVocabConfig(
        size_adaptation_strategy=strategy,
        size_adaptation_factors={'small': 0.6, 'medium': 1.0, 'large': 1.4},
        min_vocab_subset=512,
        max_vocab_subset=4096
    )
    spec_vocab = SpecVocab(config)
    candidates = torch.arange(0, 1024)  # 1024 candidate tokens
    adapted = spec_vocab._adapt_vocab_size(candidates, model_size)
    
    if strategy == 'exponential' and model_size == 'large':
        assert len(adapted) == 4096  # Max cap
    else:
        assert len(adapted) == expected