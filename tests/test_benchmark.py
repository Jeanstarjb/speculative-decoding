import pytest
from benchmark.runner import BenchmarkRunner
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
from backend.app.speculative_decoding import SpeculativeDecoder

@pytest.fixture
def test_models():
    draft = DraftModel(vocab_size=1000)
    target = TargetModel(vocab_size=1000)
    spec_vocab = SpecVocab()
    return {
        'SpecVocab': SpeculativeDecoder(draft, target, spec_vocab),
        'EAGLE-3': Eagle3Decoder(draft, target)
    }

def test_benchmark_initialization(test_models):
    config = {
        'num_prompts': 2,
        'batch_sizes': [1],
        'samples_per_config': 1,
        'max_length': 10
    }
    runner = BenchmarkRunner(test_models, config)
    results = runner.run()
    
    assert 'SpecVocab' in results
    assert 'EAGLE-3' in results
    assert all('avg_throughput' in metrics for metrics in results.values())