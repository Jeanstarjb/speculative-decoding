import yaml
import torch
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
from backend.app.speculative_decoding import SpeculativeDecoder
from benchmark.eagle3 import Eagle3Decoder
from benchmark.runner import BenchmarkRunner

if __name__ == '__main__':
    # Load configuration
    with open('benchmark/config.yaml') as f:
        config = yaml.safe_load(f)['benchmark_params']

    # Initialize models
    draft_model = DraftModel(vocab_size=32000)
    target_model = TargetModel(vocab_size=32000)
    spec_vocab = SpecVocab()

    # Create decoders
    decoders = {
        'SpecVocab': SpeculativeDecoder(draft_model, target_model, spec_vocab),
        'EAGLE-3': Eagle3Decoder(draft_model, target_model)
    }

    # Run benchmarks
    runner = BenchmarkRunner(decoders, config)
    results = runner.run()
    runner.generate_report(f'benchmark_results_{int(time.time())}.json')

    print('\nBenchmarking complete. Results saved to JSON file.')