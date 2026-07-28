import time
import torch
from tqdm import tqdm
from backend.app.spec_vocab import SpecVocab, SpecVocabConfig

class SpecVocabBenchmark:
    """Benchmarking tool for SpecVocab optimizations"""
    
    def __init__(self, model_sizes=['small', 'medium', 'large'], tasks=['translation', 'summarization']):
        self.model_sizes = model_sizes
        self.tasks = tasks
        self.results = []

    def run_benchmark(self, num_runs=100):
        """Execute benchmark across configurations"""
        config = SpecVocabConfig()
        vocab = SpecVocab(config)
        
        for _ in tqdm(range(num_runs), desc='Benchmark runs'):
            for size in self.model_sizes:
                for task in self.tasks:
                    hidden_states = torch.randn(1, 128, 32000)
                    
                    start_time = time.perf_counter()
                    selected = vocab.get_adaptive_vocab(
                        hidden_states,
                        current_layer=6,
                        model_size=size,
                        task_type=task
                    )
                    latency = time.perf_counter() - start_time
                    
                    self.results.append({
                        'model_size': size,
                        'task': task,
                        'latency': latency,
                        'vocab_size': len(selected)
                    })
        
        return self._analyze_results()

    def _analyze_results(self):
        """Analyze benchmark results across dimensions"""
        analysis = {}
        for size in self.model_sizes:
            size_data = [r for r in self.results if r['model_size'] == size]
            analysis[size] = {
                'avg_latency': sum(r['latency'] for r in size_data) / len(size_data),
                'avg_vocab_size': sum(r['vocab_size'] for r in size_data) / len(size_data)
            }
        
        for task in self.tasks:
            task_data = [r for r in self.results if r['task'] == task]
            analysis[task] = {
                'avg_latency': sum(r['latency'] for r in task_data) / len(task_data),
                'avg_vocab_size': sum(r['vocab_size'] for r in task_data) / len(task_data)
            }
        
        return analysis