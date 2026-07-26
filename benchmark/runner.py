import time
import torch
import psutil
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

class BenchmarkRunner:
    """Benchmarking system for comparing decoding strategies"""

    def __init__(self, decoders: Dict[str, object], config: Dict):
        self.decoders = decoders
        self.config = config
        self.results = defaultdict(dict)

    def _generate_prompts(self) -> List[str]:
        """Generate test prompts with varying lengths"""
        return [
            'The future of AI is ',
            'In a world where artificial intelligence ',
            'Machine learning models can ' * 5,
            'Natural language processing enables ' * 3
        ][:self.config['num_prompts']]

    def _measure_metrics: Dict):
        """Track memory usage during generation"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            metrics['gpu_mem'] = torch.cuda.max_memory_allocated()
        metrics['cpu_mem'] = psutil.Process().memory_info().rss
        return metrics

    def run(self) -> Dict:
        """Execute benchmarking across all configurations"""
        prompts = self._generate_prompts()

        for decoder_name, decoder in self.decoders.items():
            print(f'\n=== Benchmarking {decoder_name} ===')
            
            for batch_size in tqdm(self.config['batch_sizes'], desc='Batch sizes'):
                batch_metrics = {
                    'latency': [],
                    'throughput': [],
                    'acceptance_rate': []
                }

                for prompt in prompts[:self.config['samples_per_config']]:
                    start_time = time.perf_counter()
                    
                    # Run generation and capture metrics
                    with torch.inference_mode():
                        outputs = decoder.generate(
                            input_ids=torch.tensor([decoder.tokenizer.encode(prompt)]),
                            max_length=self.config['max_length'],
                            temperature=self.config['temperature']
                        )
                    
                    # Calculate metrics
                    duration = time.perf_counter() - start_time
                    batch_metrics['latency'].append(duration)
                    batch_metrics['throughput'].append(len(outputs[0]) / duration)
                    
                    # For speculative: calculate token acceptance rate
                    if hasattr(decoder, 'acceptance_rate'):
                        batch_metrics['acceptance_rate'].append(decoder.acceptance_rate)

                    # Measure memory usage
                    self._measure_resources(batch_metrics)

                # Aggregate results
                self.results[decoder_name][batch_size] = {
                    'avg_latency': sum(batch_metrics['latency']) / len(batch_metrics['latency']),
                    'avg_throughput': sum(batch_metrics['throughput']) / len(batch_metrics['throughput']),
                    'max_cpu_mem': max(batch_metrics.get('cpu_mem', [])),
                    'max_gpu_mem': max(batch_metrics.get('gpu_mem', []))
                }

        return self.results

    def generate_report(self, path: str = 'benchmark_results.json'):
        """Save benchmarking results to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)