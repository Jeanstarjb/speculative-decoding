import json
import matplotlib.pyplot as plt
import numpy as np

def generate_performance_report():
    with open('benchmark_results.json') as f:
        data = json.load(f)

    plt.figure(figsize=(12, 6))
    for strategy in data['strategies']:
        x = data['batch_sizes']
        y = [data[strategy][str(bs)]['tokens_per_sec'] for bs in x]
        plt.plot(x, y, marker='o', label=strategy)
    
    plt.title('Throughput Comparison by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens/Second')
    plt.legend()
    plt.grid(True)
    plt.savefig('throughput_comparison.png')

    with open('performance_report.md', 'w') as f:
        f.write('# Performance Benchmark Report\n\n')
        f.write(f"**Total Tests Run**: {data['total_tests']}\n\n")
        f.write('## Throughput Comparison\n')
        f.write('![Throughput Chart](throughput_comparison.png)\n\n')
