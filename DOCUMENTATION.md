# SpecVocab Integration Documentation

## Overview
Our speculative decoding system reduces inference costs by 3-5x through:
1. Dynamic vocabulary subset selection
2. Speculative parallel verification
3. Adaptive caching strategies

## Getting Started
```bash
# Clone repository
git clone https://github.com/your-org/speculative-decoding
cd speculative-decoding

# Start services
docker-compose up -d --build

# Verify installation
curl http://localhost:8000/health
```

## Core Components
### SpecVocab Module
```python
from spec_vocab import SpecVocab, SpecVocabConfig

config = SpecVocabConfig(
    adaptive_threshold=0.75,
    task_specific_scaling={'translation': 1.2}
)
spec_vocab = SpecVocab(config)
```

### API Endpoints
| Endpoint | Description |
|----------|-------------|
| POST /generate | Main text generation endpoint |
| GET /metrics | Real-time performance metrics |
| POST /benchmark | Comparative performance analysis |

## Advanced Configuration
```yaml
# configs/spec_vocab_config.yaml
optimized_presets:
  high_throughput:
    adaptive_threshold: 0.6
    max_vocab_subset: 8192
    cache_strategy: aggressive
```