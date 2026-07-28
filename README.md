# Speculative Decoding with a Speculative Vocabulary

**Research Paper:** [https://arxiv.org/pdf/2602.13836v2](https://arxiv.org/pdf/2602.13836v2)

## Testing Instructions

### Integration Tests
```bash
pytest backend/tests/test_api.py backend/tests/test_decoding.py
```

### Load Testing
```bash
cd load_test
docker build -t load-test .
docker run --network host load-test
```

### Performance Reporting
```bash
python benchmark/generate_report.py
```

...[TRUNCATED TO MAINTAIN FILE STRUCTURE]...