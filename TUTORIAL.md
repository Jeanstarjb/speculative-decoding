# Integration Tutorial

## Basic Usage
```python
from clients import SpecDecodingClient

client = SpecDecodingClient(
    base_url="http://localhost:8000",
    api_key=os.getenv("API_KEY")
)

response = client.generate(
    prompt="The future of AI is",
    max_length=100,
    temperature=0.7
)
```

## Custom Vocabulary Strategies
```python
# Custom vocabulary adapter
def medical_vocab_adapter(prompt: str) -> SpecVocabConfig:
    return SpecVocabConfig(
        task_specific_scaling={'medical_terms': 1.5},
        min_vocab_subset=1024
    )

client.set_vocab_adapter(medical_vocab_adapter)
```

## Monitoring Integration
```python
metrics = client.get_metrics()
print(f"Throughput: {metrics.throughput_tokens_sec:.1f} tokens/sec")
print(f"Acceptance Rate: {metrics.acceptance_rate:.1%}")
```

## Real-world Example: Translation Service
```python
def translate_text(text: str, target_lang: str) -> str:
    client.set_task_type("translation")
    response = client.generate(
        prompt=f"Translate to {target_lang}: {text}",
        temperature=0.9
    )
    return response.generated_text
```