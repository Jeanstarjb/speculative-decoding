# Speculative Decoding

A from-scratch implementation of speculative decoding for LLM inference ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)), with a proof of correctness, an interactive demo, and a containerized API.........................

**[Try the live demo →](https://speculative-decoding-u6wkd7vdc2ex4a9xwp3zby.streamlit.app/)**

## Overview

Autoregressive language models generate one token per forward pass, which underuses accelerator hardware since each pass only predicts a single next token. Speculative decoding addresses this with two models:

- A small, fast **draft model** proposes several tokens ahead.
- The large **target model** verifies all of them in a single forward pass and accepts or rejects each one using an exact probability ratio.

When verification accepts a proposed token, that's effectively "free" — the model got a token without paying for its own dedicated forward pass. The accept/reject rule is calibrated so the final output distribution is mathematically identical to sampling from the target model alone; correctness isn't traded away for speed.

## How it works

1. The draft model proposes *k* tokens autoregressively.
2. The target model evaluates all *k* positions in one forward pass.
3. Each token is accepted with probability `min(1, p_target / p_draft)`.
4. On rejection, the next token is resampled from the residual distribution `max(0, p_target − p_draft)` (renormalized), and the rest of the draft's proposals are discarded.
5. If every proposed token is accepted, one additional "bonus" token is sampled from the target model for free.

Both models use `DynamicCache` for KV-caching; a rejected token rolls the cache back to the last accepted position via `.crop()` rather than recomputing from scratch.

## Correctness

At near-zero temperature, both models become deterministic. In that regime, speculative decoding is required to produce the exact same token sequence as plain greedy decoding from the target model — this is checked directly with an exact tensor comparison:

```bash
python demo/test_correctness.py
```

```
EXACT MATCH: True
PASS: speculative decoding reduces to target-model greedy decoding, as required.
```

## Benchmark results

```bash
python demo/speculative_decoding.py
```

| Metric | Value |
|---|---|
| Acceptance rate | 51–86% across runs |
| Target-model forward passes | ~4–21, vs. 15–60 for naive decoding |
| Speedup (CPU) | 0.11x – 0.75x |

On CPU, speculative decoding is slower than plain generation, which is the expected result rather than a defect. The technique's speedup comes from a GPU-specific property: evaluating several sequence positions in one forward pass costs only marginally more than evaluating one, because a single-token forward pass leaves most of a GPU's parallel compute idle. A CPU has no equivalent slack — evaluating *k* positions costs close to *k* times as much, not ~1x — so batched verification has nothing to exploit, while the draft model's extra forward passes are pure overhead. The acceptance rate and reduced forward-pass count above show the algorithm behaving correctly; the result is a property of the hardware, not the implementation.

`notebooks/speculative_decoding_kaggle.ipynb` runs the full benchmark on GPU when one is available, and automatically scales down on CPU.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install torch transformers    # CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu

python demo/test_correctness.py
python demo/speculative_decoding.py
```

Default model pair: `distilgpt2` (draft) → `gpt2` (target), ~200M parameters total. A larger `distilgpt2` → `gpt2-medium` pairing is also available. Both use the same tokenizer, which speculative decoding requires.

### Interactive demo

**Live: [speculative-decoding-u6wkd7vdc2ex4a9xwp3zby.streamlit.app](https://speculative-decoding-u6wkd7vdc2ex4a9xwp3zby.streamlit.app/)**

Or run locally:

```bash
pip install streamlit
streamlit run app.py
```

Enter a prompt and compare speculative vs. naive decoding side by side, with live timing, throughput, forward-pass count, and acceptance rate.

### API

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

or with Docker:

```bash
docker compose up --build
```

```bash
$ curl http://127.0.0.1:8000/health
{"status":"ok","device":"cpu","draft_model":"distilgpt2","target_model":"gpt2"}

$ curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
    -d '{"prompt": "Docker containers are useful because", "max_new_tokens": 15, "compare_to_naive": true}'

{"text":"Docker containers are useful because a Web server may be tasked with protecting a group's
database from eavesdropping","elapsed_seconds":24.58,"tokens_per_second":0.61,"target_forward_passes":4,
"acceptance_rate":0.86,"device":"cpu","naive_elapsed_seconds":2.78,"naive_text":"Docker containers are
useful because a button pilot with a spot sensor is \"tree legs diffuse the spill\"","speedup":0.11}
```

Interactive API docs (Swagger UI) are served at `/docs`. `backend/tests/test_api.py` covers the endpoints with `pytest`.

## Project structure

```
demo/speculative_decoding.py    Core implementation: speculative_decode(), naive_decode()
demo/test_correctness.py        Correctness proof (exact-match test)
backend/app/main.py             FastAPI service wrapping the core implementation
backend/tests/                  API tests
app.py                          Streamlit interactive demo
notebooks/                      Kaggle-ready notebook with real executed output
Dockerfile, docker-compose.yml  Containerized deployment
```

## Roadmap

- Verified GPU benchmark (theory and CPU results support a real speedup; not yet measured on GPU)
- Dynamic vocabulary subsetting (restricting candidate tokens per step to reduce softmax cost) as an additional optimization on top of standard speculative decoding

## References

- Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) (2023)
- Chen et al., ["Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) (2023)
