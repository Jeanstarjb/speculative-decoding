# Speculative Decoding, From Scratch

A correct, from-scratch implementation of speculative decoding ([Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) / [Chen et al. 2023](https://arxiv.org/abs/2302.01318)), built to actually prove the technique works rather than just claim it.

## The story

This repo started as an auto-generated scaffold: a lot of Docker/FastAPI/Kubernetes/Redis boilerplate around a core decoding algorithm that, on inspection, was fake — the accept/reject step (the entire point of speculative decoding) was stubbed out to always return the target model's top token, several files called methods that didn't exist on the classes they imported, and nothing had ever actually been run.

What's in `demo/` and `notebooks/` is a real rewrite of just the core algorithm: two real pretrained models, the actual accept/reject sampling math, working KV-caching, and a correctness test that proves it's right — not benchmarked-and-hoped, proven.

## What's real here

| Path | What it is |
|---|---|
| [`demo/speculative_decoding.py`](demo/speculative_decoding.py) | The actual implementation: `speculative_decode()` and a cached `naive_decode()` baseline |
| [`demo/test_correctness.py`](demo/test_correctness.py) | Proof of correctness (see below) |
| [`notebooks/speculative_decoding_kaggle.ipynb`](notebooks/speculative_decoding_kaggle.ipynb) | Same code, packaged to run standalone (Kaggle or local) — already contains real executed output |

Draft model: `distilgpt2`. Target model: `gpt2-medium`. Same tokenizer family, which speculative decoding requires.

## Correctness, proven not assumed

At near-zero temperature, both models become deterministic — always pick their own top token. In that regime speculative decoding is mathematically required to produce the *exact same token sequence* as plain greedy decoding from the target model alone. The test checks this with an exact tensor match:

```bash
python demo/test_correctness.py
```

```
EXACT MATCH: True
PASS: speculative decoding reduces to target-model greedy decoding, as required.
```

If that test doesn't pass, nothing below it should be trusted — this is what it means to actually verify an algorithm instead of eyeballing the output text.

## Honest benchmark results (CPU)

```bash
python demo/speculative_decoding.py
```

On CPU, with proper KV-caching on both the speculative and naive paths:

| Metric | Value |
|---|---|
| Acceptance rate | 51–80% across runs |
| Target-model forward passes | ~13–21, vs. 40–60 for naive |
| **Speedup** | **0.52x – 0.75x (slower than naive)** |

Speculative decoding is *slower* than plain generation on this hardware, and that's expected, not a bug — the technique's entire speed advantage comes from a GPU-specific fact: computing several sequence positions in one forward pass costs barely more than computing one, because a single-token forward pass leaves most of a GPU idle. A CPU has no such slack — computing 4 positions costs close to 4x, not ~1x — so the "verify several draft tokens at once" trick has nothing to exploit, while the draft model's extra sequential forward passes are pure added cost. Acceptance rate and forward-pass-count numbers above confirm the algorithm itself is doing exactly what it should; the hardware is just the wrong hardware for it to pay off.

The notebook auto-detects this: it runs the full 3-prompt × 100-token benchmark on a GPU, and automatically scales down to a smaller run on CPU so it still finishes in a reasonable time.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install torch transformers    # CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu

python demo/test_correctness.py       # correctness proof (~1 min on CPU)
python demo/speculative_decoding.py   # full benchmark + sample generation
```

Or open `notebooks/speculative_decoding_kaggle.ipynb` directly — it already contains real executed CPU output, and will run the larger GPU-scale benchmark automatically if you run it somewhere with CUDA available.

## What's *not* real yet

The `backend/`, `frontend/`, `k8s/`, `benchmark/`, `configs/`, `clients/`, `load_test/`, and `spec_vocab/` directories are the original auto-generated scaffold. They're left in place for reference but are **not wired to the working implementation above** — the FastAPI service has no working `/generate` endpoint, several modules reference methods that don't exist elsewhere in the scaffold, and none of it has been run. If this gets built out further, the plan is to fold the proven `demo/speculative_decoding.py` core into a real, working version of that service rather than trust the scaffold's existing files.

The "SpecVocab" dynamic-vocabulary-subset idea from the original scaffold (restricting the candidate vocabulary per step to cut softmax cost) is a separate, legitimate optimization on top of standard speculative decoding — not yet implemented here. What's implemented is correct, standard speculative decoding.
