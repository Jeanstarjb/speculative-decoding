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
| [`app.py`](app.py) | Interactive Streamlit demo — type a prompt, see speculative vs. naive decoding side by side with real live stats |
| [`backend/app/main.py`](backend/app/main.py) | A real FastAPI service wrapping the same core (see below) |

Draft/target model pairs used throughout: `distilgpt2` → `gpt2` (light, ~200M total, the default) or `distilgpt2` → `gpt2-medium` (~440M total). Same tokenizer family in both cases, which speculative decoding requires.

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

One more real data point, from the interactive demo using a smaller target model (`distilgpt2` → `gpt2`, both under the light model-pair option): **0.21x**, an even bigger CPU slowdown than the `gpt2-medium` pairing. That's consistent with the explanation above — with a smaller target model, the draft model's own per-round overhead is a *larger* fraction of the total cost, so the CPU penalty gets worse, not better, as the target model shrinks. The technique needs a target model expensive enough (and a GPU parallel enough) for verifying several tokens at once to actually be cheap.

## Interactive demo

```bash
pip install streamlit
streamlit run app.py
```

Type a prompt, pick a model pair and settings in the sidebar, hit Generate. Shows speculative and naive output side by side with real timing, tokens/sec, target-model forward-pass count, and acceptance rate — and an honest, dynamic note about why the speedup number will look bad on CPU and should look good on GPU.

Defaults to the lighter `distilgpt2` → `gpt2` pairing (~200M params total) so it stays usable on free CPU-only hosting (e.g. Streamlit Community Cloud); the original `distilgpt2` → `gpt2-medium` pairing is available from the sidebar.

## API (FastAPI + Docker)

A real HTTP service wrapping the same proven core — no reimplementation, no Redis, no Kubernetes (deliberately: not needed to serve one model to one demo, would just be complexity for its own sake).

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

or with Docker:

```bash
docker compose up --build
```

Both `GET /health` and `POST /generate` are real and tested (`backend/tests/test_api.py`, run against the actual app with `pytest`, not mocked). Verified with a live server and real `curl` requests, not just the test suite:

```bash
$ curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
    -d '{"prompt": "The best way to learn programming is", "max_new_tokens": 15, "compare_to_naive": true}'

{"text":"The best way to learn programming is to study seven different languages and then practice
your favorite patterns or eliminate certain ideas","elapsed_seconds":5.47,"tokens_per_second":2.74,
"target_forward_passes":5,"acceptance_rate":0.77,"device":"cpu","naive_elapsed_seconds":1.85,
"naive_text":"The best way to learn programming is to study it. You need to actively \"tree\" your
learning through Innov","speedup":0.34}
```

That `speedup: 0.34` is the same honest CPU story as everywhere else in this README — real, unedited output, not cherry-picked.

Interactive API docs (via FastAPI's auto-generated Swagger UI) are at `/docs` once the server is running.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install torch transformers    # CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu

python demo/test_correctness.py       # correctness proof (~1 min on CPU)
python demo/speculative_decoding.py   # full benchmark + sample generation
```

Or open `notebooks/speculative_decoding_kaggle.ipynb` directly — it already contains real executed CPU output, and will run the larger GPU-scale benchmark automatically if you run it somewhere with CUDA available.

## What happened to the rest of the original scaffold

The original auto-generated scaffold also included `frontend/` (a React shell, superseded by the working Streamlit demo), `k8s/`, `benchmark/`, `configs/`, `clients/`, `load_test/`, and `spec_vocab/` directories, plus a top-level `tests/` that mocked out the very code it claimed to test. All of it was either non-functional (imports that didn't exist, a `run_benchmarks.py` that imported a `time` module it never actually imported) or made redundant by the real work above, so it's been removed rather than left to confuse anyone browsing the repo. `git log` has the full history if any of it is ever useful for reference.

Deliberately still not implemented: the "SpecVocab" dynamic-vocabulary-subset idea from the original repo name (restricting the candidate vocabulary per step to cut softmax cost). That's a separate, legitimate optimization on top of standard speculative decoding — not yet built. What's implemented and tested here is correct, standard speculative decoding.
