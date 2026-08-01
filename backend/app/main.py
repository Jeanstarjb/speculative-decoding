"""
A real FastAPI service wrapping the proven speculative_decode() core
from demo/speculative_decoding.py -- no reimplementation, no Redis, no
Kubernetes. Just the actual algorithm, served over HTTP.
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the repo-root `demo` package importable regardless of how this
# service is launched (uvicorn CLI, Docker CMD, pytest).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.speculative_decoding import naive_decode, speculative_decode  # noqa: E402

from .schemas import GenerateRequest, GenerateResponse, HealthResponse  # noqa: E402

MODELS: dict = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DRAFT_MODEL_NAME = "distilgpt2"
TARGET_MODEL_NAME = "gpt2"  # light pair, matches app.py's default -- fast to load, cheap to run


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODELS["tokenizer"] = AutoTokenizer.from_pretrained("gpt2")
    MODELS["draft"] = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME).to(DEVICE).eval()
    MODELS["target"] = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).to(DEVICE).eval()
    yield
    MODELS.clear()


app = FastAPI(
    title="Speculative Decoding API",
    description=(
        "Serves a correctness-tested speculative decoding implementation "
        "(see github.com/Jeanstarjb/speculative-decoding). "
        "Draft model proposes tokens, target model verifies them, and the "
        "accept/reject sampling guarantees output matching the target "
        "model alone -- this API exposes exactly that, nothing more."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if "draft" in MODELS else "loading",
        device=DEVICE,
        draft_model=DRAFT_MODEL_NAME,
        target_model=TARGET_MODEL_NAME,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "draft" not in MODELS:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly.")
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")

    tokenizer = MODELS["tokenizer"]
    prompt_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(DEVICE)

    torch.manual_seed(req.seed)
    t0 = time.perf_counter()
    ids, stats = speculative_decode(
        prompt_ids, MODELS["draft"], MODELS["target"], req.max_new_tokens,
        lookahead=req.lookahead, temperature=req.temperature,
    )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(ids[0], skip_special_tokens=True)

    response = GenerateResponse(
        text=text,
        elapsed_seconds=elapsed,
        tokens_per_second=req.max_new_tokens / elapsed,
        target_forward_passes=stats.steps,
        acceptance_rate=stats.acceptance_rate,
        device=DEVICE,
    )

    if req.compare_to_naive:
        torch.manual_seed(req.seed)
        t0 = time.perf_counter()
        naive_ids = naive_decode(prompt_ids, MODELS["target"], req.max_new_tokens, temperature=req.temperature)
        naive_elapsed = time.perf_counter() - t0
        response.naive_elapsed_seconds = naive_elapsed
        response.naive_text = tokenizer.decode(naive_ids[0], skip_special_tokens=True)
        response.speedup = naive_elapsed / elapsed

    return response
