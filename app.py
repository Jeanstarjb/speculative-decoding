"""
Interactive demo for the speculative decoding implementation in
demo/speculative_decoding.py. Imports that module directly -- this app
never reimplements the algorithm, so it can't drift from the version
that's actually been correctness-tested.
"""
import time

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo.speculative_decoding import naive_decode, speculative_decode

MODEL_PAIRS = {
    "Light (distilgpt2 -> gpt2, ~200M total, safe on free CPU hosting)": ("distilgpt2", "gpt2"),
    "Original (distilgpt2 -> gpt2-medium, ~440M total, heavier)": ("distilgpt2", "gpt2-medium"),
}

st.set_page_config(page_title="Speculative Decoding Demo", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner=False)
def load_models(draft_name: str, target_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    draft = AutoModelForCausalLM.from_pretrained(draft_name).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(target_name).to(device).eval()
    return tokenizer, draft, target


st.title("⚡ Speculative Decoding")
st.caption(
    "A small **draft model** proposes several tokens at once; the big **target model** "
    "verifies them all in one forward pass and accepts or rejects each one with the exact "
    "probability that guarantees the output matches the target model alone. "
    "[Correctness of this implementation is proven by an exact-match test.]"
    "(https://github.com/Jeanstarjb/speculative-decoding#correctness)"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.warning(
        "Running on CPU. Speculative decoding's speedup comes from a GPU-specific fact — "
        "verifying several token positions in one forward pass costs barely more than one, "
        "because a single-token pass leaves most of a GPU idle. CPUs have no such slack, so "
        "**this demo will likely be the same speed or slower than plain generation here** — "
        "that's expected, not a bug. See the README for the full explanation and a real CPU "
        "benchmark. The acceptance rate and forward-pass counts below still show the algorithm "
        "working correctly."
    )
else:
    st.success(f"Running on GPU ({torch.cuda.get_device_name(0)}) — this is where the technique should actually pay off.")

with st.sidebar:
    st.header("Settings")
    pair_label = st.selectbox("Model pair", list(MODEL_PAIRS.keys()))
    draft_name, target_name = MODEL_PAIRS[pair_label]
    max_new_tokens = st.slider("Tokens to generate", 10, 100, 30, step=10)
    lookahead = st.slider("Lookahead (draft tokens per round)", 1, 8, 4)
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, step=0.1)
    seed = st.number_input("Random seed", value=0, step=1)

prompt = st.text_area("Prompt", value="The future of artificial intelligence is", height=80)
run = st.button("Generate", type="primary")

if run:
    with st.spinner(f"Loading {draft_name} (draft) and {target_name} (target)..."):
        tokenizer, draft_model, target_model = load_models(draft_name, target_name, device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    col1, col2 = st.columns(2)

    with st.spinner("Running speculative decoding..."):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        spec_ids, stats = speculative_decode(
            prompt_ids, draft_model, target_model, max_new_tokens,
            lookahead=lookahead, temperature=temperature,
        )
        spec_time = time.perf_counter() - t0
        spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)

    with st.spinner("Running naive (cached) decoding..."):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        naive_ids = naive_decode(prompt_ids, target_model, max_new_tokens, temperature=temperature)
        naive_time = time.perf_counter() - t0
        naive_text = tokenizer.decode(naive_ids[0], skip_special_tokens=True)

    with col1:
        st.subheader("Speculative decoding")
        st.write(spec_text)
        st.metric("Time", f"{spec_time:.2f}s", f"{max_new_tokens/spec_time:.1f} tok/s")
        st.metric("Target-model forward passes", stats.steps, f"vs {max_new_tokens} for naive")
        st.metric("Draft token acceptance rate", f"{stats.acceptance_rate:.1%}")

    with col2:
        st.subheader("Naive decoding (target model only, cached)")
        st.write(naive_text)
        st.metric("Time", f"{naive_time:.2f}s", f"{max_new_tokens/naive_time:.1f} tok/s")

    st.divider()
    speedup = naive_time / spec_time
    if speedup >= 1:
        st.success(f"**Speedup: {speedup:.2f}x** faster than naive decoding.")
    else:
        st.info(
            f"**Speedup: {speedup:.2f}x** (slower than naive) — expected on CPU. "
            f"The {stats.acceptance_rate:.0%} acceptance rate shows the draft model's guesses "
            f"are genuinely being verified and used; on a GPU this same run would be faster, "
            f"not just theoretically but because the wasted parallel capacity a single-token "
            f"pass leaves behind is exactly what batched verification uses instead."
        )
