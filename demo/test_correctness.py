"""
Correctness check: at very low temperature, both models become
near-deterministic (always pick their own top token). In that regime,
speculative decoding must reduce to exactly the target model's own
greedy output -- if it doesn't, the accept/reject math has a bug.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import speculative_decode, naive_decode

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()

prompt = "Once upon a time"
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
max_new_tokens = 30
temperature = 1e-4  # near-greedy

torch.manual_seed(0)
spec_ids, stats = speculative_decode(
    prompt_ids, draft_model, target_model, max_new_tokens,
    lookahead=4, temperature=temperature,
)

torch.manual_seed(0)
naive_ids = naive_decode(prompt_ids, target_model, max_new_tokens, temperature=temperature)

spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)
naive_text = tokenizer.decode(naive_ids[0], skip_special_tokens=True)

print("Speculative:", spec_text)
print("Naive greedy:", naive_text)
print("Acceptance rate at near-zero temp:", f"{stats.acceptance_rate:.1%}")

match = torch.equal(spec_ids, naive_ids)
print("\nEXACT MATCH:", match)
if not match:
    print("spec_ids: ", spec_ids.tolist())
    print("naive_ids:", naive_ids.tolist())
    raise SystemExit(1)
print("PASS: speculative decoding reduces to target-model greedy decoding, as required.")
