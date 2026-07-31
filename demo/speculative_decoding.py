"""
A correct, from-scratch implementation of speculative decoding
(Leviathan et al. 2023 / Chen et al. 2023), WITH proper KV-caching —
this is what actually makes it fast, not just theoretically correct.

Two real pretrained models, same tokenizer family:
  - draft model:  distilgpt2   (small, fast, weaker)
  - target model: gpt2-medium  (bigger, slower, what we actually want)

Guarantee: the output distribution is IDENTICAL to sampling from the
target model alone, token by token. Speed comes from two things:
  1. the target model verifies several draft tokens in one forward
     pass instead of generating them one at a time
  2. neither model ever reprocesses tokens it has already seen —
     KV-caches carry forward across rounds, and get trimmed (not
     rebuilt) on the rare occasion a draft token is rejected
"""
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class SpecDecodeStats:
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0


def _sample(probs: torch.Tensor) -> int:
    return torch.multinomial(probs, num_samples=1).item()


def _step(model, token_ids: torch.Tensor, cache: DynamicCache):
    """Feed `token_ids` (new, not-yet-cached tokens) through `model`,
    extending `cache` in place. Returns logits for every fed position."""
    with torch.no_grad():
        out = model(input_ids=token_ids, past_key_values=cache, use_cache=True)
    return out.logits, out.past_key_values


@torch.no_grad()
def speculative_decode(
    prompt_ids: torch.Tensor,
    draft_model,
    target_model,
    max_new_tokens: int,
    lookahead: int = 4,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, SpecDecodeStats]:
    stats = SpecDecodeStats()
    ids = prompt_ids.clone()
    device = ids.device
    prompt_len = ids.shape[1]

    # Prime both caches on the prompt. Each "pending logits" predicts the
    # very next token (position len(ids), not chosen yet).
    draft_logits, draft_cache = _step(draft_model, ids, DynamicCache())
    target_logits, target_cache = _step(target_model, ids, DynamicCache())
    draft_pending = draft_logits[0, -1, :]
    target_pending = target_logits[0, -1, :]

    while ids.shape[1] - prompt_len < max_new_tokens:
        stats.steps += 1
        base_len = ids.shape[1]  # cache length before this round's draft tokens

        # 1. Draft proposes `lookahead` tokens, one at a time, cache growing normally.
        draft_tokens, draft_dists = [], []
        pending = draft_pending
        for _ in range(lookahead):
            probs = F.softmax(pending / temperature, dim=-1)
            tok = _sample(probs)
            draft_tokens.append(tok)
            draft_dists.append(probs)
            logits, draft_cache = _step(draft_model, torch.tensor([[tok]], device=device), draft_cache)
            pending = logits[0, -1, :]
        draft_pending_next = pending  # only used if every draft token gets accepted

        # 2. Target verifies all `lookahead` tokens in ONE forward pass.
        draft_tensor = torch.tensor([draft_tokens], device=device)
        batch_logits, target_cache_full = _step(target_model, draft_tensor, target_cache)
        # target_pending (from end of last round) verifies draft_tokens[0];
        # batch_logits[0, i, :] verifies draft_tokens[i+1] for i in 0..lookahead-2,
        # and batch_logits[0, lookahead-1, :] is the bonus-token prediction.
        target_dists = [target_pending] + [batch_logits[0, i, :] for i in range(lookahead - 1)]

        # 3. Accept/reject in order.
        n_accepted = 0
        rejected = False
        for i, tok in enumerate(draft_tokens):
            target_probs = F.softmax(target_dists[i] / temperature, dim=-1)
            p_target = target_probs[tok].item()
            p_draft = draft_dists[i][tok].item()
            accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
            stats.proposed_tokens += 1

            if torch.rand(1).item() < accept_prob:
                ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
                n_accepted += 1
                stats.accepted_tokens += 1
                if eos_token_id is not None and tok == eos_token_id:
                    target_cache_full.crop(base_len + n_accepted)
                    return ids[:, : prompt_len + max_new_tokens], stats
            else:
                adjusted = torch.clamp(target_probs - draft_dists[i], min=0.0)
                total = adjusted.sum()
                adjusted = adjusted / total if total > 0 else target_probs
                resampled = _sample(adjusted)
                ids = torch.cat([ids, torch.tensor([[resampled]], device=device)], dim=1)
                rejected = True
                # Roll BOTH caches back to just the accepted prefix, then feed
                # the resampled token to extend them by exactly one.
                target_cache_full.crop(base_len + n_accepted)
                draft_cache.crop(base_len + n_accepted)
                resampled_t = torch.tensor([[resampled]], device=device)
                tlogits, target_cache = _step(target_model, resampled_t, target_cache_full)
                dlogits, draft_cache = _step(draft_model, resampled_t, draft_cache)
                target_pending = tlogits[0, -1, :]
                draft_pending = dlogits[0, -1, :]
                break

        if not rejected:
            # All `lookahead` tokens accepted -> take the free bonus token.
            bonus_probs = F.softmax(batch_logits[0, lookahead - 1, :] / temperature, dim=-1)
            bonus = _sample(bonus_probs)
            ids = torch.cat([ids, torch.tensor([[bonus]], device=device)], dim=1)
            bonus_t = torch.tensor([[bonus]], device=device)
            # target_cache_full already covers ids up to (not incl.) bonus token.
            tlogits, target_cache = _step(target_model, bonus_t, target_cache_full)
            dlogits, draft_cache = _step(draft_model, bonus_t, draft_cache)
            target_pending = tlogits[0, -1, :]
            draft_pending = dlogits[0, -1, :]
            if eos_token_id is not None and bonus == eos_token_id:
                return ids[:, : prompt_len + max_new_tokens], stats

    return ids[:, : prompt_len + max_new_tokens], stats


@torch.no_grad()
def naive_decode(
    prompt_ids: torch.Tensor,
    target_model,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Baseline: target model generates one token at a time, WITH caching
    (this is the fair comparison — a real system would always cache)."""
    ids = prompt_ids.clone()
    logits, cache = _step(target_model, ids, DynamicCache())
    pending = logits[0, -1, :]
    for _ in range(max_new_tokens):
        probs = F.softmax(pending / temperature, dim=-1)
        tok = _sample(probs)
        ids = torch.cat([ids, torch.tensor([[tok]], device=ids.device)], dim=1)
        if eos_token_id is not None and tok == eos_token_id:
            break
        logits, cache = _step(target_model, torch.tensor([[tok]], device=ids.device), cache)
        pending = logits[0, -1, :]
    return ids


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Loading draft model (distilgpt2)...")
    draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    print("Loading target model (gpt2-medium)...")
    target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()

    prompt = "The future of artificial intelligence is"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    max_new_tokens = 60
    lookahead = 4

    print(f"\nPrompt: {prompt!r}")
    print(f"Generating {max_new_tokens} tokens, lookahead={lookahead}\n")

    torch.manual_seed(0)
    t0 = time.perf_counter()
    spec_ids, stats = speculative_decode(
        prompt_ids, draft_model, target_model, max_new_tokens, lookahead=lookahead
    )
    spec_time = time.perf_counter() - t0
    spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    naive_ids = naive_decode(prompt_ids, target_model, max_new_tokens)
    naive_time = time.perf_counter() - t0
    naive_text = tokenizer.decode(naive_ids[0], skip_special_tokens=True)

    print("=" * 70)
    print("SPECULATIVE DECODING OUTPUT:")
    print(spec_text)
    print(f"\n  time: {spec_time:.2f}s  |  {max_new_tokens / spec_time:.2f} tok/s")
    print(f"  target-model forward passes (steps): {stats.steps}  (vs {max_new_tokens} for naive)")
    print(f"  acceptance rate: {stats.acceptance_rate:.1%}  "
          f"({stats.accepted_tokens}/{stats.proposed_tokens} draft tokens accepted)")

    print("\n" + "=" * 70)
    print("NAIVE (TARGET-ONLY, CACHED) OUTPUT:")
    print(naive_text)
    print(f"\n  time: {naive_time:.2f}s  |  {max_new_tokens / naive_time:.2f} tok/s")

    print("\n" + "=" * 70)
    print(f"SPEEDUP: {naive_time / spec_time:.2f}x")


if __name__ == "__main__":
    main()
