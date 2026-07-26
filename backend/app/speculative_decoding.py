import torch
from typing import List
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab

class SpeculativeDecoder:
    """Implements speculative decoding with dynamic vocabulary selection"""

    def __init__(self, draft_model: DraftModel, target_model: TargetModel, 
                 spec_vocab: SpecVocab, max_length=100, temperature=1.0):
        self.draft_model = draft_model
        self.target_model = target_model
        self.spec_vocab = spec_vocab
        self.max_length = max_length
        self.temperature = temperature

    def generate(self, input_ids: torch.Tensor) -> List[int]:
        generated = input_ids[0].tolist()
        for _ in range(self.max_length):
            draft_logits = self.draft_model(input_ids)
            draft_probs = torch.softmax(draft_logits[:, -1, :]/self.temperature, -1)
            top_n = torch.topk(draft_probs, self.spec_vocab.top_n, -1).indices[0].tolist()
            
            candidates = self.spec_vocab.get_candidates(top_n)
            target_logits = self.target_model(input_ids)
            target_probs = torch.softmax(target_logits[:, -1, :]/self.temperature, -1)
            
            next_token = self._select_token(candidates, draft_probs, target_probs)
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], -1)
            self.spec_vocab.update_frequencies([next_token])
        return generated

    def _select_token(self, candidates, draft_probs, target_probs):
        best_token = candidates[0]
        max_ratio = -1
        for token in candidates:
            ratio = target_probs[0, token].item() / (draft_probs[0, token].item() + 1e-8)
            if ratio >= self.spec_vocab.threshold and ratio > max_ratio:
                max_ratio = ratio
                best_token = token
        return best_token if max_ratio >= self.spec_vocab.threshold else target_probs.argmax(-1).item()