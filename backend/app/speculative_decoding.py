import torch
from typing import List, Tuple
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
        """Generate sequence using speculative decoding pipeline"""
        current_ids = input_ids.clone()
        for _ in range(self.max_length):
            # Generate draft candidates
            with torch.no_grad():
                draft_logits = self.draft_model(current_ids)
                draft_probs = torch.softmax(draft_logits[:, -1, :] / self.temperature, dim=-1)
                draft_tokens = torch.multinomial(draft_probs, num_samples=1)

            # Get target model verification
            combined_ids = torch.cat([current_ids, draft_tokens], dim=1)
            target_logits = self.target_model(combined_ids)
            target_probs = torch.softmax(target_logits[:, -1, :] / self.temperature, dim=-1)

            # Validate and accept tokens
            accepted = draft_tokens
            if not torch.all(torch.isclose(draft_probs, target_probs, atol=0.01)):
                mismatch = torch.argmax(draft_probs != target_probs)
                accepted = draft_tokens[:, :mismatch+1]

            current_ids = torch.cat([current_ids, accepted], dim=1)
            
            # Update vocabulary frequencies
            self.spec_vocab.update_frequencies(accepted.squeeze().tolist())

            if accepted[0, -1] in [0, 1]:  # Stop tokens
                break

        return current_ids.squeeze().tolist()