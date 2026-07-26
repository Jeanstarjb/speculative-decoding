import torch
from typing import List, Tuple
from models import DraftModel, TargetModel

class Eagle3Decoder:
    """Implementation of EAGLE-3 speculative decoding baseline"""

    def __init__(self, draft_model: DraftModel, target_model: TargetModel, 
                 max_length=100, temperature=1.0, lookahead=3):
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_length = max_length
        self.temperature = temperature
        self.lookahead = lookahead
        self.acceptance_rate = 0.0

    def generate(self, input_ids: torch.Tensor) -> List[List[int]]:
        """EAGLE-3 generation algorithm with multi-token lookahead"""
        sequences = input_ids.tolist()
        
        for _ in range(self.max_length - input_ids.size(1)):
            # Generate lookahead tokens from draft model
            draft_outputs = self.draft_model(input_ids)
            draft_probs = torch.softmax(draft_outputs[:, -1, :] / self.temperature, dim=-1)
            draft_tokens = torch.topk(draft_probs, self.lookahead, dim=-1).indices

            # Get target model probabilities
            with torch.no_grad():
                target_outputs = self.target_model(input_ids)
            target_probs = torch.softmax(target_outputs[:, -1, :] / self.temperature, dim=-1)

            # Verify and accept tokens
            accepted = []
            for i in range(self.lookahead):
                if target_probs[0, draft_tokens[0, i]] >= draft_probs[0, draft_tokens[0, i]]:
                    accepted.append(draft_tokens[0, i].item())
                else:
                    break
            
            # Update acceptance rate metric
            self.acceptance_rate = 0.9 * self.acceptance_rate + 0.1 * len(accepted)/self.lookahead

            if not accepted:
                break

            # Update input_ids with accepted tokens
            input_ids = torch.cat([
                input_ids, 
                torch.tensor([accepted], device=input_ids.device)
            ], dim=-1)
            
            sequences[0].extend(accepted)

        return sequences