from .draft_model import DraftModel
import torch

class TargetModel:
    """Main language model for final inference"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)