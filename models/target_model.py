import torch
import torch.nn as nn

class TargetModel(nn.Module):
    """Full transformer-based language model with multiple decoder layers"""

    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16, 
                 num_layers: int = 12, max_seq_len: int = 2048):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer('position_ids', torch.arange(max_seq_len).unsqueeze(0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        positions = self.position_ids[:, :seq_length]
        
        x = self.token_embed(input_ids)
        x += self.pos_embed(positions)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.fc_out(x)
        return logits