import torch
import torch.nn as nn

class DraftModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.decoder_layer(x)
        logits = self.output_proj(x)
        return logits