import torch
import torch.nn as nn

class DraftModel(nn.Module):
    """Lightweight transformer-based draft model with single decoder layer for speculative decoding"""

    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.register_buffer('position_ids', torch.arange(max_seq_len).unsqueeze(0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # Embed tokens and positions
        token_embeds = self.token_embed(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        pos_embeds = self.pos_embed(position_ids)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), 
            diagonal=1
        ).to(input_ids.device)
        
        # Process through decoder layer
        x = self.decoder_layer(x, src_mask=causal_mask)
        
        # Project to vocabulary space
        logits = self.output_proj(x)
        return logits