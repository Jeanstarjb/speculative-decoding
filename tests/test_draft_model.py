import torch
from models import DraftModel

def test_draft_model_forward():
    vocab_size = 10000
    d_model = 512
    model = DraftModel(vocab_size=vocab_size, d_model=d_model)
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)