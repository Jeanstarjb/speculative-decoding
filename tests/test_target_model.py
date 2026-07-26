import torch
from models import TargetModel

def test_target_model_forward():
    vocab_size = 32000
    d_model = 1024
    model = TargetModel(vocab_size=vocab_size, d_model=d_model)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert not torch.allclose(logits, torch.zeros_like(logits))

def test_sequence_processing():
    model = TargetModel(vocab_size=1000)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    output = model(input_ids)
    assert output.requires_grad == False  # Ensure inference mode
    assert output.dtype == torch.float32