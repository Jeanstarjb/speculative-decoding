import torch
import pytest
from models import DraftModel

@pytest.fixture
def sample_input():
    return torch.randint(0, 1000, (2, 10))

def test_draft_model_forward_shape():
    model = DraftModel(vocab_size=1000)
    input_ids = torch.randint(0, 1000, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, 1000)

def test_variable_length_processing(sample_input):
    model = DraftModel(vocab_size=1000)
    for length in [5, 10, 15]:
        input_ids = sample_input[:, :length]
        logits = model(input_ids)
        assert logits.shape == (2, length, 1000)

def test_embeddings_consistency():
    model = DraftModel(vocab_size=1000)
    input_ids = torch.LongTensor([[42, 24], [15, 73]])
    output = model(input_ids)
    
    # Check token and position embeddings are applied
    with torch.no_grad():
        embeds = model.token_embed(input_ids) + model.pos_embed(torch.arange(input_ids.size(1)))
        transformer_out = model.decoder_layer(embeds)
        manual_logits = torch.matmul(transformer_out, model.token_embed.weight.T)
    
    assert torch.allclose(output, manual_logits, atol=1e-5)

def test_batch_processing():
    model = DraftModel(vocab_size=1000)
    batch_sizes = [1, 2, 4]
    for bs in batch_sizes:
        input_ids = torch.randint(0, 1000, (bs, 8))
        logits = model(input_ids)
        assert logits.size(0) == bs

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_device_compatibility(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = DraftModel(vocab_size=1000).to(device)
    input_ids = torch.randint(0, 1000, (2, 10), device=device)
    logits = model(input_ids)
    assert logits.device.type == device
