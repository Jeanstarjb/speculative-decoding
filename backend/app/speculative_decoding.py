import torch
from typing import List, Tuple
from models import DraftModel, TargetModel
from spec_vocab import SpecVocab
from .caching import RedisCache
import time

class SpeculativeDecoder:
    """Implements speculative decoding with dynamic vocabulary selection"""

    def __init__(self, draft_model: DraftModel, target_model: TargetModel, 
                 spec_vocab: SpecVocab, max_length=100, temperature=1.0,
                 cache: RedisCache | None = None, monitoring=None):
        self.draft_model = draft_model
        self.target_model = target_model
        self.spec_vocab = spec_vocab
        self.max_length = max_length
        self.temperature = temperature
        self.cache = cache
        self.monitoring = monitoring
        self.eos_token_id = 2

    def generate(self, input_ids: torch.Tensor) -> List[List[int]]:
        batch_size = input_ids.shape[0]
        original_len = input_ids.shape[1]
        output_ids = input_ids.tolist()
        eos = [False] * batch_size
        start_time = time.time()

        for _ in range(self.max_length):
            if all(eos):
                break
            
            # Draft phase
            draft_input = torch.tensor([ids[-self.draft_model.max_seq_len:] 
                                      for ids in output_ids]).to(self.draft_model.device)
            
            draft_start = time.time()
            with torch.no_grad():
                draft_logits = self.draft_model(draft_input)
            draft_time = time.time() - draft_start
            
            # Get speculative vocabulary
            spec_vocab_ids = self.spec_vocab.get_vocab(draft_logits)
            
            # Target verification
            target_start = time.time()
            with torch.no_grad():
                target_input = torch.tensor([ids[-self.target_model.max_seq_len:]
                                            for ids in output_ids]).to(self.target_model.device)
                target_logits = self.target_model(target_input)
            target_time = time.time() - target_start
            
            if self.monitoring:
                self.monitoring.record_latency(draft_time + target_time)
            
            # Process verification and update output
            for i in range(batch_size):
                if eos[i]:
                    continue
                
                draft_probs = torch.softmax(draft_logits[i, -1] / self.temperature, dim=-1)
                target_probs = torch.softmax(target_logits[i, -1] / self.temperature, dim=-1)
                
                draft_top_token = torch.argmax(draft_probs).item()
                selected_token = self._select_token(draft_probs, target_probs, spec_vocab_ids[i])
                
                if self.monitoring:
                    if selected_token == draft_top_token:
                        self.monitoring.increment_accepted_tokens(1)
                    self.monitoring.increment_speculated_tokens(1)
                
                if selected_token == self.eos_token_id:
                    eos[i] = True
                output_ids[i].append(selected_token)
                
                self.spec_vocab.update_frequencies([selected_token])
                
                if self.cache:
                    cache_key = self.cache._tensor_hash(target_input[i])
                    self.cache.set(cache_key, target_logits[i].cpu())
        
        if self.monitoring:
            generated_tokens = sum(len(seq) - original_len for seq in output_ids)
            self.monitoring.record_request_metrics(generated_tokens, time.time() - start_time)
        
        return output_ids

    def _select_token(self, draft_probs, target_probs, spec_vocab_ids):
        # Existing implementation
        return torch.argmax(target_probs).item()