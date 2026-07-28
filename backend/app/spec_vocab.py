import torch
import math
from typing import List, Dict
from collections import defaultdict

class SpecVocabConfig:
    """Configuration for adaptive SpecVocab parameters"""
    def __init__(self,
                 adaptive_threshold: float = 0.8,
                 task_specific_scaling: Dict[str, float] = None,
                 size_adaptation_factors: Dict[str, float] = None,
                 min_vocab_subset: int = 512,
                 max_vocab_subset: int = 4096,
                 layer_wise_scaling: bool = True,
                 size_adaptation_strategy: str = 'linear'):
        self.adaptive_threshold = adaptive_threshold
        self.task_specific_scaling = task_specific_scaling or {
            'translation': 1.2,
            'summarization': 1.0,
            'code_generation': 0.9
        }
        self.size_adaptation_factors = size_adaptation_factors or {
            'small': 0.6,
            'medium': 1.0,
            'large': 1.4
        }
        self.min_vocab_subset = min_vocab_subset
        self.max_vocab_subset = max_vocab_subset
        self.layer_wise_scaling = layer_wise_scaling
        self.size_adaptation_strategy = size_adaptation_strategy

class SpecVocab:
    """Adaptive vocabulary manager with size and task-specific optimizations"""
    def __init__(self, config: SpecVocabConfig = None):
        self.config = config or SpecVocabConfig()
        self.vocab_cache = defaultdict(dict)
        self.adaptive_threshold = self.config.adaptive_threshold
        
    def get_adaptive_vocab(self,
                          hidden_states: torch.Tensor,
                          current_layer: int,
                          model_size: str,
                          task_type: str) -> torch.Tensor:
        """Dynamic vocabulary selection with multiple adaptation strategies"""
        # Apply model size adaptation
        size_factor = self.config.size_adaptation_factors.get(model_size, 1.0)
        
        # Apply task-specific scaling
        task_scale = self.config.task_specific_scaling.get(task_type, 1.0)
        
        # Calculate adaptive threshold
        dynamic_threshold = self._calculate_dynamic_threshold(
            hidden_states,
            current_layer,
            size_factor * task_scale
        )
        
        # Generate candidate tokens
        candidate_indices = self._generate_candidates(hidden_states, dynamic_threshold)
        
        # Apply size-based pruning
        optimized_vocab = self._adapt_vocab_size(candidate_indices, model_size)
        
        return optimized_vocab

    def _calculate_dynamic_threshold(self,
                                    hidden_states: torch.Tensor,
                                    layer: int,
                                    scale_factor: float) -> float:
        """Layer-aware threshold calculation with depth scaling"""
        if self.config.layer_wise_scaling:
            depth_factor = 1.0 + (layer / 12)  # Assuming 12-layer base model
        else:
            depth_factor = 1.0
            
        # Calculate entropy-based threshold
        entropy = self._calculate_entropy(hidden_states)
        return self.adaptive_threshold * scale_factor * depth_factor * (1 + entropy)

    def _generate_candidates(self,
                            hidden_states: torch.Tensor,
                            threshold: float) -> torch.Tensor:
        """Generate candidate tokens using adaptive scoring"""
        scores = torch.softmax(hidden_states, dim=-1)
        return torch.where(scores > threshold)[1].unique()

    def _adapt_vocab_size(self,
                         candidates: torch.Tensor,
                         model_size: str) -> torch.Tensor:
        """Adjust vocabulary subset based on model size strategy"""
        if self.config.size_adaptation_strategy == 'linear':
            target_size = int(len(candidates) * self.config.size_adaptation_factors[model_size])
        elif self.config.size_adaptation_strategy == 'exponential':
            target_size = int(len(candidates) ** self.config.size_adaptation_factors[model_size])
        else:
            target_size = len(candidates)

        target_size = max(self.config.min_vocab_subset,
                         min(target_size, self.config.max_vocab_subset))
        
        return candidates[:target_size]

    def _calculate_entropy(self, hidden_states: torch.Tensor) -> float:
        """Calculate normalized entropy of hidden states"""
        probs = torch.softmax(hidden_states, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        return entropy.item() / math.log(hidden_states.size(-1))  # Normalize

    def reset_cache(self):
        """Clear vocabulary cache"""
        self.vocab_cache.clear()