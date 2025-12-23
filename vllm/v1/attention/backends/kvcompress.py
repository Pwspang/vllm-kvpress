import torch

# from vllm.v1.logprobs_store import global_logprobs
from typing import Callable

class KVPress:
    def __init__(
        self,
        callback: Callable,
        **kwargs,
    ):
        # for recording kept token indices
        self.evicted_token_num = 0
        self.kept_token_indices = torch.tensor([])
        self.index = 0
        self.steps = 0
        self.get_indices = callback
        

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        req_id: int,
    ):
        # This should just run per request only
        # Key Shape: [1, Num Heads, Num Tokens, Dimension]
        # Query Shape: [1, Num Heads, 1, Dimension]
        # Value Shape: [1, Num Heads, Num Tokens, Dimension]
                
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        self.steps += 1
        
        
        # Comment this out if you dont need to prune under a certain budget
        if kv_cache_len < 30:
            return key_states, value_states
        
        # Run this every some steps to reduce overhead
        if self.steps % 10 != 0:
            return key_states, value_states
        
        # Get extend mapping for current indices
        self.extend_indices(kv_cache_len, key_states.device)
        
        # Callback function to return a list of indices to keep
        # Indices should be the indices of overall token position
        indices: torch.Tensor = self.get_indices(req_id, self.kept_token_indices)
        
        # Remap back overall token position to kept_token_indices position
        mask = (self.kept_token_indices.unsqueeze(1) == indices).any(dim=1)
        indices_map = torch.nonzero(mask, as_tuple=True)[0]
        
        self.kept_token_indices = self.kept_token_indices[indices_map]
        
        # print(f"MAP: {indices_map}")
        # print(f"KEPT: {self.kept_token_indices}")
        
        
        # This should collect across all the heads
        key_states = key_states[:, :, indices_map, :]
        value_states = value_states[:, :, indices_map, :]
        
        return key_states, value_states
    
    def to_keep(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compare every element of A with B
        mask = (A.unsqueeze(1) == B.unsqueeze(0))  # shape: [len(A), len(B)]

        # Find indices in A where any match occurs
        indices = torch.nonzero(mask.any(dim=1)).squeeze(1)
        return indices
    
    def extend_indices(self, kv_cache_len: int, device) -> None:
        # Reset if kv_cache_len smaller than kept_index, this means it is new request 
        if kv_cache_len < self.kept_token_indices.numel():
            self.kept_token_indices = torch.tensor([])
            self.index = 0
        
        if self.kept_token_indices.numel() == 0:
            # Initialization for mapping
            self.kept_token_indices = torch.arange(kv_cache_len, device=device)
            self.index = self.kept_token_indices[-1]
        else:
            # Calculate how many new tokens to add
            num_new = kv_cache_len - self.kept_token_indices.numel()
            if num_new > 0:
                new_idx = torch.arange(
                    self.index + 1,
                    self.index + 1 + num_new,
                    device=self.kept_token_indices.device,
                )
                self.index = new_idx[-1]
                self.kept_token_indices = torch.cat([self.kept_token_indices, new_idx])
            # else: nothing to extend