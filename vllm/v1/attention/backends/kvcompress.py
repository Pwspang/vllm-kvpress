import torch

from vllm.v1.logprobs_store import global_logprobs

class KVPress:
    def __init__(
        self,
        budget=50,
        **kwargs,
    ):
        self.budget = budget

        # for recording kept token indices
        self.evicted_token_num = 0
        self.kept_token_indices = torch.tensor([])
        self.index = 0
        self.steps = 0

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        # This should just run per request only
        # Key Shape: [1, Num Heads, Num Tokens, Dimension]
        # Query Shape: [1, Num Heads, 1, Dimension]
        # Value Shape: [1, Num Heads, Num Tokens, Dimension]
        # logger.info(f"[Key Size] {key_states.shape} [Query Size] {query_states.shape} [Value Size] {value_states.shape}")
        
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        self.steps += 1
        
        # Dont prune anything
        if kv_cache_len < self.budget:
            self.kept_token_indices = torch.tensor([])
            return key_states, value_states
        
        if self.steps % 40 != 0:
            return key_states, value_states

        # print(f"[KEY CACHE] {key_states.shape}")
        
            
        # Get extend mapping for current indices
        self.extend_indices(kv_cache_len, key_states.device)
        
        logprob = torch.tensor(global_logprobs.get_all(), device=key_states.device)
        logprob_index = torch.index_select(logprob, dim=0, index=self.kept_token_indices)

        # # print(f"[Logprob] {logprob_index}")
                
        top_values, top_indices = torch.topk(logprob_index, self.budget)
         
        indices = self.to_keep(self.kept_token_indices, top_indices)  
        
        # # print("[KEPT TOKEN BEFORE]")
        # # print(self.kept_token_indices)
        
        self.kept_token_indices = self.kept_token_indices[indices]
        
        # print("[KEPT TOKEN AFTER]" )
        # print(self.kept_token_indices)
        
        key_states = key_states[:, :, indices, :]
        value_states = value_states[:, :, indices, :]
        
        # print(f"[KEY CACHE] {key_states.shape}")
        
        return key_states, value_states
    
    def to_keep(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compare every element of A with B
        mask = (A.unsqueeze(1) == B.unsqueeze(0))  # shape: [len(A), len(B)]

        # Find indices in A where any match occurs
        indices = torch.nonzero(mask.any(dim=1)).squeeze(1)
        return indices
    
    def extend_indices(self, kv_cache_len: int, device) -> None:
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