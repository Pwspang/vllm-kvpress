import torch

def get_indices(req_id: str, kept_token_indices: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # Should place final tensor at device location, so i dont have to move tensor around
    
    # Compress to budget of 30 
    # rand_idx = torch.randperm(kept_token_indices.size(0), device=kept_token_indices.device)[:30]
    if kept_token_indices.size(0) < 40:
        return kept_token_indices
    # print(kept_token_indices)
    return torch.arange(30, device=kept_token_indices.device)