# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L2 Norm Cache for storing and retrieving L2 norms of attention keys.

This module provides a thread-safe cache for storing L2 norms computed
during attention forward passes. The norms can be retrieved via the API
for use in KV cache eviction strategies.
"""

import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import torch
import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


# Add this near the top of your file
MAX_SEQ_LEN = 34000

@dataclass
class RequestL2NormData:
    """Stores L2 norm data using a pre-allocated fixed-size buffer."""
    # Pre-allocated buffer (approx 120KB per request)
    buffer: torch.Tensor = field(
        default_factory=lambda: torch.zeros(MAX_SEQ_LEN, dtype=torch.float32, device='cpu')
    )
    # Tracks the actual valid sequence length
    current_seq_len: int = 0
    num_layers_accumulated: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, new_norms: torch.Tensor):
        # Handle input shape and device
        if new_norms.dim() > 1:
            new_norms = new_norms.mean(dim=-1)
        new_norms = new_norms.detach().cpu()
        
        new_len = new_norms.shape[0]

        with self._lock:
            # Safety truncate
            if new_len > MAX_SEQ_LEN:
                new_norms = new_norms[:MAX_SEQ_LEN]
                new_len = MAX_SEQ_LEN

            # Initialization
            if self.num_layers_accumulated == 0:
                self.buffer[:new_len].copy_(new_norms)
                self.current_seq_len = new_len
                self.num_layers_accumulated = 1
                return

            # Update overlapping part (Running Average)
            overlap_len = min(self.current_seq_len, new_len)
            n = self.num_layers_accumulated
            
            if overlap_len > 0:
                # In-place update: (old * n + new) / (n + 1)
                self.buffer[:overlap_len].mul_(n).add_(new_norms[:overlap_len]).div_(n + 1)

            # Handle new tokens (Append)
            if new_len > self.current_seq_len:
                self.buffer[self.current_seq_len : new_len].copy_(
                    new_norms[self.current_seq_len:]
                )
                self.current_seq_len = new_len

            self.num_layers_accumulated += 1

    def get_norms(self, start_index: int = 0) -> List[float]:
        with self._lock:
            if start_index >= self.current_seq_len:
                return []
            return self.buffer[start_index:self.current_seq_len].tolist()


class L2NormCache:
    """
    Global cache for L2 norms of attention keys per request.
    
    This is a singleton that stores L2 norm data computed during attention
    forward passes. The data can be retrieved via the API for eviction decisions.
    
    Supports filtering by layer indices via `l2_norm_layers` configuration.
    """
    
    _instance: Optional['L2NormCache'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_cache()
        return cls._instance
    
    def _init_cache(self):
        """Initialize the cache."""
        self._request_data: Dict[str, RequestL2NormData] = {}
        self._data_lock = threading.Lock()
        self._enabled = True
        # Layer filtering: None means use all layers, otherwise use only specified layers
        self._l2_norm_layers: Optional[List[int]] = None
        self._skip_layers: Optional[List[int]] = None  # Alternative: skip these layers
    
    def enable(self):
        """Enable L2 norm computation."""
        self._enabled = True
        logger.info("L2 norm cache enabled")
    
    def disable(self):
        """Disable L2 norm computation."""
        self._enabled = False
        logger.info("L2 norm cache disabled")
    
    @property
    def is_enabled(self) -> bool:
        """Check if L2 norm computation is enabled."""
        return self._enabled
    
    def set_l2_norm_layers(self, layers: Optional[List[int]]):
        """
        Set which layers to use for L2 norm computation.
        
        Args:
            layers: List of layer indices to use for L2 norm computation.
                    If None, all layers are used.
        """
        self._l2_norm_layers = layers
        if layers is not None:
            logger.info(f"L2 norm computation restricted to layers: {layers}")
        else:
            logger.info("L2 norm computation enabled for all layers")
    
    def set_skip_layers(self, layers: Optional[List[int]]):
        """
        Set which layers to skip for L2 norm computation (like skip_layers in l2_compress.py).
        
        Args:
            layers: List of layer indices to skip.
                    If None, no layers are skipped.
        """
        self._skip_layers = layers
        if layers is not None:
            logger.info(f"L2 norm computation will skip layers: {layers}")
        else:
            logger.info("L2 norm computation will not skip any layers")
    
    def should_compute_for_layer(self, layer_idx: int) -> bool:
        """
        Check if L2 norms should be computed for a given layer.
        
        Args:
            layer_idx: The layer index
            
        Returns:
            True if L2 norms should be computed for this layer
        """
        if not self._enabled:
            return False
        
        # Check skip_layers first (higher priority)
        if self._skip_layers is not None and layer_idx in self._skip_layers:
            return False
        
        # Check l2_norm_layers (if specified, only compute for these layers)
        if self._l2_norm_layers is not None:
            return layer_idx in self._l2_norm_layers
        
        return True
    
    @property
    def l2_norm_layers(self) -> Optional[List[int]]:
        """Get the list of layers to use for L2 norm computation."""
        return self._l2_norm_layers
    
    @property
    def skip_layers(self) -> Optional[List[int]]:
        """Get the list of layers to skip for L2 norm computation."""
        return self._skip_layers
    
    def get_or_create_request(self, request_id: str) -> RequestL2NormData:
        """Get or create L2 norm data for a request."""
        with self._data_lock:
            if request_id not in self._request_data:
                self._request_data[request_id] = RequestL2NormData()
            return self._request_data[request_id]
    
    def update_norms(
        self,
        request_id: str,
        key_norms: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        req_indices: Optional[List[int]] = None,
        block_size: int = 16,
    ):
        """
        Update L2 norms for a request from attention layer computation.
        
        Args:
            request_id: The request ID
            key_norms: L2 norms of keys [seq_len, num_kv_heads] or [seq_len]
            seq_lens: Sequence lengths (optional)
            req_indices: Request indices in batch (optional)
            block_size: KV cache block size
        """
        if not self._enabled:
            return
        
        request_data = self.get_or_create_request(request_id)
        request_data.update(key_norms, seq_lens or torch.tensor([key_norms.shape[0]]),
                           req_indices or [0], block_size)
    
    def update_norms_batch(
        self,
        request_ids: List[str],
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        layer_idx: Optional[int] = None,
    ):
        if not self._enabled or key_cache is None:
            return
            
        if layer_idx is not None and not self.should_compute_for_layer(layer_idx):
            return

        try:
            # Filter for active requests
            active_indices = [i for i, rid in enumerate(request_ids) 
                              if rid is not None and seq_lens[i].item() > 0]
            if not active_indices:
                return

            for idx in active_indices:
                req_id = request_ids[idx]
                seq_len = int(seq_lens[idx].item())
                
                # Get valid blocks
                num_blocks = (seq_len + block_size - 1) // block_size
                block_indices = block_table[idx, :num_blocks]
                valid_mask = block_indices >= 0
                if not valid_mask.any():
                    continue
                valid_blocks = block_indices[valid_mask]

                # 1. Gather blocks [num_blocks, block_size, heads, head_size]
                gathered_blocks = key_cache.index_select(0, valid_blocks)
                
                # 2. Compute Norm per block FIRST to reduce size immediately
                # [num_blocks, block_size]
                block_norms = torch.norm(gathered_blocks.float(), p=2, dim=-1).mean(dim=-1)
                
                # 3. Flatten and slice to exact seq_len (Zero-copy view usually)
                final_norms = block_norms.flatten()[:seq_len]
                
                # 4. Update Cache
                self.get_or_create_request(req_id).update(final_norms)
                
        except Exception as e:
            logger.warning(f"Error computing L2 norms batch: {e}")
    
    def get_norms(self, request_id: str, start_index: int = 0) -> Optional[List[float]]:
        """
        Get L2 norms for a request.
        
        Args:
            request_id: The request ID
            start_index: Start index for differential retrieval
            
        Returns:
            List of L2 norms per token, or None if not available
        """
        with self._data_lock:
            if request_id in self._request_data:
                return self._request_data[request_id].get_norms(start_index)
        return None
    
    def remove_request(self, request_id: str):
        """Remove L2 norm data for a completed request."""
        with self._data_lock:
            self._request_data.pop(request_id, None)
    
    def clear(self):
        """Clear all cached data."""
        with self._data_lock:
            self._request_data.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._data_lock:
            return {
                'num_requests': len(self._request_data),
                'enabled': self._enabled,
                'requests': list(self._request_data.keys()),
            }


# Global singleton instance
_l2_norm_cache: Optional[L2NormCache] = None


def get_l2_norm_cache() -> L2NormCache:
    """Get the global L2 norm cache instance."""
    global _l2_norm_cache
    if _l2_norm_cache is None:
        _l2_norm_cache = L2NormCache()
    return _l2_norm_cache
