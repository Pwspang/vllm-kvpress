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


@dataclass
class RequestL2NormData:
    """Stores L2 norm data for a single request."""
    # L2 norms per token, averaged across heads and layers
    token_l2_norms: List[float] = field(default_factory=list)
    # Number of layers that have contributed to the norms
    num_layers_accumulated: int = 0
    # Total tokens seen so far
    num_tokens: int = 0
    # Lock for thread-safe updates
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, new_norms: torch.Tensor, seq_lens: torch.Tensor, 
               req_indices: List[int], block_size: int):
        """
        Update L2 norms with new data from an attention layer.
        
        Args:
            new_norms: Tensor of L2 norms from the key cache [num_blocks * block_size, num_kv_heads]
            seq_lens: Sequence lengths for each request
            req_indices: Indices of requests in the batch that this data belongs to
            block_size: KV cache block size
        """
        with self._lock:
            # Average across heads
            if new_norms.dim() > 1:
                norms_per_token = new_norms.mean(dim=-1)  # [num_tokens]
            else:
                norms_per_token = new_norms
            
            norms_list = norms_per_token.cpu().float().tolist()
            
            # If this is the first layer, initialize
            if self.num_layers_accumulated == 0:
                self.token_l2_norms = norms_list
            else:
                # Running average across layers
                for i in range(min(len(norms_list), len(self.token_l2_norms))):
                    old_avg = self.token_l2_norms[i]
                    new_val = norms_list[i]
                    n = self.num_layers_accumulated
                    self.token_l2_norms[i] = (old_avg * n + new_val) / (n + 1)
                
                # Extend if we have more tokens
                if len(norms_list) > len(self.token_l2_norms):
                    self.token_l2_norms.extend(norms_list[len(self.token_l2_norms):])
            
            self.num_layers_accumulated += 1
            self.num_tokens = len(self.token_l2_norms)
    
    def get_norms(self) -> List[float]:
        """Get the current L2 norms."""
        with self._lock:
            return self.token_l2_norms.copy()


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
        """
        Update L2 norms for a batch of requests from the KV cache.
        
        Args:
            request_ids: List of request IDs in the batch
            key_cache: The key cache tensor [num_blocks, block_size, num_kv_heads, head_size]
            block_table: Block table mapping requests to blocks [num_reqs, max_blocks]
            seq_lens: Sequence lengths [num_reqs]
            block_size: Block size
            layer_idx: Optional layer index for filtering
        """
        if not self._enabled or key_cache is None:
            return
        
        # Check if we should compute for this layer
        if layer_idx is not None and not self.should_compute_for_layer(layer_idx):
            return
        
        try:
            # Compute L2 norms for each request
            for req_idx, request_id in enumerate(request_ids):
                if request_id is None:
                    continue
                
                seq_len = int(seq_lens[req_idx].item())
                if seq_len == 0:
                    continue
                
                # Get blocks for this request
                num_blocks_needed = (seq_len + block_size - 1) // block_size
                req_blocks = block_table[req_idx, :num_blocks_needed]
                
                # Gather keys from cache
                # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
                keys_list = []
                for block_idx in req_blocks:
                    if block_idx >= 0 and block_idx < key_cache.shape[0]:
                        block_keys = key_cache[block_idx]  # [block_size, num_kv_heads, head_size]
                        keys_list.append(block_keys)
                
                if not keys_list:
                    continue
                
                # Concatenate and compute L2 norms
                all_keys = torch.cat(keys_list, dim=0)[:seq_len]  # [seq_len, num_kv_heads, head_size]
                
                # Compute L2 norm across head_size dimension, average across heads
                l2_norms = torch.norm(all_keys, p=2, dim=-1)  # [seq_len, num_kv_heads]
                l2_norms_avg = l2_norms.mean(dim=-1)  # [seq_len]
                
                # Update cache
                request_data = self.get_or_create_request(request_id)
                with request_data._lock:
                    norms_list = l2_norms_avg.cpu().float().tolist()
                    
                    if request_data.num_layers_accumulated == 0:
                        request_data.token_l2_norms = norms_list
                    else:
                        # Running average
                        n = request_data.num_layers_accumulated
                        for i in range(min(len(norms_list), len(request_data.token_l2_norms))):
                            old_val = request_data.token_l2_norms[i]
                            new_val = norms_list[i]
                            request_data.token_l2_norms[i] = (old_val * n + new_val) / (n + 1)
                        
                        if len(norms_list) > len(request_data.token_l2_norms):
                            request_data.token_l2_norms.extend(
                                norms_list[len(request_data.token_l2_norms):]
                            )
                    
                    request_data.num_layers_accumulated += 1
                    request_data.num_tokens = len(request_data.token_l2_norms)
                    
        except Exception as e:
            logger.warning(f"Error computing L2 norms: {e}")
    
    def get_norms(self, request_id: str) -> Optional[List[float]]:
        """
        Get L2 norms for a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            List of L2 norms per token, or None if not available
        """
        with self._data_lock:
            if request_id in self._request_data:
                return self._request_data[request_id].get_norms()
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
