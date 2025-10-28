# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlexAttention and KV cache eviction support."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple
from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               _score_mod_signature,
                                               create_block_mask,
                                               flex_attention)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

create_block_mask_compiled = torch.compile(create_block_mask,
                                           fullgraph=True,
                                           mode="reduce-overhead")
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _offsets_to_doc_ids_tensor(offsets: torch.Tensor) -> torch.Tensor:
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts)


def _create_eviction_mask_tensor(
    evictable_ranges: List[Tuple[int, int]],
    max_seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create a boolean mask tensor where True indicates evicted positions.
    
    Args:
        evictable_ranges: List of (start, end) token ranges to evict
        max_seq_len: Maximum sequence length
        device: Device for the tensor
    
    Returns:
        Boolean tensor of shape [max_seq_len] where True = evicted
    """
    if not evictable_ranges:
        return None
    
    mask = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
    
    for start, end in evictable_ranges:
        if start < max_seq_len and end <= max_seq_len:
            mask[start:end] = True
    
    return mask


class FlexAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        return  # FlexAttention supports any head size

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlexAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


# @torch.compile(fullgraph=True, mode="reduce-overhead")
def physical_to_logical_mapping(
        block_table: torch.Tensor,
        total_blocks: Optional[int] = None) -> torch.Tensor:
    """
    Creates an inverse mapping from physical block locations to logical indices.
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    physical_to_logical = torch.full((max_reqs, total_blocks),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

    logical_indices = (torch.arange(max_num_blocks,
                                    device=device).unsqueeze(0).expand(
                                        max_reqs, -1))

    physical_to_logical.scatter_(-1, block_table.to(torch.int64),
                                 logical_indices)
    # TODO Confirm - Seems like block 0 is always empty so we reset it manually
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def causal_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor,
                    kv_idx: torch.Tensor):
    return q_idx >= kv_idx


@dataclass
class FlexAttentionMetadata:
    causal: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Block info
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # Flex Metadata
    num_blocks = 0
    block_mask: Optional[BlockMask] = None
    score_mod: Optional[_score_mod_signature] = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod
    
    # KV Cache Eviction Support - use pre-computed tensor instead of list
    eviction_mask: Optional[torch.Tensor] = None  # Boolean mask [max_seq_len]
    
    # Cache the mask mod function to avoid recreating it
    _cached_mask_mod: Optional[_mask_mod_signature] = None
    _mask_mod_cache_key: Optional[int] = None  # Hash of eviction mask

    def get_causal_mask_mod(self) -> _mask_mod_signature:
        """Creates the mask_mod function for FlexAttention.
        
        Caches the mask mod function to avoid recreating on every call.
        Only rebuilds if eviction_mask changes.
        """
        # Create cache key based on eviction mask
        current_cache_key = id(self.eviction_mask) if self.eviction_mask is not None else 0
        
        # Return cached version if available and valid
        if (self._cached_mask_mod is not None and 
            self._mask_mod_cache_key == current_cache_key):
            return self._cached_mask_mod
        
        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)
        
        # Pre-capture eviction mask (if any)
        eviction_mask = self.eviction_mask
        has_eviction = eviction_mask is not None
        
        # Fast path: no eviction
        if not has_eviction:
            def final_mask_mod_no_eviction(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                physical_kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                # Map query indices to corresponding request indices
                q_req = request_lookup[q_idx]

                # Convert physical KV indices to logical indices
                physical_kv_block = physical_kv_idx // self.block_size
                physical_kv_offset = physical_kv_idx % self.block_size
                logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
                logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

                # Determine valid kv indices
                live_block = logical_block_idx >= 0
                within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
                within_lower_bound = logical_kv_idx >= 0

                is_valid = live_block & within_upper_bound & within_lower_bound

                # Convert physical query indices to logical indices
                local_q_idx = q_idx - self.query_start_loc[q_req]
                logical_q_idx = local_q_idx + self.decode_offset[q_req]

                # Apply mask modification
                return torch.where(
                    is_valid,
                    self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                    False,
                )
            
            self._cached_mask_mod = final_mask_mod_no_eviction
            self._mask_mod_cache_key = current_cache_key
            return final_mask_mod_no_eviction
        
        # Slow path: with eviction
        eviction_mask_size = eviction_mask.size(0)
        
        def final_mask_mod_with_eviction(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            # Map query indices to corresponding request indices
            q_req = request_lookup[q_idx]

            # Convert physical KV indices to logical indices
            physical_kv_block = physical_kv_idx // self.block_size
            physical_kv_offset = physical_kv_idx % self.block_size
            logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset

            # Determine valid kv indices
            live_block = logical_block_idx >= 0
            within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
            within_lower_bound = logical_kv_idx >= 0

            is_valid = live_block & within_upper_bound & within_lower_bound

            # Convert physical query indices to logical indices
            local_q_idx = q_idx - self.query_start_loc[q_req]
            logical_q_idx = local_q_idx + self.decode_offset[q_req]

            # Check if KV token is evicted (using pre-computed mask)
            # Clamp logical_kv_idx to valid range before indexing
            safe_kv_idx = torch.clamp(logical_kv_idx, 0, eviction_mask_size - 1)
            is_evicted_at_idx = eviction_mask[safe_kv_idx]
            # Only consider eviction if the index is actually valid
            is_not_evicted = ~is_evicted_at_idx | ~is_valid
            
            # Apply mask modification only for valid, non-evicted indices
            return torch.where(
                is_valid & is_not_evicted,
                self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )
        
        self._cached_mask_mod = final_mask_mod_with_eviction
        self._mask_mod_cache_key = current_cache_key
        return final_mask_mod_with_eviction

    def get_bidirectional_mask_mod(self) -> _mask_mod_signature:
        """Creates the encoder mask_mod function for FlexAttention."""
        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return request_lookup[q_idx] == request_lookup[kv_idx]

        return final_mask_mod

    def build_block_mask(self) -> BlockMask:
        """Build block mask only once and cache it."""
        if self.block_mask is not None:
            return self.block_mask
            
        if self.causal:
            mask_mod = self.get_causal_mask_mod()
            kv_len = self.total_cache_tokens
        else:
            mask_mod = self.get_bidirectional_mask_mod()
            kv_len = self.num_actual_tokens
            
        self.block_mask = create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
        )
        return self.block_mask

    def __post_init__(self):
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        self.num_blocks = self.total_cache_tokens // self.block_size
        # Don't build block mask here - build it lazily on first use
        # self.block_mask = self.build_block_mask()


class FlexAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlexAttentionMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.device = device

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False,
              evictable_token_ranges: Optional[List[Tuple[int, int]]] = None) -> FlexAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = int(common_attn_metadata.seq_lens_cpu.max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        block_size = self.kv_cache_spec.block_size
        max_possible_seq_len = self.model_config.max_model_len
        total_cache_tokens = self.cache_config.num_gpu_blocks * block_size

        inverse_block_table = physical_to_logical_mapping(
            block_table_tensor, self.cache_config.num_gpu_blocks)

        # Get the original offset tensor
        offset_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            self.device, non_blocking=True)

        # Create eviction mask tensor if evictable ranges provided
        # Only create if ranges are non-empty
        eviction_mask = None
        if evictable_token_ranges and len(evictable_token_ranges) > 0:
            eviction_mask = _create_eviction_mask_tensor(
                evictable_token_ranges,
                max_possible_seq_len,
                self.device
            )
            logger.debug(f"Created eviction mask with {len(evictable_token_ranges)} ranges")

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            eviction_mask=eviction_mask,
        )
        return out


class FlexAttentionImpl(AttentionImpl):
    sliding_window: Optional[tuple[int, int]]
    alibi_slopes: Optional[torch.Tensor]
    logits_soft_cap: Optional[float]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.ENCODER_ONLY,
                             AttentionType.DECODER):
            raise NotImplementedError(
                f"FlexAttention does not support {attn_type} attention")

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlexAttention does not support alibi slopes yet.")
        else:
            self.alibi_slopes = None
        if sliding_window is not None:
            raise NotImplementedError(
                "FlexAttention does not support sliding window yet.")
        else:
            self.sliding_window = (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "FlexAttention does not support logits soft cap yet.")

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "FlexAttention does not support kv sharing yet.")

        FlexAttentionBackend.validate_head_size(head_size)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet")

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """View a 3d tensor as 4D."""
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlexAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlexAttentionImpl")

        enable_gqa = self.num_kv_heads != self.num_heads

        if attn_metadata is None:
            # Profiling run.
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        if not attn_metadata.causal:
            assert self.attn_type == AttentionType.ENCODER_ONLY

            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key, value),
            )

        else:
            assert self.attn_type == AttentionType.DECODER
            key_cache, value_cache = kv_cache.unbind(0)

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

            # View out the block_size dim
            key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.view(-1, self.num_kv_heads,
                                           self.head_size)
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key_cache, value_cache),
            )

        query = query[:, :, :num_actual_tokens, :]

        # Build block mask lazily (cached after first build)
        if attn_metadata.block_mask is None:
            attn_metadata.build_block_mask()

        # default M=64, N=64 may run out of shared memory on some GPUs
        extra_kernel_options = defaultdict[str, int](lambda: 64)
        if query.dtype == torch.float32:
            extra_kernel_options["BLOCK_M"] //= 2
            extra_kernel_options["BLOCK_N"] //= 2
        if current_platform.is_cuda():
            device_props = torch.cuda.get_device_properties()
            max_shared_memory = device_props.shared_memory_per_block_optin
            if max_shared_memory < 144 * 1024:
                extra_kernel_options["BLOCK_M"] //= 2
                extra_kernel_options["BLOCK_N"] //= 2

        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options={
                "FORCE_USE_FLEX_ATTENTION": True,
                **extra_kernel_options
            },
        )

        # Flex doesn't have an out variant today, rely on epilogue fusion
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output[:num_actual_tokens, :, :].copy_(out)
        return output