# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for steer vector algorithms.

This module provides shared utility functions used across different algorithm implementations,
including sample information extraction, query_start_loc handling, and prefix cache support.
"""

from typing import Optional, Dict
import torch

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


def extract_samples_info(attn_metadata) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract sample boundaries and phases from attention metadata.
    
    Uses GPU batch operations to calculate sample boundaries and phases without loops.
    This is a shared utility function used by both AlgorithmTemplate and MultiVectorAlgorithm.
    
    Args:
        attn_metadata: Attention metadata from forward context
        
    Returns:
        Dict with GPU tensors:
            'query_start_loc': [num_samples+1] tensor of sample boundaries
            'num_computed': [num_samples] tensor of cached token counts (or None)
            'is_decode_mask': [num_samples] boolean tensor (True for decode samples)
        or None if query_start_loc is unavailable
    """
    query_start_loc = get_query_start_loc(attn_metadata)
    
    if query_start_loc is None or len(query_start_loc) <= 1:
        return None
    
    # Extract num_computed_tokens_cpu for prefix cache support
    num_computed_tokens_cpu = get_num_computed_tokens(attn_metadata)
    
    # Calculate sample lengths
    starts = query_start_loc[:-1]  # [num_samples]
    ends = query_start_loc[1:]      # [num_samples]
    lengths = ends - starts         # [num_samples]
    
    # Determine decode/prefill phase based on length
    # Decode phase: length == 1 (single token)
    # Prefill phase: length > 1 (multiple tokens)
    is_decode_mask = (lengths == 1)  # [num_samples], boolean tensor
    
    return {
        'query_start_loc': query_start_loc,       # [num_samples+1] on GPU
        'num_computed': num_computed_tokens_cpu,  # [num_samples] on GPU or None
        'is_decode_mask': is_decode_mask          # [num_samples] on GPU
    }


def get_query_start_loc(attn_metadata) -> Optional[torch.Tensor]:
    """
    Extract query_start_loc from attention metadata.
    
    V1 uses dict format with 'query_start_loc' field.
    
    Args:
        attn_metadata: Attention metadata from forward context
        
    Returns:
        query_start_loc tensor or None if not available
    """
    if isinstance(attn_metadata, dict):
        # V1: dict format (primary path)
        if attn_metadata:
            first_layer_metadata = next(iter(attn_metadata.values()))
            return getattr(first_layer_metadata, 'query_start_loc', None)
    else:
        # Object format fallback
        return getattr(attn_metadata, 'query_start_loc', None)
    return None


def get_num_computed_tokens(attn_metadata) -> Optional[torch.Tensor]:
    """
    Extract num_computed_tokens_cpu for prefix cache support.
    
    V1 stores num_computed_tokens_cpu in forward_context.
    
    Args:
        attn_metadata: Attention metadata from forward context
        
    Returns:
        num_computed_tokens_cpu tensor or None if not available
    """
    try:
        if get_forward_context is None:
            return None
        forward_context = get_forward_context()
        return forward_context.num_computed_tokens_cpu
    except Exception:
        # Fallback: try to extract from attn_metadata
        if isinstance(attn_metadata, dict) and attn_metadata:
            first_layer_metadata = next(iter(attn_metadata.values()))
            return getattr(first_layer_metadata, 'num_computed_tokens_cpu', None)
        else:
            return getattr(attn_metadata, 'num_computed_tokens_cpu', None) if attn_metadata else None

