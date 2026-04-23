# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Hidden States Capture
"""

import torch
import numpy as np
from typing import Dict, Any


def deserialize_hidden_states(serialized_data: Dict[int, Dict[str, Any]]) -> Dict[int, torch.Tensor]:
    """
    Deserialize hidden states from RPC-transferred format back to tensors.
    
    Args:
        serialized_data: Dictionary mapping layer_id to serialized tensor info:
            {
                'data': list (nested) - always in float32 format for compatibility,
                'shape': list of ints,
                'dtype': str (e.g., 'torch.bfloat16') - original dtype
            }
    
    Returns:
        Dictionary mapping layer_id to torch.Tensor with original dtype
    
    Example:
        >>> results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
        >>> serialized = results[0]
        >>> hidden_states = deserialize_hidden_states(serialized)
        >>> # Now hidden_states[layer_id] is a real tensor with correct dtype
    """
    tensors = {}
    
    for layer_id, tensor_info in serialized_data.items():
        # Extract info
        data_list = tensor_info['data']
        shape = tuple(tensor_info['shape'])
        dtype_str = tensor_info['dtype']
        
        # Convert dtype string to torch dtype
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float64': torch.float64,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
        }
        original_dtype = dtype_map.get(dtype_str, torch.float32)
        
        # Data is always in float32 format (for numpy compatibility)
        # Convert list to numpy array (float32), then to tensor
        np_array = np.array(data_list, dtype=np.float32)
        tensor = torch.from_numpy(np_array).view(shape)
        
        # Convert back to original dtype if needed
        if original_dtype != tensor.dtype:
            tensor = tensor.to(original_dtype)
        
        tensors[layer_id] = tensor
    
    return tensors


def print_hidden_states_summary(hidden_states: Dict[int, torch.Tensor]):
    """
    Print a summary of captured hidden states.
    
    Args:
        hidden_states: Dictionary mapping layer_id to torch.Tensor
    """
    print(f"📊 Captured {len(hidden_states)} layers:")
    for layer_id in sorted(hidden_states.keys()):
        tensor = hidden_states[layer_id]
        print(f"  Layer {layer_id:2d}: shape {tuple(tensor.shape)}, "
              f"dtype {tensor.dtype}, device {tensor.device}")

