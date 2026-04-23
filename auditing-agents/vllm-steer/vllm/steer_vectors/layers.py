# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from .algorithms import BaseSteerVectorAlgorithm, create_algorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


def extract_layer_id_from_module_name(module_name: str) -> Optional[int]:
    """
    Extract layer ID from module name.
    
    Args:
        module_name: Module name like 'model.layers.0' or 'transformer.h.12'
    
    Returns:
        Layer ID as integer, or None if not found
    
    Examples:
        'model.layers.0' -> 0
        'transformer.h.12' -> 12
        'model.embed_tokens' -> None
    """
    parts = module_name.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


@dataclass
class SteerVectorMapping:
    layer_mapping: dict[int, torch.Tensor]


class BaseLayerWithSteerVector(nn.Module):
    pass


def _extract_hidden_states_and_residual(output):
    """
    Extract hidden_states and residual from DecoderLayer output.

    Args:
        output: DecoderLayer output, possible formats:
               - (hidden_states, residual)  # Qwen2 and similar models
               - hidden_states              # Phi and similar models
               - tuple with more elements   # Other possible formats

    Returns:
        (hidden_states, residual, other_outputs, original_format)
    """
    if isinstance(output, tuple):
        if len(output) == 2:
            # Assume (hidden_states, residual) format
            hidden_states, residual = output
            if (isinstance(hidden_states, torch.Tensor) and
                    isinstance(residual, torch.Tensor) and
                    hidden_states.shape == residual.shape):
                return hidden_states, residual, None, "tuple_2"
            else:
                # If shapes don't match, may not be (hidden_states, residual) format
                return output[0], None, output[1:], "tuple_other"
        elif len(output) > 2:
            # More complex tuple, assume first element is hidden_states
            return output[0], None, output[1:], "tuple_multi"
        else:
            # Single-element tuple
            return output[0], None, None, "tuple_1"
    elif isinstance(output, torch.Tensor):
        # Direct tensor output, e.g., Phi models
        return output, None, None, "tensor"
    else:
        # Other formats, try to extract from attributes
        if hasattr(output, 'hidden_states'):
            hidden_states = output.hidden_states
            residual = getattr(output, 'residual', None)
            return hidden_states, residual, output, "object"
        else:
            # Unrecognized format, return original output
            return output, None, None, "unknown"


def _reconstruct_output(modified_hidden_states, residual, other_outputs, original_format, original_output):
    """
    Reconstruct output based on original format.

    Args:
        modified_hidden_states: Modified hidden_states
        residual: Residual (if any)
        other_outputs: Other output elements
        original_format: Original format identifier
        original_output: Original output (for reconstructing complex objects)

    Returns:
        Reconstructed output
    """
    if original_format == "tuple_2":
        return (modified_hidden_states, residual)
    elif original_format == "tuple_other":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_multi":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_1":
        return (modified_hidden_states,)
    elif original_format == "tensor":
        return modified_hidden_states
    elif original_format == "object":
        # For object format, modify the corresponding attribute
        if hasattr(original_output, 'hidden_states'):
            original_output.hidden_states = modified_hidden_states
        return original_output
    else:
        # Unknown format, return modified hidden_states
        return modified_hidden_states


class DecoderLayerWithSteerVector(BaseLayerWithSteerVector):
    """
    Generic DecoderLayer wrapper that supports intervention on full hidden states.
    Uses lazy loading mechanism to create algorithm instances only when needed, saving memory.
    """

    def __init__(self, base_layer) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.active_algorithm_name: str = "direct"
        self.algorithms: Dict[str, BaseSteerVectorAlgorithm] = {}
        self.layer_id: Optional[int] = None

    def _get_or_create_algorithm(self, name: str, **kwargs) -> BaseSteerVectorAlgorithm:
        """Lazy load or get algorithm instance by name."""
        if name not in self.algorithms:
            # Pass kwargs (e.g., normalize) from external calls to the constructor
            self.algorithms[name] = create_algorithm(name, layer_id=self.layer_id, **kwargs)
        return self.algorithms[name]

    def set_layer_id(self, layer_id: int) -> None:
        """Set layer ID for all created algorithms."""
        self.layer_id = layer_id
        for algo in self.algorithms.values():
            algo.layer_id = layer_id

    def set_steer_vector(self, index: int, **kwargs):
        """
        Generic method: set steer vector parameters for the specified algorithm.
        This method is responsible for distributing all relevant parameters to the algorithm instance.
        """
        # 1. Determine algorithm and extract its unique associated parameters
        algorithm_name = kwargs.pop("algorithm_name", "direct")
        self.active_algorithm_name = algorithm_name
        
        # Extract constructor parameters (e.g., normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")

        algo = self._get_or_create_algorithm(algorithm_name, **init_kwargs)

        # 2. Set core vector parameters (payload) and other runtime parameters
        # Pass all remaining kwargs to set_steer_vector
        algo.set_steer_vector(index, **kwargs)

        # 3. Set intervention parameters (triggers and debug flags)
        # Batch configure using InterventionController's unified interface
        algo.params.configure_from_dict(kwargs)

    def reset_steer_vector(self, index: int):
        """Reset the vector at specified index in all algorithms (or only the currently active one)."""
        # For simplicity, we reset vectors in all created algorithms
        # Could also reset only the currently active one
        for algo in self.algorithms.values():
            algo.reset_steer_vector(index)

    def set_active_tensor(self, index: int):
        """Set the active tensor for the currently active algorithm."""
        algo = self._get_or_create_algorithm(self.active_algorithm_name)
        algo.set_active_tensor(index)

    def forward(self, *args, **kwargs):
        """Wrap the forward method of DecoderLayer."""
        output = self.base_layer(*args, **kwargs)

        # Dynamically get the currently active algorithm and apply intervention
        active_algo = self._get_or_create_algorithm(self.active_algorithm_name)
        
        # Extract hidden_states and residual from decoder layer output
        hidden_states, residual, other_outputs, original_format = _extract_hidden_states_and_residual(output)

        # Construct complete hidden state
        if residual is not None:
            complete_hidden_states = hidden_states + residual
        else:
            complete_hidden_states = hidden_states

        # Apply algorithm transformation
        modified_complete_hidden_states = active_algo.apply_intervention(complete_hidden_states)

        # Reconstruct output format
        if residual is not None:
            zero_residual = torch.zeros_like(residual)
            return _reconstruct_output(modified_complete_hidden_states, zero_residual, other_outputs,
                                       original_format, output)
        else:
            return _reconstruct_output(modified_complete_hidden_states, None, other_outputs, original_format,
                                       output)


