# SPDX-License-Identifier: Apache-2.0
"""
vLLM Transformer Layer Wrappers

Uses vLLM's decoder layer class names for precise layer matching,
and vLLM's residual extraction logic for correct hidden state extraction.
"""

from typing import Optional
import torch
from torch import nn


class VLLMTransformerLayerWrapper(nn.Module):
    """
    Wrapper for vLLM transformer layers to capture hidden states.
    
    Uses the same extraction logic as steer_vectors to correctly handle
    (hidden_states, residual) output format from vLLM's optimized layers.
    """

    def __init__(self, base_layer: nn.Module, layer_id: int, layer_name: str = "", 
                 store=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.layer_name = layer_name or f"layer_{layer_id}"
        self.store = store

    def forward(self, *args, **kwargs):
        """
        Forward pass that captures the final hidden state output
        """
        # Call original layer's forward method
        output = self.base_layer(*args, **kwargs)

        # Extract hidden states using vLLM's logic
        if self.store is not None:
            hidden_states = self._extract_hidden_states_vllm_style(output)

            # Store the complete hidden states
            if hidden_states is not None:
                self.store.store_hidden_state(
                    self.layer_id,
                    hidden_states,
                    self.layer_name
                )

        return output

    def _extract_hidden_states_vllm_style(self, output) -> Optional[torch.Tensor]:
        """
        Extract hidden states using vLLM's steer_vectors extraction logic.
        
        This handles the (hidden_states, residual) format correctly.
        
        Args:
            output: Layer output, possible formats:
                   - (hidden_states, residual)  # vLLM optimized format
                   - hidden_states              # Direct tensor
                   - tuple with more elements
        
        Returns:
            Complete hidden states tensor (with residual added if applicable)
        """
        try:
            # Import vLLM's extraction function
            from vllm.steer_vectors.layers import _extract_hidden_states_and_residual
            
            # Use vLLM's extraction logic
            hidden_states, residual, other_outputs, original_format = \
                _extract_hidden_states_and_residual(output)
            
            # Compute complete hidden states
            if residual is not None:
                # Add residual to get complete accumulated state
                complete_hidden_states = hidden_states + residual
                return complete_hidden_states
            else:
                return hidden_states
                
        except ImportError:
            # Fallback to simple extraction if vLLM not available
            return self._extract_hidden_states_fallback(output)
    
    def _extract_hidden_states_fallback(self, output) -> Optional[torch.Tensor]:
        """
        Fallback extraction logic if vLLM is not available.
        
        This is a simplified version that handles common cases.
        """
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            if len(output) >= 2 and isinstance(output[0], torch.Tensor) and isinstance(output[1], torch.Tensor):
                hidden_states, residual = output[0], output[1]
                if hidden_states.shape == residual.shape:
                    return hidden_states + residual
                return hidden_states
            elif len(output) > 0 and isinstance(output[0], torch.Tensor):
                return output[0]
        elif isinstance(output, dict):
            for key in ['hidden_states', 'last_hidden_state', 'output']:
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
        elif hasattr(output, 'hidden_states'):
            if isinstance(output.hidden_states, torch.Tensor):
                return output.hidden_states
        
        return None

    def __getattr__(self, name):
        """Delegate attribute access to the base layer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

