# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .template import AlgorithmTemplate
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("lm_steer")
class LMSteerAlgorithm(AlgorithmTemplate):
    """LM-Steer algorithm: h' = h + α * ((h @ P1) @ P2^T)
    
    This algorithm demonstrates low-rank optimization:
    - Only 2 methods needed: _transform and load_from_path
    - Payload is a dict containing P1, P2 projection matrices
    - All parameter management is handled by AlgorithmTemplate
    """

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply LM-Steer transformation: h' = h + α * ((h @ P1) @ P2^T)"""
        P1 = params["projector1"]
        P2 = params["projector2"]
        scale_factor = params.get("scale_factor", 1.0)
        
        # Select the first steer vector (index 0) if multi-dimensional
        if P1.dim() > 2:
            P1 = P1[0]
        if P2.dim() > 2:
            P2 = P2[0]
        
        # Ensure data types match
        device = hidden_state.device
        dtype = hidden_state.dtype
        
        P1 = P1.to(device).to(dtype)
        P2 = P2.to(device).to(dtype)
        
        # Apply low-rank transformation: (h @ P1) @ P2^T
        transformed = torch.matmul(hidden_state, P1)  # [..., rank]
        transformed = torch.matmul(transformed, P2.transpose(-2, -1))  # [..., hidden_dim]
        
        # Add original hidden state: h' = h + α * delta
        return hidden_state + scale_factor * transformed

    @classmethod
    def load_from_path(cls, file_path: str, device: str, config=None, target_layers=None):
        """Load LM-Steer parameters from pt file."""
        import os
        
        try:
            # Load pt file, set weights_only=False to allow loading argparse.Namespace and other objects
            state_dict = torch.load(file_path, map_location=device, weights_only=False)
            
            # Extract projection matrices
            projector1 = None
            projector2 = None
            
            # Check if it's a list structure (handles gpt2.pt special structure)
            if isinstance(state_dict, list) and len(state_dict) > 1:
                # logger.info(f"Detected list structure, trying to extract parameters from element[1]")
                params_dict = state_dict[1]
                
                if isinstance(params_dict, dict) and 'projector1' in params_dict and 'projector2' in params_dict:
                    # This is the low-rank optimization form
                    # logger.info(f"Found low-rank optimization projector1 and projector2 parameters")
                    projector1 = params_dict['projector1']
                    projector2 = params_dict['projector2']
            # Check if it's a dictionary structure
            elif isinstance(state_dict, dict):
                if "projector1" in state_dict and "projector2" in state_dict:
                    # This is the low-rank optimization form
                    # logger.info(f"Found low-rank optimization projector1 and projector2 parameters")
                    projector1 = state_dict["projector1"]
                    projector2 = state_dict["projector2"]
            
            # If projection matrices not found, raise error
            if projector1 is None or projector2 is None:
                logger.error(f"Could not find projector matrices in file {file_path}")
                raise ValueError(f"Projector matrices not found in pt file")
            
            # Get data type from config, with default value
            adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16
                
            # Create payload for each target layer
            layer_payloads = {}
            
            # If target layers are not specified, assume apply to all layers
            if target_layers is None:
                # Try to get number of layers from config
                if hasattr(config, 'num_hidden_layers'):
                    target_layers = list(range(config.num_hidden_layers))
                else:
                    # Default to 32 layers
                    target_layers = list(range(32))
            
            # Ensure it's a tensor and convert data type
            projector1_tensor = projector1.to(device=device, dtype=adapter_dtype)
            projector2_tensor = projector2.to(device=device, dtype=adapter_dtype)
            
            for layer_idx in target_layers:
                layer_payloads[layer_idx] = {
                    "projector1": projector1_tensor,
                    "projector2": projector2_tensor
                }
            # logger.info(f"Loaded low-rank projection matrices P1: {projector1_tensor.shape}, P2: {projector2_tensor.shape}")
                
            return {"layer_payloads": layer_payloads}
            
        except Exception as e:
            logger.error(f"Failed to load LM-Steer parameters from {file_path}: {e}")
            raise RuntimeError(f"Failed to load LM-Steer parameters") from e 