# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Any
import torch
import numpy as np

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging
logger = logging.getLogger(__name__)
@register_algorithm("direct")
class DirectAlgorithm(AlgorithmTemplate):
    """Direct addition algorithm: h' = h + vector
    
    This algorithm demonstrates the ultimate simplicity:
    - Only 2 methods needed: _transform and load_from_path
    - All parameter management (including normalize) is handled by AlgorithmTemplate
    - Payload is a simple Tensor
    """

    def _transform(self, hidden_state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply direct addition: h' = h + vector (with optional normalization)."""
        if self.normalize:
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            transformed = hidden_state + params
            norm_post = torch.norm(transformed, dim=-1, keepdim=True)
            return transformed * norm_pre / norm_post
        else:
            return hidden_state + params

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """Load Direct steer vector from GGUF file, PT file, or ReFT directory."""
        import os
        
        config = kwargs.get("config")
        if config is None:
            raise ValueError("DirectAlgorithm.load_from_path requires 'config' in kwargs")

        if os.path.isdir(path):
            return cls._load_from_reft_dir(path, device, **kwargs)
            
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.pt':
            return cls._load_from_pt(path, device, **kwargs)
        else:  # Default to gguf format
            return cls._load_from_gguf(path, device, **kwargs)
    
    @classmethod
    def _load_from_pt(cls, path: str, device: str, **kwargs) -> dict:
        """Load Direct steer vector from PT file."""
        import torch
        
        config = kwargs.get("config")
        target_layers = kwargs.get("target_layers")
        if target_layers is None:
            raise ValueError("Loading .pt files requires 'target_layers' in kwargs")
            
        # Use the first target layer for loading PT file
        if not target_layers:
            raise ValueError("target_layers list cannot be empty")
            
        target_layer = target_layers[0]
        
        try:
            # Load tensor from PT file
            # Use weights_only=False to handle PyTorch 2.6+ behavior
            vector = torch.load(path, map_location=device, weights_only=False)
            
            # Handle numpy array and convert to tensor
            if isinstance(vector, np.ndarray):
                vector = torch.tensor(vector, device=device)
            # Ensure vector is in correct format and convert to required dtype
            elif not isinstance(vector, torch.Tensor):
                raise ValueError(f"PT file does not contain a tensor or numpy array: {type(vector)}")
                
            vector = vector.to(device).to(config.adapter_dtype)
            
            # Use the specified target layer
            sv_weights = {target_layer: vector}
            
            return {"layer_payloads": sv_weights}
            
        except Exception as e:
            raise ValueError(f"Failed to load PT file: {e}") from e
    
    @classmethod
    def _load_from_gguf(cls, path: str, device: str, **kwargs) -> dict:
        """Load Direct steer vector from GGUF file."""
        import gguf
        import numpy as np
        
        config = kwargs.get("config")
        
        reader = gguf.GGUFReader(path)
        
        # Validate file type
        archf = reader.get_field("general.architecture")
        if archf and len(archf.parts):
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "steervector" and arch != "controlvector":
                # Only log, don't enforce
                # logger.warning(".gguf file with arch %s may not be a steer vector", arch)
                pass

        sv_weights = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (ValueError, IndexError) as e:
                raise ValueError(f".gguf file has invalid direction field name: {tensor.name}") from e
            
            np_copy = np.array(tensor.data, copy=True)
            sv_weights[layer] = torch.from_numpy(np_copy).to(device).to(config.adapter_dtype)
            
        return {"layer_payloads": sv_weights}

    @classmethod
    def _load_from_reft_dir(cls, path: str, device: str, **kwargs) -> dict:
        """Load steer vector from ReFT directory (e.g., BiasIntervention)."""
        import os
        import glob
        import json
        import torch

        config = kwargs.get("config")
        target_layers = kwargs.get("target_layers")

        if not os.path.isdir(path):
            raise ValueError(f"For ReFT algorithm, path must be a directory. Got: {path}")

        bin_files = glob.glob(os.path.join(path, "*.bin"))
        if not bin_files:
            raise ValueError(f"No .bin files found in directory: {path}")
        if len(bin_files) > 1:
            raise ValueError(f"Multiple .bin files found in directory {path}. Please ensure only one exists.")
        
        bin_file_path = bin_files[0]

        config_files = [os.path.join(path, f) for f in ["reft_config.json", "config.json"] if os.path.exists(os.path.join(path, f))]
        if not config_files:
            raise ValueError(f"No config file (reft_config.json or config.json) found in directory: {path}")
        if len(config_files) > 1:
            raise ValueError(f"Multiple config files found in directory {path}. Please ensure only one exists.")
        
        config_file_path = config_files[0]

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)

        config_layer_idx = None
        if "representations" in config_data:
            representations = config_data.get("representations", [])
            if representations:
                first_repr = representations[0]
                if isinstance(first_repr, dict):
                    config_layer_idx = first_repr.get("layer")
                # Support for older list-based representation format
                elif isinstance(first_repr, list) and len(first_repr) > 0:
                    config_layer_idx = first_repr[0]


        if config_layer_idx is None:
            bin_filename = os.path.basename(bin_file_path)
            if "intkey_layer_" in bin_filename:
                try:
                    layer_str = bin_filename.split("intkey_layer_")[1].split("_")[0]
                    config_layer_idx = int(layer_str)
                except (ValueError, IndexError):
                    pass
        
        if config_layer_idx is None:
            raise ValueError(f"Could not extract layer info from config {config_file_path} or filename {os.path.basename(bin_file_path)}")

        if target_layers and config_layer_idx not in target_layers:
            raise ValueError(f"Layer mismatch: config specifies layer {config_layer_idx}, but target_layers is {target_layers}.")

        state_dict = torch.load(bin_file_path, map_location=device)
        
        vector = None
        adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16

        if len(state_dict) == 1:
            vector = list(state_dict.values())[0]
        elif 'source_representation' in state_dict:
            vector = state_dict['source_representation']
        elif 'bias' in state_dict:
            vector = state_dict['bias']
        elif 'weight' in state_dict:
            vector = state_dict['weight']
        else:
            raise ValueError(f"Could not determine the correct tensor from .bin file with multiple tensors. Keys found: {list(state_dict.keys())}")
        
        if not isinstance(vector, torch.Tensor):
            raise ValueError(f"Loaded payload is not a tensor. Type: {type(vector)}")
            
        vector = vector.to(device).to(adapter_dtype)
        
        sv_weights = {config_layer_idx: vector}
        
        return {"layer_payloads": sv_weights}

 