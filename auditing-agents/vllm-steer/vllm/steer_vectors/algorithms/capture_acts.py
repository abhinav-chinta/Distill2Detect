# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist

from .factory import register_algorithm
from .template import AlgorithmTemplate
from .utils import extract_samples_info

try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


@register_algorithm("capture_acts")
class CaptureActs(AlgorithmTemplate):
    """
    Activation capture algorithm for concurrent request batching.

    Each sample in a batch writes to its own file to prevent interference.
    Each rank writes to its own file to support tensor parallelism.
    """

    def _transform(self, hidden_states: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Capture activations for each sample in the batch.

        Args:
            hidden_states: [num_tokens, hidden_dim] for all samples
            params: {"save_dir": str, "layer_id": int}

        Returns:
            hidden_states unchanged
        """
        # Save the acts to params["path"]
        acts_path_name = f"rank_{dist.get_rank()}_layer_{params['layer_id']}_acts.pt"
        
        # If the file exists, already, load it
        if os.path.exists(os.path.join(params["path"], acts_path_name)):
            acts = torch.load(os.path.join(params["path"], acts_path_name))
            acts = torch.cat([acts, hidden_states.cpu()], dim=0)
        else:
            acts = hidden_states.cpu()

        torch.save(acts, os.path.join(params["path"], acts_path_name))

        return hidden_states

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """
        Setup activation capture for specified layers.

        Args:
            path: Directory to save activations
            device: Unused
            **kwargs: Must contain 'target_layers'

        Returns:
            {"layer_payloads": {layer_id: config}}
        """
        target_layers = kwargs.get("target_layers", [])
        if not target_layers:
            raise ValueError("CaptureActs requires 'target_layers'")

        return {"layer_payloads": {layer: {"path": path, "layer_id": layer, "counter": 0} for layer in target_layers}}
