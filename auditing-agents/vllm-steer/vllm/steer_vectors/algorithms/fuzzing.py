# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging

logger = logging.getLogger(__name__)


@register_algorithm("fuzzing")
class FuzzingAlgorithm(AlgorithmTemplate):
    """Fuzzing algorithm: h' = h + noise where noise is a random unit vector scaled by radius R.

    Samples random vectors from unit ball and scales them to have norm R.
    Different noise is sampled per token position.
    """

    def _transform(self, hidden_state: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Apply random unit ball noise scaled by radius: h' = h + R * (random_unit_vector)."""
        radius = params.get('radius', 1.0)

        # Sample Gaussian noise and normalize to unit vectors
        noise = torch.randn_like(hidden_state)
        noise = F.normalize(noise, p=2, dim=-1)

        # Scale by radius
        noise = noise * radius

        return hidden_state + noise

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load fuzzing config from PT file.

        Expected format: dict with 'radius' key specifying the norm of random vectors.
        If path points to a simple float/int, use that as radius.
        """
        import torch
        import os

        config = kwargs.get("config")
        if config is None:
            raise ValueError("FuzzingAlgorithm.load_from_path requires 'config' in kwargs")

        target_layers = kwargs.get("target_layers")
        if target_layers is None or len(target_layers) == 0:
            raise ValueError("Loading fuzzing config requires non-empty 'target_layers' in kwargs")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Fuzzing config file not found: {path}")

        try:
            data = torch.load(path, map_location=device, weights_only=False)

            # Handle different formats
            if isinstance(data, dict) and 'radius' in data:
                radius = float(data['radius'])
            elif isinstance(data, (int, float)):
                radius = float(data)
            else:
                raise ValueError(
                    f"Fuzzing config must be either a dict with 'radius' key or a numeric value. Got: {type(data)}"
                )

            # Create payload dict for each target layer
            layer_payloads = {}
            for layer_id in target_layers:
                layer_payloads[layer_id] = {'radius': radius}

            return {"layer_payloads": layer_payloads}

        except Exception as e:
            raise ValueError(f"Failed to load fuzzing config from {path}: {e}") from e
