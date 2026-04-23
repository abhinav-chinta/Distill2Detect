# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any
import torch
import numpy as np

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None

logger = logging.getLogger(__name__)


@register_algorithm("inject_acts")
class InjectActsAlgorithm(AlgorithmTemplate):
    """Inject activations at positions where specific token IDs appear.

    This algorithm injects pre-computed activations wherever specified token IDs appear
    in the sequence. Unlike 'direct' which adds the same vector to all tokens, this algorithm:
    - Injects different activation vectors for different token IDs
    - Uses forward context to find token ID matches dynamically
    - Normalizes injected activations to match the original activation norms
    - Works correctly even when multiple requests are batched together

    Payload format:
        Dict with keys:
        - 'acts_normalized': torch.Tensor of shape [n_token_ids, hidden_dim] (pre-normalized)
        - 'token_ids': List[int] of token IDs to inject activations on
        - 'steering_coef': float (default 1.0) to scale the injected activations

    The algorithm will find all positions in the current batch where any of the token_ids
    appear, and inject the corresponding activation vector at those positions.

    Example usage:
        For a verbalizer with token_id=42 repeated 3 times:
        payload = {
            'acts_normalized': F.normalize(tensor([[act1], [act2], [act3]]), dim=-1),  # shape [3, hidden_dim]
            'token_ids': [42, 42, 42],  # All the same token ID
            'steering_coef': 1.0
        }
        The algorithm will find all positions where token 42 appears and inject
        act1 at the first occurrence, act2 at the second, act3 at the third.
    """

    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply activation injection at positions matching token IDs.

        Finds a consecutive sequence of token IDs in the input and injects
        pre-computed activation vectors at those positions.

        Args:
            hidden_states: [batch_tokens, hidden_dim] - activations for all tokens in this batch

        Returns:
            Modified hidden_states with injected activations
        """
        # ========== VALIDATION ==========
        # Check if intervention is configured with trigger tokens
        if not self.params.has_any_triggers():
            return hidden_states

        # Get the payload (acts, token_ids, steering_coef)
        algo_params = self._get_params()
        if not self._is_valid(algo_params):
            return hidden_states

        # ========== GET CURRENT TOKENS ==========
        # Ask vLLM: "What token IDs are being processed right now?"
        if get_forward_context is None:
            return hidden_states

        forward_ctx = get_forward_context()
        if forward_ctx is None or forward_ctx.current_tokens is None:
            return hidden_states

        # current_tokens: tensor of token IDs being processed
        # Example: [128000, 9506, 25, 17487, 17487, 17487, 198, ...]
        #          [<BOS>, "Layer", ":", " ?", " ?", " ?", "\n", ...]
        current_tokens = forward_ctx.current_tokens
        if current_tokens.dim() == 2:
            current_tokens = current_tokens.flatten()

        # ========== EXTRACT PAYLOAD ==========
        # acts_normalized: [n, hidden_dim] - pre-computed activation vectors (normalized to unit length)
        # token_ids: [id_0, id_1, ..., id_{n-1}] - token IDs to search for consecutively
        # steering_coef: float - scaling factor for injection strength
        acts_normalized = algo_params['acts_normalized']
        token_ids = algo_params['token_ids']
        steering_coef = algo_params['steering_coef']

        # Move to GPU and correct dtype
        acts_normalized = acts_normalized.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # ========== FIND CONSECUTIVE PATTERN ==========
        n_tokens = len(token_ids)
        if n_tokens == 0:
            return hidden_states

        # Convert token_ids list to tensor for efficient comparison
        # Example: [17487, 17487, 17487] -> tensor([17487, 17487, 17487])
        target_tokens = torch.tensor(token_ids, device=current_tokens.device, dtype=current_tokens.dtype)

        # Check if sequence is long enough to contain the pattern
        seq_len = len(current_tokens)
        if seq_len < n_tokens:
            return hidden_states

        # Slide a window of size n_tokens across current_tokens to find the pattern
        # Example: Looking for [17487, 17487, 17487]
        #   Position 0: [128000, 9506, 25] ❌
        #   Position 1: [9506, 25, 17487] ❌
        #   Position 3: [17487, 17487, 17487] ✅ MATCH!
        inject_positions = []
        for start_pos in range(seq_len - n_tokens + 1):
            # Extract window: current_tokens[start_pos : start_pos + n_tokens]
            window = current_tokens[start_pos:start_pos + n_tokens]

            # Check if this window matches our target pattern
            if torch.equal(window, target_tokens):
                # Found it! Record all positions in the match
                # Example: start_pos=3, n_tokens=3 -> positions [3, 4, 5]
                inject_positions.extend(range(start_pos, start_pos + n_tokens))
                break  # Only inject at first occurrence

        # No match found, nothing to inject
        if len(inject_positions) == 0:
            return hidden_states

        # ========== INJECT ACTIVATIONS ==========
        # Convert positions list to tensor for indexing
        # Example: [3, 4, 5] -> tensor([3, 4, 5])
        inject_positions = torch.tensor(inject_positions, device=hidden_states.device, dtype=torch.long)

        # Get the original activations at these positions
        # Shape: [n_tokens, hidden_dim]
        # Example: hidden_states[[3, 4, 5]] -> [3, 4096] tensor
        original_acts = hidden_states[inject_positions]

        # Calculate the L2 norm of each original activation
        # This tells us the "magnitude" or "strength" of each activation
        # Shape: [n_tokens, 1]
        # Example: [[15.2], [14.8], [15.1]]
        original_norms = original_acts.norm(dim=1, keepdim=True)

        # Scale our normalized activations to match these norms
        # acts_normalized has norm 1.0, so multiplying by original_norms scales it
        # steering_coef allows us to make injections stronger/weaker
        # Shape: [n_tokens, hidden_dim]
        steered_acts = (acts_normalized * steering_coef * original_norms).detach()

        # Add injected activations to existing activations (not replace!)
        # hidden_states[3] = hidden_states[3] + steered_acts[0]
        # hidden_states[4] = hidden_states[4] + steered_acts[1]
        # hidden_states[5] = hidden_states[5] + steered_acts[2]
        hidden_states[inject_positions] = hidden_states[inject_positions] + steered_acts

        return hidden_states

    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """Not used - we override apply_intervention directly.

        This method is required by AlgorithmTemplate but not used by InjectActsAlgorithm
        because we need access to forward context (current_tokens) which isn't available
        in _transform. Instead, we override apply_intervention to get forward context.
        """
        raise NotImplementedError("InjectActsAlgorithm overrides apply_intervention directly")

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load activation injection data from a PT file or directory.

        The file should contain a dict with keys:
        - 'acts': torch.Tensor of shape [n_token_ids, hidden_dim]
        - 'token_ids': List[int] of token IDs to inject on
        - 'steering_coef': float (optional, default 1.0)

        Args:
            path: Path to PT file containing the injection data
            device: Device to load tensors to
            **kwargs: Additional config (target_layers, config, etc.)

        Returns:
            Dict with 'layer_payloads' mapping layer_id to payload dict
        """
        import os

        config = kwargs.get("config")
        if config is None:
            raise ValueError("InjectActsAlgorithm.load_from_path requires 'config' in kwargs")

        target_layers = kwargs.get("target_layers")
        if target_layers is None or len(target_layers) == 0:
            raise ValueError("Loading injection data requires non-empty 'target_layers' in kwargs")

        target_layer = target_layers[0]

        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")

        try:
            # Load data from PT file
            data = torch.load(path, map_location=device, weights_only=False)

            if not isinstance(data, dict):
                raise ValueError(f"Expected dict in PT file, got {type(data)}")

            if 'acts' not in data or 'token_ids' not in data:
                raise ValueError("PT file must contain 'acts' and 'token_ids' keys")

            # Extract and validate acts
            acts = data['acts']
            if isinstance(acts, np.ndarray):
                acts = torch.tensor(acts, device=device)
            elif not isinstance(acts, torch.Tensor):
                raise ValueError(f"'acts' must be tensor or numpy array, got {type(acts)}")
            acts = acts.to(device).to(config.adapter_dtype)

            # Normalize once at load time (more efficient than per-call)
            acts_normalized = torch.nn.functional.normalize(acts, dim=-1).detach()

            # Extract and validate token_ids
            token_ids = data['token_ids']
            if not isinstance(token_ids, list):
                raise ValueError(f"'token_ids' must be a list, got {type(token_ids)}")

            # Extract steering coefficient
            steering_coef = data.get('steering_coef', 1.0)

            # Validate dimensions
            if acts.ndim != 2:
                raise ValueError(f"'acts' must be 2D tensor, got shape {acts.shape}")

            n_inject = acts.shape[0]
            n_token_ids = len(token_ids)
            if n_inject != n_token_ids:
                raise ValueError(
                    f"Number of activation vectors ({n_inject}) must match "
                    f"number of token_ids ({n_token_ids})"
                )

            # Package payload with pre-normalized acts
            payload = {
                'acts_normalized': acts_normalized,
                'token_ids': token_ids,
                'steering_coef': steering_coef
            }

            # Return in expected format
            sv_weights = {target_layer: payload}
            return {"layer_payloads": sv_weights}

        except Exception as e:
            raise ValueError(f"Failed to load injection data from {path}: {e}") from e
