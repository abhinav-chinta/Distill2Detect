# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
from abc import ABC, abstractmethod

from .base import BaseSteerVectorAlgorithm
from .utils import extract_samples_info
from .parameter_control import InterventionController

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


class AlgorithmTemplate(BaseSteerVectorAlgorithm, ABC):
    """
    Steer vector algorithm template class.
    
    Provides a clean template for implementing new algorithms. Algorithm developers
    only need to focus on 3 core methods:
    
    1. _get_params(): Return algorithm parameters (vectors, matrices, etc.)
    2. _is_valid(): Check if parameters are valid
    3. _transform(): Core transformation logic
    
    Parameter management (triggers, exclusions, etc.) is handled by InterventionController,
    completely decoupled from algorithm logic.
    """
    
    def __init__(self, layer_id: Optional[int] = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        # Intervention parameters - directly exposed for clean access
        self.params = InterventionController()
        
        # Universal payload storage - can store ANY type (Tensor, dict, list, etc.)
        # Algorithms don't need to manage storage - just implement _transform and load_from_path
        self._payloads: dict[int, Any] = {}
        self._active_payload: Optional[Any] = None
        
        # Common parameters - all algorithms inherit these, but only use what they need
        self.normalize = normalize  # Direct algorithm uses this
        # Future common parameters can be added here:
        # self.clamp_range = kwargs.get('clamp_range', None)
        # self.dropout_rate = kwargs.get('dropout_rate', 0.0)
    
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """
        Universal implementation: Store payload of any type.
        
        Algorithms don't need to override this - just define what payload format
        they need in load_from_path, and use it in _transform.
        """
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if payload is None:
            raise ValueError(f"{self.__class__.__name__} requires 'payload' in kwargs")
        
        # Handle scale_factor for different payload types
        if isinstance(payload, torch.Tensor):
            # For Tensor payload: apply scale_factor directly
            payload = payload * scale_factor
        elif isinstance(payload, dict):
            # For dict payload: add scale_factor to the dict
            payload = {**payload, "scale_factor": scale_factor}
        # For other types: store as-is (algorithms handle scaling themselves)
        
        self._payloads[index] = payload
    
    def set_active_tensor(self, index: int) -> None:
        """
        Universal implementation: Activate stored payload.
        
        Algorithms don't need to override this.
        """
        self._active_payload = self._payloads.get(index)
    
    def reset_steer_vector(self, index: int) -> None:
        """
        Universal implementation: Remove payload.
        
        Algorithms don't need to override this.
        """
        if index in self._payloads:
            del self._payloads[index]
    
    def _get_params(self) -> Any:
        """
        Universal implementation: Return active payload as-is.
        
        Algorithms don't need to override this - payload format is defined
        by the algorithm's load_from_path method.
        """
        return self._active_payload
    
    def _is_valid(self, params: Any) -> bool:
        """
        Universal implementation: Check params is not None.
        
        Algorithms rarely need to override this.
        """
        return params is not None
    
    @abstractmethod
    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """
        Transform hidden state (MUST be implemented by subclass).
        
        This is the core logic of your algorithm - the only truly required method.
        """
        pass
    
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Unified intervention application logic.
        
        This method coordinates the intervention process:
        1. Get algorithm parameters
        2. Delegate to parameter controller to find intervention positions
        3. Apply transformations at those positions
        """
        # Skip if no triggers configured
        if not self.params.has_any_triggers():
            return hidden_states

        # Get algorithm parameters
        algo_params = self._get_params()
        if not self._is_valid(algo_params):
            return hidden_states

        # ========== Fast Path: Global Application ==========
        # When configured to apply to ALL tokens in BOTH prefill AND generate phases,
        # we can use direct tensor transformation for optimal performance.
        # This provides 2-3x speedup by avoiding index_select/index_copy overhead.
        #
        # Fast path requirements:
        #   - prefill_trigger_tokens contains -1 (global marker for prefill phase)
        #   - generate_trigger_tokens contains -1 (global marker for generate phase)
        #   - No exclusion filters (exclude_tokens, exclude_positions)
        #
        # Note: prefill_trigger_positions is ignored when -1 is present, as the normal
        # path returns immediately when matching all tokens.
        #
        # Design constraint: Single-phase global configs (e.g., only prefill) cannot
        # use fast path because we need forward context to distinguish phases in mixed batches.
        # The normal path handles phase separation correctly through batch metadata.
        
        if self.params.is_global_only_config():
            # Direct tensor transformation - fastest path!
            if self.params.debug:
                print(f"[{self.__class__.__name__}] ✨ Fast Path: Global application to ALL {hidden_states.shape[0]} tokens (both phases)")
            original_dtype = hidden_states.dtype
            return self._transform(hidden_states, algo_params).to(original_dtype)

        # Get forward context and samples info using helper
        ctx_info = self._get_forward_context_and_samples(hidden_states)
        if ctx_info is None:
            return hidden_states
        
        forward_ctx, samples_info, current_tokens = ctx_info
        
        # Debug: Show batch composition using helper
        if self.params.debug:
            self._debug_print_batch_info(samples_info)
        
        # Delegate to parameter controller to collect intervention positions
        positions_tensor = self.params.collect_intervention_positions(
            hidden_states=hidden_states,
            current_tokens=current_tokens,
            samples_info=samples_info
        )
        
        # ========== Batch Transform ==========
        if positions_tensor is not None and positions_tensor.numel() > 0:
            if self.params.debug:
                # Only sync for debug output (1 sync instead of N syncs)
                positions_list = positions_tensor.tolist()
                print(f"[{self.__class__.__name__}] ========== Batch Transform ==========")
                print(f"[{self.__class__.__name__}]   Total positions: {len(positions_list)}")
                print(f"[{self.__class__.__name__}]   Positions (first 20): {positions_list[:20]}{'...' if len(positions_list) > 20 else ''}")
            
            # Apply transformation using tensor (no sync needed)
            hidden_states = self._batch_transform_tensor(hidden_states, positions_tensor, algo_params)

        return hidden_states
    
    # ========== Helper Methods ==========
    def _get_forward_context_and_samples(self, hidden_states: torch.Tensor):
        """
        Get forward context and sample information.
        
        This is a shared helper that extracts forward context, current tokens,
        and sample boundaries - common operations for both single-vector and
        multi-vector interventions.
        
        Args:
            hidden_states: [total_tokens, hidden_dim]
            
        Returns:
            tuple: (forward_ctx, samples_info, current_tokens) or None if unavailable
        """
        # Get forward context
        if get_forward_context is None:
            return None

        forward_ctx = get_forward_context()
        if forward_ctx is None:
            return None
            
        current_tokens = forward_ctx.current_tokens
        attn_metadata = forward_ctx.attn_metadata

        if current_tokens is None or attn_metadata is None:
            return None
        
        # Flatten tokens if needed
        if current_tokens.dim() == 2:
            current_tokens = current_tokens.flatten()
        
        # Extract sample boundaries using GPU batch operations
        samples_info = extract_samples_info(attn_metadata)
        
        if samples_info is None:
            # In vLLM V1, query_start_loc should always be available
            raise RuntimeError(
                "Cannot extract sample information from attention metadata. "
                "This should not happen in vLLM V1 with standard attention backends. "
                "Please report this issue with your configuration details."
            )
        
        return (forward_ctx, samples_info, current_tokens)
    
    def _debug_print_batch_info(self, samples_info: dict, class_name: str = None):
        """
        Print debug information about batch composition.
        
        Args:
            samples_info: Dict with 'query_start_loc', 'num_computed', 'is_decode_mask'
            class_name: Optional class name for debug prefix (defaults to self.__class__.__name__)
        """
        if not self.params.debug:
            return
        
        if class_name is None:
            class_name = self.__class__.__name__
        
        query_start_loc = samples_info['query_start_loc']
        is_decode_mask = samples_info['is_decode_mask']
        num_computed = samples_info.get('num_computed')
        
        num_samples = len(query_start_loc) - 1
        decode_count = int(is_decode_mask.sum().item())
        prefill_count = num_samples - decode_count
        
        print(f"\n[{class_name}] ========== New Batch ==========")
        print(f"[{class_name}]   Total samples: {num_samples} ({decode_count} decode, {prefill_count} prefill)")
        
        for idx in range(num_samples):
            start = int(query_start_loc[idx].item())
            end = int(query_start_loc[idx + 1].item())
            length = end - start
            is_decode = bool(is_decode_mask[idx].item())
            phase = "DECODE" if is_decode else "PREFILL"
            
            cached = 0
            if num_computed is not None and idx < len(num_computed):
                cached = int(num_computed[idx].item())
            
            print(f"[{class_name}]   Sample {idx}: [{start}:{end}] len={length} {phase} (cached={cached})")
    
    def _batch_transform_tensor(self, hidden_states, positions_tensor, params):
        """
        Apply transformation using position tensor.
        
        Performs direct tensor operations without GPU-CPU synchronization.
        
        Args:
            hidden_states: [total_tokens, hidden_dim]
            positions_tensor: [num_positions] GPU tensor of indices
            params: Algorithm parameters
            
        Returns:
            hidden_states: Transformed hidden states
        """
        original_dtype = hidden_states.dtype
        
        # Select positions to transform
        selected = hidden_states.index_select(0, positions_tensor)
        
        # Apply transformation
        transformed = self._transform(selected, params).to(original_dtype)
        
        # Write back transformed values
        hidden_states.index_copy_(0, positions_tensor, transformed)

        return hidden_states