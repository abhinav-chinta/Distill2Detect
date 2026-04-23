# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Dict, Set, Tuple, Any
import torch
from dataclasses import dataclass

from .template import AlgorithmTemplate
from .factory import register_algorithm, create_algorithm
from .utils import extract_samples_info

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


@dataclass
class VectorInstance:
    """Represents a vector instance to be applied."""
    vector_idx: int
    algorithm: AlgorithmTemplate
    scale: float = 1.0


@register_algorithm("multi_vector")
class MultiVectorAlgorithm(AlgorithmTemplate):
    """Multi-vector control algorithm implementation, supports applying multiple vectors at the same layer."""
    
    def __init__(self, layer_id: Optional[int] = None):
        super().__init__(layer_id)
        # Store algorithm instance for each vector index
        self.vector_algorithms: Dict[int, AlgorithmTemplate] = {}
        # Store scale_factor for each vector index
        self.vector_scales: Dict[int, float] = {}
        # Conflict resolution strategy
        self.conflict_resolution: str = "priority"  # 'error', 'priority', or 'sequential'
        
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """
        MultiVectorAlgorithm is a container and does not load from a single path.
        This method is implemented to satisfy the abstract base class contract.
        """
        return {}
        
    def set_conflict_resolution(self, conflict_resolution: str) -> None:
        """Set conflict resolution strategy."""
        self.conflict_resolution = conflict_resolution

    def add_vector(self, vector_idx: int, algorithm_type: str, **kwargs) -> None:
        """Add a vector to multi-vector manager."""
        # Extract constructor arguments (e.g., normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")
        
        # Use factory to create algorithm instance
        algo = create_algorithm(algorithm_type, layer_id=self.layer_id, **init_kwargs)
        
        # Prepare unified parameters for set_steer_vector
        set_vector_kwargs = {}
        if "payload" in kwargs:
            set_vector_kwargs["payload"] = kwargs["payload"]
        if "scale_factor" in kwargs:
            set_vector_kwargs["scale_factor"] = kwargs.get("scale_factor", 1.0)

        if set_vector_kwargs:
            algo.set_steer_vector(0, **set_vector_kwargs)  # Use index 0 for internal storage
        
        # Store scale_factor separately (single-vector algorithms don't have scale attribute)
        self.vector_scales[vector_idx] = kwargs.get("scale_factor", 1.0)
        
        # Set trigger configuration using batch interface
        algo.params.configure_from_dict(kwargs)
            
        # Store the algorithm instance
        self.vector_algorithms[vector_idx] = algo
        
    def remove_vector(self, vector_idx: int) -> None:
        """Remove a vector."""
        if vector_idx in self.vector_algorithms:
            del self.vector_algorithms[vector_idx]
        if vector_idx in self.vector_scales:
            del self.vector_scales[vector_idx]

    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply multi-vector intervention.
        
        In V1 continuous batching, a batch may contain both decode and prefill samples.
        We need to handle them separately based on their individual lengths (not batch-level phase).
        """
        if not self.vector_algorithms:
            return hidden_states

        # Get forward context and samples info using helper from template
        # Use the first algorithm's helper method (they all inherit from template)
        first_algo = next(iter(self.vector_algorithms.values()))
        ctx_info = first_algo._get_forward_context_and_samples(hidden_states)
        if ctx_info is None:
            return hidden_states
        
        forward_ctx, samples_info, current_tokens = ctx_info
        
        # Debug: Show batch composition using helper
        # Note: multi_vector has its own params controller, so use self._debug_print_batch_info
        if self.params.debug:
            self._debug_print_batch_info(samples_info, class_name="MultiVector")
        
        # ========== Step 1: Collect all target positions for each vector ==========
        # Use GPU-optimized position collection from each algorithm
        vector_to_positions: Dict[int, Set[int]] = {}
        position_to_vectors: Dict[int, List[int]] = {}  # For conflict detection

        for vector_idx in sorted(self.vector_algorithms.keys()):
            algo = self.vector_algorithms[vector_idx]
            
            # Prepare algorithm parameters
            algo.set_active_tensor(0)
            params = algo._get_params()
            if not algo._is_valid(params):
                continue
            
            # 🚀 Use parameter controller to collect intervention positions
            positions_tensor = algo.params.collect_intervention_positions(
                hidden_states=hidden_states,
                current_tokens=current_tokens,
                samples_info=samples_info
            )
            
            # Single GPU↔CPU sync per vector
            if positions_tensor is not None and positions_tensor.numel() > 0:
                positions_list = positions_tensor.tolist()
                vector_to_positions[vector_idx] = set(positions_list)
                
                # Build position->vectors mapping for conflict detection
                for pos in positions_list:
                    if pos not in position_to_vectors:
                        position_to_vectors[pos] = []
                    position_to_vectors[pos].append(vector_idx)
        
        # ========== Step 2: Conflict resolution ==========
        if self.conflict_resolution == "error":
            # Check for conflicts and raise error
            for pos, vec_list in position_to_vectors.items():
                if len(vec_list) > 1:
                    raise ValueError(
                        f"Multiple vectors conflict at position {pos}: vectors {vec_list}. "
                        f"Set conflict_resolution='priority' to use the first vector, "
                        f"or 'sequential' to apply all vectors in sequence."
                    )
        
        elif self.conflict_resolution == "priority":
            # Only keep the first vector for each conflicted position
            for pos, vec_list in position_to_vectors.items():
                if len(vec_list) > 1:
                    # Remove access to this position from all vectors except the first
                    for victim_vec_idx in vec_list[1:]:
                        vector_to_positions[victim_vec_idx].discard(pos)
                    
                    if self.params.debug:
                        print(f"[MultiVector] Conflict at position {pos}: "
                              f"vectors {vec_list}, using vector {vec_list[0]} (priority mode)")
        
        elif self.conflict_resolution == "sequential":
            # No filtering needed, all vectors will be applied in order
            if self.params.debug:
                for pos, vec_list in position_to_vectors.items():
                    if len(vec_list) > 1:
                        print(f"[MultiVector] Conflict at position {pos}: "
                              f"vectors {vec_list}, applying all in sequence (sequential mode)")
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution}")
        
        # ========== Step 3: Apply vectors in order (one transform per vector) ==========
        for vector_idx in sorted(self.vector_algorithms.keys()):
            if vector_idx not in vector_to_positions:
                continue

            positions = sorted(vector_to_positions[vector_idx])
            if not positions:
                continue

            algo = self.vector_algorithms[vector_idx]
            scale = self.vector_scales.get(vector_idx, 1.0)
            
            # Convert positions to tensor
            indices_tensor = torch.tensor(positions, device=hidden_states.device, dtype=torch.long)
            
            # Prepare algorithm parameters
            algo.set_active_tensor(0)
            params = algo._get_params()
            if algo._is_valid(params):
                # Use helper method from template for batch transformation
                hidden_states = algo._batch_transform_tensor(hidden_states, indices_tensor, params)
                
                if self.params.debug:
                    print(f"[MultiVector] Applied vector {vector_idx} (scale={scale}) to {len(positions)} positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")
        
        return hidden_states
    
    # Abstract method implementations required by template (not directly used in multi-vector mode)
    def _get_params(self) -> Any:
        """Not used in multi-vector mode."""
        return None

    def _is_valid(self, params: Any) -> bool:
        """Not used in multi-vector mode."""
        return False

    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """Not used in multi-vector mode."""
        return hidden_state

    # Methods to comply with BaseSteerVectorAlgorithm interface
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """Not directly used in multi-vector mode."""
        pass

    def reset_steer_vector(self, index: int) -> None:
        """Reset all vectors."""
        self.vector_algorithms.clear()
        self.vector_scales.clear()

    def set_active_tensor(self, index: int) -> None:
        """Not directly used in multi-vector mode."""
        pass 