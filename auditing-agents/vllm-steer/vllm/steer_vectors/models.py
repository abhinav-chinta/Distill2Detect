# SPDX-License-Identifier: Apache-2.0

"""Steer Vector Model Management for vLLM V1.

This module provides a clean, well-organized implementation of steer vector
model management with clear separation of concerns:

1. Utility Classes - Helper classes for caching and ID generation
2. Wrapper Registry - Configuration-driven wrapper class mapping
3. Data Models - SteerVectorModel representing loaded vectors
4. Model Manager - Core management and lifecycle control
5. LRU Cache Extensions - Memory-efficient adapter management
6. Factory Functions - Convenient manager creation

Architecture:
    config.py → models.py → layers.py → algorithms/
    (Configure) (Orchestrate) (Implement)  (Transform)
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import gguf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from vllm.config import SteerVectorConfig
from vllm.steer_vectors.algorithms import create_algorithm
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
from vllm.steer_vectors.config import WRAPPER_REGISTRY, get_target_modules
from vllm.steer_vectors.layers import (
    DecoderLayerWithSteerVector,
    SteerVectorMapping,
    extract_layer_id_from_module_name,
)
from vllm.utils.cache import LRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Section 1: Utility Classes and Functions
# ============================================================================

class AdapterLRUCache(LRUCache[int, T]):
    """LRU cache for adapters with automatic deactivation on removal."""
    
    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: T | None):
        logger.debug("Removing steer vector adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


_GLOBAL_STEER_VECTOR_ID = 0


def get_steer_vector_id() -> int:
    """Generate a unique steer vector ID."""
    global _GLOBAL_STEER_VECTOR_ID
    _GLOBAL_STEER_VECTOR_ID += 1
    return _GLOBAL_STEER_VECTOR_ID


# ============================================================================
# Section 2: Wrapper Registry (Configuration-Driven)
# ============================================================================

# Build wrapper class mapping from configuration
# This registry is extensible - future wrapper types (attention, MLP, etc.)
# can be added through the WRAPPER_REGISTRY in config.py

_all_sv_classes = {}

# Import wrapper classes based on enabled configuration
# This is done at module import time to avoid circular dependencies
for wrapper_type, config in WRAPPER_REGISTRY.items():
    if config.get("enabled", False):
        if wrapper_type == "decoder_layer":
            _all_sv_classes[wrapper_type] = DecoderLayerWithSteerVector
        # Future wrapper types can be added here:
        # elif wrapper_type == "attention":
        #     from vllm.steer_vectors.layers import AttentionWithSteerVector
        #     _all_sv_classes[wrapper_type] = AttentionWithSteerVector
        # elif wrapper_type == "mlp":
        #     from vllm.steer_vectors.layers import MLPWithSteerVector
        #     _all_sv_classes[wrapper_type] = MLPWithSteerVector


# ============================================================================
# Section 3: Data Model - SteerVectorModel
# ============================================================================

class SteerVectorModel:
    """Represents a steer vector model that can be applied to layers.
    
    This class encapsulates the data and metadata for loaded steer vectors,
    supporting both single-vector and multi-vector configurations.
    
    Attributes:
        id: Unique identifier for this steer vector
        layer_payloads: Dict mapping layer indices to their vector payloads
        scale_factor: Global scaling factor for single-vector mode
        algorithm: Algorithm type ('direct', 'linear', etc.)
        multi_vector_data: Configuration for multi-vector mode (if applicable)
    """

    def __init__(
        self,
        steer_vector_id=None,
        layer_payloads=None,
        scale_factor=1.0,
        algorithm="direct",
        multi_vector_data=None
    ) -> None:
        self.id = steer_vector_id
        self.layer_payloads = layer_payloads
        self.scale_factor = scale_factor
        self.algorithm = algorithm
        self.multi_vector_data = multi_vector_data
        
    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector model."""
        return self.multi_vector_data is not None

    # ------------------------------------------------------------------------
    # Factory Methods - Loading from Different Sources
    # ------------------------------------------------------------------------

    @classmethod
    def from_local_checkpoint(
        cls,
        steer_vector_model_path: str,
        steer_vector_id: int,
        config: SteerVectorConfig,
        device: str = "cuda",
        scale_factor: float = 1.0,
        algorithm: str = "direct",
        target_layers: Optional[list[int]] = None,
    ) -> "SteerVectorModel":
        """Load a steer vector from a local checkpoint or HuggingFace Hub.
        
        Args:
            steer_vector_model_path: Path to the vector file (local or HF format)
            steer_vector_id: Unique ID for this vector
            config: Steer vector configuration
            device: Device to load tensors on
            scale_factor: Global scaling factor
            algorithm: Algorithm type (can be embedded in path with "|")
            target_layers: Optional list of target layer indices
            
        Returns:
            Loaded SteerVectorModel instance
        """
        try:
            # Handle algorithm parameter in path (e.g., "path/to/vector|linear")
            if "|" in steer_vector_model_path:
                steer_vector_model_path, path_algorithm = steer_vector_model_path.split("|", 1)
                algorithm = path_algorithm

            # Resolve path (local file or HuggingFace Hub)
            if os.path.exists(steer_vector_model_path):
                file_path = os.path.abspath(steer_vector_model_path)
            else:
                # Download from HuggingFace Hub
                parts = steer_vector_model_path.split("/")
                repo_id = "/".join(parts[:2])
                file_name = "/".join(parts[2:])
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    revision="main"
                )

            # Dynamically get the algorithm class from the registry
            if algorithm not in ALGORITHM_REGISTRY:
                raise ValueError(f"Unsupported algorithm for loading: '{algorithm}'")
            
            algo_class = ALGORITHM_REGISTRY[algorithm]

            # Delegate loading to the algorithm's class method
            loaded_params = algo_class.load_from_path(
                file_path, device, config=config, target_layers=target_layers
            )
            
            # Create SteerVectorModel instance from loaded parameters
            return cls(
                steer_vector_id=steer_vector_id,
                layer_payloads=loaded_params.get("layer_payloads"),
                scale_factor=scale_factor,
                algorithm=algorithm,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load steer vector from {steer_vector_model_path} "
                f"with algorithm '{algorithm}'"
            ) from e



# ============================================================================
# Section 4: Core Manager - SteerVectorModelManager
# ============================================================================

class SteerVectorModelManager:
    """Manages steer vector models for a given model.
    
    This is the core orchestrator that:
    1. Wraps model layers with appropriate intervention wrappers
    2. Manages the lifecycle of steer vector adapters
    3. Activates/deactivates vectors at runtime
    4. Coordinates between configuration, wrappers, and algorithms
    
    The manager is designed to be completely agnostic to specific wrapper
    types - it uses configuration-driven dynamic dispatch for extensibility.
    """

    def __init__(
        self,
        model: nn.Module,
        steer_vector_config: SteerVectorConfig
    ):
        self.model = model
        self._registered_adapters: Dict[int, SteerVectorModel] = {}
        self._active_adapters: Dict[int, Any] = {}
        self.steer_vector_config = steer_vector_config
        self._last_mapping = None
        self.model.steer_vector_manager = self
        self.steer_vector_index_to_id: list[Optional[int]] = [None] * self.adapter_slots
        self.modules: dict[str, nn.Module] = {}
        self._create_sv_modules()

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def adapter_slots(self) -> int:
        """Number of adapter slots available."""
        return self.capacity

    @property
    def capacity(self) -> int:
        """Maximum number of steer vectors that can be loaded."""
        return self.steer_vector_config.max_steer_vectors

    # ------------------------------------------------------------------------
    # Adapter Lifecycle Management
    # ------------------------------------------------------------------------

    def activate_adapter(
        self,
        steer_vector_id: int,
        target_layers: Optional[list[int]] = None,
        prefill_trigger_tokens: Optional[list[int]] = None,
        prefill_trigger_positions: Optional[list[int]] = None,
        prefill_exclude_tokens: Optional[list[int]] = None,
        prefill_exclude_positions: Optional[list[int]] = None,
        generate_trigger_tokens: Optional[list[int]] = None,
        debug: bool = False,
        conflict_resolution: str = "priority",
        normalize: bool = False,
    ) -> bool:
        """Activate a steer vector adapter.
        
        Args:
            steer_vector_id: ID of the steer vector to activate
            target_layers: Optional list of layer indices to apply to
            prefill_trigger_tokens: Tokens that trigger intervention in prefill
            prefill_trigger_positions: Positions that trigger intervention in prefill
            prefill_exclude_tokens: Tokens to exclude from intervention
            prefill_exclude_positions: Positions to exclude from intervention
            generate_trigger_tokens: Tokens that trigger intervention in generation
            debug: Enable debug output
            conflict_resolution: Strategy for multi-vector conflicts
            normalize: Whether to normalize vectors
            
        Returns:
            True if activation successful
        """
        if steer_vector_id in self._active_adapters:
            self._deactivate_adapter(steer_vector_id)
            del self._active_adapters[steer_vector_id]

        first_free_slot = next(
            (i for i, slot_id in enumerate(self.steer_vector_index_to_id) if slot_id is None),
            None
        )
        if first_free_slot is None:
            raise ValueError("No free steer vector slots")
        index = first_free_slot
        
        steer_vector_model = self._registered_adapters.get(steer_vector_id)
        if not steer_vector_model:
            raise ValueError(f"Steer vector {steer_vector_id} not found.")

        # Prepare unified parameter dictionary
        params = {
            "algorithm_name": steer_vector_model.algorithm,
            "scale_factor": steer_vector_model.scale_factor,
            "prefill_trigger_tokens": prefill_trigger_tokens,
            "prefill_trigger_positions": prefill_trigger_positions,
            "prefill_exclude_tokens": prefill_exclude_tokens,
            "prefill_exclude_positions": prefill_exclude_positions,
            "generate_trigger_tokens": generate_trigger_tokens,
            "debug": debug,
            "normalize": normalize,
        }

        # Apply algorithm-specific weights/parameters to all target modules
        if steer_vector_model.is_multi_vector:
            self._activate_multi_vector_adapter(index, steer_vector_model, debug, conflict_resolution)
        elif steer_vector_model.layer_payloads:
            for layer_idx, payload in steer_vector_model.layer_payloads.items():
                if target_layers and layer_idx not in target_layers:
                    continue
                for module in self._get_modules_for_layer(layer_idx):
                    module.set_steer_vector(index, payload=payload, **params)
        else:
            # Fallback for models without payloads
            pass

        self.steer_vector_index_to_id[index] = steer_vector_id
        self._active_adapters[steer_vector_id] = None
        self._set_adapter_mapping(steer_vector_id)

        logger.debug(f"Activated steer vector {steer_vector_id} in slot {index}")
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        """Deactivate a steer vector adapter."""
        if adapter_id in self._active_adapters:
            self._deactivate_adapter(adapter_id)
            del self._active_adapters[adapter_id]
            return True
        return False

    def add_adapter(self, adapter: SteerVectorModel) -> bool:
        """Add a steer vector adapter to the registry."""
        if len(self._registered_adapters) >= self.capacity:
            logger.warning(
                f"Cannot add adapter {adapter.id}: "
                f"already at capacity ({self.capacity})"
            )
            return False
        return self._add_adapter(adapter)

    def remove_adapter(self, adapter_id: int) -> bool:
        """Remove a steer vector adapter."""
        self.deactivate_adapter(adapter_id)
        if adapter_id in self._registered_adapters:
            del self._registered_adapters[adapter_id]
            return True
        return False

    def remove_all_adapters(self):
        """Remove all SteerVectorModels from the manager."""
        self._registered_adapters.clear()
        self.steer_vector_index_to_id = [None] * self.adapter_slots
        self._active_adapters.clear()

    def list_adapters(self) -> dict[int, Any]:
        """List all registered adapters."""
        return dict(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        """Get a specific adapter."""
        return self._registered_adapters.get(adapter_id)

    def pin_adapter(self, adapter_id: int) -> bool:
        """Pin an adapter (not implemented for steer vectors)."""
        raise NotImplementedError(
            "Pinning is not supported for steer vectors"
        )

    def set_adapter_mapping(self, mapping: SteerVectorMapping) -> None:
        """Set the adapter mapping (for compatibility)."""
        # Simplified implementation for V1
        pass

    # ------------------------------------------------------------------------
    # Wrapper Management (Configuration-Driven)
    # ------------------------------------------------------------------------

    def _create_sv_modules(self):
        """
        Create steer vector modules based on enabled wrapper types.
        
        This method is fully configuration-driven and automatically wraps
        all enabled wrapper types defined in WRAPPER_REGISTRY.
        """
        for wrapper_type in _all_sv_classes.keys():
            self._wrap_modules_by_type(wrapper_type)

    def _wrap_modules_by_type(self, wrapper_type: str) -> None:
        """
        Generic method to wrap modules by type using configuration.
        
        This method works for any wrapper type (decoder_layer, attention, mlp, etc.)
        without needing specific knowledge of what each wrapper does.
        
        Args:
            wrapper_type: Type of wrapper (e.g., "decoder_layer", "attention")
        """
        # Get wrapper class from registry
        wrapper_class = _all_sv_classes.get(wrapper_type)
        if not wrapper_class:
            logger.warning(f"Wrapper class not found for type: {wrapper_type}")
            return
        
        # Get target module names from configuration
        try:
            target_modules = get_target_modules(wrapper_type)
        except ValueError as e:
            logger.warning(f"Failed to get target modules for {wrapper_type}: {e}")
            return
        
        # Wrap matching modules
        wrapped_count = 0
        for module_name, module in self.model.named_modules():
            # Check if this module should be wrapped
            if any(
                class_name in module.__class__.__name__ 
                for class_name in target_modules
            ):
                # Skip if already wrapped
                if isinstance(module, wrapper_class):
                    continue
                
                # Create wrapper using registry (dynamic instantiation)
                new_module = self.replace_submodule(
                    self.model,
                    module_name,
                    wrapper_class(module)  # Generic wrapper creation!
                )
                new_module.set_layer_id(extract_layer_id_from_module_name(module_name))
                self.register_module(module_name, new_module)
                wrapped_count += 1
                logger.debug(f"Wrapped {wrapper_type}: {module_name}")
        
        # Log summary
        if wrapped_count > 0:
            logger.debug(
                f"Using {wrapper_type}-level steer vector intervention "
                f"({wrapped_count} modules wrapped)"
            )
        else:
            logger.warning(f"No {wrapper_type} modules found for steer vector intervention")

    # ------------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------------

    def _activate_multi_vector_adapter(
        self,
        index: int,
        steer_vector_model: SteerVectorModel,
        debug: bool,
        conflict_resolution: str
    ):
        """Handle multi-vector activation logic."""
        layer_to_vectors: Dict[int, List[Tuple[int, Dict]]] = {}
        
        # 1. Collect vectors to process for each layer
        for vector_idx, vector_data in enumerate(steer_vector_model.multi_vector_data):
            vector_target_layers = vector_data.get('target_layers')
            affected_layers = list(vector_data.get('payloads', {}).keys())
            
            for layer_idx in affected_layers:
                if vector_target_layers is None or layer_idx in vector_target_layers:
                    if layer_idx not in layer_to_vectors:
                        layer_to_vectors[layer_idx] = []
                    layer_to_vectors[layer_idx].append((vector_idx, vector_data))

        # 2. Configure algorithm for each layer
        for layer_idx, vectors_for_layer in layer_to_vectors.items():
            for module in self._get_modules_for_layer(layer_idx):
                if len(vectors_for_layer) == 1:
                    # Single vector: degrade to single-vector mode
                    _, vector_data = vectors_for_layer[0]
                    self._apply_single_vector_to_module(module, index, vector_data, debug, layer_idx)
                else:
                    # Multiple vectors: configure MultiVectorAlgorithm
                    module.active_algorithm_name = "multi_vector"
                    multi_vector_algo = module._get_or_create_algorithm("multi_vector")
                    multi_vector_algo.set_conflict_resolution(conflict_resolution)
                    multi_vector_algo.params.set_debug(debug)
                    multi_vector_algo.reset_steer_vector(0)
                    
                    # Add all sub-vectors
                    for vec_idx, vec_data in vectors_for_layer:
                        add_kwargs = vec_data.copy()
                        algorithm_type = add_kwargs.pop('algorithm')
                        add_kwargs['payload'] = vec_data['payloads'][layer_idx]
                        add_kwargs['scale_factor'] = vec_data.get('scale', 1.0)
                        
                        # Remove parameters that don't belong to add_vector
                        add_kwargs.pop('payloads', None)
                        add_kwargs.pop('weights', None)
                        add_kwargs.pop('loreft_params', None)
                        add_kwargs.pop('sv_vector', None)
                        
                        multi_vector_algo.add_vector(
                            vector_idx=vec_idx, 
                            algorithm_type=algorithm_type, 
                            **add_kwargs
                        )

    def _get_modules_for_layer(self, layer_idx: int) -> List[nn.Module]:
        """Get all modules for the specified layer."""
        modules = []
        for module_name, module in self.modules.items():
            if extract_layer_id_from_module_name(module_name) == layer_idx:
                modules.append(module)
        return modules

    def _apply_single_vector_to_module(
        self,
        module,
        index,
        vector_data,
        debug,
        layer_idx
    ):
        """Helper method: apply a single vector to a module."""
        params = {
            "algorithm_name": vector_data['algorithm'],
            "scale_factor": vector_data.get('scale', 1.0),
            "prefill_trigger_tokens": vector_data.get('prefill_trigger_tokens'),
            "prefill_trigger_positions": vector_data.get('prefill_trigger_positions'),
            "prefill_exclude_tokens": vector_data.get('prefill_exclude_tokens'),
            "prefill_exclude_positions": vector_data.get('prefill_exclude_positions'),
            "generate_trigger_tokens": vector_data.get('generate_trigger_tokens'),
            "debug": debug,
            "normalize": vector_data.get('normalize', False),
        }
        params["payload"] = vector_data['payloads'][layer_idx]
        module.set_steer_vector(index, **params)

    def _deactivate_adapter(self, steer_vector_id: int) -> bool:
        """Internal method to deactivate an adapter."""
        index = self.get_index_from_id(steer_vector_id)
        if index is None:
            return False
        for k, v in self.modules.items():
            v.reset_steer_vector(index)
        self.steer_vector_index_to_id[index] = None
        return True

    def _add_adapter(self, steer_vector: SteerVectorModel) -> bool:
        """Internal method to add a SteerVectorModel."""
        self._registered_adapters[steer_vector.id] = steer_vector
        return True

    def _set_adapter_mapping(self, id: int) -> None:
        """Internal method to set adapter mapping."""
        index = self.get_index_from_id(id)
        if index is None:
            logger.warning(f"No slot found for steer vector ID {id}")
            return
        for k, v in self.modules.items():
            v.set_active_tensor(index)

    def get_index_from_id(self, id):
        """Get the slot index for a given adapter ID."""
        for i in range(len(self.steer_vector_index_to_id)):
            if self.steer_vector_index_to_id[i] == id:
                return i
        return None

    def replace_submodule(
        self,
        model: nn.Module,
        module_name: str,
        new_module: nn.Module
    ) -> nn.Module:
        """Replace a submodule in a model with a new module."""
        parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
        target_name = module_name.split(".")[-1]
        setattr(parent, target_name, new_module)
        return new_module

    def register_module(self, module_name: str, module: nn.Module):
        """Register a wrapped module."""
        self.modules[module_name] = module


# ============================================================================
# Section 5: LRU Cache Extensions
# ============================================================================

class SteerVectorLRUCache(AdapterLRUCache[SteerVectorModel]):
    """LRU cache specifically for SteerVectorModel."""
    
    def __init__(self, capacity: int, deactivate_sv_fn: Callable[[int], bool]):
        super().__init__(capacity, deactivate_sv_fn)


class LRUCacheSteerVectorModelManager(SteerVectorModelManager):
    """A model manager with LRU cache for automatic memory management.
    
    This manager uses LRU (Least Recently Used) caching to automatically manage
    steer vector capacity. When the cache is full and a new vector is added,
    the least recently used vector is automatically evicted.
    """

    def __init__(
        self,
        model: nn.Module,
        steer_vector_config: SteerVectorConfig,
    ):
        super().__init__(model, steer_vector_config)
        # Replace simple dicts with LRU caches
        self._registered_adapters: SteerVectorLRUCache = SteerVectorLRUCache(
            self.capacity, self.deactivate_adapter
        )
        self._active_adapters: SteerVectorLRUCache = SteerVectorLRUCache(
            self.adapter_slots, self._deactivate_adapter
        )

    def list_adapters(self) -> dict[int, SteerVectorModel]:
        """List all registered SteerVectorModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, adapter: SteerVectorModel) -> bool:
        """Add a steer vector adapter with LRU cache management."""
        logger.debug(
            "Adding steer vector. Model id: %d, int id: %d", 
            adapter.id, adapter.id
        )
        if adapter.id not in self._registered_adapters:
            self._add_adapter(adapter)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(adapter.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        steer_vector_id: int,
        target_layers: Optional[list[int]] = None,
        prefill_trigger_tokens: Optional[list[int]] = None,
        prefill_trigger_positions: Optional[list[int]] = None,
        prefill_exclude_tokens: Optional[list[int]] = None,
        prefill_exclude_positions: Optional[list[int]] = None,
        generate_trigger_tokens: Optional[list[int]] = None,
        debug: bool = False,
        conflict_resolution: str = "priority",
        normalize: bool = False,
    ) -> bool:
        """Activate adapter with automatic LRU eviction."""
        # Automatically evict oldest active adapter if at capacity
        if (
            steer_vector_id not in self._active_adapters
            and len(self._active_adapters) >= self.adapter_slots
        ):
            self._active_adapters.remove_oldest()
        
        result = super().activate_adapter(
            steer_vector_id,
            target_layers,
            prefill_trigger_tokens,
            prefill_trigger_positions,
            prefill_exclude_tokens,
            prefill_exclude_positions,
            generate_trigger_tokens,
            debug,
            conflict_resolution,
            normalize
        )
        # We always touch to update the LRU cache order
        self._active_adapters.touch(steer_vector_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        """Remove the oldest (least recently used) adapter from registered cache."""
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False


# ============================================================================
# Section 6: Factory Functions
# ============================================================================

def create_sv_manager(
    model: nn.Module,
    steer_vector_config: SteerVectorConfig,
    steer_vector_manager_cls: type[SteerVectorModelManager] = LRUCacheSteerVectorModelManager,
) -> SteerVectorModelManager:
    """Factory function to create a steer vector manager.
    
    By default, creates an LRUCacheSteerVectorModelManager for automatic
    capacity management.
    
    Args:
        model: The neural network model to manage
        steer_vector_config: Configuration for steer vectors
        steer_vector_manager_cls: Manager class to instantiate
        
    Returns:
        Initialized steer vector manager
    """
    steer_vector_manager = steer_vector_manager_cls(
        model=model,
        steer_vector_config=steer_vector_config
    )
    return steer_vector_manager

