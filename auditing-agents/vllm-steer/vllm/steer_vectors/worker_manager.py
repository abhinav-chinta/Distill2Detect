# SPDX-License-Identifier: Apache-2.0

"""Worker-level manager for steer vectors in vLLM V1."""

import logging
from typing import Any, Dict, List, Set

import torch

from vllm.config import SteerVectorConfig
from vllm.steer_vectors.models import (
    SteerVectorModel,
    SteerVectorModelManager,
    LRUCacheSteerVectorModelManager,
    create_sv_manager
)
from vllm.steer_vectors.request import SteerVectorRequest

logger = logging.getLogger(__name__)


class WorkerSteerVectorManager:
    """WorkerSteerVectorManager that manages steer vector models on the worker side.

    Every request, the requested steer vectors will be loaded (unless they are already loaded),
    and every other steer vector will be unloaded.
    """

    _manager_cls: type[SteerVectorModelManager] = SteerVectorModelManager

    def __init__(
        self,
        device: torch.device,
        steer_vector_config: SteerVectorConfig,
        steer_vector_model_cls: type[SteerVectorModel] = SteerVectorModel
    ):
        self._adapter_manager: SteerVectorModelManager | None = None
        self._steer_vector_model_cls = steer_vector_model_cls
        self.steer_vector_config = steer_vector_config
        self.device = device

    @property
    def is_enabled(self) -> bool:
        return True

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        """Create and initialize the steer vector manager for the model."""
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._manager_cls,
        )
        self._adapter_manager = steer_vector_manager
        return steer_vector_manager.model

    def _load_adapter(
        self,
        steer_vector_request: SteerVectorRequest
    ) -> SteerVectorModel:
        """Load a steer vector from a request.
        
        This method acts as the decoupling layer between SteerVectorRequest
        and SteerVectorModel, extracting parameters from the request and
        calling the appropriate factory method.
        
        Similar to LoRA's WorkerLoRAManager._load_adapter() pattern.
        """
        try:
            if not steer_vector_request.is_multi_vector:
                # Single-vector mode: extract parameters and call from_local_checkpoint
                steer_vector = self._steer_vector_model_cls.from_local_checkpoint(
                    steer_vector_model_path=steer_vector_request.local_path,
                    steer_vector_id=steer_vector_request.steer_vector_id,
                    config=self.steer_vector_config,
                    device=str(self.device),
                    scale_factor=steer_vector_request.scale,
                    algorithm=steer_vector_request.algorithm,
                    target_layers=steer_vector_request.target_layers,
                )
            else:
                # Multi-vector mode: load each vector individually and assemble
                multi_vector_data = []
                
                for i, vector_config in enumerate(steer_vector_request.vector_configs):
                    try:
                        # Load individual vector using from_local_checkpoint
                        single_model = self._steer_vector_model_cls.from_local_checkpoint(
                            steer_vector_model_path=vector_config.path,
                            steer_vector_id=f"{steer_vector_request.steer_vector_id}_vec_{i}",
                            config=self.steer_vector_config,
                            device=str(self.device),
                            scale_factor=vector_config.scale,
                            algorithm=vector_config.algorithm,
                            target_layers=vector_config.target_layers,
                        )
                        
                        # Store vector data with its configuration
                        vector_data = {
                            'payloads': single_model.layer_payloads,
                            'scale': vector_config.scale,
                            'target_layers': vector_config.target_layers,
                            'prefill_trigger_tokens': vector_config.prefill_trigger_tokens,
                            'prefill_trigger_positions': vector_config.prefill_trigger_positions,
                            'prefill_exclude_tokens': vector_config.prefill_exclude_tokens,
                            'prefill_exclude_positions': vector_config.prefill_exclude_positions,
                            'generate_trigger_tokens': vector_config.generate_trigger_tokens,
                            'algorithm': vector_config.algorithm,
                            'path': vector_config.path,
                            'normalize': vector_config.normalize,
                        }
                        multi_vector_data.append(vector_data)
                        
                        logger.debug(
                            f"Loaded vector {i}: {vector_config.path} "
                            f"(algorithm: {vector_config.algorithm}, scale: {vector_config.scale})"
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to load vector {i} from {vector_config.path}: {e}")
                        raise RuntimeError(
                            f"Failed to load vector {i} from {vector_config.path}"
                        ) from e
                
                logger.debug(
                    f"Successfully loaded {len(multi_vector_data)} vectors for "
                    f"multi-vector request '{steer_vector_request.steer_vector_name}'"
                )
                
                # Create multi-vector model (note: no from_steer_vector_request needed!)
                steer_vector = self._steer_vector_model_cls(
                    steer_vector_id=steer_vector_request.steer_vector_id,
                    layer_payloads=None,
                    scale_factor=1.0,
                    algorithm="multi_vector",
                    multi_vector_data=multi_vector_data
                )
                
        except Exception as e:
            request_info = (
                steer_vector_request.local_path 
                if not steer_vector_request.is_multi_vector 
                else f"multi-vector request with {len(steer_vector_request.vector_configs)} vectors"
            )
            # Import traceback to get full error details
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Failed to load steer vector {request_info}:\n{error_details}")
            raise RuntimeError(
                f"Loading steer vector {request_info} failed: {str(e)}"
            ) from e
        
        return steer_vector

    def add_dummy_steer_vector(
        self,
        steer_vector_request: SteerVectorRequest
    ) -> bool:
        """Add a dummy steer vector (placeholder for future use)."""
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        """Pin an adapter (not supported for steer vectors)."""
        if self._adapter_manager is None:
            return False
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: Set[Any]) -> None:
        """Set the active adapters based on requests."""
        # Simplified implementation for V1
        # In V1, we don't use the complex set_adapter_mapping from V0
        for request in requests:
            if request is not None:
                self.add_adapter(request)

    def add_adapter(self, adapter_request: SteerVectorRequest) -> bool:
        """Add a steer vector adapter."""
        if self._adapter_manager is None:
            logger.warning("Steer vector manager not initialized")
            return False
        
        # Support replacing adapters with the same ID by removing old one first
        if adapter_request.steer_vector_id in self.list_adapters():
            logger.debug(
                f"Replacing existing steer vector with ID {adapter_request.steer_vector_id}"
            )
            self.remove_adapter(adapter_request.steer_vector_id)
        
        # Load the adapter
        adapter = self._load_adapter(adapter_request)
        
        # Add to manager
        if not self._adapter_manager.add_adapter(adapter):
            return False
        
        # Activate based on request type
        if adapter_request.is_multi_vector:
            # Multi-vector mode: activation is handled internally
            self._adapter_manager.activate_adapter(
                adapter_request.steer_vector_id,
                debug=adapter_request.debug,
                conflict_resolution=adapter_request.conflict_resolution
            )
        else:
            # Single-vector mode: use request-level parameters
            self._adapter_manager.activate_adapter(
                adapter_request.steer_vector_id,
                target_layers=adapter_request.target_layers,
                prefill_trigger_tokens=adapter_request.prefill_trigger_tokens,
                prefill_trigger_positions=adapter_request.prefill_trigger_positions,
                prefill_exclude_tokens=adapter_request.prefill_exclude_tokens,
                prefill_exclude_positions=adapter_request.prefill_exclude_positions,
                generate_trigger_tokens=adapter_request.generate_trigger_tokens,
                debug=adapter_request.debug,
                normalize=adapter_request.normalize
            )
        
        return True

    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        """Apply adapters to the model."""
        if self._adapter_manager is None:
            return
        
        # Remove adapters that are no longer requested
        current_ids = {req.steer_vector_id for req in adapter_requests if req is not None}
        registered_ids = set(self.list_adapters())
        
        for adapter_id in registered_ids - current_ids:
            self.remove_adapter(adapter_id)
        
        # Add new adapters
        for request in adapter_requests:
            if request is not None:
                self.add_adapter(request)

    def remove_adapter(self, adapter_id: int) -> bool:
        """Remove a steer vector adapter."""
        if self._adapter_manager is None:
            return False
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        """Remove all steer vector adapters."""
        if self._adapter_manager is not None:
            self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        """List all registered adapter IDs."""
        if self._adapter_manager is None:
            return set()
        return set(self._adapter_manager.list_adapters().keys())


class LRUCacheWorkerSteerVectorManager(WorkerSteerVectorManager):
    """WorkerSteerVectorManager that manages steer vector models with LRU cache.

    Uses an LRU Cache. Every request, the requested steer vectors will be loaded 
    (unless they are already loaded) and least recently used steer vectors will
    be unloaded if the cache is above capacity.
    """

    _steer_vector_manager_cls: type[
        LRUCacheSteerVectorModelManager
    ] = LRUCacheSteerVectorModelManager

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        """Create LRU cache steer vector manager."""
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._steer_vector_manager_cls
        )
        self._adapter_manager: LRUCacheSteerVectorModelManager = (
            steer_vector_manager
        )
        return steer_vector_manager.model

    def _apply_adapters(
        self,
        steer_vector_requests: Set[SteerVectorRequest]
    ) -> None:
        """Apply adapters with LRU caching."""
        steer_vectors_map = {
            steer_vector_request.steer_vector_id: steer_vector_request
            for steer_vector_request in steer_vector_requests
            if steer_vector_request
        }
        
        if self._adapter_manager is None:
            return
        
        if len(steer_vectors_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested steer vectors "
                f"({len(steer_vectors_map)}) is greater "
                "than the number of GPU steer vector slots "
                f"({self._adapter_manager.adapter_slots})."
            )
        
        for steer_vector in steer_vectors_map.values():
            self.add_adapter(steer_vector)

    def add_adapter(
        self,
        steer_vector_request: SteerVectorRequest
    ) -> bool:
        """Add adapter with LRU cache management."""
        if self._adapter_manager is None:
            return False
        
        if steer_vector_request.steer_vector_id not in self.list_adapters():
            # Remove before we load the new steer vector to save memory
            if (len(self._adapter_manager._registered_adapters) + 1 
                > self._adapter_manager.capacity):
                self._adapter_manager.remove_oldest_adapter()
            
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        else:
            # Support replacing adapters with the same ID
            logger.debug(
                f"Replacing existing steer vector with ID "
                f"{steer_vector_request.steer_vector_id}"
            )
            self._adapter_manager.remove_adapter(steer_vector_request.steer_vector_id)
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        
        if not loaded:
            return False
        
        # Activate based on mode
        if steer_vector_request.is_multi_vector:
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                debug=steer_vector_request.debug,
                conflict_resolution=steer_vector_request.conflict_resolution
            )
        else:
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                target_layers=steer_vector_request.target_layers,
                prefill_trigger_tokens=steer_vector_request.prefill_trigger_tokens,
                prefill_trigger_positions=steer_vector_request.prefill_trigger_positions,
                prefill_exclude_tokens=steer_vector_request.prefill_exclude_tokens,
                prefill_exclude_positions=steer_vector_request.prefill_exclude_positions,
                generate_trigger_tokens=steer_vector_request.generate_trigger_tokens,
                debug=steer_vector_request.debug,
                normalize=steer_vector_request.normalize
            )
        
        return loaded
