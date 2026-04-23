# SPDX-License-Identifier: Apache-2.0
"""
V1 Worker Mixin for Hidden States Capture

This mixin integrates hidden states capture into vLLM V1's GPUModelRunner,
similar to how SteerVectorModelRunnerMixin works.
"""

from typing import Dict
import torch
from torch import nn

from vllm.hidden_states.wrapper import VLLMTransformerLayerWrapper
from vllm.hidden_states.storage import HiddenStatesStore


class HiddenStatesModelRunnerMixin:
    """
    Mixin for GPUModelRunner to support hidden states capture.
    
    This works within the V1 worker process, wrapping layers at initialization
    and capturing hidden states during forward passes.
    
    Note: This mixin uses lazy initialization to avoid MRO issues.
    Attributes are created on first access, not in __init__.
    """
    
    def _wrap_model_for_hidden_states(self, model: nn.Module) -> nn.Module:
        """
        Wrap model layers for hidden states capture.
        
        This should be called during load_model() to prepare the model.
        Similar to steer vector's _wrap_model_with_steer_vectors().
        
        Args:
            model: The model to wrap
            
        Returns:
            The wrapped model
        """
        # Initialize store if not exists
        if not hasattr(self, 'hidden_states_store'):
            self.hidden_states_store = HiddenStatesStore()
            self.hidden_states_capture_enabled = False
            self._hidden_states_wrapped = False
        
        # Skip if already wrapped
        if self._hidden_states_wrapped:
            return model
        
        # Import here to avoid circular dependency
        from vllm.steer_vectors.config import SUPPORTED_DECODER_LAYERS
        
        layer_id = 0
        wrapped_count = 0
        
        def wrap_module(module: nn.Module, name: str = "") -> nn.Module:
            nonlocal layer_id, wrapped_count
            
            # Get class name
            class_name = module.__class__.__name__
            
            # Check if this is a decoder layer
            if class_name in SUPPORTED_DECODER_LAYERS:
                # Check if already wrapped
                if not isinstance(module, VLLMTransformerLayerWrapper):
                    wrapped_layer = VLLMTransformerLayerWrapper(
                        module, layer_id, name, self.hidden_states_store
                    )
                    layer_id += 1
                    wrapped_count += 1
                    return wrapped_layer
            
            # Recursively process child modules
            for child_name, child_module in list(module.named_children()):
                full_child_name = f"{name}.{child_name}" if name else child_name
                wrapped_child = wrap_module(child_module, full_child_name)
                if wrapped_child is not child_module:
                    setattr(module, child_name, wrapped_child)
            
            return module
        
        # Wrap the model and return
        wrapped_model = wrap_module(model)
        self._hidden_states_wrapped = True
        
        if wrapped_count > 0:
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info(f"Wrapped {wrapped_count} decoder layers for hidden states capture")
        else:
            import warnings
            warnings.warn("No decoder layers were wrapped for hidden states capture")
        
        return wrapped_model
    
    def enable_hidden_states_capture(self):
        """
        Enable hidden states capture.
        
        This method only controls the capture switch, not model wrapping.
        Model wrapping is done during load_model() via _wrap_model_for_hidden_states().
        """
        if not hasattr(self, 'hidden_states_store'):
            raise RuntimeError(
                "Hidden states store not initialized. "
                "This should not happen if the model was loaded properly."
            )
        
        if not self._hidden_states_wrapped:
            raise RuntimeError(
                "Model not wrapped for hidden states capture. "
                "This should not happen if the model was loaded properly."
            )
        
        self.hidden_states_store.enable_capture()
        self.hidden_states_store.clear()
        self.hidden_states_store.enable_multi_batch_mode()  # Enable multi-batch support
        self.hidden_states_capture_enabled = True
    
    def disable_hidden_states_capture(self):
        """Disable hidden states capture"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            self.hidden_states_store.disable_capture()
            self.hidden_states_capture_enabled = False
    
    def get_captured_hidden_states(self) -> Dict[int, Dict[str, any]]:
        """
        Get captured hidden states from the store.
        
        Returns tensors as serializable dictionaries for RPC transmission.
        Each tensor is converted to: {
            'data': tensor values as list (converted to float32 for compatibility),
            'shape': list(tensor.shape),
            'dtype': str(original tensor.dtype)
        }
        """
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            # Finalize multi-batch mode if enabled (combines all forward passes)
            self.hidden_states_store.finalize_multi_batch()
            
            result = {}
            for layer_id, tensor in self.hidden_states_store.hidden_states.items():
                # Move to CPU
                cpu_tensor = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                
                # Convert bfloat16 to float32 for numpy compatibility
                if cpu_tensor.dtype == torch.bfloat16:
                    cpu_tensor = cpu_tensor.to(torch.float32)
                
                # Convert to numpy and then to list for serialization
                result[layer_id] = {
                    'data': cpu_tensor.numpy().tolist(),
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype)  # Store original dtype
                }
            return result
        return {}
    
    def get_hidden_states_debug_info(self) -> Dict[str, any]:
        """Get debug information about hidden states capture"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            store = self.hidden_states_store
            return {
                'capture_enabled': store.capture_enabled,
                'multi_batch_mode': store.multi_batch_mode,
                'finalized': store.finalized,
                'num_batches_captured': len(store.batch_hidden_states),
                'batch_shapes': [
                    {layer_id: list(tensor.shape) for layer_id, tensor in batch.items()}
                    for batch in store.batch_hidden_states
                ],
                'current_shapes': {
                    layer_id: list(tensor.shape) 
                    for layer_id, tensor in store.hidden_states.items()
                },
            }
        return {}
    
    def clear_hidden_states(self):
        """Clear stored hidden states"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            self.hidden_states_store.clear()

