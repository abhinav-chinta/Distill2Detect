# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture Request

Defines the request structure for enabling hidden states capture in vLLM.
"""

from typing import Optional, List
import msgspec


class HiddenStatesCaptureRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """
    Request to enable hidden states capture for specific inference requests.
    
    Similar to LoRARequest and SteerVectorRequest, but for hidden states capture.
    """
    
    request_id: str
    """Unique identifier for this capture request"""
    
    capture_layers: Optional[List[int]] = None
    """
    List of layer indices to capture. If None, captures all layers.
    Layer indices are 0-based (0 = first layer).
    """
    
    return_cpu: bool = True
    """Whether to return tensors on CPU (default: True to save GPU memory)"""
    
    def __post_init__(self):
        """Validation after initialization"""
        if self.capture_layers is not None:
            if not isinstance(self.capture_layers, list):
                raise ValueError("capture_layers must be a list of integers")
            if any(layer < 0 for layer in self.capture_layers):
                raise ValueError("Layer indices must be non-negative")
    
    def __eq__(self, value: object) -> bool:
        """
        Equality comparison based on request_id.
        """
        return isinstance(value, self.__class__) and self.request_id == value.request_id
    
    def __hash__(self) -> int:
        """
        Hash based on request_id for use in sets and dicts.
        """
        return hash(self.request_id)
    
    def should_capture_layer(self, layer_id: int) -> bool:
        """
        Check if a specific layer should be captured.
        
        Args:
            layer_id: Layer index (0-based)
            
        Returns:
            True if this layer should be captured
        """
        if self.capture_layers is None:
            return True  # Capture all layers
        return layer_id in self.capture_layers

