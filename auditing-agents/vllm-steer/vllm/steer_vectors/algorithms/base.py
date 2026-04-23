# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import torch


class BaseSteerVectorAlgorithm(ABC):
    """
    Base interface for steer vector algorithms.
    
    This class defines the core interface that all algorithm implementations must follow.
    Parameter management is handled by InterventionController in parameter_control.py,
    allowing algorithm developers to focus purely on transformation logic.
    """

    def __init__(self, layer_id: Optional[int] = None):
        """
        Initialize algorithm with layer ID.
        
        Args:
            layer_id: Layer index where this algorithm will be applied
        """
        self.layer_id = layer_id

    @classmethod
    @abstractmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load steer vector data from file path and return a dictionary containing parameters."""
        pass

    @abstractmethod
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """Set steer vector parameters."""
        pass

    @abstractmethod
    def reset_steer_vector(self, index: int) -> None:
        """Reset steer vector."""
        pass

    @abstractmethod
    def set_active_tensor(self, index: int) -> None:
        """Set active steer vector."""
        pass

    @abstractmethod
    def apply_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply steer vector intervention."""
        pass
