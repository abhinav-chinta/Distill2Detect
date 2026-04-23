# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Type, Any

# Forward declaration to avoid circular imports
class BaseSteerVectorAlgorithm:
    pass

# Global algorithm registry
ALGORITHM_REGISTRY: Dict[str, Type["BaseSteerVectorAlgorithm"]] = {}


def register_algorithm(name: str):
    """
    Decorator for registering algorithm classes to the global registry.
    
    Args:
        name: Unique name of the algorithm (e.g., "direct", "loreft").
    """
    def decorator(cls: Type["BaseSteerVectorAlgorithm"]):
        if name in ALGORITHM_REGISTRY:
            # In practice, can be changed to a more lenient strategy, e.g., logging.warning
            raise ValueError(f"Algorithm '{name}' is already registered.")
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def create_algorithm(name: str, *args, **kwargs) -> "BaseSteerVectorAlgorithm":
    """
    Algorithm factory function that creates algorithm instances by name.
    
    Args:
        name: Name of the algorithm to create.
        *args, **kwargs: Arguments passed to the algorithm constructor.
        
    Returns:
        An instance of BaseSteerVectorAlgorithm.
        
    Raises:
        ValueError: If the algorithm name is not registered.
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: '{name}'. Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    
    # Import the actual definition of BaseSteerVectorAlgorithm
    from .base import BaseSteerVectorAlgorithm as ConcreteBase
    
    cls = ALGORITHM_REGISTRY[name]
    
    # Ensure the returned instance type is correct
    instance: ConcreteBase = cls(*args, **kwargs)
    return instance 