# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for Steer Vectors."""

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal
import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

SteerVectorDType = Literal["auto", "float16", "bfloat16", "float32"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SteerVectorConfig:
    """Configuration for Steer Vectors.
    
    Steer vectors allow runtime intervention in model behavior by adding
    control vectors to hidden states at specific layers.
    """

    max_steer_vectors: int = Field(default=1, ge=1)
    """Maximum number of steer vectors in a single batch."""
    
    max_cpu_steer_vectors: int | None = None
    """Maximum number of steer vectors to store in CPU memory. 
    Must be >= max_steer_vectors. If None, defaults to max_steer_vectors."""
    
    steer_vector_dtype: SteerVectorDType = "auto"
    """Data type for steer vectors. If 'auto', will default to base model dtype."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        factors: list[Any] = []
        factors.append(self.max_steer_vectors)
        factors.append(self.steer_vector_dtype)

        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()
        return hash_str

    @property
    def adapter_dtype(self) -> torch.dtype:
        """Backward compatibility alias for steer_vector_dtype.
        
        Returns actual torch.dtype, converting "auto" to float16 as default.
        """
        if isinstance(self.steer_vector_dtype, torch.dtype):
            return self.steer_vector_dtype
        
        # Convert string dtype to torch.dtype
        dtype_map = {
            "auto": torch.float16,  # Default to float16 for "auto"
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        if self.steer_vector_dtype in dtype_map:
            return dtype_map[self.steer_vector_dtype]
        
        # Fallback to float16 if unknown
        logger.warning(
            f"Unknown steer_vector_dtype: {self.steer_vector_dtype}, "
            f"defaulting to float16"
        )
        return torch.float16

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.max_cpu_steer_vectors is None:
            self.max_cpu_steer_vectors = self.max_steer_vectors
        if self.max_cpu_steer_vectors < self.max_steer_vectors:
            raise ValueError(
                f"max_cpu_steer_vectors ({self.max_cpu_steer_vectors}) "
                f"must be >= max_steer_vectors ({self.max_steer_vectors})"
            )
        return self


