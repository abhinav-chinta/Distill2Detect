# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for vLLM

Provides functionality to capture intermediate hidden states from transformer models
during inference. Integrated into vLLM's V1 worker architecture.
"""

from vllm.hidden_states.storage import HiddenStatesStore
from vllm.hidden_states.wrapper import VLLMTransformerLayerWrapper
from vllm.hidden_states.request import HiddenStatesCaptureRequest
from vllm.hidden_states.utils import (
    deserialize_hidden_states,
    print_hidden_states_summary,
)

__all__ = [
    "HiddenStatesStore",
    "VLLMTransformerLayerWrapper",
    "HiddenStatesCaptureRequest",
    "deserialize_hidden_states",
    "print_hidden_states_summary",
]

__version__ = "1.0.0"

