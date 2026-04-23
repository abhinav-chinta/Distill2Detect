# SPDX-License-Identifier: Apache-2.0
"""
Configuration for steer vector wrappers and supported model architectures.

This module provides centralized configuration for:
1. Supported model layer class names
2. Wrapper registry for different intervention granularities
3. Extensibility for future intervention types (attention, MLP, etc.)
"""

from typing import Dict, List, Any


# ============================================================================
# Supported DecoderLayer Class Names
# ============================================================================
# List of DecoderLayer class names for automatic recognition
# Organized alphabetically by model name for easier maintenance

SUPPORTED_DECODER_LAYERS: List[str] = [
    # A
    "ApertusDecoderLayer",
    "ArceeDecoderLayer",
    "ArcticDecoderLayer",
    "AriaTextDecoderLayer",
    
    # B
    "BaiChuanDecoderLayer",
    "BailingMoeBlock",
    "BambaAttentionDecoderLayer",
    "BambaMixerDecoderLayer",
    "BertLayer",
    "BloomBlock",
    
    # C
    "ChameleonDecoderLayer",
    "ChameleonSwinDecoderLayer",
    "CohereDecoderLayer",
    
    # D
    "DbrxBlock",
    "DeciLMDecoderLayer",
    "DecoderLayer",
    "DeepseekDecoderLayer",
    "DeepseekV2DecoderLayer",
    "Dots1DecoderLayer",
    
    # E
    "Ernie4_5_MoeDecoderLayer",
    "Ernie4_5_VLMoeDecoderLayer",
    "Exaone4DecoderLayer",
    "ExaoneDecoderLayer",
    
    # F
    "FalconDecoderLayer",
    "FalconH1AttentionDecoderLayer",
    "FalconH1SSMDecoderLayer",
    "FlashDecoderLayer",
    "FlexOlmoDecoderLayer",
    
    # G
    "Gemma2DecoderLayer",
    "Gemma3DecoderLayer",
    "Gemma3nDecoderLayer",
    "GemmaDecoderLayer",
    "Glm4DecoderLayer",
    "Glm4MoeDecoderLayer",
    "GLMBlock",
    "GPT2Block",
    "GPTBigCodeBlock",
    "GPTJBlock",
    "GPTNeoXLayer",
    "GraniteDecoderLayer",
    "GraniteMoeDecoderLayer",
    "GraniteMoeHybridAttentionDecoderLayer",
    "GraniteMoeHybridMambaDecoderLayer",
    "GraniteMoeSharedDecoderLayer",
    "Grok1DecoderLayer",
    
    # H
    "HunYuanDecoderLayer",
    
    # I
    "InternLM2VEDecoderLayer",
    "InternLMDecoderLayer",
    
    # J
    "JAISBlock",
    "JambaAttentionDecoderLayer",
    "JambaMambaDecoderLayer",
    
    # L
    "Lfm2AttentionDecoderLayer",
    "Lfm2MoeAttentionDecoderLayer",
    "Lfm2MoeShortConvDecoderLayer",
    "Lfm2ShortConvDecoderLayer",
    "Llama4DecoderLayer",
    "LlamaDecoderLayer",
    
    # M
    "Mamba2DecoderLayer",
    "MambaDecoderLayer",
    "MiniCPM3DecoderLayer",
    "MiniCPMDecoderLayer",
    "MiniMaxText01DecoderLayer",
    "MixtralDecoderLayer",
    "MolmoDecoderLayer",
    "MolmoDecoderNormAfterLayer",
    "MPTBlock",
    
    # N
    "NemotronDecoderLayer",
    "NemotronHAttentionDecoderLayer",
    "NemotronHMambaDecoderLayer",
    "NemotronHMLPDecoderLayer",
    "NemotronHMoEDecoderLayer",
    
    # O
    "Olmo2DecoderLayer",
    "OlmoDecoderLayer",
    "OlmoeDecoderLayer",
    "OPTDecoderLayer",
    "OrionDecoderLayer",
    
    # P
    "PersimmonDecoderLayer",
    "PhiLayer",
    "PhiMoEDecoderLayer",
    "Plamo2DecoderLayer",
    
    # Q
    "Qwen2DecoderLayer",
    "Qwen2MoeDecoderLayer",
    "Qwen3DecoderLayer",
    "Qwen3MoeDecoderLayer",
    "Qwen3NextDecoderLayer",
    "QWenBlock",
    
    # S
    "SeedOssDecoderLayer",
    "SolarDecoderLayer",
    "StablelmDecoderLayer",
    "Starcoder2DecoderLayer",
    "Step3TextDecoderLayer",
    
    # T
    "TransformerBlock",
    
    # W
    "WhisperDecoderLayer",
    
    # Z
    "Zamba2AttentionDecoderLayer",
    "Zamba2HybridLayer",
    "Zamba2MambaDecoderLayer",
]


# ============================================================================
# Wrapper Registry Configuration
# ============================================================================
# Registry for different intervention granularities
# This allows for future extensibility to attention-level, MLP-level, etc.

WRAPPER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # DecoderLayer-level intervention (currently supported)
    "decoder_layer": {
        "wrapper_class": "DecoderLayerWithSteerVector",
        "target_modules": SUPPORTED_DECODER_LAYERS,
        "enabled": True,
        "description": "Full decoder layer intervention on complete hidden states",
    },
    
    # Future extension examples (currently disabled):
    
    # "attention": {
    #     "wrapper_class": "AttentionWithSteerVector",
    #     "target_modules": [
    #         "Attention",
    #         "FlashAttention2",
    #         "SelfAttention",
    #         "MultiHeadAttention",
    #         # Add more attention module names...
    #     ],
    #     "enabled": False,
    #     "description": "Attention output intervention",
    # },
    
    # "mlp": {
    #     "wrapper_class": "MLPWithSteerVector",
    #     "target_modules": [
    #         "MLP",
    #         "FeedForward",
    #         "GatedMLP",
    #         # Add more MLP module names...
    #     ],
    #     "enabled": False,
    #     "description": "MLP output intervention",
    # },
    
    # "residual_stream": {
    #     "wrapper_class": "ResidualStreamWithSteerVector",
    #     "target_modules": ["ResidualConnection"],
    #     "enabled": False,
    #     "description": "Residual stream intervention",
    # },
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_enabled_wrappers() -> Dict[str, Dict[str, Any]]:
    """Get only the enabled wrapper configurations."""
    return {
        wrapper_type: config
        for wrapper_type, config in WRAPPER_REGISTRY.items()
        if config.get("enabled", False)
    }


def get_target_modules(wrapper_type: str) -> List[str]:
    """Get target module names for a specific wrapper type."""
    config = WRAPPER_REGISTRY.get(wrapper_type)
    if config is None:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")
    return config.get("target_modules", [])


def is_wrapper_enabled(wrapper_type: str) -> bool:
    """Check if a wrapper type is enabled."""
    config = WRAPPER_REGISTRY.get(wrapper_type)
    if config is None:
        return False
    return config.get("enabled", False)

