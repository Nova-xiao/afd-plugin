# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 Flash admission and loaded weights for the Mega backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.npu.async_mega.mega_layers import MegaModelSpec
from afd_plugin.connectors.npu.async_mega.mega_weights import convert_expert_weights

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.connectors.npu.async_mega.mega_layers import LayerWeights
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4DecoderLayer,
    )

FLASH_GEOMETRY = (4096, 2048, 256, 6)


def get_mega_model_spec(config: VllmConfig) -> MegaModelSpec:
    # Extend model admission here when another adapter is validated. Connector
    # and FFN execution consume the same explicit dimensions/weights contract.
    if config.model_config.hf_config.model_type == "deepseek_v4":
        return flash_mega_model_spec(config)
    raise ValueError("Mega currently supports the DeepSeek-V4-Flash model adapter")


def flash_mega_model_spec(config: VllmConfig) -> MegaModelSpec:
    """Keep the accepted checkpoint family in the model adapter."""
    hf = config.model_config.hf_config
    geometry = (
        hf.hidden_size,
        hf.moe_intermediate_size,
        hf.n_routed_experts,
        hf.num_experts_per_tok,
    )
    if (
        hf.model_type != "deepseek_v4"
        or geometry != FLASH_GEOMETRY
        or hf.expert_dtype != "fp4"
    ):
        raise ValueError("Mega model adapter requires native FP4 DeepSeek-V4-Flash")
    if config.additional_config.get("enable_fused_mc2", 0):
        raise ValueError(
            "Mega split requires enable_fused_mc2=0 for packed layer weights"
        )
    return MegaModelSpec(*geometry, activation_parameters=(hf.swiglu_limit, 1.702, 1.0))


def extract_mega_expert_weights(
    layers: Iterable[AFDDeepseekV4DecoderLayer],
    rank: int,
    hidden: int,
    intermediate: int,
) -> dict[int, LayerWeights]:
    """Extract native parameter views, convert storage, and release the old copy."""
    from vllm_ascend.quantization.quant_type import QuantType

    result: dict[int, LayerWeights] = {}
    for layer in layers:
        if layer.layer_idx in result:
            raise ValueError("Mega requires unique model layer IDs")
        moe = layer.mlp
        experts = moe.experts
        owner = experts.routed_experts
        if experts.quant_type != QuantType.W4A8MXFP or owner.dynamic_eplb:
            raise ValueError("Mega routed experts require static W4A8 MXFP weights")
        local = moe.n_local_physical_experts
        if moe.ep_rank != rank or moe.physical_expert_start != rank * local:
            raise ValueError("FFN EP rank must own contiguous experts")
        # Native post-load weights/scales use transposed views. Present the
        # backend's output-major contract, independently of native attribute names.
        weights = convert_expert_weights(
            owner.w13_weight.detach().transpose(1, 2),
            owner.w2_weight.detach().transpose(1, 2),
            owner.w13_weight_scale.detach().view(torch.uint8).transpose(1, 2),
            owner.w2_weight_scale.detach().view(torch.uint8).transpose(1, 2),
            hidden=hidden,
            intermediate=intermediate,
        )
        owner.w13_weight.data = weights[0][0].transpose(1, 2)
        owner.w2_weight.data = weights[1][0].transpose(1, 2)
        result[layer.layer_idx] = weights
    return result
