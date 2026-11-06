# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4.1 native Attention/Engram path with UMDK routed experts on FFN."""

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers import fused_moe
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm_ascend.models.deepseek_v41 import model as native
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForCausalLM,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)

from afd_plugin.config import MEGAMOE_CONNECTOR, parse_afd_config
from afd_plugin.model_executor.models.npu.deepseek_v4 import (
    AFDDeepseekV4AttentionGateRemoteMoE,
    _checkpoint_weight_roles,
    _weight_layer_path,
)


def role_weights(weights: Iterable[tuple[str, torch.Tensor]], role: str):
    """FFN owns only routed experts/router; Attention owns all V4.1 state."""
    for name, tensor in weights:
        path = _weight_layer_path(name)
        if role == "ffn":
            if path is None or path[1] not in ("mlp", "ffn"):
                continue
            if path[2] and path[2][0] == "shared_experts":
                continue
        elif role not in _checkpoint_weight_roles(name, attention_shared_experts=True):
            continue
        yield name, tensor


class AFDDeepseekV41RemoteMoE(AFDDeepseekV4AttentionGateRemoteMoE):
    """Retain native V4.1's FP32 router input and explicit SP argument."""

    def __init__(self, *, config, layer_idx, prefix, vllm_config):
        super().__init__(
            config=config, layer_idx=layer_idx, prefix=prefix, vllm_config=vllm_config
        )
        # Text serving still loads the checkpoint's vision router parameter.
        # Vision requests are excluded by the connector configuration contract.
        self.gate.bias_vl = (
            nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            if config.vision_n_layers > 0
            else None
        )

    def forward(
        self,
        hidden_states,
        input_ids=None,
        hidden_states_fp32=None,
        already_sequence_parallel=False,
    ):
        if already_sequence_parallel:
            raise ValueError("DSV4.1 MegaMoE requires TP1 without sequence parallelism")
        return super().forward(hidden_states, input_ids, hidden_states_fp32)


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    # Upstream: vllm-ascend f9c8e0bd, DeepseekV41DecoderLayer.__init__.
    # Replace only MoE construction; inherited forward retains delayed HC,
    # exact FP32 normalization, Attention and Engram. Signature is unchanged.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: skip unsplit native MoE allocation
        nn.Module.__init__(self)
        # ### PATCH END: skip unsplit native MoE allocation

        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe

        attn_cls = self.attention_cls

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )

        # ### PATCH START: remote routed experts, local router/shared experts
        self.is_moe_layer = True
        self.mlp = AFDDeepseekV41RemoteMoE(
            config=config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.mlp",
            vllm_config=vllm_config,
        )
        # ### PATCH END: remote routed experts, local router/shared experts
        self.input_layernorm = native.RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=self.norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        # Leave the TP partial sums for the reduce-scatter below. The mHC
        # and MoE paths then stay sharded between attention calls.
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        has_engram = native._engram_enabled_for_runtime(config, vllm_config)
        self.engram: native.AscendEngram | None
        if (
            has_engram
            and not is_draft_layer
            and self.layer_idx in config.engram_layer_ids
        ):
            self.engram = native.AscendEngram(config, quant_config, f"{prefix}.engram")
        else:
            self.engram = None


class AFDDeepseekV41FFNLayer(nn.Module):
    """Allocate only the native routed-expert module on an FFN rank."""

    def __init__(self, vllm_config, prefix):
        super().__init__()
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_moe_layer = True
        config = copy(
            native.normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        )
        object.__setattr__(config, "n_shared_experts", None)
        native.FusedMoEFactory = fused_moe.FusedMoEFactory
        self.mlp = native.DeepseekV41MoE(
            config=config,
            parallel_config=vllm_config.parallel_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )


class AFDDeepseekV41Model(native.DeepseekV41Model):
    decoder_layer_cls = AFDDeepseekV41DecoderLayer

    # Upstream: f9c8e0bd DeepseekV41Model.__init__, same signature.
    # Delegate the large Attention constructor unchanged to preserve Engram,
    # KV/index sharing and bounded replay. FFN needs only the weight container.
    def __init__(self, *, vllm_config, prefix=""):
        # ### PATCH START: separate role allocations
        afd = parse_afd_config(vllm_config, validate=False)
        if afd.role == "attention":
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            nn.Module.__init__(self)
            self.config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
            self.has_engram = False
            self.engram_hash = None
            self.hc_mult = self.config.hc_mult
            self.start_layer, self.end_layer, self.layers = native.make_layers(
                self.config.num_hidden_layers,
                lambda prefix: AFDDeepseekV41FFNLayer(vllm_config, prefix),
                prefix=f"{prefix}.layers",
            )
            self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors
        # ### PATCH END: separate role allocations


class AFDDeepseekV41LLMForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    model_cls = AFDDeepseekV41Model
    afd_requires_input_ids = True

    # Upstream: f9c8e0bd AscendDeepseekV41LLMForCausalLM.__init__.
    # Same signature. Preserve native Attention initialization; FFN omits the
    # output head and Attention-only runtime. No native forward runs on FFN.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: FFN-only model initialization
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        if self.afd_config.connector != MEGAMOE_CONNECTOR:
            raise ValueError(
                "DSV4.1 AFD currently requires CAMAsyncMegamoeAFDConnector"
            )
        if self.afd_config.role == "attention":
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            nn.Module.__init__(self)
            self.config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
            self.quant_config = vllm_config.quant_config
            self.requires_uncompiled_fallback = True
            self.model = self.model_cls(
                vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model")
            )
            self.lm_head = native.PPMissingLayer()
            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors
            )
            self.num_moe_layers = self.config.num_hidden_layers
            self.set_moe_parameters()
            self.moe_comm_methods = {}
        # ### PATCH END: FFN-only model initialization

    # Upstream: f9c8e0bd AscendDeepseekV41LLMForCausalLM.load_weights.
    # Same signature; filter role ownership before delegating the native Engram
    # and deferred-weight policies. The large native name/quantization loader
    # remains authoritative and is not duplicated in the plugin.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: role-owned checkpoint stream
        return super().load_weights(role_weights(weights, self.afd_config.role))
        # ### PATCH END: role-owned checkpoint stream


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class AFDDeepseekV41ForCausalLM(AscendDeepseekV41ForCausalLM):
    language_model_cls = AFDDeepseekV41LLMForCausalLM
    afd_requires_input_ids = True

    @property
    def model(self):
        """Expose the role-local layer bank to the existing FFN runner."""
        return self.language_model.model
