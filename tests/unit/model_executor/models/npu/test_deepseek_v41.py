# SPDX-License-Identifier: Apache-2.0
"""V4.1 role ownership and allocation, without executing Ascend kernels."""

from __future__ import annotations

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class NativeMoE(nn.Module):
    def __init__(self, *, config, **kwargs):
        super().__init__()
        self.expert = nn.Parameter(torch.ones(1))
        if config.n_shared_experts is not None:
            self.shared = nn.Parameter(torch.ones(1))


class NativeModel(nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.engram = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.engram

    def _make_empty_intermediate_tensors(self, *args):
        return {}


class NativeDecoder(nn.Module):
    def __init__(self, vllm_config, prefix):
        raise AssertionError("Attention must not allocate the unsplit native MoE")


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class RemoteMoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(1))


@pytest.fixture
def adapter():
    root = Path(__file__).resolve().parents[5]
    functions = {"_weight_layer_path", "_checkpoint_weight_roles", "role_weights"}
    classes = {
        "AFDDeepseekV41DecoderLayer",
        "AFDDeepseekV41FFNLayer",
        "AFDDeepseekV41Model",
    }
    nodes = []
    for module in ("deepseek_v4", "deepseek_v41"):
        path = root / f"afd_plugin/model_executor/models/npu/{module}.py"
        nodes.extend(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name in functions
            or isinstance(n, ast.ClassDef)
            and n.name in classes
        )

    def make_layers(count, factory, prefix):
        return 0, count, nn.ModuleList(factory(f"{prefix}.{i}") for i in range(count))

    native = SimpleNamespace(
        DeepseekV41Model=NativeModel,
        DeepseekV41DecoderLayer=NativeDecoder,
        RMSNorm=lambda *a, **kw: nn.Identity(),
        _engram_enabled_for_runtime=lambda *a: False,
        DeepseekV41MoE=NativeMoE,
        normalize_deepseek_v41_config=lambda c: c,
        make_layers=make_layers,
    )
    ns = dict(
        nn=nn,
        torch=torch,
        copy=copy,
        native=native,
        AFDDeepseekV41RemoteMoE=RemoteMoE,
        fused_moe=SimpleNamespace(FusedMoEFactory=object),
        parse_afd_config=lambda c, validate: c.afd,
        _ATTENTION_ROLE="attention",
        _FFN_ROLE="ffn",
        _BOTH_ROLES=frozenset(("attention", "ffn")),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "adapter", "exec"), ns)
    ns["AFDDeepseekV41DecoderLayer"].attention_cls = Attention
    return SimpleNamespace(**ns)


@pytest.mark.parametrize(
    "name,roles",
    [
        ("layers.0.ffn.experts.0.w1.weight", {"ffn"}),
        ("layers.0.ffn.shared_experts.w1.weight", {"attention"}),
        ("layers.0.ffn.gate.bias_vl", {"attention", "ffn"}),
        ("layers.0.attn.wq_a.weight", {"attention"}),
        ("layers.0.engram.embed.weight", {"attention"}),
        ("layers.0.hc_ffn_fn", {"attention"}),
        ("embed.weight", {"attention"}),
        ("head.weight", {"attention"}),
    ],
)
def test_checkpoint_ownership(adapter, name, roles):
    value = torch.ones(1)
    actual = {
        role
        for role in ("attention", "ffn")
        if list(adapter.role_weights([(name, value)], role))
    }
    assert actual == roles


def test_ffn_allocates_only_routed_expert_layers(adapter):
    hf = SimpleNamespace(n_shared_experts=1, num_hidden_layers=2, hc_mult=4)
    config = SimpleNamespace(
        afd=SimpleNamespace(role="ffn"),
        model_config=SimpleNamespace(hf_config=hf),
        parallel_config=None,
        quant_config=None,
    )
    model = adapter.AFDDeepseekV41Model(vllm_config=config, prefix="model")
    assert set(dict(model.named_parameters())) == {
        "layers.0.mlp.expert",
        "layers.1.mlp.expert",
    }
    assert hf.n_shared_experts == 1


def test_attention_preserves_native_model_forward(adapter):
    config = SimpleNamespace(afd=SimpleNamespace(role="attention"))
    model = adapter.AFDDeepseekV41Model(vllm_config=config, prefix="model")
    assert torch.equal(model(torch.tensor([2.0])), torch.tensor([3.0]))


def test_attention_decoder_does_not_allocate_native_experts(adapter):
    hf = SimpleNamespace(
        hidden_size=4,
        rms_norm_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        rope_parameters={"original_max_position_embeddings": 128},
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )
    layer = adapter.AFDDeepseekV41DecoderLayer(config, "model.layers.0")
    names = set(dict(layer.named_parameters()))
    assert {"self_attn.weight", "mlp.gate", "hc_attn_fn", "hc_ffn_fn"} <= names
    assert not any("expert" in name for name in names)
