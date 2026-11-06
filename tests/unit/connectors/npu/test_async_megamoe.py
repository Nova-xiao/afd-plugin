# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the UMDK call contract; no simulated NPU arithmetic."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def backend():
    # Execute the real connector with its external runtime imports omitted.
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_megamoe.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = dict(
        torch=torch,
        AFDConnectorBase=object,
        ROLE_SIZE=2,
        FLASH_GEOMETRY=(4096, 2048, 256, 6),
        MAX_LAYER_TENSORS=65536,
        AscendDeviceType=SimpleNamespace(A5="A5"),
        get_ascend_device_type=lambda: "A5",
    )
    exec(compile(tree, str(path), "exec"), ns)
    return SimpleNamespace(**ns)


@pytest.fixture
def config():
    hf = SimpleNamespace(
        model_type="deepseek_v41",
        hidden_size=5120,
        moe_intermediate_size=1024,
        n_routed_experts=384,
        num_experts_per_tok=6,
        num_hidden_layers=40,
        expert_dtype="fp4",
        hidden_act="silu",
    )
    p = SimpleNamespace(
        data_parallel_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        enable_eplb=False,
        enable_elastic_ep=False,
        use_sequence_parallel_moe=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        enable_dbo=False,
        use_ubatching=False,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf,
            dtype=torch.bfloat16,
            enforce_eager=True,
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda _: 0),
        ),
        parallel_config=p,
        additional_config={},
        use_v2_model_runner=True,
        speculative_config=None,
        engram_config=None,
    )


def test_accepts_v41_mxfp4_geometry(backend, config):
    afd = SimpleNamespace(
        num_attention_ranks=2,
        num_ffn_ranks=2,
        role="ffn",
        compute_gate_on_attention=True,
        async_dp=True,
    )
    backend.validate_megamoe_config(config, afd)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 5130),
        ("moe_intermediate_size", 4097),
        ("n_routed_experts", 383),
        ("num_hidden_layers", 65537),
        ("hidden_act", "gelu"),
        ("expert_dtype", "int4"),
    ],
)
def test_rejects_incompatible_kernel_contract(backend, config, field, value):
    setattr(config.model_config.hf_config, field, value)
    afd = SimpleNamespace(
        num_attention_ranks=2,
        num_ffn_ranks=2,
        role="ffn",
        compute_gate_on_attention=True,
        async_dp=True,
    )
    with pytest.raises(ValueError):
        backend.validate_megamoe_config(config, afd)


def test_dispatch_transmits_layer_and_actual_rows(backend):
    wire = []
    connector = object.__new__(backend.CAMAsyncMegamoeAFDConnector)
    connector.ticket = None
    connector.role_rank = 1
    connector.buffer = "window"
    connector.work = SimpleNamespace(
        next_sequence=lambda _: 1, publish=lambda *args: None
    )
    connector.ops = SimpleNamespace(
        megamoe_dispatch_send=lambda *args, **kw: wire.append((args, kw)) or "ticket"
    )
    context = SimpleNamespace(metadata=SimpleNamespace(stage_idx=0, layer_idx=37))
    connector.send_attn_output(
        torch.zeros(3, 64),
        context,
        topk_ids=torch.zeros(3, 6),
        topk_weights=torch.ones(3, 6),
    )
    args, kwargs = wire[0]
    assert args[0].shape == (3, 64)
    assert args[1].dtype == torch.int32
    assert args[2].dtype == torch.float32
    assert args[4:] == ([1], 1)
    assert kwargs == {"layer_index": 37}


def test_ffn_submits_all_layers_to_device_selection(backend, monkeypatch):
    wire = []
    connector = object.__new__(backend.CAMAsyncMegamoeAFDConnector)
    connector.role_rank = 0
    connector.buffer = "window"
    connector.weights = (["w10", "w11"], ["w20", "w21"], ["s10", "s11"], ["s20", "s21"])
    connector.work = SimpleNamespace(poll=lambda _: (1, 1, 32))
    connector.ops = SimpleNamespace(megamoe_fused_moe=lambda *args: wire.append(args))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False
    )
    connector.execute_next_experts()
    assert wire == [(*connector.weights, "window", [1], 32)]
