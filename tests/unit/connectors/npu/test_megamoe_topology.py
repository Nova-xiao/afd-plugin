# SPDX-License-Identifier: Apache-2.0
"""Exercise configuration validation without loading CANN or creating groups."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validate():
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_megamoe.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_megamoe_config"
        or isinstance(node, ast.Assign)
    ]
    from datetime import timedelta

    ns = dict(
        timedelta=timedelta,
        torch=SimpleNamespace(bfloat16="bf16"),
        get_ascend_device_type=lambda: "A5",
        AscendDeviceType=SimpleNamespace(A5="A5"),
    )
    exec(compile(tree, str(path), "exec"), ns)
    return ns["validate_megamoe_config"]


@pytest.mark.parametrize(
    "a,f,role,dp,valid",
    [
        (2, 2, "ffn", 2, True),
        (1, 1, "attention", 1, True),
        (3, 2, "attention", 3, True),
        (3, 2, "ffn", 2, True),
        (1, 4, "ffn", 4, True),
        (3, 2, "attention", 2, False),
        (2, 3, "ffn", 3, False),
        (0, 1, "ffn", 1, False),
        (1, 0, "attention", 1, False),
        (63, 1, "ffn", 1, True),
        (64, 1, "ffn", 1, False),
        (768, 1, "ffn", 1, False),
    ],
)
def test_role_sizes_and_tiling_limits(validate, a, f, role, dp, valid):
    obj = SimpleNamespace
    hf = obj(
        model_type="deepseek_v4",
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=6,
        expert_dtype="fp4",
    )
    parallel = obj(
        data_parallel_size=dp,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        enable_eplb=False,
        enable_elastic_ep=False,
        use_sequence_parallel_moe=False,
        eplb_config=obj(num_redundant_experts=0),
        enable_dbo=False,
        use_ubatching=False,
    )
    config = obj(
        model_config=obj(hf_config=hf, dtype="bf16", enforce_eager=True),
        additional_config={},
        parallel_config=parallel,
        use_v2_model_runner=True,
        speculative_config=None,
    )
    afd = obj(
        role=role,
        num_attention_ranks=a,
        num_ffn_ranks=f,
        async_dp=True,
        compute_gate_on_attention=True,
    )
    if valid:
        validate(config, afd)
    else:
        with pytest.raises(ValueError):
            validate(config, afd)
