# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4 model-to-Mega runner routing and microbatch completion on CPU."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)

from afd_plugin.connectors import AFDConnectorFactory  # noqa: E402
from afd_plugin.connectors.npu.async_cam_runtime import async_cam_layout  # noqa: E402
from afd_plugin.model_executor.ffn.npu import mega  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.vllm_config import attention_moe_vllm_config  # noqa: E402


@pytest.fixture
def mega_config():
    config = attention_moe_vllm_config(
        connector="CAMAsyncMegaAFDConnector", compute_gate_on_attention=True
    )
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.enable_expert_parallel = True
    return config


def test_mega_registration_constructs_weightless_experts(cam_moe, mega_config):
    block = cam_moe(gate=True, vllm_config=mega_config)
    assert isinstance(block.experts, mega.AFDAsyncMegaMoERunner)
    process_weights_after_loading(
        block.experts,
        SimpleNamespace(quantization=None, word_embeddings_untied_by_checkpoint=False),
        torch.device("cpu"),
    )
    assert list(block.experts.parameters()) == []
    assert set(block.state_dict()) == {"gate.weight", "gate.e_score_correction_bias"}
    assert (
        AFDConnectorFactory.get_attention_moe_runner("CAMAsyncMegaAFDConnector", "cuda")
        is None
    )


@pytest.mark.parametrize(
    "tokens,rank,sp,profile",
    [
        (5, 0, False, False),
        (5, 1, False, False),
        (1, 1, False, False),
        (5, 1, False, True),
        (5, 1, True, False),
        (5, 1, True, True),
    ],
)
def test_mega_routes_and_completes_on_owning_microbatch(
    cam_moe, mega_config, monkeypatch, tokens, rank, sp, profile
):  # long-test: verify routing, wire padding and owner-only completion together
    mega_config.parallel_config.all2all_backend = (
        "allgather_reducescatter" if sp else "flashinfer_all2allv"
    )
    mega_config.parallel_config.data_parallel_size = 2 if sp else 1
    hidden = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    normalized = hidden + 0.125
    input_ids = torch.arange(tokens) % 4
    width = tokens if sp else (tokens + 1) // 2
    rows = slice(0, tokens) if sp else slice(rank * width, (rank + 1) * width)
    valid = hidden[rows].shape[0]
    events = []
    metadata = install_proxy_context(monkeypatch, [], yield_module=mega)
    metadata.stage_idx = 1
    monkeypatch.setattr(
        mega, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=rank)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(_EXTRA_CTX=SimpleNamespace(in_profile_run=profile)),
    )

    class Shared(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, states):
            events.append("shared")
            torch.testing.assert_close(states, hidden)
            return states * 3

    block = cam_moe(
        gate=True,
        vllm_config=mega_config,
        routing_config={"scoring_func": "sqrtsoftplus", "num_hash_layers": 4},
        shared_cls=Shared,
    )
    block.gate.weight_fp32 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    block.gate.tid2eid.data = torch.tensor([[2, 1], [0, 2], [1, 0], [2, 0]])
    route_rows = slice(None) if sp or profile else rows

    def select(*, x, input_ids, tid2eid, **kwargs):
        events.append("route")
        torch.testing.assert_close(x, normalized[route_rows] @ block.gate.weight_fp32.T)
        torch.testing.assert_close(input_ids, (torch.arange(tokens) % 4)[route_rows])
        assert kwargs["routed_scaling_factor"] == 1.5
        return torch.full((len(x), 2), 1.5), tid2eid[input_ids].int(), None

    monkeypatch.setattr(
        torch.ops._C_ascend, "moe_gating_top_k_hash", select, raising=False
    )
    monkeypatch.setattr(
        torch.ops.vllm, "afd_balance_routing_ids_", lambda *_: None, raising=False
    )

    def send(states, context, **payload):
        events.append("send")
        assert context.metadata.layer_idx == 3
        assert context.metadata.stage_idx == 1
        assert context.metadata.seq_lens == [width]
        torch.testing.assert_close(states[:valid], hidden[rows])
        torch.testing.assert_close(
            payload["topk_ids"][:valid], block.gate.tid2eid[input_ids[rows]].int()
        )
        torch.testing.assert_close(
            payload["topk_weights"][:valid], torch.full((valid, 2), 1.5)
        )
        if valid < width:
            assert states[valid:].count_nonzero() == 0
            assert payload["topk_weights"][valid:].count_nonzero() == 0
            assert payload["topk_ids"][valid:].tolist() == [[0, 1]]

    def recv(*, ref_tensor, ubatch_idx, context):
        events.append("recv")
        assert ubatch_idx == context.metadata.stage_idx == 1
        return ref_tensor + 100

    def yield_to_peer(states, *, role):
        events.append("yield")
        metadata.stage_idx = 0
        return states

    def gather(output, dim):
        events.append("gather")
        torch.testing.assert_close(output[:valid], hidden[rows] + 100)
        return torch.cat((hidden, torch.zeros(width * 2 - tokens, 4))) + 100

    metadata.connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=recv)
    monkeypatch.setattr(mega, "maybe_apply_dbo_yield", yield_to_peer)
    monkeypatch.setattr(async_cam_layout, "tensor_model_parallel_all_gather", gather)
    output = block(hidden, input_ids=input_ids, hidden_states_fp32=normalized)
    torch.testing.assert_close(output, hidden * 4 + 100)
    assert events == ([] if valid == 0 and not profile else ["route"]) + [
        "send",
        "shared",
        "yield",
        "recv",
    ] + ([] if sp else ["gather"])
