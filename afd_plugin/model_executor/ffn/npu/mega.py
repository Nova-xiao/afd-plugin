# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention MoE execution for the asynchronous Mega connector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout import (
    CAMDispatchLayout,
    CAMDispatchPayload,
    restore_cam_dispatch_output,
)
from afd_plugin.connectors.npu.async_mega.dispatch import (
    build_mega_dispatch_payload,
    plain_tp_layout,
)
from afd_plugin.connectors.npu.async_mega.remote import send_and_build_receive
from afd_plugin.model_executor.ffn.npu.remote import (
    _cam_gate_topk,
    local_hash_input_ids,
)
from afd_plugin.model_executor.ffn.remote import (
    AFDRemoteMoERunner,
    _current_afd_metadata,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import ReplicatedLinear


class AFDAsyncMegaMoERunner(AFDRemoteMoERunner):
    """Route local token rows and return the result to the sending microbatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_sequence_parallel_moe = (
            get_current_vllm_config().parallel_config.use_sequence_parallel_moe
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        attention_gate: ReplicatedLinear | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        assert attention_gate is not None, "Async Mega requires an Attention gate"
        # The model passes its FP32 normalized gate input as router_logits,
        # following the native runner interface. Projection happens below.
        payload = self._prepare_dispatch(
            hidden_states, router_logits, input_ids, attention_gate
        )
        metadata = _current_afd_metadata()
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=metadata.stage_idx,
                seq_len=int(payload.hidden_states.shape[0]),
            )
        )
        receive = send_and_build_receive(
            metadata.connector,
            payload.hidden_states,
            context,
            topk_weights=payload.topk_weights,
            topk_ids=payload.topk_ids,
        )
        shared = shared_experts(hidden_states) if shared_experts is not None else None
        maybe_apply_dbo_yield(hidden_states, role="attention")
        output = receive()
        output = restore_cam_dispatch_output(output, payload.layout)
        return output if shared is None else output + shared

    def _prepare_dispatch(
        self,
        hidden_states: torch.Tensor,
        gate_input: torch.Tensor,
        input_ids: torch.Tensor | None,
        gate: ReplicatedLinear,
    ) -> CAMDispatchPayload:
        # Load Ascend only when executing NPU work, not during runner registration.
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        use_sp = self.use_sequence_parallel_moe
        tp = get_tp_group() if not use_sp else None
        parent_tokens = hidden_states.shape[0]
        if tp is not None and tp.world_size > 1:
            layout = plain_tp_layout(parent_tokens, tp.rank_in_group, tp.world_size)
        else:
            layout = CAMDispatchLayout(
                parent_tokens=parent_tokens,
                padded_tokens=parent_tokens,
                local_token_slice=slice(0, parent_tokens),
                requires_tp_all_gather=False,
                use_sequence_parallel=use_sp,
                tp_rank=0,
                tp_size=1,
            )
        rows = layout.local_token_slice
        local_routing = layout.requires_tp_all_gather and not _EXTRA_CTX.in_profile_run
        if local_routing:
            if gate.tid2eid is not None:
                if input_ids is None:
                    input_ids = get_forward_context().input_ids
                input_ids = local_hash_input_ids(
                    input_ids=input_ids, router_tokens=parent_tokens
                )[rows]
            hidden_states = hidden_states[rows]
            gate_input = gate_input[rows]
        if local_routing and hidden_states.shape[0] == 0:
            shape = (0, self.routed_experts.top_k)
            weights = hidden_states.new_empty(shape, dtype=torch.float32)
            ids = hidden_states.new_empty(shape, dtype=torch.int32)
        else:
            weights, ids = _cam_gate_topk(
                self.routed_experts, gate, gate_input, input_ids
            )
        if not local_routing:
            hidden_states, weights, ids = hidden_states[rows], weights[rows], ids[rows]
        return build_mega_dispatch_payload(hidden_states, weights, ids, layout)
