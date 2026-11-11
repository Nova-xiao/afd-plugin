# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 eager MegaMoE backend for DeepSeek-V4-Flash."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.npu.megamoe_work import POLL_SECONDS, ExpertWorkQueue
from afd_plugin.distributed.afd_process_group import (
    DefaultProcessGroupSwitcher,
    init_afd_process_group,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig
    from afd_plugin.connectors.metadata import AFDTransferContext

FLASH_GEOMETRY = (4096, 2048, 256, 6)
STARTUP_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class MegaMoEExtraInfo(ConnectorExtraInfo):
    attn_ranks_per_dp: int | None = None

    def to_mapping(self):
        if self.attn_ranks_per_dp is None:
            return {}
        return {"attn_ranks_per_dp": self.attn_ranks_per_dp}


def validate_megamoe_config(config: VllmConfig, afd: AFDConfig) -> None:
    hf = config.model_config.hf_config
    if (
        get_ascend_device_type() != AscendDeviceType.A5
        or hf.model_type != "deepseek_v4"
        or (
            hf.hidden_size,
            hf.moe_intermediate_size,
            hf.n_routed_experts,
            hf.num_experts_per_tok,
        )
        != FLASH_GEOMETRY
        or hf.expert_dtype != "fp4"
        or config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError(
            "MegaMoE requires A5 and native FP4 DeepSeek-V4-Flash with BF16 input"
        )
    if config.additional_config.get("enable_fused_mc2", 0):
        raise ValueError(
            "MegaMoE split requires enable_fused_mc2=0 for packed layer weights"
        )
    attn, ffn = afd.num_attention_ranks, afd.num_ffn_ranks
    world = attn + ffn
    experts = hf.n_routed_experts
    # UMDK SplitLayout / arch35 tiling bound the union group and count table.
    if not (
        attn > 0
        and ffn > 0
        and world <= 768
        and ffn <= experts <= 2048
        and experts % ffn == 0
        and experts // ffn <= 1024
        and world * (experts // ffn) <= 16384
    ):
        raise ValueError(
            f"Invalid MegaMoE split geometry: A={attn}, F={ffn}, E={experts}"
        )
    p = config.parallel_config
    if (
        p.data_parallel_size * p.tensor_parallel_size
        != (attn if afd.role == "attention" else ffn)
        or (
            p.pipeline_parallel_size,
            p.prefill_context_parallel_size,
            p.decode_context_parallel_size,
        )
        != (1, 1, 1)
        or (
            afd.role == "ffn"
            and (not p.enable_expert_parallel or p.use_sequence_parallel_moe)
        )
        or p.expert_placement_strategy != "linear"
        or p.enable_eplb
        or p.enable_elastic_ep
        or p.eplb_config.num_redundant_experts
        or (
            afd.role == "attention"
            and config.scheduler_config.max_num_batched_tokens % p.tensor_parallel_size
        )
    ):
        raise ValueError(
            "MegaMoE requires role DP * TP equal to its rank count, PP/CP=1, "
            "TP-aligned Attention token capacity and static FFN EP without SP"
        )
    if (p.enable_dbo or p.use_ubatching) and (
        afd.role != "attention" or p.num_ubatches != 2
    ):
        raise ValueError(
            "MegaMoE DBO requires two Attention ubatches and an unbatched FFN"
        )
    if (
        not config.model_config.enforce_eager
        or not config.use_v2_model_runner
        or config.speculative_config is not None
        or not afd.compute_gate_on_attention
        or not afd.async_dp
    ):
        raise ValueError(
            "MegaMoE requires eager ModelRunnerV2, async DP, Attention routing"
        )


class CAMAsyncMegamoeAFDConnector(AFDConnectorBase):
    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> ConnectorExtraInfo:
        raw = raw or {}
        if set(raw) - {"attn_ranks_per_dp"}:
            raise ValueError("Unknown MegaMoE connector_extra_config field")
        width = raw.get("attn_ranks_per_dp")
        if "attn_ranks_per_dp" in raw and (type(width) is not int or width < 1):
            raise ValueError("attn_ranks_per_dp must be a positive integer")
        return MegaMoEExtraInfo(width)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.attn_size = afd_config.num_attention_ranks
        self.ffn_size = afd_config.num_ffn_ranks
        self.world_rank = (
            role_rank if afd_config.role == "attention" else self.attn_size + role_rank
        )
        self.group = None
        self.buffer = None
        self.ticket = None
        self.sequence = 0
        self.weights = {}

    @property
    def is_initialized(self) -> bool:
        return self.buffer is not None

    def prepare_layers(self, layers) -> None:
        from afd_plugin.model_executor.npu.megamoe_weights import expert_weights

        hf = self.vllm_config.model_config.hf_config
        self.weights = {
            layer.layer_idx: expert_weights(
                layer.mlp, self.role_rank, hf.hidden_size, hf.moe_intermediate_size
            )
            for layer in layers
        }

    def init_afd_connector(self) -> None:
        if self.is_initialized:
            return
        import umdk_cam_megamoe as ops

        self.ops = ops
        stores = []
        self.group = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.attn_size + self.ffn_size,
            rank=self.world_rank,
            group_name="afd_megamoe",
            timeout=STARTUP_TIMEOUT,
            store_callback=stores.append,
        )
        try:
            store = dist.PrefixStore("afd_megamoe", stores[0])
            # FFN TP can differ from Attention TP. Share actual widths and check
            # all explicit declarations before anyone enters buffer collectives.
            attention = self.afd_config.role == "attention"
            width = (
                self.vllm_config.parallel_config.tensor_parallel_size
                if attention
                else 0
            )
            store.set(
                f"tp/{self.world_rank}",
                f"{width} {self.extra_info.attn_ranks_per_dp or 0}",
            )
            widths = [
                tuple(map(int, store.get(f"tp/{i}").split()))
                for i in range(self.attn_size + self.ffn_size)
            ]
            self.attn_tp = widths[0][0]
            if (
                self.attn_size % self.attn_tp
                or any(actual != self.attn_tp for actual, _ in widths[: self.attn_size])
                or any(declared not in (0, self.attn_tp) for _, declared in widths)
            ):
                raise ValueError("Inconsistent Attention TP / attn_ranks_per_dp")
            self.work = ExpertWorkQueue(
                store, self.attn_size // self.attn_tp, self.ffn_size, ops.split_sequence
            )
            self.sequence = 0
            # Only Attention schedulers own the token limit. All window offsets
            # must be derived from the same capacity, including on FFN workers.
            if self.afd_config.role == "attention":
                store.set(
                    f"capacity/{self.role_rank}",
                    str(
                        self.vllm_config.scheduler_config.max_num_batched_tokens
                        // self.attn_tp
                    ),
                )
            capacity = max(
                int(store.get(f"capacity/{dp}")) for dp in range(self.attn_size)
            )
            hf = self.vllm_config.model_config.hf_config
            with DefaultProcessGroupSwitcher(dist.group.WORLD, self.group):
                self.buffer = ops.create_megamoe_split_buffer(
                    self.group,
                    self.attn_size,
                    self.ffn_size,
                    hf.n_routed_experts,
                    capacity,
                    hf.hidden_size,
                    hf.moe_intermediate_size,
                    hf.num_experts_per_tok,
                    shared_experts=0,
                    combine_quant_mode=0,
                    activation_parameters=(hf.swiglu_limit, 1.702, 1.0),
                )
        except Exception:
            dist.destroy_process_group(self.group)
            self.group = None
            raise

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: torch.Tensor,
    ) -> None:
        stage = context.metadata.stage_idx
        if self.ticket is not None or stage not in (0, 1):
            raise RuntimeError(
                "MegaMoE eager supports one outstanding Attention ticket"
            )
        dp = self.role_rank // self.attn_tp
        members = self.dp_members(dp)
        sequence = self.ops.split_sequence(self.sequence + 1)
        ticket = self.ops.megamoe_dispatch_send(
            hidden_states.contiguous(),
            kwargs["topk_ids"].int().contiguous(),
            kwargs["topk_weights"].float().contiguous(),
            self.buffer,
            members,
            sequence,
        )

        # The peer receive hook drains this ticket before the next send.
        self.ticket = (stage, ticket)
        self.sequence = sequence
        if self.role_rank == members[0]:
            self.work.publish(dp, context.metadata.layer_idx, sequence)

    def dp_members(self, dp: int) -> list[int]:
        start = dp * self.attn_tp
        return list(range(start, start + self.attn_tp))

    def recv_ffn_output(
        self, ref_tensor: torch.Tensor, ubatch_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        if self.ticket is None or ubatch_idx != self.ticket[0]:
            raise RuntimeError("MegaMoE combine requires the matching ubatch ticket")
        output = self.ops.megamoe_combine_recv(self.ticket[1], self.buffer)
        self.ticket = None
        return output

    def execute_next_experts(self) -> None:
        item = self.work.poll(self.role_rank)
        if item is None:
            time.sleep(POLL_SECONDS)
            return
        dp, layer, sequence = item
        self.ops.megamoe_fused_moe(
            *self.weights[layer], self.buffer, self.dp_members(dp), sequence
        )
        # Bound device submissions and retain layer tensors until fused work
        # completes. Other Attention DPs remain independently scheduled.
        torch.npu.synchronize()

    def recv_attn_output(self, ubatch_idx=0, **kwargs):
        raise RuntimeError(
            "MegaMoE FFN runs recv, compute and send in execute_next_experts"
        )

    def send_ffn_output(self, ffn_output, context, **kwargs):
        raise RuntimeError("MegaMoE FFN sends results inside fused_moe")

    def close(self) -> None:
        if self.buffer is not None:
            torch.npu.synchronize()
            dist.barrier(group=self.group)
            self.buffer.destroy()
            self.buffer = None
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None
        self.ticket = None
        self.weights.clear()
