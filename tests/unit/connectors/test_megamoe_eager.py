# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU regression tests for the eager driver and byte-preserving FP4 packing."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.model_executor.npu.megamoe_weights import _nz_bytes  # noqa: E402
from tests.e2e.models.deepseek_v4_flash.config import (  # noqa: E402
    DSV4_MEGAMOE_SCENARIO,
)


def test_independent_dp_and_slow_readers():
    # Load only the protocol module. The connector package imports vLLM.
    path = (
        Path(__file__).resolve().parents[3]
        / "afd_plugin/connectors/npu/megamoe_work.py"
    )
    spec = importlib.util.spec_from_file_location("work", path)
    work = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(work)
    store = torch.distributed.HashStore()

    def queue():
        return work.ExpertWorkQueue(store, 2, 2, lambda n: (n - 1) % 32 + 1)

    writers = [queue(), queue()]
    readers = [queue(), queue()]
    for iteration in range(1, 70):
        active = [0, 1] if iteration % 3 == 0 else [0]
        expected = []
        for dp in active:
            sequence = writers[dp].next_sequence(dp)
            layer = iteration % 43
            writers[dp].publish(dp, layer, sequence)
            expected.append((dp, layer, sequence))
        for ffn in (1, 0):
            items = [readers[ffn].poll(ffn) for _ in active]
            assert sorted(items) == expected
            assert readers[ffn].poll(ffn) is None
        assert store.num_keys() == 0

    store.set("work/0/0", "0 1")
    with pytest.raises(RuntimeError, match="Out-of-order"):
        readers[0].poll(0)


def test_nz_packing_preserves_checkpoint_bytes():
    data = torch.randint(0, 256, (2, 32, 48), dtype=torch.uint8)
    packed = _nz_bytes(data)
    assert packed.is_contiguous()
    for e in range(2):
        for k in range(3):
            for n in range(2):
                tile = data[e, n * 16 : (n + 1) * 16, k * 16 : (k + 1) * 16]
                assert torch.equal(packed[e, k, n], tile)


def test_flash_scenario_uses_native_quantization(monkeypatch, tmp_path):
    from tests.e2e import runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model",
            "/models/DeepSeek-V4-Flash",
            "--device-backend",
            "npu",
            "--scenario",
            DSV4_MEGAMOE_SCENARIO,
            "--attention-devices",
            "0,1",
            "--ffn-devices",
            "2,3",
            "--completion-output-path",
            str(tmp_path / "out.json"),
        ],
    )
    args = runner.parse_args()
    runner.configure_scenario(args)
    runner.validate_topology(args, ["0", "1"], ["2", "3"])
    for role in ("attention", "ffn"):
        env = runner.build_env("0,1" if role == "attention" else "2,3", args, role=role)
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        cmd = runner.build_vllm_command(args, role=role)
        config = json.loads(cmd[cmd.index("--additional-config") + 1])["afd"]
        assert config["connector"] == "MegaMoESplitAFDConnector"
        assert config["role"] == role and config["compute_gate_on_attention"]
        assert cmd[cmd.index("--data-parallel-size") + 1] == "2"
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
        assert "--enforce-eager" in cmd and "--enable-expert-parallel" in cmd
        assert "--quantization" not in cmd and "--enable-dbo" not in cmd
