# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the narrow DSV4 Async CAM feature contract."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_dsv4_async_features,
    _fail_if_unsupported_dsv4_connector,
)
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo


@pytest.fixture(autouse=True)
def a3_device(monkeypatch):
    from vllm_ascend import utils

    monkeypatch.setattr(
        utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A3
    )


def _afd_config(
    *,
    compute_gate_on_attention: bool,
    connector: str = "CAMAsyncAFDConnector",
) -> SimpleNamespace:
    return SimpleNamespace(
        compute_gate_on_attention=compute_gate_on_attention,
        connector=connector,
    )


def test_dsv4_rejects_connectors_without_a_token_id_channel() -> None:
    with pytest.raises(RuntimeError, match="supports only CAMAsyncAFDConnector"):
        _fail_if_unsupported_dsv4_connector(
            _afd_config(
                compute_gate_on_attention=False,
                connector="P2pNcclAFDConnector",
            ),
        )


def test_dsv4_accepts_the_camp2p_connector() -> None:
    """CAMP2P carries the Hash ids over the A2E ids channel with the gate on FFN."""

    _fail_if_unsupported_dsv4_connector(
        _afd_config(
            compute_gate_on_attention=False,
            connector="CAMP2pAFDConnector",
        ),
    )


def test_dsv4_async_requires_attention_side_gate() -> None:
    with pytest.raises(RuntimeError, match="compute_gate_on_attention"):
        _fail_if_unsupported_dsv4_async_features(
            _afd_config(compute_gate_on_attention=False),
            AFDAsyncExtraInfo(dynamic_quant=1),
            SimpleNamespace(),
        )


@pytest.mark.parametrize(
    ("extra_info", "message"),
    [
        (
            AFDAsyncExtraInfo(dynamic_quant=0),
            "dynamicQuant=1",
        ),
    ],
)
def test_dsv4_async_rejects_deferred_features(
    extra_info: AFDAsyncExtraInfo,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _fail_if_unsupported_dsv4_async_features(
            _afd_config(compute_gate_on_attention=True),
            extra_info,
            SimpleNamespace(),
        )


def test_dsv4_async_accepts_target_baseline() -> None:
    _fail_if_unsupported_dsv4_async_features(
        _afd_config(compute_gate_on_attention=True),
        AFDAsyncExtraInfo(
            dynamic_quant=1,
            attn_ranks_per_dp=2,
        ),
        SimpleNamespace(),
    )


@pytest.mark.parametrize("dynamic_quant,accepted", [(0, True), (1, False)])
def test_a5_mxfp4_requires_bf16_cam_transport(monkeypatch, dynamic_quant, accepted):
    from vllm_ascend import utils

    from tests.unit.fakes.vllm_config import npu_vllm_config

    monkeypatch.setattr(
        utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A5
    )
    config = npu_vllm_config(enable_eplb=False)
    config.model_config.hf_config = SimpleNamespace(expert_dtype="fp4")
    monkeypatch.delenv("AFD_ASYNC_CAM_LAYERED_GMM", raising=False)
    args = (
        _afd_config(compute_gate_on_attention=True),
        AFDAsyncExtraInfo(dynamic_quant=dynamic_quant),
        config,
    )
    if accepted:
        _fail_if_unsupported_dsv4_async_features(*args)
    else:
        with pytest.raises(RuntimeError, match="dynamicQuant=0"):
            _fail_if_unsupported_dsv4_async_features(*args)
