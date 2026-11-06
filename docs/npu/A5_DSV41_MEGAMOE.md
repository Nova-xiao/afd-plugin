# A5 DSV4.1 Flash + CAMAsyncMegamoeAFDConnector

This development targets [RFC #537](https://github.com/vllm-project/afd-plugin/issues/537).
It adds the V4.1 model adapter and updates the connector to the multi-layer UMDK
interface. **A5 compilation, numerical accuracy and end-to-end serving are not
validated locally.** The local development host has no Ascend device.

## Fixed sources

| Component | Revision |
| --- | --- |
| AFD base | `update/v0.32.0`, `2917805` (existing development branch retained) |
| AFD development | `Nova-xiao/afd-plugin`, `feature/a5-dsv4-cam-async-0.32` |
| vLLM | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |
| CANN | `9.2.1` |
| UMDK | `lqy16/umdk`, `codex/megamoe-split-sequence`, `74fa9f9ad59ca57830028a459e70661c4574a18e` |
| Checkpoint | Original DeepSeek-V4.1 Flash FP8 checkpoint with FP4 routed experts |

Rebuild/install the Ascend950 OPP **and** `umdk_cam_megamoe` extension from the
same UMDK revision. An older wheel does not accept `layer_index`; an older OPP
also has a different window layout. Do not mix the two generations. Follow the
build instructions in UMDK
`src/cam/comm_operator/ascend_kernels/megamoe_support/README.md`.

## Execution and weight ownership

1. Attention retains native V4.1 sparse attention, KV/index sharing, delayed HC
   pre-mix, Engram and bounded replay. Its MoE proxy computes routing and shared
   experts, then sends the current layer's actual rows and `layer_index`.
2. FFN loads only routed experts and the native MoE router parameters needed by
   the loader. It allocates no attention, Engram, embedding or output head.
   After native quantization finalization, AFD converts the packed weights once
   and assembles four layer-ordered lists: W1, W2, S1, S2.
3. `megamoe_fused_moe` receives the **whole layer bank** on each invocation.
   The device validates the task's sequence/layer tag and selects weights from
   the layer ID published through MTE. The Store notification only schedules
   `(DP, layer, sequence)` work on the FFN host; it does not select a single-layer
   tensor list anymore.
4. Attention combines the routed outputs and adds its shared-expert output.
   Routing weights include the routed scaling factor, so it is applied once.

Each Attention DP retains one outstanding ticket. Ring sequences remain
`1..32 → 1`, independently per DP, with the UMDK device protocol protecting slot
reuse. Layer IDs are independent of sequence values. Bounded replay can change
row counts without changing the selected layer; the proxy takes its token count
from the actual activation tensor, not the scheduler's original batch size.

Implementation paths:

- `afd_plugin/model_executor/models/npu/deepseek_v41.py`: native V4.1 adapter,
  role-specific construction/loading and the existing FFN runner's layer view.
- `afd_plugin/connectors/npu/async_megamoe.py`: configuration validation,
  multi-layer bank, dispatch layer argument and fused FFN submission.
- `afd_plugin/model_executor/npu/megamoe_weights.py`: native FP4 NZ → packed-byte
  C0=16 storage (UMDK describes it as FP4 `FRACTAL_NZ_C0_32`), output-major E8M0
  group-32 scales. No requantization of checkpoint weights.

The target uses BF16 input/output, dispatch MXFP8 E4M3 activations and MXFP4 E2M1
routed weights. It does not accept A3 integer INT4/INT8 W4A8 checkpoints.
The connector checks UMDK's geometry limits for V4.1 instead of imposing V4's
4096/2048/256 dimensions. These checks are API eligibility, not a memory-fit or
model-accuracy claim for every configuration within those limits.

## Configuration and launch

The provided scenario is single-node **2A2F**, Attention DP2/TP1 and FFN EP2/TP1,
static contiguous expert placement, eager ModelRunnerV2, TP/PP/CP=1. It excludes
MTP/DSpark, DBO, graphs, dynamic EPLB and image requests. This V4.1 adapter currently
requires `CAMAsyncMegamoeAFDConnector`; it does not claim V4.1 support through
`CAMAsyncAFDConnector` or the synchronous connector.

Use the usual HCCL network environment for the host, activate both packages, then
run from the AFD repository:

```bash
python -m tests.e2e.runner \
  --model /models/DeepSeek-V4.1-Flash \
  --device-backend npu \
  --attention-devices 0,1 \
  --ffn-devices 2,3 \
  --scenario afd-dsv41-flash-megamoe-eager-2a2f \
  --served-model-name-prefix dsv41-flash \
  --afd-host 127.0.0.1 \
  --afd-port 6456 \
  --api-port-base 19380 \
  --startup-timeout 1800 \
  --completion-output-path /tmp/dsv41-megamoe-completions.json
```

The scenario sets `VLLM_USE_V2_MODEL_RUNNER=1`, eager mode, `--dtype bfloat16`, `enable_fused_mc2=0`, `enable_flashcomm1=false`,
`afd.async=true`, `compute_gate_on_attention=true`, and image limit 0. It selects
`deepseek_v41` tokenization and the native DeepSeek V4 reasoning/tool parsers.
It uses a 4096-token context, 1024-token batch budget and chunked prefill.
Set `--dtype bfloat16` in custom launch commands; the connector rejects FP16.

Engram is enabled on Attention with CPU offload and `embedding_across_dp=false`:
independent Attention DPs must not enter per-step Engram DP collectives. Native
shared-memory behavior and capacity requirements still apply. FFN does not load
Engram. Bounded replay retains the pinned native configuration/implementation;
use the native cache configuration when testing it, rather than adding an AFD
replay path. Model weights and Engram require sufficient HBM and host memory;
the scenario's resource fit is still to be checked on the target machine.

For manual launches use the same A/F counts and the fields above, omit
`connector_extra_config`, and set `--limit-mm-per-prompt '{"image":0}'` on both
roles. This path uses UMDK's three operators, not AFD's four asynchronous CAM
operators. The earlier `PIPE_V` compilation report in that four-operator package
is a separate unresolved issue and is not evidence against or validation of
this UMDK path.

## Verification to run on A5

1. Run `pytest -q tests/npu/test_megamoe_weights.py` to check FP4 physical storage
   round-trip without changing packed nibbles.
2. Reuse UMDK's `test/cam/megamoe/compare_megamoe_multilayer_dp3_cp2_ep2.py` for its
   multi-layer selection and original-MegaMoE numerical comparison, using that
   script's `--help` and documented device requirements. **It generates MXFP8
   weights and uses a different DP3/CP2/EP2 topology: its PASS does not validate
   this FP4 checkpoint or the 2A2F AFD path.**
3. Run the E2E scenario above. It submits concurrent text completions and records
   outputs. Exercise long/chunked prompts, multiple generated tokens, native
   bounded replay, empty/profile batches and requests spanning several sequence
   wraps. Startup/shutdown should be repeated after requests drain.
4. Compare the same checkpoint and prompts with the colocated #537 baseline.
   Completion success alone is not numerical accuracy acceptance. Record expert
   output error and generated-result/model accuracy before claiming support.

Local checks cover protocol arguments, weight ownership, allocation boundaries,
ring scheduling and byte tiling. They do not execute communication, Engram,
attention or fused expert kernels. No performance results are claimed.
