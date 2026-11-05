# A5 DeepSeek-V4-Flash asynchronous AFD development

## Source baseline

This branch starts at AFD `update/v0.32.0`, commit `2917805`, and targets the
[#537](https://github.com/vllm-project/afd-plugin/issues/537) software environment:

| Component | Target |
| --- | --- |
| vLLM | `5df63a7203cdb9328d824a27b682e77c7b9928be` (0.32.0) |
| vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |
| CANN | 9.2.1 |
| Model | DeepSeek-V4-Flash, original FP8 checkpoint with MXFP4 routed experts |

The model target is V4 Flash, not V4.1. The existing A3 validation record uses an
older Ascend commit and does not establish A5 support. This branch removes the
DSV4 wrapper's calls to the DSA-CP interfaces deleted in the target Ascend source.
The native loader, attention, HC, hash routing and shared-expert arithmetic remain
responsible for the model semantics.

## Two execution paths

| | `CAMAsyncAFDConnector` | `CAMAsyncMegamoeAFDConnector` |
| --- | --- | --- |
| Attention | Routing and shared experts; asynchronous dispatch/combine | Same model-side ownership |
| Transfer input | BF16 (`dynamicQuant=0`) | Dispatch quantizes BF16 to MXFP8 inside the operator |
| FFN | CAM receive, native MXFP4 expert MLP, CAM send | One fused receive + expert MLP + send operator |
| Weights | Native transposed FP4 NZ tensors and E8M0 scales | Output-major FP4 C0=32 storage and E8M0 scales, prepared once after loading |
| Scheduling | Existing CAM device work selection | Bounded per-DP/per-FFN Store descriptors select layer and generation; tensors travel through MTE |

Both use `afd.async=true` and `afd.compute_gate_on_attention=true`. Attention
DPs can progress independently. Both require eager ModelRunnerV2 for this A5
path. Use `enable_fused_mc2=0` so native loading retains the tensors consumed by
these backends. The A3 layered INT4/INT8 executor is not the A5 MXFP4 executor;
leave `AFD_ASYNC_CAM_LAYERED_GMM` disabled.

For ordinary CAM, the BF16 receive scale output is a placeholder. The connector
now passes `None` to the FFN MLP, causing the native MX method to perform group-32
MXFP8 activation quantization. Passing the placeholder would incorrectly suppress
that quantization. The FFN MLP consumes the native weights directly and does not
perform another EP dispatch. Top-k scaling remains applied once by combine.

For MegaMoE, the connector owns the shared communication context and dispatch
ticket. Each DP advances the UMDK ring sequence from 1 through 32 and back to 1.
Every FFN rank consumes the descriptor, including ranks with no selected expert.
The kernel's PUBLISHED/TAKEN/READY/CONSUMED protocol protects input/output reuse.
The FFN loop drains each fused submission before selecting another descriptor,
which bounds queued work without synchronizing independent Attention DPs.
The implementation currently accepts 2A2F, DP2/TP1, contiguous static EP2 in one
MTE supernode; no DBO, graphs, MTP or EPLB. Close the roles together after draining
requests: communication-window destruction requires the union-group barrier.

## Operator dependencies and current limits

**Plugin development is not an A5 runtime validation result.** Remote deployment,
accuracy, long-run communication and performance checks have not been performed.

- Ordinary CAM needs an **A5 build of its asynchronous communication operators**
  implementing the existing dispatch-send/recv and combine-send/recv ABI with
  BF16 transport. The local `cam_async` source inspected during development
  registers those operators only for `ascend910_93`; that A3 bundle does not
  establish an A5 implementation. Installing vLLM-Ascend and CANN alone does not
  supply this dependency. This remains an external requirement for running the
  ordinary CAM path; its kernels were not ported as part of this Plugin change.
- MegaMoE needs the Ascend950 OPP and `umdk_cam_megamoe` wheel from UMDK
  [`793d2f02`](https://gitcode.com/lqy16/umdk/commit/793d2f02c2564eda1e0f8971040f8626d89295d0),
  including the ring-sequence changes. This is separate from the ordinary CAM
  bundle. Its three Python entry points are `megamoe_dispatch_send`,
  `megamoe_fused_moe`, and `megamoe_combine_recv`.
- MegaMoE's one-time FP4 storage conversion preserves packed nibbles; its
  format-cast and CANN interpretation still need device verification. Local CPU
  checks establish byte reordering and queue behavior, not numerical support.

## Configuration contract

Supply role, rendezvous and rank counts as usual. The relevant additional-config
fields for ordinary CAM are:

```json
{
  "enable_fused_mc2": 0,
  "afd": {
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "compute_gate_on_attention": true,
    "connector_extra_config": {"dynamicQuant": 0, "attn_ranks_per_dp": 1}
  }
}
```

For MegaMoE, select `CAMAsyncMegamoeAFDConnector`, keep both booleans true,
set both role rank counts to 2, and omit `connector_extra_config`. Both roles
use DP2/TP1, static EP, BF16 model activations and eager execution. These are
configuration fragments, not complete deployment commands or verified recipes.
