# DeepSeek-V4-Flash on A5 with split MegaMoE

This eager V1 backend targets one host with two Attention dies (DP2/TP1)
and two FFN dies (static contiguous EP2/TP1). Use the original DeepSeek-V4-Flash
checkpoint with FP4 E2M1 routed weights. Both roles use BF16 activations at
the model boundary. Gate, hash routing and the shared expert run on Attention.
The FFN receives one work descriptor per participating Attention DP and layer.
It selects the local expert weights and launches the fused UMDK kernel.
The descriptor travels over the rendezvous Store. Payloads use the registered
same-supernode MTE window. This first version prioritizes correctness in eager
mode; the Store incurs host overhead per layer.

## Running the A5 acceptance case

Install this plugin and the CANN 9.2 Ascend950 operator package and Python wheel
from [UMDK commit 793d2f02](https://gitcode.com/lqy16/umdk/commit/793d2f02c2564eda1e0f8971040f8626d89295d0)
in both worker environments. Activate the vendor OPP environment. Check that
`python -c "import umdk_cam_megamoe as m; assert m.split_sequence(33) == 1"`
succeeds. The A3 CAM bundle packaged with the plugin is for a different backend.

From the plugin checkout, with four visible A5 dies and a local checkpoint:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
python -m tests.e2e.runner \
  --model /path/to/DeepSeek-V4-Flash --device-backend npu \
  --scenario afd-dsv4-flash-megamoe-eager-2a2f \
  --attention-devices 0,1 --ffn-devices 2,3 \
  --afd-host 127.0.0.1 --afd-port 6456 --startup-timeout 1800 \
  --completion-output-path /tmp/megamoe-flash-completions.json
```

The scenario starts both services with `--enforce-eager`,
`MegaMoESplitAFDConnector`, Attention routing and prefix caching disabled.
It checks concurrent answers and writes completions to the requested path.
For numerical qualification, compare fixed-token outputs with the native A5
checkpoint baseline, including a hash layer and a normal routed layer.
For a communication soak, exercise both DPs with unequal traffic and enough
tokens to cross multiple 32-generation cycles. Shut down the two roles together
after draining inference. This case has to be run on A5 to establish numerical
and device-runtime support.
