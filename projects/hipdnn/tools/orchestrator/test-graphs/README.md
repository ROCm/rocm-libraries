<!-- Copyright © Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier:  MIT -->

# Test graphs

Standalone hipDNN graph JSON files, one graph per file, for feeding the orchestrator:

```bash
python orchestrate.py run configs/flows/rtc-kernel-review.yaml \
    --input graph=test-graphs/matmul_fp32_batched.json
```

Each file is a **fully expanded** graph: no `${case.*}` placeholders, concrete dims,
strides and data types. They were materialised from the template-sweep bundles in
`dnn-providers/integration-tests/integration-test-bundles/quick/` by substituting one
sweep case into its `graph.template.json`, following the same rules the C++ harness
uses (`src/harness/bundle/IntegrationTestBundle.hpp`): a `${case.<field>}` inside a
tensor object resolves against that tensor's entry in `values.tensors[]` (matched by
`uid`), otherwise against the case's top-level `values`.

## The graphs

### `conv_fwd_pointwise_fp32_nchw.json` — convolution forward + pointwise

`ConvolutionFwdAttributes` → `PointwiseAttributes` (`relu_fwd`), fp32, NCHW.
The conv output (uid 0) is **virtual** — it is consumed by the ReLU and never written
to memory, so a correct kernel fuses the two rather than materialising an intermediate.

| uid | tensor | dims | strides |
|---|---|---|---|
| 3 | `x` | [1, 16, 16, 8] | [2048, 128, 8, 1] |
| 2 | `w` | [1, 16, 3, 3] | [144, 9, 3, 1] |
| 0 | `ConvolutionFprop_0::Y` (virtual) | [1, 1, 16, 8] | [128, 128, 8, 1] |
| 1 | `Pointwise_1::OUT_0` | [1, 1, 16, 8] | [128, 128, 8, 1] |

`conv_mode: CROSS_CORRELATION`, stride 1×1, dilation 1×1, pre/post padding 1×1 — so the
16×8 spatial extent is preserved and every output pixel on the border reads padded
input. 16 input channels, 1 output channel, 3×3 filter.

Source: `quick/ConvolutionFwdPointwise/Default`, case
`1_16_3_3_fp32_nchw_dil1x1_postpad1x1_prepad1x1_f61755`
(captured from `Smoke/IntegrationGpuConvFwdBiasActiv2dFp32.Correctness/12`).

### `batchnorm_inference_fp32_nchw.json` — batchnorm inference

`BatchnormInferenceAttributes`, fp32, NCHW. Pure inference: mean/inv_variance arrive as
inputs, nothing is updated.

| uid | tensor | dims | strides |
|---|---|---|---|
| 1 | `X` | [1, 3, 14, 14] | [588, 196, 14, 1] |
| 5 | `scale` | [1, 3, 1, 1] | [3, 1, 1, 1] |
| 2 | `bias` | [1, 3, 1, 1] | [3, 1, 1, 1] |
| 4 | `mean` | [1, 3, 1, 1] | [3, 1, 1, 1] |
| 3 | `inv_variance` | [1, 3, 1, 1] | [3, 1, 1, 1] |
| 0 | `BatchnormInference_0::Y` | [1, 3, 14, 14] | [588, 196, 14, 1] |

The per-channel parameters broadcast over N, H and W: 3 channels against a 14×14 plane,
which is not a multiple of any natural block size.

Source: `quick/BatchnormInference/Default`, case `1_3_14_14_fp32_nchw`
(captured from `Smoke/IntegrationGpuBatchnormForwardInference2dFp32.Correctness/0`).

### `batchnorm_inference_fp32_nchw_large.json` — batchnorm inference, benchmark-sized

The same graph, scaled until the kernel is what gets measured rather than the launch.

| uid | tensor | dims | strides |
|---|---|---|---|
| 1 | `X` | [32, 64, 112, 112] | [802816, 12544, 112, 1] |
| 5 | `scale` | [1, 64, 1, 1] | [64, 1, 1, 1] |
| 2 | `bias` | [1, 64, 1, 1] | [64, 1, 1, 1] |
| 4 | `mean` | [1, 64, 1, 1] | [64, 1, 1, 1] |
| 3 | `inv_variance` | [1, 64, 1, 1] | [64, 1, 1, 1] |
| 0 | `BatchnormInference_0::Y` | [32, 64, 112, 112] | [802816, 12544, 112, 1] |

103 MB in, 103 MB out. `HIP_MLOPS_ENGINE` reads ~1.08 ms here.

Shape family source: `quick/BatchnormInference/Default`, case
`1_16_112_112_fp32_nchw`, with N and C raised.

### `batchnorm_inference_fp32_nchw_xl.json` — batchnorm inference, the benchmark graph

Four times the `_large` working set. **This is the one to hand the
`test-engine-kernel` flow**; `_large` is kept because it is a real point on the
sizing ladder below, not because there is a reason to benchmark on it.

| uid | tensor | dims | strides |
|---|---|---|---|
| 1 | `X` | [64, 128, 112, 112] | [1605632, 12544, 112, 1] |
| 5 | `scale` | [1, 128, 1, 1] | [128, 1, 1, 1] |
| 2 | `bias` | [1, 128, 1, 1] | [128, 1, 1, 1] |
| 4 | `mean` | [1, 128, 1, 1] | [128, 1, 1, 1] |
| 3 | `inv_variance` | [1, 128, 1, 1] | [128, 1, 1, 1] |
| 0 | `BatchnormInference_0::Y` | [64, 128, 112, 112] | [1605632, 12544, 112, 1] |

411 MB in, 411 MB out. `HIP_MLOPS_ENGINE` reads ~4.35 ms, a stub `TEST_ENGINE` whose
kernels do nothing reads ~0.11 ms.

#### The sizing ladder, measured on gfx1151/Windows

`hipdnn_graph_bench` times submit-plus-drain, and that has a floor of roughly 0.03 ms
no matter what the kernel does. A graph under the floor measures the floor:

| shape | per-tensor | `HIP_MLOPS_ENGINE` median | verdict |
|---|---|---|---|
| 1×3×14×14 | 9 KB | ~0.03 ms | at the floor — a stub engine measures the same |
| 1×16×112×112 | 802 KB | ~0.03 ms | still at the floor |
| 8×64×112×112 | 26 MB | 0.32 ms | clear of it |
| 32×64×112×112 | 103 MB | 1.08 ms | `_large` |
| 64×128×112×112 | 411 MB | 4.35 ms | `_xl` |

**Warm-up matters at this size.** The working set is larger than anything that stays
resident, and on a shared-memory APU the first touches pay page migration. Three
warm-up iterations on `_xl` left a 40% standard deviation; ten cut it to 1.5%, and
three consecutive 50-iteration runs then landed at 4.355, 4.346 and 4.410 ms — a 1.5%
spread, which is what makes a 20%-faster gate meaningful rather than a coin toss.

### `matmul_fp32_batched.json` — batched matmul

`MatmulAttributes`, fp32, batch 3, M=16, K=32, N=128.

| uid | tensor | dims | strides |
|---|---|---|---|
| 1 | `a` | [3, 16, 32] | [512, 32, 1] |
| 2 | `b` | [3, 32, 128] | [4096, 1, 32] |
| 0 | `Matmul_0::C` | [3, 16, 128] | [2048, 128, 1] |

Note `b`: strides `[4096, 1, 32]` means its last two dimensions are **transposed** in
memory (column-major within each batch). A kernel that assumes row-major contiguity
computes the wrong answer while reading only valid addresses — which is exactly the
kind of defect the review step exists to catch.

Source: `quick/Matmul/Default`, case `3_16_128_fp32_ncl_70b725`
(captured from `Smoke/IntegrationGpuMatmulFp32.Correctness/11`).

## Choosing more cases

Cases were checked for internal consistency before selection, not taken on trust. In
`quick/ConvolutionFwdPointwise/Default`, 12 of the 36 two-dimensional cases declare an
`x` channel count that disagrees with `w` (e.g. case
`2_8_3_3_fp32_nchw_dil1x1_postpad1x1_prepad1x1`: `x` [1, 16, 16, 16] against `w`
[2, 8, 3, 3]). Those expand into a graph no correct kernel can satisfy. If you add a
graph here, verify the shape algebra — for convolution,
`out = (in + pre_pad + post_pad - (dilation * (filter - 1) + 1)) / stride + 1` — before
using it as a target.
