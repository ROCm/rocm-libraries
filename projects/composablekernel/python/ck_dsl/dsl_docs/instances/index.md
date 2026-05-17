# Instances Index

Quick reference for the shipped instance builders. Each row points to the per-family doc.

## GEMM Family

| File | Spec | Doc |
|-----------------------------------|-------------------------------------------------------------------|------------------------------|
| `gemm_universal.py` | `TileSpec`, `TraitSpec`, `DataSpec`, `UniversalGemmSpec` | `instances/gemm.md` |
| `batched_gemm.py` | `BatchedGemmSpec` | `instances/gemm.md` |
| `grouped_gemm.py` | `GroupedGemmProblem`, `GroupedGemmSpec`, `GroupedGemmLauncher` | `instances/gemm.md` |

ABI: `(A, B, C, M, N, K)` for plain GEMM, with `stride_a/b/c` added for batched.

Atom set: `16x16x16`, `16x16x32`, `32x32x8`, `32x32x16` f16.

Pipelines: `mem`, `compv3`, `compv4`. Epilogues: `default`, `cshuffle`. Layout: `RCR`.

## Convolution Family

| File | Spec | Doc |
|-----------------------------------|-------------------------------------------------------------------|------------------------------|
| `conv_implicit_gemm.py` | `ConvProblem`, `ImplicitGemmConvSpec` | `instances/convolution.md` |
| `conv_direct_grouped.py` | `DirectConvProblem`, `DirectConv16cSpec`, `DirectConv4cSpec` | `instances/convolution.md` |
| `img2col.py` | `Img2ColSpec` | `instances/convolution.md` |
| `pooling.py` | `PoolingProblem`, `Pooling2DSpec`, `PoolOp` | `instances/convolution.md` |

ABI: `(A, B, D, A_bytes, B_bytes, D_bytes)` for implicit-GEMM / direct grouped conv. Img2col writes `Y`. Pooling reads `X` and writes `Y`.

Layouts: NHWC input, KRSC weight, NHWK output for conv. Grouping via `cpg`/`kpg`.

Bake-off results (per `RUNBOOK_COMPLIANCE.md`):

```text
Implicit-GEMM conv (N=8 H=W=56 C=K=64 R=S=3):
 111 TFLOPS -> 280 TFLOPS by applying 5 runbook levers in series.

Direct grouped 16c (N=32 H=W=200 R=S=3 pad=1):
 ~92 TFLOPS -> ~214 TFLOPS.

Direct grouped 4c (same shape, cpg=kpg=4):
 ~44 TFLOPS -> ~48 TFLOPS.
```

## Attention Family

| File | Spec | Doc |
|-----------------------------------|-------------------------------------------------------------------|------------------------------|
| `attention_unified.py` | `UnifiedAttentionProblem`, `UnifiedAttention2DSpec`, `UnifiedAttention3DSpec`, `UnifiedAttentionReduceSpec` (scalar kernels) | `instances/attention.md` |
| `attention_tiled_2d.py` | `UnifiedAttention2DTiledSpec` | `instances/attention.md` |
| `attention_tiled_3d.py` | `UnifiedAttention3DTiledSpec`, `UnifiedAttentionReduceTiledSpec` | `instances/attention.md` |
| `_fmha_common.py` | `FmhaCommonSpec`, `FmhaShape`, `FmhaMaskMode` (scaffold) | `instances/attention.md` |
| `fmha_varlen.py` | `FmhaFwdVarlenSpec` (CK Tile 01 varlen) | `instances/attention.md` |
| `fmha_appendkv.py` | `FmhaAppendKvSpec` (CK Tile 01 appendkv; optional rotary) | `instances/attention.md` |
| `fmha_paged_prefill.py` | `FmhaFwdPagedPrefillSpec` (CK Tile 01 pagedkv_prefill) | `instances/attention.md` |
| `fmha_splitkv_decode.py` | `FmhaFwdSplitKvDecodeSpec` (CK Tile 01 splitkv; 2 launches) | `instances/attention.md` |
| `fmha_head_grouping.py` | `FmhaFwdHeadGroupingSpec` (CK Tile 01 head_grouping; GQA/MQA) | `instances/attention.md` |
| `fmha_bwd.py` | `FmhaBwdSpec` (CK Tile 01 bwd; atomic fp32 dQ/dK/dV) | `instances/attention.md` |
| `fmha_fwd_fp8.py` | `FmhaFwdFp8Spec` (CK Tile 01 fp8; per-tensor scales) | `instances/attention.md` |
| `sage_attention.py` | `SageAttentionSpec`, `SageQuantMode` (CK Tile 49; 4 variants) | `instances/attention.md` |
| `sparse_attention.py` | `JengaSparseSpec`, `VsaSparseSpec` (CK Tile 50; jenga + VSA) | `instances/attention.md` |

Runtime entry point: `run_unified_attention_torch(...)`.

Path selection: `select_2d_config` / `select_3d_config` / `use_2d_kernel`. The runtime picks 3D split-KV for long-context decode and 2D for chunked-prefill / sliding-window / qq-bias rows.

Coverage: fp16 / bf16, head_size in `{128, 256}`, block_size in `{16, 64}`, causal / sliding window / softcap / sinks / ALiBi / QQ-bias.

FP8 K/V cache + output scale/clamp is the next coverage step (see attention parity README).

## Small Ops

| File | Spec | Doc |
|-----------------------------------|-------------------------------------------------------------------|------------------------------|
| `elementwise.py` | `ElementwiseSpec` | `instances/small_ops.md` |
| `reduce.py` | `Reduce2DSpec`, `ReduceOp` | `instances/small_ops.md` |
| `layernorm2d.py` | `LayerNorm2DSpec` | `instances/small_ops.md` |
| `rmsnorm2d.py` | `RMSNorm2DSpec` | `instances/small_ops.md` |
| `transpose.py` | `Transpose2DSpec` | `instances/small_ops.md` |

## MoE Family

| File | Spec / launcher | Doc |
|-----------------------------------|--------------------------------------------------------------------|------------------------------|
| `topk_softmax.py` | `TopkSoftmaxSpec` (CK Tile 09) | `instances/small_ops.md` |
| `moe_smoothquant.py` | `MoeSmoothQuantSpec` (CK Tile 14) | `instances/small_ops.md` |
| `moe_sorting.py` | `MoeSortingSpec` (CK Tile 13; three-kernel pipeline) | `instances/small_ops.md` |
| `fused_moe.py` | `FusedMoeSpec`, `FusedMoeLauncher` (CK Tile 15) | `instances/small_ops.md` |

The fused-MoE forward (`fused_moe.py`) is a *composition*: it ships
the three MoE-specific kernels (gather, SwiGLU activation fusion,
topk-weighted reduce) and the launcher class describes how to stitch
them together with the upstream builders (topk_softmax
-> moe_sorting -> moe_smoothquant -> block_scale_gemm x2 -> silu_mul
-> moe_smoothquant -> block_scale_gemm -> reduce).

ABI shapes (from `helpers/spec.py::SignatureBuilder`):

```text
Elementwise unary: (A, C, N)
Elementwise binary: (A, B, C, N)
Reduce2D: (X, Y, M, N)
LayerNorm2D: (X, Gamma, Beta, Y[, mean, invstd], M, N, eps)
RMSNorm2D: (X, Gamma, Y[, inv_rms], M, N, eps)
Transpose2D: (X, Y, M, N)
```

Allowed dtypes / block sizes / vecs from `IOSpecRule`:

```text
allowed_dtypes = ("f16", "fp16", "bf16")
allowed_block_sizes = (64, 128, 256, 512, 1024)
allowed_vecs = (2, 4, 8)
```

## Cross-Family Capability Matrix

From `helpers/README.md`:

| Instance | tensor_view | transform DAG | sweep | io | reduction | spec | distribution | GEMM-shape helpers |
|-------------------------|--------------|------------------------|-------|-----|-----------|------|--------------|--------------------|
| elementwise | yes | - | - | yes | - | yes | - | - |
| layernorm2d | yes | - | yes | yes | yes | yes | - | - |
| rmsnorm2d | yes | - | yes | yes | yes | yes | - | - |
| reduce | yes | - | yes | yes | yes | yes | - | - |
| transpose | yes | - | - | yes | - | yes | - | - |
| gemm_universal | yes | partial (view strides) | - | - | - | yes | - | `MfmaAtom`, `WarpGrid`, `LdsLayout`, `CoalescedTileLoader`, `AsyncTileLoader`, `CShuffleEpilogue`, `DirectEpilogue`, `SchedulePolicy`, `SoftwarePipeline` |
| batched_gemm | - | - | - | - | - | yes | - | (wraps gemm_universal) |
| grouped_gemm | - | planned | - | - | - | yes | - | (wraps gemm_universal) |
| conv_implicit_gemm | yes (buffer) | full (unmerge+embed+pad) | - | - | - | yes | - | `AsyncTileLoader`, `CoalescedTileLoader`, `CShuffleEpilogue`, `MfmaAtom`, `WarpGrid`, `LdsLayout`, `SchedulePolicy` |
| conv_direct_grouped | - | input/output/weight + H/W pad | - | - | - | yes | - | `MfmaAtom` (4x4x4 / 16x16x{16,32}) |
| img2col | - | reuses A descriptor | - | yes | - | yes | - | - |
| pooling | - | full (input + pad) | - | yes | - | yes | - | - |
| attention_unified | - | Q + output + paged-KV | - | - | - | yes | - | `OnlineSoftmaxState`, `PagedKvDescriptor` |
| attention_tiled_2d | - | Q + output + paged-KV | - | - | - | yes | - | `TransposeLdsReader`, `OnlineSoftmaxState`, MFMA helpers |
| attention_tiled_3d | - | Q + workspace + paged-KV| - | - | - | yes | - | `TransposeLdsReader`, `OnlineSoftmaxState`, MFMA helpers |

## Building Any Instance

```python
from ck_dsl.helpers import compile_kernel
from ck_dsl.instances import (
 UniversalGemmSpec, TileSpec, TraitSpec, build_universal_gemm)

spec = UniversalGemmSpec(
 name="hero",
 tile=TileSpec(tile_m=128, tile_n=128, tile_k=32,
 warp_m=2, warp_n=2,
 warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
 trait=TraitSpec(pipeline="compv4", scheduler="intrawave",
 epilogue="cshuffle"))
kernel = build_universal_gemm(spec)
art = compile_kernel(kernel)
```

`art.hsaco` is ready for `KernelLauncher`; `art.llvm_text` is ready for `analyze_llvm_ir`; `art.timings` is the per-stage codegen budget.

For end-to-end execution, see `runtime/compile_launch_and_manifest.md` and `runtime/manifest_schema.md`.
