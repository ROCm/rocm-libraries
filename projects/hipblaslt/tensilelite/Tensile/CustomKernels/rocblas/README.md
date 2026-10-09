<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# rocBLAS custom kernels

Wave-split-K skinny GEMM kernels from rocBLAS-internal `skinnyGemm`
(`library/src/blas_ex/rocblas_gemm_ex_kernels.cpp`), for LLM decode.

`CuCount` is a kernarg, not a compile-time constant, so the persistent
stride always matches the launch grid of one workgroup per CU.

Names follow rocBLAS, where `M` is the token count: the `_m1` / `_m2` / `_m4`
suffix is hipBLASLt's `n`.

## `wvSpltK_bf16_tn_m1`, `wvSpltK_bf16_tn_m2`, `wvSpltK_bf16_tn_m4`

BF16 I/O, FP32 accumulate and alpha/beta, hipBLASLt TN with a skinny `n`. This
is the layout `torch.mm(x, w.t())` and `F.linear` reach hipBLASLt with (`m` =
output features, `n` = tokens), and the one rocBLAS served with the
`TRANSA=false` instantiation. Block `(64, 16)`, grid = device CU count. One
kernel per token count, because rocBLAS compiles a different tile per count
rather than taking it as an argument:

| Kernel | n | YTILE | UNRL |
| ------ | - | ----- | ---- |
| `wvSpltK_bf16_tn_m1` | 1 | 2 | 2 |
| `wvSpltK_bf16_tn_m2` | 2 | 2 | 2 |
| `wvSpltK_bf16_tn_m4` | 1 to 4 | 1 | 4 |

gfx950 only: the dot product is `v_dot2c_f32_bf16`, and gfx942 has no BF16 dot
instruction. The m4 tile is tuned for gfx950's 256 CUs: on the GLM-5.2 decode
shapes a wider tile leaves too few waves to hide memory latency, and with
weights streamed from HBM `YTILE=1, UNRL=4` is 1.2x faster than `YTILE=3,
UNRL=2` there, and no slower on larger decode shapes.

Every leading dimension is a kernarg. `torch.mm` on a sliced activation, for
example `q = qkv[:, :q_size]`, reaches hipBLASLt with `ldb > K`, so the kernels
cannot assume a packed layout. m4 also takes `n` as a kernarg and serves every
`n <= 4`.

Predicated in `custom.config`: `batch == 1`, `m > 8`, `K % 8 == 0`, unit strides
on all four tensors, and `n == 1` / `n == 2` for m1 / m2 or `0 < n <= 4` for m4
(hipBLASLt does not quick-return `n == 0`, which grouped GEMM relies on). K
needs no bound: the tokens are staged in 128 KB of LDS (gfx950 has 160 KB per
CU), and any past that are read from global memory, which is correct in this
layout.

## Kernarg preload

The kernels take 72 bytes of kernargs, which spill into a second 64-byte line.
They order everything the staging and the K loop read first and preload those
14 dwords into SGPRs, so only the epilogue's arguments are fetched with
`s_load`. Tensile strips the preload directives on toolchains that cannot use
them, and the compatibility prologue the compiler emits then loads the same
SGPRs.

## Library logic

hipBLASLt picks these kernels through the `Range` logic file
`gfx950/gfx950/Range/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`. Every range
has `batch == 1`, `m >= 9` and `8 <= K <= 4096`:

| Kernel | n |
| ------ | - |
| `wvSpltK_bf16_tn_m1` | 1 |
| `wvSpltK_bf16_tn_m2` | 2 |
| `wvSpltK_bf16_tn_m4` | 3 to 4 |

The ranges stop at K = 4096, short of the kernels' own bound. Past that,
hipBLASLt's own kernels match or beat wvSpltK on MI355X, by up to 1.5x.

The file carries the plain-GEMM problem type (no bias, activation or scale
vector), because a custom kernel takes the logic file's problem type and these
kernels support none of those.

Within one device's logic, hipBLASLt searches the plain-GEMM placeholder library
before the Bias/SAV ones, and `Equality` before `Range` inside a placeholder.
The Range file reuses the header of the shipped plain-GEMM logic
(`gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`), so both
share a placeholder and its tuned sizes keep their kernels; tuning a size there
is how to move it off wvSpltK. A matched range whose kernel fails its own
predicates (for example `K % 8 != 0`) returns nothing, so the problem falls
through to the next library. General-batched (pointer-array) problems skip
custom kernels altogether, since these take plain A/B/C/D addresses.

The file uses the MI350 (`0x75a0`) header. MI355X searches its own tuned logic
(`gfx950_id75a3`) first, so its tuned sizes keep their kernels. On MI350, the
ranges come before skinny sizes tuned in the same device's Bias libraries.

## Tensile metadata

Each `.s` embeds its Tensile `custom.config`, generated from the Tensile YAML of
the same suffix, `custom_rocblas_gemv_bf16_tn_m{1,2,4}.yaml`. To refresh one,
delete the existing block first: `AddCustomConfig` will not overwrite it.
Replacement assembly must stay at code object version 4 (see `../README.md`)
and keep its `.amdgcn_target` / `.amdhsa_code_object_version` directives.

```bash
python -m Tensile.AddCustomConfig \
  Tensile/CustomKernels/rocblas/wvSpltK_bf16_tn_m2.s \
  --yaml Tensile/Tests/custom/custom_rocblas_gemv_bf16_tn_m2.yaml \
  --origin rocblas \
  --repository https://github.com/ROCm/rocBLAS-internal \
  --version 1.0.0
```
