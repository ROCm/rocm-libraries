<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# rocBLAS custom kernels

Wave-split-K skinny GEMM kernels from rocBLAS-internal `skinnyGemm`
(`library/src/blas_ex/rocblas_gemm_ex_kernels.cpp`).

`CuCount` is a kernarg, not a compile-time constant, so the persistent
stride matches the launch grid on MI300 (gfx942) and MI350 (gfx950).

Names follow rocBLAS: `M` is the skinny side (tokens) and `N` the long side.
In hipBLASLt terms that is `m` and `n` for NN, and `n` and `m` for TN.

## `wvSpltK_f16_nn_m1`, `wvSpltK_f16_nn_m2`, `wvSpltK_f16_nn_m4`

FP16 I/O, FP32 alpha/beta, NN only. Block `(64, 16)`, grid = device CU count.
One kernel per M, because rocBLAS compiles a different tile per M rather than
taking it as an argument:

| Kernel | M | YTILE | UNRL | K bound |
| ------ | - | ----- | ---- | ------- |
| `wvSpltK_f16_nn_m1` | 1 | 2 | 2 | none |
| `wvSpltK_f16_nn_m2` | 2 | 2 | 2 | `K <= 16384` |
| `wvSpltK_f16_nn_m4` | 1 to 4 | 3 | 2 | `K <= 8192` |

M=4 is retuned: rocBLAS used `YTILE=7, UNRL=1`, chosen for an 80-CU part. At
304 CUs a 7-wide tile needs `N > 34000` before every wave has a tile, so small-N
decode shapes left most of the machine idle. `YTILE=3, UNRL=2` measures 1.11x
versus hipBLASLt over 20 decode shapes where the original tile measured 0.89x.
M=1 and M=2 keep the rocBLAS tile; sweeping confirmed it is already the best of
the candidates for them.

`wvSpltK_f16_nn_m4` takes M as a kernarg: tile rows past it read the last real row
of A and are never stored, so it serves every `M <= 4`, including M=3. With a
runtime M, `lda == ldc == ldd == M` is no longer a constant, so it also takes
all four leading dimensions as kernargs.

Predicated in `custom.config`: `batch == 1` (no batch strides in the kernarg
list), `N > 8` (below that the tail fixup underflows), `K % 8 == 0`, the K bound
above, and unit strides on all four tensors. m1 and m2 pin their exact `M` and
the packed layout they index without stride arguments: `lda == ldc == ldd == M`,
which is a compile-time constant there and can be compared directly. `ldb` is a
kernarg on all three, so B may have any leading dimension. m4 is predicated on
`0 < M <= 4` instead.

The K bound exists because the M >= 2 kernels read A only from LDS, which holds
`M * K <= 32768` halves; for m4 the bound is sized for the full `M = 4` tile.
rocBLAS had a global-memory fallback there, but it indexes A as row-major and is
wrong for this layout; its host path guaranteed the same bound, so the fallback
was unreachable and is dropped here. M=1 needs no bound: the layouts coincide at
one row, so its fallback stays and `K` above 32768 is correct, only slower.

These are the `TRANSA=true` rocBLAS instantiations, i.e. column-major A
(`M x K`) and D (`M x N`), which is what hipBLASLt NN produces.

Tensile retargets `.amdgcn_target` for gfx950; keep the directives.

## `wvSpltK_bf16_tn_m1`, `wvSpltK_bf16_tn_m2`, `wvSpltK_bf16_tn_m4`

BF16 I/O, FP32 accumulate and alpha/beta, hipBLASLt TN with a skinny `n`. This
is the layout `torch.mm(x, w.t())` and `F.linear` reach hipBLASLt with (`m` =
output features, `n` = tokens), and the one rocBLAS served with the
`TRANSA=false` instantiation. Same block, grid and persistent loop as the FP16
kernels.

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

`wvSpltK_f16_nn_m4` and the TN kernels take 72 bytes of kernargs, which spill into
a second 64-byte line. They order everything the staging and the K loop read
first and preload those 14 dwords into SGPRs, so only the epilogue's arguments
are fetched with `s_load`. Tensile strips the preload directives on toolchains
that cannot use them, and the compatibility prologue the compiler emits then
loads the same SGPRs.

## Library logic

hipBLASLt picks these kernels through `Range` logic files, one per problem type.
Every range has `batch == 1`:

| Logic file | Ranges (skinny side; long side; K) |
| ---------- | ---------------------------------- |
| `aquavanjaram/gfx942/Range/aquavanjaram_Cijk_Ailk_Bljk_HHS_BH_UserArgs.yaml` | m = 1, 2, 3-4; n >= 9; K >= 8, capped as above |
| `gfx950/gfx950/Range/gfx950_Cijk_Ailk_Bljk_HHS_BH_UserArgs.yaml` | same as gfx942 |
| `gfx950/gfx950/Range/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` | n = 1, 2, 3-4; m >= 9; K >= 8 |

Each file carries the plain-GEMM problem type (no bias, activation or scale
vector), because a custom kernel takes the logic file's problem type and these
kernels support none of those.

Within one device's logic, hipBLASLt searches the plain-GEMM placeholder library
before the Bias/SAV ones, and `Equality` before `Range` inside a placeholder.
Where plain-GEMM logic already ships for the same hardware, the Range file
reuses its header so both share a placeholder, and its tuned sizes keep their
kernels. A matched range whose kernel fails its own predicates (strides, K
bound) returns nothing, so the problem falls through to the next library.

The gfx950 files use the MI350 (`0x75a0`) header. MI355X searches its own tuned
logic (`gfx950_id75a3`) first, so its tuned sizes keep their kernels. On MI350
and on gfx942, the ranges come before skinny sizes tuned in the same device's
Bias libraries.

## Tensile metadata

Each `.s` embeds its Tensile `custom.config`, generated from the Tensile YAML of
the same suffix: `custom_rocblas_gemv_f16_nn_m{1,2,4}.yaml` and
`custom_rocblas_gemv_bf16_tn_m{1,2,4}.yaml`. To
refresh one, delete the existing block first: `AddCustomConfig` will not
overwrite it. Replacement assembly must stay at code object version 4 (see
`../README.md`) and keep its `.amdgcn_target` / `.amdhsa_code_object_version`
directives.

```bash
python -m Tensile.AddCustomConfig \
  Tensile/CustomKernels/rocblas/wvSpltK_f16_nn_m2.s \
  --yaml Tensile/Tests/custom/custom_rocblas_gemv_f16_nn_m2.yaml \
  --origin rocblas \
  --repository https://github.com/ROCm/rocBLAS-internal \
  --version 1.0.0
```
