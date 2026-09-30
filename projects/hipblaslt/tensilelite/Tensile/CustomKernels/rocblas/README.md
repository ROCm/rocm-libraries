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

## `wvSpltK_hf_m1`, `wvSpltK_hf_m2`, `wvSpltK_hf_m4`

FP16 I/O, FP32 alpha/beta, NN only. Block `(64, 16)`, grid = device CU count.
One kernel per M, because rocBLAS compiles a different tile per M rather than
taking it as an argument:

| Kernel | M | YTILE | UNRL | K bound |
| ------ | - | ----- | ---- | ------- |
| `wvSpltK_hf_m1` | 1 | 2 | 2 | none |
| `wvSpltK_hf_m2` | 2 | 2 | 2 | `K <= 16384` |
| `wvSpltK_hf_m4` | 1 to 4 | 3 | 2 | `K <= 8192` |

M=4 is retuned: rocBLAS used `YTILE=7, UNRL=1`, chosen for an 80-CU part. At
304 CUs a 7-wide tile needs `N > 34000` before every wave has a tile, so small-N
decode shapes left most of the machine idle. `YTILE=3, UNRL=2` measures 1.11x
versus hipBLASLt over 20 decode shapes where the original tile measured 0.89x.
M=1 and M=2 keep the rocBLAS tile; sweeping confirmed it is already the best of
the candidates for them.

`wvSpltK_hf_m4` takes M as a kernarg: tile rows past it read the last real row
of A and are never stored, so it serves every `M <= 4`, including M=3. With a
runtime M, `lda == ldc == ldd == M` is no longer a constant, so it also takes
all four leading dimensions as kernargs.

Predicated in `custom.config`: `batch == 1` (no batch strides in the kernarg
list), `N > 8` (below that the tail fixup underflows), `K % 8 == 0`, the K bound
above, and unit strides on all four tensors. m1 and m2 pin their exact `M` and
the packed layout they index without stride arguments: `lda == ldc == ldd == M`,
which is a compile-time constant there and can be compared directly. m4 is
predicated on `M <= 4` instead.

The K bound exists because the M >= 2 kernels read A only from LDS, which holds
`M * K <= 32768` halves; for m4 the bound is sized for the full `M = 4` tile.
rocBLAS had a global-memory fallback there, but it indexes A as row-major and is
wrong for this layout; its host path guaranteed the same bound, so the fallback
was unreachable and is dropped here. M=1 needs no bound: the layouts coincide at
one row, so its fallback stays and `K` above 32768 is correct, only slower.

Still only documented for m1 and m2: `ldb == K`. That is the one layout
constraint a predicate cannot express, because it compares a stride against a
runtime size rather than a constant.

These are the `TRANSA=true` rocBLAS instantiations, i.e. column-major A
(`M x K`) and D (`M x N`), which is what hipBLASLt NN produces.

Tensile retargets `.amdgcn_target` for gfx950; keep the directives.

## `wvSpltK_bf16_tn_m1`, `wvSpltK_bf16_tn_m2`, `wvSpltK_bf16_tn_m4`

BF16 I/O, FP32 accumulate and alpha/beta, hipBLASLt TN with a skinny `n`. This
is the layout `torch.mm(x, w.t())` and `F.linear` reach hipBLASLt with (`m` =
output features, `n` = tokens), and the one rocBLAS served with the
`TRANSA=false` instantiation. Same block, grid and persistent loop as the FP16
kernels; all three build from `wvSpltK_bf16_tn.cpp`.

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
on all four tensors, and `n == 1` / `n == 2` for m1 / m2 or `n <= 4` for m4. K
needs no bound: the tokens are staged in 128 KB of LDS (gfx950 has 160 KB per
CU), and any past that are read from global memory, which is correct in this
layout.

## Kernarg preload

`wvSpltK_hf_m4` and the TN kernels take 72 bytes of kernargs, which spill into
a second 64-byte line. They order everything the staging and the K loop read
first and preload those 14 dwords into SGPRs, so only the epilogue's arguments
are fetched with `s_load`. Tensile strips the preload directives on toolchains
that cannot use them, and the compatibility prologue the compiler emits then
loads the same SGPRs.

## Regenerating

Regenerate assembly (keep `.amdgcn_target` / `.amdhsa_code_object_version`),
then embed the Tensile metadata. FP16 M=1 reads its config from
`custom_rocblas_gemv.yaml`; M=2 and M=4 use the `_m2` / `_m4` files, and the
TN kernels use `custom_rocblas_gemv_bf16_tn_m{1,2,4}.yaml`.

```bash
hipcc -S --cuda-device-only --offload-arch=gfx942 -O3 \
  -o wvSpltK_hf_m2.s wvSpltK_hf_m2.cpp

hipcc -S --cuda-device-only --offload-arch=gfx942 -O3 \
  -mllvm -amdgpu-kernarg-preload-count=14 \
  -o wvSpltK_hf_m4.s wvSpltK_hf_m4.cpp

hipcc -S --cuda-device-only --offload-arch=gfx950 -O3 -DWVSPLTK_M=4 \
  -mllvm -amdgpu-kernarg-preload-count=14 \
  -o wvSpltK_bf16_tn_m4.s wvSpltK_bf16_tn.cpp

python -m Tensile.AddCustomConfig \
  Tensile/CustomKernels/rocblas/wvSpltK_hf_m2.s \
  --yaml Tensile/Tests/custom/custom_rocblas_gemv_m2.yaml \
  --origin rocblas \
  --repository https://github.com/ROCm/rocBLAS-internal \
  --version 1.0.0
```
