<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# rocBLAS custom kernels

Wave-split-K skinny GEMM kernels from rocBLAS-internal `skinnyGemm`
(`library/src/blas_ex/rocblas_gemm_ex_kernels.cpp`).

`CuCount` is a kernarg, not a compile-time constant, so the persistent
stride matches the launch grid on MI300 (gfx942) and MI350 (gfx950).

## `wvSpltK_hf_m1`, `wvSpltK_hf_m2`, `wvSpltK_hf_m4`

FP16 I/O, FP32 alpha/beta, NN only. Block `(64, 16)`, grid = device CU count.
One kernel per M, because rocBLAS compiles a different tile per M rather than
taking it as an argument:

| Kernel | M | YTILE | UNRL | K bound |
| ------ | - | ----- | ---- | ------- |
| `wvSpltK_hf_m1` | 1 | 2 | 2 | none |
| `wvSpltK_hf_m2` | 2 | 2 | 2 | `K <= 16384` |
| `wvSpltK_hf_m4` | 4 | 3 | 2 | `K <= 8192` |

M=4 is retuned: rocBLAS used `YTILE=7, UNRL=1`, chosen for an 80-CU part. At
304 CUs a 7-wide tile needs `N > 34000` before every wave has a tile, so small-N
decode shapes left most of the machine idle. `YTILE=3, UNRL=2` measures 1.11x
versus hipBLASLt over 20 decode shapes where the original tile measured 0.89x.
M=1 and M=2 keep the rocBLAS tile; sweeping confirmed it is already the best of
the candidates for them.

Predicated in `custom.config`: the exact `M`, `batch == 1` (no batch strides in
the kernarg list), `N > 8` (below that the tail fixup underflows), `K % 8 == 0`,
for M >= 2 the K bound above, and the packed layout the kernels index without
stride arguments — unit strides on all four tensors plus `lda == ldc == ldd == M`.
Because each kernel pins `M`, those leading dimensions are compile-time constants
and can be compared directly.

The K bound exists because the M >= 2 kernels read A only from LDS, which holds
`M * K <= 32768` halves. rocBLAS had a global-memory fallback there, but it
indexes A as row-major and is wrong for this layout; its host path guaranteed
the same bound, so the fallback was unreachable and is dropped here. M=1 needs
no bound: the layouts coincide at one row, so its fallback stays and `K` above
32768 is correct, only slower.

Still only documented: `ldb == K`. That is the one layout constraint a predicate
cannot express, because it compares a stride against a runtime size rather than
a constant.

These are the `TRANSA=true` rocBLAS instantiations, i.e. column-major A
(`M x K`) and D (`M x N`), which is what hipBLASLt NN produces.

Tensile retargets `.amdgcn_target` for gfx950; keep the directives.

Regenerate assembly (keep `.amdgcn_target` / `.amdhsa_code_object_version`),
then embed the Tensile metadata. M=1 reads its config from
`custom_rocblas_gemv.yaml`; M=2 and M=4 use the `_m2` / `_m4` files.

```bash
hipcc -S --cuda-device-only --offload-arch=gfx942 -O3 \
  -o wvSpltK_hf_m2.s wvSpltK_hf_m2.cpp

python -m Tensile.AddCustomConfig \
  Tensile/CustomKernels/rocblas/wvSpltK_hf_m2.s \
  --yaml Tensile/Tests/custom/custom_rocblas_gemv_m2.yaml \
  --origin rocblas \
  --repository https://github.com/ROCm/rocBLAS-internal \
  --version 1.0.0
```
