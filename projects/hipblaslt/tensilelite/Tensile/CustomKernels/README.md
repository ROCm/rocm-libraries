<!--
Copyright (C) 2025 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Custom kernel validation

In the context of TensileCreateLibrary, Custom kernel meta data is taken from the logic
file where the custom kernel is referenced. The metadata from the logic file can be
overwritten by specifying the parameters of interest in the `custom.config` section of
the kernel assembly file. The majority of the override parameters are validated at build
time when building a custom kernel with exception of the nine element MatrixInstruction.
While this validation is no longer applied during the build process it can be applied by
using the **TensileLogic** script as follows:

```
Tensile/bin/TensileLogic --check-only-custom-kernels <path to logic file>
```

where `<path to logic file>` is a path to a logic file containing a solution where
`CustomKernelName` references the custom kernel requiring validation.

# `Custom_Cijk_Alik_Bljk_I4*_SABB*` (w4a16, gfx1151)

The ten `..._I4B_...` / `..._I4H_...` kernels are the w4a16 prototype: int4
weights in A, bf16 or fp16 activations in B, and one 16-bit scale (optionally
plus a packed int4 zero-point) per K group of a row of A. They are referenced
from `library/.../Logic/asm_full/gfx1151/FreeSize/gfx1151_Cijk_Alik_Bljk_I4*.yaml`
as pre-tuned choices. TensileLite also generates this path from ordinary
solutions with `UseScaleAB: Block`; see `scripts/generate_q27b_w4a16.py` for
reproducing the Q27B prefill configurations without custom assembly.

The name encodes the configuration: `SABB<G><scale type>[ZP][U8]`, where `G`
is the K-group size, the scale type is `B` (bf16) or `H` (fp16), `ZP` marks the
asymmetric variants, and `U8` marks `UnsignedBias8` weights
(no suffix = `Signed`).

Each `.s` overrides `LocalReadVectorWidth{A,B}` back to `-1` in its
`custom.config`. Every reader of a solution state re-runs
`assignDerivedParameters`, and `calLRVW` only tolerates
`LocalReadVectorWidthA < MIInputPerThread` on the auto path; fed back its own
derived value it rejects the solution.

## Q27B group-32 unsigned decode

The Q27B Equality grid selects three decode schedules. The K=5120 shapes use
`Custom_W4A16_Decode_G32_W4_UnsignedBias8_gfx1151`. K=17408 uses the `_U1_A4`
variant, with an unroll factor of one and four independent dot-product
accumulators. K=6144 uses `_NativePerm`, with compiler-native packed FP16
arithmetic and byte permutations for the activation pairs. All three round
scaled weights to FP16 before the FP32 dot products and use the same universal
argument layout.

Regenerate the two additional kernels from the `tensilelite/` directory with
the ROCm compiler on PATH:

```sh
python Tensile/CustomKernels/Source/generate_w4a16_decode.py --group-size 32 --load-width 4 --unroll 1 --accumulators 4
python Tensile/CustomKernels/Source/generate_w4a16_decode.py --group-size 32 --load-width 4 --native-permute
```

Compare the complete Equality library with `hipblaslt-bench --rotating 512
--adaptive --use_gpu_timer`; numerical validation should also cover odd M,
padded row strides, and K values that take the scalar-load fallback.
