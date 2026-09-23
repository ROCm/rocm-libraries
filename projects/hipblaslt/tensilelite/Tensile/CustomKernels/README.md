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
