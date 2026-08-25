---
id: technique-mfma-atom
title: "MFMA atom selection (CDNA)"
type: technique
tags: [mfma-atom, mfma]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [cdna]
architectures: [gfx90a, gfx942, gfx950]
operator_families: [gemm, attention, convolution]
rocke_primitive: "MfmaAtom / warp_tile_*"
related: [hw-mfma, hw-gfx942, hw-gfx950, technique-wmma-atom]
sources: [project-rocke]
---

# MFMA atom (CDNA)

Pick M×N×K from `helpers/atoms.py`, gated by `arch_specs.json`.

| gfx | f16/bf16 dense atoms |
|---|---|
| gfx90a / gfx942 | 16×16×16, 32×32×8 |
| gfx950 | those plus K-pack 16×16×32, 32×32×16 |

gfx950 32×32×16 C-layout matches A-input — chained QK→PV without permute.
16×16 chains pay `ds_bpermute`. fp8 wide-K exists on gfx942; MX/fp4/fp6 are
gfx950.

```python
from rocke.helpers.atoms import MfmaAtom
atom = MfmaAtom.f16_32x32x16  # legal on gfx950; rejected on gfx942 by is_valid_spec
```

`probe_intrinsic_counts` must show the new `llvm.amdgcn.mfma.*` before you keep
the change. Occupancy: 32×32 raises acc VGPRs (`technique-occupancy`).
