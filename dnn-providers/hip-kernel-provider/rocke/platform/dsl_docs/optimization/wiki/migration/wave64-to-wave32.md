---
id: migration-wave64-to-wave32
title: "Wave64 reductions to wave32"
type: migration
from_arch: gfx950
to_arch: gfx1250
architectures: [gfx950, gfx1151, gfx1201, gfx1250]
architecture_families: [cdna, rdna, gfx12]
tags: [migration, wave32, gfx1250]
confidence: verified
reproducibility: snippet
related:
  - technique-wave32
  - family-small-ops
  - family-attention
sources: [project-rocke]
---

# Wave64 → wave32

Required on gfx1250 (and RDNA) for any XOR-butterfly, `ds_bpermute`, or
wave reduce. A `wave_size=64` build issues a lane-32 shuffle that silently
corrupts softmax / layernorm / reduce.

```python
spec.wave_size = 32
# block_size == num_warps * 32
```

Attention: use the gfx1250 WMMA bodies (`library/kernels/gfx1250/`), not
the CDNA MFMA tiled 2D copy with a gfx string swap. Softmax is wave32
online reduce, then P staged in LDS, then PV WMMA.
