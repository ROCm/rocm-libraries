---
id: technique-wave32
title: "Wave32 reductions and launch"
type: technique
tags: [wave32]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [rdna, gfx12]
architectures: [gfx1151, gfx1201, gfx1250]
operator_families: [small-ops, attention, gemm]
rocke_primitive: "wave_size=32"
related: [family-small-ops, hw-rdna, hw-gfx1250]
sources: [project-rocke]
---

# Wave32

RDNA and gfx1250 are wave32. XOR-butterfly reductions emit `log2(wave_size)`
shuffles. A `wave_size=64` build on wave32 hardware issues a lane-32 shuffle
that silently corrupts reduce / layernorm / rmsnorm.

```python
spec.wave_size = 32  # required on gfx1151/gfx1201/gfx1250 for reduction trees
```

Block dim must match (`num_warps * 32`). FMHA that hardcodes 64-wide butterflies
needs the WMMA / wave32 body, not the CDNA copy.
