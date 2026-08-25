---
id: technique-occupancy
title: "Occupancy vs resource budget"
type: technique
tags: [occupancy, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe]
rocke_primitive: "waves_per_eu / probe_occupancy.py"
related: [pattern-register-pressure, pattern-occupancy-loss, hw-cdna, hw-rdna]
sources: [project-rocke]
---

# Occupancy

VGPR, AGPR (CDNA), LDS, and `waves_per_eu` cap waves per CU. More waves hide
latency; more registers/LDS per wave cut waves.

```python
# TraitSpec.waves_per_eu → kernel attr "amdgpu-waves-per-eu"
trait = TraitSpec(pipeline="compv4", waves_per_eu=2)
```

Always run `utilities/tools/dsl_probes/probe_occupancy.py` before/after a tile,
atom, or pipeline change. A “better” ISA that drops below ~4 waves/CU often
loses (`pattern-occupancy-loss`).
