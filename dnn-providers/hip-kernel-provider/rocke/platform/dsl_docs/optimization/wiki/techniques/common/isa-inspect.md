---
id: technique-isa-inspect
title: "ISA and resource inspection"
type: technique
tags: [isa-inspect, common]
confidence: verified
reproducibility: runnable
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe, small-ops]
rocke_primitive: "probe_isa_inspect.py / analyze_hsaco"
related: [process-probe-sequence, technique-occupancy]
sources: [project-rocke]
---

# ISA inspection

Ask: did the compiler emit the intended MMA, load width, wait, and occupancy?
A bench delta without an ISA diff can be a skipped-work cheat or a hidden cvt.

```bash
python3 utilities/tools/dsl_probes/probe_isa_inspect.py --demo implicit_gemm
python3 utilities/tools/dsl_probes/probe_occupancy.py --demo implicit_gemm
python3 utilities/tools/dsl_probes/probe_intrinsic_counts.py --demo implicit_gemm
```

Skill: `utilities/skills/isa-inspection-rocke.md`. stinkytofu is the *generator*
side of scheduling (`project-stinkytofu`); these probes are the rocke *check*.
