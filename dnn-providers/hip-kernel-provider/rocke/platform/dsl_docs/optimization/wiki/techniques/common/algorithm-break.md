---
id: technique-algorithm-break
title: "Prototype a new algorithm (escape hatch)"
type: technique
tags: [escape-hatch, algorithm-break, common]
symptoms: [catalog-exhausted]
confidence: experimental
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe, small-ops]
rocke_primitive: "family spec default-off field + examples/<arch>/<workload>/"
related:
  - process-escape-hatch
  - pattern-catalog-exhausted
  - technique-fusion
  - technique-persistent-streamk
  - technique-isa-inspect
sources:
  - project-rocke
  - project-hipblaslt
  - project-tensilelite
  - project-stinkytofu
  - project-composablekernel
---

# Prototype a new algorithm

Use only after `process-escape-hatch` stall test passes. This is not a tile
sweep.

## Land it as a default-off spec field

Fork the instance (or add a field) so production stays on the swept winner.
Reject illegal combos in `__post_init__`. Mirror emission in the C++ engine
in the same change if IR changes.

```python
# On the existing family spec (e.g. TraitSpec, tiled attention spec):
use_new_mapping: bool = False  # default-off; dispatcher ignores it

def __post_init__(self) -> None:
    if self.use_new_mapping and not self._legal_on_this_arch():
        raise ValueError("use_new_mapping illegal on this gfx / shape")
```

Drive it from `examples/<arch>/<workload>/` or a raw-flag launch, not from
the named-preset grammar.

## Pass/fail

| Probe | Hatch keep | Still catalog (revert, try another source) |
|---|---|---|
| `probe_intrinsic_counts` | new MFMA/WMMA/async/tr class, or a kernel fused away | same histogram |
| `probe_isa_inspect` | new opcode family or loop nest | waitcnt / tile-only delta |
| numeric harness | within family tolerance | any fail |
| occupancy | allowed to drop if the new mapping is the point | unexplained spill with no mapping change |

## Idea seeds (not a checklist to apply all)

- Unused feature on `hw-<gfx>` that this kernel never emits.
- TensileLite / hipBLASLt / CK Tile / stinkytofu mapping this spec cannot name.
- Cross-family: register-PV, Stream-K, implicit vs direct conv, fused mega MoE.
- One less launch, or persistent/split-K instead of a larger CTA.
