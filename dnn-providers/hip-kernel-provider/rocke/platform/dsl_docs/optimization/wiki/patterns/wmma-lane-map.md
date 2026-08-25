---
id: pattern-wmma-lane-map
title: "Wrong WMMA K-split / lane map"
type: pattern
tags: [wmma, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [wmma-lane-map]
candidate_techniques:
  - technique-gfx1250-wmma-k32
  - technique-isa-inspect
related:
  - hw-wmma-gfx1250
  - kernel-wmma-gemm-gfx1250
  - migration-gfx1201-to-gfx1250
sources: [project-rocke]
---

# WMMA lane map

Numerics fail on a 16×16 tile, or only fail for K>16, after copying a
gfx1201 16×16×16 pack. gfx1250 f16 K=32 splits K across lane-halves;
bf16 must not go through the gfx11 i16 bitcast.

Re-run `examples/gfx1250/wmma_probe.py` with asymmetric inputs. Do not
“fix” it with a larger tile until the probe matches the reference.
