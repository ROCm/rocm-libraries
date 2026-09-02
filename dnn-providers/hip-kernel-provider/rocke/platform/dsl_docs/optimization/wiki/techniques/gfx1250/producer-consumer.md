---
id: technique-gfx1250-producer-consumer
title: "Producer / consumer waves (warp-specialization analog)"
type: technique
tags: [producer-consumer, arch-specific]
symptoms: [pipeline-stalls, valu-plumbing]
confidence: inferred
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention]
rocke_primitive: "s_barrier_bare + per-wave role (wave_id)"
related:
  - hw-cluster-barrier
  - technique-gfx1250-asynccnt-pipeline
  - technique-software-pipeline
  - process-escape-hatch
sources: [project-rocke]
prerequisites: [hw-split-waitcnt]
---

# Producer / consumer waves

Analog of KernelWiki warp-specialization: some waves DMA, some WMMA, some
softmax. gfx1250 WMMA is **per-wave**, not single-thread CTA issue, so a
role split is a software choice — there is no architectural “canonical CTA
warp count.”

gfx9 `TraitSpec.scheduler = interwave` is the closest catalog knob on MFMA
GEMM; gfx1250 WMMA builders do not automatically inherit it. A real split
needs:

- named owner for each LDS stage (acquire / wait cannot deadlock);
- `s_barrier_bare` plus explicit `s_wait_asynccnt` / `s_wait_dscnt` on the
  producer path (a full `s_barrier` drain serializes the next DMA);
- VGPR budget for both roles (`hw-vgpr-acc-gfx1250`).

If you cannot name a new opcode class or a different barrier cadence, stay
in the one-lever loop. A first producer/consumer body is
`process-escape-hatch` until it has a default-off spec field.
