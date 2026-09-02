---
id: hw-cluster-barrier
title: "Named / cluster barriers (2-SM analog — not cooperative MMA)"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [cluster-barrier, gfx1250]
hardware_features: [cluster-barrier]
confidence: inferred
related:
  - hw-gfx1250
  - technique-gfx1250-producer-consumer
  - technique-gfx1250-tile-schedule
sources: [project-rocke]
aliases: [cluster-barrier]
---

# Named / cluster barriers

Catalog: `barrier_model: split_named_cluster`. Analog of KernelWiki cluster
scope — **not** `tcgen05.mma.cta_group::2`. There is no paired-CTA WMMA in
the rocke catalog. gfx1250 is a CDNA multi-chip product; workgroup/cluster
sync is how you split producer and consumer waves, not how you issue one
MMA across two CUs.

Rocke currently lowers a full-CTA `s_barrier` (with split wait drain). Named
and cluster-scoped forms are the programming-model target, not a shipped
snippet. Until a kernel emits them, treat extra barrier scopes as
`process-escape-hatch` experiments, not copy-paste from gfx950 `s_barrier`
timing.
