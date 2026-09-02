---
id: process-probe-sequence
title: "Probe sequence before rocprof"
type: process
tags: [routing, isa-inspect]
related: [process-optimization-loop, technique-isa-inspect, technique-occupancy]
sources: [project-rocke]
---

# Probe sequence

Scripts live under `utilities/tools/dsl_probes/` (no GPU for most `--demo`s).

```text
1) probe_config_sweep --only-build
      SPEC-FAIL → coupled fields; BUILD-FAIL → isolate IR coverage
2) probe_occupancy
      spill → VGPR; waves low → LDS or tile; limiter annotated
3) probe_intrinsic_counts
      no MFMA/WMMA → wrong atom; no async DMA → pipeline; barrier flood
4) probe_targeted_bench vs baseline (same harness)
5) probe_isa_inspect
      scalar stores; ds_read without tr; waitcnt patterns
6) rocprofv3 / ATT / WaveScope
      memory_stall → technique-async-copy
      lds_stall → technique-lds-swizzle
      compute healthy but slow epilogue → technique-epilogue
```

ATT capture: `utilities/skills/capture-kernel-trace-rocke.md` and
`utilities/tools/stage2_capture/`. WaveScope `annotations.json` /
`notes.json` is a two-writer inner loop on one trace — complementary, not a
wiki page rewrite.
