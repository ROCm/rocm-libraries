---
id: pattern-wrong-waitcnt-abi
title: "gfx950 s_waitcnt copied onto gfx1250"
type: pattern
tags: [split-waitcnt, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [wrong-waitcnt-abi, missing-waitcnt]
candidate_techniques:
  - technique-gfx1250-split-waitcnt
  - technique-isa-inspect
related:
  - hw-split-waitcnt
  - hw-asynccnt
  - migration-gfx950-to-gfx1250
sources: [project-rocke]
---

# Wrong waitcnt ABI

ISA still has `s_waitcnt` immediates, or LDS/async races, after a gfx950
port. gfx1250 cannot select `llvm.amdgcn.s.waitcnt`. Fix: backend drain
(`s_wait_loadcnt`/`s_wait_dscnt`) plus `s_wait_asynccnt` on DMA. Do not
hand-encode a gfx9 mask.
