---
id: pattern-async-without-asynccnt
title: "Async DMA without s_wait_asynccnt"
type: pattern
tags: [asynccnt, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [async-without-asynccnt, missing-waitcnt]
candidate_techniques:
  - technique-gfx12-async-lds
  - technique-gfx1250-asynccnt-pipeline
  - technique-isa-inspect
related:
  - hw-asynccnt
  - hw-async-global-lds
  - pattern-missing-waitcnt
sources: [project-rocke]
---

# Async without ASYNCcnt

Intermittent wrong LDS after `global_load_async_to_lds` because the wait
was `s_waitcnt` / loadcnt / dscnt. Async copies complete on **ASYNCcnt**.
Insert `s_wait_asynccnt` (full drain or partial `n` for ping-pong), then
re-verify tails before timing.
