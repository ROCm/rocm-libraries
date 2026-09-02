---
id: pattern-tail-oob
title: "Incorrect only on padded / tail shapes"
type: pattern
tags: [buffer-rsrc]
symptoms: [tail-oob]
candidate_techniques: [technique-vectorized-io]
related: [process-optimization-loop]
sources: [project-rocke]
---

# Tail / OOB

Fast on aligned hero shapes, wrong on odd dims. Causes: vector crossing the
tail, invalid pointer, missing `buffer_rsrc` OOB clamp (`DW3` sentinel),
descriptor `valid` not tied to `pad` / `pad_dynamic`.

Fix with tiny adversarial shapes and the buffer-resource path in
`optimization_runbook.md` §6.1 / arch §21.6 — not with a bigger tile.
