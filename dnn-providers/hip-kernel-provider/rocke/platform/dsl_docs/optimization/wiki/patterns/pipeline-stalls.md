---
id: pattern-pipeline-stalls
title: "Pipeline / waitcnt stalls"
type: pattern
tags: [software-pipeline, waitcnt]
symptoms: [pipeline-stalls, missing-waitcnt]
candidate_techniques: [technique-software-pipeline, technique-async-copy, technique-isa-inspect]
related: [pattern-missing-waitcnt, process-probe-sequence]
sources: [project-rocke]
---

# Pipeline stalls

Loads issue too late, waits sit next to uses, or barriers split a ping-pong
that should be one `s_barrier_bare`. Inspect waitcnt patterns with
`probe_isa_inspect`, then deepen the software pipeline or fix wait placement.
Intermittent wrong answers → `pattern-missing-waitcnt`, not a tile tweak.
