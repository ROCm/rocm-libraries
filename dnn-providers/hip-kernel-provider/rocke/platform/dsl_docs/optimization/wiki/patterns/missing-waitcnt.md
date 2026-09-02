---
id: pattern-missing-waitcnt
title: "Intermittent wrong answers on async paths"
type: pattern
tags: [waitcnt, async-copy]
symptoms: [missing-waitcnt]
candidate_techniques: [technique-async-copy, technique-software-pipeline, technique-isa-inspect]
related: [pattern-pipeline-stalls]
sources: [project-rocke]
---

# Missing wait / barrier

Async DRAM→LDS or pipelined LDS without `s_waitcnt(vmcnt=…)` / lgkmcnt, or a
workspace freed too early. Add the wait, keep buffers alive across the launch,
then re-verify tails. Do not report speed until the race is gone.
