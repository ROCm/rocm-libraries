---
id: process-optimization-loop
title: "Step 0, one-lever loop, escape hatch"
type: process
tags: [routing, escape-hatch]
related: [process-routing, process-probe-sequence, process-escape-hatch, technique-isa-inspect]
sources: [project-rocke]
---

# The loop

Do not redesign the kernel body until the **current** implementation has been
swept. The shipped named-preset bench is not that sweep.

## Step 0 — exhaust implemented levers

1. List every spec field and default-off flag (`__post_init__` documents illegal combos).
2. Cartesian-sweep the legal product for **this shape** (batched comgr compile).
3. Correctness-prune, then time survivors with the same harness.
4. Only if the swept ceiling still misses: enter the loop below, starting from the winning config.

## One-lever loop

```text
hypothesis → verify baseline → measure baseline
  → inspect IR/ISA/resources
  → change one lever (family table + technique page)
  → re-verify → re-measure → explain with an ISA diff
  → keep or revert → record
```

If correctness fails, speed is not a win. Prefer turning a keep into a
heuristic knob rather than a one-off body patch.

## Escape hatch — when the catalog cannot move the limiter

The family table is finite. If **all four** stall boxes are true, **do not
retune tiles**. Open `process-escape-hatch` and invent a new mapping
(`technique-algorithm-break`):

1. Step 0 done for this shape (raw flags, not named presets).
2. At least three one-lever iterations, each with verify + ISA/intrinsic diff.
3. Probe signature unchanged (same occupancy / stall / MFMA-count limiter).
4. The next idea is a technique already kept or reverted this session.

A hatch experiment must add or remove an **opcode class** or change the
launch/loop nest. Query: `python3 scripts/query.py --symptom catalog-exhausted`.

Deep checklist: `optimization_runbook.md` (appendix). Do not read it linearly
when a family/technique page already names the lever.
