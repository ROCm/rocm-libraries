---
id: process-escape-hatch
title: "Escape hatch — invent when the catalog is exhausted"
type: process
tags: [routing, escape-hatch]
symptoms: [catalog-exhausted]
related:
  - process-optimization-loop
  - process-routing
  - technique-algorithm-break
  - family-overview
  - pattern-catalog-exhausted
sources: [project-rocke, project-hipblaslt, project-tensilelite, project-stinkytofu, project-composablekernel]
---

# Escape hatch

The family tables and `technique-*` pages are a **closed catalog**. When that
catalog cannot move the bottleneck, **stop retuning tiles**. The required next
step is a new algorithm or a new mapping — something this instance does not
already expose as a spec field.

Do not skip Step 0 to get here. The hatch is illegal until the stall test
passes.

## Stall test (all four)

1. **Step 0 done** for this shape: cartesian sweep of implemented spec fields
   and default-off flags, not the named-preset bench.
2. **At least three** one-lever iterations from the family table, each with a
   verify + ISA / intrinsic diff.
3. **Probe signature unchanged** — same limiter as before those iterations
   (`probe_occupancy`, `probe_intrinsic_counts`, `probe_isa_inspect`, or ATT).
4. The next idea you are about to try is a technique **already kept or
   reverted** this session (tile ±1, same pipeline, same atom family).

If any box is false, return to `process-optimization-loop`. If all four are
true, you are **required** to leave the catalog. Repeating `technique-tiling`
is a loop bug, not caution.

## What counts as a new idea

A hatch experiment must change **work mapping or loop structure**, so the ISA
histogram grows or loses an **opcode class** (new async DMA, `ds_read_tr` /
`ds_load_tr*`, AGPR path, different MFMA/WMMA K-pack, extra kernel fused away,
persistent/Stream-K grid, wave-specialized producer/consumer).

Not a hatch: another `tile_m/n/k`, `num_warps`, pad width, or waitcnt
immediate. Those stay in the one-lever loop.

## Generate one hypothesis (pick a source, do not brainstorm in a vacuum)

Walk these in order. Stop at the first hypothesis that names a **missing
opcode class or mapping** and a probe that would change if it worked.

1. **Unused hardware on this gfx.** `get_page.py hw-<gfx>` vs
   `probe_intrinsic_counts` / ISA. If the page lists async-LDS, `ds_read_tr`,
   AGPR, fp8/MX, cluster-barrier, TDM, and the kernel emits none of that
   class — that is the experiment.
2. **Steal a mapping from the monorepo, not a number.**
   `query.py --operator <fam> --type project` then `get_page.py project-*`.
   hipBLASLt / TensileLite / CK Tile / stinkytofu / rocRoller / Origami /
   MIOpen: what schedule, split, or algorithm class exists there that this
   rocke spec cannot even express?
3. **Steal from another operator family.** Attention register-PV and
   softmax–MFMA interleave; GEMM Stream-K / persistent; conv implicit vs
   direct; MoE fused mega. A lever that is catalog for them is a new
   algorithm here until it has a spec field.
4. **Change the launch/loop nest.** Persistent or Stream-K; split-K;
   grouped vs batched; two launches fused into one; online vs two-pass;
   producer/consumer waves (not just `interwave`); register-resident tile
   vs LDS-resident tile.

Write the hypothesis in one paragraph: *bottleneck → new mapping → which
opcode class or kernel count must move → which probe shows it.* Then open
`technique-algorithm-break` and prototype.

## Prototype contract

Same as the one-lever loop, plus:

- One gfx, one shape, **default-off** flag or a new example kernel. Do not
  retarget the production dispatcher until it is correct.
- Verify before timing. Wrong-and-faster is not a hatch win.
- ISA / intrinsic diff **must** show the new opcode class or a different
  launch count. If the histogram is the same, it was still a catalog lever —
  revert and pick a different source above.
- Keep: default-off spec field, C++ mirror, wiki page
  (`confidence: experimental` until a second shape/arch), family-table cell,
  `scripts/generate-indices.py`.
- Revert is success when the mapping is illegal on this gfx — record *why*
  next to the example so the next agent does not retry it.

Authoring a new instance: `platform/dsl_docs/architecture/authoring_model.md`.
Do not paste software-achieved TFLOP/s, µs, or GB/s into git.
