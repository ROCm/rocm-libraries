# StreamK on RDNA3 — findings, two proposed fixes, and method

Measurement record and tooling from a gfx1100 (RX 7900 XTX) StreamK campaign. **This is a
research record and two bug-fix proposals, not product documentation** — nothing here is
wired into the Sphinx docs under `projects/hipblaslt/docs/`.

Committed because it previously existed only on one developer's disk, including two fixes
that are still needed on `develop`.

## The two fixes — start here

Both were **re-verified against `origin/develop` @ `dab5e862a64` on 2026-08-23**: the defects
are still present and the patches still apply cleanly.

| | what | patch |
|---|---|---|
| [`UPSTREAM_PR_1_glc.md`](UPSTREAM_PR_1_glc.md) | A gfx12-only `th:TH_ATOMIC_RETURN` emitted unconditionally where gfx11 spells it `glc`. **Two independent sites**: `GlobalAtomicIncU32Saddr` (breaks `StreamK: 4`/`5` on *every* RDNA target) and `FlatAtomicDecU32` (breaks GSU `MultipleBufferSingleKernel`). Both fixed and validated. | `patches/rocisa_glc_fix.patch` |
| [`UPSTREAM_PR_2_clamp.md`](UPSTREAM_PR_2_clamp.md) | `TENSILE_STREAMK_TILES` above a problem's tile count underflows a `uint32_t` and **crashes the GPU** (25 launch failures reproduced). The SK3 path already clamps; SK4 does not. | `patches/sk4_clamp_fix.patch` |

PR 2 notes a **second uncovered site** on develop (`sk4_skTiles` in the SK5
`effectiveDynamic` block) with the same defect — read that section before filing.

## Findings

| file | question | answer |
|---|---|---|
| [`SUMMARY.md`](SUMMARY.md) | the campaign in one page | start here for orientation |
| [`REPORT.md`](REPORT.md) | SK3 vs SK4 vs SK5 on gfx1100 | **keep SK3** — best SK4 is 94-95% of best SK3 best-vs-best, and **89.4% like-for-like** |
| [`GATED_POLICY.md`](GATED_POLICY.md) | is StreamK a wall-clock win? | **no** — 96.95% geomean but **102.17% wall-clock**; the metrics disagree *in sign* |
| [`GATE_RESULT.md`](GATE_RESULT.md) | does a size gate help? | **yes, +1.3 to +2.4%** — and read its reconciliation table before quoting any gate number |
| [`SHIP_TEST.md`](SHIP_TEST.md) | should StreamK ship on navi31? | **no — parity at best**, and a default build ships no StreamK kernels anyway |

## Three things that will save you a day

**A default gfx1100 build contains no StreamK kernels.** Across `Logic/asm_full/navi31/`
there are **2560 `StreamK: 0`** against **22 `StreamK: 3`**, and all 22 live in
`Experimental/`, which `tasks.py` excludes by default. Every performance number in these
documents was measured on a purpose-built SK3 catalog, not on anything that ships.

**Per-shape geomean and flops-weighted wall-clock disagreed in sign, three separate times.**
Sub-0.1 ms shapes were 53% of the suite by count and 5% by time. Report both, and say which
one the decision rests on.

**Prefer an inert *partition* to an inert *arm*.** A region where all arms are identical *by
construction* caught 0.28 pt of arm-position drift that an A/A arm structurally cannot see —
and that drift had produced a clean, monotone, entirely false ranking.

[`RUNBOOK.md`](RUNBOOK.md) has the rest: the measurement protocol (the first one had a
**541% p95 noise floor**), the `pgrep -f` self-match trap, and why a jackknife belongs in
every wall-clock comparison on this workload.

## Tools

`analyze.py`, `gate_analyze.py`, `gated_policy.py`, `gated_robust.py`, `plateau_analyze.py`,
`ship_analyze.py`, `matched_pairs.py` — each runs on a partial CSV so a live sweep can be
inspected. The raw measurement CSVs are **not** committed (they are large and reproducible
from the runbook).
