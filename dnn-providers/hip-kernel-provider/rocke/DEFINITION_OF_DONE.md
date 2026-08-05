# Definition of Done — Kernel Authoring

A shared, consistent set of expectations for **all kernel authoring work** on the
dual-engine kernel stack (rocke today; intended to be shared with sibling kernel
authoring teams, e.g. Convolution). The goal is that "done" means the same thing
for every author, and that the list of steps lives **here** instead of in each
author's head.

> **Status / placement.** This is a standalone working document. It is not yet
> decided whether its permanent home is in-repo or a Confluence page (some steps
> reference AMD-internal dashboards/links that fit Confluence better). Treat this
> file as the source of truth until that decision is made; see
> [AICK-1872](https://amd-hub.atlassian.net/browse/AICK-1872).
>
> **Testing note.** Run [`tools/run_checks.py`](tools/run_checks.py) before a PR
> merge for sanity. It is **not comprehensive**: its on-GPU numeric lane only
> exercises the arch of whatever GPU is available on your machine, so run it on
> each arch you can.

---

## Table of Contents

- [How to use this](#how-to-use-this)
- [The one command](#the-one-command)
- [Core DoD — every change](#core-dod--every-change)
- [DoD by change type](#dod-by-change-type)
  - [A. New kernel family](#a-new-kernel-family)
  - [B. New knob / optimization](#b-new-knob--optimization)
  - [C. New feature / enablement](#c-new-feature--enablement)
  - [D. Perf optimization (no new feature)](#d-perf-optimization-no-new-feature)
- [Local testing matrix (platforms)](#local-testing-matrix-platforms)
- [Appendix — Attention (SDPA/MHA) specifics](#appendix--attention-sdpamha-specifics)
- [Team sharing](#team-sharing)

---

## How to use this

Pick the row in [DoD by change type](#dod-by-change-type) that matches your work,
do the [Core DoD](#core-dod--every-change) plus that row, and run
[the one command](#the-one-command). A change is **not done** until every
applicable box is ticked and the local runner is green. Agents doing this work
should follow the same list.

## The one command

```bash
python dnn-providers/hip-kernel-provider/rocke/tools/run_checks.py
```

It **auto-discovers** every `test_*.py` and every `<family>_emit.{py,c}` parity
pair — so adding a new test needs **no** edit to any script or checklist. It runs,
in order: relative-path guard → byte-identity gate (both `llvm20` and `llvm22`) →
platform + library parity → pytest (both suites) → on-GPU numeric (auto-skips
when no device). Scope while iterating with:

- `--steps <subset>` — pick stages, e.g. `--steps numeric` (just the on-GPU
  correctness lane) or `--steps gate,parity`.
- `--op <operator>` — scope to one operator/family across parity, pytest,
  numeric, and the gate, e.g. `--steps numeric --op fmha_bwd`.
- `--flavor <llvm>` — pin one LLVM flavor; `--list` — dry-run.

This replaces hand-maintaining a list of tests to run.

## Core DoD — every change

- [ ] **Correctness before speed.** Parity/verify harness run and within
      tolerance *before* any perf number is quoted. Never report a speedup on a
      numerically-wrong kernel.
- [ ] **Dual-engine byte-identity.** Any change to emitted IR is mirrored in
      **both** the Python and C engines, and the golden is re-blessed in the same
      change. Gate GREEN at **both** `llvm20` and `llvm22`.
- [ ] **`problem → spec` and `spec → IR` parity.** Both parity directions pass
      for the touched family (the `.py`/`.c` emitter pair covers `spec → IR`;
      dispatcher/selection parity covers `problem → spec`).
- [ ] **Tests exist and are numeric, not spec-only.** A geometry/spec assertion
      is not sufficient — a kernel/knob needs an on-GPU numeric parity assertion
      within tolerance. (A spec-only test let a correctness regression ship once;
      don't repeat it.)
- [ ] **`run_checks.py` is green** on the platforms in the
      [testing matrix](#local-testing-matrix-platforms).
- [ ] **Docs.** Optimization work leaves a replayable case study in
      `examples/<arch>/<workload>/` (or `builders/<arch>/<workload>/`); a general
      lesson is promoted to the optimization runbook.
- [ ] **Hygiene.** No local-only files (`CLAUDE.md`/`SANDBOX_NOTES.md`/
      `PROBLEM.md`/`.claude/`) or build artifacts staged; Conventional-Commit
      message with a scope; branch `users/<user>/<name>`.

## DoD by change type

This section makes explicit the mental checklist authors currently carry. Every
type also does the [Core DoD](#core-dod--every-change).

### A. New kernel family

- [ ] Spec-driven `build_*()` builder (no one-off scripts); reuse existing
      helpers/atoms first.
- [ ] **C engine mirror** in the same change (byte-identity).
- [ ] `.py`/`.c` parity pair added under `tests/parity/` (auto-picked up by the
      runner).
- [ ] Golden blessed; gate GREEN both flavors.
- [ ] **End-user visibility:** family added to the support matrix
      (`SUPPORT_MATRIX.md` / operation-support doc) so users can see it is
      supported.
- [ ] Numeric test + a benchmark scenario.

### B. New knob / optimization

- [ ] Spec field added, **default-OFF and golden-safe**; `__post_init__` rejects
      illegal combinations.
- [ ] Emission implemented and **mirrored in the C engine**.
- [ ] Knob documented in the knob reference and added to the runbook Knob Catalog.
- [ ] **Step 0 first:** before any algorithm/structure change, an exhaustive
      lever sweep proves the existing config can't already hit the target.
- [ ] Golden re-blessed **iff** output is intended to change; otherwise gate
      stays GREEN with no re-bless (proof the knob is inert by default).
- [ ] Test for the knob (golden-IR if default-ON; on-GPU numeric if it is a
      feature/numeric knob).

### C. New feature / enablement

Aviral's five steps, made explicit:

- [ ] **1. Dispatcher feature list** — register the feature so the dispatcher
      records which kernels support it (e.g. `supports_native_*` / the feature
      gate). This is the map of "every kernel and what it supports."
- [ ] **2. C-side implementation** — the enablement exists in the C engine, not
      just Python.
- [ ] **3. `problem → spec` and `spec → IR` parity** — both pass for the feature.
- [ ] **4. End-user visibility** — reflected in the operation-support /
      `SUPPORT_MATRIX.md` doc so the feature is visible to users.
- [ ] **5. Dashboards & shapes** — the perf dashboard tracks the new feature, and
      the benchmarking owner (Thomas) has the shapes needed to cover it.
- [ ] Numeric test for the feature within tolerance.

### D. Perf optimization (no new feature)

- [ ] **Step 0 exhaustive lever sweep** before concluding a gap is structural.
- [ ] Same-session A/B ratios (median of ≥3) at production-representative scale;
      absolute µs treated as illustrative.
- [ ] Correctness re-verified within tolerance.
- [ ] Replayable case study + runbook/measured-results update.
- [ ] Honest losses recorded, not just wins.

## Local testing matrix (platforms)

Run the local checks on the arches your change affects. The byte-identity gate
enumerates all supported arches intrinsically (it is comgr-compile-only, no GPU
needed); on-GPU numeric needs a device of that arch.

| Arch | Byte-identity gate (no GPU) | On-GPU numeric |
|---|---|---|
| gfx942 | required | if device available |
| gfx950 (baseline default) | required | required for attention changes |
| gfx1151 | required | if device available |
| gfx1201 | required | if device available |
| gfx1250 | required | if device available |

Both LLVM flavors (`llvm20`, `llvm22`) are part of "gate GREEN" — the runner does
both by default. If you lack a device for an arch, use the remote GPU path rather
than faking the lane or using CPU torch.

## Appendix — Attention (SDPA/MHA) specifics

- **Parity families:** the 20 `tests/parity/*_emit.{py,c}` pairs
  (`attention_unified`, `fmha_*`, `gfx9xx_attention_tiled_*`, `sage`/`sparse`).
- **Dispatcher feature gate:** `supports_native_unified_attention*` +
  `select_path` in `library/kernels/common/attention_unified.py` — the
  `(path, head_size, block_size)` selection identity (Aviral's step 1).
- **Knob reference:** `library/builders/common/README.md`.
- **Numeric lane:** `library/tests/differential/numeric_attention.py` (GPU).
- **Process map:** [`library/ENGINEERING_PROCESS.md`](library/ENGINEERING_PROCESS.md).

## Team sharing

This DoD is intended to be **shared across kernel authoring teams**, not just
attention. The [Core DoD](#core-dod--every-change) and
[change-type](#dod-by-change-type) sections are written to be kernel-agnostic;
team-specific specifics go in an appendix like the attention one above. Sibling
teams (e.g. Convolution, led by Kocot) should be able to adopt the core list and
add their own appendix. Onboarding/adoption is tracked in
[AICK-1872](https://amd-hub.atlassian.net/browse/AICK-1872).
