# Kernel Authoring — Process & Definition of Done

A shared, consistent set of expectations for **all kernel authoring work** on the
dual-engine kernel stack (rocke today; intended to be shared with sibling kernel
authoring teams, e.g. Convolution). The goal is that "done" means the same thing
for every author, that the list of steps lives **here** instead of in each
author's head, and that the command that proves it is the same for everyone.

This is a **process + policy** doc. It does not restate *how* to author or
optimize a kernel — that lives in the field manual
([authoring_model.md](platform/dsl_docs/architecture/authoring_model.md))
and the optimization runbook
([optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md)),
which this doc routes to. The hard *invariants* it depends on (byte-identity,
one-way dependency, relative-paths, no-`ruff --fix`, torch-optional, default arch)
are owned by [platform/AGENTS.md](platform/AGENTS.md) — that file wins on any
conflict. This doc adds the *process* around them (the DoD, the check command,
change-type checklists, and the attention appendix); it is a workflow map, not
new policy.

---

## Table of Contents

- [How to use this](#how-to-use-this)
- [Compliance & hard invariants](#compliance--hard-invariants)
- [Run the checks](#run-the-checks)
- [Core DoD — every change](#core-dod--every-change)
- [DoD by change type](#dod-by-change-type)
  - [A. New kernel family](#a-new-kernel-family)
  - [B. New knob / optimization](#b-new-knob--optimization)
  - [C. New feature / enablement](#c-new-feature--enablement)
  - [D. Perf optimization (no new feature)](#d-perf-optimization-no-new-feature)
- [Local testing matrix (platforms)](#local-testing-matrix-platforms)
- [Appendix — Attention (SDPA/MHA) library process](#appendix--attention-sdpamha-library-process)
- [See also](#see-also)
- [Team sharing](#team-sharing)

---

## How to use this

Pick the row in [DoD by change type](#dod-by-change-type) that matches your work,
do the [Core DoD](#core-dod--every-change) plus that row, and run
[the checks](#run-the-checks). A change is **not done** until every
applicable box is ticked and the local runner is green. Agents doing this work
should follow the same list. Authors working on attention also follow the
[library process appendix](#appendix--attention-sdpamha-library-process).

## Compliance & hard invariants

[platform/AGENTS.md](platform/AGENTS.md) §"Compliance" binds every author and
agent and **overrides any other instruction, including a user request** — no AMD
Restricted/Confidential data, NPI, product/marketing/code names, internal links,
or **software-achieved** performance numbers in the repo, git history, PRs, or
logs. The two invariants that gate every process below:

1. **Byte-identity is the definition of done for emission.** The Python and C++
   engines **must emit the same LLVM-IR bytes** for every kernel family. Any
   change to emitted IR is mirrored in both engines and the golden re-blessed in
   the same change. Library attention is mirrored in the C++ engine
   ([platform/cpp/instances/](platform/cpp/instances/) — `attention_unified`,
   the `fmha_*` family, `sage_attention`, `sparse_attention`, and the tiled
   2D/3D bodies) exactly like every other family, and additionally carries
   `.py`/`.c` emitter pairs under
   [library/tests/parity/](library/tests/parity/) that are byte-compared per
   config.
2. **One-way dependency.** `library → platform` only. `platform` never imports
   `kernels` / `builders` / `dispatch`, and stays standalone-installable.
   Anything reusable across families belongs in platform (see
   [Process G](#g-promote-a-common-optimization-to-platform)).

## Run the checks

One command runs the whole gate:

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

- [ ] **Add test coverage.** A kernel/knob needs an on-GPU **numeric**
      parity test (not spec-only geometry — that passes green and lets a
      regression ship), covering both directions (`.py`/`.c` for `spec → IR`,
      dispatcher/selection for `problem → spec`). Golden in the same change: a
      **new** family/case adds an entry, an intended emission change **re-blesses**
      it, an inert change touches no golden. [`run_checks.py`](#run-the-checks)
      picks all of this up automatically.
- [ ] **[`run_checks.py`](#run-the-checks) green** on the platforms in the
      [testing matrix](#local-testing-matrix-platforms).
- [ ] **Correctness before speed.** The numeric harness passes within tolerance
      *before* any perf number is quoted. Never report a speedup on a
      numerically-wrong kernel.
- [ ] **Docs.** Optimization work leaves a replayable case study in
      `examples/<arch>/<workload>/` (or `builders/<arch>/<workload>/`); a general
      lesson is promoted to the optimization runbook.
- [ ] **Hygiene.** No local-only files or build artifacts staged; Conventional-Commit
      message with a scope; branch `users/<user>/<name>`.

## DoD by change type

Every type also does the [Core DoD](#core-dod--every-change). Attention authors
also follow the matching [library process](#appendix--attention-sdpamha-library-process).

### A. New kernel family

- [ ] Spec-driven `build_*()` builder (no one-off scripts); reuse existing
      helpers/atoms first.
- [ ] **C engine mirror** in the same change (byte-identity).
- [ ] `.py`/`.c` parity pair added under `tests/parity/` (auto-picked up by the
      runner).
- [ ] **Dispatcher wiring:** candidates registered so the family is selectable —
      the generic `CandidateRegistry` + `dispatch_<family>` for a platform family
      ([dispatch/families/](platform/python/rocke/dispatch/families/)), or
      `ATTENTION_REGISTRY` for attention. A family not in a registry is unreachable.
- [ ] Golden created for the new family; gate GREEN both flavors.
- [ ] **End-user visibility:** family added to the support matrix
      (`SUPPORT_MATRIX.md` / operation-support doc) so users can see it is
      supported.
- [ ] **Docs discoverability:** a genuinely new family gets an
      [instances/index.md](platform/dsl_docs/instances/index.md) catalog row and
      a per-family `instances/<family>.md` (or a section in the closest existing
      one). A new *variant* of an existing family just extends that family's doc.
- [ ] A benchmark scenario for the family (the on-GPU numeric test is already
      required by the Core DoD).

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

- [ ] **Dispatcher feature list** — register the feature so the dispatcher
      records which kernels support it (e.g. `supports_native_*` / the feature
      gate). This is the map of "every kernel and what it supports."
- [ ] **End-user visibility** — feature reflected in the operation-support /
      `SUPPORT_MATRIX.md` doc so users can see it.
- [ ] **Dashboards & shapes** — the feature is tracked by the perf dashboard,
      and the representative shapes needed to cover it are handed to benchmarking.
      (Both are team-internal; measured numbers stay out of the repo.)

### D. Perf optimization (no new feature)

- [ ] **Step 0 exhaustive lever sweep** before concluding a gap is structural.
- [ ] Same-session A/B ratios (median of ≥3) at production-representative scale;
      absolute µs treated as illustrative.
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

---

## Appendix — Attention (SDPA/MHA) library process

A repeatable, end-to-end process for evolving the SDPA/MHA product under
[library/](library/): adding a kernel, adding a spec knob, sweeping an
optimization across shapes, updating docs, extending benchmarks + unit tests,
keeping parity green and re-blessing goldens, and promoting a common optimization
down to [platform/](platform/). It distills the conventions already encoded in
[library/builders/gfx950/attention/README.md](library/builders/gfx950/attention/README.md),
[library/builders/common/README.md](library/builders/common/README.md),
[platform/AGENTS.md](platform/AGENTS.md), and the optimization runbook
([platform/dsl_docs/optimization/optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md)).


### The library layer at a glance

| Layer | Location | Role |
|---|---|---|
| Kernel emitters | [library/kernels/common/](library/kernels/common/), [library/kernels/gfx950/](library/kernels/gfx950/) | Build the typed SSA `KernelDef` (e.g. [attention_unified.py](library/kernels/common/attention_unified.py), [attention_tiled_2d.py](library/kernels/gfx950/attention_tiled_2d.py), [attention_tiled_3d.py](library/kernels/gfx950/attention_tiled_3d.py)) |
| Spec seam | [library/builders/common/attention_spec_builder.py](library/builders/common/attention_spec_builder.py) | Maps `UnifiedAttentionProblem` → arch tiled-spec; owns every knob decision |
| Dispatch | [library/dispatch/attention/](library/dispatch/attention/) | `ATTENTION_REGISTRY` + `dispatch_attention`; selects `(path, head_size, block_size)` |
| Builders / harnesses | [library/builders/gfx950/attention/](library/builders/gfx950/attention/) | Parity + benchmark drivers, captured shape JSONs, case studies |
| Benchmarks | [library/benchmarks/](library/benchmarks/) | Standalone perf sweeps (e.g. [benchmark_rocke_unified_attention.py](library/benchmarks/common/benchmark_rocke_unified_attention.py)) |
| Parity emitters | [library/tests/parity/](library/tests/parity/) | `.py`/`.c` emitter pairs, byte-compared |
| Unit / numeric tests | [library/tests/](library/tests/) | Build, smoke, extended-parity, numeric ([numeric_attention.py](library/tests/differential/numeric_attention.py)) |

### A. Add a new kernel family

Follows the new-kernel rules in [platform/AGENTS.md](platform/AGENTS.md)
("New kernels must become reusable spec-driven builders under `instances/`"),
scoped to library.

1. **Decide the home.** SDPA/MHA product kernel → `library/`. Generic kernel →
   `platform/instances/`. If in doubt, it is a library kernel only if it is part
   of the attention product.
2. **Research and reuse first.** Grep [library/kernels/common/](library/kernels/common/) and
   [platform/Python/rocke/helpers/](platform/Python/rocke/helpers/) for an existing emitter,
   epilogue, atom, or loader before writing new SSA. Extend, don't duplicate.
3. **Author the spec + `build_*()` emitter** under [library/kernels/](library/kernels/) as a
   spec-driven builder (a dataclass `Spec` with a `__post_init__` validator +
   `build_<family>()` returning a `KernelDef`). No one-off scripts —
   [AGENTS.md](platform/AGENTS.md) requires reusable spec-driven builders.
4. **Wire dispatch.** Register the candidate in `ATTENTION_REGISTRY` in
   [library/dispatch/attention/](library/dispatch/attention/) and, if it introduces a new
   selection axis, extend `select_path` / `supports_native_unified_attention`
   (keep the decision a **pure** function of the problem so it mirrors on the C
   side).
5. **Add the parity pair.** Create `tests/parity/<family>_emit.py` (Python
   reference, driven by `run_emit` from
   [platform/tests/instances/parity/_emit_common.py](platform/tests/instances/parity/_emit_common.py))
   and the matching `<family>_emit.c`. Model them on
   [attention_unified_emit.py](library/tests/parity/attention_unified_emit.py) /
   [attention_unified_emit.c](library/tests/parity/attention_unified_emit.c). Cover the
   representative configs *and* edge shapes (see the `_stress_emit` pairs).
6. **Add coverage** ([Process E](#e-add-shapes-to-benchmarks-and-unit-tests)):
   a build/smoke test in [library/tests/](library/tests/), a numeric lane in
   [numeric_attention.py](library/tests/differential/numeric_attention.py)
   (GPU), and a benchmark under [library/benchmarks/](library/benchmarks/).
7. **Bless the golden** ([Process F](#f-parity-and-golden-re-bless)) and
   run the gate GREEN at **both** LLVM flavors.
8. **Document** ([Process D](#d-update-the-docs-runbook-readmes-case-study)):
   an `ALGORITHM.md` if the math is new, plus a `README.md` harness/results doc.

> A family is not "done" until it is wired into registry + tests + parity/golden
> coverage ([AGENTS.md](platform/AGENTS.md)).

### B. Add a new knob to a spec

Knobs are how the existing lever space grows. Model any new knob on the entries
documented in [library/builders/common/README.md](library/builders/common/README.md)
(`use_transposed_half_local_pv`, `use_k_single_buffer`, `use_register_pv`, …).

1. **Add the field to the spec dataclass** (e.g. `UnifiedAttention2DTiledSpec`),
   **default OFF** and golden-safe. Use a field-presence guard if the knob is
   arch-specific (the gfx950-only knobs like `use_v_double_buffer` /
   `use_sched_barrier` / `use_k_single_buffer` are injected this way so the
   gfx942 spec never declares them).
2. **Enforce legality in `__post_init__`.** Reject illegal combinations there —
   this is what makes the cartesian sweep in
   [Process C](#c-sweep-an-optimization-across-all-shapes) safe (it
   enumerates the legal product and lets `__post_init__` prune).
3. **Implement the emission** in the kernel emitter, mirrored in the C parity
   emitter. If the knob changes emitted IR when ON, that is expected — it only
   affects the golden when the dispatcher/parity config turns it ON.
4. **Wire the selector** in
   [attention_spec_builder.py](library/builders/common/attention_spec_builder.py) via an
   `_enable_<knob>()` predicate, following the branch order documented in
   [library/builders/common/README.md](library/builders/common/README.md). Default-OFF opt-in
   knobs may stay unwired (env/flag-only) until a sweep proves a win.
5. **Document the knob** in the "Knob reference" section of
   [library/builders/common/README.md](library/builders/common/README.md): what it rewrites, its
   correctness guarantee, its measured effect, and its enable condition.
6. **Parity + golden.** If the knob is turned ON in any parity config, re-bless
   ([Process F](#f-parity-and-golden-re-bless)). A default-OFF,
   unwired knob should be golden-neutral — verify the gate stays GREEN with **no**
   re-bless as the proof it is inert by default.
7. **Add it to the runbook Knob Catalog** so Step 0 sweeps discover it
   ([optimization_runbook.md §12.1](platform/dsl_docs/optimization/optimization_runbook.md)).

### C. Sweep an optimization across all shapes

This is the mandatory **Step 0** discipline: prove the current implementation
can't already hit the target with a different config before touching the
algorithm.

1. **Enumerate every applicable lever** for the target shape — walk the spec
   dataclass *and* the runbook Knob Catalog, including default-OFF flags
   (those are exactly what a heuristic may be mis-picking). The gfx950 README's
   "Exhaustive microlever sweep" lists a concrete axis set to copy
   ([library/builders/gfx950/attention/README.md](library/builders/gfx950/attention/README.md)).
2. **Run the cartesian sweep.** Enumerate the legal product (`__post_init__`
   rejects illegal combos), batch-compile via threaded comgr (hundreds-to-
   thousands of configs), **correctness-prune against an fp32 reference**, then
   time the survivors against the baseline you must beat. Use the live
   workbenches as the harness:
   `benchmark_prefill2d_live.py` (best correct variant per shape/bucket) and
   `parity_unified_attention.py` for the apples-to-apples auto/2d/3d lanes
   ([library/builders/gfx950/attention/README.md](library/builders/gfx950/attention/README.md)).
3. **Cover all shape cohorts, not one.** Run the full scenario sets
   (`--set default|creative|fmha|all`) and the production-scale caches
   (`--cap-blocks 65536`+ — small caps are artificially L2-resident and
   understate HBM-bound wins, as the README documents at length).
4. **Interpret the ceiling.** If the swept best meets the target → the heuristic
   was mis-routing; turn the swept winner into a tuned knob (Process B). If the
   swept best still falls short → the gap is genuinely structural and a body/
   algorithm redesign is justified, now with the exact resource budget to hit.
5. **Only same-session A/B ratios are load-bearing** on the auto-clocking device
   — report `baseline_us / ck_us` from the same process/stream, median of ≥3
   runs; treat absolute `us` as illustrative (see the README "Measurement
   conditions").
6. **Record the sweep** as a case study ([Process D](#d-update-the-docs-runbook-readmes-case-study)),
   including the honest losses (the gfx950 README's Triton residual is the model
   for reporting a gap you could not close).

### D. Update the docs (runbook, READMEs, case study)

Every optimization leaves three doc artifacts (per [AGENTS.md](platform/AGENTS.md)):

1. **Replayable case study** in `library/builders/<arch>/<workload>/` — the evidence,
   exact commands, traces, config table, and final keep/revert decision, next to
   the code that uses it. Models:
   [gfx1250_mha_optimization_case_study.md](library/builders/gfx1250/attention/gfx1250_mha_optimization_case_study.md),
   and the results sections of
   [library/builders/gfx950/attention/README.md](library/builders/gfx950/attention/README.md).
2. **Results/harness README update** — add the new scenario rows, the geomean,
   the measurement conditions, and the file-map entry for any new script.
3. **Runbook promotion** — a *general* lesson (a new lever, tactic, or bottleneck
   signature) is promoted into
   [platform/dsl_docs/optimization/](platform/dsl_docs/optimization/): a new
   knob into [optimization_runbook.md §12.1](platform/dsl_docs/optimization/optimization_runbook.md),
   a reusable tactic into the relevant skill doc, and the
   concept→code mapping into
   [runbook_compliance.md](platform/dsl_docs/optimization/runbook_compliance.md)
   / [runbook_mapping.md](platform/dsl_docs/optimization/runbook_mapping.md).
   Record the final numbers in
   [measured_results.md](platform/dsl_docs/optimization/measured_results.md).

Doc conventions (project rules): every `.md` with 3+ sections gets a Table of
Contents; every code reference is a clickable relative hyperlink; keep case
studies close to the builder they describe.

### E. Add shapes to benchmarks and unit tests

A new shape/cohort must land in **both** the benchmark set and the test set.

1. **Benchmark scenarios.** Add the `(q_len, kv_len)`, dtype, head geometry, and
   extras to the scenario builder consumed by the harness — `default_scenarios()`
   for the reference set, or the captured shape JSONs
   (`aiter_ua_*shapes.json`) for trace cohorts, both documented in
   [library/builders/gfx950/attention/README.md](library/builders/gfx950/attention/README.md).
   For an exploratory shape use the `creative`/`fmha` sets; promote to `default`
   only once it is a stable, load-bearing cohort.
2. **Parity emit configs.** Add the shape to the `_CONFIGS` dict of the relevant
   `tests/parity/<family>_emit.py` **and** its `.c` twin so the byte-compare
   covers it. Add genuinely small/edge shapes to the `_stress_emit` pair.
3. **Numeric coverage.** Extend the GPU numeric lane in
   [numeric_attention.py](library/tests/differential/numeric_attention.py)
   and/or the extended parity harness spawned by
   [test_extended_parity_attention.py](library/tests/test_extended_parity_attention.py).
4. **Build/smoke coverage.** If the shape exercises a new build path, add it to
   [test_attention_builds.py](library/tests/test_attention_builds.py) or the arch
   smoke test ([test_gfx950_smoke_attention.py](library/tests/test_gfx950_smoke_attention.py)).
5. **Run** `pytest library/tests` (add `platform/tests` if platform was touched);
   GPU numeric lanes need a HIP-visible device.

### F. Parity and golden re-bless

The gate is the emission definition-of-done. Run it from the **platform** tree
(the gate builds the C++ engine and drives the whole harness):

```bash
cd rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform
export ROCKE=$(pwd) PYTHONPATH=$ROCKE/Python

python tools/check_byte_identity.py                            # llvm20, build engine + gate
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # llvm22 flavor
python tools/check_byte_identity.py --only attention           # scope to a family
```

For the library attention `.py`/`.c` parity pairs, the byte-compare runs the
Python emitter and the compiled C emitter for each config index and diffs the
`.ll`. When (and only when) you **intend** to change emitted output:

1. Confirm the change is expected and reviewed — a golden diff is a claim that
   the IR *should* change.
2. Re-bless the flavor-keyed golden in the **same change** (the golden stores one
   sub-document per LLVM flavor; the gate compares only the host's autodetected
   flavor). The bless path is the `--bless`/build-golden flow in the parity
   harness ([rocke_ir_parity_harness.py](platform/tests/instances/rocke_ir_parity_harness.py),
   golden at
   [rocke_representative_ir_sha256.json](platform/tests/golden/rocke_representative_ir_sha256.json)).
3. Re-run the gate GREEN at **both** flavors before considering it done.
4. A default-OFF, unwired knob must produce **no** golden diff — a clean gate
   with no re-bless is the proof it is inert.

> Known layout-independent residuals reproduce on `develop` and are **not** your
> regression: 7 `TestDatalayoutDriftGuard` subtests on a mismatched local clang
> vintage, and 6 `conv/*` golden cases when the C++ engine is not built (see
> [BUILDING.md](BUILDING.md)).

### G. Promote a common optimization to platform

When an optimization proves reusable beyond attention, push it *down* into
[platform/](platform/) — respecting the one-way `library → platform`
dependency. Use the `helpers/` placement rule from
[AGENTS.md](platform/AGENTS.md).

**Promote to `platform/helpers/` when ALL of:**

1. It emits reusable kernel SSA (or is intentionally host-only / fusion-planner), AND
2. At least one of: used (or will be) by ≥2 kernel families; is a general
   emitter/primitive/pipeline (e.g. a software-pipeline, a coalesced tile
   loader, a transpose-LDS layout); a CK-Tile-parity primitive; or it prevents a
   class of silent bugs if duplicated (lane maps, barriers, pipelining).

**Keep it in `library/` when ANY of:** it is single-family, or it is
descriptor/addressing logic specific to attention's layout.

Promotion steps:

1. **Move the reusable SSA emitter into `platform/helpers/`** (Python) and mirror
   it in the **C++ engine** `platform/Cpp/` — a promoted SSA helper requires the
   Python + C++ mirror + byte-identity in the **same** change.
2. **Refactor the library caller** to import the platform helper (never the
   reverse). Library keeps only its attention-specific glue.
3. **Add platform gate coverage** — a case in the platform parity harness
   ([rocke_ir_parity_harness.py](platform/tests/instances/rocke_ir_parity_harness.py))
   and its golden, run GREEN at both LLVM flavors.
4. **Promote the lesson to the runbook** — a new general lever goes in
   [optimization_runbook.md §12.1](platform/dsl_docs/optimization/optimization_runbook.md);
   a reusable tactic becomes/extends a skill doc under
   [platform/dsl_docs/optimization/](platform/dsl_docs/optimization/).
5. **Verify platform still installs standalone** — it must not gain any import of
   `kernels` / `builders` / `dispatch`.

Candidates for promotion visible in the current attention work: the transposed
LDS-read layout (`ds_read_b64_tr_b16` / `TransposeLDSLayout`), the XOR-butterfly
cross-lane softmax (`ds_swizzle`), async-DMA K/V issue ordering, and the 64-bit
paged addressing primitive (`global_ptr_add` / `offset_i64*`) — several of these
already live in platform; new ones that recur across families follow the same
path.

### Attention specifics — quick reference

- **Parity families:** the 20 `library/tests/parity/*_emit.{py,c}` pairs
  (`attention_unified`, `fmha_*`, `gfx9xx_attention_tiled_*`, `sage`/`sparse`).
- **Dispatcher feature gate:** `supports_native_unified_attention*` +
  `select_path` in [library/kernels/common/attention_unified.py](library/kernels/common/attention_unified.py) — the
  `(path, head_size, block_size)` selection identity (feature step 1).
- **Knob reference:** [library/builders/common/README.md](library/builders/common/README.md).
- **Numeric lane:** [library/tests/differential/numeric_attention.py](library/tests/differential/numeric_attention.py) (GPU).

---

## See also

- **How to author a kernel** (contract → spec → grid → descriptors → compute →
  epilogue, + the authoring checklist):
  [platform/dsl_docs/architecture/authoring_model.md](platform/dsl_docs/architecture/authoring_model.md).
- **How to optimize a kernel** (Step 0 exhaustive lever sweep + The Loop):
  [platform/dsl_docs/optimization/optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md).
- **Canonical agent rules / invariants:** [platform/AGENTS.md](platform/AGENTS.md).
- **Build & test:** [BUILDING.md](BUILDING.md), [TESTING.md](TESTING.md).
- **Style:** [style/PYTHON_STYLE.md](style/PYTHON_STYLE.md),
  [style/CPP_STYLE.md](style/CPP_STYLE.md).

## Team sharing

This DoD is intended to be **shared across kernel authoring teams**, not just
attention. The [Core DoD](#core-dod--every-change) and
[change-type](#dod-by-change-type) sections are written to be kernel-agnostic;
team-specific specifics go in an appendix like the
[attention one](#appendix--attention-sdpamha-library-process) above. Sibling
teams (e.g. Convolution) should be able to adopt the core list and add their own
appendix. Onboarding/adoption is tracked in the team issue tracker.
