# rocke `library/` Engineering Process

A repeatable, end-to-end process for evolving the SDPA/MHA product under
[`library/`](.): adding a kernel, adding a spec knob, sweeping an optimization
across shapes, updating docs, extending benchmarks + unit tests, keeping parity
green and re-blessing goldens, and promoting a common optimization down to
[`platform/`](../platform).

This distills the conventions already encoded in
[builders/gfx950/attention/README.md](builders/gfx950/attention/README.md),
[builders/common/README.md](builders/common/README.md),
[platform/AGENTS.md](../platform/AGENTS.md), and the optimization runbook
([optimization_runbook.md](../platform/dsl_docs/optimization/optimization_runbook.md)).
It is a workflow map, not new policy — every hard rule below already lives in
those sources.

---

## Table of Contents

- [Ground rules (read once)](#ground-rules-read-once)
- [The library layer at a glance](#the-library-layer-at-a-glance)
- [Process A — Add a new kernel family](#process-a--add-a-new-kernel-family)
- [Process B — Add a new knob to a spec](#process-b--add-a-new-knob-to-a-spec)
- [Process C — Sweep an optimization across all shapes](#process-c--sweep-an-optimization-across-all-shapes)
- [Process D — Update the docs (runbook, READMEs, case study)](#process-d--update-the-docs-runbook-readmes-case-study)
- [Process E — Add shapes to benchmarks and unit tests](#process-e--add-shapes-to-benchmarks-and-unit-tests)
- [Process F — Parity and golden re-bless](#process-f--parity-and-golden-re-bless)
- [Process G — Promote a common optimization to platform](#process-g--promote-a-common-optimization-to-platform)
- [Definition of Done checklist](#definition-of-done-checklist)

---

## Ground rules (read once)

These constrain every process below:

1. **One-way dependency.** `library → platform` only. `platform` never imports
   `kernels` / `builders` / `dispatch`. Anything reusable across families
   belongs in platform (see [Process G](#process-g--promote-a-common-optimization-to-platform)).
2. **Byte-identity is the definition of done for emission.** Any change to
   emitted IR must be mirrored in both engines and the golden re-blessed in the
   same change. For library attention the "second engine" is the C emitter
   (`.c`) paired with each Python emitter (`.py`) under
   [library/tests/parity/](tests/parity/) — not the C++ `librocke_core.a` (which
   holds no attention mirror today).
3. **Never report speed without correctness.** Run the parity/verify harness and
   stay within the §1.4 tolerances of the runbook *before* claiming any win —
   see [optimization_runbook.md §1.4](../platform/dsl_docs/optimization/optimization_runbook.md).
4. **Never `ruff check --fix` emitter code.** The IR builder is side-effecting;
   F841 autofix silently changes kernels. Lint with `ruff check` (no `--fix`).
5. **Step 0 is mandatory before any algorithm change** — exhaust the existing
   lever space with a cartesian sweep first (see
   [Process C](#process-c--sweep-an-optimization-across-all-shapes) and
   [optimization_runbook.md Step 0](../platform/dsl_docs/optimization/optimization_runbook.md)).
6. **New optimizations leave a replayable case study** in
   `builders/<arch>/<workload>/`, and any general lesson is promoted into
   [platform/dsl_docs/optimization/](../platform/dsl_docs/optimization/).

---

## The library layer at a glance

| Layer | Location | Role |
|---|---|---|
| Kernel emitters | [kernels/common/](kernels/common/), [kernels/gfx950/](kernels/gfx950/) | Build the typed SSA `KernelDef` (e.g. [attention_unified.py](kernels/common/attention_unified.py), [attention_tiled_2d.py](kernels/gfx950/attention_tiled_2d.py), [attention_tiled_3d.py](kernels/gfx950/attention_tiled_3d.py)) |
| Spec seam | [builders/common/attention_spec_builder.py](builders/common/attention_spec_builder.py) | Maps `UnifiedAttentionProblem` → arch tiled-spec; owns every knob decision |
| Dispatch | [dispatch/attention.py](dispatch/attention.py) | `ATTENTION_REGISTRY` + `dispatch_attention`; selects `(path, head_size, block_size)` |
| Builders / harnesses | [builders/gfx950/attention/](builders/gfx950/attention/) | Parity + benchmark drivers, captured shape JSONs, case studies |
| Benchmarks | [benchmarks/](benchmarks/) | Standalone perf sweeps (e.g. [benchmark_rocke_unified_attention.py](benchmarks/common/benchmark_rocke_unified_attention.py)) |
| Parity emitters | [tests/parity/](tests/parity/) | `.py`/`.c` emitter pairs, byte-compared |
| Unit / numeric tests | [tests/](tests/) | Build, smoke, extended-parity, numeric ([tests/differential/numeric_attention.py](tests/differential/numeric_attention.py)) |

---

## Process A — Add a new kernel family

Follows the new-kernel rules in [platform/AGENTS.md](../platform/AGENTS.md)
("New kernels must become reusable spec-driven builders under `instances/`"),
scoped to library.

1. **Decide the home.** SDPA/MHA product kernel → `library/`. Generic kernel →
   `platform/instances/`. If in doubt, it is a library kernel only if it is part
   of the attention product.
2. **Research and reuse first.** Grep [kernels/common/](kernels/common/) and
   [helpers/](../platform/Python/rocke/helpers/) for an existing emitter,
   epilogue, atom, or loader before writing new SSA. Extend, don't duplicate.
3. **Author the spec + `build_*()` emitter** under [kernels/](kernels/) as a
   spec-driven builder (a dataclass `Spec` with a `__post_init__` validator +
   `build_<family>()` returning a `KernelDef`). No one-off scripts —
   [AGENTS.md](../platform/AGENTS.md) requires reusable spec-driven builders.
4. **Wire dispatch.** Register the candidate in `ATTENTION_REGISTRY` in
   [dispatch/attention.py](dispatch/attention.py) and, if it introduces a new
   selection axis, extend `select_path` / `supports_native_unified_attention`
   (keep the decision a **pure** function of the problem so it mirrors on the C
   side).
5. **Add the parity pair.** Create `tests/parity/<family>_emit.py` (Python
   reference, driven by `run_emit` from
   [platform/tests/instances/parity/_emit_common.py](../platform/tests/instances/parity/_emit_common.py))
   and the matching `<family>_emit.c`. Model them on
   [attention_unified_emit.py](tests/parity/attention_unified_emit.py) /
   [attention_unified_emit.c](tests/parity/attention_unified_emit.c). Cover the
   representative configs *and* edge shapes (see the `_stress_emit` pairs).
6. **Add coverage** ([Process E](#process-e--add-shapes-to-benchmarks-and-unit-tests)):
   a build/smoke test in [tests/](tests/), a numeric lane in
   [tests/differential/numeric_attention.py](tests/differential/numeric_attention.py)
   (GPU), and a benchmark under [benchmarks/](benchmarks/).
7. **Bless the golden** ([Process F](#process-f--parity-and-golden-re-bless)) and
   run the gate GREEN at **both** LLVM flavors.
8. **Document** ([Process D](#process-d--update-the-docs-runbook-readmes-case-study)):
   an `ALGORITHM.md` if the math is new, plus a `README.md` harness/results doc.

> A family is not "done" until it is wired into registry + tests + parity/golden
> coverage ([AGENTS.md](../platform/AGENTS.md)).

---

## Process B — Add a new knob to a spec

Knobs are how the existing lever space grows. Model any new knob on the entries
documented in [builders/common/README.md](builders/common/README.md)
(`use_transposed_half_local_pv`, `use_k_single_buffer`, `use_register_pv`, …).

1. **Add the field to the spec dataclass** (e.g. `UnifiedAttention2DTiledSpec`),
   **default OFF** and golden-safe. Use a field-presence guard if the knob is
   arch-specific (the gfx950-only knobs like `use_v_double_buffer` /
   `use_sched_barrier` / `use_k_single_buffer` are injected this way so the
   gfx942 spec never declares them).
2. **Enforce legality in `__post_init__`.** Reject illegal combinations there —
   this is what makes the cartesian sweep in
   [Process C](#process-c--sweep-an-optimization-across-all-shapes) safe (it
   enumerates the legal product and lets `__post_init__` prune).
3. **Implement the emission** in the kernel emitter, mirrored in the C parity
   emitter. If the knob changes emitted IR when ON, that is expected — it only
   affects the golden when the dispatcher/parity config turns it ON.
4. **Wire the selector** in
   [attention_spec_builder.py](builders/common/attention_spec_builder.py) via an
   `_enable_<knob>()` predicate, following the branch order documented in
   [builders/common/README.md](builders/common/README.md). Default-OFF opt-in
   knobs may stay unwired (env/flag-only) until a sweep proves a win.
5. **Document the knob** in the "Knob reference" section of
   [builders/common/README.md](builders/common/README.md): what it rewrites, its
   correctness guarantee, its measured effect, and its enable condition.
6. **Parity + golden.** If the knob is turned ON in any parity config, re-bless
   ([Process F](#process-f--parity-and-golden-re-bless)). A default-OFF,
   unwired knob should be golden-neutral — verify the gate stays GREEN with **no**
   re-bless as the proof it is inert by default.
7. **Add it to the runbook Knob Catalog** so Step 0 sweeps discover it
   ([optimization_runbook.md §12.1](../platform/dsl_docs/optimization/optimization_runbook.md)).

---

## Process C — Sweep an optimization across all shapes

This is the mandatory **Step 0** discipline: prove the current implementation
can't already hit the target with a different config before touching the
algorithm.

1. **Enumerate every applicable lever** for the target shape — walk the spec
   dataclass *and* the runbook Knob Catalog, including default-OFF flags
   (those are exactly what a heuristic may be mis-picking). The gfx950 README's
   "Exhaustive microlever sweep" lists a concrete axis set to copy
   ([builders/gfx950/attention/README.md](builders/gfx950/attention/README.md)).
2. **Run the cartesian sweep.** Enumerate the legal product (`__post_init__`
   rejects illegal combos), batch-compile via threaded comgr (hundreds-to-
   thousands of configs), **correctness-prune against an fp32 reference**, then
   time the survivors against the baseline you must beat. Use the live
   workbenches as the harness:
   [benchmark_prefill2d_live.py](builders/gfx950/attention/README.md) (best
   correct variant per shape/bucket) and
   [parity_unified_attention.py](builders/gfx950/attention/README.md) for the
   apples-to-apples auto/2d/3d lanes.
3. **Cover all shape cohorts, not one.** Run the full scenario sets
   (`--set default|creative|fmha|all`) and the production-scale caches
   (`--cap-blocks 65536`+ — small caps are artificially L2-resident and
   understate HBM-bound wins, as the README documents at length).
4. **Interpret the ceiling.** If the swept best meets the target → the heuristic
   was mis-routing; turn the swept winner into a tuned knob (Process B). If the
   swept best still falls short → the gap is genuinely structural and a body/
   algorithm redesign is justified, now with the exact resource budget to hit.
5. **Only same-session A/B ratios are load-bearing** on the auto-clocking MI355X
   — report `baseline_us / ck_us` from the same process/stream, median of ≥3
   runs; treat absolute `us` as illustrative (see the README "Measurement
   conditions").
6. **Record the sweep** as a case study ([Process D](#process-d--update-the-docs-runbook-readmes-case-study)),
   including the honest losses (the gfx950 README's Triton residual is the model
   for reporting a gap you could not close).

---

## Process D — Update the docs (runbook, READMEs, case study)

Every optimization leaves three doc artifacts (per [AGENTS.md](../platform/AGENTS.md)):

1. **Replayable case study** in `builders/<arch>/<workload>/` — the evidence,
   exact commands, traces, config table, and final keep/revert decision, next to
   the code that uses it. Models:
   [gfx1250_mha_optimization_case_study.md](builders/gfx1250/attention/gfx1250_mha_optimization_case_study.md),
   and the results sections of
   [builders/gfx950/attention/README.md](builders/gfx950/attention/README.md).
2. **Results/harness README update** — add the new scenario rows, the geomean,
   the measurement conditions, and the file-map entry for any new script.
3. **Runbook promotion** — a *general* lesson (a new lever, tactic, or bottleneck
   signature) is promoted into
   [platform/dsl_docs/optimization/](../platform/dsl_docs/optimization/): a new
   knob into [optimization_runbook.md §12.1](../platform/dsl_docs/optimization/optimization_runbook.md),
   a reusable tactic into the relevant skill doc, and the
   concept→code mapping into
   [runbook_compliance.md](../platform/dsl_docs/optimization/runbook_compliance.md)
   / [runbook_mapping.md](../platform/dsl_docs/optimization/runbook_mapping.md).
   Record the final numbers in
   [measured_results.md](../platform/dsl_docs/optimization/measured_results.md).

Doc conventions (project rules): every `.md` with 3+ sections gets a Table of
Contents; every code reference is a clickable relative hyperlink; keep case
studies close to the builder they describe.

---

## Process E — Add shapes to benchmarks and unit tests

A new shape/cohort must land in **both** the benchmark set and the test set.

1. **Benchmark scenarios.** Add the `(q_len, kv_len)`, dtype, head geometry, and
   extras to the scenario builder consumed by the harness — `default_scenarios()`
   for the reference set, or the captured shape JSONs
   (`aiter_ua_*shapes.json`) for trace cohorts, both documented in
   [builders/gfx950/attention/README.md](builders/gfx950/attention/README.md).
   For an exploratory shape use the `creative`/`fmha` sets; promote to `default`
   only once it is a stable, load-bearing cohort.
2. **Parity emit configs.** Add the shape to the `_CONFIGS` dict of the relevant
   `tests/parity/<family>_emit.py` **and** its `.c` twin so the byte-compare
   covers it. Add genuinely small/edge shapes to the `_stress_emit` pair.
3. **Numeric coverage.** Extend the GPU numeric lane in
   [tests/differential/numeric_attention.py](tests/differential/numeric_attention.py)
   and/or the extended parity harness spawned by
   [tests/test_extended_parity_attention.py](tests/test_extended_parity_attention.py).
4. **Build/smoke coverage.** If the shape exercises a new build path, add it to
   [tests/test_attention_builds.py](tests/test_attention_builds.py) or the arch
   smoke test ([tests/test_gfx950_smoke_attention.py](tests/test_gfx950_smoke_attention.py)).
5. **Run** `pytest library/tests` (add `platform/tests` if platform was touched);
   GPU numeric lanes need a HIP-visible device.

---

## Process F — Parity and golden re-bless

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
   harness ([rocke_ir_parity_harness.py](../platform/tests/instances/rocke_ir_parity_harness.py),
   golden at
   [rocke_representative_ir_sha256.json](../platform/tests/golden/rocke_representative_ir_sha256.json)).
3. Re-run the gate GREEN at **both** flavors before considering it done.
4. A default-OFF, unwired knob must produce **no** golden diff — a clean gate
   with no re-bless is the proof it is inert.

> Known layout-independent residuals reproduce on `develop` and are **not** your
> regression: 7 `TestDatalayoutDriftGuard` subtests on a mismatched local clang
> vintage, and 6 `conv/*` golden cases when the C++ engine is not built (see
> [BUILDING.md](../BUILDING.md)).

---

## Process G — Promote a common optimization to platform

When an optimization proves reusable beyond attention, push it *down* into
[platform/](../platform) — respecting the one-way `library → platform`
dependency. Use the `helpers/` placement rule from
[AGENTS.md](../platform/AGENTS.md).

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
   ([rocke_ir_parity_harness.py](../platform/tests/instances/rocke_ir_parity_harness.py))
   and its golden, run GREEN at both LLVM flavors.
4. **Promote the lesson to the runbook** — a new general lever goes in
   [optimization_runbook.md §12.1](../platform/dsl_docs/optimization/optimization_runbook.md);
   a reusable tactic becomes/extends a skill doc under
   [platform/dsl_docs/optimization/](../platform/dsl_docs/optimization/).
5. **Verify platform still installs standalone** — it must not gain any import of
   `kernels` / `builders` / `dispatch`.

Candidates for promotion visible in the current attention work: the transposed
LDS-read layout (`ds_read_b64_tr_b16` / `TransposeLDSLayout`), the XOR-butterfly
cross-lane softmax (`ds_swizzle`), async-DMA K/V issue ordering, and the 64-bit
paged addressing primitive (`global_ptr_add` / `offset_i64*`) — several of these
already live in platform; new ones that recur across families follow the same
path.

---

## Definition of Done checklist

An optimization or new kernel in `library/` is complete only when:

- [ ] Emission mirrored in the Python **and** C parity emitter (`.py` + `.c`).
- [ ] Byte-identity gate GREEN at **both** llvm20 and llvm22; golden re-blessed
      in the same change **iff** output was intended to change.
- [ ] Correctness proven within runbook §1.4 tolerances **before** any speed
      claim (parity/verify harness run).
- [ ] Step 0 exhaustive lever sweep done before any algorithm/structure change.
- [ ] New/changed knob documented in
      [builders/common/README.md](builders/common/README.md) and added to the
      runbook Knob Catalog.
- [ ] New shapes added to **both** benchmarks and unit/parity/numeric tests.
- [ ] Replayable case study left in `builders/<arch>/<workload>/`; results README
      updated with same-session ratios (honest losses included).
- [ ] General lesson promoted into
      [platform/dsl_docs/optimization/](../platform/dsl_docs/optimization/);
      reusable SSA promoted to `platform/helpers/` with the C++ mirror when it
      crosses the ≥2-family / general-primitive bar.
- [ ] `library → platform` dependency direction preserved; platform still
      standalone-installable.
- [ ] `ruff check` (no `--fix`) + `pre-commit` clean over the PR range.
