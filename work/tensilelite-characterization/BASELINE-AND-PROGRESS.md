# TensileLite codegen-coverage — baseline, progress, and provenance

Single rigorous record of **where we started, where we are, and exactly how each number was
obtained**. Companion to `PLAN-CODEGEN-WORKFLOW.md` (narrative/plan) and `coverage/` (raw
receipts). Last updated 2026-06-06.

---

## 1. The metric (read this first — numbers are meaningless without it)

- **Tool:** `coverage.py` 7.x, branch coverage **on** (`pyproject.toml`), measuring the
  `Tensile/` package (`--cov=Tensile` — a **path**, never a dotted module).
- **Reported number = the `coverage report` `TOTAL` "Cover" column**, which with branch coverage
  enabled is **branch-inclusive**: `Cover% ≈ (covered_stmts + covered_branch_edges) /
  (total_stmts + total_branches)`. It is therefore **lower** than plain line coverage.
- **Line-only coverage** (when we want it) = `(Stmts − Miss) / Stmts`. We quote both where it
  matters; the campaign's headline metric is the branch-inclusive `Cover%` per the plan.
- **Denominator caveat (important for rigor):** `coverage.py` only counts statements in files
  that were **imported** during the run. Different suites import different file sets, so the
  total-statement denominator drifts (~54,491–55,124 across the runs below, a ~1.1% band). A
  percentage-to-percentage comparison across two runs is therefore **approximate**; the
  apples-to-apples comparison is same-methodology / same-denominator, or compare covered-stmt
  counts directly. Each row below carries its own `Stmts` so the denominator is explicit.

Two measurement **methodologies** are used, and they are not identical:

| Methodology | Used for | How |
| --- | --- | --- |
| **A — tox full-suite** | develop baseline; whole-suite gate | tox `coverage-unit`/`coverage-common` envs: reinstall the package, run the whole suite under `pytest-xdist`, `coverage combine` the worker data. Safe for the full suite (xdist workers bound the per-process rocisa footprint). |
| **B — campaign isolated-run + combine** | P0–P3 seed/attribution numbers | one `docker exec … pytest --cov=Tensile` **per input**, each with its own `COVERAGE_FILE`; `coverage combine` reconstructs the union. Avoids the per-process rocisa footprint limit (P0 finding) and lets us attribute lines to single inputs. |

> A↔B are comparable only approximately (different denominators, different harness). For a
> strict develop→now delta, re-measure HEAD with **methodology A** (the tox env) — **not yet
> done on this branch post-P3** (see §4, open item).

Environment for every measurement: container **`tl-char`**, in-container project
`/work/projects/hipblaslt/tensilelite`, cp312 `pytest`/`coverage` entrypoints (the container's
`python3.11` has neither). CPU-only throughout.

---

## 2. Where we started (baselines)

### develop branch — the true starting point
Worktree `tensilelite-develop-cov` @ `8f9a5fe9ad8` (develop HEAD), methodology **A**, 2026-06-05.

| Suite | Tests | Stmts | Miss | Branch | BrPart | **Cover (branch-incl.)** | Line-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `-m unit` (develop) | 1237 passed / 201 skipped | 55124 | 40861 | 22922 | 721 | **22.47%** | 25.87% |
| common (develop) | — | 54803 | 43031 | 22806 | 1042 | **18.51%** | 21.48% |

Receipts: `coverage/develop-unit-20260605-155701.log`, `coverage/develop-common-20260605-155701.log`.

**This is the headline baseline: the develop `-m unit` suite covers 22.47% of the whole project.**

### our branch, pre-campaign-seed snapshot (for reference only — now stale)
`coverage/CURRENT-full-package.txt`, our branch, methodology A-ish (full `pytest` session,
2667 items), 2026-06-05, **before P2/P3**:

| Run | Stmts | Miss | Branch | BrPart | Cover | Line-only |
| --- | --- | --- | --- | --- | --- | --- |
| full package (pre-P2) | 54812 | 36338 | 22764 | 989 | **30.62%** | 33.70% |

Already above develop because the prior characterization work (PLAN-80 / G0) is on this branch.
Stale now — P2 (+15 seeds) and P3 landed after it.

---

## 3. Progress — per phase, with provenance

### P0 — method validation (NOT a whole-project number)
4-test **de-risk subset**, methodology B, 2026-06-04. **File-scoped to the two core codegen
files only** (this is the number people misremember as a baseline — it is not whole-project):

| File | Before | After (merged 4 tests) |
| --- | --- | --- |
| `KernelWriterAssembly.py` | 52.00% | **59.18%** |
| `KernelWriter.py` | 57.27% | **65.93%** |
| TOTAL *(those 2 files only)* | 53.88% | **61.59%** |

What P0 proved: `coverage combine` reconstructs the union of separate-process runs (merged >
best single 53.88%); the per-process rocisa footprint is per-process (fresh process per input
avoids it). Receipt/context: `toolchain-inventory.md`, commit `d6ae0b113d9`.

### P1 — attribution (covered-line counts per arch, not %)
Methodology B, 29 curated inputs/arch, isolated, 2026-06-05. Per-arch **union covered lines**
(the empirical reach of the curated corpus, used to design P2):

| arch | union covered lines |
| --- | --- |
| gfx942 | 21900 |
| gfx950 | 22121 |
| gfx90a | 18587 |

Output: `_codegen/attribution-{arch}.json` (ranked by marginal yield). Commits `0d818440d0c`,
`97a331c434f`.

### P2 — minimal custom seed set (widened, option b) → the Stage-1 ceiling
Methodology B, 15 seed inputs across gfx942/gfx950/gfx90a, isolated then combined, 2026-06-06.

| Seed set | Stmts | Miss | Branch | BrPart | **Cover** | Line-only |
| --- | --- | --- | --- | --- | --- | --- |
| single-config-per-arch (first pass) | 54491 | — | — | — | **30.53%** | — |
| **widened, multi-ProblemType (final)** | 54491 | 33335 | 22648 | 2702 | **35.89%** | 38.83% |

Why widened: the config harness reads only `BenchmarkProblems[0]` (one ProblemType/DataType per
config), so one config per arch undershot at 30.53%. Spanning distinct families per arch
(cheapest-first, mostly shipped `Tests/common/**` configs) lifted it to 35.89% (+2773 covered
stmts). Receipt: `coverage/p2/ceiling-widened.txt`. Commit `ec7524bd1be`.

> **Notable:** the 15 seeds (35.89%) already exceed the *entire develop `-m unit` suite*
> (22.47%) on whole-project coverage — the seeds drive the full codegen emit path end-to-end,
> which the develop unit tests barely touch.

### P3 — goldens (no coverage change; pins behavior + reproduces the ceiling)
Methodology B, 2026-06-06. Added order-invariant `{basename, err}` snapshot goldens to the 15
seed tests (NOT a text hash — the emitter's process-global scheduler state makes text hashes
flaky; matches `matrix.py`). Seed suite reproduces the P2 ceiling **exactly (35.89%, +0)**.
Full `-m unit` gate (our branch): **2543 passed / 201 skipped / 0 failed** (pass/fail only — no
`--cov` % captured for the full suite at P3). Receipt: `coverage/p3/master-baseline-p3.txt`,
`coverage/p3/unit-gate.txt`. Commit `46842e5`.

---

## 4. Where we are now

- **Stage-1 codegen seed set: 35.89% whole-project, CPU-only** — the committed, goldened,
  fast-reproducing efficiency baseline for Stage-2 expansion.
- **develop `-m unit` baseline: 22.47%.** Campaign delta so far (seeds vs develop): roughly
  +13 points, but across methodologies A↔B (treat as approximate).
- **Open measurement (for strict rigor):** the full `-m unit` whole-project `Cover%` on
  **HEAD** (seeds + all existing characterization) has **not** been re-run with methodology A
  since P3. The last comparable full-suite figure is the stale pre-P2 30.62%. Recommended next
  rigorous step before/with P4: re-run the tox `coverage-unit` env on HEAD to get the true
  current full-suite number and the apples-to-apples develop→now delta.

### The gap to 80% (honest framing)
From a develop baseline of ~22% (or the seed-set 35.89%), whole-project ≥80% CPU-only is a
large climb. A material share of the remainder is **not CPU-reachable today**: the **client
perf-run** and the **device probes** (incl. ISA detection) — i.e. the work gated on the
**GPU-mock prerequisite (P0.5 / Track A)**, see `GPU-MOCK-PR.md`. P4 must tag those targets
`needs_cpu_only_switch` and skip them until that PR lands; the honest ceiling without it is
expected to sit well below 80% and must be **documented with per-region evidence**, never
inflated. (Rigor rule: every reported number is measured with a saved receipt; every
"unreachable" claim carries file:line evidence.)

---

## 5. Receipt index
- develop: `coverage/develop-unit-20260605-155701.log`, `coverage/develop-common-20260605-155701.log`
- our branch pre-P2: `coverage/CURRENT-full-package.txt`
- P2: `coverage/p2/ceiling.txt`, `coverage/p2/ceiling-widened.txt`, `coverage/p2/unit-gate.txt`
- P3: `coverage/p3/master-baseline-p3.txt`, `coverage/p3/unit-gate.txt`
- attribution: `_codegen/attribution-{gfx942,gfx950,gfx90a}.json`
- prior campaign baselines (PLAN-80): `coverage/master-baseline-*.txt`
