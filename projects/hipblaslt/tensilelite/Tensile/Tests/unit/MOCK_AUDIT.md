# Audit: cms_test_utils.py mock infrastructure

> **BRANCH-SPECIFIC AUDIT — READ BEFORE VERIFYING.**
>
> Every claim in this document was derived against the
> **`users/alvasile/validator_long_term_plans` (vlt)** branch. The file
> `Tensile/Tests/unit/cms_test_utils.py` itself only exists on that branch
> (and descendants). On `develop` and most other branches, `cms_test_utils.py`
> is absent and `CMSValidationTestBase` still has 24+ live subclasses, so the
> audit's headline finding ("no subclasses exist, the mock chain is dead")
> **does not hold there.**
>
> If you ran `grep -rn "CMSValidationTestBase" Tensile/Tests/` from `develop`
> (or any branch that has not picked up the vlt cleanup) and got many hits,
> that is expected and does not contradict this audit. To verify the audit's
> findings:
>
> ```
> git checkout users/alvasile/validator_long_term_plans
> grep -rn "CMSValidationTestBase" projects/hipblaslt/tensilelite/Tensile/Tests/
> ```
>
> On vlt the only hits are the definition site plus three documentation
> references in `graph_native_validation_base.py` /
> `test_LR_Pack_interaction.py` describing the legacy approach being
> migrated away from — i.e. zero subclasses.

**Bead:** `rocm-libraries-5u7q`
**Branch audited:** `users/alvasile/validator_long_term_plans` (vlt) — work performed in worktree
**Scope:** static-analysis audit of `Tensile/Tests/unit/cms_test_utils.py`. No code
deletions. Recommendations only — follow-up beads gated on user direction.

---

## TL;DR

The 4ft cleanup deleted ~250 LoC of unused mock infrastructure. **There is more.**

> **Branch caveat (repeat of banner):** all findings below are against the
> **vlt** branch only. On `develop` the audited file does not exist and
> `CMSValidationTestBase` still has live subclasses. Do not apply this
> audit's conclusions to other branches without re-running the call-graph
> analysis there.

The single most important finding from this audit: **no test in the repository
subclasses `CMSValidationTestBase` (on the vlt branch).** As a consequence,
every helper that is reachable *only* through that base class is currently
dead code:

- `make_mock_id_map` — the central mock-builder. Its sole non-base-class caller
  is `_make_context()` in `test_CustomSchedule.py:32`, **which is itself never
  called.**
- `make_mock_mfma_code` — same: only consumed by `_make_context` and the dead
  base class.
- `_make_mock_lr` / `_make_mock_gr` / `_make_mock_grinc` / `_make_mock_swap` /
  `_make_mock_lw_swap` / `_make_mock_lcc` / `_make_mock_packs_bf16` — all
  reached only via `make_mock_id_map`.
- `subset_id_map` / `kernel_to_solution_config` / `_frozen_config_key` — used
  only by the dead `CMSValidationTestBase.validate()` real-idmap path
  (`kernel_to_solution_config` has zero callers anywhere).

What **is** load-bearing in this file:

- `_make_solution` — directly imported by 8 sites in `test_ScheduleCapture.py`.
- `generate_real_idmap` — directly imported by `test_ScheduleCapture.py:674`.
- The register-space constants (`LRA_BASE`, …) — only consumed by the now-dead
  mock builders, but they document the design and cost nothing.

Recommendation: file follow-up beads to (a) delete `CMSValidationTestBase`
and the unreferenced `_make_context` shim, then (b) delete the entire mock
chain that becomes orphaned. The audit doc preserves the categorisation so
the cleanup can land without re-derivation.

> **Clean-implementation directive note:** any follow-up bead that lands the
> deletion should produce a clean diff — no dead-code stubs, no compatibility
> shims for the helpers being removed.

---

## Inventory and call-site map

The full call graph below was derived with `grep -rn` over
`projects/hipblaslt/tensilelite/Tensile/Tests/`. Every claim about who
calls what was verified against the source — the file's own docstrings
were treated as hints, not ground truth.

### Direct entry points (imported by test files)

| Symbol                       | Imported by                                                  | Test-file callers (live)            |
| ---------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| `make_mock_id_map`           | `cms_validation_base.py`, `test_CustomSchedule.py`           | **none** (see below)                |
| `make_mock_mfma_code`        | `cms_validation_base.py`, `test_CustomSchedule.py`           | **none** (see below)                |
| `generate_real_idmap`        | `cms_validation_base.py`, `test_ScheduleCapture.py:674`      | `test_tf32_4x4_tn_capture_shape`    |
| `subset_id_map`              | `cms_validation_base.py`                                     | **none**                            |
| `_frozen_config_key`         | `cms_validation_base.py`                                     | **none**                            |
| `_make_solution`             | `test_ScheduleCapture.py` (8 sites)                          | live, multiple tests                |
| `kernel_to_solution_config`  | nothing                                                      | **none**                            |

### Internal helpers (called only from `make_mock_id_map`)

| Symbol                  | Called from                                          |
| ----------------------- | ---------------------------------------------------- |
| `_make_mock_lr`         | `make_mock_id_map` (LRA*/LRB* keys)                  |
| `_make_mock_gr`         | `make_mock_id_map` (GRA/GRB keys)                    |
| `_make_mock_grinc`      | `make_mock_id_map` (GRIncA/GRIncB keys)              |
| `_make_mock_swap`       | `make_mock_id_map` (LRSA/LRSB keys)                  |
| `_make_mock_lw_swap`    | `make_mock_id_map` (LWSA/LWSB keys)                  |
| `_make_mock_lcc`        | `make_mock_id_map` (LCC key)                         |
| `_make_mock_packs_bf16` | `make_mock_id_map` (Pack* keys)                      |

### Why every direct caller above is dead

1. **`CMSValidationTestBase`** (`cms_validation_base.py:38`) — `grep -rn
   "CMSValidationTestBase"` returns one definition site and three
   *documentation references* in `graph_native_validation_base.py` /
   `test_LR_Pack_interaction.py` describing it as the legacy approach being
   migrated **away from**. Zero subclasses remain. The `validate()` method
   at line 125 is the only routine that consumes `make_mock_id_map`,
   `make_mock_mfma_code`, `subset_id_map`, `_frozen_config_key`, and
   `generate_real_idmap` (via `_get_real_idmap`).

2. **`_make_context` in `test_CustomSchedule.py:32`** — defined; the file's
   docstring says it builds a `ValidationContext` from mocks for the
   per-shape positive tests. `grep -rn "_make_context"` returns the
   definition only — there are no callers. The remaining tests in that file
   (`TestCustomScheduleValidation`, `TestSchedulePositionOrdering`) build
   their own minimal idMaps inline (`{"LRA0": [], …}`) without calling
   `_make_context`, `make_mock_id_map`, or `make_mock_mfma_code`.

3. **`real_id_map_config`** — class attribute on `CMSValidationTestBase`
   default `None`; `grep -rn "real_id_map_config"` returns *only* the five
   self-references inside `cms_validation_base.py`. Even if a subclass
   existed it would have to opt in. Nothing opts in.

---

## Per-helper categorisation

Categories follow the bead's taxonomy:

- **Replaceable**: real codegen (`generate_real_idmap`) would produce the
  same shape; the mock can be deleted and the test rewritten against the
  real path.
- **Load-bearing**: mock constructs a shape real kernels don't produce; keep,
  and document why.
- **Dead**: gated on flags no test sets, or no callers at all.

| Helper                  | Classification | Reason                                                                                                                  | Recommendation                                           |
| ----------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `make_mock_id_map`      | **Dead**       | Only callers are dead `CMSValidationTestBase.validate()` and never-called `_make_context()`.                            | Delete with the base class.                              |
| `make_mock_mfma_code`   | **Dead**       | Same call-site situation as `make_mock_id_map`.                                                                         | Delete with the base class.                              |
| `_make_mock_lr`         | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_gr`         | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_grinc`      | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_swap`       | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_lw_swap`    | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_lcc`        | **Dead**       | Only callable through `make_mock_id_map`.                                                                               | Delete (transitive).                                     |
| `_make_mock_packs_bf16` | **Dead**       | Only callable through `make_mock_id_map`. (Note: name implies BF16-only — the actual VPermB32 shape is generic.)        | Delete (transitive).                                     |
| `subset_id_map`         | **Dead**       | Only caller is `CMSValidationTestBase.validate()`.                                                                      | Delete with the base class.                              |
| `kernel_to_solution_config` | **Dead**   | `grep -rn` finds zero callers. Defined; never used; never imported anywhere.                                            | Delete unconditionally.                                  |
| `_frozen_config_key`    | **Dead**       | Only caller is `CMSValidationTestBase._get_real_idmap`.                                                                 | Delete with the base class.                              |
| `_make_solution`        | **Load-bearing** | 8 direct call sites in `test_ScheduleCapture.py` (`TestRealKernelCapture` and downstream classes).                    | **Keep.** This is the canonical entry point for real-codegen tests. |
| `generate_real_idmap`   | **Load-bearing** | Used by `test_ScheduleCapture.py::test_tf32_4x4_tn_capture_shape` — the canonical "real codegen instead" template the bead names. | **Keep.** Convenience wrapper around `_make_solution + _getKernelSource + _last_id_map`. |
| Register-space constants (`LRA_BASE`, `LRB_BASE`, `PACK_A_DST_BASE`, `PACK_B_DST_BASE`, `MFMA_ACC_BASE`) | **Dead** (transitive) | Only consumed by the mock builders.                                                                                     | Delete with the mock builders. |

**No helper meets the "load-bearing mock" criterion** (a fixture for an
edge case real codegen can't produce). Every mock helper was infrastructure
for the abandoned `CMSValidationTestBase` validate() pipeline. The
load-bearing pieces are the *real-codegen* helpers that just happen to live
in the same file.

---

## Could `generate_real_idmap` replace each mock?

The bead asks this per-call-site, but with no live mock call sites the
question collapses. For completeness:

- The `CMSValidationTestBase.validate()` real-idmap path (lines 159–169)
  already shows the migration pattern it would have used: build a real
  idMap with `generate_real_idmap`, then `subset_id_map` it down to the
  schedule's first code path while falling back to the mock idMap for
  exotic schedule keys (e.g. `PackA3` from `ForceUnrollSubIter`). That
  fallback is the concession that the real codegen *cannot* produce
  every key the validator expects.
- That concession is itself moot now: with no live tests using the path,
  the question of "could real codegen cover these keys" never has to be
  answered. If/when a future test wants a CMS-validator regression
  fixture, it should follow the `test_validate_pack_graph.py` /
  `test_LR_Pack_interaction.py` pattern (graph-native, hand-built capture
  via `dataflow_fixtures.make_lr` / `make_mfma`), which is the post-CMS
  approach the migration is heading toward.

---

## Cost analysis

Representative timings on the worktree host (single warm pytest run; one
warm-up before each measurement, then the reported run):

| Test                                                                      | Wall time | Notes                                                       |
| ------------------------------------------------------------------------- | --------- | ----------------------------------------------------------- |
| `test_CustomSchedule.py` (10 mock-style tests, full file)                 | 0.41 s    | Mock-only path; no `isa_infrastructure`.                    |
| `test_ScheduleCapture.py::test_tf32_4x4_tn_capture_shape` (single test)   | 6.78 s    | Includes ~3.8 s `isa_infrastructure` session-fixture probe. |
| `test_ScheduleCapture.py` (74 passed, 1 skipped, full file)               | 21.21 s   | Same fixture amortised across 74 tests (~0.23 s/test once warm). |

### Timing reproducer

The numbers above were measured on the vlt branch with the commands below.
A future reader can re-run them on the same branch to verify (or refresh)
the figures. All commands assume a working tree at the repo root and an
already-built tensilelite environment (the same `tox -e unit` /
`pytest Tensile/Tests/unit/` setup the project CLAUDE.md describes).

```
# Anchor: vlt branch, with the audited file present
git checkout users/alvasile/validator_long_term_plans

cd projects/hipblaslt/tensilelite

# Always exclude the slow MatrixInstructionConversion test (per project memory)
PYTEST_IGNORE='--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py'

# Warm-up runs (discard timings) — needed because the first invocation
# pays import / bytecode-cache costs that distort the measurement.
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_CustomSchedule.py >/dev/null
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_ScheduleCapture.py::test_tf32_4x4_tn_capture_shape >/dev/null
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_ScheduleCapture.py >/dev/null

# Row 1: mock-only file (10 tests, ~0.41 s)
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_CustomSchedule.py

# Row 2: single real-codegen test (~6.78 s, ~3.8 s of which is the
#        session-scoped isa_infrastructure fixture probe)
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_ScheduleCapture.py::test_tf32_4x4_tn_capture_shape

# Row 3: full real-codegen file (74 passed + 1 skipped, ~21.21 s; fixture
#        amortised across all tests in the same session)
pytest $PYTEST_IGNORE Tensile/Tests/unit/test_ScheduleCapture.py
```

The `~3.8 s session-fixture penalty` is derived from
`(row 2 wall time) − (row 3 wall time / 74)` after subtracting per-test
collection overhead. The `5× per-test multiplier` is
`(row 3 marginal per-test ≈ 0.20 s) / (row 1 marginal per-test ≈ 0.04 s)`.
Both are rounded order-of-magnitude figures — exact values will vary with
host, kernel, and pytest cache state. Treat them as load-bearing for the
"real codegen is meaningfully slower than mocks but not catastrophically
so" qualitative claim, not as a precise benchmark.

**Per-test multiplier:** the session-scoped `isa_infrastructure` probe is
~3.8 s and is paid once per pytest session no matter how many real-codegen
tests run. Marginal cost per additional `generate_real_idmap` call is
roughly `(21.21 − 6.78) / 73 ≈ 0.20 s` once the fixture is warm. Compared
to a mock-based test at ~0.04 s amortised (10 tests in 0.41 s minus
~0.05 s collection overhead), real codegen is roughly **5× slower per
test, plus a one-time ~3.8 s session penalty.**

**However**, the cost question is academic: there are no replaceable mocks
to migrate. The recommendation is straight deletion of the mock chain, not
a swap to `generate_real_idmap`. The remaining `generate_real_idmap` users
already exist and pay the cost; deleting the dead mock path doesn't move
the wall-clock needle.

---

## Coverage delta for "load-bearing" classifications

Per the bead: for any helper kept as load-bearing, document why an
end-to-end test wouldn't catch the same regression.

- `_make_solution` and `generate_real_idmap` are kept as **infrastructure**,
  not as fixtures for negative tests. They aren't substituting for
  end-to-end coverage; they are the means by which several
  `test_ScheduleCapture.py` tests *are* end-to-end. There is no negative
  fixture in the file that "real kernels don't produce."
- The mock builders that *would* qualify under the bead's "load-bearing
  for an edge case" rubric (`_make_mock_lcc` for unusual loop-counter
  shapes; `_make_mock_swap` for ExpandPointerSwap edge cases) are not
  load-bearing today because **no test exercises them.** If a future test
  needs an edge-case fixture real kernels don't produce, it should be
  hand-built at the call site — not resurrected from the abandoned mock
  chain.

---

## Suggested follow-up beads (do not file)

These are listed for the user to file or skip; this bead does not file
them. Each is sized for one PR.

1. **Delete `CMSValidationTestBase` and `_make_context`.** Remove
   `cms_validation_base.py` outright (no subclasses exist) and remove the
   never-called `_make_context()` from `test_CustomSchedule.py`. Verify
   the unit-test suite still passes (modulo the
   `test_MatrixInstructionConversion.py` ignore). Estimated diff: ~300
   LoC deletion, no test rewrites.

2. **Delete the dead mock chain in `cms_test_utils.py`.** After (1) lands,
   delete `make_mock_id_map`, `make_mock_mfma_code`, `_make_mock_lr`,
   `_make_mock_gr`, `_make_mock_grinc`, `_make_mock_swap`,
   `_make_mock_lw_swap`, `_make_mock_lcc`, `_make_mock_packs_bf16`,
   `subset_id_map`, `_frozen_config_key`, and `kernel_to_solution_config`.
   Drop the now-orphaned `LRA_BASE` / `LRB_BASE` / `PACK_A_DST_BASE` /
   `PACK_B_DST_BASE` / `MFMA_ACC_BASE` constants. Trim the docstring at
   the top of the file (the "Register spaces are widely separated…"
   block) since it describes the deleted infrastructure. The file should
   then contain only `_make_solution` + `generate_real_idmap` and their
   imports. Estimated diff: ~350 LoC deletion.

3. **Optional: rename `cms_test_utils.py`.** With only real-codegen
   helpers remaining, the "cms_test_utils" name no longer accurately
   describes the file. A name like `real_kernel_idmap.py` or
   `kernel_writer_fixtures.py` would be more honest. Low priority.

Combined effect of (1) + (2): roughly **650 LoC removed** with zero test
behaviour change, on top of the ~250 LoC the 4ft cleanup already removed.

The "clean implementation" directive should apply to follow-up bead (2) in
particular: the deletion should produce a clean diff with no leftover
shims, deprecated wrappers, or commented-out code.

---

## Hard-rules compliance check

- **No deletions performed.** Audit-only output as required.
- **`grep` + call-graph analysis used throughout** — file's own docstrings
  treated as hints, not ground truth. The "BF16-only" claim in
  `_make_mock_packs_bf16` is one example: the actual VPermB32 shape it
  builds is dtype-agnostic.
- **Skipped/xfailed/never-collected tests treated as dead.** None found
  in this scope (no `@pytest.mark.skip` or `@pytest.mark.xfail` on any
  CMS-validator test). The "never-collected" cases here are
  `_make_context` (defined but never called) and `CMSValidationTestBase`
  subclasses (none exist).
- **TF32 4x4 case from 4ft (`test_tf32_4x4_tn_capture_shape`) noted as
  the canonical real-codegen template** and used for cost timing.
