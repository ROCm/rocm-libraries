# Mutation Production — Phase-0/Phase-1 pilot baseline (slice-1)

Worktree `tensilelite-mutation` (branch `users/davidd-amd/tensilelite-mutation`),
container `tl-mut` bound to this worktree. Engine **mutmut 3.6.0**. Validated
2026-06-07. Companion to `../../PLAN-MUTATION-PRODUCTION.md`.

## Environment (verified in-container)

- `tl-mut` = `tensilelite-char:repro`, mounted `-v <this worktree>:/work`.
- `rocisa` built once (`invoke rocisa`) + imports clean; `mutmut`, `pytest-cov` installed.
- mutmut config lives in `projects/hipblaslt/tensilelite/pyproject.toml [tool.mutmut]`.

## mutmut 3.6 reality (differs from the report's v2 framing)

- **Copy model:** mutmut copies `source_paths` into a `mutants/` dir and runs the
  selected tests there. So `source_paths = ["Tensile"]` (whole package, for imports)
  + `only_mutate = [<slice-1 files>]` (restricts what is mutated).
- Config keys: `source_paths`, `only_mutate`/`do_not_mutate` (globs),
  `pytest_add_cli_args_test_selection` (not `tests_dir`), `mutate_only_covered_lines`,
  `max_stack_depth` (default -1). `also_copy` must include `pytest.ini` (markers +
  config; not auto-copied; auto-copies only tests/, test/, setup.cfg, pyproject.toml).
- **Never set `PYTHONPATH` to the real tree** when invoking `mutmut run`: it shadows
  the `mutants/` copy and false-survives every mutant. Run with just
  `docker exec -w /work/projects/hipblaslt/tensilelite tl-mut mutmut run`.
- CLI: `mutmut run` (config-driven, only `--max-children`), `results`, `show <id>`,
  `apply <id>`, `browse`, `tests-for-mutant`.

## The rocisa × covered-lines hazard (root-caused)

`mutate_only_covered_lines = true` → mutmut runs an **in-process** coverage gather
(`code_coverage.gather_coverage`) whose `_unload_modules_not_in()` pops `rocisa`
from `sys.modules`. The later "Running stats" phase **re-imports rocisa in the same
process** → nanobind duplicate registration → **SIGABRT (exit 134, no Python
traceback)**. Per-mutant runs are fork children (one rocisa import each) → safe.

**Mitigation (current):** `mutate_only_covered_lines = false`. Trade-off: mutmut
also mutates uncovered lines → those show as `no tests` (🫥), which are coverage
gaps, not assertion gaps — triage filters them.

**To re-enable covered-lines later:** pre-import rocisa at interpreter startup
(e.g. a `sitecustomize.py` on the path) so the unload snapshot keeps rocisa loaded.

## Pilot run (slice-1, covered-lines OFF)

`mutmut run` exit 0, ~104 mutations/sec. **665 mutants** over the 5 slice-1 files
(generation: "5 files mutated, 372 ignored"):

| outcome | emoji | count |
|---------|-------|-------|
| killed | 🎉 | 450 |
| survived | 🙁 | 131 |
| no covering test | 🫥 | 84 |
| timeout | ⏰ | 0 |
| suspicious | 🤔 | 0 |

Counts cross-checked: `mutmut results` lists exactly **84 "no tests" + 131
"survived"** (textual labels); killed = 665 − 215 = **450** (matches the
`665/665 🎉450` progress line; 450+84+131 = 665). Scope verified: every mutant is
in one of the 5 slice-1 files (no leak).

Mutation score, reported two ways (no cherry-picking):
- **raw** = 450 / 665 = **67.7%** (counts the 84 no-test mutants as not-killed).
- **on covered mutants** = 450 / (450+131) = **77.5%** (excludes the 84 no-test;
  those are coverage gaps — a different problem from assertion strength).

Baseline is green independently of mutmut: the two slice-1 char dirs run
`110 passed, 70 snapshots, 0 failed` (direct pytest in `tl-mut`).

Survivors span `x_ceilDivide` (many — e.g. `mutmut_1`: `numerator < 0 or
denominator < 0` → `and`), `x_choose_multiplier`, `x_isRhel8`,
`x_wmmaV3InputVgprLayout`. These are **not yet triaged** — bucket assignment
(missing-assertion-strength / wrong-granularity / equivalent / unhelpful /
design-smell) is Phase 2 work; `mutmut_1` merely *looks like* a weak-assertion
case pending verification.

## Harness validation (the script run end-to-end)

`wf/mutmut-verify.sh` is manifest-driven + serial and materializes mutants via
`mutmut apply`. It was executed end-to-end on a 2-row manifest (artifacts in
`verify-selftest/kill_matrix.tsv`):

| mutant | role | base_rc | mut_rc | revert | verdict |
|--------|------|---------|--------|--------|---------|
| `x_ceilDivide__mutmut_10` | killed mutant | 0 | 1 | ok | **KILLED** |
| `x_ceilDivide__mutmut_1` | survivor (negative control, same covering test) | 0 | 0 | ok | **BAD** (not-killed) |

So the script correctly proves a kill (test passes clean, fails mutated, source
reverts) AND does **not** false-claim a kill for a survivor; trap-revert left the
tree clean; it exits non-zero when any row is not a kill. (Earlier I had only
validated the `mutmut show`/`apply`/`git checkout` mechanism by hand on
`mutmut_1`; this is the full-script proof.)

## Status / next

- Phase 0 scaffold: **done** (`[tool.mutmut]` config, `tox mutation-unit`,
  `wf/mutmut-verify.sh`, `.gitignore`).
- Phase 1 pilot: **done** (population + score above).
- **Next = Phase 2–4 (the dynamic workflow):** triage the 131 survivors into the 5
  buckets, author add-only killing tests for the test-fixable ones, kill-proof them
  serially via `wf/mutmut-verify.sh`, then synthesize the report. (Optionally also
  address the 84 no-test mutants as a coverage-gap follow-up.)
