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

Mutation score on covered mutants = 450 / (450+131) = **77.5%**. The 84 no-test
mutants are coverage gaps (a different problem from assertion strength).

Representative survivors (textbook missing-assertion-strength): `x_ceilDivide`
(many — e.g. `mutmut_1`: `numerator < 0 or denominator < 0` → `and`),
`x_choose_multiplier`, `x_isRhel8`, `x_wmmaV3InputVgprLayout`.

## Harness validation (the verifier's mechanism, proven)

`wf/mutmut-verify.sh` is manifest-driven + serial and materializes mutants via
`mutmut apply`. Proven on `Tensile.Common.Utilities.x_ceilDivide__mutmut_1`:
`mutmut show` → diff; `mutmut apply <id>` mutated the real source (visible via the
bind mount + host `git diff`); `git -C <src> checkout --` reverted clean. Apply →
run → revert path works.

## Status / next

- Phase 0 scaffold: **done** (`[tool.mutmut]` config, `tox mutation-unit`,
  `wf/mutmut-verify.sh`, `.gitignore`).
- Phase 1 pilot: **done** (population + score above).
- **Next = Phase 2–4 (the dynamic workflow):** triage the 131 survivors into the 5
  buckets, author add-only killing tests for the test-fixable ones, kill-proof them
  serially via `wf/mutmut-verify.sh`, then synthesize the report. (Optionally also
  address the 84 no-test mutants as a coverage-gap follow-up.)
