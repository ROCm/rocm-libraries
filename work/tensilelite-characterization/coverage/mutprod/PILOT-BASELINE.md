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

## Campaign result (Phase 2-4, dynamic workflow — DONE)

The dynamic workflow `wf/triage-workflow.js` (27 parallel per-function triage
agents → single serial kill-proof → bounded repair → serial pragma apply →
synthesis) triaged all 131 survivors and the result was certified by a fresh
full `mutmut` re-run + manual equivalence audit (not agent self-report):

| | before | after |
|---|--------|-------|
| total mutants | 665 | 654 (−11: 3 pragma'd lines) |
| killed | 450 | **566** |
| **survived** | **131** | **4** |
| no-tests | 84 | 84 |

- **131 survivors → 4.** Disposition: **118 killed** by new add-only tests
  (26 files / 74 test fns, each proven `base_rc=0 mut_rc=1` by `wf/mutmut-verify.sh`),
  **9 removed** by 3 `# pragma: no mutate` markers (I/O-noise lines), **4 remain
  equivalent**.
- All **4 remaining survivors independently verified genuinely equivalent**
  (regex under `re.IGNORECASE`; `-1→+1` default where `1` is degenerate-valid;
  unused `value` param; `open` default mode). ⇒ **100% of non-equivalent covered
  mutants killed**; covered score **77.5% → 99.3%** (566/570), raw 67.7% → 86.5%.
- Source changed **only** by 3 pragma comments (no code edits). Slice suite green:
  **184 passed / 70 snapshots / 0 failed**.
- Artifacts: `workflow/{mutation-report.json, survivor-ledger.md, recommendations.md}`
  (synthesis summary tables were stale on first write — corrected against ground truth).

## Value of mutation testing (two perspectives)

**(1) It measures test QUALITY, not quantity.** Coverage only proves a line ran;
it cannot tell whether any test would notice the line's behavior changing. Slice-1
began at ~80% coverage yet **131 covered mutants survived** — each a line executed
but never pinned. Mutation testing localized every weakness to an exact line *and*
the distinguishing input (e.g. `ceilDivide`'s `numerator < 0 or denominator < 0`
passed even flipped to `and`; `hash_combine`'s `shift` kwarg unpinned). Result:
118 "executed-but-unchecked" lines became behavior-pinning assertions (suite
110 → 184 tests; covered score 77.5% → 99.3%), with the genuinely-untestable
(4 equivalent), no-contract noise (9 pragma'd), and refactor candidates (design
smells) cleanly separated. Far more actionable than a coverage %.

**(2) It makes LLM-generated tests TRUSTWORTHY and blocks fraudulent claims.** An
LLM can write an always-true test, assert the wrong thing, or just *claim* a kill.
Mutation testing replaces trust with a deterministic gate: a kill counts only if
`base_rc==0 and mut_rc!=0 and revert=='ok'`, checked by `wf/mutmut-verify.sh` —
never the model's say-so. What the gate + discipline actually caught here (none
taken on faith):

- **9 of 118** LLM-authored tests **failed the gate on the first pass** → rejected
  as not-real-kills, repaired, re-proven (verify → verify2 kill-matrices).
- the model's **4 equivalence claims were independently audited** by a human/Claude
  pass (the report's rule: never let the model decide equivalence) — one looked
  killable until full path analysis confirmed it equivalent.
- an **inflated synthesis summary** (agent counts summed 133 ≠ 131; over-stated
  kill count) was **caught by a validating step and corrected** to ground truth.
- the final result was **certified by an independent fresh `mutmut` re-run**,
  immune to any agent self-report.

LLMs propose; the deterministic harness disposes. That loop is what lets AI test
generation scale without scaling unverified claims. (The tutorial notebook
`parametric-chaos-notebook/characterization-testing.ipynb` §7 makes the gate
runnable — it rejects a fabricated kill live.)

## Out of scope / next

- **84 no-test mutants** = coverage gaps (lines no test exercises), a different
  problem from assertion strength — deferred as a coverage follow-up.
- Next slices (smallest-first): `Common/ValidParameters.py`, `CustomYamlLoader.py`,
  `BenchmarkSplitter.py`, `Configuration.py`, `LibraryIO.py`, then SolutionStructs.
- CI: pilot stays report-only; set a per-slice floor at CI phase 3 once stable.
