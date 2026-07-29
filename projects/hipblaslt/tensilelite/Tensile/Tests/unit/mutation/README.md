# Mutation-testing tooling

Reproducibility harness for the mutmut-based mutation-hardening series
(AIHPBLAS-3868). Co-located with the characterization suite it exercises
(`../characterization/`). Test-support tooling only; no product source.

Run everything from the repository root. Scratch output lands under
`work/mutation/` (git-ignored working area, safe to delete).

## Scripts

- `covering-set-discover.sh` - build the real covering test set for a target
  module by resolving its importers, not just the char-dir tests.
- `pyproject-mutmut.sh` - stamp a per-slice mutmut config (keeps
  `mutate_only_covered_lines = false`; `true` hides survivors).
- `slice-preflight.sh` - sanity-check a slice config before a run.
- `mutmut-slice.js` - plan/split a module into runnable mutation slices.
- `solution-subslice.py` - window the Solution.py giants by covered-line
  regions (the partition mechanism that replaces the broken pragmas).
- `mutmut-verify.sh` / `mutant-identity.py` - per-mutant trampoline verify
  (`MUTANT_UNDER_TEST`); a clean-source PASS does not prove a kill.
- `mutmut-results-adapter.py` - normalize mutmut output.
- `rank-modules.py` / `rank-refresh.sh` - rank modules by survivor pressure
  and maintain the ranking history.
- `triage-workflow.js` - drive survivor triage.
- `ci-mutant-regression.py` + `validate-report.js` + `schema/` - CI
  regression gate and report schema.

## Selftests

`tests/*selftest*` plus `tests/*dryrun*` exercise the scripts against the
committed `tests/fixtures/`. Run the full set:

```sh
for f in tests/*selftest*.sh tests/verify-selftest-strict.sh; do bash "$f"; done
for f in tests/*selftest*.py;  do python3 "$f"; done
for f in tests/*selftest*.mjs tests/*dryrun*.mjs; do node "$f"; done
```

All must pass from a clean checkout.
