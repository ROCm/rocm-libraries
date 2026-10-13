# Running the SLO linter in CI

Not wired into any CI yet. This page is the contract any CI can rely on, plus
copy-paste snippets for when we turn it on.

## The contract

- **Exit codes.** 0 = ran fine. 1 = something got worse, only with `--fail-on increase`.
  2 = the tool itself broke (bad args, a file that does not parse, git or coverage.py
  missing when asked).
- **Warn only by default.** `check` exits 0 even with warnings. Enforcement later is one
  flag, `--fail-on increase`, with no code change.
- **Inputs are files, not services.** No network, no GPU, no build. Python 3.10+ stdlib only,
  plus coverage.py for `--coverage`. Without `--coverage` the SLO2xx checks are skipped, and
  JUnit leaves them out.
- **Git is optional.** Without git, `check` compares every file to `baseline.json`. With
  `--base REF`, it judges only the files changed since `REF` and follows renames, so a
  moved file keeps its baseline.
- **Paths come from the tool's own location** (two folders up is the rocke root), so it runs
  from a source tree or an installed copy. `--root` overrides it.

## Outputs, pick any

| Flag | For |
|---|---|
| (none) | Plain text for people. |
| `--json FILE` | Every finding, machine readable. `-` writes to stdout. |
| `--junit FILE` | One test case per check id. CTest, Jenkins, GitLab and GitHub dashboards read it. |
| `--github` | `::warning file=...,line=...::` lines that show on the PR diff. Use `--path-prefix dnn-providers/hip-kernel-provider/rocke/` so paths match the repo. |
| `--markdown FILE` | Before/after table for a PR description or job summary. |

## Snippets

### CTest (TheRock)

TheRock runs tests from an installed copy, where there is no git. Two steps:

1. Install `tools/slo/` next to the other installed Python tests (it is not installed today).
2. Add a test entry next to the other Python ones in `platform/CMakeLists.txt`:

```cmake
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "add_test(rocke_slo_check python \"${ROCKE_TEST_CTEST_REL}tools/slo/slo.py\" check --junit rocke_slo.xml)\n")
string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
       "set_tests_properties(rocke_slo_check PROPERTIES LABELS \"ckc;python;quick\")\n")
```

Then add `rocke_slo_check` to the `quick` tier in
`ROCKE_ENGINE_test_categories_external.yaml`, or it drops out of `ctest -L <tier>`.

### GitHub Actions

```yaml
- name: rocKE SLO check (warn only)
  working-directory: dnn-providers/hip-kernel-provider/rocke
  run: |
    python tools/slo/slo.py check --base origin/develop \
      --github --path-prefix dnn-providers/hip-kernel-provider/rocke/ \
      --markdown "$GITHUB_STEP_SUMMARY"
```

`--base` needs the base branch fetched (`fetch-depth: 0`, or fetch `origin/develop`).
Add `--coverage` (and `pip install coverage`) to also check that the IR harness runs the
changed lines.

### Jenkins

```groovy
sh 'cd dnn-providers/hip-kernel-provider/rocke && python tools/slo/slo.py check --junit slo.xml'
junit allowEmptyResults: false, testResults: 'dnn-providers/hip-kernel-provider/rocke/slo.xml'
```

### Local and `tools/run_checks.py`

`library/tests/test_slo_ratchet.py` runs `check` against the baseline, so
`python tools/run_checks.py` (which auto-discovers `library/tests`) runs it too. It
reports worse numbers as pytest warnings and fails only if the tool cannot run.

The tool's own tests are separate: `pytest tools/slo/tests`.

## Updating the baseline

```
python tools/slo/slo.py baseline --update --coverage
```

Run it on the commit you want as the new starting point, and commit `baseline.json` in the
same PR. It warns when a number got looser than before, so a loosened limit is a visible
edit in review.
