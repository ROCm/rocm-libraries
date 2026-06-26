# tensilelite quality + AI-friendliness check

A single report-only check (`check_quality.py`) that folds two measurements
together over the `Tensile/` Python tree (source only — `Tensile/Tests` and
`Tensile/CustomKernels` are excluded):

- **complexity / size** — function CCN and file NLOC violator counts (via `lizard`)
- **AI-friendliness** — 21 AST readability signals (via `llm_readability_report.py`)

It prints current values against the targets in `quality_targets.json` and
**always passes** for now. Nothing is enforced yet — the point at this stage is
visibility and agreeing on realistic targets, not blocking commits. Targets are
initialised to the current measurement ("hold the line"); tighten them
deliberately over time, and flip on enforcement when ready.

```bash
# print the report (exit 0)
python scripts/quality/check_quality.py
# set targets to the current measurement
python scripts/quality/check_quality.py --update
# opt-in: exit 1 if any metric is over target (for CI once targets are agreed)
python scripts/quality/check_quality.py --enforce
```

The scanner (`llm_readability_report.py`) is a stdlib-only AST walker;
`check_quality.py` additionally needs `lizard` for the CCN/NLOC counts.

Measured 2026-06-25 on the develop worktree: 136 source files, 3045 functions.

## Complexity / size (lizard)

tensilelite is large, mature code — max function CCN is **859** and 31 files
exceed 509 NLOC — so absolute pass/fail thresholds (rocMETRICS-style) don't fit.
These are tracked as violator counts. Only Python is measured (`lans=["python"]`),
so the C/C++ headers under `Tensile/Source` are excluded.

| Metric | Current | Threshold |
| ------ | ------- | --------- |
| functions CCN > 12 | 318 | cyclomatic complexity > 12 |
| functions CCN > 20 | 183 | cyclomatic complexity > 20 |
| files NLOC > 509 | 31 | non-blank/comment lines > 509 |
| files NLOC > 1000 | 13 | non-blank/comment lines > 1000 |

## AI-friendliness (21 AST signals)

Signal numbers follow the rocMETRICS llm-readability study.

| Metric | Current | Notes |
| ------ | ------- | ----- |
| files >= 1000 LOC | 22 | worst: `KernelWriterAssembly.py` 19,625 |
| files > ~5000 tokens | 45 | char/4 estimate |
| functions nesting depth >= 5 | 197 | worst: depth 34 |
| swallowed errors | 26 | `except: pass` / `return None/False/0` / bare `BaseException` |
| bloated `__init__.py` | 1 | over 50 LOC or 20 re-exports |
| impl-before-interface files | 92 | files leading with impl rather than type declarations |
| modules without a test | 56 | no `test_<module>.py` anywhere under `Tensile/Tests` |
| lateral cross-feature imports | 20 | `Common`/`Utilities` excluded as infra |
| cross-feature literal dispatch | 4 | same string `==`-compared across >=3 features |
| duplicated long literals | 38 | >=40-char string in >=2 files |
| shallow features (impl/iface ratio) | 2 | `Common`, `Toolchain` |
| shallow features (impl/symbol) | 0 | |
| parallel-impl module pairs | 0 | >=40% public-symbol overlap across features |
| typing.Any uses | 1 | |
| typing.cast uses | 0 | |
| `# type: ignore` comments | 4 | |
| generic filenames | 0 | `utils`/`helpers`/`common`/… |
| generic feature dirs | 0 | |
| tests importing private paths | 1 | `_private`/`internal`/`detail`, scanned under `Tensile/Tests` |
| adapter-seam violations | 0 | allowlist empty in v1 |

## Layout adaptations from rocMETRICS

- Source root is `Tensile/`; feature dirs are its subdirs. `Tests` and
  `CustomKernels` are in `SKIP_DIRS`.
- Tests live *inside* the tree at `Tensile/Tests` (not a sibling), so
  `_guess_tests_root` checks inside the src root first and the seam-test matcher
  matches `test_<stem>.py` by basename anywhere under the test root (tensilelite
  nests tests under `Tests/unit/<Area>/`).
- `Common` and `Utilities` are treated as shared infra (`INFRA_DIR_EXCLUDE`) so
  importing them is not counted as lateral coupling.
- `ADAPTER_ALLOWLIST` is empty — no adapter seams designated yet.

## Not in scope (deferred)

C++/assembly analysis, PMD token-duplication, the PR-delta comment workflow, and
the CI-only Tier-B signals (bespoke-helper O(n²), co-change locality, PR-diff).
tensilelite already runs flake8/black/isort, so those linters are not duplicated
here.
