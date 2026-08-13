# Covering-set validation

Use this procedure before mutating a TensileLite Python module. Candidate
discovery is a review aid, not a correctness proof; measured coverage is the
gate.

## 1. Assemble candidates

Start with explicit, readable evidence:

1. Include `Tensile/Tests/unit/test_<Module>.py` when present.
2. Search unit and characterization tests for direct imports and references to
   the target module.
3. Include existing characterization directories whose fixtures or public API
   calls exercise the target indirectly, even when they do not import it.
4. Use prior committed slice selections when available; explain any additions
   or removals.

Text search can miss aliases, re-exports, multiline imports, fixtures, and
runtime imports. Do not treat an `rg` or grep result as complete. Review the
selected tests and the target module together.

## 2. Run only the reviewed set

Run pytest in the Linux/WSL mutation environment with an explicit path list.
For example:

```bash
PROJ=/work/projects/hipblaslt/tensilelite
OUT=work/mutation/covering/utilities
mkdir -p "$OUT"

set +e
docker exec \
  -e PYTHONPATH="$PROJ" \
  -w "$PROJ" \
  tl-mut \
  pytest -p no:cacheprovider -m unit \
    --cov=Tensile/Common \
    --cov-report=term-missing \
    --cov-fail-under=0 \
    Tensile/Tests/unit/characterization/CommonUtilities \
    >"$OUT/coverage.log" 2>&1
rc=$?
set -e
```

Adjust the `--cov` path and test list for the target. Do not use a dotted Python
module as the coverage source in this repository; prior runs observed rocisa
unload/re-import failures with that form.

The command disables the repository's aggregate coverage threshold because this
measurement intentionally selects a package directory but gates one exact file.
Do not treat `--cov-fail-under=0` as a waiver: step 3 below must independently
require the target row to meet the slice threshold.

## 3. Fail closed

All conditions are mandatory:

1. `rc == 0`.
2. The coverage report contains the exact target file row, for example
   `Tensile/Common/Utilities.py`.
3. The target row's coverage is at or above the reviewed threshold (80% unless
   the slice records another value).
4. The selected list is non-empty and contains no accidental bare full-suite
   fallback.

Never use the aggregate `TOTAL` row when the target row is missing. A missing
row means **defer**, regardless of total package coverage.

## 4. Record the decision

Write a slice-local `covering-set.json` or equivalent reviewed artifact with:

```json
{
  "module": "Tensile/Common/Utilities.py",
  "source_sha": "<commit>",
  "selected": [
    "Tensile/Tests/unit/characterization/CommonUtilities"
  ],
  "command": "<exact command>",
  "exit_code": 0,
  "target_row": "<verbatim coverage row>",
  "coverage_percent": 100.0,
  "threshold": 80.0,
  "status": "ok",
  "reason": "exact target row meets threshold"
}
```

Use `status: "defer"` for a failed run, missing row, empty selection, or coverage
below threshold. Include the actual reason; do not coerce missing data to zero
or package total.

## 5. Revalidate when inputs move

Re-run covering-set validation when:

- the target module changes materially;
- tests or fixtures in the selected set move or change;
- the base branch changes relevant imports or validators;
- the Python/coverage/pytest environment changes; or
- a prior mutation run reports unexpected `no tests` results.

The committed selection is the reproducibility artifact; the fresh measurement
is evidence that it remains adequate.
