# Selecting tests for one module

A covering set is the reviewed, bounded list of tests used for one target
Python module. Its line-coverage measurement is a scheduling heuristic: it
shows which source lines the selected tests execute, but it does not show that
the tests assert the behavior of those lines or will detect every mutation.
Only a per-mutant run can establish whether the selected tests detect a given
source change.

## 1. Find candidate tests

Start with evidence that another developer can review:

1. Search recursively below `Tensile/Tests/unit/` for
   `test_<Module>.py`. Do not assume the file is directly below `unit/`; for
   example, the direct tests for `Tensile/Common/Utilities.py` are in
   `Tensile/Tests/unit/Common/test_Utilities.py`.
2. Search unit tests and tests that record current behavior for imports or
   references to the target module.
3. Include existing test directories whose fixtures or public API calls reach
   the target indirectly, even when they do not import it.
4. Reuse a previously committed test selection when it still applies. Explain
   every addition or removal.

Keep the set inside the `-m unit` suite. The YAML-driven directories under
`Tensile/Tests/common` execute the same production code, but each config
compiles kernels and benchmarks them on real hardware, and a mutation run
repeats the whole set once per mutant. Including them makes a run
hardware-gated and turns minutes into hours. Accept the consequence: a kill
recorded here means the unit suite detected the change, so a mutant that only
the end-to-end suite would catch is reported as a survivor. Pull a `common`
directory in under rule 3 only when a specific target needs it, and record why
the rerun cost is justified.

Text search can miss aliases, re-exports, multiline imports, fixtures, and
imports performed while the program runs. An `rg` or grep result is not a
complete test selection. Read the selected tests and the target module
together.

## 2. Run only the reviewed tests

Run pytest in the Linux or Windows Subsystem for Linux (WSL) mutation
environment. List every selected path explicitly. For example:

```bash
PROJ=/work/projects/hipblaslt/tensilelite
OUT=work/mutation/covering/utilities
mkdir -p "$OUT"

set +e
docker exec -e PYTHONPATH="$PROJ" -w "$PROJ" tl-mut \
  pytest -p no:cacheprovider -m unit --cov=Tensile/Common \
  --cov-report=term-missing --cov-fail-under=0 \
  Tensile/Tests/unit/Common/test_Utilities.py \
  Tensile/Tests/unit/characterization/CommonUtilities \
  >"$OUT/coverage.log" 2>&1
rc=$?
set -e
```

Change the `--cov` path and test list for the target. Use a package path, not a
dotted Python module, as the coverage source. Earlier runs observed rocisa
unload/re-import failures with a dotted module.

The command sets `--cov-fail-under=0` because the repository-wide percentage
does not apply to this focused run. The next step checks the scheduling
threshold against the exact target-file row instead.

## 3. Decide whether the slice is ready

Require all four scheduling conditions before starting the mutation run:

1. Pytest exits with status `0`.
2. The coverage report contains the exact target file, such as
   `Tensile/Common/Utilities.py`.
3. That file's executed-line percentage meets the reviewed scheduling
   threshold. Use 80% unless the slice records another value.
4. The selected list contains at least one explicit test path and is not an
   accidental full-suite fallback.

Do not use the package `TOTAL` row when the target file is missing. A missing
target row means **Deferred**, even when the total package percentage is high.
Meeting these conditions makes the slice ready to run; it does not certify the
selected tests as behaviorally complete.

## 4. Save the decision

Write `covering-set.json`, or an equivalent record, in the slice output
directory:

```json
{
  "module": "Tensile/Common/Utilities.py",
  "source_sha": "<commit>",
  "selected": [
    "Tensile/Tests/unit/Common/test_Utilities.py",
    "Tensile/Tests/unit/characterization/CommonUtilities"
  ],
  "command": "<exact command>",
  "exit_code": 0,
  "target_row": "<verbatim coverage row>",
  "coverage_percent": 100.0,
  "threshold": 80.0,
  "status": "ready",
  "reason": "exact target row meets the scheduling threshold"
}
```

Use `status: "defer"` when pytest fails, the target row is missing, the
selection is empty, or coverage is below the threshold. Record the actual
reason. Do not replace missing data with zero or with the package total.

`status: "ready"` authorizes a bounded mutation run only. Do not cite it or
the line-coverage percentage as evidence that the selected tests detect a
mutation. Record that evidence per mutant from the unchanged-source and
mutated-source test outcomes.

## 5. Measure again when inputs change

Repeat this validation when:

- the target module changes in a way that may affect behavior;
- a selected test or fixture moves or changes;
- the base branch changes relevant imports or validation;
- the Python, coverage, or pytest environment changes; or
- a mutation run unexpectedly reports `no tests`.

The committed selection records which tests were chosen. A new measurement
shows whether they still meet the scheduling threshold for the target file.
