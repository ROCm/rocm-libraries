# Planning one mutation slice

A slice is one bounded mutation run against a selected source file and test set.
Write the slice record before changing configuration. Follow its steps in order,
and restore every edited file after success, failure, or interruption.

Do not generate shell command strings or claim that an automated executor exists
until integration tests prove execution, restoration, and output creation.

## 1. Record the required inputs

Record at least:

```json
{
  "slice_id": "utilities",
  "source_sha": "<full git SHA>",
  "only_mutate": [
    "Tensile/Common/Utilities.py"
  ],
  "test_selection": [
    "Tensile/Tests/unit/characterization/CommonUtilities"
  ],
  "container": "tl-mut",
  "source_root": "projects/hipblaslt/tensilelite",
  "container_project": "/work/projects/hipblaslt/tensilelite",
  "max_children": 32,
  "coverage_threshold": 80.0,
  "artifact_dir": "work/mutation/slices/utilities"
}
```

Use a complete Git commit ID for `source_sha`, not a branch name. A commit ID
keeps the source version fixed even when a branch moves.

Select one module unless several modules implement one behavior that cannot be
tested separately. Explain why a slice needs more than one module.

## 2. Validate the inputs before mutation

Require every check before editing `pyproject.toml`:

1. `git rev-parse HEAD` equals `source_sha`.
2. Every `only_mutate` file exists under `source_root`.
3. Every selected test path exists and pytest can collect it.
4. The selected tests pass on unchanged source.
5. The covering-test check in
   [covering-set.md](covering-set.md) passes.
6. The named container is running and mounts this worktree at
   `container_project`.
7. Version-controlled product source has no existing edits.
8. `max_children` is an integer from 1 through 32.
9. `artifact_dir` belongs only to this slice and will not overwrite another
   run.

When a check fails, stop with **Deferred**, **Inconclusive**, or **Blocked**.
Do not continue by guessing a value.

## 3. Run the phases in order

### Phase 0 — Record the environment

- Record the source commit ID and branch.
- Record the container image, container ID, mutmut version, and current time.
- Record the selected modules and tests.
- Record whether version-controlled source is unchanged.
- Write this environment record before editing configuration.

### Phase 1 — Configure mutmut

- Back up `pyproject.toml` byte-for-byte.
- Set the reviewed `only_mutate` and test-selection arrays.
- Preserve unrelated mutmut settings.
- Keep `mutate_only_covered_lines = false` because the project file documents
  a rocisa problem with the other setting.
- Run the selected tests on unchanged source after configuration.

### Phase 2 — Run mutmut

- Run `mutmut run --max-children <reviewed value>` in the recorded container.
- Record the command, exit status, start and end times, ordinary results, and
  complete results.
- Do not run another process that applies a mutant or edits the same source.

### Phase 3 — Review results and add tests

- Follow [survivor-triage.md](survivor-triage.md).
- Follow [test-authoring.md](test-authoring.md) for rows marked `add-test`.
- Keep exactly one review row for every input mutant.

### Phase 4 — Verify decisions

- Verify every claimed kill independently.
- Require a concrete explanation for every claim that a mutant behaves
  identically for valid inputs.
- Keep timeouts, collection errors, internal errors, interruptions, and unknown
  results **Inconclusive**.

### Phase 5 — Restore files

- Restore the byte-identical `pyproject.toml` backup after every outcome.
- Confirm that `pyproject.toml` matches `HEAD` unless an approved
  configuration change is the intended deliverable.
- Confirm that no version-controlled product source contains a mutant or
  temporary marker.

### Phase 6 — Report and hand off

- Check that every input result is counted exactly once.
- Record the paths to the result files, review table, and verifier output.
- End with **Certified**, **Deferred**, **Inconclusive**, or **Blocked**, followed
  by the exact reason.

## 4. Restore files after failures

Restoration must behave like a Python `finally` block:

- When backup fails, do not edit configuration.
- When configuration fails, attempt restoration and stop.
- When unchanged-source tests fail, restore files and mark the run invalid.
- When mutmut fails, save its diagnostic output, restore files, and mark the run
  **Inconclusive**.
- When restoration fails, report that source or configuration is still
  changed. Never certify the run.
- Do not reduce a restoration failure to a warning.

After any failure, report both the original error and the restoration result.

## 5. Reusing records from past runs

Do not turn historical mutation counts into executable expectations unless the
past run record is committed, complete, and intentionally maintained as a
stable check.

A reusable past-run record needs:

- the complete source commit ID;
- the mutmut version and relevant configuration;
- the exact module and test selections;
- the original and reviewed result files;
- the container or environment identity;
- counts that account for every mutant; and
- links to decisions about identical behavior and excluded lines.

When a new real run proves an older trial inaccurate, replace or remove the old
trial. Do not keep tests that require known-wrong totals or refer to uncommitted
files under `work/`.

## 6. Keep future command execution safe

- Pass subprocess arguments as arrays. Do not use `eval` or build one shell
  string from unquoted user or operator values.
- Validate strings, paths, arrays, percentages, and worker counts before
  execution.
- Treat configuration as written by a trusted operator, but still reject
  malformed values.
- Do not add an `--execute` option that only aborts. A read-only plan is
  documentation. A real executor needs integration tests for every phase and
  restoration path.

## 7. Hand off the plan

Before execution, summarize:

```text
slice ID
source commit ID
target module or modules
selected tests
target-file coverage and required percentage
container, image, and mutmut version
maximum parallel workers
output directory
files that must be restored
```

Ask the user before changing product source, expanding the requested scope, or
performing an external operation that was not already authorized.
