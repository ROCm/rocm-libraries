# Planning a mutation slice

Define a slice as reviewed data and an ordered fail-closed procedure. Do not
generate shell command strings or advertise an executor until a real integration
test proves execution, rollback, and artifact creation.

## 1. Required slice record

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

Use a full immutable SHA, not a branch name, for `source_sha`. Keep
`only_mutate` to one module unless multiple modules form one inseparable
behavior contract; explain any multi-module slice.

## 2. Validate before mutation

Require all of these before editing `pyproject.toml`:

1. `git rev-parse HEAD` equals `source_sha`.
2. Every `only_mutate` file exists under `source_root`.
3. Every selected test path exists and collects successfully.
4. The clean selected tests pass.
5. Covering-set validation passes using
   [covering-set.md](covering-set.md).
6. The container exists, is running, and mounts the intended worktree at
   `container_project`.
7. Tracked product source is clean.
8. `max_children` is an integer in the reviewed range `1..32`.
9. `artifact_dir` is slice-specific and does not overwrite another run.

If any condition fails, stop with **Deferred**, **Inconclusive**, or **Blocked**;
do not continue with guessed defaults.

## 3. Ordered phases

Execute these phases in order:

### Phase 0 — Preflight

- Capture source SHA/branch, container image and ID, mutmut version, timestamp,
  selected modules/tests, and tracked-source cleanliness.
- Write the environment artifact before mutating configuration.

### Phase 1 — Configure

- Back up `pyproject.toml` byte-for-byte.
- Set the reviewed `only_mutate` and test-selection arrays.
- Preserve unrelated mutmut settings, especially the rocisa-compatible
  `mutate_only_covered_lines = false` value.
- Re-run the clean selected tests after configuration.

### Phase 2 — Execute

- Run `mutmut run --max-children <reviewed value>` in the pinned container.
- Capture command, exit status, start/end time, raw results, and full results.
- Do not start another mutation/source-edit actor concurrently.

### Phase 3 — Triage and author tests

- Follow [survivor-triage.md](survivor-triage.md).
- Follow [test-authoring.md](test-authoring.md) for `add-test` rows.
- Maintain exact input/ledger conservation throughout.

### Phase 4 — Verify

- Verify every claimed kill independently.
- Audit equivalent rows with concrete valid-input reasoning.
- Keep timeout, collection, internal, interrupted, and unknown outcomes
  inconclusive.

### Phase 5 — Restore

- Restore the byte-exact pyproject backup in success, failure, or interruption.
- Confirm `pyproject.toml == HEAD` unless an explicitly reviewed configuration
  change is itself the deliverable.
- Confirm no tracked product source contains an applied mutant or temporary
  fence.

### Phase 6 — Certify and hand off

- Validate report accounting.
- Record result/ledger/verifier artifact paths.
- State Certified, Deferred, Inconclusive, or Blocked with the exact reason.

## 4. Failure behavior

Restoration is mandatory and must behave like `finally`:

- If backup fails, do not edit configuration.
- If configuration fails, attempt restoration and stop.
- If baseline tests fail, restore and mark the run invalid.
- If mutmut fails, capture diagnostics, restore, and mark inconclusive.
- If restoration fails, report a blocking source/configuration leak; never
  certify the run.
- A warning is not sufficient for restoration failure.

After any failure, report both the primary error and restoration status.

## 5. Historical receipts

Do not encode historical mutation counts as executable expectations unless the
receipt is committed, provenance-complete, and intended as a stable gate.

Every reusable receipt needs:

- immutable source SHA;
- mutmut version and complete relevant configuration;
- exact module and test selections;
- raw and normalized result artifacts;
- environment/container identity;
- accounting that closes; and
- links to equivalence and pragma decisions.

When a newer real run proves an older pilot inaccurate, supersede or remove the
pilot. Do not keep tests asserting known-inaccurate totals or references to
uncommitted `work/` files.

## 6. Command safety

- Pass subprocess arguments as arrays when implementing automation; do not use
  `eval` or concatenate unquoted operator data into shell strings.
- Validate strings, paths, arrays, numeric thresholds, and worker counts before
  execution.
- Treat configuration as operator-authored but still reject malformed values.
- Do not add an `--execute` mode that deliberately aborts. A read-only plan is
  documentation; a real executor must have integration tests for every phase
  and rollback path.

## 7. Planning handoff

Before execution, summarize:

```text
slice ID
source SHA
target module(s)
selected tests
target coverage and threshold
container/image/mutmut version
max children
artifact directory
expected restore target
```

Ask for user confirmation when the slice changes product source, broadens scope,
or requires an external operation not already authorized.
