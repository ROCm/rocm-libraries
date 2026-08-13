# Writing and certifying a mutation report

Build the report from saved command output and result files, not from memory or
summary prose. The original mutmut results, one-row-per-mutant review table,
verifier output, environment record, and restoration result are the sources of
truth.

## 1. Keep tool results separate from review decisions

### Results produced by mutmut

Keep mutmut's original status for every mutant:

```text
mutmut_killed
survived
no_tests
timeout
skipped
caught_by_type_check
suspicious
interrupted
segfault
not_checked
other
```

Require:

```text
sum(original mutmut status counts) == total_mutants
```

Call the first field `mutmut_killed`, not `strict_killed`. Mutmut 3.6 may
label pytest internal-error status 3 as `killed`. The tool's label alone does
not prove that a test detected changed behavior.

### Decisions after review

Assign exactly one final decision to every mutant in the original not-killed
input set:

```text
killed_by_new_tests
equivalent
survived
no_tests
timeout
skipped
caught_by_type_check
suspicious
interrupted
segfault
not_checked
design_smell
deferred
other_inconclusive
```

Require both checks:

```text
sum(final decision counts) == original not-killed review rows
set(report IDs) == set(review-table IDs)
```

Do not combine timeouts, collection errors, internal errors, interruptions,
segmentation faults, or unknown results with killed or equivalent results.

## 2. Record how the run was produced

Record:

- the report format and version;
- the slice ID and complete source commit ID;
- the source root and target modules;
- the exact test selection and covering-test record;
- the container name, image ID or digest, status, and mounted project path;
- the Python, pytest, coverage, and mutmut versions;
- the relevant mutmut settings and maximum parallel workers;
- start and end times in Coordinated Universal Time (UTC);
- the unchanged-source test command and result;
- mutation commands and exit statuses; and
- the final file-restoration result.

A missing source commit ID, environment identity, unchanged-source result, or
restoration result makes the report **Inconclusive**.

## 3. Link every evidence file

Include paths for:

- the environment record written before the run;
- the reviewed slice record;
- the covering-test measurement;
- the original `mutmut results` output;
- the complete `mutmut results --all true` output;
- the one-row-per-mutant review table;
- the verified-kill table or report;
- tests and `pragma: no mutate` lines added;
- explanations for identical behavior or production-design problems; and
- the final version-controlled file-status record.

A missing or uncommitted path under `work/` is not durable evidence. Copy it
to the agreed handoff location before citing it.

## 4. Report scores without hiding results

Scores are optional. Counts and unresolved results are required.

For every score, include:

```text
name
numerator
denominator
formula
excluded categories and reason
value
```

Never:

- hide no-test, timeout, suspicious, or other unresolved counts;
- call a score strict when it counts unverified `mutmut_killed` results;
- compare scores with different denominators as though they are the same
  measurement;
- let `# pragma: no mutate` remove mutants without recording the change; or
- round counts before calculating the value.

When a pragma changes the mutant population, record the totals before and after
the pragma. Also report a comparison value that uses the original denominator.

## 5. Example report

The file format may change, but it must preserve the separation and counting
rules above:

```json
{
  "schema_version": "tensilelite-mutation-report/1",
  "slice_id": "utilities",
  "source_sha": "0123456789abcdef0123456789abcdef01234567",
  "environment": {
    "container": "tl-mut",
    "container_image_id": "sha256:example",
    "mutmut_version": "3.6.0",
    "max_children": 32
  },
  "initial_results": {
    "total_mutants": 10,
    "mutmut_killed": 6,
    "survived": 2,
    "no_tests": 1,
    "timeout": 1,
    "skipped": 0,
    "caught_by_type_check": 0,
    "suspicious": 0,
    "interrupted": 0,
    "segfault": 0,
    "not_checked": 0,
    "other": 0
  },
  "triage": {
    "input_non_killed": 4,
    "killed_by_new_tests": 1,
    "equivalent": 1,
    "survived": 1,
    "no_tests": 0,
    "timeout": 1,
    "skipped": 0,
    "caught_by_type_check": 0,
    "suspicious": 0,
    "interrupted": 0,
    "segfault": 0,
    "not_checked": 0,
    "design_smell": 0,
    "deferred": 0,
    "other_inconclusive": 0
  },
  "changes": {
    "tests_added": 1,
    "pragmas_added": 0
  },
  "artifacts": {
    "environment": "env.json",
    "covering_set": "covering-set.json",
    "raw_results": "results.txt",
    "ledger": "survivor-ledger.json",
    "kill_matrix": "kill_matrix.tsv",
    "restoration": "restore.txt"
  },
  "certification": {
    "state": "Certified",
    "reason": "all accounting closes and restoration is clean"
  }
}
```

## 6. Choose the final state

### Certified

Require every item:

- source, environment, and mutmut settings are recorded;
- the unchanged-source tests passed;
- the selected tests met the target-file coverage requirement;
- original mutmut status counts equal the total mutant count;
- the review table contains every original not-killed mutant exactly once;
- final decision counts equal the number of review rows;
- every new kill has verifier evidence;
- every equivalent result has a reviewed explanation;
- unresolved results remain visible; and
- configuration and source files were restored.

### Deferred

Use **Deferred** when the required coverage, reviewed scope, or test selection is
not ready. State exactly what is missing.

### Inconclusive

Use **Inconclusive** when an environment, test collection, timeout, internal,
interruption, unknown, missing-file, or restoration failure prevents a
trustworthy result.

### Blocked

Use **Blocked** only when a named external dependency or user decision is
required. Do not use it for unfinished local work.

## 7. Review the report before handoff

Check every item:

1. Every count is a non-negative integer.
2. Both counting equations pass.
3. Every mutant ID is unique and appears exactly once where required.
4. Every evidence path exists.
5. The source commit ID and container identity are present.
6. Every score can be recalculated from displayed fields.
7. Pragmas are visible in the counts and explanation.
8. The final state matches the evidence.
