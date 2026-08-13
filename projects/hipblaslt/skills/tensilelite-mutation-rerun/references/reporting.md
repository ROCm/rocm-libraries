# Reporting and certification

Build each mutation report from evidence artifacts, not freehand prose. The raw
mutmut results, conserved survivor ledger, strict verifier matrix, source/config
provenance, and restoration receipt are the sources of truth.

## 1. Keep engine and triage accounting separate

### Initial engine results

Preserve mutmut's native statuses without silently reclassifying them:

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
sum(initial engine status counts) == total_mutants
```

Call the first field `mutmut_killed`, not `strict_killed`. Mutmut 3.6 can map
pytest internal-error status 3 to `killed`; the native label alone is not strict
kill evidence.

### Final triage dispositions

For every ID in the initial non-killed input ledger, assign exactly one final
disposition:

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

Require:

```text
sum(final disposition counts) == initial non-killed ledger rows
set(report IDs) == set(ledger IDs)
```

Do not merge timeout, collection, internal, interrupted, segfault, or unknown
results into killed or equivalent.

## 2. Required provenance

Record:

- report schema/version identifier;
- slice ID and immutable source SHA;
- source root and target modules;
- exact test selection and covering-set artifact;
- container name, image ID/digest, status, and mounted project path;
- Python, pytest, coverage, and mutmut versions;
- relevant mutmut configuration and max children;
- start/end UTC timestamps;
- clean baseline command and result;
- mutation commands and exit statuses; and
- final restoration/cleanliness result.

A report with missing source SHA, environment identity, baseline result, or
restoration evidence is **Inconclusive**, not certified.

## 3. Required artifacts

Link paths for:

- preflight environment record;
- reviewed slice record;
- covering-set measurement;
- raw `mutmut results` output;
- full `mutmut results --all true` output;
- survivor conservation ledger;
- strict kill matrix/report;
- tests and source pragmas added;
- equivalence/design-smell rationales; and
- final tracked-worktree status receipt.

Do not cite an uncommitted or missing `work/` path as durable evidence without
copying it into the agreed handoff location.

## 4. Scores

Scores are optional. Counts and unresolved outcomes are mandatory.

For every reported score include:

```text
name
numerator
denominator
formula
excluded categories and justification
value
```

Never:

- hide no-test, timeout, suspicious, or other inconclusive counts;
- call a score strict when its numerator is native `mutmut_killed` without
  return-code verification;
- compare scores with different denominators as if they were the same metric;
- let `# pragma: no mutate` silently remove mutants and inflate the score; or
- round counts before computing the value.

When pragmas change the mutant population, record pre-pragma and post-pragma
totals plus a pragma-free shadow metric using the original denominator.

## 5. Example report shape

The exact format may evolve, but preserve this separation and accounting:

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

## 6. Certification states

### Certified

Require all of:

- source/environment/config provenance complete;
- clean baseline passed;
- covering threshold passed;
- initial status accounting closes;
- survivor ledger conservation closes;
- final disposition accounting closes;
- claimed new kills have strict verifier evidence;
- equivalent rows have reviewed rationales;
- unresolved statuses remain visible; and
- configuration/source restoration is clean.

### Deferred

Use when the covering threshold, reviewed scope, or required test selection is
not ready. State the exact missing evidence.

### Inconclusive

Use for environment, collection, timeout, internal, interrupted, unknown,
missing artifact, or restoration failures that prevent a trustworthy result.

### Blocked

Use only when a named external dependency or user decision is required. Do not
use Blocked as a substitute for incomplete local work.

## 7. Final review checklist

Before handing off a report, verify:

1. Every count is a non-negative integer.
2. Both accounting equations close.
3. Ledger IDs are unique and conserved.
4. Artifact paths exist.
5. Source SHA and container identity are non-empty.
6. Any score can be recomputed from displayed fields.
7. Pragmas are visible in counts and rationale.
8. Certification state matches the evidence.
