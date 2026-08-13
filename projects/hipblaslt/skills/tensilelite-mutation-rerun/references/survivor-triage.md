# Survivor triage and conservation

Use this procedure after a slice run produces any non-killed result. Mutmut's
native CLI output is the source input; do not invent or require an intermediate
JSON format merely to group work.

## 1. Capture the input set

Save the complete non-killed output before triage:

```bash
mutmut results > work/mutation/<slice>/results.txt
```

Also retain full accounting separately:

```bash
mutmut results --all true > work/mutation/<slice>/results-all.txt
```

Parse the output conservatively. Each non-empty result line has a mutant ID and
status, for example:

```text
    Tensile.Common.Utilities.x__mutmut_1: survived
```

Record every ID exactly once before grouping. Preserve unknown statuses as
inconclusive rather than dropping them.

Use these disposition classes:

- `survived`: tests passed with the mutant active.
- `no tests`: mutmut did not identify a relevant test.
- `timeout`: execution did not finish within the configured limit.
- `skipped`, `caught by type check`, or `not checked`: preserve separately and
  verify the reason before assigning a disposition.
- `check was interrupted by user`, `segfault`, `suspicious`, or another
  infrastructure status: inconclusive; do not count as killed.
- unknown status: inconclusive until understood.

Mutmut 3.6 maps some pytest/process return codes to `killed`, including pytest
internal-error status 3. Do not accept the native label as final kill proof.
Re-verify claimed kills with the strict criteria in step 5.

## 2. Inspect source evidence

For every captured ID:

```bash
mutmut show <mutant-id>
mutmut tests-for-mutant <mutant-id>
```

Record the source file, enclosing function or method, changed expression, and
mutmut-selected tests. Mutant names usually contain function context, but verify
that context against the diff and current source. Ordinal suffixes are not
stable identities across source edits.

Grouping by source function is only a work-allocation convenience. Never merge
or discard ledger rows because two mutants look similar.

## 3. Maintain a conservation ledger

Use one row per input mutant with at least these fields:

| Field | Meaning |
|---|---|
| `mutant_id` | Exact mutmut ID from the captured run |
| `initial_status` | Status before triage |
| `source_file` | Source-relative path |
| `function` | Verified enclosing function/method or `<module>` |
| `change` | Concise description of the mutation |
| `disposition` | `add-test`, `equivalent`, `design-smell`, `inconclusive`, or `defer` |
| `test_node` | Exact pytest node for an added test, when applicable |
| `clean_rc` | Clean-source pytest return code |
| `mutant_rc` | Mutated-source pytest return code |
| `revert` | `clean` or a description of leaked source changes |
| `evidence` | Kill proof or equivalence rationale |

Before reporting results, require:

```text
set(input mutant IDs) == set(ledger mutant IDs)
len(input IDs) == len(ledger rows)
```

Reject duplicate IDs, missing rows, and unknown extra rows. Do not rely on an
agent prompt or report prose to enforce conservation.

## 4. Decide the next action

### Add a test

Choose an input that distinguishes original and mutant behavior. State the
mutation question explicitly: *what source change makes this assertion fail?*

Run the exact node on clean source first, then rerun the named mutant with one
child:

```bash
pytest -p no:cacheprovider -m unit -q <test-file>::<test-name>
mutmut run <mutant-id> --max-children 1
```

When a verifier manifest exists, use `mutmut-verify.sh` to capture clean,
mutated, and restoration evidence.

### Prove equivalence

Require a concrete proof over valid inputs. Acceptable evidence explains why the
changed expression cannot alter an observable return value, exception, state,
or required output. “No test found” and “hard to exercise” are not equivalence
proofs.

### Mark a design smell

Use this only when a behavior-distinguishing test would require changing the
production design rather than choosing a better input. Link follow-up tracking;
do not silently convert the mutant to equivalent.

### Keep inconclusive

Collection errors, usage errors, timeouts, interrupts, source leaks, and unknown
statuses remain inconclusive. Fix the environment and rerun; never count them as
kills.

## 5. Verify strict outcomes

A claimed kill requires all of:

1. The exact test node passes on clean source.
2. The same node returns pytest assertion status `1` with the mutant active.
3. Mutated source is restored byte-for-byte.

Return code `0` means survived. Return codes for collection, usage, internal
errors, interrupts, and timeouts are inconclusive.

## 6. Report without loss

Report counts for the complete input set:

```text
total non-killed input
killed by added tests
equivalent with rationale
no-test remaining
timeout remaining
other inconclusive
design-smell/deferred
```

The category counts must sum to the input total. Include ledger and verifier
artifact paths in the handoff.
