# Reviewing mutation results

Use this procedure after a mutation run produces a result that was not killed.
A mutant is one temporary source change. It is killed when tests detect the
changed behavior and survives when the tests still pass. Other results, such as
timeouts and tool errors, need separate review.

Use the native mutmut command output as input. Do not create another file format
only to divide the review work.

## 1. Save every result before grouping work

Save the results that were not killed:

```bash
mutmut results > work/mutation/<slice>/results.txt
```

Save the complete result list separately:

```bash
mutmut results --all true > work/mutation/<slice>/results-all.txt
```

Read the output conservatively. Each non-empty result line contains a mutant ID
and a status:

```text
    Tensile.Common.Utilities.x__mutmut_1: survived
```

Record every ID exactly once before grouping work. Keep an unknown status as
**Inconclusive** instead of dropping it.

Interpret the common statuses as follows:

- `survived`: the tests passed while the mutant was active.
- `no tests`: mutmut did not find a test for the mutant.
- `timeout`: the command did not finish before the configured limit.
- `skipped`, `caught by type check`, or `not checked`: keep the original
  status and verify its reason before making a final decision.
- `check was interrupted by user`, `segfault`, `suspicious`, or another
  tool or environment status: **Inconclusive**.
- any unknown status: **Inconclusive** until it is understood.

Mutmut 3.6 labels some pytest or process errors as `killed`. For example,
pytest internal-error status 3 may receive that label. Do not treat the mutmut
label alone as proof. Verify a claimed kill with the checks in section 5.

## 2. Inspect each source change

Run both commands for every recorded ID:

```bash
mutmut show <mutant-id>
mutmut tests-for-mutant <mutant-id>
```

Record the source file, enclosing function or method, changed expression, and
tests selected by mutmut. A mutant name often contains a function name, but
verify it against the displayed change and current source. The numeric suffix
can change after the source file changes.

Grouping by function is useful for assigning work. It must not merge or remove
rows for similar-looking mutants.

## 3. Keep one review row per mutant

Create a table with at least these fields:

| Field | Meaning |
|---|---|
| `mutant_id` | Exact mutmut ID from the saved result |
| `initial_status` | Status before review |
| `source_file` | Path relative to the repository |
| `function` | Verified enclosing function or method, or `<module>` |
| `change` | Short description of the temporary source change |
| `disposition` | `add-test`, `equivalent`, `design-smell`, `inconclusive`, or `defer` |
| `test_node` | Exact pytest test name for an added test, when applicable |
| `clean_rc` | Pytest return code on unchanged source |
| `mutant_rc` | Pytest return code with the mutant active |
| `revert` | `clean` or a description of source that was not restored |
| `evidence` | Detection proof or explanation of identical behavior |

Before reporting results, require both checks:

```text
set(input mutant IDs) == set(review-table mutant IDs)
len(input IDs) == len(review-table rows)
```

Reject duplicate IDs, missing rows, and extra rows. Instructions in a prompt or
summary text are not enough to enforce this check.

## 4. Choose the next action

### Add a test

Choose an input that makes the original and changed code behave differently.
Before writing the assertion, answer:

> What specific source change makes this assertion fail?

Run the exact test on unchanged source first. Then rerun one named mutant with
one worker:

```bash
pytest -p no:cacheprovider -m unit -q <test-file>::<test-name>
mutmut run <mutant-id> --max-children 1
```

When a verifier manifest exists, use `mutmut-verify.sh` to record the
unchanged-source result, changed-source result, and restoration result.

### Explain identical behavior

Require a concrete explanation for all valid inputs. The explanation must show
why the changed expression cannot alter an observable return value, exception,
state, or required output. “No test found” and “hard to exercise” do not prove
identical behavior.

### Record a design problem

Use `design-smell` only when detecting the change would require a production
design change rather than a better test input. Link follow-up work. Do not
silently call the mutant equivalent.

### Keep the result inconclusive

Collection errors, command-usage errors, timeouts, interruptions, source that
was not restored, and unknown statuses remain **Inconclusive**. Correct the
environment and rerun. Never count these results as kills.

## 5. Verify every claimed kill

Require all three results:

1. The exact test passes on unchanged source.
2. The same test exits with pytest assertion status `1` while the mutant is
   active.
3. The source file is restored byte-for-byte.

Return code `0` means the mutant survived. Collection, usage, internal,
interruption, and timeout return codes are **Inconclusive**.

## 6. Report every input result

Report counts for the complete saved input set:

```text
total results not killed
killed by added tests
identical behavior with explanation
no-test remaining
timeout remaining
other inconclusive
design problem or deferred
```

The category counts must add up to the input total. Include the review table and
verifier output paths in the handoff.
