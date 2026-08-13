# Writing tests that detect a mutant

Use this procedure when review shows that a test suite missed a temporary source
change. The goal is one small, maintainable assertion that distinguishes the
current behavior from that mutant. The goal is not a larger test count or a
higher coverage percentage.

## 1. Prepare one behavior group

Group mutants only after the review table contains every input mutant. For one
function or closely related behavior:

1. Read the current source and its callers.
2. Run `mutmut show <id>` for every mutant in the group.
3. Read nearby tests and fixtures. Follow their imports, pytest markers,
   repeated-input style, cleanup, and saved expected results.
4. Decide which mutants can share one input and assertion. Keep separate cases
   when they check different behavior.

A group name is only a way to assign work. Keep one review row for every
mutant.

## 2. Choose where the test belongs

Prefer these locations in order:

1. Strengthen an existing focused test when the new assertion checks the same
   behavior.
2. Add another input to an existing focused test.
3. Add one focused test file for one related function or behavior group.

Do not create one file per mutant. Follow repository pytest naming so automated
checks discover the test. Python test files use `test_*.py`. Add
`@pytest.mark.unit` when the surrounding test suite requires it.

## 3. State what source change the test detects

Before writing an assertion, answer:

> What specific source change makes this assertion fail?

Reject an assertion that only proves a function exists, a mock was called, a
value is not null, or the function did not crash. Do not mock the behavior being
tested. Explain any numeric value, input combination, or tolerance whose reason
is not obvious.

Test the behavior implemented by the current source. Do not assert a preferred
result that the source does not provide.

## 4. Keep the test repeatable and independent

- Restore changed global values, environment variables, module caches, files,
  and monkeypatch state after each test.
- Do not depend on test order or state left by a process-wide scheduler.
- Prefer direct function calls and small real objects over broad end-to-end
  mocks.
- Add inputs to one test instead of copying the same test body.
- For syrupy saved expected results, update only the exact reviewed test.
  Follow the hipBLASLt snapshot rules; never update every snapshot at once.

## 5. Test unchanged source first

Run the exact test on unchanged source:

```bash
pytest -p no:cacheprovider -m unit -q <test-file>::<test-name>
```

The test must pass before a mutant is applied. Record the command and return
code in every review row that uses the test.

## 6. Verify mutants one at a time

Applying a mutant, running pytest, and restoring the file all touch the same
version-controlled source. Use only one process for this sequence.

For a quick mutmut check:

```bash
mutmut run <mutant-id> --max-children 1
```

For complete evidence, create a verifier manifest and run
`mutmut-verify.sh`. Report `KILLED` only when:

1. the test has the expected result on unchanged source;
2. the test exits with pytest assertion status `1` while the mutant is active;
   and
3. the source file is restored.

One test may detect several related mutants. Verify and record each mutant ID
separately.

## 7. Limit attempts to repair the test

When the first assertion does not detect the mutant:

1. Read the displayed source change and observed behavior again.
2. Correct the input or assertion once when the intended distinction was wrong.
3. Rerun the unchanged-source test and mutant verification.
4. If the mutant still survives, stop inventing assertions. Record evidence for
   identical behavior, a production-design problem, deferred work, or an
   inconclusive result.

Do not weaken assertions or repeat edits until mutmut reports the desired
status.

## 8. Limit parallel work

Parallel work is allowed only for reading code or writing separate test files
that share no fixtures or global state. Never run these operations at the same
time:

- `mutmut apply`;
- a named mutant rerun that writes changed source;
- `mutmut-verify.sh`;
- a source `pragma: no mutate` edit;
- a `pyproject.toml` mutation-setting edit; or
- source restoration.

When two authors need the same test file, assign the entire behavior group to
one author or edit the file serially.

## 9. Review requests to exclude a line

Do not add `# pragma: no mutate` only because a mutant survived. Require a
reviewed explanation that the change behaves identically or that testing it
would preserve intentionally unhelpful behavior. Record the exact line and run
the affected tests after the edit.

The pragma removes possible mutants from future runs. Keep that population
change visible in the review table.

## 10. Update the review table

For every mutant, record:

- the final decision and result;
- the exact test or the explanation for identical behavior or a design problem;
- unchanged-source and mutant return codes;
- whether source restoration succeeded;
- the test file changed; and
- any remaining risk or follow-up issue.

Check again that every input mutant has exactly one row. A new test without a
matching row, or a row omitted from verification, makes the campaign result
incomplete.
