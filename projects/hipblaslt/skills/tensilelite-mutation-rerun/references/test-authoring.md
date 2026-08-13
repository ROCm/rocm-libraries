# Authoring mutation-killing tests

Use this procedure for ledger rows whose disposition is `add-test`. The goal is
the smallest maintainable assertion that distinguishes current behavior from a
specific mutant, not maximum test count or line coverage.

## 1. Prepare one behavior group

Group mutants only after the complete ledger exists. For one source function or
coherent behavior:

1. Read the current source and its callers.
2. Run `mutmut show <id>` for every group member.
3. Read nearby tests and fixtures to learn established imports, markers,
   parameterization, isolation, and snapshot style.
4. Identify which mutants can share an input/assertion and which require
   distinct behavior cases.

Do not trust the group label as mutant identity. Keep one ledger row per mutant.

## 2. Choose the test location

Prefer, in order:

1. Strengthen an existing focused test when the assertion belongs to the same
   behavior contract.
2. Add a parametrized case to an existing focused file.
3. Add one new focused test file for a coherent function/behavior group.

Do not generate one file per mutant. Follow repository pytest naming so CI and
the PR policy bot discover the test (`test_*.py` for Python tests). Use
`@pytest.mark.unit` where the surrounding suite requires it.

## 3. State the mutation question

Before writing an assertion, answer:

> What specific source change makes this assertion fail?

Reject assertions that only prove a callable exists, a mock was called, a value
is non-null, or the function did not crash. Avoid mocking the behavior under
test. Record the rationale for non-obvious numeric values, parameter choices,
or tolerances.

Characterize actual current behavior. Do not encode an idealized result merely
because it seems preferable.

## 4. Keep tests deterministic and isolated

- Restore globals, environment variables, module caches, files, and monkeypatch
  state after each test.
- Avoid order dependence and process-global scheduler residue.
- Prefer direct functions and small real objects over broad end-to-end mocks.
- Parameterize repeated cases instead of copy/paste.
- For syrupy snapshots, update only the exact reviewed node and follow the
  hipBLASLt golden discipline. Never blanket-update snapshots.

## 5. Validate clean behavior first

Run the exact node on clean source:

```bash
pytest -p no:cacheprovider -m unit -q <test-file>::<test-name>
```

The clean node must pass before applying a mutant. Record the command and return
code in every ledger row mapped to the node.

## 6. Verify mutants serially

Mutation application, pytest execution, and restoration share tracked source.
Use one serial verifier actor only.

For a quick native mutmut check:

```bash
mutmut run <mutant-id> --max-children 1
```

For strict evidence, create a verifier manifest and run `mutmut-verify.sh`.
Accept `KILLED` only when clean status matches expectation, mutated pytest status
is assertion failure `1`, and restoration is clean.

One test node may kill several related mutants, but verify and record every ID
individually.

## 7. Bound repair

If the first assertion does not kill the mutant:

1. Re-read the diff and observed behavior.
2. Correct the input or assertion once when the behavior distinction was wrong.
3. Re-run clean and strict mutant verification.
4. If still not killed, stop inventing assertions and reclassify with evidence
   as equivalent, design smell, deferred, or inconclusive.

Do not loop indefinitely or weaken assertions until a desired status appears.

## 8. Parallelism rules

Parallel work is permitted only for read-only analysis or authoring distinct
test files with no shared fixtures/global state. Never run these concurrently:

- `mutmut apply`;
- named mutant reruns that materialize source;
- `mutmut-verify.sh`;
- source pragma edits;
- pyproject mutation configuration edits; or
- source restoration.

If two authors need the same test file, serialize their edits or assign the
whole behavior group to one author.

## 9. Pragmas and equivalence

Do not add `# pragma: no mutate` simply because a mutant survived. Require a
reviewed equivalence or intentionally-unhelpful-behavior rationale, record the
exact line, and run the affected suite after the edit. A pragma changes the
mutation search space and must remain visible in the ledger.

## 10. Update the ledger

For each mutant, record:

- final disposition and verdict;
- exact test node or equivalence/design rationale;
- clean and mutant return codes;
- restoration result;
- test file changed; and
- any residual risk or follow-up tracker.

Re-run the conservation check after authoring and repair. Tests authored without
a ledger row, or ledger rows silently omitted from verification, invalidate the
campaign result.
