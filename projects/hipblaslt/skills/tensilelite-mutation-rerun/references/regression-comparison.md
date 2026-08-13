# Comparing mutation reports

A report comparison may become an automated GitHub check only after the
repository has three working pieces:

1. one standard report generator;
2. a machine identifier that distinguishes every mutant; and
3. a GitHub workflow that runs the comparison.

Until all three exist, treat comparison as reviewed analysis. Keep every
uncertain result visible.

## 1. Compare compatible reports only

Before matching mutants, require both the reference report and current report
to include:

- the report format and version;
- the complete source commit ID;
- the target module set;
- the mutmut version, mutation operations, and relevant settings;
- `only_mutate`, exclusions, and covered-line behavior;
- the selected tests and covering-test record;
- the Python and pytest environment; and
- paths to the complete original results and review table.

The source commit IDs normally differ. The reference must still be an intended
ancestor or another comparison point approved by a reviewer. Mutmut version,
target scope, and mutation settings must be compatible.

When those inputs differ enough to change which mutants are produced, stop.
Require an approved replacement reference report instead of calling the
population change a test regression.

## 2. Give every mutant a unique machine identifier

A machine identifier must distinguish identical-looking changes that occur more
than once in one file. Include all four parts:

1. The source path relative to the repository.
2. The full class and function name, or `<module>`.
3. The mutation operation and changed body, represented by a full
   cryptographic digest.
4. A stable location inside that function, such as a parsed syntax-tree path
   plus the occurrence number for repeated nodes.

A line number is useful for display, but it cannot be the only location. Line
numbers change when unrelated lines are added.

Normalize leading `./`, path separators, and diff prefixes such as `a/` and
`b/`. Reject absolute paths and paths that use `..` to leave the repository.

Reject an identifier when:

- the changed body is empty;
- the original function or surrounding source cannot be found;
- the source path is invalid;
- a required field is missing; or
- two rows produce the same machine identifier.

A duplicate identifier means the report is invalid. Never merge duplicate rows
by choosing the better or worse status. For example, merging two reference rows
into one current row could hide a lost verified kill.

## 3. Remove unstable display details without changing meaning

The identifier calculation may remove details that can change without changing
the mutation:

- mutmut's numeric suffix and display header;
- Git blob hashes;
- file-header prefixes; and
- diff line numbers.

It must preserve details that affect meaning:

- every added and removed line;
- leading indentation;
- multiline changed bodies;
- the mutation operation, when available; and
- the function and repeated-occurrence location.

Store the full digest in the report. Shorten it only when displaying it to a
person.

## 4. Classify result changes explicitly

Compare reviewed final results, not only the original mutmut labels.

| Reference result | Current result | Required decision |
|---|---|---|
| verified kill | verified kill | No regression |
| verified kill | survived, no-tests, timeout, skipped, suspicious, interrupted, segfault, not-checked, unknown, or missing | Regression or approved replacement reference |
| equivalent | verified kill | Review the old identical-behavior explanation again |
| equivalent | equivalent | Accept only when the explanation still applies |
| missing from reference | any | Review why the mutant population changed |
| new mutant | not killed | New test work; fail only when an approved policy requires it |

Never change a verified kill to equivalent or exclude it with a pragma without
review. Never treat an unknown status as harmless.

## 5. Explain missing and reformatted mutants

A mutant missing from the current report is not automatically success or
failure. Record one reason:

- the source behavior was intentionally removed;
- the mutmut version or mutation operation changed;
- the target or settings changed;
- source formatting or refactoring changed the structural identifier;
- the identifier algorithm failed; or
- one report is incomplete.

Require an approved replacement reference report when the mutant population
changes. A new not-killed mutant in the same file may indicate that reformatting
hid a regression, but the file name alone is not proof. Compare the function,
changed body, and structural location.

## 6. Add automation in stages

Use these stages in order:

1. **Report only:** produce compatible reports and reviewed classifications.
   Never fail automated checks.
2. **Fail on verified regressions:** begin only after identifier uniqueness and
   report completeness are proven.
3. **Fail on unexplained population changes:** require an approved replacement
   reference for missing or new identities.
4. **Require no new not-killed mutants:** add this only after the team accepts
   the maintenance cost.

Do not call a program an automated check when no GitHub workflow runs it.

## 7. Approve a replacement reference report

Record:

- the old and new source commit IDs;
- why the mutant population or identifiers changed;
- mutmut, setting, or test-selection changes;
- counts of retained, removed, and new identities;
- the decision for every prior verified kill; and
- reviewer or owner approval.

Never replace the reference report only to make an automated check pass.

## 8. Test future comparison code

Before enabling executable comparison, add ordinary pytest tests that prove:

- identical changes in different functions remain distinct;
- identical changes at different locations in one function remain distinct;
- two reference rows cannot collapse into one current row and still pass;
- empty or malformed diffs are rejected;
- duplicate machine identifiers are rejected;
- multiline or indentation-sensitive changes remain distinct;
- equivalent path spellings become one repository-relative path;
- absolute paths and parent traversal are rejected;
- incompatible report formats, mutmut versions, or settings are rejected;
- a prior verified kill changing to any not-killed status fails;
- missing, new, and reformatted mutants follow the stated policy; and
- command errors and report-only or blocking exit codes remain stable.

Implement the standard report generator first, the comparison program second,
and the GitHub workflow last. Hand-written comparison fixtures alone do not
prove that the complete workflow works.

## 9. Report the comparison

Include:

```text
compatible reports: yes/no and reason
unique identifiers: yes/no and duplicate count
verified kills retained
verified regressions
identical-behavior decisions to review again
missing identities grouped by reason
new identities grouped by status
formatting or identifier uncertainties
replacement reference required: yes/no
automation stage and exit decision
```

When report compatibility, identifier uniqueness, or required files cannot be
verified, mark the comparison **Inconclusive**. Do not return a passing
regression result.
