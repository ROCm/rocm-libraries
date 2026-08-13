# Regression comparison and mutant identity

Mutation-result comparison can become a CI gate only after the repository has a
canonical artifact producer, a collision-resistant identity, and an actual CI
consumer. Until then, perform comparisons as reviewed analysis and preserve all
ambiguity in the report.

## 1. Require compatible artifacts

Before matching mutants, require both baseline and current artifacts to include:

- artifact schema/version;
- immutable source SHA;
- target module set;
- mutmut version and mutation operator/configuration;
- `only_mutate`, exclusions, and covered-lines behavior;
- test selection and covering-set provenance;
- Python/pytest environment identity; and
- complete raw result and ledger paths.

The source SHAs normally differ, but the baseline must be an intended ancestor
or reviewed comparison point. Mutmut version, target scope, and mutation settings
must be compatible. If they differ materially, stop and require an explicit
rebaseline; do not interpret population churn as a test regression.

## 2. Identity requirements

A machine identity must distinguish repeated identical expressions in one file.
Include all of:

1. Canonical repository-relative source path.
2. Qualified class/function scope or `<module>`.
3. Canonical mutation operation/body using the full cryptographic digest.
4. A structural occurrence discriminator within the scope, such as a stable AST
   path/fingerprint plus repeated-node occurrence index.

Line number may be retained for display and diagnostics but must not be the only
occurrence discriminator. Normalize `./`, path separators, diff `a/` and `b/`
prefixes, and reject absolute or parent-traversing paths.

Reject an identity when:

- the canonical changed body is empty;
- the original anchor/scope cannot be resolved;
- the file path is invalid;
- required identity fields are missing; or
- two artifact rows produce the same machine identity.

Duplicate identities are an artifact error. Never merge them by best/worst
status: two baseline rows collapsing to one current row can hide a lost kill.

## 3. Canonicalization constraints

Canonicalization may remove unstable metadata such as:

- mutmut ordinal/name headers;
- git blob hashes;
- file header prefixes; and
- hunk line numbers.

Preserve semantic content:

- all added and removed lines;
- leading indentation;
- multiline mutation bodies;
- mutation operator/type when available; and
- structural scope/occurrence context.

Use the full digest in artifacts. Abbreviate only for human display.

## 4. Status transition policy

Define policy against final reviewed statuses, not only native mutmut labels.

Minimum transitions:

| Baseline | Current | Classification |
|---|---|---|
| strict/verified kill | strict/verified kill | OK |
| strict/verified kill | survived, no-tests, timeout, skipped, suspicious, interrupted, segfault, not-checked, unknown, or absent | Regression or explicit rebaseline required |
| equivalent | verified kill | Re-audit equivalence rationale |
| equivalent | equivalent | OK only when rationale remains applicable |
| absent | any | Explicit source/operator population review |
| new mutant | non-killed | New coverage work; fail only under reviewed policy |

Never silently downgrade a verified kill to equivalent or pragma. Never treat an
unknown status as benign.

## 5. Absent and reformatted mutants

Absence is not automatically success or failure. Classify why the baseline
identity disappeared:

- source behavior removed intentionally;
- mutation operator/version changed;
- target/configuration changed;
- formatting/refactor changed structural identity;
- identity algorithm failed; or
- artifact is incomplete.

Require a reviewed rebaseline for population changes. A same-file new survivor
may be evidence of a reformat-masked regression, but file coincidence alone is
not proof; compare structural scope and mutation body.

## 6. Gate stages

Use an explicit maturity ladder:

1. **Report only:** produce compatible, reviewed classifications; never fail CI.
2. **Fail on verified regressions:** only after identity uniqueness and artifact
   completeness are proven.
3. **Fail on population ambiguity:** require explicit rebaseline for absent/new
   identities under high-confidence scope.
4. **Strict ratchet:** optionally fail on new non-killed mutants after the team
   accepts the maintenance policy.

Do not name a tool a CI gate when no CI workflow invokes it.

## 7. Rebaseline protocol

Every rebaseline must record:

- old and new source SHAs;
- reason for population/identity changes;
- mutmut/config/test-selection changes;
- count of retained, removed, and new identities;
- reviewed disposition of prior verified kills; and
- reviewer/owner approval.

Never overwrite the baseline merely to make CI green.

## 8. Acceptance tests for future code

Before implementing or enabling executable comparison, require normal pytest
tests for at least:

- two identical mutations in different functions remain distinct;
- two identical mutations at different locations in one function remain
  distinct;
- two baseline rows and one current row cannot pass as one retained kill;
- empty/malformed diffs are rejected;
- duplicate machine identities are rejected;
- multiline and indentation-sensitive mutations remain distinct;
- path variants normalize to one canonical path;
- absolute and parent-traversing paths are rejected;
- incompatible schema/mutmut/config artifacts are rejected;
- verified-kill-to-every-non-killed-status transitions fail;
- absent/new/reformat cases follow explicit policy; and
- CLI error/report-only/gating exit codes are stable.

Add a canonical producer first, then the comparator, then the CI invocation.
Tests over hand-authored comparator fixtures alone do not prove the end-to-end
gate works.

## 9. Review output

For each comparison report:

```text
compatible artifacts: yes/no + reason
unique identities: yes/no + duplicate count
retained verified kills
verified regressions
equivalence re-audits
absent identities by reason
new identities by status
reformat/identity ambiguities
rebaseline required: yes/no
gate stage and exit decision
```

If compatibility, identity uniqueness, or artifact completeness fails, the
comparison is **Inconclusive** and must not return a green regression verdict.
