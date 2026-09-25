# ADR 0014: Apply the logic filter to validation and generation

Status:  Accepted
Defect:  none — the command option intentionally narrows both operations

## Context

The device-library build accepts a logic-file glob. The create-library command
used that glob, while the preceding `tensilelite logic --check-all` command
validated every file for the selected architecture. The parser snapshot also
changed when the validation command gained its `logic_filter` default.

## Decision

Translate the create-library stem filter into the equivalent recursive YAML
glob for validation, and record the validation parser's default in its saved
expected result.

## Consequences

A filtered build validates exactly the files it can generate. An unfiltered
build retains the `**/*.yaml` validation default. Future changes to this option
must update both command shapes and supersede this record if their scopes
intentionally diverge.
