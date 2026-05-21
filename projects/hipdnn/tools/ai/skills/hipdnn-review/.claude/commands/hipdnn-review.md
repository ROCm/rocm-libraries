---
description: Review a hipDNN pull request, branch, or local diff
argument-hint: "[PR URL | branch:<name> | local] [base:<branch>] [focus:<area>] [diff-only]"
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

Use the `hipdnn-review` skill to review the requested hipDNN change set.

Parse `$ARGUMENTS` as the review target and options. Use the skill's review format, lead with findings, and include the testing assessment.
