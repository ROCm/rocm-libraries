---
description: Draft or revise a structured pull request title and body
argument-hint: "[PR URL | branch:<name> | local] [risk:<1-5>] [testing:<summary>]"
allowed-tools: Bash, Read, Grep, Glob
---

Use the `pr-summary` skill to draft or revise pull request text.

Parse `$ARGUMENTS` as the PR, branch, or local-diff target plus any risk/testing hints. Preserve verified testing facts and avoid inventing validation.
