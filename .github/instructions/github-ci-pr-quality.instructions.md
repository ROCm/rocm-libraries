---
applyTo: ".github/**"
---

# GitHub CI PR review guidance (for Copilot code review)

These instructions apply when reviewing pull requests that touch `.github/**` (workflows, composite actions, CI scripts, CODEOWNERS, and related configuration).

## Use the GitHub CI PR-quality skill

This repository maintains an agent skill named `github-ci-pr-quality`. A skill is a self-contained set of review instructions plus a rubric that an AI reviewer follows to produce a consistent, substance-focused review. This one covers where CI tests belong, runner sizing and run cost, trigger scope, shared CI state, and the required cross-read of each affected project's `TESTING.md`.

When reviewing a CI pull request, read and follow that skill as your review rubric:

- GitHub CI overlay (start here): `skills/github-ci-pr-quality/SKILL.md`
- Base skill it extends (in `ROCm/TheRock`, not present in this repo): https://github.com/ROCm/TheRock/blob/main/skills/rocm-pr-quality/SKILL.md and https://github.com/ROCm/TheRock/blob/main/skills/rocm-pr-quality/reference.md

Two points to check on every CI pull request: it should not add a new workflow file for a single test when an existing lane fits, and it should not use a large prod runner for light CPU-only work without a stated reason.

The overlay tightens the base skill for CI and never relaxes a base rule; on any conflict, the skill files are the source of truth. This guidance is advisory: it points at the rubric and raises the review floor, but does not by itself approve or block a merge.
