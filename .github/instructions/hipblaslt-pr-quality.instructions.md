---
applyTo: "projects/hipblaslt/**"
---

# hipBLASLt PR review guidance (for Copilot code review)

These instructions apply when reviewing pull requests that touch `projects/hipblaslt/**` (the hipBLASLt component, including `tensilelite/`).

## Use the hipBLASLt PR-quality skill

hipBLASLt maintains an agent skill named `hipblaslt-pr-quality`. A skill is a self-contained set of review instructions plus a rubric that an AI reviewer follows to produce a consistent, substance-focused review; among other actions it defines how to **review** a pull request (change classification, the required-test bar, test-substance checks, device/arch coverage, coupling risks, and severity-ordered findings).

When reviewing a hipBLASLt PR, read and follow that skill as your review rubric:

- hipBLASLt overlay (start here): `projects/hipblaslt/skills/hipblaslt-pr-quality/SKILL.md`
- Base skill it extends (in `ROCm/TheRock`, not present in this repo): https://github.com/ROCm/TheRock/blob/develop/skills/rocm-pr-quality/SKILL.md and https://github.com/ROCm/TheRock/blob/develop/skills/rocm-pr-quality/reference.md

The overlay tightens the base skill for hipBLASLt and never relaxes a base rule; on any conflict, the skill files are the source of truth. This guidance is advisory: it points at the rubric and raises the review floor, but does not by itself approve or block a merge.
