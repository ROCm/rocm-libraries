---
name: pr-summary
description: "Draft or revise pull request titles and bodies with a concise standard format: summary, risk assessment, testing summary, testing checklist, and technical changes. Use when preparing a new draft or ready-for-review PR, opening a PR from a branch, updating an existing draft/open PR, reopening a PR, or when the user provides a GitHub PR URL/branch and asks for PR summary, risk, testing, or description text. New PRs should be draft by default unless the user explicitly asks to open them ready for review."
argument-hint: "[PR URL | branch:<name> | local] [risk:<1-5>] [testing:<summary>]"
allowed-tools: Bash, Read, Grep, Glob
---

# PR Summary

## Required Inputs

Before drafting or editing PR text, make sure these details are known. If any are missing, ask the user for them first. Accept `N/A`, `none`, or `not run` as user answers, but do not render empty `N/A` fields in the PR body.

- Tracking references: Jira key, GitHub issue, prior PR, RFC/design doc, or none.
- Risk hint: ask for it if absent; the user may answer `N/A` or decline. The scale is 1-5, where 1 is minimal risk and 5 is very high risk.
- Testing performed: test groups, exact commands when available, status, ASICs for hardware tests, and links for relevant workflow/TheRock/CI/manual validation runs.
- ASIC coverage and blast radius: which GPU architectures (ASICs) the change can affect and which must be verified before merge. Judge this from the diff content, not the file path (see Blast Radius and ASIC Coverage). Ask the user only when the content's ASIC impact is genuinely unclear.

Do not block on details that can be discovered reliably from the PR/branch diff, commits, or repo files.

## Workflow

1. Inspect the PR or branch diff with `gh pr view`, `gh pr diff`, `git diff --stat`, and targeted file reads as needed.
2. Evaluate changed subsystems, public API/schema/build impact, feature flags, compatibility risk, and test coverage.
3. For new PRs, create a draft PR by default. Open the PR ready for review only when the user explicitly asks.
4. Draft a title and body, or revise the existing PR title/body, using the rules below.
5. Preserve accurate user-provided testing facts and links. Do not invent completed testing.

## Title And References

- Jira keys belong in the title only. Never put Jira keys, Jira URLs, branch names containing Jira keys, copied commit subjects containing Jira keys, or Jira issue references in the PR body.
- GitHub issues, prior PRs, RFCs, design docs, workflow runs, TheRock runs, and benchmark dashboards may appear in the body only as Markdown links.
- If the PR implements an RFC, link the RFC document in the Summary or Technical Changes section. Prefer an in-repository Markdown link when available.
- If a reliable non-Jira reference link cannot be found, ask the user or omit the reference instead of naming it unlinked.
- Put GitHub issue/PR references in the title only if the user explicitly asks or the repository convention requires it; otherwise link them in the body where relevant.

## Output Shape

- Creating a new PR: provide/apply `Title:` and `Body:`. The body must be Markdown. Create it as draft unless the user explicitly asks for a ready-for-review PR.
- Revising an existing PR, draft or open: preserve the title unless the user asks to change it; rewrite the body into the standard format.
- Body-only requests: return only the Markdown body.

## Blast Radius and ASIC Coverage

Judge ASIC coverage from what the diff actually changes, not from the file's location. Read the content to decide whether it is ASIC-independent, behavior-shifting, or arch-scoped, then map it to the coverage that must pass before merge. These coverage requirements are merge gates: surface them in the PR body so reviewers and mergers can confirm them.

- ASIC-independent wiring, plumbing, or core capability that does not change kernel selection, support surface, or default behavior (for example, frontend-to-backend wiring): standard PR CI is sufficient; no dedicated multi-arch sweep is required.
- Frontend or default-setting changes, or dispatch-behavior changes, that existing test cases exercise: a multi-arch CI run is needed, because the change can alter existing test-case behavior and surface arch-specific failures.
- Provider changes that add or extend ops or support surface: a full multi-arch sweep is needed, unless the ops are scoped to specific architectures, in which case only those architectures.
- Arch-specific changes (for example, a gfx950-only kernel or code path): only the affected ASICs.
- Expanding or newly enabling generic integration test cases (for example, activating a suite in a provider lane): a full sweep across all supported GFX families is needed — not just the default CI test families — because a newly added or activated generic case can fail on any target.
- Docs-only, comments-only, or skill-only changes: no ASIC coverage.

A behavior-shifting change (defaults or dispatch) with no existing covering tests still needs the corresponding run plus new tests to exercise it; absent coverage is a gap to close, not a reason to skip the run.

Discover the supported architectures rather than hardcoding them; the set drifts. The GPU families and test labels are discoverable from the in-repo multi-arch CI workflow (`.github/workflows/therock-multi-arch-ci.yml`, whose `workflow_dispatch` inputs include `linux_amdgpu_families` and `linux_test_labels`) and TheRock's GPU-family matrix. The workflow's default `linux_amdgpu_families` is a routine-CI subset; when the content requires a full sweep, take the required set from the supported family matrix, not from that default. Family names such as `gfx90a`, `gfx942`, `gfx950` (specific architectures) and `gfx94X`, `gfx120X` (family-group keys covering multiple architectures) are illustrative only.

When the required coverage exceeds what local hardware or default PR CI provides, the gap can be closed by launching a TheRock multi-arch integration CI run on the needed architectures. Record any such required-but-not-yet-passed run as an unchecked gate in the Testing Checklist.

## Risk Assessment

Use the user risk hint as input, then independently evaluate the diff.

- `1` / Minimal risk: docs-only, comments-only, metadata-only, or isolated non-shipping test changes.
- `2` / Low risk: narrow implementation change, low blast radius, good local coverage, no API/schema/build behavior impact.
- `3` / Medium risk: core subsystem changes, new feature paths, schema/API additions, dispatch/build/test infrastructure changes, or feature-flagged changes with meaningful integration surface.
- `4` / High risk: broad behavior changes, compatibility-sensitive public API changes, performance-critical code paths, complex migrations, or incomplete direct test coverage.
- `5` / Very high risk: cross-project architectural changes, default behavior changes in critical paths, ABI-breaking changes, large unproven refactors, or changes with known unresolved failures.

Factor the change's blast radius and required ASIC coverage (see Blast Radius and ASIC Coverage) into the level: a change that needs a multi-arch sweep or a specific-ASIC run that has not yet passed carries more residual risk than one fully covered by passing PR CI.

Keep the risk section to one short paragraph.

Example: `Medium risk. This touches provider dispatch and build wiring behind an opt-in flag; local unit tests passed, with PR CI pending.`

## PR Body Format

Use this exact section order:

```markdown
## Summary

<1-3 sentences describing purpose, motivation, and what the PR enables. Link named non-Jira references.>

## Risk Assessment

<Risk level and concise rationale.>

## ASIC Coverage

<Blast radius of the change and the ASICs that must be verified before merge. State whether passing PR CI is sufficient, a specific-ASIC run is required, or a full multi-arch sweep is required, and why. Omit this section only for docs-only, comments-only, or skill-only changes with no ASIC impact.>

## Testing Summary

- <Testing category and what it covers.>
- <Another testing category and what it covers.>

## Testing Checklist

- [x] <Test group> - `<command>` - Status: Passed
- [x] <Hardware test group> - `<command>` - ASICs: <ASIC list> - Status: Passed
- [ ] <Multi-arch sweep, only if the blast radius requires one> - TheRock multi-arch CI - ASICs: <families> - Status: Pending
- [ ] PR CI - GitHub PR checks - Status: Pending
- [ ] <Pending/not-run/failed test group> - Status: <Pending/Not run/Failed>

## Technical Changes

- <Top-level technical what/why change.>
- <Another top-level technical what/why change.>
```

## Checklist Rules

- Use `[x]` only for passed/completed validation.
- Use `[ ]` for pending, not run, failed, or unknown validation.
- Include command details directly after the test group when an exact command or useful command summary is known; do not prefix with `Command:`. Use backticks for local commands. Omit command text entirely when no useful command summary exists.
- Include `ASICs: ...` only for hardware tests where ASICs apply.
- Include `Link: ...` only when there is a relevant non-Jira link.
- Represent each ASIC verification the change requires as its own checklist gate, derived from the ASIC Coverage section. For a specific-ASIC requirement, name the ASIC; for a full sweep, add a multi-arch sweep gate listing the families. Leave the gate `[ ]` until that coverage has actually passed.
- Include the multi-arch sweep gate only when the blast radius requires it; omit it for ASIC-independent changes that passing PR CI fully covers. When the required coverage exceeds passing PR CI, keep the run as a pending gate rather than marking the change fully covered.

## Examples

Title with Jira:
```text
[hipDNN] ALMIOPEN-1234 Add plugin dispatch validation
```

Linked references in the body:
```markdown
This PR implements [RFC 0008](../../docs/rfcs/0008_OverridableTensorShapesDesign.md) support for override validation and follows up on [PR #1234](https://github.com/ROCm/rocm-libraries/pull/1234).
```

ASIC Coverage section (provider op added, full sweep required):
```markdown
Full multi-arch sweep required. This adds a provider op whose support surface is not arch-scoped, so it must pass on every supported GFX target before merge; local validation on gfx90a alone is insufficient.
```

Testing checklist:
```markdown
- [x] Commit hooks - `git commit` - Status: Passed
- [x] hipDNN unit tests - `ninja -C build hipdnn-unit-check` - ASICs: gfx90a - Status: Passed
- [x] TheRock package validation - Status: Passed - Link: [TheRock run](https://github.com/ROCm/TheRock/actions/runs/123456789)
- [ ] Multi-arch sweep - TheRock multi-arch CI - ASICs: gfx90a, gfx942, gfx950 - Status: Pending
- [ ] PR CI - GitHub PR checks - Status: Pending
```

Minimal docs-only PR:
```markdown
Title: [hipDNN] Add PR summary AI skill
State: Draft

## Summary

This PR updates hipDNN contributor documentation so agents can discover project-local AI skills for repeatable workflows.

## Risk Assessment

Minimal risk. This is documentation-only and does not affect product code, build configuration, public APIs, or tests.

## Testing Summary

- Markdown/frontmatter validation confirmed the new skill metadata is valid.

## Testing Checklist

- [x] Skill validation - `quick_validate.py <skill-path>` - Status: Passed
- [ ] PR CI - GitHub PR checks - Status: Pending

## Technical Changes

- Adds a project-local AI skill under `tools/ai/skills/`.
- Updates AI rules to list when agents should suggest the skill.
```

## Do Not Include

- Jira keys, Jira URLs, or Jira issue references in the body.
- Raw URLs unless explicitly requested; prefer Markdown links with labels like `[RFC 0008](...)`, `[PR #1234](...)`, `[issue #5678](...)`, `[workflow run](...)`, or `[TheRock run](...)`.
- `ASICs: N/A`, `Link: N/A`, or other empty placeholder fields.
- Unverified testing claims.
- File-by-file changelogs for large PRs.
