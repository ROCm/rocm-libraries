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

## Risk Assessment

Use the user risk hint as input, then independently evaluate the diff.

- `1` / Minimal risk: docs-only, comments-only, metadata-only, or isolated non-shipping test changes.
- `2` / Low risk: narrow implementation change, low blast radius, good local coverage, no API/schema/build behavior impact.
- `3` / Medium risk: core subsystem changes, new feature paths, schema/API additions, dispatch/build/test infrastructure changes, or feature-flagged changes with meaningful integration surface.
- `4` / High risk: broad behavior changes, compatibility-sensitive public API changes, performance-critical code paths, complex migrations, or incomplete direct test coverage.
- `5` / Very high risk: cross-project architectural changes, default behavior changes in critical paths, ABI-breaking changes, large unproven refactors, or changes with known unresolved failures.

Keep the risk section to one short paragraph.

Example: `Medium risk. This touches provider dispatch and build wiring behind an opt-in flag; local unit tests passed, with PR CI pending.`

## PR Body Format

Use this exact section order:

```markdown
## Summary

<1-3 sentences describing purpose, motivation, and what the PR enables. Link named non-Jira references.>

## Risk Assessment

<Risk level and concise rationale.>

## Testing Summary

- <Testing category and what it covers.>
- <Another testing category and what it covers.>

## Testing Checklist

- [x] <Test group> - `<command>` - Status: Passed
- [x] <Hardware test group> - `<command>` - ASICs: <ASIC list> - Status: Passed
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

## Examples

Title with Jira:
```text
[hipDNN] ALMIOPEN-1234 Add plugin dispatch validation
```

Linked references in the body:
```markdown
This PR implements [RFC 0008](../../docs/rfcs/0008_OverridableTensorShapesDesign.md) support for override validation and follows up on [PR #1234](https://github.com/ROCm/rocm-libraries/pull/1234).
```

Testing checklist:
```markdown
- [x] Commit hooks - `git commit` - Status: Passed
- [x] hipDNN unit tests - `ninja -C build hipdnn-unit-check` - ASICs: gfx90a - Status: Passed
- [x] TheRock package validation - Status: Passed - Link: [TheRock run](https://github.com/ROCm/TheRock/actions/runs/123456789)
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
