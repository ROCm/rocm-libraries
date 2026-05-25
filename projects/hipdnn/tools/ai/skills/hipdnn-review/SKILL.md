---
name: hipdnn-review
description: Review a hipDNN pull request, branch, or local diff for correctness, API compatibility, provider behavior, resource management, code reuse, and testing coverage/quality. Uses local source/worktrees for cross-reference by default. Use when asked to review hipDNN code, review a PR, or assess whether a change is ready to merge.
argument-hint: "[PR URL | branch:<name> | local] [base:<branch>] [focus:<area>] [diff-only]"
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# hipDNN Review

Review hipDNN changes with a code-review stance. Lead with findings, ordered by severity, and ground each finding in file and line references. Prioritize correctness, compatibility, resource ownership, provider behavior, test quality, and maintainability over style-only comments.

## Inputs

Infer the review target from the user request:

- **Pull request URL**: review a GitHub pull request.
- **Local diff**: review the current worktree changes.
- **Branch name**: review a local or remote branch.
- **Base branch**: default to the pull request base, then `origin/develop`, then `develop`.
- **Focus area**: optional emphasis such as `testing`, `backend`, `frontend`, `provider`, `build`, or `reuse`.
- **Diff-only**: use only when the user explicitly wants to avoid local checkout or worktree setup.

Ask the user which change set to review if it cannot be inferred.

## Setup

1. Determine the repository root:
   ```bash
   git rev-parse --show-toplevel
   ```

2. Inspect changed files:
   - Pull request:
     ```bash
     gh pr view <pr-url> --json title,body,files,additions,deletions,changedFiles,baseRefName,headRefName
     gh pr diff <pr-url> --name-only
     ```
   - Local diff:
     ```bash
     git diff --name-only <base>...
     git diff --stat <base>...
     ```
   - Branch:
     ```bash
     git diff --name-only <base>...<branch-ref>
     git diff --stat <base>...<branch-ref>
     ```

3. Save the full diff to a file instead of pasting it into the conversation:
   ```bash
   DIFF_FILE=$(mktemp "${TMPDIR:-/tmp}/review.XXXXXX.diff")
   ```
   Use repository or workspace artifact directories when active instructions require them.

4. Prefer local source for cross-reference. A review based only on a web page or raw diff has lower confidence unless the user requested diff-only mode.

5. Read only the files needed to validate the changed behavior and nearby patterns. Prefer `rg` for call sites, similar implementations, tests, and ownership patterns; fall back to `grep` if needed.

Do not modify files during review.

## Scope Buckets

Classify changed files before reviewing:

- **Frontend**: `projects/hipdnn/frontend/`, `projects/hipdnn/python/`, public frontend headers, graph/node/attribute wrappers, public C++ or Python API.
- **Backend**: `projects/hipdnn/backend/`, descriptors, engines, plugin loading, pack/unpack logic, backend C API.
- **Data and FlatBuffers SDK**: `projects/hipdnn/data_sdk/`, `projects/hipdnn/flatbuffers_sdk/`, `.fbs` schemas, generated-object wrappers.
- **Plugin SDK**: `projects/hipdnn/plugin_sdk/`, plugin interfaces, ABI/API contracts, behavior notes.
- **Providers**: `dnn-providers/`, provider registration, applicability, execution, workspace, external library calls.
- **Build and infrastructure**: `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, CI, packaging, scripts.
- **Tests**: unit/integration tests, test SDK helpers, GTest fixtures, generated test data.
- **Docs and tools**: documentation, RFCs, codegen, developer tooling.

## Review Checklist

### Correctness

- Validate the changed code path end to end, not just changed lines.
- Check new branches, status returns, enum conversions, descriptor fields, and default values.
- Confirm error paths return meaningful hipDNN status codes and do not leave partially initialized state.
- For public API changes, verify naming, lifetime rules, nullability, and consistency with existing API style.
- For schemas or serialized data, check backward/forward compatibility, defaults, and generated wrapper behavior.

### Resource Ownership

hipDNN CI commonly runs sanitizer-enabled tests. Treat leaks and ownership ambiguity as substantive review issues.

- Owning raw pointers must be wrapped in RAII immediately.
- FlatBuffers `UnPack()` returns owning raw pointers; prefer generated helpers returning `std::unique_ptr`, or wrap manually.
- Backend descriptor attributes that allocate descriptors must transfer ownership clearly and be wrapped immediately by callers.
- Avoid manual `delete`; it is fragile with assertions, exceptions, and early returns.
- Check provider handles, workspace buffers, streams, and library resources for correct lifetime and failure cleanup.

### Provider Behavior

- Verify provider applicability predicates match the implementation's actual support.
- Confirm registration uses the correct operation type, tensor layouts, data types, compute types, and behavior notes.
- Check workspace size calculation, stream usage, async behavior, and external library API calls.
- Ensure unsupported cases fail predictably instead of dispatching to an invalid or partial implementation.

### Compatibility Claims

- For hipDNN APIs corresponding to cuDNN APIs, compare signature, parameter semantics, defaults, status behavior, ownership/lifetime rules, and documented constraints.
- Prefer source-level comparison against the public cuDNN frontend repository when available. Use NVIDIA's published cuDNN API documentation as supporting reference for released behavior.
- If no authoritative source can be located, flag the compatibility point for human verification rather than relying on memory.
- Flag public API changes that silently diverge from equivalent cuDNN behavior unless the divergence is explicit, documented, and intentional.

### Code Reuse

- Search for existing helpers before recommending or accepting duplicated logic.
- Flag copy-paste implementations when an existing descriptor, wrapper, validator, test fixture, or provider helper can be reused.
- Recommend new abstractions only when duplication is meaningful and the abstraction matches existing project patterns.

### Build and Packaging

- Check CMake target visibility, component boundaries, install/export behavior, and generated-file dependencies.
- Avoid hardcoded local paths, machine-specific assumptions, and build-order dependencies.
- For CI or script changes, verify the command shape matches the repository's supported Linux and Windows workflows.

## Testing Review

Testing review is required for every hipDNN review, even when no test files changed. Do not equate "tests were added" with "behavior is covered"; read assertions and map them to changed code paths.

For documentation-only, AI-skill-only, or comment-only changes with no product behavior, focus on validation appropriate to the artifact, such as metadata parsing, link/installability, or dry-run invocation.

Assess:

- Covered behavior: meaningful coverage that exists.
- Missing coverage: exact scenarios or code paths that need tests.
- Weak tests: shallow assertions, excessive mocking, nondeterminism, or poor isolation.
- Recommended tests: concrete test names or scenarios to add.

## Delegation

When the active host and user request permit reviewer delegation, split broad reviews by affected scope bucket and add cross-cutting testing and reuse reviews. If delegation is unavailable, disallowed, or unnecessary, perform a direct single-pass review using the same checklist.

## Severity

- **Critical**: likely correctness failure, data corruption, leak/crash in normal use, ABI/API break, or a defect serious enough to block merge.
- **Major**: real behavioral risk, missing essential validation, meaningful test gap, compatibility issue, or maintainability issue likely to cause defects.
- **Minor**: localized quality issue, unclear docs, low-risk edge case, or non-blocking cleanup.
- **Suggestion**: optional improvement, refactor, or broader follow-up.

## Final Response

Use this shape:

```markdown
## Findings

- **Major** `[file:line]` Finding title.
  Explain the behavioral risk and why it matters. Include the fix direction when clear.

## Testing Assessment

- Covered: ...
- Missing: ...
- Weak tests: ...
- Recommended: ...

## Open Questions

- ...

## Summary

Briefly summarize the reviewed scope and overall readiness.
```

If there are no findings, say that clearly and mention residual testing risk.
