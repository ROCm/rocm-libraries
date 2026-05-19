---
name: hipdnn-review
description: Review a hipDNN pull request or local diff for correctness, API compatibility, provider behavior, resource management, code reuse, and testing coverage/quality. Uses local source/worktrees for cross-reference by default. Use when asked to review hipDNN code, review a PR, or assess whether a change is ready to merge.
argument-hint: "[PR URL | branch:<name> | local] [base:<branch>] [focus:<area>] [diff-only]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Review

Review hipDNN changes with a code-review stance: findings first, ordered by severity, with file and line references. Prioritize correctness, compatibility, resource ownership, test coverage, and maintainability over style-only comments.

## Inputs

- **PR URL**: GitHub PR URL to review.
- `local`: Review the current worktree diff instead of a PR.
- `branch:<name>`: Review a local branch or use it as the PR worktree name when needed.
- `base:<branch>`: Base branch for comparison. Default: the PR base branch, then `origin/develop`, then `develop`.
- `focus:<area>`: Optional review emphasis such as `testing`, `backend`, `frontend`, `provider`, `build`, or `reuse`.
- `diff-only`: Opt out of local checkout/worktree setup and review only the PR/diff plus already-available files. Use only when the user wants to avoid local clone/worktree cost.

If neither a PR URL nor `local` is supplied, ask which change set to review.

## Setup

1. Determine the repository root:
   ```bash
   git rev-parse --show-toplevel
   ```
2. Inspect changed files:
   - PR:
     ```bash
     gh pr view <PR_URL> --json title,body,files,additions,deletions,changedFiles,baseRefName,headRefName
     gh pr diff <PR_URL> --name-only
     ```
   - Local:
     ```bash
     git diff --name-only <base>...
     git diff --stat <base>...
     ```
3. Fetch the diff without pasting the full diff into the response:
   - PR:
     ```bash
     gh pr diff <PR_URL> > /tmp/hipdnn-review.diff
     ```
   - Local:
     ```bash
     git diff <base>... > /tmp/hipdnn-review.diff
     ```
4. Prefer a local source checkout for cross-reference. A review based only on the PR page or raw diff is incomplete unless `diff-only` was requested.
   - For PR reviews, make the PR head and base source available locally before spawning reviewers. Use an existing local worktree/checkout if one already matches the PR head; otherwise fetch the PR/head branch and create or update a local worktree/checkout according to the repository's normal workspace conventions.
   - For local reviews, use the current worktree and the selected base branch for cross-reference.
   - If local source setup is unavailable or skipped, state that the review is `diff-only` and lower confidence accordingly.
5. Read only the files needed to validate the changed behavior and nearby patterns. Prefer `rg` for call sites, similar implementations, tests, and ownership patterns; fall back to `grep` if `rg` is unavailable.

Do not modify files during review.

## Scope Buckets

Classify changed files before reviewing so each affected area gets the right scrutiny.

- **Frontend**: `projects/hipdnn/frontend/`, `projects/hipdnn/python/`, public frontend headers, graph/node/attribute wrappers, public C++/Python API.
- **Backend**: `projects/hipdnn/backend/`, descriptors, engines, plugin loading, pack/unpack logic, backend C API.
- **Data/FlatBuffers SDK**: `projects/hipdnn/data_sdk/`, `projects/hipdnn/flatbuffers_sdk/`, `.fbs` schemas, generated-object wrappers.
- **Plugin SDK**: `projects/hipdnn/plugin_sdk/`, plugin interfaces, ABI/API contracts, behavior notes.
- **Providers**: `dnn-providers/`, provider registration, applicability, execution, workspace, external library calls.
- **Build/Infra**: `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, CI, packaging, scripts.
- **Tests**: unit/integration tests, test SDK helpers, GTest fixtures, generated test data.
- **Docs/Tools**: documentation, RFCs, codegen, developer tooling.

## Review Checklist

### Correctness

- Validate the changed code path end to end, not just the changed lines.
- Check all new branches, status returns, enum conversions, descriptor fields, and default values.
- Confirm error paths return meaningful hipDNN status codes and do not leave partially initialized state.
- For public API changes, verify naming, lifetime rules, nullability, and compatibility with existing API style.
- For schemas or serialized data, check backward/forward compatibility, defaults, and generated wrapper behavior.

### Resource Ownership

hipDNN CI commonly runs sanitizer-enabled tests. Treat leaks and ownership ambiguity as substantive review issues.

- Owning raw pointers must be wrapped in RAII immediately.
- FlatBuffers `UnPack()` returns owning raw pointers; prefer generated helpers returning `std::unique_ptr`, or wrap manually.
- Backend descriptor attributes that allocate descriptors must transfer ownership clearly and be wrapped immediately by callers.
- Avoid manual `delete`; it is fragile with `ASSERT_*`, exceptions, or early returns.
- Check provider handles, workspace buffers, streams, and library resources for correct lifetime and failure cleanup.

### Provider Behavior

- Verify provider applicability predicates match the implementation's actual support.
- Confirm registration uses the correct operation type, tensor layouts, data types, compute types, and behavior notes.
- Check workspace size calculation, stream usage, async behavior, and external library API calls.
- Ensure unsupported cases fail predictably instead of dispatching to an invalid or partial implementation.

### Compatibility Claims

- For existing public-facing hipDNN API that corresponds to an equivalent cuDNN API, check whether the signature, parameter semantics, defaults, status behavior, ownership/lifetime rules, and documented constraints preserve seamless porting expectations.
- Flag public API changes that silently diverge from the equivalent cuDNN API unless the divergence is explicit, documented, and intentional.
- New hipDNN-only API does not need to match cuDNN by default. Review it for consistency with hipDNN design, and only apply cuDNN parity expectations when the API or documentation claims cuDNN compatibility.
- If docs mention cuDNN migration, compatibility, or parity, ensure the wording is precise and does not promise unimplemented behavior.

### Code Reuse

- Search for existing helpers before recommending or accepting duplicated logic.
- Flag copy-paste implementations when an existing descriptor, wrapper, validator, test fixture, or provider helper can be reused.
- Recommend new abstractions only when duplication is meaningful and the abstraction matches existing project patterns.

### Build And Packaging

- Check CMake target visibility, component boundaries, install/export behavior, and generated-file dependencies.
- Avoid hardcoded local paths, machine-specific assumptions, or build-order dependencies.
- For CI or script changes, verify the command shape matches the repository's supported Linux and Windows workflows.

## Testing Review

Testing review is required for every hipDNN review, even when no test files changed. Do not equate "tests were added" with "the behavior is covered"; read the assertions and map them back to the changed code paths.

### Coverage Questions

- What new or modified public API, descriptor field, provider capability, schema field, behavior note, or build option needs coverage?
- Which changed branches, error paths, unsupported cases, and boundary values are untested?
- Are both positive and negative cases covered?
- Are integration tests needed because the behavior crosses frontend/backend/provider boundaries?
- Are GPU tests needed, or is a unit-level mock/fixture sufficient?
- For provider changes, do tests cover applicability rejection and successful execution where feasible?
- For serialization changes, do tests cover missing/default fields, round trip behavior, and invalid input?
- For bug fixes, is there a regression test that fails without the fix?

### Quality Questions

- Do tests assert observable behavior, outputs, status codes, ownership, and state changes rather than only "does not crash"?
- Are assertions strong enough to catch the likely regression?
- Are tests deterministic and isolated from global state, test order, current working directory, environment variables, and GPU availability unless explicitly integration-scoped?
- Are fixtures and helpers used consistently with nearby tests?
- Are multi-type or multi-shape cases expressed with typed/parameterized tests when that improves coverage without obscuring intent?
- Do sanitizer-sensitive tests use RAII so a failed assertion does not leak descriptors or unpacked FlatBuffers objects?
- Are skipped or disabled tests justified, narrow, and not masking the changed behavior?

### Testing Output

In the final review, include a **Testing Assessment** section with:

- Covered behavior: meaningful coverage that exists.
- Missing coverage: exact scenarios or code paths that need tests.
- Weak tests: tests with shallow assertions, excessive mocking, nondeterminism, or poor isolation.
- Recommended tests: concrete test names or scenarios to add.

## Default Multi-Agent Review

Default to a multi-agent review when the runtime supports reviewer delegation. Tell the user which reviewers will run and proceed unless they opt out or ask for a single-pass review. If the active environment requires explicit permission before spawning agents, state that multi-agent review is the skill default and ask for permission before spawning.

Use focused reviewers by changed-file bucket:

- **Frontend/API reviewer**: frontend, Python frontend, public headers, graph/node/attribute wrappers, public API compatibility, cuDNN-porting expectations for existing equivalent API.
- **Backend reviewer**: backend descriptors, engines, plugin loading, status propagation, resource ownership, and backend C API behavior.
- **Data/FlatBuffers reviewer**: data SDK, FlatBuffers SDK, schemas, generated wrappers, serialization compatibility, unpacking/ownership.
- **Plugin SDK reviewer**: plugin interfaces, ABI/API contracts, behavior notes, plugin-facing ownership and error semantics.
- **Provider reviewer**: provider registration, applicability, execution, workspace, streams, external library calls, unsupported-case behavior.
- **Build/Infra reviewer**: CMake, presets, CI, packaging, install/export behavior, scripts, generated-file dependencies.
- **Docs/Tools reviewer**: documentation, RFCs, codegen, developer tooling, user-facing claims.
- **Testing reviewer**: coverage and test quality. Always run this reviewer, even when no test files changed.
- **Reuse reviewer**: duplication and existing helper opportunities. Run this for broad PRs, repeated patterns, generated boilerplate, or any change spanning multiple implementation buckets.

Each reviewer should:

- Review only its assigned bucket, but read enough adjacent code to validate behavior.
- Use the local PR/head source and local base source for cross-reference. Do not rely only on the diff unless the review is explicitly running in `diff-only` mode.
- Compare against existing base-branch patterns and tests.
- Return findings with severity, file/line references, and concrete rationale.
- Include "No findings" when nothing actionable is found.
- Not modify files.

Direct single-pass review is acceptable when the user opts out of multi-agent review, the change is trivial, or agent delegation is unavailable. Even in direct mode, keep the same bucket checklist and always include the testing assessment.

## Final Response Format

Lead with findings. If there are no findings, say so clearly and mention residual testing risk.

```markdown
## Findings

- **Major** `[file:line]` Finding title.
  Explain the behavioral risk and why it matters. Include the specific fix direction when clear.

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

Severity guidance:

- **Critical**: likely correctness failure, data corruption, leak/crash in normal use, ABI/API break, or invalid merge blocker.
- **Major**: real behavioral risk, missing essential validation, meaningful test gap, compatibility issue, or maintainability issue likely to cause defects.
- **Minor**: localized quality issue, unclear docs, low-risk edge case, or non-blocking cleanup.
- **Suggestion**: optional improvement, refactor, or broader follow-up.

Keep comments specific and actionable. Avoid speculative findings unless the risk is concrete and supported by code references.
