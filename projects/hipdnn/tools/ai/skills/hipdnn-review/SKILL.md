---
name: hipdnn-review
description: Review a hipDNN pull request or local diff for correctness, API compatibility, provider behavior, resource management, code reuse, and testing coverage/quality. Uses local source/worktrees for cross-reference by default. Use when asked to review hipDNN code, review a PR, or assess whether a change is ready to merge.
argument-hint: "[PR URL | branch:<name> | local] [base:<branch>] [focus:<area>] [diff-only]"
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# hipDNN Review

Review hipDNN changes with a code-review stance: findings first, ordered by severity, with file and line references. Prioritize correctness, compatibility, resource ownership, test coverage, and maintainability over style-only comments.

## Inputs

- **PR URL**: GitHub PR URL to review.
- `local`: Review the current worktree diff instead of a PR.
- `branch:<name>`: Review a local branch instead of a PR.
- `base:<branch>`: Base branch for comparison. Default: the PR base branch, then `origin/develop`, then `develop`.
- `focus:<area>`: Optional review emphasis such as `testing`, `backend`, `frontend`, `provider`, `build`, or `reuse`.
- `diff-only`: Opt out of local checkout/worktree setup and review only the PR/diff plus already-available files. Use only when the user wants to avoid local clone/worktree cost.

If none of PR URL, `local`, or `branch:<name>` is supplied, ask which change set to review.

## Setup

If `diff-only` was passed, skip local worktree creation/update. Still capture a unique diff file and state in the final review that confidence is lower because local source cross-reference was intentionally skipped.

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
   - Branch:
     ```bash
     git diff --name-only <base>...<branch-ref>
     git diff --stat <base>...<branch-ref>
     ```
3. Choose a unique diff output path without pasting the full diff into the response. Use `mktemp` by default:
   ```bash
   DIFF_FILE=$(mktemp "${TMPDIR:-/tmp}/hipdnn-review.XXXXXX.diff")
   ```
   If repository or workspace instructions define a separate artifact directory, use that location instead. Do not create workspace-specific artifact folders inside a normal source checkout unless the local instructions explicitly require it.
   Pass the chosen `DIFF_FILE` path to every reviewer.
4. Fetch the diff:
   - PR:
     ```bash
     gh pr diff <PR_URL> > "$DIFF_FILE"
     ```
   - Local:
     ```bash
     git diff <base>... > "$DIFF_FILE"
     ```
   - Branch:
     ```bash
     git diff <base>...<branch-ref> > "$DIFF_FILE"
     ```
5. Prefer a local source checkout for cross-reference. A review based only on the PR page or raw diff is incomplete unless `diff-only` was requested.
   - Follow repository or workspace instructions for source worktree layout. If no local convention is documented, use an existing checkout instead of inventing a new directory layout.
   - For PR reviews, make the PR head and base source available locally before spawning reviewers. Use an existing local worktree/checkout if one already matches the PR head; otherwise fetch the PR/head branch and create or update a local worktree/checkout according to the repository's normal workspace conventions.
   - Fetch the selected base branch and update the base worktree when it can be fast-forwarded. If the base worktree is stale, dirty, detached unexpectedly, or cannot be fast-forwarded, report that limitation before reviewing.
   - For `branch:<name>` reviews, fetch the branch and base when possible, then resolve the reviewed branch reference before diffing: use the local branch if present, otherwise `origin/<name>` when that remote-tracking ref exists, otherwise the fetched ref such as `FETCH_HEAD`. Report which branch ref was reviewed.
   - For `local` reviews, use the current worktree and the selected base branch for cross-reference.
   - If local source setup is unavailable or skipped, state that the review is `diff-only` and lower confidence accordingly.
6. Read only the files needed to validate the changed behavior and nearby patterns. Prefer `rg` for call sites, similar implementations, tests, and ownership patterns; fall back to `grep` if `rg` is unavailable.

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
- When validating cuDNN compatibility, prefer source-level comparison against the equivalent cuDNN frontend/API implementation, such as the public cuDNN frontend repository, when web access or a local checkout is available. Use NVIDIA's published cuDNN API documentation as a supporting reference for released behavior. If neither an authoritative source page nor source implementation can be located, do not infer cuDNN semantics from memory; flag the compatibility point for human verification instead.
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

For documentation-only, AI-skill-only, or comment-only changes with no product behavioral surface, the Testing Assessment may state that no product behavior is exercised. Do not invent behavioral test gaps for non-code changes; focus instead on validation appropriate to the artifact, such as formatting, metadata parsing, link/installability, or a dry-run invocation.

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

## Severity

- **Critical**: likely correctness failure, data corruption, leak/crash in normal use, ABI/API break, or a defect serious enough to block merge.
- **Major**: real behavioral risk, missing essential validation, meaningful test gap, compatibility issue, or maintainability issue likely to cause defects.
- **Minor**: localized quality issue, unclear docs, low-risk edge case, or non-blocking cleanup.
- **Suggestion**: optional improvement, refactor, or broader follow-up.

## Default Multi-Agent Review

Prefer a multi-agent review using `Task` when reviewer delegation is available and allowed by the active system, user, and repository instructions. Tell the user which reviewers will run and proceed only when delegation is permitted by those active policies. If delegation is unavailable, disallowed, or not explicitly permitted in a stricter host environment, state that multi-agent review is preferred by the skill and use the direct single-pass fallback.

Use the Scope Buckets above as the canonical taxonomy. Spawn focused reviewers for affected implementation buckets, plus the required cross-cutting reviewers:

- **Bucket reviewers**: one reviewer per affected Scope Bucket when that bucket has meaningful changes.
- **Testing reviewer**: coverage and test quality. Always run this reviewer, even when no test files changed; apply the docs-only carveout above when appropriate.
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

Keep comments specific and actionable. Avoid speculative findings unless the risk is concrete and supported by code references.
