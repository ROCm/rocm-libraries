---
name: hipdnn-code-quality
description: Review a hipDNN pull request, branch, or local diff purely for code quality and maintainability - duplication, shared-helper reuse, design patterns, performance, readability, comment complexity, file size, and naming. Diff-scoped against the base/merge base, same target resolution as hipdnn-review. Uses local source for cross-reference. Use when asked to assess code quality, style, maintainability, duplication, or readability of a hipDNN change - not correctness or test coverage.
argument-hint: "[PR URL | branch:<name> | local] [base:<branch>] [focus:<axis>] [diff-only]"
allowed-tools: Bash, Read, Grep, Glob, Task
---

# hipDNN Code Quality

Review a hipDNN change set for **maintainability and code quality only**. This is the quality-axis companion to `hipdnn-review`: same diff scope, opposite emphasis. Do not review correctness, compatibility, resource-ownership defects, provider behavior, or test coverage here - those belong to `hipdnn-review`. Lead with findings, ordered by severity, each grounded in file and line references, each with a concrete fix direction.

Positive comments and compliments are not findings. Omit them.

## Relationship to hipdnn-review

- `hipdnn-review` prioritizes correctness and deprioritizes style. This skill is the inverse.
- Reuse `hipdnn-review`'s **Scope Buckets** and **Severity** ladder verbatim (restated below). Do not invent a second taxonomy.
- If, while auditing quality, you find a genuine correctness/leak/compat defect, note it once under Open Questions and recommend a `hipdnn-review` pass. Do not expand this review into a correctness review.

## Inputs

Infer the review target from the user request, identically to `hipdnn-review`:

- **Pull request URL**: review a GitHub pull request.
- **Local diff**: review the current worktree changes.
- **Branch name**: review a local or remote branch.
- **Base branch**: default to the pull request base, then `origin/develop`, then `develop`.
- **Focus axis**: optional emphasis - `duplication`, `naming`, `readability`, `performance`, `size`, `comments`, `patterns`, or `reuse`.
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
   DIFF_FILE=$(mktemp "${TMPDIR:-/tmp}/quality.XXXXXX.diff")
   ```
   Use repository or workspace artifact directories when active instructions require them.

4. Prefer local source for cross-reference. Quality findings - especially duplication and reuse - require reading the surrounding files and searching the tree for existing helpers, not just the diff hunks.

5. Read the changed files in full where practical (for files over 1000 lines, focus on changed hunks plus generous surrounding context). Use `grep`/`glob` to find existing helpers, similar implementations, and naming precedents; use structural search for pattern smells.

6. Run the bundled metric pass first, then read. The script is diff-scoped and
   uses only in-harness techniques (no clang-tidy/lizard/cloc); every line it
   emits is a *candidate*, not a verdict - confirm each against the source
   before reporting.
   ```bash
   python3 scripts/quality_scan.py --repo <repo-root> --base <base-branch>
   # or an explicit set:
   python3 scripts/quality_scan.py --repo <repo-root> --files <f1> <f2> ...
   # machine-readable:
   python3 scripts/quality_scan.py --repo <repo-root> --base <base-branch> --json
   ```
   It reports, for changed source files only: large files, long functions
   (namespace- and Allman-brace-aware), duplicated regions (changed files
   matched against the whole tree, so a copy of untouched code is still found;
   `#include`/`using`/brace-only boilerplate is excluded), and a naming-mix
   hint. Thresholds: `--fn-threshold` (default 120), `--file-threshold` (800),
   `--dup-window` (6). Treat its output as a worklist to verify by reading.

Do not modify files during review.

## Scope Buckets

Classify changed files before reviewing (same buckets as `hipdnn-review`):

- **Frontend**: `projects/hipdnn/frontend/`, `projects/hipdnn/python/`, public frontend headers, graph/node/attribute wrappers, public C++ or Python API.
- **Backend**: `projects/hipdnn/backend/`, descriptors, engines, plugin loading, pack/unpack logic, backend C API.
- **Data and FlatBuffers SDK**: `projects/hipdnn/data_sdk/`, `projects/hipdnn/flatbuffers_sdk/`, `.fbs` schemas, generated-object wrappers.
- **Plugin SDK**: `projects/hipdnn/plugin_sdk/`, plugin interfaces, ABI/API contracts.
- **Providers**: `dnn-providers/`, provider registration, applicability, execution, workspace, external library calls.
- **Build and infrastructure**: `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, CI, packaging, scripts.
- **Tests**: unit/integration tests, test SDK helpers, GTest fixtures.
- **Docs and tools**: documentation, RFCs, codegen, developer tooling.

Exclude generated files (FlatBuffers `_generated.h`, codegen output) from quality findings; note if the diff hand-edits generated code.

## Quality Checklist

Only maintainability axes. Each finding must point at specific lines and name the concrete improvement.

### Duplication

- Start from `quality_scan.py`'s "Duplicated regions" list, then confirm each by
  reading both sites - the script finds normalized line-region matches but
  cannot tell an incidental match from a real copy.
- Flag near-duplicates the script misses: same structure with renamed
  variables, repeated if-else ladders, repeated descriptor pack/unpack
  boilerplate, repeated provider applicability scaffolding, and near-identical
  sibling files (observed: `data_sdk/types/Fp6E2M3.hpp` and `Fp6E3M2.hpp` share
  large blocks - the pattern to watch for is a new type/node/descriptor cloned
  from a sibling instead of sharing a template or base).
- `ast_grep` helps for exact shapes (e.g. `$A && $A()`), but its C++ grammar
  fails to parse many hipDNN files (heavy templates, macros, the
  `hipdnn_compatibility/cudnn` headers) and silently skips them - never treat an
  `ast_grep` "no matches" as proof of absence; corroborate with the builtin
  `grep`. Broad shape patterns like `switch ($X) { $$$ }` mostly match
  legitimate enum converters, not smells - use narrow patterns.
- Distinguish incidental duplication (leave it) from meaningful duplication
  worth a shared helper (flag it). Repeated test setup is common and often fine;
  flag it only when a shared fixture would clearly reduce churn.

### Shared-Helper Reuse

- Before accepting new local logic, search for an existing helper, wrapper, validator, converter, or fixture that already does it. Use `lsp symbols`/`grep` across the bucket.
- Flag hand-rolled logic that reimplements an existing utility (status conversion, enum<->string, shape math, RAII wrappers, test builders).
- Flag a new helper placed in the wrong layer (e.g. a frontend-only concept added to backend, or a generic util buried in one provider). Recommend the correct shared location.

### Design Patterns and Structure

- Check that new code follows the established pattern for its bucket (how existing descriptors/engines/providers/nodes are structured), rather than a second parallel convention beside it.
- Flag god-functions, deep nesting (>3-4 levels), long parameter lists, boolean-parameter flags that would read better as enums, and control flow that would be clearer with early returns.
- Flag leaky abstractions, public exposure of internals that should be private, and headers pulling in heavy includes they do not need.
- Recommend a new abstraction only when duplication is meaningful and the abstraction matches existing project patterns; do not over-engineer.

### Performance (readability-adjacent, not correctness)

- Flag avoidable allocations and copies: pass-by-value of large objects that should be `const&`, unnecessary `std::string`/container temporaries, copies in range-for that should be `const auto&`, repeated re-computation inside loops that is loop-invariant.
  A quick grep for by-value container/string params (then confirm each is not
  intentionally moved-from): builtin `grep` for
  `std::(string|vector|map|set)\b[^&]*\)` in changed files and check the param
  lacks `&`/`const&`. Verified to surface real cases (e.g. `std::string message`
  ctor params, `std::vector<Rule> rules` setters).
- Flag missing `reserve()` before known-size fills, repeated map lookups that could be a single `find`, and O(n^2) patterns over sizes that can grow.
- Keep these as maintainability/perf-hygiene notes; deep algorithmic-correctness analysis stays in `hipdnn-review`.

### Readability

- Naming - judge against the file's own dominant convention, never a repo-wide
  rule. hipDNN's conventions (from `projects/hipdnn/CLAUDE.md`): backend/data_sdk
  use `camelCase` functions/variables, `CamelCase` types, `_`-prefixed private
  members; the **frontend deliberately uses `snake_case`** (`get_q`,
  `pre_validate_node`, `create_hipdnn_handle`) to mirror the cuDNN frontend API.
  So flag naming only when it diverges from the surrounding file/bucket - a
  `camelCase` helper dropped into a `snake_case` frontend file, or vice versa,
  misleading/inconsistent abbreviations, or a new public name inconsistent with
  its siblings. `quality_scan.py`'s "naming mix" hint is a starting point, not a
  finding: many mixed files are legitimate (frontend wrappers over the C API).
- Clarity: magic numbers/strings that should be named constants, unclear boolean expressions, expressions that would read better decomposed, inconsistent formatting the project's style would normalize.
- Comments: flag comments that restate the code, are stale relative to the code they sit above, or are absent where a non-obvious invariant/rationale needs one. Flag commented-out code and leftover scaffolding/TODOs without owners.
- Comment complexity: a block that needs a paragraph to explain what it does is often a signal to simplify the code; note it.

### File Size and Shape

- Flag files the diff pushes to outlier size for their bucket, and functions that grow past a readable length. Compute rough LOC deltas instead of guessing:
  ```bash
  git diff --stat <base>... | sort -t'|' -k2 -n | tail
  wc -l <changed-file>
  ```
- Flag new files that duplicate the responsibility of an existing file, or a single file accreting multiple unrelated responsibilities that should be split.

### Project-Specific Conventions


These are style rules hipDNN enforces (`projects/hipdnn/CLAUDE.md`); a diff that
breaks one is a legitimate quality finding. Search with the builtin `grep`.

- **Mandatory braces**: `if`/`for`/`while` bodies always use braces, even
  single-line. Flag brace-less bodies introduced by the diff.
- **Header hygiene**: every `.h`/`.hpp` starts with the copyright header then
  `#pragma once` (exclude generated files). Flag a new header missing either.
- **Explicit casts**: the code compiles with `-Wconversion`/`-Wsign-conversion`;
  prefer explicit `static_cast<>` over implicit conversions. Flag new implicit
  narrowing/sign conversions. Separately, `const_cast` that casts away const
  (observed: `const_cast<void*>(...)` in descriptor/plugin code) is a design
  smell worth calling out.
- **RAII**: owning raw pointers must be wrapped immediately; no manual `delete`.
  A leak/ownership defect is a correctness issue - note it and route to
  `hipdnn-review` rather than fixing scope here, but a manual `new`/`delete`
  pair that should be a smart pointer is a maintainability finding.
- **Tests**: prefer `TYPED_TEST` over copy-pasted per-datatype tests; test-suite
  names follow the documented `Test`/`Integration` + optional `Gpu` + Feature +
  optional datatype scheme. Flag duplicated per-type test bodies and off-scheme
  suite names.

### Tooling Notes

Hard constraints learned from running these detectors against the tree:

- **Use the builtin `grep`/`ast_grep`/`glob` tools, not shell `grep`/`find`.**
  Ad-hoc shell `grep` with `-v` filters silently over-excludes (it dropped real
  TODOs in testing). The builtin `grep` is the source of truth for text search.
- **`ast_grep` silently skips unparseable files.** Its C++ grammar chokes on
  hipDNN's templates, macros, and `hipdnn_compatibility/cudnn` headers - it
  reported parse failures on the majority of files in some directories. A "no
  matches" result is never proof of absence; corroborate structural claims with
  the builtin `grep`.
- **No external analyzers.** `clang-tidy`, `cppcheck`, `lizard`, `cloc`, and
  `cpplint` are out of scope by design. `quality_scan.py` reimplements the
  metrics that matter (LOC, long functions, duplication) with pure in-harness
  techniques so the skill is portable and dependency-free.
- **Every metric is a candidate.** The script errs toward recall; the reviewer
  supplies precision by reading each hit. Do not paste raw script output as
  findings.

## Delegation

When the host and user request permit reviewer delegation and the diff is large or spans multiple buckets, split by scope bucket, each subagent applying this quality checklist to its files plus the minimum surrounding context needed to judge duplication and reuse. The orchestrator consolidates: deduplicate findings, merge related ones, and surface cross-cutting quality issues (a pattern duplicated across several buckets, systemic naming drift, repeated missing reuse). Otherwise perform a direct single-pass review.

## Severity

Same ladder as `hipdnn-review`, applied to quality impact:

- **Critical**: rare here - reserve for a quality defect severe enough to block merge on its own (e.g. a large verbatim copy of an existing subsystem, or a structural choice that will force a painful rework).
- **Major**: meaningful maintainability risk likely to cause future defects or repeated churn - significant duplication, wrong-layer helper, a second parallel convention, a god-function.
- **Minor**: localized quality issue - naming, magic numbers, small readability nits, isolated small duplication.
- **Suggestion**: optional refactor or broader cleanup follow-up.

## False-Positive Scrub

Before reporting:

- Re-read each finding against the exact hunk and surrounding file context.
- Confirm claimed duplication actually exists elsewhere in the tree (you searched and found it), not from memory.
- Drop style nits the project's own existing code does not follow - do not impose a convention the codebase rejects.
- Downgrade severity when the issue is real but overclassified.

## Final Response

Use this shape:

```markdown
## Findings

- **Major** `[file:line]` Finding title.
  Explain the maintainability impact and the concrete refactor. Reference the existing helper/pattern/precedent by path when relevant.

## Metrics Snapshot

- Largest changed files / notable LOC growth: ...
- Duplication hotspots: ...

## Open Questions

- ... (including any correctness/leak/compat concern to route to hipdnn-review)

## Summary

Briefly summarize the reviewed scope and overall maintainability of the change.
```

If there are no findings, say so clearly.
