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

6. Check for an existing clang-tidy run first (PR CI, or a local
   `ninja tidy`/`run-clang-tidy` invocation) and read its output before doing
   any manual style review. hipDNN's `.clang-tidy` (209 explicit checks,
   `WarningsAsErrors: "*"`) already gates function size, identifier casing,
   brace style, pass-by-value params, redundant casts, and dozens of
   bugprone/modernize/performance checks - see Deferred to Clang-Tidy below.
   Do not re-derive those findings by hand; cite the clang-tidy diagnostic
   instead.

7. Run the bundled metric pass. The script is diff-scoped and text-based (no
   clang-tidy/lizard/cloc dependency) and covers only what clang-tidy
   structurally cannot: cross-file duplication and outlier file size. Every
   line it emits is a *candidate*, not a verdict - confirm each against the
   source before reporting.
   ```bash
   python3 scripts/quality_scan.py --repo <repo-root> --base <base-branch>
   # or an explicit set:
   python3 scripts/quality_scan.py --repo <repo-root> --files <f1> <f2> ...
   # machine-readable:
   python3 scripts/quality_scan.py --repo <repo-root> --base <base-branch> --json
   ```
   It reports, for changed source files only: large files (`--file-threshold`,
   default 800) and duplicated regions (changed files matched against the
   whole tree, so a copy of untouched code is still found; `--dup-window`,
   default 6; `#include`/`using`/brace-only boilerplate excluded). Treat its
   output as a worklist to verify by reading. The whole-tree index rebuild
   takes roughly 20s per run on the full hipDNN tree - a known cost, not a
   hang; do not re-run it per bucket if delegating to subagents, run it once
   and hand each subagent its slice of the output.

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

## Deferred to Clang-Tidy

hipDNN's `.clang-tidy` (209 explicit checks, `WarningsAsErrors: "*"`, run via
`cmake/ClangTidy.cmake`) already gates these on every compiled target. Do not
re-derive them by hand or via `quality_scan.py` - cite the clang-tidy
diagnostic (from PR CI or a local run) instead of writing a duplicate finding:

- **Function size** - `readability-function-size` (AST-exact: statements,
  branches, params, nesting - not a line-count heuristic).
- **Identifier casing** - `readability-identifier-naming`, configured
  `camelBack` functions/vars/params/members, `CamelCase` classes/structs/enums,
  `UPPER_CASE` enum constants, `_`/`s_` prefixes. This is enforced repo-wide by
  the linter; this skill's own naming guidance (below) is narrower by design -
  judging whether a *convention itself* fits its surrounding file, which
  clang-tidy's fixed per-project casing rule cannot do. See the caveat at the
  end of this section.
- **Brace style** - `readability-braces-around-statements`.
- **Pass-by-value / copies** - `performance-unnecessary-value-param`,
  `performance-for-range-copy`, `performance-unnecessary-copy-initialization`,
  `modernize-pass-by-value`, `performance-inefficient-vector-operation`.
- **Implicit/narrowing conversions** - `bugprone-narrowing-conversions`,
  `modernize-use-integer-sign-comparison` (backed by the project's
  `-Wconversion`/`-Wsign-conversion` compile flags).
- **Redundant casts, redundant control flow, else-after-return, simplifiable
  booleans** - `readability-redundant-casting`, `readability-redundant-control-flow`,
  `readability-else-after-return`, `readability-simplify-boolean-expr`.
- **Duplicate includes** - `readability-duplicate-include`.

**Caveat**: `readability-identifier-naming`'s `FunctionCase: camelBack` is
configured at the repo root with no frontend-specific override found under
`frontend/`, yet the frontend deliberately uses `snake_case`
(`pre_validate_node`, `create_hipdnn_handle`) per `projects/hipdnn/CLAUDE.md`.
clang-tidy runs per compiled target (`clang_tidy_check()` in
`cmake/ClangTidy.cmake`); the frontend is header-only, so its naming may only
be checked indirectly through targets that include it, or not at all. Do not
assume naming enforcement is uniform across buckets - if a clang-tidy naming
diagnostic and this skill's guidance disagree for a frontend file, trust the
project's stated convention (`CLAUDE.md`) over an assumption about linter
coverage, and flag the discrepancy rather than silently picking one.

What clang-tidy does **not** cover, and this skill exists for:

- Cross-file and whole-tree duplication (single-TU tool, cannot see across
  files or find near-identical sibling files).
- Shared-helper reuse and wrong-layer placement (semantic judgment about which
  existing utility elsewhere in the tree should have been reused).
- Design-pattern consistency across a bucket.
- `readability-magic-numbers`, `readability-identifier-length`,
  `readability-function-cognitive-complexity`, `misc-include-cleaner`, and
  `bugprone-easily-swappable-parameters` are explicitly **disabled** in
  `.clang-tidy` (documented in its header as deliberate, not oversight) -
  magic numbers, nesting/branching complexity independent of raw size, unused
  includes, and swappable same-type parameters are open territory this skill
  covers instead.

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
- Flag god-functions, long parameter lists, boolean-parameter flags that would read better as enums (`bugprone-easily-swappable-parameters` is disabled in `.clang-tidy` - two adjacent same-type params, especially `bool`, are this skill's job to catch), and control flow that would be clearer with early returns.
- Deep nesting is a separate axis from raw function size (`readability-function-cognitive-complexity` is disabled in `.clang-tidy` on purpose - see Deferred to Clang-Tidy). Flag it directly rather than folding it into a length check: grep changed files for lines indented 5+ levels (`^(\s{4}){5,}\S` at 4-space indent, or the project's actual indent width) as a leads-generator, then read each hit to judge whether it is a real complexity smell or just a deeply-scoped switch/lambda.
- Unused or unnecessary includes are open territory (`misc-include-cleaner` is disabled in `.clang-tidy`). For a changed header, spot-check a few `#include`d symbols against the file body with `grep` - an include with none of its plausible symbols used anywhere in the file is a candidate to flag, not a certainty (a single grep can miss macro-only or ADL-only uses).
- Flag leaky abstractions and public exposure of internals that should be private.
- Recommend a new abstraction only when duplication is meaningful and the abstraction matches existing project patterns; do not over-engineer.

### Performance (readability-adjacent, not correctness)

- Most avoidable-copy patterns (by-value params that should be `const&`, unnecessary temporaries, range-for copies, inefficient vector ops) are covered by enabled clang-tidy checks - see Deferred to Clang-Tidy. Cite the diagnostic instead of re-deriving it by hand.
- What clang-tidy won't tell you: repeated re-computation inside a loop that is loop-invariant, missing `reserve()` before a known-size fill, and repeated map lookups that could collapse to a single `find`/structured binding. These need reading the surrounding loop, not a grep.
- Keep these as maintainability/perf-hygiene notes; deep algorithmic-correctness analysis stays in `hipdnn-review`.


### Readability

- Naming casing itself is clang-tidy's job (`readability-identifier-naming`,
  repo-wide `WarningsAsErrors`) - do not re-flag a casing violation it would
  already catch. This skill's naming job is narrower: judge whether the
  *convention in force* fits the file, since hipDNN intentionally runs two
  conventions (backend/data_sdk `camelCase`, frontend `snake_case` mirroring
  cuDNN - see `projects/hipdnn/CLAUDE.md` and the Deferred to Clang-Tidy
  caveat above). Flag a `camelCase` helper dropped into a `snake_case`
  frontend file or vice versa, misleading/inconsistent abbreviations, or a new
  public name inconsistent with its siblings.
- Magic numbers/strings (`readability-magic-numbers` is disabled in
  `.clang-tidy` on purpose - open territory). Grep changed lines for bare
  numeric literals in comparisons/arithmetic outside array indices and test
  files: `[^\w.]([2-9]|[1-9][0-9]+)[^\w.]` as a leads-generator: `0`/`1` are
  usually fine, small loop bounds are usually fine, an unexplained `4096`,
  `0x7fff`, or a repeated magic threshold is a candidate. Flag repeated
  literal strings (error-message fragments, attribute-name strings) that
  should be a shared constant.
- Unclear boolean expressions, expressions that would read better decomposed,
  inconsistent formatting the project's style would normalize.
- Comments: flag comments that restate the code, are stale relative to the
  code they sit above, or are absent where a non-obvious invariant/rationale
  needs one. Comment complexity - a block that needs a paragraph to explain
  what it does is often a signal to simplify the code, not document it further.
- TODO/commented-out code: grep changed files for
  `TODO|FIXME` and check each has an issue/JIRA reference nearby; an
  unowned TODO in shipped (non-test) logic is a Minor finding, not just a
  style nit (verified real examples exist in `EnginePluginResourceManager.cpp`
  and `TestEngineConfigDescriptor.cpp`). Grep for
  `^\s*//\s*(if|for|while|return|auto)\b` to surface probable commented-out
  code; confirm each hit isn't documentation prose that happens to start with
  a keyword before flagging.
- Python (`projects/hipdnn/python/`, part of the Frontend bucket):
  `quality_scan.py` only scans `.cpp/.cc/.hpp/.h` - it silently produces no
  signal for a diff that only touches `.py` files. Review Python changes by
  hand against PEP 8 naming (`snake_case` functions/variables, `PascalCase`
  classes - distinct from the C++ conventions above) and the same duplication/
  shared-helper-reuse questions applied to the Python surface.
- Doxygen coverage: `projects/hipdnn/CLAUDE.md`'s Conditional Guidelines
  require Doxygen (`/** @brief */` for classes/functions/files, `///<` for
  enum values) on new or changed public API under `backend/include/`,
  `frontend/include/hipdnn_frontend/` (excluding `detail/`), and
  `plugin_sdk/include/`, excluding generated files. For a changed public
  header in those paths, check each new/changed public declaration has a
  Doxygen comment; a missing one is a legitimate finding here, not a nitpick -
  it is a stated project rule.

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

- **Header hygiene**: every `.h`/`.hpp` starts with the copyright header then
  `#pragma once` (exclude generated files). Flag a new header missing either
  (not a clang-tidy check).
- **Casting away const**: `const_cast` that discards `const` is a design smell
  worth calling out (observed: `const_cast<void*>(...)` in descriptor/plugin
  code); it is distinct from the narrowing/sign-conversion checks clang-tidy
  already runs (see Deferred to Clang-Tidy) - `const_cast` usage is not in the
  enabled check list.
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
- **Lean on the project's clang-tidy config, don't reimplement it.** hipDNN's
  `.clang-tidy` (209 explicit checks, `WarningsAsErrors: "*"`) already gates
  function size, identifier casing, brace style, pass-by-value params, and
  more - see Deferred to Clang-Tidy. `quality_scan.py` itself stays dependency-
  free (no clang-tidy/lizard/cloc invocation) and covers only what clang-tidy
  structurally cannot: cross-file duplication and outlier file size.
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
