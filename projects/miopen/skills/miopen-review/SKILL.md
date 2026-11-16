---
name: miopen-review
description: "Lightweight review of a MIOpen PR or local commits at a level chosen from contract impact: low (one integrated reviewer, no subagents) or medium (orient step, parallel reviewer lanes for correctness, risk and test coverage, clean code and C++20, and documentation and PR text, plus one devil's-advocate pass), both primed with MIOpen's contracts, with optional bounded reproduction on a GPU host the user supplies and draft comments posted only after explicit approval. Recommends miopen-deep-review for changes that touch persistent or cross-project contracts; much cheaper than that skill."
argument-hint: "[local | <PR number> | <PR URL>] [level:low|medium] [verify:<host or container>]"
allowed-tools: Bash, Read, Grep, Glob, Task
---

# MIOpen Review (lightweight)

A single-round review at one of two levels, low or medium (Step 0 picks it). It keeps no ledger, writes
nothing to the product tree, and posts nothing without approval. For a multi-round review where every
finding is verified and tracked, use `miopen-deep-review` instead; Step 0 says when to recommend it.

Project contracts live in [`projects/miopen/AGENTS.md`](../../AGENTS.md). That file is the source of
truth; lanes read the sections named below rather than relying on a copy here.

Prerequisites: `git`, and an authenticated `gh` for PR mode. Subagents are optional: where the host
has none, run the lanes one after another in the same order.

Install: agents do not discover skills from the repository tree. Symlink this directory into your agent's
skills directory, for example `ln -s "$PWD/projects/miopen/skills/miopen-review" ~/.claude/skills/miopen-review`
(Claude) or into `~/.codex/skills/` (Codex).

## Step 0 - Scope and review level

`ARG` empty, `local` or `.` selects **local mode**: review `BASE..HEAD`, where `BASE` is
`git merge-base HEAD develop`. `ARG` that is a number, `#number` or a GitHub PR URL selects **PR
mode** (`gh` takes either form). Anything else: ask, then stop.

Gather in parallel: PR metadata (`gh pr view <PR> --json number,title,author,baseRefName,headRefName,url,additions,deletions,changedFiles,body`)
or the commit log, and the file-level summary. Stop and report if there is nothing to review or `gh`
fails.

Write the full diff once to a scratch file outside the product tree (`$DIFF`, for example under
`mktemp -d`). Every lane reads `$DIFF` and the working tree instead of fetching the diff again; this
is the main cost saving.

### Review level

Pick the level from contract impact and uncertainty, never from line count: a 700-line test file can be
low risk, a six-line kernel change deep. Judge from the description and the file list, and read code
only where the list leaves doubt. **Default to medium; any uncertainty means medium.** Never lower the
level mid-review. If a lane finds contract impact, raise it and say so in the report. The user can force
a level with `level:`.

| | Low | Medium | Deep (recommend `miopen-deep-review`) |
| --- | --- | --- | --- |
| Suitable changes | Prose, isolated tests, proven host-only leaves with unchanged product contracts | Bounded internal behavior, error handling, tooling, test registration, localized existing-solver fixes | Kernel algorithms, indexing or synchronization; persistent compatibility; new support surfaces; broad selection or default changes; cross-project contracts |
| Reading boundary | Every changed hunk and its complete enclosing block, plus the direct contracts it relies on | Complete changed routines and affected execution paths, direct callers and dependencies, relevant alternatives | Owning subsystem and impacted transitive consumers |
| Execution | One integrated reviewer: you, no subagents, no devil's-advocate agent | Orient step, the lanes in Step 2 (pruned to those that apply), devil's-advocate pass | The deep-review campaign |
| Tests | A meaningful oracle and the smallest changed-path check | Changed-behavior to test map, boundary and error cases, direct-consumer check | Affected variant, build and architecture matrix, deeper coverage challenges |

Examples.

- Low: correcting explanatory documentation (verify its claims against source, and run executable
  instructions where relevant); adding an isolated CPU test without changing tolerances, filtering,
  registration or support claims; refactoring a private host utility after proving its callers and
  observable behavior are unchanged.
- Medium: a localized internal cleanup or error path; a bounded existing-solver applicability boundary
  (check accepted and rejected cases, workspace sizes and performance evidence); changing gtest selection
  or registration (prove which cases are selected, and that they run); a Python review helper (run it and
  its direct consumers).
- Deep: changing registration order (`src/solver.cpp`), `SolverDbId()`, problem keys, or perf-config
  formats; kernel indexing, numerical algorithms, barriers, wave or ISA assumptions, or shared kernel
  parameters; adding an architecture or materially widening supported shapes or dtypes; broad changes to
  Find, heuristics, cache behavior or default solver selection; the public/private/provider ABI
  (`include/miopen/miopen.h`, `src/private/`, `test/public_abi/`); or changing the observable behavior of a
  public API (return status, size or workspace queries, defaults), even when the header is unchanged.

For a deep-level change, tell the user and recommend `miopen-deep-review`. This is a recommendation only:
nothing starts a deep review, and the user decides. If they continue, review at medium and say in the
report that the change is in the deep class. If the user says a deep review has already run, skip the
recommendation, review at medium, and treat that review's findings as known: do not re-report them.

## Step 1 - Orient (medium)

At level low skip to Step 2.

One cheap pass (a single agent, or you): list the changed areas using the table in
`projects/miopen/AGENTS.md` *Key Directories*, note which are product code, tests, build/CMake,
kernels, docs, or generated data, and pick the lanes that apply. Skip a lane that cannot find
anything: no test lane for a docs-only change, no correctness lane for a pure rename of comments.
Record the lane list and the `$DIFF` path in a short context block and give it to every lane.

## Step 2 - Review

**Level low.** You are the single reviewer. Read every changed hunk with its complete enclosing block and
the direct contracts it relies on, and cover the four focus areas below (correctness, risk and test
coverage, clean code, documentation and PR text) in one pass, applying the shared rules. Then go to
Step 3.

**Level medium.** Run the lanes below in parallel.

Each prompt is self-contained: scope description, `$DIFF` path, the context block, the focus below
and the shared rules. Stronger model for lanes 1 and 3 (they decide what is a real defect); a cheaper
model is enough for lanes 2 and 4 (enumeration and cross-checking).

**Shared rules for every lane**

- Report only what this change introduces or worsens. Pre-existing issues are out of scope unless the
  change makes them materially easier to hit; mark such a finding `[pre-existing]`.
- Give `file:line`, the problem, a concrete failure scenario or consequence, and a concrete fix.
  Severity is `blocker`, `major`, `minor` or `nit`.
- Read surrounding code and call sites before claiming a bug. Do not pattern-match on the diff alone.
- Cap the report at about 300 words unless the findings need more. No preamble.

**Lane 1 - Correctness (stronger model)**

- Off-by-one, inverted condition, integer overflow and narrowing, sign errors.
- Lifetime and ownership: use after free, dangling references, wrong `std::move`, handle and stream
  lifetime through asynchronous work (see *Ownership/async* in `AGENTS.md`).
- Synchronization, data races, wrong or missing barriers; GPU specifics: launch bounds, LDS sizing,
  alignment, host/device pointer mixups, wave-size assumptions per architecture.
- Floating point: tolerance, NaN, denormals, datatype-specific behavior.
- Error handling: `MIOPEN_THROW` versus status returns, `miopen::try_` coverage at API boundaries.
- Logic that does not match the commit message, comment or name.
- MIOpen contracts, from `AGENTS.md` *Architecture & Data Flow* and *Code Conventions*:
  - `IsApplicable` must establish correctness, not only performance; default tuning configs must be valid
    without benchmarking; workspace size must not depend on the performance config.
  - Solver IDs are persistent: never renumbered, a removed ID becomes `++id`, `SolverDbId` is kept on
    rename.
  - FindDb (`ProblemDescription::Serialize`), PerfDb (`VisitAll`) and invoker (`MakeNetworkConfig`) keys stay
    distinct; a change to what a key covers invalidates stored data.
  - A solver is reachable only if registered in `src/solver.cpp` and added to the right
    `SolverContainer` in `src/mlo_dir_conv.cpp`; tunable solvers also need the `AnySolver` case in
    `src/fin/fin_interface.cpp`.
  - Host code uses `#if MIOPEN_USE_...`, not `#ifdef`.
  - Environment variables are cached on first read.

**Lane 2 - Risk and test coverage (cheaper model)**

- Find the tests for the changed code (`test/gtest/`, `test/`, `driver/`) and what they cover.
- New paths, branches and error returns without a test; edge inputs (zero, empty, max, odd strides,
  alignment, NaN) not exercised; removed tests or weakened assertions.
- Public behavior, ABI or default-value changes without a test that pins the contract.
- Solver and kernel changes: applicability and numerical-correctness coverage per the fixtures in
  `test/gtest/unit_conv_solver*.hpp`; datatype and architecture coverage versus what the change affects;
  independent references and datatype-appropriate tolerances.
- Test hygiene from `AGENTS.md` *Testing & QA*: suite naming (`CPU_`/`GPU_`, `Smoke`/`Standard`/`Full`/`Perf`/`Unit`
  prefixes), tier placement and added runtime against the tier budget, device eligibility, skips that hide
  untested devices.
- For each gap: the risky `file:line`, the uncovered scenario, and the test to add (file, name, assertion).
  Mark each `must-have` or `nice-to-have`.

**Lane 3 - Clean code and C++20 (stronger model)**

- Clean code introduced or worsened by the change: naming, dead code, duplication, long functions, leaky
  abstractions, magic numbers, style that differs from the surrounding file.
- Modernization only when it is a real win and the file already uses the idiom or the target supports it.
  Check `CMAKE_CXX_STANDARD` first. A short idiomatic loop rewritten with ranges is churn, not a finding.
  Tag findings `[clean]` or `[modern]`.
- Project conventions from `AGENTS.md` *Code Conventions*: formatting (`.clang-format`, Allman braces, 100
  columns), the two-line AMD copyright and SPDX header on new files, `miopen...` naming, logging macros, env
  var declaration helpers.

**Lane 4 - Documentation and PR text (cheaper model)**

- In-code: new public APIs, changed semantics with stale comments, new env vars, CMake options and flags
  missing from `docs/reference/env_variables.rst` or other docs, new failure modes, non-obvious hardware or
  performance choices with no *why*.
- Missing `CHANGELOG.md` entry for a user-visible change; removed or renamed solver env vars.
- PR text (PR mode) or commit messages (local mode): states the motivation (the why, not a restatement of
  the diff); every factual claim matches the diff; scope claims ("only affects X") hold; risk claims ("no
  allocation on the hot path") are not contradicted by the code; the test plan matches the tests actually
  added; a JIRA ID is present where the project convention needs one; stale references to renamed or removed
  things. Tag `[pr-text]`. Trim excess how-detail that duplicates the code.
- Do not nitpick prose, formatting or grammar.

## Step 3 - Consolidate, then devil's advocate (medium)

Level low: skip this step, except to re-read each finding against the real code and drop any you cannot
support.

Merge duplicates, keeping both perspectives and every `file:line`. Drop `nit` findings before the next
pass; they are listed at the end, unchallenged.

One more agent (stronger model) gets the consolidated findings and `$DIFF`, and verifies each against the
real code. For each finding it answers, with a reason:

- **Confirmed**; **Overstated** (real but minor); **Wrong** (not an issue); **Out of scope** (pre-existing).
- It checks: is the bug reachable given the call sites and preconditions; is the fix worse than the
  code; is the missing test covered indirectly; is the modernization really a win; is the doc gap
  inferable from the signature.
- It also lists **blind spots**: issues the lanes should have caught. It does not re-argue Confirmed
  findings beyond one line.

## Step 4 - Optional reproduction (bounded)

Run only if the user supplied a GPU host or container (`verify:` argument or in conversation). Never
guess credentials, and do not acquire hardware on your own.

- Candidates: `blocker` and `major` findings that are empirically testable (correctness, build break, test
  failure, sanitizer-detectable, GPU-specific). Never clean-code, modernization or doc findings.
- Build only MIOpen from the exact code under review (PR head SHA or local `HEAD`) in a separate git
  worktree, never the user's checkout, using the recommended environment in `AGENTS.md` *Development Commands*.
  Build only the targets needed and run the smallest test (existing test with a filter, or a minimal driver
  command). Use `--no-tests=error` on targeted CTest runs. Isolate user DB and cache paths as `AGENTS.md`
  *Runtime/Tooling Preferences* describes.
- Capture the real output. If it reproduces and the fix is small, apply the fix in a scratch worktree, rebuild,
  re-run the repro and the adjacent tests; only then call the fix `verified on <arch>`, otherwise `untested`.
- For a timing claim, also measure the base (merge-base, or `develop` in local mode) the same way on the
  same node and GPU, and report both with arch and the exact commits.
- Outcome tag: `[reproduced]`, `[not reproduced]`, `[inconclusive]` (say exactly why), or `[static]`.
- Without a host, every candidate stays `[static]`. Say so in the summary; do not imply it was tested.

## Step 5 - Report

Print it inline; write no file. Keep it scannable.

```
*Review - <branch or PR #N: title>* (<head> -> <base>, +<add>/-<del>, <files> files)
Confirmed: X | Overstated: Y | Rejected: Z | Blind spots: B
<1-2 sentences: the headline finding, or "no blockers">
```

Then a flat list, sorted by verdict (Confirmed, Overstated, Rejected) and severity within it, one
finding per two lines:

```
<icon> <SEVERITY> - <file:line> - <description> [reproduced|not reproduced|inconclusive|static]
       -> <one-line fix>
```

`✅` Confirmed, `⚠️` Overstated, `❌` Wrong or out of scope (add the reason in parentheses). A Confirmed
finding that did not reproduce is flagged explicitly. A `Blind spots` section follows only if there are
any. End with the level you used and why, and the deep-review recommendation from Step 0 if it applied.

## Step 6 - Draft comments (no posting)

Turn Confirmed and Overstated findings, blind spots and anything reproduced into comments a human can act
on. Drop rejected findings, and drop `[not reproduced]` ones unless the user asks to keep them.

- Per comment: a short italicized title, anchor (`file:line` or general), and one short paragraph
  (what is wrong, why it matters, the scenario); evidence when reproduced (command, arch, short real
  output); the proposed fix; severity as a plain word where useful. Factual and collegial, no filler.
- Use italicized titles for inline and summary comments, not Markdown headings (`#`, `##`, etc.).
  Keep severity out of titles; do not use oversized numbered headings.
- Where the fix is a small self-contained edit to lines in the PR diff, add a GitHub suggestion block under a
  marker line: `Suggested fix - one-click accept (verified on <arch>)` or `(untested)`. Larger or multi-hunk
  fixes are prose or a plain snippet. List the suggestions in a numbered veto table (`S1`, `S2`, ...).
- One top-level summary comment: overview, count of suggestions and which are verified, and a line stating
  the review was AI-assisted.

Show each draft comment with a small control number on its own line, separate from the comment body.
Keep surviving numbers stable when comments are dropped or edited, so later instructions remain
unambiguous. These numbers are local review controls, not part of the text approved for posting.
For example:

```text
Draft 1
Anchor: <file:line>

*Regression test selection*

<comment paragraph>
```

End with a short instruction: `You can say "drop 1", "modify 2: ...", or "veto S1". Nothing posted.`
Iterate until the user explicitly approves the final comment bodies.

## Step 7 - Post (only after explicit approval)

Never post, submit or edit anything on GitHub without explicit approval of the exact final draft;
"looks good so far" is not approval, and a changed draft needs re-confirmation. PR mode only; in local
mode the approved drafts stay in the response.

Before posting, strip the local control numbers (`Draft 1`, etc.), anchor labels, suggestion veto
IDs/table, and editing instructions. Preserve the approved italicized titles, comment paragraphs,
evidence, and GitHub suggestion blocks; use anchors as API metadata, not comment text. Posted inline
comments and the summary have no draft numbering or Markdown headings. This presentation-only
stripping does not require another approval; any change to the approved comment bodies does.

Post inline comments and the summary as one pending review, adding to the user's existing pending
review if there is one, and submit as `COMMENT` unless the user asked otherwise. Never push to the PR
branch. Report the review URL and post nothing beyond the approved text.
