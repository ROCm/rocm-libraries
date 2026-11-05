---
name: miopen-review
description: "Lightweight single-pass review of a MIOpen PR or local commits: one orient step, four parallel reviewer lanes (correctness, risk and test coverage, clean code and C++20, documentation and PR text) primed with MIOpen's contracts, one devil's-advocate pass, optional bounded reproduction on a GPU host the user supplies, then draft comments that are posted only after explicit approval. Cheaper and faster than miopen-deep-review; use that one for large or high-risk changes."
argument-hint: "[local | <PR number> | <PR URL>] [verify:<host or container>]"
allowed-tools: Bash, Read, Grep, Glob, Task
---

# MIOpen Review (lightweight)

A single-pass review. It runs one round, keeps no ledger, writes nothing to the product tree, and
posts nothing without approval. For a multi-round review where every finding is verified and
tracked, use `miopen-deep-review` instead (Step 0 says when).

Project contracts live in [`projects/miopen/AGENTS.md`](../../AGENTS.md). That file is the source of
truth; lanes read the sections named below rather than relying on a copy here.

Prerequisites: `git`, and an authenticated `gh` for PR mode. Subagents are optional: where the host
has none, run the lanes one after another in the same order.

Install: agents do not discover skills from the repository tree. Symlink this directory into your agent's
skills directory, for example `ln -s "$PWD/projects/miopen/skills/miopen-review" ~/.claude/skills/miopen-review`
(Claude) or into `~/.codex/skills/` (Codex).

## Step 0 - Scope and escalation

`ARG` empty, `local` or `.` selects **local mode**: review `BASE..HEAD`, where `BASE` is
`git merge-base HEAD develop`. `ARG` that is a number, `#number` or a GitHub PR URL selects **PR
mode** (`gh` takes either form). Anything else: ask, then stop.

Gather in parallel: PR metadata (`gh pr view <PR> --json number,title,author,baseRefName,headRefName,url,additions,deletions,changedFiles,body`)
or the commit log, and the file-level summary. Stop and report if there is nothing to review or `gh`
fails.

Write the full diff once to a scratch file outside the product tree (`$DIFF`, for example under
`mktemp -d`). Every lane reads `$DIFF` and the working tree instead of fetching the diff again; this
is the main cost saving.

**Escalate, do not continue silently.** Tell the user and recommend `miopen-deep-review` when the
change:

- touches solver registration or IDs (`src/solver.cpp`), FindDb/PerfDb formats or keys, or the
  public/private ABI surface (`include/miopen/miopen.h`, `src/private/`, `test/public_abi/`), or
- changes the observable behavior of a public API (return status, size or workspace queries, defaults),
  even when the header and its paths are unchanged, or
- is too large or spread out for one pass to cover, by judgment: many files across several areas of
  product code. Bulk that is mostly one test file does not count.

This is a recommendation only. Nothing starts a deep review, and the user decides. If they continue,
say in the report that the change is in the escalation class. If the user says a deep review has already
run, skip the recommendation and review anyway.

## Step 1 - Orient

One cheap pass (a single agent, or you): list the changed areas using the table in
`projects/miopen/AGENTS.md` *Key Directories*, note which are product code, tests, build/CMake,
kernels, docs, or generated data, and pick the lanes that apply. Skip a lane that cannot find
anything: no test lane for a docs-only change, no correctness lane for a pure rename of comments.
Record the lane list and the `$DIFF` path in a short context block and give it to every lane.

## Step 2 - Reviewer lanes (parallel)

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

## Step 3 - Consolidate, then devil's advocate

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
# Review - <branch or PR #N: title> (<head> -> <base>, +<add>/-<del>, <files> files)
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
any. End with the escalation note from Step 0 if it applied.

## Step 6 - Draft comments (no posting)

Turn Confirmed and Overstated findings, blind spots and anything reproduced into comments a human can act
on. Drop rejected findings, and drop `[not reproduced]` ones unless the user asks to keep them.

- Per comment: anchor (`file:line` or general); one short paragraph (what is wrong, why it matters, the
  scenario); evidence when reproduced (command, arch, short real output); the proposed fix; severity as a
  plain word. Factual and collegial, no filler.
- Where the fix is a small self-contained edit to lines in the PR diff, add a GitHub suggestion block under a
  marker line: `Suggested fix - one-click accept (verified on <arch>)` or `(untested)`. Larger or multi-hunk
  fixes are prose or a plain snippet. List the suggestions in a numbered veto table (`S1`, `S2`, ...).
- One top-level summary comment: overview, count of suggestions and which are verified, and a line stating
  the review was AI-assisted.

Show the numbered draft and iterate ("drop 3", "soften 5", "veto S2") until the user explicitly approves.

## Step 7 - Post (only after explicit approval)

Never post, submit or edit anything on GitHub without explicit approval of the exact final draft;
"looks good so far" is not approval, and a changed draft needs re-confirmation. PR mode only; in local
mode the approved drafts stay in the response.

Post inline comments and the summary as one pending review, adding to the user's existing pending
review if there is one, and submit as `COMMENT` unless the user asked otherwise. Never push to the PR
branch. Report the review URL and post nothing beyond the approved text.
