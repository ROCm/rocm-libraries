<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Lessons learned: the rocKE review and merge pipeline

One source of truth for what this pipeline has learned, so no agent repeats a known mistake or re-discovers a known
fact.

**Who reads what.** The qwen worker reads only the review book; its §11 mirrors the Worker section (rule text, same
IDs). Verifiers read Verifiers and Domain facts before every draft (their team member file says so). The orchestrator
reads all of it at session start, on `resume`, and before every verifier brief.

**Format.** `L-NNN · rule · enforcement`: the rule in the present tense, then how a tool enforces it, when one does (a
guard candidate names its Open problems ID). IDs are stable (0xx worker, 1xx verifier, 2xx orchestrator, 3xx domain, 4xx
enforced, 5xx open); IDs 500–599 are taken: a new entry takes the next free 6xx number, and an entry keeps its ID when
it changes section. A lesson that recurs after it was written becomes a guard with a test (Open problems), then moves to
Enforced. CI and repo facts live here, not in memory (L-222). Facts of one site (its hosts, people, tickets, runner
pools, services) live in the site lessons, `<SITE_LESSONS_FILE>` in the monitor's state, same format, never packaged;
`lessons_for.py` appends their matching sections to each role's extract. The evidence behind each lesson is in the
archived verifier notes (`<WORKING_DIR>/lessons/notes/`); merged and retired texts, and evidence cut from here, are in
`<WORKING_DIR>/lessons/archive/lessons_archive_<week>.md`.

## Worker (qwen) — mirrored in review book §11

Frozen: no new worker rules (a drafter does not follow prose); a recurring worker mistake becomes a `rk draft-lint`
check (L-501).

- L-001 · Never write ACCEPT or REJECT. The header ends `— <light> First-round|Follow-up: <2–5 words of your own>` and the
  footer starts with the same word; use meta.md's `Round:` when present, and its `Line 1:` as your first line, exactly ·
  `rk post` refuses a wrong line 1, header or round (L-401); meta.md stages `Round:` and `Line 1:` (L-513)
- L-002 · Copy nothing from the book's examples: no phrase, number, SHA, job name or ticket tag. Every number comes from
  your own tool output or the PR files; you have no source for a behind/ahead count, so never write one · guard
  candidate: delete the book's examples
- L-003 · "(ran)" only on a point your own run_pytest, rule_scan or merge_conflicts call printed; read_file and grep_code
  are "(read)"; an empty grep proves nothing; the PR body, replies and commits are "author-reported" ·
  `rk draft-lint` flags it (L-501)
- L-004 · Bots / CI only from checks.txt: anything not completed → 🟡 "running" (`<k> ✓ · <m> running · <f> ✗`); name
  each failing job; cancelled is not failed; the PR bot's policy table is not CI; never "unrelated", "pre-existing" or
  "infra": write "cause not checked" · `rk draft-lint` flags it (L-501)
- L-005 · Merge: only what merge_conflicts() printed (`merge: clean (ran)` or `merge: <n> files conflict: <files>
  (ran)`); in a conflict hunk HEAD is the PR's side, the other side develop's
- L-006 · Never write rows for our earlier items (<user_login>, meta.md's `Operator:`) or credit them to
  whoever quoted them: you cannot see them; the verifier adds them · guard candidate: L-502
- L-007 · Before a net-new N<k>, search threads.md, reviews.md and comments.md for its file and topic; anyone's item is a
  prior row under their name
- L-008 · A net-new finding needs file:line at head, the quoted line and a concrete failing input, or it is left out; a
  SKIPPED test proves nothing. Read the cited line at head with read_file first: a file it cannot find is not at head,
  and a line a later commit fixed is a prior row (L-015). A finding whose fix is "none needed" is not a finding
- L-009 · The author is meta.md's `Author:`, never the branch name; drop the title's `type(scope):` prefix
- L-010 · Approvals: `APPROVED at <sha>`, never Stale, Current, FIXED or ADDRESSED. Fixes: `FIXED at head (read)`, with a
  SHA from commits.md only, never a reply's, never one SHA for several fixes; when you cannot tie a fix to one commit,
  write `FIXED at head (read)` with no SHA
- L-011 · Utility: what the PR buys and costs, in your own words; never "correctness-validated", "well-tested" or
  "CI-verified" unless your own run_pytest ran the tests
- L-012 · Output only the review: no "Saved to" line, no note about files (you cannot write files)
- L-013 · CI pitfalls in new code: multiprocessing "fork", command lines over 32,767 chars, argparse errors that print
  `str(exc)` of a path (Windows); a child Python with a bare `env={...}` (Linux). Leave a read or write without
  `encoding=` out of your findings: it is one only if the file has non-ASCII bytes, which your tools cannot show
- L-014 · A <JIRA_PROJECT> tag (`<JIRA_PROJECT>-<digits>`) only from the PR's title, body or commits; never from an example, a bot or a
  GitHub issue number; with none, leave it out · `rk post` refuses a tag the PR does not carry (L-401)
- L-015 · A bot's or reviewer's overview describes the head it was written at: check commits.md for later commits before
  repeating a failure it names; a fixed one is a prior row (`FIXED at head (read)`), never your header
- L-016 · rule_scan hits are hints: perf needs an absolute time or throughput of a kernel run (ratios, defaults,
  fixtures, mocked values, memory sizes, and time limits or budgets in comments, YAML and CMake are not); text I/O needs
  a non-ASCII file (L-013); tracker IDs count only in rocke/ code · `rk draft-lint` flags it (L-501)
- L-017 · One prior row per reviewer, login and state from reviews.md; a CHANGES_REQUESTED review always gets its own row
  (it blocks the merge); a comment headed "CHANGES REQUESTED" is not a review state. A row's item count (×N) is the
  number of that reviewer's threads in threads.md; a quote is copied from threads.md or comments.md, never written
  from memory
- L-018 · Our checks: only what your own tools printed (merge, rules 1–3, your own test run), marked as yours; build,
  ctest, byte identity, goldens and compile come from a checks job you cannot see: leave them to the verifier. A run
  whose cases skipped (no cmake, no torch) is NOT RUN, never clean
- L-019 · A prior item stays OPEN unless the author replied to it or a commit after it touches its line: no reply and
  no later commit is never DEFERRED, MOOT, ADDRESSED or FIXED, and an item raised after the head's last commit cannot
  be FIXED · `rk draft-lint` flags it (L-501)

## Verifiers

### Every draft
- L-101 · Strip the worker's known boilerplate: ACCEPT/REJECT, a missing line 1, a footer round unlike the header, book
  phrases (L-002), example <JIRA_PROJECT> tags, behind/ahead counts, "Saved to", example-shaped Bots / CI and Our checks rows,
  text-I/O and perf findings from rule_scan hints (check the bytes, L-123). A draft that stopped at its wall-time cap
  or in a repetition loop (agent_log ends at the cap; NOT CHECKED or repeated rows) is no draft: its light and "no
  net-new findings" are unjudged · `rk draft-lint` flags it (L-501)
- L-576 · A quality-lane claim is a lead like a draft's: check each `file:line` at head (the quoted code can be right at
  a wrong line), read its run's output file before carrying its "(ran)" (it can have errored), check a "second home"'s
  scope (a precedent for another question is no duplicate), and re-run an internal-pass reproduction through the
  public entry and its defaults
- L-102 · "(ran)" stays only if checks.md or your own run shows it at this head (agent_log.txt lists the worker's
  calls; an empty grep backs nothing) · `rk draft-lint` flags it (L-501)
- L-560 · Name time limits and waits without numbers ("its CTest time limit", "the watchdog waits"): `rk post`
  reads `<n> s` as a measured figure and stops · enforced: `rk post`
- L-543 · No `#<digits>` anywhere in a post body, not only the header: GitHub turns each one into a link and a
  cross-reference on that issue or PR (`#1–#13` pings issues 1 and 13; a quoted "(#NNNN)" pings 8348). Write bare
  numbers ("NNNN", "items 1–13", "the tracking ID") · `rk post --premises` refuses a `#<n>` outside line 3
- L-546 · A CI failure you predict, not one a shard log shows, is never 🔴: post it 🟡 "predicted: confirm on the
  shard", or wait for the log. Read a whole health_status block, never one line of it (L-309)
- L-547 · Never write "stale" in a post, a header or a notice (it reads as a judgement of the author's work); write
  "out of date", "older head", "out of step" · `rk post --premises` refuses it, so a slip costs an edit and a re-run
- L-548 · A red check at a PR head can come from a workflow_dispatch run the author started on the branch (extra
  architectures). GitHub's rollup and required checks read only the pull_request suites; checks.md's "CI at head" and
  `rk post --premises` count both. Split runs by event before calling a head red, and name the dispatch in the Bots /
  CI row; the spec's `FAILOK` takes the verified failing names (`rk ci-fails N` lists them)
- L-549 · A fix line that moves tests to "both engines" must also keep them in a CI lane: CI's installed `rocke_pytest`
  has no `rocke_engine` binding, so a helper that needs it skips there. Say "and still run in CI (installed, without
  the binding)" in the fix, and check a shard log at the next round
- L-550 · An installed CTest entry is not CI coverage by itself: CI runs `ctest -L '^standard$'`, and an entry gets
  that label only through a tier in `ROCKE_ENGINE_test_categories_external.yaml`. Before writing "CI runs it", find
  the entry in the YAML's tiers or in a shard log's test list; our installed run proves only that it passes
- L-103 · Status every item of our previous post and <user_login>'s own comments at this head; retract (⚪) ours that
  were wrong · guard candidate: L-502
- L-106 · Approvals: `🟢 APPROVED `sha``; at an older head "older head; still counts on develop", plus what it does not
  cover (L-218); never "Stale"; never ask for re-approval except after a hold (L-317)
- L-111 · Disprove each worker finding; reproduce reviewers', Copilot's and authors' claims with a probe before FIXED,
  docs-only answers included: a doc can claim less than the gates admit
- L-113 · Post text: point form (status word, 1–3 `<br>•` points of 4–10 words), reason 2–5 words; <user_login>, never
  "the user"; no @-mentions (decorators without `@`); PR numbers without `#` (L-543) ·
  `rk post --premises` refuses "the user", `@` before a letter and `#<n>` outside line 3
- L-115 · Return block 2 and save it as notes.md with `brief:`, `CI pending:`, `lesson candidates:` and `premises:` (head,
  develop, CI counts by state, threads, stop timestamp, probes)
- L-118 · Re-read the timeline just before you finish: reviews with their verdicts (a grader's re-review can settle an
  item while its thread stays open), comments by `created_at` (`since` filters on edits), thread states, pushes, the
  body and `auto_merge`; then merge-tree the head with live develop once more (a same-area landing minutes before
  hand-back turns a clean merge into conflicts) · `rk post --premises` stops on a review or comment newer than `STOP`
  and on a changed conflict set
- L-123 · Judge each rule_scan hit (L-016): perf only for absolute kernel times; text I/O only if the file read is
  non-ASCII (check bytes); hits in moved code may predate the PR (check the merge base). Measured times and sizes count
  too, outside kernel times: Compliance keeps comments, docstrings and commit messages qualitative and allows only
  ratios in the body. Never echo the figures in a post · rule_scan flags a figure with s/min/KB/MB/GB on a line with a
  measuring word (or the 3 lines after one), and every such figure in the body
- L-158 · A fix line is advice the author applies literally, so check it like code before posting · guard candidate:
  L-578:
  - a test move or tier change: run the entry as CI's image does (installed, LLVM tools hidden, no `/opt/rocm`);
  - name every clause the defect needs: each engine, every rule the upstream added (not only the one cited), every ABI
    home (abi.h, abi.py, the canary), each half of a two-part finding as its own row;
  - reusing another open PR's helper names the owner PR and the route (rebase after it lands, or stack), never "copy";
    keeping a public name keeps its signature (a new one under `extern "C"` is an ABI bump); weigh the helper's imports;
  - a skip guard says whether CI still runs the test: CI has no torch, so `importorskip("torch")` means never (L-309);
    move the import into the function that needs it. A new shared home is named only after `rk facts`' overlap owners
    show none (L-553), and a direction (refuse or admit) is probed on the builder with the old validator first;
  - when it names code, compile and lint it under the repo's config (clang-tidy, `-Werror`), and fix the cause, not
    one more list entry (an empty member init like `prepared{}` trips `readability-redundant-member-init`; "extend the
    second list" grows a second home); when it points at an existing helper, compare its types and rounding with the
    code it replaces (an `int` helper loses a 64-bit product); "keep both" conflict advice needs a build of the splice
    (a host test hunk ending inside a function, or two `abi.h` history lines, do not compile as plain concatenations);
  - before accepting "the hook forced it" (a restyle, a rename), read the hook's config and a develop PR's hook log on
    the same file: the config can disable the very checks the sweep claims
- L-159 · A PR that widens either engine's validator (a newly admitted knob value, a dtype that varies on its own):
  lower one spec per new value in both engines and compare, and check `kernel_name()` tells the variants apart. Byte
  identity only sees configs, so an admitted spec no config uses is invisible. The same for a candidate that promises
  a fixed grid (an AOT twin): sweep its knob space, not only its default spec; the registry scope must enforce the pin,
  not a manifest `ignored` list
- L-580 · A vocabulary, rename or "every site" finding takes its site list from a tokenized sweep of code, comments,
  strings, docs and tests, at head and at the merge base, in both engines; never by eye and never by re-running last
  round's narrower scan. FIXED needs the same sweep. Name each site (`file:line`, the exact pair for a duplicate) the
  first time it is posted: a list that grows round by round reads as a moved goalpost
- L-595 · Scope creep across rounds: when each round adds asks beyond a PR's stated fix (design, duplication it only
  continues, wider coverage), a scoped fix stays open for over a week and its author pushes back that the review loop
  bloated it. Start from the PR's title, body and the problem it says it fixes. Block only on Needed-to-merge items
  (wrong results, a break the PR causes, a regression in code it touches, a test it adds that cannot fail, a claim
  false at head), all of them in one round so the author fixes them in one revision; list significant non-blocking
  items once as lit follow-ups (🟠/🟡), ticketed when the author agrees or does not answer, and leave trivial ones out;
  never re-raise a follow-up, or an item the author declined as out of scope. Correctness and breaking changes stay
  non-negotiable (team file §2, §3, §5; `rules.md`, Reviews respect the PR's stated scope)

### Heads, rebases and stacks
- L-104 · Head moved while you verify: cover the delta (range-diff, merge-tree, CI at the new head) and post at the new
  head; restart only for a substantive change. The checks' `wt<N>` stays at the old head until the toolkit re-picks the
  PR: run the probe premises on a sparse worktree at the new head (`worktree add --no-checkout`, `sparse-checkout set
  --no-cone /dnn-providers/hip-kernel-provider/`), never on `wt<N>`
- L-124 · After a rebase or squash, map each fix onto the new range (`git log -S<token> <base>..<head>`); check the
  worker's SHAs against the range-diff before "correcting" them
- L-128 · A push after a nudge can be a rebase only: compare the PR's own commits by range-diff and blobs first · the
  follow-up classifier compares each file's patch as a +/- multiset (L-415)
- L-132 · When develop moves under a draft, merge-tree the head against today's develop; equal rocke subtrees (your
  scratch merge, a base PR's subtree) carry every earlier run, and a rocke change between the checks and the post
  needs the golden gate and the differential on the new merge (Python-only, a sparse scratch merge, about 3 min); a
  develop merge that resolves a golden conflict with develop's hashes is suspect when the PR changes that IR:
  re-record. Read what the develop change adds, not only its conflicts: a new caller, a new home for a helper the fix
  line names, or a baked value turned into a kernarg breaks the PR's code outside the hunks · guard candidate: L-529
- L-140 · A rebase that replays scoped commits drops edits that lived only in merges: check `git show --remerge-diff
  <old merge>` at the new head ("not our ref <sha>" on the blob:none mirror is noise)
- L-157 · Check a rebase in one step: `git merge-tree --write-tree --merge-base <old base> <new base> <old head>`, then
  diff that tree with the new head: only the author's rebase edits remain
- L-105 · Stacked PRs: range-diff the PR's own commits against its base; merge-tree the base alone before blaming this PR
  for a conflict or CI; no PR Bot, Freshness or pre-commit runs. A base other than develop: the checks describe
  develop's diff until L-599 lands: judge `git diff --name-only $(git merge-base <base> <head>) <head>`, run the
  base's own interfaces, and mark develop-relative sections NOT RUN
- L-135 · A history finding (numbers in commit messages) is checked against the history at head, never carried by
  patch equality: a squash can fix it · guard candidate: L-521
- L-136 · Every follow-up: compare the body, and an armed auto-merge's stored title and message, with the PR's state
  (the squash commits them); stale is 🟡 · the checks compare an armed auto-merge's stored message with the live one
  (L-412)

### Checks, CI and coverage
- L-571 · A failing build or clang-tidy job: list every `error:` in the whole log (`grep -n 'error:' <log> | sort -u`),
  not the first 25 lines `rk job-log` shows: the author fixes the one we name and the next push stops at the next. A
  red on a PR that touches provider C++: open the Linux hipDNN superbuild log first (L-306); lint new
  preprocessor-guarded code with the feature off as well
- L-107 · A check that ran nothing is NOT RUN: read each denominator and SKIP reason. Forms: llvm-objdump off PATH
  (SUBSKIP, parent PASSED), module-level `importorskip("torch")`, the forced llvm22 (L-154), a conftest sample, a capped
  list, a Python-only builder (L-130), a harness outside pytest, your own script comparing nothing · guard candidate:
  L-506
- L-108 · Attribute every CI failure (job, step or test, evidence). A red `… Summary` beside a later run at the same head
  is a cancelled run (a push, or the bot's `organization: ROCm` label, which can land seconds or most of an hour after
  open); its finished shards still count. After TheRock's auto re-run, read `actions/runs/<id>/attempts/1/jobs`; while
  a run is in progress use `gh api …/actions/jobs/<id>/logs` (finished CTest entries are already there). Check
  `rk breakages` and L-314 first. A hand re-run of failed jobs hides attempt 1 the same way: the check-run view shows
  only the queued re-run, and jobs that passed in attempt 1 reappear with attempt-2 ids but attempt-1 logs and times.
  Before calling a red head "infra", read `attempts/<n-1>/jobs` for every job being re-run: attempt 1 can hold the PR's
  own failures. A test red on one shard and ✓ on the other in one run is the runner's (both run the same rocKE CTest
  list); a short CTest "***Failed" can be a pytest-timeout ("+++ Timeout +++") · guard candidate: L-508
- L-109 · CI still running at post time: "running" with counts, and `CI pending:` in your notes (L-209)
- L-110 · ABI: develop's number and "takes develop + 1 when it lands"; never a lane number, never the tool name `rk abi`
  in a post (the order is in your notes only). abi.h's "when to bump" rule decides, not the layout: a struct that
  gains, loses or reorders a field needs a bump even when the field lands in tail padding and sizeof is unchanged (an
  old caller leaves the byte unset). Offsets and sizes in 2 s: `clang++ -fsyntax-only -Xclang -fdump-record-layouts`
  on develop's and the head's headers (pull them with `git show <sha>:<path>`, never `git archive`); measure a history
  line's "changes its size" before citing it. A develop PR that makes the same bump merges abi.py silently and
  conflicts only in abi.h: the fix names every ABI home; a test pin `ROCKE_ABI_VERSION == N` breaks on the next bump
  (develop pins with `>=`) · guard candidate: L-579
- L-119 · Credit "runs in CI" only from a shard log's test list (CI runs the installed tree under `ctest -L` on the
  tier labels; pull_request runs take `^standard$|^quick$`); cases no lane collects are not coverage (L-310). The
  scheduled nightly builds the provider without rocKE, so a comprehensive-only entry runs in no CI job. GPU lanes:
  read the shard log's skip reasons and the run's job list (gfx950 jobs come and go; an author's workflow_dispatch run
  can add some, L-548); develop's push runs carry rocKE shards only some days. A test that asserts and then calls a
  tool-gated helper reads SKIPPED with its assertions run, and a both-engine test can check the Python half before its
  `importorskip("rocke_engine")`: run a Python mutant installed before calling it untested in CI · guard candidate:
  L-506 (`rk lane-of`)
- L-131 · Attribute a PR's CI failure by its run's merge ref: compare the develop parent of the failing run with the
  last green one (L-323)
- L-142 · CI's `rocke_pytest` runs the INSTALLED tree: a test that reaches a source-only path passes here and fails
  there. A moved or NEW test runs installed for the first time: installed, `platform_root()` is `bin/`, so
  `platform_root()/python/...` and `dsl_docs_dir()/...` do not exist (package data: `importlib.resources.files(
  "rocke.<package>")`; docs: skip with a reason when absent), a `Path(__file__).parents[N]` path breaks, and a child's
  PYTHONPATH built from the source layout fails; run it there with the platform tools hidden before calling it FIXED
  (L-158). By hand: the recipe card's installed run (R5), with a moved-aside negative control · the checks' "Changed
  tests in the installed tree" (L-526)
- L-150 · Counts: CTest's "100% … out of N" includes skips; pytest 9 prints a subtest as SUBFAILED[label] and passes the
  parent, so "2 failed" can be one test; the worker counts per invocation. A probe that greps pytest output counts
  SUBFAILED too: one that counts only FAILED lines can read REPRODUCES on a fix
- L-154 · "Failing (on the merge only)" = fails on the merge, passes on develop; says nothing about the head. The checks
  force llvm22: llvm23+-only tests skip there, and a test that skips at CI's llvm24 passes there; run the changed tests
  once unforced on this LLVM 24 host · guard candidate: L-535
- L-138 · An A/B timing counts only when both sides run at once under the same load
- L-161 · A PR that caches a kernel on more arches than its numeric test's arch gate admits ships kernels with no
  runnable numeric test there: compare the test's gate (`_WGRAD_ARCHES`-style tuples, skip markers) with the AOT grid's
  arch list
- L-165 · A PR that adds integration-test bundles must also update the exact-set tests over the bundle tree: hipDNN's
  `TestReferenceOpCoverage.RaggedBundlesAreNotListedAsGaps` names every quick SdpaFwd BSHD bundle. Our checks do not
  build hipDNN and the author's `verify_golden_bundles` passes, so grep `dnn-providers/integration-tests/tests` for
  literal bundle ids whenever the diff adds bundles, before CI says so

### Merge semantics, goldens and probes
- L-563 · A conflict with develop is 🟠: the author merges develop, keeping what the row names from each side. 🔴 needs
  a defect that breaks develop, CI or users once merged, such as a resolution that fails tests or a mismatched ABI pair
- L-112 · Merge semantics on the merged tree: grep removed or renamed names, port move-vs-edit hunks, re-bless what
  develop added, find develop files a hand merge reverts, build any resolution you recommend, grep develop for
  positional brace-inits of a struct the PR extends (Windows -Werror)
- L-143 · A clean merge can be wrong: re-record each touched family on the simulated merge; compare parity config ids
  on both sides; on arch-domain or target changes run `platform/tests/tools/test_arch_domain_artifact.py` and
  `platform/tests/core/test_arch_domain_lane.py` on the merge. The author's own clean develop merge does the same at
  the head: the PR's added golden cases keep hashes from before develop's change and CI goes red. Attribute it by
  reverting the suspect develop commit's hunks and recomputing; the fix is a `--write` that moves only the PR's
  entries · the checks run the arch-data companion tests (L-437)
- L-116 · A PR that changes the flavor columns or drops an alias re-records everything develop added since its base;
  first check whether develop already resolves the host to the new lane (then the gap is develop's)
- L-125 · A PR that drops its bundled copy of a change develop took separately can lose review fixes that lived only
  in the copy
- L-130 · Byte identity covers only families with parity pairs: a Python-only builder, or one the C++ engine lowers via
  serialized IR, is uncovered, and the `.ll` gate inlines constants, so a constants-only order change never shows
  there. "Supersedes N" can park a develop fix behind a stacked base. Which configs reach a line, the Python twin and
  a whole AOT grid: the recipe card
- L-133 · Docs PRs: check each described behaviour and each replay command path against develop at merge
- L-137 · A PR that ships an AOT grid: compile the whole grid for spills (SGPR too), not only the parity table, and name
  the spilling default picks · guard candidate: L-522
- L-145 · A catalog or instance-list change: compare record SETS (not counts or order) across every pinned inventory,
  the shipped KDPs' instance ids included
- L-147 · A fix to one path: probe sibling paths in both engines separately (`_lower_kernel_to_llvm_python` vs
  `rocke_engine.lower_serialized_ir`; `backend="both"` compares lowerers, not C++ builders); for "knob A implies B",
  compare a seeded spec sample at head and on develop
- L-152 · A saved probe is a test the next round trusts blindly: it keys on the defect, never on a spelling a correct
  fix may change · guard candidate: L-577:
  - it calls the code under test (no copied logic) through its entry point or symbol (a test helper as
    `tests.<package>.<module>`, not a file path), matches names on word boundaries, and simulates the host inside it
    with each absence taken from a real shard log;
  - it never keys on a token, a function or hook name, a file, a literal count, a clamp's spelling, a test name or an
    exception type a correct fix may keep or change: a one-home (DRY) probe traces the data or the logic's home, a
    refusal probe accepts any loud refusal, a classifier reads each item's own field (`spec.path`);
  - it parses with the real parser (YAML anchors and block scalars, argparse, pytest's collection), counts `SUBFAILED`,
    and maps a crash or import error to exit 3, never 1;
  - it pins the round's DEVBASE, exits 2 on a tree without the PR's change or the develop change a merge-only finding
    needs (by ancestry, outside conflict hunks), and compiles shipped objects only for the arches their catalog ships;
    its control set leaves out inputs the upstream rule itself rejects, picked from the data by that rule;
  - before saving, run it on DEVBASE (2), the tree (1), a simulated fix of our fix line in another spelling (0) and
    with the guard deleted (1); re-key it when our fix line changes, retire it two rounds after FIXED, and rerun its
    controls when its verdict contradicts the author's reply
- L-163 · After a stacked base lands as a squash, the dependent PR's develop merge keeps every file the base deleted
  before landing: a one-sided add, no conflict, so neither git nor the checks flag it. Diff the PR's added files against
  the base PR's final tree
- L-166 · A land order between two open PRs can merge without a conflict and still not compile: grep the trial merge
  tree, outside the conflict hunks, for the functions and macros either PR deletes or renames, then syntax-check the
  touched TUs with the conflicts resolved to one side. Re-fetch the sibling heads right before the trial (a head can
  move minutes after the first fetch). Say which PR does the port in the plan's order (`rk merge-plan`), not which one
  owns the helper
- L-167 · A 🔴 build finding quotes the compiler's own line: the job, `file:line:col: error: … [-Werror,-W…]`, and the
  flag or config that makes it an error. Say when the cited line is only where the compiler reports it (the closing
  brace of a multi-line init), and keep FIXED items out of the failing list: an author whose build has the feature off
  cannot see the error otherwise
- L-160 · Re-bless claims ("only our entries moved", "same ISA") are cheap to verify: leaf-diff the golden JSONs against
  develop, then compile old vs new IR of the changed configs with the checks' emitter shim (`p_emit_compile.py`; about
  1 min for 24 configs × 4 flavors). "Goldens re-blessed" can mean the differential golden only: check the
  representative shards too · the checks print a re-record command that sets PYTHONPATH
- L-155 · A new arch-domain decl key needs a row in every column (`name_absent` where that LLVM lacks the intrinsic, plus
  `canonical`): 🟠 until then
- L-156 · Perf numbers in a PR body's edit history: never point at them in a post; tell <user_login>
- L-569 · The code quality pass has no call target, and runs on every follow-up except develop merges and docs: a pass
  with a call target stops with calls left and loses about a third of the 🟠, while the same pass without one finds
  nearly all at about a third more of its tokens. A verifier aims at about 40 calls but never skips a test run a finding
  depends on
- L-570 · Judgement misses that recur with calls to spare: mutate every check the PR adds or touches, not only those a
  brief or a prior row names, and every branch of a multi-branch fix (a mutant of one branch can pass while the
  both-branches mutant is caught); check a both-engine claim ("texts match case for case") on the merge when develop
  changed the same rules, not only at head; a PR that says "Fixes #N" while sibling sites keep the bug owns them: a
  Needed-to-merge finding (its claim is false at head); an old pattern it only adds a reader to is a follow-up, not a
  pre-existing lead (L-595)
- L-567 · A fix for our own item gets the code quality pass like any code change, and a meaning change across both
  engines needs a deep pass (L-569): the 🟠 a follow-up loses are quality findings inside fixes for our items
- L-581 · A mutant counts only when a positive control shows it is live: `import a.b as c` can bind a re-exported
  function (patch through `sys.modules`); a test run from another tree loads that tree's conftest first; `git checkout
  -- f` restores a staged mutant (use `HEAD --`); a C++ mutant that drops a parameter's use fails `-Werror` (keep
  `(void)x;`); SKIPPED is not caught; `pytest -k` matches module names and `-rs` prints no FAILED lines (select by node
  id, read the count, use `-rfE`). Run both remove-the-guard and silence-the-guard mutants.
  A sweep survivor is a lead (L-598): before "untested", re-run it with torch and the shared engine against every
  CI-run file that imports the module, with no develop-known failure in the set

### Running things here
- L-114 · Python `-B`, never `-I`; never `HIP_VISIBLE_DEVICES=-1` (L-319); `/opt/rocm/core-10.1/lib/llvm/bin` on PATH
  before HSACO gates (L-309); `worker_grace.py` before heavy steps
- L-134 · Merged-tree runs in a scratch `git worktree add` + `git merge --no-commit` (sparse set with `library/`); never
  a `git archive` export, not even of a header directory (read files with `git show <sha>:<path>`, list them with
  `git ls-tree`); never retry a refused run another way, mark the evidence (read)
- L-141 · No `rm -rf` or bulk deletes in run scripts (one refusal loses the whole script); `git worktree remove` in a
  separate script; re-check scripts copied from earlier rounds
- L-162 · `-x c++` applies to every later input: `c++ -x c++ drv.c tu.o lib.a` makes cc1plus parse the archive as C++
  and hang; put `-x none` before objects and archives
- L-164 · Never read `FETCH_HEAD` in the shared mirror `<ROCKE_SRC_DIR>`: other verifiers and the checks fetch
  into it concurrently, so it can name another branch's head right after your fetch. Use `refs/remotes/origin/...`,
  `pull/N/head` fetched into a named ref, or `git ls-remote`. The mirror is a blob:none clone: `git grep <commit>`
  over it fetches every blob and stalls; grep a checked-out worktree or restrict the paths (prove a path-scoped grep
  complete with `GIT_NO_LAZY_FETCH=1 git cat-file --batch-check`). Never run `git config` in a worktree of the mirror
  (it writes the mirror's config: use `git -c`), never fetch a PR ref without `+` (`pull/N/head:refs/x` keeps the old
  ref after a force-push, silently), and never re-run a step of the shared build (`ninja -t commands`' archive step
  rebuilds the checks' `librocke_core.a`)
- L-582 · A script checks every value it computes before using it: capture `rc=$?` on its own line (inside
  `echo "$(cmd) rc=$?"` it is the substitution's status); `git commit-tree` with no identity prints nothing, and
  `git worktree add ""` then checks out the mirror's HEAD, so test `[ -n "$c" ]`, `rev-parse HEAD = $c` and one file
  only the PR has; resolve list paths before `cd` and refuse "0 files" (pytest with no file arguments collects the
  whole tree); key CMake scratch dirs on the full tree path or SHA, not its basename; print counts and a control for
  every compare (an empty ISA or IR file compares "same"); compute a sweep's size before enumerating it on the shared
  host

## Orchestrator

- L-201 · Verifier facts come only from `rk facts N` (head, every force-push, our posts, the ABI lane, develop
  issues, reviews, open threads, <user_login>'s comments in full): `rk review-prep` writes them to `facts.txt` in the
  review directory, and the brief tells the verifier to print that file whole as its first read. The orchestrator
  never recalls, retypes, summarizes or cuts them (a `head` or `sed` cut can hide open threads or comments); it
  describes a PR from its diff stat, not a guess, and takes our open items from the file's "our last post" section.
  Thread lines in it are cut at about 200 characters: never tell a verifier the threads are there "in full"; say to
  fetch the bodies with `gh api`. A push is described only from `rk classify` and the range-diff (after a rebase every
  commit carries a push-time date, so "the last N commits" is not "N new commits"); a develop break inside the PR's
  CI merge ref "needs a push", never "a re-run" (L-323); the develop SHA comes from `checks/develop_sha`, not from the
  overlap scan · `rk facts` (L-536)
- L-202 · A shell step: a quoted heredoc or a script file, never inline shell text. A helper script is named uniquely
  for its PR and round, never a renumbered copy; never batch a Write with the Bash that runs it; post with
  `rk spec-post`, which writes the spec and checks the verifier's premises · `rk post --premises` re-checks them (L-509)
- L-224 · A premise trip is a stop, not a re-verify: if CI only added re-run attempts or finished jobs, refresh Bots / CI
  from the live check-runs and re-run; re-verify on a new failure, comment or push. Leave hygiene greps to `rk post`
  ("404s" read as seconds). While CI runs, post its counts filled in at post time (placeholders like `%OKCOUNT%`; an
  `@` placeholder trips the mention check) and premise only on: no new failure, still running, the named result not in
  yet. Count per check name at its latest attempt, as `rk ci-fails` does: raw check runs include superseded attempts,
  so an exact count trips with nothing changed. Premise on failures, cancels and the running set; fill the ✓ count at
  post time · `rk post --premises` checks failures, cancels and the running set, and fills the ✓ count (L-509)
- L-203 · Never block the live monitor on AskUserQuestion; ask in plain text and keep ticking
- L-204 · A user question or correction the board should have answered is a toolkit gap: fix it the same session, with a
  test
- L-205 · Put a user preference where its writer reads it (book §5, verifier file), not only in memory
- L-206 · Ship a known mechanical fix within a day, or log the deferral here with its weekly cost · the weekly review
  records which proposed guards were built
- L-207 · When prose recurs, enforce it with a test, fixes to unversioned tools included (L-226)
- L-208 · Never hand-write a list or fact the toolkit can print (families, ABI lanes, CI picture, brief facts)
- L-209 · After a post with CI pending, re-check that head's CI next ticks; post one CI update only if a failure is the
  PR's · guard candidate: L-507
- L-210 · Keep offered <JIRA_PROJECT> candidates and develop leads on one persisted list; show it to <user_login> once a day · guard
  candidate: L-511
- L-213 · Judge the worker out of the box: a lint annotates, never rewrites; `rk save N verify` after every post ·
  `rk post --premises` runs `rk save N verify`
- L-215 · After each post, archive notes.md to `<WORKING_DIR>/lessons/notes/<day>_pr<N>_<round>.md` ·
  `rk post --premises` archives it
- L-216 · Subagents cannot write a file named `report.md`: name outputs otherwise, or save the returned text
- L-217 · Two numbers for one thing come from one function, with a test comparing the views
- L-218 · When an approved PR absorbs another PR's code, tell <user_login>: the approval then carries code nobody
  approved; never ask for re-approval · guard candidate: L-528
- L-219 · `rk post` allows six posts an hour: check the ledger before a burst; merge candidates first
- L-220 · Start the review driver only with `run_in_background`, never `&` (a detached copy keeps the lock and never
  notifies)
- L-591 · Brief the verifier and the code quality pass with distinct worktree prefixes under the review directory
  (`wt_v*` for the verifier, `wt_q*` for the pass): when both create `tmp/v<N><round>/wt`, they share one worktree and
  one edits the other's files · guard candidate: `rk review-prep` makes the `wt_v` and `wt_q` paths, and the GUARD hook
  refuses a worktree add anywhere else
- L-593 · Copy FAILOK and RUNOK names from `rk ci-fails N` exactly as it prints them: it cuts long job names with
  `...`, and the full name from GitHub never matches, so the premise stops the post
- L-574 · One driver run can print several `REVIEW_READY` lines (it pipelines: a draft finishes while the next
  starts). Read the whole driver output for them, not its last lines, or run `rk review-status` (its "awaiting
  verification" lines) at every driver exit: a ready review otherwise sits unseen
- L-575 · A `git worktree add <sha>` run from inside the shared mirror, with no path, creates a worktree directory
  named after the SHA in the mirror's root. `rk post --premises` then stops on `git diff <mb> <sha>` with "ambiguous
  argument: both revision and filename". Remove such directories with
  `git -C <ROCKE_SRC_DIR> worktree remove --force <dir>`. Briefs say "under your review directory"; check
  `git -C <ROCKE_SRC_DIR> status --short` for `??` entries after agent runs · guard candidate: pass `--` in the
  toolkit's git diff calls, and refuse a worktree path inside the mirror (L-588)
- L-222 · Host, CI and repo facts go into this file the day they are learned; memory files point here
- L-223 · Start a review by hand (`rk review-start`) only on a settled head; `--now` only with a reason ·
  `rk review-start` refuses an unsettled head (L-534)
- L-225 · Before `rk fix-link` or "Fixes #N" advice, read the issue's later comments: issues widen, and a "Fixes" for
  part of an issue closes it with failures still open
- L-226 · A fix to a tool outside the toolkit repo (`<WORKING_DIR>/tools`) gets a test and a commit, never only a `.bak`
  · guard candidate: L-537
- L-227 · Every review has a code quality pass: the rocKE Code Quality Reviewer runs beside the verifier, who folds its
  findings in; the post gets a Code quality row. Non-negotiables: single responsibility; DRY (one home per fact, search
  for an existing helper first, merge duplicates); primitives first with clear API layers; interfaces judged from <user_login>'s side; descriptive names; every claim backed by a test; root-cause fixes in both engines. A clear violation the
  PR introduces is 🟠 and blocking (blocks G1 and trips the auto-merge guard); a pattern it merely continues, or an
  improvement beyond the change, is a follow-up (L-595). Run the same input on develop before grading a quality gap
  blocking: a gap the PR neither adds nor worsens is a follow-up at most · `rk post --premises` refuses a post whose Code quality row is missing
  or pending (review_procedure.md step 2)
- L-229 · Post premises: a CI count covers check runs only, never commit statuses; `git merge-tree --write-tree
  --name-only` lists the conflicted files from line 2 to the first blank line (after it come the messages); a saved
  probe takes the tree root as its argument (exit 2 without it) · `rk post --premises` re-checks the CI count and the
  conflict set
- L-230 · The harness can refuse a lane's write of its own report, findings or notes file ("subagents must return
  findings as text"), while post files and quality.md usually go through. The lane then returns the content inline:
  save it to the file yourself at once (the weekly review reads notes.md), and point the next brief at it
- L-552 · When you forward quality.md, ask for the lane's findings by their tags (`blocking` as N rows, `follow-up` as
  F rows with their 🟠/🟡 light) and leave the split to the verifier (team file §2, §7: every blocking finding is an N
  row; trivial ones are dropped). Never write "add Q3–Q5 as rows": a verifier follows the brief over its own split
- L-583 · A model under assessment drafts in shadow beside the live pipeline: nothing it writes reaches a post, and
  the verifier scores it only after the post is written. A live stage's model and line 1 change only on <user_login>'s
  instruction naming that change ("assess X live" means shadow). A shadow draft is scored only when its path reaches
  the verifier before hand-back; a late or capped shadow scores `none`, so watch `rk drafts`' late and missing counts
  daily while a shadow runs, and stop it when it scores nothing
- L-584 · The dashboards carry only the cards <user_login> asked for (commands.md lists them; cost tables sit on the
  admin dashboard's Review spend card); a new card, section or nav entry needs their request, and toolkit briefs say
  so. Other analysis stays in `rk` commands, never in chat
- L-585 · A model comparison or pipeline test names the model ID each call really used (the transcript's `"model"`
  field); the `sonnet`, `haiku` and `opus` aliases resolve through Windows user environment variables read at start-up
- L-586 · Review depth follows a PR's size and impact (a small change to a shared header, the ABI, engine core, CTest
  or tier lists, goldens or an interface other open PRs use counts as large); the cost target is a fleet average.
  Never cut coverage silently to meet a number: fail, or name what was not covered
- L-553 · Put the recorded overlap owners in the verifier brief: the `state/overlaps_decided.json` keys whose members
  include the PR, or the merged PR whose follow-ups it carries. The checks' overlap section shows a recorded owner only
  when the PR is a member, so a verifier can re-decide a group recorded for the PR whose follow-ups it carries · guard
  candidate: `rk facts N` prints the decisions whose keys the PR's overlap groups share
- L-554 · Look at a dashboard fix before you call it done: render the page and read the screenshot (a fix read from the
  CSS alone can miss the cause, such as a class name another rule also matches). Headless Chrome works here:
  `chrome.exe --headless=new --hide-scrollbars --window-size=1500,860 --screenshot=<png> file:///<html>`, and a copy
  with `html{margin-top:-Npx}` brings a lower card into view. Give a page element's class a name no other rule uses
- L-556 · A tick that keeps `state/lock` for minutes with its CPU busy (`/proc/<pid>/status` State R): send it
  `kill -INT <pid>`. It prints its Python traceback and frees the lock; it saves nothing, so the next tick redoes its
  work. A regex that repeats a group of overlapping alternatives can hang on one long input: make each token `(?>…)`
  and test a 400-token failing input
- L-558 · An infra incident on a job that develop's path filters skip cannot clear itself: `ci_cleared` needs a later
  develop SHA that ran the job green, and later commits that skip the job leave the queue frozen. Once the log shows
  infra: `rk incident-note <id> summary|resolution|void "…"`, then `rk incident-close <id>`. The red SHA can be older
  than the incident's `last_green`, so check the order with `git merge-base --is-ancestor`. When the run's jobs list
  answers 502, find the job with one small request: `commits/<sha>/check-runs?check_name=<url-encoded name>` · guard
  candidate: a develop red whose failed rocKE jobs all ran on a configured degraded pool, with only CTest/pytest
  timeouts in the log, opens no incident and prints one line
- L-559 · Side-project agents must not reach this host's live services: a test that falls back to the default model
  address `127.0.0.1:11434` makes Ollama load another model and evict the live reviewer's (live qwen drafts then fail
  with HTTP 500), and backticks sent as inline shell text run as commands. Brief such agents: tests point every model
  address at a closed port (`127.0.0.1:9`) and put a stand-in `ollama` first on PATH; no inline shell text with
  backticks or `$( )` (a quoted heredoc or a script only); no `docker` commands unless the brief names them
- L-601 · Briefs for background research or experiment agents say "do not start subagents", and live reviews keep two
  subagent slots free: the session caps concurrent subagents, and one research agent's helpers can block a verifier
- L-602 · Never print environment variables, tokens, keys or webhook URLs, and never copy a credentials folder: name a
  secret by its config key or file, and mask any token a tool echoes

## Domain facts that change

### Develop layout and lanes
- L-301 · Test lists: C++ registrations in `platform/tests/cmake/*.cmake` (plus `_FRAGMENTS`), library suites in
  `library/tests/ctest_lists.cmake` (included by `platform/CMakeLists.txt`; `rocke_pytest` passes `--ignore
  tests/library`), library C++ host tests in `library/cpp/tests/CMakeLists.txt`; tiers in
  `ROCKE_ENGINE_test_categories_external.yaml`, matched by full-match regex (`shared/ctest/parse_ctest_categories.py`;
  a duplicate YAML key keeps the last). The registration contract test runs in CI's installed `rocke_pytest`, and
  its tier check skips without PyYAML
- L-302 · Goldens: the representative golden is per-family shards (`tests/golden/rocke_representative_ir_sha256/`; conv
  families under `library/tests/golden/`): re-record only the touched family on the merged tree (`--write <dir> --only
  <fam>`, then an empty `git diff` and a corrupted-hash control: `--check` refuses `--only`); a PR that changes
  `aliases` needs a full `--write`. The differential canonical section is CI-tiered too: re-bless it with ll
  (`run_diff --mode ir --canonical --python-only --record-golden --only <fam>`; check with `--check-golden`)
- L-303 · Develop's binary and recipe ABI numbers move several times a week: read them from `rk facts`' `rk abi` section
  only; lane numbers are tentative (`rk abi`), and an approved PR in the binary lane can lose its number to develop
  before it merges (each renumber also falsifies an armed squash message that states it). ABI consumers outside rocke:
  hipDNN kernel-ingestor packs (`*Native.cpp`), `test_hkp_pack_rocke.py`, `library/tests/test_sdpa_reference_contract.py`.
  A new ingestor pack must join `packDispatchSymbols` in `TestKernelIngestorEngine.cpp`, else both hipDNN superbuilds
  fail. A provider-wide plugin API bump opens every engine in it to newly gated graph features (ragged offsets, custom
  alignment); each ingestor engine also has its own schema gate (`GenericPlanBuilder::isApplicable` →
  `understandsGraph`, its UED `sdk_version`): check each matcher against both on the merge
- L-304 · Four LLVM columns (llvm20/22/23/24), llvm24 its own. CI's compiler is AMD clang 24.0.0git, so CI resolves
  llvm24. A PR adding cases, keys or targets needs llvm24 data on the merge; a shard that keeps an `llvm24→llvm23` alias
  fails to load on the merge (re-record it); a family serving older flavors only is refused on CI's compiler and its
  goldens and gates skip
- L-305 · Differential goldens run in CI (`rocke_differential_golden_ll`/`_canonical`): an IR change without a re-bless
  fails; unblessed configs count toward `_ll`'s 300 s limit; compare NEW counts with develop's at the same SHA
- L-310 · A test runs in no CI lane when it is: in no CTest list (`platform/tests/cmake/*.cmake`,
  `library/tests/ctest_lists.cmake`, the explicit lists in `platform/CMakeLists.txt`); excluded from install (that
  file's install EXCLUDEs); in a module marked `pytestmark = pytest.mark.integration` (`-m "not integration"`); a
  CTest entry with no tier match in the tier YAML; a parity pair (byte identity only, which no workflow runs) or a C++
  half the differential gate skips (it runs `--python-only`); a both-engine case (no `rocke_engine` or
  `rocke_library_engine` in CI); comprehensive-only (L-119). Not laneless: package-internal suites run as children of
  installed entries (`test_unit`, `tests/in_tree`), and the hipDNN superbuilds run build-tree ctests: read a
  superbuild log's `Test #N` line before writing "no lane runs it" · guard candidate: L-506 (`rk lane-of`)
- L-335 · rocKE's code standards (their only home; read them in the repo, never copy them): `rocke/style/PYTHON_STYLE.md`,
  `rocke/style/CPP_STYLE.md`, `rocke/style/API_DESIGN_STANDARD.md`, `rocke/KERNEL_AUTHORING.md`, the `AGENTS.md` of
  `rocke/`, `rocke/platform/` and `rocke/library/dispatch/`, `platform/python/rocke/dispatch/ARCHITECTURE.md`, and the
  repo's review rules `.github/instructions/rocke-pr-review.instructions.md` (a reviewer bot's §A–§S grid is its own
  checklist with no repo file; §D.16 is `library/dispatch/AGENTS.md:278-284`)
- L-600 · Conv code lives under `library/`: Python routes in `library/kernels/conv/` (`engine_bridge.py` loads the
  engine through `import_engine(module=…)`), shared conv helpers in `library/kernels/common/`, C++ headers and
  instances in `library/cpp/{include,instances}`, the binding module `rocke_library_engine` in
  `library/cpp/bindings/`, goldens in `library/tests/golden/` and `library/tests/parity/golden/llvm_gfx_all.json`
  (`run_diff` routes library families there). Platform TUs cannot include library headers, and `rocke_build_id` must
  cover library TUs. The checks build `rocke_engine` only: library binding tests skip there as in CI (recipe R12)

### CI
- L-306 · Workflows at a PR head: TheRock Multi-Arch CI (+ ASAN), Component CI, hipDNN Superbuild CI, pre-commit,
  clang-tidy, Libraries PR Bot (`therock-pr-bot`), PR Base Freshness, Auto Label PR. A conflicting PR runs no
  pull_request workflows; stacked PRs get no PR Bot, Freshness or pre-commit; fork PRs' "action_required" runs wait
  for a maintainer, and a push by `copilot-swe-agent[bot]` leaves every pull_request workflow at action_required and
  disables auto-merge. Multi-Arch fetches every `.dvc` pointer: an unpushed one fails every build stage (404) and,
  merged, breaks develop; the hipDNN superbuilds pull only the descriptors and golden bundles. The Linux hipDNN
  superbuild co-compiles clang-tidy-20 (WarningsAsErrors) over the TUs its product builds and stops at the first error;
  Windows runs no tidy, and the hipDNN summary is not required, so nothing stops a merge over it. The superbuilds run
  build-tree ctests unfiltered and `hipdnn_ingestor_generator_pytest` under `-p hipdnn_strict_skips`: green proves
  those tests ran
- L-317 · Develop's ruleset: required checks Math CI Summary, pre-commit, Multi-Arch CI Summary, Component CI Summary; 1
  approval plus code-owner review; threads resolved; stale approvals count; `require_extra_approval_for_unattributed_
  changes` (semantics unconfirmed). Math CI Summary sometimes never posts; `reviewDecision` can be null (G2 then checks
  CODEOWNERS). Authors can dismiss our hold's change request and merge. After a push to another SHA and back, read the
  required statuses BY NAME at the returning head: green re-runs can sit beside red cancelled Summaries, and a bot that
  posted at the old head may never post at it
- L-308 · Windows CI: default `read_text()`/`write_text()` use cp1252 (non-ASCII files fail); no "fork"; command lines ≤
  32,767 chars; `PYTEST_ADDOPTS` is split POSIX-style (backslashed paths break); `windows.h` macros (`far`, `near`,
  `min`, `max`) break identifiers of those names; an open `NamedTemporaryFile` cannot be reopened; no
  `os.sched_getaffinity`; pytest-timeout runs its thread method, so one timeout ends the whole pytest run; a taken GPU
  trap hangs the child on gfx110X (gfx1100–gfx1103 and gfx11-generic get a simulated trap); `-Werror` with
  missing-field-initializers and range-loop-construct. The test runners have cmake, ctest, ninja and MinGW-w64 GCC on
  PATH, no MSVC, hipcc or llvm-objdump: read a whole health_status block (L-546). GDN/KDA lanes skip on Windows
- L-321 · Linux CI: a child Python with a bare `env={...}` exits 127 (setup-python needs LD_LIBRARY_PATH)
- L-309 · What CI cannot see: no torch (numerics skip; a module-level `importorskip("torch")` skips the CPU tests above
  it too: rerun with torch before calling them tested); no matplotlib; no `rocke_engine` or `rocke_library_engine`
  (C++-engine and both-engine cases skip); no MMA-pair compile; no llc. llvm-objdump is off PATH (HSACO gates SUBSKIP
  and the test reads PASSED), though a lookup through `$ROCM_PATH/llvm/bin` finds it on the Linux gfx94X shards, where
  a fixed `/opt/rocm` fallback fails rather than skips. GPU tests without torch do run on gfx94X. CI's packages are in
  the shard logs (`uv pip install -r requirements-test*.txt`); Linux runners have c++ but no cmake or ninja
- L-314 · Known infrastructure failures are matched by signature: the config's `ci_infra_signatures` (pattern, job
  scope, and the develop commit that ends it, when one does), shown as tags in checks.md's "CI at head". A failure
  whose every line matches a signature is infra: no issue, no CI update, a re-run. Add a new recurring signature to
  the config the day it is seen, never to this file. A signature cannot settle two traps: a failed Linux build stage
  skips every Linux test job behind it (no rocKE shard ran: the Bots / CI row says a re-run is owed), and a run on an
  older merge ref keeps a develop break develop has since fixed (attribute by ancestry, L-323: "needs a push") ·
  guard candidate: L-508
- L-564 · A shard stopped at "The action 'Test' has timed out after 30 minutes", or a CTest entry at its time limit with
  0 failed, is the runner pool's unless the PR's own entries add the time. One shard takes 15 to 30 minutes depending
  on the pool, and pools drift week to week, so compare entry times, not pool names: read the job's `runner_name`,
  then join per-entry times (`N/M Test #K: name … sec`; gtest `[ OK ] case (ms)`) with develop's same shard on the
  same pool in the same window, over several runs. One slow entry in one shard is noise until the PR's other shard
  shows it too. Only gtest cases split across shards: every pytest CTest entry runs whole in every shard, so a new
  standard-tier pytest entry adds its time to every shard, and splitting it fixes its CTest limit, not the step
  limit. Develop fixes hit the same limits: say so and name the open CI fix · guard candidate: L-508
  (`rk shard-times N`)
- L-323 · A PR's CI runs on the merge ref made when the run started, and re-runs reuse it: later develop changes stay
  invisible until the PR's next push; our checks' merge with today's develop is the only guard. A land-order item ("X
  must regenerate after Y lands") reopens the moment Y lands, and X's own CI cannot see it: X's run on a merge from
  before Y can go green and still turn develop red. A C ctest fed recorded hook answers cannot see Python hook drift
  either

### This host
- Host facts (Pythons, LLVM and ROCm paths, venvs, the GPU, services, dashboards, the tick loop) belong to one site:
  they live in the site lessons' "This host" section, appended here where there are site lessons. No host is CI:
  its LLVM, Python and GPU differ (L-154, L-309)

## Enforced by the toolkit

- L-401 · `rk post` refuses a wrong line 1, non-allow-listed links, absolute perf numbers, linked <JIRA_PROJECT> tags, a <JIRA_PROJECT> tag
  not in the PR, a second post at one head, the hourly rate, a moved head, a header without `<light>
  First-round|Follow-up`, naming another PR, the wrong round · post.py
- L-403 · Workers: builds and a worker's first 5 min are kept apart (`worker_grace.py`); a hung runner is resent
  after `ollama stop` in 120 s; prompt progress resets the stall, and finalize keeps the tools
- L-404 · Picking: a bounded skip loop; settle re-checked and counted from the commit date; REVIEW_MANUAL once settled,
  covered heads skipped; a re-record after our 🔴/🟠 gets a follow-up; a dropped pipeline kills its jobs; checks wait
  for a pending sanity
- L-405 · Checks cannot pass on nothing: 0 compiled fails; no config cap; SUBFAILED read; deleted tests dropped; a head
  containing develop is the merge; with no merged tree, tests still run at head; capped lists and conftest samples say
  so; logs kept; a capped compile family list says so in the report and the status
- L-406 · Failures develop shares are not the PR's: compile, tests; develop-skipped is "not compared"; host hipcc and
  torch-import failures skipped and named
- L-445 · Every flavor: loops follow the tree's columns; every-family byte identity at every flavor; compile with rocKE's
  codegen options; one golden gate as CI's rocke_golden_static, load errors and drift by family vs develop;
  differential summary per flavor
- L-437 · Saved probes of our open findings re-run in each review's checks; companion contract tests for CMake
  registration and arch-data changes
- L-557 · A probe that crashed reads `ERROR rc=1 (crashed)`, not REPRODUCES: an uncaught Python exception exits 1, the
  code for "still reproduces". Probes exit 3 when they cannot tell
- L-447 · ABI: struct layouts compared; claims checked against develop's history on the tick, the dashboards and G9;
  portable_ir's BINARY_ABI / RECIPE_ABI must equal abi.h's counters, a G9 note otherwise (no CI lane builds the shared
  engine, which refuses to load when the two numbers differ)
- L-417 · rule_scan (tools/): data-file and env-assignment noise cut, every section shown; perf unit then figures; PR
  body scanned; text-I/O hint list
- L-421 · Worker tools (tools/): run_pytest in the checks' env without `-I`; grep_code splits `a|b`; checks.txt lists
  real conclusions
- L-407 · Redundant builds: no re-run of a recorded analysis; batches carry across non-rocKE commits; batch members
  get no solo gate; no gate on a PR GitHub reports merged or closed (this covers the race between the PR search and the
  develop read); only a prefix of a validated batch carries its validation
- L-408 · Gate plans: shared helpers select their families; differential flavors from the tree's CTest entry; compile
  at the family's flavor and target
- L-409 · `gh` retries every failure but a JSON 4xx other than 429 (L-505): 5xx, HTTP/2 stream errors, timeouts,
  non-JSON pages, truncated JSON; it halves GraphQL pages past the time limit
- L-410 · Incidents: CONFLICT once per transition; the concluded attempt's jobs; close on their own signal, 10-char SHA
  match; infra jobs and cleared reds are no incident, voids skip stats; an attributable break gets its own incident;
  aged no-culprit told once
- L-411 · Held PRs: HOLD_REVIEW_NEEDED; HELD_MERGED; the board follows a PR back from a hold until re-approval;
  HOLD_DISMISSED is never a re-approval; MERGED_OVER_RED. The auto-merge guard turns an auto-merge off once per head at
  most, and never one <user_login> armed (it alerts); a re-arm at a head it disarmed reads "alert the user"
- L-509 · `rk post --premises` owns the post: it re-checks the verifier's premises live (head, develop, the conflict
  set, CI failures, cancels and the running set by latest attempt, threads, reviews and comments after `STOP`), the
  hygiene words, measured figures, verdict words, `#<n>` cross-links, the Code quality row and the footer; then it
  saves the verify, records the drafter lines and the cost, and archives notes.md
- L-501 · `rk draft-lint N` checks a first-pass draft against its own inputs and writes `checks/lint.md` beside it,
  never editing the draft: verdict words, line 1, the round against meta.md, the author, <JIRA_PROJECT> tags, "(ran)" with no
  matching run and test counts no pytest summary printed, merge words, Bots / CI words and counts against
  checks.txt, rule_scan hints as findings, our own items, fix SHAs, ×N counts and statuses other than OPEN with no
  reply or later commit (L-019), and a degenerate draft (repeated rows, a forced finalize, the step cap, a very long
  think, one file read in most steps). REVIEW_READY runs it on every draft of the local review loop and names the
  report; the verifier opens it after its blind pass
- L-594 · `rk classify N --record` reads the PR's live head and refuses (exit 4, nothing recorded) while it is not
  the head the tick saw: run between a push and the next tick, it would record the older head as covered
- L-590 · The checks print, as leads for the verifier: clashes with other open PRs (both land orders, silent clashes
  that merge clean), a mutation sweep of the checks the PR adds, CTest labels and tiers, references to moved or
  deleted files, and an engine import probe, each against the PR's merge base; and the PR's head is read from the
  worktree
- L-412 · G2/G3: open threads block, no fresh approval suggested; required checks by name, "CI incomplete" when one is
  missing; <user_login>'s own PR waits on another owner; guard explanations never rate-limited; armed auto-merge message vs
  live (a single commit's co-author line has no dashed rule)
- L-414 · Develop issues: dup check; labels by kind, ci-red only with post-merge; only named fixes count; known-failure
  mentions are no signal; no PR notice on <user_login>'s own PR
- L-449 · Scope: develop fetched after the PR refs; OUT_OF_SCOPE names the cause
- L-416 · Board: "merged today" counts develop merges only; ages and rolling counts are no change; LOSING only past a
  20-review backlog; a mergeability GitHub is recomputing (UNKNOWN) keeps its last definite answer in the ready list and
  G4
- L-415 · Isolation: pristine trees, fresh builds, wiped outputs; the follow-up classifier compares each file's own
  patch as a +/- multiset
- L-534 · Settle: one rule (`store.settle_left`) for the plan, the driver and `rk review-start`, which refuses an
  unsettled head (exit 4) unless `--now`; 2+ pushes in 90 min (as ticks saw them) settle 45 min after the last; develop
  fixes and <user_login>'s priorities keep 10 min
- L-536 · `rk facts N`: the brief facts, read fresh and read-only, every force-push (`filteredCount`), every
  hand-written comment of <user_login>'s in full; exit 1 if a section could not be read
- L-505 · `gh` retries every failure but a JSON 4xx other than 429: DNS, refused connections and TLS timeouts included;
  a permanent GraphQL error costs ~20 s of retries
- L-513 · The worker's meta.md stages `- Round:` and `- Line 1:` (the toolkit passes REVIEW_ROUND and REVIEW_LINE1 to
  `<WORKING_DIR>/tools/fetch_pr.py`, which is unversioned: L-537); used by L-001
- L-526 · The PR's changed tests run in the installed tree, as CI's rocKE lanes do (`tools/installed_tests.py`): the
  checks' build is installed, `ctest --show-only=json-v1` maps each changed test file to the entries that collect it,
  and they run with the entry's command and environment (the checks' ROCKE_* and HIP_VISIBLE_DEVICES dropped); over 40
  files the entries run whole; a failure develop's installed run shares is labelled develop's; `INSTALLED
  pass|fail|skip` joins G6. Not covered: changed test helpers or conftest.py alone, install-rule changes without a test
  change
- L-540 · Develop drift under an unchanged head (`rkmon_lib/drift.py`): whenever develop's rocKE tree moves past the
  tree a PR's tests last ran on, a light job runs the PR's test files on its head merged with develop and on develop
  alone (git archive, no build), for every PR our review clears that merges cleanly, approved or not, also while
  frozen; a failure on the merge only fails G6, blocks before approval, and prints DRIFT_FAIL with the develop commits
  touching its files. The job runs only the files the merge has. Not covered: PRs without test files; behaviour a test
  outside the PR's own files sees. Before telling an author, read the job's `workers/drift/<d>/pr<N>/report.txt`: a
  `RUN-ERROR(rc=4)` is pytest refusing its arguments, not a test failing

## Open problems (guard candidates, ranked by time saved)

- L-577 · `rk probe-check DIR TREE`: saved only after 2 on DEVBASE, 1 on the tree and 0 on a simulated fix; run_probes
  runs each probe with HIP_VISIBLE_DEVICES and ROCKE_LLVM_FLAVOR unset and the tree root as argument; pytest rc 5 and
  a crash read 3; a probe that greps `^FAILED` without `SUBFAILED` is flagged · 4–7 h/week
- L-578 · Fix-line premise: `rk post --premises` refuses a post whose notes lack a `fixcheck: N<k> ran|read <how>` line
  for each net-new 🔴/🟠 whose fix names code, a tier or a test move (L-158) · 3–6 h/week and an author round each
- L-508 · Known-failure tags in "CI at head" from `ci_infra_signatures` (pattern, job scope, ending commit) and open
  develop issues; `rk shard-times N` joins per-entry times with develop's same shard and pool · 3–5 h/week
- L-502 · Our-items skeleton in checks/ours.md from our last post; `rk post` refuses a follow-up without a row for each
  open 🔴/🟠 of ours, and an ABI number that is neither develop's nor "develop + 1" · 3–5 h/week
- L-588 · Worker and facts tooling: the worker's run_pytest gets `<WORKING_DIR>/venv/bin` on PATH and a pybind dir only
  when its build stamp matches the checks' tree; fetch_pr's commits.md lists the newest commits (and says "first N of
  M"); `rk facts` pages review threads past 100 and prints each thread's last reply and each grader re-review's
  verdict lines; the toolkit's git diff calls pass `--` and refuse a worktree path inside the mirror (L-575) · 2–4 h/week
- L-579 · ABI premise: `rk post --premises` refuses "no bump" while the checks' ABI section lists a changed C struct,
  unless notes carry an `abi:` line naming abi.h's rule; the checks compare abi.h's define, its newest history entry
  and abi.py's BINARY_ABI on the merge (a half renumber is 🔴, CI cannot see it)
- L-596 · Merge-only coverage: the checks run, on the merge, every CTest/pytest file that imports a module the PR changes
  and develop's tests changed since the PR's CI merge ref; `DEVTESTS_FAIL` lines name the develop commit
- L-597 · Superbuild tidy in the checks: clang-tidy-20 with the provider `.clang-tidy` over the PR's touched TUs, feature
  macros on and off, every `error:` listed; `rk job-log --errors` lists a CI log's whole error set and its unreached TUs
- L-598 · Sweep survivors: re-run each against every CI-run importer, with venv-torch and the shared engine; a set with a
  develop-known failure drops that file, never the sweep; mutate carried fields of widened shared helpers
- L-599 · Base-branch PRs: files.txt, plan, sweep, tiers, premises, the differential's flags and the fetch use the PR's
  `base.ref`, never a literal develop
- L-506 · `rk lane-of <test file|node id|ctest entry>`: prints its CTest entry and tier labels, installed or excluded,
  module markers and module-level skips, the latest shard and superbuild log lines naming it, and a verdict
  `CI: runs|skips(<reason>)|no lane(<form>)`; checks.md's coverage map uses it · 3–5 h/week and the retractions
- L-535 · Flavor-unforced pass: rerun the PR's changed tests with `ROCKE_LLVM_FLAVOR` unset and list skips seen only
  there
- L-532 · After a develop golden-layout or flavor-ladder change, re-run the merge golden gate (and L-535) for open PRs
  that touch goldens or served-flavor lists; print `GOLDEN_GAP #N`
- L-538 · Companion tests: conv dispatch policy → `test_conv_reference_contract.py` and GDN decode cases →
  `test_gdn_decode_ir_cpp_parity.py` are built. Left: `lower_hip*` → `test_hip_source_identity.py` once #NNNN lands
  it; IngestorGenerator → its suite (outside rocke/, needs venv-gen: its own section)
- L-537 · Put `<WORKING_DIR>/tools` under git, its tests in `rk selftest` (L-226)
- L-511 · Persist <JIRA_PROJECT> candidates and develop leads (first seen, recurrences, <user_login>'s answer) on the board
- L-503 · REVIEW_DELTA: a rebase-only or develop-merge head move keeps the draft and re-runs only the checks
- L-507 · CI_UPDATE_DUE: a post made with CI pending is re-checked until CI completes; red → a plan line
- L-573 · Scan gaps verifiers find by hand: the clash scan misses CTest entry names used as strings in the other PR's
  tests (`== "rocke_pytest"`) and attribute names passed as strings (`mock.patch.object(mod, "_name", …)`: clean
  merge, AttributeError at run time), constants another PR asserts on with no shared file, changed C/C++ function
  signatures whose callers the other PR adds, and open PRs created after the scan ran; the overlap scan misses
  the same new C struct fields and binding keys in two PRs, one removed block extracted into differently named
  helpers, renamed twins of a mirrored constant, lanes added through a pytest runner, and reads R100 moves as new
  code; the fix sweep matches a PR that only edits `platform/CMakeLists.txt` and names a failing test while saying it
  "fails identically on develop"
- Backlog (texts in the archive): L-504 build precondition; L-510 HOLD_REBASED; L-514 PROPOSE churn; L-515 exact
  shard hot spots; L-516 restrict review dismissals (<user_login>'s repo call); L-517 gc.auto (one command); L-519
  develop NEW counts cached; L-520 hkp census replay; L-521 history findings per push; L-522 AOT grid sweep; L-525 our
  findings' families in plans; L-528 approval coverage; L-529 tree = trial merge; L-530 descriptor id vs DVC; L-531
  required statuses in "CI at head"; L-533 torch-gated modules in venv-torch

## Merged and retired IDs (full texts and reasons in the archive)

- The texts of merged, retired and moved IDs, and the lesson each merged ID belongs to, are in
  `<WORKING_DIR>/lessons/archive/lessons_archive_<week>.md`.
- Merged: L-337, L-542 and L-551 → L-314; L-555, L-572 and L-592 → L-564; L-565 and L-566 → L-152. Retired: L-539,
  L-561 (enforced by `tests/conftest.py` and `rk selftest`). Moved to the site lessons: L-221, L-228, L-315, L-319,
  L-327, L-332, L-336, L-340, L-541, L-544, L-562, L-568, L-587.
