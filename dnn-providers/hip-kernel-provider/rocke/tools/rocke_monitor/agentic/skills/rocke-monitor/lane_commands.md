<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

## Commands card for the review passes (from `rk --help` and the toolkit source)

The verifier and the code quality reviewer learn these here, not from the toolkit's source. All are read-only. In a
row, `rk` stands for `<RK>`: run `<RK> facts 101`.

| Command | Prints |
|---|---|
| `rk facts N` | `# Facts for #N, read <UTC>`, then a `== ` heading per section: head and state; every force-push (newest last; the ones after our last review marked so); the last 6 commits; our reviews (the posts in the ledger, then the files in the outputs folder, each a `copy` or `kept unposted`); our last review's header, flagged rows and findings; rk abi; rk breakages; the latest review per reviewer; unresolved threads (the first 2 comments of each, cut at 200 characters: read a whole body with `gh api`); <user_login>'s hand-written comments, in full; other people's comments since our last review, in full (bots left out); the worker inputs. Exit 1: a section says `(not read: …)` |
| `rk ci-fails N` | `head` and the live head's SHA; then, by check name, at its latest attempt (the one that started last), `RUN⇥name⇥status⇥url` while it is queued or running, or `FAIL⇥name⇥conclusion⇥url` when it failed; last, every attempt and the names, counted (`runs 14 names 11`). A url's last number is the job's ID |
| `rk job-log ID` | the log's line count, then up to 25 failing lines, each after its line number and a colon (cut at 260 characters); the whole log stays a day in `joblog_<ID>.txt` in the temporary folder (`TMPDIR`), to read around a line |
| `rk heads N…` | per PR, live: the number, the head's first 10 characters, the state, whether it merged, GitHub's mergeable state and the last update (`101 3f2a9c1d7e open merged=false mergeable=clean updated=<UTC>`) |
| `rk overlaps N --section` | checks.md's overlap section for N, as of the last tick: each group (`[key …]`), `here:` and the other PRs' `file:line` with their similarity, the land order and a `decided:` owner; then "Copies of a develop pattern" |
| `rk wait-quality DIR` (Bash timeout 600000) | the verifier's wait for the code quality pass's `quality.md` in the review directory DIR: `ready: …` once it is there, newer than `facts.txt`, and its Code quality row is not `pending`; `waiting: …` after 4.5 minutes (call it again); `timeout: …` (exit 1) 20 minutes after the first call |
| `python3 -B <PACKAGE_DIR>/tools/worker_grace.py [MAX_S]` (default 900) | `worker_grace: no local review loop running`, or the youngest worker process's age and `go` (exit 0); else `waiting` lines, then `gave up` (exit 1). It can wait about 5 minutes: put it first in the background script of a build or a big test run |
| `rk lane-of TARGET... [--pr N] [--collect]` | TARGET is a test file, a node id or a CTest entry. Per target `CI: runs (<entry>, <tier>)`, `CI: skips(<reason>)`, `CI: no lane(<why>)` or `CI: errors(<why>)`, then its entries, tiers, lanes and `-m`, answered from the tree at develop's head, or at PR N's (as in CI, without the modules of `ci_absent_modules`, such as torch and the built engines; an entry whose only tiers are comprehensive or full runs in no CI job unless a lane selects them); `--collect` collects it as CI would. Exit 0 runs, 1 not, 3 NOT RUN |
| `rk probe-check DIR TREE --simfix PATCH` | per probe `<probe>: DEVBASE 2 · tree 1 · simfix 0 · OK`, or `NOT OK` and why, then `flag:` lines (a host path, a `^FAILED` grep without `SUBFAILED`, `HIP_VISIBLE_DEVICES=-1`, a script that never reads its tree root `$1`); without TREE, the flags only. Run it before saving a probe. Exit 1: a probe is not OK |
| `rk probe-port DIR` | per probe file `converts` or `needs a hand edit`, with each path and why; then the counts: a saved probe's host paths rewritten as the probe contract's variables. Without `--write` it is a dry run; `--write`, which rewrites the files, is <user_login>'s |
| `rk shard-times N [--every]` | per failed or timed-out shard job at the live head (`--every`: every shard): its pool, minutes, CTest entry times and entries cut short, against develop's same shard on the same pool within a day; last line `VERDICT PR-added +a..b s per shard`, `VERDICT pool (develop same pool m–n min)` or `VERDICT not compared: …` |
| `rk werror N` | the checks' Superbuild tidy run by hand, on the review's shared build while its slot is kept: each new clang-tidy or `clang++ -Werror` error as `file:line:col` over the TUs the PR touches, feature macros off and on, the ones develop has too counted apart; last line `TIDY pass|fail|skip` |
| `rk job-log ID --errors [--pr N]` | every distinct `error:` line of a CI job's log, in log order, after its line number and a colon; then a `never compiled:` line for each of PR N's changed C/C++ sources with no object file in the log |

**The `premises:` line** of `notes.md` (`premises: head 3f2a9c1d7e; develop 9b1e04c2aa; CI 12 ✓ · 0 running · 1 ✗ ·
0 cancelled; threads 2/5; stop <UTC>; probes none`) becomes the spec `rk post --premises` re-checks right before the
post; the first value that changed stops it:
- `head` (HEAD10): the live head.
- `develop` (DEVBASE): develop may not have changed the PR's own files since. A PR on a feature branch writes `base`,
  the branch and its commit instead (`base <branch> 1a2b3c4d5e`), for that branch.
- `CI` (✓, running, ✗, cancelled): check runs per name at the latest attempt, as `rk ci-fails` counts them, no commit
  statuses (L-224, L-229). Failures must stay exactly the FAIL names you accounted for (FAILOK), nothing may run beyond
  your `CI pending:` jobs (RUNOK), and cancels stay at most the cancelled count (MAXCANCEL); the ✓ count is filled in
  at post time.
- `threads` (THREADS): the open or the resolved threads, and the total; both must still match.
- `stop` (STOP): no review, review comment or human comment may be newer (L-118).
- `probes`: what the next round's checks re-run, or `none`.
- `fixcheck: N<k> ran|read <how>` (notes): `rk post --premises` stops a 🔴/🟠 fix that guards a test on a module CI lacks
  (`importorskip("torch")`), checks a "CI still runs it" with `rk lane-of --collect` at the head, and needs this line
  for any other skip guard, tier change or test move.

It also re-checks the merge's conflict set (`git merge-tree --write-tree --name-only` with the base branch's head:
the files from line 2 to the first blank line), the ticket tag in the body and the post's hygiene (verifier §6).

**The checks' guard lines** (checks.md; leads unless marked):
- `DEVTESTS_FAIL tests/test_x.py (develop 9b1e04c2aa)`: a develop test CI never ran with the PR fails on the merge
  only; the commit is the develop commit that changed it. `IMPORTERS_FAIL`, a test file and, in brackets, the changed
  modules it imports: a test the PR does not change fails on the merge only.
- `homes` and the ABI lane, then `define N · history N · abi.py N · canary N · OK`, or `HALF RENUMBER:` and what
  disagrees (the ABI section, a flag): a half renumber is 🔴, and CI cannot see it. `rk clash N`'s `ABI NUMBER CLASH`:
  another open PR claims the same counter, and whichever lands second renumbers.
- "CI at head" tags a failing job `known infrastructure`, with the signature and its note, from `ci_infra_signatures`
  and the site's `ci_infra_site_signatures` (job scope, `unless` text, ending commit), or when develop fails the same;
  a job that passed only on a re-run reads `FLAKE`.
- The CI-condition run: each changed test run without torch, built engines or ROCm tools; `0 of n run` is a lead, a
  skipped tier or contract test reads NOT RUN. Superbuild tidy: `TIDY pass|fail|skip` (`rk werror N`).
- The mutation sweep's survivors re-run against every importer: `SURVIVED (CI importers, torch, engine)` or
  `vacuous: torch-only path`.

**Recipes** (the shape of a run that settles a claim; a planned command replaces its line once built):
- R1 rebase or edits: `git merge-tree --write-tree --merge-base <old base> <new base> <old head>` diffed with the new
  head (empty: pure rebase). R2 a develop merge: `git show --remerge-diff` per merge since our last review; mutate
  rules it adds
- R3 silent clash, land order: `merge-tree` develop+A+B both orders; grep outside the hunks for removed or renamed
  names, CTest names as strings, `mock.patch.object` string attributes; `c++ -fsyntax-only` with the build's ninja
  flags, merged headers first on `-I`; the sibling's tests on the double merge
- R4 keep-both advice: a diff3 ours-then-theirs splice (never `merge-file --union`), YAML through the real parser
  (`safe_load` keeps the last duplicate key); compile or run the splice
- R5 run installed as CI does: `cp -a` the checks' install; move or mutate one file; run the entry's own command
  (PYTHONPATH = installed tests and library plus site-packages; a plain `ctest` there lacks `rocke`); restore, `cmp`
- R6 a C++ check is tested: compile a copy of the TU with the checks' flags (keep `(void)x;`), link it ahead of the
  checks' `librocke_core.a`, run the CI entry; never re-run the archive step. R7 a Python check: an in-memory pytest
  plugin or `sitecustomize` execs the mutated source as the module (tree untouched); node ids, not `-k`
- R8 a mutant's verdict is real: the control reads "N passed, 0 errors" first; one mutant per branch and per copy;
  remove-the-guard and silence-the-guard; boundaries (`>` for `>=`, LIMIT±1); count `SUBFAILED`
- R9 configs reaching C++ lines: build with `--coverage`, run the gate, read the gcov JSON. R10 processes reaching a
  Python function: a logging `sitecustomize.py` on PYTHONPATH (run_diff's workers inherit it) and a control kernel
- R11 C++ IR over a whole AOT grid: a C driver generated from the Python specs (`dataclasses.asdict`), the old TU
  linked ahead of the archive for "before". R12 library engine tests: one `c++` link of `library/cpp/bindings/*.cpp`
  against the shared build's `librocke_library.a` and `librocke_core.a`
- R13 engines agree, a refactor is neutral: a seeded spec sample, a knob-off control; verdicts, refusal texts and
  `.ll` sha256 in both engines, distinct-hash count printed. R14 ABI layout: `-Xclang -fdump-record-layouts` on
  develop's and the head's headers (`git show`); simulate keep-both, abi.h-only and full-renumber resolutions
- R15 a shard timeout: `rk shard-times N`. R16 re-run jobs: the earlier attempt's jobs
  (`actions/runs/RUN/attempts/1/jobs` when attempt 2 is the latest; attempt-2 ids with attempt-1 times did not
  re-run); a running job's log already has its finished CTest entries
- R17 every build or tidy error: `rk job-log ID --errors --pr N`; clang-tidy a fix snippet with the provider
  `.clang-tidy` and a control
- R18 golden claims: leaf-diff the JSONs base vs head, ll vs ir_canonical keys; re-record on a scratch merge, `cmp`.
  R19 spills: whole-grid -O3 with rocKE's per-kernel COMGR options, each record for its own arches only
- R20 Windows text I/O: `LC_ALL=en_US.CP1252 PYTHONUTF8=0 PYTHONCOERCECLOCALE=0` with `LOCPATH`, a failing control
  `read_text`. R21 numerics, no GPU: a numpy model, adversarial inputs; an IR def-chain trace from each load to its
  guard
- R22 a stacked base landing: simulate its squash (`merge-tree` develop+base, `commit-tree`), merge the head into it.
  R23 catalog or tuning-id changes: record sets and config keys, never counts. R24 a red test turned green: diff the
  touched tests for removed ids and new skip markers, and run the removed case
