# Runbook: deep review

Each stage ends in a **Gate:**; do not start the next stage until it holds. This runbook
needs subagents; without them, stop and tell the user to run `hipdnn-pr-quality`. Variables:
[lanes/common.md](lanes/common.md). Your formats: [report.md](report.md).

## Pin and intake

`<pr>` is the PR number, `<N>` the round. `<owner>/<repo>` comes from the PR URL, else
`origin` of `$CLONE`, else `ROCm/rocm-libraries`; pass `-R <owner>/<repo>` to every `gh pr` and
`gh issue` call. `pin.py`, `partition.py`, `ledger.py` and `comments.py` are in `$SKILL/scripts/`;
a script name below stands for `python3 "$SKILL/scripts/<name>"` (`python` or `py -3` on Windows, as in SKILL.md *Platforms*). Each prints terse results and
exits non-zero with a one-line reason. After pinning, commands run from `$EVIDENCE/r<N>/`
(sourcing `env.sh` changes to it); GNU coreutils, grep and sed are assumed.

`$KEY` is `pr<pr>`, else the branch name or `<base>..<head>` text with characters outside
`A-Za-z0-9._-` replaced (`pin.py key <text>` prints it and fails when no valid ref component
results; `init` derives it itself). `$TARGET` is the PR's base branch (a stacked PR's is not
`develop`), `develop` for `branch:<name>` unless told otherwise (`--target`), or the left side
of a range.

```bash
python3 <skill dir>/scripts/pin.py init --evidence <evidence dir> --scratch <scratch dir> \
  --round <N> --owner-repo <owner>/<repo> [--clone <dir>] \
  --pr <pr>   # or: --branch <name> [--target <base>]   or: --range <base>..<head>
```

`init` creates both directories and `PIN.md`, then:

- uses `$CLONE` (`--clone`, else the `CLONE` environment variable, else `$SCRATCH/clone`) as
  given, else makes a blobless clone with full history (never `--depth` or `--shallow-since`:
  a wrong merge base with exit 0), and sets `gc.auto 0`;
- fetches head and target into `refs/review/$KEY/` with `+` refspecs; a range side that is a
  SHA is set with `update-ref`, not fetched;
- appends `TARGET`, `TARGET_TIP` (the target's tip), `HEAD`, `HEAD_SHORT` and `BASE` (the merge base; the left side of a
  range) to `PIN.md` and writes `env.sh`; on an empty diff it prints `empty diff: PR already
  merged by merge commit?` to stderr;
- writes `files.txt`, `numstat.txt`, `commits.txt` and `sparse.txt`: the project root of each
  touched project (`projects/<x>`, `dnn-providers/<x>`, `shared/<x>`; all of `projects/hipdnn`
  when it is touched), each other touched directory, and root `tools/` when
  `projects/*/tools/` changes;
- writes `drift.txt` from `git merge-tree --write-tree`: the target tip, `Merge: clean | conflicts`, the conflicted paths, the
  commits in `BASE..TARGET_TIP` that touch a changed path or a file beside one, and `Trigger: yes | no (<reason>)`. The trigger
  is `yes` when the merge conflicts or a commit touches the change; it is `no` for a range review;
- adds the sparse worktrees `$REPO` (at `$HEAD`) and `$REPO_BASE` (at `$BASE`) and checks
  both are at their SHAs;
- for a PR saves `pr/pr.json` (with GitHub's `mergeStateStatus` and `mergeable`; nothing gates on them), `body.md`, `inline.json`, `files.json`, `checks.tsv` and the plain-text `branch.txt`, `pr-text.txt`, `pr-comment.txt` and `pr-inline.txt`, and
  checks `$HEAD` against `headRefOid` and the file counts (below).

`. <evidence dir>/r<N>/env.sh` exports every `PIN.md` variable among `SKILL EVIDENCE SCRATCH
CLONE KEY TARGET TARGET_TIP HEAD HEAD_SHORT BASE REPO REPO_BASE KNOWLEDGE`, fails when `EVIDENCE`,
`SCRATCH`, `CLONE` or `KEY` is unset, and changes to the round directory. It works in a fresh
shell and in a persistent one; every later block starts by sourcing it.

A stacked PR's parent: `gh pr list -R <owner>/<repo> --head "$TARGET" --json number,url --jq '.[0]'`
(`none` for `develop`).

With no PR, save the intent document as `intent.md`. An empty description is an `ORCH`
`process` finding.

Complete `PIN.md` by appending the `report.md` fields `init` does not write: `Change`,
`Round`, `Repo`, `Files` (range and branch reviews: the `files.txt` count), `Stack parent`, `Description`, `Checks`, `PR bot`, `Build host`,
`Leak terms`. `Build host: yes` needs `cmake`, `ninja`, a C++ compiler, a ROCm install
(Linux: `${ROCM_PATH:-/opt/rocm}`; Windows: the wheel SDK, which `windows_rocm_setup.py --provision never --venv-path $SCRATCH/rocm-venv` from [hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) validates; when only the wheel SDK is missing, record `yes (RUN provisions the SDK)`, since `RUN` provisions it with `--provision auto`; Clang present per that skill) and 25 GB free in `$SCRATCH`; else `no (<missing>)`. `PR bot` is
the `therock-pr-bot` check. Round N > 1 also records the previous head and base, saves `git
-C "$CLONE" range-diff <old-base>..<old-head> <new-base>..<new-head>` to `range-diff.txt` and
writes `affected.txt` (*Iterate*); `partition.py packets` lists them when `range-diff.txt` exists.

**Gate:** `pin.py init` exits 0: both SHAs and both sparse worktrees exist; for a PR, `$HEAD`
equals `headRefOid` and the `--no-renames` file list equals `pr/files.json`'s (renamed files'
old paths added); a mismatch fails, naming the first differences and both counts. A match
records `Files: <count> (GitHub <count>)`. At 3000 files GitHub truncates: it records
`Files: unverifiable (>3000)` and relies on the SHA match. On any failure: `pin.py cleanup`
(*Iterate*), then re-run `init` once to re-fetch; if it persists, stop and report both lists
(or the moved head; re-pin only if the user wants the new head). When `init` rejects the key,
stop and tell the user; nothing was fetched.

## Orient

Export `$KNOWLEDGE` and append it to `PIN.md` as `KNOWLEDGE: <path>`, outside `$CLONE`, `$REPO` and `$REPO_BASE`, as the first writable of
`knowledge:<dir>`; a durable-knowledge directory the host's agent instructions name;
`${TMPDIR:-/tmp}/hipdnn-review-knowledge-${USER:-$USERNAME}/`; `$EVIDENCE/knowledge/`. Record the choice
and why an earlier one failed in `PIN.md`.

`partition.py areas` assigns each path of `files.txt` to one area and writes `areas.tsv`
(`path<TAB>slug`). `AREA_ROWS` in `scripts/partition.py` is the only copy of the area table:
the first matching row wins, so read it there. A path no row matches is `docs` when it is
documentation ([policy.md](policy.md) *Review rules*), else `<project>-<first directory below
the project root>`. Docs detection (`is_docs`): Markdown and reStructuredText
files, `LICENSE`, `CONTRIBUTING`, `SECURITY` and `README` files of any extension, and `.txt`
files under a `docs/` or `doc/` directory; config, CMake and requirements files are not
documentation. A documentation file under a row's directory keeps that row's slug but still
counts as documentation for the all-documentation check in *Partition*.

A `docs` area gets no scout; a documentation-only change gets a `SYNTH`-only `CONTEXT.md`.
For each other area, the prior file is the newest published `$KNOWLEDGE/*/<slug>.md` whose
`Describes` sha passes `git -C "$CLONE" merge-base --is-ancestor <sha> "$BASE"` (order by
`git log -1 --format=%ct <sha>`).

Dispatch one `SCOUT<k>` per area ([lanes/orient.md](lanes/orient.md)) with its prior file or
none; a scout that fails twice leaves its area without a file and `CONTEXT.md` says so. When
all have returned, dispatch `SYNTH`.

**Gate:** every area whose scout did not fail twice has `orient/<slug>.md` whose `Describes` sha is `$BASE`,
`CONTEXT.md` exists, and every repository path they cite exists at `$BASE`; this prints
nothing then:

```bash
. <evidence dir>/r<N>/env.sh; : "${BASE:?}"
cat CONTEXT.md orient/*.md | "$GREP" -oE '`[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)+(:[0-9]+)?`' |
  tr -d '`' | sed 's/:[0-9]*$//' | sort -u |
  while read -r p; do git -C "$CLONE" cat-file -e "$BASE:$p" 2>/dev/null || echo "not at base: $p"; done
```

Fix each `not at base` citation by hand or re-dispatch the scout. Then publish `CONTEXT.md`
and `orient/*.md` to `$KNOWLEDGE/<KEY>-r<N>/` (the store's own conventions win), replacing
only this review's entry.

## Partition

`partition.py plan` proposes `owners.tsv` (`owner<TAB>path`) from `areas.tsv` and `numstat.txt`
and prints one summary line per correctness lane (`CORR<k>: <files> files, <lines> lines,
areas <slugs>`); it never reads the diff. It packs areas in first-appearance order into
lanes of about 2000 lines, splits an area over the target by file, never splits a file (a
file over the target gets its own lane), and counts lines as added plus deleted (a rename is
two `numstat.txt` rows; a binary file `-<TAB>-<TAB>path` is one owned file of zero lines).
It cannot judge context: edit `owners.tsv` by hand for what only judgment decides.

- A total at or under the target is one lane. Split only for distinct contexts (areas that
  share no contracts) and record why in the ledger Lanes table.
- Keep a header with its implementation and a producer with its consumer. `plan` keeps a changed test with its changed source:
  `test_<x>.py` or `Test<X>.cpp` pairs with every changed non-test file whose stem is `<x>` (case-insensitive) when they share a directory (`X.cpp` and `X.hpp`),
  in their area group; a test with no such file, or same-stem files in several directories, stays where its own area puts it. A pair over the target gets its own lane.
- Generated code, golden data and bundles get their own `CORR<k>`: list them in a `--sampled`
  file, one row per path: `path<TAB>generator-or-source-of-truth`, plus a third column
  `sample` on the files the lane reads. `plan` puts every listed path in the last lane, and
  `packets` marks it `Sampled: <generators>; sample <files>` (and fails when the lane has no
  `sample` row). Pass the same file to both.
- When every changed file is documentation (`is_docs`, *Orient*), `plan` gives all paths to
  `DOCS` and no `CORR` lane runs.

Choose the concern lanes from the table below, which owns every lane's scope. Run a lane when
its trigger holds, else mark it `N/A`; when unsure, run it.

| Lane | Brief | Reviews | Runs when |
|---|---|---|---|
| `CORR<k>` | [correctness](lanes/correctness.md) | Logic, error paths, ownership, code quality of one partition | Non-documentation file changes |
| `TESTQ` | [test-quality](lanes/test-quality.md) | Whether each test fails on a plausible bug | Non-documentation file changes |
| `TESTB` | [test-breadth](lanes/test-breadth.md) | Which changed behaviors have tests at all | Non-documentation file changes |
| `AUTH` | [author-testing](lanes/author-testing.md) | What the author and CI ran against what the change needs; M4, M5, PR bot. Its claims reconciliation against what `RUN` executed is the follow-up `AUTH-f1` | Always |
| `BUILD` | [build-packaging](lanes/build-packaging.md) | CMake targets, option defaults and registration, install rules, exported config, CI files | `CMakeLists.txt`, `*.cmake`, `*.cmake.in`, any `CMakePresets.json`, `*test_categories*.yaml`, `.github/` or `version.json` changes |
| `DEPS` | [dependencies](lanes/dependencies.md) | New dependencies and boundary crossings | CMake, `pyproject.toml`, `requirements*.txt`, `.github/` or C/C++/HIP source (`.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`, `.hxx`, `.hip`, `.cu`, `.cuh`) changes |
| `API` | [public-api](lanes/public-api.md) | Public surfaces, behavior changes in source, docs of new public declarations | A file in the brief's surfaces table, or non-test source under `projects/hipdnn/` or `dnn-providers/`, changes |
| `ROCK` | [therock](lanes/therock.md) | Whether TheRock CI builds and tests the change: skip-ci, job, family and tier selection, flags, RPATH, CTest files, artifact split, target-neutral outputs, Windows lookup and path length, tier timeout baselines. Its timeout projection against what `RUN` measured is the follow-up `ROCK-f1` | Non-documentation file changes |
| `LEAK` | [leaks](lanes/leaks.md) | Private information in the change and in review output | Always |
| `DUP` | [dedup](lanes/dedup.md) | Reuse of existing code; repetition within the change | Code is added |
| `DOCS` | [docs](lanes/docs.md) | Accuracy and concision of docs and comments, including edited public headers | A documentation or source file changes |
| `DRIFT` | [drift](lanes/drift.md) | The change merged onto the target's current tip: conflicts, broken target-side contracts, tests on the merge | `drift.txt` says `Trigger: yes` |
| `RUN` | [build-validate](lanes/build-validate.md) | Building the head and running the relevant tests | Always |

`ORCH` (process findings you file), `SCOUT<k>`, `SYNTH` ([orient](lanes/orient.md)) and `VER`
([verify](lanes/verify.md)) are not partition choices. Merge two lanes into one agent when the
host's agent limit requires it; split a concern lane over ~4000 changed lines by area into
`<LANE><k>` (copy its packet and keep that area's files by hand). Then:

```bash
. <evidence dir>/r<N>/env.sh
partition.py plan [--sampled <file>]       # writes owners.tsv; edit it for judgment
partition.py gate                          # after every edit of owners.tsv
partition.py packets [--sampled <file>] --lanes TESTQ,TESTB,AUTH,... [--parent <PR number>]
ledger.py seed "$EVIDENCE" <N> --lane 'DEPS=N/A (no manifest or source changes)' --lane ...
```

`packets` writes `packets/<LANE>.md` ([lanes/common.md](lanes/common.md) *Context packet*) for
each lane of `owners.tsv` and each concern lane in `--lanes` (changed files in scope with their
`numstat` rows, areas and Orient paths; BUILD and DEPS scope to their trigger files and print
`<LANE>: no changed file in scope, N/A` when none; `DRIFT` adds `drift.txt` and prints
`DRIFT: N/A (<reason>)` when its trigger is `no` (give that reason to `seed` with `--lane`); every other concern lane takes all changed
files, so name only lanes whose trigger holds; `--parent` is the stack parent). In round 2 on,
write `range-diff.txt` and `affected.txt` first so the packets list them.

`ledger.py seed` creates `LEDGER.md` ([report.md](report.md)) in round 1; from round 2 it
appends this round's Coverage and Lanes tables and keeps every existing row (it fails on an
already seeded round). Coverage gets one row per changed file (`pending`); Lanes gets a `Ran`
row per packet and a row per `--lane NAME=<status text>`: give the reason for every `N/A`,
merge and split there (`merged into <LANE>`, `split by area`), and edit a `CORR` row's Status
by hand to record why a split was made. `LEDGER.md` holds only headings and tables; the
scripts reject any other line. Append `Lines: <total>` (the total printed by `gate`) to
`PIN.md`.

**Gate:** `partition.py gate` exits 0 only when each `numstat.txt` path is owned by exactly one
correctness lane (or by `DOCS`), every owned path is in `numstat.txt` and the owned lines sum
to the total (it prints `unowned:`, `owned twice:`, `not in numstat:` or `owned <n> of <m>
lines` otherwise, and `<n> files, <m> lines` always), and `packets` and `seed` exited 0.

## Fan out

Write `$EVIDENCE/r<N>/dispatch-context.md` once per round, before the first dispatch: the block
every agent shares.

```markdown
Env:       export SKILL=… EVIDENCE=… SCRATCH=… CLONE=… KNOWLEDGE=… REPO=… REPO_BASE=…
                  HEAD=… HEAD_SHORT=… BASE=… KEY=… TARGET=… TARGET_TIP=…   (absolute); <owner>/<repo>; PR <pr> | none
Approved:  <verbatim commands with architecture and walltime> | none
Builds:    every scratch configure (RUN, ROCK, VER) passes -DENABLE_CLANG_TIDY=OFF, and
           -DCMAKE_CXX_CLANG_TIDY= where the environment or preset sets it; each attempt sets
           its own TMPDIR (lanes/build-validate.md)
Agents:    <maximum concurrent agents>
Cores:     <CPU cores for builds and tests>
Mutation:  <host-provided recipe or wrapper command for mutating a built tree> | none
```

`Agents:`, `Cores:` and `Mutation:` are inputs from the user or the host; the skill assumes no values. Dispatch each lane, scout, verifier and the
writer as its own fresh agent with shell and write access; fill this template as its first
message. Agents see only their dispatch: each begins by reading `dispatch-context.md`.

**Queue.** Run at most `Agents:` agents at once, in this priority order: `RUN` first, then `DRIFT`, the
longest lanes (largest `CORR<k>`, then `TESTQ`), `DUP` and `DOCS` last. `RUN`, `DRIFT`, an approved
`ROCK` build, `ROCK`'s configure diff (`Build host: yes`), or any other lane that builds or mutates, run concurrently in their own `$SCRATCH` trees and
split `Cores:` between them while they overlap, `RUN` taking the larger share.

```markdown
Role:      <LANE> | SCOUT<k> | SYNTH | VER-<batch> | WRITER; round <N>; follow-up f<k> if any
Context:   read $EVIDENCE/r<N>/dispatch-context.md first (Env, Approved, Builds, Agents, Cores, Mutation)
Read:      lanes, verifiers: $SKILL/lanes/common.md, findings.md, policy.md, then <brief path>.
           Scouts, SYNTH: lanes/common.md, lanes/orient.md. WRITER: report.md *REPORT.md*, policy.md
Inputs:    under $EVIDENCE/r<N>/: PIN.md; pr/body.md or intent.md; CONTEXT.md and the area
           files for its areas (verifier: the areas of its findings' Where paths);
           packets/<LANE>.md; LEDGER.md rows in scope (round 2 on).
           Scout: area, slug, changed paths, prior area file or none. SYNTH: orient/*.md.
           Verifier: finding IDs and report paths. WRITER: LEDGER.md, verify/, PIN.md,
           pr/pr.json (title), CONTEXT.md, lanes/*.md (M-rule lines, RUN execution counts,
           ROCK not-run items). `RUN`, `ROCK`, `AUTH`: the **Options and guards** and
           **Consumers** sections of `CONTEXT.md` and the area files. Follow-up: the lane report
           path plus the approved request, decision or Handoff.
Output:    <exact file>; scratch $SCRATCH/<ID>/; first provisional ID p<n>
Overrides: anything that differs from dispatch-context.md for this agent (a narrower
           Approved command, its share of Cores) | none
Return:    at most 10 lines: IDs, tiers, one-line claims, STOP or approval requests.
           Scout, SYNTH, WRITER: 5 lines. Verifier: one `<ID>: <verdict> <tier>` per finding.
```

Put **Approval requests** to the user in one batch. A `STOP:` becomes a ledger row (kind
`process`, `OPEN`) and an open question. Do not wait for answers: record a pending approval
as `UNVERIFIED (approval pending)`, run every unblocked lane and verification, and let
blocked items wait. *Synthesize* then writes `REPORT.md` with `Closure: OPEN` listing what
waits; the answer resumes the round (*Iterate*, same head).

A follow-up dispatch (approved request, decision, Handoff) carries the lane's report path and
the item only, and writes `lanes/<LANE>-f<k>.md`, numbering IDs after the lane's last `p<n>`.

`AUTH`, `TESTQ` and `DRIFT` run in the first fan-out without waiting for `RUN`. When `RUN` returns, dispatch
`AUTH-f1` with the `AUTH` and `RUN` report paths: it reconciles the claims and the
**Consumers** list against `RUN`'s executed counts ([lanes/author-testing.md](lanes/author-testing.md)).
When `TESTQ` wrote `M1: UNVERIFIED`, also dispatch `TESTQ-f1` with the `TESTQ` and `RUN` report paths: it settles M1 from
`RUN`'s regression-test runs ([lanes/test-quality.md](lanes/test-quality.md)). When `ROCK` wrote a `Timing-base:` line, also dispatch
`ROCK-f1` with the `ROCK` and `RUN` report paths: it projects each entry's per-shard time from `ROCK`'s CI baselines and `RUN`'s case
times ([lanes/therock.md](lanes/therock.md) section 6). A lane's newest follow-up report wins when
`ledger.py mlines` reads an `M<n>:` line. Re-dispatch each follow-up that ran (`AUTH-f1`, `TESTQ-f1`, `ROCK-f1`) when `RUN` is re-dispatched with an approval.

An agent that fails, times out or returns a non-conforming report twice is `FAILED
(<reason>)` in the Lanes table; split an over-large lane by area before the second attempt.
Its Coverage rows stay `pending`; you file an `ORCH` `process` finding at `IMPORTANT`
(`<lane> did not complete: <reason>`), listed under `Closure: OPEN`. A failed verifier's
findings are `UNVERIFIED`.

**Gate:** every dispatched lane returned a conforming report ([findings.md](findings.md)
*Lane report*), `N/A` or `STOP:`, or is `FAILED`; `ledger.py coverage "$EVIDENCE" <N>` (it
copies each lane's read depth and tests into this round's Coverage rows) prints no `pending:`
row except one carried forward (edit it to `carried from r<M>` by hand) or owned by a `FAILED`
lane. Re-dispatch a non-conforming report once, naming the rule.

## Merge and verify

1. Collect each lane's `M<n>:` lines ([findings.md](findings.md) *Lane report*) with
   `ledger.py mlines "$EVIDENCE" <N>`: it prints one line per rule (the last line per rule
   wins; an `N/A` owner lane is a vacuous `PASS`), a `file ORCH process ...` line for each
   `FAIL` (`BLOCKING`) and `UNVERIFIED` (the tier [policy.md](policy.md) *M-rules* implies), and
   exits 1 when a lane that owns a rule has no such line (non-conforming). Write those filings,
   each `FAILED` lane's `ORCH` finding and each `STOP:` row into `$EVIDENCE/r<N>/lanes/ORCH.md` in the lane
   report format (`ORCH-p<n>`, Kind `process`).
2. `ledger.py ids "$EVIDENCE" <N>` assigns final `<LANE>-<n>` IDs (all `CORR<k>` share
   `CORR-<n>`), writes `ID-MAP.md` ([report.md](report.md) *Final IDs*) and adds a candidate
   row per finding; re-running keeps assigned IDs; it exits 1 when a report sits below a nested `r<k>/`
   directory or holds `ID:` lines it cannot parse. Cluster findings with the same location and
   claim, and those a verifier reported as `ORCH: cluster <ID> with <ID>` under **Handoffs**: keep one, put its ID in the `SUPERSEDED-BY` cell of each other row of `ID-MAP.md`,
   and run `ids` again, which sets their status `SUPERSEDED-BY:<ID>`. `ledger.py coverage`
   re-copies read depth and tests after a follow-up.
3. Route **Handoffs** to a follow-up dispatch of the target lane or into the ledger as a
   candidate. A `SCOUT` handoff corrects an Orient file: edit a line in `orient/<slug>.md`
   and its published copy; for more, re-dispatch that scout, then `SYNTH`.
4. Send conflicting findings and each lane's own **Needs human decision** entries to the
   ledger's Open questions table (edit it by hand), citing [policy.md](policy.md) *Precedence*.

Verify every candidate with an agent that did not report it ([lanes/verify.md](lanes/verify.md)):
`BLOCKING` and `IMPORTANT` in batches of at most 10 per verifier, grouped by lane;
`SUGGESTION` and `FUTURE WORK` in batches of at most 30. Re-dispatch a verifier whose files
lack the fields in [findings.md](findings.md) *Verifier output*. Verify `VER` candidates (`verify/VER<batch>.md`) with
a different verifier. Send to a second verifier each `REFUTED` `BLOCKING` or `IMPORTANT`
finding and each verdict whose final tier is on the other side of the `IMPORTANT`/`SUGGESTION`
line from the reported tier ([findings.md](findings.md) *Verdicts*).

**Gate:** `ledger.py verdicts "$EVIDENCE" <N>` exits 0, which it does only when it prints no
`missing:` or `second verifier needed:` line. It sets verdict, status, final tier and (`PARTIAL`) the narrowed
claim on each ledger row from `verify/<ID>.md`; `verify/<ID>-2.md` replaces the first verdict
where it differs. Every candidate not `SUPERSEDED-BY` another has `verify/<ID>.md` (and
`<ID>-2.md` where a second verifier ran) and a ledger verdict. Dispatch the second verifier
for each `second verifier needed:` line and run it again; it exits 1 on any such line and
lists each non-conforming verify file for re-dispatch.

## Synthesize

Dispatch one `WRITER` for `REPORT.md` ([report.md](report.md) *REPORT.md*), or write it
yourself from the same inputs. Group the `OPEN` and `UNVERIFIED` findings by theme
into the fix list, ordered by tier then consequence; batch `SUGGESTION` and `FUTURE WORK` by
theme; fold M-rule failures into the **Verdict** and fix groups. The `M<n>:` lines come from
`TESTQ` (M1, M3), `TESTB` (M2) and `AUTH` (M4, M5); test execution for M1-M3 comes from `AUTH`.
List each human decision under **Decisions needed**. Set `Closure:` from `ledger.py counts
"$EVIDENCE" <N>`, which implements [SKILL.md](SKILL.md) *Completion*: it prints per-tier counts
and `Closure: OPEN; waiting: ...` or `Closure: CLOSED`. Every Open questions row and every
Decisions row with an empty Decision cell keeps `Closure: OPEN`; move an answered question
from Open questions into Decisions. Run `comments.py draft "$EVIDENCE" <N>` for `COMMENTS.md`
(needs `ID-MAP.md`): an `OPEN` or `UNVERIFIED` `BLOCKING` or `IMPORTANT` finding whose first
`Where` is `path:line` inside a changed hunk becomes an inline entry, every other one a
`## General` item; an `UNVERIFIED` one is marked `(UNVERIFIED: <verifier's Reason>)`. An `OPEN`
`SUGGESTION` or `FUTURE WORK` finding is one item per tier and kind. Review and edit the
drafts. Re-dispatch `LEAK` for
its output sweep ([lanes/leaks.md](lanes/leaks.md); output `$EVIDENCE/r<N>/lanes/LEAK-sweep.md`) before the user sees either file; redact each hit it reports in the output and in the published `$KNOWLEDGE` copy (the sweep reads both and fails when the copy is missing), then sweep again. Evidence subdirectories are local-only and not swept.
`comments.py check-sweep "$EVIDENCE" <N>` exits 0 only when the sweep says `Sweep: clean` and is newer than every file it read; a later edit makes it stale.

**Gate:** every statement in `REPORT.md` traces to a ledger row or a lane report, its
`Closure:` line equals the `ledger.py counts` line, and `comments.py check-sweep` exits 0.

## Iterate

On a new head, start round N+1 at *Pin and intake*. In *Orient*, copy the previous round's
`orient/` and `CONTEXT.md`; scout an area the change adds and every area whose file's
`Describes` sha is not the new `$BASE` (the copied file is its prior file), then `SYNTH` again.
The affected files differ between the heads and belong to either round's change:

```bash
. <evidence dir>/r<N>/env.sh
git -C "$CLONE" diff --name-only --no-renames <old-head> <new-head> | sort -u |
  comm -12 - <(sort -u "../r<N-1>/files.txt" files.txt) > affected.txt
```

Findings whose `Where` is in `affected.txt`, and every `process` finding, are affected. Re-run
each lane the delta touches, every lane `FAILED` earlier (closing its `ORCH` finding when it
completes), and always `AUTH` and `LEAK`: a new head brings new commits, PR text and CI runs.
Re-run `DRIFT` whenever the round's `drift.txt` says `Trigger: yes`: the target moves between rounds.
Re-verify every affected `OPEN` or `UNVERIFIED` finding. A lane not re-run carries forward
(Coverage and Lanes rows copied, marked `carried from r<M>`).

On the same head, after a decision, approval, new evidence or to re-run a `FAILED` lane, reopen the round at *Merge and
verify*: record the decision, run the follow-ups, re-verify what it affects and repeat
*Synthesize*, updating the round's files in place.

Close a finding only when its proof ([findings.md](findings.md) *Proof per kind*) no longer
holds at the new head; author rationale alone closes it only as a human `ACCEPTED-RISK`. Record
the closure with `ledger.py set "$EVIDENCE" <N> <ID> FIXED@<sha> --class rerun|text-read|not-exercised --note <proof>`,
or `ledger.py set ... <ID> ACCEPTED-RISK --note <the human's decision>` (it also appends a
Decisions row); both set `Round closed`. `verdicts` never touches a `FIXED` or `ACCEPTED-RISK`
row, so when a later head breaks a closed finding, `ledger.py set ... <ID> OPEN` reopens it and
`verdicts` re-verifies it. Record other decisions and open questions as rows of the ledger's
Decisions and Open questions tables, by hand; an answered question moves from Open questions
into Decisions.

**Gate:** [SKILL.md](SKILL.md) *Completion*. On closing, `pin.py cleanup --evidence "$EVIDENCE"
--round <N>` removes the review worktrees under `$SCRATCH` (including the lanes'), the `fullclone` checkouts
`$SCRATCH/RUN/head`, `RUN/base`, `ROCK/src` and `DRIFT/merge`, and this review's refs (a head that exists only as a ref is lost); it keeps
`$EVIDENCE` and prints `removed <n> worktrees, <m> refs`.
