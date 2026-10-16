# Runbook: deep review

Each stage ends in a **Gate:**; do not start the next stage until it holds. This runbook
needs subagents; without them, stop and tell the user to run `hipdnn-pr-quality`. Variables:
[lanes/common.md](lanes/common.md). Your formats: [report.md](report.md).

## Pin and intake

`<pr>` is the PR number, `<N>` the round. `<owner>/<repo>` comes from the PR URL, else
`origin` of `$CLONE`, else `ROCm/rocm-libraries`; pass `-R <owner>/<repo>` to every `gh pr` and
`gh issue` call. Commands run from `$EVIDENCE/r<N>/`; GNU coreutils, grep and sed are assumed.

`$KEY` is `pr<pr>`, else the branch name or `<base>..<head>` text with characters outside
`[A-Za-z0-9._-]` replaced by `-`. `$TARGET` is the PR's base branch, `develop` for
`branch:<name>` unless told otherwise, or the left side of a range. A tool call is a new
shell: this block saves the variables to `PIN.md` and a loader, and every later block starts
by sourcing it, which stops when a variable is unset:

```bash
mkdir -p <scratch dir> <evidence dir>
export SKILL=<skill dir> EVIDENCE=$(cd <evidence dir> && pwd -P) SCRATCH=$(cd <scratch dir> && pwd -P)
export CLONE=${CLONE:-$SCRATCH/clone} KEY=<review key>
export REPO="$SCRATCH/r<N>/head" REPO_BASE="$SCRATCH/r<N>/base"
mkdir -p "$EVIDENCE/r<N>" && cd "$EVIDENCE/r<N>"
for v in SKILL EVIDENCE SCRATCH CLONE KEY REPO REPO_BASE; do echo "$v: ${!v}"; done > PIN.md
cat > env.sh <<'EOF'
. <(awk -F': *' '/^(SKILL|EVIDENCE|SCRATCH|CLONE|KEY|TARGET|HEAD|BASE|REPO|REPO_BASE|KNOWLEDGE):/{print "export " $1 "=\"" $2 "\""}' "$(dirname "${BASH_SOURCE[0]}")/PIN.md")
: "${EVIDENCE:?}" "${SCRATCH:?}" "${CLONE:?}" "${KEY:?}"; cd "$(dirname "${BASH_SOURCE[0]}")"
EOF
```

Use `$CLONE` as given, else make a blobless clone with full history (never `--depth` or
`--shallow-since`: a wrong merge base with exit 0). Fetch head and target into
`refs/review/$KEY/` with `+` refspecs, using the block for the target type, then resolve:

```bash
. <evidence dir>/r<N>/env.sh
git -C "$CLONE" rev-parse --git-dir >/dev/null 2>&1 ||
  git clone --filter=blob:none --no-checkout "https://github.com/<owner>/<repo>.git" "$CLONE"
git -C "$CLONE" config gc.auto 0
# PR (a stacked PR's TARGET is not develop)
TARGET=$(gh pr view <pr> -R <owner>/<repo> --json baseRefName --jq .baseRefName)
git -C "$CLONE" fetch origin "+refs/heads/$TARGET:refs/review/$KEY/target" "+refs/pull/<pr>/head:refs/review/$KEY/r<N>"
# branch:<name>
TARGET=${TARGET:-develop}
git -C "$CLONE" fetch origin "+refs/heads/$TARGET:refs/review/$KEY/target" "+refs/heads/<name>:refs/review/$KEY/r<N>"
# <base>..<head>; a side that is a SHA: update-ref "refs/review/$KEY/target" (or .../r<N>) to it, no fetch
TARGET=<base>
git -C "$CLONE" fetch origin "+refs/heads/<base>:refs/review/$KEY/target" "+refs/heads/<head>:refs/review/$KEY/r<N>"

HEAD=$(git -C "$CLONE" rev-parse "refs/review/$KEY/r<N>^{commit}")
BASE=$(git -C "$CLONE" merge-base "refs/review/$KEY/target" "$HEAD")   # PR, branch
BASE=$(git -C "$CLONE" rev-parse "refs/review/$KEY/target^{commit}")   # range: the left side
export TARGET HEAD BASE; printf 'TARGET: %s\nHEAD: %s\nBASE: %s\n' "$TARGET" "$HEAD" "$BASE" >> PIN.md
[ "$BASE" != "$HEAD" ] || echo "empty diff: PR already merged by merge commit?"
```

A stacked PR's parent: `gh pr list -R <owner>/<repo> --head "$TARGET" --json number,url --jq '.[0]'`
(`none` for `develop`).

The sparse set is the project root of each touched project (`projects/<x>`,
`dnn-providers/<x>`, `shared/<x>`; all of `projects/hipdnn` when it is touched), each other
touched directory, and root `tools/` when `projects/*/tools/` changes.

```bash
. <evidence dir>/r<N>/env.sh; : "${HEAD:?}" "${BASE:?}" "${REPO:?}" "${REPO_BASE:?}"
git -C "$CLONE" diff --name-only --no-renames "$BASE" "$HEAD" > files.txt
{ awk -F/ 'NF == 1 { next }
    $1 == "projects" || $1 == "dnn-providers" || $1 == "shared" { if (NF > 2) print $1 "/" $2; next }
    { sub("/[^/]*$", ""); print }' files.txt
  /usr/bin/grep -qE '^projects/[^/]+/tools/' files.txt && echo tools; } | sort -u > sparse.txt
git -C "$CLONE" diff --numstat --no-renames "$BASE" "$HEAD" > numstat.txt
git -C "$CLONE" log --format='%H %s' "$BASE..$HEAD" > commits.txt
git -C "$CLONE" worktree add --no-checkout --detach "$REPO" "$HEAD"
git -C "$CLONE" worktree add --no-checkout --detach "$REPO_BASE" "$BASE"
for w in "$REPO" "$REPO_BASE"; do git -C "$w" sparse-checkout set --cone --stdin < sparse.txt
  git -C "$w" checkout HEAD; done
```

For a PR, save its description, comments, inline threads, files and checks under `pr/`:

```bash
. <evidence dir>/r<N>/env.sh; : "${HEAD:?}"
mkdir -p pr; gh pr view <pr> -R <owner>/<repo> --json title,body,baseRefName,headRefName,headRefOid,files,comments,reviews > pr/pr.json
jq -r .body pr/pr.json > pr/body.md
gh api --paginate "repos/<owner>/<repo>/pulls/<pr>/comments" | jq -s add > pr/inline.json
gh api --paginate "repos/<owner>/<repo>/pulls/<pr>/files" | jq -s add > pr/files.json
gh api --paginate "repos/<owner>/<repo>/commits/$HEAD/check-runs" \
  --jq '.check_runs[] | [.name,.conclusion,.html_url] | @tsv' > pr/checks.tsv
[ "$(jq -r .headRefOid pr/pr.json)" = "$HEAD" ] || echo "PR head moved since pin"
```

With no PR, save the intent document as `intent.md`. An empty description is an `ORCH`
`process` finding.

Complete `PIN.md` (append). `Build host: yes` needs `cmake`, `ninja`, a C++ compiler, a ROCm install
(`${ROCM_PATH:-/opt/rocm}`) and 25 GB free in `$SCRATCH`; else `no (<missing>)`. `PR bot` is
the `therock-pr-bot` check. Round N > 1 also records the previous head and base, saves `git
-C "$CLONE" range-diff <old-base>..<old-head> <new-base>..<new-head>` to `range-diff.txt` and
writes `affected.txt` (*Iterate*).

**Gate:** both SHAs and both sparse worktrees exist; for a PR, `$HEAD` equals `headRefOid`
(else re-pin if the user wants the new head) and these counts match:

```bash
. <evidence dir>/r<N>/env.sh; : "${HEAD:?}" "${BASE:?}"
git -C "$CLONE" diff --name-only -M "$BASE" "$HEAD" | wc -l; jq length pr/files.json
```

On a mismatch, compare with `--no-renames` on both sides (add renamed files' old paths).
At 3000 files GitHub truncates: record `Files: unverifiable (>3000)` and rely on the SHA
match. Any other difference: re-fetch; if it persists, stop and report both lists.

## Orient

Export `$KNOWLEDGE` and append it to `PIN.md` as `KNOWLEDGE: <path>`, outside `$CLONE`, `$REPO` and `$REPO_BASE`, as the first writable of
`knowledge:<dir>`; a durable-knowledge directory the host's agent instructions name;
`${TMPDIR:-/tmp}/hipdnn-review-knowledge-$USER/`; `$EVIDENCE/knowledge/`. Record the choice
and why an earlier one failed in `PIN.md`.

Assign each changed file to one area (first matching row); a path no row matches takes
`<project>-<first directory below the project root>` as its slug (`hipdnn-docs`):

| Area | Slug | Paths |
|---|---|---|
| Frontend | `frontend` | `projects/hipdnn/frontend/`, `projects/hipdnn/python/` |
| Backend | `backend` | `projects/hipdnn/backend/` |
| Data and FlatBuffers SDK | `data-sdk` | `projects/hipdnn/data_sdk/`, `projects/hipdnn/flatbuffers_sdk/` |
| Plugin, test SDK | `plugin-sdk`, `test-sdk` | `projects/hipdnn/plugin_sdk/`, `projects/hipdnn/test_sdk/` |
| Integration tests | `integration-tests` | `dnn-providers/integration-tests/` |
| Provider `<x>` | `provider-<x>` | `dnn-providers/<x>/` |
| Core tests, build | `core-tests`, `core-build` | `projects/hipdnn/tests/`; `projects/hipdnn/cmake/`, `projects/hipdnn/CMakeLists.txt`, `projects/hipdnn/scripts/` |
| Build and CI | `build-ci` | `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `.github/` |
| Shared `<x>` | `shared-<x>` | `shared/<x>/` |
| Samples, tooling | `samples`, `tooling` | `projects/hipdnn/samples/`; `projects/hipdnn/tools/`, `tools/` |
| Docs | `docs` | documentation files ([policy.md](policy.md) *Review rules*) |

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
cat CONTEXT.md orient/*.md | /usr/bin/grep -oE '`[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)+(:[0-9]+)?`' |
  tr -d '`' | sed 's/:[0-9]*$//' | sort -u |
  while read -r p; do git -C "$CLONE" cat-file -e "$BASE:$p" 2>/dev/null || echo "not at base: $p"; done
```

Fix each `not at base` citation by hand or re-dispatch the scout. Then publish `CONTEXT.md`
and `orient/*.md` to `$KNOWLEDGE/<KEY>-r<N>/` (the store's own conventions win), replacing
only this review's entry.

## Partition

A rename is two `numstat.txt` rows; a binary file (`-<TAB>-<TAB>path`) is one owned file of
zero lines; lines are added plus deleted. Form correctness lanes `CORR1..k` by area:

- Target ~2000 lines each; a total at or under it is one lane. Split only for distinct
  contexts (areas that share no contracts) and record why in the ledger Lanes table.
- Keep a header with its implementation, a producer with its consumer, a test with its code.
  Never split a file; a file over the target gets its own lane.
- Generated code, golden data and bundles get their own `CORR<k>`, packet marked `Sampled:
  <generator or source of truth>; sample <files>`. When every changed file is documentation,
  `DOCS` owns all paths and no `CORR` lane runs.

Write each packet ([lanes/common.md](lanes/common.md) *Context packet*) from `owners.tsv`,
`numstat.txt` and each file's area; do not read the diff. This table owns every lane's scope.
Run a lane when its trigger holds, else mark it `N/A`; when unsure, run it.

| Lane | Brief | Reviews | Runs when |
|---|---|---|---|
| `CORR<k>` | [correctness](lanes/correctness.md) | Logic, error paths, ownership, code quality of one partition | Non-documentation file changes |
| `TESTQ` | [test-quality](lanes/test-quality.md) | Whether each test fails on a plausible bug | Non-documentation file changes |
| `TESTB` | [test-breadth](lanes/test-breadth.md) | Which changed behaviors have tests at all | Non-documentation file changes |
| `AUTH` | [author-testing](lanes/author-testing.md) | What the author and CI ran against what the change needs; M4, M5, PR bot | Always |
| `BUILD` | [build-packaging](lanes/build-packaging.md) | CMake targets, option defaults and registration, install rules, exported config, CI files | `CMakeLists.txt`, `*.cmake`, `CMakePresets.json`, `.github/` or `version.json` changes |
| `DEPS` | [dependencies](lanes/dependencies.md) | New dependencies and boundary crossings | CMake, `pyproject.toml`, `requirements*.txt` or `.github/` changes |
| `API` | [public-api](lanes/public-api.md) | Public surfaces, behavior changes in source, docs of new public declarations | A file in the brief's surfaces table or non-test source under `projects/hipdnn/` changes |
| `ROCK` | [therock](lanes/therock.md) | Whether TheRock CI builds and tests the change: skip-ci, job, family and tier selection, flags, RPATH, CTest files, artifact split, Windows lookup | Non-documentation file changes |
| `LEAK` | [leaks](lanes/leaks.md) | Private information in the change and in review output | Always |
| `DUP` | [dedup](lanes/dedup.md) | Reuse of existing code; repetition within the change | Code is added |
| `DOCS` | [docs](lanes/docs.md) | Accuracy and concision of docs and comments, including edited public headers | A documentation or source file changes |
| `RUN` | [build-validate](lanes/build-validate.md) | Building the head and running the relevant tests | Always |

`ORCH` (process findings you file), `SCOUT<k>`, `SYNTH` ([orient](lanes/orient.md)) and `VER`
([verify](lanes/verify.md)) are not partition choices. Merge two lanes into one agent when the
host's agent limit requires it; split a concern lane over ~4000 changed lines by area into
`<LANE><k>`. Write `owners.tsv` (`owner<TAB>path` for every path of a correctness lane, or of
`DOCS`) and the `LEDGER.md` seed ([report.md](report.md)); from round 2 append this round's Coverage and Lanes sections
and keep every existing row. Its Lanes table gives the reason for every `N/A`, merge and
split.

**Gate:** each `numstat.txt` path is owned by exactly one correctness lane (or by `DOCS`) and
the owned lines sum to the total. This exits 0 only then:

```bash
. <evidence dir>/r<N>/env.sh
[ -s owners.tsv ] || { echo "owners.tsv empty"; false; }
awk -F'\t' '
  FNR == NR && FILENAME == ARGV[1] { own[$2]++; next }   # owners.tsv
  { n++; seen[$3] = 1; l = ($1 == "-" ? 0 : $1 + $2); total += l
    if ($3 in own) owned += l; else { print "unowned: " $3; bad = 1 } }
  END { for (p in own) {
          if (own[p] > 1)   { print "owned twice: " p; bad = 1 }
          if (!(p in seen)) { print "not in numstat: " p; bad = 1 } }
        if (owned != total) { print "owned " owned " of " total " lines"; bad = 1 }
        print n " files, " total " lines"; exit bad }
' owners.tsv numstat.txt
```

## Fan out

Dispatch each lane, scout, verifier and the writer as its own fresh agent with shell and
write access, concurrently up to the host's limit; fill this template as its first message.
`RUN` and an approved `ROCK` build run concurrently in their own `$SCRATCH` trees and split
the cores.

```markdown
Role:      <LANE> | SCOUT<k> | SYNTH | VER-<batch> | WRITER; round <N>; follow-up f<k> if any
Read:      lanes, verifiers: $SKILL/lanes/common.md, findings.md, policy.md, then <brief path>.
           Scouts, SYNTH: lanes/common.md, lanes/orient.md. WRITER: report.md *REPORT.md*, policy.md
Env:       export SKILL=… EVIDENCE=… SCRATCH=… CLONE=… KNOWLEDGE=… REPO=… REPO_BASE=…
                  HEAD=… BASE=… KEY=… TARGET=…   (absolute); <owner>/<repo>; PR <pr> | none
Inputs:    under $EVIDENCE/r<N>/: PIN.md; pr/body.md or intent.md; CONTEXT.md and the area
           files for its areas (verifier: the areas of its findings' Where paths);
           packets/<LANE>.md; LEDGER.md rows in scope (round 2 on).
           Scout: area, slug, changed paths, prior area file or none. SYNTH: orient/*.md.
           Verifier: finding IDs and report paths. WRITER: LEDGER.md, verify/, PIN.md,
           pr/pr.json (title), CONTEXT.md, lanes/*.md (M-rule lines, RUN execution counts,
           ROCK not-run items). Follow-up: the lane report path plus the approved request,
           decision or Handoff.
Output:    <exact file>; scratch $SCRATCH/<ID>/; first provisional ID p<n>
Approved:  <verbatim commands with architecture and walltime> | none
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

An agent that fails, times out or returns a non-conforming report twice is `FAILED
(<reason>)` in the Lanes table; split an over-large lane by area before the second attempt.
Its Coverage rows stay `pending`; you file an `ORCH` `process` finding at `IMPORTANT`
(`<lane> did not complete: <reason>`), listed under `Closure: OPEN`. A failed verifier's
findings are `UNVERIFIED`.

**Gate:** every dispatched lane returned a conforming report ([findings.md](findings.md)
*Lane report*), `N/A` or `STOP:`, or is `FAILED`; every Coverage row has a read depth, is
carried forward or belongs to a `FAILED` lane. Re-dispatch a non-conforming report once, naming the rule.

## Merge and verify

1. Assign final IDs and write `ID-MAP.md` ([report.md](report.md) *Final IDs*); copy each
   lane's read depth and tests into the ledger Coverage rows. Cluster findings with the same
   location and claim; keep one, mark the rest `SUPERSEDED-BY:<ID>`. File each lane's
   M-rule `FAIL` line as an `ORCH` `process` finding at `BLOCKING` ([policy.md](policy.md) *M-rules*).
2. Route **Handoffs** to a follow-up dispatch of the target lane or into the ledger as a
   candidate. A `SCOUT` handoff corrects an Orient file: edit a line in `orient/<slug>.md`
   and its published copy; for more, re-dispatch that scout, then `SYNTH`.
3. Send conflicting findings and each lane's own **Needs human decision** entries to the
   ledger's Open questions, citing [policy.md](policy.md) *Precedence*.

Verify every candidate with an agent that did not report it ([lanes/verify.md](lanes/verify.md)):
`BLOCKING` and `IMPORTANT` in batches of at most 10 per verifier, grouped by lane;
`SUGGESTION` and `FUTURE WORK` in batches of at most 30. Re-dispatch a verifier whose files
lack the fields in [findings.md](findings.md) *Verifier output*. Verify `VER` candidates with
a different verifier; send each `REFUTED` `BLOCKING` or `IMPORTANT` one to a second
([findings.md](findings.md) *Verdicts*).

**Gate:** every candidate not `SUPERSEDED-BY` another has `verify/<ID>.md` (and `<ID>-2.md`
where a second verifier ran) and a ledger verdict.

## Synthesize

Dispatch one `WRITER` for `REPORT.md` ([report.md](report.md) *REPORT.md*), or write it
yourself from the same inputs. Group the `OPEN` and `UNVERIFIED` findings by theme
into the fix list, ordered by tier then consequence; batch `SUGGESTION` and `FUTURE WORK` by
theme; fold M-rule failures into the **Verdict** and fix groups. M1 and M3 evidence comes from
`TESTQ`, M2 from `TESTB`, test execution for all three and M4-M5 from `AUTH`; an `N/A` lane
gives a vacuous `PASS`. List each human decision under **Decisions needed**. Set `Closure:`
from [SKILL.md](SKILL.md) *Completion*. Write `COMMENTS.md` as drafts. Re-dispatch `LEAK` for
its output sweep ([lanes/leaks.md](lanes/leaks.md); output `$EVIDENCE/r<N>/lanes/LEAK-sweep.md`) before the user sees either file; redact each hit it reports in the output and in the published `$KNOWLEDGE` copy, then sweep again.

**Gate:** every statement in `REPORT.md` traces to a ledger row or a lane report, it has its
`Closure:` line, and the `LEAK` sweep of the output is clean.

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
Re-verify every affected `OPEN` or `UNVERIFIED` finding. A lane not re-run carries forward
(Coverage and Lanes rows copied, marked `carried from r<M>`).

On the same head, after a decision, approval, new evidence or to re-run a `FAILED` lane, reopen the round at *Merge and
verify*: record the decision, run the follow-ups, re-verify what it affects and repeat
*Synthesize*, updating the round's files in place.

Close a finding only when its proof ([findings.md](findings.md) *Proof per kind*) no longer
holds at the new head; author rationale alone closes it only as a human `ACCEPTED-RISK`.

**Gate:** [SKILL.md](SKILL.md) *Completion*. On closing, remove the review worktrees
(including the lanes') and this review's refs (a head that exists only as a ref is lost);
keep `$EVIDENCE`.

```bash
. <evidence dir>/r<N>/env.sh
git -C "$CLONE" worktree list --porcelain |
  awk -v c="$CLONE" -v s="${SCRATCH%/}/" 'sub(/^worktree /, "") && $0 != c && index($0, s) == 1' |
  xargs -r -d '\n' -n1 git -C "$CLONE" worktree remove --force
git -C "$CLONE" for-each-ref --format='delete %(refname)' "refs/review/$KEY/" |
  git -C "$CLONE" update-ref --stdin
```
