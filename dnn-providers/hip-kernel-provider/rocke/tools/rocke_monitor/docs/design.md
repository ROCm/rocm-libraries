<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: design

The rocKE monitor keeps `develop` green while rocKE pull requests land. It reviews every new or changed PR, plans
and gates the merges against today's develop, checks develop after every landing, and puts every decision that is not
its own in front of one operator. This document says why it is built this way: what each part is for and how the
parts fit. [QUICK_USER_GUIDE.md](../QUICK_USER_GUIDE.md) is how to run it, [README.md](../README.md) the reference for
every command and setting, and [operations_guide.md](operations_guide.md) the overview, its guardrails and limits
included.

## Goal

- **Review** every new in-scope PR and every substantive change to one: the toolkit's scripted checks, an optional
  first-pass draft, then Claude's verifier, which reproduces or cuts every claim before one comment is posted per head.
- **Plan and gate** the merges: a merge plan of conflict-free waves on today's develop, and gates G1–G9 checked against
  today's develop, not the base a PR's CI last saw.
- **Check develop** after every rocKE landing, find the culprit of a break, and see the break owned and fixed.
- **Leave the decisions to the operator**: approvals, merges, priorities, and when each feature goes live.

The monitor starts safe: the deploy template has review automation and the merge queue in `prototype` (all their work
done, and what they send kept on this machine as unsent items), and auto-merge, the alerts, the team dashboard and the
tickets off. Each feature goes live, or on, only on the operator's word.

## What GitHub guards, and the gap

| Fact | Consequence |
|---|---|
| develop's branch protection requires a set of status checks and a code-owner review | Necessary, not sufficient |
| It does not require a PR to be up to date with develop, and there is no merge queue | A PR merges on green checks from a **stale base**. Golden, ABI and semantic conflicts with work that landed since are never re-tested. This is the gap the monitor closes |
| Auto-merge is allowed | `gh pr merge --auto --squash --match-head-commit <sha>` can arm one PR at a time |
| The monitor's account comments, files issues and arms auto-merge; it has no maintain or admin role | It can never bypass a protection, and never tries |
| develop CI runs on every push | It shows a develop failure late; a local sanity run after each landing shows it sooner |
| CI's test lanes have no PyTorch and no built C++ engine module, and some architecture lanes skip on PRs | Arch coverage and engine parity have to be checked per PR; green CI does not show them |

## Architecture

| Part | What it does | Never does |
|---|---|---|
| **The toolkit, `rk`** (Python; `rkmon.py` and `rkmon_lib/`) | Every mechanical step: GitHub reads (GETs only), events, the plan of work, the merge plan, gates, batches, ABI lanes, develop sanity runs and bisects, drift checks, the review pipeline's staging and checks, dashboards, state, journal and ledger. It also has the only write path: a fixed set of guarded `rk` commands | Write on its own: every outward write is an `rk` command that the Claude session runs, that a tick run by fixed rules runs (`rk rules-tick`), or that the operator types |
| **The orchestrator** (one long-lived Claude Code session, `rk orchestrate`, running the `/rocke-monitor` skill on `features.review_automation.orchestrator.model`) | Runs `rk tick` every `monitor.tick_interval_minutes` minutes, gets a turn only for the tick lines that need Claude, verifies each finished review with its two review agents, decides, and carries out each decision with `rk` commands | Approve or merge a PR, dismiss a review, bypass a protection, write to GitHub or send anything other than through `rk` |
| **The tick triage** | Reads each tick's output before the session: by default the toolkit's rule filter (`rk tick-filter`), with no model; optionally a small model (`models.tick`) that also does the routine work with read-only `rk` commands | Hand the session fewer lines than the rule filter would |
| **The review driver** (`rk review-driver`) | Keeps the parallel reviews full every minute: stages each picked PR, starts its first passes and its checks job, and prints `REVIEW_READY #N` when they are done. The orchestrator keeps it running | Verify or post |
| **The review agents** (Claude subagents) | The rocKE Review Verifier (`features.review_automation.passes.verification_pass.model`) reviews blind, then reproduces or cuts each draft claim, folds in the code quality pass and writes the post. The Code Quality Reviewer (`features.review_automation.passes.code_quality_pass.model`) runs the code quality pass beside it | Write to GitHub: their `gh` is the read login |
| **The optional passes** (`features.review_automation.passes`) | Each runs on a model only once one is named; by default (`default`) the Claude pass it helps does its work. The first review pass, the reviewer model, drafts each review: the local review loop (a model on an Ollama server or behind an OpenAI-compatible API) or the tool loop (a Claude model or another binding); its tools only read the PR's trees: no GitHub, no network, no shell. The CI triage, first code quality and notice draft passes hand a Claude pass leads, which it checks before it uses one ([README.md, "The two model roles"](../README.md#the-two-model-roles)) | Verify, decide a gate or post: those roles are always Claude's |
| **The scaler** | Moves the parallel reviews and the parallel builds in effect one step at a time, each within its range, by the machine's load, and starts a review or a build only when it fits | Stop running work |
| **The dashboards** | The Markdown and HTML dashboards each full tick; the admin dashboard (the operator's page, with Approve and Decline) served by `rk serve`; optionally a team dashboard for other teams, with a priority request box | Grant a request: each one waits for the operator's Approve or Decline |

### The deployment

The monitor runs in one container. The image has the toolkit, Python with the checks' venvs, git, gh, the Claude
Agent SDK with the Claude Code CLI, cmake, ninja and a compiler, and, as the settings choose, a ROCm release (with its
development files, so hipcc compiles `<hip/hip_runtime.h>`) and the atlassian CLI. The container has no GPU: GPU-only
test cases run on a Slurm cluster's GPU nodes with remote GPU tests on (`features.review_automation.remote_gpu_tests`).
The state lives in the volume `/work`, the confirmed deploy settings in the read-only `/etc/rkmon/settings`, the
releases of the package's code that root installs over the image's own in `/code`, the monitor's secrets in
`/secrets`, and the writer's (the GitHub write login, the atlassian login, the Teams webhook and the web host's key)
in `/secrets-write`; nothing of a deployment is baked into the image.

`docker/rkmon` drives two jobs, on any machines, each step ending with the next command. The image is built from a
build settings file (`rkmon build --settings FILE`: one `docker build`, each setting a build argument and each download
location a build secret, in no layer), tested (`rkmon image test`) and carried as a file (`rkmon image save`) or
through a registry. The deploy machine builds nothing: `rkmon template deploy` (a deploy settings file to fill in),
`rkmon check` (the host and the deploy settings file), `rkmon confirm` (the file recorded, then the settings in effect
shown, checked and confirmed: start and apply refuse other settings), `rkmon image load` (the image, checked: built
from this package's image recipe), `rkmon login` (each login and secret the settings need), `rkmon start` (the first
time, the confirmed settings delivered, and the deployment's `.env`, with the users' ids, and compose override written
from them), `rkmon doctor` and `rkmon first-tick`. `rkmon next` says where a deployment is on that path.
Later: `rkmon confirm`, then `rkmon apply`, change the running deployment (new settings, a new image's tag, or a
newer package's code that `rkmon package update` brought), with the doctor before and after, and put back what ran
before on a failure; `rkmon rollback` goes back. `rkmon adopt` moves an existing monitor's state in, and the
deployment then starts in standby, sending nothing, until `rkmon standby off`; `rkmon secret` puts a secret in, never
printed.

Inside the container, the entry point keeps the parts the features run (`rk features --runs PART`) and starts again
any that ends: the admin dashboard's service (with `features.dashboards.admin.enabled`), the orchestrator (while
review automation is on), and the tick loop that runs without the orchestrator. That loop runs a tick every
`monitor.tick_interval_minutes` minutes while the orchestrator is not set up, and otherwise only when the
orchestrator's ticks are late (`rk tick-due`: no tick for twice the interval plus 5 minutes, 25 by default). With the
merge queue on and no Claude session (review automation off), it runs each tick by fixed rules (`rk rules-tick`):
each line whose action is fixed runs its `rk` command, and every other line waits for a person on the admin
dashboard's Needs-you list. The container's restart policy brings it back after a reboot, and the orchestrator
resumes its session.

## The tick

Every `monitor.tick_interval_minutes` minutes (10 by default) the orchestrator runs `rk tick`:

1. **Idle or full.** The tick first reads a change fingerprint: the in-scope PRs' heads, CI, labels and mergeability,
   develop and its watched CI runs, the open develop issues, the priority requests, the jobs, the files the reviews in
   flight wait on, the configuration and the code. When nothing changed and no timer is due, the tick is idle: one
   line, and the dashboards' "checked" stamps. A full tick runs at least every `monitor.full_tick_max_interval_minutes`
   minutes.
2. **A full tick** polls GitHub (GETs only) and turns what changed into events; reads finished drift checks and the
   develop issues; marks merge-only pushes; recomputes the merge queue and the merge plan; sweeps open PRs for fixes to
   develop issues; advances the develop guard, the merge queue and the review pipeline (each while its feature is
   on); cleans up; polls remote GPU jobs and priority requests; refreshes the overlap scan; writes the board, the state,
   the journal and the dashboards; and prints its lines (`!! ` alerts, `REVIEW_READY`, `PROPOSE`, `DRIFT_FAIL`, ...).
3. **The tick triage** passes the session only the lines that need Claude: an audit, any outward action, or a line
   no rule knows. A standing line (one the tick repeats while its condition lasts) goes once, until it goes away and
   comes back.
4. **The session** answers those lines with `rk` commands, following the skill's procedures, and reports critical
   items first.

A quiet tick costs one command and one line, and no Claude turn.

## Reviews

### The pipeline

- **Pick.** In plan order: the review queue first, then develop fixes and the operator's priorities (P0), final
  reviews of PRs close to merge (P1), other near-ready follow-ups (P2), first rounds (P3) and other follow-ups (P4).
  A head must have settled: `features.review_automation.schedule.push_cooldown_minutes` (30) after the push, 45
  minutes after the last of a burst, at most 10 for a develop fix or a priority, a burst's too. A follow-up waits
  `features.review_automation.schedule.follow_up_cooldown_minutes` (60) after our last review of the PR, posted or
  kept unposted, except a merge-only push and the PRs in `followup_gap_exempt` (develop fixes and priorities by
  default). The review driver never starts a PR over
  `features.review_automation.large_prs.in_depth_review_over_lines` (30,000 changed lines) from the plan: the tick
  prints `REVIEW_MANUAL #N`, and the orchestrator starts it, a first round only once the operator approves it (Giant
  PRs, below).
- **Classify** (follow-ups). Comparing the PR's own changes at our reviewed head and now: **rebase-only** (or, after a
  🟢/🟡 review, a rebase plus golden re-records only) keeps our verdict and posts nothing; **merge-only** (the new
  commits only bring develop in) runs the checks at once; **substantive** runs the full pipeline.
- **Stage, draft and check, in parallel.** As one of the parallel reviews, the PR is staged (its context fetched with
  GETs, a worktree at the head and one at the head merged with develop). With a first review pass, its draft starts,
  beside the first reads of the CI triage and first code quality passes when they have a model; by default there is
  no first-pass drafter, and the verifier reviews alone. As one of the parallel builds, the checks job builds and
  tests the merged tree: native build and ctest, byte identity at every emission flavor, goldens and the differential
  golden, the PR's own tests and those that import what it changes, the installed-tree run, the compile checks, saved
  probes of our open findings, CI at the head (earlier attempts included), and scripted leads that cost no model
  (clashes with other open PRs, a mutation sweep, CTest tiers, moved references, docs against code, an engine import
  probe, a simulated Windows lane, the peer-review checks, the ABI scan, overlapping work in other open PRs).
- **Verify.** On `REVIEW_READY #N`, the orchestrator runs `rk dispatch-pack N` and briefs the verifier, and, unless
  the round is exempt, the code quality pass beside it. The verifier writes its own findings before it opens any draft,
  then reproduces or downgrades every draft claim, gives every prior item (other reviewers', bots', and ours) its
  status, and writes the post and its notes, with the premises the post depends on.
- **Post.** `rk spec-post N --round R` builds the post's spec from the verifier's notes and live reads, stops on any
  difference, and posts through `rk post --premises`: the head, develop's change to the PR's files, the conflict set,
  CI and the threads are re-checked, then line 1, the public-repo hygiene, one post per head and the hourly cap.

### What a post says

- **Scope first.** The review starts from the PR's stated scope. **Needed to merge** lists every blocker, each as its
  own row with its light, with no cap: wrong results, a build, CI or ABI break the PR causes, a regression in code it
  touches, a test it adds that cannot fail, a claim in its body or commits that is false at head. **Highly recommended
  follow-up (not blocking)** lists only significant items outside that scope (pre-existing problems, design and
  duplication improvements, coverage beyond the fix), each with a 🟠 or 🟡 light for its importance; no nits.
  Correctness and breaking changes are never follow-ups, and an out-of-scope item is never a blocker.
- **The header light** is the worst open Needed-to-merge severity, prior rows included; follow-ups never set it.
- **Evidence.** Every claim is marked `(ran)` only when a check output shows the run at this head, otherwise `(read)`.
  The words ACCEPT and REJECT are never used: a post flags whether a closer look is needed, and is never an approval.
- **Line 1** gives the automated reviewer name and the drafter's model, or the verifier's when there is no
  first-pass drafter; the footer credits the verifier.

### Rounds

- **Merge-only heads are carried.** A merge-only head whose checks are clean is recorded as covered by our review, and
  nothing is posted. So is one whose only flags repeat what our review saw: CI failures that are all known
  infrastructure, the same unanswered threads and title problem, and the same failing tests and PR references our
  review already flagged or our last review still lists. A carried head keeps our verdict, and the drift check,
  the conflict tracking and CI keep watching it.
- **Micro rounds.** When a merge-only head would be carried but has new items our review did not see (unanswered
  threads, a title problem, or reviews and comments others left since our last review), the verifier alone (no
  code quality pass) covers those items and carries every other row of our last review. A head whose CI failures are all
  known infrastructure and that has new items is a micro round too.
- **Giant PRs.** A PR over `features.review_automation.large_prs.in_depth_review_over_lines` gets one in-depth review,
  which starts only once the operator approves it: until then nothing of it runs, and Needs you asks for the approval
  with the PR's size, about how many agents it runs and their cost (Approve on the admin dashboard or
  `rk review-approve N`; Decline there or `rk review-skip N`). The approval stands for that round, whatever the head
  does before the review starts. `rk review-split N` cuts its own diff into parts, each filled up to 8,000 changed
  lines by default area by area (`features.review_automation.large_prs.split_part_lines`; `--max-lines` overrides it),
  keeping files that change together in one part and summarizing generated, golden and data files; one verifier per
  part, which runs the part's code quality checks too, in waves of at most 20 agents, then one cross-cutting verifier
  writes the single post. The orchestrator starts every later round too (`REVIEW_MANUAL`), with no approval, as a light
  follow-up: one verifier, no code quality pass, no split, checking each of our open findings at the new head and the
  delta for breaks, conflicts, CI and new threads.
- **The code quality pass** runs on every first round and on every follow-up whose new commits change more than docs or
  tests answering our own findings; never on a micro round.

## Merge gates

Every tick works the gates out again, at the PR's head on today's develop. G5–G7 reuse the results of the latest
checks run on the PR merged with develop, by a gate job or by the review's own checks, recorded with the head, the
develop SHA and develop's rocKE tree: they stay fresh while the head is the same and develop's rocKE tree is unchanged
(a develop move outside rocKE cannot change them), and are pending otherwise.

| Gate | Passes when |
|---|---|
| **G1 Review** | Our review covers the head (a rebase-only or carried merge-only head keeps it), with no open Needed-to-merge 🔴/🟠 |
| **G2 Humans** | A code-owner approval, no outstanding change request, every review thread resolved |
| **G3 CI** | CI is green at the head and every required check has reported; a cancelled run whose check also passed is ignored |
| **G4 Fresh merge** | The PR merges cleanly with today's develop. GitHub does not require it, so the monitor does |
| **G5 Goldens** | On the merge with today's develop: the representative golden and the differential golden (a differential not run is never a pass) |
| **G6 Build and engines** | On the merge: native build, ctest, byte identity at every flavor, library parity, the PR's tests, and its changed tests in the installed tree. A failed drift check fails it too |
| **G7 Arch coverage** | The compile per target of every touched parity family; nothing compiled is pending, never a pass. Run evidence for a numerics change is judged in the review: a gap is a 🟠 finding, which blocks through G1, unless the operator waives it |
| **G8 Dependencies** | Its stack base has landed, it has no do-not-merge label, and the queue is not frozen |
| **G9 ABI lane** | It heads its ABI lane and claims develop's number + 1 |

**Arch coverage in detail.** The affected arches come from the touched paths: an arch directory means that arch; core,
lowering or IR code means every production target. Every affected arch needs compile evidence (byte identity or the
golden, and a compile showing no new spills or scratch), which G7 records. The run evidence a numerics-relevant change
needs per arch is a CI lane that really ran the tests, the author's hardware evidence in the PR, or a remote GPU run.

**The drift check.** Gate jobs run only for candidates, so a PR waiting for approval would otherwise keep its last
review's results however far develop moved. Whenever develop's rocKE tree moves past the tree a PR's tests last ran on,
a light job runs the PR's own test files on its head merged with develop and on develop alone, for every PR our review
clears that merges cleanly, approved or not. A failure on the merge only fails G6, takes the PR off the approval lists
and prints `DRIFT_FAIL #N` with the develop commits that likely caused it.

## The merge plan

The plan orders the near-merge PRs (our 🟢/🟡 review or an approval, no do-not-merge label, plus every develop fix
and priority) on today's develop:

1. **Conflicts.** Each PR is merged with develop (`git merge-tree`, no checkout); two PRs that share a file have their
   two merges merged. A textual conflict, or an **ABI pair** (two PRs that bump the same ABI counter: the later one
   renumbers), is a conflict. Shared files that are no conflict (a golden shard, a CMake list) are shown as care, never
   used to order.
2. **Priorities:** pinned groups, the top priorities (develop fixes and the operator's priorities), held PRs, and ready
   PRs the plan has kept waiting a day, in that order. Each joins the earliest wave with no higher priority it conflicts
   with, so conflicting priorities split across waves, the higher one first. A top priority that conflicts with develop
   is placed the same way, by the files and ABI lanes it shares (the PRs that share them land after it), and ends its
   wave after its rebase: in wave 1 it rebases now, in a later wave once the wave before it has landed.
3. **Waves:** each wave then takes in the best set of the other PRs left, weighted by readiness, that conflict with none
   of its members and with no priority still waiting (they land after it). A wave lands first the PRs whose landing
   makes no later PR rebase, then the others, then the top priorities that must rebase; within each part, the priorities
   first, then the readiest, then the least weighted rebasing.
4. **Hubs last:** a PR with five or more textual conflicts with other planned PRs.

Only a PR in wave 1 that merges clean with develop, with no PR ahead of it conflicting, may land now. Compatible ready
PRs (up to four) are validated together by one batch gate job, on each prefix when a member touches a golden or an ABI
counter, and land back to back with one develop sanity run. ABI numbers are assigned at landing, in plan order. Every
planned PR gets rebase advice: none needed, wait for another PR then rebase, rebase now, or renumber after another PR
lands. Humans can still merge in any order; every merge re-plans.

## The develop guard

Every develop SHA that brings rocKE changes gets a sanity run; a burst of landings is tested once, at its newest SHA.
A run covers the native build and ctest, the representative golden, the differential golden, byte identity for the
families the landed commits touched, and the ABI history. develop's watched CI workflows
(`develop_watched_workflows`) are followed to completion. Develop checks, the open develop issues and the fix sweep
run whenever the develop guard is on (`features.merge_queue.develop_guard`), in prototype too, and so do
incidents and the frozen queue; the issues the monitor files and its fix links go out only while the merge
queue is live.

A red run, or develop CI red in a rocKE job, opens an incident and freezes the queue. The culprit is the only landing
since the last green run, or the first red merge of a bisect over the first-parent merges since then; when a later
merge only turned the failing check on, that merge is the culprit and the earlier ones contributed. A develop CI run
on a commit from before an incident's merged fix, failing only in jobs the incident already saw, is old news
(`DEVELOP_RED_STALE`) and opens nothing.

## Incidents

| When | What happens |
|---|---|
| The break opens | The queue freezes, a PR the monitor armed is disarmed (`!! DISARM_NEEDED`), and the operator is told at once |
| The culprit is known | Develop breakages are GitHub issues, one source of truth. Once develop alone reproduces the break, Claude files one while the merge queue is live with its develop guard (or links an existing one), with the develop guard's `ci_red_label` when CI is red, or with its `stability_label`, assigned to the breaking PR's author, when no CI lane sees it. One develop-health notice goes on the culprit PR. A CI-red break with no fix planned also alerts the team chat |
| A fix PR opens | It is linked to the issue, gets review P0 and merge tier 0, and leads the merge plan |
| After `revert_offer_after_min` (two hours) | While the merge queue is live, the tick offers revert evidence for the operator. A revert is the develop-green team's to push, never the monitor's; once the fix has merged, the offer is replaced by `INCIDENT_FIX_MERGED` |
| develop is green again | The incident closes, the queue unfreezes, the journal records it |

Problems found along the way that are neither a develop break nor a PR's own finding get a proposed JIRA ticket, filed
only on the operator's Approve. Our should-fix items still open on develop after their PR merged get a follow-up
ticket, assigned to the PR's author. Each type has its switch in ticket creation (`features.ticket_creation`:
`found_issue_tickets`, `follow_up_tickets`), and a new ticket is linked to the epic (`epic_key`) when one is set.

## GitHub access

- **Two logins.** The read login (a fine-grained, read-only token) serves every read: the ticks, the jobs, the Claude
  session and its agents. The write login is used only by the guarded `rk` write commands. In the container the split
  is enforced: the write login lives in its own volume, readable only by the writer (the user `rkwrite`), and the
  write commands run as the writer through a root-owned broker (`docker/rk_write.py`, one sudo rule). The broker runs
  the writer's commands alone: those `rkmon_lib/outward.py`'s table gives the writer, the GitHub writes and the sends
  to the team chat, the web host and JIRA among them. Every tick probes both logins at most once an hour; a read login
  that can write is a standing `!! GH_LOGIN` alert.
- **PR code runs in the PR runners' slots.** Every command that builds or runs a PR's code (builds, tests, probes, the
  mutation sweep) runs in the container as a PR runner (`rkpr1` to `rkpr16`), in a slot of its own whose folders are
  made fresh for each job: it reads no secret and writes only its slot's folders, and the monitor reads what it keeps
  of the job's results as data.
- **The session's guard.** A hook on every shell command of the session and its agents refuses an outward action that
  is not an `rk` command (a gh write, a push, a network write, ssh, Slurm commands, the atlassian CLI's own write unless
  a ticket feature is on: `rk atlassian` weighs each of its writes itself), the `rk` commands only the operator or rk
  itself runs (`rkmon_lib/outward.py`'s table), and anything that would change state the review's agents and jobs
  share.
- **The session's file tools.** The file tools of the session and its agents write only in the folders the monitor
  works in (`paths.*`), each allowed by one rule on its absolute path (`rkmon_lib/orchestrator.py`, `write_rule`):
  Claude Code's dontAsk mode refuses every tool call the allow list lacks.
- **The ledger.** Every outward write is one ledger line, checked before each write, so nothing is done twice. A
  write whose answer was lost is read back from GitHub or JIRA before it is called failed.

## Capacity: parallel reviews, parallel builds and the RAM disk

The machine comes first. How hard a review or a build loads it depends on the pull request, and the machine may run
other work too, so no fixed count suits every hour: the parallel reviews and the parallel builds start at their
settings, then follow the machine's load one step at a time, each within its range. A lower setting holds back only
new work: what runs finishes. The RAM disk keeps the reviews' worktrees and builds in memory, for speed; it counts as
used memory, and what does not fit waits rather than spill to the disk.

- Why the reviews scale this way: item 12 of
  [review_pipeline.md, "2. Why it is built this way"](review_pipeline.md#2-why-it-is-built-this-way).
- How each setting moves, and what the RAM disk holds:
  [review_pipeline.md, "3.3 Parallel reviews"](review_pipeline.md#33-parallel-reviews) and
  [review_pipeline.md, "3.5 Local checks: the parallel builds"](review_pipeline.md#35-local-checks-the-parallel-builds).
- The RAM disk's size, and the container's memory it counts against:
  [QUICK_DEPLOY_GUIDE.md, "Machine sizing"](../QUICK_DEPLOY_GUIDE.md#machine-sizing).

## State, resume and compaction

Everything lives in the monitor's home (`/work/monitor` in the container).

| File | What it keeps |
|---|---|
| `state/state.json` (whole-file writes under a lock) | The frozen queue and the cleared PRs; develop's last green SHA; every PR (head, our last review and verdict, gates and drift results per head and develop SHA); the holds; the jobs; the queue, batches, ABI lanes and the merge plan; develop fixes; priority requests; the review pipeline |
| `state/journal.md` | Append-only, one line per decision, action or event |
| `state/actions.jsonl` | The ledger: every outward action with its key and URL, checked before each write |
| `state/jobs/<id>/` | One background job: its script, log, process id and exit code |
| `state/incidents.json` | Develop incidents: SHAs, culprit, contributors, fixer, issue, times |
| `state/scaler.jsonl`, `state/claude_usage.jsonl`, `state/triage.jsonl` | The scaler's decisions; the Claude tokens per turn and per subagent; what the tick triage did |
| `state/orchestrator.json` | The Claude session to resume |

- **Resume.** Every stage is idempotent and on disk. The orchestrator resumes its session at each start and sends
  `/rocke-monitor resume`, which re-checks what was in flight; a review awaiting verification that no agent is
  working on is handed to the session again. The ledger blocks duplicates.
- **Compaction.** Each tick reloads the skill's tick card and reads state from disk, so nothing lives only in the
  conversation. The session compacts at `claude.auto_compact_window` tokens.

## Guardrails

The monitor works on other people's pull requests, in a public repository, under the operator's GitHub login, so its
guardrails keep each decision with a person and make every write deliberate:

- **People decide.** It never approves, merges directly or dismisses a review, so whether a PR merges stays a
  person's call ([What GitHub guards, and the gap](#what-github-guards-and-the-gap)). What only a person can put right
  reaches the operator at once: develop goes red, a PR merges over a hold or an open 🔴/🟠, a posted comment proves
  wrong, a GitHub or JIRA login fails, or the read/write split breaks.
- **One write path.** Every outward write is a guarded `rk` command, checked against the ledger first, so the post
  guards, the features' switches and the stops hold for every write, and nothing goes out twice
  ([GitHub access](#github-access)).
- **Nothing goes out unasked.** A feature sends only once the operator has set it live
  ([Features and modes](#features-and-modes)), so the operator sees its work before anyone else does.

Each guardrail, with what it refuses, and what the monitor may do on its own, only on the operator's word, or never:
[operations_guide.md, "5. Guardrails"](operations_guide.md#5-guardrails).

## Token economy and review spend

- **Free:** polling, the merge plan, gates, checks, sanity runs and every scripted lead (the toolkit); idle ticks (no
  Claude turn); the rule filter.
- **Paid:** verification, judgement and outward actions (Claude). Verification runs in the two review agents, so the
  session's context stays small; the session's prompt cache lives `claude.prompt_cache_ttl` (1h), the agents'
  `claude.subagent_prompt_cache_ttl` (5m).
- **Scaled to the PR.** A carried merge-only head costs nothing; a micro round is the verifier alone; a giant PR's cost
  grows with its parts, and its later rounds are light.
- **Review spend.** With the prices in `features.review_automation.spend`, `rk spend`, the admin dashboard's Review
  spend card and the weekly report price each review, posted, kept unposted or closed without a post
  (`rk review-close`: a review of its own, whose verifier runs are never booked on the PR's next review), by part
  (verifier, second verifier, code quality pass, orchestration), per UTC day and ISO week, from the Claude Code
  session transcripts and the post ledger.
  `state/spend_history.json` carries the runs and turns of a monitor this one took over from, whose transcripts this
  machine cannot read. `rk usage` counts tokens either way.

## Features and modes

The deploy settings' features decide every capability the code runs, one rule each (`rkmon_lib/features.py`,
CAPABILITIES); `rk features` prints each capability's state, what it needs and what is missing. Five features send,
and each runs in `prototype` or `live`: review automation, the merge queue, the alerts, the team dashboard and ticket
creation. A capability's rule reads its feature's switches, never the mode, so a feature in prototype does all its
work; each write it makes meets the one outward gate (`post.outward_gate`), which keeps it as an unsent item.
Standby is the whole monitor's: while it is on, no feature sends, whatever its mode. What goes out in each, and what
an unsent item keeps, is in the README's [Prototype, live and standby](../README.md#prototype-live-and-standby).

| Feature | In prototype | Live, it also |
|---|---|---|
| Review automation (`features.review_automation`) | Stages, drafts, checks and verifies reviews and decides overlap owners; each verified review is kept unposted | Posts each verified review and each overlap notice |
| Merge queue (`features.merge_queue`) | Computes the queue, the gates, the batches and the merge plan, shown on the dashboards, and proposes the order; the auto-merge guard only alerts; with `develop_guard`, develop is checked after every landing, and a red run opens an incident and freezes the queue | Arms auto-merge for cleared PRs (`auto_merge`), puts a change request on a PR that fails a gate by its own changes (`auto_hold`), posts queue and drift notices, turns others' risky auto-merge off with a comment, and, with `develop_guard`, files develop issues and links their fixes |
| Alerts (`features.alerts`) | Writes each message | Sends it to the team chat |
| Team dashboard (`features.dashboards.team`) | Makes the page after every tick and shows it on the admin dashboard; with `delivery` `serve`, serves it on 127.0.0.1 only | Pushes the page to the web host (`delivery` `publish`), or serves it at its own address (`serve`) |
| Ticket creation (`features.ticket_creation`) | Writes the follow-up tickets, and proposes the found-issue tickets for the operator's Approve | Files the follow-up tickets, and each found-issue ticket the operator approves |

Start in prototype. Take review automation live once its reviews match the team's review quality, and rehearse an
incident (`rk sanity <old sha> --rehearsal`) before the merge queue goes live with its develop guard. Arming and queue
notices go out under the monitor's account on other people's PRs: tell the team before turning `auto_merge` on or
the merge queue live.

## Known limits

Most limits follow from two choices above. The monitor is advisory: it has no maintain or admin role, so it reports
what it cannot prevent, such as a merge over an open 🔴 or an auto-merge that lands ahead of the plan. And its
container has no GPU, so arch coverage is mostly compile evidence unless remote GPU tests are on. Each limit, and
what it means for the operator:
[operations_guide.md, "7. Limits and known gaps"](operations_guide.md#7-limits-and-known-gaps).
