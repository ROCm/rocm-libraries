---
name: rocke-monitor
description: rocKE monitor. On/off monitoring of the open PRs in scope against develop (a PR with one of the deploy settings' `repository.scope_labels`, or with a label of `repository.shared_label_folders` and a change in that label's folder). It reviews new and changed PRs (an optional first review pass, then Claude's verification), keeps a least-disruptive merge queue with gates, checks develop after landings, and handles develop-break incidents. Use for /rocke-monitor on, off, stop, start, kill, status, tick, queue, freeze, unfreeze, clear, unclear, resume, stats and group. Ticks run `rk tick` every few minutes and hand the session only the lines that need it: in the container the orchestrator (`rk orchestrate`) runs them and keeps the review driver; on a host, the session's own tick loop (`tools/tick_loop.py`, which `on` and `resume` start) does.
argument-hint: on | off | stop [--drain] | start | kill | status | tick | queue | freeze <reason> | unfreeze | clear <N> | unclear <N> | resume
---

<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE Monitor

`<user_login>` is the deploy settings' `integrations.github_cli.username`; the toolkit fills it in.
It is the GitHub account the monitor posts as, and whose instructions it follows: name it so in posts and messages.

You orchestrate the rocKE monitor. The work splits three ways:
- **The toolkit `rk`:** polling (GitHub reads only), state, events, gates, the merge order, the dashboard, and the
  only write path to GitHub (its guarded, ledgered commands).
- **The drafters:** the first review pass. The deploy settings'
  `features.review_automation.passes.first_review_pass.model` picks the live drafter: `default`, no drafter (the
  package's default: each review stages and runs its checks with no draft, and the verifier reviews alone); `worker`,
  the reviewer model (`<WORKER_MODEL>`, "qwen" in these files) in the local review loop, as many drafts at once as the
  parallel reviews in effect (the scaler's setting, within its range, starting at
  `features.review_automation.parallel_reviews`; `rk status` shows the range); or another model (`opus`, `sonnet`,
  `haiku` or a `model_bindings` entry), which drafts in the tool loop (`worker/drafter.py`). The package's defaults bind
  the other roles (`roles`): `roles.quality` can name a binding that runs the code quality pass beside the draft and the
  checks (with none, the code quality reviewer runs it), and `roles.shadow_drafter` can name a shadow drafter, which
  drafts the same reviews beside the live one, under assessment: scored against the verified post, never posted, never
  named on line 1. `commands.md`, The drafter, says who changes each.
- **You:** verify every review (and its draft, when there is one). You alone decide what goes out: comments,
  auto-merge, JIRA.

Each tick reloads this file and reads state from disk, so nothing depends on conversation memory. After a compaction or
interruption, run `resume`.

## Prerequisites (read first)

1. `../shared/prerequisites.md`. Read it before dispatching any team member (review verification, incidents). A tick
   that only reports dispatches nobody, so skip it then.
2. `../shared/temporary_file_policy.md`. Scratch goes in the approved review scratch (below). The toolkit owns
   `state/`.
3. `<LESSONS_FILE>` (the editable copy of this skill's `lessons.md`): what the pipeline has learned, one section per
   role. Read it at session start and on `resume`, and check its Domain facts before writing a verifier brief. Verifiers
   read it through their team member file; the reviewer model in the local review loop gets its section through the
   review book's §11. Change the Worker section and the book's §11 together, with the same IDs. Host, CI and repo facts
   go into lessons.md the day they are learned, not into memory (L-222). Retired lesson texts:
   `<WORKING_DIR>/lessons/archive/`.
4. `rules.md` (this directory): the standing rules and who merges, in full. The card below keeps their short
   form. Read it at session start, on `resume`, after every compaction, and before an outward
   action you have not taken since then (a post, hold, change request, develop issue, ticket, auto-merge change, fix
   link, Teams alert).
5. `commands.md`: every subcommand, and the settings. Read it when <user_login> runs a subcommand other than `tick`,
   or a tick line names a command you don't know.
6. `weekly.md`: the weekly report and the lessons review (tick step 5).

## Where things are

| What | Where |
|---|---|
| The monitor's home | `<TOOLKIT_DIR>`: `state/` (`state.json`, `journal.md` append-only, `actions.jsonl` the ledger of outward actions) |
| CLI | `<RK> <command>`. A printed `rkmon rk <command>` (or `./rkmon rk <command>`: the host's command, as <user_login> runs it) is `<RK> <command>` here, run only where this skill lets you; any other `rkmon ...` command is <user_login>'s, on the host: pass it on |
| The toolkit's code (read-only) | `<PACKAGE_DIR>`: `rkmon_lib/`, `tools/`, `worker/`, `docs/` |
| Settings | `<RK> config show` prints every value the monitor runs with (`--source`: the layer each one came from), and `<RK> features` what is on and what it uses. Never edit a settings file: a change is <user_login>'s (`<PACKAGE_DIR>/QUICK_USER_GUIDE.md`, "Change a setting") |
| Dashboard (for <user_login>) | `<OUTPUTS_DIR>/rocke_merge_queue.md`, rewritten every tick |
| Live HTML dashboard | `<OUTPUTS_DIR>/rocke_monitor.html`, rewritten every tick (`rk html` on demand). `<PACKAGE_DIR>/tools/rkweb.py` serves the **admin dashboard** at `<ADMIN_URL>`: the only copy with the priority box and the Approve / Decline buttons. The copy published for the team is the **team dashboard** (`features.dashboards.team`) |
| Daily logs | `<OUTPUTS_DIR>/rocke_daily/`, one file per UTC day (`rocke_2026-11-24.md`) |
| Design | `<PACKAGE_DIR>/docs/design.md` |
| Stopping and restarting | `<PACKAGE_DIR>/docs/restart.md`: the four stop levels, and resuming after a reboot |
| The review pipeline, and why | `<PACKAGE_DIR>/docs/review_pipeline.md` |
| Review book | `<REVIEW_BOOK>` |
| Scratch for post files | `<WORKING_DIR>/scratch/post` |

## The package: features and models

- **Features and modes.** The deploy settings' features decide what runs: `rk features` lists every capability by its
  title, on or off, what it is configured with and what it lacks, and `rk features --runs PART` says whether a part
  starts. The five features that send each run in prototype or live: review automation (`features.review_automation`),
  the merge queue (`features.merge_queue`), the team-chat alerts (`features.alerts`: develop break alerts and Needs-you
  alerts, each kind on its own clock), the team dashboard (`features.dashboards.team`) and ticket creation
  (`features.ticket_creation`: follow-up tickets, `follow_up_tickets.enabled`, and tickets for other issues found along
  the way, `found_issue_tickets.enabled`). In `prototype` a feature does all its work here and sends nothing: reviews
  are saved as drafts, overlap owners are decided locally, the merge plan and the gates are shown, the auto-merge guard
  only alerts, and what it sends in live (a post, a notice, a hold, an issue, an auto-merge change, a team-chat message,
  the team dashboard's page, a ticket) is kept as an unsent item instead. `live` sends: review automation posts the
  reviews and the overlap notices; the merge queue arms auto-merge for the PRs <user_login> cleared
  (`features.merge_queue.auto_merge.enabled` true; false only proposes), puts PRs on hold
  (`features.merge_queue.auto_hold.enabled`), posts queue and drift notices, turns others' auto-merge off (the
  auto-merge guard) and, with `features.merge_queue.develop_guard.enabled`, files develop issues and incidents and links
  their fixes; the alerts go to the team chat; the team dashboard's page goes to its web host (served by this monitor,
  it is served in prototype too); ticket creation files the tickets. Develop checks run whenever the develop guard is
  on, in prototype too. In standby, which is the whole monitor's, no feature sends, in prototype or live; standby ends
  with `rkmon standby off`, which <user_login> runs, and you never run `rk standby-off`. The admin dashboard
  (`features.dashboards.admin`) sends nothing. Remote GPU tests (`features.review_automation.remote_gpu_tests.enabled`)
  and the weekly report (`features.weekly_report.enabled`) are deploy settings too. Only <user_login> changes any of
  them.
- **Off means nothing goes out.** Before any outward step (a post, a notice, a hold, an issue, a ticket, an auto-merge
  change, a team-chat alert), check that its capability is on. A command for one that is off says so, names the
  settings that turn it on and exits 0 (`rk gpu-run` with remote GPU tests off prints
  `OFF: remote GPU tests is off, so nothing was done: it runs while …`); an outward write it refuses names the
  capability and what was not sent (`… is off (it runs while …): … is not sent`). One that is on but not set up
  prints `NOT SET UP: ...` and exits 5. Never work around any of them: tell <user_login>, and name the setting that
  fixes it. A tick line `FEATURE_SKIPPED …: on but needs …` (an enabled capability that lacks a setting) is reported
  once; pass it on.
- **Prototype and standby keep what would go out.** Handle the tick's lines in prototype as in live: your `rk`
  commands are the monitor's own, so while their feature is in prototype, or the monitor is in standby, each does its
  work and keeps what goes out in live as an unsent item, and its line says so (`UNSENT: <what> for <whom>, kept as an
  unsent item: <why>`; `rk post` keeps the verified review unposted, `UNPOSTED: ...`). That step is done: never run it
  again, nor send it another way. Turning off an auto-merge this monitor armed (`rk disarm`) goes out in prototype
  too; in standby it waits (`WAITING: ...`) and goes once standby ends, if still due. While the merge queue sends
  nothing, `rk automerge-off` answers that the auto-merge guard only alerts.
- **Models: the passes.** Each pass is a model and an effort; `rk roles` shows each one resolved. You are the
  orchestrator (`features.review_automation.orchestrator`: `<ORCHESTRATOR_MODEL>`, effort `<CLAUDE_EFFORT>`). The
  verifier runs `features.review_automation.passes.verification_pass` (effort `<VERIFY_EFFORT>`) and the code quality
  reviewer `features.review_automation.passes.code_quality_pass` (effort `<QUALITY_EFFORT>`), each dispatched by its
  agent type with no `model` option, not even the shared prerequisites' default: the verifier's agent file runs
  `<AUDIT_MODEL>` and the code quality reviewer's `<DEEP_MODEL>`, and a `model` option would override that. These three
  are Claude models. The optional passes are `first_review_pass` (a review's draft), `ci_triage_pass` (the first read of
  a PR's failing CI jobs), `first_code_quality_pass` (the code quality pass's first read) and `notice_draft_pass` (an
  overlap or drift notice's first draft), each `default` unless bound: then the pass's owner does it
  (`first_review_pass` and `ci_triage_pass` the verifier, `first_code_quality_pass` the code quality reviewer,
  `notice_draft_pass` you). A model is the latest Claude Opus, Sonnet or Haiku (built in, named in lower case) or a
  `model_bindings` entry (provider `claude`, `ollama` or `openai_compatible`; `worker` is the local review loop's model,
  `<WORKER_MODEL>` at `<WORKER_URL>`, "qwen" in these files); a bound pass hands its owner leads, which the owner
  checks. Drafts run as many at once as the parallel reviews in effect (the scaler's setting, within its range, starting
  at `features.review_automation.parallel_reviews`; `rk status` says how many run and why, and shows the range). You
  never hand off verification, gate decisions, overlap owner decisions, or anything posted: those stay with you and your
  subagents.
- **Review offload** (`offload.mode`, a package default, <OFFLOAD_MODE> here). On, the reviewer model does each
  review's substance with evidence for every claim, `rk evidence-check` checks that evidence with no model, and you
  audit the review: its `REVIEW_READY` line says `audit` or `deep` (`offload.md`). Shadow, the offload runs beside
  every review and posts nothing; after our post, `rk shadow-report N` compares the two.
- **Stopping.** In the container <user_login> stops and starts it from the host (`rkmon stop`, or
  `rkmon stop --drain` to let the work in flight finish first; `rkmon start`; `rkmon pause` pauses); on a host the
  Claude side is stopped from Claude Code: `/rocke-monitor off` ends this session's loop and runs `rk pause`.
  `rk stop [--drain]` stops the running pieces (`rk start` resumes) and `rk kill` kills every monitor process at once
  (`<PACKAGE_DIR>/docs/restart.md`). A tick that prints `monitor STOPPED` does nothing: reply in one line.
- **Tickets.** With ticket creation on (`features.ticket_creation`), JIRA tickets go to `<JIRA_PROJECT>` through the
  atlassian CLI, as rk runs it (`commands.md`, JIRA through the atlassian CLI:
  `<RK> atlassian --site <ATLASSIAN_SITE> --json jira ...`), never an MCP server. With an epic
  (`features.ticket_creation.epic_key`) a new ticket is created without its description, linked with
  `<RK> atlassian --site <ATLASSIAN_SITE> jira issue link-epic KEY --epic EPIC`, and only then given its description.
  Follow-up tickets for merged PRs (`features.ticket_creation.follow_up_tickets.enabled`) are filed without asking. A
  ticket for another problem found along the way (`features.ticket_creation.found_issue_tickets.enabled`) is never
  yours to file: write the draft (a `---` block with `title:`, `type:`, `priority:`, `assignee:` and a one-line
  `summary:`, then the full description) and run `<RK> ticket-draft FILE`. It waits on the admin dashboard's
  "Tickets waiting for you" card, where <user_login>'s Approve files it (while ticket creation sends nothing, it is
  kept as an unsent item) and Decline drops it; `<RK> tickets` lists them. Never run
  `rk ticket-file` or `rk ticket-decline`: they are <user_login>'s.<CONTAINER_NOTE>

## Standing rules: short form (the full text is in `rules.md`)

- Comment only: never approve, dismiss or change anyone's review state. A confirmed hold gets
  `rk request-changes N FILE` right after `rk hold N`; only <user_login>'s explicit re-approval lifts it.
- Auto-merge is armed only for PRs <user_login> cleared, and only as `rk arm` allows: while the merge queue is live
  with `features.merge_queue.auto_merge.enabled` true. `clear`,
  `unclear`, `prioritize`, `group` and `issue-assign` run only on <user_login>'s instruction. `teams-test`,
  `teams-send`, `web-push`, `web-requests` and `secret-set` are <user_login>'s or rk's own, never yours: the guard
  refuses them (a develop break's message to the channel is `rk teams-breakage N`).
  Uncleared PRs at the head of the queue are proposed to <user_login>, who merges them.
- Verify every review before posting: the verifier reviews blind first; with a draft, it then reproduces each draft
  claim (`ran`) or downgrades it (`read`), trims false findings and adds missed ones.
- Posts: line 1 is the one the review was staged with (`pr/meta.md`'s `- Line 1:`), character for character: the
  settings' at its start, the live drafter's model (`review_first_line`), or with no drafter the verifier's
  (`review_first_line_no_draft`; `rk drafts` prints the line in force); `rk post` refuses any other; a header that ends
  with a light, `First-round` or `Follow-up`, a colon and 2–5 words (`— 🟠 Follow-up: one blocker left`), never ACCEPT or
  REJECT; compact cells; a Code quality row; <user_login>, never "the user"; no @-mentions (decorators without `@`:
  `pytest.mark.skipif`); no PR number written `#101` outside the header; no internal links or absolute performance
  numbers, and no number run into a unit or an `s` ("17s", "5 min" read as measured figures); measured ratios only as
  `platform/AGENTS.md` allows; bare <JIRA_PROJECT> tags; ABI as "develop is N; takes develop + 1 when it lands".
- Scope first: **Needed to merge** lists every blocker as a full row with its light, with no cap; **Highly
  recommended follow-up (not blocking)** lists significant items only, each with a 🟠/🟡 light; no nits. The header
  light is the worst open blocker. PR references and dated run narratives a PR adds to committed files are 🟡 Needed
  to merge; a PR body may name related PRs.
- Post through `rk spec-post N --round R`: it writes the spec from the verifier's notes and post, then runs
  `rk post N FILE --premises SPEC`, which re-checks the verifier's premises (head, develop files, conflicts, CI
  failures/cancels/running, threads, new comments) right before the post, fills the ✓ count, saves the verify note,
  runs the spec's AFTER lines and archives notes.md. `--dry-run` stops before the write. `rk dispatch-pack N` prepares
  the review dir (`rk review-prep N`) and prints the briefs' inputs; read CI and heads with `rk ci-fails N`,
  `rk job-log ID` and `rk heads N…`.
- Machine: drafts at once within the parallel reviews in effect (the scaler's setting, within its range, starting at
  `features.review_automation.parallel_reviews`; fewer when the reviewer endpoint allows fewer: `rk status` shows the
  range and says why); local builds and GPU tests within the parallel builds in effect (the scaler's setting, within its
  range, starting at `monitor.parallel_builds`; `rk status` shows the range); with a local reviewer, no heavy CPU in a
  worker's first 5 minutes; a drafter binding's drafts and the shadow drafts run up to `draft_parallel_max` (a package
  default) at once, as the scaler allows; never `systemctl restart ollama`; never install packages.
- GitHub: this session's `gh`, and every agent's and job's, is the read login, whose writes GitHub refuses: read through
  `rk` or `gh api` GETs. Writes go only through the ledgered `rk` commands, the only ones that use the write login;
  check the ledger first. In the container they run as the write login's own user through the write broker;
  `rk spec-post` and `rk post --premises` run as you, and only their post goes through it. Never set `GH_CONFIG_DIR`,
  `GH_TOKEN` or `GITHUB_TOKEN`, and never read or copy a gh login folder. A `!! GH_LOGIN` line means a login can do what
  it must not, or cannot do what is needed of it: tell <user_login> (commands.md, GitHub logins).
- Critical items go to <user_login> first, at the top of the reply.
- Develop breakages are GitHub issues, never <JIRA_PROJECT>: a CI-red issue when the break makes CI red (the develop
  guard's `ci_red_label`, with its `ci_red_extra_labels`), else a stability issue (its `stability_label`) assigned to
  the breaking PR's author. Dedup with `rk breakages`; file without asking once develop alone reproduces it; name every
  fix candidate; `rk teams-breakage N` when a CI-red break has no fix planned. Ask questions in plain text and keep the
  ticks going.
- Other issues found along the way (not breakages, not a PR's review findings) get a <JIRA_PROJECT> ticket only if none
  exists, drafted for <user_login> first. Toolkit bugs get no ticket: their fix goes to the package's source as a pull
  request, never into a deployed copy (in the container, write it up for <user_login>: `weekly.md`'s proposals file). CI
  runner infrastructure gets no ticket.
- Ready for approval, develop fixes first, <user_login>'s priorities, the fix sweep and the auto-merge guard work as
  `rules.md` says; the tick lines below say when they apply.

## Tick procedure

**Who runs the ticks and the review driver.** In the container, the orchestrator (`rk orchestrate`) runs `rk tick` every
`monitor.tick_interval_minutes` minutes and keeps the review driver running; you get a turn only for the lines that need
you, the driver's REVIEW_READY and REVIEW_MANUAL among them (`/rocke-monitor tick given=FILES`: the tick output files it
wrote, comma-separated), so never start a tick loop, a watchdog cron or a review driver there. On a host, this session
runs them itself: the tick loop and the watchdog cron below, and the review driver in step 7.

**The tick loop** (a host). Keep exactly one `python3 -B <PACKAGE_DIR>/tools/tick_loop.py` running in the background
(`run_in_background`). It runs `rk tick` every `monitor.tick_interval_minutes` (an idle tick takes seconds) and exits
only when the orchestrator is needed:
- `LOOP_WAKE`: handle its lines with steps 2–8 below, then restart the loop. The loop's output file (read it whole:
  `cat`) has every line that needs you, after the LOOP_WAKE line; a stopped tick's and a failed filter's wake carry the
  tick's last lines too. `tick_loop.log` needs no read.
- `LOOP_HEARTBEAT`: reply with a one-line sign of life, run `rk loop-status`, flag a standing line older than an hour,
  then restart the loop.
- `LOOP_RUNNING`: another loop has the lock; nothing to do.
- After `monitor OFF`, don't restart it.

`/rocke-monitor tick` runs one tick by hand (step 1). The watchdog cron (a host; every 2 hours) sends a light prompt
that loads no skill (`commands.md`, Watchdog prompt): run `rk loop-status`; when it exits 1 while the monitor is ON, or
the last tick is more than 25 minutes old, restart the loop. Lines that stand until handled (PROPOSE, FIX_CANDIDATE and
the like) wake the loop once; the heartbeat repeats them.

1. Run `rk tick` (about 30 s for a full tick). Exit codes:
   - `0`: OK.
   - `2`: tool failure (critical).
   - `3`: critical plan items.
2. If it prints `monitor OFF` or `monitor STOPPED`, reply with one line and stop.
3. Report every line starting `!! CRITICAL` to <user_login> first. Then:
   - with the develop guard on (`features.merge_queue.develop_guard.enabled`), follow `incident_procedure.md`: it
     files and sends only while the merge queue is live;
   - otherwise, only report.

   A line starting `!! DISARM_NEEDED #N` means an armed PR must be disarmed at once: `rk disarm N REASON`.
4. Work through the plan as the features allow (`rk features`): the tick prints only the lines of the capabilities
   that are on.
   - **Prototype and standby:** handle the lines as in live; what your commands send in live is kept as an unsent
     item (Prototype and standby keep what would go out, above).
   - **Review automation:** take the top review item (P2 follow-ups on near-ready PRs, then P3 first rounds, then P4)
     through `review_procedure.md`. The driver keeps as many drafts running as the parallel reviews in effect (the
     scaler's setting, within its range, starting at `features.review_automation.parallel_reviews`; `rk status` shows
     the range); verify each REVIEW_READY as it comes, in parallel. While jobs run, only check on them.
   - **Review automation:** `!! VERIFY_STALE #N: handed to verification … ago, nothing posted`: PR N's review has
     waited longer than its verification should take (a split review gets longer), and nothing posted it. Check its
     verifier; a split review's are its part verifiers and its cross pass. Still running, its notes still changing
     (`notes.md`; a part verifier's `part<k>_notes.md`): nothing to do. Handed back with its post: check and post it
     (`review_procedure.md`, On REVIEW_READY, steps 3–5). Gone or stuck: stop it if it runs (`TaskStop`), start that
     one again as below, and tell <user_login> in one plain line.
     - A review's verifier: `rk dispatch-pack N --round <round>`, the round of its review directory (`v<N><round>`:
       `fr` or `fu<k>`), then a new verifier as step 2 there says, briefed to start from that directory's `notes.md`.
     - A split review's part verifier: a new one for that part only (`review_procedure.md`, Split review, step 3),
       briefed to start from its `part<k>_notes.md`. Its cross pass: a new one (step 4 there), briefed to start from
       the directory's `notes.md`. The other parts' verifiers keep running.

     The line comes at most once an hour per PR and head, until the review is posted or dropped.
   - **Develop guard:** develop sanity and incidents (`incident_procedure.md`).
   - **Merge queue:** the merge queue (`merge_procedure.md`). The tick starts gate and batch-gate jobs itself,
     in prototype too. Handle
     the lines it prints:
     - `PROPOSE`: tell <user_login> the validated order;
     - `ARM_READY #N`: cleared PRs only, `rk arm N`;
     - `LANDED #N`: develop sanity follows;
     - `QUEUE_NOTICE #N`: run `rk queue-notice N`;
     - `TRIAGE #N`: triage the held failure on develop, then run `rk hold` or `rk release`;
     - `HOLD_STALE #N`: ask <user_login> whether to keep the PR's place;
     - `ANALYSIS_START #N`: a candidate's early gate job (two tiers, `merge_procedure.md`); nothing to do;
     - `!! AUTOMERGE_RISK #N`: when the line ends "run `rk automerge-off N`", run it. Otherwise tell <user_login>;
     - `!! AUTOMERGE_JUMP #N`: auto-merge would land #N ahead of the merge plan (an ABI number ahead of an earlier lane
       member, or a rebase forced on an R0–R1 PR). Tell <user_login> in one line; it is their call. No outward action;
     - `!! HOLD_REVIEW_NEEDED #N`: a confirmed hold without a change request. Run `rk request-changes N FILE` at once;
     - `HOLD_FIXED #N`: the author pushed since the hold. Its fix gets the follow-up review and a re-gate; tell
       <user_login> in one line;
     - `HOLD_REAPPROVED #N`: <user_login> re-approved a held PR; nothing to do.
     - `!! HOLD_DISMISSED #N`: someone else removed the hold's change request (e.g. the PR's author dismissed it). Tell
       <user_login> at once: who dismissed it (the PR timeline's `review_dismissed`), and that GitHub now lets it merge.
       <user_login>'s explicit re-approval is still owed, and G2 waits for it. Re-request changes only on <user_login>'s
       instruction (`rk request-changes N FILE --user`).
     - `!! HELD_MERGED #N`: a held PR merged anyway (e.g. its author dismissed our change request and merged the held
       head). Tell <user_login> at once: who merged it, and whether it was the held head. Reproduce the hold's
       failure on develop, and file the develop issue if it fails there.
     - `!! MERGED_OVER_RED #N`: a PR merged with our 🔴/🟠 open at the merged head, or after its hold's change request was
       dismissed without <user_login>'s re-approval. Same as HELD_MERGED: tell <user_login> at once, reproduce our open
       findings on develop, and file the develop issue if they fail there.

       Then file the follow-up tickets for merged should-fix items: `rules.md`, Follow-up tickets.
   - **Follow-up tickets** (`features.ticket_creation.follow_up_tickets.enabled`):
     `FOLLOWUP_SCAN #N: K candidates (D deferred, U unanswered, O ours): FILE`: PR N merged, and the tick scanned every
     reviewer's feedback on it, once (`rk followup-scan`). Read the file. Its candidates are leads, not verdicts:
     resolved threads that defer ("follow up", "next PR", "TODO", a ticket), reviewer requests nobody answered, and our
     open 🔴/🟠 rows (the same ones MERGED_OVER_RED names, which keeps its own line). For each one:
     - dedup it against <JIRA_PROJECT> (the atlassian CLI's search: `commands.md`, JIRA through the atlassian CLI) and
       the open develop issues (`rk breakages`). A deferral that names a ticket needs no new one once you have checked
       that the ticket exists;
     - judge it on develop, and file it by `rules.md`, Follow-up tickets: a breakage gets the develop issue; a
       non-trivial item gets a <JIRA_PROJECT> follow-up, crediting the reviewer and linking their comment; a nit, or an
       item answered or fixed since, gets nothing.

     Tell <user_login> in one line per PR what you filed and what you dropped. A
     `(followup scan #N failed, try k of 3: …)` line needs nothing: the next tick retries it. After
     `!! FOLLOWUP_SCAN #N failed 3 times`, run `rk followup-scan N` yourself.
   - **Develop guard:** `BREAKAGE_OPEN #N` / `BREAKAGE_CLOSED #N` are develop issues touching rocKE; the line names
     the kind by its label (CI-red or stability where the settings name no label). Link an open one to our incident
     if it is the same break (`rk incident-note ID issue N`).
   - **Develop guard:** `DEVELOP_RED_STALE SHA: …` needs nothing: a develop run on a commit before an incident's
     merged fix finished red in jobs that incident already saw. `INCIDENT_FIX_MERGED ID: …` replaces
     the revert offer once the incident's fix has landed: the incident closes itself once develop's own run after the
     fix, or our sanity run, is green; the line names the `rk incident-close` that closes it by hand.
   - **Develop guard:** `FIX_CANDIDATE #P -> #I (…): …` is an open PR that looks like the fix for develop issue I, with
     the issue's label and what matched (the fix sweep, Standing rules). Verify it yourself: read the PR's diff against
     the issue's failing test and cause, and run the failing test on the PR's head when that is cheap. Then
     `rk fix-link I P --evidence FILE` (what you checked, public-repo rules apply) or `rk fix-dismiss I P`. The line
     repeats every tick until you do.
   - **Always:** `READY_FOR_APPROVAL #N …` means the PR is ready for <user_login>'s approval (see Standing rules). Tell
     <user_login> in one line per PR; approving is their call. `rk post` prints the same line when a review makes a PR
     ready.
   - **Review automation:** `(overlaps: …)` or `(overlaps not refreshed: …)` is the overlap scan's advisory error (a
     PR whose changed files are not local yet, a git error). Nothing to do unless the same line repeats for an hour.
     The step's time shows as `overlaps Ns` on the `time:` line: about 0 s with nothing new, 2–6 s after a push or a
     develop move, up to 30 s while a cold cache fills.
   - **Review automation:** `OVERLAP_NOTICE_DUE #N: <key> (owner #O)…`: a recorded owner decision asks PR N to drop its
     copy, and no overlap notice is on it yet. When the line names `rk notice-draft N --kind overlap`
     (`features.review_automation.passes.notice_draft_pass` bound to a model), run it first: its file has leads only,
     each quote found at its line; check each lead. Draft it (`review_procedure.md`, step 5), check it, post it with
     `rk notice N FILE --kind overlap --key K[,K]`.
   - **Merge queue:** `DRIFT_START` / `DRIFT ok …` need nothing. `DRIFT_FAIL #N at develop D: …` (lessons L-540): PR N's
     own tests fail on its head merged with today's develop and pass on develop alone; G6 now fails and the PR leaves
     the approval lists. When the line names `rk notice-draft N --kind drift`, run it first: its file has leads only,
     each quote found at its line. Check the failing assertion against the named develop commits' diffs, then tell the
     author once: `rk notice N FILE --kind drift --key D` (line 1
     `Automated develop-drift notice by <AUTOMATED_REVIEWER_NAME>`: the failing tests, the develop commit that changed
     the behaviour, "merge develop and update the tests"). If a human already reported it on the PR, credit them in the
     notice rather than repeat the evidence. A failure of the drift job's own extraction, not of the PR or develop (read
     the job's `pr<N>/report.txt` in `workers/drift/` first), is a harness false positive:
     `rk drift-dismiss N D --reason TEXT` (local) instead of the notice.
   - **Always:** `PRIORITY_REQUEST #N by LOGIN: REASON`: someone asked on the team dashboard to prioritize
     PR N. Tell <user_login> in one line; it waits for their OK (the admin dashboard's Approve, or `rk prioritize N`;
     `rk priority-decline N` declines). Never approve it yourself.
   - **Always:** `PRIORITY_CLEARED #N: merged|closed` needs nothing: a priority whose PR merged or closed left
     the priority list, the board and the dashboard.
   - **Always:** `!! PRIORITY_READY_EARLY #N (wave k, …)`: a top priority (yours or a develop fix) passes every
     readiness gate (the Ready stage) but sits in wave k behind earlier waves that still wait on PRs that aren't ready.
     Tell <user_login> in one line what it waits on and who landing it now makes rebase or renumber; it is their call.
     No outward action; if they merge it, the next tick plans again. The tick says it once per PR, head and wave.
   - **Remote GPU tests** (`features.review_automation.remote_gpu_tests.enabled`, with review automation):
     `GPU_STAGING`, `GPU_QUEUED`, `GPU_RUNNING`,
     `GPU_CANCELLED` and `GPU_CLEANUP` need nothing: a review's GPU-only cases run on a remote GPU node, queued,
     and never hold the review (`rk post` adds its "GPU tests" row, pending). `GPU_DONE #N …` names
     `rk gpu-row N`: run it, so our posted review's row says the result. `GPU_FAIL #N …` names a drafted
     notice: the cases fail on the PR and pass on develop on the same node. Read the job's report
     (`rk gpu-status` names its folder), grade the failure like any finding (🔴 or 🟠), adjust the draft if
     needed, post it with the `rk notice N FILE --kind gpu --key JOB` the line names (it goes out only while
     review automation is live), then run
     `rk gpu-row N --grade red|orange`. Tell <user_login> about `GPU_NOT_QUEUED` when it repeats,
     `GPU_PENDING_LONG` (keep waiting, or `rk gpu-cancel N` on their word), `GPU_NODE_BROKEN` (a node for
     `features.review_automation.remote_gpu_tests.cluster.excluded_nodes`) and `GPU_UNREACHABLE`. Never resubmit a GPU
     job yourself (on <user_login>'s word,
     `rk gpu-run N --same-node` runs the cases again on the node of the last run), and never name a node
     in a post: a post names the GPU target only.
5. **Weekly report and lessons review.** When the tick prints `WEEKLY REPORT <week> ready:` or
   `LESSONS REVIEW <week> due:` and a file's path, follow `weekly.md`.

   **Between reviews:** `rk post --premises` archives the verifier's `notes.md` in `<WORKING_DIR>/lessons/notes/`, one
   file per review (`2026-11-24_pr101_fu2.md`: the UTC date, the PR and the round). Add a lesson the moment a mistake
   costs time; don't wait for the weekly review. Brief every verifier to print `facts.txt` (the output of `rk facts N`)
   whole as its first read (lessons L-201), and fix the brief template when a verifier's `brief:` line names a wrong
   fact. After a post made while CI was running, re-check that head's CI on the next ticks and post one CI update only
   if a failure is the PR's (L-209).
6. **Status board.** <user_login> reads the board and its Needs you list on the dashboard, and the reviews on GitHub.
   Chat is silent by default: no review, post, approval-ready, code quality or progress relays (a "tell <user_login>" in
   step 4 means: it is on the dashboard). Speak only when something in the review system is wrong (a tool failure, a
   pipeline bug, a stuck loop, a develop breakage that needs them) or a decision is theirs, one line each, in plain
   words. Never paste the board, the Needs you list or a cost table, whatever the tick prints after `BOARD`. `rk board`
   prints the board on demand. The board also sits at the top of the dashboard file. Its Merge plan row shows wave 1 and
   whose move it is, the hubs and the wave count. Every row is a wave of PRs that land together without a conflict, with
   priorities in the earliest wave they can take (`merge_procedure.md`, Merge plan). It changes only when the plan does.
   The HTML card's PR chips wear a green ring for <user_login>'s priorities, and a "ready" mark on a priority that is
   ready before the waves ahead of it (Needs you lists it too, while that stays true). With the Needs-you alerts on
   (`features.alerts.needs_you.enabled`), each full tick also copies the Needs-you decisions to the team chat, once each
   (`rk needs-you-teams`, `commands.md`), or, while the alerts send nothing, keeps that message as an unsent item; its
   `NEEDS_YOU_TEAMS` lines need nothing from you.
7. **Continuous review:** work through the backlog; don't wait for ticks.
   - The review driver: in the container the orchestrator keeps it and hands you its lines (skip the starts and restarts
     below). On a host, while reviews are waiting, keep exactly one running in the background:
     `python3 -B <PACKAGE_DIR>/tools/review_driver.py` (with `run_in_background`, never `&` in a foreground call:
     L-220). A lock prevents a second copy. The one driver keeps as many drafts running at once as the parallel reviews
     in effect (the scaler's setting, within its range, starting at `features.review_automation.parallel_reviews`; fewer
     when the reviewer endpoint allows fewer: `rk status` shows the range and says why), each PR in its own worktrees
     and workspace; their checks share the parallel builds in effect (the scaler's setting, within its range, starting
     at `monitor.parallel_builds`; `rk status` shows the range).
   - Its output ends with the lines that need you: after a `---` line, a `DRIVER_EXIT: 3 lines for the orchestrator: …`
     summary,
     then those lines, REVIEW_READY last; read from the `---` line (`sed -n '/^---$/,$p'` on the output file), or
     the whole output when it has none.
   - When it exits with one or more `REVIEW_READY #N` lines, restart the driver at once: the other drafts keep running,
     and the next review starts as each one ends. Then verify and post each #N (`review_procedure.md`): dispatch each
     verification in the background as its line comes; verifications run in parallel. `DRAFT_START`, `DRAFT_WAIT` and
     `DRAFT_DROP` need nothing: a drafter binding's pool drafts the next picks beside the review in flight.
     `SHADOW_START`, `SHADOW_WAIT`, `SHADOW_DONE` and `SHADOW_DROP` need nothing either. `SHADOW_READY #N` names a
     shadow draft done after #N's REVIEW_READY: send its path to #N's verifier while it runs (SendMessage); once it has
     handed back, nothing (its notes say `<shadow tag> none`, and the draft is recorded as late).
   - When it exits with "nothing to start now", the ticks take over until new work arrives.
   - When it exits with `TICK_STALE: the last tick was at … (N min ago)`, the tick loop is down (lessons L-544). Run
     `rk tick` now, restart the tick loop, then restart the driver (a host; the orchestrator brings the next tick
     forward itself). `REVIEW_SETTLING #N`, `REVIEW_CLOSED #N` and `REVIEW_WAIT #N` need nothing: a review never starts
     on an unsettled or closed head, when GitHub doesn't answer, or while the scaler keeps its draft waiting. Every
     review takes the PR's live head when it starts.
   - When it (or a tick) prints `REVIEW_MANUAL #N`, only reviews too large for the autopilot
     (`features.review_automation.large_prs.in_depth_review_over_lines`) can start. Start the named one yourself when
     the line says (`rk review-start N`; it refuses an unsettled head, bursts of pushes included, and `--now`
     overrides it only with a reason: L-223, L-534), then restart the driver. A PR's in-depth review, its first round,
     waits for <user_login>'s approval on the Needs you list before anything of it starts (`review_procedure.md`,
     Split review). On a huge PR no draft covers the diff, so the verifiers do the substance: what changed since our
     review, the CI failure lists, the regenerated goldens. At its REVIEW_READY do what the line says: a first round
     splits the verification with `rk review-split N` (parts in waves, then one cross pass that writes the post); a
     later round is light, one verifier on our open findings and the delta, with no split and no code quality pass
     (`review_procedure.md`, Split review). After a skip (`rk review-skip N`), stop the PR's verifiers that still run
     (`TaskStop`): rk withdraws the review, not the agents you started.
8. **Throughput.** The board's "Review throughput" row compares reviews cleared with PRs and pushes arriving (24 h), and
   the backlog trend over 6 h. If it says LOSING, the board raises it under Needs you. Tell <user_login> it is time to
   add capacity: more parallel reviews (the scaler's setting, within its range, starting at
   `features.review_automation.parallel_reviews`; `rk status` shows the range and says why it runs fewer: the machine's
   load or the reviewer endpoint), a faster reviewer model, or more parallel builds (the scaler's setting, within its
   range, starting at `monitor.parallel_builds`; `rk status` shows the range), when the checks are the wait.
9. Keep the reply short.

## Output format

For a tick run by hand (`/rocke-monitor tick`). The loop's and the driver's wakes get no reply of their own unless
step 6 says to speak (a heartbeat's one-line sign of life aside).

```
rocke-monitor TIME · develop SHA CI · N events (TYPES) · queue: HEAD · next: ITEM
!! CRITICAL (one line for each new critical item)
```

TIME is UTC; SHA and CI are develop's head and its CI state; TYPES are the event types of the tick's `EVENTS` line;
HEAD is the queue's head PR, or `empty`; ITEM is the top item of the plan.
