#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk: the rocKE monitor's command. It reads GitHub under the read login; the guarded write commands (post,
notices, issues, arm and the rest: rkmon_lib/post.py) alone write, under the write login (rkmon_lib/gh.py). In the
container those run as the write user, through the broker (docker/rk_write.py).

  init [--if-new] [--on] [--force]   create state/ and seed it from GitHub without emitting events; --if-new: only
                       when there is no state yet, else it says so and exits 0 (the container's `docker/rkmon start`
                       runs it each time); --on: the state it seeds starts ON, and kept state is never switched (a
                       paused monitor stays paused, and it says how to resume); --force: seed a new state in place of
                       one that exists, which it replaces (without it, a state that exists is kept)
  tick [--force] [--full] [--dry-run]   one monitor iteration (no-op while OFF, unless --force, or STOPPED). Idle when
                       nothing changed since the last full tick and no timer ran out (rkmon_lib/idle.py): one line, the
                       dashboards' "checked" stamps. Otherwise full: poll, events, queue, plan, journal, dashboards.
                       --full runs a full tick anyway; --dry-run says which it would be, and why, and writes nothing.
                       Exit 2 when a tool failed (gh or git: !! CRITICAL TOOL failure), 3 when the plan has a critical
                       item or the develop guard opened an incident, named a culprit or offered a revert, else 0
  tick-now             ask the tick loop that runs here for a tick at once: the orchestrator (rk orchestrate),
                       tools/tick_loop.py or the shell tick loop (rk tick-wait) takes the request within seconds and
                       starts a tick (after the one in progress, if any) through its own path, so its lines go where
                       every tick's go. ASKED, exit 0: one of them runs here; WAITING, exit 3: none does, and the next
                       of them to start takes it at once; REFUSED, exit 4: the monitor is stopped (rk start resumes
                       it); ERROR, exit 1: the request could not be written
  tick-wait --loop PID SECONDS   the wait of the shell tick loop (docker/entrypoint.sh's tick loop that runs without
                       the orchestrator, whose pid is PID) between its ticks: it marks the loop for rk tick-now, and
                       takes rk tick-now's request for it while PID runs and no orchestrator session does (that one
                       takes it itself). TAKEN, exit 0: the loop starts a tick at once; exit 1 once SECONDS went by
                       with none taken
  tick-filter [--rc N] [--seen FILE] [--driver]   the lines of a tick's output (stdin) that need the orchestrator
                       (rkmon_lib/tickfilter.py): exit 0 when there are any, 1 when there are none. --rc: the tick's
                       exit code. --seen: a standing line (one the tick repeats while its condition is true) that FILE
                       has already stays quiet; a full tick's standing lines replace FILE. --driver: the input is the
                       review driver's (its last rk review-advance): its lines that need the orchestrator, then a
                       summary line, the REVIEW_READY lines last
  rules-tick [--clear]   a tick run by fixed rules, for a deployment with no Claude session (the merge queue on and
                       review automation off; rkmon_lib/rules_tick.py): rk tick, then, for each of its lines a fixed
                       rule covers (arm a cleared batch, disarm a PR the queue let go of, turn off a risky auto-merge,
                       start a bisect, post or update a queue notice), that rule's rk command, whose own guards still
                       decide. Every other line, and a rule whose command refused, waits for a person on the admin
                       dashboard's Needs-you list and in rk status. Exit: the tick's. --clear: that list emptied
  loop-config          tools/tick_loop.py's settings, in seconds and bytes (the minutes between ticks,
                       monitor.tick_interval_minutes, and the package's own loop_heartbeat_min, tick_timeout_min and
                       loop_log_max_mb: docs/internal_settings.md), and its files, rk tick-now's request (TICK_NOW) and
                       the shell tick loop's mark (SHELL_LOOP_PID, which rk tick-wait writes) among them, as shell
                       assignments
  worker-tools [--install]   the worker tools in paths.worker_tools that are missing or not the running code's (exit 1
                       when any); --install copies the running code's worker/ there (writable; a locally edited file is
                       backed up once)
  pr-slot take|keep|free|status [--pid PID] [--pr N] [--head SHA] [--build DIR] [--record FILE]   the PR runners'
                       slots (rkmon_lib/prcode.py, rkmon_lib/slots.py). In the container PR code (the checks, gates,
                       sanity runs, bisects, drift checks and probes) runs as a PR runner, in the runner's slot: it
                       reads no secret and writes only the slot's folders (out, build, tmp, home). take --pid PID: a
                       slot for a job script's process PID, free again once that process is gone, printed as shell
                       assignments (SLOT, SD, SO, SB, SI: the runner, the slot's folder, its out and build, what the
                       monitor hands the job); exit 1 when none was taken. keep --pid PID --pr N --head SHA --build DIR
                       --record FILE: that slot kept for the review of SHA (no job runs in it; the other runners read
                       its build) until the review ends, or a job finds no free slot and it is the one kept longest.
                       free --pid PID: that slot freed. status: each slot. On a host there are no slots: PR code runs
                       as the session user
  pr-run [--slot NAME [--end]] [--env FILE] [-C DIR] [--out DIR] -- CMD [ARGS...]   CMD as PR code in a slot of its own
                       (on a host as the session user), with the environment PR code gets (worker/pr_env.py): how the
                       session runs PR code by hand (a test file again, a probe). The slot's folders are made fresh,
                       every process is stopped when CMD ends, and the slot is freed. --slot NAME (pr<N>-<word>): named
                       work, which keeps its slot and files for its next commands until --end (with no CMD: the end
                       alone), the end of PR N's review in flight, or the end of the process that started the work.
                       -C: where CMD runs (absolute, or a folder of the slot's: out, build, tmp, home; default this
                       folder). --env FILE: its environment, NUL-separated NAME=VALUE (/dev/stdin: read from stdin).
                       --out DIR: the slot's out copied into DIR after, as data, in new files only (a name DIR has
                       already is refused). The checks' re-run commands start with it. Exit: CMD's; 1 when CMD ended
                       well but a result was refused, 3 when no slot could be taken
  pr-fetch -- CMD [ARGS...]   a check's --fetch, as the monitor (a job script's prf, before the check runs in a slot,
                       where git fetches nothing): CMD with --fetch before its --, lazy fetches on, no input, in a time
                       limit; what it prints goes to stderr. Exit: CMD's; 124 when it ran out of time, 127 when it did
                       not start
  pr-tree DIR...       each DIR (missing, or empty) made a folder for a tree PR code reads: in the container the PR
                       runners' group's, 2750, no ACL, the one kind of folder a PR runner reads in the scratch folder;
                       on a host a plain folder. An agent's worktree is made so before git adds it (the session's
                       guard refuses `git worktree add` without it), and so is a folder for files PR code reads. Exit
                       0; 1 when one could not be made so (a line on stderr says why)
  loop-status          whether a tick loop runs, the last tick and full tick, the next deadlines, the day's tick counts
                       (exit 0 when a loop runs, 1 when none does)
  poll | queue         the individual steps, for debugging
  status [--queue]     how the monitor operates right now, from local files only (no remote call): on or off, stopped
                       or draining; the pieces running (the review driver, the tick loop that runs here as rk tick-now
                       finds it: the orchestrator, tools/tick_loop.py or the container's tick loop, and the last tick,
                       the admin dashboard's service, the orchestrator, the gate, develop sanity); the parallel reviews
                       and the parallel builds, each current setting against its range with why the monitor set it so
                       and its last changes; the review queue, the work in flight and the reviews stalled on
                       verification; the RAM disk's use; then the summary from stored state, and the GitHub logins'
                       last check. --queue: the summary's queue computed again (git work), and the dashboard written
                       with it. Whether it is configured right and reaches everything: doctor
  gh-logins [--side read|write|session] [--json]   which GitHub login each side uses and whether the split is right,
                       probed now (a comment with no body and a branch on a commit that does not exist, each refused
                       either way: 422 may, 403 may not; rkmon_lib/ghlogins.py): exit 1 when the read or the session
                       login can write, or the write login cannot comment while something that is on writes to GitHub,
                       or push while something that is on turns auto-merge on or off. --side: that side only (--json:
                       its probe as JSON; the container's broker runs `--side write` as the write user)
  pause | resume       pause the monitor and resume it: while paused (OFF) the tick decides nothing and starts nothing
                       new; running jobs finish (the Claude loop checks this every tick)
  stop [--drain [--timeout-minutes N]]   stop the running pieces now: the review driver, the work in flight (the
                       drafts, the gate, sanity and check jobs) and the admin dashboard server.
                       --drain first waits as rk drain does: nothing new starts, and the work in flight finishes,
                       for at most N minutes (default update.drain_timeout_min), what is left being stopped then; it
                       refuses while another drain is in force; when the configuration cannot be loaded, no review
                       can finish: it says NOT DRAINED, why and the fix, and stops at once
  start                start again after a stop or a kill: the admin dashboard server; then the Claude side resumes
                       from Claude Code
  kill [--wait S]      kill every process the monitor started, at once, and print what it killed. It
                       first waits up to S seconds (default 30) for a GitHub write in flight, so none is cut in half,
                       and never touches a process it did not start (the Ollama server, Claude Code, your shells)
  drain [--timeout-minutes N]   nothing new starts while the work in flight finishes (rkmon_lib/drain.py), for at
                       most N minutes (default update.drain_timeout_min). Exit 0 once nothing is in flight, the flag
                       left in place until rk undrain lifts it (rk start clears it too, and it lapses 60
                       minutes after the drain), or once it sees the monitor stopped (nothing is left to drain);
                       exit 1, the flag lifted, when the work is still in flight after N minutes; REFUSED, exit 4,
                       while another drain is in force
  undrain              lift the flag rk drain set: new work starts again, but a pause you set stays
  freeze REASON | unfreeze      stop or resume merge arming (reviews continue)
  clear N | unclear N  pre-clear a PR for auto-merge (the user's call only)
  prioritize N [WHY...] | unprioritize N   float PR N to the top of the review and merge queues, as a develop fix
                       is (the user's call only)
  serve [--port P] [--bind ADDR] [--public-port P]   the admin dashboard's service (tools/rkweb.py starts it and
                       stops it): the dashboard, its "Request PR priority" box and the Approve / Decline of other
                       teams' requests, of the proposed tickets and of the in-depth reviews. Every request carries the
                       access token, and it does not start without one (where it listens and who can reach it:
                       README.md, "The admin dashboard"). With features.dashboards.team.delivery serve it serves the
                       team dashboard too, read-only but for its request box
                       (features.dashboards.team.priority_requests.enabled). --port: the port it listens on; --bind:
                       the address it listens on (default: loopback on a host, every address in the container);
                       --public-port: the host port a container's dashboard is published on (default:
                       features.dashboards.admin.port)
  orchestrate [--check | --running]   the container's orchestrator: one long-lived Claude session that runs the
                       skill's resume; then rk tick every monitor.tick_interval_minutes minutes, read first by the tick
                       triage (the rule filter, rk tick-filter), so the session gets a turn only for the lines that
                       need Claude, and the review driver (rkmon_lib/orchestrator.py); --check says what it lacks
                       (exit 5) or that it is ready; --running, whether a session runs here (exit 0, with its pid) or
                       not (exit 1)
  tick-due [--minutes N]   exit 0 when the last tick is over N minutes old (default 25) or there is none, else 1:
                       the container runs rk tick itself when the orchestrator's ticks stall
  tick-min             the minutes between the container's ticks: monitor.tick_interval_minutes (the orchestrator
                       and the tick loop that runs without it both take it)
  stopped              exit 0, and say since when, while an rk stop or rk kill is in force (rk start ends it), else 1:
                       the container's restart loops stay down meanwhile
  applied [--load] [--json]   your confirmed settings the monitor runs, from their applied record (exit 0 with
                       one, else 1); rk start records them. --load loads the configuration now and records its layer,
                       so that a delivery that needs no restart is taken up at once: exit 1 when nothing was recorded,
                       and REFUSED with the fix, exit 4, when it cannot run. --json: the record, whether a pause
                       is set and whether the monitor is in standby, as the host's deploy side reads them, or the
                       refusal and its fix
  roles                each pass of the deploy settings resolved (pass, model, provider, effort: a default pass
                       names whose model and effort it takes, e.g. `ci_triage_pass  default → opus  claude  high (from
                       verification_pass)`), then the other roles with their effort and where it comes from, then the
                       last run of the CI triage, first code quality and notice draft passes
  first-pass N --head SHA   the CI triage pass and the first code quality pass at that head, when they name a model
                       (the review driver's job): leads with checked quotes into workers/prN/first_pass/, which rk
                       review-prep copies for the verifier and the code quality reviewer; a miss is logged, never an
                       error
  notice-draft N --kind overlap|drift   the first draft of PR N's overlap or drift notice by the notice draft pass
                       (leads only, for the orchestrator to check): <tmp>/notice_draft_prN_<kind>.md; with the pass at
                       default, or a model that fails, it says to write the notice without one (exit 0)
  role-run planner|analyst --input FILE --output FILE   one call of the planner's or the analyst's model
                       (rkmon_lib/roles.py; the tick's job, whose answer the next tick reads): one chat on the role's
                       endpoint and model, the answer's JSON, or why there is none, written to the output FILE (exit 0
                       whenever it is written)
  insights [--days N]  where the time goes, per PR (default 14 days): rebases, conflicts with develop, review and
                       fix rounds, CI runner waits (read-only)
  usage [--days N] [--by-review]   the tokens the models used, per UTC day (default 7 days): the orchestrator's
                       Claude session (input, output, cache read and written) and its subagents, the reviews posted,
                       the reviewer model's; --by-review: per review we posted, Claude's
                       (its audits' turns and the subagents its briefs named) and the reviewer model's (read-only)
  priority-requests    other teams' priority requests (from the team dashboard, read each tick): the pending ones,
                       then those answered or approved in the last day. Approve one with `prioritize N`
  priority-decline N   decline PR N's pending priority request (recorded, dropped from the list; local)
  ticket-draft FILE|-  add a proposed ticket for a problem found along the way (ticket creation, out_of_scope_issue): a
                       --- block (title, type, priority, assignee, summary; labels, component, pr, source) and then the
                       full description, or a JSON object. It waits on the admin dashboard's "Tickets waiting for you"
                       card for the user's OK (local; a proposed ticket whose title waits, or was filed or declined
                       lately, is refused; one whose title waits on a proposed ticket that cannot be filed takes its
                       place)
  tickets              the proposed tickets on the card: waiting, being filed, filed in the last day (read-only)
  ticket-file ID [--again]   the user's fallback for the card's Approve: file proposed ticket ID through the atlassian
                       CLI (guarded, ledgered; only while ticket creation is on; in the container the writer files it).
                       One that ticket-draft's checks refuse is not filed: it waits, with their words as its note.
                       --again: a proposed ticket whose filing was cut off (check JIRA first). Never the monitor's
                       Claude session: its guard refuses it
  ticket-decline ID [WHY...]   the card's Decline: drop proposed ticket ID (journaled; local)
  jira-account add LOGIN ACCOUNT | jira-account list   the JIRA account (an email or an account id) that the follow-up
                       tickets of GitHub login LOGIN's items are assigned to, recorded in the internal state (a pair
                       recorded already is replaced); list: every pair the configuration runs with (local)
  ledger [N]           show the last N outward actions
  log TYPE PR TEXT...  add an entry of type TYPE to the daily log by hand, such as a ticket, for PR (a number, or -)
  save PR TYPE TEXT...   record in the daily log an issue the system kept out of develop, for PR (a number, or -):
                       TYPE finding, fixed, gate, develop, verify or other (any other word is other)
  daylog [DAY]         the daily log of DAY (YYYY-MM-DD, UTC; default today) written again: its file, then its summary
                       line
  abi                  show the ABI lanes (binary, wire, family strings) from the last queue computation
  abi-ok N             after review: accept PR N's ABI notes (e.g. an append-only header change) at its current head
  batches              show the merge batches from the last queue computation
  merge-plan           show the merge plan from the last queue computation: the order, the conflict-free waves, who
                       each landing makes rebase, and every near-merge PR's rebase advice (read-only)
  board                the status board in Markdown: a row per part of the monitor, then the Needs-you list
                       (read-only)
  html                 the dashboards' pages written now (the admin dashboard's file, and the team dashboard's copy
                       that goes to the web host), then the admin dashboard's file
  group N [N ...]      pin a user group: validated together as one batch and landed in this order
  stats [--week YYYY-Www]   write the weekly statistics report (default: the last complete week) and print it
  spend [--day YYYY-MM-DD | --week YYYY-Www | --since YYYY-MM-DD] [--by part|kind|size|pr] [--experiments] [--write]
                       review spend (rkmon_lib/spend.py): per UTC day and ISO week the reviews posted, kept unposted
                       or closed without a post, covered heads (merge-only / rebase, no Claude spend), reviews $,
                       overhead $ and total $, and the mean, median, p90, min and max per review; the selection's
                       outliers (above Q3 + 1.5 x IQR) and its breakdown (--by, default part). --experiments adds
                       experiment and other tasks' spend, apart. --write also writes spend_YYYY-MM-DD.md for each
                       complete day of the selection and spend_YYYY-Www.md for its weeks (paths.spend_dir).
                       Refreshes the dashboards' card (local: no GitHub; reads the transcripts and git only)
  scoreboard [--days N] [--state DIR]   per PR author, the last N days (default 30): PRs merged, rework for others
                       (rebases forced + re-reviews caused), develop breaks, landed ahead of plan, merged with open
                       review items, follow-ups, sorted by rework, with the team's totals and the trend; read from
                       state/scoreboard_events.jsonl only (rkmon_lib/scoreboard.py; read-only)
  scoreboard-backfill [--days N] [--dry-run] [--no-github] [--state DIR]   derive the scoreboard's events of the last N
                       days (default 21) from every record (events, ledger, daily log, conflict history, journal, tick
                       log, follow-up tickets, incidents) and append the ones not recorded yet; GitHub GETs fill the
                       merges and develop issues the records lack (--no-github: none). --dry-run prints the counts and
                       writes nothing. --state: another state folder (a copy), for a rehearsal
  conflict-files [--days N]   the files PRs conflicted on with develop most in the last N days (default 7), the PRs,
                       and the develop PRs that last changed each (read-only)
  ci-times [--days N] [--collect]   CI time per PR (push → CI result per head) for heads pushed in the last N days
                       (default 7); --collect, or no heads recorded yet, reads the heads not recorded first (capped)
  review-start N [--kind first-round|follow-up] [--now] [--again]   start PR N's review now (in prototype or live), as
                       one of the parallel reviews; refused while all of them are busy (queue it:
                       review-queue), and for an in-depth review the operator has not approved (review-approve); an
                       unsettled head (features.review_automation.schedule.push_cooldown_minutes), or a follow-up
                       inside the gap after our last review of the PR
                       (features.review_automation.schedule.follow_up_cooldown_minutes), is refused unless --now.
                       --again: a re-review at the head of our last review, posted or kept unposted, or at a head we
                       posted on (the head of our last review is not carried as covered; no gap)
  review-queue [N ...] [--again] [--drop]   the review queue, which the driver starts before the plan, in order: list
                       it, with what each entry waits for; add PRs at its end (one queued already keeps its place and
                       takes --again or not); --again: a re-review at the head of our last review or at a head we posted
                       on (neither our review there nor the gap keeps it back; the settle does); --drop: remove them. An
                       entry leaves once its review starts or its PR closes; a push during that review, before its post,
                       puts it back at the front as it was, as the same round when review-start N --kind forced one. An
                       in-depth review waits for the operator's approval (review-approve): adding one prints its size,
                       agents and cost (local)
  review-status        every review in flight by its part (the dashboards' In review count); the work in flight:
                       stage, worker and checks jobs, log tails; the review queue; the pool's and the shadow drafts
  review-driver        the continuous review driver (tools/review_driver.py runs it): advances the work in flight
                       every minute, up to the parallel reviews at once; exits on REVIEW_READY or REVIEW_MANUAL, when
                       nothing can start, or after 2 hours. One at a time (a lock)
  review-advance       one advance of the reviews, as the review driver makes one every minute: the work in flight moved
                       on, and reviews started up to the parallel reviews; nothing while the monitor is stopped or the
                       last tick is too old (TICK_STALE)
  review-skip N | review-unskip N   keep PR N out of (or back in) the autopilot picks
  review-close N       close PR N's review awaiting verification or in flight without a post, when a review our account
                       posted on GitHub covers its head and stands for ours: its first line gives one of this
                       deployment's names, or none (an older monitor's, or one posted by hand), never another's. That
                       review is our last review of the PR, as a post would be, so the PR is not reviewed again at
                       that head and a new head gets a follow-up. Refused, nothing changed, in a mirror, when no such
                       review covers it, when nothing awaits verification or is in flight, or when the review moved on
                       to another head while GitHub was read. Reads GitHub before it takes the state lock; writes
                       nothing to GitHub
  review-approve N     approve PR N's in-depth review, a first round of a PR over
                       features.review_automation.large_prs.in_depth_review_over_lines: its first review, which starts
                       at its turn, or one forced once it is reviewed, which review-start N --kind first-round starts.
                       The PR leaves review_skip. The operator's, as the admin dashboard's Approve; decline with
                       review-skip N (local)
  review-recheck N     run the checks of PR N's draft awaiting verification again, at the same head, after a fix to the
                       checks themselves: it needs a free parallel build and room on the RAM disk, and is refused while
                       the monitor is stopped or drains. The next advance records the result (RECHECK_DONE #N)
  drafts               the draft stage: the roles' model bindings (never a key), the stage, the ceiling, the pool's and
                       the shadow drafts, the last drafts with their method, tokens, requests, waits, minutes and prior
                       rows by settler, the last allow or deny decisions for the load, and the drafters-vs-reference
                       card (read-only)
  draft-lint N [--dir DIR]   a draft of the first review pass checked against its own inputs (lessons L-501):
                       verdict words, line 1, round, ticket tags, "(ran)" with no matching tool call, CI and merge
                       words against checks.txt and merge_conflicts, rule_scan hints as findings, our own items, prior
                       statuses and counts against threads.md and commits.md, a degenerate draft; flags only (the
                       draft is unchanged), written to <folder>/checks/lint.md (default folder workers/prN); exit 0
                       no flags, 1 flags, 2 no draft. REVIEW_READY runs it on every draft of the local review loop
  werror N [--tree DIR] [--build DIR] [--base REV] [--dev-tree DIR] [--files F ...] [--budget S]   clang's
                       diagnostics the GCC build misses (tools/tidy_check.py, the checks' "Superbuild tidy"
                       section): clang-tidy with the provider's .clang-tidy and clang++ -fsyntax-only -Wall -Wextra
                       -Werror over the TUs PR N's change touches (its changed sources, and the TUs including a
                       changed header), feature macros off and on, on wt<N>m (else wt<N>) with its native build's
                       compile_commands.json (the review's shared build while its slot is kept, else the build
                       folder's; --build names another); every distinct error, file:line; --dev-tree: an error
                       develop's copy has too is not counted. As PR code, in a slot of its own. Exit 1 on an error.
                       Read-only
  probe-check DIR TREE [--probe NAME] [--devbase REV | --devbase-tree DIR] [--simfix PATCH | --simfix-tree DIR]
                       [--base BRANCH]   a saved probe checked before it is saved (rkmon_lib/probes.py, lessons
                       L-577): each probe of DIR run through tools/run_probes.py on DEVBASE (the merged tree's
                       first parent, else the merge base with --base, default develop), on TREE and on a simulated
                       fix (PATCH applied to a copy of TREE), each tree but TREE a git archive in a scratch folder;
                       it must read 2, 1 and 0: `<probe>: DEVBASE 2 · tree 1 · simfix 0 · OK`. A crash, pytest's
                       rc 5 and any other code read 3. Flags: a hardcoded host path, a ^FAILED grep without
                       SUBFAILED, HIP_VISIBLE_DEVICES=-1, a script that never reads $1. Without TREE: the flags
                       only, of DIR or of every probe folder under it. Exit 0 all OK, 1 one is not, 2 nothing to
                       check
  probe-port DIR [--write] [--root NAME]   saved probes written for a live layout rewritten to the probe
                       environment contract (rkmon_lib/probes.py): paths under a home folder's NAME (default
                       rocke_review; --root adds others) and under /opt/rocm become PROBE_PYTHON, RKMON_BUILD_DIR,
                       RKMON_REPO_DIR, RKMON_ROCM_LLVM_BIN or TMPDIR; per file: converts, needs a hand edit (each
                       path and why) or nothing; then the counts per probe and per file. A dry run unless --write,
                       which rewrites only the files that convert cleanly. Exit 0 no hand edit left, 1 some, 2 no
                       probe
  stage-ours N DIR     our last review of PR N, posted or kept unposted (its head, when, and its body: a post's read
                       from GitHub or its copy, a kept one's from its file) into DIR/ours.md, for a draft's prior
                       items (draft_run.py runs it; writes only that file)
  classify N [--record]   compare PR N's own change at our reviewed head and now (merge-only, rebase-only vs
                       substantive); --record classifies the live head (read from GitHub, fetched into the clone),
                       refuses when it is not the head the tick saw (run `rk tick` first), records a merge- or
                       rebase-only head as covered by our review, and closes a review of that head awaiting
                       verification (or in flight) without a post
  review-prep N [--round fr|fr<k>|fu<k>] [--out DIR]   a verification's inputs in one review directory (default
                       <tmp>/v<N><round>; round fr before our first review, posted or kept, else the next follow-up):
                       facts.txt, classify.txt, overlap_section.md, files.txt; then the prep summary, with the
                       verifier's and the code quality pass's worktree paths in it (wt_v<name> and wt_q<name>, each
                       made a tree with rk pr-tree first: the session's guard refuses a worktree added elsewhere or not
                       made so) (read-only but that directory)
  review-split N [--max-lines L] [--round R] [--out DIR] [--max-agents K] [--team-file F] [--head SHA]   a review too
                       large for one verifier (REVIEW_MANUAL) cut into parts, each filled to at most L changed lines
                       (default 8000) area by area, files that change together kept together, generated/golden/data
                       files summarized: <review dir>/parts/part<k>.md (+ .files), cross.md and plan.json, then the
                       parts, the dispatch plan and its cost: a verifier per part, which runs its part's code quality
                       checks too, in waves of at most K agents (default 20), then the cross-cutting verifier that
                       writes the post, each agent with the description and brief to dispatch it with, which name the
                       PR and the round (rkmon_lib/split.py; read-only but parts/)
  facts N              the facts for a verifier brief, read now: head and state, every force-push, the last commits,
                       our reviews, our account's reviews this deployment did not post, the ABI lanes, develop issues,
                       the latest review per reviewer, unresolved threads, the user's hand-written comments in full,
                       the worker's inputs (read-only)
  heads N [N ...]      each PR's head, state, merged, mergeable state and last update, read live (read-only)
  ci-fails N           PR N's live head, then its check runs' latest attempts that failed (FAIL) or still run (RUN),
                       each with its url, and the counts: what a post spec's FAILOK and RUNOK name (read-only)
  shard-times N [--every]   whether PR N's CTest shards take longer by the PR or by the runner pool
                       (rkmon_lib/shard_times.py): each failed or timed-out shard job at the live head (--every: each
                       shard) with its pool, time and CTest entry times, against develop's runs of the same shard on
                       the same pool within a day; the last line is `VERDICT PR-added +a..b s per shard`, `VERDICT pool
                       (develop same pool m–n min)` or `VERDICT not compared` (read-only)
  spec N --round R [--dir DIR] [--light TEXT] [--save TEXT] [--cost-verifier T] [--cost-quality T] [--out FILE]
                       the post spec from the review directory (default <tmp>/v<N><round>, paths.tmp): notes.md's
                       premises:, FAILOK, CI pending:, conflict set and decide: lines, the post's Bots / CI anchor (a
                       "; n ✓" added to a row without one), and live reads: FAILOK and RUNOK as rk ci-fails, CONFLICTS
                       from merge-tree with the PR's base, the body's tracker tag, the threads. CHECK: lines say where
                       the notes and the live reads differ (exit 3); writes the spec (default <tmp>/spec<N><round>.json)
  dispatch-pack N [--round R]   review-prep, then what the briefs need in one compact print: head and state, our last
                       review's rows and findings, threads, the operator's comments, the delta since our last review,
                       the checks' flags and shared build, CI's FAIL and RUN lines, the overlap decisions naming N, a
                       micro round's items, and the lane line, whether the review needs the code quality pass and why
                       (`lane: needed (...)` or `lane: exempt (...)`)
  spec-post N --round R [--dir DIR] [--light TEXT] [--save TEXT] [--cost-verifier T] [--cost-quality T] [--out FILE]
        [--again] [--dry-run] [--devbase-recheck]   rk spec with its options, then, when it made the spec with no
                       CHECK line, post N with it (--premises; --again and --dry-run as rk post's); CHECK lines stop it
                       (exit 3). With --devbase-recheck, a stop only on develop's change to the PR's files prints
                       develop's diff stat for them and posts nothing
  wait-quality DIR [--max-min M] [--slice-min S]   block until the code quality pass's quality.md in DIR is final:
                       ready, waiting (after S min, default 4.5: call again) or timeout (M min, default 20, since the
                       first call)
  job-log ID [--errors [--pr N | --files F ...]]   a CI job's log: its line count, then its failing lines (the
                       whole log is kept in <tmp>/joblog_ID.txt; read-only). --errors: every distinct `error:` line
                       in log order, then each changed C/C++ source (PR N's, or --files) the log never compiled (no
                       /<name>.o in it)
  evidence-check N [--dir DIR]   a draft's evidence checked again with no model: quotes at their lines, cheap runs
                       re-run, unchanged files' blobs and staged facts read again; the report (evidence_check.md) and
                       prior_settled.md in the run's folder (default workers/prN), each failed or missing claim marked
                       in its draft, the run kept under lessons/drafts; exit 0 PASS, 1 FAIL, 2 no draft run
  evidence-check N --offload [--round R]   the review offload's evidence, checked again with no model: each quote at
                       its file:line in the tree and commit it names, each cheap command re-run under a time cap and
                       its output compared, each claim without evidence listed; PASS or FAIL, and the auditor's
                       spot-check sample, in the run's evidence_check.md; the live run is kept in lessons/offload/
                       (exit 0 PASS, 1 FAIL, 2 nothing to check; GitHub reads only)
  shadow-report [N ...]   the shadow trial: per PR, the offload's findings against the review we posted (read from
                       GitHub): matched, missed and extra, with their evidence check; then a summary over the last
                       offload.trial_prs PRs, in <OUTPUTS_DIR>/rocke_shadow/ (GitHub reads only)
  followup-scan N [--out FILE]   the follow-up ticket candidates in every reviewer's feedback on PR N: deferrals
                       ("follow up", "next PR", "TODO", a ticket; in resolved threads too), requests nobody answered,
                       and our open 🔴/🟠 rows, into FILE (default state/followups/followups_N.md); the tick runs it once
                       when an in-scope PR merges (read-only but that file)
  overlaps [N] [--section]   work an open PR adds that another open PR also adds (same name, same logic) and copies of
                       a develop pattern, with the land order and recorded owners: every group, or PR N's; --section
                       prints checks.md's section for N (read-only)
  clash N [--head SHA] [--heads N=SHA,..] [--develop REV] [--others N,N] [--budget S] [--max-fetch N] [--cache DIR]
        [--new] [--build]   PR N against
                       every other open PR (rkmon_lib/clashes.py): both land orders on develop (conflicting files,
                       modify/delete pairs), the silent clashes that merge clean (a name one removes that the other's
                       new lines use, a file one moves or deletes that the other includes or imports), leads (logic
                       one changes under the other's new tests) and who ports what; its ABI lane: another open PR
                       claiming the same abi.h counter (ABI NUMBER CLASH: whichever lands second renumbers); the
                       checks' "Clashes with other open PRs" section; also a name one removes that the other's new
                       Python lines spell as a string (a CTest name its CMake lines drop, an attribute patched by
                       name). --heads: snapshot heads. --max-fetch: the most PRs whose changed files it fetches when
                       they are not local, one request each (default clashes.max_fetch: 4); a PR past it is NOT RUN.
                       --new: only the open PRs the checks' scan did not compare (opened since; `rk post --premises`
                       names them). --build: each pair merge (this PR first) extracted to a scratch folder, the C/C++
                       files both change compiled -fsyntax-only with the shared build's flags and both PRs' changed
                       tests collected, as PR code in a slot of its own (BUILD pass|fail|skip). Read-only but temporary
                       merge objects (no ref), the scratch folder and the cache (default cache/clash, paths.cache)
  mutants N [--head SHA] [--since SHA] [--budget S] [--cap K] [--per-mutant S] [--out DIR]   the checks' mutation
                       sweep for PR N (tools/mutation_sweep.py): each check its Python lines add or change (--since:
                       only those the push since SHA adds or changes, as a follow-up's checks do), disabled once,
                       against its tests, in a scratch copy of the head (never wt<N>); survivors are checks no test
                       catches, each re-run against every test file importing its module, under the checks venv and
                       paths.torch_python (SURVIVED (CI importers, torch, engine), or vacuous: torch-only path).
                       Report and logs in DIR (default <tmp>/mutants_prN)
  ctest-labels N [--head SHA] [--budget S] [--out DIR]   the checks' "CTest labels and tiers" (tools/ctest_labels.py):
                       PR N's merge base and head, each configured (not built) as hip-kernel-provider configures rocKE,
                       their CTest entries in the build and the installed tree; leads: an entry that loses a YAML tier
                       in one tree, leaves a tree, or comes with none. Scratch copies, removed after; the configure
                       logs and entries in DIR (default <tmp>/ctest_labels_prN)
  lane-of TARGET... [--pr N | --rev REV] [--collect] [--log FILE] [--json]   where a test runs in CI, from the tree
                       (tools/lane_of.py): TARGET a test file (in the rocKE tree), a node id or a CTest entry; at
                       develop's head, PR N's head or REV. Its CTest entries and tiers, the lanes of ci_test_tiers that
                       select them, the entry's -m markers, module-level skips and imports of ci_absent_modules: one
                       line each, "CI: runs (<entry>, <tier>)", "CI: skips(<reason>)", "CI: no lane(<why>)" or
                       "CI: errors(<why>)". --collect also collects it as CI would (those modules unimportable, skip
                       marks evaluated); --log reads a CI shard log for the entry's lines. Exit 0 when every target
                       runs in CI, 1 when one does not, 3 NOT RUN. Scratch copies, removed after; each commit's
                       entries cached in cache/lane_of
  winsim [TARGET...] [--pr N | --rev REV] [--base REF] [--only SIMS] [--scan] [--budget S]   the Windows lane,
                       simulated on Linux (tools/winsim.py): the rocKE tree at PR N's head (or REV) in a scratch copy,
                       each TARGET (test files or node ids; default with --pr: the test files the PR adds or changes)
                       run plainly and under tools/pytest_plugins/rkmon_winsim.py (text opens in cp1252, CRLF text
                       writes, a NamedTemporaryFile refused a second open while open; --only os adds a Windows os.name,
                       os.sep and os.linesep): a test that passes plainly and fails simulated is a lead, named with the
                       simulation that breaks it. With --pr (or --scan) the C/C++ lines the PR adds are scanned for
                       windows.h's macro names (far, near, IN, OUT, ERROR, DELETE, min( and max( without NOMINMAX, ...),
                       a lead only in a file that includes a Windows header (itself or through the tree's headers).
                       The tests run as PR code, in a slot of its own (in the container a PR runner's; on a host as
                       you, with PR code's environment). Last line WINSIM pass|fail|skip; exit 0 nothing found, 1 a
                       lead, 3 NOT RUN
  moved-refs N [--head SHA] [--budget S]   the checks' "References to moved or deleted files" (tools/moved_refs.py):
                       the references PR N's head still makes to the old path of each file it moves or deletes (docs,
                       includes, imports, CMake, YAML) that resolve to nothing, by file:line, and the links in the docs
                       it changes whose target is missing. Git only, read-only
  doc-refs N [--head SHA] [--budget S]   the checks' "Docs against code" (tools/doc_refs.py): the doc lines at PR N's
                       head that name a function, class, env var, flag or arch the PR removes or renames (its code
                       has it nowhere), and the env vars, IR ops and dtypes it adds that the reference docs
                       (env_flags.md, op_vocabulary.md, dtype_support.md) do not name. Leads. Git only, read-only
  import-probe N [--head SHA] [--budget S] [--out DIR]   the checks' "Engine import probe" (tools/import_probe.py):
                       each Python module PR N changes under library/ and platform/python/, imported at the head and at
                       the merge base with only rocke_engine built (a stand-in), its cheapest backend="python" entry
                       point called; leads: what needs an unbuilt engine at the head and not at the base. Scratch
                       copies, removed after; logs in DIR (default <tmp>/import_probe_prN)
  peer-checks N [--check NAME] [--head SHA]   the checks' peer-review sections (rkmon_lib/peer_checks.py), every one or
                       NAME's: live review threads no owner answered, tracker IDs of any prefix, ROCKE_* env reads
                       env_flags.md lacks, new library tests in no CMake or CTest list, C/C++ getenv, a title that is
                       not Conventional Commits, split tautologies, bare sibling test imports, workload sizes in kernel
                       names, pull-request references, timings in source comments, new platform -> library back-edges,
                       kernarg changes with the ABI string kept, over-written fixed C arrays, NaN-blind failure tests,
                       unguarded optional test imports, repeated numbered headings (flags), and the leads (bool() of
                       request fields, torch device queries, file(GLOB) without CONFIGURE_DEPENDS; a lowerer change
                       with no parity pair; unused unpack targets). Git, the staged title (else GitHub's) and GitHub
                       thread reads; read-only
  overlap-decide KEY OWNER [--members N,N] [--note TEXT] [--by LOGIN] [--same-as KEY]   record which PR owns a
                       group's shared code, as a review proposed it (local: no GitHub write); --same-as files an
                       earlier decision again under this key, and its notices count here
  post N FILE [--again]   post a verified review comment (guards + ledger); the only write path for reviews
  post N FILE --premises SPEC [--again] [--dry-run]   post only while the verifier's premises hold (the spec `rk spec`
                       writes): head, develop, conflicts, CI, threads, comments, the ticket tag, then the post's checks
                       (the ✓ count filled in), the fix-line premise (rkmon_lib/fixline.py: a fix that guards a test on
                       a module CI lacks, or a "CI still runs it" that lane-of denies at the head, stops it; another
                       skip guard, a tier change or a test move needs a `fixcheck: N<k> ran|read <how>` notes line), the
                       ABI premise (rkmon_lib/abipremise.py: a "no bump" while the checks' ABI section lists a change
                       without a bump, or an ABI number neither develop's nor develop + 1, stops it unless the notes
                       carry an `abi:` line naming abi.h's rule), `rk post`, `rk save N verify`, the spec's AFTER lines
                       and the notes archive. --again: a second post at a head we posted on (a re-review), the state's
                       head the live one; every other premise and guard applies, the hourly cap too. While review
                       posting is off the review is kept unposted instead, and --again keeps a second one at the head of
                       our last review, kept or posted. --dry-run (with or without --premises) stops before the post,
                       after `rk post`'s guards
  jobs [ID]            list recent jobs, or show one job's log tail
  cleanup [--dry-run] [--all]   what reviews of merged or closed PRs left on disk removed, as each tick does
                       (rkmon_lib/cleanup.py): their worktrees, builds and workers/pr<N> folders, for as many PRs as a
                       tick takes (cleanup.PER_TICK) unless --all, then the finished jobs and sanity runs older than
                       the jobs' retention; --dry-run: what it would remove, with nothing removed
  mirror-report --live-state DIR --mirror DIR [--live-claude DIR] [--json] [--out FILE]   the A/B comparison of a
                       live monitor and its mirror (a monitor beside the live one that sends nothing): each PR reviewed
                       at one head on both sides, its findings, priors, hygiene, Claude's dollars per pass and session
                       from the transcripts (the live ones in --live-claude, its Claude projects folder), the drafting
                       models' tokens, minutes, the totals per arm, the judge's worklist and every lost review (reads
                       only; no configuration needed)
  sanity [SHA] [--base SHA] [--rehearsal] [--now]   start a develop sanity run now on SHA (default: the develop head),
                       for the families the commits landed since --base touched (default: the last green sanity run);
                       --rehearsal: a drill on an old SHA, with no outward action ever taken; --now: as for bisect
  sanity-status        the last green develop sanity run, the develop heads waiting for one, and the run in progress
                       with the last lines of its log (read-only)
  guard-advance        process finished sanity/bisect runs and start pending ones (what each tick does while the
                       develop guard is on)
  incidents            list develop-break incidents
  bisect [--now]       start a bisect for the open develop-break incident; refused while the monitor is stopped or
                       draining, and, unless --now, while all parallel builds are busy or the RAM disk has no room for
                       it, as a gate or a sanity run is
  incident-note ID KEY VALUE   record issue/comment/revert_offered/culprit/fixed_by on an incident
  breakages            open develop issues of both types (rocKE first), and their incidents
  issue-file FILE [--new]      open a develop issue (front matter: title, labels with exactly one of the two types'
                       labels, assign, incident; guarded, ledgered)
  issue-comment N FILE --key KEY   comment on develop issue N, e.g. our bisect evidence (once per key)
  issue-relabel N --to stability|ci_red [--assign LOGIN]   swap issue N's develop label (guarded, ledgered)
  fix-link I P [--evidence FILE]   after verifying it: PR P fixes develop issue I (issue comment, PR notice, develop
                       fix at once; guarded, ledgered)
  fix-dismiss I P      stop the FIX_CANDIDATE line for (I, P); unlinks a link made by mistake (local)
  drift-dismiss N D --reason TEXT   PR N's drift failure on develop D (a commit prefix) is a harness false positive:
                       G6 and the merge blockers stop counting that run (local; recorded in state/drift_dismissed.jsonl)
  approvals            PRs ready for the user's approval, then what is left on every PR close to merge (who is next)
  automerge-off N      turn off auto-merge someone else enabled on a PR with a recorded risk at its head (the
                       auto-merge guard does so itself while features.merge_queue.mode is live; on the user's own
                       PRs, while features.merge_queue.auto_merge_guard.on_your_pull_requests is same_as_others)
  automerge-off N --user FILE   on the user's instruction: turn auto-merge off and post FILE's reason (no risk needed)
  gate-cancel REASON   stop the running gate job and free its place among the parallel builds
  teams-breakage N [--force]   alert the team chat: CI-red develop issue N, no fix planned
  teams-test           one card to the team chat, marked as a test that needs no action and naming the alerts the
                       channel gets: whether the webhook reaches it (REFUSED, exit 4, while there is no webhook)
  needs-you-teams [--dry-run]   the Needs-you decisions not sent yet, as one Teams message to the team chat, now
                       (needs features.alerts; ledgered); --dry-run prints it and writes nothing. Each full tick
                       sends them by itself, at most once per features.alerts.message_cooldown_minutes
  teams-send --type develop-break|needs-you   one message to the team chat channel, the message on stdin (JSON, as
                       rkmon_lib/post.py teams_message makes it), with the webhook read here: in the container the
                       writer's half of every team chat message (rkmon_lib/writer.py). Exit 4 when refused
  web-push FILE        the team dashboard's page FILE to the web host (publish.html.dest), with the key read here: in
                       the container the writer's half of every push. Exit 1 when it failed
  web-requests         the web host's priority requests file (publish.html.requests), printed, with the key read here
                       (read-only): in the container the writer's half of every read of it. Exit 1 when it failed
  atlassian [--site SITE] [--json] [--timeout-s N] WORDS...   one of the atlassian CLI's commands the writer lists
                       (rkmon_lib/writer.py ATLASSIAN_COMMANDS: the checks' reads, the project's create metadata, a
                       ticket's reads, filing, epic link, edits and comments), with its login: in the container the
                       writer runs it, so no other user reads the login. Anything off the list is refused (exit 4);
                       else the CLI's answer and exit code
  issue-assign N LOGIN   assign issue N to LOGIN (on the user's instruction; ledgered)
  label-create NAME --description TEXT --color HEX   create a label in the develop issues' repository (on the user's
                       instruction; ledgered)
  request-changes N FILE [--user]   request changes on PR N from our account: a confirmed hold, or --user on the
                       user's instruction (guarded, ledgered). Only the user's explicit re-approval lifts it
  hold N [REASON]      confirm PR N's hold, one the merge queue took pending triage, as the PR's fault, or hold PR N by
                       hand (a new hold needs a REASON). Unless the PR is the user's own or has our change request
                       already, it prints the next step: rk request-changes
  release N [REASON]   clear PR N's hold when develop or our tool is at fault: a PR held for a failed gate is gated
                       again
  holds                every hold on record: its PR, status, since when, from where, at which head and why; one whose
                       PR was pushed or closed since is marked ended (read-only)
  lessons [--week YYYY-Www]   collect a week's raw material for the lessons-learned review (default: last week)
  incident-close ID [FIXED_BY]   close an incident by hand
  notice N FILE --kind incident|queue|overlap|drift|gpu --key KEY   post a develop-health, merge-queue, overlap, drift
                       or GPU-test notice (guarded, ledgered)
  queue-advance        what each tick does while the merge queue is on: gate the head batch, report PROPOSE /
                       ARM_READY / DISARM_NEEDED
  gate N [--now]       start a gate job for PR N against today's develop; --now: as for bisect
  arm N | disarm N [REASON]   enable auto-merge for a cleared, validated PR (guarded) / disable it
  queue-notice N       create or update our in-place merge-queue status comment on PR N (posted while
                       features.merge_queue.mode is live and features.merge_queue.queue_notices.enabled is true)
  incident-add START END CULPRIT FIXED_BY SUMMARY... [--contributing N,N]   record a develop break (use - for
                       unknown; END - if ongoing); --contributing: the other PRs that contributed to it
  ungroup N            remove the user group that contains PR N
  selftest [--jobs N] [TEST...]   run the unit tests, or the tests named (a module, class or test, as unittest
                       names it: tests.test_selftest), in a home folder of their own; a test that leaves anything
                       in its home folder, its TMPDIR or its monitor home, or that writes in the package's folder,
                       fails the run; --jobs N runs the suite's modules N at a time, each in a home folder of its own

Remote GPU tests (features.review_automation.remote_gpu_tests.enabled, with review automation on: a Slurm cluster
over SSH, config gpu; rkmon_lib/gpu.py):
  gpu-run N [--arch A] [--case ID ...] [--same-node | --node NAME]   queue PR N's GPU-only cases (those its last
                       checks skipped as they need a GPU, or the cases named) on a remote GPU node, in the background;
                       in prototype too (no GitHub write). --same-node: on the node its last job on that target ran on (the
                       local records keep node names; posts name only the GPU target); --node NAME: on that node
  gpu-run --develop --arch A --case ID [--case ID ...]   queue develop's named cases the same way
  gpu-poll             what the tick does for the GPU jobs: read the queue, collect finished jobs, clean up leftovers
  gpu-status           every GPU job recorded here: state, Slurm job, account, node, result (local; node names show)
  gpu-queue            every job of the cluster login in the queue, this monitor's marked (read-only; empty: clean)
  gpu-cancel N|JOB|all give up on queued or running GPU jobs (scancel; the review's row then says not run)
  gpu-row N [--grade red|orange|yellow]   set the GPU tests row of our posted reviews of PR N to what its jobs say,
                       in place (guarded, ledgered; in prototype it only journals what it would do)
  gpu-check            what the login host has, read-only: the partition's nodes per GPU type, the accounts, the
                       Slurm tools, git and the remote Python with pytest
  A failure that is the PR's (develop passes it on the same node) is posted with `notice N FILE --kind gpu --key JOB`.

Setup and checks (they work before the configuration is complete):
  guide [build|deploy] print the operator's quick guide (QUICK_USER_GUIDE.md of the code that runs: in the container, the
                       image's or the active release's), or with build the image builder's guide (QUICK_BUILD_GUIDE.md),
                       with deploy the container's (QUICK_DEPLOY_GUIDE.md); README.md beside them is the full reference
  logins [--json]      every login and secret something may need, in the secrets registry's order
                       (rkmon_lib/secrets_registry.py), each with what needs it among what is on, its state (SET,
                       MISSING, WRONG, NOT NEEDED, or UNKNOWN when it could not be checked yet) and the command that
                       sets it; exit 1 while one that is needed is missing or wrong (docker/rkmon login walks them)
  secret-set NAME [--remove]
                       a secret from stdin into its file (mode 600), never printed, nor its length; NAME is any secret
                       the secrets registry lists (a login is rkmon login NAME's); an empty input, or one that is not
                       the secret's shape, is refused; --remove removes its file from the secrets folder
                       (docker/rkmon secret NAME runs it). In the container a secret the writer keeps is set by the
                       writer only, as docker/rkmon secret NAME asks from the host: the monitor's rk refuses one
                       (exit 4)
  writer-secrets [--json]
                       whether each secret the writer keeps is set, where, and when it last changed; never a value (in
                       the container the writer answers it: the monitor cannot read the writer's volume)
  claude-test          the claude login's test call: one message of one token to the Claude session's address, on its
                       model, sent as the session sends its requests (the login's key, never shown, as
                       integrations.claude_code's settings say); its row, PASS, WARN or FAIL with the fix, exit 4 on a
                       FAIL (docker/rkmon login claude runs it once the key is set)
  read-login           a GitHub token on stdin as the read login, never printed: logged in to a folder of its own
                       first, where the probe asks whether it can write (a comment with no body and a branch on a
                       commit that does not exist, each refused either way), then that folder's login takes the read
                       login's folder's place, whole; one that can write or one GitHub does not take is refused, as is
                       every token while repository.name names no public repository (the monitor watches public ones
                       only): exit 4, and nothing changes (docker/rkmon login github_read runs it)
  host-key NAME [--json|--record]
                       the own keys of the host the secret NAME's ssh key is for (web_host_ssh_key: the web host;
                       gpu_ssh_key: the GPU login host), the only keys ssh trusts it by: as ssh-keyscan reads them
                       here, with their fingerprints and whether the state records them; exit 4 when none is read.
                       --record: the known_hosts lines on stdin recorded as its own keys, refused (exit 4) unless each
                       is a key of that host (docker/rkmon secret NAME runs it once the key is set, and docker/rkmon
                       login for a key set already; each records the keys you accept)
  web-host-install     the request service (tools/webhost_server.py) on the team dashboard's web host, as its login's
                       user service rocke-monitor.service, from the settings rk runs on, over the web host's key
                       and its own keys the state records: INSTALLED, or FAILED (exit 4; 3 while the key or the web
                       host's own key is not set) or ERROR (exit 1) naming the step that failed; NOTHING TO DO while
                       the monitor serves the team dashboard itself, or priority requests are off; refused in standby
                       (exit 4); INTERRUPTED (exit 130) at a Ctrl-C, with what it leaves on the web host. In the
                       container the writer runs it, as docker/rkmon web-host install asks from the host: the
                       monitor's rk refuses it (exit 4)
  adopt plan|apply --stage DIR   an earlier monitor's state, staged by docker/rkmon adopt with its config.json and
                       standby set in it, into this deployment: what would change, or the change (the mark of the
                       adoption written first, which keeps standby on until docker/rkmon standby off; this
                       deployment's copies set aside, then the one-time move made due again when config.json came).
                       When it stops, what was done, where that leaves the state, and the fix: REFUSED, exit 4, when
                       nothing changed, else ERROR, exit 1 (rkmon_lib/adopt.py)
  standby-off          the end of standby, which docker/rkmon standby off runs, never the Claude session
                       (rkmon_lib/standby.py): the one-time move from an older install ended, when one is open (its
                       lines: what it renamed, removed and kept), then standby.on cleared, then the mark of an
                       adoption removed, last; RELEASED, exit 0, or NOTHING TO DO while the monitor is not in standby.
                       While the mark says rkmon adopt stopped part-way, REFUSED, exit 4, its fix adopt again. A move,
                       a state or a mark that cannot change keeps standby on: REFUSED with the fix, exit 4, when
                       nothing changed, else ERROR, exit 1, after what was done
  provision [--check]  the work volume set up from your confirmed settings, asking nothing (rkmon_lib/provision.py): the
                       folders, the worker tools, the clone, the code owners, the checks' virtual environment, the
                       agentic files and the checks of the local review loop's model, then its record in the internal
                       state. Each start runs it (rkmon start), a later one to put back what has gone missing: it is
                       not a command for you. --check: exit 0 when the work volume is provisioned (one an older
                       package set up counts), else 1
  doctor [--fresh] [--fails | --json]   whether the monitor is configured right and reaches everything, each check PASS,
                       FAIL or WARN with the fix: this machine's prerequisites, every configured value (secrets only as
                       set or missing, with their path), and read-only checks of each configured service (GitHub, JIRA,
                       the team chat's reachability, the model gateways, the GPU cluster, the Claude session's settings,
                       the RAM disk; in the container the Claude session's key, the claude login's test call); the
                       service checks are cached a few minutes (--fresh: check again). Its output is for this machine
                       only. --fails: only the names of the checks that fail, one per line; --json: the rows, as one
                       JSON list on one line. Exit 3 when a prerequisite fails (a row of prerequisites.txt), 4 when
                       anything else fails (the configuration, its folders or its secrets), else 0
  features [--runs PART]   the setup summary: each feature's switches, what starts, every capability (on, off,
                       incomplete) and every value they use; with no configuration yet, every optional feature off,
                       the fix and the defaults. While the configuration cannot be read, it says that nothing runs,
                       why and the fix, exit 1. --runs PART (web, claude, reviews, rules, merge-queue,
                       develop-checks): exit 0 when the deployment runs that part, 1 when it does not, 4 when there is
                       no configuration or it cannot be read (none runs)
  config show [--source]   the running configuration, one value a line under the name its file gives it, never a
                       secret; --source: before each value, the layer it came from (deploy, repository, package,
                       state, derived, or config.json). A key of your deploy settings file is deploy when the file
                       sets it and package when it leaves it out, as the first start or the last apply delivered
                       the file: a key you add at its default changes nothing, so it stays package until a later
                       change is applied. Read-only; exit 1, its problems after it, when it cannot run yet; REFUSED,
                       exit 4, why and the fix, when it cannot be loaded
  env [--json]         the RKMON_* environment the jobs give the worker tools (paths and settings)
  paths                every resolved path
  home                 the monitor's home (config.json and state/) and how it was found
  code install | back  refused, exit 4: in the container root alone changes the code (rkmon_lib/releases.py): rkmon
                       apply and every rkmon start on the host put this package's code in, and rkmon rollback goes back
                       to the code kept
  md2adf FILE|-        a Markdown file as the Atlassian Document Format (rkmon_lib/adf.py), for a JIRA description:
                       rk atlassian ... jira issue create ... --field-json "description=$(rk md2adf FILE)"

Stopping the monitor, from the gentlest: `rk pause` (`rk resume` ends the pause), `rk stop [--drain]` (stop now;
`rk start` resumes), `rk kill` (kill every monitor process at once; `rk start` resumes). rk stop, rk kill, rk undrain
and rk stopped read only what names the folders (config.json, or the container's layout), never the internal state or
your confirmed settings: a configuration that cannot be loaded never keeps the monitor from stopping. The Claude side is
stopped from Claude Code: `/rocke-monitor off` ends its loop and runs `rk pause`.

`rk --home DIR COMMAND` (or RKMON_HOME=DIR) runs with the home DIR. Otherwise the home is the toolkit's own folder when
it has a config.json, else the folder the pointer file ~/.config/rkmon/home names. The toolkit's folder itself is never
written.
"""

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rkmon_lib import events as ev_mod  # noqa: E402
from rkmon_lib import prcode, review_keep, rules_tick, slots, verify_stale  # noqa: E402
from rkmon_lib import (
    gh,
    gitops,
    jobs,
    plan as plan_mod,
    post as post_mod,
    queue as queue_mod,
    report,
    review,
)  # noqa
from rkmon_lib import (
    breakage,
    cleanup,
    daylog,
    guard,
    lessons,
    merge,
    merges,
    snapshot,
    stats,
    webdash,
)  # noqa: E402
from rkmon_lib import (
    abi,
    ci_runs,
    ci_times,
    conflicts,
    drift,
    facts,
    first_pass,
    overlaps,
    priority,
    publish,
    webserve,
)  # noqa: E402
from rkmon_lib import (
    config,
    container_layout,
    control,
    drain,
    features,
    followups,
    gh_cap,
    gpu,
    host_keys,
    paths,
    premises,
    prep,
    procs,
    registry,
)  # noqa: E402
from rkmon_lib import (
    analyst,
    idle,
    mergeplan,
    needs_teams,
    planner,
    roles,
    tickfilter,
)  # noqa: E402
from rkmon_lib import (
    assessment,
    bindings,
    draft_lint,
    draft_stage,
    evidence,
    mirror,
    scaler,
    spend,
)  # noqa: E402
from rkmon_lib import (
    clashes,
    dispatch,
    lanes,
    ghlogins,
    outward as outward_mod,
    retention,
    scoreboard,
    spec as spec_mod,
    split,
)  # noqa: E402
from rkmon_lib.store import stopped  # noqa: E402
from rkmon_lib.store import drain_stale, draining  # noqa: E402
from rkmon_lib.store import (
    ROOT,
    Store,
    load_config,
    minutes_since,
    now,
    settle_left,
)  # noqa: E402

# The checks that compare a PR's head with its merge base, by command: their tools (rkmon_lib/review.py runs each).
HEAD_CHECKS = {
    "ctest-labels": "ctest_labels.py",
    "moved-refs": "moved_refs.py",
    "doc-refs": "doc_refs.py",
    "import-probe": "import_probe.py",
}
NO_OUT = ("moved-refs", "doc-refs")  # git only: no folder of their own
PR_CODE_CHECKS = (
    "ctest-labels",
    "import-probe",
)  # they run the head's code: in a PR runner's slot, their folder its out


def resolve_gone(cfg: dict, scope_cache: dict | None = None):
    def _resolve(n: int) -> dict:
        p = gh.rest(
            f"repos/{cfg['repo']}/pulls/{n}",
            jq="{state, merged_at, merge_commit_sha, by: (.merged_by.login // null), labels: [.labels[].name]}",
        )
        if p["merged_at"]:
            return {
                "state": "merged",
                "merged_at": p["merged_at"],
                "merge_sha": (p["merge_commit_sha"] or "")[:10],
                "by": p["by"],
            }
        if p["state"] == "closed":
            return {"state": "closed"}
        folders = config.scope(cfg).folders(p["labels"])
        known = snapshot.answers(cfg, scope_cache or {}, n)
        if folders and all(known.get(f) is False for f in folders):
            return {
                "state": "open",
                "note": f"shared label only, and its diff changes no file in {' or '.join(folders)}",
            }
        return {"state": "open", "note": "label removed or now a draft search miss"}

    return _resolve


def refresh_ours(cfg: dict, store: Store, st: dict, keys: set[str]) -> int:
    """Re-read our latest review comment for the given PRs; resolve the reviewed head when the review is new. While
    review posting is off, a review kept unposted (post.save_draft) stays our last review until GitHub shows a post of
    ours made since. Once posting is on, our last review is the one GitHub shows, or none: every PR whose last review
    is a kept one is read again, so its next post is a first round when GitHub shows no post of ours, and a follow-up
    to the one GitHub shows (another monitor's, under the same login) when it does. A review of our account that
    `rk review-close` took as our last review stays it until GitHub shows a review of this deployment's own posted
    after it was taken (post.taken_stands).
    """
    posting = features.active(cfg, "review-posting")
    if posting:
        keys = set(keys) | {
            k for k, rec in st["prs"].items() if post_mod.kept_unposted(rec.get("ours"))
        }
    n_new = 0
    for key in sorted(keys):
        rec = st["prs"][key]
        if rec.get("status") != "open":
            continue
        ours = snapshot.our_latest_review(cfg, rec["n"])
        if mirror.hold_ours(cfg, st, rec["n"], ours):
            # In a mirror (a monitor beside the live one that sends nothing) the post is the live monitor's of this
            # round, not the last post this review follows.
            continue
        old = rec.get("ours")
        # Posting off, and GitHub shows no post of ours since the review kept unposted: it is our last review.
        if (
            not posting
            and post_mod.kept_unposted(old)
            and (ours is None or ours["at"] < old["at"])
        ):
            continue
        if post_mod.taken_stands(old, ours):
            continue
        if ours and (not old or old.get("id") != ours["id"] or not old.get("head")):
            ours = snapshot.resolve_reviewed_head(cfg, store, rec["n"], ours)
            n_new += 1
        elif ours and old:
            ours["head"], ours["head_src"] = old.get("head"), old.get("head_src")
            if old.get("equivalent_heads"):
                ours["equivalent_heads"] = old["equivalent_heads"]
        rec["ours"] = ours
    return n_new


TIMING: dict[str, float] = {}


def timed(name: str, fn, *a, **kw):
    t0 = time.monotonic()
    try:
        return fn(*a, **kw)
    finally:
        TIMING[name] = TIMING.get(name, 0.0) + time.monotonic() - t0


def do_poll(cfg: dict, store: Store, st: dict, seed: bool = False) -> list[dict]:
    prs = timed("search", snapshot.list_prs, cfg)
    prs = timed(
        "scope", snapshot.scope_filter, cfg, prs, st.setdefault("scope_cache", {})
    )
    prs = timed(
        "rollup", snapshot.correct_rollups, cfg, prs, st.setdefault("rollup_cache", {})
    )
    before = {k: (v.get("updated_at"), v.get("status")) for k, v in st["prs"].items()}
    dev = timed("develop", snapshot.develop_snapshot, cfg, st)
    events = timed(
        "diff",
        ev_mod.diff_and_apply,
        st,
        {"prs": prs, "develop": dev},
        resolve_gone(cfg, st["scope_cache"]),
    )
    events = (
        ev_mod.take_pending(st) + events
    )  # pushes a review start found since the last tick (review.start_review)
    # Re-read our review only where something changed: any update (our own posts update the PR too) or an event.
    touched = {
        str(p["n"]) for p in prs if before.get(str(p["n"])) != (p["updated_at"], "open")
    }
    touched |= {
        str(e["pr"]) for e in events if e.get("pr") and str(e["pr"]) in st["prs"]
    }
    timed("ours", refresh_ours, cfg, store, st, touched)
    if seed:
        store.journal(
            "seed", f"seeded {len(prs)} open rocKE PRs; develop {dev['sha'][:10]}"
        )
        st["develop"]["pending_sanity"] = []
        return []
    ts = now()
    for e in events:
        store.journal("event", json.dumps(e, ensure_ascii=False))
    store.append_jsonl("events.jsonl", [dict(e, ts=ts) for e in events])
    return events


def refresh_threads(cfg: dict, st: dict) -> None:
    """Unresolved review threads (the develop ruleset requires them resolved), read for every PR heading for a merge:
    our latest verdict is 🟢/🟡. "Ready for your approval" needs them all resolved. Cached per PR update.
    """
    for rec in st["prs"].values():
        if (
            rec.get("status") != "open"
            or rec.get("draft")
            or (rec.get("ours") or {}).get("verdict") not in queue_mod.GOOD_VERDICTS
        ):
            rec.pop("threads", None)
            continue
        if (rec.get("threads") or {}).get("updated_at") != rec.get("updated_at"):
            rec["threads"] = {
                "updated_at": rec.get("updated_at"),
                "open": snapshot.unresolved_threads(cfg, rec["n"]),
            }


def refresh_required(cfg: dict, st: dict) -> None:
    """Required status checks per head (develop's ruleset), for PRs heading for a merge and develop fixes: GitHub's
    rollup covers only checks that reported, so a required check that never started would look green.
    Two GETs per PR, cached per head until every required check has passed (re-read on PR updates meanwhile).
    """
    rc = st.get("required_checks")
    if not rc or minutes_since(rc["at"]) > DAILY_MIN:
        st["required_checks"] = rc = {
            "at": now(),
            "names": snapshot.required_contexts(cfg),
        }
    fixes = st.get("develop_fixes") or {}
    for k, rec in st["prs"].items():
        if (
            rec.get("status") != "open"
            or rec.get("draft")
            or rec.get("ci_rollup") != "SUCCESS"
            or not (
                (rec.get("ours") or {}).get("verdict") in queue_mod.GOOD_VERDICTS
                or k in fixes
            )
        ):
            continue
        cur = rec.get("required") or {}
        done = cur.get("head") == rec["head"] and all(
            v == "success" for v in (cur.get("checks") or {}).values()
        )
        if done or (
            cur.get("head") == rec["head"]
            and cur.get("updated_at") == rec.get("updated_at")
        ):
            continue
        rec["required"] = {
            "head": rec["head"],
            "updated_at": rec.get("updated_at"),
            "checks": snapshot.required_status(cfg, rec["head"], rc["names"]),
        }


DAILY_MIN = (
    24 * 60
)  # the code-owner teams and the required checks are read again this often


@idle.timer("daily reads")
def _daily_due(cfg: dict, st: dict, store):
    """When the code-owner teams and the required checks are read again (DAILY_MIN)."""
    for key, label in (
        ("codeowners", "code owners"),
        ("required_checks", "required checks"),
    ):
        at = (st.get(key) or {}).get("at")
        if at:
            yield label, idle.plus(at, DAILY_MIN)


def do_queue(cfg: dict, st: dict) -> list[dict]:
    co = st.get("codeowners")
    if not co or minutes_since(co.get("tried") or co["at"]) > DAILY_MIN:
        # A refused read never fails the tick: the last list stands, or with none the teams are none, said once with
        # the fix; either way the next try is a day later.
        for line in snapshot.refresh_codeowners(cfg, st, now()):
            print(line)
    refresh_threads(cfg, st)
    queue_mod.refresh_develop_fixes(st, guard.incidents(Store()))
    refresh_required(cfg, st)
    q = queue_mod.build_queue(cfg, st, st["develop"]["sha"])
    st["queue"] = [
        dict(
            {
                k: r[k]
                for k in ("n", "position", "blockers", "disruption", "cleared", "stage")
                if k in r
            },
            pending=[
                k
                for k, v in sorted(r["gates"].items())
                if k in report.STATIC and v[0] == "pending"
            ],
            dyn_pending=[
                k for k in ("G5", "G6", "G7") if r["gates"][k][0] == "pending"
            ],
            gs={k: v[0] for k, v in sorted(r["gates"].items())},
        )
        for r in q
    ]
    return q


def approval_news(cfg: dict, st: dict, only: int | None = None) -> list[str]:
    """READY_FOR_APPROVAL lines for PRs that became ready for the user's approval (or changed head) since the last
    report; `only` limits it to one PR (after a post) without marking the others as reported.
    """
    appr = report.approval_candidates(cfg, st)
    seen = set(st.get("approvals_seen") or [])
    new = [
        a
        for a in appr
        if f"{a['n']}:{a['head'][:12]}" not in seen and only in (None, a["n"])
    ]
    keep = {f"{a['n']}:{a['head'][:12]}" for a in appr} if only is None else seen
    st["approvals_seen"] = sorted(keep | {f"{a['n']}:{a['head'][:12]}" for a in new})
    return [f"READY_FOR_APPROVAL {report.approval_line(a)}" for a in new]


def metrics_snapshot(st: dict, q: list[dict], plan: list[dict] | None = None) -> dict:
    d = st["develop"]
    open_recs = [r for r in st["prs"].values() if r.get("status") == "open"]
    plan = plan or []
    return {
        "ts": now(),
        "develop": (d.get("sha") or "")[:12],
        "review_backlog": sum(1 for p in plan if p["kind"] in report.REVIEW_KINDS),
        "follow_ups": sum(1 for p in plan if p["kind"] == "FOLLOW_UP"),
        "first_rounds": sum(1 for p in plan if p["kind"] == "FIRST_ROUND"),
        "develop_ci": (
            snapshot.ci_verdict(d["ci"].get(d.get("sha"), {}))
            if d.get("sha")
            else "unknown"
        ),
        "open": sum(1 for r in open_recs if not r["draft"]),
        "drafts": sum(1 for r in open_recs if r["draft"]),
        "conflicting": sum(1 for r in open_recs if r["mergeable"] == "CONFLICTING"),
        "ready": sum(1 for r in q if r.get("position")),
        "batches": len(st.get("batches") or []),
        "abi_binary_claims": len(
            ((st.get("abi_lanes") or {}).get("binary") or {}).get("groups", [])
        ),
    }


def insight_steps(cfg: dict, st: dict, store: Store) -> list[str]:
    """The history behind the weekly insights: the merge-conflict files every tick, CI
    times at most hourly. Both are advisory: a failure is reported in a line, never raised out of the tick.
    """
    lines = []
    for name, step in (("conflicts", conflicts.collect), ("ci", ci_times.refresh)):
        try:
            res = timed(name, step, cfg, st, store) or {}
        except (
            Exception
        ) as exc:  # noqa: BLE001 - advisory history: never fail a tick on it
            lines.append(f"({name} not recorded: {str(exc)[:200]})")
            continue
        if res.get("errors") or res.get("error"):
            lines.append(
                f"({name}: {'; '.join(res.get('errors') or [res.get('error')])[:200]})"
            )
    return lines


def role_step(cfg: dict, st: dict, store: Store) -> list[str]:
    """The optional model roles (rkmon_lib/roles.py), after the plan and the insights' history: the planner is asked
    for a proposal on the plan's facts when they have none yet, and the analyst for a reading of the per-PR insights
    when one is due; a proposal the plan set aside, or a reading refused, goes to the journal once. Advisory: a
    failure is a line, never raised out of the tick."""
    lines = []
    try:
        from rkmon_lib import mergeplan

        payload = mergeplan.PLANNER["payload"]
        if (
            payload
        ):  # the planner is on: how the plan used its proposal (rk roles shows it)
            r = roles.state(st, "planner")
            r["use"] = use = (st.get("merge_plan") or {}).get("planner") or ""
            if use.startswith(planner.SET_ASIDE) and r.get("aside") != payload["fp"]:
                r["aside"] = payload["fp"]
                store.journal("role", f"{use}; the toolkit's own plan stands")
        started = planner.start(cfg, st, store, payload)
        if started:
            store.journal("role", started)
        if roles.on(cfg, "analyst"):
            from rkmon_lib import insights

            stats = insights.pr_statistics(cfg, st, store)
            if analyst.due(cfg, st, stats):
                started = analyst.start(cfg, st, store, stats)
                if started:
                    store.journal("role", started)
            analyst.reading(
                st, store
            )  # a new reading is checked once; a refused one is journaled
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        lines.append(f"(the model roles: {str(exc)[:200]})")
    return lines


def request_step(cfg: dict, st: dict, store: Store) -> list[str]:
    """Merged or closed priorities leave the lists first (priority.prune), then other teams' priority requests from the
    published page (their clicks wait for the user's OK), read from the web host every tick. Advisory: a failure is reported in a line, never raised.
    """
    try:
        lines = priority.prune(st, store)
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        lines = [f"(merged priorities not cleared: {str(exc)[:200]})"]
    try:
        return lines + priority.pull(cfg, st, store, publish.pull_requests)
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        return lines + [f"(priority requests not read: {str(exc)[:200]})"]


def review_step(cfg: dict, st: dict, store: Store, plan: list[dict]) -> list[str]:
    """The reviews: the pipeline's advance (review.advance), then the stuck-verification alarms on the reviews it
    leaves waiting for their post (rkmon_lib/verify_stale.py): each review's VERIFY_STALE line, then the stall's
    critical line. An alarm's failure is a line, never raised.
    """
    lines = review.advance(cfg, st, store, plan)
    for alarm, what in (
        (verify_stale.lines, "stuck verifications"),
        (verify_stale.stalled_lines, "stalled verifications"),
    ):
        try:
            lines = lines + alarm(cfg, st)
        except Exception as exc:  # noqa: BLE001 - never fail a tick on it
            lines = lines + [f"({what} not checked: {str(exc)[:200]})"]
    return lines


def write_weekly(
    cfg: dict, st: dict, store: Store, label: str | None
) -> tuple[str, pathlib.Path]:
    start, end, lab = stats.week_bounds(label)
    text = stats.render(stats.compute(cfg, st, store, start, end), lab)
    out = pathlib.Path(paths.get(cfg, "weekly_dir"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rocke_weekly_{lab}.md"
    path.write_text(text, encoding="utf-8")
    return lab, path


def maybe_weekly_report(cfg: dict, st: dict, store: Store, t=None) -> str | None:
    """On the first tick after the configured weekday/hour (UTC), write last week's report once; a week missed while
    the monitor was down is written late, oldest first (stats.owed_weeks)."""
    import datetime as dt

    w = cfg.get("weekly_report")
    if not w:
        return None
    t = t or dt.datetime.now(dt.timezone.utc)
    due = not (
        t.weekday() < w["weekday"]
        or (t.weekday() == w["weekday"] and t.hour < w["hour_utc"])
    )
    out = []
    for week in stats.owed_weeks(st.get("weekly_done"), t, due):
        try:
            lab, path = write_weekly(cfg, st, store, week)
        except (gh.GhError, OSError) as exc:
            out.append(f"!! WEEKLY report {week} failed: {exc}")
            break
        st["weekly_done"] = lab
        store.journal("report", f"weekly report {lab}: {path}")
        line = f"WEEKLY REPORT {lab} ready: {path} (present it to the user)"
        try:  # the lessons-learned review of the agentic process
            start, end, _ = stats.week_bounds(lab)
            raw = lessons.collect(cfg, store, start, end, lab)
            line += f"\nLESSONS REVIEW {lab} due: {raw} (distil it into the skill's lessons.md, then present the review)"
        except OSError as exc:
            line += f"\n!! LESSONS raw material for {lab} failed: {exc}"
        out.append(line)
    return "\n".join(out) or None


def drafts_lines(cfg: dict, store: Store) -> list[str]:
    """`rk drafts`: the roles' bindings, the draft stage and its ceiling, the pool and the shadow drafts, the last
    drafts, the scaler's last decisions, and the drafters-vs-reference card."""
    st = store.load()
    stage = draft_stage.for_config(cfg)
    shadow = draft_stage.shadow_binding(cfg)
    out = bindings.describe(cfg) + [
        f"draft stage: {stage.name}; at most {bindings.parallel_max(cfg)} drafts at once"
        + ("" if stage.parallel else " (one at a time for this stage)")
        + (f"; shadow drafter: {shadow}, every live review" if shadow else "")
    ]
    out += [f"problem: {p}" for p in bindings.problems(cfg)]
    for k, e in sorted(draft_stage.pool(st).items(), key=lambda kv: int(kv[0])):
        out.append(
            f"pool: #{k} {e['kind']} at {e['head'][:10]} since {e['started']} "
            f"({jobs.status(store, e.get('job'))})"
        )
    for k, e in sorted(draft_stage.shadow_pool(st).items(), key=lambda kv: int(kv[0])):
        out.append(
            f"shadow: #{k} {e['kind']} at {e['head'][:10]} since {e['started']} "
            f"({jobs.status(store, e.get('job'))})"
        )
    for d in (st.get("drafts") or [])[-8:]:
        p = d.get("prior") or {}
        who = (
            f"{d.get('binding') or '-'}{' shadow' if d.get('role') == 'shadow_drafter' else ''} "
            f"{d.get('method') or draft_stage.OLD_METHOD}"
        )
        out.append(
            f"draft: #{d['pr']} {d['at']} {who} {d['status']}: {d['tokens']} tokens, {d['requests']} "
            f"requests, waited {d['waits_s']:.0f}s, {d['wall_s'] / 60:.0f} min"
            + (", late" if d.get("late") else "")
            + (
                f"; prior rows: {p.get('script', 0)} script, {p.get('model', 0)} model, {p.get('look', 0)} to "
                f"look at"
                if p
                else ""
            )
            + (f"; {d['evidence']}" if d.get("evidence") else "")
        )
    for r in scaler.recent(store, 5, kind="draft"):
        out.append(
            f"load: {r['at']} #{r['pr']} {r.get('role') or 'drafter'} "
            f"{'allow' if r['allow'] else 'deny'}: {r['reason']}"
        )
    return out + assessment.card_md(cfg, st, store)


@idle.timer("weekly report", again=True)
def _weekly_due(cfg: dict, st: dict, store):
    """When last week's report (and its lessons review) falls due: this week's configured weekday and hour (UTC), or
    next week's once it is written. Past due, every tick is full until the report is written. Only with the
    weekly-report feature on."""
    import datetime as dt

    w = cfg.get("weekly_report")
    if not w or not features.active(cfg, "weekly-report"):
        return
    t = dt.datetime.now(dt.timezone.utc)
    monday = dt.datetime.combine(
        t.date() - dt.timedelta(days=t.weekday()), dt.time(), dt.timezone.utc
    )
    due = monday + dt.timedelta(days=w["weekday"], hours=w["hour_utc"])
    if stats.owed_weeks(
        st.get("weekly_done"), t, False
    ):  # a week missed while down: due now
        due = monday
    elif st.get("weekly_done") == stats.week_bounds(None, t)[2]:
        due += dt.timedelta(days=7)
    yield "weekly report", due.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_dashboard(cfg, store, st, plan, q, board_md=None):
    path = report.dashboard_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.dashboard(
            cfg,
            st,
            plan,
            q,
            store.journal_tail(15),
            board_md,
            spend.card_md(spend.load_card(store)),
            store,
        )
    )


def paused_until() -> str:
    """How a pause ends, as the operator types it here (doctor.rk_command)."""
    from rkmon_lib.doctor import rk_command

    return f"paused until `{rk_command('resume')}`"


def cmd_tick(args, cfg, store):
    """`rk tick`: `run_tick`, once the GitHub request counter has the configuration's cap, the monitor on or off
    (rkmon_lib/gh_cap.py: a GH_CAP_OFF line first when the counter cannot write its record); then a GH_CAP_WAIT line
    when gh calls waited for the cap while the tick ran.
    """
    if args.dry_run:
        return tick_dry_run(args, cfg, store)
    t0, started = time.monotonic(), time.time()
    cap, state = registry.number(cfg, gh_cap.CONFIG_KEY), gh_cap.state_folder()
    for line in gh_cap.set_cap(state, cap) if state else []:
        print(line)
    try:
        return run_tick(args, cfg, store, t0)
    finally:
        waited = gh_cap.waits(state, started) if state else []
        if waited:
            print(f"GH_CAP_WAIT while this tick ran, {gh_cap.wait_text(waited, cap)}")


def run_tick(args, cfg, store, t0: float) -> int:
    """A tick: idle when nothing changed since the last full tick and no timer ran out (idle.py), else full. T0: when
    `rk tick` began (time.monotonic)."""
    with store.locked():
        st = store.load()
        if not st["enabled"] and not args.force:
            print(f"TICK {now()} monitor OFF: nothing done ({paused_until()})")
            return 0
        halt = stopped(st)
        if (
            halt
        ):  # rk stop or rk kill: nothing at all until rk start (--force overrides OFF only)
            from rkmon_lib.doctor import rk_command

            print(
                f"TICK {now()} monitor STOPPED since {halt.get('at')} (rk {halt.get('how') or 'stop'}): nothing done "
                f"(use `{rk_command('start')}`)"
            )
            return 0
        for line in tick_checks(st, store) + ghlogins.tick_lines(cfg, store):
            print(line)
        d = timed("fingerprint", idle.decide, cfg, st, store, full=args.full)
        if d.idle:
            for line in idle.idle_tick(cfg, st, store, time.monotonic() - t0):
                print(line)
            return 0
        rc = full_tick(cfg, store, st)
        idle.record_full(cfg, store, d, time.monotonic() - t0)
        return rc


def gh_cap_log(cfg: dict, store: Store) -> None:
    """The gh calls that waited for the GitHub request cap and the day log does not have yet (gh_cap.report), as one
    day log entry (kind wait), which the admin dashboard's log shows: a full tick's step, before the dashboards.
    """
    state = gh_cap.state_folder()
    told = gh_cap.report(state, time.time()) if state else []
    if told:
        first, last = (
            time.strftime("%H:%MZ", time.gmtime(w["at"])) for w in (told[0], told[-1])
        )
        when = f"at {first}" if first == last else f"between {first} and {last}"
        text = gh_cap.wait_text(told, registry.number(cfg, gh_cap.CONFIG_KEY))
        daylog.add(store, cfg, "wait", None, f"{text}, {when}")


def cmd_worker_tools(args, cfg: dict) -> int:
    """`rk worker-tools [--install]`: the jobs run the copies in paths.worker_tools, so they must be the running code's
    worker/ (rkmon_lib/deploy.py)."""
    from rkmon_lib import deploy
    from rkmon_lib.doctor import rk_command

    try:
        if args.install:
            for line in deploy.install_worker_tools(cfg):
                print(line)
        stale = deploy.stale_worker_tools(cfg)
    except (OSError, paths.PathsError) as exc:
        print(f"REFUSED: {exc}")
        return 4
    for name, why in stale:
        print(f"{why:<8} {name}")
    print(
        f"worker tools: {len(stale)} file(s) missing or not the running code's"
        + (
            f"; `{rk_command('worker-tools', '--install')}` installs them"
            if stale
            else ": up to date"
        )
    )
    return 1 if stale else 0


def _err(line: str) -> None:
    print(line, file=sys.stderr)


def cmd_pr_slot(args) -> int:
    """`rk pr-slot take|keep|free|status`: a job script's slot (rkmon_lib/prcode.py)."""
    if args.action == "status":
        print("\n".join(prcode.status()))
        return 0
    if args.pid is None:
        print(f"rk pr-slot {args.action}: give --pid PID")
        return 2
    if args.action == "take":
        runner, lines = prcode.take(
            args.pid, job=os.environ.get(jobs.JOB_ENV, ""), pr=args.pr or "", say=_err
        )
        for line in lines:  # stdout is the assignments
            _err(line)
        if not runner:
            return 1
        print("\n".join(prcode.assignments(runner)))
        return 0
    if args.action == "keep":
        missing = [
            f"--{k}" for k in ("pr", "head", "build", "record") if not getattr(args, k)
        ]
        if missing:
            print(f"rk pr-slot keep: give {' '.join(missing)}")
            return 2
        ok, lines = prcode.keep(args.pid, args.pr, args.head, args.build, args.record)
        print("\n".join(lines))
        return 0 if ok else 1
    print("\n".join(prcode.free(args.pid)))
    return 0


def cmd_pr_fetch(args) -> int:
    """`rk pr-fetch -- CMD [ARGS...]`: a check's `--fetch` as the monitor (prcode.fetch_for), a job script's `prf`."""
    argv = list(args.command)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        _err("rk pr-fetch: give the command after --: rk pr-fetch -- CMD [ARGS...]")
        return 2
    return prcode.fetch_for(argv)


def cmd_pr_tree(args) -> int:
    """`rk pr-tree DIR...`: each DIR made a folder for a tree PR code reads (prcode.tree), an agent's worktree before
    git adds it."""
    bad = [why for d in args.dirs if (why := prcode.tree(d))]
    for why in bad:
        _err(f"rk pr-tree: {why}")
    return 1 if bad else 0


def cmd_pr_run(args, store) -> int:
    """`rk pr-run [--slot NAME [--end]] [--env FILE] [-C DIR] [--out DIR] -- CMD [ARGS...]`: CMD as PR code in a PR
    runner's slot, on a host as the session user (rkmon_lib/prcode.py): the session's way to run PR code by hand.
    """
    argv = list(args.command)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if args.end and not args.slot:
        print("rk pr-run: --end ends named work: give --slot NAME")
        return 2
    if not argv and not args.end:
        print(
            "rk pr-run: give the command after --: rk pr-run [--slot NAME] -- CMD [ARGS...]"
        )
        return 2
    pr = ""
    if args.slot:
        m = slots.NAME_RE.match(args.slot)
        if not m:
            print(
                f"rk pr-run: {args.slot} is no work name: pr<N>-<word> (lower-case letters, digits, '.', '_' and '-')"
            )
            return 2
        pr = m.group(1)
    if not argv:
        print("\n".join(prcode.end_work(args.slot)))
        return 0
    env = None
    if args.env:
        try:
            env = prcode.read_env(args.env)
        except (OSError, ValueError) as exc:
            print(f"rk pr-run: the environment in {args.env} was not read ({exc})")
            return 2
    head = ""
    if args.slot:
        try:
            head = review_keep.head_in_flight(store.load(), int(pr))
        except ValueError:
            head = ""  # no state to read: the work stays until it ends
    return prcode.call(
        argv,
        env,
        args.where or os.environ.get("RKMON_CALLER_DIR") or os.getcwd(),
        args.out,
        say=_err,
        name=args.slot or "",
        pr=pr,
        head=head,
        end=args.end,
    )


def drain_line(st: dict) -> str | None:
    """The line for a drain in force (`rk drain`, `rk stop --drain`), or None."""
    d = draining(st)
    if d and not drain_stale(d):
        return (
            f"DRAINING since {d.get('since')} ({d.get('why') or 'rk stop --drain'}): nothing new starts until it "
            "ends; the work in flight finishes"
        )
    return None


def tick_checks(st: dict, store: Store) -> list[str]:
    """First in every tick, idle or full, under its lock: jobs that a tick or command started but never saved are
    killed (jobs.reap_orphans: the steps start their work again, tracked), then a drain is said while it holds, or
    cleared once it is stale (its process died, or it ran past its time). Saves the state when it clears a drain;
    the orphans are read against the state as saved, so they go first."""
    lines = jobs.reap_orphans(store, st)
    d = draining(st)
    why = drain_stale(d) if d else ""
    if why:
        st.pop("draining", None)
        store.save(st)
        store.journal(
            "mode",
            f"drain cleared ({d.get('why') or 'rk stop --drain'}, since {d.get('since')}): {why}",
        )
        lines.append(
            f"!! DRAIN_CLEARED ({d.get('why') or 'rk stop --drain'}, since {d.get('since')}): {why}; new work "
            "starts again"
        )
    elif d:
        lines.append(drain_line(st))
    return lines


def tick_dry_run(args, cfg, store) -> int:
    """`rk tick --dry-run`: which tick the next one would be, and why; it reads what a tick reads and writes nothing."""
    st = store.load()
    if not st["enabled"] and not args.force:
        print(f"DRY RUN: the monitor is OFF: a tick does nothing ({paused_until()})")
        return 0
    t0 = time.monotonic()
    d = idle.decide(cfg, st, store, full=args.full)
    print(
        f"DRY RUN ({time.monotonic() - t0:.1f} s): "
        + (idle.idle_line(st) if d.idle else "a full tick: " + "; ".join(d.reasons))
    )
    print(
        "parts: "
        + ", ".join(
            f"{k} {v:.1f} s" for k, v in sorted(d.secs.items(), key=lambda kv: -kv[1])
        )
    )
    for line in idle.status_lines(cfg, st):
        print(line)
    return 0


def full_tick(cfg: dict, store: Store, st: dict) -> int:
    """The full tick, under the lock: poll, events, drift, develop issues, queue, plan, guard, merge queue, reviews,
    cleanup, GPU jobs, requests, follow-up scans, history, the model roles, overlaps, the board and the dashboards.
    """
    events = do_poll(cfg, store, st)
    try:  # what the merge plan said about each PR that just merged, before the queue plans again (scoreboard.py)
        scoreboard.note_landings(st, events, cfg)
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        print(f"(scoreboard landings not noted: {str(exc)[:160]})")
    # A finished drift check first, so the queue's gates read its results (drift.py).
    try:
        drift_lines = (
            timed("drift", drift.collect, cfg, st, store)
            if features.on(cfg, "merge-plan")
            else []
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a drift problem never fails the tick; it shows
        drift_lines = [f"(drift results not read: {str(exc)[:160]})"]
    try:
        breakage_lines = (
            timed("breakages", breakage.poll, cfg, st)
            if features.on(cfg, "develop-checks")
            else []
        )
    except gh.GhError as exc:  # the issue list is advisory: never fail a tick on it
        breakage_lines = [f"(develop issues not read: {exc})"]
    # The optional model roles' answers first (rkmon_lib/roles.py), so the plan below can use the planner's.
    role_lines = (
        roles.poll(cfg, st, store) if any(roles.on(cfg, r) for r in roles.ROLES) else []
    )
    # Each moved head is marked merge-only or not before the plan reads it: a merge-only follow-up skips the gap, and
    # its checks decide whether anything is posted. A deployment that makes no reviews carries rebase-only and
    # merge-only pushes for G1 itself (queue.carry_heads).
    try:
        if features.on(cfg, "review-drafts"):
            merge_lines = timed("mergeonly", review.mark_merges, cfg, st)
        elif not features.on(cfg, "review-drafts") and features.on(cfg, "merge-plan"):
            merge_lines = timed("carry", queue_mod.carry_heads, cfg, st)
        else:
            merge_lines = []
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a marking problem never fails the tick: the follow-up just stays due
        merge_lines = [f"(merge-only marking not run: {str(exc)[:160]})"]
    q = timed("queue", do_queue, cfg, st)
    # A top priority ready now behind waves that wait on others, whatever the mode: the user's call.
    ready_lines = mergeplan.ready_lines(st) if features.on(cfg, "merge-plan") else []
    try:  # after the queue: its ABI scan has fetched every open PR's head, and the develop fixes are current
        fix_lines = (
            timed("fixes", breakage.fix_sweep, cfg, st)
            if features.on(cfg, "develop-checks")
            else []
        )
        if not features.active(
            cfg, "fix-links"
        ):  # off: the candidates are listed for the operator, nothing acts
            fix_lines = breakage.listed_only(fix_lines)
    except (gh.GhError, gitops.GitError) as exc:
        fix_lines = [f"(fix sweep failed: {exc})"]
    plan = plan_mod.make_plan(
        cfg, st, events, q, plan_mod.last_review_posts(store.ledger())
    )
    for p in plan:
        if p["critical"]:
            store.journal("CRITICAL", f"{p['kind']}: {p['text']}")
    seen = {tuple(x) for x in st.get("abi_renumber_logged", [])}
    new = [
        p
        for p in plan
        if p["kind"] == "ABI_RENUMBER"
        and (p["pr"], st["prs"][str(p["pr"])]["head"]) not in seen
    ]
    store.append_jsonl(
        "events.jsonl",
        [{"ts": now(), "kind": "ABI_RENUMBER_NEEDED", "pr": p["pr"]} for p in new],
    )
    st["abi_renumber_logged"] = sorted(
        seen | {(p["pr"], st["prs"][str(p["pr"])]["head"]) for p in new}
    )
    store.append_jsonl("metrics.jsonl", [metrics_snapshot(st, q, plan)])
    report_line = (
        maybe_weekly_report(cfg, st, store)
        if features.active(cfg, "weekly-report")
        else None
    )
    # The develop guard: develop is checked after every merge, whatever the mode (live files what it finds).
    guard_lines = (
        timed("guard", guard.advance, cfg, st, store, events)
        if features.active(cfg, "develop-checks")
        else []
    )
    queue_lines = (
        timed("merge", merge.advance, cfg, st, store, q, events)
        if features.on(cfg, "merge-plan")
        else []
    )
    review_lines = (
        timed("review", review_step, cfg, st, store, plan)
        if features.on(cfg, "review-drafts")
        else []
    )
    if features.on(
        cfg, "merge-plan"
    ):  # after review.advance: a worker it just started holds drift off (prefill)
        try:
            drift_lines += drift.start(cfg, st, store, q)
        except Exception as exc:  # noqa: BLE001 - as above
            drift_lines.append(f"(drift check not started: {str(exc)[:160]})")
    cleanup_lines = (
        timed(
            "cleanup",
            cleanup.sweep,
            cfg,
            st,
            store,
            pathlib.Path(paths.get(cfg, "working_dir")),
        )
        if features.on(cfg, "review-drafts") or features.on(cfg, "merge-plan")
        else []
    )
    try:  # the remote GPU jobs (rkmon_lib/gpu.py): queued, never waited for; a cluster that is down fails no tick
        gpu_lines = (
            timed("gpu", gpu.tick, cfg, st, store)
            if features.active(cfg, gpu.FEATURE)
            else []
        )
    except Exception as exc:  # noqa: BLE001
        gpu_lines = [f"(gpu: the GPU jobs were not polled: {str(exc)[:200]})"]
    approval_lines = approval_news(cfg, st)
    request_lines = (
        timed("requests", request_step, cfg, st, store)
        if features.on(cfg, "priority-requests")
        else []
    )
    # Every reviewer's open items on each PR that merged, once; never fails a tick.
    followup_lines = (
        timed("followups", followups.tick, cfg, st, store, events)
        if features.active(cfg, "follow-up-tickets")
        else []
    )
    daylog.from_events(store, cfg, st, events)
    daylog.from_lines(
        store,
        cfg,
        st,
        breakage_lines
        + guard_lines
        + queue_lines
        + merge_lines
        + review_lines
        + cleanup_lines,
    )
    gh_cap_log(cfg, store)
    if features.on(cfg, "admin-dashboard"):  # the dashboards' figures
        try:  # merges per day for the board and the page: advisory, never fail a tick on it
            timed("merges", merges.refresh, cfg, st)
        except gh.GhError as exc:
            print(f"(merges per day not refreshed: {exc})")
        for line in insight_steps(cfg, st, store):
            print(line)
    role_lines += timed("roles", role_step, cfg, st, store)
    if features.active(cfg, "overlap-management"):  # advisory, never fails
        try:
            for err in timed("overlaps", overlaps.refresh, cfg, st, store)["errors"][
                :2
            ]:
                print(f"(overlaps: {err})")
        except Exception as exc:  # noqa: BLE001
            print(f"(overlaps not refreshed: {str(exc)[:200]})")
        try:  # PRs a recorded owner decision asks to drop their copy, without our overlap notice yet
            for line in overlaps.notices_due(
                st, store, hint=lambda n: first_pass.notice_hint(cfg, n, "overlap")
            ):
                print(line)
        except Exception as exc:  # noqa: BLE001
            print(f"(overlap notices not listed: {str(exc)[:200]})")
    feature_lines = features.tick_reports(
        cfg, st
    )  # before the save: each is printed once
    spend_lines = timed(
        "spend", spend.tick, cfg, st, store
    )  # the card hourly, yesterday's files once (advisory)
    spend_lines += timed(
        "scoreboard", scoreboard.tick, cfg, st, store
    )  # the admin page's scoreboard (advisory)
    try:  # the state folder's logs stay bounded (rkmon_lib/retention.py)
        spend_lines += timed("retention", retention.rotate, cfg, store)
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        spend_lines.append(f"(retention: {str(exc)[:160]})")
    st["last_tick"] = now()
    st["last_plan"] = plan
    bd = report.board(cfg, st, store, plan, q)
    try:  # the Needs-you decisions, copied once to the Teams channel (features.alerts)
        needs_lines = needs_teams.tick(cfg, st, store, bd)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the Teams copy never fails a tick; its items stay unsent
        needs_lines = [  # never the webhook
            f"(needs-you Teams message not sent: {post_mod.redact(str(exc))[:200]})"
        ]
    seen = st.get("board") or {}
    show = (
        seen.get("fp") != bd["fp"]
        or minutes_since(seen.get("shown") or "2000-01-01T00:00:00Z")
        >= report.BOARD_EVERY_MIN
    )
    if show:
        st["board"] = {"fp": bd["fp"], "shown": now()}
    store.save(st)
    if features.on(
        cfg, "admin-dashboard"
    ):  # with the admin dashboard off, none is written
        write_dashboard(cfg, store, st, plan, q, bd["md"])
        try:
            webdash.write(cfg, st, store, plan, q, bd)
        except Exception as exc:  # noqa: BLE001 - the HTML page must never break a tick
            print(f"(html dashboard not written: {exc})")
    print(report.tick_summary(cfg, st, events, plan, q))
    # A feature that is off, or on but not set up, prints none of its action lines; one that lacks something is
    # reported once (rkmon_lib/features.py).
    keep = lambda lines: features.filter_lines(cfg, lines)  # noqa: E731
    for line in feature_lines:
        print(line)
    for line in keep(breakage_lines + fix_lines):
        print(line)
    for line in keep(guard_lines):
        print(
            (
                "!! CRITICAL "
                if line.startswith(("INCIDENT_OPEN", "CULPRIT", "REVERT_OFFER"))
                else ""
            )
            + line
        )
    for line in keep(queue_lines):
        print(
            (
                "!! "
                if line.startswith(
                    ("DISARM_NEEDED", "AUTOMERGE_RISK", "AUTOMERGE_JUMP")
                )
                else ""
            )
            + line
        )
    for line in keep(ready_lines):
        print("!! " + line)
    for line in keep(
        followup_lines
        + drift_lines
        + merge_lines
        + review_lines
        + gpu_lines
        + cleanup_lines
        + approval_lines
        + request_lines
        + role_lines
        + spend_lines
        + needs_lines
    ):
        print(line)
    if report_line:
        print("\n".join(keep(report_line.splitlines())))
    if show:
        print(
            f"BOARD ({'changed' if seen.get('fp') != bd['fp'] else 'hourly'}): show it to the user"
        )
        print(bd["md"])
    else:
        print(f"BOARD unchanged since {seen.get('shown')}")
    print("time: " + " ".join(f"{k} {v:.0f}s" for k, v in TIMING.items()))
    return (
        3
        if any(p["critical"] for p in plan)
        or any(
            line.startswith(("INCIDENT_OPEN", "CULPRIT", "REVERT_OFFER"))
            for line in keep(guard_lines)
        )
        else 0
    )


def cmd_init(args, cfg, store):
    with store.locked():
        if store.state_path.exists() and not args.force:
            if args.if_new:
                from rkmon_lib.doctor import rk_command, wrapper_name

                print(
                    f"state exists: kept (`{rk_command('init', '--force')}` re-seeds it)"
                )
                st = store.load()
                if getattr(args, "on", False) and not st.get("enabled"):
                    if st.get("paused"):
                        # rkmon start never undoes a pause: rk pause holds until someone runs rk resume
                        print(
                            f"paused (rk pause, {st['paused'].get('at')}): the monitor stays off; "
                            f"`{rk_command('resume')}` ends the pause (or `{wrapper_name()} start --on`)"
                        )
                    else:  # seeded, never switched on (a start cut short before its rk resume)
                        st["enabled"] = True
                        store.save(st)
                        store.journal("mode", "monitor ON (rk init --if-new --on)")
                        print("monitor ON")
                return 0
            print("state exists; use --force to re-seed")
            return 1
        from rkmon_lib.store import empty_state

        st = empty_state()
        if getattr(args, "on", False):
            st["enabled"] = True
            store.journal("mode", "monitor ON (rk init --on)")
        do_poll(cfg, store, st, seed=True)
        q = do_queue(cfg, st)
        plan = plan_mod.make_plan(
            cfg, st, [], q, plan_mod.last_review_posts(store.ledger())
        )
        st["last_plan"] = plan
        store.save(st)
        write_dashboard(cfg, store, st, plan, q)
        print(report.tick_summary(cfg, st, [], plan, q))
        return 0


def cmd_status(args, cfg, store):
    st = store.load()
    for line in control.status_lines(cfg, st, store):
        print(line)
    plan = st.get("last_plan", [])
    if args.queue:  # recompute (git work) and refresh the dashboard
        q = queue_mod.build_queue(cfg, st, st["develop"]["sha"])
        write_dashboard(cfg, store, st, plan, q)
    else:  # the rows stored by the last tick
        q = st.get("queue", [])
    print(report.tick_summary(cfg, st, [], plan, q).replace("TICK", "STATUS", 1))
    print(f"last tick: {st.get('last_tick')}; journal: {store.journal_path}")
    for line in ghlogins.cached_lines(cfg, store):
        print(line)
    return 0


def cmd_flag(args, cfg, store):
    with store.locked(wait=True):
        st = store.load()
        if args.cmd in ("pause", "resume"):
            st["enabled"] = args.cmd == "resume"
            # A pause is recorded, so `rk init --if-new --on` (rkmon start) never undoes it.
            if args.cmd == "pause":
                st["paused"] = {"at": now()}
            else:
                st.pop("paused", None)
            store.journal("mode", f"monitor {'ON' if st['enabled'] else 'OFF'}")
        elif args.cmd == "freeze":
            st["frozen"] = {"since": now(), "reason": " ".join(args.reason)}
            store.journal("mode", f"FROZEN: {st['frozen']['reason']}")
        elif args.cmd == "unfreeze":
            st["frozen"] = None
            store.journal("mode", "unfrozen")
        elif args.cmd in ("clear", "unclear"):
            s = set(st.get("cleared", []))
            (s.add if args.cmd == "clear" else s.discard)(args.n)
            st["cleared"] = sorted(s)
            store.journal(
                "mode",
                f"{args.cmd} #{args.n} for auto-merge (cleared: {st['cleared']})",
            )
        elif args.cmd in ("prioritize", "unprioritize"):
            # Float a PR to the top: review P0 with the short settle, merge tier 0 and analysis first, as for a
            # develop fix. Only on the user's instruction; the dashboard's "Request PR priority" box does the same
            # through `rk serve`.
            priority.set_priority(
                st,
                store,
                args.n,
                " ".join(getattr(args, "why", [])),
                on=args.cmd == "prioritize",
                by=config.operator_login(cfg),
            )
        store.save(st)
        print(
            f"enabled={st['enabled']} frozen={bool(st['frozen'])} cleared={st['cleared']} "
            f"prioritized={sorted(int(k) for k in st.get('user_priority') or {})}"
        )
    return 0


def heavy_start_refusal(cfg, st, store, n: int = 0, now_: bool = False) -> str | None:
    """Why `rk gate`, `rk sanity` or `rk bisect` may not start its heavy job now (None when it may): the monitor is
    stopped or drains, in store.halted's words (a drain's words say that nothing new starts), or, unless --now, every
    heavy slot the scaler allows is busy or the RAM disk has no room for it, as the tick's own starts check (merge.py,
    guard.py).
    """
    why = control.halted(st)
    if why:
        return why
    if now_:
        return None
    busy = jobs.heavy_busy(cfg, st, store)
    if busy:
        return f"all parallel builds are busy ({busy['id']} among them): try again later, or --now to start it beside them"
    ok, why = scaler.room(cfg, st, store, "heavy", n)
    if not ok:
        return f"{why}: try again later, or --now"
    return None


def cmd_post_checked(args, cfg, store):
    """`rk post N FILE --premises SPEC` (rkmon_lib/premises.py), and `--dry-run`. The checks read GitHub and git without
    the state lock, so a tick is never skipped for them; the post takes it, as `rk post`.
    `--again` (a re-review at a head we posted on or, while review posting is off, at the head of our last review)
    goes through to premises.check and to `rk post` or the keep.
    """
    if args.premises and os.environ.get(gh.BROKERED) == "1":
        # As the write user, the premises' git and AFTER lines would run what the session wrote: never.
        print(
            "REFUSED: rk post --premises runs as the session user; only its post goes through the write broker"
        )
        return 4
    if not args.premises:  # a dry run of a plain post: its guards only
        try:
            post_mod.preflight(
                cfg, store.load(), store, args.n, args.file, again=args.again
            )
        except (post_mod.PostRefused, gh.GhError) as exc:
            print(f"REFUSED: {exc}")
            return 4
        print(
            f"dry run: rk post's guards pass; stopped before the post ({post_mod.write_login()})"
        )
        return 0
    lines, code = premises.post_checked(
        cfg,
        store,
        args.n,
        args.file,
        args.premises,
        again=args.again,
        dry=args.dry_run,
        news=approval_news,
    )
    print("\n".join(lines))
    return code


# Commands that work before the configuration is complete: they report what is missing instead of refusing.
NO_CONFIG = (
    "doctor",
    "config",
    "env",
    "paths",
    "home",
    "selftest",
    "features",
    "tick-filter",
    "mirror-report",
    "guide",
    "probe-check",
    "probe-port",
    "applied",
)
# The emergency brake: it reads only what names the folders (config.folders), never the internal state or the
# delivered layer, so that neither keeps the monitor from stopping.
BRAKE = ("stop", "kill", "undrain", "stopped")
# Run with what names the folders alone too: rk adopt fills this deployment's folders, also before its first start,
# while there is no configuration yet (docker/rkmon adopt; the first start sets the work volume up around them).
FOLDERS_ONLY = (*BRAKE, "adopt")
# Loaded without the check (config.load(check=False)): a configuration with a problem runs them, and one that cannot
# be read refuses them; the entry point runs no tick while tick-min refuses (docker/entrypoint.sh), and its tick loop
# waits between its ticks with tick-wait, a wait that a problem in the configuration must not end at once.
UNCHECKED = ("tick-min", "tick-wait", "worker-tools", "pr-slot", "pr-run")
# Commands that run for a feature (rkmon_lib/features.py): off, they do nothing and exit 0; on but not set up, they exit
# 5 and name what is missing. Never a traceback. The sends among them come from rkmon_lib/outward.py: each one that runs
# whole for one feature (with review posting off, a post keeps a draft instead; a notice's arguments pick its features,
# and one with two runs while either is on, or for the one an issue command's arguments pick: feature_gate).
FEATURE_OF = {
    "hold": "holds",
    "overlap-decide": "overlap-management",
    "ticket-draft": "found-issue-tickets",
    **{
        c: gpu.FEATURE
        for c in (
            "gpu-run",
            "gpu-poll",
            "gpu-status",
            "gpu-queue",
            "gpu-cancel",
            "gpu-check",
        )
    },
    **{
        r.name: r.features[0]
        for r in outward_mod.TABLE
        if r.gate == outward_mod.OWN and len(r.features) == 1 and not r.drafts
    },
}
NOTICE_FEATURE = (
    post_mod.NOTICE_FEATURE
)  # one home: post.py's outward_gate reads it too


def feature_gate(args, cfg: dict) -> tuple[int, str] | None:
    """None when the command may act, else (exit code, message)."""
    name = FEATURE_OF.get(args.cmd)
    if args.cmd == "needs-you-teams" and args.dry_run:
        return None  # the dry run reads the state, says whether the feature is on, and writes nothing
    if (
        args.cmd == "disarm"
    ):  # protective: a PR the state still records as armed is disarmed with the feature off
        try:
            armed = ((Store().load().get("queue_state") or {}).get("armed") or {}).get(
                "pr"
            )
        except (OSError, ValueError):
            armed = None
        if armed == args.n:
            return None
    if args.cmd == "notice":
        for name in NOTICE_FEATURE.get(args.kind) or ():
            gate = features.refusal(cfg, name)
            if gate:
                return gate
        return None
    elif args.cmd == "automerge-off":
        gate = features.refusal(cfg, name)
        if gate:
            return gate
        if not getattr(args, "user", None) and features.guard_mode(cfg) == "alert":
            return features.EXIT_OFF, (
                f"OFF: {features.guard_alert_words(cfg)}: auto-merge was left on."
            )
    elif len(getattr(outward_mod.row(args.cmd), "features", ())) > 1:
        # Two features: it runs while either is on (rk teams-test: both alerts send to the channel its card checks), or
        # for the one an issue command's arguments pick, the issue's type.
        kinds, kind = set(outward_mod.row(args.cmd).features), None
        if args.cmd == "issue-relabel":
            kind = args.to
        elif args.cmd == "issue-file":
            try:
                kind = breakage.parse_file(args.file, cfg)["kind"]
            except (OSError, ValueError):
                pass  # the command itself names what is wrong with the file
        if kind:
            kinds = {
                "stability-issues" if kind == breakage.STABILITY else "incident-issues"
            }
        if not any(features.on(cfg, k) for k in kinds):
            k = sorted(kinds)[0]
            return features.refusal(cfg, k)
        return next(
            (
                r
                for k in sorted(kinds)
                if features.on(cfg, k)
                for r in [features.refusal(cfg, k)]
                if r
            ),
            None,
        )
    return features.refusal(cfg, name) if name else None


def atlassian_words(args) -> list[str]:
    """`rk atlassian`'s words as writer.atlassian_call reads them: rk's own options, then the CLI's words."""
    from rkmon_lib import writer

    return (
        ([f"--site={args.site}"] if args.site else [])
        + (["--json"] if args.json else [])
        + (
            [f"{writer.TIMEOUT_OPTION}={args.timeout_s}"]
            if args.timeout_s is not None
            else []
        )
        + args.words
    )


def write_gate(args, cfg: dict, st: dict) -> str:
    """The line that stops the send ARGS asks for, "" when it may go out: for a send the writer makes for another
    process (rkmon_lib/outward.py: gate GATED), a mirror (a monitor beside the live one that sends nothing), the one
    outward gate (post.outward_gate: the send's feature off, or in prototype, refuses it, or keeps it, and its held
    line says so) and a stop (control.refusal). rk asks them in the process that sends, before the send
    (in the container, the writer), whoever asked for the send. An `rk atlassian` that writes nothing to JIRA passes;
    its writes are follow-up tickets' (a found issue's ticket is `rk ticket-file`). The other sends' own commands gate
    them (gate OWN).
    """
    from rkmon_lib import writer

    row = outward_mod.row(args.cmd)
    if not row or row.gate != outward_mod.GATED:
        return ""
    needs = row.features
    if args.cmd == writer.TEAMS_SEND:
        needs = post_mod.TEAMS_FEATURES[args.type]
    elif args.cmd == writer.ATLASSIAN:
        try:
            writes = writer.atlassian_call(atlassian_words(args)).writes
        except ValueError:
            return ""  # off the list: `rk atlassian` refuses it (writer.refusal), and runs nothing
        needs = needs if writes else ()
    if not needs:
        return ""
    why = mirror.guard(cfg, args.cmd)
    if why:
        return f"REFUSED: {why}"
    try:
        post_mod.outward_gate(
            cfg,
            st,
            row.what,
            needs,
            post_mod.writer_item(
                cfg,
                args.cmd,
                atlassian_words(args) if args.cmd == writer.ATLASSIAN else (),
            ),
            store=Store(),
        )
    except post_mod.Held as held:
        return str(held)  # kept as an unsent item, not refused
    except post_mod.PostRefused as exc:
        return f"REFUSED: {exc}"
    why = control.refusal(args.cmd, st)
    return f"REFUSED: {why}" if why else ""


def cmd_writer_half(args, cfg: dict) -> int:
    """The sends whose secret the writer alone reads in the container, each its half there (rkmon_lib/writer.py), and
    the send itself on a host: a team chat message (`rk teams-send`, the message on stdin: post.teams_send), the team
    dashboard's page (`rk web-push FILE`: publish.push_page), the web host's requests file (`rk web-requests`:
    publish.read_requests), and one of the atlassian CLI's listed commands (`rk atlassian`: atlassian.listed). The web
    host's login, host and path come from the configuration only as publish.target_refusal lets them.
    """
    from rkmon_lib import atlassian, publish, writer

    if args.cmd in ("web-push", "web-requests"):
        why = publish.target_refusal(
            cfg, "dest" if args.cmd == "web-push" else "requests"
        )
        if why:
            print(writer.refusal(why))
            return 4
    if args.cmd == "teams-send":
        try:
            code = post_mod.teams_send(
                cfg, args.type, sys.stdin.read(post_mod.TEAMS_MESSAGE_MAX_BYTES + 1)
            )
        except post_mod.PostRefused as exc:
            print(f"REFUSED: {exc}")
            return 4
        print(f"HTTP {code}")
        return 0
    if args.cmd == "web-push":
        why = publish.push_page(cfg, args.file)
        print(f"{publish.NOT_PUSHED}{why}" if why else "pushed")
        return 1 if why else 0
    if args.cmd == "web-requests":
        text, why = publish.read_requests(cfg)
        if why:
            print(f"{publish.NOT_READ}{why}")
            return 1
        sys.stdout.write(text)
        return 0
    try:
        p = atlassian.listed(cfg, atlassian_words(args))
    except ValueError as exc:
        print(writer.refusal(str(exc)))
        return 4
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    return p.returncode


def cmd_tickets(args, cfg: dict, store) -> int:
    """The proposed tickets: the session proposes them (rk ticket-draft), the user decides on the admin dashboard's
    card; ticket-file and ticket-decline are the user's command-line fallback."""
    from rkmon_lib import tickets

    me = config.operator_login(cfg) or "the operator"
    if args.cmd == "tickets":  # read-only
        print("\n".join(tickets.lines(cfg, store.load())))
        return 0
    if args.cmd == "ticket-file":  # under the write lock already (control.WRITES)
        ok, msg, _extra = tickets.approve(
            cfg, store, args.id, by=me, lock_writes=False, again=args.again
        )
        print(msg if ok else f"REFUSED: {msg}")
        return 0 if ok else 4
    if args.cmd == "ticket-decline":
        with store.locked(wait=True):
            st = store.load()
            ok, msg = tickets.decline(
                cfg, st, store, args.id, by=me, why=" ".join(args.why)
            )
            store.save(st)
        print(msg if ok else f"REFUSED: {msg}")
        return 0 if ok else 4
    try:
        text = (
            sys.stdin.read()
            if args.file == "-"
            else pathlib.Path(args.file).read_text(encoding="utf-8")
        )
        draft = tickets.parse(text)
    except (OSError, UnicodeDecodeError, tickets.DraftError) as exc:
        print(f"REFUSED: the proposed ticket: {exc}")
        return 4
    with store.locked(wait=True):
        st = store.load()
        ok, msg, _tid = tickets.add(cfg, st, store, draft)
        store.save(st)
    print(msg if ok else f"REFUSED: {msg}")
    return 0 if ok else 4


def cmd_jira_account(args, cfg: dict) -> int:
    """`rk jira-account add LOGIN ACCOUNT | list`: the JIRA account that the follow-up tickets of a GitHub
    login's items are assigned to, in the internal state (rkmon_lib/state_store.py). add records the pairs the
    configuration runs with (config.json's too) and LOGIN's, which replaces one recorded already; list prints each
    pair, a login a line. A login or an account the state's schema refuses is refused, and nothing is written; the
    fix is this command again, with a login and an account the schema takes. An internal state that cannot be read, or
    cannot be written (state_store.unwritable), is refused the same way, with its own fix: the reason, then the fix, a
    line each (applied.cannot_run).
    """
    from rkmon_lib import checks, doctor, state_store

    accounts = config.jira_accounts(cfg)
    if args.action == "list":
        pairs = [f"{login} {account}" for login, account in sorted(accounts.items())]
        print("\n".join(pairs) or "no JIRA account is recorded")
        return 0
    again = checks.Steps(
        (
            doctor.rk_command("jira-account", "add", "LOGIN", "ACCOUNT"),
            "again, with LOGIN a GitHub login (letters, digits and hyphens) and ACCOUNT the JIRA account (no spaces)",
        )
    )
    pairs, home = dict(accounts, **{args.login: args.account}), config.home()
    try:
        state_store.set(home, "jira.accounts", pairs, fix=again)
    except state_store.StateError as exc:
        return config_refused(exc)
    except OSError as exc:
        return config_refused(state_store.unwritable(home, exc))
    was = accounts.get(args.login)
    print(
        f"{args.login}: JIRA account {args.account} recorded"
        + (f", in place of {was}" if was not in (None, args.account) else "")
    )
    return 0


def cmd_features(cfg: dict) -> int:
    """`rk features`: the setup summary (rkmon_lib/summary.py): every feature, then every value the features use;
    secrets only as set or missing."""
    from rkmon_lib import doctor, summary

    print(
        summary.render(
            cfg,
            doctor.claude_code(cfg),
            doctor.gh_status(cfg),
            control.state_summary(cfg),
        ),
        end="",
    )
    return 0


def features_unloaded(exc: config.ConfigError) -> int:
    """`rk features` while the configuration cannot be loaded (EXC), in applied.cannot_run's words, why and then the
    fix: with no configuration yet, every optional feature is off, and the summary of the defaults follows, exit 0;
    one that cannot be read runs nothing, exit 1."""
    # a call-time import: only a configuration that cannot be loaded reads it
    from rkmon_lib import applied

    if not isinstance(exc, config.NoConfiguration):
        for line in applied.cannot_run(exc, "Nothing runs"):
            print(line)
        return 1
    for line in applied.cannot_run(exc, "Every optional feature is off"):
        print(line)
    print()
    return cmd_features(config.normalize(config.load_defaults()))


def cmd_gpu(args, cfg: dict, store) -> int:
    """The remote GPU test commands (rkmon_lib/gpu.py). Read-only ones first; the others hold the state lock."""
    if args.cmd == "gpu-check":
        lines = gpu.check(cfg)
        print("\n".join(lines))
        return 4 if any(x.startswith("FAIL") for x in lines) else 0
    if args.cmd == "gpu-queue":
        try:
            print("\n".join(gpu.queue_lines(cfg)))
        except gpu.GpuError as exc:
            print(f"REFUSED: {exc}")
            return 4
        return 0
    if args.cmd == "gpu-status":
        print("\n".join(gpu.status_lines(cfg, store.load())))
        return 0
    with store.locked(wait=True):
        st = store.load()
        try:
            if args.cmd == "gpu-run":
                if not args.develop and args.n is None:
                    raise gpu.GpuError(
                        "name a PR (rk gpu-run N), or use --develop with --case"
                    )
                lines = gpu.run_by_hand(
                    cfg,
                    st,
                    store,
                    None if args.develop else args.n,
                    args.arch,
                    args.case,
                    develop_only=args.develop,
                    same_node=args.same_node,
                    node=args.node or "",
                )
            elif args.cmd == "gpu-poll":
                lines = gpu.tick(cfg, st, store)
            elif args.cmd == "gpu-cancel":
                lines = gpu.cancel(cfg, st, store, args.target)
            else:  # gpu-row: an outward write (it holds the write lock: control.WRITES)
                lines = gpu.update_row(cfg, st, store, args.n, args.grade)
        except (gpu.GpuError, post_mod.PostRefused) as exc:
            store.save(st)
            print(f"REFUSED: {exc}")
            return 4
        store.save(st)
    print("\n".join(lines) or "nothing to do")
    return 0


def _driver_acts(text: str, out=print) -> None:
    """After the driver's last advance, printed whole: the lines the orchestrator acts on (`rk tick-filter --driver`),
    one DRIVER_EXIT summary line, the others, and the REVIEW_READY lines last, so the output's tail is all it needs to
    read."""
    acts = tickfilter.driver_lines(text)
    if acts:
        out("---\n" + "\n".join(acts))


def cmd_review_driver(
    cfg: dict, store, sleep=time.sleep, interval: int = 60, rounds: int = 120, out=print
) -> int:
    """The continuous review driver (it works through the review backlog without waiting for ticks). One at a time: it
    holds state/review_driver.lock. Every `interval` seconds it advances the parallel reviews (up to
    features.review_automation.parallel_reviews drafts at once) and prints what changed. It exits, which wakes the
    orchestrator, as soon as a draft is ready for verification (REVIEW_READY, every one of that advance printed), only
    reviews too large for the autopilot are left (REVIEW_MANUAL), nothing can start or run now, the monitor is stopped,
    the last tick is over review.TICK_STALE_MIN minutes old (TICK_STALE: the tick loop may be down), or `rounds` rounds
    pass (2 hours: a stuck job; the orchestrator checks rk review-status). The drafts in flight keep running when it
    exits: they are detached jobs. A configuration that cannot be loaded stops it at once, exit 4, in
    applied.cannot_run's words: why, then the fix the error names."""
    import fcntl

    lock = open(store.dir / "review_driver.lock", "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        out("DRIVER: another review driver is already running")
        return 0
    try:
        last = ""
        for _ in range(rounds):
            try:
                cfg = load_config()
            except config.ConfigError as exc:
                # a call-time import: only a refusal reads it
                from rkmon_lib import applied

                for line in applied.cannot_run(
                    exc, "DRIVER: the configuration cannot run"
                ):
                    out(line)
                return 4
            with store.locked(wait=True):
                st = store.load()
                if stopped(st):
                    out(f"DRIVER: {control.halted(st)}")
                    return 0
                stale = review.tick_stale(
                    st
                )  # the state may be out of date: the orchestrator hears of it
                if stale:
                    out(stale)
                    _driver_acts(stale, out)
                    return 0
                lines = review.advance(cfg, st, store, st.get("last_plan", []))
                daylog.from_lines(store, cfg, st, lines)
                store.save(st)
            if any(
                line.startswith(("REVIEW_READY", "REVIEW_MANUAL")) for line in lines
            ):
                out("\n".join(lines))
                _driver_acts("\n".join(lines), out)
                return 0
            if not lines:
                out(
                    "DRIVER: nothing to start now (backlog empty, heads still settling, follow-ups waiting for the gap, "
                    "in-depth reviews waiting for the operator's approval, or the monitor draining); the ticks take "
                    "over"
                )
                return 0
            head1 = lines[0][:160]
            if head1 != last:
                out(f"{now()[11:16]}Z {head1}")
                last = head1
            sleep(interval)
        out(f"DRIVER: still running after {rounds * interval // 60} min")
        st = store.load()
        for p in review.pipes(st):
            out(
                f"#{p['pr']} {p.get('kind')} stage={p.get('stage')} worker {jobs.status(store, p.get('worker'))}, "
                f"checks {jobs.status(store, p.get('checks'))}"
            )
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def cmd_guide(root: pathlib.Path = ROOT, out=None, which: str = "user") -> int:
    """`rk guide [build|deploy]`: the operator's quick guide, or with build the image builder's, with deploy the
    deployer's (rkmon_lib/image.py: USER_GUIDE, BUILD_GUIDE, DEPLOY_GUIDE), at the top of the code that runs (in the
    container, the image's or the active release's: rk runs that code). An install without it says so (exit 1).
    """
    from rkmon_lib import image

    out = out or sys.stdout
    guides = {"build": image.BUILD_GUIDE, "deploy": image.DEPLOY_GUIDE}
    guide = root / guides.get(which, image.USER_GUIDE)
    if not guide.is_file():
        out.write(
            f"no guide in this install ({guide} is missing): the package's {guide.name} and README.md have it\n"
        )
        return 1
    try:
        out.write(guide.read_text(encoding="utf-8"))
        out.flush()
    except (
        BrokenPipeError
    ):  # `rk guide | head`: the reader stopped early, which is no failure
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


def cmd_secret_set(
    name: str, cfg: dict, stdin=None, out=None, remove: bool = False
) -> int:
    """`rk secret-set NAME [--remove]`: the secret on stdin into its file (secrets_registry.path: its place in its
    keeper's secrets folder, which the load derives), written as a private file (private_files.write: permissions 600,
    the folder 700, never through a link). NAME is any secret the secrets registry lists for this configuration, a
    model binding's key among them, or a login kept in a file of its own (secrets_registry.held_files: the claude login,
    which `rkmon login claude` sets through here); a login kept in a folder, or a name it does not list, is a usage
    error (2), with why (secrets_registry.name_refusal: an older name of an entry says the entry's name and the command
    that sets it). The value is never printed, nor its length: an empty one, or one that is not the secret's shape
    (secrets_registry.check_value), is refused and nothing is written, and so is a file that breaks a private file's
    rule, with the edit that names another secrets folder (secrets_registry.retry). In the container a secret the
    writer keeps is set by the writer only, as the host asks it (`rkmon secret NAME`: writer.FROM_HOST); the monitor's
    rk refuses one (exit 4), with that command (writer.set_from_host).

    With REMOVE, the secret's file at its place (secrets_registry.path) is removed, a link there itself, never what it
    points to: that place is its own in the secrets folder (secrets_registry.home), as the load keeps no other place a
    configuration gives (rkmon_lib/derived.py)."""
    from rkmon_lib import checks, logins, private_files, writer, secrets_registry as sr

    stdin = stdin or sys.stdin
    out = out or sys.stdout
    entry = sr.entry(cfg, name)
    if entry is None or sr.held_files(entry) is not None:
        names = ", ".join(e.name for e in sr.entries(cfg) if sr.held_files(e) is None)
        print(
            f"rk secret-set: {sr.name_refusal(cfg, name)} (the names it takes: {names})",
            file=sys.stderr,
        )
        return 2
    if not logins.here(entry):  # the writer's, and this rk is the monitor's
        out.write(f"REFUSED: {writer.set_from_host(name)}; nothing changed\n")
        return 4
    try:
        dest = sr.path(cfg, entry)
    except paths.PathsError as exc:
        out.write(f"REFUSED: {name}: nothing changed: {exc}\n")
        return 4
    if remove:
        return _secret_remove(entry, pathlib.Path(dest), out)
    value = stdin.read()
    row = sr.check_value(entry, value)
    if row.status != checks.Status.PASS:
        out.write(f"REFUSED: {name}: nothing written:\n{checks.render([row])}\n")
        return 4
    try:
        dest = private_files.write(
            dest,
            value.strip() + "\n",
            paths.PACKAGE,
            replace=True,
            retry=sr.retry(),
        )
    except private_files.Refused as exc:
        out.write(f"REFUSED: {name}: nothing written:\n{exc}\n")
        return 4
    out.write(f"{name}: set ({dest}, mode 600; not shown)\n")
    return 0


def _secret_remove(entry, p: pathlib.Path, out) -> int:
    """`rk secret-set NAME --remove`: the secret's file P, at its place, removed (cmd_secret_set's docstring)."""
    if p.is_dir() and not p.is_symlink():
        out.write(f"REFUSED: {entry.name}: {p} is a folder; nothing removed\n")
        return 4
    if not (p.is_symlink() or p.exists()):
        out.write(f"{entry.name}: not set ({p}); nothing removed\n")
        return 0
    try:
        p.unlink()
    except OSError as exc:
        out.write(f"ERROR: {entry.name}: {p}: {exc.strerror}; nothing removed\n")
        return 1
    out.write(f"{entry.name}: removed ({p})\n")
    return 0


def adopt_targets(cfg: dict) -> dict:
    """This deployment's folder of each part an adopted state fills (rkmon_lib/adopt.py PARTS)."""
    from rkmon_lib import adopt, paths as paths_mod

    out = {"state": str(config.state_dir()), "home": str(config.home())}
    keys = {
        "probes": "probes",
        "lessons": "lessons",
        "cache": "cache",
        "posts": "posts",
        "posted": "posted_dir",
        "daily": "daily_dir",
        "spend": "spend_dir",
        "workers": "workers",
        "repo": "repo",
        "scratch": "scratch",
        "tmp": "tmp",
        "outputs": "outputs_dir",
        "root": "working_dir",
    }
    for label, _how, _key, _under in adopt.PARTS:
        if label in keys:
            try:
                out[label] = paths_mod.get(cfg, keys[label])
            except (KeyError, paths_mod.PathsError):
                pass
    return out


def cmd_adopt(args, cfg: dict, store) -> int:
    """`rk adopt plan|apply --stage DIR`: docker/rkmon adopt's two steps in the container (rkmon_lib/adopt.py). An apply
    that stops (adopt.Stopped) prints what it did, then why, where that leaves the state, and the fix: REFUSED, exit
    4, when nothing changed (adopt.NOTHING), else ERROR, exit 1; an OSError before it (the state's lock, the plan's
    read of the stage) is the same REFUSED, nothing changed.
    """
    from rkmon_lib import adopt, outcome, paths as paths_mod

    def stopped(exc) -> int:
        # a call-time import: doctor is a large module, read only on a failure
        from rkmon_lib.doctor import wrapper_name

        if exc.done:
            print("\n".join(exc.done))
        word = "REFUSED" if exc.where == adopt.NOTHING else "ERROR"
        text = f"{exc}; {exc.where}"
        return outcome.finish(word, text, fix=exc.error.fix(wrapper_name()))

    targets = adopt_targets(cfg)
    try:
        if args.action == "plan":
            print("\n".join(adopt.plan(args.stage, targets)))
            print("PLAN only: nothing changed")
            return 0
        running = [
            e["kind"]
            for e in control.processes(cfg, store)
            if e["kind"] in ("tick", "driver", "orchestrator", "review-advance")
        ]
        if running:
            print(
                f"REFUSED: the monitor runs here ({', '.join(sorted(set(running)))}): stop it first "
                "(docker/rkmon stop)"
            )
            return 4
        with store.locked(wait=True):
            lines = adopt.apply(args.stage, targets, paths_mod.get(cfg, "working_dir"))
        print("\n".join(lines))
        return 0
    except adopt.Stopped as exc:
        return stopped(exc)
    except adopt.AdoptError as exc:
        print(f"REFUSED: {exc}")
        return 4
    except OSError as exc:
        error = adopt.stopped_by(exc, args.stage)
        return stopped(adopt.Stopped(error, (), adopt.NOTHING))


def job_log_files(cfg: dict, store, n: int) -> list[str]:
    """`rk job-log --pr N`: the files PR N changes (from its merge base), for the never-compiled list."""
    if not n:
        return []
    rec = (store.load().get("prs") or {}).get(str(n))
    if not rec:
        return []
    try:
        return review._changed_files(cfg, dict(rec, n=n))
    except Exception:  # a read that fails leaves the list out, never the errors
        return []


def cmd_werror(args, cfg: dict) -> int:
    """`rk werror N`: tools/tidy_check.py on PR N's merged worktree (else its head's) and its native build: the review's
    shared build while its slot is kept for it, else the build folder's. As PR code, in a slot of its own (the build's
    compile commands are what PR code wrote), after its --fetch (git in the slot fetches nothing).
    """
    import os

    from rkmon_lib import probes

    p = paths.resolve(cfg)
    work = pathlib.Path(p["working_dir"])
    tree = pathlib.Path(args.tree) if args.tree else work / f"wt{args.n}m"
    if not args.tree and not tree.is_dir():
        tree = work / f"wt{args.n}"
    if not tree.is_dir():
        print(
            f"no tree for #{args.n}: {tree} (rk review-prep or the checks make it; or give --tree)"
        )
        return 2
    build = pathlib.Path(args.build) if args.build else None
    shared = review.shared_build_record(cfg, args.n)
    if (
        not build
        and shared[1:]
        and os.path.normpath(shared[1]) == os.path.normpath(tree)
    ):
        if slots.kept_build(shared[0]):
            build = pathlib.Path(shared[0]) / "core"
        else:
            print(
                f"the review's shared build {shared[0]} is gone (its slot was freed): give --build"
            )
    build = build or pathlib.Path(p["build"]) / tree.name / "core"
    base = args.base or probes.devbase_rev(tree, "develop")
    argv = [
        sys.executable,
        "-I",
        "-B",
        str(ROOT / "tools" / "tidy_check.py"),
        "--tree",
        str(tree),
        "--build",
        str(build),
        "--rocke-path",
        paths.repository_folder(cfg),
        "--repo",
        p["repo"],
        "--base",
        base,
        "--llvm-bin",
        p.get("rocm_llvm_bin") or "",
        "--budget",
        str(args.budget),
    ]
    if args.dev_tree:
        argv += ["--dev-tree", args.dev_tree]
    if args.files:
        argv += ["--files", *args.files]
    env = prcode.with_trees(
        {**os.environ, "TMPDIR": p["tmp"]},
        p["repo"],
        str(tree),
        *([args.dev_tree] if args.dev_tree else []),
    )
    prcode.fetch_for(argv, env)
    r = prcode.run(argv, env=env, timeout=args.budget + 120)
    print(r.stdout.rstrip("\n") or r.stderr.rstrip("\n"))
    last = (r.stdout.strip().splitlines() or [""])[-1]
    return 1 if last.startswith("TIDY fail") else 0


def cmd_probes(args) -> int:
    """rk probe-check and rk probe-port (rkmon_lib/probes.py). The probes run with the job environment the
    configuration gives (its venv, ROCm's LLVM tools, the builds, the repository), when there is one.
    """
    import os

    from rkmon_lib import probes

    folder = pathlib.Path(args.dir)
    if args.cmd == "probe-port":
        lines, code = probes.port(
            folder, write=args.write, names=args.root or probes.LIVE_ROOT_NAMES
        )
        print("\n".join(lines))
        return code
    if not args.tree:
        lines, code = (
            probes.static(folder) if folder.is_dir() else ([f"no folder: {folder}"], 2)
        )
        print("\n".join(lines))
        return code
    env, dirs, scratch = dict(os.environ), [], None
    try:
        cfg = config.load(check=False)
        env.update({k: v for k, v in config.env(cfg).items() if v})
        dirs = paths.sparse_dirs(cfg) if paths.repository_folder(cfg) else []
        scratch = paths.get(cfg, "scratch") or None
    except (config.ConfigError, paths.PathsError, KeyError):
        pass
    if scratch:
        try:
            pathlib.Path(scratch).mkdir(parents=True, exist_ok=True)
        except OSError:
            scratch = None
    lines, code = probes.check(
        folder,
        pathlib.Path(args.tree),
        only=args.probe,
        devbase=args.devbase,
        devbase_tree=pathlib.Path(args.devbase_tree) if args.devbase_tree else None,
        simfix=pathlib.Path(args.simfix) if args.simfix else None,
        simfix_tree=pathlib.Path(args.simfix_tree) if args.simfix_tree else None,
        base=args.base,
        dirs=dirs,
        env=env,
        scratch=pathlib.Path(scratch) if scratch else None,
    )
    print("\n".join(lines))
    return code


def cmd_stop(args, cfg: dict, store) -> int:
    """`rk stop [--drain [--timeout-minutes N]]` (control.stop), CFG naming the folders alone (config.folders). The
    drain advances the work in flight, which takes the whole configuration, for the GitHub calls too
    (gh.configure); when it cannot be loaded, no review in flight can finish, as no tick or review driver runs either:
    NOT DRAINED, why and the fix, and the stop goes at once."""
    whole = None
    if args.drain:
        try:
            whole = config.load(check=False)
        except config.ConfigError as exc:
            # a call-time import: only a drain that cannot load the configuration reads it
            from rkmon_lib import applied

            for line in applied.cannot_run(exc, drain.NOT_DRAINED):
                print(line)
        else:
            gh.configure(whole)
    return control.stop(
        cfg if whole is None else whole,
        store,
        drain=whole is not None,
        timeout_minutes=args.timeout_minutes,
    )


def cmd_host_key(args, cfg: dict) -> int:
    """`rk host-key NAME [--json | --record]`: the own keys of the host whose ssh key the secret NAME holds
    (rkmon_lib/host_keys.py), the only keys ssh trusts the host by. `rkmon secret NAME FILE` runs it once the key is
    set, and `rkmon login` for a key set already, in the walk's one-off container on the deployment's network, where
    ssh reaches the host. Without --record: the host's keys as ssh-keyscan reads them here, their fingerprints, and
    whether the state records them (host_keys.read), in words or, with --json, as JSON; exit 0 when a key is read,
    else 4, saying why. With --record:
    the known_hosts lines on stdin, those you accepted, recorded as the host's own keys in the internal state, in the
    place of those recorded before; REFUSED, exit 4, and nothing recorded, for lines that are not keys of the host
    your settings name (host_keys.refusal), or a state that refuses them, with its fix.
    """
    from rkmon_lib import registry, state_store, wrapping

    h = host_keys.host(args.name)
    name, port = host_keys.target(cfg, h)
    if args.record:
        lines = [x.strip() for x in sys.stdin.read().splitlines() if x.strip()]
        why = host_keys.refusal(lines, name, port)
        if why:
            print(f"REFUSED: {h.words}'s own keys are not recorded: {why}")
            return 4
        try:
            host_keys.record(config.home(), h, lines)
        except state_store.StateError as exc:
            return config_refused(exc)
        where = state_store.file_key(registry.key(h.state))  # the state's own name
        print(
            f"{h.words}'s own keys recorded ({wrapping.counted(len(lines), 'key')}, {where}): ssh trusts {name} by "
            "them alone"
        )
        return 0
    got = host_keys.read(cfg, h)
    if args.json:
        print(json.dumps(got))
    elif got["lines"]:
        said = (
            "the state records them"
            if got["recorded"]
            else "the state does not record them"
        )
        if got["others"]:
            said += f"; it records {wrapping.counted(got['others'], 'other key')} of it"
        print(f"{h.words}'s own keys ({name}), as ssh-keyscan reads them here: {said}")
        print("".join(f"  {f}\n" for f in got["fingerprints"] or got["lines"]), end="")
    else:
        print(f"no key of {h.words} was read: {got['why']}")
    return 0 if got["lines"] else 4


def config_refused(exc) -> int:
    """REFUSED, exit 4: the configuration cannot be loaded (EXC, a config.ConfigError): applied.refuse, why and the
    fix the error names."""
    from rkmon_lib import applied  # a call-time import: only a refusal reads it

    return applied.refuse(exc)


def cmd_noconfig(args) -> int:
    if args.cmd == "applied":  # the applied record; --load loads the configuration
        from rkmon_lib import applied

        return applied.main(load=args.load, as_json=args.json)
    if args.cmd == "guide":  # a file of the code that runs: no configuration, no state
        return cmd_guide(which=args.which)
    if args.cmd == "mirror-report":  # reads the two sides' records only
        from rkmon_lib import mirror_report

        return mirror_report.main(args)
    if args.cmd == "tick-filter":  # reads stdin only: no configuration, no state
        lines = (
            tickfilter.driver_lines(sys.stdin.read())
            if args.driver
            else tickfilter.actionable(
                sys.stdin.read(),
                args.rc,
                pathlib.Path(args.seen) if args.seen else None,
            )
        )
        if lines:
            print("\n".join(lines))
        return 0 if lines else 1
    if args.cmd == "selftest":
        from rkmon_lib import selftest

        if args.jobs < 1:
            print("REFUSED: --jobs must be 1 or more")
            return 4
        # A child process runs the suite (or the tests named), with a home folder, a TMPDIR and a monitor home of its
        # own (--jobs N: one per module, N at a time): a test that leaves anything there, or that writes in the
        # package's folder, fails the run.
        try:
            r = selftest.run(ROOT, names=args.tests, jobs=args.jobs)
        except selftest.CannotSplit as exc:
            print(f"REFUSED: {exc}; run rk selftest without --jobs")
            return 4
        return 0 if r.returncode == 0 else 1
    if args.cmd == "features":
        # Read-only. --runs: whether the deployment runs this part (docker/entrypoint.sh); none while there is no
        # configuration or it cannot be read (exit 4), which rk doctor shows with its fix. Without --runs, the setup
        # summary says why and the fix (features_unloaded).
        try:
            cfg = config.load(check=False)
        except config.ConfigError as exc:
            return config_refused(exc) if args.runs else features_unloaded(exc)
        if args.runs:
            return 0 if features.components(cfg).get(args.runs) else 1
        return cmd_features(cfg)
    if args.cmd == "home":
        where, why = config.home_and_why()
        print(
            f"{where}\n  found by: {why}\n  config: {where / 'config.json'}"
            f"{'' if (where / 'config.json').exists() else ' (missing)'}\n  pointer file: "
            f"{config.pointer_path()}"
        )
        return 0
    if args.cmd == "doctor":
        from rkmon_lib import doctor

        return doctor.main(fresh=args.fresh, fails=args.fails, as_json=args.json)
    if (
        args.cmd == "config"
    ):  # read-only: the running configuration, a configuration that cannot run included
        from rkmon_lib import config_show

        return config_show.main(source=args.source)
    if args.cmd in (
        "probe-check",
        "probe-port",
    ):  # a probe folder and trees; any configuration, or none
        return cmd_probes(args)
    try:
        cfg = config.load(check=False)
    except config.ConfigError as exc:
        return config_refused(exc)
    if args.cmd == "paths":
        for key, value in paths.resolve(cfg).items():
            print(f"{key:<14} {value or '(not set)'}")
        return 0
    env = config.env(cfg)
    print(json.dumps(env, indent=1) if args.json else config.export_line(cfg))
    return 0


def cmd_review_queue(args, cfg, store):
    """`rk review-queue [N ...] [--again] [--drop]`: list the review queue, add PRs at its end, or drop them (the state
    and the journal only). Of a PR added whose in-depth review waits for the operator's approval, it says so, with the
    review's size, agents and cost (review.approval_wait)."""
    if not args.ns:
        if args.again or args.drop:
            print(
                "REFUSED: --again and --drop name PRs: rk review-queue N [N ...] --again | --drop"
            )
            return 4
        print("\n".join(review.queue_lines(cfg, store.load(), store)))
        return 0
    if args.again and args.drop:
        print("REFUSED: --again does not go with --drop")
        return 4
    with store.locked(wait=True):
        st = store.load()
        if args.drop:
            line = "; ".join(
                (
                    f"#{n} dropped"
                    if review.unqueue(st, store, n, "dropped by hand")
                    else f"#{n} was not queued"
                )
                for n in dict.fromkeys(args.ns)
            )
        else:
            try:
                line = review.queue_add(st, store, args.ns, args.again)
            except ValueError as exc:
                print(f"REFUSED: {exc}")
                return 4
        store.save(st)
    asks = [
        review.approval_wait(cfg, st, store, n)
        for n in dict.fromkeys(args.ns)
        if not args.drop and review.needs_approval(cfg, st, n)
    ]
    print("\n".join([line] + asks + review.queue_lines(cfg, st, store)))
    return 0


# Commands and options rk does not take that a user may type ("stop --wait": the option --wait of rk stop; "settings
# export": a command of two words), each with the rk command to run instead and what it does: the usage error names
# that command as the operator types it here.
INSTEAD = {
    "off": (("pause",), "pauses the monitor"),
    "on": (("resume",), "ends a pause"),
    "stop --wait": (
        ("stop", "--drain", "--timeout-minutes", "N"),
        "waits at most N minutes for the work in flight",
    ),
    "settings export": (("config", "show"), "prints the running configuration"),
}
# The rk command the host's commands replace: in the container, root alone changes the code, at the
# host's request (rkmon_lib/releases.py), so its usage error names the host's commands; with --rollback, the way back.
ON_THE_HOST = "update"
# The command the deploy path replaces in rk, as in rkmon: with any of its options, its usage error names
# what replaces it on the host, in the words of its row of rkmon's GONE (rkmon_lib/commands.py).
SETUP = "setup"


def instead(argv: list[str]) -> str | None:
    """The usage error for a command or option of INSTEAD in `argv` (rk's arguments), or for ON_THE_HOST or SETUP, or
    None."""
    if not argv:
        return None
    if argv[0] == ON_THE_HOST:
        return f'unknown command "{argv[0]}": {_on_the_host("--rollback" in argv[1:])}'
    if argv[0] == SETUP:
        return f'unknown command "{argv[0]}": {_set_up_on_the_host()}'
    if argv[0] in INSTEAD:
        return f'unknown command "{argv[0]}": {_run_instead(argv[0])}'
    command = " ".join(argv[:2])
    if len(argv) > 1 and not argv[1].startswith("-") and command in INSTEAD:
        return f'unknown command "{command}": {_run_instead(command)}'
    for arg in argv[1:]:
        option = arg.split("=", 1)[0]
        if f"{argv[0]} {option}" in INSTEAD:
            return f'{argv[0]}: unknown option "{option}": {_run_instead(f"{argv[0]} {option}")}'
    return None


def _run_instead(key: str) -> str:
    """INSTEAD's command for KEY as the operator types it here (doctor.rk_command), and what it does."""
    from rkmon_lib.doctor import rk_command

    args, does = INSTEAD[key]
    return f"{rk_command(*args)} {does}"


def _on_the_host(back: bool) -> str:
    """The host's commands in ON_THE_HOST's place, as the operator types them (releases.host_commands), and what they
    do: with BACK (its --rollback), the way back to the code kept."""
    from rkmon_lib.releases import host_commands

    does = "goes back to the code kept" if back else "install a newer package's code"
    return f"on the host, {host_commands(back)} {does}"


def _set_up_on_the_host() -> str:
    """What replaces SETUP, on the host: its row of commands.GONE, with the host's program as the operator types it
    (doctor.wrapper_name)."""
    from rkmon_lib.commands import GONE
    from rkmon_lib.doctor import wrapper_name

    (row,) = [g for g in GONE if g.words == SETUP]
    return f"on the host, {row.replacement(wrapper_name())}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if (
        argv[:1] == ["--home"] and len(argv) > 1
    ):  # rk --home DIR COMMAND: this run's home
        os.environ["RKMON_HOME"] = argv[1]
        argv = argv[2:]
    why = instead(argv)
    if why:
        print(f"rk: {why}", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(
        prog="rk",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--if-new",
        action="store_true",
        help="only when there is no state yet; else say so and exit 0",
    )
    p.add_argument(
        "--on",
        action="store_true",
        help="the seeded state starts ON; kept state is never switched",
    )
    p = sub.add_parser(
        "rules-tick",
        help="a tick run by fixed rules, for a deployment with no Claude session",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="empty the list of lines waiting for a person",
    )
    p = sub.add_parser("tick")
    p.add_argument("--force", action="store_true", help="run while the monitor is OFF")
    p.add_argument(
        "--full",
        action="store_true",
        help="a full tick, whatever the change fingerprint says",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="say whether the tick would be idle or full, and why",
    )
    sub.add_parser("tick-now")
    p = sub.add_parser("tick-wait")
    p.add_argument(
        "--loop",
        type=int,
        required=True,
        metavar="PID",
        help="the pid of the shell tick loop that waits",
    )
    p.add_argument(
        "seconds", type=float, metavar="SECONDS", help="the longest it waits"
    )
    p = sub.add_parser("tick-filter")
    p.add_argument("--rc", type=int, default=None, help="the tick's exit code")
    p.add_argument(
        "--seen",
        default=None,
        metavar="FILE",
        help="standing lines already reported, which stay quiet; a full tick's standing lines replace it",
    )
    p.add_argument(
        "--driver",
        action="store_true",
        help="the review driver's exit: its lines that need the orchestrator, a summary line, REVIEW_READY last",
    )
    sub.add_parser("loop-config")
    p = sub.add_parser("worker-tools")
    p.add_argument(
        "--install",
        action="store_true",
        help="copy the running code's worker/ into paths.worker_tools",
    )
    p = sub.add_parser("pr-slot")
    p.add_argument("action", choices=("take", "keep", "free", "status"))
    p.add_argument("--pid", type=int, default=None, help="the job script's process")
    p.add_argument("--pr", default="", help="the PR the job is for")
    p.add_argument("--head", default="", help="keep: the head of the review")
    p.add_argument(
        "--build", default="", help="keep: the slot's build, the review's shared build"
    )
    p.add_argument(
        "--record", default="", help="keep: the file that names the shared build"
    )
    p = sub.add_parser("pr-run")
    p.add_argument(
        "--slot",
        default="",
        metavar="NAME",
        help="named work, pr<N>-<word>: its slot stays for its next commands",
    )
    p.add_argument(
        "--end", action="store_true", help="end the named work after CMD (no CMD: now)"
    )
    p.add_argument(
        "--env",
        default="",
        metavar="FILE",
        help="the environment, NUL-separated NAME=VALUE",
    )
    p.add_argument("-C", dest="where", default="", metavar="DIR", help="where CMD runs")
    p.add_argument(
        "--out",
        default="",
        metavar="DIR",
        help="copy the slot's out here after, in new files",
    )
    p.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD")
    p = sub.add_parser("pr-fetch")
    p.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD")
    p = sub.add_parser("pr-tree")
    p.add_argument("dirs", nargs="+", metavar="DIR")
    sub.add_parser("loop-status")
    sub.add_parser("poll")
    sub.add_parser("queue")
    p = sub.add_parser("status")
    p.add_argument(
        "--queue", action="store_true", help="recompute the queue (git work)"
    )
    p = sub.add_parser("gh-logins")
    p.add_argument("--side", choices=("read", "write", "session"), default=None)
    p.add_argument("--json", action="store_true", help="with --side: its probe as JSON")
    sub.add_parser(
        "pause", help="pause: the tick decides nothing and starts nothing new"
    )
    sub.add_parser("resume", help="end a pause")
    sub.add_parser("unfreeze")
    stop_parser = sub.add_parser(
        "stop", help="stop the running pieces now (rk start resumes)"
    )
    stop_parser.add_argument(
        "--drain",
        action="store_true",
        help="start nothing new; let the work in flight finish first, as rk drain does",
    )
    stop_parser.add_argument(
        "--timeout-minutes",
        type=float,
        metavar="N",
        help="with --drain: the most minutes to wait (default update.drain_timeout_min)",
    )
    sub.add_parser("start", help="start again after rk stop or rk kill")
    p = sub.add_parser("kill", help="kill every process the monitor started, at once")
    p.add_argument(
        "--wait",
        type=float,
        default=30,
        metavar="S",
        help="seconds to wait for a GitHub write in flight first (default 30)",
    )
    p = sub.add_parser(
        "drain", help="nothing new starts while the work in flight finishes"
    )
    p.add_argument(
        "--timeout-minutes",
        type=float,
        metavar="N",
        help="the most minutes to wait (default update.drain_timeout_min)",
    )
    sub.add_parser("undrain", help="lift the flag rk drain set")
    sub.add_parser(
        "review-driver", help="the continuous review driver (tools/review_driver.py)"
    )
    p = sub.add_parser("freeze")
    p.add_argument("reason", nargs="+")
    for name in ("clear", "unclear"):
        p = sub.add_parser(name)
        p.add_argument("n", type=int)
    p = sub.add_parser("prioritize")
    p.add_argument("n", type=int)
    p.add_argument("why", nargs="*")
    p = sub.add_parser("unprioritize")
    p.add_argument("n", type=int)
    p = sub.add_parser("serve")
    p.add_argument(
        "--port", type=int, default=None, help="default: config admin_dashboard.port"
    )
    p.add_argument(
        "--bind",
        default=None,
        help=f"default: config admin_dashboard.bind ({container_layout.LOOPBACK}; "
        f"{container_layout.LISTEN_ADDRESS} in a container)",
    )
    p.add_argument(
        "--public-port",
        type=int,
        default=None,
        help="the host port a container's dashboard is published on (default: features.dashboards.admin.port)",
    )
    sub.add_parser("priority-requests")
    p = sub.add_parser("priority-decline")
    p.add_argument("n", type=int)
    p = sub.add_parser("ticket-draft")
    p.add_argument(
        "file",
        help="the proposed ticket (a --- block, then the description), or - for stdin",
    )
    sub.add_parser("tickets")
    p = sub.add_parser("ticket-file")
    p.add_argument("id")
    p.add_argument(
        "--again",
        action="store_true",
        help="file a proposed ticket whose filing was cut off (check JIRA first)",
    )
    p = sub.add_parser("ticket-decline")
    p.add_argument("id")
    p.add_argument("why", nargs="*")
    jira_account_parser = sub.add_parser(
        "jira-account", help="the JIRA account a GitHub login's follow-up tickets go to"
    )
    jira_account_parser.add_argument("action", choices=("add", "list"))
    jira_account_parser.add_argument("login", nargs="?", default="")
    jira_account_parser.add_argument("account", nargs="?", default="")
    p = sub.add_parser("ledger")
    p.add_argument("n", type=int, nargs="?", default=20)
    sub.add_parser("abi")
    sub.add_parser("batches")
    sub.add_parser("merge-plan")
    p = sub.add_parser("abi-ok")
    p.add_argument("n", type=int)
    p = sub.add_parser("group")
    p.add_argument("ns", type=int, nargs="+")
    p = sub.add_parser("stats")
    p.add_argument("--week", default=None)
    p = sub.add_parser("spend")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--day", default=None, help="YYYY-MM-DD (UTC)")
    g.add_argument("--week", default=None, help="YYYY-Www (ISO, UTC)")
    g.add_argument("--since", default=None, help="YYYY-MM-DD: the days from it on")
    p.add_argument("--by", default="part", choices=spend.BY)
    p.add_argument(
        "--experiments",
        action="store_true",
        help="add experiment and other tasks' spend, apart",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="write the selection's day and week files too",
    )
    p = sub.add_parser("scoreboard")
    p.add_argument("--days", type=int, default=scoreboard.DEFAULT_DAYS)
    p.add_argument("--state", default=None, help="another state folder (a copy)")
    p = sub.add_parser("scoreboard-backfill")
    p.add_argument("--days", type=int, default=21)
    p.add_argument(
        "--dry-run", action="store_true", help="print the counts; write nothing"
    )
    p.add_argument("--no-github", action="store_true", help="the local records only")
    p.add_argument(
        "--state", default=None, help="another state folder (a copy), for a rehearsal"
    )
    p = sub.add_parser("conflict-files")
    p.add_argument("--days", type=int, default=conflicts.DAYS)
    p = sub.add_parser("ci-times")
    p.add_argument("--days", type=int, default=ci_times.DAYS)
    p.add_argument(
        "--collect",
        action="store_true",
        help="read the heads not recorded yet first (capped)",
    )
    p = sub.add_parser("lessons")
    p.add_argument("--week", default=None)
    p = sub.add_parser("review-start")
    p.add_argument("n", type=int)
    p.add_argument("--kind", default=None, choices=tuple(review.ROUNDS))
    p.add_argument(
        "--now",
        action="store_true",
        help="start an unsettled head, or a follow-up inside the gap after our last review, anyway",
    )
    p.add_argument(
        "--again",
        action="store_true",
        help="a re-review at the head of our last review or at a head we posted on (no gap)",
    )
    p = sub.add_parser("review-queue")
    p.add_argument("ns", type=int, nargs="*")
    p.add_argument(
        "--again",
        action="store_true",
        help="a re-review at the head of our last review or at a head we posted on: neither our review "
        "there nor the gap keeps it back",
    )
    p.add_argument("--drop", action="store_true", help="remove the PRs from the queue")
    sub.add_parser("review-status")
    sub.add_parser("drafts")
    p = sub.add_parser("stage-ours")
    p.add_argument("n", type=int)
    p.add_argument("dir")
    p = sub.add_parser("draft-lint")
    p.add_argument("n", type=int)
    p.add_argument(
        "--dir",
        default=None,
        help="the draft's folder (default: workers/prN, paths.workers)",
    )
    p = sub.add_parser("probe-check")
    p.add_argument("dir")
    p.add_argument("tree", nargs="?", default="")
    p.add_argument("--probe", default="")
    p.add_argument("--devbase", default="")
    p.add_argument("--devbase-tree", default="")
    p.add_argument("--simfix", default="")
    p.add_argument("--simfix-tree", default="")
    p.add_argument("--base", default="develop")
    p = sub.add_parser("probe-port")
    p.add_argument("dir")
    p.add_argument("--write", action="store_true")
    p.add_argument("--root", action="append", default=[])
    for name in (
        "review-skip",
        "review-unskip",
        "review-recheck",
        "review-approve",
        "review-close",
    ):
        p = sub.add_parser(name)
        p.add_argument("n", type=int)
    p = sub.add_parser("classify")
    p.add_argument("n", type=int)
    p.add_argument("--record", action="store_true")
    p = sub.add_parser("facts")
    p.add_argument("n", type=int)
    p = sub.add_parser("review-prep")
    p.add_argument("n", type=int)
    p.add_argument(
        "--round",
        default=None,
        help="fr (fr<k> for a re-review) or fu<k> (default: fr before our first review, posted or kept, else the "
        "next fu<k>)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="the review directory (default: <tmp>/v<N><round>, paths.tmp)",
    )
    p = sub.add_parser("review-split")
    p.add_argument("n", type=int)
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="the most changed lines a part has (default: features.review_automation.large_prs.split_part_lines)",
    )
    p.add_argument(
        "--round", default=None, help="fr, fr<k> or fu<k> (default: as review-prep)"
    )
    p.add_argument(
        "--out",
        default=None,
        help="the review directory (default: <tmp>/v<N><round>, paths.tmp)",
    )
    p.add_argument(
        "--max-agents", type=int, default=split.MAX_AGENTS, help="subagents at once"
    )
    p.add_argument(
        "--team-file",
        default=None,
        help="the verifier team file (default: paths.verifier_team_file)",
    )
    p.add_argument(
        "--head",
        default=None,
        help="the head to split (default: the live head, fetched)",
    )
    p = sub.add_parser("heads")
    p.add_argument("ns", type=int, nargs="+")
    p = sub.add_parser("ci-fails")
    p.add_argument("n", type=int)
    p = sub.add_parser("shard-times")
    p.add_argument("n", type=int)
    p.add_argument("--every", action="store_true")
    p = sub.add_parser("spec")
    p.add_argument("n", type=int)
    p.add_argument("--round", required=True)
    p.add_argument(
        "--dir",
        default=None,
        help="the review directory (default: <tmp>/v<N><round>, paths.tmp)",
    )
    p.add_argument("--light", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--cost-verifier", type=int, default=None)
    p.add_argument("--cost-quality", type=int, default=None)
    p.add_argument(
        "--out",
        default=None,
        help="the spec (default: <tmp>/spec<N><round>.json, paths.tmp)",
    )
    # One call per orchestrator step (rkmon_lib/dispatch.py): the briefs' pack, the spec and its post, the lane's wait.
    p = sub.add_parser("dispatch-pack")
    p.add_argument("n", type=int)
    p.add_argument(
        "--round", default=None, help="fr, fr<k> or fu<k> (default: as rk review-prep)"
    )
    p = sub.add_parser("spec-post")
    p.add_argument("n", type=int)
    p.add_argument("--round", required=True)
    p.add_argument("--dir", default=None)
    p.add_argument("--light", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--cost-verifier", type=int, default=None)
    p.add_argument("--cost-quality", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--again", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--devbase-recheck",
        action="store_true",
        help="after a stop on develop's change to the PR's files",
    )
    p = sub.add_parser("wait-quality")
    p.add_argument("dir")
    p.add_argument("--max-min", type=float, default=20)
    p.add_argument("--slice-min", type=float, default=None)
    p = sub.add_parser("job-log")
    p.add_argument("id", type=int)
    p.add_argument("--errors", action="store_true")
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--pr", type=int, default=0)
    p = sub.add_parser("werror")
    p.add_argument("n", type=int)
    p.add_argument("--tree", default="")
    p.add_argument("--build", default="")
    p.add_argument("--base", default="")
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--dev-tree", default="")
    p.add_argument("--budget", type=int, default=900)
    p = sub.add_parser("evidence-check")
    p.add_argument("n", type=int)
    p.add_argument(
        "--dir", default=None, help="the run's folder (default: the PR's review folder)"
    )
    p.add_argument(
        "--offload", action="store_true", help="the review offload's run, not a draft's"
    )
    p.add_argument(
        "--round",
        default=None,
        help="with --offload: fr or fu<k>, that round's run (default: the live run)",
    )
    p = sub.add_parser("shadow-report")
    p.add_argument("ns", type=int, nargs="*")
    p = sub.add_parser("followup-scan")
    p.add_argument("n", type=int)
    p.add_argument(
        "--out",
        default=None,
        help="the candidates file (default: followups_<N>.md in followup_scan.dir)",
    )
    sub.add_parser("review-advance")
    p = sub.add_parser("post")
    p.add_argument("n", type=int)
    p.add_argument("file")
    p.add_argument("--again", action="store_true")
    p.add_argument(
        "--premises",
        metavar="SPEC",
        default=None,
        help="the verifier's premises, re-checked first",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="every check and rk post's guards, then stop before the post",
    )
    p = sub.add_parser("jobs")
    p.add_argument("id", nargs="?")
    p = sub.add_parser(
        "mirror-report",
        help="the A/B comparison of a live monitor and its mirror (mirror.enabled)",
    )
    p.add_argument(
        "--live-state",
        required=True,
        help="the live monitor's state folder (review_cost.jsonl)",
    )
    p.add_argument(
        "--mirror", required=True, help="the mirror's out folder (mirror.out)"
    )
    p.add_argument(
        "--live-claude",
        default=None,
        help="the live monitor's Claude projects folder (its transcripts); default: this machine's",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("sanity")
    p.add_argument("sha", nargs="?")
    p.add_argument("--base", default=None)
    p.add_argument(
        "--rehearsal",
        action="store_true",
        help="drill on an old SHA: no outward action is ever taken",
    )
    NOW_HELP = "start it even when all parallel builds are busy or the RAM disk has no room for it"
    p.add_argument("--now", action="store_true", help=NOW_HELP)
    sub.add_parser("sanity-status")
    sub.add_parser("incidents")
    p = sub.add_parser("bisect")
    p.add_argument("--now", action="store_true", help=NOW_HELP)
    sub.add_parser("guard-advance")
    sub.add_parser("queue-advance")
    for name in ("gate", "arm", "queue-notice"):
        p = sub.add_parser(name)
        p.add_argument("n", type=int)
        if name == "gate":
            p.add_argument("--now", action="store_true", help=NOW_HELP)
    p = sub.add_parser("disarm")
    p.add_argument("n", type=int)
    p.add_argument("reason", nargs="*")
    for name in ("hold", "release"):
        p = sub.add_parser(name)
        p.add_argument("n", type=int)
        p.add_argument("reason", nargs="*")
    sub.add_parser("holds")
    sub.add_parser("board")
    sub.add_parser("html")
    p = sub.add_parser("cleanup")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("log", help="add a daily-log entry by hand, e.g. a ticket")
    p.add_argument("type")
    p.add_argument("pr", help="PR number or -")
    p.add_argument("text", nargs="+")
    p = sub.add_parser(
        "save",
        help="record an issue the system avoided (finding, fixed, gate, develop, verify, other)",
    )
    p.add_argument("pr", help="PR number or -")
    p.add_argument("type")
    p.add_argument("text", nargs="+")
    p = sub.add_parser("daylog")
    p.add_argument("day", nargs="?")
    p = sub.add_parser("incident-note")
    p.add_argument("id")
    p.add_argument("key")
    p.add_argument("value", nargs="+")
    p = sub.add_parser("incident-close")
    p.add_argument("id")
    p.add_argument("fixed_by", type=int, nargs="?")
    p = sub.add_parser("notice")
    p.add_argument("n", type=int)
    p.add_argument("file")
    p.add_argument(
        "--kind",
        required=True,
        choices=post_mod.NOTICE_CHOICES,
    )
    p.add_argument("--key", required=True)
    p = sub.add_parser("incident-add")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("culprit")
    p.add_argument("fixed_by")
    p.add_argument("summary", nargs="+")
    p.add_argument("--contributing", default="")
    p = sub.add_parser("ungroup")
    p.add_argument("n", type=int)
    sub.add_parser("breakages")
    sub.add_parser("approvals")
    p = sub.add_parser("issue-file")
    p.add_argument("file")
    p.add_argument("--new", action="store_true")
    p = sub.add_parser("issue-comment")
    p.add_argument("n", type=int)
    p.add_argument("file")
    p.add_argument("--key", required=True)
    p = sub.add_parser("issue-relabel")
    p.add_argument("n", type=int)
    p.add_argument("--to", required=True, choices=(breakage.STABILITY, breakage.RED))
    p.add_argument("--assign", default=None)
    p = sub.add_parser("fix-link")
    p.add_argument("issue", type=int)
    p.add_argument("pr", type=int)
    p.add_argument(
        "--evidence",
        default=None,
        help="a file with what verified it (added to the issue comment)",
    )
    p = sub.add_parser("fix-dismiss")
    p.add_argument("issue", type=int)
    p.add_argument("pr", type=int)
    p = sub.add_parser("drift-dismiss")
    p.add_argument("n", type=int)
    p.add_argument("develop")
    p.add_argument(
        "--reason",
        required=True,
        help="why the failure is the harness's, not the PR's or develop's",
    )
    p = sub.add_parser("automerge-off")
    p.add_argument("n", type=int)
    p.add_argument(
        "--user",
        metavar="FILE",
        default=None,
        help="on the user's instruction: turn auto-merge off and post this file's reason",
    )
    p = sub.add_parser("gate-cancel")
    p.add_argument("reason", nargs="+")
    p = sub.add_parser("teams-breakage")
    p.add_argument("n", type=int)
    p.add_argument("--force", action="store_true")
    sub.add_parser("teams-test")
    p = sub.add_parser("needs-you-teams")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("issue-assign")
    p.add_argument("n", type=int)
    p.add_argument("login")
    p = sub.add_parser("label-create")
    p.add_argument("name")
    p.add_argument("--description", required=True)
    p.add_argument("--color", required=True)
    p = sub.add_parser("request-changes")
    p.add_argument("n", type=int)
    p.add_argument("file")
    p.add_argument(
        "--user",
        action="store_true",
        help="on the user's explicit instruction, without a hold",
    )
    p = sub.add_parser("overlaps")
    p.add_argument("n", type=int, nargs="?")
    p.add_argument(
        "--section", action="store_true", help="checks.md's section for PR N"
    )
    p = sub.add_parser("clash")
    p.add_argument("n", type=int)
    p.add_argument("--head", default=None)
    p.add_argument("--heads", default="", help="N=SHA,N=SHA: other PRs' heads")
    p.add_argument("--develop", default=None)
    p.add_argument("--others", default=None, help="N,N: only these PRs")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--max-fetch", type=int, default=None)
    p.add_argument("--cache", default=None, help="default cache/clash (paths.cache)")
    p.add_argument(
        "--new",
        action="store_true",
        help="only the open PRs the checks' scan did not compare (opened since)",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="also build each pair merge (shared C/C++ files) and collect both PRs' changed tests",
    )
    p = sub.add_parser("mutants")
    p.add_argument("n", type=int)
    p.add_argument("--head", default=None)
    p.add_argument("--since", default=None, help="a follow-up's last reviewed head")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--per-mutant", type=float, default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("lane-of")
    p.add_argument("targets", nargs="+", metavar="TARGET")
    p.add_argument("--pr", type=int, default=None, help="at PR N's head (the state's)")
    p.add_argument("--rev", default=None, help="at this commit (default develop)")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--log", default=None, help="a CI shard log")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("winsim")
    p.add_argument("targets", nargs="*", metavar="TARGET")
    p.add_argument("--pr", type=int, default=None, help="at PR N's head (the state's)")
    p.add_argument("--rev", default=None, help="at this commit")
    p.add_argument(
        "--base",
        default=None,
        help="the PR's base (default: its base branch, else develop)",
    )
    p.add_argument(
        "--only", default=None, help="the simulations: text,crlf,tempfile,os"
    )
    p.add_argument(
        "--scan", action="store_true", help="scan the added C/C++ (on with --pr)"
    )
    p.add_argument("--budget", type=float, default=None)
    for name in HEAD_CHECKS:
        p = sub.add_parser(name)
        p.add_argument("n", type=int)
        p.add_argument("--head", default=None)
        p.add_argument("--budget", type=float, default=None)
        if name not in NO_OUT:
            p.add_argument("--out", default=None)
    p = sub.add_parser("peer-checks")
    p.add_argument("n", type=int)
    p.add_argument("--head", default=None)
    p.add_argument(
        "--check",
        default="all",
        help="one of rkmon_lib/peer_checks.CHECKS (default: every one)",
    )
    p = sub.add_parser("overlap-decide")
    p.add_argument("key")
    p.add_argument("owner", type=int)
    p.add_argument(
        "--members", default="", help="N,N (default: the group's current members)"
    )
    p.add_argument("--note", default="")
    p.add_argument(
        "--by",
        default=None,
        help="who decided (default: integrations.github_cli.username)",
    )
    p.add_argument(
        "--same-as", default="", help="KEY: the same decision as one recorded under KEY"
    )
    p = sub.add_parser(
        "gpu-run",
        help="queue GPU-only test cases on a remote GPU node "
        "(features.review_automation.remote_gpu_tests.enabled)",
    )
    p.add_argument("n", type=int, nargs="?")
    p.add_argument("--arch", default=None)
    p.add_argument(
        "--case", action="append", default=[], help="a test id to run (repeatable)"
    )
    p.add_argument(
        "--develop", action="store_true", help="develop's named cases, not a PR's"
    )
    p.add_argument(
        "--same-node",
        action="store_true",
        help="on the node its last job on that target ran on",
    )
    p.add_argument("--node", default=None, help="on this node (sbatch --nodelist)")
    for name in ("gpu-poll", "gpu-status", "gpu-queue", "gpu-check"):
        sub.add_parser(name)
    p = sub.add_parser("gpu-cancel")
    p.add_argument("target", help="PR number, Slurm job, job id, or all")
    p = sub.add_parser("gpu-row")
    p.add_argument("n", type=int)
    p.add_argument(
        "--grade",
        default=None,
        choices=tuple(gpu.GRADES),
        help="the failure's grade, as you graded it",
    )
    p = sub.add_parser("selftest")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="run the whole suite's modules in N processes at a time, the largest first, each with a home folder of "
        "its own (1, the default: the suite in one process; the tests named run in one process)",
    )
    p.add_argument(
        "tests",
        nargs="*",
        metavar="TEST",
        help="a test module, class or test to run instead of the whole suite, as unittest names it from the package's "
        "folder (tests.test_selftest, tests.test_selftest.TestSelftest)",
    )
    sub.add_parser("paths")
    sub.add_parser("home")
    p = sub.add_parser(
        "guide",
        help="print the operator's guide (QUICK_USER_GUIDE.md), with `build` the image builder's guide "
        "(QUICK_BUILD_GUIDE.md), with `deploy` the container's deploy guide (QUICK_DEPLOY_GUIDE.md)",
    )
    p.add_argument(
        "which", nargs="?", choices=("user", "build", "deploy"), default="user"
    )
    p = sub.add_parser(
        "logins",
        help="every login and secret, in the secrets registry's order, with what needs it, its state and its fix",
    )
    p.add_argument("--json", action="store_true")
    p = sub.add_parser(
        "secret-set",
        help="a secret from stdin into its file (mode 600; never printed): docker/rkmon secret NAME runs it",
    )
    p.add_argument("name", help="a secret the secrets registry lists")
    p.add_argument(
        "--remove", action="store_true", help="remove its file instead (no stdin)"
    )
    p = sub.add_parser(
        "writer-secrets",
        help="whether each secret the writer keeps is set, where, and when it last changed (never a value)",
    )
    p.add_argument("--json", action="store_true")
    sub.add_parser(
        "claude-test",
        help="the claude login's test call, sent as the Claude session sends its requests: its row; exit 4 on a FAIL",
    )
    sub.add_parser(
        "read-login",
        help="a GitHub token on stdin as the read login, refused when it can write (never printed): docker/rkmon "
        "login github_read runs it",
    )
    p = sub.add_parser(
        "host-key",
        help="the own keys of the host the secret NAME's ssh key reaches, as read here; --record: those on stdin",
    )
    p.add_argument("name", choices=tuple(h.secret for h in host_keys.HOSTS))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument(
        "--record",
        action="store_true",
        help="record the known_hosts lines on stdin as the host's own keys",
    )
    sub.add_parser(
        "web-host-install",
        help="the request service on the team dashboard's web host, from the settings rk runs on (docker/rkmon "
        "web-host install runs it)",
    )
    p = sub.add_parser(
        "teams-send",
        help="one message to the team chat channel, on stdin (in the container, the writer's half of each)",
    )
    p.add_argument("--type", required=True, choices=tuple(post_mod.TEAMS_FEATURES))
    p = sub.add_parser(
        "web-push",
        help="the team dashboard's page to the web host (in the container, the writer's half of each push)",
    )
    p.add_argument("file")
    sub.add_parser(
        "web-requests",
        help="the web host's priority requests file, printed (in the container, the writer's half of each read)",
    )
    p = sub.add_parser(
        "atlassian",
        help="one of the atlassian CLI's commands the writer lists, with its login (in the container, as the writer)",
    )
    p.add_argument("--site", default="")
    p.add_argument("--json", action="store_true")
    p.add_argument("--timeout-s", type=int, default=None)
    p.add_argument("words", nargs=argparse.REMAINDER, metavar="WORDS")
    p = sub.add_parser(
        "adopt",
        help="an earlier monitor's state, staged by docker/rkmon adopt: plan or apply",
    )
    p.add_argument("action", choices=("plan", "apply"))
    p.add_argument("--stage", required=True, help="the staging folder")
    p = sub.add_parser("doctor")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="check every service again (not from the few-minute cache)",
    )
    only = p.add_mutually_exclusive_group()
    only.add_argument(
        "--fails",
        action="store_true",
        help="only the names of the checks that fail, one per line",
    )
    only.add_argument(
        "--json",
        action="store_true",
        help="only the rows, as one JSON list on one line, each with its fix (rkmon apply reads them)",
    )
    p = sub.add_parser("features")
    p.add_argument(
        "--runs",
        choices=features.COMPONENTS,
        help="exit 0 when the deployment runs this part, 1 when it does not, 4 when the configuration cannot be "
        "read (none runs)",
    )
    p = sub.add_parser("config", help="the running configuration")
    p.add_argument("action", choices=("show",))
    p.add_argument(
        "--source", action="store_true", help="the layer each value came from"
    )
    p = sub.add_parser("env")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("orchestrate")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--running", action="store_true")
    p = sub.add_parser("tick-due")
    p.add_argument("--minutes", type=int, default=25)
    sub.add_parser("tick-min")
    sub.add_parser("stopped")
    p = sub.add_parser(
        "applied",
        help="your confirmed settings the monitor runs (applied record)",
    )
    p.add_argument(
        "--load",
        action="store_true",
        help="load the configuration now and record its layer as applied",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="the record, whether a pause is set and whether the monitor is in standby",
    )
    p = sub.add_parser("usage")
    p.add_argument("--days", type=int, default=7)
    p.add_argument(
        "--by-review", action="store_true", help="per review we posted, not per day"
    )
    sub.add_parser("roles")
    p = sub.add_parser("first-pass")
    p.add_argument("n", type=int)
    p.add_argument("--head", default="")
    p = sub.add_parser("notice-draft")
    p.add_argument("n", type=int)
    p.add_argument("--kind", required=True, choices=("overlap", "drift"))
    p = sub.add_parser("insights")
    p.add_argument("--days", type=int, default=14)
    p = sub.add_parser("role-run")
    p.add_argument("role", choices=roles.ROLES)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("provision")
    p.add_argument("--check", action="store_true")
    sub.add_parser("standby-off")
    p = sub.add_parser("code")
    p.add_argument("action", choices=("install", "back"))
    p.add_argument("words", nargs=argparse.REMAINDER)
    p = sub.add_parser("md2adf")
    p.add_argument("file")
    args = ap.parse_args(argv)
    if args.cmd == "stop" and args.timeout_minutes is not None and not args.drain:
        stop_parser.error("--timeout-minutes goes with --drain")
    if args.cmd == "jira-account" and len(
        [w for w in (args.login, args.account) if w]
    ) != (2 if args.action == "add" else 0):
        jira_account_parser.error("add takes LOGIN and ACCOUNT; list takes neither")
    if args.cmd == "md2adf":  # a file in, JSON out: no configuration, no state
        from rkmon_lib import adf

        return adf.main([args.file])
    if args.cmd == "pr-fetch":  # a check's --fetch: no configuration, no state
        return cmd_pr_fetch(args)
    # folders made trees PR code reads: no configuration, no state
    if args.cmd == "pr-tree":
        return cmd_pr_tree(args)
    # the container's code is root's to change (rkmon_lib/releases.py): a refusal, with no configuration and no state
    if args.cmd == "code":
        from rkmon_lib import releases

        return releases.main(args.action)
    # the first start's setup of the work volume: it loads the configuration itself, and --check reads only the state
    if args.cmd == "provision":
        from rkmon_lib import provision

        return provision.main(check=args.check)
    if args.cmd in NO_CONFIG:
        return cmd_noconfig(args)
    # In the container the writer's commands run as the writer, the one user that reads their secrets
    # (rkmon_lib/writer.py; docker/rk_write.py runs the command, with the image's code and every guard).
    from rkmon_lib import writer

    routed = writer.routed(argv)
    if routed:
        sys.stdout.flush()
        try:
            return subprocess.run(routed).returncode
        except OSError as exc:
            print(writer.refusal(f"it did not start ({type(exc).__name__}: {exc})"))
            return 4
    try:
        # The emergency brake (BRAKE) runs with what names the folders alone: a configuration that cannot be loaded
        # never keeps it from stopping the monitor; and so does rk adopt (FOLDERS_ONLY).
        if args.cmd in FOLDERS_ONLY:
            cfg = config.folders()
        elif args.cmd in UNCHECKED:
            cfg = config.load(check=False)
        else:
            cfg = load_config()
    except config.ConfigError as exc:
        return config_refused(exc)
    gh.configure(
        cfg
    )  # the gh binary and its login's folder, for every gh call of this command
    try:
        return run_command(args, cfg)
    finally:
        gh.configure({})


# The long-running commands register their process while they run (rkmon_lib/procs.py), so rk stop, rk kill and
# rk status find them.
REGISTERED = {
    "tick": "tick",
    "review-advance": "review-advance",
    "serve": "serve",
    "review-driver": "driver",
    "stop": "stop",
    "drain": drain.KIND,
    "orchestrate": control.ORCHESTRATOR,
}


def registered_kind(args) -> str:
    """What the command's process registers as while it runs (REGISTERED), "" for nothing: `rk orchestrate --check`
    and `--running` only ask, so neither counts as a Claude session (control.session_pids).
    """
    if args.cmd == "orchestrate" and (args.check or args.running):
        return ""
    return REGISTERED.get(args.cmd, "")


def outward(args) -> bool:
    """Does this command write outward, under the write lock (control.WRITES)? Not with a --dry-run that sends nothing
    (rkmon_lib/outward.py: dry_run)."""
    row = outward_mod.row(args.cmd)
    return (
        bool(row) and row.gate == outward_mod.OWN and not (row.dry_run and args.dry_run)
    )


def run_command(args, cfg: dict) -> int:
    # A mirror (a monitor beside the live one that sends nothing) refuses an outward write before any request.
    refusal = mirror.guard(cfg, args.cmd) if outward(args) else None
    if refusal:
        print(f"REFUSED: {refusal}")
        return 4
    gate = feature_gate(args, cfg)
    if gate:
        code, msg = gate
        print(msg)
        return code
    store = Store()
    try:
        st0 = store.load()
    except ValueError:
        st0 = {}
    why = (
        control.refusal(args.cmd, st0)
        if outward(args) or args.cmd in control.STARTERS
        else ""
    )
    # stopped: no outward write, no new job, until rk start (the writer's sends: write_gate)
    if why:
        print(f"REFUSED: {why}" if args.cmd in control.WRITES else f"STOPPED: {why}")
        return 4 if args.cmd in control.WRITES else 0
    refusal = write_gate(args, cfg, st0)  # the writer's sends for another process
    if refusal:
        print(refusal)
        return 0 if post_mod.held_line(refusal) else 4
    run = None
    kind = registered_kind(args)
    if kind:
        try:
            run = control.run_dir(cfg)
            procs.register(run, kind, getattr(args, "cmd", ""))
        except paths.PathsError:
            run = None
    try:
        if not outward(args):
            return _run_command(args, cfg, store)
        # Every outward write holds the write lock from its first write to its ledger entry: rk stop and rk kill wait
        # for it, so no GitHub write is cut in half (rkmon_lib/control.py).
        with control.write_lock(store.dir, control.WRITE_WAIT_S) as held:
            if not held:
                print(
                    "REFUSED: the write lock is still held (another outward write, or rk stop or rk kill); try again"
                )
                return 4
            if control.refusal(args.cmd, store.load()):  # stopped while it waited
                print(f"REFUSED: {control.refusal(args.cmd, store.load())}")
                return 4
            return _run_command(args, cfg, store)
    finally:
        if run:
            procs.unregister(run)


def _run_command(args, cfg: dict, store) -> int:
    try:
        if args.cmd == "tick":
            return cmd_tick(args, cfg, store)
        if args.cmd == "rules-tick":
            if args.clear:
                print(f"cleared {rules_tick.clear(store)} line(s) waiting for a person")
                return 0
            return rules_tick.run(cfg, store)
        if args.cmd == "worker-tools":
            return cmd_worker_tools(args, cfg)
        if args.cmd == "pr-slot":
            return cmd_pr_slot(args)
        if args.cmd == "pr-run":
            return cmd_pr_run(args, store)
        if (
            args.cmd == "logins"
        ):  # read-only: each login's state (the GitHub probes refuse to make anything)
            from rkmon_lib import logins

            return logins.main(cfg, as_json=args.json)
        if args.cmd == "secret-set":
            return cmd_secret_set(args.name, cfg, remove=args.remove)
        # one message of one token with the key the session reads, never shown
        if args.cmd == "claude-test":
            from rkmon_lib import checks, doctor, reach

            row = reach.claude_test(cfg, doctor.claude_login_fix(cfg))
            print(checks.render([row]))
            return 4 if row.status == "FAIL" else 0
        if (
            args.cmd == "read-login"
        ):  # a token on stdin as the read login, never one that can write; never printed
            return ghlogins.set_read(cfg, sys.stdin.read())
        if args.cmd == "host-key":  # a host's own keys, read or recorded (--record)
            return cmd_host_key(args, cfg)
        if args.cmd == "web-host-install":  # the writer's, in the container
            from rkmon_lib import web_host_install

            return web_host_install.install(cfg)
        if (
            args.cmd == "writer-secrets"
        ):  # read-only: whether each is set, never a value (the writer answers it in the container)
            from rkmon_lib import logins

            kept = logins.writer_secrets(cfg)
            print(
                json.dumps(kept)
                if args.json
                else "\n".join(
                    f"{name}: {'set' if f['set'] else 'not set'} ({f['where']})"
                    for name, f in kept.items()
                )
            )
            return 0
        if args.cmd in outward_mod.names(handler=outward_mod.HALF):
            return cmd_writer_half(args, cfg)
        if args.cmd == "adopt":
            return cmd_adopt(args, cfg, store)
        if args.cmd == "loop-config":
            # and rk tick-now's request, which the loop looks for while it waits, and the shell tick loop's mark
            # (rkmon_lib/control.py)
            files = {
                "TICK_NOW": control.TICK_NOW,
                "SHELL_LOOP_PID": control.SHELL_LOOP_PID,
            }
            print(
                "\n".join(
                    idle.loop_config(cfg, store)
                    + [
                        f"{k}={shlex.quote(str(store.dir / name))}"
                        for k, name in files.items()
                    ]
                )
            )
            return 0
        if args.cmd == "loop-status":
            pid = idle.loop_pid(store)
            running = (
                "no tick loop running"
                if pid is None
                else "tick loop running" + (f" (pid {pid})" if pid else "")
            )
            print(f"{running}; log {store.dir / idle.LOOP_FILES['LOOP_LOG']}")
            print("\n".join(idle.status_lines(cfg, store.load())))
            return 0 if pid is not None else 1
        if args.cmd == "stop":
            return cmd_stop(args, cfg, store)
        if args.cmd == "start":
            return control.start(cfg, store)
        if args.cmd == "kill":
            return control.kill_all(cfg, store, wait=args.wait)
        if args.cmd == "drain":
            return drain.main(cfg, store, timeout_min=args.timeout_minutes)
        if args.cmd == "undrain":
            return drain.undrain(store)
        if args.cmd == "review-driver":
            return cmd_review_driver(cfg, store)
        if args.cmd == "init":
            return cmd_init(args, cfg, store)
        if args.cmd == "status":
            return cmd_status(args, cfg, store)
        if (
            args.cmd == "gh-logins"
        ):  # the probes only (gh.probe): GitHub refuses both, nothing is made
            if args.side and args.json:
                print(ghlogins.side_json(cfg, args.side))
                return 0
            lines = ghlogins.status_lines(cfg, store, force=True)
            if args.side:
                lines = [x for x in lines if args.side in x]
            print("\n".join(lines))
            return 1 if any(x.startswith("!! ") for x in lines) else 0
        if args.cmd in (
            "pause",
            "resume",
            "freeze",
            "unfreeze",
            "clear",
            "unclear",
            "prioritize",
            "unprioritize",
        ):
            return cmd_flag(args, cfg, store)
        # the container's Claude session (rkmon_lib/orchestrator.py)
        if args.cmd == "orchestrate":
            from rkmon_lib import orchestrator

            if args.running:  # read-only
                pids = control.session_pids(cfg)
                print(
                    f"ORCHESTRATOR running (pid {', '.join(map(str, pids))})"
                    if pids
                    else "ORCHESTRATOR not running"
                )
                return 0 if pids else 1
            if not args.check and stopped(store.load()):
                print(
                    f"{now()} ORCH not started: the monitor is stopped (rk start resumes it)"
                )
                return 0
            return orchestrator.main(cfg, check_only=args.check)
        if args.cmd == "tick-min":  # read-only
            from rkmon_lib import orchestrator

            print(orchestrator.tick_min(cfg))
            return 0
        if args.cmd == "stopped":  # read-only
            s = stopped(store.load())
            print(
                f"stopped by rk {s.get('how') or 'stop'} at {s.get('at')}; rk start resumes"
                if s
                else "not stopped"
            )
            return 0 if s else 1
        if args.cmd == "tick-now":
            return control.ask_tick(cfg, store)
        if args.cmd == "tick-wait":
            return control.wait_tick(cfg, store, args.seconds, args.loop)
        if args.cmd == "tick-due":
            st = store.load()
            last = st.get("last_tick")
            age = minutes_since(last) if last else None
            print(
                f"last tick {last or 'never'}"
                + (f", {age:.0f} min ago" if age is not None else "")
            )
            return 0 if age is None or age >= args.minutes else 1
        if args.cmd == "roles":  # read-only
            print("\n".join(roles.table_lines(cfg, store)))
            print("\n".join(roles.status_lines(cfg, store.load(), store)))
            return 0
        if args.cmd in (
            "first-pass",
            "notice-draft",
        ):  # writes only its leads file and its record
            st = store.load()
            print(
                "\n".join(
                    first_pass.job(cfg, st, store, args.n, args.head)
                    if args.cmd == "first-pass"
                    else first_pass.notice(cfg, st, store, args.n, args.kind)
                )
            )
            return 0
        if (
            args.cmd == "role-run"
        ):  # the background job of a model role (rkmon_lib/roles.py)
            return roles.run_job(cfg, args.role, args.input, args.output)
        if args.cmd == "insights":  # read-only
            from rkmon_lib import insights

            st = store.load()
            stats = insights.pr_statistics(cfg, st, store, days=args.days)
            got = analyst.reading(st) if args.days == insights.DAYS else None
            print("\n".join(insights.stats_md(stats, got)))
            return 0
        if args.cmd == "usage":  # tokens only: no price, no cost
            from rkmon_lib import usage

            print(
                "\n".join(
                    (usage.by_review_report if args.by_review else usage.report)(
                        cfg, store, args.days
                    )
                )
            )
            return 0
        if args.cmd == "serve" and stopped(
            store.load()
        ):  # rk stop/kill: it starts again with rk start
            print(
                f"{now()} RKWEB not started: the monitor is stopped (rk start resumes it)"
            )
            return 0
        if args.cmd == "serve":  # the dashboard's "Request PR priority" box
            return webserve.serve(cfg, store, args.port, args.bind, args.public_port)
        if args.cmd == "priority-requests":  # read-only
            print("\n".join(priority.lines(cfg, store.load())))
            return 0
        if args.cmd in (
            "ticket-draft",
            "tickets",
            "ticket-file",
            "ticket-decline",
        ):  # rkmon_lib/tickets.py
            return cmd_tickets(args, cfg, store)
        if args.cmd == "jira-account":  # local: the internal state
            return cmd_jira_account(args, cfg)
        if args.cmd == "standby-off":  # local: the monitor's home
            from rkmon_lib import standby

            return standby.off(cfg)
        if args.cmd == "priority-decline":  # local: state and journal
            with store.locked(wait=True):
                st = store.load()
                ok, msg = priority.decline(cfg, st, store, args.n)
                store.save(st)
            print(msg if ok else f"REFUSED: {msg}")
            return 0 if ok else 4
        if args.cmd == "ledger":
            for r in store.ledger()[-args.n :]:
                print(json.dumps(r, ensure_ascii=False))
            return 0
        if args.cmd in ("abi", "batches"):
            st = store.load()
            if args.cmd == "batches":
                for i, b in enumerate(st.get("batches") or [], 1):
                    print(
                        f"{i}. [{b.get('origin', 'auto')}] {' -> '.join('#' + str(n) for n in b['members'])}  "
                        f"lines={b['lines']} prefix_checks={b['check_prefixes']} status={b['status']}"
                        + (
                            f"  notes: {'; '.join(b['notes'])}"
                            if b.get("notes")
                            else ""
                        )
                    )
                return 0
            for line in abi.lane_lines(st):
                print(line)
            return 0
        if args.cmd == "overlaps":  # read-only: no fetch, no state written
            if args.section and args.n is None:
                print("--section needs a PR number")
                return 4
            st = store.load()
            with overlaps.no_fetch():
                lines = (
                    overlaps.section(cfg, st, store, args.n)
                    if args.section
                    else overlaps.render_all(cfg, st, store, args.n)
                )
            print("\n".join(lines))
            return 0
        if (
            args.cmd == "clash"
        ):  # read-only: no GitHub, no state; temporary merge objects (no ref) and the cache
            heads = {
                int(k.strip().lstrip("#")): v.strip()
                for k, _, v in (
                    x.partition("=") for x in args.heads.split(",") if x.strip()
                )
            }
            if args.head:
                heads[args.n] = args.head
            others = (
                [
                    int(x.strip().lstrip("#"))
                    for x in args.others.split(",")
                    if x.strip()
                ]
                if args.others is not None
                else None
            )
            st = store.load()
            if args.new:
                at, new = clashes.unscanned(cfg, st, args.n)
                if at is None:
                    print(
                        f"no clash scan record in #{args.n}'s checks ({clashes.record_path(cfg, args.n)}): "
                        "scanning every open PR"
                    )
                elif not new:
                    print(
                        f"every open PR was compared by the checks' clash scan ({at}): nothing new"
                    )
                    return 0
                else:
                    print(
                        f"open PRs the checks' clash scan ({at}) did not compare: "
                        + " ".join(f"#{k}" for k in new)
                    )
                    others = (
                        sorted(set(new) & set(others)) if others is not None else new
                    )
            res, lines = clashes.run(
                cfg,
                st,
                args.n,
                heads=heads,
                develop=args.develop,
                others=others,
                cache=args.cache or f"{cfg['paths']['cache']}/clash",
                budget_s=args.budget,
                max_fetch=args.max_fetch,
            )
            if args.build:
                lines += clashes.build(cfg, res)
            print("\n".join(lines))
            return 0
        if (
            args.cmd == "mutants"
        ):  # read-only: a scratch copy of the head, removed after; the report in --out
            head = args.head or (store.load()["prs"].get(str(args.n)) or {}).get("head")
            if not head:
                print(f"#{args.n} has no head in the state: give --head SHA")
                return 4
            p = cfg["paths"]
            cmd = [
                "nice",
                "-n",
                "10",
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "tools" / "mutation_sweep.py"),
                "--repo",
                p["repo"],
                "--head",
                head,
                "--tree",
                paths.worktree(cfg, f"wt{args.n}"),
                "--rocke-path",
                paths.repository_folder(cfg),
                "--python",
                f"{p['venv']}/bin/python",
            ]
            if paths.resolve(cfg).get("torch_python"):
                cmd += ["--torch-python", paths.resolve(cfg)["torch_python"]]
            for flag, value in (
                ("--budget", args.budget),
                ("--cap", args.cap),
                ("--per-mutant", args.per_mutant),
                ("--since", args.since),
            ):
                if value is not None:
                    cmd += [flag, str(value)]
            # In a slot of its own, after its `--fetch` (the blobs of the clone it reads: git in the slot fetches
            # nothing): the sweep writes its report in the slot's out, copied into --out as data.
            env = prcode.with_trees(
                {**os.environ, "TMPDIR": p["tmp"]},
                p["repo"],
                paths.worktree(cfg, f"wt{args.n}"),
            )
            prcode.fetch_for(cmd, env)
            return prcode.call(
                lambda s: [*cmd, "--out", s.out],
                env,
                out=args.out or f"{p['tmp']}/mutants_pr{args.n}",
            )
        if (
            args.cmd == "lane-of"
        ):  # a scratch copy of the tree in a slot of its own; the monitor keeps its listing in the cache
            if args.pr is not None and args.rev:
                print("give --pr or --rev, not both")
                return 4
            rev = args.rev
            if args.pr is not None:
                rev = (store.load()["prs"].get(str(args.pr)) or {}).get("head")
                if not rev:
                    print(f"#{args.pr} has no head in the state: give --rev SHA")
                    return 4
            return lanes.run_lane_of(
                cfg,
                args.targets,
                rev=rev,
                collect=args.collect,
                log=args.log,
                as_json=args.json,
                capture=False,
                timeout=None,
            ).returncode
        if (
            args.cmd == "winsim"
        ):  # a scratch copy of the tree, removed after; PR code in a slot of its own
            if args.pr is not None and args.rev:
                print("give --pr or --rev, not both")
                return 4
            if args.pr is None and not args.rev:
                print("give --pr N, or --rev REV with the test files to run")
                return 4
            rev, base = args.rev, args.base
            if args.pr is not None:
                rec = store.load()["prs"].get(str(args.pr)) or {}
                rev = rec.get("head")
                if not rev:
                    print(f"#{args.pr} has no head in the state: give --rev SHA")
                    return 4
                base = base or (
                    None if gitops.is_develop(rec.get("base")) else rec.get("base")
                )
            if args.pr is None and not args.targets and not args.scan:
                print("give the test files to run, or --scan")
                return 4
            return lanes.winsim(
                cfg,
                args.targets,
                rev,
                gitops.base_ref(base) if base else lanes.prtree.DEVELOP,
                scan=args.pr is not None or args.scan,
                only=args.only,
                budget=args.budget,
            )
        if (
            args.cmd in HEAD_CHECKS
        ):  # read-only: scratch copies of the head and its merge base, removed after
            head = args.head or (store.load()["prs"].get(str(args.n)) or {}).get("head")
            if not head:
                print(f"#{args.n} has no head in the state: give --head SHA")
                return 4
            p = cfg["paths"]
            cmd = [
                "nice",
                "-n",
                "10",
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "tools" / HEAD_CHECKS[args.cmd]),
                "--repo",
                cfg["paths"]["repo"],
                "--head",
                head,
                "--rocke-path",
                paths.repository_folder(cfg),
            ]
            if args.cmd == "import-probe":
                cmd += ["--python", f"{p['venv']}/bin/python"]
            if args.budget is not None:
                cmd += ["--budget", str(args.budget)]
            # The checks' PATH: the venv's cmake, ctest and ninja.
            env = {
                **os.environ,
                "TMPDIR": p["tmp"],
                "PATH": f"{p['venv']}/bin:{os.environ.get('PATH', '')}",
            }
            if args.cmd in PR_CODE_CHECKS:
                # In a slot of its own, after its `--fetch` (as the sweep's): the tool writes in the slot's out, copied
                # into --out as data.
                prcode.fetch_for(cmd, env)
                return prcode.call(
                    lambda s: [*cmd, "--out", s.out],
                    prcode.with_trees(env, p["repo"]),
                    out=args.out
                    or f"{p['tmp']}/{args.cmd.replace('-', '_')}_pr{args.n}",
                )
            return subprocess.run(cmd, env=gh.job_env(env)).returncode
        if (
            args.cmd == "peer-checks"
        ):  # read-only: git, the staged title and GitHub GETs
            head = args.head or (store.load()["prs"].get(str(args.n)) or {}).get("head")
            if not head:
                print(f"#{args.n} has no head in the state: give --head SHA")
                return 4
            p = cfg["paths"]
            meta = pathlib.Path(f"{p['workers']}/pr{args.n}/pr/meta.md")
            cmd = [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "tools" / "peer_checks.py"),
                args.check,
                "--repo",
                cfg["paths"]["repo"],
                "--head",
                head,
                "--rocke-path",
                paths.repository_folder(cfg),
                "--slug",
                cfg["repo"],
                "--pr",
                str(args.n),
            ]
            from rkmon_lib.peer_checks import tracker_allow

            cmd += ["--allow", ",".join(tracker_allow(cfg))]
            if meta.exists():
                cmd += ["--meta", str(meta)]
            else:
                # As JSON: a plain jq string comes back stripped, and a leading space is one of the title's faults.
                cmd += [
                    "--title",
                    gh.rest(
                        f"repos/{cfg['repo']}/pulls/{args.n}", jq="{title: .title}"
                    )["title"],
                ]
            return subprocess.run(
                cmd, env=gh.job_env({**os.environ, "TMPDIR": p["tmp"]})
            ).returncode
        if (
            args.cmd == "drift-dismiss"
        ):  # local only: no GitHub write; its own record, never the outward-action ledger
            with store.locked(wait=True):
                st = store.load()
                try:
                    print(
                        drift.dismiss(
                            st,
                            store,
                            args.n,
                            args.develop,
                            args.reason,
                            config.operator_login(cfg),
                        )
                    )
                except ValueError as exc:
                    print(f"REFUSED: {exc}")
                    return 4
                store.save(st)
            return 0
        if (
            args.cmd == "overlap-decide"
        ):  # local only: no GitHub write, so no ledger entry
            with store.locked(wait=True):
                st = store.load()
                try:
                    rec = overlaps.decide(
                        st,
                        store,
                        args.key,
                        args.owner,
                        [
                            int(x.strip().lstrip("#"))
                            for x in args.members.split(",")
                            if x.strip()
                        ],
                        args.note,
                        args.by or config.operator_login(cfg),
                        args.same_as,
                    )
                except ValueError as exc:
                    print(f"REFUSED: {exc}")
                    return 4
            print(
                f"decided: {args.key}: #{rec['owner']} owns it (members "
                f"{', '.join('#%d' % n for n in rec['members'])})"
                + (f"; the same decision as {args.same_as}" if args.same_as else "")
            )
            return 0
        if args.cmd == "merge-plan":  # read-only
            print(
                "\n".join(report.plan_md(cfg, store.load()))
                or "no merge plan yet: the next tick computes it"
            )
            return 0
        if args.cmd == "abi-ok":
            with store.locked(wait=True):
                st = store.load()
                rec = st["prs"][str(args.n)]
                st.setdefault("abi_waived", []).append([args.n, rec["head"]])
                store.journal(
                    "mode",
                    f"abi-ok #{args.n} at {rec['head'][:10]}: {(rec.get('abi_cache') or {}).get('notes')}",
                )
                store.save(st)
            print(f"#{args.n} ABI notes accepted at {rec['head'][:10]}")
            return 0
        if args.cmd in ("group", "ungroup"):
            with store.locked(wait=True):
                st = store.load()
                groups = [
                    g
                    for g in st.get("groups", [])
                    if (args.ns[0] if args.cmd == "group" else args.n) not in g
                ]
                if args.cmd == "group":
                    groups = [[n for n in g if n not in args.ns] for g in groups]
                    groups = [g for g in groups if g] + [args.ns]
                st["groups"] = groups
                store.journal(
                    "mode",
                    f"{args.cmd} {args.ns if args.cmd == 'group' else args.n}: groups now {groups}",
                )
                store.save(st)
            print(f"groups: {groups}")
            return 0
        if args.cmd == "board":
            st = store.load()
            print(
                report.board(
                    cfg, st, store, st.get("last_plan", []), st.get("queue", [])
                )["md"]
            )
            return 0
        if args.cmd == "needs-you-teams":
            if args.dry_run:  # reads the state and writes nothing
                st = store.load()
                bd = report.board(
                    cfg, st, store, st.get("last_plan", []), st.get("queue", [])
                )
                print("\n".join(needs_teams.command(cfg, st, store, bd, dry_run=True)))
                return 0
            with store.locked(wait=True):
                st = store.load()
                bd = report.board(
                    cfg, st, store, st.get("last_plan", []), st.get("queue", [])
                )
                try:
                    print(
                        "\n".join(
                            needs_teams.command(cfg, st, store, bd, dry_run=False)
                        )
                    )
                except (post_mod.PostRefused, ValueError) as exc:
                    print(f"REFUSED: {exc}")
                    return 4
                store.save(st)
            return 0
        if args.cmd == "html":
            st = store.load()
            plan, q = st.get("last_plan", []), st.get("queue", [])
            print(
                webdash.write(
                    cfg, st, store, plan, q, report.board(cfg, st, store, plan, q)
                )
            )
            return 0
        if args.cmd == "holds":
            st = store.load()
            live = queue_mod.active_holds(st)
            for k, h in sorted(
                (st.get("holds") or {}).items(), key=lambda kv: int(kv[0])
            ):
                print(
                    f"#{k} {h['status']} since {h['since']} ({h.get('source', '?')}) at {h['head'][:10]}"
                    + ("" if k in live else " [ended: pushed or closed]")
                    + f": {h['reason']}"
                )
            return 0
        if args.cmd in ("hold", "release"):
            with store.locked(wait=True):
                st = store.load()
                rec = st["prs"].get(str(args.n))
                if not rec:
                    print(f"#{args.n} is not tracked")
                    return 4
                holds, reason = st.setdefault("holds", {}), " ".join(args.reason)
                if args.cmd == "hold":
                    h = holds.get(str(args.n))
                    if not h and not reason:
                        print("a new hold needs a reason")
                        return 4
                    h = h or {"head": rec["head"], "source": "manual"}
                    h.update(status="held", since=now(), stale_told=False)
                    if reason:
                        h["reason"] = reason
                    holds[str(args.n)] = h
                    store.journal(
                        "queue", f"#{args.n} held at {h['head'][:10]}: {h['reason']}"
                    )
                    daylog.add(
                        store,
                        cfg,
                        "hold",
                        args.n,
                        f"confirmed (the PR's fault): {h['reason']}",
                    )
                    daylog.save(
                        store,
                        cfg,
                        args.n,
                        "gate",
                        f"merge blocked by a failing gate: {h['reason']}",
                    )
                    print(f"#{args.n} held at {h['head'][:10]}: {h['reason']}")
                    mine = next(
                        (
                            r["state"]
                            for r in rec.get("reviews") or []
                            if r["login"] == config.operator_login(cfg)
                        ),
                        None,
                    )
                    if (
                        rec.get("author") != config.operator_login(cfg)
                        and mine != "CHANGES_REQUESTED"
                    ):
                        print(
                            f"NEXT: request changes so it cannot merge until fixed: `rk request-changes {args.n} FILE` "
                            f"(line 1: {post_mod.notice_line(cfg, 'hold')}; what fails, how to fix it, and to re-request a "
                            "review once fixed)"
                        )
                else:
                    h = holds.pop(str(args.n), None)
                    if h and h.get("source") == "gate":
                        rec["gates"] = None  # re-gate it, e.g. after a tool fix
                    qs = st.setdefault("queue_state", {})
                    qs["failed_batches"] = [
                        f
                        for f in qs.get("failed_batches", [])
                        if str(args.n) not in f["id"].split("-")[1:]
                    ]
                    store.journal(
                        "queue",
                        f"#{args.n} released"
                        + (f": {reason}" if reason else "")
                        + (
                            f" (was {h['status']}: {h['reason']})"
                            if h
                            else " (it had no hold)"
                        ),
                    )
                    daylog.add(store, cfg, "release", args.n, reason or "released")
                    print(f"#{args.n} released" + ("" if h else " (it had no hold)"))
                store.save(st)
            return 0
        if args.cmd == "post" and (args.premises or args.dry_run):
            return cmd_post_checked(args, cfg, store)
        if args.cmd == "review-queue":  # local: the state and the journal
            return cmd_review_queue(args, cfg, store)
        if args.cmd == "review-approve":  # local: the state and the journal
            with store.locked(wait=True):
                st = store.load()
                ok, msg = review.approve(cfg, st, store, args.n)
                store.save(st)
            print(msg if ok else f"REFUSED: {msg}")
            return 0 if ok else 4
        if (
            args.cmd == "review-close"
        ):  # reads GitHub before the state lock, then checks again and writes under it
            try:
                found, why = review.close_reads(cfg, store.load(), store, args.n)
            except gh.GhError as exc:
                found, why = {}, (
                    f"our account's reviews on #{args.n} were not read ({str(exc)[:120]}): nothing closed"
                )
            ok, lines = False, [why]
            if not why:
                with store.locked(wait=True):
                    st = store.load()
                    ok, lines = review.close_posted(cfg, st, store, args.n, found)
                    if ok:
                        store.save(st)
            for line in lines:
                print(line if ok else f"REFUSED: {line}")
            return 0 if ok else 4
        if args.cmd in (
            "review-start",
            "review-skip",
            "review-unskip",
            "review-recheck",
            "post",
        ):
            refused = False
            with store.locked(wait=True):
                st = store.load()
                if args.cmd == "review-recheck":
                    # In store.halted's words alone, as heavy_start_refusal's: a drain's words say that nothing new
                    # starts (a stop refused it before: control.STARTERS). The slot and room checks are recheck's own.
                    why = control.halted(st)
                    if why:
                        print(f"REFUSED: {why}")
                        return 4
                    for line in review.recheck(cfg, st, store, args.n):
                        print(line)
                elif args.cmd == "review-start":
                    rec = st["prs"][str(args.n)]
                    busy, eff = review.pipes(st), review.pipes_in_effect(cfg)
                    if control.halted(st):
                        print(f"REFUSED: {control.halted(st)}")
                        return 4
                    if review.pipe_of(st, args.n):
                        print(
                            f"REFUSED: #{args.n} is being reviewed already (rk review-status)"
                        )
                        return 4
                    # An in-depth review starts only once the operator approves it, whatever the options: neither a
                    # forced --kind follow-up of a PR's first review nor a forced --kind first-round of a reviewed PR
                    # gets around it.
                    kind = args.kind or review.round_kind(rec)
                    if review.needs_approval(cfg, st, args.n, kind):
                        print(
                            f"REFUSED: {review.approval_wait(cfg, st, store, args.n)}"
                        )
                        return 4
                    if len(busy) >= eff:
                        queue = f"rk review-queue {args.n}{' --again' if args.again else ''}"
                        print(
                            f"REFUSED: all {eff} parallel reviews are busy "
                            f"({', '.join('#%d' % p['pr'] for p in busy)}); queue it: {queue}. Parallel reviews: "
                            f"{config.pipes_line(cfg)}"
                        )
                        return 4
                    # The automatic path's settle, by hand too (lessons L-534: a review started a minute after a push
                    # dies with the next push).
                    left, waiting = settle_left(cfg, st, rec)
                    if left > 0 and not args.now:
                        print(
                            f"REFUSED: #{args.n}'s head {rec['head'][:10]} has not settled ({waiting}); it settles "
                            f"in {left:.0f} min. Start it then, or now anyway with --now"
                        )
                        return 4
                    # The driver's gap applies by hand too: one follow-up review per PR every followup_min_gap_min. A
                    # re-review (--again) does not wait for it.
                    until = (
                        plan_mod.gap_until(
                            cfg, st, rec, plan_mod.last_review_posts(store.ledger())
                        )
                        if kind == "follow-up" and not args.now and not args.again
                        else None
                    )
                    if until:
                        print(
                            f"REFUSED: #{args.n}'s follow-up waits for the {plan_mod.gap_label(cfg)} gap after our "
                            f"last review of it; it may start after {until[11:16]}Z. Start it then, or now "
                            "anyway with --now"
                        )
                        return 4
                    # It starts at the live head; a newer one than the state's must settle too, unless --now.
                    lines = review.start_review(
                        cfg, st, store, args.n, kind, force=args.now, again=args.again
                    )
                    review.first_started(st, store, args.n, lines)
                    for line in lines:
                        print(line)
                    refused = bool(lines) and lines[-1].startswith(review.NOT_STARTED)
                elif args.cmd == "post" and mirror.on(
                    cfg
                ):  # the post, its inputs and cost.json, not GitHub
                    text = pathlib.Path(args.file).read_text(encoding="utf-8")
                    try:
                        print(
                            "\n".join(
                                mirror.divert(
                                    cfg,
                                    st,
                                    store,
                                    args.n,
                                    {"text": text},
                                    None,
                                    again=args.again,
                                )
                            )
                        )
                    except (post_mod.PostRefused, gh.GhError) as exc:
                        print(f"REFUSED: {exc}")
                        return 4
                elif args.cmd == "post" and post_mod.keeps(cfg):
                    try:
                        print(
                            post_mod.keep_as_draft(
                                cfg, st, store, args.n, args.file, again=args.again
                            )
                        )
                    except post_mod.PostRefused as exc:
                        print(f"REFUSED: {exc}")
                        return 4
                elif args.cmd == "post":
                    try:
                        print(
                            "posted:",
                            post_mod.post(
                                cfg, st, store, args.n, args.file, again=args.again
                            ),
                        )
                    except post_mod.PostRefused as exc:
                        print(f"REFUSED: {exc}")
                        return 4
                    # Say at once when a review makes a PR an approval candidate.
                    for line in approval_news(cfg, st, only=args.n):
                        print(line)
                elif args.cmd == "review-skip":
                    for line in review.skip(st, store, args.n):
                        print(line)
                    print(f"review_skip: {st['review_skip']}")
                else:
                    st["review_skip"] = sorted(
                        set(st.get("review_skip", [])) - {args.n}
                    )
                    store.journal("review", f"{args.cmd} #{args.n}")
                    print(f"review_skip: {st['review_skip']}")
                store.save(st)  # a refused start keeps the live head it found
            return 4 if refused else 0
        if args.cmd == "review-status":
            st = store.load()
            flying = review.pipes(st)
            rows = report.in_review(
                cfg, st, store
            )  # the dashboards' In review box counts these
            print(
                f"in review: {len(rows)} ({report.in_review_counts(rows)})"
                + "".join(f"; #{r['pr']} {r['part']}" for r in rows)
            )
            print(f"parallel reviews: {config.pipes_line(cfg)}")
            for line in mirror.status_lines(cfg, st):
                print(line)
            print(f"skip: {st.get('review_skip', [])}")
            print("\n".join(review.queue_lines(cfg, st, store)))
            for r in review.waiting_on_verifier(st):
                print(
                    f"awaiting verification: #{r['pr']} {r['kind']} at {r['head'][:10]} since {r['at']}"
                    + (
                        f" (offload {r['offload']}"
                        + (f": {'; '.join(r['why'])}" if r.get("why") else "")
                        + ")"
                        if r.get("offload")
                        else ""
                    )
                )
            for k, e in sorted(draft_stage.pool(st).items(), key=lambda kv: int(kv[0])):
                print(
                    f"pool draft: #{k} {e['kind']} at {e['head'][:10]} since {e['started']}: "
                    f"{jobs.status(store, e.get('job'))} {(e.get('job') or {}).get('id', '')}"
                )
            for k, e in sorted(
                draft_stage.shadow_pool(st).items(), key=lambda kv: int(kv[0])
            ):
                print(
                    f"shadow draft: #{k} {e['kind']} at {e['head'][:10]} since {e['started']}: "
                    f"{jobs.status(store, e.get('job'))} {(e.get('job') or {}).get('id', '')}"
                )
            if not flying:
                print("no review in flight")
                return 0
            for pipe in flying:
                print(
                    f"#{pipe['pr']} {pipe['kind']} stage={pipe['stage']} head={pipe['head'][:10]} "
                    f"since={pipe['since']}"
                    + (
                        f" offload={pipe['offload']} round={pipe.get('round')}"
                        if pipe.get("offload")
                        else ""
                    )
                )
                for which in ("worker", "checks") + (
                    ("finish",) if pipe.get("finish") else ()
                ):
                    j = pipe.get(which)
                    print(f"  {which}: {jobs.status(store, j)} {j['id'] if j else ''}")
                    if j:
                        for line in jobs.tail(store, j, 4).splitlines():
                            print(f"    {line}")
                if pipe.get("plan"):
                    print(f"  checks plan: {pipe['plan']}")
            return 0
        if args.cmd == "werror":  # read-only: clang over the tree, its build's commands
            return cmd_werror(args, cfg)
        if args.cmd == "draft-lint":  # writes only <folder>/checks/lint.md
            ws = (
                pathlib.Path(args.dir)
                if args.dir
                else pathlib.Path(paths.get(cfg, "workers")) / f"pr{args.n}"
            )
            lines, code, _ = draft_lint.run(args.n, ws, config.tracker_keys(cfg))
            print("\n".join(lines))
            return code
        if args.cmd == "followup-scan":  # writes only the candidates file
            out = pathlib.Path(args.out) if args.out else None
            path, counts = followups.scan(cfg, store.load(), args.n, out)
            print(followups.line(args.n, counts, path))
            return 0
        if (
            args.cmd == "facts"
        ):  # brief facts, pasted verbatim (lessons L-201, L-536); writes nothing
            lines, failed = facts.render(cfg, store.load(), store, args.n)
            print("\n".join(lines))
            return 1 if failed else 0
        if (
            args.cmd == "review-prep"
        ):  # the briefs' inputs in one directory; writes nothing else
            st = store.load()
            rnd = args.round or prep.default_round(st, args.n, cfg)
            if not prep.ROUND.fullmatch(rnd):
                print(f"REFUSED: --round {rnd}: fr (or fr<k> for a re-review) or fu<k>")
                return 4
            lines, ok = prep.prepare(
                cfg,
                st,
                store,
                args.n,
                (
                    pathlib.Path(args.out)
                    if args.out
                    else prep.review_dir(cfg, args.n, rnd)
                ),
            )
            print("\n".join(lines))
            return 0 if ok else 1
        if args.cmd == "review-split":  # the mirror; writes only <review dir>/parts/
            st = store.load()
            out, rnd = split.default_out(cfg, st, args.n, args.round)
            if args.max_lines is None:  # the deploy settings' part size
                args.max_lines = split.part_lines(cfg)
            if (
                not prep.ROUND.fullmatch(rnd)
                or args.max_lines < 100
                or args.max_agents < 2
            ):
                print(
                    f"REFUSED: --round {rnd} (fr, fr<k> or fu<k>), --max-lines {args.max_lines} (100 or more), "
                    f"--max-agents {args.max_agents} (2 or more)"
                )
                return 4
            lines, code = split.run(
                cfg,
                st,
                args.n,
                pathlib.Path(args.out) if args.out else out,
                rnd,
                args.max_lines,
                args.max_agents,
                args.team_file,
                args.head,
                store=store,
            )
            print("\n".join(lines))
            return code
        if (
            args.cmd == "spec"
        ):  # GitHub GETs and the mirror; writes the spec (and the post's ✓ count when missing)
            cost = (
                None
                if args.cost_verifier is None and args.cost_quality is None
                else {"verifier": args.cost_verifier, "quality": args.cost_quality}
            )
            folder = args.dir or str(prep.review_dir(cfg, args.n, args.round))
            lines, code = spec_mod.write(
                cfg, args.n, args.round, folder, args.out, args.light, args.save, cost
            )
            print("\n".join(lines))
            return code
        if (
            args.cmd == "dispatch-pack"
        ):  # rk review-prep, then the briefs' inputs in one compact print
            st = store.load()
            rnd = args.round or prep.default_round(st, args.n, cfg)
            if not prep.ROUND.fullmatch(rnd):
                print(f"REFUSED: --round {rnd}: fr (or fr<k> for a re-review) or fu<k>")
                return 4
            lines, ok = dispatch.pack(
                cfg, st, store, args.n, rnd, prep.review_dir(cfg, args.n, rnd)
            )
            print("\n".join(lines))
            return 0 if ok else 1
        if (
            args.cmd == "spec-post"
        ):  # rk spec, then rk post --premises when the spec has no CHECK line
            cost = (
                None
                if args.cost_verifier is None and args.cost_quality is None
                else {"verifier": args.cost_verifier, "quality": args.cost_quality}
            )
            folder = args.dir or str(prep.review_dir(cfg, args.n, args.round))
            lines, code = dispatch.spec_post(
                cfg,
                store,
                args.n,
                args.round,
                folder,
                args.out,
                args.light,
                args.save,
                cost,
                args.again,
                args.dry_run,
                args.devbase_recheck,
                approval_news,
            )
            print("\n".join(lines))
            return code
        if (
            args.cmd == "wait-quality"
        ):  # reads the review directory; writes only its wait marker
            line, code = dispatch.wait_quality(
                pathlib.Path(args.dir),
                args.max_min,
                args.slice_min or dispatch.SLICE_MIN,
            )
            print(line)
            return code
        if args.cmd == "shard-times":  # live reads
            from rkmon_lib import shard_times

            lines, ok = shard_times.report(cfg, args.n, every=args.every)
            print("\n".join(lines))
            return 0 if ok else 1
        if args.cmd in ("heads", "ci-fails", "job-log"):  # live reads
            lines, ok = (
                facts.heads(cfg, args.ns)
                if args.cmd == "heads"
                else (
                    ci_runs.fails(cfg, args.n)
                    if args.cmd == "ci-fails"
                    else ci_runs.job_log(
                        cfg,
                        args.id,
                        errors=args.errors,
                        files=args.files or job_log_files(cfg, store, args.pr),
                    )
                )
            )
            print("\n".join(lines))
            return 0 if ok else 1
        if args.cmd == "evidence-check" and not (
            args.offload or args.round
        ):  # a draft's evidence: writes only the
            lines, code = evidence.cmd(
                cfg, args.n, args.dir
            )  # run's folder and its kept copy
            print("\n".join(lines))
            return code
        if args.cmd == "stage-ours":  # GitHub GETs; writes only DIR/ours.md
            print(
                draft_stage.stage_ours(
                    cfg, store.load(), args.n, pathlib.Path(args.dir)
                )
            )
            return 0
        if args.cmd == "drafts":  # read-only
            print("\n".join(drafts_lines(cfg, store)))
            return 0
        if (
            args.cmd == "evidence-check"
        ):  # the review offload's evidence, with no model; writes its report
            if args.round and not prep.ROUND.fullmatch(args.round):
                print(f"REFUSED: --round {args.round}: fr (or fr<k>) or fu<k>")
                return 4
            from rkmon_lib import offload_evidence

            lines, code = offload_evidence.cmd(cfg, args.n, args.round)
            print("\n".join(lines))
            return code
        if (
            args.cmd == "shadow-report"
        ):  # the shadow trial's comparison; GitHub reads, writes its pages
            from rkmon_lib import shadow

            lines, code = shadow.cmd(cfg, store.load(), store, args.ns or None)
            print("\n".join(lines))
            return code
        if args.cmd == "classify":
            with store.locked(wait=True):
                st = store.load()
                rec = st["prs"][str(args.n)]
                if args.record:
                    # --record classifies the live head, which the tick must have seen: run between a push and the
                    # next tick, it would otherwise record the state's older head (lessons L-594).
                    try:
                        live = review.live_head(cfg, args.n)
                    except gh.GhError as exc:
                        print(
                            f"REFUSED: #{args.n}'s live head was not read ({str(exc)[:120]}); nothing recorded"
                        )
                        return 4
                    if not snapshot.same_commit(live, rec["head"]):
                        print(
                            f"REFUSED: #{args.n}'s live head is {live[:10]}, but the state's head is "
                            f"{rec['head'][:10]}: the tick has not seen {live[:10]} yet. Run `rk tick`, then "
                            f"`rk classify {args.n} --record` again; nothing recorded"
                        )
                        return 4
                c = review.classify_pr(cfg, rec)
                print(json.dumps(c, indent=1))
                if (
                    args.record
                    and c.get("new")
                    and not snapshot.same_commit(c["new"], live)
                ):
                    print(
                        f"REFUSED: the clone's PR ref is {c['new'][:10]}, not the live head {live[:10]}; "
                        "nothing recorded"
                    )
                    return 4
                if args.record and c["kind"] in (
                    "rebase-only",
                    "rebase+re-record",
                    "merge-only",
                ):
                    if not snapshot.covers(rec["ours"], c["new"]):
                        rec["ours"].setdefault("equivalent_heads", []).append(c["new"])
                    store.journal(
                        "review",
                        f"#{args.n} {c['kind']} since our review: {c['new'][:10]} recorded as "
                        "equivalent (manual classify)",
                    )
                    daylog.add(
                        store,
                        cfg,
                        "rebase",
                        args.n,
                        f"{c['kind']} since our review; verdict kept at {c['new'][:10]}",
                    )
                    # A review of that head already drafted or in flight is moot: it is closed without a post.
                    closed = review.close_review(
                        st, store, args.n, c["new"], f"{c['kind']} since our review"
                    )
                    store.save(st)
                    print("recorded as equivalent: our review stands for the new head")
                    for line in closed:
                        print(line)
            return 0
        if args.cmd in ("queue-advance", "gate", "arm", "disarm", "queue-notice"):
            with store.locked(wait=True):
                st = store.load()
                q = queue_mod.build_queue(cfg, st, st["develop"]["sha"])
                try:
                    if args.cmd == "queue-advance":
                        for line in merge.advance(cfg, st, store, q, []):
                            print(line)
                    elif args.cmd == "gate":
                        why = heavy_start_refusal(cfg, st, store, args.n, args.now)
                        if why:
                            print(f"REFUSED: {why}")
                            return 4
                        for line in merge.start_gate(cfg, st, store, args.n):
                            print(line)
                    elif args.cmd == "arm":
                        print(
                            post_mod.outcome(
                                "armed:", post_mod.arm(cfg, st, store, args.n)
                            )
                        )
                    elif args.cmd == "disarm":
                        print(
                            post_mod.disarm(
                                cfg,
                                st,
                                store,
                                args.n,
                                " ".join(args.reason) or "manual",
                            )
                        )
                    else:
                        want = merge.desired_notices(cfg, st, q)
                        if args.n not in want:
                            print(f"#{args.n} needs no queue notice right now")
                            return 0
                        state, detail = want[args.n]
                        text = merge.notice_text(cfg, st, args.n, state, detail)
                        import hashlib

                        key = hashlib.sha1(
                            f"{state}:{detail}:{st.get('_notice_order', {}).get(str(args.n))}".encode()
                        ).hexdigest()[:10]
                        print(
                            post_mod.outcome(
                                "notice:",
                                post_mod.queue_notice(
                                    cfg, st, store, args.n, text, key
                                ),
                            )
                        )
                except post_mod.PostRefused as exc:
                    print(f"REFUSED: {exc}")
                    return 4
                store.save(st)
            return 0
        if args.cmd == "guard-advance":
            with store.locked(wait=True):
                st = store.load()
                for line in guard.advance(cfg, st, store, []):
                    print(line)
                store.save(st)
            return 0
        if args.cmd == "review-advance":
            with store.locked(wait=True):
                st = store.load()
                if stopped(st):
                    print(f"STOPPED: {control.halted(st)}")
                    return 0
                stale = review.tick_stale(
                    st
                )  # the state may be out of date: start nothing, change nothing
                if stale:
                    print(stale)
                    return 0
                lines = review.advance(cfg, st, store, st.get("last_plan", []))
                for line in lines:
                    print(line)
                daylog.from_lines(store, cfg, st, lines)
                store.save(st)
            return 0
        if args.cmd == "cleanup":
            with store.locked(wait=True):
                st = store.load()
                for line in cleanup.sweep(
                    cfg,
                    st,
                    store,
                    pathlib.Path(paths.get(cfg, "working_dir")),
                    limit=None if args.all else cleanup.PER_TICK,
                    dry=args.dry_run,
                ):
                    print(line)
                if not args.dry_run:
                    store.save(st)
            return 0
        if args.cmd in ("log", "save"):
            with store.locked(wait=True):
                pr = None if args.pr == "-" else int(args.pr.lstrip("#"))
                if args.cmd == "save":
                    rec = daylog.save(store, cfg, pr, args.type, " ".join(args.text))
                else:
                    rec = daylog.add(store, cfg, args.type, pr, " ".join(args.text))
            print(daylog.logged_line(rec, args.pr))
            return 0
        if args.cmd == "daylog":
            day = args.day or now()[:10]
            path = daylog.render(store, cfg, day)
            print(f"{path}\n{daylog.summary(daylog.entries(store, day))}")
            return 0
        if args.cmd.startswith("gpu-"):
            return cmd_gpu(args, cfg, store)
        if args.cmd == "jobs":
            d = store.dir / "jobs"
            metas = sorted(
                (
                    json.loads((p / "meta.json").read_text())
                    for p in d.iterdir()
                    if (p / "meta.json").exists()
                ),
                key=lambda m: m["started"],
            )
            if args.id:
                m = next(m for m in metas if m["id"] == args.id)
                print(jobs.status(store, m))
                print(jobs.tail(store, m, 40))
                return 0
            for m in metas[-15:]:
                print(f"{m['started']} {jobs.status(store, m):<14} {m['id']}")
            return 0
        if args.cmd in (
            "sanity",
            "bisect",
            "incident-note",
            "incident-close",
            "notice",
        ):
            with store.locked(wait=True):
                st = store.load()
                if args.cmd in ("sanity", "bisect"):
                    why = heavy_start_refusal(cfg, st, store, 0, args.now)
                    if why:
                        print(f"REFUSED: {why}")
                        return 4
                if args.cmd == "sanity":
                    sha = args.sha or st["develop"]["sha"]
                    full = gitops.git(cfg["paths"]["repo"], "rev-parse", sha).strip()
                    base = (
                        gitops.git(cfg["paths"]["repo"], "rev-parse", args.base).strip()
                        if args.base
                        else None
                    )
                    for line in guard.start_sanity(
                        cfg, st, store, full, base=base, rehearsal=args.rehearsal
                    ):
                        print(line)
                elif args.cmd == "bisect":
                    for line in guard.start_bisect(cfg, st, store):
                        print(line)
                elif (
                    args.cmd == "notice"
                    and args.kind == "gpu"
                    and not features.on(cfg, "review-posting")
                ):
                    store.journal(
                        "gpu",
                        f"#{args.n} PROTOTYPE: would post the GPU test notice {args.file} ({args.key})",
                    )
                    print(
                        f"PROTOTYPE: review automation is in prototype: the GPU test notice for #{args.n} was not "
                        "posted (journaled)"
                    )
                elif args.cmd == "notice":
                    try:
                        print(
                            post_mod.outcome(
                                "posted:",
                                post_mod.notice(
                                    cfg,
                                    st,
                                    store,
                                    args.n,
                                    args.file,
                                    args.kind,
                                    args.key,
                                ),
                            )
                        )
                    except post_mod.PostRefused as exc:
                        print(f"REFUSED: {exc}")
                        return 4
                else:
                    inc = guard.incidents(store)
                    rec = next((i for i in inc if i.get("id") == args.id), None)
                    if not rec:
                        print(f"no incident {args.id}")
                        return 1
                    if args.cmd == "incident-close":
                        rec.update(status="closed", end=rec.get("end") or now())
                        if args.fixed_by:
                            rec["fixed_by"] = args.fixed_by
                        if (st.get("frozen") or {}).get("incident") == rec["id"]:
                            st["frozen"] = None
                    else:
                        val = " ".join(args.value)
                        if args.key in ("culprit", "fixed_by", "issue"):
                            rec[args.key] = int(val.lstrip("#"))
                            if args.key == "issue":
                                rec.setdefault("actions", []).append("issue")
                        elif args.key == "revert_offered":
                            rec.setdefault("actions", []).append("revert_offered")
                        else:
                            rec[args.key] = val
                            rec.setdefault("actions", []).append(args.key)
                    guard.save_incidents(store, inc)
                    store.journal("INCIDENT", f"{args.cmd} {args.id}: {vars(args)}")
                    print(json.dumps(rec, indent=1)[:800])
                store.save(st)
            return 0
        if args.cmd in ("sanity-status", "incidents"):
            st = store.load()
            g = st.get("guard", {})
            if args.cmd == "sanity-status":
                run = g.get("running")
                print(
                    f"last green sanity: {(g.get('last_sanity_green') or '-')[:12]}; pending: "
                    f"{[x[:10] for x in st['develop'].get('pending_sanity') or []]}"
                )
                if run:
                    print(
                        f"running at {run['sha'][:12]}: {jobs.status(store, run['job'])}; landed {run['landed']}; "
                        f"families {run['families']}; full={run['full']}"
                    )
                    for line in jobs.tail(store, run["job"], 6).splitlines():
                        print(f"   {line}")
                return 0
            for i in guard.incidents(store):
                print(
                    f"{i.get('id', '-')} {i.get('status', 'closed')} start={i['start'][:16]} end={(i.get('end') or '-')[:16]} "
                    f"culprit=#{i.get('culprit')} fixed_by=#{i.get('fixed_by')} jira={i.get('jira')} :: "
                    f"{(i.get('summary') or '')[:100]}"
                )
            return 0
        if args.cmd == "stats":
            st = store.load()
            lab, path = write_weekly(cfg, st, store, args.week)
            print(path.read_text(encoding="utf-8"))
            print(f"(written to {path})")
            return 0
        if (
            args.cmd == "spend"
        ):  # local: the transcripts, the ledger and git; writes its caches and files only
            try:
                print(spend.cli(cfg, store, args), end="")
            except ValueError as exc:  # no price table, no projects folder
                print(f"spend: {exc}")
                return 1
            return 0
        if args.cmd in ("scoreboard", "scoreboard-backfill"):
            sstore = Store(pathlib.Path(args.state)) if args.state else store
            if args.cmd == "scoreboard":  # read-only: the events file
                print("\n".join(scoreboard.lines(scoreboard.load(sstore), args.days)))
                return 0
            if args.days < 1:
                print("REFUSED: --days must be 1 or more")
                return 4
            # The reading (GitHub GETs, git reads) runs without the lock, so no tick waits for it.
            res = scoreboard.backfill(
                cfg, sstore.load(), sstore, args.days, not args.no_github
            )
            if args.dry_run:
                new = scoreboard.new_events(sstore, res["found"])
            else:
                with sstore.locked(wait=True):
                    st = sstore.load()
                    new = scoreboard.record(sstore, st, res)
                    sstore.save(st)
            print("\n".join(scoreboard.backfill_lines(sstore, res, new, args.dry_run)))
            return 0
        if args.cmd == "lessons":
            start, end, lab = stats.week_bounds(args.week)
            print(lessons.collect(cfg, store, start, end, lab))
            return 0
        if args.cmd in ("conflict-files", "ci-times"):  # on demand
            import datetime as dt

            end = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(days=args.days)
            if args.cmd == "conflict-files":
                hot = conflicts.hot_files(
                    store, start, end, rocke_path=paths.repository_folder(cfg)
                )
                try:
                    hot["offenders"] = conflicts.offenders(
                        cfg["paths"]["repo"],
                        [f["path"] for f in hot["files"]],
                        start,
                        end,
                    )
                except conflicts.GIT_ERRORS as exc:
                    print(f"(develop's last changes not read: {exc})")
                print(
                    "\n".join(
                        conflicts.render_md(
                            hot, f"## Merge-conflict hot files, last {args.days} days"
                        )
                    )
                )
                return 0
            if args.collect or not ci_times.summary(store, start, end)["heads"]:
                with store.locked(wait=True):
                    st = store.load()
                    res = ci_times.collect(cfg, st, store, since=start)
                    store.save(st)
                print(
                    f"(read {res['recorded']} heads in {res['calls']} API calls; {res['pending']} still running, "
                    f"{res['left']} left for the next pass"
                    + (f"; stopped: {res['error']}" if res["error"] else "")
                    + ")"
                )
            st = store.load()
            who = (
                lambda n: f"#{n} ({(st['prs'].get(str(n)) or {}).get('author', '?')})"
            )  # noqa: E731
            print(
                "\n".join(
                    ci_times.render_md(
                        ci_times.summary(store, start, end),
                        who,
                        f"## CI time per PR, heads pushed in the last {args.days} days",
                    )
                )
            )
            return 0
        if args.cmd == "incident-add":
            with store.locked(wait=True):
                inc = stats.store_incidents(store)
                rec = {
                    "start": args.start,
                    "end": None if args.end == "-" else args.end,
                    "culprit": None if args.culprit == "-" else int(args.culprit),
                    "fixed_by": None if args.fixed_by == "-" else int(args.fixed_by),
                    "contributing": [int(x) for x in args.contributing.split(",") if x],
                    "summary": " ".join(args.summary),
                    "recorded": now(),
                }
                inc.append(rec)
                guard.save_incidents(store, inc)
                store.journal("incident", json.dumps(rec))
            print(json.dumps(rec))
            return 0
        if args.cmd == "breakages":
            with store.locked(wait=True):
                st = store.load()
                for line in breakage.poll(cfg, st):
                    print(line)
                store.save(st)
            for line in breakage.issue_lines(
                st["post_merge"]["open"], guard.incidents(store)
            ):
                print(line)
            return 0
        if args.cmd == "gate-cancel":
            with store.locked(wait=True):
                st = store.load()
                print(merge.cancel_gate(st, store, " ".join(args.reason)))
                store.save(st)
            return 0
        if args.cmd == "approvals":
            st = store.load()
            appr = report.approval_candidates(cfg, st)
            for a in appr:
                print(report.approval_line(a))
            print(
                f"({len(appr)} ready for your approval)\n\nClose to merge, what is left (who is next):"
            )
            for r in report.close_to_merge(cfg, st)[:25]:
                left = (
                    "; ".join(f"{b['what']} [{b['who']}]" for b in r["blockers"])
                    or "nothing we can see"
                )
                print(f"#{r['n']} {r['verdict']} {r['author']}: {left}")
            return 0
        if args.cmd in ("fix-link", "fix-dismiss"):
            with store.locked(wait=True):
                st = store.load()
                try:
                    if args.cmd == "fix-link":
                        print(
                            post_mod.outcome(
                                "linked:",
                                post_mod.fix_link(
                                    cfg, st, store, args.issue, args.pr, args.evidence
                                ),
                            )
                        )
                    else:
                        print(breakage.fix_dismiss(st, store, args.issue, args.pr))
                except (post_mod.PostRefused, OSError) as exc:
                    print(f"REFUSED: {exc}")
                    return 4
                finally:  # a link recorded before a failed write stands; a re-run does the rest
                    store.save(st)
            return 0
        if args.cmd in outward_mod.names(handler=outward_mod.POST_LOCKED):
            with store.locked(wait=True):
                st = store.load()
                try:
                    if args.cmd == "request-changes":
                        print(
                            post_mod.outcome(
                                "changes requested:",
                                post_mod.request_changes(
                                    cfg, st, store, args.n, args.file, user=args.user
                                ),
                            )
                        )
                    elif args.cmd == "issue-assign":
                        print(post_mod.issue_assign(cfg, store, args.n, args.login))
                    elif args.cmd == "issue-relabel":
                        print(
                            post_mod.issue_relabel(
                                cfg, st, store, args.n, args.to, args.assign
                            )
                        )
                    elif args.cmd == "label-create":
                        print(
                            post_mod.label_create(
                                cfg, store, args.name, args.description, args.color
                            )
                        )
                    elif args.cmd == "teams-test":
                        print(post_mod.teams_test(cfg, store))
                    elif args.cmd == "teams-breakage":
                        print(
                            post_mod.teams_breakage(
                                cfg, st, store, args.n, force=args.force
                            )
                        )
                    elif args.cmd == "issue-file":
                        print(
                            post_mod.outcome(
                                "filed:",
                                post_mod.issue_file(
                                    cfg, st, store, args.file, new=args.new
                                ),
                            )
                        )
                    elif args.cmd == "issue-comment":
                        print(
                            post_mod.outcome(
                                "commented:",
                                post_mod.issue_comment(
                                    cfg, st, store, args.n, args.file, args.key
                                ),
                            )
                        )
                    elif args.user:
                        print(
                            post_mod.automerge_off_user(
                                cfg, st, store, args.n, args.user
                            )
                        )
                    else:
                        print(post_mod.automerge_off(cfg, st, store, args.n))
                except (post_mod.PostRefused, ValueError) as exc:
                    print(f"REFUSED: {exc}")
                    return 4
                store.save(st)
            return 0
        with store.locked():
            st = store.load()
            if args.cmd == "poll":
                for e in do_poll(cfg, store, st):
                    print(json.dumps(e, ensure_ascii=False))
            else:
                for r in do_queue(cfg, st):
                    print(
                        f"{r.get('position') or '-':>3} #{r['n']:<6} {r['author']:<20} d={r['disruption']} "
                        f"{'; '.join(r['blockers'])[:110]}"
                    )
            store.save(st)
        return 0
    except BlockingIOError:
        print(f"TICK {now()} {tickfilter.TICK_RUNNING}; skipped")
        return 0
    except (gh.GhError, gitops.GitError) as exc:
        store.journal("CRITICAL", f"tool failure: {exc}")
        print(f"!! CRITICAL TOOL failure: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
