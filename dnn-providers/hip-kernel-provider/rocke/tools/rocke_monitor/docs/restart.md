<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Stopping and restarting the monitor

How to stop the rocKE monitor, how far each kind of stop goes, and how to bring it back after a reboot, a crash or a
container restart. On this page `rk` is the monitor's own command, `<RK>` (that is, `<TOOLKIT_DIR>/rk`); `rk --help`
lists every command. In the container, `docker/rkmon rk ARGS` runs any of them (`docker/rkmon help` lists the
wrapper's own commands).

## Is it safe to restart?

Yes. Every outward GitHub write is one `rk` call, made under the write lock and recorded in the ledger, so no write is
ever half done, and nothing is posted twice after a restart. A restart only interrupts local work: a draft the reviewer
model is writing, a gate, sanity or check job, the review driver, the tick loop, and the Claude session (the
container's orchestrator, or Claude Code on a host install) with any subagent it runs.

1. **Note what is in flight:** `rk status`, `rk review-status`, `rk jobs`.
2. **Stop cleanly:** `docker/rkmon stop --drain` starts nothing new, lets the work in flight finish, then stops the
   monitor and its container; `rk stop --drain` does the same for the monitor alone. How long each waits, and what each
   does when the work takes longer or the stop cannot be made: [The four stop levels](#the-four-stop-levels) and
   [In the container](#in-the-container).
3. **With the reviewer model on this machine's GPU,** an idle GPU is the gentlest state to shut down from. Restarting
   the model server while the GPU is hung in a prompt can take the whole host down.

## The four stop levels

| Level | Command | What it does | What stays | Back with |
|---|---|---|---|---|
| Pause | `rk pause` | The tick makes no decisions and starts nothing new | Running jobs finish; the admin dashboard, and in the container the orchestrator, keep running | `rk resume` |
| Stop after the work in flight | `rk stop --drain` | Nothing new starts; the work in flight finishes, then it stops as `rk stop` does | As for a stop | `rk start` |
| Stop | `rk stop` | Stops the running pieces now, and names each in a `STOPPED` line | State and configuration; reviews awaiting verification | `rk start` |
| Kill | `rk kill` | Kills every process the monitor started, at once, and prints what it killed | State and configuration | `rk start` |

- `rk stop --drain` waits for the work in flight for at most `--timeout-minutes N` minutes, or the package's default
  ([Internal settings](internal_settings.md#package-defaults)), with its lines: `drain: ...`, then `DRAINED`, or
  `DRAIN TIMEOUT` and the work still in flight. Then it stops as `rk stop` does, what is still in flight with it, and
  lifts the drain in the same write that records the stop, so that nothing starts in between. It stops nothing while
  another drain is in force (`REFUSED`, exit 4), when `rk start` lifts its drain meanwhile (`UNDRAINED`, exit 1), or
  when Ctrl-C ends its drain, which lifts the drain. When the configuration cannot be loaded, no review in flight can
  finish: it says `NOT DRAINED`, why and the fix, and stops at once.
- `rk stop` stops the review driver, the work in flight, the gate, sanity and check jobs, the admin dashboard
  server, and in the container the orchestrator; its last line says what became of the Claude side. When the state
  lock is still in use after 4 minutes, it says `NOT STOPPED` (exit 1): the monitor is not marked stopped, only its
  tick, review advance and review driver were ended, and a drain of `rk stop --drain` stays in force, so nothing new
  starts, until `rk start`; run `rk stop` again, or `rk kill`.
- `rk kill` first waits a short, bounded time (`--wait` seconds, default 30) for a GitHub write in flight, so none is
  cut in half. It never touches a process the monitor did not start: the Ollama server, Claude Code, your shells.
- `rk status` shows which state the monitor is in: on or off; stopped or draining; the pieces running (see
  [Checks](#checks)).
- A stop or a kill ends only the processes the monitor registered: its jobs, the review driver, `rk serve`, a running
  `rk tick` or `rk review-advance`, `rk stop --drain`, and in the container the orchestrator (`rk orchestrate`), each
  with the processes it started. A stop also ends a tick still running after two minutes.
- After a stop or a kill, each review in flight starts again at its head after `rk start`, before any new review,
  unless its PR closed or its head moved meanwhile; a draft awaiting verification stays, and its review directory comes
  back from its disk copy when the RAM disk was emptied; a gate is gated again; a develop sanity run goes back to
  pending; a bisect is dropped (run `rk bisect` again).
- A reboot ends a stop or a kill, as each is recorded with the boot it happened in; in the container, the container's
  start runs `rk start` too. To keep the monitor from starting work after a reboot, pause it (`rk pause`), or stop its
  container (`docker/rkmon stop`), which stays stopped across reboots. `rk stopped` exits 0 and says since when while a
  stop or a kill is in force.
- While the monitor is stopped, `rk tick` and the review driver do nothing, and the guarded GitHub writes are refused
  until `rk start` (taking auto-merge off, `rk disarm` and `rk automerge-off`, still works).
- Every outward GitHub write is one `rk` call under the write lock, recorded in the ledger. That is why a stop or a kill
  first waits, up to 30 seconds, for a write in flight (one still running then is left alone to finish), and why a
  restart never repeats one.
- Remote GPU jobs (`features.review_automation.remote_gpu_tests`) run on the cluster, not here: a stop or a kill leaves
  them to finish (no stop waits on the network), and the first tick after `rk start` collects them. `rk gpu-cancel all`
  gives up on them.
- **The Claude side.** On a host install, the Claude Code session's loops are stopped from Claude Code:
  `/rocke-monitor off` ends the session's tick loop and watchdog cron, then runs `rk pause`; `/rocke-monitor stop` and
  `/rocke-monitor kill` end them too, then run `rk stop` or `rk kill`. `rk` never touches Claude Code on a host: a loop
  left running keeps ticking, and its ticks do nothing while the monitor is off or stopped. In the container the
  Claude session is the orchestrator's, a process the monitor started: `rk stop`'s last line says whether the stop
  ended the session, or left it idle because the session ran the stop itself, and that `rk start`, or
  `docker/rkmon start` on the host, brings it back ([In the container](#in-the-container)).

## After a restart, on a host install

A host install runs the monitor without a container, until it moves into one
([QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor"](../QUICK_DEPLOY_GUIDE.md#moving-from-an-existing-monitor)).
Nothing of it comes back by itself:

1. Run `rk start`. It clears a stop or a drain, says which settings it loaded, and starts the admin dashboard's server
   again. `<PACKAGE_DIR>/tools/rkweb.py` starts that server alone, and says "already running" if it is up. A monitor
   you paused (`rk pause`) stays paused: `rk resume` ends the pause.
2. Open `<ADMIN_URL>` in a browser on this machine: the admin dashboard, the only page with the Approve and Decline
   buttons for priority requests, proposed tickets and the in-depth reviews of large pull requests.
3. Open Claude Code in `<CLAUDE_WORKSPACE>` and run `/rocke-monitor resume`. It reads `rk status`, the journal tail and
   the jobs in flight, and continues from the recorded stage; the ledger keeps anything from being posted twice. It
   starts the tick loop (`<PACKAGE_DIR>/tools/tick_loop.py`, in the background) when `rk loop-status` shows none, and
   creates the watchdog cron (every 2 hours) when there is none. The session keeps one review driver running while
   reviews wait, and the next review advance retries, once, a draft whose job died.

The tick loop runs `rk tick` at each tick interval (10 minutes by default) and wakes the session only for the lines
that need it, or with a heartbeat. The watchdog cron runs `rk loop-status` and starts the loop again when none runs, or
when the last tick is more than 25 minutes old. Both live in the Claude Code session: a new session needs
`/rocke-monitor resume`, and so does one after a compaction or any interruption. Each tick reloads the skill and reads
the state from the disk, so nothing depends on the conversation's memory.

## In the container

The container has no tick loop script and no watchdog cron: its entry point keeps the monitor running. At each start
it runs `rk start`, installs the worker tools again, then keeps the parts the deployment's features run
(`rk features --runs PART`):

- the loop that restarts the admin dashboard, which, with the admin dashboard on, starts its service again within a
  minute whenever it is down;
- the orchestrator (`rk orchestrate`), a long-lived Claude session, while review automation is enabled and once
  `rk orchestrate --check` says it is set up (it names what is missing, such as the session's key, and the fix);
- the tick loop that runs without the orchestrator: every `monitor.tick_interval_minutes` minutes (10; `rk tick-min`
  prints the value in effect) it runs a tick itself while the orchestrator is not set up, or when the last tick is over
  twice that plus 5 minutes old (25 minutes: `rk tick-due`), so the dashboards stay current. With review automation
  off and the merge queue on, each tick is run by fixed rules (`rk rules-tick`). It waits between its ticks with
  `rk tick-wait`, which takes `rk tick-now`'s request, as the orchestrator does: a tick at once.

Each part that dies is started again, after a pause that grows from 10 seconds to 5 minutes, except while an `rk stop`
or `rk kill` is in force: then the orchestrator, the admin dashboard's service and the ticks stay down until
`rk start`. The restart policy (`docker/compose.yaml`, `unless-stopped`) brings the container back when Docker starts
after a reboot, and after a crash, but not after you stopped it yourself.

**While the configuration cannot be read**, `rk start` is refused (`REFUSED`, exit 4, with why and the fix), and
nothing runs: no admin dashboard, no Claude session, no tick. The orchestrator and the tick loop each say so once in
their logs and look again every minute; `docker/rkmon doctor` shows the FAIL and its fix.

`docker/rkmon` is the package's wrapper of docker compose (from any folder; each command prints the docker command it
runs):

| In the container | Command | What happens |
|---|---|---|
| To pause | `docker/rkmon pause` | `rk pause` in the running container: `PAUSED`, or `NOTHING TO DO` when it is paused already; `docker/rkmon resume` ends the pause |
| To stop | `docker/rkmon stop` | Stops the monitor, then the container, which stays stopped, across reboots too, until `docker/rkmon start` |
| To stop after the work in flight | `docker/rkmon stop --drain` | Lets the work in flight finish, then stops as `docker/rkmon stop` does |
| To start | `docker/rkmon start` | Starts the container with the settings you confirmed: [QUICK_DEPLOY_GUIDE.md, "Step 6: start"](../QUICK_DEPLOY_GUIDE.md#step-6-start) |
| To change the image | `docker/rkmon image load IMAGE_FILE` (or `image pull`), `image.tag` set to its tag, then `docker/rkmon confirm` and `docker/rkmon apply` | Makes the container again on the new image, or puts back what ran before (`ROLLED BACK`) |
| To kill | `docker/rkmon rk kill` | `rk kill` in the container; the container stays up, and nothing comes back by itself |
| To remove | `docker/rkmon remove` | Removes the container; its volumes stay, unless `--delete-data` |

- `docker/rkmon stop`: the entry point stops the orchestrator, the loop that restarts the admin dashboard and the tick
  loop, then runs `rk stop` (after any GitHub write in flight) and waits up to 2 minutes more for a write `rk stop`
  left to finish before it ends what is left; docker waits up to 10 minutes (`stop_grace_period`, over `rk stop`'s
  worst case: a write in flight and the state lock). It ends with `STOPPED`; `NOTHING TO DO` when the container does
  not run; `ERROR` when docker cannot stop it.
- `docker/rkmon stop --drain` runs `rk drain` in the container, its lines shown, then `drain: done`, for at most
  `--timeout-minutes N` minutes (`docker/rkmon stop --help` shows the default); then `rk stop` in the container, which
  lifts the drain as it records the stop, so that nothing starts in between; then the container stops. When the work
  has not finished in time, it stops nothing and lifts the drain (`ERROR`, exit 1; the next line offers twice the
  time, or the same command later), and so when another command lifts the drain (`ERROR`, exit 1); while another drain
  is in force, it stops nothing (`REFUSED`, exit 4). Ctrl-C during the drain lifts it (`INTERRUPTED`, exit 130,
  nothing changed); once the monitor or the container is stopping, Ctrl-C waits for that stop to end, then starts the
  monitor again while the container runs.
- `docker/rkmon start`: its entry point runs `rk start`, which records the settings it loaded, your confirmed settings,
  as the ones applied (`rk applied` shows the record), then keeps the parts above. A paused monitor stays paused:
  `docker/rkmon start --on` ends the pause too. A monitor moved from an existing one, in place or with
  `docker/rkmon adopt`, starts in standby: `rk start` makes the one-time move of the older install's settings first, and
  the monitor sends nothing until `docker/rkmon standby off`
  ([QUICK_USER_GUIDE.md, "Standby"](../QUICK_USER_GUIDE.md#standby)).
- A new image: the load or the pull checks that it was built from this package's image recipe. Then apply runs the
  doctor and a drain (the work in flight finishes, for at most `--timeout-minutes N` minutes:
  `docker/rkmon apply --help` shows the default), writes the settings, makes the container again on the new image,
  then checks the monitor's load of the settings and runs the doctor again; a load that fails or a new doctor failure
  puts back what ran before (`ROLLED BACK`). New code with no new image: [README.md, "Updating"](../README.md#updating).
- `docker/rkmon rk kill` kills the orchestrator and its Claude session among the rest. The entry point then leaves the
  orchestrator, the admin dashboard's service and the ticks down, and the review driver, new jobs and the guarded
  writes do nothing, until `docker/rkmon start` or `docker/rkmon rk start`. `docker/rkmon rk stop` in a running
  container works the same way.
- `docker/rkmon remove --delete-data` removes the volumes too (the state, the clone, the secrets and the logins): it
  asks first on a terminal, and needs `--yes` otherwise. Before the first start there is no container, and
  `--delete-data` removes the volumes `docker/rkmon login` made.

`docker/rkmon status` shows the deployment's state, then, while the container runs, `rk status`; `docker/rkmon doctor`
shows the state's row, then `rk doctor`'s report; `docker/rkmon logs -f` follows the container's log: the entry point's
own lines and the orchestrator's lean per-tick output. Each part logs to `/work/logs`: `start.log`, `stop.log`,
`web.log` (the loop that restarts the admin dashboard), `orchestrator.log` (the orchestrator's output), `tick.log` (the
tick loop that runs without the orchestrator) and `rk-write.log` (each request to the write broker); `restarts`
records each restart of a kept part, which the health check counts. In the container `rk status` shows
`tick loop: not running`: that line is about a host install's `tools/tick_loop.py`, which the container does not use,
as the orchestrator and the entry point tick instead; its last tick is the time to watch.

**What a container restart keeps.** Everything on the five volumes: `/work` (the monitor's home `/work/monitor` with
`state/`, the clone, the workers' folders, the saved probes, the posts, the caches, the logs, and Claude Code's own
state with the orchestrator's session), `/secrets`, `/secrets-write` (the writer's secrets), `/code` (the code
releases) and `/etc/rkmon/settings` (your confirmed settings). The RAM disk does not survive: it is a tmpfs,
emptied whenever the container stops. What was on it (the PR worktrees and builds behind their links, and the
temporary folder with the review directories) is made again on demand: `worker/mkwt.py` makes a worktree again when
its link points at nothing, and `rk review-prep` makes a review directory again; that of a review still in flight
comes back from its disk copy (`<WORKING_DIR>/kept_reviews`, rkmon_lib/review_keep.py) with what the verifier wrote
there (REVIEW_RESTORED). A review the stop ended starts again at its head (REVIEW_RESUME). A draft that was awaiting
verification gets its worktrees again (`mkwt.py`, the head merged with today's develop) before `rk review-recheck`,
`rk dispatch-pack` or the code quality pass reads them. What the monitor learned of each review's and each build's use
of the RAM disk (`state/scaler_state.json`) survives. After a crash, when no `rk stop` ran, the reviews whose jobs died
show as lost, and the next review advance retries each draft once.

**The orchestrator after a restart** resumes its Claude session (`state/orchestrator.json`) and sends `/rocke-monitor
resume`: it checks again what was in flight, as subagents that were running when the container stopped are not
restored. It runs `rk tick` itself every `monitor.tick_interval_minutes` minutes (10), counted from the last tick on
record, so a tick that is due runs at once, busy session or not; it hands the session only the lines that need Claude,
which the tick triage picks (by default the rule filter, `rk tick-filter`, with no model). It keeps the review driver
running: after each tick while review automation is on, the monitor is on and not stopped, and again at once after a
REVIEW_READY. When its ticks stall, as with a session that cannot start, the tick loop that runs without the
orchestrator takes over, and the review driver exits with TICK_STALE when the last tick is more than 25 minutes old.
Until the Claude session has its key (`docker/rkmon login claude`), the container verifies nothing and posts nothing,
and only the ticks and the dashboards run.

**What runs follows the features** (`rk features` says what a configuration runs, `rk features --runs PART` answers
with its exit code). With every feature on, every part above runs. With the admin dashboard off there is no web
server: the loop that restarts the admin dashboard starts nothing. With review automation off there is no Claude
session: with the merge queue on, each tick is run by fixed rules (`rk rules-tick`), and what needs judgement waits for
a person (the admin dashboard's Needs-you list, `rk status`). The templates that start each kind of deployment:
[README.md: Starting templates](../README.md#starting-templates).

**When a change takes effect**, the reviewer model's among them, and whether it needs a restart:
[QUICK_USER_GUIDE.md, "Change a setting"](../QUICK_USER_GUIDE.md#change-a-setting).

**Compaction and the prompt cache.** The package's defaults set when the session compacts its context and how long
its prompt cache and its subagents' last ([Internal settings](internal_settings.md#package-defaults)). `rk orchestrate`
writes them into the container's `<CLAUDE_WORKSPACE>/.claude/settings.json`, as Claude Code's `autoCompactWindow`,
`promptCacheTtl` and `subagentPromptCacheTtl`, when the session starts, backing up a file it changes, and `rk doctor`
warns when the file does not carry them. On a host the workspace is a person's, and nothing writes there. A compaction
needs no action: each tick reloads the skill and reads the state from the disk.

## Checks

Two commands, for two questions:

- **`rk status`: how the monitor operates right now**, from local files only (no remote call): on or off, stopped or
  draining; the pieces running (the review driver, the tick loop with the last tick and the last full tick, the admin
  dashboard's service, the orchestrator, the gate, develop sanity); the settings; the review queue, the reviews in
  flight, the drafts awaiting verification, the builds running and the remote GPU jobs; the RAM disk's use and the PRs
  on it; then the summary from the stored state.
- **`rk doctor`: whether the monitor is configured right and reaches everything**, read-only: this machine's
  prerequisites, every configured value (secrets only as set or missing, with their path), the secrets' permissions,
  the features, the dashboards' ports, in the container the Claude session, its settings and the tick triage; then
  live checks of GitHub (the login, the API calls left), JIRA, the team chat (its webhook's host answers;
  nothing is posted), the model gateways (the drafting bindings', keys never shown), in the container the Claude
  session's key (the `claude` login's test call: one message, sent as the session sends its requests, the key never
  shown), the GPU cluster, and the RAM disk. The live checks are cached for five minutes (`rk doctor --fresh` checks
  again). It exits 0 when nothing fails, 3 for a missing prerequisite, 4 for anything else that fails. Its output is
  for this machine only.

**The settings in `rk status`.** The monitor moves the parallel reviews and the parallel builds within their ranges,
from the settings that start them, by this machine's load:
[README.md, "Parallel reviews and parallel builds"](../README.md#parallel-reviews-and-parallel-builds). `rk status`
shows each against its range, with the last sample's load and the reason, and how many run at most when the reviewer
endpoint allows fewer; then the starting settings, the footprint estimates on the RAM disk, and the last changes. Every
change is a line of `state/scaler.jsonl`.

**Only with the RAM disk on** (`host.resources.ramdisk_gib` above 0):

- `rk status` says `ramdisk: NOT MOUNTED` when no tmpfs is mounted at the RAM disk's folder: nothing that goes on it
  starts, and nothing spills to the disk. In the container, `host.resources.ramdisk_gib` in your deploy settings file
  (the compose override that mounts it is made from it), then `docker/rkmon confirm`, whose next line says how the
  change applies; [QUICK_DEPLOY_GUIDE.md, "Machine sizing"](../QUICK_DEPLOY_GUIDE.md#machine-sizing) has the memory it
  needs. On a host, mount a tmpfs of your own. `rk doctor` checks the mount, its size, room for one review and one
  build, and the container's memory limit.
- A review or a build starts only when the RAM disk has room for its learned footprint and the memory stays under the
  monitor's limit for this machine; otherwise it waits. A finished review leaves the RAM disk at once.

**Only when the local review loop (the first review pass on the model binding `worker`) runs on Ollama on this
machine:**

- `systemctl is-active ollama` prints `active`; systemd starts it at boot. If it does not, start it as root:
  `systemctl start ollama`. The monitor only checks it, and never restarts it.
- `curl -s <WORKER_URL>/api/tags` lists `<WORKER_MODEL>`.
- Never restart Ollama while it is processing a prompt. Stop a hung runner with `ollama stop <WORKER_MODEL>`, or as root
  with `python3 <PACKAGE_DIR>/worker/kill_runner.py`, never with a service restart.
- About one fresh model load in eight hangs on its first prompt batch: the GPU idle, a few CPU threads spinning. The
  worker sees it in Ollama's journal and resends after `ollama stop`; how long it waits and how often it resends are
  the package's defaults ([Internal settings](internal_settings.md#package-defaults)). If a draft still makes no
  progress, stop the runner before you touch Ollama.

**Only when the reviewer model is remote** (Ollama on another host, or an OpenAI-compatible endpoint): there is nothing
to restart here, and `rk doctor` checks the endpoint read-only. Its key, when it needs one, is a secret, which a
restart keeps ([README.md, "Logins and secrets"](../README.md#logins-and-secrets)); `rk features` shows whether it is
set. Waits on the endpoint's shared rate budget are logged in `<RUN_DIR>/budget.log`.

**Only on a host install on WSL:** WSL stops the distribution about a minute after the last `wsl.exe` session ends, and
that kills every running job: drafts, gates, sanity and check jobs. Keep a `wsl.exe` session open, in a window you
leave open, while the monitor runs.

## Where things live

The monitor's files, on the host and in the container:
[README.md, "Where things live"](../README.md#where-things-live).

Tip: point a Claude Code memory note at `<PACKAGE_DIR>/docs/restart.md`, so a new session finds this page.
