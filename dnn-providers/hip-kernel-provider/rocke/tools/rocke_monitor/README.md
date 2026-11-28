<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor

A monitor for rocKE's open pull requests. It reviews each new or changed pull request in scope, keeps a merge
queue with gates, checks develop after every merge, and shows all of it on two dashboards. It works in ticks, every ten
minutes by default: each tick reads GitHub and updates the reviews, the merge queue and the dashboards. Claude verifies
every review before it goes out: another model may write a first draft, and Claude checks each claim in it. A Claude
session running the `/rocke-monitor` skill makes the judgement calls; the `rk` command does the work.

It runs in a container; a host install, with no services, is for a machine where a container is not an option
([Container and host](#container-and-host)). It starts safe: the deploy settings template starts each feature in
prototype, where the monitor does all its work and keeps each post, message and ticket on this machine, unsent.
Nothing is posted, armed, held, filed or sent until you set a feature to live ([Prototype, live and
standby](#prototype-live-and-standby)).

Build an image with [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md), deploy it with
[QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md), and run it day to day with [QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md).
This README is the reference: what the monitor does, its commands, settings, logins, ports and files, and the terms the
docs use.

## Documentation

Which document to read, and what each one is for:

<!-- BEGIN GENERATED: doc-map
     source: tools/generated.py, DOCS
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Doc | Reader | Its one job |
|---|---|---|
| [README.md](README.md) | everyone | the reference |
| [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md) | the image builder | from a build settings file to a shipped image |
| [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md) | the first-time deployer | from an image to a first tick |
| [QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md) | the operator | day to day, after the first tick |
| [docs/FAQ.md](docs/FAQ.md) | deployers, pull request authors, team leads | short answers, each ending in a link to the fact's home |
| [docs/restart.md](docs/restart.md) | the operator during a restart | is it safe now; the stop levels; what a restart keeps; what to check after |
| [docs/design.md](docs/design.md) | maintainers, stakeholders | why it is built this way |
| [docs/operations_guide.md](docs/operations_guide.md) | stakeholders | the overview: the problems it solves, what it produces, guardrails, benefits, limits, and the policies the operator keeps |
| [docs/review_pipeline.md](docs/review_pipeline.md) | maintainers, reviewers | how a review is made; known issues |
| [docs/maintaining.md](docs/maintaining.md) | maintainers | build, test, publish and regenerate the package, and make the package tarball |
| [docs/internal_settings.md](docs/internal_settings.md) | maintainers | the internal keys: package defaults, state and derived, with their sources and descriptions |
| [agentic/README.md](agentic/README.md) | the Claude session | the agents' guidance |
| [.claude/skills/rkmon-setup/SKILL.md](.claude/skills/rkmon-setup/SKILL.md) | the deployer's own Claude Code session | walk a deployer through deploy steps 1 and 2 |

<!-- END GENERATED: doc-map -->

## Architecture

One toolkit, `rk`, does every mechanical step: it reads GitHub, plans, stages and checks reviews, and writes the
dashboards, and every write that leaves this machine is one of its commands. A Claude session running the
`/rocke-monitor` skill makes the judgement calls: it verifies each review, decides the gates, and writes the posts and
notices, and it sends nothing but through `rk` commands. Ticks drive both. Without review automation no Claude session
runs: each tick is run by fixed rules, and what needs judgement waits for you on the admin dashboard's Needs-you list.

### The components

| Component | What it is | Where it runs |
|---|---|---|
| The toolkit, `rk` | `rk`, `rkmon.py` and `rkmon_lib/`: every action is one `rk` command. It reads GitHub under the read login; its outward writes are a fixed set of commands ([The commands that send](#the-commands-that-send)), each with its feature gate, the write lock and a ledger line, and the writer runs them under the write login | The image's code in `/opt/rocke_monitor`, or the release `rkmon start` or `rkmon apply` installed over it in `/code` (`rk --which-code` names the code that runs) |
| The entry point | `docker/entrypoint.sh`, under tini. As root, it gives the monitor's user the deployment's ids; then, as the monitor, it runs `rk start` and keeps three pieces running, each started again when it ends, except after `rk stop` or `rk kill`: the admin dashboard's service, the orchestrator, and the tick loop that runs without the orchestrator. The container's restart policy brings the container back | The container; each piece's log is in `/work/logs` ([In the container](#in-the-container)) |
| The tick, `rk tick` | One iteration: idle when nothing changed since the last full tick (one line), else full: poll, events, plan, queue, develop guard, review advance, dashboards, then the lines that need Claude | Run by the orchestrator, or by the tick loop that runs without it |
| The orchestrator | `rk orchestrate`: one long-lived Claude session, through the Claude Agent SDK, running the `/rocke-monitor` skill on `features.review_automation.orchestrator`'s model. It runs `rk tick` every `monitor.tick_interval_minutes` and gets a turn only for the lines the rule filter (`rk tick-filter`) passes on; it verifies each `REVIEW_READY`, decides, and sends through `rk` | Kept running by the entry point while review automation is enabled, once `rk orchestrate --check` passes |
| The tick loop that runs without the orchestrator | A tick every `monitor.tick_interval_minutes` while the orchestrator is not set up, or when its ticks are late. Without review automation it runs each tick by fixed rules (`rk rules-tick`): it arms a cleared batch, disarms a PR the queue let go of, turns off a risky auto-merge, starts a bisect and posts the queue notices, each through its own `rk` command and that command's guards; every other line waits for you on the Needs-you list and in `rk status` | Kept running by the entry point |
| The review driver | Keeps the parallel reviews full: stages each PR, starts its first passes and its checks job, and prints `REVIEW_READY #N` when they are done | `rk review-driver`, one at a time, kept running by the orchestrator |
| Jobs | Detached processes: drafts, checks, gates, develop sanity runs, bisects, drift checks; a job that builds or runs GPU tests takes one of the parallel builds | `state/jobs/` in the monitor's home (`rk jobs` lists them) |
| Parallel reviews and parallel builds | How many reviews, and how many local builds with their tests, run at once: each starts at your setting and follows this host's load one step at a time, within its range ([Parallel reviews and parallel builds](#parallel-reviews-and-parallel-builds)) | `features.review_automation.parallel_reviews`, `monitor.parallel_builds` |
| The verification and code quality passes | Two Claude subagents for each review. The verification pass (the agent file `rocke-review-verifier`) reviews the PR on its own, checks every claim of the first review pass against the code, drops what it cannot confirm, and writes the review that is posted; the code quality pass (`rocke-code-quality-reviewer`) reviews the PR's design beside it and gives its findings to the verification pass | Agent files that `rk provision` installs in the Claude workspace, `/work/claude-workspace` |
| The optional passes | Four passes that another model may run first, each set in `features.review_automation.passes`: the first review pass (a draft of the review), the CI triage pass (what caused each failing CI job), the first code quality pass (leads for the code quality pass) and the notice draft pass (a first draft of an overlap or a drift notice). None of them decides: the verification pass checks every claim of a draft, and each owner checks every lead (a lead whose quotes are not in its sources is dropped first). The first review pass's tools only read the PR's trees: no shell, no GitHub, no network | A model binding (`model_bindings`) or a Claude model; a review's drafts and leads are in its folder under `/work/workers/` |
| The dashboards and the web server | Each full tick writes the dashboards. `rk serve`, which `tools/rkweb.py` starts, serves the admin dashboard, with its Approve and Decline buttons and its priority box, and, with delivery `serve`, the team dashboard; with delivery `publish`, the team dashboard is pushed to a web host over SSH after every tick ([The dashboards](#the-dashboards)) | The entry point's loop that restarts the admin dashboard's service |
| The team chat | A Teams channel's webhook: a message when develop's CI is red and no fix is planned, and one for each decision that waits for you | `features.alerts`; the secret `teams_webhook` |
| JIRA | Tickets through the atlassian CLI, which the writer runs: the follow-up tickets of merged PRs, and an out-of-scope issue's proposed ticket once you approve it (`rk ticket-file`) | `features.ticket_creation`, `integrations.atlassian_cli.jira_url`; the CLI's login, `rkmon login atlassian` |
| Remote GPU tests | GPU-only test cases on a Slurm cluster's GPU nodes, over SSH | `features.review_automation.remote_gpu_tests` |
| The update | `rkmon package update` puts a newer package in the package folder; `rkmon confirm`, then `rkmon apply`, install its code over the image's, with the doctor before and after, and put back what ran before on a failure; `rkmon rollback` goes back ([Updating](#updating)) | `docker/rkmon` on the host; in the container, root installs each release in `/code/releases/` (`docker/rk_root.py`) |

### How they connect

- The tick reads GitHub, the state and the jobs, saves the state, writes the dashboards, and prints its lines: events,
  proposals, `!! ` alerts, `REVIEW_READY`, the board.
- The rule filter (`rk tick-filter`) passes the orchestrator only the lines that need Claude, and the orchestrator
  answers them with `rk` commands, following the skill's procedures. Without review automation, `rk rules-tick` reads
  the same lines by its fixed rules.
- The review driver advances the work in flight every minute and hands each finished review to the orchestrator as
  `REVIEW_READY`.
- Every outward write is one `rk` command: its feature on and live, the monitor not in standby, no stop in force, the
  write lock, then one ledger line (`state/actions.jsonl`) and one journal line. A restart never repeats a write: a
  comment, issue or ticket whose answer was lost (a timeout) is read back from GitHub or JIRA before it is called
  failed, and one that cannot be settled stays in doubt (`state/writes_in_doubt.json`; a ticket stays being filed)
  until the next try reads first.
- The admin dashboard's buttons go to `rk serve`, which does what `rk prioritize`, `rk priority-decline`,
  `rk ticket-file` and `rk ticket-decline` do.

The whole system:

```text
 operator: rkmon, a browser                          other teams: a browser (team dashboard, request box)
      |                                                     |
      v                                                     v
 +------------------------------------------------------------------------------------------------+
 | the container                                                                                  |
 |                                                                                                |
 |  +-----------------------------+    rk commands     +------------------------------------+     |
 |  | the orchestrator            |------------------->| toolkit: rk                        |     |
 |  |   /rocke-monitor skill      |                    |   tick: poll, plan, queue, guard,  |     |
 |  |   verification pass         |<-------------------|     dashboards                     |     |
 |  |   code quality pass         |  tick lines,       |   review driver                    |     |
 |  +-----------------------------+  REVIEW_READY      |   guarded writes, ledger, journal  |     |
 |        |                                            +------------------------------------+     |
 |        |                                              |              |              |          |
 |        |                                              v              v              v          |
 |        |                                       jobs: drafts,   /work: state,   rk serve:       |
 |        |                                       checks, gates,  clone,          admin and team  |
 |        |                                       sanity, drift   worktrees, logs dashboards      |
 +--------|---------------------------------------------------------------------------------------+
          |              |                 |                 |                  |
          v              v                 v                 v                  v
   Claude API      GitHub: reads;     the optional       JIRA: the          Teams webhook;
   (Anthropic or   the guarded rk     passes' models     atlassian CLI      GPU nodes: SSH
   a gateway)      writes only        (model bindings)   (ticket features)  and Slurm
```

One review, from pick to post:

```text
 tick: the plan picks PR N (in scope, head settled for push_cooldown_minutes, not reviewed at this head)
   |                                                        or by hand: rk review-start N, rk review-queue N
   v
 review driver: a free place among the parallel reviews
   |
   v
 stage: fetch_pr.py (reads) -> workers/pr<N>/pr/ ; mkwt.py -> wt<N> (the head), wt<N>m (the head merged with develop)
   |
   +-------------------------------+---------------------------------+
   v                               v                                 v
 first review pass (optional)    checks job (a parallel build)     CI triage pass and first code
   another model's draft           build, ctest, goldens, tests,     quality pass (optional): leads
                                   installed tests, scripted         from another model
                                   sections, CI, other open PRs
   -> review_agent.md              -> checks/checks.md               -> ci_triage.md, quality_leads.md
   +-------------------------------+---------------------------------+
   v
 REVIEW_READY #N -> the orchestrator
   v
 rk dispatch-pack N -> verification pass: a blind review, then every claim of the draft reproduced or cut
                    -> code quality pass beside it (rk wait-quality) -> notes.md and the post
   v
 rk spec-post N --round R: the spec from the notes and live reads; a CHECK line stops it
   v
 rk post --premises: head, develop, conflicts, CI and threads checked again; line 1, the public-repository checks,
                     one post per head, an hourly cap
   v
 live: one comment on PR N, a ledger line, a local copy   |   prototype or standby: kept unposted, PR_UNPOSTED_<N>.md
```

The tick:

```text
 every monitor.tick_interval_minutes minutes: the orchestrator, or the tick loop that runs without it
   |
   v
 rk tick --- OFF, or STOPPED? --- yes ---> one line; nothing else
   |
   v
 the change fingerprint: PR heads, CI, develop, issues, requests, jobs, files, configuration, code; and the timers
   |                                    |
   | nothing changed, no timer due      | a change, a timer due, or monitor.full_tick_max_interval_minutes since a full tick
   v                                    v
 idle tick: the "checked" stamps,     full tick: poll GitHub (reads) -> events -> drift results -> develop issues
 one TICK idle line                     -> merge-only marks -> merge queue and plan -> fix sweep -> weekly report
                                        -> develop guard -> merge queue advance (merge queue on)
                                        -> review advance (review automation on) -> cleanup -> GPU jobs
                                        -> priority requests -> follow-up scans -> insights -> overlaps -> board
                                        -> Needs-you message -> state saved -> dashboards written (and published)
                                        -> the tick's lines
                                          |
                                          v
                                rk tick-filter: the lines that need Claude
                                          |
                                          v
                                the orchestrator: /rocke-monitor tick -> audits, decisions, rk write commands
                                (no Claude session: rk rules-tick runs its fixed rules, and the rest waits for you)
```

### Where state and configuration live

The monitor runs on your confirmed deploy settings, which the settings volume delivers, over the package's defaults:
each key you may set, with its default and meaning, is in the [Settings reference](#settings-reference). Its state is
in its home's `state/`, and everything else it writes is in the work volume. [Where things live](#where-things-live)
has each folder and file. A host install has no deploy settings: its settings are a `config.json` in the monitor's
home, read only while no deploy settings are delivered.

### Container and host

| | Container | Host install |
|---|---|---|
| Setup | An image built from a build settings file ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)), then a deployment from a deploy settings file ([QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md)) | The package folder, run as it is, and a `config.json` in the monitor's home |
| The Claude session | `rk orchestrate`, while review automation is enabled | Your own Claude Code session, running the `/rocke-monitor` skill |
| Ticks | The orchestrator, or the tick loop that runs without it | `tools/tick_loop.py`, which your session keeps running in the background |
| After a stop or a reboot | The entry point starts each piece again, and the container's restart policy brings it back | Nothing starts by itself: no services. `/rocke-monitor start` runs `rk start`, which starts the admin dashboard's service, and starts the tick loop again |
| Users | The monitor, the writer and the PR runners, each a user of its own | One user runs everything, PR code included, and both GitHub logins are that user's: they are kept apart by convention only ([The GitHub CLI](#the-github-cli)) |
| GPU tests | On the cluster only (`features.review_automation.remote_gpu_tests`): the container has no GPU | The host's own GPU, when it has one, or the cluster |
| Updates | `rkmon package update`, then `rkmon confirm` and `rkmon apply` ([Updating](#updating)) | None: `rk` has no update command |

### The container deployment

The container has two jobs on the host, on any machines: building an image, and deploying a container from one. On
each, `docker/rkmon` (a Python wrapper of docker; [the wrapper's commands](#the-containers-wrapper-dockerrkmon)) turns a
settings file into everything docker needs, and never runs anything the operator could not run by hand: it prints each
docker command first.

```text
 the build machine                                     the deploy machine (builds nothing)
 your build settings file (permissions 600)            your deploy settings file (permissions 600)
   |  rkmon build --settings FILE: one docker build,     |  rkmon confirm --settings FILE: your settings checked,
   |  each setting a --build-arg, each download          |  shown and confirmed, then recorded in this package
   |  location a --secret (in no layer, history, label)  |  folder (docker/.rkmon); nothing else is written
   v                                                     |
 image <image.name>:<tag>, every image full: ROCm, gh,   |
 the atlassian CLI, and the Claude Agent SDK with        |
 Claude Code                                             |
   |                                                     |  rkmon image load IMAGE_FILE (or pull): checks that it
   |  rkmon image test, then rkmon image save IMAGE_FILE |  was built from this package's image recipe, and TLS
   +-------------------- the file (or a registry) ------>+  from inside it
                                                         |  rkmon start, the first time: docker/.env,
                                                         |  docker/compose.override.yaml and the settings volume's
                                                         |  deploy.json written from your confirmed settings (none
                                                         |  committed); this package's code put in /code;
                                                         |  rk provision and rk init; then docker compose up -d
                                                         v
 the container: the compose service "rkmon" of the project that container.name names
   entry point (tini): as root, the deployment's ids given to the monitor's user (docker/rk_ids.py); then, as the
   monitor, rk start, and it keeps the admin dashboard's service, the orchestrator and the tick loop that runs
   without it
   users: the monitor (the orchestrator, its agents, the jobs), the writer (the outward writes, through
          docker/rk_write.py), the PR runners (PR code, one per slot, through docker/rk_prcode.py)
```

- **Two settings files.** The build settings file (`kind: build_settings`: `image`, `operating_system`, `rocm`,
  `github_cli`, `atlassian_cli`, `claude_agent_sdk` and `python_packages`; [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md))
  is the one source of an image, and every image is full; its labels are in [Image labels](#image-labels). The deploy
  settings file (`kind: deploy_settings`: `container`, `image`, `updates`, `repository`, `host`, `monitor`,
  `features`, `integrations` and `model_bindings`; [Settings reference](#settings-reference)) is the one source of a
  deployment. Each has a JSON Schema in `schema/`, for the checks and for your editor (the Settings reference's "In an
  editor").
- **The steps** of each path are checked, and each ends with the next step's command: [The build
  path](#the-build-path), [The deploy path](#the-deploy-path) and [The deployment's states](#the-deployments-states).
- **The volumes and ports**, on the host and in the container: [Ports, volumes and
  docker/.env](#ports-volumes-and-dockerenv); what each volume has: [In the container](#in-the-container).
- **Several deployments**, on one host or on one repository: [QUICK_DEPLOY_GUIDE.md, "Several
  deployments"](QUICK_DEPLOY_GUIDE.md#several-deployments).

## What it can and cannot do

What it can do, each part only while its feature says so: the capabilities table in [Features](#features). Beyond the
features, `docker/rkmon`:

- builds an image on one machine and deploys it on another, each step checked and ending with the next step's command
  ([The build path](#the-build-path), [The deploy path](#the-deploy-path));
- updates the monitor's code from a newer package, or moves it to a new image, with the doctor before and after and a
  rollback ([Updating](#updating));
- takes over an earlier monitor's state from a copy of its `/work` (`rkmon adopt --from DIR`): it shows the plan
  first, sets this deployment's own copies aside, and starts the deployment in standby;
- runs several deployments on one host, each with its own name, port and image tag ([QUICK_DEPLOY_GUIDE.md, "Several
  deployments"](QUICK_DEPLOY_GUIDE.md#several-deployments)).

What it never does:

- It never approves a PR or merges one directly. Its reviews are comments; its one other review state is a change
  request on a confirmed hold (`features.merge_queue.auto_hold`), which only your re-approval lifts. Each guardrail:
  [docs/operations_guide.md, "5. Guardrails"](docs/operations_guide.md#5-guardrails).
- It never pushes to a git remote: it fetches only. Its GitHub calls are reads (the `gh` wrapper refuses write methods
  and GraphQL mutations), except the guarded write commands ([The commands that send](#the-commands-that-send)).
- It never changes your own review states and never turns off an auto-merge you turned on yourself: it tells you.
- It never posts a draft as it is, nor a post that fails `rk post`'s guards: line 1, no link outside the public
  allow-list, no absolute performance numbers, no ticket tag inside a link, one post per head (unless `--again`), and
  an hourly cap.
- It never lets a model that is not Claude verify, decide a gate or post: the verification pass, the code quality pass
  and the orchestrator are always Claude models. Another model only writes a first draft or hands leads, which are
  checked, and the first review pass's tools only read the PR's trees: no shell, no GitHub, no network.
- It never changes the image's code, never puts a secret in a settings file or a log, and never installs a system
  package.
- It never writes to Confluence.
- It reaches JIRA only through the atlassian CLI, and writes to it only while ticket creation is live: it files
  nothing while ticket creation is off or in prototype, in standby, or while the monitor is stopped.
- Its public posts name a GPU by its gfx target only, never a node or a GPU product.
- The orchestrator sends nothing except through `rk`: its guard refuses gh writes, pushes to a git remote, network
  writes by curl or wget, ssh, scp, Slurm commands, and the atlassian CLI's own writes unless follow-up tickets go out
  (`rk atlassian` weighs each of its writes itself). It refuses the `rk` commands only you, or `rk` itself, run
  (`rkmon_lib/outward.py`'s table): it never files or declines a proposed ticket, sends a raw team chat message,
  pushes the team dashboard, reads the web host's requests, sends the test card, sets a secret or the read login,
  records a host's own keys, installs the web host's request service, or ends standby.
- The file tools of the orchestrator and its agents never write outside the folders the monitor works in, each
  allowed by one rule on its absolute path.

What needs you:

- the build settings file, and in it the atlassian CLI's download and its SHA-256, your network's root certificates
  when it inspects TLS, and your own download places, index and mirror where the public ones are out of reach; the
  deploy settings file;
- the features and when each goes live, and every login and secret ([Logins and secrets](#logins-and-secrets)):
  nothing in the package makes a GitHub token;
- clearing a PR for auto-merge (`rkmon rk clear N`), floating a priority (`rkmon rk prioritize N`), and approving or
  declining other teams' priority requests and the proposed tickets;
- approving PRs, re-approving a held PR once it is fixed, and reverting a breaking merge;
- approving or declining the in-depth review of a PR too large for the autopilot
  (`features.review_automation.large_prs.in_depth_review_over_lines`): it waits for you under Needs you, with its size
  and cost (the admin dashboard's Approve and Decline, or `rkmon rk review-approve N` and `rkmon rk review-skip N`);
- turning off someone's auto-merge without a failing gate (`rkmon rk automerge-off N --user FILE`), assigning an issue
  (`rkmon rk issue-assign N LOGIN`) or creating a label (`rkmon rk label-create NAME --description TEXT --color HEX`):
  the monitor never does these on its own;
- updates: `rkmon package update`, a new image (`rkmon build`, `rkmon image load`), `rkmon apply` and
  `rkmon rollback`.

Its limits:

- It reviews only open, non-draft PRs into develop that carry a scope label (`repository.scope_labels`), or a shared
  label (`repository.shared_label_folders`) with a change under that label's folder, once the head settled
  (`features.review_automation.schedule.push_cooldown_minutes`), with at most one follow-up post per
  `features.review_automation.schedule.follow_up_cooldown_minutes`.
- It needs the network: GitHub, Claude (directly or through your gateway), and the endpoint of each model binding a
  pass uses.
- It counts tokens (`rkmon rk usage`); review spend in dollars needs the prices in `features.review_automation.spend`.
- In the container, every run of PR code (tests, builds, probes) is a PR runner's: a user of its own that reads no
  secret ([The GitHub CLI](#the-github-cli)). On a host install, PR code runs as the user that runs everything else.
- The container has no GPU: GPU-only test cases run only with the remote GPU tests
  (`features.review_automation.remote_gpu_tests`).
- It runs on Linux or in WSL2, never natively on Windows ([Platforms](#platforms)); the container runs under Docker
  with compose v2 (and buildx on the build machine), or Podman (`CONTAINER_ENGINE=podman`).
- The image build needs the network: the base image, the operating system's packages, the ROCm installer, gh, the
  atlassian CLI, the Claude Agent SDK and the other Python packages, each from its public place or the one your build
  settings give (the atlassian CLI's always). Behind a TLS-inspecting proxy it needs that proxy's root certificates
  (`operating_system.ca_certificates`), and so does the deployment, whose image trusts the build's certificates only.

## The build path

An image is built from a build settings file, on any machine with Docker and buildx, in five steps, and carried to the
deploy machine as a file or through a registry. [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md) walks them one by one.
`rkmon next --settings FILE`, with your build settings file, says where a build is on the path, with the command to run
next, and `rkmon help build-path` lists the steps.

<!-- BEGIN GENERATED: build-steps
     source: rkmon_lib/steps.py, BUILD_STEPS
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| # | Step | Command | What it does |
|---|---|---|---|
| 1 | template | `rkmon template build FILE` | write a build settings file for you to fill in |
| 2 | check | `rkmon check --settings FILE` | check this machine and your build settings; change nothing |
| 3 | build | `rkmon build --settings FILE` | build the image from your build settings |
| 4 | test | `rkmon image test --settings FILE` | test the image on this machine |
| 5 | ship | `rkmon image save IMAGE_FILE --settings FILE`<br>or: `rkmon image push --settings FILE` | save the image to a file, or push it to your registry |

<!-- END GENERATED: build-steps -->

## The deploy path

A deployment runs the monitor in a container, from an image built elsewhere or on the same machine, in eight steps.
[QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md) walks them one by one. `rkmon next` says where a deployment is on the
path, with the command to run next, and `rkmon help deploy-path` lists the steps.

<!-- BEGIN GENERATED: deploy-steps
     source: rkmon_lib/steps.py, DEPLOY_STEPS
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| # | Step | Command | What it does |
|---|---|---|---|
| 1 | template | `rkmon template deploy [--profile PROFILE] FILE` | write a deploy settings file for you to fill in |
| 2 | check | `rkmon check --settings FILE` | check this host and your deploy settings; change nothing |
| 3 | confirm | `rkmon confirm --settings FILE` | check your deploy settings, show them, and confirm them |
| 4 | image | `rkmon image load IMAGE_FILE`<br>or: `rkmon image pull`; nothing when the image is here<br>(`rkmon image check` checks it) | make sure the image is on this host and suits this package |
| 5 | login | `rkmon login` | set each login and secret your settings need |
| 6 | start | `rkmon start` | start the container with your confirmed settings |
| 7 | doctor | `rkmon doctor` | check the running monitor and everything it reaches |
| 8 | first tick | `rkmon first-tick` | wait for the monitor's first tick |

<!-- END GENERATED: deploy-steps -->

## The deployment's states

A deployment is always in one of these states, each with its sentence and its next line. `rkmon status` starts with
the sentence (`state: ...`), and every command on the deploy side ends with the next line.

<!-- BEGIN GENERATED: states
     source: rkmon_lib/states.py, TABLE
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| # | State | What it says | Next line |
|---|---|---|---|
| 1 | Another deployment's container name | the container name rocke-monitor is used by another deployment on this host | `next: rkmon confirm --settings ~/rkmon-deploy.yaml` |
| 2 | No settings confirmed | no deploy settings are confirmed in this package folder | `next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check --settings FILE)` |
|  |  | the settings file you confirmed, \~/rkmon-deploy.yaml, is gone | `next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check --settings FILE)` |
| 3 | An apply, rollback or adopt stopped part-way | rkmon apply stopped part-way | `next: rkmon apply (it finishes the change, or puts the previous settings back on a new failure)` |
|  |  | rkmon rollback stopped part-way | `next: rkmon rollback (it finishes putting back the code and settings from before the last apply)` |
| 4 | Your file is in another layout | \~/rkmon-deploy.yaml is in layout 1, older than this package's layout 2 | `next: rkmon migrate --settings ~/rkmon-deploy.yaml` |
|  |  | \~/rkmon-deploy.yaml is in layout 3, newer than this package's layout 2 | `next: rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one` |
| 5 | Not confirmed | your settings file changed after the last confirm | `next: rkmon confirm` |
|  |  | this package changed after the last confirm | `next: rkmon confirm` |
| 6 | No image | the image rocke-monitor:YYYY.MM.DD is not on this host | `next: rkmon image load IMAGE_FILE` |
|  |  | the image rocke-monitor:YYYY.MM.DD is not on this host | `next: rkmon image pull` |
| 7 | Logins missing | your confirmed settings need github_read, which is missing or wrong | `next: rkmon login` |
| 8 | Never started | this deployment's container does not exist yet | `next: rkmon start` |
|  |  | this deployment's container runs, but its work volume is not set up (it was started some other way than rkmon start) | `next: rkmon start` |
| 9 | Changed outside your file | changed outside your file: docker/.env differs from what rkmon last wrote there | `next: rkmon apply (--dry-run shows what it puts back)` |
| 10 | Put back | the last change you confirmed was put back by rkmon rollback | `next: rkmon apply` |
|  |  | the last change you confirmed was put back after a failed doctor check: model fast: no answer from the binding's base_url | `next: rkmon apply, once what failed is fixed (a fix in your file: rkmon confirm first)` |
| 11 | A change waits | a change waits: you confirmed 1 change, and the monitor runs the settings before it | `next: rkmon apply` |
|  |  | a change waits: you confirmed 1 change, and the stopped container has the settings before it | `next: rkmon apply` |
| 12 | Stopped | the container is stopped | `next: rkmon start` |
| 13 | Not loaded | the container has run for 2 minutes or more, and the monitor has not loaded your confirmed settings | `next: rkmon apply (or rkmon rollback, for the version before)` |
| 14 | Standby | the monitor runs your confirmed settings, in standby: it sends nothing until you run rkmon standby off | `next: rkmon standby off, once rkmon status and the dashboards read right` |
| 15 | Paused | you paused the monitor: it starts no new work | `next: rkmon resume` |
| 16 | Runs your confirmed settings | the monitor runs your confirmed settings | `done: the monitor runs your confirmed settings` |

Until deploy step 8 has passed after a start (a tick has run since a start: the first command that sees one keeps that in this folder's records, until `rkmon remove --delete-data` removes the deployment's volumes), a next command that is a deploy step's carries that step's number and name: in state 7, `next: step 5, login: rkmon login`. While the container runs, the line of states 14 and 16 is `next: step 7, doctor: rkmon doctor` until the doctor has passed since the start, then `next: step 8, first tick: rkmon first-tick` until a tick has run.

While standby is on, every state's sentence says so: state 14's itself, and every other's ends "; in standby: it sends nothing until you run rkmon standby off".

A command that cannot run in the deployment's state refuses: it names the state's sentence, the fix when there is one, and the state's next line. In state 1:

```text
REFUSED: the container name rocke-monitor is used by another deployment on this host; nothing changed
fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rocke-monitor-2 (docker compose ls --all
     lists the names in use)
next: step 3, confirm: rkmon confirm --settings ~/rkmon-deploy.yaml
```

A refusal exits 3 in states 2, 4, 5, 6, 7 and 8, where an earlier step is not done, and 4 in any other.

<!-- END GENERATED: states -->

## Run it in a container

The monitor runs in a container of one image. Any machine with Docker and buildx builds the image ([The build
path](#the-build-path)), and any host with Docker and its compose plugin runs it ([The deploy
path](#the-deploy-path)); [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md) and [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md)
walk each step. This section is the reference for the image, the container and what runs in it. Below, `rkmon` is
`docker/rkmon`, the wrapper of docker that runs every step ([The container's
wrapper](#the-containers-wrapper-dockerrkmon)).

### The image

Every image is full: ROCm, Claude Code (through the Claude Agent SDK), the GitHub CLI, the atlassian CLI, git, cmake,
ninja, a compiler, and a Python of 3.12 or newer with the checks' virtual environment. A deployment chooses its
profile and its features when it runs, never what the image carries, and nothing of a deployment is in the image: its
settings, its state, its logins and secrets, and its user ids come in as the container starts.

- **The build.** `rkmon build --settings FILE` runs one `docker build` of `docker/Dockerfile` from your build settings
  ([Build settings](#build-settings)), after build step 2's checks: each value becomes a build argument, and each
  download location a build secret, in no layer, no history and no label. No build argument carries a secret or a
  private value. `rkmon build --settings FILE --dry-run` prints the command, and `--no-cache` builds every layer
  again. [QUICK_BUILD_GUIDE.md, "Step 3: build"](QUICK_BUILD_GUIDE.md#step-3-build) says what the image records of
  its build, and what Docker keeps of it on the build machine.
- **Nothing of where it was built.** Every docker build runs with `BUILDX_GIT_INFO=0`, `BUILDX_GIT_LABELS=0` and
  `--provenance=false` (`rkmon_lib/build_plan.py`), so the image carries neither the package folder's git remote and
  commit nor a provenance record.
- **The stages.** `base` (the operating system, your network's root certificates, the tools, Python); then `rocm`,
  `venvs`, `claude`, `atlassian` and `gh`, each from `base` and built at once; then the image, on the `rocm` stage,
  with each other stage's files copied in. The package's code comes last, so a code change rebuilds no tool.
  `docker/Dockerfile` says what each stage installs and checks.
- **Your network's root certificates** (`operating_system.ca_certificates`). `rkmon build` hands the build copies of
  them in a private folder, which it makes for the build and removes after it, and the image trusts them for every
  download and every tool. They are never in the build context of the package's code: [QUICK_BUILD_GUIDE.md, "Behind
  a network that inspects TLS"](QUICK_BUILD_GUIDE.md#behind-a-network-that-inspects-tls).
- **The build context** is an allow-list (`.dockerignore`): only the package's own files go in, and a file that may
  hold a secret or a private value stays out wherever it is, your settings files among them.
- **ROCm.** The image records the ROCm release it installed, with its clang, in `/etc/rkmon/rocm.json`. `rk doctor`'s
  clang parity row and the setup summary say when that clang is not the one the review gates are tuned on. The build
  checks that hipcc compiles for the GPU targets of the review's GPU tests: `rkmon build --dry-run` shows them, as
  `HIP_CHECK_TARGETS`.

### The container

- **Its users.** The monitor runs the session, its agents and the jobs. The writer is the only user that reads the
  writer's logins and secrets (`/secrets-write`), and runs the commands that use them. The PR runners, one for each
  slot, run the code of pull requests: each reads no secret and writes only its slot's folders.
- **Their ids.** At each start the entry point, as root, gives the monitor and its group the ids
  `host.users.monitor_uid` and `host.users.monitor_gid` (by default yours, so a file of yours that the monitor reads
  is its own); the writer and the PR runners keep the package's ids. When the ids change, the volumes' files follow.
  An id that would mix the users, or a system account's (below 1000, or 65534), is refused, with the setting to change
  (`docker/rk_ids.py`). Then each command runs as its user, and `rk` never runs as root.
- **Its limits.** The compose file drops every Linux capability but eight, which only a process that runs as root has
  (`rkmon_lib/container_layout.py`, CAPABILITIES): CHOWN, DAC_OVERRIDE, FOWNER and FSETID, with which each start gives
  the users their ids and the folders their owners and permissions; SETUID and SETGID, with which it drops to a user,
  and which sudo needs to run the write broker as the writer and PR code as a PR runner; AUDIT_WRITE, for sudo's record
  of each run; and KILL, with which tini, the first process, passes a stop signal on. At most 4096 processes run at
  once (`pids_limit`). Its CPUs and memory are `host.resources.cpus` and `host.resources.memory_gib`
  ([QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing)).
- **Its health.** The image's health check (`docker ps` shows it) says unhealthy while a piece the entry point keeps
  running crash-loops (5 restarts in 30 minutes, which `/work/logs/restarts` records), or while whether the work
  volume is set up cannot be told (its line says why); it says healthy while the work volume is not set up yet, or
  while the monitor is stopped.
- **Its stop.** A stop of the container, `rkmon stop` or docker's own, runs `rk stop` in it, which may wait for a write
  in flight and for the state lock: docker waits up to 10 minutes for it. The restart policy starts the container
  again after a reboot, unless you stopped it; its start runs `rk start`, and the orchestrator resumes its session.
  Each way to stop the monitor: [Stopping the monitor](#stopping-the-monitor).

### A start

`rkmon start`, deploy step 6, starts the container with your confirmed settings; `rkmon help start` says what it
refuses.

1. The first start puts your confirmed settings in place: `docker/.env`, `docker/compose.override.yaml` and the
   settings volume's `deploy.json` ([Ports, volumes and docker/.env](#ports-volumes-and-dockerenv)). After it, only
   `rkmon apply` and `rkmon rollback` write them.
2. When the image's own code is not this package's, the start installs this package's code in the code volume,
   `/code`, except after `rkmon rollback` or an apply that rolled back: the code they put back runs.
3. `rk provision`, in a one-off container, sets the work volume up from your confirmed deploy settings, asking
   nothing: the folders, the worker tools, the monitor's clone of the repository and its develop worktree, the
   code-owner areas, the checks' virtual environment, the agentic files in the Claude workspace
   (`/work/claude-workspace`), and the checks of the first review pass's model when a model binding runs it. The first
   start sets the work volume up, the next finishes one cut short, and a later one puts back what has gone missing and
   changes nothing in place. Once the work volume is set up, a start removes nothing: a clone or a worktree it finds
   part-made is a WARN, whose fix names what to remove by hand (in `rkmon shell`), then `rkmon stop --drain` and
   `rkmon start`. `rkmon apply` and `rkmon rollback` run it too, in the running container, before the monitor loads
   the settings they put in place, for 30 minutes at most: a run that fails, or is not done in time, says so in one
   `set-up:` line, and the command goes on.
4. The first start's `rk init` seeds the monitor's state from GitHub, the monitor on. A monitor you paused with
   `rkmon pause` stays paused; `rkmon start --on` ends the pause too.
5. The container starts, and a stop or a drain left behind is cleared. A container started some other way waits for
   its first start, with nothing running: the start stops it, then makes the first start.

### What runs in the container

The entry point runs `rk start`, then keeps three pieces running ([The components](#the-components)), each started
again when it ends, except after `rkmon rk stop` or `rkmon rk kill`: they stay down until `rkmon rk start`, and
`rkmon rk stopped` says so. At each start it installs the worker tools again from the code that runs
(`rk worker-tools --install`).

- **The admin dashboard's service**, while the admin dashboard is on (`features.dashboards.admin.enabled`).
- **The orchestrator**, while review automation is on. Until `rk orchestrate --check` passes, it waits, and looks
  again every minute, so a key set meanwhile with `rkmon login claude` needs no restart; a session that runs reads a
  new key as it next starts. Its session sends nothing but through `rk` ([What it can and cannot
  do](#what-it-can-and-cannot-do)), and its guard also refuses what would change what the review's agents and jobs
  share, each refusal saying what to run instead (`rkmon_lib/workguard.py`).
- **The tick loop that runs without the orchestrator**: while the orchestrator is not set up, or its last tick is
  more than twice `monitor.tick_interval_minutes` plus 5 minutes old (`rk tick-due`), it runs a tick every
  `monitor.tick_interval_minutes` minutes (`rk tick-min` prints them). With the merge queue on and no Claude session,
  each tick is run by fixed rules (`rk rules-tick`); otherwise it is `rk tick`. While the configuration cannot be
  read, no tick runs: the loop says so once, and looks again every minute. Each tick has a time limit
  (`rk loop-config` prints it), past which the tick's whole process group is stopped; while the limit cannot be read,
  no tick runs that round, and the log says why.

`rkmon rk tick-now` asks for a tick at once, from whichever of the two runs the ticks. Each piece's log is in
`/work/logs` ([In the container](#in-the-container)); the logs are cut at their size limit at each start and at each
round of the tick loop, never while the configuration cannot be read. The orchestrator's lines (one for each tick,
the `!!` lines, and what went to the session) are the container's log too: `rkmon logs -f`.

To change, update or remove a deployment: [QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md) and [Updating](#updating).

## Quick start: to a first tick in prototype

1. **Build the image** from a build settings file: [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md).
2. **Deploy it** from a deploy settings file: the eight steps of [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md), from
   `docker/rkmon template deploy` to `docker/rkmon first-tick`. Your logins and secrets are set with `docker/rkmon
   login` and `docker/rkmon secret`, never in the file. With Claude Code on the deploy host, `/rkmon-setup` in the
   package folder walks you through writing and checking the file.
3. **The first tick** runs in prototype: it reads GitHub, writes the dashboards and the merge plan, prepares reviews and
   keeps them unposted, and sends nothing.
4. **Check it.** `docker/rkmon doctor` re-runs every check and reaches every configured service; `docker/rkmon rk
   features` shows the setup summary; `docker/rkmon status` shows how the monitor operates right now.

## Image labels

`rkmon build` gives each image these labels and no others, and no label contains an address or a user id.
`rkmon image ls` and step 4 of [the deploy path](#the-deploy-path) show four of them: the package version, the build
date, the base and ROCm. No label decides whether an image suits a package folder: step 4 checks that the image was
built from this package's image recipe.

| Label | What it says |
|---|---|
| `org.opencontainers.image.title` | `rocke-monitor`: an image of this package, whatever its name (`rkmon image ls` finds it by this label) |
| `org.opencontainers.image.description`, `org.opencontainers.image.licenses` | What the image is, and its license |
| `org.opencontainers.image.version`, `org.rocke.package.version` | The package version it was built from |
| `org.opencontainers.image.created`, `org.rocke.build.date` | When it was built |
| `org.rocke.base` | The base image it was built on |
| `org.rocke.rocm.version` | The ROCm release it carries |
| `org.rocke.rocm.clang_commit` | The llvm-project commit its clang must name: the one of the ROCm release the review gates are tuned on; blank for another release |
| `org.rocke.rocm.library_targets` | The GPU targets ROCm's libraries are installed for |
| `org.rocke.atlassian_cli.version`, `org.rocke.github_cli.version`, `org.rocke.claude_agent_sdk.version`, `org.rocke.claude_code.version` | The versions of the atlassian CLI, gh, the Claude Agent SDK and Claude Code it carries |
| `org.rocke.build_settings.fingerprint` | A SHA-256, written `sha256:…`, of the build settings it was built from, without the private download addresses |

## Where things live

Where the monitor and `rkmon` keep what they use, on the host and in the container; the outputs folder; your private
files; and where a private value may show.

### On the host

| What | Where |
|---|---|
| The package folder | Where you unpacked the package: `rkmon` runs from it. It keeps no file of your deployment except the package's own private files ([Your private files](#your-private-files)); keep your own files outside it. Its code is every file the package's `.dockerignore` lets into an image's build, a file you add to `docker/` too: a change to one is a change of the package, which `rkmon next` shows as step 3 REDO until `rkmon confirm` |
| Your settings files | Your own files, where [Your private files](#your-private-files) allows; the guides name them `~/rkmon-build.yaml` and `~/rkmon-deploy.yaml` |
| `docker/.env`, `docker/compose.override.yaml` | In the package folder: what docker compose reads beside `docker/compose.yaml` to run this deployment's container, made from your confirmed deploy settings (`rkmon_lib/deploy_plan.py`). The first `rkmon start`, `rkmon apply` and `rkmon rollback` write both, with the settings volume's `deploy.json` ([In the container](#in-the-container)): the first start and apply keep the text each had before as `docker/.env.prev` and `docker/compose.override.yaml.prev`, which rollback puts back. `rkmon confirm` writes neither |
| `docker/.rkmon/records.json` | In the package folder: the host's records, what only the host keeps about this package folder's deployment. `rkmon confirm` records each confirm there (your settings file, the settings' fingerprint and the file's digest, the package version, the container's name and image, and how many changes it confirmed), `rkmon doctor` its last result, and the first `rkmon start`, `rkmon apply` and `rkmon rollback` the fingerprint of each file they write, so that `rkmon status` shows a hand edit. An apply, a rollback or an adopt also records that it is under way while it runs: one that stops part-way leaves that record, and `rkmon next` names the command that finishes it. An apply or a rollback records too whether this package's code went in meanwhile, which the finishing command's put-back then puts back too, and the change it puts back |
| `docker/.rkmon/package.lock`, `docker/.rkmon/package.lock.<hex>` | In the package folder: the package folder's lock, an empty file, which `rkmon`'s commands share while they run and `rkmon package update` takes alone ([The container's wrapper](#the-containers-wrapper-dockerrkmon)); and beside it, one file for each command that has the lock, naming the command as it was typed and since when, as a refusal shows them. The kernel lets the lock go as the command ends, however it ends: a command that has ended blocks nothing, and the next command to take the lock removes its file |
| The image | Docker's own store of images, as `<name>:<tag>` (`image.name` and `image.tag`); `rkmon image ls` lists this package's images on this host, whatever their names |

### In the container

| Folder or file | What it has |
|---|---|
| `/work` | The work volume, which the monitor and the writer share: the working folder, and the outputs folder ([The outputs folder](#the-outputs-folder)). In it are the monitor's clone of the repository (`repo/`) and its worktrees, the tools the monitor's jobs run (`tools/`, copied from the package's `worker/`), each review's folder, with the pull request's data, its draft and its checks (`workers/`), the agentic files filled in for this deployment (`agentic/`), `scratch/`, `cache/`, and `run/`: the reviews' shared rate budget and the processes the monitor started. An older package's code, `code/`, stays there until the one-time move ends ([The one-time move's copies](#the-one-time-moves-copies)). `rkmon shell` opens a shell in the container |
| `/work/monitor` | The monitor's home. Its `state/` has the state (`state.json`), the journal (`journal.md`), the ledger of the outward writes (`actions.jsonl`), the jobs (`jobs/`), the internal state (`internal_state.json`) and the applied record (`applied_settings.json`): the fingerprint and the package version of the confirmed deploy settings the monitor runs, and when it first started on them, which `rk start` and `rk applied --load` write as they load them, and `rk applied` shows. Its `state/site/lessons.md` has the site lessons: your site's facts and lessons (its hosts, people, tickets and runner pools), which each review role reads beside the package's own lessons and the weekly lessons-learned review edits; in the state, so an update keeps them. A `config.json` there is an older install's, which the first start splits once, by the one-time move ([QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor"](QUICK_DEPLOY_GUIDE.md#moving-from-an-existing-monitor)): `state/one_time_move.json` keeps the move's record and report, and ending standby renames `config.json` to `config.json.migrated`, read-only. Only a work volume that had its `config.json` before this package's first start moves: a `config.json` written later never does, and only `rkmon adopt` makes the move due again |
| `/work/logs` | The logs of the container's start and stop (`start.log`, `stop.log`), and of what the entry point keeps running: the loop that restarts the admin dashboard when it is down (`web.log`), the orchestrator (`orchestrator.log`), and the tick loop that runs without the orchestrator, while it is not set up or is late, or by rules when there is no Claude session (`tick.log`). A log past its size limit is copied to its name with `.1` added, then emptied. The orchestrator's lines are the container's log too, which `rkmon logs` shows |
| `/code` | The code volume: the releases of the package's code that `rkmon start` and `rkmon apply` installed, as root, over the image's own, one folder each in `releases/`; `current` is the one that runs, and `previous` the code that ran before the last install (a release, or the image's own), which `rkmon rollback` puts back; `moved` has the one-time move's marks, one for each copy of a secret the move has dealt with, which it never makes again ([The one-time move's copies](#the-one-time-moves-copies)). Root's, and read-only to every other user |
| `/work/claude/config` | Claude Code's own configuration folder |
| `/work/claude-workspace` | The Claude workspace the agentic files are installed in: the `/rocke-monitor` skill, and the reviewers' team members and agent types |
| `/opt/rocke_monitor` | The image's own code, read-only: the monitor runs it while `/code` has no whole release installed over this image ([Updating](#updating)) |
| `/secrets` | The secrets volume the monitor reads: the GitHub read login (`/secrets/gh-read`), the Claude session's key, the `claude` login (`/secrets/claude_key`), and a file for each of the monitor's other secrets, named as the secret: `admin_access_token`, `gpu_ssh_key`, and the key each model binding names (the first review pass's: `first_review_pass_api_key`) (`rkmon login` shows each, with what needs it) |
| `/secrets-write` | The writer's secrets volume, which only the writer reads: the GitHub write login (`/secrets-write/gh`), the atlassian CLI's login (`/secrets-write/atlassian`), and a file for each of the writer's other secrets, named as the secret: `teams_webhook` and `web_host_ssh_key` |
| `/etc/rkmon/settings` | The settings volume (`rkmon-settings` in `docker/compose.yaml`), read-only here. Its `deploy.json` has your confirmed deploy settings, every key with its value (its default where your file sets none), and which keys your file sets, with the package version and the deploy settings' layout they were confirmed for, when, their fingerprint, and the container's project, ports and image. While `features.review_automation.skills_folder` is set, `skills/` beside it has your team's skills, which the session's render lays over the package's. The monitor and the writer read it, by the monitor's group; no PR runner can. Only a one-off container writes it, as root, at the first `rkmon start`, `rkmon apply` and `rkmon rollback` (`rkmon_lib/deliver.py`): the first start and apply keep the file they replace as `deploy.previous.json` and the skills as `skills.previous/`, which rollback puts back |
| `/etc/rkmon/build.json` | The image's build record: [QUICK_BUILD_GUIDE.md, "Step 3: build"](QUICK_BUILD_GUIDE.md#step-3-build) says what it has |
| `/etc/rkmon/rocm.json` | The ROCm release the image's build installed: its version, its clang, its LLVM tools folder and the GPU targets of its libraries, which `rkmon doctor` and the setup summary read |
| `/ramdisk` | The RAM disk: memory used as a disk, `host.resources.ramdisk_gib` GiB of it; when that is 0, an ordinary folder in the container. In `/ramdisk/slots` each PR runner has a folder, where its job builds and keeps its temporary files. What it has goes with the container |

`rkmon login` and `rkmon secret NAME` install the secrets of both secrets volumes.

### The outputs folder

The files the monitor writes for you. In the container the outputs folder is `/work`, the working folder itself.

| File or folder | What it is |
|---|---|
| `rocke_merge_queue.md` | The Markdown dashboard, rewritten every tick: the admin dashboard's main cards in Markdown (develop's status, the status board, the pull requests that can merge now, the merge plan and the review spend) |
| `rocke_monitor.html` | The admin dashboard's page, as a file |
| `rocke_daily/` | The daily log: what the monitor did, one file per UTC day, named by its date |
| `rocke_weekly/` | With `features.weekly_report` on, each week's statistics report (`rocke_weekly_<week>.md`) and notes for a lessons-learned review (`rocke_lessons_raw_<week>.md`) |
| `rocke_spend/` | The review spend, what the reviews cost: one file per UTC day and one per week (`features.review_automation.spend`) |
| `PR_FIRST_ROUND_<N>.md`, `PR_FOLLOWUP_<N>.md` | A copy of each review the monitor posted, by the pull request's number: its first round, and its follow-ups |
| `PR_UNPOSTED_<N>.md` | An unposted review: one the monitor verified and kept, posting nothing, while review automation is in prototype, the monitor is in standby, or its posting is not set up; the next one for the same pull request is `PR_UNPOSTED_<N>_2.md`, and so on |
| `rocke_insights_*.md` | Insights reports: the insights card on the dashboards shows the tables of the newest one |

### Your private files

A private file is one that may have a private value in it
([Where a private value may show](#where-a-private-value-may-show)): your settings files, a file you install a secret
from, and the package's own `docker/.env`, `docker/compose.override.yaml`, `docker/.env.prev`,
`docker/compose.override.yaml.prev` and `docker/.rkmon/records.json`. Every private file:

- is a regular file, not a link, and yours;
- has permissions 600, so that only you can read it (400 passes too, and a write makes it 600);
- when it is one of your own files, is not in the package folder, a git work tree or a cloud-synced folder
  (OneDrive, Dropbox, iCloud Drive, Google Drive, Box): files there can end up in an image, a commit, the code
  `rkmon start` and `rkmon apply` install, or another machine;
- when it is one of the package's own, has its fixed place in the package folder: the package folder must be outside
  cloud-synced folders, and the package's ignore lists keep those files out of git and out of the image.

`rkmon` refuses a private file that breaks a rule: the refusal names the rule and the command that fixes it. `rkmon`
checks where the file really is, after following any link in its path. It writes a private file as a new file with
permissions 600, then puts it in the file's place, so that no one finds it half written.

### Where a private value may show

A private value names a site, a person or an internal system: an address (of a site, a host, a registry, a mirror or
a download), a login or a user name, an internal model's name, a header's name, a site's labels, a JIRA key or
component, a folder inside the repository, or the repository's name. It is not a secret: a secret is never in a
settings file. A private value shows only:

- on your terminal, in what `rkmon check`, `rkmon confirm`, `rkmon apply`, `rkmon migrate`, `rkmon status` and
  `rkmon doctor` print;
- on the admin dashboard, which only you reach ([The admin dashboard](#the-admin-dashboard));
- in your private files;
- in the system it belongs to, when its feature works there: the repository's name, folders and labels, and the
  automated reviewer name on the first line of each review, on GitHub; the ticket fields in JIRA; the web host's
  address, user and folder on that host; the GPU cluster's settings on that cluster; a gateway's address and header at
  that gateway;
- in what a feature kept instead of sending it, in prototype or while the monitor is in standby, as it would have
  gone out: the admin dashboard's "Not sent" card, the state folder's `unsent.jsonl` and the unposted reviews
  (`PR_UNPOSTED_<N>.md`; [Prototype, live and standby](#prototype-live-and-standby));
- in the messages the monitor sends to people, and only these: links to the pull requests and issues it works on, the
  team dashboard's address (`features.dashboards.team.public_url`), your GitHub username
  (`integrations.github_cli.username`) and the messages' label (`features.alerts.message_label`).

It never shows in the image or its labels, in a log or journal of the settings commands, in a doc, in a template or in
a test's data.

## Features

The deploy settings' `features` decide everything the monitor does. Each key, its default and its meaning are in
[Deploy settings](#deploy-settings).

- `features.review_automation`: a review of each new or changed PR in scope, verified by Claude. `enabled` starts the
  reviews and the container's Claude session; off, there are no reviews and no Claude session, and while the merge
  queue is on, fixed rules run each tick. `mode`: `prototype` prepares and verifies each review and keeps it on this
  machine unposted (`PR_UNPOSTED_<N>.md`), and overlap detection and owner decisions run locally; `live` posts each
  verified review and each overlap notice. Its schedule, parallel reviews, large-PR limits, passes, remote GPU tests
  (`remote_gpu_tests`) and review spend (`spend`) are its own keys.
- `features.merge_queue`: the landing order of the PRs in scope, its gates and merge plan, and priority requests.
  `mode`: `prototype` computes and shows the plan, the gates and the order, and on GitHub arms nothing, puts no PR on
  hold, posts no comment and files nothing (the auto-merge guard only alerts); `live` does these, each part by its
  switch: `auto_merge` (a PR you cleared gets GitHub's auto-merge), `auto_hold` (a PR whose failure is confirmed as its
  own is put on hold), `queue_notices`, `drift_notices`, and `develop_guard` (develop is checked after every merge and
  a break is traced to its PR; live, it is filed as a GitHub issue, an incident when CI is red, and its fixes are
  linked).
- `features.dashboards`: `admin` (your page, on its `port`) and `team` (the page for other teams, by its `delivery`:
  `publish` to a web host, or `serve` here, at its `serve.address` and `serve.port`).
- `features.alerts`: messages to the team chat channel (its webhook is the secret `teams_webhook`), each alert
  (`develop_breaks`, `needs_you`) with its own switch and its own clock, `message_cooldown_minutes` apart at least.
  Each message's first line, "Automated message by" and a name, says which deployment sent it: its
  `features.review_automation.automated_reviewer_name`, else its `container.name`.
- `features.ticket_creation`: JIRA tickets through the atlassian CLI (`integrations.atlassian_cli.jira_url`, its
  login `rkmon login atlassian`) in `project_key`; with an `epic_key`, each ticket is created, linked to the epic
  (`atlassian jira issue link-epic KEY --epic EPIC`), and only then given its description. `follow_up_tickets` and
  `found_issue_tickets` each have a switch.
- `features.weekly_report`: the weekly statistics report and the material for a lessons-learned review.

The alerts, the team dashboard and ticket creation have a `mode` too: five features send, each in prototype or live
([Prototype, live and standby](#prototype-live-and-standby)). The integrations have no switch: a feature that needs
one requires its settings.

Each capability below runs while its features say so. A capability that is off never runs and never fails: its command
prints `OFF: ...` and exits 0. One that is on but not set up says `NOT SET UP: ...`, names the fix and exits 5, and the
tick reports it once, in a line that starts `FEATURE_SKIPPED` and names the capability. `rk features` prints the features' switches, then each
capability's state, what it is configured with and what is missing. Prototype and live run the same capabilities,
review posting aside (it runs only while review automation sends); in prototype, and in standby, what a capability
would post, send or file is kept ([Prototype, live and standby](#prototype-live-and-standby)).

<!-- BEGIN GENERATED: capabilities
     source: rkmon_lib/features.py, CAPABILITIES
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Capability | Runs while | What it does | What it needs |
|---|---|---|---|
| Reviews | features.review_automation.enabled is true | Each new or changed PR in scope is reviewed: the first review pass (features.review_automation.passes.first_review_pass), then the verification pass, which checks every claim (features.review_automation.passes.verification_pass), several reviews at once (features.review_automation.parallel_reviews at first, then adjusted to how busy this host is). The result is kept on this machine. | nothing with no first-pass drafter (the default: the verifier reviews alone); a model's endpoint and key when a model drafts |
| Review posting | features.review_automation is enabled and live | Each verified review is posted on its PR as one comment, by the GitHub login the monitor posts as. In prototype, each stays on this machine as an unposted review. | integrations.github_cli.username, and gh rights to comment |
| Overlap management | features.review_automation.enabled is true (its notices go out while it is live) | Duplicate or conflicting work across open PRs is found and one owner is recorded for each shared piece. While review automation is live, the PRs that should drop their copy get an overlap notice. | comment rights (live) |
| Remote GPU tests | features.review_automation is enabled and features.review_automation.remote_gpu_tests.enabled is true | The GPU-only test cases of each review run on a GPU node of a Slurm cluster over SSH: queued, never holding the review, whose GPU tests row is updated in place when they finish. Without it, GPU-only cases skip here. | the cluster's settings (features.review_automation.remote_gpu_tests.cluster) and the SSH key for its login host (a secret) |
| Merge plan | features.merge_queue.enabled is true | The landing order, the gates and the merge queue view, computed every tick. It only reads GitHub. | the repository and the monitor's clone |
| Priority requests | features.merge_queue.enabled is true | Priority requests are taken in at all times: from the admin dashboard, from rkmon rk prioritize, and from the team dashboard's request box. You approve or decline each one. | nothing |
| Auto-merge | features.merge_queue is enabled and its auto_merge.enabled is true | A PR you cleared (rkmon rk clear N) gets GitHub's own auto-merge once it is validated, so GitHub merges it only when every gate it enforces passes. It never merges directly, never uses admin rights, never forces. | your login's merge rights |
| Auto-merge guard | features.merge_queue.enabled is true (it comments and turns auto-merge off while the merge queue is live) | When someone else's auto-merge is on and one of our gates fails, the auto-merge guard alerts you (prototype) or comments in plain words and turns that auto-merge off (live). It never turns off an auto-merge you turned on yourself. | comment and merge rights (live) |
| Auto-hold | features.merge_queue is enabled and its auto_hold.enabled is true | A PR whose gate failure is confirmed as its own is held, and a change request from your account keeps it from merging until it is fixed and you re-approve it. | review rights |
| Queue notices | features.merge_queue is enabled and its queue_notices.enabled is true | PRs near the front of the merge queue get one status comment, kept current, saying where they stand. | comment rights |
| Drift notices | features.merge_queue is enabled and its drift_notices.enabled is true | When today's develop breaks a PR's own tests, its author gets a notice naming the develop commit that changed the behaviour. | comment rights |
| Develop guard | features.merge_queue is enabled and its develop_guard.enabled is true | After every merge into develop, develop is checked (native build, goldens, byte identity, ABI), and a break is traced to its PR. | the checks venv and ROCm's LLVM tools |
| Develop issues | features.merge_queue is enabled and its develop_guard.enabled is true | A develop break that CI does not show (a test no CI lane runs) is filed as a GitHub issue and assigned to the breaking PR's author. | gh rights to open issues |
| Incidents | features.merge_queue is enabled and its develop_guard.enabled is true | A red develop sanity run, or develop's CI red in a rocKE job, opens an incident: culprit, revert offer, fix tracking. A CI-red develop break is filed as a GitHub issue. | gh rights to open issues and comment |
| Fix links | features.merge_queue is enabled and its develop_guard.enabled is true | An open PR that fixes a develop issue is linked to it once verified, with a comment, and goes to the front of the queues. | gh rights to comment |
| Admin dashboard | features.dashboards.admin.enabled is true | Your page, served by the monitor (where it listens and who can reach it: README.md, "The admin dashboard"). It is the one place to approve or decline priority requests, the tickets waiting for you and the in-depth reviews of large PRs. | the access token (a secret), which every request carries |
| Team dashboard | features.dashboards.team.enabled is true | The dashboard for other teams, with its request box and no admin controls: published on a team web host, or served by this monitor on a port of its own (features.dashboards.team.delivery). | publish: its public_url and its publish block (web_host, username, site_folder) and, in live, the web host's SSH key (a secret); serve: its serve block (address and port) |
| Develop break alerts | features.alerts.develop_breaks.enabled is true | When develop's CI is red and no fix is planned, the team chat channel gets a short message. | in live, the channel's webhook URL (a secret) |
| Needs-you alerts | features.alerts.needs_you.enabled is true | The decisions on the dashboard's Needs-you list (a merge to propose, a stale hold, a priority request, a PR ready for your approval, an in-depth review to approve, reviews that have waited on verification for hours, and the like) go once each to the team chat channel. | in live, the channel's webhook URL (a secret) |
| Follow-up tickets | features.ticket_creation.follow_up_tickets.enabled is true | When a PR merges with our significant review items still open, each gets a follow-up ticket for the PR's author. | features.ticket_creation.project_key and the atlassian CLI (integrations.atlassian_cli.jira_url; in live, logged in) |
| Out-of-scope issue tickets | features.ticket_creation.found_issue_tickets.enabled is true | An issue found outside any PR's review gets a proposed ticket, which waits on the admin dashboard's Tickets card: your Approve files it, your Decline drops it. | features.ticket_creation.project_key and the atlassian CLI (integrations.atlassian_cli.jira_url; in live, logged in) |
| Weekly report | features.weekly_report.enabled is true | Each week the monitor writes a statistics report and the raw material for a lessons-learned review. | nothing |
| Claude session | features.review_automation.enabled is true (in the container) | Claude's part of each tick runs unattended: in the container the orchestrator's long-lived session (rk orchestrate), which runs while review automation is enabled. | Claude Code and its login (the claude login, the session's key) |

<!-- END GENERATED: capabilities -->

Set them in your deploy settings file, then `rkmon confirm` and `rkmon apply`. `rkmon check` and `rkmon confirm`
check their combination, each problem with its fix. `rkmon confirm` refuses a FAIL, such as ticket creation with no
JIRA project, or the team dashboard without the admin dashboard, and lets a WARN through, such as `auto_merge` enabled
while the merge queue is in prototype. With the merge queue live and review automation in prototype, a WARN says that
the queue gates on unposted reviews their authors never saw; its fix sets review automation live, or keeps the merge
queue in prototype.

### Prototype, live and standby

Five features have a `mode`, "prototype" or "live": review automation, the merge queue, the alerts, the team
dashboard and ticket creation. Standby covers the whole monitor, never one feature. The start that makes the one-time
move from an older install puts the monitor in standby, with the reason "moved from an older install", and so does
`rkmon adopt`, with the reason "adopted from an earlier monitor". The monitor stays in standby until you end it with
`rkmon standby off`, which ends the move and the adoption too: [QUICK_USER_GUIDE.md,
"Standby"](QUICK_USER_GUIDE.md#standby) says when.
Each review the older monitor posted before the move, with no automated reviewer name in its first line, counts as
this deployment's own; ending standby moves that time to its own, so a review the older monitor posted meanwhile
counts too.

| What | Prototype | Live | Standby |
|---|---|---|---|
| The feature's work: reviews, plans, alert texts, the team dashboard's page, proposed tickets | done | done | done |
| What it would post, send, push or file | kept as an unsent item | sent | kept as an unsent item |
| The disarm: turning off an auto-merge the monitor armed | sent | sent | waits, then sent on the first tick after standby ends, if still due |
| A write you type, with an `rk` command that writes | refused, naming the feature in prototype | sent | refused, naming standby |

An unsent item keeps its fields, and the journal gets one line for it, without its content; items are kept for one
log rotation, and none is ever sent later. The team dashboard's page counts as one item. The admin dashboard's "Not
sent" card lists them, each with why it was kept: its feature in prototype, such as "review automation in prototype",
or "the monitor in standby" ([The dashboards](#the-dashboards)). Going live sends what comes next, never a kept item.

An unposted review (`PR_UNPOSTED_<N>.md`), one the monitor verified and kept, counts as the monitor's last review of
its pull request: gate G1 reads it, and the next review of the pull request is its follow-up. The pull request's author
never saw it, and nothing that reads the reviews on GitHub does: the follow-up tickets of a merged pull request, and
the open findings the auto-merge guard's comment names, come from posted reviews alone.

While the monitor is in standby, the admin dashboard's top line and the doctor's `WARN` row `standby` say since when
and why nothing is sent, with what ends standby: its command, once `rkmon status` and the dashboards read right. A
team dashboard served on this host shows other teams a short notice in its place ([The team
dashboard](#the-team-dashboard)).

## Starting templates

`rkmon template deploy --profile NAME FILE` writes the deploy settings template with the features set for a common use
(the deploy schema's profiles, `x-profiles` in `schema/deploy_settings.schema.json`). A profile is never a setting: the
file's features decide, and `rkmon confirm` checks their combination. Every template starts review automation and the
merge queue in `prototype`.

<!-- BEGIN GENERATED: profiles
     source: schema/deploy_settings.schema.json, x-profiles
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Profile | What it runs | Its values | Write it with |
|---|---|---|---|
| `full` | reviews, the merge queue and the admin dashboard | every key's default | `rkmon template deploy --profile full FILE` |
| `review-only` | reviews and the admin dashboard | `features.merge_queue.enabled: false` | `rkmon template deploy --profile review-only FILE` |
| `merge-queue` | the merge queue and the admin dashboard, without Claude | `features.review_automation.enabled: false`<br>`features.merge_queue.drift_notices.enabled: false` | `rkmon template deploy --profile merge-queue FILE` |

<!-- END GENERATED: profiles -->

What follows from the features:

- **`rkmon login`, `rkmon check` and `rk doctor`** ask for and check only what the features run: a login
  nothing uses is `NOT NEEDED`, a check of a part that does not run is `SKIP`. In `rkmon login` such a login is `OFF`,
  but for the atlassian login's `WARN` while ticket creation is in prototype ([QUICK_DEPLOY_GUIDE.md, "Step 5:
  login"](QUICK_DEPLOY_GUIDE.md#step-5-login)).
- **What starts:** with the admin dashboard off no web server starts (`rk serve`, and the container's loop that
  restarts it); with review automation off no review driver and no Claude session start. The tick skips the steps of
  the features that are off (the merge plan and queue, the develop guard, the reviews, the requests, the dashboards).
  `rk features --runs PART` (`web`, `claude`, `reviews`, `rules`, `merge-queue`, `develop-checks`) answers with its
  exit code whether the deployment runs that part.
- **No Claude session (the merge queue with review automation off):** each tick runs as `rk rules-tick`, which the
  container's tick loop starts. For each line whose action is fixed it runs that action's command, with the command's
  own guards: `ARM_READY` (`rk arm`), `DISARM_NEEDED` (`rk disarm`), an `AUTOMERGE_RISK` that names `rk
  automerge-off`, `BISECT_NEEDED` (`rk bisect`) and `QUEUE_NOTICE` (`rk queue-notice`). Every other line the
  orchestrator would take up (a merge proposal, an auto-merge that jumps the plan, a develop break or incident, a
  GitHub login problem, a line no rule knows), and a rule whose command refused, waits for a person on the admin
  dashboard's Needs-you list and in `rk status`; none is dropped, and nothing is filed.
- **The review gate without reviews:** where the monitor reviews, gate G1 is our own 🟢/🟡 review at the head. A
  deployment that makes no reviews gates on the repository's own human review instead: G1 passes when no change
  request stands, GitHub's review decision is APPROVED (its required reviews, code owners' included, are met; a
  repository that requires none needs one approval) and a person's approval (not the author's, not a bot's) is at the
  head. A review comment posted under `integrations.github_cli.username`, by another deployment or by hand, neither
  passes nor fails G1, so beside a `review-only` deployment, its reviews do not stand in for an approval; that
  account's change request or approval counts like anyone's. A rebase-only or merge-only push keeps the approval: the
  tick checks each moved head once (`CARRIED #N` lines; at most five checks a tick) and reads the approval at the old
  head as at the new one. A substantive push needs a new approval.
- **Two deployments on one host:** neither reads the other. `rkmon start` and `rkmon apply` check this host alone:
  the container name, compose project and volumes are this deployment's own, and the ports it opens are free. Nothing
  checks what two deployments do on one repository, such as which one runs its merge queue: QUICK_DEPLOY_GUIDE.md,
  "Several deployments", has the rules that are yours to keep.

## The setup summary

`rk features` prints it at any time (from the host: `rkmon rk features`):

1. **Features:** each feature's switches (`features.review_automation`: on, prototype; ...), what starts (the web
   server, the Claude session, the review driver, and ticks run by Claude or by fixed rules), then every capability in
   one table: its state (on, off, or on and incomplete), when it runs, what it is configured with, and what is missing
   with the command that fixes it; then two lists, "Configured and on" and "Not configured or off". A first-pass model
   that failed the start's test chat or tool-call probe shows here: Reviews is "on (incomplete)", with the reason.
2. **Configuration:** every value the features use: where things are; GitHub (the repository, rocKE's folder, where
   develop issues go, your login, the scope labels, the code-owner teams, the GitHub CLI); the dashboards; JIRA; team
   chat; the models (each pass's model and effort; a first-pass model's source, endpoint and model, what the start's
   checks found and the limits in use: context length, max output, rate limits; the routing); ROCm (its LLVM tools,
   the image's ROCm release and the reference clang); the parallel reviews (asked, in effect, and why) and the
   parallel builds; the remote GPU tests, while they are on; the monitor's stop state; and every secret, shown only as
   "set" or "missing" with its place, or as "not used" with why.

## The dashboards

The **admin dashboard** is your page: everything the dashboards show, and the only place with Approve and Decline
(priority requests, the tickets waiting for you, and the in-depth reviews of large pull requests, on their Needs you
lines). `rk serve` serves it, and the entry point keeps it running. Only the admin dashboard has the Scoreboard
(`rk scoreboard`: per PR author, what lands and the rework it causes for others, from `state/scoreboard_events.jsonl`),
Review spend, "Not sent" and Today's log, the last card; while the monitor is in standby, its top line says so. The
**team dashboard** is the same page for other teams, without those cards:

- **Admin dashboard** (`features.dashboards.admin.enabled`): where it listens, who can reach it and the access token
  every request carries are in [The admin dashboard](#the-admin-dashboard), below.
- **Team dashboard** (`features.dashboards.team.enabled`): how it reaches other teams, and what a web host needs for
  it, are in [The team dashboard](#the-team-dashboard), below.

The container publishes them on this host where your confirmed settings say: [Ports, volumes and
docker/.env](#ports-volumes-and-dockerenv). `rkmon doctor` checks the admin dashboard's reach and its access token,
each served dashboard's port (answering, free, or held by another program), and that the container publishes them
where your confirmed settings say.

**Tickets waiting for you.** With `features.ticket_creation.found_issue_tickets` enabled, a problem found along the
way (not a develop breakage, not a PR's review finding) gets a proposed ticket, never a ticket: the monitor's Claude
session writes it (`rk ticket-draft FILE`: a `---` block with `title:`, `type:`, `priority:`, `assignee:` and a
one-line `summary:`, then the full description), and it waits on the admin dashboard's "Tickets waiting for you" card,
one row each, with a link to its full text. **Approve** files it through the atlassian CLI
(`integrations.atlassian_cli.jira_url`, logged in) in `features.ticket_creation.project_key`, linked to
`features.ticket_creation.epic_key` when one is set (created, then linked, then given its description), with the
default issue type, the components, the labels and the required fields `features.ticket_creation` names. It records
the key in the ledger and shows it in the row; **Decline** drops it, journaled. One action per proposed ticket. While
ticket creation is in prototype, or the monitor in standby, Approve files nothing: it keeps the proposed ticket as an
unsent item, listed under "Not sent". Nothing is filed while ticket creation is off or the monitor is stopped.
`rk tickets` lists them; `rk ticket-file ID` and `rk ticket-decline ID` are your command-line fallback, which the
session's guard never lets it run. Follow-up tickets for merged PRs (`follow_up_tickets`) are not proposed: they are
filed without asking. Ticket creation is set up only when `features.ticket_creation.project_key` and
`integrations.atlassian_cli.jira_url` are set and the atlassian CLI, which every image carries, is logged in
(`rkmon login atlassian`): until then `rk features` and `rkmon doctor` name what is missing, and a ticket command
refuses with `NOT SET UP`.

**Not sent.** The admin dashboard's "Not sent" card lists what a feature in prototype, or the monitor in standby,
kept instead of sending, newest first: when, its type, what it was for, why it was kept, and its content as it would
have gone out. It shows the 100 newest; the rest are in the state folder's `unsent.jsonl`. A review kept unposted and
the team dashboard's page link to their file, which the admin dashboard's service serves with the access token: the
review as text, the page with no script running. While the team dashboard does not send, the card links its page as
the last tick made it, published or served here alike. The card shows while an item is kept, or while one may be: in
standby, or while a feature that sends is in prototype.

### The admin dashboard

It listens on this host's 127.0.0.1 only, on `features.dashboards.admin.port` (18614 unless you changed it). From your
own machine, open an SSH tunnel, `ssh -L 18614:127.0.0.1:18614 <deploy-host>`, then open http://127.0.0.1:18614/. The
browser asks for a user name and a password (HTTP Basic): type any user name, and the access token as the password
(`rkmon secret admin_access_token --generate` made it, and showed it once). Every request needs it, from this host
too. Without it the service does not start, and `rkmon doctor` says so. No page carries the token.

### The team dashboard

The team dashboard is the admin dashboard's page for other teams, without its cards for you alone, its buttons or its
access token, and it needs the admin dashboard on. With `features.dashboards.team.priority_requests.enabled`, the page
has a box where other teams ask for a pull request to go first; each request waits for your Approve or Decline on the
admin dashboard. Its keys are under `features.dashboards.team` ([Deploy settings](#deploy-settings)), and
`features.dashboards.team.delivery` says how the page reaches other teams:

- **`serve`**: the monitor serves the page itself on this host, at `features.dashboards.team.serve.address` and
  `features.dashboards.team.serve.port` (127.0.0.1 and 18615 unless you change them), and runs the request service
  itself, which takes the priority requests. Nothing runs on another host.
- **`publish`**: after every tick, the monitor copies the page with rsync over SSH to the web host
  `features.dashboards.team.publish.web_host`, as the user `features.dashboards.team.publish.username` (`<web-user>`),
  into the folder `features.dashboards.team.publish.site_folder`, and copies the priority requests back; the web host
  serves that folder at `features.dashboards.team.public_url`. The SSH key is the secret `web_host_ssh_key`, and
  nothing is pushed until the web host's own keys are recorded ([The hosts' own keys](#the-hosts-own-keys)).

Until `features.dashboards.team.mode` is "live", nothing goes to the web host, and a served page is served on
127.0.0.1 only, whatever `features.dashboards.team.serve.address` says ([QUICK_USER_GUIDE.md, "Go
live"](QUICK_USER_GUIDE.md#go-live)).

While the monitor is in standby ([Prototype, live and standby](#prototype-live-and-standby)), nothing goes to the web
host either, and a served page answers every request, whatever its method and path, with 503 and a short notice
alone: the monitor is in standby, and its team dashboard opens there once its operator ends standby. It takes in no
priority request meanwhile. It reads standby as each request comes, so the page is back as soon as standby ends, with
no restart. You still see the page from the admin dashboard's "Not sent" card ([The dashboards](#the-dashboards)).

**What the web host needs**, with `publish`: rsync, and an SSH login for `<web-user>` that accepts your key (its public
half in that user's `~/.ssh/authorized_keys`). With priority requests on, the page is served by the monitor's request
service, `tools/webhost_server.py`, in place of any other web server: it serves the site folder and takes the
requests, which it keeps at the top of the user's home folder, outside the site folder. It runs on the web host's
Python 3.6 or newer, as `<web-user>`'s user service `rocke-monitor.service`, which keeps running after that user
logs out. It serves the site folder over plain http at the root of its own port, so
`features.dashboards.team.public_url` is then `http://<web-host>:<port>/`, with no path, on a port from 1024 to 65535
that other teams can reach.

**Install the request service** with `rkmon web-host install`, once the web host's key is set and its own keys are
recorded ([QUICK_DEPLOY_GUIDE.md, "Step 5: login"](QUICK_DEPLOY_GUIDE.md#step-5-login)); in prototype it readies the
web host for live. The writer runs it in a one-off container, from your confirmed settings, over SSH with that key.
It looks first (what serves the port, the user service, lingering, Python's version, the site folder), then copies
the service and its user service into the site folder's first folder, makes the site folder when it is missing, puts
the user service in place, keeps the user's services running after logout (lingering), starts the service and checks
that it answers. It ends:

- `INSTALLED`; or `NOTHING TO DO` with delivery `serve` or priority requests off, as nothing on the web host needs
  the service then;
- `FAILED`, nothing changed, naming why: the web host's key or its own keys not set yet (exit 3, with the command
  that sets them); no python3 on the web host, or one older than 3.6, found at the look; another program on the port
  (when it is this login's own `python3 -m http.server`, stop it, then install again); lingering that the login
  cannot turn on, with the command for the host's admin;
- `ERROR`, part-way, saying what it leaves on the web host: a copy that failed, an install that ssh lost or that ran
  out of time, or a service that did not come up (the server it replaced then runs again, as it was);
- `INTERRUPTED` at a Ctrl-C (exit 130), saying what it leaves on the web host, then the command again;
- `REFUSED` in standby, as the monitor you moved from may still use the web host.

Once the team dashboard is live, `rkmon doctor` checks that the service answers at
`features.dashboards.team.public_url`, with this install as its fix.

## Settings files

A settings file has a whole setup, so a container on another machine is set up from it. No secret, key or login is
ever in one: set them on the new machine (`rkmon login`, `rkmon secret`).

- `rkmon confirm --settings FILE` records the file for the deployment (its image, sizing, ports and ids), and the
  first `rkmon start` sets up from it; `rkmon template deploy FILE` writes one to start from (QUICK_DEPLOY_GUIDE.md
  lists what it needs). Once the container runs, a change is the file changed, then confirmed and applied
  ([QUICK_USER_GUIDE.md, "Change a setting"](QUICK_USER_GUIDE.md#change-a-setting)).
- Moving an earlier monitor's state in, from a copy of its /work: `rkmon adopt --from DIR` (QUICK_DEPLOY_GUIDE.md);
  the deployment starts in standby.
- From a monitor that runs in a container and keeps its settings in `config.json`: three runs of
  `python3 tools/translate_production.py --container NAME`, each `PASSED` line naming the next run.
  1. `--report-only` prints what each of its settings becomes in a deploy settings file of layout 2
     (`schema_version: 2`, `schema/deploy_settings.schema.json`), and writes nothing. The report says what each rule
     did, where each key that is not carried goes, which logins and secrets the file's features need, and what the
     file's own check found.
  2. `--round-trip` takes that file back to the names the monitor runs with and compares the two, key by key, with
     what each feature does and the usage the monitor recorded, priced both ways. Each difference is a row: PASS when
     the package's list of intended differences (`tools/round_trip_differences.json`) gives its reason, or a file of
     rows you agreed, `--agreed FILE` (yours only; a row's form: `tools/round_trip.py`), and FAIL otherwise, exit 4.
     With no usage recorded it fails too, and the row's fix names where the transcripts are looked for. It may take
     up to 30 minutes.
  3. `--out FILE` writes the file, yours only. With `--round-trip`, FILE gets the round trip's report instead, and the
     terminal only its outcome.

  Run it on the container's host, from this folder. It runs a one-off container of NAME's own image, never pulled,
  with no network, as the monitor's user with no capability, with one CPU and 2 GiB of memory at most: NAME's work and
  secrets volumes are mounted read-only, and this folder beside them. There it reads `config.json` and the shape of
  the older secret that held the Claude session's settings (names and settings, never the key); nothing in NAME or its
  volumes changes. A setting the file cannot carry stops each run (exit 4), and nothing is written. Past the time
  limit (ERROR, exit 1), or on Ctrl-C (INTERRUPTED, exit 130), it removes the one-off container, and its next line
  runs the same command again. `python3 tools/translate_production.py --help` says the rest.

Keep your settings files outside the package folder: each is a private file ([Your private
files](#your-private-files)), as is each file `rkmon secret` reads a secret from. tools/translate_production.py writes
its `--out` FILE as one, refuses (exit 4) a FILE in a folder the rules forbid, and reads an `--agreed` FILE only when it
keeps them.

The format (`rkmon_lib/settings_file.py`, `schema/deploy_settings.schema.json`; YAML in a small subset, or JSON), by
its top-level keys; for your editor to check it as you type, see "In an editor" in the [Settings
reference](#settings-reference):

| Key | What it has |
|---|---|
| `kind: deploy_settings`, `schema_version: 2` | What the file is, the first keys after the comments; a file without them is refused |
| `container`, `image`, `updates`, `repository`, `host`, `monitor`, `features`, `integrations`, `model_bindings` | The schema's keys ([Settings reference](#settings-reference)); a value left out takes its default |
| `secret_values:`, `secrets:` | Never in a settings file: a file with either section is refused, naming the section and how many entries it has, never a value. The fix moves each value into a file only you can read and deletes the section |

`rkmon template deploy FILE` writes the file a deployment starts from, each key with its description from the schema
(`--profile NAME`, NAME a profile of [Starting templates](#starting-templates));
[QUICK_DEPLOY_GUIDE.md, "The keys you must fill in"](QUICK_DEPLOY_GUIDE.md#the-keys-you-must-fill-in) lists the keys
that have no default. A file in an earlier layout is refused, nothing in it read: its fix is the command that moves it
to this layout, or, for a file with no `kind` line, a new file from the template. The image has a settings file of its
own, the build settings file (`kind: build_settings`, its sections at the top level; `rkmon template build FILE`
writes one; [Build settings](#build-settings) lists its keys).

## The two model roles

Each review has two model roles. The **reviewer model** may draft it: the first review pass's model, any model, drafts
each review once you name one ([Swapping the reviewer model](#swapping-the-reviewer-model)). Claude checks and decides:
the orchestrator, whose model is the **orchestrator model**, the verification pass and the code quality pass run on
Claude models, and only they verify, decide the gates and post. Three more optional passes may do a first read for one
of them. Each is a pass, with a model and an effort, in your deploy settings file (their defaults:
[Deploy settings](#deploy-settings)):

| Pass | Its key | What it does | Its model |
|---|---|---|---|
| Orchestrator | `features.review_automation.orchestrator` | The Claude session: the ticks, the verification's decisions, the gates, the posts | Claude |
| Verification pass | `features.review_automation.passes.verification_pass` | Reviews each pull request blind, then checks each claim of a first draft against the code, and writes the review that gets posted | Claude |
| Code quality pass | `features.review_automation.passes.code_quality_pass` | Reviews each pull request's design (single responsibility, one home per fact, layers, names, tests) for the verification pass | Claude |
| First review pass | `features.review_automation.passes.first_review_pass` | The first draft of each review, for the verification pass | Any |
| CI triage pass | `features.review_automation.passes.ci_triage_pass` | A first read of the pull request's failing CI jobs, for the verification pass | Any |
| First code quality pass | `features.review_automation.passes.first_code_quality_pass` | A first read of the pull request's changes by the code quality rules, for the code quality pass | Any |
| Notice draft pass | `features.review_automation.passes.notice_draft_pass` | The first draft of an overlap or drift notice, for the orchestrator | Any |

A pass's `model` is `opus`, `sonnet` or `haiku` (built in: `claude-opus-5-5`, `claude-sonnet-5-5` and
`claude-haiku-5-5`) or the name of a model binding. Its `effort` is `low`, `medium`, `high`, `xhigh` or `max`; a model
that is not Claude takes only `low`, `medium` or `high`. The four optional passes are `default` until you name a model:
then the pass they help does their work, with its own model and effort (`default` as an effort takes that pass's too).
The settings' checks refuse a model that is not Claude on the first three. `rk roles`, `rkmon confirm` and `rk doctor`
show every pass resolved; here the CI triage pass runs on a binding named `local-llama`:

```
pass                     model                      provider  effort
orchestrator             opus                       claude    high
verification_pass        opus                       claude    high
code_quality_pass        opus                       claude    medium
first_review_pass        default → opus             claude    high (from verification_pass)
ci_triage_pass           local-llama → llama3.1:8b  ollama    -
first_code_quality_pass  default → opus             claude    medium (from code_quality_pass)
notice_draft_pass        default → opus             claude    high (from orchestrator)
```

**Model bindings** (`model_bindings.<name>`) are the models a pass may name besides the built-in ones, each under a
name you choose: [Model bindings](#model-bindings) lists each provider's keys, and [QUICK_USER_GUIDE.md, "Use another
model for a pass"](QUICK_USER_GUIDE.md#use-another-model-for-a-pass) walks the change. A binding named `worker` is the
local review loop below.

**The orchestrator model** (`features.review_automation.orchestrator.model`) is the Claude Code session's. The current
Claude Opus and Sonnet families are known to drive the monitor well; for another model, the setup summary
(`rk features`) shows a warning: the skill files and the verifiers' instructions are long and use many tools, so a less
capable model may skip steps or misread the rules.

**The optional passes' leads.** The CI triage, first code quality and notice draft passes each do a first read of a
task a Claude pass owns. Their model hands that pass leads, never findings: each lead quotes the lines that show it,
each quote is checked at its line with no model (a lead whose quotes are not found is dropped, and the file says so),
and the Claude pass checks each lead before it uses one and still does its whole task.

| Pass | When it runs | Its leads |
|---|---|---|
| CI triage pass | The review driver's job `rk first-pass N --head SHA`, once the pull request is staged, beside the draft and the checks | `ci_triage.md` in the review directory, for the verification pass |
| First code quality pass | The same job | `quality_leads.md` in the review directory, for the code quality pass |
| Notice draft pass | `rk notice-draft N --kind overlap\|drift`, which the orchestrator runs when an `OVERLAP_NOTICE_DUE` or `DRIFT_FAIL` line names it, while the pass has a model | `<tmp>/notice_draft_pr<N>_<kind>.md`, for the orchestrator, which writes the notice it posts |

A first read never delays its task: a model that is not at hand, fails, answers with no JSON twice or runs past its
time or token cap is a miss, one line in `state/first_pass.jsonl` and the journal, and the Claude pass works without
it (`rk review-prep` and `rk notice-draft` say so). `rk doctor` checks each optional pass's model, and `rk roles` shows
each one's last run.

**The local review loop** is the first review pass on the binding named `worker` (`model_bindings.worker`): a model you
run, on an Ollama server (this host's, at an address the container reaches, or another host's) or behind an
OpenAI-compatible API (a gateway, vLLM, LiteLLM), drafts each review, first rounds and follow-ups, with a fixed menu of
tools that only read. An answer's reasoning never joins its draft. A start checks the model while its last check is
missing, was of another address or model, or did not pass: its model list, a short test chat and a tool-call probe
shaped like the reviews' own requests. A model that fails the test chat or the probe is refused: no review starts
while it is, and `rk features` names why. A check that gets no answer, or a rate limit, refuses nothing, and the next
start checks again. The start's warning and `rkmon doctor` give the fix, by cause:

- HTTP 401 or 403, the key or how it is sent: set the key again (`rkmon secret NAME FILE`, NAME the binding's
  `api_key_secret_name`), then `rkmon stop --drain` and `rkmon start`; or correct how the binding sends the key
  (`api_key_sent_as`, `api_key_header`), then `rkmon confirm` and `rkmon apply`.
- A server error (HTTP 5xx): put the server right, then `rkmon stop --drain` and `rkmon start`.
- A failed tool-call probe: a model that can call tools in the binding, then `rkmon confirm` and `rkmon apply`.
- A model the server does not list: serve it there, then stop and start; or name a model the server serves, then
  confirm and apply.
- Any other refusal of the test chat, such as a route the server does not have: correct the binding, then confirm and
  apply.

How the loop works and how to choose its model: `docs/review_pipeline.md` §3.1, §3.9 and §5.

## Swapping the reviewer model

The reviewer model is one setting, `features.review_automation.passes.first_review_pass.model`. Whatever it is, the
verification pass checks every claim of the draft against the code and writes the post. Line 1 of each review,
`Automated review by <automated_reviewer_name>: Model - <model>, effort: <effort>`, names the model that drafted:

| The model | Who drafts | Line 1's model |
|---|---|---|
| `default` | Nobody: the verification pass reviews alone | The verification pass's model in words (`Claude Opus`), at its effort |
| `worker` | The local review loop ([The two model roles](#the-two-model-roles)) | The binding's model ID, at effort `max` |
| `opus`, `sonnet`, `haiku` or another binding | The tool loop, on that model; a Claude model is reached through `integrations.claude_code`, as the Claude session is | The model's ID |

To swap, set it (for a binding of your own, add the binding first: `rkmon template section model_bindings` prints its
three forms, and its key goes in with `rkmon secret NAME FILE`, NAME its `api_key_secret_name`, never in the file),
then make the change as [QUICK_USER_GUIDE.md, "Change a setting"](QUICK_USER_GUIDE.md#change-a-setting) says. Post or
drop the reviews awaiting verification first: `rk post` checks line 1 against the first review pass in effect.
`rk roles`, `rk doctor` and `rk features` (Reviews: the first review pass and line 1) check the swap.

## Insights

The dashboards' insights card shows our recent 🔴/🟠 findings by cause, the open develop issues, the merge-conflict hot
files and the CI time per PR, and where the time goes per PR over the last 14 days: how often each PR was rebased
(force-pushes and develop merges), how often and for how long it conflicted with develop, our review rounds and the
fix rounds after them, and its CI jobs' waits for a runner (and per runner label); the top five PRs of each, and the
totals. `rk insights [--days N]` prints the same, and the weekly report carries its week's. All of it comes from what
the monitor already records.

**Review spend** (`rk spend`, the admin dashboard's Review spend card and the weekly report's section) is what each
review, posted, kept unposted or closed without a post, cost, by part and per UTC day and ISO week, from this
machine's Claude Code transcripts, priced with `features.review_automation.spend`; with no prices it is off, and
`rk usage` counts tokens either way. A monitor that took over another's state also counts that monitor's runs from
`state/spend_history.json`, when there is one, up to the time the file names. How each part is counted:
`docs/review_pipeline.md` §3.12.

## Parallel reviews and parallel builds

The parallel reviews (how many reviews are prepared at once, each PR in its own worktrees) and the parallel builds (how
many local builds, each with its tests, run at once) each start at your setting,
`features.review_automation.parallel_reviews` and `monitor.parallel_builds` (their ranges and defaults:
[Deploy settings](#deploy-settings)). The monitor then moves each one step at a time, within its range, by this host's
load, and starts a review or a build only when it fits; lowering a setting stops no work. How each one moves, and what
goes on the RAM disk: [docs/review_pipeline.md, "3.3 Parallel reviews"](docs/review_pipeline.md#33-parallel-reviews)
and [docs/review_pipeline.md, "3.5 Local checks: the parallel
builds"](docs/review_pipeline.md#35-local-checks-the-parallel-builds); the review driver, which keeps the parallel
reviews full: [docs/review_pipeline.md, "3.4 The review driver"](docs/review_pipeline.md#34-the-review-driver).

- **The first review pass's endpoint** may cap the parallel reviews. A local Ollama drafts one review at a time
  unless its binding's `limits.max_parallel_requests` allows more: raise it once you have checked the model's size,
  the free memory and Ollama's own parallel setting. A remote endpoint's rate limits, which all reviews share, are its
  binding's `limits.requests_per_minute` and `limits.tokens_per_minute`.
- **The RAM disk** (`host.resources.ramdisk_gib`, 0 for none) keeps the reviews' worktrees and builds in memory. It
  counts against the container's memory limit
  ([QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing)), and the compose override mounts
  it from your confirmed settings, with `exec`, as `docker/compose.local.example.yaml` shows. `rk doctor` checks that
  it is mounted, large enough, has room left, and runs a script written there.
- **Each change** of a setting is a line of `state/scaler.jsonl` with its inputs and its reason. `rk status` shows
  each current setting against its range, with the load it was set from and its last changes:

```
parallel reviews 3 of 1–5 — CPU 42%, mem 51%, RAM disk 9.2/24 GiB
```

## Verifying and posting: the orchestrator's steps

- Each step of a verification is one `rk` command (`rk --help` lists each with its options). How the steps fit
  together: `docs/review_pipeline.md` §3.6 (one PR, start to finish, with the premises `rk post --premises` checks
  before a post) and §3.12 (the orchestrator's steps, drift and review spend).
- The checks' scripted sections are leads that cost no model: `docs/review_pipeline.md` §3.11.
- A review too large for one verifier (`REVIEW_MANUAL`): `rk review-split N` cuts it into parts, each filled up to
  8000 changed lines by default area by area (`features.review_automation.large_prs.split_part_lines`; `--max-lines`
  overrides it), with one verifier per part, which runs the part's code quality checks too, and one more that checks
  what spans the parts and writes the single post (`docs/review_pipeline.md` §3.12).

## Ticks: idle or full, the tick filter and the tick loop

- `rk tick` runs one tick, idle or full (`rkmon_lib/idle.py`). Every tick first reads what may have changed: the open
  PRs' heads, last updates, drafts, mergeability, CI, auto-merge and labels; the base branch's head and the CI runs of
  its recent commits; the open develop issues; the team dashboard's priority requests; the monitor's state and the jobs
  it names; the files the work in flight waits on; the ledger and the other files commands write; the settings; the
  code. The tick is idle when all of it reads as at the last full tick and none of that tick's timers has run out (a
  head settling, a follow-up gap ending, a draft held for its model's budget free to start, the board falling due, a
  retry): it refreshes the dashboards' "checked" stamps and prints one line, such as
  `TICK idle: no change since 14:05Z (12 PRs, develop 3f2a9c1d0e)`. Anything else, a part it cannot read included,
  makes it a full tick. A full tick runs at least every `monitor.full_tick_max_interval_minutes` minutes (0: every tick
  is full), and on `rk tick --full`; `rk tick --dry-run` says which the next tick would be, and why, and writes
  nothing. The board's Monitor row counts the day's idle and full ticks and their average times.
- Each stage registers what its tick reads, its timers and its "checked" stamps in its own module (rkmon_lib/idle.py's
  `part`, `file_part`, `timer` and `stamp`), beside the rule each one stands for, so a new stage needs no change to
  the tick itself.
- **Where ticks run.** In the container, the orchestrator (`rk orchestrate`) runs `rk tick` every
  `monitor.tick_interval_minutes` minutes, and the rule filter (`rk tick-filter`) reads each tick's output first, so
  the Claude session gets a turn only for the lines that need Claude; an idle tick reaches neither. The container's
  own tick loop runs the ticks while the orchestrator is not set up or its ticks are late, and each tick by fixed
  rules (`rk rules-tick`) while no Claude session runs ([What runs in the container](#what-runs-in-the-container)).
  On a host install, `tools/tick_loop.py`, which your Claude Code session starts
  ([Container and host](#container-and-host)), runs `rk tick` at the tick interval and appends each tick's output to
  `state/tick_loop.log` in the monitor's home. It exits, which wakes the session, when a tick prints a line that needs
  the orchestrator (LOOP_WAKE), when a tick fails or runs too long and is stopped, and at its heartbeat
  (LOOP_HEARTBEAT, with the tick counts); one loop runs at a time, and a second prints LOOP_RUNNING and exits 2.
- `rk tick-now` asks whichever tick loop runs here for a tick at once; `rk loop-status` says whether one runs, the
  last tick and full tick, the next deadlines and the day's tick counts.
- `rk tick-filter` reads a tick's output and prints the lines that need the orchestrator, from one table of line types
  (`rkmon_lib/tickfilter.py`: each with its class, the step that prints it and what it asks for). A line no row knows
  goes to the orchestrator. A standing line, one the tick repeats while its condition is true (PROPOSE,
  FIX_CANDIDATE, OVERLAP_NOTICE_DUE …), wakes the orchestrator once, not on every tick; the host loop's heartbeat
  repeats those still standing.

## Remote GPU tests (optional)

With `features.review_automation.remote_gpu_tests.enabled` true and review automation on, the test cases a review's
checks skip because this host has no GPU run on a GPU node of a Slurm cluster reached over SSH. Nothing waits for
them: the review is posted with a "GPU tests" row marked pending, which is filled in when the job ends. How the cases
are chosen, run, judged against develop and cleaned up after:
[docs/review_pipeline.md, "3.7 Remote GPU tests (optional)"](docs/review_pipeline.md#37-remote-gpu-tests-optional).

Every value is your site's, under `features.review_automation.remote_gpu_tests` ([Deploy settings](#deploy-settings)
lists each key): the login host and the user there, the partition, the accounts in the order to try, each GPU target
and the cluster's GPU type for it, the nodes to avoid, the time limit, the folder on the cluster and the Python on the
nodes. The login's SSH key is the secret `gpu_ssh_key`, and no job is sent until the login host's own keys are recorded
([The hosts' own keys](#the-hosts-own-keys)).

The commands (`rk --help` gives each one's options):

- `rk gpu-check` shows, read-only, what the login host has: the nodes of each GPU type, your accounts, the tools.
- `rk gpu-row N` sets the "GPU tests" row of our posted reviews of PR N to what its jobs say, when the tick prints
  `GPU_DONE #N`.
- `rk gpu-run N` queues PR N's GPU-only cases by hand: `--same-node` on the node of its last job on that target,
  `--node NAME` on that node. A node asked for that fails the GPU probe is not swapped for another: the job ends, and
  the tick says so. `rk gpu-run --develop --arch A --case ID` queues develop's named cases the same way.
- `rk gpu-cancel N` gives up on PR N's queued or running jobs, such as one the tick reports as waiting longer than
  `features.review_automation.remote_gpu_tests.wait_report_after_minutes`.
- `rk gpu-status` lists every job recorded here, with its node; `rk gpu-queue` lists the cluster login's jobs in the
  queue; `rk gpu-poll` does what the tick does for the jobs.

While review automation is in prototype, `rk gpu-row` and the GPU notice only journal what they would do; `rk gpu-run`
still queues a job, since it writes nothing to GitHub.

## Stopping the monitor

Each way to stop the monitor, from a pause to a kill, with what it keeps and the command that brings it back:
[docs/restart.md, "The four stop levels"](docs/restart.md#the-four-stop-levels). In the container, the wrapper's
`rkmon pause`, `rkmon stop` and `rkmon start` run them from the host
([The container's wrapper](#the-containers-wrapper-dockerrkmon)).

## Updating

An update puts a newer package's code in the container. The commands each case takes, in their order:
[QUICK_USER_GUIDE.md, "Move to a new version"](QUICK_USER_GUIDE.md#move-to-a-new-version). What each one does, on the
host, in the package folder's `docker/`:

1. `rkmon package update --from NEW --dry-run`, NEW being the newer package's folder or tarball, shows what would
   change, and its next line names the commands the update needs. Without `--from`, the package comes from the
   repository and branch the `updates` section of your deploy settings file names, fetched with git (a repository
   that needs a login takes git logged in to it): the one folder of the branch's newest commit with a package in it.
   The update stops when that commit has none, or more than one.
2. `rkmon package update --from NEW` puts the newer package in this package folder's place. Your private files and
   every other local file of `docker/` go with it, and the folder it replaced stays beside it as `<folder>.prev`.
   When the files the image keeps its own copies of differ, its `image` row is a WARN: the newer package needs a new
   image, built from it ([QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)) and named by `image.tag`. Its next line names
   the step after it: `rkmon migrate` when your deploy settings file needs the newer layout, else `rkmon confirm`.
3. `rkmon confirm` shows the change of package, `This package: <version> -> <version>`, with
   `(other code under the same version)` when only the code changed, and each setting the newer package changes.
4. `rkmon apply` makes the change: the doctor; a drain, so that the work in flight finishes; the newer code
   installed as a release over the image's own; the settings written, the earlier ones kept; the container made
   again; the monitor's load of the settings; and the doctor again. A load that fails, or a check that fails after it
   and passed before, puts the earlier code and settings back: `ROLLED BACK`, with what failed and its fix.

A deployment of the package rocke_pr_monitor, whose image is rocke-pr-monitor, needs no update in place to this
package: it moves by a new deployment beside it, as
[QUICK_DEPLOY_GUIDE.md, "Beside the old monitor, on this host or another machine"](QUICK_DEPLOY_GUIDE.md#beside-the-old-monitor-on-this-host-or-another-machine)
says.

`rkmon rollback` puts back the code and the settings from before the last apply, and
`rkmon package update --from <folder>.prev` the package folder from before the update. Each command's options:
[The container's wrapper](#the-containers-wrapper-dockerrkmon).

**The code in the container.** The image keeps its own code in `/opt/rocke_monitor`, and the code volume, `/code`,
the releases `rkmon start` and `rkmon apply` install over it ([Where things live](#where-things-live)). `rk` and the
entry point run `current`'s release while it was installed over this very image and is whole, else the image's own
code; `rk --which-code` names the folder, and `rkmon --image-code rk ...` runs the image's own. The image recipe is
what the image keeps its own copies of (`IMAGE_BOUND` in `rkmon_lib/code_volume.py`): the files it is built from, and
the few it runs from its own folder. Code that changes them runs only on a new image. A damaged release (a file `rk`
needs, missing or empty: `rk`, `rkmon.py`, `rkmon_lib/store.py`) never runs: `rk` runs the image's code and says so on
stderr, naming `rkmon rollback`. Root alone changes the code volume, at the host's request: in the container,
`rk code install` and `rk code back` only refuse.

## Platforms

Linux, natively, or a WSL2 distribution on Windows, for the container's host and a host install alike. Running
natively on Windows is out of scope.

The package's scripts are Python: Python 3.10 or newer, except `tools/pytest_vs_base.py`, which also runs on the
remote GPU nodes, and `docker/rkmon` and `tools/make_package.py`, with each file of the package they load, which run on
the older Python that [QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing) names for the
host and the build machine (`tests/test_python_floor.py` checks each file against its Python). Nine files stay
shell, and `tests/test_shell_scripts.py` fails on any other shell file:

| File | Why it is shell |
|---|---|
| `rk` | The monitor's command: it picks the code a command runs (`rkmon_lib/code_volume.py`) before Python starts |
| `docker/entrypoint.sh` | The container's entry point (tini runs it), inside the Linux image only |
| `docker/install_rocm.sh`, `docker/install_atlassian.sh`, `docker/install_sdk.sh` | Build steps of the Linux image (`docker/Dockerfile` runs them), inside the image only |
| `docker/pkgs.sh`, `docker/param.sh`, `docker/fetch.sh`, `docker/masked.sh` | The Linux image's package-manager helper, a build step's download location (its build secret over its argument), the download of the build steps, and a build step's tool that shows each download location by its build secret's name: `docker/Dockerfile` runs them before any Python exists |

A job runs each worker tool from the working folder's `tools/`, where each start installs the package's `worker/`
([What runs in the container](#what-runs-in-the-container)), or from the package's own `worker/` while that copy
lacks the file. `rk doctor`'s `worker tools` row names each one missing, or not the running code's, with its fix,
`rk worker-tools --install`.

## The GitHub CLI

Every GitHub call goes through `gh`, and `rkmon_lib/gh.py` is the one place that starts it. The image carries it; if
it is missing or older than 2.40, build the image again (QUICK_BUILD_GUIDE.md) and roll it out by its new tag
(QUICK_DEPLOY_GUIDE.md).

gh runs under one of two logins, each a token with the permissions [Logins and secrets](#logins-and-secrets) gives,
kept where [Where things live](#where-things-live) says:

- **The read login**: a fine-grained token that can only read. Every read uses it: the ticks, the reviews' tools
  (`tools/ghr.py` is the shell tools' gh), the Claude session and its agents (`GH_CONFIG_DIR` in their environment), and
  every job. `rkmon login github_read` asks for the token, typed or from stdin, and logs it in once the probe, in a
  folder of its own, finds that the token cannot write. It refuses a token that can write, and one that cannot see the
  repository (GitHub's 404, also when the issue the probe comments on is gone): the fix of each is another token, then
  the command again, and a refusal changes nothing
  ([QUICK_DEPLOY_GUIDE.md, "Step 5: login"](QUICK_DEPLOY_GUIDE.md#step-5-login)). gh's login in the read login's
  folder is the token's one copy; on a host, the gh split below also copies it into gh's own folder.
  A token with no organization access cannot read organization data. The tick then reads each pull request's requested
  reviewers over REST, kept per pull request in the working folder's `cache/review_requests.json` until the pull request
  changes. The code-owner teams' members keep the last list read, tried again a day later; while none has been read yet,
  each tick stops at that read, with `!! CRITICAL TOOL failure:` and GitHub's refusal, until the token can read the
  organization's members (Members: read).
- **The write login**: the login the monitor posts under. Only rk's guarded GitHub write commands use it (`rk post`,
  `rk notice`, `rk arm`, `rk issue-file`, ...: the writes in `rkmon_lib/post.py`), after their guards; none of them
  approves, merges at once or dismisses a review. Log it in with `rkmon login github_write` ([QUICK_DEPLOY_GUIDE.md,
  "Step 5: login"](QUICK_DEPLOY_GUIDE.md#step-5-login)).

A job never gets a login folder other than the read one, a token variable or the atlassian CLI's settings; a job that
builds or runs a PR's code (the checks, the gates, the develop sanity runs, the bisects, the drift checks, the probes)
gets no model gateway key, no Claude session variable and no variable named like a token, a key or a secret either.

**In the container PR code runs as a PR runner, in a slot of its own.** Every run of PR code, the jobs' and your
own, runs as one of the PR runners: a user that reads neither secrets volume and writes only its slot's folders, made
fresh for each job. How, and what a runner can and cannot reach:
[docs/review_pipeline.md, "7. Guardrails"](docs/review_pipeline.md#7-guardrails), section 7.2. `rk pr-run -- CMD`
runs a command of yours that way (the re-run commands the checks print start with it), and `rk pr-slot status` lists
the slots. On a host every job runs as the session user, PR code with the same list of the environment.

**In the container the split is enforced.** The write login is in its own volume, `/secrets-write` (mode 700), owned
by a second user, the writer, that shares the monitor's group; the session, its agents and the jobs run as the
monitor, which cannot read it. The writer keeps the atlassian login, the team chat webhook and the web host's ssh key
there too (`rkmon_lib/secrets_registry.py`: each secret's keeper). Each of them, and each of its logins, is set from
the host only, as the writer itself (`rkmon secret NAME`, `rkmon login NAME`): the monitor's rk and the broker refuse a
secret of the writer's from the monitor, which is the session's user too. `rk writer-secrets` says whether each is set
(never a value). rk's GitHub write commands, and every send to the team chat, the web host or JIRA that needs one of
the writer's secrets, run as the writer through the write broker: one sudo rule lets the monitor run the image's
`docker/rk_write.py`, root-owned, which runs only the writer's commands (the table in `rkmon_lib/outward.py` gives
each command, where it sends and what gates it; `rkmon_lib/writer.py` checks each one's words), with the image's own
code and gh, in a fixed environment, and logs each to `/work/logs/rk-write.log`. Everything else it refuses, and so
it does a file argument the session user could not read itself (the write login above all) and a configuration whose
folders lie in the writer's folders. A file argument is opened once, by a walk that follows no symbolic link, checked
on what was opened, and copied into a folder in the writer's home that only the writer reads; rk reads that copy, from
that folder, so a link swapped in after the check, or a file made later under a name that named nothing, is never
what it reads. A run's copies are removed when it ends; those of a run cut off before its end go with a later run,
once they are an hour old. rk sends a write command there by itself (`RKMON_WRITE_BROKER`). `rk post --premises` and
`rk spec-post` stay the session user's (their premises read git and their AFTER lines run commands): only their post
goes through the broker, as a plain `rk post`. The writer asks the features' gate itself before each send: nothing
goes out for a feature that is off, or in prototype, whoever asked for it.

**On a host the split is by convention.** Both logins are the same Linux user's: the session or a job could read the
write login's folder if it went looking. What keeps writes to rk's guarded commands is that nothing else is pointed
at it. For the enforced split, run the monitor in the container.

`rk gh-logins` probes each login with two writes GitHub refuses either way, so nothing is made: a comment with no body
on an issue of the repository, and a branch on a commit that does not exist. GitHub answers 422 when the login may make
it and 403 when it may not. The read login, and gh's default login for the session when it is another folder, must make
neither. The write login must be able to comment while a feature that is on writes to GitHub, and to push while
auto-merge is on or the merge queue is live (the auto-merge guard then turns auto-merge off), since GitHub turns
auto-merge on or off only for a login that may push. One that may comment but not push still posts reviews, but cannot
turn auto-merge on or off. In the container the write login is probed through the broker. Every tick runs the check at
most once an hour, or at once when a login file changes, and prints `GH_LOGINS`; a broken split is a standing
`!! GH_LOGIN` line. `rk doctor` runs the same probes. While the read login is missing, or can write, the deployment is
in state 7, Logins missing ([The deployment's states](#the-deployments-states)), whose next line is `rkmon login`.

On a host, `tools/gh_split_apply.py` gives gh's own folder (`~/.config/gh`, or `$XDG_CONFIG_HOME/gh`) a copy of the
read login, so a gh that names no folder (a Claude session started by hand, its agents, a shell) only reads. It moves
the write login from gh's own folder to the write login's folder, where rk's guarded writes look, and leaves the read
login in its own folder (`rk gh-logins` shows both folders). It refuses while there is no read login
(`rk read-login` sets it) or while an rk command runs (`--force` skips that check), and probes each login before and
after: a check that fails before the move changes nothing, and one that fails after it puts the old layout back.
`tools/gh_split_rollback.py` undoes it: the copy goes, and gh's own folder has the write login again. A read login
set later reaches gh's own folder by a rollback, then an apply.

In the container, each start moves a write login it finds in `/secrets/gh` to `/secrets-write/gh`, owned by the
writer, by itself: the start's root step (`docker/rk_root.py`), which the entry point runs as root before the monitor
starts. It never follows a link and never prints the login. While no read login exists, each start of the container
says the one step that adds it, `docker/rkmon login github_read`, run on the host.

## The agentic files

`agentic/` has the Claude Code side: the `/rocke-monitor` skill with its procedures, rules and lessons, the two
reviewer team members and their agent types, and the reviewer model's review book. They are templates: at every start,
`rk provision` fills them in for this deployment into the working folder's `agentic/` (the orchestrator model and every
path among them) and installs them into the container's Claude workspace, unless what is installed there is this
configuration's already; it puts back an installed file that has gone missing, and changes none in place. The
orchestrator renders and installs them again as it starts when they are older than the configuration, the package's
files or your team's skills (`agentic/README.md`).

## Command reference

Two commands run the monitor: `docker/rkmon` on the host, and `rk` in the container, which `docker/rkmon rk ARGS`
runs from the host.

### The container's wrapper, `docker/rkmon`

`docker/rkmon` runs on the host, from any folder (a link to it on your PATH works too), and runs docker compose in its
own `docker/` folder; `rkmon help` prints its commands. Every command prints each docker command it runs before it
runs it. A command for the container runs in it when it is up (`docker compose exec`), else in a one-off container
(`docker compose run --rm`). `--dry-run`, before or after the command, shows what the command would do and changes
nothing: the docker commands it would run are printed, not run (`adopt --dry-run` shows its plan). `--image-code`
before the command runs `rk` on the image's own code, not the release in `/code` (`RKMON_CODE=image`). A command in the
running container runs as the monitor; a one-off's entry point maps the users' ids and drops to the monitor by itself. Ctrl-C during a change (`start`, `apply`, `rollback`, `package update`) ends it
with `INTERRUPTED` (exit 130), where it stopped and the `next:` line, such as
`INTERRUPTED: the start stopped while setting up the work volume`; at any other time Ctrl-C ends the command (exit 130).
While this folder's container name is another deployment's on this host (`rkmon_lib/identity.py`), each command
that would reach the container refuses, with a name of its own as the fix; `status` and `next` say that what they
cannot read is unknown, and `logs`, `login` (with no NAME) and `first-tick` say so before they refuse. No command
reads the other deployment's container or names its folder, and a port counts as this deployment's only while its
own container publishes it.

`docker/rkmon` runs on the host's Python ([QUICK_DEPLOY_GUIDE.md, "Machine
sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing) says which). `CONTAINER_ENGINE=podman` runs podman in place of docker,
and `NO_COLOR` turns off the dim printing of each docker command. Your shell's `RKMON_*` and `COMPOSE_*` variables are
dropped as it starts, so that docker compose runs what your confirmed settings put in `docker/.env` and the override;
docker's connection variables (`DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG`) reach it as they are.

Each command but `help`, `template section`, `status`, `next`, `logs`, `image ls` and `package update` shares the
package folder's lock (`rkmon_lib/package_lock.py`) from its start to its end, an open `rkmon shell` or `rkmon rk`
too. `rkmon package update` takes it alone, and none waits: while an update runs, each of those commands refuses at
once with `REFUSED: a package update runs; nothing changed` (exit 4), then the same command again, and an update
refuses while another command runs.

<!-- BEGIN GENERATED: rkmon-commands
     source: rkmon_lib/commands.py, TABLE
     refresh: python3 -B tools/make_package.py --refresh-generated -->

Each command's help, with its options and examples: `rkmon help COMMAND`.

| Command | What it does |
|---|---|
| `rkmon help [COMMAND]` | the paths and the commands, one command's help, or one topic |
| `rkmon template build FILE` | write a build settings file for you to fill in |
| `rkmon template deploy FILE` | write a deploy settings file for you to fill in |
| `rkmon template section KEY` | print one section of the deploy file in full, every key with its comment |
| `rkmon check` | check this machine and your settings; change nothing but the ticket fields JIRA requires, added on your yes |
| `rkmon build --settings FILE` | build the image from your build settings |
| `rkmon image test` | test the image on this machine |
| `rkmon image save IMAGE_FILE` | save the image to a file, to carry to the deploy machine |
| `rkmon image push --settings FILE` | push the image to the registry your build settings name |
| `rkmon image load IMAGE_FILE` | load the image from a file, and check that it suits this package |
| `rkmon image pull` | pull the image from the registry your deploy settings name, and check it |
| `rkmon image check` | check that the image is on this host and suits this package |
| `rkmon confirm` | check your deploy settings, show them, and confirm them |
| `rkmon apply` | make the confirmed change on this deployment |
| `rkmon rollback` | go back to the code and settings before the last apply |
| `rkmon status` | the deployment's state, then the monitor's status |
| `rkmon logs [-f]` | the container's log |
| `rkmon pause` | start no new work; the work in flight finishes |
| `rkmon resume` | start new work again, after a pause |
| `rkmon stop [--drain]` | stop the container |
| `rkmon start` | start the container with your confirmed settings |
| `rkmon login NAME` | set each login and secret your settings need, or one login |
| `rkmon secret NAME [FILE]` | set one secret, from FILE, stdin or typed; never shown |
| `rkmon standby off` | let a monitor in standby send again |
| `rkmon shell` | a shell in the container, with the tools' settings |
| `rkmon rk ARGS` | run an rk command in the container |
| `rkmon next` | where this deployment is on its path, and the command to run next |
| `rkmon doctor` | check the running monitor and everything it reaches |
| `rkmon first-tick` | wait for the monitor's first tick |
| `rkmon package update [--from DIR\|TARBALL]` | replace a package folder with a newer package |
| `rkmon migrate --settings FILE` | rewrite a settings file in this package's layout |
| `rkmon adopt --from DIR` | move an earlier monitor's state into this deployment |
| `rkmon web-host install` | install the request service on the team dashboard's web host |
| `rkmon image ls` | this package's images on this host, whatever their names |
| `rkmon remove [--delete-data]` | remove the container; its volumes stay, unless --delete-data |

<!-- END GENERATED: rkmon-commands -->

### `rk`, in the container

`rk` does every step of the monitor's work; from the host, run it as `docker/rkmon rk ARGS`. `rk --which-code` prints
the folder of the code `rk` runs (in the container, the image's or the active release's). Its help, `rk --help`:

<!-- BEGIN GENERATED: rk-commands
     source: rkmon.py, rk --help
     refresh: python3 -B tools/make_package.py --refresh-generated -->

```text
rk: the rocKE monitor's command. It reads GitHub under the read login; the guarded write commands (post,
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
```

<!-- END GENERATED: rk-commands -->

### The commands that send

The `rk` commands that post, comment or change something on GitHub, send a team chat message, push the team
dashboard to its web host, or write to JIRA. Each one sends only while its feature is live and the monitor is not in
standby; otherwise the monitor keeps what it sends as an unsent item, and a command you type is refused
([Prototype, live and standby](#prototype-live-and-standby)). A feature that is off or not set up:
[Features](#features). The last column says whether a stop (`rkmon stop`, `rk stop`) refuses the command.

<!-- BEGIN GENERATED: outward-commands
     source: rkmon_lib/outward.py, TABLE
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Command | Sends to | Gate | Feature | A stop refuses it |
|---|---|---|---|---|
| `rk post` | GitHub | the command's own, before each send | Review posting | yes |
| `rk spec-post` | GitHub | the command's own, before each send | Review posting | yes, but not with `--dry-run`, which sends nothing |
| `rk notice` | GitHub | the command's own, before each send | by its `--kind`:<br>`incident`: Incidents<br>`queue`: Queue notices<br>`overlap`: Overlap management<br>`drift`: Drift notices<br>`gpu`: Remote GPU tests and Review posting | yes |
| `rk arm` | GitHub | the command's own, before each send | Auto-merge | yes |
| `rk disarm` | GitHub | the command's own, before each send | Auto-merge | no: it only takes auto-merge off |
| `rk queue-notice` | GitHub | the command's own, before each send | Queue notices | yes |
| `rk automerge-off` | GitHub | the command's own, before each send | Auto-merge guard | no: it only takes auto-merge off |
| `rk issue-file` | GitHub | the command's own, before each send | Incidents or Develop issues (its arguments pick one) | yes |
| `rk issue-comment` | GitHub | the command's own, before each send | Incidents or Develop issues (its arguments pick one) | yes |
| `rk issue-relabel` | GitHub | the command's own, before each send | Incidents or Develop issues (its arguments pick one) | yes |
| `rk issue-assign` | GitHub | the command's own, before each send | Incidents or Develop issues (its arguments pick one) | yes |
| `rk label-create` | GitHub | the command's own, before each send | Incidents or Develop issues (its arguments pick one) | yes |
| `rk request-changes` | GitHub | the command's own, before each send | Auto-hold | yes |
| `rk fix-link` | GitHub | the command's own, before each send | Fix links | yes |
| `rk gpu-row` | GitHub | the command's own, before each send | Remote GPU tests | yes |
| `rk teams-breakage` | the team chat | the command's own, before each send | Develop break alerts | yes |
| `rk teams-test` | the team chat | the command's own, before each send | Develop break alerts or Needs-you alerts | yes |
| `rk needs-you-teams` | the team chat | the command's own, before each send | Needs-you alerts | yes, but not with `--dry-run`, which sends nothing |
| `rk ticket-file` | JIRA | the command's own, before each send | Out-of-scope issue tickets | yes |
| `rk teams-send` | the team chat | at the send, before a team chat message goes out, whoever asked for it | by its `--type`:<br>`develop-break`: Develop break alerts<br>`needs-you`: Needs-you alerts | yes, at its gate |
| `rk web-push` | the web host | at the send, before the team dashboard's page goes out, whoever asked for it | Team dashboard | yes, at its gate |
| `rk atlassian` | JIRA | at the send, before a JIRA write goes out, whoever asked for it | Follow-up tickets | yes, at its gate |

<!-- END GENERATED: outward-commands -->

## Settings reference

There are two settings files, each YAML with a JSON Schema of its own in `schema/`: the deploy settings file (`kind:
deploy_settings`; `rkmon template deploy FILE` writes one, [QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md)) and the
build settings file (`kind: build_settings`; `rkmon template build FILE`, [QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)).
`rkmon check --settings FILE` checks either one, one plain line per problem, naming the key. A key you leave out takes
its default. The tables below are made from the schemas, as the templates' comments are, and `rkmon template section
KEY` prints any section of the deploy settings file in full, each key with its comment. Every other configuration
key, in no settings file and set by no person, is in [docs/internal_settings.md](docs/internal_settings.md), the
maintainers' list.

**In an editor.** The files name no schema, so map their names to the package's schemas in your editor's settings:
it then shows each key's description, type, allowed values and default as you type, and marks a key the schema does
not know. With the YAML language server (the YAML extension of VS Code, among others), that is its `yaml.schemas`
setting, each schema by its full path in the package folder:

```json
"yaml.schemas": {
  "<PACKAGE_DIR>/schema/deploy_settings.schema.json": "**/rkmon-deploy*.yaml",
  "<PACKAGE_DIR>/schema/build_settings.schema.json": "**/rkmon-build*.yaml"
}
```

Name your files to match, as the guides do: `~/rkmon-deploy.yaml` and `~/rkmon-build.yaml`.

### Deploy settings

Every key of the deploy settings file but the model bindings, in the file's order. A key in bold heads a section.

<!-- BEGIN GENERATED: deploy-settings
     source: schema/deploy_settings.schema.json
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Key | What it is |
|---|---|
| `kind` | The type of settings file. Don't change it. |
| `schema_version` | The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout. |
| **`container`** | Container: the Docker container this monitor runs in. |
| `container.name` | The container's name. The monitor keeps its data under this name, so choose it once: rkmon confirm refuses a new name while this container exists. To run a second monitor on this host, see QUICK_DEPLOY_GUIDE.md, "Several deployments". While features.review_automation.automated_reviewer_name is blank, every review gives this name on its first line. Lowercase letters, digits, - and \_. Default: "rocke-monitor". |
| **`image`** | Image: the container image to run, \<registry>/\<name>:\<tag>, as your build settings name it. |
| `image.registry` | The registry to pull the image from, such as "registry.example.com/my-team". Run docker login for it first. Leave blank if you load the image from a file instead (rkmon image load FILE). |
| `image.name` | The image's name, as in your build settings, without its registry, which goes in image.registry. Default: "rocke-monitor". |
| `image.tag` | The image's tag, as in your build settings. To move to a newer image, change the tag (QUICK_USER_GUIDE.md, "Change a setting"). Required. |
| **`updates`** | Updates: where rkmon package update fetches newer code for this package. Leave this section out to update only from a folder or an archive (rkmon package update --from). |
| `updates.repository` | The GitHub repository to fetch this package from, as \<org>/\<repo>. Example: "ROCm/rocm-libraries". Required. |
| `updates.branch` | The branch to fetch the newest code from. Example: "develop". Required. |
| **`repository`** | Repository: the GitHub repository the monitor watches, and which of its pull requests the monitor works on. |
| `repository.name` | The repository, as \<org>/\<repo>. Example: "my-org/my-repo". Required. |
| `repository.folder` | The folder your project lives in, from the repository's root, without a / at either end. The monitor uses it to tell your project's files from the rest of the repository. Example: "projects/my-project". Required. |
| `repository.scope_labels` | The labels that put a pull request in scope: the monitor works on every open pull request that has at least one of them. To see the repository's labels, run: gh label list -R \<org>/\<repo>. Example: \["my-project"]. Required. |
| `repository.shared_label_folders` | Labels that several teams share, each with a folder. A pull request with one of these labels is in scope only if it also changes files in that label's folder. Write each one as "label": "folder", the folder from the repository's root. Example: {"needs-ci": "projects/my-project"}: a "needs-ci" pull request is in scope only if it changes files in projects/my-project. Default: none. |
| **`host`** | Host: how much of this host the container may use, and which user it runs as. |
| **`host.resources`** | Resources: the container's share of this host. QUICK_DEPLOY_GUIDE.md, "Machine sizing", helps you choose. |
| `host.resources.cpus` | The number of CPUs the container may use. Default: 4. |
| `host.resources.memory_gib` | The container's memory limit, in GiB. At least ramdisk_gib + 8. Default: 8. |
| `host.resources.ramdisk_gib` | A RAM disk that makes reviews and builds faster, in GiB. It is part of memory_gib. Allow about 1.5 GiB for each review and 2 GiB for each build that run at the same time. 0: no RAM disk. Default: 0. |
| **`host.users`** | Users: the user and group the monitor runs as inside the container. |
| `host.users.monitor_uid` | The user id, 1000 or more, other than 65534, the system's "nobody" user, and none of the package's own users' (the GitHub writer's and the PR runners'). Default: null, which means your own user id on this host. |
| `host.users.monitor_gid` | The group id, 1 or more: any group's, a system group's included, but none of the PR runners' groups'. Default: null, which means your own group id on this host. |
| **`monitor`** | Monitor: the monitor works in ticks. Each tick, it reads GitHub and updates the reviews, the merge queue and the dashboards. |
| `monitor.tick_interval_minutes` | Minutes between ticks. Default: 10. |
| `monitor.full_tick_max_interval_minutes` | Most ticks read only what changed on GitHub. At least this often, in minutes, a tick reads everything again. 0: every tick reads everything. Default: 60. |
| `monitor.parallel_builds` | How many local builds, each with its tests, may run at the same time at first, from 1 to 8. Reviews, the merge queue and the develop guard all use these builds. The monitor then adjusts the number, within 1 to 8, to how busy this host is. Default: 1. |
| **`features`** | Features: what the monitor does. Each feature that posts or sends runs in prototype mode or live mode:<br>  "prototype": it does its work and shows you the result on the admin dashboard, but posts and sends nothing.<br>  "live": it posts and sends. |
| **`features.review_automation`** | Review automation: Claude reviews every new or updated pull request in scope. |
| `features.review_automation.enabled` | true: review pull requests. false: no reviews and no Claude; decisions that need judgement wait for you on the admin dashboard. Required. |
| `features.review_automation.mode` | "prototype": reviews are written and shown on the admin dashboard, but not posted. "live": reviews are posted on GitHub as integrations.github_cli.username, and so are notices about pull requests that change the same code. Required when review automation is on. |
| `features.review_automation.automated_reviewer_name` | The name at the top of every review this monitor posts: "Automated review by \<automated_reviewer_name>". Letters, digits, spaces, - and \_. Example: "Platform team". Leave blank to use the container's name. |
| `features.review_automation.skills_folder` | A folder with your team's own review skills, as Markdown (.md) files only. Its layout must match the package's agentic/ folder, and it may contain only team_members/, agents/, review_book.md and skills/rocke-monitor/review_procedure.md. A file with the same name as a package file replaces it; a file with a new name is added. Leave blank to use the package's skills only. |
| **`features.review_automation.orchestrator`** | Orchestrator: the main Claude session. It runs the reviews, checks and posts them, and makes the merge queue's judgement calls. |
| `features.review_automation.orchestrator.model` | Its model: "opus", "sonnet", "haiku", or the name of a Claude model in model_bindings. Default: "opus". |
| `features.review_automation.orchestrator.effort` | How hard it thinks: "low", "medium", "high", "xhigh" or "max". Higher costs more and takes longer. Default: "high". |
| **`features.review_automation.schedule`** | Schedule: when reviews start. |
| `features.review_automation.schedule.push_cooldown_minutes` | How long to wait after a pull request's latest push before reviewing it, in minutes, so that a burst of pushes gets one review. A burst waits 45 minutes after its last push, and your priorities and the pull requests that fix develop wait at most 10 minutes after any push, a burst's too. Set above 45, it applies to bursts too. 0: a single push doesn't wait. Default: 30. |
| `features.review_automation.schedule.follow_up_cooldown_minutes` | The shortest time between two follow-up reviews of the same pull request, in minutes. Your priorities, pull requests that fix develop, and pushes that only merge develop in don't wait. 0: don't wait. Default: 60. |
| `features.review_automation.parallel_reviews` | How many reviews may run at the same time at first, from 1 to 5. The monitor then adjusts the number, within 1 to 5, to how busy this host is. Default: 3. |
| **`features.review_automation.large_prs`** | Large pull requests: how very large pull requests are reviewed. |
| `features.review_automation.large_prs.in_depth_review_over_lines` | A pull request with more changed lines than this gets one in-depth review, split into parts, and after that only light follow-ups. The in-depth review starts only once you approve it: the admin dashboard's Needs you card gives its size and cost, with Approve and Decline, or run: rkmon rk review-approve N. Default: 30000. |
| `features.review_automation.large_prs.split_part_lines` | The most changed lines in each part of an in-depth review, from 6000 to 10000. Default: 8000. |
| **`features.review_automation.passes`** | Passes: the steps of the review work, each with its model and effort.<br>  Models: "opus", "sonnet", "haiku", or the name of an entry in model_bindings.<br>  Efforts: "low", "medium", "high", "xhigh" or "max". Ollama and OpenAI-compatible models take only "low", "medium" or "high".<br>  The verification and code quality passes always run, each on a Claude model. The four optional passes after them let another model do part of the work first; "default" means no other model, so the pass in charge does that work itself. |
| **`features.review_automation.passes.verification_pass`** | Verification pass: reviews each pull request on its own, checks every claim of the first review pass (if there is one) against the code, drops what it can't confirm, and writes the review that gets posted. |
| `features.review_automation.passes.verification_pass.model` | Its model, which must be a Claude model. Default: "opus". |
| `features.review_automation.passes.verification_pass.effort` | Its effort. Default: "high". |
| **`features.review_automation.passes.code_quality_pass`** | Code quality pass: reviews each pull request's design (single responsibility, no duplication, layering, names, tests) and gives its findings to the verification pass. |
| `features.review_automation.passes.code_quality_pass.model` | Its model, which must be a Claude model. Default: "opus". |
| `features.review_automation.passes.code_quality_pass.effort` | Its effort. Default: "medium". |
| **`features.review_automation.passes.first_review_pass`** | First review pass (optional): another model writes a first draft of each review, which the verification pass then checks claim by claim. |
| `features.review_automation.passes.first_review_pass.model` | The model that writes the draft. Default: "default", which means no draft: the verification pass reviews alone. |
| `features.review_automation.passes.first_review_pass.effort` | Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't Claude. |
| **`features.review_automation.passes.ci_triage_pass`** | CI triage pass (optional): another model reads the pull request's failing CI jobs and tells the verification pass what caused each failure. |
| `features.review_automation.passes.ci_triage_pass.model` | The model. Default: "default", which means none: the verification pass reads the failing CI jobs itself. |
| `features.review_automation.passes.ci_triage_pass.effort` | Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't Claude. |
| **`features.review_automation.passes.first_code_quality_pass`** | First code quality pass (optional): another model checks the pull request's changes against the code quality rules first, and gives what it finds to the code quality pass to confirm. |
| `features.review_automation.passes.first_code_quality_pass.model` | The model. Default: "default", which means none: the code quality pass does all of the work. |
| `features.review_automation.passes.first_code_quality_pass.effort` | Its effort. Default: "default", which means the code quality pass's effort, or none for a model that isn't Claude. |
| **`features.review_automation.passes.notice_draft_pass`** | Notice draft pass (optional): another model writes the first draft of the notices the monitor posts on pull requests: when two pull requests change the same code, and when a change on develop breaks a pull request's tests. The orchestrator checks and posts them. |
| `features.review_automation.passes.notice_draft_pass.model` | The model. Default: "default", which means none: the orchestrator writes the notices itself. |
| `features.review_automation.passes.notice_draft_pass.effort` | Its effort. Default: "default", which means the orchestrator's effort, or none for a model that isn't Claude. |
| **`features.review_automation.remote_gpu_tests`** | Remote GPU tests: some tests need a GPU, and on a host without one they are skipped. With this on, the monitor runs them on a Slurm cluster instead: it logs in to the cluster's login node over SSH and submits each test run as a Slurm job, which runs on a GPU node. A review doesn't wait for the results: its GPU tests row says "pending" and is filled in when the job finishes. |
| `features.review_automation.remote_gpu_tests.enabled` | true: run GPU tests on the cluster. false: skip them, as on any host without a GPU. Required when review automation is on. |
| **`features.review_automation.remote_gpu_tests.cluster`** | Cluster: how the monitor reaches the cluster, and what each job asks for. Your cluster's documentation or its admins have these values. |
| `features.review_automation.remote_gpu_tests.cluster.login_host` | The host name of the cluster's login node, where the monitor submits jobs: letters, digits, '.', '-' and '\_', starting with a letter or a digit, with no user@ and no port. Example: "login.cluster.example.com". Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.cluster.ssh_port` | The login node's SSH port. Default: 22. |
| `features.review_automation.remote_gpu_tests.cluster.ssh_user` | The user name the monitor logs in to the login node with: letters, digits, '.', '-' and '\_', starting with a letter or '\_'. Every pull request's tests run as this user on the GPU nodes, so use an account made for the monitor, not your own. Install this user's SSH key with: rkmon secret gpu_ssh_key FILE. Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.cluster.partition` | The Slurm partition to submit jobs to (sbatch --partition): letters, digits, '.', ',', '&', '\|', '!', '-' and '\_'. Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.cluster.accounts` | The Slurm accounts to charge jobs to (sbatch --account), in order: if the cluster refuses one, the monitor tries the next. Each is letters, digits, '.', '-' and '\_'. Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.cluster.qos` | The Slurm quality of service for the jobs (sbatch --qos): letters, digits, '.', ',', '&', '\|', '!', '-' and '\_'. Leave blank for the account's default. |
| `features.review_automation.remote_gpu_tests.cluster.constraint` | A Slurm node constraint (sbatch --constraint), for example to use only nodes that can read your home folder: letters, digits, '.', ',', '&', '\|', '!', '-' and '\_'. Leave blank for none. |
| `features.review_automation.remote_gpu_tests.cluster.gpu_types` | For each GPU target the tests use ("gfx", then lowercase letters and digits), the cluster's name for that type of GPU, as in sbatch --gres=gpu:\<type>:\<count>: letters, digits, '.', ':', '-' and '\_'. Example: {"gfx942": "mi300x"}. Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.cluster.default_target` | The GPU target for tests that don't say which one they need: one of gpu_types' targets. Leave blank to use the first entry in gpu_types. |
| `features.review_automation.remote_gpu_tests.cluster.gpus_per_job` | GPUs per job. The tests use one. Default: 1. |
| `features.review_automation.remote_gpu_tests.cluster.time_limit` | Each job's time limit, as Slurm writes it ("HH:MM:SS", "D-HH:MM:SS" or minutes). Allow about 1.5 times a normal test run. Default: "01:00:00". |
| `features.review_automation.remote_gpu_tests.cluster.excluded_nodes` | GPU nodes to avoid, such as broken ones, each named as Slurm names nodes: letters, digits, '.', ',', '\[', ']', '-' and '\_', such as "node\[01-04]". Each job first checks that its node's GPU works; when a node fails that check, the monitor tells you the node's name, so you can add it here. Default: none. |
| **`features.review_automation.remote_gpu_tests.node`** | Node: where jobs keep their files on the cluster, and how each GPU node is prepared. |
| `features.review_automation.remote_gpu_tests.node.work_folder` | The folder on the cluster for the monitor's copy of the repository and each job's files, on one line: relative to ssh_user's home folder, or absolute. The GPU nodes must be able to read it, for example on a shared home folder. Default: "rkmon-gpu". |
| `features.review_automation.remote_gpu_tests.node.python` | The Python on the GPU nodes that has the tests' packages, pytest included, on one line; usually a virtual environment's bin/python. Example: "\~/venvs/tests/bin/python". Required when remote GPU tests are on. |
| `features.review_automation.remote_gpu_tests.node.setup_command` | A shell command each job runs first, on one line, such as one that loads the cluster's modules. Example: "module load rocm". Leave blank for none. |
| `features.review_automation.remote_gpu_tests.node.environment` | Environment variables each job sets, as NAME: "value": each NAME letters, digits and '\_', starting with a letter or '\_', and each value on one line. Example: {"HIP_VISIBLE_DEVICES": "0"}. Default: none. |
| `features.review_automation.remote_gpu_tests.node.clone_url` | Where the cluster clones the repository from, if not GitHub (for example, a mirror inside your network): an address with no spaces. Leave blank to clone from GitHub. |
| `features.review_automation.remote_gpu_tests.jobs_per_review` | The most GPU targets one review runs tests on, one job each. The cluster's own limits decide how many jobs run at the same time. Default: 2. |
| `features.review_automation.remote_gpu_tests.wait_report_after_minutes` | If a job is still waiting for a GPU node after this many minutes, the monitor tells you, once. The job keeps waiting. 0: never tell you. Default: 240. |
| `features.review_automation.remote_gpu_tests.bad_node_resubmits` | How many times to resubmit a job on another node when its node fails the GPU check. Default: 1. |
| **`features.review_automation.spend`** | Spend: what the reviews cost in US dollars, shown on the dashboards and in the weekly report. |
| `features.review_automation.spend.enabled` | true: work out the cost from the prices in usd_per_million_tokens. false: don't show costs. Required when review automation is on. |
| `features.review_automation.spend.usd_per_million_tokens` | The price in US dollars per million tokens of each model the monitor uses, from Anthropic's price list or your gateway's. Name each model by its id: claude-opus-5-5 for "opus", claude-sonnet-5-5 for "sonnet", claude-haiku-5-5 for "haiku", and the model_id of a model_bindings entry. Add one entry per model, as shown below; a model with no entry has its cost shown as unknown. Default: none. |
| `features.review_automation.spend.usd_per_million_tokens.claude-opus-5-5.input` | Input tokens. Required. |
| `features.review_automation.spend.usd_per_million_tokens.claude-opus-5-5.output` | Output tokens. Required. |
| `features.review_automation.spend.usd_per_million_tokens.claude-opus-5-5.cache_read` | Tokens read from the prompt cache. Required. |
| `features.review_automation.spend.usd_per_million_tokens.claude-opus-5-5.cache_write_5_minutes` | Tokens written to the prompt cache for 5 minutes. Required. |
| `features.review_automation.spend.usd_per_million_tokens.claude-opus-5-5.cache_write_1_hour` | Tokens written to the prompt cache for 1 hour. Required. |
| **`features.merge_queue`** | Merge queue: puts the pull requests in scope in merge order, runs its gates (the checks a pull request must pass before it merges), and makes a merge plan (which pull requests can merge together without conflicts). Your priorities go first (rkmon rk prioritize N, or a priority request you approve on the admin dashboard). |
| `features.merge_queue.enabled` | true: run the merge queue (only one monitor per repository may). false: no merge queue; your priorities still decide which pull requests are reviewed first. Required. |
| `features.merge_queue.mode` | "prototype": the queue, its gates and the merge plan are worked out and shown to you, but nothing happens on GitHub: no auto-merge is turned on, and nothing is held, commented on or filed. "live": the queue does these things on GitHub, as the settings below allow. Required when the merge queue is on. |
| **`features.merge_queue.auto_merge`** | Auto-merge: when a pull request you cleared for merging (rkmon rk clear N) passes the gates, the monitor turns on GitHub's auto-merge for it. GitHub's branch protection still has the final say. |
| `features.merge_queue.auto_merge.enabled` | true: turn on auto-merge (in prototype, the dashboard only shows where it would). false: only propose the merge order to you. Required when the merge queue is on. |
| **`features.merge_queue.auto_merge_guard`** | Auto-merge guard: when someone turns on auto-merge for a pull request that fails a gate, the monitor comments on the pull request and turns its auto-merge off (in prototype, it only tells you). It never turns off an auto-merge you turned on yourself. It runs whenever the merge queue runs. |
| `features.merge_queue.auto_merge_guard.on_your_pull_requests` | What to do when someone else turns on auto-merge for a pull request you wrote: "alert_only" (only tell you) or "same_as_others" (comment and turn it off, as for anyone else's pull request). Default: "alert_only". |
| **`features.merge_queue.auto_hold`** | Auto-hold: when a pull request fails a gate because of its own changes, the monitor posts a "Request changes" review on it from your account. That blocks the merge until the author fixes it and you approve it again. |
| `features.merge_queue.auto_hold.enabled` | true: hold such pull requests (in prototype, only show them). false: only report the failure to you. Required when the merge queue is on. |
| **`features.merge_queue.queue_notices`** | Queue notices: each pull request near the front of the queue gets one comment about its place in the queue, updated as things change. |
| `features.merge_queue.queue_notices.enabled` | true: post them (in live). false: none. Required when the merge queue is on. |
| **`features.merge_queue.drift_notices`** | Drift notices: when a change on develop breaks a pull request's own tests, the monitor tells its author which develop commit changed the behaviour. Claude writes these, so they need review automation. |
| `features.merge_queue.drift_notices.enabled` | true: post them (in live). false: none. Required when the merge queue is on. |
| **`features.merge_queue.develop_guard`** | Develop guard: after every merge, the monitor checks develop and traces any break to the pull request that caused it. In live, each break becomes a GitHub issue, and an open pull request that fixes it is linked to the issue. Without review automation, breaks wait for you on the admin dashboard instead. |
| `features.merge_queue.develop_guard.enabled` | true: check develop. false: don't. Required when the merge queue is on. |
| `features.merge_queue.develop_guard.issue_repository` | The repository to file develop issues in, as \<org>/\<repo>. Leave blank to use repository.name. |
| `features.merge_queue.develop_guard.ci_red_label` | The label for an issue about a break that turns develop's CI red. The label must already exist in issue_repository (rkmon rk label-create NAME --description TEXT --color HEX creates one). Example: "ci-red". Required when the develop guard is on and the merge queue is live. |
| `features.merge_queue.develop_guard.ci_red_extra_labels` | More labels for those issues. Each must already exist. Default: none. |
| `features.merge_queue.develop_guard.stability_label` | The label for an issue about a break that CI doesn't catch; the issue is assigned to the author of the pull request that caused it. The label must already exist in issue_repository. Example: "stability". Required when the develop guard is on and the merge queue is live. |
| **`features.dashboards`** | Dashboards: the admin dashboard (yours) and the team dashboard (for other teams). |
| **`features.dashboards.admin`** | Admin dashboard: everything the monitor shows, with Approve and Decline buttons for priority requests, proposed tickets and the in-depth reviews of large pull requests. Only this host can open it, at http://127.0.0.1:\<port>/. It asks for a user name and a password: type any user name, and the access token as the password (rkmon secret admin_access_token --generate makes one). To open it from another machine, use an SSH tunnel (README.md, "The admin dashboard"). |
| `features.dashboards.admin.enabled` | true: serve the admin dashboard. false: none (the team dashboard and found-issue tickets need it). Required. |
| `features.dashboards.admin.port` | The port to serve it on. Default: 18614. |
| **`features.dashboards.team`** | Team dashboard: a copy of the dashboard for other teams to read, without the admin buttons. |
| `features.dashboards.team.enabled` | true: deliver the team dashboard (it needs the admin dashboard). false: none. Required. |
| `features.dashboards.team.mode` | "prototype": the page is made every tick and shown on the admin dashboard, but not published. "live": the page is delivered as delivery says. Required when the team dashboard is on. |
| `features.dashboards.team.public_url` | The web address other teams open the page at; the Needs you alerts (below) link to it. Example: "https://dashboards.example.com/my-team/". With priority requests on and delivery "publish", the monitor's request service serves the page over plain http, at the root of its own port: give http, a port of 1024 or more and no path, with a host that is an IP address or a name of letters, digits, "." and "-". Example: "http://dashboards.example.com:8080/". Required when delivery is "publish", or when serve's address is "0.0.0.0" or "::". Otherwise, leave blank to use serve's address and port. |
| **`features.dashboards.team.priority_requests`** | Priority requests: a box on the page where other teams can ask for a pull request to be prioritized. Each request waits for your Approve or Decline on the admin dashboard. With delivery "publish", the web host needs the monitor's request service (README.md, "The team dashboard", says how to install it). |
| `features.dashboards.team.priority_requests.enabled` | true: show the box. false: no box. Required when the team dashboard is on. |
| `features.dashboards.team.delivery` | How the page reaches other teams: "publish" (copied to a web host over SSH every tick) or "serve" (served by this monitor from this host: anyone who can reach that address and port can open it). Fill in the matching block below and delete the other. Required when the team dashboard is on. |
| **`features.dashboards.team.publish`** | Publish: the web host to copy the page to. Install the SSH key for it with: rkmon secret web_host_ssh_key FILE |
| `features.dashboards.team.publish.web_host` | The web host's name: letters, digits, '.', '-' and '\_', starting with a letter, a digit or '\_'. Example: "web.example.com". Required when delivery is "publish". |
| `features.dashboards.team.publish.username` | The user name to log in to the web host with: letters, digits, '.', '-' and '\_', starting with a letter, a digit or '\_'. Required when delivery is "publish". |
| `features.dashboards.team.publish.site_folder` | The folder the web host serves the page from, relative to that user's home folder: folder names of letters, digits, '.', '-' and '\_', each starting with a letter, a digit or '\_', joined by '/'. Example: "public_html/my-team". Required when delivery is "publish". |
| **`features.dashboards.team.serve`** | Serve: use this block instead of publish when delivery is "serve". |
| `features.dashboards.team.serve.address` | The address of this host to serve the page on: "127.0.0.1" (this host only), one of this host's addresses, or "0.0.0.0" or "::" (all of them; then set public_url). In prototype the page is served on "127.0.0.1" only, whatever this says. Default: "127.0.0.1". |
| `features.dashboards.team.serve.port` | The port to serve it on. Default: 18615. |
| **`features.alerts`** | Alerts: short messages to a Microsoft Teams channel, through a Teams Workflows webhook. Install the webhook's address with: rkmon secret teams_webhook FILE. Each alert has its own on/off switch. |
| `features.alerts.mode` | "prototype": messages are written and listed on the admin dashboard, but not sent. "live": messages are sent. Required when an alert is on. |
| `features.alerts.message_label` | Text to start every message with, such as your team's name. Leave blank for none. |
| `features.alerts.message_cooldown_minutes` | The shortest time between two messages of the same alert, in minutes; anything new in between goes into the next one. 0: don't wait. Default: 30. |
| **`features.alerts.develop_breaks`** | Develop breaks: a message when develop's CI is red and no fix is planned. It needs the develop guard. |
| `features.alerts.develop_breaks.enabled` | true: send this alert. false: breaks only show on the dashboards. Required. |
| **`features.alerts.needs_you`** | Needs you: a message for each decision that waits for you on the admin dashboard (a merge to approve, a hold that has waited too long, a priority request, a pull request ready for your approval, an in-depth review to approve, reviews that have waited on verification for hours). Each is sent once, addressed to your GitHub username, with a link to the team dashboard when its public_url is set. |
| `features.alerts.needs_you.enabled` | true: send this alert. false: decisions only show on the dashboards. Required. |
| **`features.ticket_creation`** | Ticket creation: JIRA tickets, filed with the atlassian tool on the site in integrations.atlassian_cli.jira_url. Install its login with: rkmon login atlassian (the login needs permission to create tickets). Each type of ticket has its own on/off switch. |
| `features.ticket_creation.mode` | "prototype": tickets are written and listed on the admin dashboard, but not filed. "live": tickets are filed. Required when a type of ticket is on. |
| `features.ticket_creation.project_key` | The key of the JIRA project to file tickets in. Example: "PROJ". Required when a type of ticket is on. |
| `features.ticket_creation.issue_type` | The issue type of new tickets, as your JIRA project writes it, case and all, and never a sub-task type: a sub-task needs a parent, which the monitor never gives. Example: "Story". Leave blank for "Task". |
| `features.ticket_creation.epic_key` | The epic to link new tickets to, by its key. Example: "\<PROJECT>-123". With an epic, a new ticket gets its description only after it is linked, so a project that requires a description when a ticket is created refuses the ticket. Leave blank for no epic. |
| `features.ticket_creation.components` | Components to set on every new ticket, each one your JIRA project allows. A project that requires components on a new ticket needs at least one here. Default: none. |
| `features.ticket_creation.labels` | Labels to set on every new ticket. A project that requires labels on a new ticket needs at least one here. Default: none. |
| `features.ticket_creation.required_fields` | Other fields your JIRA project requires on a new ticket: a line for each, its field id and its value, such as customfield_10010: "High". Set only the fields JIRA requires and gives no value of its own, never one the monitor or JIRA fills: the monitor fills the project, the summary, the issue type, the priority, the labels and components their keys list, and the description while epic_key is blank; JIRA fills the reporter. rkmon check asks JIRA which fields these are. On a terminal it then asks you for each one's value, from those JIRA allows, and on your yes adds their lines here, keeping the file as it was as FILE.bak: in place of {}, after the last entry, or with required_fields itself at the end of features.ticket_creation. It adds none to a required_fields written as {...} with entries: add those lines by hand. Default: none. |
| **`features.ticket_creation.follow_up_tickets`** | Follow-up tickets: when a pull request merges with review items still open, each one becomes a ticket for the pull request's author. Open items are the monitor's own blocking and should-fix findings, review comments answered with a promise to fix them later ("next PR", "TODO"), and reviewers' requests nobody answered. Never nits. Tickets are filed without asking you, so agree on this with your team first. To assign a ticket, the monitor needs the author's JIRA account (rkmon rk jira-account add LOGIN ACCOUNT); without it, the ticket is filed unassigned. |
| `features.ticket_creation.follow_up_tickets.enabled` | true: file them (in live). It needs review automation. false: only list the items for you. Required. |
| **`features.ticket_creation.found_issue_tickets`** | Found-issue tickets: when the monitor finds a problem that belongs to no pull request and isn't a develop break, such as a flaky test, it proposes a ticket on the admin dashboard. Approve files it; Decline drops it. |
| `features.ticket_creation.found_issue_tickets.enabled` | true: propose tickets. It needs the admin dashboard. false: only report such problems to you. Required. |
| **`features.weekly_report`** | Weekly report: every week, a statistics report and notes for a lessons-learned review, saved in the monitor's rocke_weekly/ folder (README.md, "Where things live"). |
| `features.weekly_report.enabled` | true: write them every week. false: don't; rkmon rk stats and rkmon rk lessons write them when you ask. Required. |
| **`integrations`** | Integrations: the tools the monitor uses. rkmon login lists the logins they need and installs each one. |
| **`integrations.github_cli`** | GitHub CLI: how the monitor reads and writes GitHub. It needs two logins: rkmon login github_read for reading, and rkmon login github_write for posting. README.md, "Logins and secrets", lists the permissions each one needs. |
| `integrations.github_cli.username` | Your GitHub username. The monitor posts as this user, and treats this user's actions as yours: your comments, your approvals (an approval lifts a hold), and auto-merges you turned on (which the guard leaves alone). Needs you alerts are addressed to this user. Required. |
| `integrations.github_cli.max_requests_per_hour` | The most GitHub API requests this monitor may make in an hour. GitHub's hourly limit is per account: if several monitors share one account, give each a cap so that together they stay under the limit. A gh command that reads several pages of results counts once toward this cap, though GitHub counts each page: leave headroom. 0: no cap. Default: 0. |
| **`integrations.atlassian_cli`** | Atlassian CLI: how the monitor files JIRA tickets. Install its login with: rkmon login atlassian |
| `integrations.atlassian_cli.jira_url` | Your JIRA site's address. Example: "https://\<site>.atlassian.net". Required when a type of ticket is on. |
| **`integrations.claude_code`** | Claude Code: how the monitor reaches Claude. Install the key with: rkmon login claude (an Anthropic API key, or your LLM gateway's token). |
| `integrations.claude_code.base_url` | The address to reach Claude at, if you use an LLM gateway (Claude Code's ANTHROPIC_BASE_URL). Example: "https://llm-gateway.example.com". Leave blank to use Anthropic's own API. |
| `integrations.claude_code.api_key_sent_as` | How the key is sent: "header" (in the header that api_key_header names), "bearer" (as "Authorization: Bearer \<key>") or "header_and_bearer" (both). Your gateway's documentation says which. Default: "header". |
| `integrations.claude_code.api_key_header` | The header the key goes in, for "header" and "header_and_bearer". Leave blank for Anthropic's own header, "x-api-key". |
| `integrations.claude_code.api_key_scheme` | Text to put before the key in that header, such as "Bearer". Leave blank to send the key alone. |

<!-- END GENERATED: deploy-settings -->

### Model bindings

The `model_bindings` section of the deploy settings file, by provider.

<!-- BEGIN GENERATED: model-bindings
     source: schema/deploy_settings.schema.json, model_bindings
     refresh: python3 -B tools/make_package.py --refresh-generated -->

Model bindings: other models the passes can use, each under a name you choose. "opus", "sonnet" and "haiku" are built in, reached through integrations.claude_code, and need no entry. To add a model, replace {} with entries in one of the three forms that rkmon template section model_bindings prints. Default: none.

A Claude model, reached through integrations.claude_code:

| Key | What it is |
|---|---|
| `model_bindings.<name>.provider` | "claude". Required. |
| `model_bindings.<name>.model_id` | The model's id. Example: "claude-sonnet-5-5". Required. |

A model behind an OpenAI-compatible API (an LLM gateway, vLLM, LiteLLM):

| Key | What it is |
|---|---|
| `model_bindings.<name>.provider` | "openai_compatible". Required. |
| `model_bindings.<name>.model_id` | The model's id, as the API lists it. Required. |
| `model_bindings.<name>.base_url` | The API's base address. Example: "https://llm-gateway.example.com/v1". Required. |
| `model_bindings.<name>.api_key_secret_name` | The name of the secret that contains the API key (not the key itself): lowercase letters, digits and \_, ending in \_api_key. Install the key with: rkmon secret \<name>\_api_key FILE. Leave blank if the API needs no key. |
| `model_bindings.<name>.api_key_sent_as` | How the key is sent: "bearer" (as "Authorization: Bearer \<key>", with api_key_header and api_key_scheme not used), "header" (in the header that api_key_header names, after api_key_scheme) or "header_and_bearer" (both). Default: "bearer". |
| `model_bindings.<name>.api_key_header` | The header the key goes in; with api_key_sent_as "bearer" it is not used. Required when api_key_sent_as is "header" or "header_and_bearer". |
| `model_bindings.<name>.api_key_scheme` | Text to put before the key in api_key_header's header, such as "Bearer"; with api_key_sent_as "bearer" it is not used. Leave blank to send the key alone. |
| **`model_bindings.<name>.limits`** | Limits: what the API allows. |
| `model_bindings.<name>.limits.requests_per_minute` | Requests per minute. 0: unknown (the monitor still slows down when the API says it is busy). Default: 0. |
| `model_bindings.<name>.limits.tokens_per_minute` | Tokens per minute. 0: unknown. Default: 0. |
| `model_bindings.<name>.limits.max_parallel_requests` | Requests at the same time, across all passes. 0: no limit. Default: 0. |
| `model_bindings.<name>.stall_timeout_seconds` | If a request gets no new output for this many seconds, it is stopped and sent again. Default: 300. |

A model on an Ollama server:

| Key | What it is |
|---|---|
| `model_bindings.<name>.provider` | "ollama". Required. |
| `model_bindings.<name>.model_id` | The model, as ollama list shows it. Example: "llama3.1:8b". Required. |
| `model_bindings.<name>.server_url` | The Ollama server's address. Inside the container, localhost is the container itself, so use the host's address. Example: "http://\<host address>:11434". Required. |
| **`model_bindings.<name>.limits`** | Limits: what the server allows. |
| `model_bindings.<name>.limits.max_parallel_requests` | Requests at the same time. One model in memory serves one request well. 0: no limit. Default: 1. |
| `model_bindings.<name>.stall_timeout_seconds` | If a request gets no new output for this many seconds, it is stopped and sent again. Default: 300. |

<!-- END GENERATED: model-bindings -->

### Build settings

Every key of the build settings file, in the file's order, with what `rkmon build` gives the build from it ([The
image](#the-image)). A key in bold heads a section.

<!-- BEGIN GENERATED: build-settings
     source: schema/build_settings.schema.json, and rkmon_lib/build_plan.py, ARGS, DOWNLOADS, CA_CONTEXT
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Key | Given to the build as | What it is |
|---|---|---|
| `kind` | — | The type of settings file. Don't change it. |
| `schema_version` | — | The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout. |
| **`image`** |  | Image: the name the built image gets, \<registry>/\<name>:\<tag>. Your deploy settings name the same image. |
| `image.registry` | — | The registry to push the image to, such as "registry.example.com/my-team". Run docker login for it first. Leave blank to move the image as a file instead (rkmon image save here, rkmon image load on the deploy host). |
| `image.name` | — | The image's name, without its registry, which goes in image.registry. Default: "rocke-monitor". |
| `image.tag` | — | The image's tag. Give every build a new tag, such as "2026.11.15-2", so a deployment can move to it and back. Leave blank to use this package's version, such as "2026.11.15". |
| **`operating_system`** |  | Operating system: the Linux image the build starts from. |
| `operating_system.base_image` | build argument `BASE_IMAGE` | The base image: "ubuntu:26.04", "ubuntu:24.04", "rockylinux:9" or "almalinux:9". Default: "ubuntu:26.04". |
| `operating_system.package_mirror` | build secret `package_mirror` | A mirror of Ubuntu's package archive, if your network can't reach Ubuntu's own. Ubuntu base images only. Example: "http://mirror.example.com/ubuntu". Leave blank to use Ubuntu's own archive. |
| `operating_system.ca_certificates` | build context `ca-certificates` | Extra root certificates (PEM files on this machine), for a network that inspects TLS traffic. If downloads fail with certificate errors, ask your IT team for these. Example: \["\~/certs/my-company-root.pem"]. Default: none. |
| **`rocm`** |  | ROCm: the ROCm release installed in the image. |
| `rocm.version` | build argument `ROCM_VERSION` | The ROCm version. Keep "10.1.0" unless your team needs another: the review checks are set up for its compiler. Required. |
| `rocm.package_url` | build secret `rocm_package_url` | Where to download the ROCm installer (the .run file) from, if not AMD's public site. Leave blank to use AMD's public installer for this version, which is checked only when you set package_sha256. |
| `rocm.package_sha256` | build argument `ROCM_PACKAGE_SHA256` | The installer's SHA-256 checksum: 64 hex digits, in quotes (run: sha256sum FILE). Required when you set package_url; set it anyway to check AMD's public installer. |
| **`github_cli`** |  | GitHub CLI: the gh tool, which the monitor uses to read and write GitHub. |
| `github_cli.version` | build argument `GITHUB_CLI_VERSION` | The gh version to install, such as "2.80.0". Leave blank for the newest release (rkmon check warns that two builds may then differ). |
| `github_cli.package_url` | build secret `github_cli_package_url` | Where to download the gh release archive (gh\_\<version>\_linux\_\<arch>.tar.gz) from, if not GitHub. Leave blank to download it from GitHub, checked against GitHub's published checksums. |
| `github_cli.package_sha256` | build argument `GITHUB_CLI_PACKAGE_SHA256` | The archive's SHA-256 checksum. Required when you set package_url. |
| **`atlassian_cli`** |  | Atlassian CLI: the atlassian tool, whose commands are auth, confluence, jira, rest, site, update and version. The monitor uses it to file JIRA tickets. Every image includes it, so that one image works for every deployment. |
| `atlassian_cli.version` | build argument `ATLASSIAN_CLI_VERSION` | The version to require, such as "1.2.3" (atlassian version prints it). Leave blank to accept any version. |
| `atlassian_cli.package_url` | build secret `atlassian_cli_package_url` | Where to download the atlassian tool's Linux binary from (the program file itself, not an archive). Ask your team where your organization keeps it. Required. |
| `atlassian_cli.package_sha256` | build argument `ATLASSIAN_CLI_PACKAGE_SHA256` | The binary's SHA-256 checksum. Download the file once and run: sha256sum FILE. Required. |
| **`claude_agent_sdk`** |  | Claude Agent SDK: the Python package that includes Claude Code, which runs the reviews. |
| `claude_agent_sdk.version` | build argument `CLAUDE_AGENT_SDK_VERSION` | The SDK version, such as "1.2.3". Leave blank for the newest release (rkmon check warns that two builds may then differ). |
| `claude_agent_sdk.package_url` | build secret `claude_agent_sdk_package_url` | Where to download the SDK's wheel (the .whl file) from, if not the package index below. Leave blank to install it from the package index, which is not checked against a checksum. |
| `claude_agent_sdk.package_sha256` | build argument `CLAUDE_AGENT_SDK_PACKAGE_SHA256` | The wheel's SHA-256 checksum. Required when you set package_url. |
| **`python_packages`** |  | Python packages: where the build downloads Python packages from. |
| `python_packages.package_index` | build secret `python_package_index` | A Python package index to use instead of pypi.org, such as "https://pypi.example.com/simple/". Leave blank to use pypi.org. |

<!-- END GENERATED: build-settings -->

## Logins and secrets

Each login and secret a deployment may need: whom it is kept for, the command that installs it and what needs it;
then the GitHub logins, each with the permissions its token needs on the repository and on the organization. None is
ever in a settings file, and none is shown once it is in. [QUICK_DEPLOY_GUIDE.md, "Make the GitHub
tokens"](QUICK_DEPLOY_GUIDE.md#make-the-github-tokens) says how to make the two GitHub tokens, and
[QUICK_DEPLOY_GUIDE.md, "Step 5: login"](QUICK_DEPLOY_GUIDE.md#step-5-login) installs each login and secret.

<!-- BEGIN GENERATED: logins
     source: rkmon_lib/secrets_registry.py
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Type | Name | Kept for | Installed by | Needed by |
|---|---|---|---|---|
| login | `github_read` | the monitor | `rkmon login github_read` | the monitor, always |
| login | `github_write` | the writer | `rkmon login github_write` | review posting; auto-merge; auto-merge guard (turning auto-merge off); auto-hold; develop issues; incidents; queue notices; drift notices; fix links |
| login | `atlassian` | the writer | `rkmon login atlassian` | follow-up tickets; out-of-scope issue tickets |
| login | `claude` | the monitor | `rkmon login claude` | Claude session |
| secret | `admin_access_token` | the monitor | `rkmon secret admin_access_token --generate` | admin dashboard |
| secret | `teams_webhook` | the writer | `rkmon secret teams_webhook` | develop break alerts; needs-you alerts |
| secret | `web_host_ssh_key` | the writer | `rkmon secret web_host_ssh_key` | team dashboard (publish) |
| secret | `gpu_ssh_key` | the monitor | `rkmon secret gpu_ssh_key` | remote GPU tests |

| Login | What to make | Repository permissions | Organization permissions |
|---|---|---|---|
| `github_read` | you, on GitHub: a fine-grained token with \<org> as its resource owner that only reads \<org>/\<repo> and \<org>'s team members | Actions, Commit statuses, Contents, Issues, Metadata, Pull requests: read | Members: read |
| `github_write` | you: gh's own sign-in, or a fine-grained token you make on GitHub with \<org> as its resource owner, for \<org>/\<repo> and the develop guard's issue repository | Contents, Issues, Pull requests: write; Metadata: read | none |

The monitor watches public repositories: no fine-grained token can read a private repository's check runs.

<!-- END GENERATED: logins -->

### The hosts' own keys

The monitor reaches two hosts over ssh: the team dashboard's web host, with `web_host_ssh_key`, and the GPU cluster's
login host, with `gpu_ssh_key`. ssh trusts each host by its own keys as the monitor's internal state records them, and
by no other: nothing is pushed to the web host, and no GPU job is sent, until they are recorded. They are in no
settings file, and you never write them by hand: [QUICK_DEPLOY_GUIDE.md, "Step 5:
login"](QUICK_DEPLOY_GUIDE.md#step-5-login) says how they are recorded.

When a host's own keys change, ssh does not trust it: the team dashboard's push, or the remote GPU tests, fail
host key verification until the new ones are recorded. `rkmon secret NAME FILE`, NAME the ssh key for that host
(`web_host_ssh_key` or `gpu_ssh_key`) and FILE that key, is the one way to record them: once the key is set, it
shows the keys the host has now and how many recorded ones they replace, and records them once you accept them.
Unless the host was set up anew, ask its admin why its keys changed first. `rkmon login` records a host's keys only
while none is recorded, so it never offers to replace them.

### The one-time move's copies

A deployment that moves in place keeps its volumes, and in them the secrets an older package kept under other names,
or in the monitor's volume. At each start of the container, its root step copies each one to where this package keeps
it, while that new copy is missing:

| Older copy | New copy | What is copied |
|---|---|---|
| the team chat's webhook, `/secrets/teams_webhook` | `/secrets-write/teams_webhook` | the file |
| the web host's SSH key, under its older name in `/secrets` | `/secrets-write/web_host_ssh_key` | the file |
| an older package's Claude settings, in `/secrets` | `/secrets/claude_key`, the `claude` login | the key alone, when your deploy settings can carry the rest of those settings; else nothing, and the line says what they cannot carry, with the fix `rkmon login claude` |
| a model binding's key under an older name in `/secrets`: `NAME_key` or `NAME`, for a key named `NAME_api_key` | the name the binding's `api_key_secret_name` gives, such as `/secrets/NAME_api_key` | the file, while review automation is on and one of its passes runs on that binding; else nothing, and the line says that no feature uses it, at every start until the move ends |

Each new copy is mode 600, and its volume's user owns it: the write user in `/secrets-write`, the monitor's user in
`/secrets`. Each copy gets a line that names both files, never a value. The older copy stays until the move ends,
when you end standby ([Prototype, live and standby](#prototype-live-and-standby)). So does the older package's code in
the work volume, `/work/code`, which the older package runs if you go back to it: `rkmon standby off` removes it with
the older copies, never before, and its report says so in a line of its own.

Each copy is made once. Once a start has dealt with one (it made the copy, or found the new copy there, no older copy,
an empty one, or one it refused), a mark in the code volume, which only root writes, keeps every later start from
making it: a file written later at an older name is never copied. While the new copy is missing, the start's line
names the older copy and the fix, the command that sets the secret.

The move refuses an older copy that is a link, a socket or another special file, or a file over 64 KiB, and two older
copies of one key that differ; the line gives the fix. A copy that cannot be read or written is an `ERROR`, with the
same fix: the root step ends with exit 1, the monitor does not start, and the container waits; `rkmon logs` shows the
lines. Set the secret, then `rkmon stop` and `rkmon start`: a later start deals with that copy again, as it does with
one that no feature uses.

`gpu_ssh_key` keeps its name and its volume. The atlassian login is never copied, since Atlassian rotates its token
and two copies log each other out: `rkmon login atlassian` sets it. GitHub's write login moves on its own
([The GitHub CLI](#the-github-cli)).

## Ports, volumes and docker/.env

What the container publishes on the host, and the volumes it mounts (what is in each volume:
[In the container](#in-the-container)):

| What | On the host | In the container |
|---|---|---|
| The admin dashboard | `127.0.0.1`, port `features.dashboards.admin.port` (18614 unless you changed it) | port 18613 |
| The team dashboard, while this host serves it | `features.dashboards.team.serve.address`, port `features.dashboards.team.serve.port` (18615 unless you changed it) | port 18615 |
| The work volume | `<container.name>_rkmon-work` | `/work` |
| The secrets volume | `<container.name>_rkmon-secrets` | `/secrets` |
| The writer's secrets volume | `<container.name>_rkmon-secrets-write` | `/secrets-write` |
| The code volume | `<container.name>_rkmon-code` | `/code` |
| The settings volume | `<container.name>_rkmon-settings` | `/etc/rkmon/settings`, read-only |

Docker names each volume after the compose project, `container.name`. The volumes stay when the container goes, unless
`rkmon remove --delete-data` removes them.

Never edit `docker/.env` or `docker/compose.override.yaml`: the first `rkmon start`, and after it `rkmon apply` and
`rkmon rollback`, write both from the deploy settings you confirmed, and `rkmon apply` puts back a hand edit.

<!-- BEGIN GENERATED: env-and-ports
     source: rkmon_lib/deploy_plan.py, ENV_KEYS, OVERRIDE_KEYS
     refresh: python3 -B tools/make_package.py --refresh-generated -->

Each variable of docker/.env, and the deploy settings it is made from:

| Variable | Made from |
|---|---|
| `COMPOSE_PROJECT_NAME` | `container.name` |
| `RKMON_IMAGE` | `image.name` |
| `RKMON_TAG` | `image.tag` |
| `RKMON_ADMIN_HOST` | no setting: always `127.0.0.1` |
| `RKMON_PORT` | `features.dashboards.admin.port` |
| `RKMON_TEAM_HOST` | `features.dashboards.team.serve.address`; only while the container serves the team dashboard (in `features.dashboards.team`: `enabled`, `mode`, `delivery`) |
| `RKMON_TEAM_PORT` | `features.dashboards.team.serve.port`; only while the container serves the team dashboard (in `features.dashboards.team`: `enabled`, `mode`, `delivery`) |
| `RKMON_CPUS` | `host.resources.cpus` |
| `RKMON_MEMORY` | `host.resources.memory_gib` |
| `RKMON_UID` | `host.users.monitor_uid` (at its default: your own user id on this host) |
| `RKMON_GID` | `host.users.monitor_gid` (at its default: your own group id on this host) |

docker/compose.override.yaml is made from `host.resources.ramdisk_gib`, `features.dashboards.team.enabled`, `features.dashboards.team.mode` and `features.dashboards.team.delivery`. While the container serves the team dashboard, with `host.resources.ramdisk_gib` at 2.5, it adds this to docker/compose.yaml:

```yaml
services:
  rkmon:
    ports:
      - target: 18615
        published: "${RKMON_TEAM_PORT}"
        host_ip: "${RKMON_TEAM_HOST}"
    tmpfs:
      - /ramdisk:size=3g,mode=1777,exec
```

<!-- END GENERATED: env-and-ports -->

## Exit codes

What each exit code of `rkmon` means (`rkmon help exit-codes` prints them too). An `rk` command's own exit codes are in
its line of `rk --help` ([`rk`, in the container](#rk-in-the-container)).

<!-- BEGIN GENERATED: exit-codes
     source: rkmon_lib/outcome.py, EXIT_CODES
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Exit code | What it means |
|---|---|
| 0 | done, or nothing to do |
| 1 | an error part-way, a roll back, or you said no |
| 2 | a usage error, including a question with no terminal and no --yes |
| 3 | an earlier step is not done (docker, buildx, a settings file, a confirm, an image, a login, a first start) |
| 4 | refused, or a check failed, and nothing changed |
| 130 | interrupted |

<!-- END GENERATED: exit-codes -->

## Glossary and placeholders

The terms the docs, the settings files and `rkmon` use, then the placeholders they write.

| Term | What it means |
|---|---|
| Admin dashboard | Your web page on the deploy host: everything the monitor shows, with Approve and Decline for priority requests, proposed tickets and the in-depth reviews of large pull requests. It listens only on the deploy host, at `features.dashboards.admin.port`, and asks for your access token ([The admin dashboard](#the-admin-dashboard)) |
| Apply | `rkmon apply`: the change you confirmed, made on the deployment. It puts this package's code and your confirmed settings in place, keeps the ones before them, and makes the container again when a change needs it; a failure puts the kept ones back, and so does `rkmon rollback` |
| Auto-hold; hold | A Request changes review the monitor posts from your account on a pull request that fails a gate because of its own changes. The hold blocks the merge until the author fixes it and you approve it again (`features.merge_queue.auto_hold`) |
| Auto-merge; arm; clear | GitHub's own auto-merge. To clear a pull request is to give your OK for it to merge (`rkmon rk clear N`); once it passes the gates, the monitor arms it: it turns its auto-merge on (`features.merge_queue.auto_merge`) |
| Auto-merge guard | When someone else turns on auto-merge for a pull request that fails a gate, the monitor comments on it and turns its auto-merge off; in prototype mode it only tells you. It never turns off an auto-merge you turned on (`features.merge_queue.auto_merge_guard`) |
| Automated reviewer name | The name on the first line of every review this monitor posts (`features.review_automation.automated_reviewer_name`) |
| Bisect | The search, among the merges since develop was last green, for the first one that broke it: the culprit |
| Blind review | The verification pass's own review of a pull request, made before it reads any first draft |
| Build settings file; deploy settings file | The two YAML files you write: one for an image, one for a deployment. Neither ever has a secret in it |
| CI red; stability issue | The two types of develop break: CI red, when develop's CI fails; a stability issue, when CI does not catch the break. Each has its GitHub issue, labelled `features.merge_queue.develop_guard.ci_red_label` or `features.merge_queue.develop_guard.stability_label`; a stability issue is assigned to the culprit's author |
| Confirm | Deploy step 3: `rkmon confirm --settings FILE` checks your deploy settings, shows them and records your OK. `rkmon start` starts the container only with the settings you confirmed |
| Culprit | The merge that broke develop: the only candidate, or the first failing one a bisect finds |
| Delivery | How the team dashboard reaches other teams (`features.dashboards.team.delivery`): publish, copied to a web host; or serve, served by the monitor from the deploy host |
| Deployment | One monitor: its container, set up from one package folder with one deploy settings file |
| Develop | The repository's branch that pull requests merge into |
| Develop break | A merge after which develop fails |
| Develop guard | After every merge, the monitor checks develop and traces a break to the pull request that caused it (`features.merge_queue.develop_guard`) |
| Do-not-merge label | A label that keeps a pull request out of the merge queue while it is on: one with do not merge, don't merge, DNM, on hold, WIP or not ready in its name, in capitals or not, with spaces, - or _ between the words |
| Drain | Nothing new starts while the work in flight finishes, for at most a given number of minutes (`rkmon rk drain`) |
| Drift notice | A comment telling a pull request's author that a change on develop broke the pull request's own tests, and which develop commit changed the behaviour (`features.merge_queue.drift_notices`) |
| Effort | How hard a model thinks: low, medium, high, xhigh or max. Higher costs more and takes longer |
| Fingerprint | A SHA-256, written `sha256:…`, of your deploy settings: each key's value (its default where you leave it blank), except the `updates` section, which only the host reads, with the deploy schema's version, the package's version and the SHA-256 of the package's code (each file its `.dockerignore` lets into an image's build), and while `features.review_automation.skills_folder` is set, each file of the team's skills folder. So a new package changes it, other code under the same version too, and so does a changed, added or removed skill. `rkmon confirm` records the fingerprint of what you confirm, and the monitor the fingerprint of the settings it runs (`rkmon rk applied`) |
| Fix link | The link `rkmon rk fix-link` makes, once a fix is verified, from a develop break's issue to the open pull request that fixes it; that pull request then goes first in the merge queue |
| Follow-up review | A later round of a pull request's review, after new pushes |
| Follow-up ticket | A JIRA ticket for a review item still open when its pull request merged, for the pull request's author (`features.ticket_creation.follow_up_tickets`) |
| Found-issue ticket; proposed ticket | A ticket for a problem the monitor found that belongs to no pull request and is no develop break, such as a flaky test (`features.ticket_creation.found_issue_tickets`). Until you decide, it is a proposed ticket on the admin dashboard: Approve files it, Decline drops it |
| Gate | A check a pull request must pass before it merges |
| Image; tag | What the container runs, `<name>:<tag>` (`image.name`, `image.tag`), built from a build settings file |
| In-depth review | The review of a pull request with more changed lines than `features.review_automation.large_prs.in_depth_review_over_lines`: one review in parts, then only light follow-ups |
| Incident | A develop break while it lasts: the merge queue is frozen until a green check of develop closes it |
| Issue type; required fields | The JIRA issue type of the monitor's tickets (`features.ticket_creation.issue_type`), and the fields your JIRA project requires on a new ticket (`features.ticket_creation.required_fields`) |
| Layer | One of the sources of the running configuration, each value from one: deploy (your confirmed deploy settings as the first `rkmon start`, `rkmon apply` and `rkmon rollback` put them in the settings volume's `deploy.json`: the delivered layer), repository, package, state, derived, or `config.json`. `rkmon rk config show --source` names each value's layer |
| Leads | What an optional pass hands its owner, the pass it works for: points to check, each quoting the lines that show it. The owner checks each lead before it uses one |
| Login; secret | What the monitor signs in with (GitHub's read login and write login, the atlassian CLI's login, Claude's credential), and the keys and tokens it uses. `rkmon login` and `rkmon secret NAME` install them; a settings file never has one |
| Masked name | What the image's build shows in a line of apt, pip or the ROCm installer in place of a host, which may be an internal one: a download location of your build settings file as its build secret's name in angle brackets, such as `<package_mirror>`; any other host as `<host>`, and an IP address as `<address>`, as in `[IP: <address>]` and `(<address>)`. The public hosts the tools read with no build secret, PyPI's and Ubuntu's, show as they are |
| Merge plan | Which pull requests can merge together without conflicts, and in what order |
| Merge queue | The pull requests in scope, in the order they should merge, with their gates and the merge plan (`features.merge_queue`) |
| Merge-only push | A push that only merges develop into a pull request, or only rebases it: the review and the approval given before the push still count |
| Model binding | A model a pass can use besides opus, sonnet and haiku, under a name you choose (`model_bindings`) |
| Monitor; writer; PR runners | The container's users. The monitor runs the Claude session, its agents and the jobs; the writer makes the outward writes, and is the only one that reads the writer's secrets; the PR runners run pull requests' code, and read no secret |
| Needs-you list | The decisions only you can make, under Needs you at the top of the admin dashboard, and in `rkmon rk status` |
| Not sent; unsent item | What a feature in prototype, or the monitor in standby, kept instead of posting, sending, pushing or filing it: the admin dashboard's "Not sent" card lists each one, and none is ever sent later ([Prototype, live and standby](#prototype-live-and-standby)) |
| Operator | You: the person who deploys and runs the monitor and makes the decisions it leaves to you. The monitor posts as your GitHub username (`integrations.github_cli.username`) |
| Orchestrator | The main Claude session (`features.review_automation.orchestrator`): it runs the reviews, checks and posts them, and makes the merge queue's judgement calls |
| Outputs folder | The folder of the files the monitor writes for you: `/work` in the container ([The outputs folder](#the-outputs-folder)) |
| Overlap notice | A comment telling a pull request's author that another open pull request changes the same code, and which of the two keeps it |
| Package; package folder | The monitor's code, docs and tests, as a tarball or a folder; the package folder is where you unpacked it, and `rkmon` runs from it |
| Parallel builds | How many local builds, each with its tests, run at the same time; the reviews, the merge queue and the develop guard share them. The monitor raises or lowers the number, within its range, as the host's load allows (`monitor.parallel_builds`) |
| Parallel reviews | How many reviews run at the same time. The monitor raises or lowers the number, within its range, as the host's load allows (`features.review_automation.parallel_reviews`) |
| Pass | One step of the review work, with its model and effort (`features.review_automation.passes`). The verification pass and the code quality pass always run, each on a Claude model; the four optional passes (the first review pass, the CI triage pass, the first code quality pass and the notice draft pass) let another model do part of the work first |
| Priority; priority request | A pull request you put first in the merge queue (`rkmon rk prioritize N`); and another team's request, from the team dashboard, for a pull request to be prioritized, which waits for your Approve or Decline on the admin dashboard |
| Profile | A starting template for one type of deployment, which `rkmon template deploy` writes: full, review-only or merge-queue. It is never a setting |
| Prototype mode; live mode | Each feature that posts or sends runs in one or the other: in prototype mode it does its work and shows you the result on the admin dashboard, and posts and sends nothing; in live mode it posts and sends, as its parts' own switches allow. Their keys are `features.review_automation.mode`, `features.merge_queue.mode`, `features.alerts.mode`, `features.dashboards.team.mode` and `features.ticket_creation.mode` |
| Queue notice | One comment on a pull request near the front of the merge queue about its place, kept up to date (`features.merge_queue.queue_notices`) |
| RAM disk | Memory the container uses as a fast disk for its jobs' builds and temporary files (`host.resources.ramdisk_gib`); it is part of the container's memory |
| Repository defaults | What belongs to one repository rather than to the monitor, such as its CI job names, its test tiers, its hot spots and its tests' wording: the package's `repository/<org>__<repo>/defaults.json`, for the folders of that repository it was written for. The package ships its home repository's alone; any other repository or folder takes the package's generic values, and `rk doctor` and `rkmon check` warn, naming them |
| Request service | The monitor's server on the web host: with delivery publish, it serves the team dashboard and takes the priority requests made on it |
| Review automation | Claude reviews every new or updated pull request in scope (`features.review_automation`) |
| Revert offer | Once an incident has lasted two hours by default, the monitor makes a revert of the culprit on this host and tests it; with your OK, it posts the result on the develop break's issue as evidence. It never pushes a revert: that is the develop-green team's call |
| rocKE | AMD's kernel DSL (domain-specific language) and library, whose pull requests the monitor works on |
| rocKE monitor | This package: it reviews rocKE's pull requests, keeps their merge queue and checks develop after every merge |
| Skills folder | A folder of your team's own review skills (`features.review_automation.skills_folder`) |
| Standby | The whole monitor does its work and sends nothing, as a feature in prototype does, until `rkmon standby off` ends it. The one-time move from an older install and `rkmon adopt` start a deployment in standby ([Prototype, live and standby](#prototype-live-and-standby)) |
| Team dashboard | A copy of the dashboard for other teams to read, without the admin dashboard's buttons (`features.dashboards.team`) |
| Team deployment | A deployment that reviews one team's pull requests: those that carry its scope labels (`repository.scope_labels`). How several deployments share a repository: [QUICK_DEPLOY_GUIDE.md, "Several deployments"](QUICK_DEPLOY_GUIDE.md#several-deployments) |
| Tick; full tick; idle tick | One round of the monitor's work, every `monitor.tick_interval_minutes` minutes. A tick is idle when nothing it reads has changed since the last full tick: it only refreshes the dashboards' "checked" time and prints one line. Every other tick is a full tick, and so is one at least every `monitor.full_tick_max_interval_minutes` minutes |
| Unposted review | A review the monitor verified and kept, posting nothing, while review automation is in prototype, the monitor is in standby, or its posting is not set up: `PR_UNPOSTED_<N>.md` in the outputs folder. It counts as the monitor's last review of its pull request, which its author never saw ([Prototype, live and standby](#prototype-live-and-standby)) |
| Working folder | The folder the monitor works in: `/work` in the container ([In the container](#in-the-container)) |
| Worktree | A checkout of one commit of the monitor's clone, in the working folder, where a review, a gate or a check of develop builds and runs the code |

| Placeholder | What it stands for |
|---|---|
| `<arch>` | The processor architecture in the name of gh's release archive (`amd64`, `arm64`) |
| `<automated_reviewer_name>` | The automated reviewer name (`features.review_automation.automated_reviewer_name`) |
| `<branch>` | A git branch's name |
| `<cache>` | The monitor's cache folder, `cache/` in the working folder (`rk paths` names it) |
| `<command>` | An `rk` command, with its arguments |
| `<count>` | How many GPUs a job asks for, as in `gpu:<type>:<count>` |
| `<day>` | A UTC day, as YYYY-MM-DD |
| `<deploy-host>` | The machine the container runs on |
| `<detail>` | The rest of a check's last line, after `pass`, `fail` or `skip`: what it found, or why it did not run |
| `<dir>` | A review directory: the folder of one round of a pull request's review, which `rk review-prep` fills in |
| `<effort>` | An effort level: `low`, `medium`, `high`, `xhigh` or `max` |
| `<entry>` | A CTest entry: the name by which CI runs a test |
| `<file>` | A file's path in the repository, from its root |
| `<folder>` | A folder: the one the text around it names, such as a package folder (`<folder>.prev`) |
| `<gateway-host>` | The host name of your organization's Claude gateway |
| `<head>` | The hash of a pull request's head commit: the one the text around it names |
| `<hex>` | Hexadecimal digits that make a file's name its own |
| `<how>` | How a fix was checked, in a few words, as the verifier's notes give it |
| `<install>` | The scratch folder the checks' native build is installed into (`cmake --install`'s prefix), shown in its place in the installed tests' lines |
| `<jira-site>` | The host name of your JIRA site |
| `<k>` | A number in a sequence: a re-review's or a follow-up's (`fr<k>`, `fu<k>`), a part of a split review (`part<k>.md`), or an item of the verifier's notes (`N<k>`) |
| `<key>` | An API key |
| `<kind>` | One value of a fixed list, by the line or the command that shows it: a notice's kind (`rk notice --kind`: `incident`, `queue`, `overlap`, `drift` or `gpu`), a review's round in `REVIEW_READY` (`first-round` or `follow-up`), or a develop issue's label in `FIX_CANDIDATE` |
| `<label>` | A label of the repository |
| `<lessons>` | The lessons folder, `lessons/` in the working folder, where review runs are kept for the lessons-learned review (`rk paths` names it) |
| `<model>` | A model, by the name its provider gives it |
| `<module>`, `<package>` | A Python module's or package's name, dotted as an import names it |
| `<N>` | A pull request's number |
| `<name>` | An image's name (`image.name`), a model binding's name (`model_bindings`), or a name the text around it gives |
| `<org>/<repo>` | A GitHub repository, as the part of its address after github.com/ (`my-org/my-repo`); where no other repository is named, the one the monitor watches (`repository.name`) |
| `<outputs>` | The outputs folder ([The outputs folder](#the-outputs-folder)) |
| `<port>` | A port number |
| `<price>` | A price in US dollars per million tokens |
| `<probe>` | A saved probe's name |
| `<PROJECT>` | A JIRA project's key |
| `<reason>`, `<why>` | A reason, in words |
| `<registry>` | A container registry's address (`image.registry`) |
| `<registry-host>` | The host name of the container registry the image is pushed to or pulled from, the first part of `image.registry`, as `docker login` takes it |
| `<round>`, `<R>` | A review's round: `fr`, the first review; `fr<k>`, a re-review; `fu<k>`, the k-th follow-up |
| `<sha>`, `<sha10>` | A commit's hash; `<sha10>`, its first 10 digits |
| `<sha256>` | A file's SHA-256, its 64 hexadecimal digits |
| `<site>` | Your Atlassian site's name, the first part of its address |
| `<tag>` | An image's tag (`image.tag`), or an Ollama model's tag |
| `<target>` | What `rk lane-of` is asked about: a test file, a test's node id or a CTest entry |
| `<tier>` | A CI test tier, as the repository's CI configuration names it |
| `<tmp>` | The monitor's folder for temporary files: `tmp/` in the working folder, or on the RAM disk while it is on (`rk paths` names it) |
| `<type>` | A GPU type, as the GPU cluster names it, as in `gpu:<type>:<count>` |
| `<url>`, `<URL>` | A web address: the one the text around it names |
| `<user>` | A machine's user name, in the path of a home folder that a line shows masked (`/home/<user>`) |
| `<user_login>` | Your GitHub username (`integrations.github_cli.username`) |
| `<value>` | A value: the one the text around it names, such as one of your own for a JIRA field that takes any (`features.ticket_creation.required_fields`) |
| `<version>` | A version number: the one of the tool or the package the text names |
| `<web-host>` | The team's web host, which the team dashboard is copied to with delivery publish |
| `<web-user>` | The user the team dashboard is copied to the web host as (`features.dashboards.team.publish.username`) |
| `<week>` | An ISO week, as YYYY-Www |
| `<word>` | A word of your own that names a piece of work, as in `pr<N>-<word>`: lower-case letters, digits, `.`, `_` and `-` |
| `<workers>` | The folder of the reviews' drafts and checks, `workers/` in the working folder (`rk paths` names it) |
