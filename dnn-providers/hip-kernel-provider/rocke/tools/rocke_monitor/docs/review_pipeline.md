<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# The review pipeline: design, passes and lessons

How the rocKE monitor reviews a pull request, for maintainers. Scripted checks build and test the pull request
merged with today's develop, and two Claude agents review it before anything is posted: the verification pass, and the
code quality pass beside it. A first review pass by another model is optional: when one runs, the verification pass
reviews blind first, then verifies or deletes every claim of its draft. This page explains why the pipeline is built
this way, how its parts fit together, what the local review loop does, the guardrails, the rocKE CI knowledge the
verification needs, the known issues, and an honest assessment of a first review pass.

Building and deploying the monitor: [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md) and
[QUICK_DEPLOY_GUIDE.md](../QUICK_DEPLOY_GUIDE.md). The day-to-day procedures are the skill's files (section 6);
stopping and resuming is [restart.md](restart.md). The terms the docs share are in the README's
[Glossary and placeholders](../README.md#glossary-and-placeholders); the terms only this page uses are in its
glossary (section 11). When this page and the skill's files or the code disagree, the files and the code win.

| § | Section | § | Section |
|---|---|---|---|
| 1 | What the pipeline does | 7 | Guardrails |
| 2 | Why it is built this way | 8 | rocKE CI knowledge for verification |
| 3 | Architecture and roles | 9 | Known issues and their fixes |
| 4 | Setup and tools | 10 | An honest assessment of a first review pass |
| 5 | The local review loop in detail | 11 | Glossary |
| 6 | Procedures | | |

Up to two models do the work. The **orchestrator** is the Claude Code session that runs the loop, decides and posts,
on `features.review_automation.orchestrator.model`; its two review passes, the verification pass and the code quality
pass, are Claude agents (section 3.2). A **first review pass** (`features.review_automation.passes.first_review_pass`)
is optional, and off by default: another model writes first drafts, through the local review loop (the model binding
named `worker`: `worker/review_agent.py`, which drives the model with a fixed menu of tools, section 5) or through a
model binding's tool loop (`worker/drafter.py`, section 3.9).

## 1. What the pipeline does

- **Checks, then Claude's review.** Each new or changed pull request is staged, and a checks job builds and tests it
  merged with today's develop: builds, tests, goldens, the scripted checks (section 3.11), CI logs and the other open
  pull requests. The verification pass then reviews it blind, reproduces each claim with a run or a code read, and
  writes one review per pull request head; the code quality pass judges its design beside it. With a first review
  pass, its draft is one more input: the verification pass trims its false alarms and adds what it missed.
- **Scope.** Open pull requests into develop that are not drafts and carry a scope label
  (`repository.scope_labels`), or a shared label (`repository.shared_label_folders`) together with a change in that
  label's folder.
- **Rounds.** A first round for a pull request we have not reviewed yet; a follow-up after the author pushed
  substantive changes, at most one per `features.review_automation.schedule.follow_up_cooldown_minutes` after our last
  review on the pull request, posted or kept. A rebase-only push keeps the verdict, and nothing is posted. A merge-only
  push (develop merged in) whose checks are clean is carried with nothing posted; one whose only new items are threads,
  a title problem, or others' new reviews or comments gets a micro round (section 3.11).
- **Output.** With review automation live (`features.review_automation.mode`), the verified review is posted on the
  pull request as one comment, under the write login (`integrations.github_cli.username`), and a copy goes to the
  outputs folder (`<OUTPUTS_DIR>`) as `PR_FIRST_ROUND_<N>.md` or `PR_FOLLOWUP_<N>.md`. In prototype, in standby, or
  while posting is not set up, it is kept instead, unposted, as `PR_UNPOSTED_<N>.md` in the outputs folder, and the
  line `UNPOSTED: …` says why. A kept review is never posted later. Around the reviews, the monitor can post overlap
  notices, file follow-up tickets and more, each while its feature allows it (the README's
  [Features](../README.md#features)).
- **The post's shape.** Line 1 names the deployment's automated reviewer and the model that drafted the review:
  `Automated review by <automated_reviewer_name>: Model - <model>, effort: <effort>`. The name is
  `features.review_automation.automated_reviewer_name`, else the container's name. The model is the first review
  pass's, at its effort (`review_first_line`), or, with no first review pass, the verification pass's
  (`review_first_line_no_draft`: `verifier_name`, at `features.review_automation.passes.verification_pass.effort`).
  Then come a header with a traffic light; a table of every prior item with its verified status, ending with the
  Bots / CI, Our checks and Code quality rows (and a Follow-ups row once we have listed follow-ups); the post's two
  parts, **Needed to merge** and **Highly recommended follow-up (not blocking)** (section 6.3); a utility line, a
  legend, and a footer that names the verification and says the post is not an approval or a rejection.
- **Its own reviews.** By line 1 the monitor knows its own reviews among its account's: a review is its own when line
  1 gives one of its names (today's, or one it has posted under: `reviews.automated_reviewer_names_used`), or gives
  none and was posted before the time `reviews.unnamed_are_ours_until` gives (an older monitor's reviews, which the
  one-time move takes over; while it is blank, none is). A mirror of a live monitor (`mirror.enabled`, section 3.10)
  takes every review whose line 1 gives no name for its own. A review that gives another name, such as another
  deployment's under the same account, is never its own: the monitor follows up, gates on and counts only its own.
  One exception: when a review of the account that stands for the monitor's own (its line 1 gives one of the
  monitor's names, or none: an older monitor's posted after the move, or one posted by hand) covers the head of the
  monitor's review in hand, `rk review-close N` closes that review without a post and takes the account's review as
  the monitor's last review of the pull request (section 3.6). Its follow-ups answer that review and G1 reads it,
  until a review of the monitor's own is posted after it. Another deployment's review never stands for the monitor's,
  and a mirror closes none.
- **What it never does.** It never approves, merges directly, resolves threads, edits or deletes anyone's comments,
  dismisses reviews, or pushes code. Its only review state is a change request on a confirmed hold, and only with the
  merge queue live and its holds on (`features.merge_queue.auto_hold.enabled`).
- **The rule above all.** A draft is never posted raw. A first review pass fabricates evidence: "ACCEPT", example
  text copied from its rubric, "all CI green" while CI is still running. The verification pass verifies or deletes
  every sentence.

The header's light is the worst open Needed-to-merge severity; follow-ups never set it, so a pull request whose
blockers are all fixed reads green, with its follow-ups listed. The header ends `<light> First-round: <2–5 word reason>`
or `<light> Follow-up: <2–5 word reason>`. The post shows the lights as colored circles; in words:

| Light | Meaning |
|---|---|
| Green | Nothing open that blocks the merge |
| Yellow | A minor blocker (for example a pull-request reference the PR adds to a committed file) |
| Orange | A real defect or a significant gap: should be fixed before the merge |
| Red | Wrong results on a reachable path, a crash, a build or CI break on merge, or absolute performance numbers in public history |

The legend adds white (one of our own earlier items, retracted) and F (a follow-up, not blocking: its orange or yellow
light marks importance only). The words ACCEPT and REJECT are never used in posts.

## 2. Why it is built this way

1. **A review signal, not a gate.** The aim is an early, verified sign of what a pull request needs before it merges,
   across many pull requests a day. A comment, rather than a review state, keeps the merge decision with people: the
   monitor may not approve or reject a pull request, only leave a comment with its findings.
2. **A first review pass is optional.** A first review pass gives a second, independent opinion before the
   verification pass looks, cheaply when it runs on a local model (no cost per token, and the code stays on the
   machine) or on a remote endpoint. The bake-off (section 5.1) showed a local model catches some real issues with a
   high false-alarm rate, and nearly every posted finding comes from the verification pass and the checks (section
   10). So a draft is paired with verification rather than trusted, and by default no first review pass runs: the
   verification pass reviews alone (section 3.9). Drafting on a local Ollama model, on an OpenAI-compatible endpoint
   or on another Claude model stays available.
3. **Verification is mandatory.** A first review pass:
   - restates other reviewers' items as new;
   - invents test results and merge outcomes;
   - calls CI failures the pull request caused "unrelated";
   - copies example text from its rubric.

   So every posted claim is derived again by the verification pass, with a tool run ("ran") or a code read ("read"),
   and the verification pass reviews blind before it opens any draft.
4. **Our own earlier reviews are kept out of the first review pass.** A model that sees them repeats them nearly word
   for word, including hardware evidence it could not have produced. Our own items are filtered out of what it sees,
   so its pass stays independent.
5. **A fixed menu of tools, never a shell.**
   - A free-form "bash" tool is unsafe.
   - An off-the-shelf local agent harness runs tools with no approval by default. With a GitHub login that can write on
     the same machine, a local model could post.
   - So the local review loop gets five tools that build their argv from validated parameters (`shell=False`), and
     only read or run the repository's own tests.
6. **Our own loop drives the model, with explicit request settings.**
   - Ollama's OpenAI-compatible endpoint cannot pass `num_batch`. At the default batch and the full context, a large
     prompt can wedge the GPU.
   - Restarting Ollama on a wedged GPU can crash the whole host.
   - Ollama's native `/api/chat`, with a `num_batch` proven safe on a ladder test, avoids it. The raw model is used,
     never a wrapper Modelfile, with its limits set per request.
   - The same loop speaks the OpenAI-compatible API to a remote endpoint, whose server keeps its own batching.
7. **Judge the pull request as it would land.** A textually clean merge can still break, for example through a stale
   symbol or a test stub that misses a new step. So every review simulates the merge with today's develop and runs its
   checks there.
8. **CI's gaps decide the checks.** rocKE's CI runs hand-picked test file lists, never builds the pybind engine, has no
   torch on its GPU lanes and no `llvm-objdump`, and runs the platform tests from the installed tree, not the source
   tree. A green CI can hide broken or never-collected tests, so our checks target exactly those gaps (section 8).
9. **A public repository.** Posts carry no internal links and no absolute performance numbers (git history keeps them
   forever), measured ratios only where the repository's `platform/AGENTS.md` allows them, and only bare ticket tags
   such as `<JIRA_PROJECT>-NNNN`.
10. **Out-of-the-box evaluation.** A noisy model is not rescued with post-filters or extra verification passes: it is
    judged on its raw output at default settings. The local review loop's prompt stays the same for every pull
    request, so its output stays comparable.
11. **Claude's tokens go to judgment, not to re-reading.** A review's Claude tokens are the verification pass's, the
    code quality pass's and the orchestrator session's; `rk usage` counts them and `rk spend` prices them (section
    3.12). These measures keep them on judgment without changing what is posted:

    | Measure | What it does |
    |---|---|
    | A long prompt cache for the session | Claude Code's `promptCacheTtl` for the monitor's session (`claude.prompt_cache_ttl`), so the cache survives the minutes between ticks, and `subagentPromptCacheTtl` for the review agents (`claude.subagent_prompt_cache_ttl`: an agent's turns run back to back). `rk orchestrate` writes both into the Claude workspace's `.claude/settings.json` |
    | Compaction | The orchestrator's session compacts itself at `claude.auto_compact_window` tokens of context, which `rk orchestrate` writes as Claude Code's `autoCompactWindow`; the review agents' types keep a window of their own |
    | Dedicated agent types | `.claude/agents/`: each pass's model and effort, no workspace `CLAUDE.md` or memory (`omitClaudeMd: true`), only the pass's tools |
    | Each pass's effort | The verification pass at `features.review_automation.passes.verification_pass.effort`, the code quality pass at `features.review_automation.passes.code_quality_pass.effort` |
    | One call per step | `rk dispatch-pack`, `rk spec-post` and the verification pass's `rk wait-quality` (section 3.12) replace several reads and calls each |
    | One shared build | The checks' native build of the merged tree is the review's shared build: both passes reuse it read-only instead of building the head again |
    | A per-role lessons extract | `lessons_for.py verifier\|quality` prints only the sections a pass reads |
    | Long commands in the background | A command that may run past 4 minutes goes to the background and is checked within 4, so a pass's cache does not expire while it waits |
    | A lean tick card | `SKILL.md` keeps the per-tick procedure and the short form of the rules; `rules.md`, `commands.md` and `weekly.md` are read on demand |
    | Short hand-backs | The passes write the post, `notes.md` and `quality.md` to files and hand back a short summary |

12. **Capacity comes from parallel reviews; the machine stays protected.** One review and one build at a time keep up
    only while pushes are few; when pushes outpace the reviews, the backlog grows and the board says LOSING. So
    several reviews run at once (the parallel reviews, section 3.3), and builds and GPU tests keep their own limit
    (the parallel builds, section 3.5), because they load this machine whatever the first review pass is. The scaler
    moves both within their ranges from the machine's load. A local Ollama holds one model in memory, on a GPU that can
    wedge, so it allows one draft at a time unless the operator confirms more. A remote endpoint costs no local memory
    but has rate limits, so every review draws on one shared budget.
13. **Verification never moves to another model.** The verification is the real review: no local model in the
    bake-off matched Claude's quality, and a first review pass fabricates evidence. So no setting routes verification,
    a gate decision or a post to another model: they stay with Claude.
14. **Scope first.** Every review starts from the pull request's title, its body and the problem it says it fixes.
    The post's Needed to merge part lists every blocker, each with its light, so the author can fix them in one
    revision; significant items outside the pull request's scope go to Highly recommended follow-up (not blocking),
    and nits are left out. The author's reasoned scope call on a follow-up stands. Once the pull request merges, the
    agreed follow-up items it did not fix become follow-up tickets, from posted reviews only. Correctness and breaking
    changes are never follow-ups.

## 3. Architecture and roles

```text
 OPERATOR ---- features, priorities, merges, approvals ------------------------+
   |                                                                           |
   v                                                                           |
 ORCHESTRATOR: Claude Code, on the orchestrator's model and effort             |
   - runs the tick loop and starts the review driver                           |
   - reviews every PR with two passes: verification, code quality              |
   - decides the gates; writes only through guarded rk commands                |
   |                      |                              |                     |
   v                      v                              v                     |
 FIRST REVIEW PASS      THE CONTAINER                  GITHUB                  |
 (optional; none by     - the monitor's clone          - reads: gh, GET only   |
 default): the local    - wt<N>, wt<N>m, wtdev         - writes: guarded rk    |
 review loop (binding   - venvs, native builds           commands only <-------+
 worker: Ollama or      - ROCm; remote GPU               (write lock, ledger)
 OpenAI-compatible)       tests (section 3.7)
 or a model binding
 tools: a fixed menu (the local review loop's: read_file, grep_code,
 run_pytest, rule_scan, merge_conflicts); no network, no GitHub, no shell
```

| Role | What it is | Does | Never does |
|---|---|---|---|
| Operator | The person who runs the monitor; `<user_login>` is the account it posts as (`integrations.github_cli.username`) | Sets the features and their modes; decides priorities, merges, approvals and policy | — |
| Toolkit `rk` | The package's code (`<PACKAGE_DIR>`), with its home in `<TOOLKIT_DIR>` | Polls GitHub with GETs, picks reviews, stages them, runs the drafts and the checks jobs, gates candidates, keeps the merge plan and the dashboards; holds the guarded write commands | Writes on its own: every outward write is an `rk` command that the orchestrator, the operator or a tick run by fixed rules runs, for a feature that is live |
| First review pass (optional) | None by default; else the local review loop (the model binding `worker`) or another model binding: a model on an Ollama server, behind an OpenAI-compatible endpoint, or a Claude model (`features.review_automation.passes.first_review_pass`, section 3.9) | Independent first drafts: first rounds and follow-ups | Anything outward: its tools reach no GitHub, no network and no shell |
| Orchestrator | `features.review_automation.orchestrator.model`, in the Claude Code session | Runs the tick loop; reviews each pull request through two passes, the rocKE Review Verifier and the Code Quality Reviewer; decides; posts | Approves or rejects; hands verification, a gate decision or a post to another model |
| The container | The Linux environment the monitor runs in; on a host install, the host's own Linux (native, or WSL2) | Holds the clone, the worktrees, the builds and venvs, and ROCm; on a host install, the host's GPU too, when it has one | — |
| GitHub | `<ROCKE_REPO>` | Read through `gh`; written only through the guarded `rk` commands | — |

### 3.1 The local review loop's model

The model binding `worker` holds the model the local review loop drafts with
(`features.review_automation.passes.first_review_pass.model: worker`, section 3.9): first rounds and follow-ups. With
another first review pass, or none, it drafts nothing. The deploy settings file's `model_bindings.worker` names it, on
one of two providers (`rkmon template section model_bindings` gives each one's keys, with their comments):

| Provider | Where the model is | Its key |
|---|---|---|
| `ollama` | An Ollama server, the host's or another machine's, by an address the container reaches (inside the container, localhost is the container itself): `model_bindings.worker.server_url` | None |
| `openai_compatible` | An API gateway (on-prem or hosted), vLLM or LiteLLM: `model_bindings.worker.base_url`, its route prefix included | The secret `model_bindings.worker.api_key_secret_name` names (`rkmon secret NAME FILE` installs it) |

- **A gateway behind an API-management layer.** The key goes as a bearer token by default
  (`model_bindings.worker.api_key_sent_as` `bearer`); with `header` it goes in the header `api_key_header`, after
  `api_key_scheme` (blank sends the key alone, as a subscription-key header wants), and with `header_and_bearer` both
  ways, for a gateway that wants it twice. The base URL may carry a path prefix, for example
  `https://gateway.example.com/some-route`: `/v1/models` and `/v1/chat/completions` are built on whatever base is given,
  and a base that already ends in `/v1` is not doubled. Model ids keep their case and dots, as the endpoint lists them.
  Neither the key nor the headers are ever logged.
- **A gateway that is not /v1-shaped.** Some serve chat at `<base>/chat/completions`, with no `/v1`, and list their
  models outside the chat route's prefix. The start's checks try `/v1/chat/completions`, then `/chat/completions`, and
  the model list at `<base>/v1/models`, `<base>/models`, then the host's own `/v1/models` and `/models`; they record
  what answered (`worker.chat_path`, `worker.models_url`). A list may give each model a display name besides the ID
  requests take (a deployment ID): the binding's `model_id` is the ID, the start records the name
  (`worker.model_name`), and requests send the ID exactly as listed, whatever case the answer echoes.
- **Rate limits.** The shared budget uses the binding's `limits.requests_per_minute` and `limits.tokens_per_minute`.
  An endpoint that sends no `x-ratelimit-*` headers relies on them, and on backing off when it answers HTTP 429. An
  answer's tokens are what its usage reports (reasoning tokens included), else an estimate.
- **Answers.** The reasoning a server returns beside the answer (`reasoning`, `reasoning_content`) never joins the
  draft; fields a server adds (vLLM's `stop_reason`, `token_ids`, `routed_experts`) are ignored. An endpoint that
  answers in a simplified shape with no `choices` (no tool calls) is refused by the tool-call probe with that reason.
- **One gateway, two roles.** A gateway that serves both Claude and on-prem models lets the first review pass reuse
  the gateway address and key that Claude Code already uses. The orchestrator still runs in Claude Code; only the
  first drafts go to the on-prem model.
- **The probes.** A start sends a tiny chat and a tool-call probe shaped like the review loop's own calls, while the
  last check is missing, was of another address or model, or did not pass. A model that fails either is refused for
  the first review pass, with the reason. Review automation then shows as "on (incomplete)", with that reason, in the
  setup summary and in `rk features`, until a model passes.
- **What the start records:** `worker.probe` (what the checks found), `worker.chat_path`, `worker.models_url` and
  `worker.model_name`. Whatever the provider, the local review loop drives the model with the fixed tool menu of
  section 5.5.

### 3.2 The orchestrator model

`features.review_automation.orchestrator` (its `model` and `effort`) is the model of the Claude Code session that runs
the tick loop, the verification, the gate decisions and the posting. The verification pass runs on
`features.review_automation.passes.verification_pass` and the code quality pass on
`features.review_automation.passes.code_quality_pass`; their agent files, as filled in for the deployment
([agentic/README.md](../agentic/README.md)), take `model:` and `effort:` from those. The skill dispatches each agent at
its pass's effort (`rk roles` shows them), and never leaves an agent's effort unset, which would inherit the session's.
The session compacts itself at `claude.auto_compact_window` tokens of context; the passes' agent types keep a window
of their own.

- These three take a Claude model: `opus`, `sonnet` or `haiku` (the latest Claude Opus, Sonnet and Haiku), or a
  `model_bindings` entry whose provider is `claude`. The Claude Opus and Sonnet families are known to drive the skill,
  its procedures and the agent files well; for any other choice the setup summary (`rk features`) shows one warning:
  the skill files and the verification agents are long and use many tools, so a less capable model may skip steps or
  misread the rules.
- The optional passes, `first_review_pass` (section 3.9), `ci_triage_pass`, `first_code_quality_pass` and
  `notice_draft_pass` (each under `features.review_automation.passes`; `rkmon_lib/first_pass.py`), are `default`
  unless set: the pass in charge (the verification pass, the code quality pass, the orchestrator) does the whole task,
  with its own model and effort. Given a model, the optional pass does the first part and hands the pass in charge
  leads whose quotes were checked at their lines (`ci_triage.md` and `quality_leads.md` in the review directory,
  `rk notice-draft N --kind overlap|drift`); the pass in charge checks each lead, and an optional pass that fails or
  runs past its cap is a logged miss the pass in charge works without.
- `rk roles` shows every pass's model and effort, resolved.
- The tick triage is the toolkit's rule filter (`rkmon_lib/tickfilter.py`, `rk tick-filter`), with no model: it
  reads each tick first, so the session gets a turn only for the lines that need it (an audit, an outward action, or a
  line no rule knows).
- No setting routes verification, a gate decision, an overlap owner decision or a post to another model.
- Line 1 of a post gives the automated reviewer name and the model that drafted it, or, with no first review pass, the
  verification pass's (`verifier_name`); the footer names the model that verified it or, with the review offload on,
  audited it.
- Tokens are counted by `rk usage` and `rk usage --by-review`, with no price. Review spend in US dollars (`rk spend`,
  section 3.12) runs with review spend on (`features.review_automation.spend.enabled`, each model's price in
  `features.review_automation.spend.usd_per_million_tokens`).

### 3.3 Parallel reviews

The parallel reviews are how many reviews are in flight at once (each staged, drafted when a first review pass runs,
and given its checks), within their range (the README's [Settings reference](../README.md#settings-reference)).
`features.review_automation.parallel_reviews` is the starting setting; the scaler (section 3.5) moves the current
setting from the machine's load. What runs is the smaller of the current setting and what the first review pass's
endpoint allows.

- **A local Ollama** allows one draft at a time, because it holds one model in memory, unless its binding's
  `limits.max_parallel_requests` allows more.
- **A remote endpoint** costs no local memory, but every review shares its rate limits through one budget: a file under
  `<RUN_DIR>`, locked while it is read and written, that counts the requests and the tokens of the last minute against
  its binding's `limits.requests_per_minute` and `limits.tokens_per_minute`. An HTTP 429, its Retry-After, and
  rate-limit headers that say the endpoint is spent make every review wait, with backoff. Every wait on the budget is
  logged in `<RUN_DIR>/budget.log`.
- **With no first review pass,** no endpoint caps the parallel reviews; a model binding's own limits are the
  limiter's and the scaler's.
- The setup summary says what is in effect at the start, and why: "asked N, running M, because …"; `rk status` shows
  the current setting against its range, the scaler's reason and its last changes.

### 3.4 The review driver

One driver process keeps the parallel reviews busy: `<PACKAGE_DIR>/tools/review_driver.py`, which `rk orchestrate`
starts in the background, with a lock so that only one runs.

- It keeps up to the parallel reviews in effect in flight, each pull request with its own worktree pair (`wt<N>` and
  `wt<N>m`) and its own folder (`<WORKING_DIR>/workers/pr<N>`).
- At every step (once a minute) it gives the scaler a sample (section 3.5), and a review that is over leaves the
  RAM disk at once.
- It prints `REVIEW_READY #N` for each review ready to verify: its draft (or, with no first review pass, its staging)
  and its checks done. The orchestrator verifies each one as it comes; verifications run in parallel, as background
  agents.
- The review driver never starts a pull request larger than
  `features.review_automation.large_prs.in_depth_review_over_lines` from the plan: the tick prints `REVIEW_MANUAL #N`,
  and the orchestrator starts it (`rk review-start N`). Its first round, the in-depth review, starts only once the
  operator approves it: until then nothing of it runs and no line names it to the orchestrator, Needs you asks for the
  approval with the pull request's size, about how many agents it runs and their cost (Approve on the admin dashboard
  or `rk review-approve N`; Decline there or `rk review-skip N`), its review queue entry waits
  (`waits: your approval`), and `rk review-start N` refuses it. The approval stands for that round, whatever the head
  does before the review starts. A first round forced on it once reviewed (`rk review-start N --kind first-round`) is
  an in-depth review too, and waits for an approval the same way. Its verification is split (`rk review-split`,
  section 3.12); its later rounds are light and need no approval.
- The review queue (`rk review-queue N [N ...] [--again]`, state `review_first`) goes before the plan: the driver
  starts its entries first, in order, each once it may start (its head settled, no review of it in flight or awaiting
  verification; without `--again`, also no review of ours at its head, posted or kept, and, for a follow-up, the gap
  after our last review passed). An entry leaves the queue once its review starts; a push to the pull request during
  that review, before its post, puts the entry back at the front of the queue as it was, as the same round when
  `rk review-start N --kind` forced one (the driver's line says so: `back in the review queue`). `--again` reviews a
  head we already reviewed once more, and its post goes out with `rk post N FILE --premises SPEC --again`.
  `rk review-start N` refuses while all parallel reviews are busy and names the queue.

### 3.5 Local checks: the parallel builds

Builds and GPU tests (review checks, gates, develop sanity runs, bisects) have their own limit, the parallel builds,
within their range (the README's [Settings reference](../README.md#settings-reference)); `monitor.parallel_builds` is
the starting setting.

- A draft that finishes while all parallel builds are busy waits for one.
- The parallel reviews never change the parallel builds.
- **The scaler** (`rkmon_lib/scaler.py`) moves the parallel reviews and the parallel builds in effect one step at a
  time within their ranges, from samples at least 50 seconds apart:
  - down after two samples in a row over: the CPU (the load average over the cores; in a CPU-limited container, the
    larger of that and its own CPU time over its limit) or the memory (used over total; tmpfs pages count as used, so
    the RAM disk's files do; in a memory-limited container, the fuller of the machine and its own limit) above
    `max_host_usage_pct`, or, for the parallel reviews, the first review pass's endpoint spent (an HTTP 429, or 80% of
    its budget, in the last 5 minutes);
  - up after three samples in a row with headroom, 5 minutes after the last change: every piece of the setting busy,
    the CPU and the memory at least 10 points under the ceiling with one more piece's RAM disk footprint counted, room
    on the RAM disk, the endpoint not spent, and the parallel reviews under what the first review pass's endpoint
    allows;
  - otherwise it holds. Lowering never stops running work: what is in flight finishes, and nothing new starts until
    fewer run than the new setting. A new starting value in the configuration starts again from it.

  Every change is a line of `state/scaler.jsonl` (`kind: setting`) with its inputs and its reason; the scaler's own
  state (the current settings, the streaks, the last sample, the footprints) is `state/scaler_state.json`, apart from
  `state.json`, so a sample never makes the next tick full.
- **The RAM disk** (`host.resources.ramdisk_gib` in the deploy settings; `rkmon_lib/ramdisk.py`). A tmpfs the
  container mounts, from the compose override made from `host.resources.ramdisk_gib`, of the shape
  `docker/compose.local.example.yaml` shows; on a host install, a tmpfs mount of yours. On it: each pull request's
  worktrees (`mkwt.py` makes them at `<ramdisk>/wt<N>`, and `<WORKING_DIR>/wt<N>` links to each); on a host install,
  their builds (`build_native.py` builds in `<ramdisk>/build/wt<N>m`, and the build folder links to it); in the
  container, the PR runners' slots (`/ramdisk/slots`: each job's results, builds, temporary files and home, and the
  build a slot keeps for its review); and `paths.tmp`, unless the
  configuration names another folder for it (the review directories and the TMPDIR of the checks and the review
  tools). Never on it: the clone, `state/`, the logs, the saved probes, the posts, the caches, the `workers/pr<N>` folders and
  the shared trees.
  - Each sample measures what each review in flight (its trees, its review tools' build, its review directories) and
    each build (its build folder) holds there, with the slots its pull request holds. A piece's peak raises the
    running high estimate of its type when it ends (a seed estimate until measured; a lower peak pulls it down by a
    tenth of the difference; `kind: footprint` lines).
  - A review, a pool draft or a build starts only when the RAM disk's free space, less what the running pieces may still
    grow to (a review awaiting verification, whose trees do not grow, at most half a GiB; the whole never more than the
    RAM disk; `rk status` shows it), holds its estimate, and the memory stays under the ceiling with that counted as
    used. Otherwise it waits (`REVIEW_WAIT`, "checks wait"), and nothing spills to the disk.
  - A finished review (not in flight, not awaiting verification, no gate or pool draft) leaves the RAM disk at the
    next step: its worktrees, its builds and their links; its review directories go once they have been quiet for two
    hours. A restart empties it: a draft awaiting verification gets its worktrees again (`mkwt.py`, the head merged
    with today's base) before `rk review-recheck`, `rk dispatch-pack` or the code quality pass reads them.
  - `rk status` shows the RAM disk's use; `rk doctor` checks that it is mounted, as large as
    `host.resources.ramdisk_gib`, and within the container's memory limit.
- **Prefill protection, with a first review pass on a local model only** (an endpoint on this machine's loopback
  address): no build starts in a local draft's first 5 minutes, and no local draft starts in a build's first 10
  minutes. Builds during a large prompt's first minutes can wedge a fresh runner (section 9, issue 29). A remote model
  needs neither rule.

### 3.6 One PR, start to finish

1. **Pick.** The tick's plan picks pull requests in priority order once the head has settled:
   `features.review_automation.schedule.push_cooldown_minutes` after a push, `settle_burst_min` minutes after the last
   of a burst of pushes, and at most `settle_develop_fix_min` minutes for a develop fix or an operator priority.
2. **Stage and draft.** The driver gives the pull request a place among the parallel reviews. The job fetches its data
   with GETs only (`fetch_pr.py`, into `workers/pr<N>/pr/`) and makes the worktrees (`mkwt.py`). With no first review
   pass that is all (`draft_run.py N --stage-only`). With the local review loop, `review_agent_run.py N` also copies
   the review book and runs the loop, which writes `review_agent.md`, `agent_log.txt` and `stats_agent.json`; with
   another model binding, `draft_run.py N` runs its tool loop (section 3.9).
3. **Checks.** As one of the parallel builds, a checks job builds and tests the PR merged with today's develop, on a
   pristine tree, into `checks/checks.md`: the native build and ctest (the review's shared build), byte identity at
   every LLVM flavor, goldens, the PR's tests against develop's, its changed tests in the installed tree, compile
   checks, saved probes, CI at head (earlier attempts included), the rule and ABI scans, the overlapping work in other
   open PRs, and the scripted sections of section 3.11.
4. **Ready.** The driver prints `REVIEW_READY #N`, with its inputs (with no first review pass, "no drafter: verify
   without a draft … Verify alone"). A merge-only head whose checks are clean is carried instead, with nothing posted,
   or gets a micro round (section 3.11).
5. **Verify.** `rk dispatch-pack N` prepares the review directory and prints what the briefs need. The orchestrator
   briefs the verification pass (the rocKE Review Verifier) with the paths of the checks, `facts.txt` (`rk facts N`),
   the classification and, when a first review pass ran, the draft, and dispatches the code quality pass (the Code
   Quality Reviewer) beside it. A follow-up whose new commits are only develop merges, docs, or test files and docs that
   answer our own open findings gets no code quality pass; nor does a micro round. The verification pass reviews blind,
   then reproduces or downgrades every draft claim, waits for the code quality pass's `quality.md` (`rk wait-quality`),
   folds its findings in, and writes the post and its notes. When the facts list a review of our account at that head
   that this deployment did not post and that stands for ours (an older monitor's under the same account, or one
   posted by hand; the facts mark another deployment's "not ours to take"), and the review's content confirms that
   head, the verification pass writes no post and hands back `close, no post`, and the orchestrator runs
   `rk review-close N`: the review closes without a post, and the account's review is our last review of the pull
   request (section 1, Its own reviews), so the pull request is not reviewed again at that head and its next head gets
   a follow-up. A review that three hours after its hand-off is neither posted nor kept unposted prints
   `` !! VERIFY_STALE #N: handed to verification <age> ago, nothing posted (head <head>); if our account posted at that head already, run `rk review-close N` ``,
   at most once an hour per PR and head, until it is posted or dropped (a mirror's line ends at the head: it closes
   none); a split review gets three hours for each wave of part verifications and three more for the cross pass
   (`rkmon_lib/verify_stale.py`).
6. **Post.** `rk spec-post N --round R` writes the post's spec from the verification notes and live reads, and posts
   only when nothing differs (no `CHECK:` line). The premises stop the post if the head, the base branch's changes to
   the PR's own files, the conflict set, CI or the threads changed since the verification, and a fix line fails its
   premise (`rkmon_lib/fixline.py`: a skip guard on a module CI lacks; a "CI still runs it" that `rk lane-of` denies at
   the head; a skip guard, tier change or test move without a `fixcheck: N<k> ran|read <how>` notes line), or the
   ABI premise fails (`rkmon_lib/abipremise.py`: a "no bump" while the checks' ABI section lists a change without a
   bump, or an ABI number neither develop's nor develop + 1, without an `abi:` notes line naming abi.h's rule).
   Otherwise `rk post` checks its own guards again, posts (with review automation live; otherwise it keeps the review
   unposted) and writes the ledger; any overlap owner the post proposed is recorded (`rk overlap-decide`, an `AFTER`
   line of the spec).
7. **Record.** A copy goes to `<OUTPUTS_DIR>`; the corrections are counted (`rk save N verify ...`); the verification
   notes are archived in `<WORKING_DIR>/lessons/notes/`; each draft's assessment lines are recorded; a finding that
   reproduces quickly is saved as a probe that later checks re-run; each new follow-up item is tracked until it is
   fixed, deferred by the author, or ticketed.

### 3.7 Remote GPU tests (optional)

The container has no GPU, so a pull request's GPU-only test cases skip there. With remote GPU tests on
(`features.review_automation.remote_gpu_tests.enabled`, with review automation enabled), those cases run on a GPU node
of a Slurm cluster reached over SSH (`rkmon_lib/gpu.py`). Every site value is a setting under
`features.review_automation.remote_gpu_tests`: the cluster's login host, partition, accounts and GPU types under
`cluster`, and the node's work folder, Python and setup under `node`.

1. **Choose.** The checks' pytest records every case it skips and why (`checks/tests/tests_skipped.tsv`). Only the cases
   whose reason names a GPU, a device, a gfx target, HIP or ROCm go, each to the target its reason names (else
   `features.review_automation.remote_gpu_tests.cluster.default_target`). Builds, CPU tests, lint and goldens stay
   local.
2. **Queue, never wait.** When the checks end, a local background job asks the login host to fetch the PR's head and
   develop from GitHub itself, check rocKE's provider folder out at their merge and at develop, and submit one batch job
   per target (one GPU). The accounts are tried in order. The draft goes on to verification at once.
3. **Post on time.** `rk post` adds a "GPU tests" row to the review's first table: pending, with the cases and the
   target.
4. **Judge on the node.** The job probes the node (the target's device is visible; hipcc, when there is one, compiles
   the HIP header), then runs the cases on the merge's tree with `tools/pytest_vs_base.py`, which runs each failure
   again on develop's tree in the same job. It sets `ROCKE_REQUIRE_<TARGET>=1`, so a case for the target fails rather
   than skips on a node without its GPU. A failure develop passes there is the PR's; one develop fails too is not.
   A node that fails the probe fails nobody: the job goes again without it, up to
   `features.review_automation.remote_gpu_tests.bad_node_resubmits` times, and the tick names the node for
   `features.review_automation.remote_gpu_tests.cluster.excluded_nodes`.
5. **Report.** Each tick polls the queue. When a job ends, its results come back to `<WORKING_DIR>/workers/gpu/<job>/`
   and its folder on the login host is removed. `GPU_DONE #N` asks for `rk gpu-row N`, which edits the posted row in
   place (guarded and ledgered). For the PR's own failure, `GPU_FAIL #N` names a drafted notice: the orchestrator
   grades it like any finding and posts it with `rk notice N FILE --kind gpu`. While review automation is in
   prototype, both only journal what they would do.
6. **Clean up.** A job of this monitor that no record names is cancelled; a job pending longer than
   `features.review_automation.remote_gpu_tests.wait_report_after_minutes` is reported once (`GPU_PENDING_LONG`, and
   on the board), never resubmitted on its own.

Public text never names a node or a GPU product: the row and the notice give the gfx target only. The node's name
stays in the local records, so `rk gpu-run N --same-node` (or `--node NAME`) can run the cases again on that machine.

### 3.8 The review offload (`offload.mode`)

An option in which the first review pass's model does the reviews, first rounds included, and Claude audits them;
Claude reviews in depth only when the evidence or the pull request calls for it. `offload.mode` takes three values:
`off` (the default: the pipeline above), `shadow` (the trial) and `on`.

1. **The draft job** (`on`) runs `worker/offload_agent.py` in place of `review_agent.py`: three passes of the same
   constrained tool loop, each its own conversation:
   - **draft**: the review book's review, written as findings (N1, ...), the checks it ran (D1, ...) and the utility
     line, in `draft.md`;
   - **quality**: the code quality pass, by the filled-in code quality team member with a short preamble that maps its
     shell steps to the tools (Q1, ...), in `quality.md`;
   - **regrade**: every prior item at this head, by the filled-in verifier team member's §3, from `rk review-prep`'s
     `facts.txt` and `classify.txt`, running the saved probes (P1, ...), in `regrade.md`.
2. **The finish job**, a light job once the checks are done: the **assemble** pass writes the post in the verifier's
   Output Format from the three files and `checks.md` (`post_draft.md`, with `notes_draft.md` and its `map:` lines),
   then `rk evidence-check N --offload`. The review then prints `REVIEW_READY #N (<kind>, audit)` or
   `REVIEW_READY #N (<kind>, deep: <why>)`.
3. **The audit.** The orchestrator dispatches the verifier with an "offload audit" brief (its team file's Offload
   audit section, the skill's `offload.md`): every 🔴/🟠 claim's evidence confirmed by re-running its cheap command or
   reading its quoted lines, the sample of 🟡/🟢 claims `evidence_check.md` draws (`offload.spot_check`, at least
   `offload.spot_check_min`), the format, the hygiene and the premises; the post goes through `rk post --premises`.
4. **Deep.** The full review (section 3.6) with the offload's files as extra inputs, on the code quality pass's model,
   when the evidence check fails, a pass wrote nothing (the model went down), a first round is large or touches the
   risky paths (`offload.deep_lines`, `deep_files`, `deep_hot_spots`, `deep_paths`), or the audit finds claims that
   contradict or a draft that does not fit the diff.

**Evidence.** The passes' tools: `read_file`, `read_lines` (numbered lines of the head, the merge with develop or the
merge base, to quote exactly), `grep_code`, `git_diff`, `git_log`, `run_pytest` (files or node ids), `rule_scan`,
`merge_conflicts`, `write_probe` and `run_probe` (its own probes and the saved ones), `run_mutant` (a pytest plugin that
undoes a claim: the tests run without it, then with it; KILLED, SURVIVED or INVALID), `ci_fails`, `job_log`,
`read_job_log` and `record_evidence`. Every run gets an id and a record in `runs.jsonl` (the tool, its arguments, the
argv, the exit, the seconds, a digest of the output reduced to what must match, an excerpt). `record_evidence` ties a
claim to a run, or to a quote it checks at once (a wrong quote is refused, with what the line says), in
`evidence.jsonl`; a claim left without evidence is recorded there as such. The model's own probes and mutants run only
in the sandbox (`worker/review_sandbox.py`): bubblewrap (the file system read-only but the review folder, no network, its own
process ids), else an unprivileged user namespace (no network; the working folder and the homes read-only); with
neither, those two tools refuse, and the saved probes run as the checks run them.

**`rk evidence-check N --offload [--round R]`** (`rkmon_lib/offload_evidence.py`), with no model: each quote must stand at its
`file:line` in the commit the run recorded (read from the monitor's clone, never a lazy fetch); each run's command,
rebuilt by the tool's own builder, must be the recorded one, and a run that took at most `offload.check_rerun_max_s`
is re-run under a time cap (`offload.check_total_s` in all, 🔴/🟠 claims first) and must give the same reduced output;
CI reads that changed since are CHANGED, not failures. A claim is PASS, FAIL, CHANGED, NOT RE-RUN or MISSING; the run
is FAIL when a claim fails or a 🔴/🟠 claim has no evidence. The report goes in the run's folder, and the run is kept
in `offload/pr<N>_<round>/` in the lessons folder.

**The shadow trial.** With `shadow`, the draft job runs `review_agent.py` as always, then the three passes into
`workers/pr<N>/offload/` (a failure there leaves the draft alone); the finish job assembles and checks them before the
usual `REVIEW_READY`. Nothing of it is posted. `rk shadow-report [N ...]` (`rkmon_lib/shadow.py`) compares each kept
run with the review we posted for that round, read from GitHub: findings matched (by file, line and the words of the
problem), missed and extra, each offload finding with its evidence check; both header lights; the route it would have
taken; the first review pass's model's tokens. Then a summary over the last `offload.trial_prs` pull requests:
findings found (🔴/🟠 apart), extra findings and whether their evidence held, evidence checks passed, routes, tokens
per review.

**Rates and tokens.** The passes share the endpoint's one rate budget by role (`offload.budget_shares`: draft,
quality, regrade; the assembly counts as the draft's): while requests wait, a role under its share goes before one
that has used it, and alone a role may use the whole budget. Every request's tokens go to `<RUN_DIR>/tokens.jsonl`
with its pass and step (`reviewer_client.Budget.count`; `rk usage` reads it), and each run's totals per pass to
`tokens_total.json` in its folder (`reviewer_client.TokenLog`): the numbers for sizing the parallel reviews. No price
is applied to the first review pass's model's tokens.

### 3.9 The first review pass, the draft stage and the model bindings

**The first review pass** (`features.review_automation.passes.first_review_pass.model` in the deploy settings,
`rkmon_lib/bindings.py`) chooses the model that drafts each review: `default` (the package's: no draft, and the
verification pass reviews alone), `worker` (the local review loop, section 3.1), or a model binding's name, a
built-in Claude model (`opus`, `sonnet`, `haiku`) included (the tool loop, below). What each choice puts on line 1
and in the footer, and how to swap it, is in the README's
[Swapping the reviewer model](../README.md#swapping-the-reviewer-model). A choice that names no binding stops the
monitor before a review starts (`config.problems`), and `rk drafts` and `rk doctor` flag a binding that is wrong.

A review's first review pass is the draft stage (`rkmon_lib/draft_stage.py`). Each review runs it through one
contract: the job stages the pull request (its data in `workers/pr<N>/pr/`; our last review on it, posted or kept,
in `pr/ours.md` by `rk stage-ours`: the head it covers, a `- Posted:` or `- Kept unposted:` line, and its body under
`## Review`; its trees `wt<N>` and `wt<N>m`), writes `workers/pr<N>/staged`, and leaves
`workers/pr<N>/review_agent.md` (the draft, in the post format); a stage that keeps evidence also leaves
`prior_leads.md`, `evidence.jsonl`, `runs.jsonl`, `draft_run.json`, `draft_findings.json`, `tokens_total.json`,
`evidence_check.md` and `prior_settled.md`. The stage follows from the first review pass (`draft_stage`, a derived
setting):

| Stage | When | The draft |
|---|---|---|
| `local-worker` | the first review pass on `worker` | the local review loop (`review_agent_run.py`, with the review offload's passes when `offload.mode` asks for them); no build in the local model's first 5 minutes; no draft twice: the pull request leaves the picks until `rk review-unskip N` |
| `tool-loop` | the first review pass on another binding | `worker/drafter.py` on that binding (`tools/draft_run.py`); its checks start once it has staged; several run at once; no draft: one retry, then REVIEW_READY with the draft marked missing ("no draft (the drafter failed twice): verify without it") |
| `none` | no first review pass (`default`) | staging only; REVIEW_READY "no drafter: verify without a draft"; the verification pass reviews alone |

**With no first review pass** the driver stages the pull request (its data and trees) and runs its checks, and the
verification pass reviews alone. There is no draft pool. Line 1 names the verifier (`review_first_line_no_draft`):
`verifier_name` (the verification pass's model in words) at the verification pass's effort, after the automated
reviewer name. The footer reads `_<First-round|Follow-up> review: verified by the orchestrator (<verifier_name>).
Checks: …_`, with no "first pass by". `rk post` takes that line 1, and `rk post --premises` stops a footer that
credits a first pass. No draft lint runs, and the assessment records no draft. The local review loop then drafts
nothing, though its model may still serve the optional passes (section 3.2).

**Model bindings** (`rkmon_lib/bindings.py`, `rkmon_lib/deploy_tree.py`, `worker/llm_client.py`). The deploy
settings' `model_bindings` name the models the passes may use; their three forms (`claude`, `openai_compatible`,
`ollama`) and their keys are in the README's [Model bindings](../README.md#model-bindings). `rkmon_lib/deploy_tree.py`
translates them into `config.json`'s own shape, with the binding named `worker` as config `worker`. A `claude`
binding is reached as the Claude session reaches Claude (`integrations.claude_code`'s address and the `claude`
login's key), with the effort as `output_config.effort` and no temperature or thinking setting. A turn goes back with
its thinking blocks; when the endpoint refuses them (the tool loop hides older tool results, which edits the history
around them), the request goes once more without them. A key is read at each request and never logged. `rk drafts`
shows the first review pass and the bindings (never a key) and flags what is wrong with them.

**The tool loop** (`worker/drafter.py`, `worker/prior_settle.py`, `worker/llm_client.py`, `worker/review_tools.py`).
Its local work is git and grep. A draft runs in phases; each cap is a key of the internal `drafter` settings, whose
defaults are in [docs/internal_settings.md](internal_settings.md):

| Phase | Who | What | Its caps |
|---|---|---|---|
| settle | script | the merge with develop, `rule_scan`, the CI row; the prior items, and the ones that need no model settled with evidence: approvals and the Threads row from the staged data, our last review's rows and findings whose cited files are unchanged since its head (the blobs at the head and at develop, then and now), its closed rows that cite no file, and items whose saved probes all give FIXED or REPRODUCES; `prior_leads.md` lists them | none: seconds; the probes run beside the prior tasks, each within `drafter.probe_timeout_s` |
| prior | model | one short task per item left open: its card holds the item, our last review's status of it, what changed since, and the code it cites; the task settles it FIXED, OPEN or PARTIAL with a quote at file:line or a run, or SKIP (no review point) | `drafter.prior_task_steps` requests a task (the last offers only `settle`), `drafter.prior_parallel` tasks at once, at `drafter.prior_effort`; the phase within `drafter.prior_wall_min` and `drafter.prior_token_budget` |
| hunt | model | net-new findings, the files changed since our last review (Δ) first; every 🔴/🟠 needs its evidence before it may finish | runs at the same time as the prior phase, within its own `drafter.hunt_wall_min` and `drafter.hunt_token_budget`, at the first review pass's effort |
| assemble | script | the post: its prior table (each row's status, settler and evidence id), the data rows, the findings | — |

The run as a whole stays within `drafter.review_wall_min` and `drafter.review_token_budget`; each phase's caps fit
inside them (`rk drafts` flags one that does not), so the hunt keeps its own time and tokens whatever the prior tasks
spend. Every request starts from the repository profile's rules card and a task card of at most about
`drafter.card_tokens`; the model reads hunks on demand, the hunt's tool results older than the last
`drafter.keep_results` are hidden behind a note of the call, and a hunt out of rounds wraps up at
`drafter.mechanical_effort`. Findings, prior statuses and evidence are tool calls, checked at once (a quote must stand
at its line). The items that hold no review point of their own get no row: the pull request author's replies, and the
texts the profile's `skip_rx` names. The repository profile (`profile_dir`: the package's `repository/<org>__<repo>/`
for the configured repository, else the generic profile; `drafter.json` and `drafter_rules.md`) holds what is
specific to the repository; the tracker keys a header tag may carry are the configuration's (`tracker_keys`,
`features.ticket_creation.project_key`). `draft_run.json` records each phase's status, time, requests and tokens, and
the method (`phased`). The tool loop reads its folders under the working folder (`paths.working_dir`): the trees, the
saved probes, the temp folder, the venv and the tools at their default places.

**The limiter** (`worker/llm_client.py`). Every model caller of a binding shares one bucket per provider, key and model,
under `drafter.limiter_dir` (blank: `<WORKING_DIR>/llm_limits`): requests and tokens per rolling minute against the
binding's `limits.requests_per_minute` and `limits.tokens_per_minute`, the requests in flight against its
`limits.max_parallel_requests`, a wait for every caller after an HTTP 429 (its Retry-After, else a doubling back-off),
and, when callers wait, the best-ranked first: a develop fix's or a priority's draft, then the draft closest to done;
the last `drafter.reserve_pct` of a minute goes only to the best-ranked caller. A draft's threads share its limiter,
each with its own phase: a request ends by the sooner of its phase's deadline and the run's (the socket waits no
longer, and a stream still running then is cut), and a phase stopped by its caps keeps what it recorded.
`llm_limits/tokens.jsonl` has one line per request (its pass, step and task), `limiter.log` one per wait.

**Drafts at once** (the pool). With the tool loop, beside the parallel reviews the driver drafts the next reviews it
would pick, one more per advance, up to `draft_parallel_max`. Before each draft starts it asks the scaler
(`rkmon_lib/scaler.py`), for the first review pass's binding: the bucket's last 5 minutes (the share of the budget
used, the waits, the 429s), the host's CPU and memory against `max_host_usage_pct`, and the queue (the backlog, its
oldest wait, the develop fixes and priorities in it). A draft starts while the last 5 minutes used under 80% of the
budget, no 429 came, and the drafts running are under the ceiling; with the RAM disk on, a pool draft also needs room
there for a review (section 3.5). The host's load moves the parallel reviews and the parallel builds (section 3.5),
not this answer. Every answer is a line of `state/scaler.jsonl` (`kind: draft`). A review adopts its pull request's
pool draft at the same head; a pool draft whose pull request closed, moved or is covered is stopped, and every pool
draft stops when the stage runs no pool.

**The probes.** The saved probes of our open findings (`<WORKING_DIR>/probes/pr<N>/`) run once a review: the tool
loop's settle phase runs each on the tree the checks probe (the merge with develop when it is clean, else the head)
and writes their lines to `probes.txt` with the tree's commit; the checks' re-probed section gives those lines when
they are for the same tree and the same probes, else runs them.

**The probe environment contract.** A saved probe runs on any machine the monitor runs on (the container, or a host
install), so it names no machine's folder: no home folder, no `/opt/rocm/core-<release>`, no working folder. `tools/run_probes.py` runs each as `bash <probe>.sh <tree root>` and
gives it these, from the job's settings:

| Variable | What it is |
|---|---|
| `$1` | the tree the probe checks (the merge with develop, or the head) |
| `PROBE_PYTHON` | the checks venv's python (`paths.venv`; `python3` without one): rocKE's tests and imports run with it |
| `RKMON_ROCM_LLVM_BIN` | ROCm's LLVM tools (`paths.rocm_llvm_bin`: clang, llc, ...) |
| `RKMON_BUILD_DIR` | the tree's native build (`build_native.py`'s `<build>/<tree's folder>`, with `core/` and `pybind/`) |
| `RKMON_REPO`, `RKMON_REPO_DIR` | the repository (owner/name) and the monitor's clone of it |

A value the job does not know is empty; a probe that needs it says so and exits 2, which reads as ERROR, never as
FIXED or REPRODUCES.

Every probe runs clean: `HIP_VISIBLE_DEVICES` and `ROCKE_LLVM_FLAVOR` are dropped whatever the caller set, and pytest's
rc 5 (no test ran) reads `ERROR rc=5 (no test ran)`. Before a probe is saved, `rk probe-check DIR TREE --simfix PATCH`
(`rkmon_lib/probes.py`) runs it on DEVBASE, the tree and a simulated fix, each but the tree a `git archive` in a
scratch folder: it must read 2, 1 and 0 (`<probe>: DEVBASE 2 · tree 1 · simfix 0 · OK`; a crash and any other code
read 3), and carry no flag (a hardcoded host path, a `^FAILED` grep without `SUBFAILED`, `HIP_VISIBLE_DEVICES=-1`, a
script that never reads `$1`). `rk probe-port DIR` rewrites older probes to this contract (a dry run unless
`--write`) and lists what needs a hand edit: a torch or second venv, a review's worktree, the toolkit's folder, a path
inside a longer string.

**The evidence check** (`rk evidence-check N [--dir DIR]`, `rkmon_lib/evidence.py`). The draft job runs it before it
ends. Each claim's quote must stand at its line in the clone at the recorded commit; each cheap tool run must give the
same output again (a saved probe's run: the probe and its tree must be the ones that ran); an unchanged claim's blobs
and a staged fact must read the same again. Each failed or missing claim gets a mark in its row of the draft, and the
run is kept in `<WORKING_DIR>/lessons/drafts/pr<N>_<head>/`. It also writes `prior_settled.md`
(`rkmon_lib/prior_settled.py`): the prior rows settled with passing evidence, one of them named for a spot check, and
the rows to look at (evidence failed or missing, a status that flips a 🔴/🟠 against our last review, or no phase
settled them).

**Records.** Each draft that ends (ready, missing, dropped) is one entry of state `drafts`: its role, binding and
method, tokens, requests, waits, wall time, attempts, its phases' times and tokens, its prior rows by settler,
lateness (a draft the review waited for after its checks), its findings and evidence check. `rk post N FILE --premises
SPEC` reads the verifier's assessment lines in `notes.md`, one set per draft, each line starting with its tag (a
binding's name, here `cand`, or for the local review loop `worker.tag`, by default the leading letters of its model
id, here `qwen`; a draft's findings N<k> are G<k>):

```
cand G1 match N2 · cand G2 new N3 · cand G3 false <reason> · cand G4 dropped <reason>
cand missed N1 🟠 · cand prior P3 right · cand prior <row's first cell> wrong · cand none
qwen G1 match N1 · qwen missed N2 🟠 · qwen prior <row> right
```

into `state/drafter_assessment.jsonl`, one record per draft with lines: the pull request, round and head, the draft's
role, binding, tag and method, its status at post time (ready, late, missing), the counts by severity, the prior
statuses right and wrong (by settler, script or model, for a phased draft), and the draft's tokens, wall time and
phases. `rk drafts` prints the card "<TAG> drafter vs <reference>": a column per tag, each on its current method only,
with a line counting the drafts of other methods it leaves out; prior items (script and model statuses right, the rows
each settled and their evidence pass rate, the rows the verifier judged), new findings (🔴/🟠 recall, 🟡 recall, false
claims per draft), tokens and minutes per phase, and the drafts missing or late at post time; with it, the pool, the
last drafts and the scaler's last decisions. No dashboard shows it.

The tests are offline: model endpoints point at the closed port 127.0.0.1:9, the client's requests go to fakes, and
no test reads a real key.
### 3.10 A mirror of a live monitor (`mirror.enabled`): an A/B comparison

`rkmon_lib/mirror.py`. With `mirror.enabled` (the one switch) this monitor reviews the same pull requests at the same
heads as a live monitor, writes nothing outward, and keeps each review with its cost, for comparing the two review
pipelines.

- **Every outward write is refused before any request.** Each command that sends (the table in `rkmon_lib/outward.py`:
  each row with a destination) stops with a `REFUSED:` line before it sends: at the top of `rk`'s command dispatch,
  and the writer's sends for another process at their gate (`write_gate`). `rk post` and `rk spec-post` are not
  refused: their post is written to the mirror's folder instead. Below the commands, `gh.run` (every gh write), a team
  chat message (`post.teams_post`, in the same words) and a ticket's filing refuse too, so a write from a tick or the
  admin dashboard goes nowhere either.
- **The feed.** The live monitor appends one line to its `state/mirror_feed.jsonl` at each review start
  (`rkmon_lib/review_cost.py`: `{pr, head, round, kind, again, merge_only, at}`); this monitor reads it from
  `mirror.feed` (mounted read-only). The review driver starts reviews from its lines alone, oldest first, never the
  plan's, the review queue's or the draft pool's: each at exactly its head (a line whose pull request has another head
  on GitHub, or is closed, is dropped), past the settle, and as a re-review when the live monitor already posted at
  that head. Lines from before the mirror's first advance, and lines older than `mirror.max_age_min`, are not taken.
- **The code quality pass** runs as the live monitor's does, so the two sides differ only in what the comparison
  sets out to compare; the records name each review's arm (A).
- **Blind to the live review, with the same last post.** The live monitor posts as the same login. While a mirrored
  review of a pull request runs, the tick keeps our last post on it as it was when the feed line came (`hold_ours`),
  the post's premises leave our own login out of the "new reviews and comments" count (`premises.new_activity`), and
  REVIEW_READY tells the verifier not to read our comments newer than the feed line. The facts and the review
  directory list our posts as the live monitor's ledger has them up to the feed line (its `actions.jsonl`, beside the
  feed), and our last post's text is the one GitHub holds under our login: the live monitor's post, the one the live
  review follows up on. The mirror takes the live monitor's reviews for its own (section 1, the post's shape): a live
  monitor whose line 1 gives a name has that name in the mirror's deploy settings file too
  (`features.review_automation.automated_reviewer_name`), and a review whose line 1 gives no name is the mirror's own
  whenever it was posted (`reviews.unnamed_are_ours_until` does not apply to a mirror).
- **The post.** `rk post N FILE [--premises SPEC]` runs its premises and rk post's guards as always (a refusal is a
  refusal), then writes the would-be post (`post.md`), the verifier's `notes.md`, the spec, the draft (`draft/`), the
  quality binding's code quality pass (`quality/`, for a review that got it) and `cost.json` into
  `<mirror.out>/pr<N>_<round>_<head10>/` (blank: `<WORKING_DIR>/mirror`; the feed line's round), marks the pair done
  and clears the review as a post does; the premises' finish (the save, the AFTER lines, the notes archive, the
  drafts' assessment, the cost record) runs as after a post.
- **cost.json**: the arm; whether the review got the quality binding's code quality pass (`quality_on_binding`); the
  drafting model's tokens in, out and reasoning, and requests, for the draft (per phase);
  Claude's from the transcripts (`rkmon_lib/transcripts.py`), counted as `rk mirror-report` counts the live side's:
  for each pass of the pull request from the review's start (verification, code quality, notice) and for the
  orchestrator session's calls from the start to the post, the tokens processed by kind (uncached, cache read,
  5-minute and 1-hour cache writes), the output, the dollars at the Claude Opus rates of config `prices` (the table
  `rk spend` reads), the input-equivalent tokens (ieq) and the effort each call ran at, with how many other mirrored
  reviews ran in the session's window; and the minutes of the draft, the checks, the code quality pass, the
  verification and the whole review. The session's and its subagents' prompt caches live
  `claude.prompt_cache_ttl` and `claude.subagent_prompt_cache_ttl`.
- **The checks.** With `mirror.checks` (the live monitor's workers folder, mounted read-only) a mirrored review's
  checks are the live review's for the same head, copied once its report is whole (a light job, no parallel build,
  within `mirror.checks_wait_min`): the same code gives the same result, so the two sides are compared on their
  reviewing, not on two builds. Blank, the mirror runs its own checks.
- **Drops.** Only reviews of one head on both sides are pairs. A feed line whose pull request moved or closed, a
  review dropped while it runs (its head moved), and a review written here whose live review was dropped (the pull
  request's head moved with no live post at the reviewed head in the live ledger) are dropped; a dropped written
  review gives its place to the next feed line. Each drop is a line of `<mirror.out>/drops.jsonl` with its reason.
- At `mirror.pairs` reviews written it takes no more feed lines and says so once (`MIRROR_DONE`). `rk review-status`
  shows the mirror's line.

On the live side, `rk post --premises` records each post in `state/review_cost.jsonl`: the pull request, round, head
and post, the draft's tokens and minutes, when the review started (its feed line) and when it was posted, and the
spec's optional `COST` (`{"verifier": N, "quality": N}`, the task notifications' totals, kept for reference).
`rk mirror-report --live-state DIR --mirror DIR --live-claude DIR` pairs the two sides by pull request and head (the
live post of the feed line the mirror took) and counts the live side's Claude tokens from its transcripts as the
mirror counts its own: the pull request's passes from the review's start to its post, and the monitor session's calls
in that window, which the report divides over the reviews that ran in it. It prints each pair's posts compared, the
dollars per pass and session, the drafting models' tokens, the minutes, the totals per arm with a day's dollars at the
live monitor's rate of reviews, the judge's worklist, the checks (a pass on the 1-hour cache, a pass with no
transcript, a review that got the quality binding's code quality pass with no record of it) and every lost review with
its reason.

### 3.11 The checks' scripted sections

Besides the merge, the builds, the goldens, the tests and the compile lanes, every review's checks run scripted
sections that read the pull request at its head against its merge base with its base branch (`base.ref`: develop, or
the feature branch it targets; `mkwt.py` merges that branch, RKMON_BASE). Each is time-boxed, writes one status line,
and its last line reads `<TAG> pass|fail|skip <detail>`; a tool that wrote none reads skip, never clean. checks.md's
second line is `base: <branch> @ <sha>`, the base and the commit every comparison is with; CI at head compares with
that branch's runs; and the differential golden first probes the tree's `run_diff.py --help` for the flags the CI
commands pass (a flag it lacks reads NOT RUN, and G5 waits):

| Section | Tool (by hand) | What it finds |
|---|---|---|
| PR clashes | `tools/clash_scan.py` (`rk clash N`; `rkmon_lib/clashes.py`, config `clashes`) | both land orders with every other open PR, and the silent clashes that merge clean: a name one removes that the other's new lines use, a C/C++ function whose parameter list one changes and the other calls with the old arity, a file one moves that the other includes; its ABI lane: another open PR claiming the same abi.h counter (`ABI NUMBER CLASH`, the second to land renumbers) |
| Overlapping work in other open PRs | `rk overlaps N --section` | the same new name or the same logic in another open PR, with the land order and any recorded owner; leads for the code quality pass |
| Mutation sweep | `tools/mutation_sweep.py` (`rk mutants N`) | each check the PR adds or changes, disabled once, against the PR's tests: a survivor is a check no test catches; a follow-up sweeps its push only and lists the earlier rounds' survivors still at its head (`<CA>/mutants`); each survivor runs again against every test file importing its module (a file failing unmutated is dropped, not the set), with the merged tree's engine, under the checks venv and then `paths.torch_python`: killed by the CI importers, `vacuous: torch-only path`, or `SURVIVED (CI importers, torch, engine)` |
| Since our last review | `tools/since_review.py` | a follow-up's range-diff and the PR's own files whose patch changed |
| CTest labels and tiers | `tools/ctest_labels.py` (`rk ctest-labels N`; config `ci_test_tiers`) | each changed test's CTest entries, tiers and the PR-CI lanes that run it |
| References to moved or deleted files | `tools/moved_refs.py` (`rk moved-refs N`) | lines still naming a file the PR moves or deletes |
| Docs against code | `tools/doc_refs.py` (`rk doc-refs N`) | doc lines naming what the PR removes or renames, and new names the reference docs lack |
| Engine import probe | `tools/import_probe.py` (`rk import-probe N`) | Python entry points that need an engine module only the library builds |
| Tests that import the changed modules | `tools/importer_tests.py` | the tests the PR does not change that import a module it changes, run on the merge (capped, time-boxed); `IMPORTERS_FAIL <file> (<module>)` names each one that fails on the merge only |
| Develop's tests changed since CI's merge | `tools/devtests.py` | the test files develop added or changed after the develop the PR's CI merged (the PR's `base.sha`, else the merge base), run on the merge and compared with develop (capped, time-boxed); `DEVTESTS_FAIL <file> (develop <sha>)` names each one that fails on the merge only and the develop commit that changed it |
| Superbuild tidy | `tools/tidy_check.py` (`rk werror N`) | after the native build, over its `compile_commands.json`: clang-tidy with the provider's `.clang-tidy` (a link the sparse tree cannot follow is read from the tree's commit, whose blob the monitor fetches before the check runs as PR code) and `clang++ -fsyntax-only -Wall -Wextra -Werror` over the TUs the PR touches, feature macros off and on; an error develop's copy of the TU has too is not counted (leads: the local clang may be newer than CI's) |
| Changed tests in the installed tree | `tools/installed_tests.py` | each changed test run in CI's `rocke_pytest` layout, with CI's environment, against develop's installed run |
| CI-condition run | `tools/ci_condition.py` (config `ci_absent_modules`; `tools/pytest_plugins/rkmon_ci.py`, `rkmon_norocm.py`) | each changed test file run as CI's test environment runs it: no torch, no built engines, ROCm's tools hidden; "0 of n run" and a failure only there are leads, a skip in a tier or contract test reads NOT RUN |
| Our open findings, re-probed | `tools/run_probes.py` | each saved probe of our open findings on the merge and on the head alone: REPRODUCES, FIXED or ERROR, with its exit code and tree |
| CI at head | `tools/ci_head.py` (config `ci_infra_signatures`, and the ones learned on this site, `site.ci_infra_signatures`) | per failing job: its first error, the failing tests, develop's latest run of the job and a known-infrastructure tag (the signature and its note, within its job scope and before its ending develop commit); failures hidden in earlier attempts, a job that passed only on a re-run tagged FLAKE (`tools/ci_attempts.py`) |
| ABI | `tools/abi_scan.py` | abi.h's counters, struct layouts and exported functions in public and generated headers, and abi.h against abi.py; on the merge, each counter's homes across all files (abi.h define and newest history line, abi.py's mirror, the highest layout canary): `HALF RENUMBER` when they disagree; a note or a half renumber is a check flag, and a note is what G9 waits on |
| Peer checks | `tools/peer_checks.py` (`rk peer-checks N`; `rkmon_lib/peer_checks.py`, config `tracker_allow`), one section each | what peer reviewers catch. Flags: unanswered review threads, tracker IDs in added lines, `ROCKE_*` env reads the env docs lack, new library tests in no CMake or CTest list, C/C++ `getenv` in added lines, a title that is not `type(scope): subject`; split tautologies, bare sibling test imports, workload sizes in kernel names, pull-request references in added lines, measured timings in source comments, new platform-to-library back-edges, kernarg or signature changes with the ABI string kept, fixed arrays written past their slots, NaN-blind tolerance tests, optional imports with no importorskip, repeated numbered headings. Leads: request fields forced to `bool`, torch device queries and unconfigured `file(GLOB)` in library code; a shared lowerer changed with no parity pair; an unpacked name never read |
| Rules, the PR body, auto-merge | `rule_scan.py`, `tools/automerge_msg.py` | tracker IDs, absolute performance numbers and environment reads in added lines; measured figures under a named key or column in `.json`, `.jsonl` and `.md` files (`experiments/` included); PR references (`#NNNN` in Markdown, `PR NNNN`, `/pull/N`) and dated narratives in any committed file; performance numbers in the PR body (a squash merge commits it); an armed auto-merge's stored message against the live one |

The clash, mutation, since, labels, moved-file, docs, import-probe, importer, develop-tests, CI-condition, tidy and
peer-lead sections are leads for the verifier, never a merge-only head's flag; dropping a lead takes a run that
disproves it. The ABI scan and the peer flags are check flags like the others. The checks' native build of the merged
tree is the review's shared build (`checks/shared_build`), which the verification pass and the code quality pass reuse
read-only. The test runs get cmake and PyYAML when `paths.cmake` and `paths.pyyaml` name them; a skip for a tool CI's
lanes have and this run lacks reads `NOT RUN: <tool> missing`, never a pass (`tools/pytest_plugins/rkmon_skips.py`),
and every skipped test is listed with its reason.

The overlapping work section (`rkmon_lib/overlaps.py`) reads, each full tick, the definitions each open PR's head adds
against its merge base: Python functions, classes, methods and module-level constants; C and C++ functions and
declarations, macros, types and constants; CMake test names. It groups them across PRs by name, and by a sketch of their
bodies, so a copy under another name is found too (6 lines or more, 85% similar or more). A name develop already defines
is not new work, and in the same file it is an edit. A copy of a pattern develop already repeats is a lead for the code
quality pass (at most 5 a PR), not a question between PRs. A tick reads at most `overlaps.max_heads` new heads in
`overlaps.budget_s` seconds and caches each head's definitions, with no GitHub API call.

One test's CI lane, on demand: `rk lane-of <test file | node id | CTest entry>` (`tools/lane_of.py`; config
`ci_test_tiers`, `ci_absent_modules`) reads it from the tree at develop's head, a PR's head (`--pr N`) or a commit: its
CTest entries and tiers, the lanes that select them, the entry's `-m` markers, module-level skips and imports of modules
CI lacks; `--collect` also collects it as CI would (`tools/pytest_plugins/rkmon_ci.py`). One line a target: `CI: runs
(<entry>, <tier>)`, `CI: skips(<reason>)`, `CI: no lane(<why>)` or `CI: errors(<why>)`.

A merge-only head (§6.2) is carried with nothing posted when its checks are clean, or when its only flag is CI failures
that are all known infrastructure (a configured log signature, or the same first error on develop or on other open PRs
today: `rkmon_lib/known_infra.py`, a CI_INFRA line). Its threads and title flags count only for items our review did not
see (`rkmon_lib/peer_carry.py`): items our review saw carry the head; new ones make a micro round, the verification
pass alone on them with no code quality pass, carrying every other row of our last review. A carried head keeps our
verdict; only a post changes it.

### 3.12 The orchestrator's steps, large reviews, drift and review spend

- **One call per step** (`rkmon_lib/dispatch.py`, `rkmon_lib/spec.py`):
  - `rk dispatch-pack N [--round R]` runs `rk review-prep N` (the review directory, `<paths.tmp>/v<N><round>`, with
    `facts.txt`, `classify.txt`, `overlap_section.md` and `files.txt`), makes the pull request's worktrees again when
    they are gone, and prints compactly what the briefs need, each section once: the head and state, our last review's
    flagged rows and findings, the unresolved threads and the operator's hand-written comments, what changed since our
    last review (the kind of change, the pull request's own changed files, its own non-merge commits since our reviewed
    head, and the merges), the checks' flags, leads and shared build, CI's FAIL and RUN lines (`rk ci-fails`), the
    overlap owner decisions that name the pull request, a micro round's items, and last whether the round gets a code
    quality pass (`lane: needed|exempt (<why>)`).
  - `rk spec N --round R` writes the post's spec (by default `spec<N><R>.json` in `paths.tmp`) from the review directory
    and live reads. From the verifier's `notes.md`: its `premises:`, `FAILOK`, `CI pending:` and `conflict set` lines,
    and its `decide:` lines, each an `AFTER` line (`rk overlap-decide`) that runs once the post is saved. From the
    verified post: its header's light (unless `--light`) and its Bots / CI row, whose ✓ count the post fills in with the
    live count; a row with no count gets one appended, the one change `rk spec` makes to the post. Live: the head, the
    base branch, the body's ticket tag `TAG` (a key of the repository profile's or the configuration's `tracker_keys`,
    else any Jira-style key), CI as `rk ci-fails` reads it (`FAILOK`, `RUNOK`), the conflict set (`git merge-tree` of
    the head with the base branch's head) and the threads. What the notes and the live reads disagree on is a `CHECK:`
    line, and the spec keeps the notes' side, so the post stops on it: a failure the notes do not account for, a run
    they did not see, more cancels, other threads, another conflict set or another head. Exit 0 with no `CHECK:` line, 3
    with one, 4 when the spec cannot be made.
  - `rk spec-post N --round R` runs `rk spec` and, only when it wrote no `CHECK:` line,
    `rk post N FILE --premises SPEC`. It takes `rk spec`'s options, and `rk post`'s `--again` (a re-review: a second
    review at a head we already reviewed) and `--dry-run` (`rk post`'s guards, then no post). With `--devbase-recheck`,
    a post that stops only because develop changed some of the PR's own files prints develop's diff stat for them and
    re-runs every other premise at today's develop: nothing is posted. When the change leaves the verified result
    standing, set `develop <sha>` (or `base <branch> <sha>`) in `notes.md`'s `premises:` line and run `rk spec-post`
    again (it writes the spec from the notes), or set the written spec's `DEVBASE` and post with
    `rk post N FILE --premises SPEC`. A `FAILOK` or `RUNOK` of `none`, `(none)`, `-` or nothing names no job.
  - `rk wait-quality DIR [--max-min M] [--slice-min S]` blocks the verifier until the code quality pass's `quality.md`
    is final (it exists, is newer than `facts.txt`, and its Code quality row is not `pending`): `ready`. Each call
    returns after at most S minutes (4.5 by default: `waiting`), so a call stays under the Bash tool's 10-minute limit
    and the agent's prompt cache stays warm; the wait, counted from the first call, ends after M minutes (20 by default:
    `timeout`).
- **A review too large for one verifier** (REVIEW_MANUAL, over
  `features.review_automation.large_prs.in_depth_review_over_lines`):
  `rk review-split N [--max-lines L] [--max-agents K]` (`rkmon_lib/split.py`) cuts the pull request's own diff into
  parts, filling each up to L changed lines area by area, keeping files that change together in one part (a C++ mirror
  and its Python twin, a header and its source, a test and its module, a list and the files it names), and summarizes
  generated, golden and data files rather than reading them. L defaults to
  `features.review_automation.large_prs.split_part_lines` (8000 by default). The split writes only in `parts/`:
  `part<k>.md` for each part (quoting the checklist lines of every area in it from the verifier's team file,
  `paths.verifier_team_file`, and the code quality checks from the code quality reviewer's team file beside it) and
  `part<k>.files` (its paths), `cross.md` and `plan.json`. It prints the dispatch plan: one verification pass per part,
  which runs the part's code quality checks too, in waves of at most K agents (20 by default), then one cross-cutting
  verification pass that checks what spans parts and writes the single post. It gives each agent the description and
  the brief to dispatch it with, which name the pull request and the round, so the orchestrator knows the review has
  its verifiers and does not hand it back to the session while they run. A follow-up's unchanged parts get no
  verification pass: the cross pass carries their prior rows. The cost scales with the pull request: k parts take
  k + 1 agents, each about one review's cost.
- **Drift, like with like** (`rkmon_lib/drift.py`): whenever develop's rocKE tree moves past the tree a pull request's
  tests last ran on, its test files run on its head merged with develop and on develop alone; a failure on the merge
  only fails G6 and prints `DRIFT_FAIL`. Develop, the merge and the head are the same extraction (config
  `drift_extract`); a test that also fails on the head alone is the head's own, never drift;
  `rk drift-dismiss N D --reason TEXT` records a harness false positive, and G6 and the merge blockers stop counting
  that run. One drift job runs at a time, with up to `drift_batch` PRs in merge-plan order, also while the queue is
  frozen.
- **Review spend** (`rkmon_lib/spend.py`, `rk spend`): what each review cost, posted, kept unposted or closed without
  a post (`rk review-close`), by part (the verification pass, a second verification, the code quality pass,
  orchestration), from the Claude transcripts (`paths.claude_projects`, else this machine's), the post ledger and,
  for a review kept unposted or closed, its line in `state/review_cost.jsonl`, priced with config `prices`;
  unattributed runs and turns are overhead, and heads carried as covered cost nothing. Per UTC day and ISO week it
  shows the reviews, the covered heads, the reviews' cost, the overhead and the total, and the mean, median, p90,
  minimum and maximum per review; for the selection (`--day`, `--week` or `--since`), its outliers (above Q3 + 1.5 x
  IQR) and a breakdown by part, kind, size or pull request (`--by`, by part by default). `--experiments` adds the
  experiments' and other tasks' spend, apart; `--write` also writes `spend_<day>.md` for each complete day of the
  selection and `spend_<week>.md` for its weeks in `paths.spend_dir`. Each run refreshes the dashboards' spend card.
  Without a price table it is off; `rk usage` stays in tokens either way.

**`rk status` and `rk doctor`.** `rk status` says how the monitor operates right now, from local files only: on or off,
stopped or draining, the pieces running (the review driver, the tick loop and the last tick, the admin dashboard's
service, the orchestrator, the gate, develop sanity), the parallel reviews and the parallel builds in effect, each
against its range with the scaler's reason and last changes, the review queue, the work in flight, the RAM disk's
use, then the summary from stored state and the GitHub logins' last check. `rk doctor` says
whether the monitor is set up right and reaches everything, each check PASS, FAIL or WARN with its fix: the machine's
prerequisites, every configured value (a secret only as set or missing, with its path), and read-only checks of each
configured service (GitHub, JIRA, the team chat, the model gateways, the GPU cluster, the Claude session's
settings and the RAM disk; in the container also the Claude session's key, with the claude login's test call). It keeps
the services' results a few minutes in `state/doctor_cache.json`; `rk doctor --fresh` checks again. `rk status` writes
nothing, and `rk doctor` only that cache.

## 4. Setup and tools

The wrapper (`docker/rkmon`) and the `rk` command set the deployment up and run every script; the scripts live in the
package, and no source is reproduced here.

| To | Use |
|---|---|
| Build its image, on any machine | the steps of [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md) |
| Deploy it in a container from that image, on a machine that builds nothing | the steps of [QUICK_DEPLOY_GUIDE.md](../QUICK_DEPLOY_GUIDE.md) |
| Change a setting | [QUICK_USER_GUIDE.md: Change a setting](../QUICK_USER_GUIDE.md#change-a-setting) |
| Use another model for a pass | [QUICK_USER_GUIDE.md: Use another model for a pass](../QUICK_USER_GUIDE.md#use-another-model-for-a-pass) |
| List every command | `rk --help` |
| Check that it is set up right and reaches everything, read-only | `rk doctor` |
| See every feature and every value in use (the setup summary) | `rk features` |
| See how it operates right now: the state, the pieces running, the settings against their ranges, the reviews | `rk status` |
| See the first review pass, the bindings, the drafts and the scaler's decisions | `rk drafts` |
| Read about the review tools in `worker/` | [worker/README.md](../worker/README.md) |
| Look up a setting, its default and what it does | The README's [Settings reference](../README.md#settings-reference) (the deploy and build settings), [docs/internal_settings.md](internal_settings.md) (the internal ones) |
| Publish the package into its repository | [docs/maintaining.md: Publishing the package](maintaining.md#publishing-the-package) |

Every start sets the work volume up, the agentic files included (`rk provision`;
[agentic/README.md](../agentic/README.md)).

Environment facts the monitor cannot fix for you:

- **The container.** One image holds the toolkit, its venv, gh, Claude Code and, when built so, ROCm; the
  configuration and state live in the work volume (`/work`), the monitor's secrets in `/secrets`, and the writer's
  (gh's write login, the atlassian login, the Teams webhook, the web host's key) in `/secrets-write`, which only the
  writer reads. The orchestrator is one long-lived Claude session (`rk orchestrate`) that runs the ticks and keeps the
  review driver running. The container has no GPU, so GPU-only cases skip there unless remote GPU tests are on
  (section 3.7), and the RAM disk is a tmpfs the compose override mounts, sized by `host.resources.ramdisk_gib`
  (section 3.5).
- **Ollama on WSL.** Ollama's installer skips ROCm on WSL, because no GPU is visible on PCI there. Install the
  matching ROCm build from the same Ollama release, check its checksum, and move the bundled `libhsa-runtime64.so*`
  aside so that the system's WSL HSA runtime is used. Repeat both steps after every Ollama upgrade.
- **Keep the GPU driver current.** A GPU wedge on an outdated driver can escalate into a host crash (section 9,
  issue 1).
## 5. The local review loop in detail

This section is the local review loop (`worker/review_agent.py`), the draft stage `local-worker` (section 3.9). The
tool loop on another binding is section 3.9; with no first review pass, neither runs.

### 5.1 Choosing a local model

**The bake-off.** Every candidate got the same prompt on known-answer tasks:

- T1: tests a PR skipped;
- T2: a subtle claim about active lanes in an `__all` reduction;
- T3: bool truthiness on an ABI field;
- three or four clean controls, where the right answer is that nothing is wrong.

| Model (open-weight tag) | Catches and false alarms | Time per task set, against the chosen model | Verdict |
|---|---|---|---|
| `qwen3.6:35b-a3b-coding-mtp-q4_K_M` (mixture of experts) | T1; one false orange (T3) | — | **Chosen**: the best balance of speed and quality |
| `qwen3.8:27b` (dense) | T1; 2 false alarms | About 3 to 6 times longer; needed `num_batch` 32 | Precise but slow |
| `qwen3.6:27b-coding` (dense) | T1; 1 false red | Longer still; `num_batch` 32 | Slow |
| `gpt-oss:120b` | Medium effort: T1 as a nit, 3 false oranges, an invented C++ file. High effort: missed T1 | Similar | Rejected |
| `qwen3-coder-next` (no thinking) | T1 and part of T3, but a false red or orange on every control | Shorter | Rejected |
| `gemma4:26b-a4b`, `gpt-oss:20b` | Ran out of thinking; no catches | — | Rejected |

No local model caught T2, and none matched Claude's quality. That is why a first review pass only gives leads.

**Choosing your own.**

- Run the same kind of bake-off: one fixed prompt, known-answer tasks from your own PR history, and clean controls.
  Count the catches and the false alarms, and compare the time each candidate takes on the same task set.
- The model must pass the start's chat and tool-call probes (section 3.1). Passing them shows that it can drive the
  loop, not that it reviews well.
- The prompt budget (section 5.3) follows the model's context (`worker.context`): a model with a smaller context gets a
  smaller prompt, with more of a large diff left to its tools.

### 5.2 Inference settings

Every setting is a key of the internal `worker` settings. The deploy settings' model binding `worker` sets the model,
its address and its key, and of the keys below `backend` (from its `provider`), `stall_s` (from its
`stall_timeout_seconds`), and `max_parallel`, `rpm` and `tpm` (from its `limits`);
[docs/internal_settings.md](internal_settings.md) gives the others' defaults.

| Setting | What it does | Why |
|---|---|---|
| `backend` | `ollama` (Ollama's native `/api/chat`, streamed JSON lines) or `openai` (`/v1/chat/completions`, streamed server-sent events) | The native API carries `num_ctx` and `num_batch` per request; the OpenAI-compatible one cannot |
| `context` | The context to ask for, in tokens (Ollama's `num_ctx`) | Review prompts are large (section 5.3). An OpenAI-compatible server keeps its own |
| `num_batch` | Ollama only: the prompt batch size | Ladder-tested: one large prompt at the full context for each batch size in a rising ladder, with the GPU watched; use the largest that stays clean, or one step below it. Ollama's default batch at the full context wedged the GPU |
| `think` | Reasoning effort: `high`, `medium`, `low` or `false` | The first review pass's effort sets it (`features.review_automation.passes.first_review_pass.effort`, when it is high, medium or low). Unset, as on the pass's `default`, the request carries none and the model takes its own default, which line 1 says (`effort: the model's default`). An Ollama server gets it as it is, and line 1 names it (`false`: `effort: none`). An OpenAI-compatible server gets `reasoning_effort` for high, medium or low, which line 1 names; with any other, line 1 says `effort: the model's default` |
| `temperature`, `max_output_tokens` | Sampling, and the longest answer in one turn | — |
| `keep_alive` | Ollama only: how long the model stays loaded after a request | It unloads soon after a draft, which frees the memory |
| `max_steps` | The most tool-calling rounds in one draft | Then one forced final answer (section 5.4) |
| `stall_s`, `max_resends` | A streamed request that sends nothing for `stall_s` is abandoned and sent again, up to `max_resends` attempts | A stalled request would cost the whole draft |
| `hang_s`, `max_hangs` | Local Ollama only: a runner that took the prompt and logged no progress in Ollama's journal for `hang_s` is hung; the model is unloaded (`ollama stop`) and the request sent again, up to `max_hangs` times | About one fresh model load in eight hangs on its first prompt batch; without it, each hang costs the full stall timeout |
| `max_parallel`, `rpm`, `tpm` | What the endpoint allows: drafts at once, and requests and tokens per minute | The parallel reviews and the shared budget (section 3.3) |

### 5.3 What the local review loop sees

`fetch_pr.py` writes these files with GETs only under `workers/pr<N>/pr/`, and the loop stages them into the user
message in this order:

1. `meta.md`: the title, author, state, mergeability, head and base SHAs, size, labels and description.
2. `reviews.md`: reviews from **other** reviewers only, with the commit each one saw.
3. `threads.md`: the inline review threads, marked resolved, open or outdated, **without our own comments**.
4. `comments.md`: conversation comments from others.
5. `commits.md`, `files.txt` and `checks.txt`.
6. `diff.patch`: the full diff, with two reductions:
   - **Collapse:** golden-hash JSON hunks, and large generated data (`.json`, `.yaml`, `.csv`, `.txt` or `.kdp` files
     with 300 or more changed lines, or an extension changed in 50 or more files), become a one-line summary that
     names the most-changed keys.
   - **Budget:** the whole prompt (the book and the context) is capped at about three characters for each token of the
     model's context, and never more than 650,000 characters (the default context's cap); each review, thread or
     comment section at 8,000. The diff gets the rest, never less than a third of the cap (at most 150,000). Over
     budget, every file keeps its header, hunks are added code files first while they fit, and the rest are marked.

The review book (`<REVIEW_BOOK>`) is appended to the **system** prompt. When GitHub refuses a very large diff, the
fetch rebuilds it from the monitor's clone.

### 5.4 The prompts (verbatim; keep them unchanged)

**System prompt**, followed by `===== REVIEW BOOK =====` and the book:

> You are an INDEPENDENT first-round code reviewer. Read the review book below and follow it: its workflow, repo
> rules, judgement standards, and especially its section 5 output format.
>
> You have a fixed set of tools (read_file, grep_code, run_pytest, rule_scan, merge_conflicts) that work on the PR's
> head worktree on the review machine (tests that need a GPU skip when it has none, each skip printed with its reason).
> USE them to verify things yourself — run the relevant tests, grep the code, check merge conflicts — do not just trust
> the diff or other reviewers. REQUIRED before you conclude: if the PR adds or changes any test, use run_pytest to
> actually run the most relevant changed test file(s), and base your 'Our checks' row on what the run printed (pass/fail
> counts). Always run rule_scan and merge_conflicts at least once. Never state a test outcome, coverage claim, or merge
> result you did not produce with a tool; if you did not run it, write 'not run'. Other reviewers' comments are provided
> ONLY so you can check whether they were properly addressed, not to copy. Form your OWN findings and NEVER claim
> evidence you did not produce with a tool.
>
> When done, reply with ONLY the review in the book's section 5 Markdown format and no tool call.

**User prompt**, followed by the staged context:

> Next review is: https://github.com/<ROCKE_REPO>/pull/<N>/changes. Please consider the existing comments from other
> reviewers and whether they've been fixed. We want our own findings; we don't want to duplicate existing reporting.
> HOWEVER, I want to also make sure previous comments have been addressed properly. I also want a judgement call on
> the utility of the PR and whether the changes are positive.

**Forced final turn.** If the loop ends without a review (the model spent its budget thinking, or reached
`max_steps`), this is sent with thinking off. The tools stay in the request, because dropping them made the runner
process the whole conversation again; a tool call asked for anyway is refused, once.

> Stop investigating. Output ONLY your first-round review now, in the review book's section 5 Markdown format
> (prior-items table ending with Bots/CI and Our checks, net-new findings table or 'Net-new: none.', utility line,
> legend). No preamble, no further tool calls.

### 5.5 The fixed tool menu

The model never supplies a shell string: each tool builds its argv from validated parameters, with `shell=False`.

| Tool | What it does | Safety |
|---|---|---|
| `read_file(path)` | Reads a file from `wt<N>`: the first 8,000 characters | The path must match `^[A-Za-z0-9_][A-Za-z0-9_./-]*$`, contain no `..`, not be absolute, and resolve inside the worktree |
| `grep_code(pattern, path?)` | `grep -rnI` in `wt<N>`; each unescaped `\|` alternative becomes its own `-e` pattern | A pattern of at most 200 characters; a validated path; a 60-second timeout |
| `run_pytest(paths)` | Runs the repository's own `.py` test files from `wt<N>` with the checks venv, on the review machine (a test that needs a GPU skips when it has none) | Existing `.py` files in the worktree only; a 1,200-second timeout; `-p no:cacheprovider -o addopts=` |
| `rule_scan()` | `rule_scan.py` on the PR's diff: tracker IDs, PR numbers in code, absolute performance numbers, kernel-selecting environment reads, text I/O without an encoding, measured figures in data files, PR references and dated narratives in committed files | — |
| `merge_conflicts()` | Lists the unmerged files in `wt<N>m` | — |

- Every tool's output is cut to about 6,000 characters (its head and its tail).
- The tools' environment drops `GH_TOKEN` and `GITHUB_TOKEN`, sets `ROCKE_LLVM_FLAVOR=llvm22` when it is unset, and
  sets `TMPDIR` to `paths.tmp`.
- `run_pytest` runs from rocKE's folder, with the platform and library trees and the monitor's pytest plugin on
  `PYTHONPATH`, plus the pybind module when one is built for the worktree.
- None of the tools writes a file, reaches the network or posts anything. The tests `run_pytest` runs are PR code:
  in the container they run as a PR runner, in a slot of its own, which reads no secret (section 7.2).

### 5.6 Outputs

| File | Contents |
|---|---|
| `review_agent.md` | The draft |
| `agent_log.txt` | Every step: the length of the thinking, the tool calls, and the first 300 characters of each result |
| `stats_agent.json` | `steps`, `tool_runs`, `wall_s`, `num_batch`, `final_chars` |
| `tokens_total.json` | The local model's tokens in all, per pass (each request has its line in `<RUN_DIR>/tokens.jsonl`, with its pass and step) |

With the review offload (section 3.8), the offload passes write `draft.md`, `quality.md`, `regrade.md`,
`post_draft.md`, `notes_draft.md`, `evidence.jsonl`, `runs.jsonl`, `offload_run.json` (the commits and each pass's
status), `agent_log_<pass>.txt` and `evidence_check.md` / `.json`, in `workers/pr<N>` with `offload.mode` `on`, or in
its `offload/` with `shadow`.

### 5.7 Typical behaviour

From a sample of 58 drafts by a local model:

- **Effort:** a median of 8 tool calls over about 4.5 steps; some drafts make only 2 or 3 calls.
- **Format:** it follows the format well, but it:
  - writes ACCEPT / REJECT headers, which the book does not use;
  - copies the book's example text ("Clean merge with develop (+17)");
  - appends a made-up "Saved to ..." line;
  - claims "(ran)" for things it did not run;
  - reports "all CI green" while CI is still running;
  - credits fixes to the wrong commit;
  - restates other reviewers' items as net-new;
  - calls CI failures the PR caused "unrelated".
- **Genuine catches are occasional:** about 3 in one batch of 29 PRs, for example an absolute timing in a docstring,
  synthetic timing figures in test comments, and a dead parameter. Credit them when you keep them.
- **Kept versus trimmed:** in most batches the verifier keeps 0 or 1 of a draft's findings per PR. Its own review and
  the checks produce nearly all the posted findings.

## 6. Procedures

The procedures live in the skill, which `rk provision` renders from the package's `agentic/` templates and installs
into `<CLAUDE_WORKSPACE>/.claude/`. When this page and those files disagree, the files win.

| Procedure | File |
|---|---|
| The per-tick procedure, the short form of the rules, the reply format | `skills/rocke-monitor/SKILL.md` |
| Every standing rule in full | `skills/rocke-monitor/rules.md` |
| Every subcommand of `/rocke-monitor` | `skills/rocke-monitor/commands.md` |
| Picking, verifying and posting reviews; follow-ups; merge-only carry and micro rounds; split reviews; CI updates; overlap notices | `skills/rocke-monitor/review_procedure.md` |
| The read-only `rk` commands the review passes use | `skills/rocke-monitor/lane_commands.md` |
| The review offload's audit | `skills/rocke-monitor/offload.md` |
| The merge queue, the merge plan, holds, ABI lanes | `skills/rocke-monitor/merge_procedure.md` |
| The develop guard, incidents, develop issues | `skills/rocke-monitor/incident_procedure.md` |
| The weekly report and the lessons review | `skills/rocke-monitor/weekly.md` |
| What the pipeline has learned, by reader, and the per-role extract | `skills/rocke-monitor/lessons.md`, `lessons_for.py` |
| The verification pass's and the code quality pass's instructions, and the post format | `team_members/rocke_review_verifier.md`, `rocke_code_quality_reviewer.md` |
| The review passes' agent types: model, tools, settings | `agents/rocke-review-verifier.md`, `rocke-code-quality-reviewer.md` |
| The local review loop's rubric | `<REVIEW_BOOK>` |

### 6.1 What "verified" means

Each claim in a post is backed by a run at this head ("ran") or a code read ("read"). The checks job automates much of
the battery below; the verifier picks what the PR calls for. The bold items apply to nearly every PR.

1. **Merge simulation and conflicts:** `wt<N>m`, GitHub's mergeable state, and the develop commits since the merge
   base that touch the same files (the semantic-conflict candidates). For a PR that conflicts, compare its head with
   its merge base.
2. **Prior items:** for each one, locate the code at head, find the fixing commit, and prove the fix with a test, a
   small probe, or a grep of every sibling site; then judge whether the item itself is valid. Never trust "resolved"
   or "outdated" flags, author replies or commit titles.
3. **Native build and ctest on the merge.**
4. **Targeted tests on the merge against develop:** the test files the PR touches and those covering the touched
   area, on both trees, with the failure sets compared (`faildiff2.py`).
5. **Engines and IR:** byte identity of the C++ and Python engines at each emission flavor; the representative golden
   at every flavor, read line by line (a load error looks like drift in its first line); family goldens; a
   differential sweep for PRs that change IR; C++ parity tests, which need the pybind module CI never builds.
6. Behavioural differentials: for dispatch, selection or policy changes, run the same request grid through develop
   and through the merge in separate processes, and compare the kernel name, spec hash, compile key, grid, block and
   signature.
7. GPU runs: the repository's GPU tests on a GPU (section 3.7); probes that show the bug on develop and the fix on
   the merge; bots' numeric claims reproduced on hardware.
8. ABI consumers: when a kernel ABI changes, check the consumers outside rocKE in a checkout that includes
   `projects/hipdnn`: hipDNN's kernel-ingestor packs (`src/engines/kernel_ingestor_engine/packs/*Native.cpp`),
   `descriptor-packaging/tests/rocke/test_hkp_pack_rocke.py` and `library/tests/test_sdpa_reference_contract.py`.
9. **CI enrollment:** is each new or moved test file in the hand-written lists in `platform/CMakeLists.txt`, and each
   new lane in the tier YAML? Does a test need torch, the pybind module or `llvm-objdump`, which CI lacks? Confirm it
   in the CI log at head.
10. **Installed tree:** the checks run every changed test in the `cmake --install` layout that CI's `rocke_pytest`
    uses, and compare the failures with develop's installed run. Not covered: a change to test helpers or
    `conftest.py` alone, and install-rule changes without a test change; check those by hand.
11. **Repository rules:** `rule_scan.py` on the diff, a grep of added lines for links, then the rest of the
    repository's review rules (`.github/instructions/rocke-pr-review.instructions.md`) by reading.
12. Other open PRs: the merge plan merges each pair that shares a file on today's develop, and the checks list the
    overlapping work. For one specific pair, a three-way scratch merge (develop, this PR, the other PR) is the proof.
13. Bots' claims: reproduce them with small stub harnesses or pytest plugins, rather than reasoning about them.
14. Semantic-merge scans for older PRs: removed or renamed APIs and fields, moved-versus-edited code, goldens that
    need the same re-bless, accidental reverts of develop files, validators that must cover new develop atoms.
15. Build every new variant on the merge when develop rewrote the area: a textually clean merge once left a stale name.

### 6.2 Follow-ups

1. Pin the head we reviewed before the re-fetch moves the PR's ref, so its commits stay reachable.
2. `git range-diff` the old head against the new one: `=` unchanged, `!` rewritten (read the inner diff), `>` new.
   Verify every changed hunk; mark unchanged commits "still fixed (range-diff)".
3. Re-verify every earlier item at the new head, ours and others'. Our earlier review is the checklist. Authors sometimes
   cite SHAs from before a rebase; map them to the new commits.
4. Note the approvals given before the new head. A stale approval still counts on develop, so never ask for a
   re-approval (a held PR is the exception: it needs the operator's explicit re-approval).
5. Re-run the merge simulation and the relevant checks: an earlier green can turn orange or red from merge semantics
   alone.
6. The header says "Follow-up"; first-round files are never overwritten.
7. Our earlier follow-up items (F rows) are not raised again: they sit in one Follow-ups row, each with its state (a
   ticket key once filed, FIXED, DEFERRED, or awaiting the author). An item the author declined with a reason reads
   DEFERRED (author: out of scope), unless a real break overrides it.

Not every push gets a follow-up. A rebase-only push records the new head as equivalent; a merge-only push whose checks
are clean is carried with nothing posted, and one whose only new items are threads or a title problem gets a micro
round (section 3.11). A follow-up waits `features.review_automation.schedule.follow_up_cooldown_minutes` after our
last review of the pull request, posted or kept unposted, except a develop fix's and the operator's priorities
(`followup_gap_exempt`); a CI update starts no gap.

### 6.3 The post

The verifier's team member file holds the exact format. In short:

1. Line 1 gives the automated reviewer name and the model that drafted the review: `review_first_line` (for the
   local review loop `Automated review by <automated_reviewer_name>: Model - <WORKER_MODEL>, effort: <effort>`, the
   effort its server runs at: `think` in section 5.2), or, with no first review pass, the verifier
   (`review_first_line_no_draft`). Line 2 is blank.
2. The header: `## PR #<N> — <title> (<author>, <JIRA_PROJECT>-NNNN) — <light> First-round: <2–5 word reason>`, or
   `Follow-up:` in place of `First-round:`. The light is the worst open Needed-to-merge item.
3. The prior-items table: one row per prior item (ours, other people's, Copilot's and relevant bots'), in order, ending
   with **Bots / CI** (what ran, where, at which head, with the count of passing runs), **Our checks** (each check
   with its result, and "(ran)" or "(read)"), **Code quality** (from the code quality pass) and, once an earlier post of
   ours listed follow-ups, **Follow-ups**. Cells are point form: the light and the status word, then one to three short
   points.
4. **Needed to merge**: every open blocker, each as a full row with its light (no cap), numbered N1, N2 and so on,
   each with the problem, its impact and the fix: wrong results; a build, CI or ABI break the PR causes; a regression
   in code it touches; a test it adds that cannot fail; a claim in its body or commits that is false at head. With
   none: `**Needed to merge:** none.`
5. **Highly recommended follow-up (not blocking)**: this round's new significant items, numbered on from earlier
   rounds as F1, F2 and so on, each with an orange or yellow light for its importance, what it gains and the fix; no
   nits, style or wording. Left out when there are none.
6. A utility line, the legend, and an italic footer that names the first review pass (when one drafted it) and the
   verification by Claude, lists the checks, and says the post flags whether a closer look is needed; it is not an
   approval or a rejection.

Each F item of a posted review is tracked in `state/followups/tickets_<N>.md` until it is settled: fixed in the PR,
declined by the author with a reason (DEFERRED, no ticket), or, when the author agrees or does not answer, a follow-up
ticket for the PR's author (with `features.ticket_creation.follow_up_tickets.enabled`).

## 7. Guardrails

### 7.1 GitHub and authority

| # | Guardrail | How it is enforced |
|---|---|---|
| 1 | **Comment only.** Never approve, dismiss a review, merge directly, resolve threads, edit or delete other people's comments, or push. The only review state is a change request on a confirmed hold, with the merge queue live and `features.merge_queue.auto_hold.enabled` | The guarded `rk` commands are the only write path; reads go through a GitHub module that refuses write methods and GraphQL mutations; the standing rules (`rules.md`) |
| 2 | Never touch the operator's existing review states | They are reported to the operator, who manages them |
| 3 | Line 1 of every post gives the automated reviewer name and the model that drafted it (`review_first_line`), or, with no first review pass, the verifier (`review_first_line_no_draft`) | `rk post` checks line 1; `rk post --premises` also stops a footer that credits a first pass when none drafted |
| 4 | Every post ends with a footer that names the verification and says the post is not an approval or a rejection | The post format in the verifier's team member file |
| 5 | GitHub credentials serve only GET requests and the guarded writes | Each write is one `rk` call under the write lock, checked against the ledger first, recorded in it, and held to hourly limits (`max_posts_per_hour` for reviews) |
| 6 | A first review pass has no GitHub access | No GitHub tool; `GH_TOKEN` and `GITHUB_TOKEN` dropped from its tools' environment; in the container the PR code its tools run runs as a PR runner, which reads no login (section 7.2) |
| 7 | Every outward action belongs to a feature, and writes to GitHub only once the operator sets it live | `features` in the deploy settings (review automation and the merge queue start in prototype); a command whose capability is off says `OFF` and does nothing |
| 8 | Verification, gate decisions, overlap owner decisions and posts stay with Claude | The verification pass, the code quality pass and the orchestrator take Claude models only; no setting hands them to another model |
| 9 | Overlap notices only after an owner decision is recorded: one per PR and group, naming both copies, the owner, the landing order and the rebase | `rk notice --kind overlap` refuses a key already posted (the ledger); `rk overlap-decide` refuses an owner that is not a member |
| 10 | Follow-up tickets only for significant follow-up items the author agreed to or left unanswered, and for non-trivial items still open on develop after a merge; deduplicated, assigned to the PR's author and linked to the PR and our comment; never for nits or items the author declined with a reason; never for CI runner infrastructure, which another team owns | The standing rules. Follow-up tickets are filed by the orchestrator, through the atlassian CLI; a found issue's proposed ticket waits on the admin dashboard for the operator's Approve. Never a first review pass, the review passes (read tools only) or the toolkit on its own |
| 11 | Auto-merge is turned off only by the auto-merge guard or on the operator's explicit instruction, and armed only for PRs the operator cleared; never a direct merge, admin rights or a force | `rk automerge-off` (once per head, ledgered) and `rk arm`, which re-checks every condition; the merge queue's Auto-merge and Auto-merge guard (live) |

### 7.2 Containment of PR code and the first review pass

- The model never supplies a shell string. All five tools build their argv from validated parameters with
  `shell=False`; a path must be relative, contain no `..`, match a safe character set, and resolve inside the PR's
  worktree.
- The tools can read the worktree, grep it, run the repository's own pytest files, run the rule scan, and list merge
  conflicts. None of them writes a file, reaches the network or posts anything.
- The tool loop's tools (`worker/review_tools.py`, section 3.9) are a fixed, read-only menu built the same way:
  git, grep, file reads and the rule scan, with no test or build. The offload passes' own probes and mutants run only in
  the sandbox (section 3.8).
- **PR code's runners.** In the container every run of PR code goes through one launcher, `docker/rk_prcode.py` (root's:
  one sudo rule lets the monitor's user run it as a PR runner, and nothing else), and runs as a PR runner (`rkpr1` to
  `rkpr16`), in that runner's slot (`rkmon_lib/prcode.py`, `rkmon_lib/slots.py`, `worker/pr_user.py`), which reads
  neither secrets volume (`/secrets`, `/secrets-write`) nor anything else the monitor's user keeps to itself, and writes
  only its slot's folders, made fresh for each job: the jobs (the checks, the gates, the develop sanity runs, the
  bisects, the drift checks, the probes), the first review pass's tools' `run_pytest`, `run_probe` and `run_mutant` (and
  `rk evidence-check`'s re-runs of them), the tool loop's saved probes, `worker/run_suite.py`, the rk commands that run
  PR code (`rk probe-check`, `rk mutants`, `rk werror` and the others `rkmon_lib/prcode.py` names), and the session's
  own runs through `rk pr-run -- CMD` (the checks' re-run commands start with it). Its git fetches nothing and trusts
  only the job's trees, and the monitor reads what it keeps of a job's results as data. The image test's `isolation`
  stage proves it with a PR test file and a probe that try to read both volumes. When a command ends, the launcher stops
  every process it left; a job's kill, `rk stop` and `rk kill` stop the runners' processes through it too.
- **Outside its slot.** A PR runner cannot use `/tmp` or `/var/tmp` (root's, the monitor's group's, mode 1770): a job's
  temporary files go to its slot's `tmp`, its `TMPDIR`. `/dev/shm`, `/dev/mqueue` and System V IPC stay every user's,
  since the container cannot give a runner its own (Python's multiprocessing locks need `/dev/shm`): there a runner
  writes its own entries, which the launcher removes, with the System V shared memory, semaphores and message queues the
  runner owns or made, each time it begins, keeps or ends the runner's slot (`rkmon_lib/container_layout.py`). One risk
  remains: a PR job can make an object of a fixed name in `/dev/shm` or `/dev/mqueue` for a develop job that runs at the
  same time to open. Each runner's own group, which no other user is in, narrows that to an object the PR job opens to
  every user.
- **The slots.** A job takes a free slot, which the launcher empties and makes fresh as the job begins, and frees it
  when it ends (`rk pr-slot status` lists the slots). The checks' slot is kept for its review: its build is the review's
  shared build, which the other runners read and none writes, and no job runs in it. The review's end (its post, its PR
  closed, or a new head) frees it, and so does a job that finds no slot free, which takes the slot kept longest; with
  none kept, the job waits for one.
- **On a host install** there are no PR runners: PR code (tests, `conftest.py`, builds, probes) runs as the account
  that runs the monitor. Those runs get only the read login's folder and no key or token variable (`gh.job_env`), but
  that account can read the other files in `<SECRETS_DIR>`, the write login's included, so a malicious pull request
  (from a fork, for example) could read or use them. Hardening for a host:
  - run the monitor as a Linux user with no other secrets, or use the container;
  - give the operator's login a token scoped to the one repository.
- **A remote model endpoint receives the prompt:** the review book and the pull request's context. Use an endpoint
  your organization allows for this code. Its key is never in a settings file: it is a secret, installed with
  `rkmon secret <name>_api_key FILE`.
- **The local review loop's prompt is the same for every pull request,** so its results stay comparable and cannot
  drift from the request.

### 7.3 Public-repository hygiene (every post)

- No internal links, host names or job links. Never construct a link from memory; when one is needed, ask the
  operator for the exact address. `rk post` refuses links outside an allow-list of public hosts.
- No absolute performance numbers (times, throughput, bandwidth), not even the ones being flagged: describe them
  instead ("absolute kernel timings in the PR body").
- Measured ratios only where the repository's `platform/AGENTS.md` allows them (PR descriptions and qualifying result
  files). In our comments, say it in words ("slower at the measured configurations") or point at the ratio the PR's
  own description states. Counts and structural ratios are fine ("10 of 16 cases").
- Ticket references are bare tags only (`<JIRA_PROJECT>-NNNN`), and only the PR's own.
- People are named by their GitHub handle, never as a mention that notifies them, with they/them pronouns.
- The same holds for what a PR commits: pull-request references (`#NNNN`, `PR #N`, `/pull/N` links) and dated
  narratives of particular runs that a PR adds to committed files (code, comments, docs, runbooks) are a Needed-to-merge
  yellow finding, listing every file:line; the fix keeps the standing fact. The peer check `prnumbers` flags the
  references. Our own posts may name PRs.

### 7.4 Integrity

- **Claims are not evidence.** Author replies, commit titles, and GitHub's resolved or outdated flags prove nothing; a
  formatter's pass marks threads outdated without fixing them.
- Mark every FIXED from a test run or a code read at head. Otherwise write "unverified", or lower the status.
- **Disprove before you report.** Check whether a finding is older than the PR (diff against develop), shared by
  develop, specific to this host (line endings, file locks, missing tools, another Python version than CI's), or
  already reported.
- If one of our earlier items was wrong, retract it explicitly (white) and offer the operator a correction text.
- Never post a draft raw. Strip ACCEPT / REJECT, copied example text, "Saved to" lines, "(ran)" claims nobody
  produced, and wrong CI statements.
- **Diagnose before iterating.** When something works in one place and not in another, diff the two cases before
  forming a hypothesis.

### 7.5 Machine and GPU safety

- **With a local Ollama,** never restart the service on a wedged GPU: that can turn a soft hang into a fatal host
  crash (a Windows bugcheck). Stop the runner instead (`ollama stop`, or `kill_runner.py` as root, which kills only the
  runner), and reboot cleanly if the GPU stays wedged.
- One local draft at a time unless the operator confirmed more
  (`model_bindings.worker.limits.max_parallel_requests`), always with an explicit `num_batch`; never Ollama's default
  batch at the full context.
- Builds within the parallel builds in effect (`monitor.parallel_builds` is where they start, within their range) and,
  with a local model, the prefill rules of section 3.5. The scaler lowers the parallel reviews and the parallel builds when the
  CPU or the memory runs over `max_host_usage_pct`; lowering never stops running work.
- `TMPDIR` is `paths.tmp`, never `/tmp`, which kernel-cache tests can fill. With the RAM disk on, `paths.tmp` lives
  on it, sized, and a job starts only when it has room (section 3.5).
- On WSL, the page cache is not given back to Windows by itself. If Windows runs short of memory, drop the caches as
  root (`sync; echo 1 > /proc/sys/vm/drop_caches`) or turn on WSL's `autoMemoryReclaim`.
- Never install packages on the bare machine: everything runs in the container or, on a host install, in venvs.
  Never put a host install's working folder in a git work tree or a cloud-synced folder.

### 7.6 Auto-merge rules

GitHub enforces CI, approvals and conflicts on its own, but it merges on a stale base and ignores our findings. With
the merge queue on:

| Rule | What happens |
|---|---|
| The auto-merge guard (an alert in prototype; live, it turns the auto-merge off) | When a PR whose auto-merge is on fails a gate GitHub does not enforce (our open red or orange at its head, a confirmed hold, a hold label), the tick prints `!! AUTOMERGE_RISK #N`. Live, `rk automerge-off N` turns auto-merge off and posts one comment that says, in plain words, what fails and how to fix it; once per head. Untriaged failures never count |
| The operator's own auto-merge | Never turned off: it gets an alert, and the operator decides. On a PR in the merge plan's wave 1 whose only open gate is our orange, there is no alert at all; if it merges, the follow-up tickets cover what is left |
| On the operator's instruction | `rk automerge-off N --user FILE` turns someone's auto-merge off without a gate failure, for example to land other PRs first, and posts the operator's reason in plain words |
| Ahead of the plan | `!! AUTOMERGE_JUMP #N`: auto-merge would land a PR ahead of the merge plan. The operator is told; nothing is posted |
| Arming (Auto-merge: live, `features.merge_queue.auto_merge.enabled`) | Only for PRs the operator cleared (`rk clear N`), one at a time, as the next member of a validated batch, with every condition re-checked by `rk arm` |

## 8. rocKE CI knowledge for verification

**Workflows at a PR head**

- **TheRock Multi-Arch CI** has build stages per architecture family (gfx94X, gfx950, gfx125X, gfx110X, ...). Its
  "Test hipkernelprovider" jobs run on Linux gfx94X and Windows gfx110X, two shards each, and sometimes on gfx120X.
- **TheRock Multi-Arch CI ASAN**, **TheRock CI**, and the **hipDNN Superbuild** workflows (Linux and Windows).
- **pre-commit**, **clang-tidy**, **Component CI**, and policy and housekeeping checks (a PR bot, base freshness,
  auto-labelling).
- **gfx950 and gfx1151 test jobs usually appear only in manually dispatched runs.** A build stage for an architecture
  does not mean tests ran there.

**What the test jobs run**

- `ctest -L '^quick$'` or `-L '^standard$'`; the tier varies from run to run.
- Tier membership comes **only** from `ROCKE_ENGINE_test_categories_external.yaml` (the tier YAML). A lane missing
  from it runs only in quick (if it has the hard-coded label) and silently drops out of the other tiers.

**Test lanes**

| Lane | What it runs | Gotchas |
|---|---|---|
| `rocke_pytest` | The **installed** platform tests (`build/bin/hip_kernel_provider/tests`, Linux and Windows), with `--ignore tests/library`, `-m "not integration"` and a 60-second test timeout | On Linux it gets `PATH` and `HIP_DEVICE_LIB_PATH` from `$ROCM_PATH`. A test that reaches a source-only path passes in the source tree and fails here; the checks run every changed test in the installed tree |
| Library lanes (`rocke_library_<area>_pytest`) | **Hand-picked file lists** in `platform/CMakeLists.txt` (`_ROCKE_LIBRARY_*_TEST_RELPATHS`); the attention lane points at a folder | **A new `library/tests` file runs nowhere unless it is listed** |
| `rocke_library_gpu_pytest` | The library GPU tests | **No torch on the GPU lanes**, so torch tests skip. A CTest entry whose only file skips at module level collects 0 items: pytest exits 5, and CTest reports Failed |
| Package-internal tests (`platform/python/rocke/**/tests`) | Nothing, unless CMake stages them under `tests/` | — |
| Differential goldens (`platform/tests/instances/differential/`) | The CTests `rocke_differential_golden_ll` and `rocke_differential_golden_canonical`, in the hipkernelprovider shards | A PR that changes develop's IR without a re-bless fails here; new configs that are not blessed yet are reported but do not fail |

**Tools and builds CI lacks**

- **The pybind `rocke_engine` is never built in CI**, so the C++/Python parity tests always skip there.
- **`llvm-objdump` is absent in CI**, so the resource and no-spill gates skip ("missing tool: llvm-objdump"). Run them
  locally with `<ROCM_LLVM_BIN>` on PATH.

**Reading CI status correctly**

- **A cancelled first run:** red summary or notification checks often come from a run cancelled at its start at the
  same head. Read each run's conclusion; the tick ignores a cancelled run whose check also passed.
- **A TheRock CI "success" that finishes in about a minute** is a path-filter no-op, not test evidence.
- **A PR that conflicts with develop has no current CI against develop.**
- **Develop itself breaks sometimes.** Check develop's runs before blaming a PR. A failure whose first error is the same
  on develop or on other open PRs today is known infrastructure or develop's own (`rkmon_lib/known_infra.py`).
- **A CI failure predicted from a read** stays yellow ("predicted: confirm on the shard") until a shard's log shows it.
- **A re-run hides earlier attempts.** A re-run of failed jobs, automatic or by hand, replaces them in the check-run
  view and in a run's default job list. Before calling a red head infra, read `actions/runs/<id>/attempts/<n-1>/jobs`
  for every job that was re-run; the checks' "CI at head" section lists the jobs an earlier attempt failed
  (`tools/ci_attempts.py`).
- **CI runner infrastructure is not the PR's** (the runners and their checkout, container and pod failures).
  Attribute it as infra in the Bots / CI row, with the evidence, and tell the operator in one line when it is
  widespread.

**Fetching logs**

- `gh api repos/<ROCKE_REPO>/actions/jobs/<id>/logs > job.log`, then grep for `Test #NN: <lane> ... Passed`, `Failed`
  or `Skipped`; per-test lines (`path::test PASSED`, `SKIPPED` or `FAILED`); the `rootdir:` line; and
  `SKIPPED [k] ... <reason>`.
- The rocKE pytest lanes are verbose per test, so a log proves whether a specific test passed or skipped on real
  hardware.

**This host against CI**

- Pin `ROCKE_LLVM_FLAVOR`: the host's LLVM may be newer than some of rocKE's committed flavors (llvm20, llvm22, llvm23
  and llvm24 on today's develop; `LLVM_EMISSION_FLAVORS` in `platform/python/rocke/core/llvm_profile.py` is the
  source). The checks run the PR's changed tests at llvm22; tests that skip at CI's flavor but pass at llvm22 are a
  miss to look for.
- Generators that need a matching toolchain report UNVALIDATED (exit 77, a CTest "Skipped"). When CI's own clang is
  newer than the flavor, those reproduction gates skip in CI too.

## 9. Known issues and their fixes

Each row is a fact about this setup's tools and machines, what it costs when it is ignored, and the rule or the code
that answers it.

| # | Issue | Impact | Fix, or the rule |
|---|---|---|---|
| 1 | **A host crash.** An agent harness on Ollama's defaults (the full context and the default batch) can wedge the GPU with a full-size prompt, and restarting Ollama on the wedged GPU can escalate into a Windows bugcheck, more so on an outdated GPU driver | The whole machine reboots | Drive `/api/chat` directly with a ladder-tested `num_batch`; never restart Ollama on a wedge (`ollama stop`, `kill_runner.py`); one local draft at a time unless confirmed; keep the GPU driver current |
| 2 | Dense 27B models hang on prompt batches of 128 or more on WSL's ROCm | Stalled drafts | `num_batch` 32 for dense models; the mixture-of-experts model is fine at 256 |
| 3 | `OLLAMA_NUM_BATCH` is ignored, and OpenAI-compatible endpoints cannot pass `num_batch` | The batch cannot be capped through a harness | `options.num_batch` per request on `/api/chat`; the raw model, never a wrapper Modelfile |
| 4 | **A first pass that sees our earlier reviews repeats them**, hardware evidence included | False independence: our evidence presented as its own | `fetch_pr.py` leaves out the operator's reviews, comments and thread comments |
| 5 | Made-up content: "ACCEPT", "+17" copied from the book's example, "Saved to ...", false "(ran)", "all CI green" while CI runs, fixes credited to the wrong commit, CI failures the PR caused called "unrelated" | Misleading if posted | The verifier reviews blind, verifies everything and posts only what it re-derived; the local review loop's prompt stays unchanged, so its results stay comparable |
| 6 | A local model reads code instead of running tests | Shallow first passes | The prompt requires running the changed tests, `rule_scan` and `merge_conflicts`; the checks and the verifier run their own |
| 7 | An empty final answer: the model spends its output budget thinking | No draft | A forced final turn with thinking off |
| 8 | **A local model's request times out** while heavy CPU suites or an automatic `git gc` run beside it | A lost draft | Keep the CPU load light while a local model drafts; the parallel builds and the prefill rules (section 3.5) enforce it; `gc.auto 0` (issue 28) |
| 9 | Huge PRs overflow the context; the model answers "no PR URL was provided" | A useless draft | The prompt budget, the section caps, the collapse of golden and generated data, a code-first diff, and a diff rebuilt from the clone when GitHub refuses a large one |
| 10 | `/tmp` fills up during kernel-cache tests | False failures | `TMPDIR` is `paths.tmp`: on disk, or on the RAM disk, whose room is checked before a job starts |
| 11 | WSL's page cache starves Windows of memory | A slow host | Drop the caches as root before a heavy run, or turn on `autoMemoryReclaim` |
| 12 | `wsl.exe -- bash -c '...'` from Git Bash mangles `$` and awk fields; a `pkill -f` inside it kills itself | Broken commands | Script files only; `--exec` when inline; `-u root` instead of sudo |
| 13 | Carriage returns in scripts written on Windows; a regex replacement with backslashes breaks a patch | Script errors | Strip carriage returns, and the toolkit's tests fail on one in any tracked file; use function replacements in Python's `re.sub` |
| 14 | A Windows temp cleanup deletes files under the Windows temp folder, venvs included | Broken pytest | Keep the environment on the Linux side |
| 15 | An older `gh` lacks the `baseRefOid` field | A failed fetch | Read `.base.sha` through `gh api` |
| 16 | The operator's hand-written comments and reviews are our review too | A PR wrongly treated as never reviewed | Count the operator's comments and reviews as well |
| 17 | A conflicted merge state left in a reused worktree | Worktree failures | `merge --abort` and `reset --hard` before the worktrees are made again |
| 18 | A non-sparse worktree starts a full checkout of the monorepo ("Filename too long" on Windows) | Wasted time and disk | Always fresh, sparse worktrees |
| 19 | Shallow history: a helper can read the base SHA as the old head, and shallow merge bases are wrong | A wrong "content changed" report | Take the head SHA from the Head line, not the base; `range-diff` per PR; a clone date far enough back (`clone_since`) |
| 20 | Without `llvm-objdump` on PATH, the resource gates skip silently, as they do in CI | Missed spills | `<ROCM_LLVM_BIN>` on PATH for every check; read CI logs for "missing tool" skips |
| 21 | A runtime-only ROCm (the runfile installer's default component, `core`) has no HIP headers, so hipcc tests skip locally | Lost coverage | Install the `core-dev` component too (`rocm compo=core,core-dev`; the image's `docker/install_rocm.sh` does and stops the build without them); `rk doctor` checks hipcc for gfx942 and gfx950 |
| 22 | No `llc` in a runtime-only ROCm | Tests gated on `llc` skip | ROCm clang standing in for `llc`; say so in posts |
| 23 | `HIP_VISIBLE_DEVICES=-1` aborts the ROCm runtime as soon as code touches HIP (a torch import included) | False failures | No check sets it: the checks, the sanity runs, the drift runs, the saved probes and the tools each unset it and hide the GPU with an empty `CUDA_VISIBLE_DEVICES` (`rkmon_lib/nogpu.py`) |
| 24 | A newer local Python than CI's gives different tracebacks | Environment results blamed on the PR | Compare with CI before blaming the PR |
| 25 | A GitHub MCP server's job-log tool keeps only the last few thousand lines | Earlier CTest lines lost | `gh api .../actions/jobs/<id>/logs` for the full log (`rk job-log ID`) |
| 26 | A post whose line 1 names a model that did not draft it misattributes the work | Wrong attribution | Line 1 names the model that drafted, or the verifier when nothing drafted; the footer credits the verification (G3) |
| 27 | A naive "keep both" conflict resolution does not compile | A wrong recommendation | Build every proposed resolution in a scratch merge first |
| 28 | After many PR refs are fetched at once, `git gc --auto` repacks the blobless clone for hours at a full core | Slower drafts and checks; part of issue 8 | `gc.auto 0` on the clone (the deployment sets it); `git gc` by hand when the machine is idle |
| 29 | A fresh Ollama runner can wedge in its first big prompt (it logs "new prompt", then nothing) while a native build runs | A lost draft per wedge | Streamed requests, abandoned after `stall_s` of silence; the model unloaded (`ollama stop`) so the resend gets a fresh runner. A runner stuck at full CPU with the model "Stopping…" is killed by its process id, never by a service restart. The prefill rules (section 3.5) |
| 30 | Each hung runner costs the full stall timeout before the resend | Slow recovery | Hang detection from Ollama's journal (`hang_s`), then a resend after `ollama stop`, up to `max_hangs` times |
| 31 | A resend queues behind a wedged runner that stays busy after the cancel | A lost draft | A local Ollama unloads the model before every resend |
| 32 | A forced final turn without tools makes the runner process the whole conversation again | A lost draft to the stall timeout | The tools stay in the final request; a tool call asked for then is refused, once |
| 33 | Basic grep reads `a\|b` literally and matches nothing | False "not found" leads | `grep_code` makes each unescaped `\|` alternative its own `-e` pattern |
| 34 | Python with `-I` drops `PYTHONPATH`: every test that imports the library tree fails | False findings | `run_pytest` sets `PYTHONPATH` explicitly, with the monitor's pytest plugin |
| 35 | A re-run of failed jobs hides attempt 1 in the check-run view, so a red head can be called infra while attempt 1 had the PR's own test failing | A wrong CI attribution | Read the earlier attempts before calling infra; the checks list them (`ci_attempts.py`) |
| 36 | A changed test can pass in the source layout and fail in CI's installed `rocke_pytest` | Our checks pass tests that CI then fails | The checks run every changed test in the installed tree, with the checks' own `ROCKE_*` settings dropped, as CI sets none, and the GPU hidden as the checks hide it |
| 37 | Open PRs add the same helpers under one name or several, and develop may already hold copies of a helper | Several homes for one fact | The overlap scan and the owner rule: one owner per shared piece, and a notice on each PR that should drop its copy |
| 38 | A 5-minute prompt cache expires between 10-minute ticks | The orchestrator re-reads its context uncached at every tick | A 1-hour cache for the session (`claude.prompt_cache_ttl`; the review passes keep `claude.subagent_prompt_cache_ttl`, their turns running back to back), written into the session's settings file |
| 39 | Review pass calls that carry workspace instructions and memory they never use, and verifiers that read the whole lessons file | Cost | Dedicated agent types for the passes, without the workspace instructions; the per-role lessons extract |
| 40 | When pushes outpace the reviews, the backlog grows and the board says LOSING | Late reviews | Parallel reviews, a shared rate budget for a remote endpoint, separate parallel builds, and the scaler that moves both from the machine's load (sections 3.3 to 3.5) |
| 41 | One verifier's red and orange recall falls well above about 1.5k changed lines: the files that hold the misses go unread | Missed blockers on giant PRs | The split review (`rk review-split`, section 3.12): parts one verifier each reads whole, then a cross pass |
| 42 | A drift run that extracts develop and the merge differently, or reads a head-only failure as develop's doing, reports drift that is not there | A false G6 failure | Like-with-like extraction (`drift_extract`), the head-alone rule, `rk drift-dismiss` for a harness false positive (section 3.12) |
| 43 | Every develop merge into a PR would otherwise cost a full review | Reviews spent on heads nothing changed | Merge-only carry with clean checks or known infrastructure CI, and micro rounds for new threads or a title problem (section 3.11) |
| 44 | Resuming a verifier that handed back reloads its whole context, often from a cold cache | A re-check that costs more than a review | A fresh verifier with a narrow brief (its notes, the post, what changed); resume only one that handed back minutes ago |
| 45 | A verifier that waits for the code quality pass past 5 minutes loses its prompt cache, and folding the pass in after a hand-back re-writes its context | Cost | `rk wait-quality` returns within 4.5 minutes and is called again until the pass is final |
| 46 | `cut -c` counts bytes: a status line cut inside a multi-byte character ("…", "—") is not UTF-8, and a strict reader stops the tick on it | A failed tick | The checks' and the sanity run's `st` write whole characters only (`u8`), every piped cut is `cutc`, and the reader replaces a bad byte |
| 47 | A container restart empties the RAM disk, and a draft awaiting verification loses its worktrees | A verification or a re-check on no tree | `rk review-recheck`, `rk dispatch-pack` and the code quality pass make them again with `mkwt.py` first (section 3.5) |
| 48 | A hygiene check that matches product words inside other words ("navigation", "instinctively", "stalemate") | A post stopped for nothing | The check matches whole words (a product name may take a model number after it) |
| 49 | On a host install under WSL, with no client attached, WSL stops the distro about a minute after each tick, and that kills every job | Lost gates, sanity runs and checks | A keep-alive `wsl.exe` session |

## 10. An honest assessment of a first review pass

- **Value:**
  - an independent first opinion, produced cheaply, and privately with a local model;
  - occasional real catches;
  - a structured draft in the post's format;
  - sometimes a useful pointer to which tests to run.
- **Precision is low.** In most batches the verifier keeps 0 or 1 of a local model's findings per PR, and nearly every
  posted finding comes from the verifier's own review and the checks. Over a sample of 116 local-model drafts,
  verification corrects about 10.6 of a draft's claims and adds about 8.6 it missed. The verifier is the real
  reviewer, and its blind pass comes before any draft.
- **So the first review pass is optional,** and off by default (section 3.9): the verifier reviews alone, with the same
  checks, and line 1 names the verifier. A first review pass stays available for a deployment that wants its leads or
  is assessing a model, and each draft is scored against the verified post (section 3.9, Records).
- **The local review loop never sees our earlier items, by design;** re-reviews and follow-ups depend on the verifier
  to carry them. The tool loop settles them from our last review (section 3.9).
- **More parallel reviews raise throughput, not precision.** Every extra review is one more verification, and the
  verification is where both the cost and the quality are.
- **A stronger model may need fewer corrections,** but the rule does not change: nothing it writes is posted until the
  verifier has verified it.
- **A local model writes ACCEPT / REJECT verdicts anyway,** though the review book's lessons forbid them (section 5.7).
  The post format is the verifier team file's Output Format.

## 11. Glossary

The terms the whole package uses (operator, orchestrator, pass, parallel reviews and builds, RAM disk, merge plan,
overlap notice, follow-up ticket and the others) are in the README's
[Glossary and placeholders](../README.md#glossary-and-placeholders). These are the terms only this page uses:

| Term | Meaning |
|---|---|
| hkp | hip-kernel-provider, the hipDNN plugin that ships rocKE kernels |
| TheRock | ROCm's build and CI system; its "Test hipkernelprovider" jobs run the rocKE lanes |
| Local review loop | `worker/review_agent.py`: a first review pass on the model binding `worker`, which drives the model with a fixed menu of tools (section 5). The assessment tags its drafts with `worker.tag` (by default the leading letters of its model id, such as `qwen`) |
| Tool loop | `worker/drafter.py`: a first review pass on any other model binding, in phases, with evidence for every claim (section 3.9) |
| Review offload | `offload.mode`: the first review pass's model does a review's substance with evidence for every claim, and Claude audits it (section 3.8); `shadow` runs it beside the usual review and posts nothing |
| Audit; deep review | Claude's check of an offload review: its 🔴/🟠 claims' evidence, a sample of the rest, the format and premises (on the verifier's model); a deep review is the full review instead (on the code quality pass's model) |
| Evidence check | `rk evidence-check N`: a draft's quotes read again and its cheap runs re-run, with no model: PASS or FAIL per claim (`--offload`: the review offload's) |
| Agent type | A Claude Code subagent definition in `.claude/agents/`: its model, its tools, its settings |
| Review book | The local review loop's rubric and the house review standard (`<REVIEW_BOOK>`) |
| First round | Our first review of a pull request; a follow-up review re-reviews it after substantive pushes |
| Merge-only carry; micro round | A head whose new commits only merge develop in, recorded as covered by our review with nothing posted when its checks are clean (or its only flag is known infrastructure CI); when its only new items are threads or a title problem, a micro round: the verifier alone on those items |
| Needed to merge; follow-up item | The post's two parts: every blocker, each with its light, which alone sets the header light; and significant non-blocking items (F rows), listed once and then tracked to a fix, an author's deferral or a ticket |
| Split review; part; cross pass | A review too large for one verifier (`rk review-split`): parts one verifier each reads whole, running the part's code quality checks too, then one cross-cutting verifier that writes the single post |
| Shared build | The checks' native build of the merged tree, which the verification pass and the code quality pass reuse read-only |
| Drift | A PR's own tests failing on its head merged with today's develop while develop alone passes; it fails G6 (`DRIFT_FAIL`) |
| (ran) / (read) | Evidence from a run at this head; evidence from reading the code |
| Review driver | `tools/review_driver.py`: the one background process that keeps the parallel reviews busy and prints `REVIEW_READY #N` for each review ready to verify |
| Scaler | `rkmon_lib/scaler.py`: moves the parallel reviews and the parallel builds in effect one step at a time within their ranges from the machine's CPU and memory against `max_host_usage_pct`, and decides whether another pool draft may start; lowering drains, never stops running work |
| Remote GPU tests; GPU tests row | `features.review_automation.remote_gpu_tests.enabled`, with review automation enabled: a review's GPU-only cases (those the checks skipped for want of a GPU) run on a GPU node of a Slurm cluster over SSH, queued; the review's "GPU tests" row says pending until `rk gpu-row` updates it in place (section 3.7) |
| Node probe | What a GPU job checks first (the target's device is visible, hipcc compiles the HIP header): a node that fails it fails no test |
| Prefill protection | With a local model only: no build in a local draft's first 5 minutes, and no local draft in a build's first 10 |
| Shared budget | A file under `<RUN_DIR>`, locked while in use, that counts a remote endpoint's requests and tokens per minute for every review, and holds the waits that a 429 or the rate-limit headers impose |
| Ledger | `state/actions.jsonl` in `<TOOLKIT_DIR>`: the record of every outward action, checked before each write |
| Flavor | The LLVM IR dialect rocKE emits for a toolchain generation (llvm20, 22, 23, 24); pin it with `ROCKE_LLVM_FLAVOR` |
| Byte identity | The C++ and Python lowering engines produce identical IR (`check_byte_identity.py`) |
| Golden | Committed hashes of emitted IR, per case and flavor: representative, or per family |
| Differential anchors | `llvm_gfx_all.json`: IR hashes at llvm20, checked locally with `sweep2.py` |
| Merge simulation | `wt<N>m`: the PR head with today's develop merged in, where the checks run |
| Semantic merge break | A textually clean merge that breaks behaviour, for example a stale symbol or a stub that misses a new step |
| Tier YAML | `ROCKE_ENGINE_test_categories_external.yaml`, which maps CTest lanes to the quick, standard and other CI tiers |
| Installed tree | The `cmake --install` layout that CI's `rocke_pytest` collects; the checks run every changed test there |
| Stale approval | An approval recorded at a commit older than the head; on develop it still counts |
| Prompt cache TTL | How long the model provider keeps a prompt prefix cached: `claude.prompt_cache_ttl` for the orchestrator's session and `claude.subagent_prompt_cache_ttl` for the review passes |
| Review spend | `rk spend`: what each posted review cost in US dollars, by part, from the Claude transcripts and config `prices`; off without a price table |
