<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE Monitor and Automated PR Review

> The overview of the rocKE monitor, for the people it serves and the people who decide on it: the problems it solves, what it produces, its parts in brief, its guardrails, benefits and limits, and the policies the operator keeps. To build, deploy or run it, start at the README's [Documentation](../README.md#documentation). The terms and placeholders this page shares with the other docs are in the README's [Glossary and placeholders](../README.md#glossary-and-placeholders); the terms only this page uses are in its [Glossary](#glossary).

| | |
|---|---|
| **What** | The rocKE monitor: it reviews rocKE's pull requests, keeps their merge queue, and checks develop after every merge |
| **Repository** | One GitHub repository, `<org>/<repo>` (`repository.name`), and rocKE's folder in it (`repository.folder`) |
| **Scope** | Open pull requests into develop that are not drafts and carry a scope label (`repository.scope_labels`), or a shared label (`repository.shared_label_folders`) and a change in that label's folder |
| **Runs as** | One container on one Linux host, which `rkmon` sets up and drives from the host; a long-lived Claude session inside the container does the judgement. A host install, without the container, is for a machine where a container is not an option |
| **Starts as** | Every feature that posts or sends starts in prototype: it does its work and keeps what it would send, and nothing goes out until the operator sets the feature live ([Prototype, live and standby](../README.md#prototype-live-and-standby)) |
| **Readers** | Managers: sections 1–3, 6 and 7. Engineers: sections 1–8. Maintainers: the appendices too |

<!-- toc -->

---

## 1. Summary

- **What it is.** The rocKE monitor keeps develop, the shared integration branch, green while rocKE pull requests land (rocKE is AMD's kernel DSL and library). It runs in a container: a toolkit does every mechanical step, run by fixed rules, and a Claude session inside the container does the judgement.
- **The toolkit** (`rk`) reads GitHub under a read-only login, computes the merge gates and a merge plan of conflict-free waves on today's develop, and finds work that two open pull requests both add. It builds and tests every reviewed pull request merged with develop, runs waiting pull requests' own tests again whenever develop moves, and checks develop after every merge. It keeps its state, a journal and a ledger of every outward action on disk.
- **The orchestrator**, one long-lived Claude session (`rk orchestrate`), is woken by a tick loop that runs every few minutes. A rule filter reads each tick first, so the session spends a turn only on the lines that need Claude. For each finished review it runs two passes, each a Claude agent: the verification pass, which reviews the pull request blind, reproduces or cuts every claim and writes the post, and the code quality pass beside it. It posts one comment-only review per pull request head, files develop-break issues and follow-up tickets, and acts only through guarded, rate-limited and ledgered `rk` commands, which run as a user of their own under a separate write login.
- **A first review pass is optional.** By default the verification pass reviews alone. Another model, a model binding (a model on an Ollama server or behind an OpenAI-compatible endpoint, or another Claude model), can write a first draft; the verification pass then checks it claim by claim.
- **Capacity follows the host.** Several reviews and several builds run at once; the monitor raises or lowers both, within their ranges, as the host's load allows ([Parallel reviews and parallel builds](../README.md#parallel-reviews-and-parallel-builds)).
- **One operator** reads a status board, a Needs-you list, the dashboards, the daily logs and a weekly report, and keeps every merge, approval, priority and policy decision. The admin dashboard is the operator's page, with Approve and Decline for priority requests, proposed tickets and the in-depth reviews of large pull requests, and the unsent items: what a feature in prototype, or the monitor in standby, kept instead of sending. An optional team dashboard shows the same page to other teams, with one approval box per code-owner team and a box where anyone can ask for a pull request to go first; the operator decides each request.

## 2. The problems it solves

| Problem | What happens without it | What the monitor does |
|---|---|---|
| **Review backlog and latency** | Dozens of rocKE pull requests are open at any time, and every push makes a review stale | Reviews every new pull request and every substantive push by itself, develop fixes and near-merge pull requests first, several at once |
| **Develop breaks** | develop has no strict status checks and no merge queue, so pull requests merge on a **stale base** | Gates every candidate against today's develop, checks develop after each rocKE merge, and files one GitHub issue per break |
| **Stale results on an unchanged head** | A pull request whose head does not move keeps its old test results, and its CI keeps testing an old merge | A drift check runs each reviewed pull request's own tests again whenever develop's rocKE code moves, approved or not. A failure blocks the pull request before anyone approves it |
| **Merge conflicts and rebases** | A few hot files (goldens, the ABI header, CMake lists) cause most conflicts, and nobody can say which merge order causes the least rebasing | A merge plan: conflict-free waves on today's develop, with rebase advice for every near-merge pull request; batches of compatible pull requests; a weekly hot-file report |
| **Duplicated work** | Open pull requests add the same helper, macro or test, under one name or several, so develop ends up with two homes for one fact | Every review lists the overlapping work; one owner is proposed, and the other pull requests get a notice with the landing order and the rebase that removes their copy |
| **CI blind spots** | Many rocKE tests run in no CI lane. CI has no PyTorch, no build of the C++ engine's Python module and no `llvm-objdump`, and some architecture lanes skip on pull requests | Builds and tests every pull request merged with develop: byte identity, goldens, its own tests and those that import what it changes, the installed-tree run, spill checks, probes; GPU-only cases on remote GPU nodes when remote GPU tests are on |
| **Inconsistent review quality** | Reviews vary; author replies and "resolved" flags are not evidence | One format and one evidence bar; every claim reproduced or downgraded; a code quality pass; scope first, so the author sees every blocker at once and nothing else blocks |
| **Tribal knowledge** | The know-how lives in one person's head and in chat history | A runbook skill, procedures and a lessons file that every agent reads; a package any team can deploy |

**What GitHub enforces, and what the monitor adds** (the gates G1–G9 are listed in [design.md, "Merge gates"](design.md#merge-gates))

| Check | GitHub on `develop` | The monitor |
|---|---|---|
| Required checks green | Yes, at the pull request's head | Also confirms that every required check reported (G3). A cancelled run whose check also passed is not a failure |
| Code-owner approval, resolved threads | Yes | Names who has to act (G2) |
| Up to date with develop | No | Gates the pull request merged with today's develop (G4–G7), and runs its own tests again each time develop's rocKE code moves (the drift check, part of G6) |
| Goldens | Only in the pull request's own CI run, against develop as it was when that run started | G5, on the merge with today's develop |
| Byte identity between the two engines; ABI numbering across pull requests | No | G6, G9 |
| Arch coverage | A lane that did not run is invisible | G7 asks for a compile per target of every touched parity family. Run evidence for a numerics change is judged in the review: a gap is a 🟠 finding, which fails G1 |
| Review findings | Only a change request blocks | An open Needed-to-merge 🔴/🟠 fails G1. The auto-merge guard turns someone else's auto-merge off, or alerts |
| Merge order across open pull requests | No | The merge plan: textual conflicts and two claims on one ABI counter order the waves |
| The same code added by two open pull requests | No | Overlap leads in every review; one owner per shared piece |

## 3. What information it produces

The files it writes for the operator are in the outputs folder, `/work` in the container ([The outputs folder](../README.md#the-outputs-folder)).

| Output | What it holds | Where and when |
|---|---|---|
| **Review comment** | One comment-only review: the verified status of every prior item, a **Needed to merge** table with every blocker, a **Highly recommended follow-up (not blocking)** table with significant out-of-scope items only, and a utility line ([its shape](#review-comment)) | On the pull request, for each new pull request and substantive push, while review automation is live (`features.review_automation.mode`); a copy in the outputs folder. While review automation is in prototype, the monitor is in standby, or posting is not set up, the verified review is kept in the outputs folder instead, unposted. A kept review is never posted later: the monitor records only what it posts |
| **Status board and Needs you** | The rows below, then what only the operator can decide, pending priority requests, proposed tickets and in-depth reviews waiting for approval included, then the pull requests ready for each team's approval | In the tick's output when it changes, and at least hourly; in the container's log (`rkmon logs`); at the top of the Markdown dashboard |
| **Dashboards** | Needs you; the proposed tickets waiting for the operator; the way from review to merge; the reviews; one approval box per code-owner team; the mergeable candidates and their gates; the merge plan (the "Request PR priority" box and the requests, the waves, a conflict grid with Ⓐ marking ABI pairs, the ABI lanes, the auto-merge that is on, and the rebase advice, collapsed); insights; develop's health. On the admin dashboard only: the Scoreboard, Review spend, "Not sent" (the unsent items) and Today's log, and, while the monitor is in standby, a top line that says so. Charts are color-blind-safe and have table views | Rewritten by each full tick, as `rocke_monitor.html` and `rocke_merge_queue.md` in the outputs folder. The admin dashboard is served on the deploy host ([The admin dashboard](../README.md#the-admin-dashboard)); the team dashboard, when on, is copied to a web host or served on a port of its own ([The team dashboard](../README.md#the-team-dashboard)) |
| **Merge plan** | The near-merge pull requests in conflict-free waves, in landing order: the priorities in the earliest waves they can join, hubs last; whom each landing makes rebase, and who acts; auto-merge that may land early; rebase advice for every pull request ([section 4.8](#merge-plan)) | `rk merge-plan`, the board's Merge plan row, and the dashboards, every full tick |
| **Daily log** | One row per event, plus **saves**: one record per issue caught (a finding posted, a finding fixed, a gate block, a develop break, a corrected draft), totaled on the board as "Issues avoided" | The outputs folder's `rocke_daily/`, one file per UTC day |
| **Weekly report** | Pull requests, reviews, conflicts, CI times, develop breaks with culprit and fixer, friction, pull request age, review spend; beside it, the raw material for the weekly lessons review | The outputs folder's `rocke_weekly/`, at the first full tick after the week's report time, with `features.weekly_report.enabled` |
| **Develop-break issue** | One GitHub issue per break, in the repository's own issue style: labelled CI red (`features.merge_queue.develop_guard.ci_red_label`) when develop's CI fails, or stability (`features.merge_queue.develop_guard.stability_label`), assigned to the breaking pull request's author, when no CI lane sees the break | In `features.merge_queue.develop_guard.issue_repository` (blank: the monitored repository), once a break on develop is proven, while the merge queue is live with its develop guard |
| **Team-channel message** | A card about a CI-red develop break with no fix planned, asking who can take it; and the Needs-you decisions, each once. Each card names the deployment; two messages of the same alert are at least `features.alerts.message_cooldown_minutes` apart, and what is new in between goes into the next one | The team channel, while the alerts are live (`features.alerts`) |
| **Follow-up ticket** | One ticket for each non-trivial 🔴/🟠 of ours still open on develop after its pull request merged, and for each agreed follow-up item: assigned to the pull request's author, linked to the merged pull request and to our review comment. Only posted reviews count: the author never saw a kept one | The JIRA project `<PROJECT>` (`features.ticket_creation.project_key`), through the atlassian CLI, linked to the epic when one is set (`features.ticket_creation.follow_up_tickets`) |
| **Proposed ticket** | A ticket for a problem found along the way that is neither a develop break nor a pull request's finding | The admin dashboard's "Tickets waiting for you" card: Approve files it, Decline drops it (`features.ticket_creation.found_issue_tickets`) |
| **Overlap notice** | On a pull request that adds code another open pull request owns: both copies with `file:line`, the owner and the resolution, the landing order the merge plan suggests, and the rebase that removes the copy ([section 4.10](#overlapping-work)) | On the pull request, once per overlap group, while review automation is live (the owner decisions are made in prototype too) |
| **Drift notice** | On a pull request whose own tests fail on its head merged with today's develop, and pass on develop alone: the failing tests, the develop commit that changed the behavior, and the fix ("merge develop and update the tests") ([design.md, "Merge gates"](design.md#merge-gates)) | On the pull request, once per develop commit, while the merge queue is live |
| **Merge proposals and approvals** | `PROPOSE #A → #B`: merge this validated batch, in this order. `READY_FOR_APPROVAL #N`: only the operator's approval is missing. One "ready for approval" box per code-owner team: our review is current and clean, every thread is resolved, and that team's approval is what is missing. `rk approvals`: what is left on each pull request close to merge, and who acts, in merge-plan order | Tick lines, the board and the dashboards |
| **Priority requests** | A "Request PR priority" box at the top of the merge plan. On the admin dashboard, the operator's request takes effect at once, and the operator approves or declines other teams' requests there. On the team dashboard, anyone can ask, with their name and a reason, and the request waits for the operator ([section 4.9](#priorities)) | The admin and team dashboards; the board's Needs you list; `rk priority-requests` |
| **Insights** | Merge-conflict hot files, CI time per pull request, our findings by cause, and where the time goes per pull request: rebases, conflicts, review and fix rounds, CI runner waits | `rk insights`, `rk conflict-files`, `rk ci-times`, the dashboards, the weekly report |
| **Review spend and usage** | What each review cost, posted, kept or closed, by part, per day and week, with outliers; the tokens every model used | `rk spend` and the admin dashboard's Review spend card, with review spend on (`features.review_automation.spend`); `rk usage`, in tokens, always |
| **Scoreboard** | Per pull request author: pull requests merged, rework caused for others (rebases forced, re-reviews), develop breaks, pull requests landed ahead of the plan, merges with open review items | `rk scoreboard`; the admin dashboard only |

**The status board**, row by row:

| Row | Shows |
|---|---|
| Monitor | Whether the monitor is on, each feature's mode, the pull requests cleared for auto-merge, and the day's idle and full ticks |
| Reviews | The parallel reviews running, the drafts awaiting verification, what is waiting (follow-ups held by the gap among them), the heads carried as merge-only, and the pull requests too large for the review driver |
| Review throughput | The oldest and median wait; the reviews cleared against the pull requests and pushes arriving in 24 hours; the trend. It says LOSING once the backlog passes 20 reviews and keeps growing: then add capacity |
| Issues avoided (7 d) | The saves by type: findings posted, our findings fixed before merge, develop breaks caught, merges blocked by a gate, drafts corrected |
| Merge queue | The queue head and its state; the ready, candidate and held pull requests; the closest pull requests and the gate blocking each; the merges today and in 7 days |
| Merge plan | Wave 1 and who acts on it (the operator, a code-owner team, us, the authors, CI), with any top priority that lands after its rebase; the hubs, the number of waves, "ABI in turn" when a lane has two planned claims, and how many pull requests must rebase first. It changes only when the plan does |
| Holds | The held pull requests, when there are any |
| Develop guard | develop's CI, the last green sanity run, the open incidents, the open develop issues by label |
| Top priority (or Develop fixes) | The pull requests that fix a develop break and those the operator prioritized, what is left on each, and who acts |

Ages, rolling counts and the count in a LOSING trend do not count as changes. Under the board, **Needs you** lists in plain words what only the operator can decide: merge a validated pull request, or clear it for auto-merge (the line gives the command); approve the pull requests only the operator's approval would land; an auto-merge that risks a gate or would land ahead of the plan; a hold that has gone stale or lost its change request; a revert offer; a pending priority request or proposed ticket; the in-depth review of a pull request too large for the autopilot, with its size, agents and cost (Approve and Decline on the admin dashboard, or the commands the line gives; the team dashboard and the team channel get its size and agents alone, as the cost stays on the admin side); more review capacity. Then come the "ready for approval" lists, one per code-owner team. With the alerts live, each decision also goes to the team channel once.

<a id="review-comment"></a>
**The review comment.** Every review is one plain comment per pull request head:
- line 1 is `Automated review by <automated_reviewer_name>: Model - <model>, effort: <effort>`, naming the first review pass's model when it drafts, else the verification pass's model and effort;
- the header says "First-round" or "Follow-up", never ACCEPT or REJECT, and carries a traffic light: the worst open Needed-to-merge severity, prior rows included;
- prior items come first, each with a verified status: every other reviewer's item, the bots', and our own earlier items (FIXED with its commit, OPEN, DEFERRED with the author's reason and our judgement, or ⚪ RETRACT for one of ours that was wrong);
- every claim is marked `(ran)`, when a check at this head shows the run, or `(read)`;
- cells are compact point form: the light and a status word, then one to three short points.

The verification pass's team member file holds the template ([Output Format](../agentic/team_members/rocke_review_verifier.md#output-format)).

| Light | Meaning |
|---|---|
| 🔴 blocking | Wrong results on a reachable path, a crash, a build or CI break on merge, a security issue, or absolute performance numbers in public history |
| 🟠 should-fix | A real defect or a significant gap |
| 🟡 minor | Minor, but still Needed to merge or a significant follow-up |
| 🟢 info, good | Nothing open; fixed items and verified checks |
| ⚪ retracted | One of our own earlier items was wrong |

In the follow-up table a light marks importance only: it never sets the header light.

---

## 4. The parts, in brief

Each part in one paragraph, with a link to where it is told in full.

### 4.1 Architecture: the parts

The toolkit (`rk`) does every mechanical step and holds the only write path: a fixed set of guarded `rk` commands. The orchestrator, one long-lived Claude session (`rk orchestrate`), judges and acts only through those commands. The tick triage, the review driver, the verification and code quality passes, the optional passes, the scaler and the dashboards each do one job beside them, and in the container the entry point keeps the parts the features need running. Each part, what it does and what it never does: [design.md, "Architecture"](design.md#architecture) and ["The deployment"](design.md#the-deployment).

### 4.2 The tick

The tick loop runs `rk tick` every `monitor.tick_interval_minutes` minutes. A tick with nothing new is idle: one line, and no Claude turn. A full tick polls GitHub with reads only, recomputes the merge queue and the merge plan, advances each feature that is on, and prints its lines; the tick triage hands the session only the lines that need Claude. The steps: [design.md, "The tick"](design.md#the-tick); each line, and what the session does with it: the skill's tick card ([SKILL.md](../agentic/skills/rocke-monitor/SKILL.md)).

### 4.3 The review pipeline

Once its head has settled, each picked pull request is staged, drafted by a first review pass when there is one, and built and tested merged with develop. The verification pass then reviews it blind, with the code quality pass beside it, and `rk spec-post` posts one comment, or keeps the review unposted ([section 3](#3-what-information-it-produces) says when). A rebase-only head, and a merge-only head whose checks are clean, keep our verdict with no new review; a pull request over the in-depth limit gets one in-depth review, split into parts, once the operator approves it. One pull request from start to finish: [review_pipeline.md, 3.6](review_pipeline.md#36-one-pr-start-to-finish); the stages and the rounds: [design.md, "Reviews"](design.md#reviews); what a post says: [design.md, "What a post says"](design.md#what-a-post-says).

### 4.4 The merge queue and gates

The merge queue gates every candidate at its head on today's develop with the gates G1–G9; the gates, how long their results stay fresh, and the drift check that runs a waiting pull request's own tests again when develop's rocKE code moves, are in [design.md, "Merge gates"](design.md#merge-gates). Ready pull requests that the merge plan lets land now are batched and proposed to the operator; with the merge queue live and auto-merge on, the pull requests the operator cleared are armed for auto-merge, one at a time. What each feature does in prototype and live: [design.md, "Features and modes"](design.md#features-and-modes); the procedure: [merge_procedure.md](../agentic/skills/rocke-monitor/merge_procedure.md).

### 4.5 The develop guard and incidents

With the develop guard on, in prototype too, every develop move that brings rocKE commits gets a sanity run. A red run, or develop's CI red in a rocKE job, opens an incident and freezes the queue. The monitor finds the culprit, and, while the merge queue is live, files one develop issue and links the fix; reverts are the develop-green team's call, and the monitor pushes none. In full: [design.md, "The develop guard"](design.md#the-develop-guard) and ["Incidents"](design.md#incidents); the procedure: [incident_procedure.md](../agentic/skills/rocke-monitor/incident_procedure.md).

### 4.6 The auto-merge guard and holds

The auto-merge guard turns off someone else's auto-merge on a pull request that fails a gate GitHub does not enforce, with one plain-words comment (in prototype, it only alerts); auto-merge the operator turned on only gets an alert. With the merge queue live and `features.merge_queue.auto_hold.enabled`, a pull request whose gate fails by its own changes is held, with a change request that only the operator's re-approval lifts. A merge over a hold or an open 🔴/🟠 is reported at once, and what is left of our posted findings becomes a develop issue or a follow-up ticket. The rules: [review_pipeline.md, 7.6](review_pipeline.md#76-auto-merge-rules) and [Appendix B](#appendix-b-policies-the-operator-keeps), policies 2, 18 and 24; the procedure: [merge_procedure.md](../agentic/skills/rocke-monitor/merge_procedure.md).

<a id="lessons-loop"></a>
### 4.7 The lessons loop

Every agent reads one lessons file in the skill, the verification and code quality passes through a per-role extract (`lessons_for.py`). A lesson is written the day it is learned; with the weekly report on, the week's corrections are reviewed and exact edits proposed, and a lesson that recurs becomes a toolkit guard with a test ([Appendix B](#appendix-b-policies-the-operator-keeps), policy 22). The procedure: [weekly.md](../agentic/skills/rocke-monitor/weekly.md).

<a id="merge-plan"></a>
### 4.8 The merge plan

Every full tick plans the near-merge pull requests in conflict-free waves on today's develop: two pull requests conflict when git cannot merge them together, or when both bump the same ABI counter. The priorities (pinned groups, top priorities, held pull requests, and ready ones the plan has kept waiting a day) each join the earliest wave with no higher priority they conflict with, and hubs come last. Wave 1 is what can go in now, and every planned pull request gets rebase advice. In full: [design.md, "The merge plan"](design.md#the-merge-plan); the readiness weights and the procedure: [merge_procedure.md](../agentic/skills/rocke-monitor/merge_procedure.md).

<a id="priorities"></a>
### 4.9 Priorities and develop fixes

A develop fix, a pull request that fixes a develop break, is a top priority by itself, and the operator makes any other pull request one with `rk prioritize N`. A top priority is reviewed first, gets its merge analysis first, and leads the merge plan ([Appendix B](#appendix-b-policies-the-operator-keeps), policy 16). Every full tick's fix sweep looks for develop fixes among the open pull requests, and Claude verifies each before linking it ([incident_procedure.md](../agentic/skills/rocke-monitor/incident_procedure.md)). Other teams ask with the "Request PR priority" box on the team dashboard ([The team dashboard](../README.md#the-team-dashboard)), and each request waits for the operator's Approve or Decline on the admin dashboard (policy 25; how requests are taken: [Guardrails](#5-guardrails), "Dashboard services"). A priority is a local decision: state and journal, with no GitHub write.

<a id="overlapping-work"></a>
### 4.10 Overlapping work across PRs

When open pull requests add the same code, under one name or several, one of them owns it, and the others drop their copy. The overlap scan lists such code in every review's checks ([review_pipeline.md, 3.11](review_pipeline.md#311-the-checks-scripted-sections)). The code quality pass proposes the owner: the pull request that puts the code in its natural shared home, else the one that lands first. The posted review records it (`rk overlap-decide`), and every other copy gets one overlap notice with both copies, the owner, the landing order and the rebase that removes the copy ([Appendix B](#appendix-b-policies-the-operator-keeps), policy 23; the procedure: [review_procedure.md](../agentic/skills/rocke-monitor/review_procedure.md)).

<a id="token-efficiency"></a>
### 4.11 Token efficiency and review spend

Scripts do the polling, ordering, gates, builds and tests, so an idle tick costs no Claude turn, and Claude's tokens go to judgement and verification, which run in agents of their own. What keeps the cost down: [review_pipeline.md, 2](review_pipeline.md#2-why-it-is-built-this-way), item 11, and [design.md, "Token economy and review spend"](design.md#token-economy-and-review-spend). `rk usage` counts every model's tokens; with review spend on (`features.review_automation.spend.enabled`), `rk spend` and the admin dashboard's Review spend card price each review, posted, kept or closed, by part ([review_pipeline.md, 3.12](review_pipeline.md#312-the-orchestrators-steps-large-reviews-drift-and-review-spend)).

---

## 5. Guardrails

- **Comment only.** The monitor never approves, dismisses reviews, resolves threads, edits other people's comments, or changes the operator's review states. Its only review state is a change request on a confirmed hold.
- **One write path.** Reads are GETs under a read-only login: the toolkit's GitHub module refuses write methods and GraphQL mutations. Every write is a guarded `rk` command, checked against a ledger first, so nothing is done twice; a write whose answer was lost is read back before it is called failed. Each write command belongs to a feature and sends only while that feature is live (or, for a feature with no mode, on); nothing goes out in prototype, in standby, or while the monitor is stopped.
- **The read/write split.** The write login is used only by the guarded write commands. In the container it lives in a volume of its own, which only the writer reads, and the write commands run as the writer through a root-owned broker that runs the writer's commands alone (`rkmon_lib/outward.py`'s table: the GitHub writes, and the sends to the team chat, the web host and JIRA that need the writer's secrets). Every tick probes both logins at most once an hour ([README.md, "The GitHub CLI"](../README.md#the-github-cli)); a read login that can write is a standing `!! GH_LOGIN` alert. On a host install both logins belong to one Linux user, and PR code runs as that user: the split holds by convention only. For the enforced split, run the container.
- **The session's guard.** A hook on every shell command of the Claude session and its agents refuses outward actions that are not `rk` commands: GitHub writes through gh, pushes to a git remote, network writes by curl or wget, ssh, scp, Slurm commands, and the atlassian CLI's own writes unless follow-up tickets are on (`rk atlassian` weighs each of its writes itself, on each call). It refuses the `rk` commands only the operator or rk itself runs (`rkmon_lib/outward.py`'s table), after `rk`, `rkmon.py` or the writer's entry point: a proposed ticket filed or declined, the test card, a writer's secret set, the read login set, a host's own keys read or recorded, the request service installed on the web host, the end of standby, a raw team chat message, the team dashboard's push and the web host's requests read. It also refuses what would change state that the reviews' agents and jobs share.
- **Pull request code isolation.** Every command that builds or runs a pull request's code runs as a PR runner, in a slot of its own, which reads neither secrets volume and writes only its slot's folders.
- **Post guards.** `rk post` refuses:
  - a wrong line 1, links outside an allow-list of public hosts, absolute performance numbers, or a linked ticket tag;
  - a header with the wrong pull request or round, a second post at one head (unless a re-review), or a moved head;
  - more than `max_posts_per_hour` posts an hour.
- **Machine limits.** The monitor keeps the host under its load ceiling (`max_host_usage_pct`); nothing starts that does not fit, and nothing spills from the RAM disk to the disk. A pristine tree and a fresh build for every run. Never install packages on the bare machine.
- **Dashboard services.** Every request to the admin dashboard carries the access token; where it listens and who can reach it is the README's [The admin dashboard](../README.md#the-admin-dashboard). The team dashboard has no operator controls and no token. Its request service (`tools/webhost_server.py`: on the web host, or in the monitor with delivery serve) takes at most five requests an hour from one address, and 100 a day in all, and the page tells the requester why a request failed. A request changes nothing until the operator approves it: a second request for the same pull request joins the first, and one for a pull request that is already a top priority, or that is not an open pull request the monitor tracks, is answered at once.
- **Recoverable.** Every stage is idempotent and on disk, so a restart continues where it stopped. The ways to pause and stop it, from pausing new work to stopping merges only: [docs/restart.md, "The four stop levels"](restart.md#the-four-stop-levels).

| The monitor may, on its own (with the feature live) | Only on the operator's instruction | Never |
|---|---|---|
| Post review comments and CI updates | Clear a pull request for auto-merge; prioritize; pin a merge group | Approve, or dismiss a review |
| Request changes on a confirmed hold | Change a feature or its mode; end standby; request changes outside a hold | Merge by itself (it only arms auto-merge, for cleared pull requests) |
| File, comment on and relabel develop issues once a break is proven; assign a stability issue to the breaking pull request's author | Assign issues to anyone else; create labels; post revert evidence | Bypass branch protection or use admin merges |
| Link a verified fix pull request to its issue | Send a test card to the team channel | Edit or delete other people's comments |
| Turn auto-merge off on a pull request that fails a gate, when someone other than the operator turned it on (the operator's own auto-merge only gets an alert) | Turn someone's auto-merge off to land other pull requests first (`--user`) | Change the operator's own review states |
| Tell the team channel about a CI-red break with no fix; send the Needs-you decisions | File a proposed ticket (the operator's Approve on the admin dashboard) | Write to JIRA outside its ticket features |
| Arm auto-merge for a cleared, validated pull request; start an in-depth review the operator approved | Start gate jobs by hand; approve or decline an in-depth review | Ticket CI runner infrastructure (another team owns it) |
| Post overlap, drift and GPU-test notices; record the owner a posted review proposed | Approve or decline another team's priority request | Push to a git remote |
| File follow-up tickets for our should-fix items still open on develop after a merge, assigned to the pull request's author | Update the monitor, or move it onto a new image (`rkmon package update`, `rkmon image load`, `rkmon confirm`, `rkmon apply`) | Put a secret in a file the package, git or a log holds |

---

## 6. Benefits

- **Breaks are found sooner, and owned:** one sanity run after the merge, then one issue, a culprit and a fixer.
- **Pull requests are judged as they would land,** so semantic merge breaks, stale goldens and ABI collisions surface before the merge.
- **Waiting pull requests stay tested:** their own tests run again whenever develop's rocKE code moves, so a green pull request cannot quietly go stale while it waits for approval.
- **Reviews are consistent and evidence-based,** scope first: the author sees every blocker at once, and nothing out of scope blocks. The operator's attention goes to a short Needs-you list instead of every open pull request.
- **Knowledge survives a handover:** it lives where the agents read it, and the package sets up a new deployment from an image and a deploy settings file.

**Where to read the numbers.** The monitor measures its own effect from its records. Only the Scoreboard names people, per pull request author: `rk scoreboard` prints it, and of the dashboards only the admin dashboard shows it.

| Measure | Where |
|---|---|
| **Issues caught ("saves")**: findings posted, earlier findings confirmed fixed before merge, merges blocked by a gate, develop breaks caught, drafts corrected. A finding counts once when posted and again when fixed, so this counts records, not distinct issues | The board's "Issues avoided (7 d)" row; the daily logs |
| **Reviews posted**, by round and light; **time to our first review** | The weekly report (`rk stats`) |
| **Review throughput**: reviews cleared against pull requests and pushes arriving, the oldest and median wait, the trend | The board's Review throughput row |
| **Merges** per day and week | The weekly report; the board's Merge queue row |
| **Findings fixed** | The daily logs' saves (fixed); the weekly report |
| **Outward actions**, all ledgered | `rk ledger` |
| **Cost per review** | `rk spend`; the Review spend card |
| **Rework per author** | `rk scoreboard`; the Scoreboard card |

## 7. Limits and known gaps

- **Capacity.** The parallel reviews and parallel builds stay within their ranges and the container's CPU and memory limits. When pushes outpace the reviews and the backlog keeps growing past 20 reviews, the board says LOSING: raise the resources or the limits.
- **A first draft is a lead generator.** It is checked claim by claim, and the verification pass's blind review comes first, so a weak draft costs time, not quality.
- **What it cannot see.** Arch coverage is mostly evidence, not execution: the container has no GPU, CI's GPU lanes have no PyTorch, and some architecture lanes skip on pull requests. GPU-only cases run only with remote GPU tests on (`features.review_automation.remote_gpu_tests.enabled`).
- **It shortens breaks; it does not prevent them.**
- **Advisory.** People can merge over an open 🔴. The monitor detects this, but cannot prevent it.
- **The merge plan is advice.** GitHub lets an approved, green pull request merge in any order, and auto-merge can land a pull request ahead of the plan. The monitor reports it and plans again.
- **Overlap detection reads names and token sketches.** The same logic, restructured, can slip through, and every lead needs a reviewer's judgement.
- **Installed-tree gaps.** The installed-tree check does not cover a change to test helpers or `conftest.py` alone, or an install-rule change without a test change.
- **Drift-check gaps.** The drift check runs only the pull request's own test files. A pull request without test files gets no drift check, and a break that only another test would show slips through.
- **In-depth reviews cost in proportion.** A split review costs about one review per part plus the cross pass; when that is far above a normal review's cost, the orchestrator says so before it starts, and the operator may skip it.
- **Kept reviews stay local.** A review kept unposted, in prototype or standby, gates the merge queue but never reaches the author, and follow-up tickets never come from it. With the merge queue live and review automation in prototype, the settings' checks warn that the queue gates on reviews the authors never saw.
- **Cost.** Verification uses paid Claude tokens; the two review passes are most of the spend ([section 4.11](#token-efficiency)).

<a id="documentation"></a>
## 8. Where the documentation lives

The README's [Documentation](../README.md#documentation) lists every doc, who reads it and its one job. The skill files and the package's code are the source of truth; this page, the design and the review book are refreshed from them. Where to find what this page only touches on:

| Topic | Where |
|---|---|
| What the build and deploy machines need, machine sizing, and every step from a build settings file to the first healthy tick | [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md); [QUICK_DEPLOY_GUIDE.md](../QUICK_DEPLOY_GUIDE.md) |
| Running it day to day, going live, and the common problems with their fixes | [QUICK_USER_GUIDE.md](../QUICK_USER_GUIDE.md) |
| Stopping and resuming | [docs/restart.md](restart.md) |
| Moving from an existing monitor | [QUICK_DEPLOY_GUIDE.md, "Moving from an existing monitor"](../QUICK_DEPLOY_GUIDE.md#moving-from-an-existing-monitor) |
| Where things live: on the host, in the container, the outputs folder | The README's [Where things live](../README.md#where-things-live) |
| The logins and secrets, and what each needs | The README's [Logins and secrets](../README.md#logins-and-secrets) |
| Every command; every setting | The README's [Command reference](../README.md#command-reference) and [Settings reference](../README.md#settings-reference); the internal keys in [docs/internal_settings.md](internal_settings.md) |
| The agentic files and how each start installs them | The README's [The agentic files](../README.md#the-agentic-files); [agentic/README.md](../agentic/README.md) |
| The review pipeline in depth: the passes, the parallel reviews, CI knowledge | [docs/review_pipeline.md](review_pipeline.md) |
| The design, and why | [docs/design.md](design.md) |

---

# Appendices

## Appendix A. Components reference

### A.1 Toolkit modules (`rkmon_lib/`)

`rk` runs `rkmon.py`, the command's dispatcher; the library lives in `rkmon_lib/`. The main modules:

| Module | Purpose |
|---|---|
| `store.py`, `state_store.py`, `config.py`, `paths.py` | The on-disk state, the journal and the ledger; the internal state and the unsent items; the configuration: loading it, the values derived from it, and its checks; every path in one place |
| `gh.py`, `ghlogins.py`, `gh_cap.py` | GitHub through `gh` under two logins: reads refuse write methods and GraphQL mutations; whether the read/write split holds; the cap on GitHub requests an hour |
| `gitops.py` | Fetches and read-only plumbing on the clone; merge simulation with `git merge-tree --write-tree` |
| `snapshot.py`, `events.py`, `idle.py` | One poll of the open in-scope pull requests and develop; events from the difference with the stored state; the idle tick's change fingerprint |
| `plan.py` | What the tick should do next, in priority order P0–P4 |
| `queue.py`, `mergeplan.py`, `batch.py`, `abi.py`, `merge.py` | The gates; the merge plan; merge batches; ABI lanes and G9; the merge queue: gate jobs, holds, proposals, arming, the auto-merge guard |
| `review.py`, `draft_stage.py`, `bindings.py` | The review pipeline: pick, classify, stage, draft (or none), checks, `REVIEW_READY`, and a review closed without a post (`rk classify N --record`, `rk review-close N`); the draft stage; the passes' model bindings |
| `scaler.py`, `ramdisk.py`, `review_keep.py` | The parallel reviews and parallel builds by load; the RAM disk and its learned footprints; the review folders kept on the disk |
| `peer_carry.py`, `known_infra.py` | A merge-only head's flags against what our review saw (carry, or a micro round); known CI infrastructure failures |
| `split.py` | `rk review-split`: an in-depth review cut into parts and a cross pass |
| `prep.py`, `dispatch.py`, `facts.py` | The review folder; `rk dispatch-pack` and `rk wait-quality`; `rk facts` |
| `spec.py`, `premises.py`, `fixline.py`, `abipremise.py`, `post.py` | The post's spec; posting only while the premises hold; the fix-line and ABI premises; every write, guarded and ledgered, and the reviews kept unposted |
| `verify_stale.py` | `!! VERIFY_STALE`: a review handed to verification with nothing posted or kept, naming `rk review-close N` for a head our account posted at already (a mirror's line names none) |
| `drift.py`, `guard.py`, `breakage.py` | The drift check; develop sanity runs, incidents and bisects; develop issues and the fix sweep |
| `overlaps.py`, `clashes.py`, `peer_checks.py` | Overlapping work across pull requests; clashes with other open pull requests; the peer-review checks |
| `orchestrator.py`, `triage.py`, `tickfilter.py`, `workguard.py` | The container's Claude session and its Bash guard; the tick triage; the tick lines the session answers; the Bash guard's shared-state half |
| `prcode.py`, `slots.py` | Pull request code in the container: as a PR runner, in a slot of its own; which job holds which slot, and the slot kept for a review |
| `features.py`, `outward.py` | Every capability, its feature and what it lacks; the outward commands, who may run each, and the writer's |
| `priority.py`, `tickets.py`, `webserve.py`, `publish.py` | Priorities and priority requests; proposed tickets; the admin dashboard's service; the team dashboard's delivery |
| `report.py`, `webdash.py` | The tick summary, the status board, the Markdown dashboard; the HTML dashboard in its two variants |
| `daylog.py`, `stats.py`, `lessons.py`, `insights.py`, `conflicts.py`, `ci_times.py`, `merges.py` | The daily log and saves; the weekly report; the lessons raw material; insights; conflict hot files; CI times; merges per day |
| `usage.py`, `spend.py`, `spend_scan.py`, `scoreboard.py` | Tokens per day and review; review spend from the transcripts; the per-author scoreboard |
| `gpu.py` | Remote GPU tests on a Slurm cluster |
| `cleanup.py`, `retention.py`, `jobs.py`, `procs.py`, `control.py` | Cleanup of merged or closed pull requests; bounded logs; detached jobs; the processes the monitor started; the stop levels |
| `package_update.py`, `adopt.py`, `one_time_move.py`, `deployment.py`, `image.py`, `doctor.py`, `reach.py` | `rkmon package update`; `rkmon adopt`; the one-time move; building an image and deploying a container, each a numbered path with its guards; the container image: its name, its bases and the package's version; `rk doctor` and its live checks |
| `provision.py` | The work volume set up from your confirmed settings at each start (`rk provision`) |
| `releases.py`, `code_volume.py` | The container's code: the image's own, or the release installed over it in `/code` |
| `privacy.py`, `agentic.py` | The privacy check; the agentic files as templates, filled in and installed |

### A.2 Commands and settings

Every `rkmon` and `rk` command, with its options and what it prints: the README's [Command reference](../README.md#command-reference), and `rkmon help` in a terminal. Every deploy and build setting, with its default: the README's [Settings reference](../README.md#settings-reference); every internal key: [docs/internal_settings.md](internal_settings.md).

<a id="state-files"></a>
### A.3 State files

All in the monitor's home's `state/`, `/work/monitor/state` in the container. Never committed.

| File | Holds |
|---|---|
| `state.json` | The source of truth: flags (enabled, frozen, cleared, the operator's priorities); develop (head, last green, CI, pending sanity); every pull request (head, our last review, posted or kept, and its verdict, covered heads, gates and drift results per head and develop SHA, holds); jobs; the queue, batches, ABI lanes and the merge plan with its caches; develop fixes; priority requests and proposed tickets; the review pipeline. Written whole under a lock |
| `internal_state.json`, `unsent.jsonl` | The internal state: what the monitor records for itself, such as the automated reviewer name and the hosts' own keys ([docs/internal_settings.md](internal_settings.md)); the unsent items, one per line |
| `journal.md` | Append-only, one line per decision, action or event |
| `actions.jsonl` | **The ledger:** every outward action with its key and URL, checked before acting; it never rotates |
| `writes_in_doubt.json` | A write whose answer was lost, until the next try reads it back |
| `events.jsonl`, `metrics.jsonl`, `daylog.jsonl` | Every event seen; one metrics snapshot per tick; daily-log entries, saves included |
| `incidents.json` | Develop incidents: SHAs, culprit, contributors, fixer, issue, times |
| `scaler.jsonl`, `scaler_state.json` | The parallel reviews' and builds' decisions with their reasons; their current numbers and learned footprints |
| `claude_usage.jsonl`, `review_cost.jsonl` | Claude's tokens per turn and agent; each review's pass costs |
| `spend_history.json` | Carried review-spend history of a monitor this one took over from (optional) |
| `scoreboard_events.jsonl` | The scoreboard's events |
| `followups/` | Per pull request, our follow-up items and their state |
| `overlap_cache.json.gz`, `overlaps_decided.json` | The definitions each pull request head adds; the recorded owner of each overlap group |
| `gh_logins.json`, `gh_requests.json`, `doctor_cache.json` | The last probe of the two GitHub logins; the GitHub requests of the last hour; the doctor's last live checks |
| `priority_requests.pulled.jsonl` | The priority requests read from the web host |
| `one_time_move.json` | The one-time move's record and report, after a move from an existing monitor |
| `orchestrator.json` | The Claude session to resume. One recorded on another host is dropped and a new one starts: a Claude conversation stays on the host that made it |
| `jobs/<id>/` | One background job: `cmd.sh` (its script), `log`, `pid`, `rc` (its exit code) and `meta.json` |

The append-only logs move to `<name>.1` past `retention.log_max_mb`; their readers read both.

### A.4 Review tools, toolkit helpers and the container's files

**Review tools** (the package's `worker/` folder, which each start installs into `/work/tools`): [worker/README.md](../worker/README.md) lists each one, what calls it and what it does.

**Toolkit helpers** (the package's `tools/`)

| Script | Purpose |
|---|---|
| `review_driver.py` | The continuous review driver behind `rk review-driver` |
| `tick_loop.py` | The orchestrator's tick loop: `rk tick` on its interval, waking the session for the lines that need Claude |
| `draft_run.py`, `quality_run.py` | The draft stage's job; the code quality pass's job on a model binding |
| `pytest_vs_base.py`, `installed_tests.py`, `importer_tests.py`, `devtests.py`, `ci_condition.py` | The pull request's tests against develop's; in the installed tree; the tests that import changed modules; develop's tests CI never ran with the pull request; the CI-condition run |
| `golden_vs_base.py`, `dg_summary.py`, `p_emit_compile.py`, `abi_check.py`, `abi_scan.py`, `tidy_check.py` | The golden gate as CI runs it; the differential summary; compiles of the touched families; the ABI history and scan; clang diagnostics the GCC build misses |
| `clash_scan.py`, `mutation_sweep.py`, `ctest_labels.py`, `lane_of.py`, `moved_refs.py`, `doc_refs.py`, `import_probe.py`, `peer_checks.py`, `winsim.py`, `since_review.py` | The checks' scripted sections |
| `ci_head.py`, `ci_attempts.py`, `ci_sha.py`, `prthreads.py`, `ghr.py` | CI at a head; failures an earlier attempt hid; CI at a SHA; threads since a date; gh under the read login |
| `run_probes.py` | Runs the saved probes again: exit 1 reproduces, 0 fixed, anything else is an error |
| `ctest_read_only.py` | ctest on a build it may only read (the review's shared build, in the checks' kept slot), through `rk pr-run`: ctest writes its own files in the command's slot |
| `worker_grace.py` | With a first review pass on a local model: no CPU-hungry work in its first five minutes |
| `rkweb.py` | The admin dashboard's service starter |
| `webhost_remote.py`, `webhost_server.py`, `rocke-monitor.service` | On the team dashboard's web host: the half of `rkmon web-host install` that runs there, the request service, and its systemd user unit |
| `make_package.py`, `publish_package.py`, `generated.py`, `doc_lints.py` | Building and publishing the package; the docs' generated sections; the docs' lints |

**The container's files** (the package's `docker/`): `rkmon` (the host's command), `Dockerfile` (with `pkgs.sh`, `param.sh`, `fetch.sh` and `masked.sh`), `compose.yaml`, `entrypoint.sh`, `install_rocm.sh`, `install_atlassian.sh`, `install_sdk.sh`, `claude_cli.py`, `rk_ids.py` (the deployment's user ids), `rk_layout.py`, `rk_root.py` (the start's root step), `rk_write.py` (the write broker), `rk_gh.py` and `rk_prcode.py` (the PR-code launcher).

### A.5 Skill and procedure files (`/work/claude-workspace/.claude/skills/rocke-monitor/`)

| File | Contents |
|---|---|
| `SKILL.md` | The lean tick card, reloaded every tick: the parts, where things are, the standing rules in short form, the tick procedure and its lines, the output format |
| `rules.md` | Every standing rule in full; who merges; follow-up tickets. Read at the session's start, on `resume`, after a compaction, and before any outward action not taken since |
| `commands.md`, `lane_commands.md` | Every subcommand and the settings schema; the commands the review agents use |
| `review_procedure.md` | The pipeline's stages, what the checks cover, verify / check / post, micro rounds, split reviews, overlap decisions and notices, isolation, manual controls |
| `merge_procedure.md` | The merge queue: gate jobs, do-not-merge labels, GitHub's own blocks, the auto-merge guard, the merge plan, batches, holds, ABI lanes, propose, arm, disarm, after a merge |
| `incident_procedure.md` | The develop guard: routing, sanity checks, tick lines, what Claude does at each step |
| `weekly.md` | The weekly report and the lessons review |
| `lessons.md`, `lessons_for.py` | The lessons file ([section 4.7](#lessons-loop)); the per-role extract |
| `offload.md` | The review offload's procedure |
| `../shared/` | The workspace's shared policies the skill reads |

### A.6 Team member files and agent types (`.claude/team_members/`, `.claude/agents/`)

| File | Pass | What it does |
|---|---|---|
| `rocke_review_verifier.md` | The verification pass | Reviews blind first, then audits the draft when there is one: evidence for every claim, findings disproved before they are kept, every prior item given a status, misses added, scope first (Needed to merge, then significant follow-ups), severity calibrated, the public-repository rules, the code quality pass folded in, overlap findings posted. Also the part review and the cross pass of a split review. Writes the post and its notes (with a `premises:` line) to files and returns a short summary |
| `rocke_code_quality_reviewer.md` | The code quality pass | Single responsibility, one home per fact, primitives and layers, interfaces from the caller's side, names, test-backed claims, root-cause fixes in both engines, against the repository's own standards; it also judges the overlap leads and proposes an owner. Writes `quality.md` |

**Agent types** (`.claude/agents/`), one per review agent: `rocke-review-verifier.md` and `rocke-code-quality-reviewer.md`. Each sets its model (`features.review_automation.passes.verification_pass.model`, `features.review_automation.passes.code_quality_pass.model`), its effort, no workspace instructions or memory, the agent's tools only, and its own compaction window. Its body tells the agent to load the team member file, and repeats the few standing rules an agent needs (GitHub reads only, through the read login; no writes; nothing guessed; nothing installed).

These files are filled in for the deployment from the package's `agentic/` folder: the README's [The agentic files](../README.md#the-agentic-files).

<a id="review-book"></a>
### A.7 The review book

The rubric of the first drafts that read it, in the agentic files (`review_book.md`): the draft on the configuration's `worker` model includes it whole in its system prompt, and the review offload's draft pass adds it after its own instructions. Without a first review pass nobody reads it, and the first review pass on a model binding uses the repository's rules card instead (`repository/<org>__<repo>/drafter_rules.md`). The verification pass writes the post in its own team member file's format, not the book's.
- §0–§2: what the draft reviews and delivers, and its non-negotiables (read-only, the public-repository rules, integrity);
- §3–§4: the standing request and the workflow: intake, the code at head, verifying prior items, the standard checks, the hunt for new findings (repository rules 1–11, rocKE specifics, the merge with develop, tests and CI hygiene), disproving before reporting, severity and scope;
- §5–§6: the exact draft format, table rules and status words, and the notes after it;
- §7–§8: persisting (the draft writes no file) and re-reviews;
- §9–§10: the draft's staged inputs, tools and limits, and the CI evidence it reads;
- §11: lessons and pitfalls, which mirror the lessons file's Worker section; where a rule there and an example elsewhere in the book disagree, the rule wins;
- §12: a self-check.

## Appendix B. Policies the operator keeps

These are the standing rules, in general form. They always hold.

1. **Comment only.** Never approve. Never request changes, except on a confirmed hold or on the operator's explicit instruction. Never dismiss reviews or edit other people's comments. Never change the operator's own review states.
2. **Holds request changes.** A confirmed hold gets a change request at once. The author fixes it and asks for a review again; only the operator's explicit re-approval lifts it.
3. **Verify everything.** Every claim is reproduced ("ran") or downgraded ("read") before anything is posted; the verification pass reviews blind before it reads any draft. Trim false findings, and add missed ones.
4. **Scope first.** A post starts from the pull request's stated scope. Needed to merge lists every blocker, each with its light, with no cap; significant out-of-scope items go under Highly recommended follow-up (not blocking), with a 🟠/🟡 light; nits, style and wording are left out. Correctness and breaking changes are never follow-ups.
5. **Post format.**
   - Line 1 gives the automated reviewer name and the first review pass's model, or, with none, the verification pass's model and effort. The header carries a light and "First-round" or "Follow-up", never ACCEPT or REJECT. Cells are compact point form.
   - Name the operator by login, never as "the user". No @-mentions.
   - For ABI changes, state develop's number and "takes develop + 1 when it lands", never a tentative lane number.
6. **Public repository.** No internal links or host names, no absolute performance numbers, bare ticket tags only, no GPU product names (gfx targets only). Measured ratios only where the repository's own rules allow them (pull request descriptions and qualifying result files): our comments describe performance in words or point at the pull request's own ratio. Counts and structural ratios are fine.
7. **Post through the premises.** Post only with `rk spec-post` (or `rk post --premises`), which reads the head, develop, the conflicts, CI and the threads again right before posting. A premise stop is a stop, not a new verification.
8. **Merge-only heads are carried, not reviewed again,** when the checks only repeat what our review saw; new items alone (threads, a title problem, others' new reviews or comments) make a micro round.
9. **Pull requests over the in-depth limit get one in-depth review**, which starts only once the operator approves it; every later round is a light follow-up, which needs no approval.
10. **Machine limits.** The monitor decides what runs at once; nothing that does not fit starts. Never install packages on the bare machine. With a model on a local Ollama server, never restart the Ollama service while a prompt runs.
11. **GitHub access.** Read through `rk` or `gh` under the read login. Write only through the guarded `rk` commands, after checking the ledger.
12. **Critical items first,** at the top of every reply, before any other work.
13. **Develop breaks are GitHub issues,** one source of truth, never tickets.
    - Labelled CI red when develop's CI fails; stability, assigned to the breaking pull request's author, when it does not.
    - File only breaks already on develop, and only after checking for an existing issue; file without asking once the break is proven.
    - Link every fix. Tell the team channel only about a CI-red break with no fix planned.
    - A problem caught before merge goes in the pull request instead.
14. **Other issues** get a proposed ticket only if none exists, filed on the operator's Approve. Toolkit bugs are fixed directly. CI runner infrastructure belongs to another team and gets no ticket.
15. **Ready for approval** means our review is current and clean, every thread is resolved, nobody else requests changes, and an approval is all that is missing; the dashboards list it under the code-owner team whose approval it needs. Never ask for a re-approval of a stale approval.
16. **Develop fixes first.** A pull request that fixes a develop break, or that the operator prioritizes, gets review P0 with a shorter settle, merge tier 0, its merge analysis first, and first place in the merge plan: the pull requests it conflicts with wait for it, and its auto-merge is expected.
17. **Fix sweep.** Match open pull requests against open develop issues every full tick. Verify each candidate before linking it.
18. **Auto-merge guard.** A pull request with someone else's auto-merge on that fails a gate GitHub does not enforce gets a plain-words comment, and auto-merge goes off (an alert only, in prototype). Auto-merge the operator turned on is never turned off: it gets an alert, and the operator decides; for a pull request in the merge plan's wave 1 whose only open gate is our 🟠, there is no alert either. Turning someone's auto-merge off for any other reason needs the operator's instruction (`--user`, with the reason).
19. **Do-not-merge labels** ([README.md, "Glossary and placeholders"](../README.md#glossary-and-placeholders) lists them) take a pull request out of merge candidacy until they are removed.
20. **Who merges.**
    - Arm only pull requests the operator cleared, one at a time, while every condition holds.
    - Only pull requests the merge plan lets land now (in wave 1, merging clean with develop, with nothing ahead of them conflicting) are batched and proposed.
    - Clearing, prioritizing, grouping and feature changes happen only on the operator's instruction.
    - Queue notices go out only while the merge queue is live, which the operator sets once the team has been told.
21. **Never block the loop on a question.** Ask in plain text and keep ticking.
22. **Lessons are written the day they are learned,** in `lessons.md`, not in chat memory. A lesson that recurs becomes a toolkit guard with a test.
23. **Overlapping work has one owner.** When open pull requests add the same code, the posted review proposes one owner, it is recorded, and every other copy gets one overlap notice with the landing order and the rebase that removes it.
24. **Follow-up tickets.** Our non-trivial 🔴/🟠 items still open on develop after their pull request merged, and agreed follow-up items, get a follow-up ticket, assigned to the pull request's author and linked to the pull request and our comment; a failing test or red CI is a develop issue instead, and nits get nothing. They are filed without asking once confirmed on develop, and only from posted reviews.
25. **Priority requests wait for the operator.** A request from another team changes nothing until the operator approves it. The monitor never approves or declines one by itself.
26. **Updates come from a shell.** `rkmon package update`, `rkmon image load`, `rkmon confirm` and `rkmon apply` are the operator's, never the Claude session's.

<a id="glossary"></a>
## Glossary

The terms the other docs share are in the README's [Glossary and placeholders](../README.md#glossary-and-placeholders). These are the terms only this page and the skill files use:

| Term | Meaning |
|---|---|
| **ABI lane** | A serialized ABI counter; numbers are assigned at landing |
| **ABI pair (Ⓐ)** | Two pull requests that bump the same ABI counter: a conflict in the plan, because the later one renumbers |
| **Agent; agent type** | A separate Claude instance the orchestrator starts for one task, such as a pass; its definition (`.claude/agents/`): model, effort, tools |
| **Batch; batch gate; prefix check** | Pull requests validated together; the job that merges them onto develop in order; checking each intermediate state |
| **Byte identity** | The C++ and Python lowering engines produce identical IR |
| **Candidate; ready** | Passes the review, approval and hold gates; passes every static gate |
| **Care** | A shared file that is no conflict (a golden shard, a CMake list): shown in the merge plan, never acted on |
| **Carried head** | A merge-only (or rebase-only) head recorded as covered by our review, with no post |
| **Clone** | The monitor's own blobless, shallow clone of the repository, `repo` in the work volume; every worktree is made from it |
| **Contributing merge** | A merge that played a part in a develop break the culprit caused |
| **Develop fix; fix candidate; fix sweep** | A pull request that fixes a develop break; a pull request that looks like one; the per-tick search for them |
| **Develop-green team** | The people who keep develop green and decide on reverts |
| **Develop sanity run** | The check of a new develop commit after rocKE merges |
| **Drift check** | A light job that runs a reviewed pull request's own tests on its head merged with today's develop, whenever develop's rocKE code moves, approved or not |
| **Dashboard file; Markdown dashboard** | `rocke_monitor.html` in the outputs folder, for reading offline; `rocke_merge_queue.md`, the board and the queue as text |
| **First round** | Our first review of a pull request; every later one is a [follow-up review](../README.md#glossary-and-placeholders) |
| **Flavor** | The LLVM IR dialect rocKE emits for a toolchain generation |
| **G1–G9** | The merge gates, worked out at a pull request's head on today's develop ([design.md, "Merge gates"](design.md#merge-gates)) |
| **Golden; differential golden** | Committed hashes of emitted IR, per case and flavor; the anchors CI compares IR against |
| **Hot spot** | A file many pull requests edit (a golden, the ABI header, the platform CMake); a change to it forces other pull requests to rebase |
| **Hub** | A pull request with 5 or more textual conflicts with other planned pull requests, planned last |
| **Installed tree** | The layout `cmake --install` produces, which CI's `rocke_pytest` runs; a test can pass in the source tree and fail there |
| **Ledger; journal** | The record of every outward action; the line-per-decision log |
| **Merge simulation** | `wt<N>m`: the pull request's head with today's develop merged in, where the checks run |
| **Micro round** | A merge-only head with new items our review did not see (threads, a title problem, others' new reviews or comments): the verification pass alone covers them |
| **Needed to merge; follow-up item** | A blocker within the pull request's scope, listed with no cap; a significant item outside it, not blocking |
| **Old red** | A develop CI failure on a commit from before an incident's merged fix, only in jobs that incident already saw fail (`DEVELOP_RED_STALE`): old news, not a new incident |
| **Overlap group; owner** | Code that two open pull requests both add; the pull request that keeps it |
| **Premise** | A fact a post depends on (head, develop, conflicts, CI, threads), checked again right before posting |
| **Probe** | A saved script that reproduces one of our open findings, run again in later checks |
| **Prompt cache TTL** | How long a model's cached context lives: long for the session, short for the review agents |
| **Propose** | Tell the operator to merge a validated batch, in order |
| **Readiness class** | T0 top priority, R0 ready, R1 approvals or threads left, R3 approved with fixes or our follow-up left, R4 further out |
| **Rebase advice** | Per pull request: no rebase, wait for another pull request and then rebase, rebase now, or renumber after another pull request lands |
| **Rebase-only; merge-only; substantive** | A push that changes only context, not the pull request's own lines (after a 🟢/🟡 review, golden re-records too); a push whose new commits only bring develop in; any other push |
| **(ran) / (read)** | Evidence from a run at this head, against evidence from reading code |
| **Review book** | The rubric of the first drafts that read it ([A.7](#review-book)) |
| **Review spend** | What each review cost, by part, priced from the Claude transcripts |
| **Save** | One record of an issue the monitor caught (a finding posted or fixed, a gate block, a develop break, a corrected draft); the board totals them as "Issues avoided" |
| **Scaler** | The part of the monitor that moves the parallel reviews and parallel builds with the host's load, one step at a time |
| **Settle; burst; gap** | How old a head must be before review; the longer wait after repeated pushes; the wait between follow-up reviews |
| **Slot** | A PR runner's folder for one job (`/ramdisk/slots/<runner>`): its results, build, temporary files and home, made fresh for each job |
| **Split review; part; cross pass** | An in-depth review: parts of at most `features.review_automation.large_prs.split_part_lines` changed lines, one verification pass each, and one cross-cutting verification pass that writes the post |
| **Tick triage** | What reads each tick before the session: the rule filter (`rk tick-filter`) |
| **Traffic lights** | 🔴 blocking · 🟠 should-fix · 🟡 minor · 🟢 info or good · ⚪ retracted |
| **Triage; release** | Checking whose fault a gate failure is, on develop alone; clearing a hold that was not the pull request's fault |
| **Wave** | A set of pull requests with no conflict between them, which land in its order with no rebase among them; a pull request of wave 1 may land now when it merges clean with develop and no pull request ahead of it conflicts |
| **Write broker** | The root-owned command that runs the guarded writes as the writer |
