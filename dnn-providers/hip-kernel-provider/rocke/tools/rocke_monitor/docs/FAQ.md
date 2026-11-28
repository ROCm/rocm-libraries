<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: questions and answers

Short answers to the questions people ask about the monitor, grouped by who asks them. Each answer links to the
section that has the detail. In the commands, `./rkmon` is the package's `docker/rkmon`, run from that folder on the
host where the container runs; `./rkmon rk ...` runs an `rk` command inside the container. "The operator" is the person
who runs a deployment; `<user_login>` is the GitHub login the monitor works as (`integrations.github_cli.username`).

- [For people deploying and operating it](#for-people-deploying-and-operating-it)
- [For PR authors receiving its reviews](#for-pr-authors-receiving-its-reviews)
- [For team leads](#for-team-leads)

## For people deploying and operating it

### Can I build on one machine and deploy on another?

Yes: building an image and deploying a container are two jobs, and one machine can do both. The image travels as a
file (`./rkmon image save`, then `./rkmon image load`) or through a registry (`./rkmon image push`, then
`./rkmon image pull`). The deploy machine builds nothing: it needs Docker with compose, the network to GitHub and
your Claude gateway, and, behind a network that inspects TLS, an image built with that network's root certificate.
See [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md) and [QUICK_DEPLOY_GUIDE.md](../QUICK_DEPLOY_GUIDE.md).

### What is a build parameter, and what is a deploy setting?

A build parameter shapes the image and lives in the build settings file (`kind: build_settings`): the image's name,
its base image and CA certificates, and the tools it installs; a change there is a new image. A deploy setting shapes
one deployment and lives in the deploy settings file (`kind: deploy_settings`), in these parts, in this order:
`container`, `image`, `updates`, `repository`, `host`, `monitor`, `features`, `integrations`, then `model_bindings`.
Each key of both files: [README.md: Settings reference](../README.md#settings-reference).

### Where does the ROCm installer come from?

From AMD's public folder of ROCm runfile installers: the build takes the newest build of `rocm.version` there (or the
build it names, such as `10.1.0-5`). Set `rocm.package_url` only when your network keeps its own copy, with its
`rocm.package_sha256`; the build passes that address as a build secret, so it is in no image layer, history or label.
Every image carries its own ROCm. See [README.md: Build settings](../README.md#build-settings).

### Which starting template do I need?

`./rkmon template deploy --profile NAME FILE` writes the deploy settings template with its features set for one
type of deployment. A profile is never a setting: once the file is written, its features decide, and step 3
(`./rkmon confirm`) checks their combination. Each profile, and what it runs:
[README.md: Starting templates](../README.md#starting-templates); beside another deployment of the same repository:
[Can two monitors share a repository or a host?](#can-two-monitors-share-a-repository-or-a-host).

### What is `integrations.github_cli.username`, and who is "the operator"?

The operator is the person who runs a deployment and makes its decisions: which features run and when each goes
live, merges, priorities, clearing PRs for auto-merge, approving proposed tickets. `integrations.github_cli.username` is
the GitHub login the monitor posts as and whose instructions it follows; the write login (`./rkmon login github_write`)
is that account. The monitor never turns off that login's own auto-merge, and never touches its review states. See
[README.md: Settings reference](../README.md#settings-reference).

### What do prototype and live mean, and when do I go live?

Five features run in prototype or live: review automation, the merge queue, the alerts, the team dashboard and ticket
creation. In `prototype`, where every template starts, a feature does its work and keeps what it would post, send or
file, under the admin dashboard's "Not sent"; in `live`, it sends what comes next, and no PR author ever sees what it
kept. After a day or two, go live feature by feature: [QUICK_USER_GUIDE.md: Go live](../QUICK_USER_GUIDE.md#go-live).
What goes out in each: [README.md: Prototype, live and standby](../README.md#prototype-live-and-standby).

### How big a machine do I need?

The container's limits are deploy settings (`host.resources.cpus`, `host.resources.memory_gib`), and so are the
numbers of parallel reviews and parallel builds it starts with; the monitor raises or lowers those as the host's load
allows, so small limits only slow it down. A RAM disk (`host.resources.ramdisk_gib`) makes reviews and builds faster.
Step 2 checks this host against the sizes in
[QUICK_DEPLOY_GUIDE.md: Machine sizing](../QUICK_DEPLOY_GUIDE.md#machine-sizing).

### How do I open the admin dashboard?

A deployment with `features.dashboards.admin.enabled` false has none. The address to open, the SSH tunnel from
another machine, and the access token the browser asks for (any user name, the token as the password):
[README.md: The admin dashboard](../README.md#the-admin-dashboard).

### Which model and what effort does each pass use, and how do I change them?

`./rkmon rk roles` prints each pass with its model, provider and effort, resolved. The orchestrator
(`features.review_automation.orchestrator`) and each pass under `features.review_automation.passes` have a `model`
(`opus`, `sonnet`, `haiku` or a `model_bindings` name) and an `effort`. The four optional passes are `default` until
you name a model: the pass they help does their work. Each pass, its key and what it does:
[README.md: The two model roles](../README.md#the-two-model-roles).

### How do I use my own model for a specific pass?

Add a `model_bindings` entry for your model's `provider`, install its key if it has one, and name it at the pass:
`features.review_automation.passes.ci_triage_pass: { model: local-llama, effort: default }`. Only Claude runs
`verification_pass` and `code_quality_pass`; any model may take `first_review_pass`, `ci_triage_pass`,
`first_code_quality_pass` or `notice_draft_pass`, and Claude checks its leads. The binding, its key and the change:
[QUICK_USER_GUIDE.md: Use another model for a pass](../QUICK_USER_GUIDE.md#use-another-model-for-a-pass).

### What if the start refuses the local review loop's model?

The doctor's reviewer row gives the cause and its fix. HTTP 401 or 403: the key set again (`./rkmon secret NAME`, NAME
the binding's `api_key_secret_name`), then `./rkmon stop --drain` and `./rkmon start`; or the binding's
`api_key_sent_as` and `api_key_header`, then `./rkmon confirm` and `./rkmon apply`. A server error, or a model it does
not serve or that cannot call tools: the server put right, or the binding. A rate limit (HTTP 429) or no answer refuses
nothing: a later stop and start check again. See [README.md: The two model roles](../README.md#the-two-model-roles).

### My user id is below 1000. What now?

The monitor's user and the GitHub writer each need an id of 1000 or more (and not 65534), each their own and none of the
PR runners' (the package's, `rkmon_lib/container_layout.py`). Set `host.users.monitor_uid` to a free id of 1000 or more,
then run step 3 again. See
[QUICK_DEPLOY_GUIDE.md: Step 3: confirm](../QUICK_DEPLOY_GUIDE.md#step-3-confirm).

### What if a token or login expires?

Run `./rkmon login`: it shows each login and secret your settings need, with the command that sets it, and on a
terminal offers to run each one that is not set or is wrong. `./rkmon doctor` fails a login that has expired, with its
fix. `./rkmon login claude` says when the Claude session reads the new key, and the atlassian CLI renews an expired
access token itself. Each login and secret: [README.md: Logins and secrets](../README.md#logins-and-secrets).

### Why does ticket creation say `NOT SET UP`?

Follow-up tickets and out-of-scope issue tickets need `features.ticket_creation.project_key` and
`integrations.atlassian_cli.jira_url`, and, once ticket creation is live, the atlassian CLI logged in
(`./rkmon login atlassian`); every image carries the CLI. `./rkmon rk features` names what is missing. See
[QUICK_USER_GUIDE.md: Turn on a feature](../QUICK_USER_GUIDE.md#turn-on-a-feature).

### Is PR code isolated from the secrets?

In the container, yes. Every run of PR code (builds, tests, probes, the checks' re-runs, `rk pr-run` by hand) runs as a
PR runner, in a slot of its own made fresh for each job. It reads no secret, and of the monitor's files only the trees
made for it (the clone, the worktrees; in the scratch folder, the folders `rk pr-tree` made) and what the monitor hands
its job; it writes only its slot. Only the guarded `rk` write commands reach the GitHub write login, through a broker.
See [docs/design.md: GitHub access](design.md#github-access).

### Can two monitors share a repository or a host?

Yes. On one host, each needs a package folder, a `container.name` and ports of its own, which `./rkmon check` and
`./rkmon start` check. No deployment reads another, so on one repository the rules are yours: one runs the merge queue
and the others use the `review-only` profile, each with an automated reviewer name of its own; a merge queue gates only
on its own deployment's reviews, so with none it waits for people's approvals. See
[QUICK_DEPLOY_GUIDE.md: Several deployments](../QUICK_DEPLOY_GUIDE.md#several-deployments).

## For PR authors receiving its reviews

### What is this review, and who posts it?

An automated review, posted as one comment from the operator's account, never an approval or a rejection. Line 1
gives the name of the monitor that posted it and the model that drafted it (or the verifier, when nothing drafted
first): `Automated review by <automated_reviewer_name>: Model - <model>, effort: <effort>`. Scripts build and test
your PR merged with today's develop, and Claude checks every claim before it is posted: each is marked `(ran)` when
a check ran it at your head, else `(read)`. See [docs/review_pipeline.md: 6.3 The post](review_pipeline.md#63-the-post).

### What do "Needed to merge" and "Highly recommended follow-up (not blocking)" mean?

**Needed to merge** lists every blocker within your PR's scope, each with its light: wrong results, a build, CI or ABI
break the PR causes, a regression in code it touches, a test it adds that cannot fail, a false claim in its body or
commits. **Highly recommended follow-up (not blocking)** lists significant items outside its scope: fix them, decline
them with a reason, or leave them (an unfixed one may become a follow-up ticket for you after the merge). Lights: 🔴
blocking, 🟠 should-fix, 🟡 minor, 🟢 info or good. See [docs/design.md: What a post says](design.md#what-a-post-says).

### How should I answer the review?

On the PR, in one comment, item by item by its number (N1, N2, F1): the commit that fixes it, or your reason when you
think no change is needed or it belongs in a later PR. Then push your fixes. You do not need to ask for a re-review: a
follow-up starts by itself once your new head has settled and the follow-up cooldown has passed, and it checks every
earlier item at the new head, your answers included. See
[docs/review_pipeline.md: 6.2 Follow-ups](review_pipeline.md#62-follow-ups).

### Why does the review list other reviewers' and Copilot's comments?

So one table shows everything still open on the PR. The first table has a row for each earlier item, ours, other
people's, Copilot's and relevant bots', each with its status (fixed, open, declined with a reason), followed by the CI
row, our checks and the code quality pass. Unresolved review threads also block the merge gate. See
[docs/review_pipeline.md: 6.3 The post](review_pipeline.md#63-the-post).

### Why does the review flag pull-request references, timings or internal links in my PR?

The repository is public. Pull-request references (`#NNNN`, `PR #N`, `/pull/N` links) and dated stories of particular
runs that a PR adds to committed files are a 🟡 Needed-to-merge item, every file:line listed: the committed text states
the standing fact. So are internal links, which the repository's `platform/AGENTS.md` keeps out. Absolute performance
numbers are 🔴, blocking: git keeps them in the public history even after a later commit removes them. See
[docs/review_pipeline.md: 7.3 Public-repository hygiene (every post)](review_pipeline.md#73-public-repository-hygiene-every-post).

### What is an overlap notice, and who lands first?

Two open PRs add the same code: the same name, or a near copy under another name. The review proposes one owner for
each shared piece (the PR that puts it in its natural shared home, else the one that lands first), and each other PR
gets one overlap notice: both copies with file:line, the owner, the landing order, and the rebase that removes its
copy. Two PRs that bump the same ABI counter conflict too: the later one to land renumbers. See
[docs/review_pipeline.md: 3.11 The checks' scripted sections](review_pipeline.md#311-the-checks-scripted-sections).

### The review says CI is red, but my checks are green. Why?

The review reads CI at your head when it runs, the earlier attempts included; a check that passed later shows in the
next round. A cancelled run whose check also passed does not count against the PR at the merge gate. See
[docs/design.md: Merge gates](design.md#merge-gates).

### Does it ever approve or merge my PR?

It never approves a PR, dismisses a review or merges directly. With the operator's merge queue live and its auto-merge
enabled, it turns on GitHub's own auto-merge only for a PR the operator cleared, so GitHub's gates still decide. Its
only review state is a change request when a failure is confirmed to be your PR's own (a hold), which only the
operator's re-approval lifts. See [README.md: What it can and cannot do](../README.md#what-it-can-and-cannot-do).

### Why did the monitor turn off my auto-merge?

Your PR's auto-merge was on while it failed a gate GitHub does not check: an open 🔴 or 🟠 in our review at its head,
a confirmed hold, or a hold label ([README.md: Glossary and placeholders](../README.md#glossary-and-placeholders)).
The operator's auto-merge guard turned it off and commented what fails and how to fix it: fix that, then turn
auto-merge on again yourself. After you edit the PR's title or description, turn auto-merge off and on again, so the
merge takes the new text. See [docs/review_pipeline.md: 7.6 Auto-merge rules](review_pipeline.md#76-auto-merge-rules).

### I force-pushed and GitHub shows new commits. Why no follow-up review?

Most likely the push was a rebase only: GitHub shows every rebased commit as new, but your PR's own diff did not
change, so the monitor records the new head as covered by its review and posts nothing. To check, with the head our
post names: `git range-diff origin/develop <reviewed head> <new head>` (fetch both heads first); `=` marks a commit
whose content is the same, `!` one that changed. A push that only merges develop in is carried too once its checks
are clean. See [docs/design.md: Reviews](design.md#reviews).

### Why hasn't my PR been reviewed yet?

It may be out of scope (a draft, or without one of the scope labels), settling (a head is reviewed 30 minutes after
its last push), in the follow-up cooldown (60 minutes after our last review of it), a rebase only (above), skipped by
the operator (`rk review-skip`), a giant PR awaiting the operator's approval (below), or waiting its turn for a free
review. The waits are the operator's settings under `features.review_automation.schedule`, whose descriptions give
their exceptions. See [docs/design.md: The pipeline](design.md#the-pipeline).

### How do I dispute a finding?

Reply on the PR to the item, with your reason or the evidence (a test, a log, the line that shows it). The next round
checks every earlier item again at the new head, reads your reply, and marks each one: fixed, still open, or declined
with your reason (a follow-up item you declined stays declined, with no ticket). A reply alone does not start a new
round; your next push does, or ask the operator to re-review the same head. See
[docs/review_pipeline.md: 6.2 Follow-ups](review_pipeline.md#62-follow-ups).

### What happens with giant PRs?

A PR over `features.review_automation.large_prs.in_depth_review_over_lines` (30,000 changed lines) gets one in-depth
review, split by area into parts, one verifier per part, then one post for the whole PR. It costs far more than a
normal review, so it starts only once the operator approves it, and the operator may decline it. Every later round is
a light follow-up that checks the open findings and what changed, and needs no approval. See
[docs/design.md: Rounds](design.md#rounds).

### How do I ask for priority?

Use the "Request PR priority" box on the team dashboard when your operator publishes one, or ask the operator. Each
request waits for the operator's Approve or Decline. A priority PR is reviewed first, settles in at most 10 minutes,
skips the follow-up cooldown and leads the merge plan. A PR that fixes a develop break gets that priority by itself.
See [docs/design.md: The pipeline](design.md#the-pipeline) and [docs/design.md: The merge
plan](design.md#the-merge-plan).

### Why did the monitor request changes on my PR?

A gate job failed, and running the failing check on develop alone showed that the failure is your PR's own: the PR
is held. It keeps its place in the plan; fix the failure and push, and the PR is gated again. The change request
stays until the operator re-approves. What a hold is:
[README.md: Glossary and placeholders](../README.md#glossary-and-placeholders).

## For team leads

### What does it do, and what does it never do?

It reviews each new or changed PR in scope, keeps a merge plan that lands PRs on today's develop in the least
disruptive order, checks develop after every merge, and shows it all on dashboards. It never approves, merges directly,
pushes, uses admin rights or edits anyone's comment; every outward action is a guarded `rk` command that runs only
while its feature is live, and a new deployment starts in prototype, where it posts, sends and files nothing. See
[README.md: What it can and cannot do](../README.md#what-it-can-and-cannot-do).

### What happens when develop breaks?

After every merge, develop gets a sanity run. A red run, or develop's CI red in one of your project's jobs, opens an
incident: the merge queue freezes, a PR the monitor armed is disarmed, and the operator is told. With the merge queue
live and its develop guard on, the monitor files one GitHub issue naming the breaking PR, with the label the guard's
settings give (`ci_red_label` or `stability_label`); a fix PR goes first. It never pushes a revert. See
[docs/design.md: Incidents](design.md#incidents).

### What does a review cost, and where do I see the spend?

The monitor counts every model's tokens (`rk usage`). With the prices in `features.review_automation.spend`, it prices
each review by part (verifier, second verifier, code quality pass, orchestration) per day and week: `rk spend`, the
admin dashboard's Review spend card and the weekly report. A review is one posted, kept unposted, or closed without
a post (`rk review-close`), priced as a review of its own. A carried rebase-only or merge-only head costs nothing, and
a large PR costs more. See [docs/design.md: Token economy and review spend](design.md#token-economy-and-review-spend).

### Can my team run its own monitor?

Yes, on any repository and folder (`repository.name`, `repository.folder`). The package's repository defaults (its CI
jobs and test tiers, its hot spots, its GPU tests and how develop is checked) are written for rocKE's own folder; any
other repository or folder gets the package's generic values, and `./rkmon check` warns, naming each. Build an image
([QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md)), deploy it ([QUICK_DEPLOY_GUIDE.md](../QUICK_DEPLOY_GUIDE.md)) and
run it day to day ([QUICK_USER_GUIDE.md](../QUICK_USER_GUIDE.md)).
