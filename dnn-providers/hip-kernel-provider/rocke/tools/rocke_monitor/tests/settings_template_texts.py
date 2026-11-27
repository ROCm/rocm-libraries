# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The settings templates' texts, golden: the build file's, the deploy file's for each of its three profiles and its
every-key text, as rkmon_lib/settings_file.py writes them from the schemas in schema/, with VERSION, the package's
version, as the deploy file's image.tag. TEXTS has each template's text under its label, the one
rkmon_lib/settings_templates.py gives it; tests/test_settings_templates.py compares each with the text the writer
writes. `python3 -B tools/make_package.py --refresh-generated` writes this file again from the writer
(tools/template_texts.py), VERSION kept: when a schema's text changes, these texts change with it.
"""

VERSION = "2026.11.15"

BUILD = """\
# rocKE monitor: build settings.
# These settings decide how the monitor's container image is built. QUICK_BUILD_GUIDE.md walks you through them, and
# README.md explains each one.
# This file is yours: keep it out of git and out of the package folder. Never put a password, token or key in it.
# Download addresses in this file are used only while building: the image never stores them.
# Put text in double quotes. Replace every <...>, then run: rkmon check --settings FILE

# The type of settings file. Don't change it.
kind: "build_settings"
# The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout.
schema_version: 2

# Image: the name the built image gets, <registry>/<name>:<tag>. Your deploy settings name the same image.
image:
  # The registry to push the image to, such as "registry.example.com/my-team". Run docker login for it first. Leave
  # blank to move the image as a file instead (rkmon image save here, rkmon image load on the deploy host).
  registry: ""
  # The image's name, without its registry, which goes in image.registry. Default: "rocke-monitor".
  name: "rocke-monitor"
  # The image's tag. Give every build a new tag, such as "2026.11.15-2", so a deployment can move to it and back.
  # Leave blank to use this package's version, such as "2026.11.15".
  tag: ""

# Operating system: the Linux image the build starts from.
operating_system:
  # The base image: "ubuntu:26.04", "ubuntu:24.04", "rockylinux:9" or "almalinux:9". Default: "ubuntu:26.04".
  base_image: "ubuntu:26.04"
  # A mirror of Ubuntu's package archive, if your network can't reach Ubuntu's own. Ubuntu base images only. Example:
  # "http://mirror.example.com/ubuntu". Leave blank to use Ubuntu's own archive.
  package_mirror: ""
  # Extra root certificates (PEM files on this machine), for a network that inspects TLS traffic. If downloads fail
  # with certificate errors, ask your IT team for these. Example: ["~/certs/my-company-root.pem"]. Default: none.
  ca_certificates: []

# ROCm: the ROCm release installed in the image.
rocm:
  # The ROCm version. Keep "10.1.0" unless your team needs another: the review checks are set up for its compiler.
  # Required.
  version: "10.1.0"
  # Where to download the ROCm installer (the .run file) from, if not AMD's public site. Leave blank to use AMD's
  # public installer for this version, which is checked only when you set package_sha256.
  package_url: ""
  # The installer's SHA-256 checksum: 64 hex digits, in quotes (run: sha256sum FILE). Required when you set
  # package_url; set it anyway to check AMD's public installer.
  package_sha256: ""

# GitHub CLI: the gh tool, which the monitor uses to read and write GitHub.
github_cli:
  # The gh version to install, such as "2.80.0". Leave blank for the newest release
  # (rkmon check warns that two builds may then differ).
  version: ""
  # Where to download the gh release archive (gh_<version>_linux_<arch>.tar.gz) from, if not GitHub. Leave blank to
  # download it from GitHub, checked against GitHub's published checksums.
  package_url: ""
  # The archive's SHA-256 checksum. Required when you set package_url.
  package_sha256: ""

# Atlassian CLI: the atlassian tool, whose commands are auth, confluence, jira, rest, site, update and version. The
# monitor uses it to file JIRA tickets. Every image includes it, so that one image works for every deployment.
atlassian_cli:
  # The version to require, such as "1.2.3" (atlassian version prints it). Leave blank to accept any version.
  version: ""
  # Where to download the atlassian tool's Linux binary from (the program file itself, not an archive). Ask your team
  # where your organization keeps it. Required.
  package_url: "<URL>"
  # The binary's SHA-256 checksum. Download the file once and run: sha256sum FILE. Required.
  package_sha256: "<sha256>"

# Claude Agent SDK: the Python package that includes Claude Code, which runs the reviews.
claude_agent_sdk:
  # The SDK version, such as "1.2.3". Leave blank for the newest release (rkmon check warns that two builds may then
  # differ).
  version: ""
  # Where to download the SDK's wheel (the .whl file) from, if not the package index below. Leave blank to install it
  # from the package index, which is not checked against a checksum.
  package_url: ""
  # The wheel's SHA-256 checksum. Required when you set package_url.
  package_sha256: ""

# Python packages: where the build downloads Python packages from.
python_packages:
  # A Python package index to use instead of pypi.org, such as "https://pypi.example.com/simple/". Leave blank to use
  # pypi.org.
  package_index: ""
"""

DEPLOY_FULL = """\
# rocKE monitor: deploy settings (profile "full": reviews, the merge queue and the admin dashboard).
# These settings decide what one monitor on this host does. QUICK_DEPLOY_GUIDE.md walks you through them, and
# README.md explains each one.
# This file is yours: keep it out of git and out of the package folder. Never put a password, token or key in it:
# rkmon login installs those.
# Every feature starts in "prototype" mode: it does its work and shows you the result on the admin dashboard, but
# posts and sends nothing. Switch a feature to "live" when you are happy with it (QUICK_USER_GUIDE.md, "Go live").
# Put text in double quotes. Replace every <...>, then run: rkmon check --settings FILE
# To change a setting later, see QUICK_USER_GUIDE.md, "Change a setting".

# The type of settings file. Don't change it.
kind: "deploy_settings"
# The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout.
schema_version: 2

# Container: the Docker container this monitor runs in.
container:
  # The container's name. The monitor keeps its data under this name, so choose it once:
  # rkmon confirm refuses a new name while this container exists. To run a second monitor on this host, see
  # QUICK_DEPLOY_GUIDE.md, "Several deployments". While features.review_automation.automated_reviewer_name is blank,
  # every review gives this name on its first line. Lowercase letters, digits, - and _. Default: "rocke-monitor".
  name: "rocke-monitor"

# Image: the container image to run, <registry>/<name>:<tag>, as your build settings name it.
image:
  # The registry to pull the image from, such as "registry.example.com/my-team". Run docker login for it first. Leave
  # blank if you load the image from a file instead (rkmon image load FILE).
  registry: ""
  # The image's name, as in your build settings, without its registry, which goes in image.registry. Default:
  # "rocke-monitor".
  name: "rocke-monitor"
  # The image's tag, as in your build settings. To move to a newer image, change the tag (QUICK_USER_GUIDE.md,
  # "Change a setting"). Required.
  tag: "2026.11.15"

# Repository: the GitHub repository the monitor watches, and which of its pull requests the monitor works on.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The folder your project lives in, from the repository's root, without a / at either end. The monitor uses it to
  # tell your project's files from the rest of the repository. Example: "projects/my-project". Required.
  folder: "<folder>"
  # The labels that put a pull request in scope: the monitor works on every open pull request that has at least one of
  # them. To see the repository's labels, run: gh label list -R <org>/<repo>. Example: ["my-project"]. Required.
  scope_labels: ["<label>"]
  # Labels that several teams share, each with a folder. A pull request with one of these labels is in scope only if
  # it also changes files in that label's folder. Write each one as "label": "folder", the folder from the
  # repository's root. Example: {"needs-ci": "projects/my-project"}: a "needs-ci" pull request is in scope only if it
  # changes files in projects/my-project. Default: none.
  shared_label_folders: {}

# Host: how much of this host the container may use, and which user it runs as.
host:
  # Resources: the container's share of this host. QUICK_DEPLOY_GUIDE.md, "Machine sizing", helps you choose.
  resources:
    # The number of CPUs the container may use. Default: 4.
    cpus: 4
    # The container's memory limit, in GiB. At least ramdisk_gib + 8. Default: 8.
    memory_gib: 8
    # A RAM disk that makes reviews and builds faster, in GiB. It is part of memory_gib. Allow about 1.5 GiB for each
    # review and 2 GiB for each build that run at the same time. 0: no RAM disk. Default: 0.
    ramdisk_gib: 0
  # Users: the user and group the monitor runs as inside the container.
  users:
    # The user id, 1000 or more, other than 65534, the system's "nobody" user, and none of the package's own users'
    # (the GitHub writer's and the PR runners'). Default: null, which means your own user id on this host.
    monitor_uid: null
    # The group id, 1 or more: any group's, a system group's included, but none of the PR runners' groups'. Default:
    # null, which means your own group id on this host.
    monitor_gid: null

# Monitor: the monitor works in ticks. Each tick, it reads GitHub and updates the reviews, the merge queue and the
# dashboards.
monitor:
  # Minutes between ticks. Default: 10.
  tick_interval_minutes: 10
  # Most ticks read only what changed on GitHub. At least this often, in minutes, a tick reads everything again. 0:
  # every tick reads everything. Default: 60.
  full_tick_max_interval_minutes: 60
  # How many local builds, each with its tests, may run at the same time at first, from 1 to 8. Reviews, the merge
  # queue and the develop guard all use these builds. The monitor then adjusts the number, within 1 to 8, to how busy
  # this host is. Default: 1.
  parallel_builds: 1

# Features: what the monitor does. Each feature that posts or sends runs in prototype mode or live mode:
#   "prototype": it does its work and shows you the result on the admin dashboard, but posts and sends nothing.
#   "live": it posts and sends.
features:
  # Review automation: Claude reviews every new or updated pull request in scope.
  review_automation:
    # true: review pull requests. false: no reviews and no Claude; decisions that need judgement wait for you on the
    # admin dashboard. Required.
    enabled: true
    # "prototype": reviews are written and shown on the admin dashboard, but not posted. "live": reviews are posted on
    # GitHub as integrations.github_cli.username, and so are notices about pull requests that change the same code.
    # Required when review automation is on.
    mode: "prototype"
    # The name at the top of every review this monitor posts: "Automated review by <automated_reviewer_name>".
    # Letters, digits, spaces, - and _. Example: "Platform team". Leave blank to use the container's name.
    automated_reviewer_name: ""
    # A folder with your team's own review skills, as Markdown (.md) files only. Its layout must match the package's
    # agentic/ folder, and it may contain only team_members/, agents/, review_book.md and
    # skills/rocke-monitor/review_procedure.md. A file with the same name as a package file replaces it; a file with a
    # new name is added. Leave blank to use the package's skills only.
    skills_folder: ""
    # Orchestrator: the main Claude session. It runs the reviews, checks and posts them, and makes the merge queue's
    # judgement calls.
    orchestrator:
      # Its model: "opus", "sonnet", "haiku", or the name of a Claude model in model_bindings. Default: "opus".
      model: "opus"
      # How hard it thinks: "low", "medium", "high", "xhigh" or "max". Higher costs more and takes longer. Default:
      # "high".
      effort: "high"
    # Schedule: when reviews start.
    schedule:
      # How long to wait after a pull request's latest push before reviewing it, in minutes, so that a burst of pushes
      # gets one review. A burst waits 45 minutes after its last push, and your priorities and the pull requests that
      # fix develop wait at most 10 minutes after any push, a burst's too. Set above 45, it applies to bursts too. 0:
      # a single push doesn't wait. Default: 30.
      push_cooldown_minutes: 30
      # The shortest time between two follow-up reviews of the same pull request, in minutes. Your priorities, pull
      # requests that fix develop, and pushes that only merge develop in don't wait. 0: don't wait. Default: 60.
      follow_up_cooldown_minutes: 60
    # How many reviews may run at the same time at first, from 1 to 5. The monitor then adjusts the number, within 1
    # to 5, to how busy this host is. Default: 3.
    parallel_reviews: 3
    # Large pull requests: how very large pull requests are reviewed.
    large_prs:
      # A pull request with more changed lines than this gets one in-depth review, split into parts, and after that
      # only light follow-ups. The in-depth review starts only once you approve it: the admin dashboard's Needs you
      # card gives its size and cost, with Approve and Decline, or run: rkmon rk review-approve N. Default: 30000.
      in_depth_review_over_lines: 30000
      # The most changed lines in each part of an in-depth review, from 6000 to 10000. Default: 8000.
      split_part_lines: 8000
    # Passes: the steps of the review work, each with its model and effort.
    #   Models: "opus", "sonnet", "haiku", or the name of an entry in model_bindings.
    #   Efforts: "low", "medium", "high", "xhigh" or "max". Ollama and OpenAI-compatible models take only "low",
    #   "medium" or "high".
    #   The verification and code quality passes always run, each on a Claude model. The four optional passes after
    #   them let another model do part of the work first; "default" means no other model, so the pass in charge does
    #   that work itself.
    passes:
      # Verification pass: reviews each pull request on its own, checks every claim of the first review pass (if there
      # is one) against the code, drops what it can't confirm, and writes the review that gets posted.
      verification_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "high".
        effort: "high"
      # Code quality pass: reviews each pull request's design (single responsibility, no duplication, layering, names,
      # tests) and gives its findings to the verification pass.
      code_quality_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "medium".
        effort: "medium"
      # First review pass (optional): another model writes a first draft of each review, which the verification pass
      # then checks claim by claim.
      first_review_pass:
        # The model that writes the draft. Default: "default", which means no draft: the verification pass reviews
        # alone.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # CI triage pass (optional): another model reads the pull request's failing CI jobs and tells the verification
      # pass what caused each failure.
      ci_triage_pass:
        # The model. Default: "default", which means none: the verification pass reads the failing CI jobs itself.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # First code quality pass (optional): another model checks the pull request's changes against the code quality
      # rules first, and gives what it finds to the code quality pass to confirm.
      first_code_quality_pass:
        # The model. Default: "default", which means none: the code quality pass does all of the work.
        model: "default"
        # Its effort. Default: "default", which means the code quality pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # Notice draft pass (optional): another model writes the first draft of the notices the monitor posts on pull
      # requests: when two pull requests change the same code, and when a change on develop breaks a pull request's
      # tests. The orchestrator checks and posts them.
      notice_draft_pass:
        # The model. Default: "default", which means none: the orchestrator writes the notices itself.
        model: "default"
        # Its effort. Default: "default", which means the orchestrator's effort, or none for a model that isn't
        # Claude.
        effort: "default"
    # Remote GPU tests: some tests need a GPU, and on a host without one they are skipped. With this on, the monitor
    # runs them on a Slurm cluster instead: it logs in to the cluster's login node over SSH and submits each test run
    # as a Slurm job, which runs on a GPU node. A review doesn't wait for the results: its GPU tests row says
    # "pending" and is filled in when the job finishes.
    remote_gpu_tests:
      # true: run GPU tests on the cluster. false: skip them, as on any host without a GPU. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required when review automation is on.
      enabled: false
    # Spend: what the reviews cost in US dollars, shown on the dashboards and in the weekly report.
    spend:
      # true: work out the cost from the prices in usd_per_million_tokens. false: don't show costs. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required when review automation is on.
      enabled: false
  # Merge queue: puts the pull requests in scope in merge order, runs its gates (the checks a pull request must pass
  # before it merges), and makes a merge plan (which pull requests can merge together without conflicts). Your
  # priorities go first (rkmon rk prioritize N, or a priority request you approve on the admin dashboard).
  merge_queue:
    # true: run the merge queue (only one monitor per repository may). false: no merge queue; your priorities still
    # decide which pull requests are reviewed first. Required.
    enabled: true
    # "prototype": the queue, its gates and the merge plan are worked out and shown to you, but nothing happens on
    # GitHub: no auto-merge is turned on, and nothing is held, commented on or filed. "live": the queue does these
    # things on GitHub, as the settings below allow. Required when the merge queue is on.
    mode: "prototype"
    # Auto-merge: when a pull request you cleared for merging (rkmon rk clear N) passes the gates, the monitor turns
    # on GitHub's auto-merge for it. GitHub's branch protection still has the final say.
    auto_merge:
      # true: turn on auto-merge (in prototype, the dashboard only shows where it would). false: only propose the
      # merge order to you. Required when the merge queue is on.
      enabled: false
    # Auto-merge guard: when someone turns on auto-merge for a pull request that fails a gate, the monitor comments on
    # the pull request and turns its auto-merge off (in prototype, it only tells you). It never turns off an
    # auto-merge you turned on yourself. It runs whenever the merge queue runs.
    auto_merge_guard:
      # What to do when someone else turns on auto-merge for a pull request you wrote: "alert_only" (only tell you) or
      # "same_as_others" (comment and turn it off, as for anyone else's pull request). Default: "alert_only".
      on_your_pull_requests: "alert_only"
    # Auto-hold: when a pull request fails a gate because of its own changes, the monitor posts a "Request changes"
    # review on it from your account. That blocks the merge until the author fixes it and you approve it again.
    auto_hold:
      # true: hold such pull requests (in prototype, only show them). false: only report the failure to you. Required
      # when the merge queue is on.
      enabled: true
    # Queue notices: each pull request near the front of the queue gets one comment about its place in the queue,
    # updated as things change.
    queue_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: true
    # Drift notices: when a change on develop breaks a pull request's own tests, the monitor tells its author which
    # develop commit changed the behaviour. Claude writes these, so they need review automation.
    drift_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: true
    # Develop guard: after every merge, the monitor checks develop and traces any break to the pull request that
    # caused it. In live, each break becomes a GitHub issue, and an open pull request that fixes it is linked to the
    # issue. Without review automation, breaks wait for you on the admin dashboard instead.
    develop_guard:
      # true: check develop. false: don't. Required when the merge queue is on.
      enabled: true
      # The repository to file develop issues in, as <org>/<repo>. Leave blank to use repository.name.
      issue_repository: ""
      # The label for an issue about a break that turns develop's CI red. The label must already exist in
      # issue_repository (rkmon rk label-create NAME --description TEXT --color HEX creates one). Example: "ci-red".
      # Required when the develop guard is on and the merge queue is live.
      ci_red_label: ""
      # More labels for those issues. Each must already exist. Default: none.
      ci_red_extra_labels: []
      # The label for an issue about a break that CI doesn't catch; the issue is assigned to the author of the pull
      # request that caused it. The label must already exist in issue_repository. Example: "stability". Required when
      # the develop guard is on and the merge queue is live.
      stability_label: ""
  # Dashboards: the admin dashboard (yours) and the team dashboard (for other teams).
  dashboards:
    # Admin dashboard: everything the monitor shows, with Approve and Decline buttons for priority requests, proposed
    # tickets and the in-depth reviews of large pull requests. Only this host can open it, at
    # http://127.0.0.1:<port>/. It asks for a user name and a password: type any user name, and the access token as
    # the password (rkmon secret admin_access_token --generate makes one). To open it from another machine, use an SSH
    # tunnel (README.md, "The admin dashboard").
    admin:
      # true: serve the admin dashboard. false: none (the team dashboard and found-issue tickets need it). Required.
      enabled: true
      # The port to serve it on. Default: 18614.
      port: 18614
    # Team dashboard: a copy of the dashboard for other teams to read, without the admin buttons.
    team:
      # true: deliver the team dashboard (it needs the admin dashboard). false: none. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required.
      enabled: false
  # Alerts: short messages to a Microsoft Teams channel, through a Teams Workflows webhook. Install the webhook's
  # address with: rkmon secret teams_webhook FILE. Each alert has its own on/off switch.
  alerts:
    # Develop breaks: a message when develop's CI is red and no fix is planned. It needs the develop guard.
    develop_breaks:
      # true: send this alert. false: breaks only show on the dashboards. Required.
      enabled: false
    # Needs you: a message for each decision that waits for you on the admin dashboard (a merge to approve, a hold
    # that has waited too long, a priority request, a pull request ready for your approval, an in-depth review to
    # approve). Each is sent once, addressed to your GitHub username, with a link to the team dashboard when its
    # public_url is set.
    needs_you:
      # true: send this alert. false: decisions only show on the dashboards. Required.
      enabled: false
  # Ticket creation: JIRA tickets, filed with the atlassian tool on the site in integrations.atlassian_cli.jira_url.
  # Install its login with: rkmon login atlassian (the login needs permission to create tickets). Each type of ticket
  # has its own on/off switch.
  ticket_creation:
    # Follow-up tickets: when a pull request merges with review items still open, each one becomes a ticket for the
    # pull request's author. Open items are the monitor's own blocking and should-fix findings, review comments
    # answered with a promise to fix them later ("next PR", "TODO"), and reviewers' requests nobody answered. Never
    # nits. Tickets are filed without asking you, so agree on this with your team first. To assign a ticket, the
    # monitor needs the author's JIRA account (rkmon rk jira-account add LOGIN ACCOUNT); without it, the ticket is
    # filed unassigned.
    follow_up_tickets:
      # true: file them (in live). It needs review automation. false: only list the items for you. Required.
      enabled: false
    # Found-issue tickets: when the monitor finds a problem that belongs to no pull request and isn't a develop break,
    # such as a flaky test, it proposes a ticket on the admin dashboard. Approve files it; Decline drops it.
    found_issue_tickets:
      # true: propose tickets. It needs the admin dashboard. false: only report such problems to you. Required.
      enabled: false
  # Weekly report: every week, a statistics report and notes for a lessons-learned review, saved in the monitor's
  # rocke_weekly/ folder (README.md, "Where things live").
  weekly_report:
    # true: write them every week. false: don't; rkmon rk stats and rkmon rk lessons write them when you ask.
    # Required.
    enabled: false

# Integrations: the tools the monitor uses. rkmon login lists the logins they need and installs each one.
integrations:
  # GitHub CLI: how the monitor reads and writes GitHub. It needs two logins: rkmon login github_read for reading, and
  # rkmon login github_write for posting. README.md, "Logins and secrets", lists the permissions each one needs.
  github_cli:
    # Your GitHub username. The monitor posts as this user, and treats this user's actions as yours: your comments,
    # your approvals (an approval lifts a hold), and auto-merges you turned on (which the guard leaves alone). Needs
    # you alerts are addressed to this user. Required.
    username: "<user_login>"
    # The most GitHub API requests this monitor may make in an hour. GitHub's hourly limit is per account: if several
    # monitors share one account, give each a cap so that together they stay under the limit. A gh command that reads
    # several pages of results counts once toward this cap, though GitHub counts each page: leave headroom. 0: no cap.
    # Default: 0.
    max_requests_per_hour: 0
  # Atlassian CLI: not used while no type of ticket is on.
  atlassian_cli: {}
  # Claude Code: how the monitor reaches Claude. Install the key with: rkmon login claude (an Anthropic API key, or
  # your LLM gateway's token).
  claude_code:
    # The address to reach Claude at, if you use an LLM gateway (Claude Code's ANTHROPIC_BASE_URL). Example:
    # "https://llm-gateway.example.com". Leave blank to use Anthropic's own API.
    base_url: ""
    # How the key is sent: "header" (in the header that api_key_header names), "bearer" (as
    # "Authorization: Bearer <key>") or "header_and_bearer" (both). Your gateway's documentation says which. Default:
    # "header".
    api_key_sent_as: "header"
    # The header the key goes in, for "header" and "header_and_bearer". Leave blank for Anthropic's own header,
    # "x-api-key".
    api_key_header: ""
    # Text to put before the key in that header, such as "Bearer". Leave blank to send the key alone.
    api_key_scheme: ""

# Model bindings: other models the passes can use, each under a name you choose. "opus", "sonnet" and "haiku" are
# built in, reached through integrations.claude_code, and need no entry. To add a model, replace {} with entries in
# one of the three forms that rkmon template section model_bindings prints. Default: none.
model_bindings: {}
"""

DEPLOY_REVIEW_ONLY = """\
# rocKE monitor: deploy settings (profile "review-only": reviews and the admin dashboard).
# These settings decide what one monitor on this host does. QUICK_DEPLOY_GUIDE.md walks you through them, and
# README.md explains each one.
# This monitor reviews the pull requests that have its scope labels, with your team's own review skills if
# skills_folder names them. It never merges a pull request, puts one on hold or turns on its auto-merge. Its reviews
# run on its own Claude login, so that login's account pays for them.
# This file is yours: keep it out of git and out of the package folder. Never put a password, token or key in it:
# rkmon login installs those.
# Every feature starts in "prototype" mode: it does its work and shows you the result on the admin dashboard, but
# posts and sends nothing. Switch a feature to "live" when you are happy with it (QUICK_USER_GUIDE.md, "Go live").
# Put text in double quotes. Replace every <...>, then run: rkmon check --settings FILE
# To change a setting later, see QUICK_USER_GUIDE.md, "Change a setting".

# The type of settings file. Don't change it.
kind: "deploy_settings"
# The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout.
schema_version: 2

# Container: the Docker container this monitor runs in.
container:
  # The container's name. The monitor keeps its data under this name, so choose it once:
  # rkmon confirm refuses a new name while this container exists. To run a second monitor on this host, see
  # QUICK_DEPLOY_GUIDE.md, "Several deployments". While features.review_automation.automated_reviewer_name is blank,
  # every review gives this name on its first line. Lowercase letters, digits, - and _. Default: "rocke-monitor".
  name: "rocke-monitor"

# Image: the container image to run, <registry>/<name>:<tag>, as your build settings name it.
image:
  # The registry to pull the image from, such as "registry.example.com/my-team". Run docker login for it first. Leave
  # blank if you load the image from a file instead (rkmon image load FILE).
  registry: ""
  # The image's name, as in your build settings, without its registry, which goes in image.registry. Default:
  # "rocke-monitor".
  name: "rocke-monitor"
  # The image's tag, as in your build settings. To move to a newer image, change the tag (QUICK_USER_GUIDE.md,
  # "Change a setting"). Required.
  tag: "2026.11.15"

# Repository: the GitHub repository the monitor watches, and which of its pull requests the monitor works on.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The folder your project lives in, from the repository's root, without a / at either end. The monitor uses it to
  # tell your project's files from the rest of the repository. Example: "projects/my-project". Required.
  folder: "<folder>"
  # The labels that put a pull request in scope: the monitor works on every open pull request that has at least one of
  # them. To see the repository's labels, run: gh label list -R <org>/<repo>. Example: ["my-project"]. Required.
  scope_labels: ["<label>"]
  # Labels that several teams share, each with a folder. A pull request with one of these labels is in scope only if
  # it also changes files in that label's folder. Write each one as "label": "folder", the folder from the
  # repository's root. Example: {"needs-ci": "projects/my-project"}: a "needs-ci" pull request is in scope only if it
  # changes files in projects/my-project. Default: none.
  shared_label_folders: {}

# Host: how much of this host the container may use, and which user it runs as.
host:
  # Resources: the container's share of this host. QUICK_DEPLOY_GUIDE.md, "Machine sizing", helps you choose.
  resources:
    # The number of CPUs the container may use. Default: 4.
    cpus: 4
    # The container's memory limit, in GiB. At least ramdisk_gib + 8. Default: 8.
    memory_gib: 8
    # A RAM disk that makes reviews and builds faster, in GiB. It is part of memory_gib. Allow about 1.5 GiB for each
    # review and 2 GiB for each build that run at the same time. 0: no RAM disk. Default: 0.
    ramdisk_gib: 0
  # Users: the user and group the monitor runs as inside the container.
  users:
    # The user id, 1000 or more, other than 65534, the system's "nobody" user, and none of the package's own users'
    # (the GitHub writer's and the PR runners'). Default: null, which means your own user id on this host.
    monitor_uid: null
    # The group id, 1 or more: any group's, a system group's included, but none of the PR runners' groups'. Default:
    # null, which means your own group id on this host.
    monitor_gid: null

# Monitor: the monitor works in ticks. Each tick, it reads GitHub and updates the reviews, the merge queue and the
# dashboards.
monitor:
  # Minutes between ticks. Default: 10.
  tick_interval_minutes: 10
  # Most ticks read only what changed on GitHub. At least this often, in minutes, a tick reads everything again. 0:
  # every tick reads everything. Default: 60.
  full_tick_max_interval_minutes: 60
  # How many local builds, each with its tests, may run at the same time at first, from 1 to 8. Reviews, the merge
  # queue and the develop guard all use these builds. The monitor then adjusts the number, within 1 to 8, to how busy
  # this host is. Default: 1.
  parallel_builds: 1

# Features: what the monitor does. Each feature that posts or sends runs in prototype mode or live mode:
#   "prototype": it does its work and shows you the result on the admin dashboard, but posts and sends nothing.
#   "live": it posts and sends.
features:
  # Review automation: Claude reviews every new or updated pull request in scope.
  review_automation:
    # true: review pull requests. false: no reviews and no Claude; decisions that need judgement wait for you on the
    # admin dashboard. Required.
    enabled: true
    # "prototype": reviews are written and shown on the admin dashboard, but not posted. "live": reviews are posted on
    # GitHub as integrations.github_cli.username, and so are notices about pull requests that change the same code.
    # Required when review automation is on.
    mode: "prototype"
    # The name at the top of every review this monitor posts: "Automated review by <automated_reviewer_name>".
    # Letters, digits, spaces, - and _. Example: "Platform team". Leave blank to use the container's name.
    automated_reviewer_name: ""
    # A folder with your team's own review skills, as Markdown (.md) files only. Its layout must match the package's
    # agentic/ folder, and it may contain only team_members/, agents/, review_book.md and
    # skills/rocke-monitor/review_procedure.md. A file with the same name as a package file replaces it; a file with a
    # new name is added. Leave blank to use the package's skills only.
    skills_folder: ""
    # Orchestrator: the main Claude session. It runs the reviews, checks and posts them, and makes the merge queue's
    # judgement calls.
    orchestrator:
      # Its model: "opus", "sonnet", "haiku", or the name of a Claude model in model_bindings. Default: "opus".
      model: "opus"
      # How hard it thinks: "low", "medium", "high", "xhigh" or "max". Higher costs more and takes longer. Default:
      # "high".
      effort: "high"
    # Schedule: when reviews start.
    schedule:
      # How long to wait after a pull request's latest push before reviewing it, in minutes, so that a burst of pushes
      # gets one review. A burst waits 45 minutes after its last push, and your priorities and the pull requests that
      # fix develop wait at most 10 minutes after any push, a burst's too. Set above 45, it applies to bursts too. 0:
      # a single push doesn't wait. Default: 30.
      push_cooldown_minutes: 30
      # The shortest time between two follow-up reviews of the same pull request, in minutes. Your priorities, pull
      # requests that fix develop, and pushes that only merge develop in don't wait. 0: don't wait. Default: 60.
      follow_up_cooldown_minutes: 60
    # How many reviews may run at the same time at first, from 1 to 5. The monitor then adjusts the number, within 1
    # to 5, to how busy this host is. Default: 3.
    parallel_reviews: 3
    # Large pull requests: how very large pull requests are reviewed.
    large_prs:
      # A pull request with more changed lines than this gets one in-depth review, split into parts, and after that
      # only light follow-ups. The in-depth review starts only once you approve it: the admin dashboard's Needs you
      # card gives its size and cost, with Approve and Decline, or run: rkmon rk review-approve N. Default: 30000.
      in_depth_review_over_lines: 30000
      # The most changed lines in each part of an in-depth review, from 6000 to 10000. Default: 8000.
      split_part_lines: 8000
    # Passes: the steps of the review work, each with its model and effort.
    #   Models: "opus", "sonnet", "haiku", or the name of an entry in model_bindings.
    #   Efforts: "low", "medium", "high", "xhigh" or "max". Ollama and OpenAI-compatible models take only "low",
    #   "medium" or "high".
    #   The verification and code quality passes always run, each on a Claude model. The four optional passes after
    #   them let another model do part of the work first; "default" means no other model, so the pass in charge does
    #   that work itself.
    passes:
      # Verification pass: reviews each pull request on its own, checks every claim of the first review pass (if there
      # is one) against the code, drops what it can't confirm, and writes the review that gets posted.
      verification_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "high".
        effort: "high"
      # Code quality pass: reviews each pull request's design (single responsibility, no duplication, layering, names,
      # tests) and gives its findings to the verification pass.
      code_quality_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "medium".
        effort: "medium"
      # First review pass (optional): another model writes a first draft of each review, which the verification pass
      # then checks claim by claim.
      first_review_pass:
        # The model that writes the draft. Default: "default", which means no draft: the verification pass reviews
        # alone.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # CI triage pass (optional): another model reads the pull request's failing CI jobs and tells the verification
      # pass what caused each failure.
      ci_triage_pass:
        # The model. Default: "default", which means none: the verification pass reads the failing CI jobs itself.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # First code quality pass (optional): another model checks the pull request's changes against the code quality
      # rules first, and gives what it finds to the code quality pass to confirm.
      first_code_quality_pass:
        # The model. Default: "default", which means none: the code quality pass does all of the work.
        model: "default"
        # Its effort. Default: "default", which means the code quality pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # Notice draft pass (optional): another model writes the first draft of the notices the monitor posts on pull
      # requests: when two pull requests change the same code, and when a change on develop breaks a pull request's
      # tests. The orchestrator checks and posts them.
      notice_draft_pass:
        # The model. Default: "default", which means none: the orchestrator writes the notices itself.
        model: "default"
        # Its effort. Default: "default", which means the orchestrator's effort, or none for a model that isn't
        # Claude.
        effort: "default"
    # Remote GPU tests: some tests need a GPU, and on a host without one they are skipped. With this on, the monitor
    # runs them on a Slurm cluster instead: it logs in to the cluster's login node over SSH and submits each test run
    # as a Slurm job, which runs on a GPU node. A review doesn't wait for the results: its GPU tests row says
    # "pending" and is filled in when the job finishes.
    remote_gpu_tests:
      # true: run GPU tests on the cluster. false: skip them, as on any host without a GPU. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required when review automation is on.
      enabled: false
    # Spend: what the reviews cost in US dollars, shown on the dashboards and in the weekly report.
    spend:
      # true: work out the cost from the prices in usd_per_million_tokens. false: don't show costs. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required when review automation is on.
      enabled: false
  # Merge queue: puts the pull requests in scope in merge order, runs its gates (the checks a pull request must pass
  # before it merges), and makes a merge plan (which pull requests can merge together without conflicts). Your
  # priorities go first (rkmon rk prioritize N, or a priority request you approve on the admin dashboard).
  merge_queue:
    # true: run the merge queue (only one monitor per repository may). false: no merge queue; your priorities still
    # decide which pull requests are reviewed first. To turn it on, see QUICK_USER_GUIDE.md, "Turn on a feature".
    # Required.
    enabled: false
  # Dashboards: the admin dashboard (yours) and the team dashboard (for other teams).
  dashboards:
    # Admin dashboard: everything the monitor shows, with Approve and Decline buttons for priority requests, proposed
    # tickets and the in-depth reviews of large pull requests. Only this host can open it, at
    # http://127.0.0.1:<port>/. It asks for a user name and a password: type any user name, and the access token as
    # the password (rkmon secret admin_access_token --generate makes one). To open it from another machine, use an SSH
    # tunnel (README.md, "The admin dashboard").
    admin:
      # true: serve the admin dashboard. false: none (the team dashboard and found-issue tickets need it). Required.
      enabled: true
      # The port to serve it on. Default: 18614.
      port: 18614
    # Team dashboard: a copy of the dashboard for other teams to read, without the admin buttons.
    team:
      # true: deliver the team dashboard (it needs the admin dashboard). false: none. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required.
      enabled: false
  # Alerts: short messages to a Microsoft Teams channel, through a Teams Workflows webhook. Install the webhook's
  # address with: rkmon secret teams_webhook FILE. Each alert has its own on/off switch.
  alerts:
    # Develop breaks: a message when develop's CI is red and no fix is planned. It needs the develop guard.
    develop_breaks:
      # true: send this alert. false: breaks only show on the dashboards. Required.
      enabled: false
    # Needs you: a message for each decision that waits for you on the admin dashboard (a merge to approve, a hold
    # that has waited too long, a priority request, a pull request ready for your approval, an in-depth review to
    # approve). Each is sent once, addressed to your GitHub username, with a link to the team dashboard when its
    # public_url is set.
    needs_you:
      # true: send this alert. false: decisions only show on the dashboards. Required.
      enabled: false
  # Ticket creation: JIRA tickets, filed with the atlassian tool on the site in integrations.atlassian_cli.jira_url.
  # Install its login with: rkmon login atlassian (the login needs permission to create tickets). Each type of ticket
  # has its own on/off switch.
  ticket_creation:
    # Follow-up tickets: when a pull request merges with review items still open, each one becomes a ticket for the
    # pull request's author. Open items are the monitor's own blocking and should-fix findings, review comments
    # answered with a promise to fix them later ("next PR", "TODO"), and reviewers' requests nobody answered. Never
    # nits. Tickets are filed without asking you, so agree on this with your team first. To assign a ticket, the
    # monitor needs the author's JIRA account (rkmon rk jira-account add LOGIN ACCOUNT); without it, the ticket is
    # filed unassigned.
    follow_up_tickets:
      # true: file them (in live). It needs review automation. false: only list the items for you. Required.
      enabled: false
    # Found-issue tickets: when the monitor finds a problem that belongs to no pull request and isn't a develop break,
    # such as a flaky test, it proposes a ticket on the admin dashboard. Approve files it; Decline drops it.
    found_issue_tickets:
      # true: propose tickets. It needs the admin dashboard. false: only report such problems to you. Required.
      enabled: false
  # Weekly report: every week, a statistics report and notes for a lessons-learned review, saved in the monitor's
  # rocke_weekly/ folder (README.md, "Where things live").
  weekly_report:
    # true: write them every week. false: don't; rkmon rk stats and rkmon rk lessons write them when you ask.
    # Required.
    enabled: false

# Integrations: the tools the monitor uses. rkmon login lists the logins they need and installs each one.
integrations:
  # GitHub CLI: how the monitor reads and writes GitHub. It needs two logins: rkmon login github_read for reading, and
  # rkmon login github_write for posting. README.md, "Logins and secrets", lists the permissions each one needs.
  github_cli:
    # Your GitHub username. The monitor posts as this user, and treats this user's actions as yours: your comments,
    # your approvals (an approval lifts a hold), and auto-merges you turned on (which the guard leaves alone). Needs
    # you alerts are addressed to this user. Required.
    username: "<user_login>"
    # The most GitHub API requests this monitor may make in an hour. GitHub's hourly limit is per account: if several
    # monitors share one account, give each a cap so that together they stay under the limit. A gh command that reads
    # several pages of results counts once toward this cap, though GitHub counts each page: leave headroom. 0: no cap.
    # Default: 0.
    max_requests_per_hour: 0
  # Atlassian CLI: not used while no type of ticket is on.
  atlassian_cli: {}
  # Claude Code: how the monitor reaches Claude. Install the key with: rkmon login claude (an Anthropic API key, or
  # your LLM gateway's token).
  claude_code:
    # The address to reach Claude at, if you use an LLM gateway (Claude Code's ANTHROPIC_BASE_URL). Example:
    # "https://llm-gateway.example.com". Leave blank to use Anthropic's own API.
    base_url: ""
    # How the key is sent: "header" (in the header that api_key_header names), "bearer" (as
    # "Authorization: Bearer <key>") or "header_and_bearer" (both). Your gateway's documentation says which. Default:
    # "header".
    api_key_sent_as: "header"
    # The header the key goes in, for "header" and "header_and_bearer". Leave blank for Anthropic's own header,
    # "x-api-key".
    api_key_header: ""
    # Text to put before the key in that header, such as "Bearer". Leave blank to send the key alone.
    api_key_scheme: ""

# Model bindings: other models the passes can use, each under a name you choose. "opus", "sonnet" and "haiku" are
# built in, reached through integrations.claude_code, and need no entry. To add a model, replace {} with entries in
# one of the three forms that rkmon template section model_bindings prints. Default: none.
model_bindings: {}
"""

DEPLOY_MERGE_QUEUE = """\
# rocKE monitor: deploy settings (profile "merge-queue": the merge queue and the admin dashboard, without Claude).
# These settings decide what one monitor on this host does. QUICK_DEPLOY_GUIDE.md walks you through them, and
# README.md explains each one.
# This monitor runs the merge queue and the develop guard by fixed rules. It writes no reviews, so its review gate
# counts only approvals by people. Decisions that need judgement, such as a hold to confirm or a develop break to
# file, wait for you on the admin dashboard.
# This file is yours: keep it out of git and out of the package folder. Never put a password, token or key in it:
# rkmon login installs those.
# Every feature starts in "prototype" mode: it does its work and shows you the result on the admin dashboard, but
# posts and sends nothing. Switch a feature to "live" when you are happy with it (QUICK_USER_GUIDE.md, "Go live").
# Put text in double quotes. Replace every <...>, then run: rkmon check --settings FILE
# To change a setting later, see QUICK_USER_GUIDE.md, "Change a setting".

# The type of settings file. Don't change it.
kind: "deploy_settings"
# The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout.
schema_version: 2

# Container: the Docker container this monitor runs in.
container:
  # The container's name. The monitor keeps its data under this name, so choose it once:
  # rkmon confirm refuses a new name while this container exists. To run a second monitor on this host, see
  # QUICK_DEPLOY_GUIDE.md, "Several deployments". While features.review_automation.automated_reviewer_name is blank,
  # every review gives this name on its first line. Lowercase letters, digits, - and _. Default: "rocke-monitor".
  name: "rocke-monitor"

# Image: the container image to run, <registry>/<name>:<tag>, as your build settings name it.
image:
  # The registry to pull the image from, such as "registry.example.com/my-team". Run docker login for it first. Leave
  # blank if you load the image from a file instead (rkmon image load FILE).
  registry: ""
  # The image's name, as in your build settings, without its registry, which goes in image.registry. Default:
  # "rocke-monitor".
  name: "rocke-monitor"
  # The image's tag, as in your build settings. To move to a newer image, change the tag (QUICK_USER_GUIDE.md,
  # "Change a setting"). Required.
  tag: "2026.11.15"

# Repository: the GitHub repository the monitor watches, and which of its pull requests the monitor works on.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The folder your project lives in, from the repository's root, without a / at either end. The monitor uses it to
  # tell your project's files from the rest of the repository. Example: "projects/my-project". Required.
  folder: "<folder>"
  # The labels that put a pull request in scope: the monitor works on every open pull request that has at least one of
  # them. To see the repository's labels, run: gh label list -R <org>/<repo>. Example: ["my-project"]. Required.
  scope_labels: ["<label>"]
  # Labels that several teams share, each with a folder. A pull request with one of these labels is in scope only if
  # it also changes files in that label's folder. Write each one as "label": "folder", the folder from the
  # repository's root. Example: {"needs-ci": "projects/my-project"}: a "needs-ci" pull request is in scope only if it
  # changes files in projects/my-project. Default: none.
  shared_label_folders: {}

# Host: how much of this host the container may use, and which user it runs as.
host:
  # Resources: the container's share of this host. QUICK_DEPLOY_GUIDE.md, "Machine sizing", helps you choose.
  resources:
    # The number of CPUs the container may use. Default: 4.
    cpus: 4
    # The container's memory limit, in GiB. At least ramdisk_gib + 8. Default: 8.
    memory_gib: 8
    # A RAM disk that makes reviews and builds faster, in GiB. It is part of memory_gib. Allow about 1.5 GiB for each
    # review and 2 GiB for each build that run at the same time. 0: no RAM disk. Default: 0.
    ramdisk_gib: 0
  # Users: the user and group the monitor runs as inside the container.
  users:
    # The user id, 1000 or more, other than 65534, the system's "nobody" user, and none of the package's own users'
    # (the GitHub writer's and the PR runners'). Default: null, which means your own user id on this host.
    monitor_uid: null
    # The group id, 1 or more: any group's, a system group's included, but none of the PR runners' groups'. Default:
    # null, which means your own group id on this host.
    monitor_gid: null

# Monitor: the monitor works in ticks. Each tick, it reads GitHub and updates the reviews, the merge queue and the
# dashboards.
monitor:
  # Minutes between ticks. Default: 10.
  tick_interval_minutes: 10
  # Most ticks read only what changed on GitHub. At least this often, in minutes, a tick reads everything again. 0:
  # every tick reads everything. Default: 60.
  full_tick_max_interval_minutes: 60
  # How many local builds, each with its tests, may run at the same time at first, from 1 to 8. Reviews, the merge
  # queue and the develop guard all use these builds. The monitor then adjusts the number, within 1 to 8, to how busy
  # this host is. Default: 1.
  parallel_builds: 1

# Features: what the monitor does. Each feature that posts or sends runs in prototype mode or live mode:
#   "prototype": it does its work and shows you the result on the admin dashboard, but posts and sends nothing.
#   "live": it posts and sends.
features:
  # Review automation: Claude reviews every new or updated pull request in scope.
  review_automation:
    # true: review pull requests. false: no reviews and no Claude; decisions that need judgement wait for you on the
    # admin dashboard. To turn it on, see QUICK_USER_GUIDE.md, "Turn on a feature". Required.
    enabled: false
  # Merge queue: puts the pull requests in scope in merge order, runs its gates (the checks a pull request must pass
  # before it merges), and makes a merge plan (which pull requests can merge together without conflicts). Your
  # priorities go first (rkmon rk prioritize N, or a priority request you approve on the admin dashboard).
  merge_queue:
    # true: run the merge queue (only one monitor per repository may). false: no merge queue; your priorities still
    # decide which pull requests are reviewed first. Required.
    enabled: true
    # "prototype": the queue, its gates and the merge plan are worked out and shown to you, but nothing happens on
    # GitHub: no auto-merge is turned on, and nothing is held, commented on or filed. "live": the queue does these
    # things on GitHub, as the settings below allow. Required when the merge queue is on.
    mode: "prototype"
    # Auto-merge: when a pull request you cleared for merging (rkmon rk clear N) passes the gates, the monitor turns
    # on GitHub's auto-merge for it. GitHub's branch protection still has the final say.
    auto_merge:
      # true: turn on auto-merge (in prototype, the dashboard only shows where it would). false: only propose the
      # merge order to you. Required when the merge queue is on.
      enabled: false
    # Auto-merge guard: when someone turns on auto-merge for a pull request that fails a gate, the monitor comments on
    # the pull request and turns its auto-merge off (in prototype, it only tells you). It never turns off an
    # auto-merge you turned on yourself. It runs whenever the merge queue runs.
    auto_merge_guard:
      # What to do when someone else turns on auto-merge for a pull request you wrote: "alert_only" (only tell you) or
      # "same_as_others" (comment and turn it off, as for anyone else's pull request). Default: "alert_only".
      on_your_pull_requests: "alert_only"
    # Auto-hold: when a pull request fails a gate because of its own changes, the monitor posts a "Request changes"
    # review on it from your account. That blocks the merge until the author fixes it and you approve it again.
    auto_hold:
      # true: hold such pull requests (in prototype, only show them). false: only report the failure to you. Required
      # when the merge queue is on.
      enabled: true
    # Queue notices: each pull request near the front of the queue gets one comment about its place in the queue,
    # updated as things change.
    queue_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: true
    # Drift notices: when a change on develop breaks a pull request's own tests, the monitor tells its author which
    # develop commit changed the behaviour. Claude writes these, so they need review automation.
    drift_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: false
    # Develop guard: after every merge, the monitor checks develop and traces any break to the pull request that
    # caused it. In live, each break becomes a GitHub issue, and an open pull request that fixes it is linked to the
    # issue. Without review automation, breaks wait for you on the admin dashboard instead.
    develop_guard:
      # true: check develop. false: don't. Required when the merge queue is on.
      enabled: true
      # The repository to file develop issues in, as <org>/<repo>. Leave blank to use repository.name.
      issue_repository: ""
      # The label for an issue about a break that turns develop's CI red. The label must already exist in
      # issue_repository (rkmon rk label-create NAME --description TEXT --color HEX creates one). Example: "ci-red".
      # Required when the develop guard is on and the merge queue is live.
      ci_red_label: ""
      # More labels for those issues. Each must already exist. Default: none.
      ci_red_extra_labels: []
      # The label for an issue about a break that CI doesn't catch; the issue is assigned to the author of the pull
      # request that caused it. The label must already exist in issue_repository. Example: "stability". Required when
      # the develop guard is on and the merge queue is live.
      stability_label: ""
  # Dashboards: the admin dashboard (yours) and the team dashboard (for other teams).
  dashboards:
    # Admin dashboard: everything the monitor shows, with Approve and Decline buttons for priority requests, proposed
    # tickets and the in-depth reviews of large pull requests. Only this host can open it, at
    # http://127.0.0.1:<port>/. It asks for a user name and a password: type any user name, and the access token as
    # the password (rkmon secret admin_access_token --generate makes one). To open it from another machine, use an SSH
    # tunnel (README.md, "The admin dashboard").
    admin:
      # true: serve the admin dashboard. false: none (the team dashboard and found-issue tickets need it). Required.
      enabled: true
      # The port to serve it on. Default: 18614.
      port: 18614
    # Team dashboard: a copy of the dashboard for other teams to read, without the admin buttons.
    team:
      # true: deliver the team dashboard (it needs the admin dashboard). false: none. To turn it on, see
      # QUICK_USER_GUIDE.md, "Turn on a feature". Required.
      enabled: false
  # Alerts: short messages to a Microsoft Teams channel, through a Teams Workflows webhook. Install the webhook's
  # address with: rkmon secret teams_webhook FILE. Each alert has its own on/off switch.
  alerts:
    # Develop breaks: a message when develop's CI is red and no fix is planned. It needs the develop guard.
    develop_breaks:
      # true: send this alert. false: breaks only show on the dashboards. Required.
      enabled: false
    # Needs you: a message for each decision that waits for you on the admin dashboard (a merge to approve, a hold
    # that has waited too long, a priority request, a pull request ready for your approval, an in-depth review to
    # approve). Each is sent once, addressed to your GitHub username, with a link to the team dashboard when its
    # public_url is set.
    needs_you:
      # true: send this alert. false: decisions only show on the dashboards. Required.
      enabled: false
  # Ticket creation: JIRA tickets, filed with the atlassian tool on the site in integrations.atlassian_cli.jira_url.
  # Install its login with: rkmon login atlassian (the login needs permission to create tickets). Each type of ticket
  # has its own on/off switch.
  ticket_creation:
    # Follow-up tickets: when a pull request merges with review items still open, each one becomes a ticket for the
    # pull request's author. Open items are the monitor's own blocking and should-fix findings, review comments
    # answered with a promise to fix them later ("next PR", "TODO"), and reviewers' requests nobody answered. Never
    # nits. Tickets are filed without asking you, so agree on this with your team first. To assign a ticket, the
    # monitor needs the author's JIRA account (rkmon rk jira-account add LOGIN ACCOUNT); without it, the ticket is
    # filed unassigned.
    follow_up_tickets:
      # true: file them (in live). It needs review automation. false: only list the items for you. Required.
      enabled: false
    # Found-issue tickets: when the monitor finds a problem that belongs to no pull request and isn't a develop break,
    # such as a flaky test, it proposes a ticket on the admin dashboard. Approve files it; Decline drops it.
    found_issue_tickets:
      # true: propose tickets. It needs the admin dashboard. false: only report such problems to you. Required.
      enabled: false
  # Weekly report: every week, a statistics report and notes for a lessons-learned review, saved in the monitor's
  # rocke_weekly/ folder (README.md, "Where things live").
  weekly_report:
    # true: write them every week. false: don't; rkmon rk stats and rkmon rk lessons write them when you ask.
    # Required.
    enabled: false

# Integrations: the tools the monitor uses. rkmon login lists the logins they need and installs each one.
integrations:
  # GitHub CLI: how the monitor reads and writes GitHub. It needs two logins: rkmon login github_read for reading, and
  # rkmon login github_write for posting. README.md, "Logins and secrets", lists the permissions each one needs.
  github_cli:
    # Your GitHub username. The monitor posts as this user, and treats this user's actions as yours: your comments,
    # your approvals (an approval lifts a hold), and auto-merges you turned on (which the guard leaves alone). Needs
    # you alerts are addressed to this user. Required.
    username: "<user_login>"
    # The most GitHub API requests this monitor may make in an hour. GitHub's hourly limit is per account: if several
    # monitors share one account, give each a cap so that together they stay under the limit. A gh command that reads
    # several pages of results counts once toward this cap, though GitHub counts each page: leave headroom. 0: no cap.
    # Default: 0.
    max_requests_per_hour: 0
  # Atlassian CLI: not used while no type of ticket is on.
  atlassian_cli: {}
  # Claude Code: not used while review automation is off.
  claude_code: {}

# Model bindings: not used while review automation is off. Default: none.
model_bindings: {}
"""

DEPLOY_EVERY_KEY = """\
# rocKE monitor: deploy settings (profile "full": reviews, the merge queue and the admin dashboard).
# These settings decide what one monitor on this host does. QUICK_DEPLOY_GUIDE.md walks you through them, and
# README.md explains each one.
# This file is yours: keep it out of git and out of the package folder. Never put a password, token or key in it:
# rkmon login installs those.
# Every feature starts in "prototype" mode: it does its work and shows you the result on the admin dashboard, but
# posts and sends nothing. Switch a feature to "live" when you are happy with it (QUICK_USER_GUIDE.md, "Go live").
# Put text in double quotes. Replace every <...>, then run: rkmon check --settings FILE
# To change a setting later, see QUICK_USER_GUIDE.md, "Change a setting".

# The type of settings file. Don't change it.
kind: "deploy_settings"
# The layout version of this file. Don't change it: rkmon migrate updates it when a new package changes the layout.
schema_version: 2

# Container: the Docker container this monitor runs in.
container:
  # The container's name. The monitor keeps its data under this name, so choose it once:
  # rkmon confirm refuses a new name while this container exists. To run a second monitor on this host, see
  # QUICK_DEPLOY_GUIDE.md, "Several deployments". While features.review_automation.automated_reviewer_name is blank,
  # every review gives this name on its first line. Lowercase letters, digits, - and _. Default: "rocke-monitor".
  name: "rocke-monitor"

# Image: the container image to run, <registry>/<name>:<tag>, as your build settings name it.
image:
  # The registry to pull the image from, such as "registry.example.com/my-team". Run docker login for it first. Leave
  # blank if you load the image from a file instead (rkmon image load FILE).
  registry: ""
  # The image's name, as in your build settings, without its registry, which goes in image.registry. Default:
  # "rocke-monitor".
  name: "rocke-monitor"
  # The image's tag, as in your build settings. To move to a newer image, change the tag (QUICK_USER_GUIDE.md,
  # "Change a setting"). Required.
  tag: "2026.11.15"

# Updates: where rkmon package update fetches newer code for this package. Leave this section out to update only from
# a folder or an archive (rkmon package update --from).
updates:
  # The GitHub repository to fetch this package from, as <org>/<repo>. Example: "ROCm/rocm-libraries". Required.
  repository: "<org>/<repo>"
  # The branch to fetch the newest code from. Example: "develop". Required.
  branch: "<branch>"

# Repository: the GitHub repository the monitor watches, and which of its pull requests the monitor works on.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The folder your project lives in, from the repository's root, without a / at either end. The monitor uses it to
  # tell your project's files from the rest of the repository. Example: "projects/my-project". Required.
  folder: "<folder>"
  # The labels that put a pull request in scope: the monitor works on every open pull request that has at least one of
  # them. To see the repository's labels, run: gh label list -R <org>/<repo>. Example: ["my-project"]. Required.
  scope_labels: ["<label>"]
  # Labels that several teams share, each with a folder. A pull request with one of these labels is in scope only if
  # it also changes files in that label's folder. Write each one as "label": "folder", the folder from the
  # repository's root. Example: {"needs-ci": "projects/my-project"}: a "needs-ci" pull request is in scope only if it
  # changes files in projects/my-project. Default: none.
  shared_label_folders: {}

# Host: how much of this host the container may use, and which user it runs as.
host:
  # Resources: the container's share of this host. QUICK_DEPLOY_GUIDE.md, "Machine sizing", helps you choose.
  resources:
    # The number of CPUs the container may use. Default: 4.
    cpus: 4
    # The container's memory limit, in GiB. At least ramdisk_gib + 8. Default: 8.
    memory_gib: 8
    # A RAM disk that makes reviews and builds faster, in GiB. It is part of memory_gib. Allow about 1.5 GiB for each
    # review and 2 GiB for each build that run at the same time. 0: no RAM disk. Default: 0.
    ramdisk_gib: 0
  # Users: the user and group the monitor runs as inside the container.
  users:
    # The user id, 1000 or more, other than 65534, the system's "nobody" user, and none of the package's own users'
    # (the GitHub writer's and the PR runners'). Default: null, which means your own user id on this host.
    monitor_uid: null
    # The group id, 1 or more: any group's, a system group's included, but none of the PR runners' groups'. Default:
    # null, which means your own group id on this host.
    monitor_gid: null

# Monitor: the monitor works in ticks. Each tick, it reads GitHub and updates the reviews, the merge queue and the
# dashboards.
monitor:
  # Minutes between ticks. Default: 10.
  tick_interval_minutes: 10
  # Most ticks read only what changed on GitHub. At least this often, in minutes, a tick reads everything again. 0:
  # every tick reads everything. Default: 60.
  full_tick_max_interval_minutes: 60
  # How many local builds, each with its tests, may run at the same time at first, from 1 to 8. Reviews, the merge
  # queue and the develop guard all use these builds. The monitor then adjusts the number, within 1 to 8, to how busy
  # this host is. Default: 1.
  parallel_builds: 1

# Features: what the monitor does. Each feature that posts or sends runs in prototype mode or live mode:
#   "prototype": it does its work and shows you the result on the admin dashboard, but posts and sends nothing.
#   "live": it posts and sends.
features:
  # Review automation: Claude reviews every new or updated pull request in scope.
  review_automation:
    # true: review pull requests. false: no reviews and no Claude; decisions that need judgement wait for you on the
    # admin dashboard. Required.
    enabled: true
    # "prototype": reviews are written and shown on the admin dashboard, but not posted. "live": reviews are posted on
    # GitHub as integrations.github_cli.username, and so are notices about pull requests that change the same code.
    # Required when review automation is on.
    mode: "prototype"
    # The name at the top of every review this monitor posts: "Automated review by <automated_reviewer_name>".
    # Letters, digits, spaces, - and _. Example: "Platform team". Leave blank to use the container's name.
    automated_reviewer_name: ""
    # A folder with your team's own review skills, as Markdown (.md) files only. Its layout must match the package's
    # agentic/ folder, and it may contain only team_members/, agents/, review_book.md and
    # skills/rocke-monitor/review_procedure.md. A file with the same name as a package file replaces it; a file with a
    # new name is added. Leave blank to use the package's skills only.
    skills_folder: ""
    # Orchestrator: the main Claude session. It runs the reviews, checks and posts them, and makes the merge queue's
    # judgement calls.
    orchestrator:
      # Its model: "opus", "sonnet", "haiku", or the name of a Claude model in model_bindings. Default: "opus".
      model: "opus"
      # How hard it thinks: "low", "medium", "high", "xhigh" or "max". Higher costs more and takes longer. Default:
      # "high".
      effort: "high"
    # Schedule: when reviews start.
    schedule:
      # How long to wait after a pull request's latest push before reviewing it, in minutes, so that a burst of pushes
      # gets one review. A burst waits 45 minutes after its last push, and your priorities and the pull requests that
      # fix develop wait at most 10 minutes after any push, a burst's too. Set above 45, it applies to bursts too. 0:
      # a single push doesn't wait. Default: 30.
      push_cooldown_minutes: 30
      # The shortest time between two follow-up reviews of the same pull request, in minutes. Your priorities, pull
      # requests that fix develop, and pushes that only merge develop in don't wait. 0: don't wait. Default: 60.
      follow_up_cooldown_minutes: 60
    # How many reviews may run at the same time at first, from 1 to 5. The monitor then adjusts the number, within 1
    # to 5, to how busy this host is. Default: 3.
    parallel_reviews: 3
    # Large pull requests: how very large pull requests are reviewed.
    large_prs:
      # A pull request with more changed lines than this gets one in-depth review, split into parts, and after that
      # only light follow-ups. The in-depth review starts only once you approve it: the admin dashboard's Needs you
      # card gives its size and cost, with Approve and Decline, or run: rkmon rk review-approve N. Default: 30000.
      in_depth_review_over_lines: 30000
      # The most changed lines in each part of an in-depth review, from 6000 to 10000. Default: 8000.
      split_part_lines: 8000
    # Passes: the steps of the review work, each with its model and effort.
    #   Models: "opus", "sonnet", "haiku", or the name of an entry in model_bindings.
    #   Efforts: "low", "medium", "high", "xhigh" or "max". Ollama and OpenAI-compatible models take only "low",
    #   "medium" or "high".
    #   The verification and code quality passes always run, each on a Claude model. The four optional passes after
    #   them let another model do part of the work first; "default" means no other model, so the pass in charge does
    #   that work itself.
    passes:
      # Verification pass: reviews each pull request on its own, checks every claim of the first review pass (if there
      # is one) against the code, drops what it can't confirm, and writes the review that gets posted.
      verification_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "high".
        effort: "high"
      # Code quality pass: reviews each pull request's design (single responsibility, no duplication, layering, names,
      # tests) and gives its findings to the verification pass.
      code_quality_pass:
        # Its model, which must be a Claude model. Default: "opus".
        model: "opus"
        # Its effort. Default: "medium".
        effort: "medium"
      # First review pass (optional): another model writes a first draft of each review, which the verification pass
      # then checks claim by claim.
      first_review_pass:
        # The model that writes the draft. Default: "default", which means no draft: the verification pass reviews
        # alone.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # CI triage pass (optional): another model reads the pull request's failing CI jobs and tells the verification
      # pass what caused each failure.
      ci_triage_pass:
        # The model. Default: "default", which means none: the verification pass reads the failing CI jobs itself.
        model: "default"
        # Its effort. Default: "default", which means the verification pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # First code quality pass (optional): another model checks the pull request's changes against the code quality
      # rules first, and gives what it finds to the code quality pass to confirm.
      first_code_quality_pass:
        # The model. Default: "default", which means none: the code quality pass does all of the work.
        model: "default"
        # Its effort. Default: "default", which means the code quality pass's effort, or none for a model that isn't
        # Claude.
        effort: "default"
      # Notice draft pass (optional): another model writes the first draft of the notices the monitor posts on pull
      # requests: when two pull requests change the same code, and when a change on develop breaks a pull request's
      # tests. The orchestrator checks and posts them.
      notice_draft_pass:
        # The model. Default: "default", which means none: the orchestrator writes the notices itself.
        model: "default"
        # Its effort. Default: "default", which means the orchestrator's effort, or none for a model that isn't
        # Claude.
        effort: "default"
    # Remote GPU tests: some tests need a GPU, and on a host without one they are skipped. With this on, the monitor
    # runs them on a Slurm cluster instead: it logs in to the cluster's login node over SSH and submits each test run
    # as a Slurm job, which runs on a GPU node. A review doesn't wait for the results: its GPU tests row says
    # "pending" and is filled in when the job finishes.
    remote_gpu_tests:
      # true: run GPU tests on the cluster. false: skip them, as on any host without a GPU. Required when review
      # automation is on.
      enabled: false
      # Cluster: how the monitor reaches the cluster, and what each job asks for. Your cluster's documentation or its
      # admins have these values.
      cluster:
        # The host name of the cluster's login node, where the monitor submits jobs: letters, digits, '.', '-' and
        # '_', starting with a letter or a digit, with no user@ and no port. Example: "login.cluster.example.com".
        # Required when remote GPU tests are on.
        login_host: ""
        # The login node's SSH port. Default: 22.
        ssh_port: 22
        # The user name the monitor logs in to the login node with: letters, digits, '.', '-' and '_', starting with a
        # letter or '_'. Every pull request's tests run as this user on the GPU nodes, so use an account made for the
        # monitor, not your own. Install this user's SSH key with: rkmon secret gpu_ssh_key FILE. Required when remote
        # GPU tests are on.
        ssh_user: ""
        # The Slurm partition to submit jobs to (sbatch --partition): letters, digits, '.', ',', '&', '|', '!', '-'
        # and '_'. Required when remote GPU tests are on.
        partition: ""
        # The Slurm accounts to charge jobs to (sbatch --account), in order: if the cluster refuses one, the monitor
        # tries the next. Each is letters, digits, '.', '-' and '_'. Required when remote GPU tests are on.
        accounts: []
        # The Slurm quality of service for the jobs (sbatch --qos): letters, digits, '.', ',', '&', '|', '!', '-' and
        # '_'. Leave blank for the account's default.
        qos: ""
        # A Slurm node constraint (sbatch --constraint), for example to use only nodes that can read your home folder:
        # letters, digits, '.', ',', '&', '|', '!', '-' and '_'. Leave blank for none.
        constraint: ""
        # For each GPU target the tests use ("gfx", then lowercase letters and digits), the cluster's name for that
        # type of GPU, as in sbatch --gres=gpu:<type>:<count>: letters, digits, '.', ':', '-' and '_'. Example:
        # {"gfx942": "mi300x"}. Required when remote GPU tests are on.
        gpu_types: {}
        # The GPU target for tests that don't say which one they need: one of gpu_types' targets. Leave blank to use
        # the first entry in gpu_types.
        default_target: ""
        # GPUs per job. The tests use one. Default: 1.
        gpus_per_job: 1
        # Each job's time limit, as Slurm writes it ("HH:MM:SS", "D-HH:MM:SS" or minutes). Allow about 1.5 times a
        # normal test run. Default: "01:00:00".
        time_limit: "01:00:00"
        # GPU nodes to avoid, such as broken ones, each named as Slurm names nodes: letters, digits, '.', ',', '[',
        # ']', '-' and '_', such as "node[01-04]". Each job first checks that its node's GPU works; when a node fails
        # that check, the monitor tells you the node's name, so you can add it here. Default: none.
        excluded_nodes: []
      # Node: where jobs keep their files on the cluster, and how each GPU node is prepared.
      node:
        # The folder on the cluster for the monitor's copy of the repository and each job's files, on one line:
        # relative to ssh_user's home folder, or absolute. The GPU nodes must be able to read it, for example on a
        # shared home folder. Default: "rkmon-gpu".
        work_folder: "rkmon-gpu"
        # The Python on the GPU nodes that has the tests' packages, pytest included, on one line; usually a virtual
        # environment's bin/python. Example: "~/venvs/tests/bin/python". Required when remote GPU tests are on.
        python: ""
        # A shell command each job runs first, on one line, such as one that loads the cluster's modules. Example:
        # "module load rocm". Leave blank for none.
        setup_command: ""
        # Environment variables each job sets, as NAME: "value": each NAME letters, digits and '_', starting with a
        # letter or '_', and each value on one line. Example: {"HIP_VISIBLE_DEVICES": "0"}. Default: none.
        environment: {}
        # Where the cluster clones the repository from, if not GitHub (for example, a mirror inside your network): an
        # address with no spaces. Leave blank to clone from GitHub.
        clone_url: ""
      # The most GPU targets one review runs tests on, one job each. The cluster's own limits decide how many jobs run
      # at the same time. Default: 2.
      jobs_per_review: 2
      # If a job is still waiting for a GPU node after this many minutes, the monitor tells you, once. The job keeps
      # waiting. 0: never tell you. Default: 240.
      wait_report_after_minutes: 240
      # How many times to resubmit a job on another node when its node fails the GPU check. Default: 1.
      bad_node_resubmits: 1
    # Spend: what the reviews cost in US dollars, shown on the dashboards and in the weekly report.
    spend:
      # true: work out the cost from the prices in usd_per_million_tokens. false: don't show costs. Required when
      # review automation is on.
      enabled: false
      # The price in US dollars per million tokens of each model the monitor uses, from Anthropic's price list or your
      # gateway's. Name each model by its id: claude-opus-5-5 for "opus", claude-sonnet-5-5 for "sonnet",
      # claude-haiku-5-5 for "haiku", and the model_id of a model_bindings entry. Add one entry per model, as shown
      # below; a model with no entry has its cost shown as unknown. Default: none.
      usd_per_million_tokens: {}
      #   claude-opus-5-5:
      #     # Input tokens. Required.
      #     input: "<price>"
      #     # Output tokens. Required.
      #     output: "<price>"
      #     # Tokens read from the prompt cache. Required.
      #     cache_read: "<price>"
      #     # Tokens written to the prompt cache for 5 minutes. Required.
      #     cache_write_5_minutes: "<price>"
      #     # Tokens written to the prompt cache for 1 hour. Required.
      #     cache_write_1_hour: "<price>"
  # Merge queue: puts the pull requests in scope in merge order, runs its gates (the checks a pull request must pass
  # before it merges), and makes a merge plan (which pull requests can merge together without conflicts). Your
  # priorities go first (rkmon rk prioritize N, or a priority request you approve on the admin dashboard).
  merge_queue:
    # true: run the merge queue (only one monitor per repository may). false: no merge queue; your priorities still
    # decide which pull requests are reviewed first. Required.
    enabled: true
    # "prototype": the queue, its gates and the merge plan are worked out and shown to you, but nothing happens on
    # GitHub: no auto-merge is turned on, and nothing is held, commented on or filed. "live": the queue does these
    # things on GitHub, as the settings below allow. Required when the merge queue is on.
    mode: "prototype"
    # Auto-merge: when a pull request you cleared for merging (rkmon rk clear N) passes the gates, the monitor turns
    # on GitHub's auto-merge for it. GitHub's branch protection still has the final say.
    auto_merge:
      # true: turn on auto-merge (in prototype, the dashboard only shows where it would). false: only propose the
      # merge order to you. Required when the merge queue is on.
      enabled: false
    # Auto-merge guard: when someone turns on auto-merge for a pull request that fails a gate, the monitor comments on
    # the pull request and turns its auto-merge off (in prototype, it only tells you). It never turns off an
    # auto-merge you turned on yourself. It runs whenever the merge queue runs.
    auto_merge_guard:
      # What to do when someone else turns on auto-merge for a pull request you wrote: "alert_only" (only tell you) or
      # "same_as_others" (comment and turn it off, as for anyone else's pull request). Default: "alert_only".
      on_your_pull_requests: "alert_only"
    # Auto-hold: when a pull request fails a gate because of its own changes, the monitor posts a "Request changes"
    # review on it from your account. That blocks the merge until the author fixes it and you approve it again.
    auto_hold:
      # true: hold such pull requests (in prototype, only show them). false: only report the failure to you. Required
      # when the merge queue is on.
      enabled: true
    # Queue notices: each pull request near the front of the queue gets one comment about its place in the queue,
    # updated as things change.
    queue_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: true
    # Drift notices: when a change on develop breaks a pull request's own tests, the monitor tells its author which
    # develop commit changed the behaviour. Claude writes these, so they need review automation.
    drift_notices:
      # true: post them (in live). false: none. Required when the merge queue is on.
      enabled: true
    # Develop guard: after every merge, the monitor checks develop and traces any break to the pull request that
    # caused it. In live, each break becomes a GitHub issue, and an open pull request that fixes it is linked to the
    # issue. Without review automation, breaks wait for you on the admin dashboard instead.
    develop_guard:
      # true: check develop. false: don't. Required when the merge queue is on.
      enabled: true
      # The repository to file develop issues in, as <org>/<repo>. Leave blank to use repository.name.
      issue_repository: ""
      # The label for an issue about a break that turns develop's CI red. The label must already exist in
      # issue_repository (rkmon rk label-create NAME --description TEXT --color HEX creates one). Example: "ci-red".
      # Required when the develop guard is on and the merge queue is live.
      ci_red_label: ""
      # More labels for those issues. Each must already exist. Default: none.
      ci_red_extra_labels: []
      # The label for an issue about a break that CI doesn't catch; the issue is assigned to the author of the pull
      # request that caused it. The label must already exist in issue_repository. Example: "stability". Required when
      # the develop guard is on and the merge queue is live.
      stability_label: ""
  # Dashboards: the admin dashboard (yours) and the team dashboard (for other teams).
  dashboards:
    # Admin dashboard: everything the monitor shows, with Approve and Decline buttons for priority requests, proposed
    # tickets and the in-depth reviews of large pull requests. Only this host can open it, at
    # http://127.0.0.1:<port>/. It asks for a user name and a password: type any user name, and the access token as
    # the password (rkmon secret admin_access_token --generate makes one). To open it from another machine, use an SSH
    # tunnel (README.md, "The admin dashboard").
    admin:
      # true: serve the admin dashboard. false: none (the team dashboard and found-issue tickets need it). Required.
      enabled: true
      # The port to serve it on. Default: 18614.
      port: 18614
    # Team dashboard: a copy of the dashboard for other teams to read, without the admin buttons.
    team:
      # true: deliver the team dashboard (it needs the admin dashboard). false: none. Required.
      enabled: false
      # "prototype": the page is made every tick and shown on the admin dashboard, but not published. "live": the page
      # is delivered as delivery says. Required when the team dashboard is on.
      mode: "prototype"
      # The web address other teams open the page at; the Needs you alerts (below) link to it. Example:
      # "https://dashboards.example.com/my-team/". With priority requests on and delivery "publish", the monitor's
      # request service serves the page over plain http, at the root of its own port: give http, a port of 1024 or
      # more and no path, with a host that is an IP address or a name of letters, digits, "." and "-". Example:
      # "http://dashboards.example.com:8080/". Required when delivery is "publish", or when serve's address is
      # "0.0.0.0" or "::". Otherwise, leave blank to use serve's address and port.
      public_url: ""
      # Priority requests: a box on the page where other teams can ask for a pull request to be prioritized. Each
      # request waits for your Approve or Decline on the admin dashboard. With delivery "publish", the web host needs
      # the monitor's request service (README.md, "The team dashboard", says how to install it).
      priority_requests:
        # true: show the box. false: no box. Required when the team dashboard is on.
        enabled: false
      # How the page reaches other teams: "publish" (copied to a web host over SSH every tick) or "serve" (served by
      # this monitor from this host: anyone who can reach that address and port can open it). Fill in the matching
      # block below and delete the other. Required when the team dashboard is on.
      delivery: "publish"
      # Publish: the web host to copy the page to. Install the SSH key for it with: rkmon secret web_host_ssh_key FILE
      publish:
        # The web host's name: letters, digits, '.', '-' and '_', starting with a letter, a digit or '_'. Example:
        # "web.example.com". Required when delivery is "publish".
        web_host: ""
        # The user name to log in to the web host with: letters, digits, '.', '-' and '_', starting with a letter, a
        # digit or '_'. Required when delivery is "publish".
        username: ""
        # The folder the web host serves the page from, relative to that user's home folder: folder names of letters,
        # digits, '.', '-' and '_', each starting with a letter, a digit or '_', joined by '/'. Example:
        # "public_html/my-team". Required when delivery is "publish".
        site_folder: ""
      # Serve: use this block instead of publish when delivery is "serve".
      # serve:
      #   # The address of this host to serve the page on: "127.0.0.1" (this host only), one of this host's addresses,
      #   # or "0.0.0.0" or "::" (all of them; then set public_url). In prototype the page is served on "127.0.0.1"
      #   # only, whatever this says. Default: "127.0.0.1".
      #   address: "127.0.0.1"
      #   # The port to serve it on. Default: 18615.
      #   port: 18615
  # Alerts: short messages to a Microsoft Teams channel, through a Teams Workflows webhook. Install the webhook's
  # address with: rkmon secret teams_webhook FILE. Each alert has its own on/off switch.
  alerts:
    # "prototype": messages are written and listed on the admin dashboard, but not sent. "live": messages are sent.
    # Required when an alert is on.
    mode: "prototype"
    # Text to start every message with, such as your team's name. Leave blank for none.
    message_label: ""
    # The shortest time between two messages of the same alert, in minutes; anything new in between goes into the next
    # one. 0: don't wait. Default: 30.
    message_cooldown_minutes: 30
    # Develop breaks: a message when develop's CI is red and no fix is planned. It needs the develop guard.
    develop_breaks:
      # true: send this alert. false: breaks only show on the dashboards. Required.
      enabled: false
    # Needs you: a message for each decision that waits for you on the admin dashboard (a merge to approve, a hold
    # that has waited too long, a priority request, a pull request ready for your approval, an in-depth review to
    # approve). Each is sent once, addressed to your GitHub username, with a link to the team dashboard when its
    # public_url is set.
    needs_you:
      # true: send this alert. false: decisions only show on the dashboards. Required.
      enabled: false
  # Ticket creation: JIRA tickets, filed with the atlassian tool on the site in integrations.atlassian_cli.jira_url.
  # Install its login with: rkmon login atlassian (the login needs permission to create tickets). Each type of ticket
  # has its own on/off switch.
  ticket_creation:
    # "prototype": tickets are written and listed on the admin dashboard, but not filed. "live": tickets are filed.
    # Required when a type of ticket is on.
    mode: "prototype"
    # The key of the JIRA project to file tickets in. Example: "PROJ". Required when a type of ticket is on.
    project_key: ""
    # The issue type of new tickets, as your JIRA project writes it, case and all, and never a sub-task type: a
    # sub-task needs a parent, which the monitor never gives. Example: "Story". Leave blank for "Task".
    issue_type: ""
    # The epic to link new tickets to, by its key. Example: "<PROJECT>-123". With an epic, a new ticket gets its
    # description only after it is linked, so a project that requires a description when a ticket is created refuses
    # the ticket. Leave blank for no epic.
    epic_key: ""
    # Components to set on every new ticket, each one your JIRA project allows. A project that requires components on
    # a new ticket needs at least one here. Default: none.
    components: []
    # Labels to set on every new ticket. A project that requires labels on a new ticket needs at least one here.
    # Default: none.
    labels: []
    # Other fields your JIRA project requires on a new ticket: a line for each, its field id and its value, such as
    # customfield_10010: "High". Set only the fields JIRA requires and gives no value of its own, never one the
    # monitor or JIRA fills: the monitor fills the project, the summary, the issue type, the priority, the labels and
    # components their keys list, and the description while epic_key is blank; JIRA fills the reporter.
    # rkmon check asks JIRA which fields these are. On a terminal it then asks you for each one's value, from those
    # JIRA allows, and on your yes adds their lines here, keeping the file as it was as FILE.bak: in place of {},
    # after the last entry, or with required_fields itself at the end of features.ticket_creation. It adds none to a
    # required_fields written as {...} with entries: add those lines by hand. Default: none.
    required_fields: {}
    # Follow-up tickets: when a pull request merges with review items still open, each one becomes a ticket for the
    # pull request's author. Open items are the monitor's own blocking and should-fix findings, review comments
    # answered with a promise to fix them later ("next PR", "TODO"), and reviewers' requests nobody answered. Never
    # nits. Tickets are filed without asking you, so agree on this with your team first. To assign a ticket, the
    # monitor needs the author's JIRA account (rkmon rk jira-account add LOGIN ACCOUNT); without it, the ticket is
    # filed unassigned.
    follow_up_tickets:
      # true: file them (in live). It needs review automation. false: only list the items for you. Required.
      enabled: false
    # Found-issue tickets: when the monitor finds a problem that belongs to no pull request and isn't a develop break,
    # such as a flaky test, it proposes a ticket on the admin dashboard. Approve files it; Decline drops it.
    found_issue_tickets:
      # true: propose tickets. It needs the admin dashboard. false: only report such problems to you. Required.
      enabled: false
  # Weekly report: every week, a statistics report and notes for a lessons-learned review, saved in the monitor's
  # rocke_weekly/ folder (README.md, "Where things live").
  weekly_report:
    # true: write them every week. false: don't; rkmon rk stats and rkmon rk lessons write them when you ask.
    # Required.
    enabled: false

# Integrations: the tools the monitor uses. rkmon login lists the logins they need and installs each one.
integrations:
  # GitHub CLI: how the monitor reads and writes GitHub. It needs two logins: rkmon login github_read for reading, and
  # rkmon login github_write for posting. README.md, "Logins and secrets", lists the permissions each one needs.
  github_cli:
    # Your GitHub username. The monitor posts as this user, and treats this user's actions as yours: your comments,
    # your approvals (an approval lifts a hold), and auto-merges you turned on (which the guard leaves alone). Needs
    # you alerts are addressed to this user. Required.
    username: "<user_login>"
    # The most GitHub API requests this monitor may make in an hour. GitHub's hourly limit is per account: if several
    # monitors share one account, give each a cap so that together they stay under the limit. A gh command that reads
    # several pages of results counts once toward this cap, though GitHub counts each page: leave headroom. 0: no cap.
    # Default: 0.
    max_requests_per_hour: 0
  # Atlassian CLI: how the monitor files JIRA tickets. Install its login with: rkmon login atlassian
  atlassian_cli:
    # Your JIRA site's address. Example: "https://<site>.atlassian.net". Required when a type of ticket is on.
    jira_url: ""
  # Claude Code: how the monitor reaches Claude. Install the key with: rkmon login claude (an Anthropic API key, or
  # your LLM gateway's token).
  claude_code:
    # The address to reach Claude at, if you use an LLM gateway (Claude Code's ANTHROPIC_BASE_URL). Example:
    # "https://llm-gateway.example.com". Leave blank to use Anthropic's own API.
    base_url: ""
    # How the key is sent: "header" (in the header that api_key_header names), "bearer" (as
    # "Authorization: Bearer <key>") or "header_and_bearer" (both). Your gateway's documentation says which. Default:
    # "header".
    api_key_sent_as: "header"
    # The header the key goes in, for "header" and "header_and_bearer". Leave blank for Anthropic's own header,
    # "x-api-key".
    api_key_header: ""
    # Text to put before the key in that header, such as "Bearer". Leave blank to send the key alone.
    api_key_scheme: ""

# Model bindings: other models the passes can use, each under a name you choose. "opus", "sonnet" and "haiku" are
# built in, reached through integrations.claude_code, and need no entry. To add a model, replace {} with entries in
# one of the three forms that rkmon template section model_bindings prints. Default: none.
model_bindings: {}
# A Claude model, reached through integrations.claude_code:
#   <name>:
#     # "claude". Required.
#     provider: "claude"
#     # The model's id. Example: "claude-sonnet-5-5". Required.
#     model_id: "<model>"
# A model behind an OpenAI-compatible API (an LLM gateway, vLLM, LiteLLM):
#   <name>:
#     # "openai_compatible". Required.
#     provider: "openai_compatible"
#     # The model's id, as the API lists it. Required.
#     model_id: "<model>"
#     # The API's base address. Example: "https://llm-gateway.example.com/v1". Required.
#     base_url: "<url>"
#     # The name of the secret that contains the API key (not the key itself): lowercase letters, digits and _, ending
#     # in _api_key. Install the key with: rkmon secret <name>_api_key FILE. Leave blank if the API needs no key.
#     api_key_secret_name: "<name>_api_key"
#     # How the key is sent: "bearer" (as "Authorization: Bearer <key>", with api_key_header and api_key_scheme not
#     # used), "header" (in the header that api_key_header names, after api_key_scheme) or "header_and_bearer" (both).
#     # Default: "bearer".
#     api_key_sent_as: "bearer"
#     # The header the key goes in; with api_key_sent_as "bearer" it is not used. Required when api_key_sent_as is
#     # "header" or "header_and_bearer".
#     api_key_header: ""
#     # Text to put before the key in api_key_header's header, such as "Bearer"; with api_key_sent_as "bearer" it is
#     # not used. Leave blank to send the key alone.
#     api_key_scheme: ""
#     # Limits: what the API allows.
#     limits:
#       # Requests per minute. 0: unknown (the monitor still slows down when the API says it is busy). Default: 0.
#       requests_per_minute: 0
#       # Tokens per minute. 0: unknown. Default: 0.
#       tokens_per_minute: 0
#       # Requests at the same time, across all passes. 0: no limit. Default: 0.
#       max_parallel_requests: 0
#     # If a request gets no new output for this many seconds, it is stopped and sent again. Default: 300.
#     stall_timeout_seconds: 300
# A model on an Ollama server:
#   <name>:
#     # "ollama". Required.
#     provider: "ollama"
#     # The model, as ollama list shows it. Example: "llama3.1:8b". Required.
#     model_id: "<model>:<tag>"
#     # The Ollama server's address. Inside the container, localhost is the container itself, so use the host's
#     # address. Example: "http://<host address>:11434". Required.
#     server_url: "<url>"
#     # Limits: what the server allows.
#     limits:
#       # Requests at the same time. One model in memory serves one request well. 0: no limit. Default: 1.
#       max_parallel_requests: 1
#     # If a request gets no new output for this many seconds, it is stopped and sent again. Default: 300.
#     stall_timeout_seconds: 300
"""

TEXTS = {
    "build": BUILD,
    "deploy full": DEPLOY_FULL,
    "deploy review-only": DEPLOY_REVIEW_ONLY,
    "deploy merge-queue": DEPLOY_MERGE_QUEUE,
    "deploy every key": DEPLOY_EVERY_KEY,
}
