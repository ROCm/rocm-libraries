<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# The agentic files

The Claude Code side of the rocKE monitor: the skill that runs each tick (`skills/rocke-monitor/`: `SKILL.md`, its
procedures, rules, lessons, the weekly review), the two reviewer team members and their agent types
(`team_members/`, `agents/`), their rows for the workspace's dispatch table (`team_members/dispatch_rows.md`), the
workspace's shared policies the skill reads (`skills/shared/`), and the local review loop's rubric (`review_book.md`).
They are the package's own: change them here, in `agentic/`, and commit them. A team's own review skills live outside
the package, in the folder `features.review_automation.skills_folder` names, laid out as this one (what it may hold:
that key's row in the package's README, [Deploy settings](../README.md#deploy-settings)). When these files are filled
in for the monitor, the team's are laid over them: a file with the path of one here takes its place, and one with a
new path is added. A team changes its skills in that folder, then runs `./rkmon confirm` and `./rkmon apply`.

These are templates: every site-specific value is a placeholder from the table below. The operator is `<user_login>`
everywhere an agent reads (the deploy settings' `integrations.github_cli.username`; the toolkit fills it in), and
other people are named by their role (the author, a contributor, a teammate). `rk provision`, which every start of the
monitor runs to set the work volume up, fills them in with this deployment's configuration into
`<WORKING_DIR>/agentic/` (outside the checkout; a blank value reads "(not set up)"), and installs the skill, team
members and agent types into `<CLAUDE_WORKSPACE>/.claude/`. A changed file is backed up first, beside it, with `.bak-`
and the time in UTC after its name (`SKILL.md.bak-20261124093000`); a file an earlier install put there that the new
render lacks is removed (backed up first the same way when edited since); shared policies the workspace already has
are kept; and the registry rows go between the dispatch table's generated-registry sentinels, in place of the rocKE
rows there. The orchestrator (`rk orchestrate`) renders and installs them again as it starts when they are older than
the configuration, and runs `/rocke-monitor` in that workspace.

`tests/test_agentic.py` checks them (what it checks: docs/maintaining.md, "Changing the agentic files"). The privacy
check reads every build.

<!-- BEGIN GENERATED: agentic-placeholders
     source: rkmon_lib/agentic.py, PLACEHOLDERS
     refresh: python3 -B tools/make_package.py --refresh-generated -->

| Placeholder | What it is |
|---|---|
| `<user_login>` | the operator's GitHub login (integrations.github_cli.username): the account the monitor posts as, and whose instructions it follows |
| `<AUTOMATED_REVIEWER_NAME>` | this deployment's automated reviewer name, which line 1 of each review and the first line of each notice give (features.review_automation.automated_reviewer_name, else container.name) |
| `<WORKING_DIR>` | the folder the monitor owns |
| `<OUTPUTS_DIR>` | where the operator's files go |
| `<TOOLKIT_DIR>` | the monitor's home: rk and state/ (on a host, config.json too) |
| `<PACKAGE_DIR>` | the package's code (read-only) |
| `<RK>` | the monitor's rk command |
| `<ROCKE_REPO>` | rocKE's GitHub repository, as owner/name |
| `<ROCKE_PATH>` | rocKE's folder in it |
| `<ROCKE_SRC_DIR>` | the monitor's own clone |
| `<ROCKE_URL>` | rocKE's folder on GitHub |
| `<ROCKE_GITHUB_ISSUES>` | where develop issues are filed |
| `<SECRETS_DIR>` | where this deployment's secrets live |
| `<REVIEW_BOOK>` | the local review loop's rubric (review_book.md) |
| `<REVIEW_CHECKLIST>` | the shared rocKE review checklist |
| `<WORKER_MODEL>` | the reviewer model (the first pass) |
| `<WORKER_URL>` | the reviewer model's endpoint |
| `<ORCHESTRATOR_MODEL>` | the orchestrator model's id, as its session sends it (features.review_automation.orchestrator.model: a built-in model's name gives its id); the verifier and code quality agents run features.review_automation.passes.verification_pass.model and features.review_automation.passes.code_quality_pass.model |
| `<ORCHESTRATOR_NAME>` | the orchestrator model in words, for a review's footer |
| `<AUDIT_MODEL>` | Claude's model for the verifier, which checks the first pass's review (features.review_automation.passes.verification_pass.model) |
| `<AUDIT_NAME>` | the audit model in words, for a review's footer |
| `<AUDIT_ALIAS>` | the audit model by the name the Agent tool's `model` option takes, its family (opus, sonnet, haiku or fable); blank for a model of none |
| `<DEEP_MODEL>` | Claude's model for the code quality pass and the in-depth work on a big inconsistency (features.review_automation.passes.code_quality_pass.model) |
| `<DEEP_NAME>` | the deep review's model in words, for its footer |
| `<DEEP_ALIAS>` | the deep model (the code quality pass's, a deep review's) by the name the Agent tool's `model` option takes, its family (opus, sonnet, haiku or fable); blank for a model of none |
| `<CLAUDE_EFFORT>` | the orchestrator session's effort (features.review_automation.orchestrator.effort) |
| `<VERIFY_EFFORT>` | the verifier agent's effort (features.review_automation.passes.verification_pass.effort) |
| `<QUALITY_EFFORT>` | the code quality agent's effort (features.review_automation.passes.code_quality_pass.effort) |
| `<AUDIT_EFFORT>` | the offload audit's effort (the package defaults' effort.audit, else claude.effort) |
| `<DEEP_EFFORT>` | a deep review's effort (the package defaults' effort.deep, else claude.effort) |
| `<OFFLOAD_MODE>` | the review offload (the package defaults' offload.mode): off, shadow or on |
| `<OFFLOAD_SPOT_CHECK>` | the auditor's sample of the 🟡/🟢 claims (the package defaults' offload.spot_check and offload.spot_check_min) |
| `<CHECKS_LIMIT>` | the parallel builds asked for (monitor.parallel_builds) |
| `<RUN_DIR>` | the shared rate budget and the process registry |
| `<WORKER_USER>` | the Linux account that runs the monitor |
| `<CLAUDE_WORKSPACE>` | the Claude Code project with the monitor's skill |
| `<LESSONS_FILE>` | the editable lessons file the skill reads and the weekly review edits |
| `<SITE_LESSONS_FILE>` | the site lessons: one site's lessons, in the monitor's state (paths.site_lessons), which lessons_for.py adds to each role's extract |
| `<CONTAINER_NOTE>` | in a container: how the orchestrator drives the skill (empty elsewhere) |
| `<ATLASSIAN_SITE>` | the JIRA site's name, which the atlassian CLI's --site takes (from integrations.atlassian_cli.jira_url) |
| `<ROCM_LLVM_BIN>` | ROCm's LLVM tools |
| `<JIRA_PROJECT>` | the JIRA project |
| `<JIRA_URL>` | the JIRA site |
| `<JIRA_COMPONENT>` | new tickets' component |
| `<JIRA_LABELS>` | new tickets' labels |
| `<JIRA_ISSUE_TYPE>` | new tickets' issue type (features.ticket_creation.issue_type) |
| `<JIRA_REQUIRED_FIELDS>` | the fields every new ticket must carry, as JSON: field id -> value (features.ticket_creation.required_fields) |
| `<TEAM_CHANNEL>` | the team chat while the develop break alerts are on (its webhook decides the channel) |
| `<TEAMS_WEBHOOK_FILE>` | the channel's webhook (a secret) |
| `<TEAM_WEB_HOST>` | the team dashboard's web host |
| `<TEAM_DASHBOARD_URL>` | the team dashboard's address |
| `<TEAM_DASHBOARD_DEST>` | where each tick pushes it |
| `<WEBHOST_KEY_FILE>` | the web host's ssh key (a secret) |
| `<ADMIN_PORT>` | the admin dashboard's port |
| `<ADMIN_URL>` | the admin dashboard |

<!-- END GENERATED: agentic-placeholders -->
