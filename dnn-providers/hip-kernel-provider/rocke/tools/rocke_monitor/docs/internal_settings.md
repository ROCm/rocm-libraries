<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Internal settings

The configuration keys that are in no settings file: the package's defaults, what the monitor records in its internal state, and the values derived from others. Each key's description and default come from the registry, `rkmon_lib/registry.py`, their one home: a package or state key's description closes with its default. This file is generated from it: change the registry, then run `python3 -B tools/make_package.py --refresh-generated`. The settings files' keys are described in their schemas, `schema/deploy_settings.schema.json` and `schema/build_settings.schema.json`.

## Package defaults

Values every deployment shares and no person sets, each in `config.defaults.json`.

| Key | What it is |
|---|---|
| `test_sparse` | Repository folders outside the watched folder's provider that every review's worktrees check out for the tests, such as a shared CTest category parser. Default: none. |
| `worker.context` | The context length to ask for, in tokens (Ollama num_ctx; an OpenAI-compatible server keeps its own). Default: 262144. |
| `worker.temperature` | Sampling temperature. Default: 0.3. |
| `worker.max_output_tokens` | The most tokens one answer may have. Default: 16000. |
| `worker.num_batch` | Ollama only: the prompt batch size. Default: 256. |
| `worker.keep_alive` | Ollama only: how long the model stays loaded after a request. Default: "120s". |
| `worker.max_steps` | The most tool-calling rounds in one draft. Default: 20. |
| `worker.hang_s` | Local Ollama only: seconds a runner may take a prompt without progress before it counts as hung. Default: 120. |
| `worker.max_hangs` | Local Ollama only: hung runners resent before the draft gives up. Default: 6. |
| `worker.max_resends` | Failed requests (no answer, dropped connection) resent before the draft gives up. Default: 3. |
| `draft_parallel_max` | The most drafts that run at once, the pool's, the running reviews' and the shadow drafts together: a ceiling, whatever room the machine has. Default: 6. |
| `drafter` | The settings of the first review pass's and the first code quality pass's tools (worker/drafter.py, worker/quality_lane.py) and of the tools they share: the run's token budget and wall-time cap (review_token_budget, review_wall_min) and each phase's own (prior\_\*, hunt\_\*); the prior tasks' steps, how many run at once and their effort; the mechanical turns' effort; the cards' size; the tool results kept; the hunt's rounds; the saved probes' time cap; the limiter's reserve and folder (limiter_dir, \<paths.root>/llm_limits while it is blank); the restage age; the evidence check's re-run caps; the code quality pass's run (quality\_\*); and the first passes' caps (first_pass\_\*: rkmon_lib/first_pass.py). An entry left out takes its default. Default: {"review_token_budget": 1000000, "review_wall_min": 45, "prior_wall_min": 15, "hunt_wall_min": 35, "prior_token_budget": 400000, "hunt_token_budget": 500000, "prior_task_steps": 3, "prior_parallel": 4, "prior_effort": "medium", "mechanical_effort": "low", "card_tokens": 3000, "keep_results": 4, "hunt_steps": 30, "prior_max_tokens": 8000, "hunt_max_tokens": 16000, "probe_timeout_s": 600, "reserve_pct": 20, "limiter_dir": "", "restage_min": 60, "check_rerun_max_s": 120, "check_total_s": 600, "quality_wall_min": 20, "quality_token_budget": 400000, "quality_steps": 24, "quality_max_tokens": 16000, "first_pass_wall_min": 5, "first_pass_token_budget": 200000, "first_pass_max_tokens": 6000}. |
| `mirror.enabled` | The one switch: every command that writes outward is refused before any request (gh, the Teams webhook and ticket filing refuse too); the review driver starts reviews only from the feed's lines, each at its head; rk post writes the would-be post with its notes, draft, the quality binding's code quality pass (for a review that got it) and cost.json to the out folder instead of posting. Default: false. |
| `mirror.feed` | The live monitor's state/mirror_feed.jsonl, mounted read-only. Blank means no feed: no mirrored review starts. |
| `mirror.out` | Where the mirrored reviews go. Blank means \<WORKING_DIR>/mirror. |
| `mirror.pairs` | The reviews to write; then no more feed lines are taken (MIRROR_DONE). Default: 15. |
| `mirror.max_age_min` | A feed line older than this many minutes is dropped, not started. Default: 360. |
| `mirror.checks` | The live monitor's workers folder, mounted read-only: a mirrored review's checks are the live review's for the same head, copied once whole, not built again. Blank means it builds its own. |
| `mirror.checks_wait_min` | How long a mirrored review waits for the live review's checks. Default: 240. |
| `mirror.arm_a` | Without a quality binding (roles.quality) every review is arm A, which runs Claude's code quality pass as the live monitor does. With one, the reviews of arm A: the first ones alternate A and B (the binding's code quality pass), A first, until arm A has this many, then B. Default: 0, which means every review is arm B. |
| `checks.min_free_memory_gb` | Free memory (GB) a native build needs; below it the build is skipped. Default: 16. |
| `offload.mode` | off (the reviewer model drafts; Claude verifies every claim and runs the code quality pass), shadow (the trial: the offload passes also run for each review and post nothing; rk shadow-report compares them with what we posted) or on (\<OFFLOAD_MODE>). Default: "off". |
| `offload.spot_check` | The share of the 🟡/🟢 claims the auditor re-checks, from 0 to 1 (every 🔴/🟠 claim is checked). Default: 0.25. |
| `offload.spot_check_min` | The fewest 🟡/🟢 claims the auditor re-checks (all of them when there are fewer). Default: 2. |
| `offload.deep_lines` | A first round over this many changed lines gets a deep review (the full Claude review). Default: 2000. |
| `offload.deep_files` | A first round over this many changed files gets a deep review. Default: 40. |
| `offload.deep_hot_spots` | The hot spots (by their names in hot_spots) a first round may not touch without a deep review. Default: \["ABI", "CMK", "CAT"]. |
| `offload.deep_paths` | Regular expressions of paths a first round may not touch without a deep review. Default: \["(^\|/)CMakeLists\\\\.txt\$", "\\\\.cmake\$"]. |
| `offload.budget_shares.draft` | The draft pass's share (the assembly counts as the draft's). Default: 0.5. |
| `offload.budget_shares.quality` | The code quality pass's share. Default: 0.25. |
| `offload.budget_shares.regrade` | The re-grade pass's share. Default: 0.25. |
| `offload.max_steps` | The most tool-calling rounds in one offload pass. Default: 40. |
| `offload.probe_timeout_s` | A probe's or mutant's time limit, in seconds (it runs in the sandbox). Default: 600. |
| `offload.check_rerun_max_s` | rk evidence-check re-runs a recorded run that took this many seconds or fewer; longer ones are left to the auditor. Default: 120. |
| `offload.check_total_s` | The most seconds rk evidence-check spends re-running commands for one review. Default: 900. |
| `offload.trial_prs` | How many PRs the shadow report's summary covers (the latest). Default: 10. |
| `gpu.skip_pattern` | Which skip reasons mark a GPU-only case (a regular expression): only those cases run remotely, and none while it is blank. Default: "(?i)\\\\b(gpus?\|devices?)\\\\b". |
| `gpu.base_env` | The environment variables each remote GPU job sets first, before gpu.env's, which may change them: the repository's own, such as the backend its tests take on the GPU nodes. Default: none. |
| `gpu.base_env.ROCKE_BACKEND` | The home repository's GPU jobs set ROCKE_BACKEND=python first (its repository defaults): rocKE runs its Python backend on the GPU nodes, so no native build is needed there (where a test allows it). ROCKE_BACKEND=cpp in gpu.env picks the native engine. |
| `gpu.max_cases` | Past this many cases on one target, their test files run instead. Default: 300. |
| `review_checklist` | Where the shared rocKE review checklist lives (\<REVIEW_CHECKLIST>): a path inside the repository or a public URL. The review book points the first-pass model and the verifier at it, so the book keeps only the monitor's own process. Blank means the review book alone. |
| `clashes` | The clash scan's limits: budget_s, the seconds it may take; max_fetch, how many pull requests' changed files it may fetch; merge_timeout_s, the seconds each git merge it runs may take. An entry left out takes its default. Default: {"budget_s": 240, "max_fetch": 4, "merge_timeout_s": 60}. |
| `max_posts_per_hour` | At most this many reviews posted per hour. Default: 6. |
| `huge_pr_lines` | A review of a PR this large keeps the merge analysis waiting while it runs. Default: 8000. |
| `hot_spots` | Files many PRs touch (golden data, the ABI header, CMake lists), by kind: RG and DG goldens, ABI, CAT (the arch catalog), CMK. A list names files, or folders ending in /; \<ROCKE_PATH> stands for rocke_path. Default: none. |
| `rebase_noise_paths` | Files whose change alone does not make a follow-up review substantive; while it is empty, the RG and DG hot spots. Default: none. |
| `libparity_check` | Run the library kernels' C++/Python parity test when a PR changes them. Default: true. |
| `models.tick` | The tick triage's model (rkmon_lib/triage.py), as an opt-in: "worker" (the local review loop's model) or a model_bindings name. It never follows the reviewer by itself. Blank means the toolkit's rule filter (rk tick-filter), with no model. |
| `models.verify` | Always "claude": the orchestrator never hands verification to another model. Default: "claude". |
| `models.quality` | Always "claude": the code quality review runs in the Claude session. Default: "claude". |
| `models.post` | Always "claude": whatever is posted goes out through the Claude session. Default: "claude". |
| `models.analytics` | The analytics model the planner and the analyst use, on the reviewer's endpoint: "worker" (the reviewer's own model), or {"model": a model the endpoint lists, "share": each role's share of the endpoint's rate limits, "timeout_min": how long an answer may take}. Blank means off. |
| `models.planner` | The planner's model, which proposes the merge plan's waves: "off", or a model of its own in the forms of models.analytics. Its proposal is the plan only when it keeps every rule of the toolkit's own plan; it never arms, merges or posts. Blank means the analytics model. |
| `models.analyst` | The analyst's model, which ranks and annotates the per-PR insights: "off", or a model of its own in the forms of models.analytics, with "every_h", the hours between readings. Blank means the analytics model. |
| `effort.audit` | The verifier's offload audit (\<AUDIT_EFFORT>). Blank means claude.effort. |
| `effort.deep` | A deep review's verifier and code quality pass (\<DEEP_EFFORT>); claude.effort while it is blank. Default: "medium". |
| `effort.tick` | The tick triage's requests when models.tick names a model: low, medium or high. Blank means claude.effort on a Claude binding, and none on any other model. |
| `roles` | Which binding (by its name in model_bindings) each role uses: shadow_drafter (a binding under assessment: it drafts every live review's PR at the same head, in its own folder, beside the live draft and never in its place), quality (the code quality pass: the review driver runs it on this binding beside the draft and the checks, worker/quality_lane.py, and REVIEW_READY names its quality.md; with none, the orchestrator's code quality reviewer runs it), analytics and triage. drafter names the reviewer's binding while the key reviewer is blank. Default: none. |
| `claude.effort` | Claude's base effort: low, medium, high, xhigh or max. A Claude role with no key of its own in effort takes it. Default: "high". |
| `claude.auto_compact_window` | The tokens of context at which the orchestrator's session compacts itself (Claude Code's autoCompactWindow setting; the session runs on a 1M-context model): a whole number within config.COMPACT_RANGE. In the container, rk orchestrate writes it into the Claude workspace's .claude/settings.json as the session starts, the settings the session loads; the review agents keep their own window (their agent files). Default: 300000. |
| `claude.prompt_cache_ttl` | How long the orchestrator's session keeps its prompt cache: 5m or 1h (Claude Code's promptCacheTtl setting). An hour lets the cache last through the minutes between ticks. In the container, rk orchestrate writes it into the session's .claude/settings.json beside autoCompactWindow. Default: "1h". |
| `claude.subagent_prompt_cache_ttl` | How long the session's subagents (the verifier, the code quality reviewer) keep their prompt cache: 5m or 1h (Claude Code's subagentPromptCacheTtl setting). A subagent's turns run back to back; the hour-long cache is the orchestrator session's. Written beside promptCacheTtl. Default: "5m". |
| `weekly_report.weekday` | The weekday (0 = Monday) after which last week's report is written. Default: 0. |
| `weekly_report.hour_utc` | The hour (UTC) on that weekday after which the report is written. Default: 14. |
| `weekly_report.friction_window_h` | Hours around an event the report looks at for friction. Default: 6. |
| `followup_gap_exempt` | The follow-ups that never wait for the gap: develop_fix (a PR that fixes a develop break), priority (a PR the operator floated to the top) and PR numbers. Default: \["develop_fix", "priority"]. |
| `settle_burst_min` | Minutes after the last push of a burst before its review starts, when that is later than settle_min. A burst is BURST_PUSHES or more pushes within BURST_WINDOW_MIN minutes, as the ticks saw them (rkmon_lib/store.py). Default: 45. |
| `settle_develop_fix_min` | The most minutes a PR that fixes a develop break, or one the operator floated to the top, waits to settle, bursts included: it is reviewed after settle_min or this, whichever is shorter. Default: 10. |
| `loop_heartbeat_min` | tools/tick_loop.py exits this many minutes after it started (LOOP_HEARTBEAT), so the session that started it hears from it. Default: 180. |
| `tick_timeout_min` | A tick of tools/tick_loop.py still running after this long is stopped, and the loop exits. Default: 20. |
| `loop_log_max_mb` | The size state/tick_loop.log is kept under. Default: 20. |
| `analysis_max_wait_min` | The longest a merge candidate's analysis waits behind reviews. Default: 90. |
| `revert_offer_after_min` | Minutes after a develop break opens before the tick asks for a revert offer. Default: 120. |
| `hold_stale_h` | Hours a held PR sits without a push before the operator is asked about it. Default: 24. |
| `ci_test_tiers` | The PR-CI lanes that run the watched folder's CTest entries, for the checks' CTest labels and tiers section and rk lane-of: each {jobs, tree, tier, exclude}. Default: none. |
| `ci_absent_modules` | The Python modules CI's test environment lacks (top-level names): rk lane-of reads a module-level importorskip of one as a skip in CI and a module-level import of one as a collection error, its --collect makes them unimportable, and the fix-line premise refuses a fix that guards a test on one. Default: none. |
| `ci_infra_signatures` | Known CI infrastructure failures (rkmon_lib/known_infra.py): a merge-only head whose only flag is such a failure is carried with a CI_INFRA line, and the checks' CI at head section tags the job. Each entry is a plain log substring or {pattern, regex, jobs, unless, ends_at, note}: pattern the text to find in the log (a regular expression when regex is true); jobs a regular expression over the job's name, any job when it is blank or left out; unless a regular expression whose presence in the log leaves the job unknown; ends_at a develop commit: the signature does not count for a head that contains it; note what it is. Default: none. |
| `drift_extract` | What a drift run extracts beside the watched folder, as repository paths ({provider}: the watched folder's parent; a path ending in /\* takes that folder's files only); while it is empty, the watched folder alone. Default: none. |
| `drift_batch` | PRs per develop-drift job (their own tests re-run on today's develop). Default: 6. |
| `develop_watched_workflows` | The develop CI workflows whose result counts as develop's CI (names in the repository's GitHub Actions). Default: none. |
| `develop_rocke_job_pattern` | A regular expression for the CI job names that test the watched folder. Blank means every job of the watched workflows. |
| `hold_label_patterns` | Regular expressions for labels that take a PR out of merge candidacy (do not merge, on hold, WIP). Default: \["do\[ \_-]\*not\[ \_-]\*merge", "don'?t\[ \_-]\*merge", "\\\\bdnm\\\\b", "on\[ \_-]\*hold", "\\\\bwip\\\\b", "not\[ \_-]\*ready"]. |
| `merge_plan.max_merges` | The most simulated merges (git merge-tree: each PR with develop, and the pairs on develop) a tick runs for the merge plan (rkmon_lib/mergeplan.py), the readiest PRs first; a pair not checked yet counts as a conflict and marks the plan partial. Default: 200. |
| `merge_plan.budget_s` | The seconds a tick spends on the merge plan's simulated merges. Default: 30. |
| `merge_plan.timeout_s` | One simulated merge's time limit, in seconds. Default: 120. |
| `conflicts.max_checks` | Merge checks per tick for the merge-conflict history (rkmon_lib/conflicts.py, state/conflicts.jsonl); what is left waits for the next tick. Default: 25. |
| `conflicts.max_fetches` | PR fetches per tick for the merge-conflict history. Default: 5. |
| `conflicts.budget_s` | Seconds per tick for the merge-conflict history. Default: 60. |
| `conflicts.timeout_s` | One merge check's time limit, in seconds. Default: 120. |
| `conflicts.refresh_h` | Hours between two rows for a PR head that keeps conflicting with develop. Default: 6. |
| `conflicts.scan_days` | Days of develop merges on tracked PRs' branches scanned for the conflicts their authors resolved. Default: 8. |
| `ci_times.every_min` | Minutes between two collections of the CI-time history (rkmon_lib/ci_times.py, state/ci_runs.jsonl). Default: 60. |
| `ci_times.max_calls` | GitHub API calls per collection of the CI-time history. Default: 30. |
| `ci_times.budget_s` | Seconds per collection of the CI-time history. Default: 90. |
| `ci_times.days` | Days of heads the CI-time history reads and keeps. Default: 8. |
| `ci_times.jobs_runs` | The longest workflows of a head whose jobs are read. Default: 2. |
| `ci_times.job_min_s` | The least seconds a workflow runs to have its jobs read. Default: 600. |
| `ci_times.weekly_calls` | The GitHub API calls the weekly report's view may make for the week's heads not recorded yet. Default: 150. |
| `overlaps.max_heads` | New PR heads the overlap scan reads per tick (rkmon_lib/overlaps.py; no GitHub API call). Default: 12. |
| `overlaps.max_fetches` | Of the new heads, those that may fetch blobs not local yet. Default: 3. |
| `overlaps.budget_s` | The seconds the overlap scan spends per tick. Default: 30. |
| `admin_dashboard.url` | The address a browser opens when the page is reachable beyond this machine, e.g. http://monitor.example.com:\<port>/: the service answers only requests that name it (or the loopback). Blank means the loopback address at the dashboard's port. |
| `scoreboard.lookback_days` | Days of records each full tick derives the scoreboard's events from. Default: 8. |
| `scoreboard.conflict_window_h` | Hours after a landing within which a PR turning CONFLICTING may count as a rebase that landing forced. Default: 3. |
| `scoreboard.ahead_h` | rk scoreboard-backfill: hours before a merge within which a tick-log AUTOMERGE_JUMP or PRIORITY_READY_EARLY alert counts as landing ahead of plan. Default: 12. |
| `scoreboard.git_budget_s` | Seconds of git a full tick may spend on scoreboard questions not answered before; the rest waits for the next tick. Default: 0.4. |
| `tracker_allow` | Tracker-like prefixes (the letters before a tracker ID's dash) the peer check of tracker IDs in added lines lets pass, each the name of a standard, an encoding, a register or a product (SEM, the semantic register IDs, makes most raw hits). An empty list takes the package's. Default: \["SEM", "IEEE", "GQA", "SWA", "MFMA", "SHA", "UTF", "ISO", "RFC", "CVE", "X86", "FP", "INT", "UINT", "BF", "TF", "GFX", "LLVM", "MI", "HIP", "CDNA", "RDNA", "CUDA", "SM", "PEP", "POSIX", "ANSI", "ECMA", "AVX", "SSE", "DDR", "HBM", "PCIE", "WG", "LDS", "GEMM", "MLA", "DSA", "GDN", "KDA", "TOP", "RC"]. |
| `incident_issue.time_limit_labels` | Develop issue labels that mean a CI time limit (a Test step or a per-test limit), not a code break: an open issue with one, or an incident with no culprit whose issue has one, makes develop's status CI-failures, not Broken. Default: none. |
| `incident_issue.title_pattern` | A regular expression, matched without regard to case, for the titles of the develop issues that concern the watched folder, beside the scope's labels: the tick reports such an issue when it opens or closes, and the fix sweep and the lists of develop issues take it as the folder's. Blank means the scope's labels alone. |
| `incident_issue.test_name_pattern` | A regular expression for the repository's own test names, beside test functions and test classes (test\_ and Test names), which every repository's issues name: a draft develop issue that shares such a name, or a PR number, with an open one is a possible duplicate of it (rk issue-file). Blank means test functions and test classes alone. |
| `incident_issue.umbrella_test_names` | Test names that run most of the watched folder's tests: two develop issues that share one of them only are no duplicates, as naming one identifies no breakage. Default: none. |
| `tracker_keys` | Ticket-key prefixes (KEY of KEY-NNN) that must never appear in code or inside a link: while it is empty, any Jira-style ID; jira.project is always included. Default: none. |
| `jira.priority_map` | A draft's priority -> the JIRA priority to file it with, e.g. {"P2": "Major"}; while it is empty, the draft's own word. Default: none. |
| `atlassian.proxy_url` | The OAuth proxy the CLI logs in through on a host install (ATLASSIAN_PROXY_URL); the container's CLI never takes one from here. Blank means the CLI's own default. |
| `followup_scan.defer_patterns` | Regexes (case-insensitive) for a comment that defers an item: follow up, later, a next PR, TODO, out of scope, will address, a ticket. A ticket of the JIRA project (jira.project, tracker_keys) defers too. An empty list takes the package's. Default: \["\\\\bfollow\[ -]?ups?\\\\b", "\\\\blater\\\\b", "\\\\b(separate\|next\|future\|another\|subsequent) (PR\|pull request)", "\\\\bTODO\\\\b", "\\\\bout of scope\\\\b", "\\\\bwill address\\\\b", "\\\\btickets?\\\\b"]. |
| `followup_scan.request_patterns` | Regexes (case-insensitive) for a reviewer's request: a question, a please, should or consider, a suggestion block, a line that opens with an imperative (not a label, such as Update:). An empty list takes the package's. Default: \["\\\\?", "\\\\bplease\\\\b", "\\\\bshould", "\\\\b(can\|could\|would) (you\|we)\\\\b", "\\\\bneeds? to\\\\b", "\\\\bmust\\\\b", "\\\\bconsider\\\\b", "\\\\bsuggest", "\\\\binstead\\\\b", "\\\\bmissing\\\\b", "\\\\bnit\\\\b", "\\\\bblock(ing\|er)\\\\b", "^\\\\s\*(add\|remove\|rename\|move\|drop\|avoid\|use\|fix\|update\|document\|split\|keep)\\\\b(?!:)"]. |
| `followup_scan.ticket_pattern` | The regex (case-sensitive) of a ticket key a comment names: capital letters and digits, a capital first, then a dash and a number of two digits or more, so UTF-8 is not one. A blank one takes the package's. Default: "\\\\b\[A-Z]\[A-Z0-9]{1,9}-\\\\d{2,}\\\\b". |
| `max_host_usage_pct` | The CPU and memory use, in percent, the monitor keeps this machine under (\`rkmon_lib/scaler.py\`): it lowers the parallel reviews and the parallel builds when either is over, raises them only with headroom, and starts a review or a build only when the memory, the RAM disk counted as used, stays under it. Default: 75. |
| `resources.work_memory_gib` | The container's memory for the work itself, in GiB, beside the RAM disk: its memory limit (host.resources.memory_gib) is at least the RAM disk's size (host.resources.ramdisk_gib) plus this. Default: 8. |
| `ramdisk.review_estimate_gib` | How much of the RAM disk a review takes, in GiB, before the monitor has measured one; the sizes it measures then replace it (rkmon_lib/ramdisk.py, SEED_GB). Default: 1.5. |
| `ramdisk.build_estimate_gib` | How much of the RAM disk a build takes, in GiB, before the monitor has measured one; the sizes it measures then replace it (rkmon_lib/ramdisk.py, SEED_GB). Default: 2. |
| `gh.probe_issue` | The issue or PR of \<ROCKE_REPO> the logins' check comments on with an empty body, which GitHub refuses either way (nothing is made): 422 when the login may comment, 403 when it may not. Its other probe, a branch with no name, asks the same of a push and needs no issue. Default: 1. |
| `retention.log_max_mb` | The size past which each full tick moves an append-only state log (the journal, the .jsonl logs and web.log: rkmon_lib/retention.py, LOGS) to \<name>.1; the readers read both. Default: 20. |
| `retention.job_days` | Days after which a job folder goes: a finished job's, and one with no exit code that nothing names and whose process is gone. Default: 7. |
| `retention.kept_reviews_h` | Hours the disk copy of a finished review's directory (\<WORKING_DIR>/kept_reviews, rkmon_lib/review_keep.py) stays once nothing in it changed; a review in flight is always kept. Default: 24. |
| `update.drain_timeout_min` | The most minutes a drain waits for the work in flight to finish, unless its --timeout-minutes sets the wait: rk drain's, which an apply and a rollback run, as the host's stop --drain does, and rk stop --drain's. Default: 60. |
| `update.poll_s` | Seconds between two looks at the work in flight while a drain waits for it: rk drain's, which an apply and a rollback run, as the host's stop --drain does, and rk stop --drain's. Default: 20. |

## Internal state

What the monitor found or recorded, in its internal state file, whose schema is `schema/internal_state.schema.json`. Beside each key, its key in that file.

| Key | In the state file | What it is |
|---|---|---|
| `codeowner_teams` | `repository.codeowner_teams` | The GitHub teams that approve each code-owner area of the repository, by area, as its CODEOWNERS file names them. Default: none. |
| `clone_since` | `repository.clone_since` | The oldest commit date (YYYY-MM-DD) that the monitor's clone of the repository and its pull request fetches keep. Default: "". |
| `base_branch` | `repository.base_branch` | The branch the monitor watches for changes. Default: "", which means the repository's default branch. |
| `worker.model_name` | `model_bindings.*.display_name` | The model's name as the server's model list gives it, when it differs from the model's id. Default: "". |
| `worker.chat_path` | `model_bindings.*.chat_path` | The server's chat route, after the binding's address, that answered the probe. Default: "". |
| `worker.models_url` | `model_bindings.*.models_url` | The address where the server lists its models. Default: "". |
| `worker.probe` | `model_bindings.*.probe` | What the last probe of the server found: when (at); the address, model and kind of server probed (url, model, backend); how it sent the key (key_sent: a digest of the headers that carry it, which never shows the key, or none with no key); its test chat (chat: ok, or why not, as the server answered it, which reads "not checked: " and why when it got no HTTP answer or a rate limit stopped it); the tool-call probe (tools: ok, why not, or blank when not reached); the context length the model list or the server reports (context); how long the test chat took (latency_s); and where its model list and its chat route answered (models_url, chat_path). Each start probes the local review loop's server again while this is of another address or model, or of another key or another way of sending it, or its test chat or its tool-call probe is not ok. Default: none. |
| `publish.html.host_keys` | `team_dashboard.web_host_keys` | The web host's own SSH keys, as known_hosts lines, recorded once you accept them. Default: none. |
| `remote_gpu_tests.login_host_keys` | `remote_gpu_tests.login_host_keys` | The GPU cluster login node's own SSH keys, as known_hosts lines, recorded once you accept them. Default: none. |
| `jira.accounts` | `jira.accounts` | Each author's JIRA account, by GitHub username, as rkmon rk jira-account add LOGIN ACCOUNT records it. Default: none. |
| `ci_infra_site_signatures` | `site.ci_infra_signatures` | The CI failures this site's own infrastructure causes, each in the form of the repository's CI infrastructure signatures, learned on this site. Default: none. |
| `reviews.automated_reviewer_names_used` | `reviews.automated_reviewer_names_used` | Every automated reviewer name this monitor's reviews have carried, so that its earlier reviews stay its own after the name changes. Default: none. |
| `reviews.unnamed_are_ours_until` | `reviews.unnamed_are_ours_until` | The time before which the reviews an older monitor posted, whose line 1 has no automated reviewer name, are this deployment's own, as an ISO 8601 time in UTC. A mirror of a live monitor (mirror.enabled) takes each of them for its own, whatever this time. The one-time move sets it to the move's time, and rkmon standby off sets it again, to its own time. Default: "", which means none of them is this deployment's own, unless it is a mirror. |
| `standby.on` | `standby.on` | true: the monitor works but posts and sends nothing, keeping each item as an unsent item, until you run rkmon standby off. Default: false. |
| `standby.reason` | `standby.reason` | Why the monitor is in standby. Default: "". |
| `standby.since` | `standby.since` | When standby began, as an ISO 8601 time in UTC. Default: "". |
| `provisioned.at` | `provisioned.at` | When the last run of rk provision finished, as an ISO 8601 time in UTC; a work volume an older package set up gets it when it is first read, and one rkmon adopt filled when the next start sets it up. Default: "", which means not set up yet. |
| `provisioned.package_version` | `provisioned.package_version` | The package of the last run of rk provision, or the one that first read a work volume an older package set up. Default: "". |
| `provisioned.older_install` | `provisioned.older_install` | true: the work volume was an older install's when this package first read it (its config.json was there before any record), so the one-time move splits that config.json, once; rkmon adopt sets it again when it copies an earlier monitor's config.json. A config.json written later on a volume this package set up is never moved. Default: false. |
| `provisioned.adopted_at` | `provisioned.adopted_at` | When rkmon adopt last filled the work volume with an earlier monitor's state, as an ISO 8601 time in UTC: until the next start has set the volume up, the config.json adopt copied is the one-time move's, not an older install's setup. Default: "", which means none was adopted. |

## Derived

Values computed at each load from the keys before them and the container's layout, one function each in `rkmon_lib/derived.py`. A key's default is what a configuration file has when it leaves the key to its rule; the paths that are in no configuration file have none.

| Key | Default | What it is |
|---|---|---|
| `ramdisk.path` | `"/ramdisk"` | The RAM disk's folder. Derived: the container's RAM disk folder (rkmon_lib/container_layout.py). |
| `paths.working_dir` | `""` | The one folder the monitor owns (\<WORKING_DIR>): every file the monitor writes lives in it. Derived: in the container, its work volume; elsewhere blank, and config.json names it. |
| `claude_workspace` | `""` | The Claude Code project folder whose .claude/ gets the monitor's skill, team members and agent types (\<CLAUDE_WORKSPACE>), as Claude Code sees it. Derived: in the container, \<WORKING_DIR>/claude-workspace; elsewhere blank, and config.json names it. |
| `paths.root` |  | The working folder under its older name, which the drafter's modules read. Derived: paths.working_dir. |
| `paths.toolkit` | `""` | The monitor's home: its rk command and state/, and on a host config.json (\<TOOLKIT_DIR>). The toolkit's code stays where it is, read-only; rk finds this home through --home, RKMON_HOME or the pointer file (\~/.config/rkmon/home). Derived: \<WORKING_DIR>/monitor. |
| `paths.repo` | `""` | The monitor's own clone of the repository, blobless and shallow; every PR worktree is made from it (\<ROCKE_SRC_DIR>). Derived: \<WORKING_DIR>/repo. |
| `paths.workers` | `""` | Each review's worker folder: the PR's data, the draft and the checks. Derived: \<WORKING_DIR>/workers. |
| `paths.tmp` | `""` | Temporary files of the checks and the worker's test runs (kept off a small /tmp). Derived: \<WORKING_DIR>/tmp, or tmp in the RAM disk while the RAM disk is on. |
| `paths.probes` | `""` | Saved probes of open findings, re-run on every follow-up. Derived: \<WORKING_DIR>/probes. |
| `paths.lessons` | `""` | The verifiers' archived notes (lessons/notes) and retired lessons. Derived: \<WORKING_DIR>/lessons. |
| `paths.cache` | `""` | Results cached per develop commit (golden and installed-test baselines). Derived: \<WORKING_DIR>/cache. |
| `paths.scratch` | `""` | Scratch for develop sanity runs, bisects, compile checks and post files. Derived: \<WORKING_DIR>/scratch. |
| `paths.worker_tools` | `""` | The worker tools (review launcher, fetch, worktrees, native build, rule scan). Derived: \<WORKING_DIR>/tools. |
| `paths.venv` | `""` | The Python environment the checks run in (pytest, cmake, ninja, pybind11). Derived: \<WORKING_DIR>/venv; in the container, the image's. |
| `paths.build` | `""` | Native builds, one per worktree. Derived: \<WORKING_DIR>/build. |
| `paths.agentic` | `""` | The agentic files filled in with this machine's settings (the agentic install writes them, rkmon_lib/agentic.py; never commit them). Derived: \<WORKING_DIR>/agentic. |
| `paths.secrets_dir` | `""` | Where this deployment's secrets live (\<SECRETS_DIR>): the admin dashboard's token, the Teams webhook, the web host's ssh key, the reviewer endpoint's key and headers. Mode 700, files 600; never inside a git work tree or a cloud-synced folder. Derived: \<WORKING_DIR>/.config; in the container, its secrets volume. |
| `paths.run` | `""` | The reviews' shared rate budget and the registry of the processes the monitor started, which rk stop and rk kill read (\<RUN_DIR>). Derived: \<WORKING_DIR>/run. |
| `paths.ramdisk` |  | The RAM disk's folder, which paths.tmp goes in. Derived: ramdisk.path while the RAM disk is on, else blank. |
| `paths.outputs_dir` | `""` | Where the files written for the operator go (dashboard, page, daily and weekly logs, copies of posted reviews). Derived: the folder of paths.dashboard when config.json names it, else paths.working_dir. |
| `paths.dashboard` | `""` | The Markdown dashboard rewritten every tick. Derived: \<OUTPUTS_DIR>/rocke_merge_queue.md. |
| `paths.html` | `""` | The HTML dashboard rewritten every tick. Derived: \<OUTPUTS_DIR>/rocke_monitor.html. |
| `paths.daily_dir` | `""` | The daily logs, one file per UTC day. Derived: \<OUTPUTS_DIR>/rocke_daily. |
| `paths.insights_dir` | `""` | Where the insights reports are read from. Derived: the outputs folder. |
| `paths.posted_dir` | `""` | Where a copy of each posted review goes (PR_FIRST_ROUND\_\<N>.md, PR_FOLLOWUP\_\<N>.md). Derived: the outputs folder. |
| `paths.spend_dir` | `""` | The review spend files, one per UTC day and per ISO week. Derived: \<OUTPUTS_DIR>/rocke_spend. |
| `paths.weekly_dir` |  | The weekly reports' folder. Derived: the folder config.json names as weekly_report.dir, else \<OUTPUTS_DIR>/rocke_weekly. |
| `paths.posts` | `""` | Where \`rk post N FILE --premises SPEC\` keeps the verified post it is about to post, pr\<N>\_\<the round>.md (one per round: a second copy stops the post). Derived: \<WORKING_DIR>/scratch/post. |
| `paths.site_lessons` |  | The site lessons (\<SITE_LESSONS_FILE>): one site's facts and lessons (its hosts, people, tickets and runner pools) in the skill's lessons.md format, which the skill's lessons_for.py adds to each role's extract and the weekly review edits; in the monitor's state, so an update keeps them. Derived: \<TOOLKIT_DIR>/state/site/lessons.md. |
| `paths.rocm_llvm_bin` | `""` | ROCm's LLVM tool folder with clang and llc (\<ROCM_LLVM_BIN>). The develop guard's checks and the checks' compile lanes need it; without it they do not run (the compile gate stays pending), and the rest of the monitor runs as set. Derived: in the container, the folder the image's ROCm record names (the image's build writes the record), over one config.json names, as the image's ROCm is the only one there; elsewhere blank, and config.json names it. |
| `paths.review_book` | `""` | The local review loop's rubric (\<REVIEW_BOOK>). Derived: the filled-in copy, \<WORKING_DIR>/agentic/review_book.md. |
| `paths.package` |  | The monitor's code: read-only, never written at run time. Derived: the folder it runs from. |
| `paths.gh` | `""` | The gh binary every GitHub call goes through. In the container it stays blank: every call goes through the image's own gh, behind the GitHub request cap's counter. Its logins: gh.read_config_dir and gh.write_config_dir. Derived: blank, which means gh on PATH. |
| `paths.verifier_team_file` | `""` | The rocKE review verifier's team member file, which \`rk review-split\` gives each part's verifier; the code quality reviewer's team file, whose checks each part quotes too, is read beside it. Derived: \<CLAUDE_WORKSPACE>/.claude/team_members/rocke_review_verifier.md, where the agentic install puts it. |
| `paths.claude_projects` | `""` | The Claude Code projects folder the monitor's sessions write their transcripts to, which review spend reads. Derived: blank, which means \$CLAUDE_CONFIG_DIR/projects, else \~/.claude/projects. |
| `paths.cmake` | `""` | A folder with cmake and ctest, put first on PATH for the checks' test runs (CI's test lanes have them). Derived: blank, which leaves them to PATH and the venv. |
| `paths.pyyaml` | `""` | PyYAML's yaml package folder (or the site-packages folder it is in), linked into the checks' test runs alone. Derived: blank, which leaves it to the venv. |
| `paths.torch_python` | `""` | A python with torch (never the checks venv, which CI's lanes match: no torch). The mutation sweep re-runs its survivors under it, so a check only torch tests reach reads "vacuous: torch-only path". Derived: blank, which means the re-run uses the checks venv alone. |
| `weekly_report.dir` | `""` | Where the weekly reports go. Derived: paths.weekly_dir. |
| `followup_scan.dir` | `""` | Where the scans' files go (followups\_\<N>.md). Derived: the follow-ups folder in the monitor's state folder, \<TOOLKIT_DIR>/state/followups. |
| `codeowner_org` | `""` | The GitHub organization of the code-owner teams. Derived: the owner part of repo. |
| `clone_url` | `""` | Where the monitor clones \<ROCKE_REPO> from: a mirror's address, or a local path (file://...). Derived: GitHub's address of repo, https://github.com/\<ROCKE_REPO>. |
| `profile_dir` | `""` | The repository profile: the first review pass's settings and rules for this repository (drafter.json, drafter_rules.md). Derived: the folder of the repository defaults for repo and rocke_path (the package's repository/\<org>\_\_\<repo>), blank where none apply, which gives the generic profile. |
| `models.draft` | `"worker"` | Who drafts a review's first pass: "worker" (the local review loop's model), a {backend, url, model} object for another endpoint, "" (nobody: the orchestrator does it), or "claude" (the orchestrator). Derived: "worker". |
| `admin_dashboard.port` | `""` | Its port (\<ADMIN_PORT>). In the container the service listens on it inside, and public_port is the host's. Derived: 18613. |
| `admin_dashboard.bind` | `""` | Where the monitor service listens: 127.0.0.1 (this machine only), or another address of this machine (0.0.0.0: all of them), which makes the page reachable from other machines (README.md, "The admin dashboard"); with 0.0.0.0 it also needs admin_dashboard.url. In the container the host publishes it on its own 127.0.0.1 alone. Derived: 0.0.0.0 in the container; elsewhere blank, which means 127.0.0.1. |
| `team_dashboard.port` | `""` | serve: its port. In the container the page listens on it inside, and public_port is the host's. Derived: 18615. |
| `team_dashboard.bind` | `""` | serve: where it listens: 127.0.0.1 (this machine only), or an address of this machine (0.0.0.0: all of them) for other teams; in the container, publish_host says where the host publishes it. Derived: 0.0.0.0 in the container; elsewhere blank, which means 127.0.0.1. |
| `publish.html.dest` | `""` | The team dashboard's page on the web host, relative to the login's home. Derived: index.html in the site folder (publish.html.site), blank while that is blank. |
| `gpu.known_hosts` | `""` | The known_hosts file ssh reads for the GPU cluster's login host: the remote GPU tests write the login host's own keys the state records to it (remote_gpu_tests.login_host_keys), and ssh trusts no other. Derived: gpu_known_hosts in paths.working_dir. |
| `publish.html.known_hosts` | `""` | The known_hosts file ssh reads for the web host: the team dashboard's pushes write the web host's own keys the state records to it (publish.html.host_keys), and ssh trusts no other. Derived: web_host_known_hosts in paths.working_dir. |
| `gpu.ssh_key` | `""` | The SSH private key for the GPU cluster's login host. A key others may read is used through a private copy beside it (mode 600), since ssh refuses it. Derived: the place the secrets registry sets for the secret gpu_ssh_key, \<SECRETS_DIR>/gpu_ssh_key, blank while no secrets folder is known. |
| `publish.html.key` | `""` | The ssh key for the web host (\<WEBHOST_KEY_FILE>; a dedicated key). Derived: the place the secrets registry sets for the secret web_host_ssh_key: in the container the writer's own, /secrets-write/web_host_ssh_key; elsewhere \<SECRETS_DIR>/web_host_ssh_key, blank while no secrets folder is known. |
| `admin_dashboard.access_token_file` | `""` | The access token every request carries, from this machine too (HTTP Basic: any user name, the token as the password): a secret, never in a page. Derived: the place the secrets registry sets for the secret admin_access_token, \<SECRETS_DIR>/admin_access_token, blank while no secrets folder is known. |
| `teams.webhook_file` | `""` | The file with the channel's incoming-webhook URL (\<TEAMS_WEBHOOK_FILE>): a secret, mode 600. Derived: the place the secrets registry sets for the secret teams_webhook: in the container the writer's own, /secrets-write/teams_webhook; elsewhere \<SECRETS_DIR>/teams_webhook, blank while no secrets folder is known. |
| `gh.read_config_dir` | `""` | The read login's gh folder (GH_CONFIG_DIR): a read-only token. While that folder has no login, the write login reads and writes, as one login (GH_LOGINS says so), until the host's login command sets the read login (github_read). Derived: the place the secrets registry sets for the login github_read, \<SECRETS_DIR>/gh-read, blank while no secrets folder is known. |
| `gh.write_config_dir` | `""` | The write login's gh folder: the login rk's GitHub write commands post under. Derived: the place the secrets registry sets for the login github_write: in the container /secrets-write/gh, the write user's only; elsewhere \<SECRETS_DIR>/gh, blank while no secrets folder is known. |
| `worker.api_key_file` | `""` | The file the reviewer model's API key is read from (a secret, mode 600). Derived: the place the secrets registry sets for the secret worker.api_key_secret names, blank while it names none or no secrets folder is known. |
| `worker.tag` | `""` | The tag the drafter assessment gives the reviewer model's drafts. Derived: the leading letters of its model's name. |
| `draft_stage` | `""` | The draft stage the review driver runs. Derived: none (no draft: the checks only) when nothing drafts, with the default reviewer; else tool-loop (the drafter binding through worker/drafter.py) when reviewer names a binding, else local-worker (the reviewer model's worker). |
| `verifier_name` | `""` | The model the footer of every post names as the one that verified it, and line 1 names when nothing drafts. Derived: the audit model in words (models.audit). |
| `review_first_line_no_draft` | `""` | Line 1 of every posted review when nothing drafts (the default reviewer, or draft_stage "none": the verifier reviews alone). Derived: \`Automated review by \<automated_reviewer_name>: Model - \<model>, effort: \<effort>\`, with verifier_name at the verifier's effort (features.review_automation.passes.verification_pass.effort, else claude.effort); a given line is not kept. |
| `review_first_line` | `""` | Line 1 of every posted review while a drafter drafts. Derived: \`Automated review by \<automated_reviewer_name>: Model - \<model>, effort: \<effort>\`, with the drafter's model (its binding's name, or the local review loop's worker.model) at the effort the draft runs at: the first review pass's (features.review_automation.passes.first_review_pass.effort; on default, a Claude model's is the verification pass's and another model's is none), the local review loop's worker.think, or the model's default where none is sent; a given line is not kept. |

## Sections

The parts of the configuration the keys above are in.

| Section | What it is |
|---|---|
| `admin_dashboard` | The admin dashboard: the operator's page, served by the monitor service (tools/rkweb.py): the only page with Approve and Decline (priority requests, tickets waiting for you, in-depth reviews). Every request carries the access token; where it listens and who can reach it: README.md, "The admin dashboard". |
| `atlassian` | The atlassian CLI (JIRA and Confluence from the command line). Its login lives in \<SECRETS_DIR>/atlassian (ATLASSIAN_CONFIG_DIR; in the container, the writer's /secrets-write/atlassian), never here. |
| `checks` | The local checks: native builds and GPU tests (review checks, gates, develop sanity, bisects), the parallel builds. |
| `ci_times` | CI-time history limits (rkmon_lib/ci_times.py, state/ci_runs.jsonl). |
| `claude` | The monitor's own Claude session (the container's orchestrator, rk orchestrate). Its key is the claude login, never here; where it goes and how it is sent are integrations.claude_code's settings. |
| `conflicts` | Merge-conflict history limits (rkmon_lib/conflicts.py, state/conflicts.jsonl). What is left waits for the next tick. |
| `effort` | Each Claude role's own effort, over the base claude.effort: low, medium, high, xhigh or max (a binding's request: low, medium or high); blank: claude.effort. A bound model that is not Claude takes the role's own effort, low, medium or high, and none while it is blank. rk roles (and rk doctor) shows every role's model and effort, and where the effort comes from. |
| `followup_scan` | rk followup-scan (each merged PR's open review items, for the follow-up tickets): what defers an item, what is a request, a ticket key's pattern, and its folder. |
| `gh` | The two GitHub logins (rkmon_lib/gh.py; README, The GitHub CLI): every read, the Claude session's, its agents' and the jobs' runs under the read login, which may not write; rk's guarded write commands alone use the write login. \`rk gh-logins\` checks both. |
| `gpu` | Optional: remote GPU tests on a Slurm cluster reached over SSH (Remote GPU tests: while review automation is enabled and gpu.enabled is true). The cluster's values are this site's, from the deploy settings file (features.review_automation.remote_gpu_tests); none is in the package. Jobs queue and never hold a review. |
| `incident_issue` | Develop breakages are GitHub issues in the repository. |
| `jira` | Optional: JIRA for follow-up tickets and other issues found along the way. Off while project is blank. Follow-up tickets are filed through the atlassian CLI; a found issue's ticket waits on the admin dashboard's Tickets card, whose Approve files it through the atlassian CLI. No token is stored here. |
| `merge_plan` | Merge-plan limits per tick (rkmon_lib/mergeplan.py). |
| `mirror` | A mirror of a live monitor, for an A/B comparison with it (rkmon_lib/mirror.py, docs/review_pipeline.md section 3.10; \`rk mirror-report\` compares the two sides). |
| `models` | Who does each role. |
| `offload` | The review offload: the reviewer model does each review's substance (its draft with evidence, the code quality pass, the re-grade of prior items, the assembled post), a deterministic check re-runs or re-reads its evidence (rk evidence-check), and Claude audits it, or reviews it in depth when the rules say so. |
| `offload.budget_shares` | Each offload pass's share of the reviewer endpoint's rate budget (worker.rpm, worker.tpm): while requests wait, a pass under its share goes before one that has used it; alone, a pass may use the whole budget. |
| `overlaps` | Overlap-scan limits per tick (rkmon_lib/overlaps.py; no GitHub API call). |
| `paths` | Folders and files. Only working_dir is required; a blank key means its default. |
| `provisioned` | Provisioned: the work volume set up from your confirmed settings by rk provision, which each start runs: the first sets it up, and a later one puts back what has gone missing. |
| `publish` | Optional: publish the team dashboard (a copy of the HTML page without the admin controls) on a team web host. |
| `publish.html` | The web host. Off while host is blank. |
| `ramdisk` | Optional: a tmpfs for the reviews' short-lived files (rkmon_lib/ramdisk.py): each PR's worktrees and builds, and paths.tmp (the review directories, the checks' and the review tools' TMPDIR). Never the clone, state/, the logs, the saved probes, the posts or the caches. A finished review's worktrees and builds leave it at once. |
| `remote_gpu_tests` | Remote GPU tests: what the monitor records for the GPU cluster. |
| `resources` | The container's share of this machine: its CPUs and its memory limit, from host.resources in the deploy settings file, and the memory the work needs beside the RAM disk. |
| `retention` | How the state folder stays bounded (rkmon_lib/retention.py): log rotation, job folder pruning and the kept review copies. The ledger, actions.jsonl, is never rotated. |
| `reviews` | Reviews: what the monitor records about the reviews it posts. |
| `scoreboard` | The admin dashboard's scoreboard (rk scoreboard): how far back each tick looks, how its rules attribute a conflict or an alert to a landing, and the git time a tick may spend. |
| `standby` | Standby: the whole monitor sending nothing, each feature in prototype or live mode alike. |
| `team_dashboard` | The team dashboard's place (features.dashboards.team): publish pushes the page to the team web host (publish.html); serve has the monitor service serve it here, read-only but for its request box, on a port of its own. |
| `teams` | Optional: a team chat channel, told about CI-red develop breaks with no fix planned while features.alerts.develop_breaks is true, and about the Needs-you decisions while features.alerts.needs_you is true, both only while features.alerts.enabled is true. Its webhook (a secret) decides the channel; channel is the text each message starts with, blank for none. |
| `update` | The drain's limits (rk drain, which an apply and a rollback run; rkmon_lib/drain.py), and the repository and branch of the updates section of your settings file. |
| `weekly_report` | The weekly statistics report and lessons review. |
| `worker` | The local review loop (reviewer worker): the first-pass model that drafts each review (first rounds and follow-ups) on a machine you choose (a local Ollama, or an OpenAI-compatible server such as vLLM, TGI, LiteLLM or an on-prem gateway). The orchestrator (Claude) then verifies every claim before anything is posted. The tick triage, the planner and the analyst may use it too (models.tick, models.analytics). |
