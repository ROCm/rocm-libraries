<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Worker tools

The tools the monitor's jobs run outside the toolkit, and the modules the toolkit loads from here too. Each start of
the container installs them into `paths.worker_tools` (`<WORKING_DIR>/tools`) from the code it runs
(`rk worker-tools --install`). The jobs call them with the RKMON_* environment, which `rk env` prints: every path, the
repository and rocKE's folder in it, the operator's login, the reviewer model's settings, the tracker keys, the read
login and the review offload's settings. Run by hand, `rkenv.py` reads the same values from the toolkit's `rk env`.
The tools are Python: run each as `python3 -B <tool>.py ARGS`.

| Tool | Called by | What it does |
|---|---|---|
| `review_agent_run.py N` | the review job's draft by the local review loop (`rkmon_lib/draft_stage.py`) | Prepares `workers/prN`: fetches the PR's data (`fetch_pr.py`), copies the review book, makes the worktrees (`mkwt.py`), then runs the local model (`review_agent.py`). Writes `review_agent.md`, the draft. With the review offload (`RKMON_OFFLOAD_MODE`): on, the offload passes instead; shadow, after the draft, into `offload/` |
| `review_agent.py WS N` | `review_agent_run.py` | The local review loop: the local model gets a fixed menu of tools (read a file, grep, run the repo's own tests, `rule_scan`, merge conflicts), never a shell; it answers in the review book's format |
| `offload_agent.py WS N PASS...` | `review_agent_run.py`, the finish job (the review offload) | The offload passes: draft, quality, regrade (from `review_agent_run.py`), assemble (the finish job, after the checks), each a constrained tool loop with evidence for every claim (`evidence.jsonl`, `runs.jsonl`); writes `draft.md`, `quality.md`, `regrade.md`, `post_draft.md` |
| `reviewer_tools.py` | `review_agent.py`, `offload_agent.py`, `rk evidence-check` | The vetted tools on a PR's trees (`Tree`): each builds its argv from validated parameters, with `shell=False`; the same builders rebuild a recorded run for the evidence check; numbered line reads and the quote check; what a run's output must keep on a re-run |
| `review_sandbox.py` | `reviewer_tools.py` | The sandbox of the model's own probes and mutants: bubblewrap, else an unprivileged user namespace; no network, nothing writable but the review folder, a time limit |
| `reviewer_client.py` | `review_agent.py`, `offload_agent.py`, an optional pass on the binding `worker` (`rkmon_lib/first_pass.py`), the start's checks (`rk provision`), `rk doctor` | The local review loop's model client (Ollama or an OpenAI-compatible endpoint), the shared rate budget with its per-role shares, and the token records (`Budget.count`: each request in `<RUN_DIR>/tokens.jsonl`; `TokenLog`: a review's totals per pass, `tokens_total.json`) |
| `ollama_stream.py` | `reviewer_client.py`, `review_agent.py` | Streamed `/api/chat` with a fast check for a wedged runner (it reads the Ollama service journal) |
| `drafter.py DIR N` | the review job's tool-loop and shadow drafts (`tools/draft_run.py`) | The tool loop, the first review pass on a model binding: the binding's model writes the draft in the post format the verification pass reads (`review_agent.md`), with evidence for every claim (`evidence.jsonl`, `runs.jsonl`) |
| `prior_settle.py` | `drafter.py`, `rk evidence-check` | A draft's prior items (the approvals, the rows of our last review, the other reviewers' items since it), and the rules a script settles some of them by before any model |
| `quality_lane.py DIR N` | the review job's code quality pass on a binding (`tools/quality_run.py`) | The first code quality pass on a model binding: the binding's model judges the PR's code by the code quality reviewer's instructions, with the tool loop's tools, and writes `quality.md` for the verification pass |
| `review_tools.py` | `drafter.py`, `prior_settle.py`, `quality_lane.py`, `rk evidence-check`, `tools/run_probes.py` | The tool loop's tools: a fixed menu of vetted, read-only tools on one PR's trees (git, grep, file reads, `rule_scan.py`), each building its argv from validated parameters, with `shell=False`; the same builders rebuild a recorded run for the evidence check |
| `llm_client.py` | `drafter.py`, `quality_lane.py`, the toolkit's calls to a model binding | The model bindings' client: one `chat` call over a binding's provider (an OpenAI-compatible route, or the Anthropic Messages API), and the shared limiter every caller of a binding uses; it never prints, logs or writes a key |
| `custom_headers.py` | `llm_client.py`, `rkmon_lib/orchestrator.py`, `docker/rk_root.py` | The one reader of `Name: value` header lines: Claude Code's ANTHROPIC_CUSTOM_HEADERS, and a model binding's headers |
| `fetch_pr.py N OUT` | `review_agent_run.py`, `tools/draft_run.py` | The PR's metadata, reviews, threads, comments, commits, files, diff and check runs, as Markdown, with the operator's own posts left out (the first review pass reviews independently) |
| `mkwt.py N` | the review, gate and recheck jobs | Fetches the PR head and develop into the clone; makes sparse worktrees `wt<N>` (head) and `wt<N>m` (head merged with develop, or with the PR's base branch that RKMON_BASE names), refreshes `wtdev`; lists the conflicting files |
| `rule_scan.py DIFF` | the checks job, the first review pass's tools | Repository rules on added lines: tracker IDs (RKMON_TRACKER_KEYS), PR numbers in code, absolute performance numbers, kernel-selecting environment reads, text I/O without an encoding |
| `build_native.py WT` | the checks, gate and sanity jobs | A from-scratch native build of a worktree (core library, tests, the pybind engine) in `$RKMON_BUILD/<WT>` (in the container the job's slot's `build`; on a host the build folder, the build itself on the RAM disk when there is one), on every CPU the job may use |
| `run_suite.py WT plat\|lib` | the verification pass, by hand | The platform or library pytest suite of a worktree; the tests run in a PR runner's slot of their own where there are PR runners (`pr_user.py`) |
| `faildiff2.py A B` | the verification pass, by hand | The failure IDs only one of two suite runs has |
| `golden_all.py ROCKE` | the verification pass, by hand, through `rk pr-run --` | Every representative-IR case against the tree's golden, per LLVM flavor |
| `sweep2.py`, `run_emit_wrap.py` | the verification pass, by hand, through `rk pr-run --` | A differential IR sweep of chosen families against a golden |
| `pr_user.py` | `mkwt.py`, `reviewer_tools.py`, `review_tools.py`, `run_suite.py`, `rkmon_lib/prcode.py` | How the tools run PR code: in the container a command that runs PR code runs in a PR runner's slot of its own through the package's `rk pr-run` (a runner reads no secret), its environment on stdin; on a host through `pr_env.py`, as the session user. Either way it gets only what `pr_env.py` lists of the environment. An installed copy reads the runners' group, a slot's home and the package's `rk` from `pr_user.json`, which its install writes beside it, and runs no file it finds |
| `pr_env.py` | `pr_user.py`, the toolkit's runs of PR code (`rkmon_lib/prcode.py`) | The environment PR code runs with: an allow-list of variables (the locale, the search paths, rocKE's switches, what the job's tools read), none shaped like a secret's |
| `pr_results.py` | `docker/rk_prcode.py`, `rkmon_lib/prcode.py`, `rkmon_lib/slots.py` | The results a job left in its PR runner's slot, read as data and copied into the monitor's folders: no link followed, only the slot owner's regular files, within limits of size and count |
| `perms.py` | the job scripts, the worker tools, `rk`, the start's root step (`docker/rk_layout.py`) | The modes of the container's folders: umask 027; what the monitor and the writer share, files 0660 and folders 2770; the trees PR code reads, the PR runners' group's, read only |
| `kill_runner.py` | the operator, as root | Gentle teardown of a wedged model runner (never `systemctl restart ollama`) |
| `rkenv.py` | the tools that read the monitor's settings | The RKMON_* settings, from the job or from `rk env` |
| `steps.py` | the scripts above, `tools/draft_run.py`, `tools/quality_run.py` | Runs a script's steps as a shell script did: each program's output straight to the script's own, in order, and a failing step's exit code |
