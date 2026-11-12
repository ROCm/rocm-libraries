# Agent harness for hipBLASLt and TensileLite

Guidance for an AI agent working on hipBLASLt or TensileLite. The rules below apply to every task. The A/B
flow applies only to changes that produce asm. Everything else is read only when needed ("Read when").

## Scope and precedence

- **Applies to:** `projects/hipblaslt/**`, including `tensilelite/`.
- **Component docs win.** Build, PR, license, and test rules are in `../AGENTS.md`, `../TESTING.md`,
  `../tensilelite/AGENTS.md`, `../tensilelite/TESTING.md`, and the skills in `../skills/`. This file only
  says how to verify a change. On any conflict, the component doc wins. Fix this file.
- **Not the kernel corpus harness.** `TESTING.md` describes a test harness for shipped kernels. This
  is a different thing: the workflow an AI agent follows around a change.

## Read when

| Read | When |
| --- | --- |
| [`ab.md`](ab.md) | you verify a change that produces asm (kernel codegen, scheduling): the A/B flow |
| [`compare.md`](../../../shared/stinkytofu/harness/compare.md) | you run tier 1: how to compare asm, which configs to run |
| [`state.md`](../../../shared/stinkytofu/harness/state.md) | `progress.md` exists, or you write results |
| [`clone.md`](../../../shared/stinkytofu/harness/clone.md) | `git fetch` fails or prints an error, or the clone is sparse or partial |
| [`PROFILING.md`](PROFILING.md) | before a perf A/B or a profiling run |
| [`DATABASE.md`](../../../shared/stinkytofu/harness/DATABASE.md) | `$HARNESS_DB` is set, or you are asked to remember something |
| [`MAINTAINING.md`](../../../shared/stinkytofu/harness/MAINTAINING.md) | you are asked to change this harness |

## Rules (every task)

1. **Check for new commits first.** At task start, run `git fetch origin develop` and list the new
   commits that touch the files in scope (`git log HEAD..FETCH_HEAD -- <path>`). `git fetch` only
   updates the remote refs and does not change your files, so run it even when you are asked not to
   change anything. If there are any, tell the developer before starting. A fix may already have landed.
   Rebasing during development is the developer's choice. Before merge, the base skill `rocm-pr-quality`
   (pre-merge gate) does the matching overlap check.
   - If `progress.md` exists, read it before starting, to see where earlier work stopped (`state.md`).
2. **Outward actions need the developer.** Push, open, edit, or comment on a PR, or post to a tracker,
   only when the developer says so. Outside contributors use a GitHub issue. AMD staff use JIRA or a
   GitHub issue.
3. **Respect other users of shared machines.** If a tool refuses or reports that a resource is held,
   read its message and stop. Do not retry in a loop or look for a way around it. To stop a process,
   send SIGTERM, wait, and only then SIGKILL.
4. **Confidential details stay out of this repo.** No host names, ssh aliases, internal URLs,
   hardware-specific bug notes, or machine-specific thresholds in tracked files, logs, results, or
   notes. Before you push, open a PR, or post anything public, check what you are adding for IP
   addresses, internal URLs, email addresses, and internal names. If the database is set, use its leak
   check (`config/` and its check script). A flag means ask the developer. A public link or a version
   number that looks like an IP is a likely false alarm, so ask, and do not decide alone.
5. **The database is private and is loaded by default.** At session start, if `$HARNESS_DB` is set,
   read its `index.md`, then load only the entries that matter for the task. Do not copy entries into
   tracked files, PRs, issues, or comments. If it is not set, say so once. If it is `none`, or the
   developer said not to use it, skip it without comment. If the developer says to remember something,
   or not to, follow `DATABASE.md`. At the end of a task, if you learned something that may be worth
   keeping, tell the developer and ask whether to note it down. Write an entry only if they agree.
6. **The A/B flow is a default.** The developer can approve an override of any step of it (the order, the
   inputs, the thresholds, which steps run). When they do, ask which they mean: this session only, every
   session (saved locally, where the developer says), or an improvement to the shared harness (then you
   may edit the harness, through a PR). An override never changes the shared text by itself. It never
   changes the safety rules: outward actions, shared machines, confidentiality, and memory.
7. **Mind the machine's memory.** Builds, Tensile generation, and test runs (Python especially) can
   use a lot of memory, and running out can kill the whole session. Cap the parallelism to fit (for
   example `--jobs 4`, or `CMAKE_BUILD_PARALLEL_LEVEL`), run one heavy job at a time, and check the
   free memory (`free -g`) before a long run. Never build the baseline and the candidate at the same
   time.
