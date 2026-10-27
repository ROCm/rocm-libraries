---
name: hipdnn-trigger-ci
description: Dispatch TheRock Multi-Arch CI or the hipDNN superbuild CI on a rocm-libraries branch with chosen GPU families, projects and test labels, then check status or watch the run. Always dry-runs first; a real dispatch needs explicit user approval.
argument-hint: "[--branch <branch>|--pr <pr-number>] [dispatch -w <workflow>|status|watch] [-f <input>=<value> ...] [--reuse-build <run-id>] [--dry-run|--yes [--cancel-active] [--allow-head-mismatch]] [--run-id <run-id>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Trigger CI

Use this skill when the user asks to trigger, re-run with different settings, check or watch GitHub Actions CI for a rocm-libraries branch or PR. It wraps `gh workflow run`, `gh run list`, `gh pr checks` and `gh run watch` for `ROCm/rocm-libraries`.

A dispatch starts real CI on shared runners. **Always run the `--dry-run` form first, show the printed `gh workflow run` command to the user, and dispatch for real only after they explicitly approve that exact command.** `dispatch` requires either `--dry-run` or `--yes`; `--yes` is the real dispatch, so pass it only after approval. `status` and `watch` are read-only.

## Inputs

Infer options from the user request:

- **Workflow** (`dispatch -w`):
  - `multi-arch` → `.github/workflows/therock-multi-arch-ci.yml`. Builds ROCm with TheRock and runs component tests on the chosen GPU families. This is the one to use for hipDNN/provider test labels.
  - `hipdnn-superbuild` → `.github/workflows/hipdnn-superbuild-ci.yml`.
- **Branch**: `--branch <branch>` (a global option, placed before the subcommand). If omitted, `--pr <pr-number>` resolves the PR head branch; otherwise the current git branch is used, under its upstream name when it tracks one. If that upstream is on a remote other than `ROCm/rocm-libraries` (a fork), or the `--pr` is from a fork, the script stops and asks for `--branch`; `--pr <pr-number> status` still works on fork PRs. The branch must already be pushed to `ROCm/rocm-libraries` for a dispatch to find it. The run tests the branch head on GitHub, not the checkout: when the branch comes from the checkout and that head is not the local `HEAD` (unpushed commits, or a checkout behind the remote), `dispatch` warns, in the dry-run too, and `--yes` refuses unless `--allow-head-mismatch` is passed. Prefer pushing or pulling; add the flag only after the user approves testing the GitHub head. `--reuse-build` is checked against the GitHub head as well.

  **A dispatch can cancel other runs.** Both workflows use the concurrency group `<workflow>-<PR number or commit SHA>` with `cancel-in-progress: true`. A dispatch has no PR number, so it shares the group of every queued or in-progress run of the same workflow at the branch's head commit that was not started by a pull request: an earlier dispatch on the same commit (for example the same branch with other families or labels), and the push postsubmit or scheduled run of that commit. Pull-request runs are keyed by PR number and are not affected. `dispatch` lists those runs as a warning, in the dry-run too; `--yes` then refuses unless `--cancel-active` is also passed. Show the list to the user and add `--cancel-active` only after they approve cancelling those runs. `dispatch` also refuses `develop` and `release/*`, whose postsubmit runs share the group, unless `--allow-shared-branch` is passed; pass it only when the user explicitly asks to dispatch on such a branch.
- **Inputs** (`-f <input>=<value>`, repeat per input): the `workflow_dispatch.inputs` of the workflow file on the target branch. Read them there; do not rely on remembered names. `dispatch` reads the same file at the same branch and rejects an unknown name (listing the current inputs), an input given twice, and a missing input that has no default. Omitted inputs keep the workflow default.
- **Values**: the workflow passes GPU families and test labels to TheRock, so the valid values are defined there, at the TheRock ref the target branch uses. Today that ref is `therock-ref` in `.github/actions/ci-env/action.yml`; if that moves, follow the workflow's checkout of TheRock. Read the files at that ref (`gh api -H "Accept: application/vnd.github.raw" "repos/ROCm/TheRock/contents/<path>?ref=<therock-ref>"`):
  - GPU families: `build_tools/github_actions/amdgpu_family_matrix.py`.
  - Test labels and components: `build_tools/github_actions/fetch_test_configurations.py` (`test_matrix`).
  - Test tier: `build_tools/github_actions/configure_multi_arch_ci.py` (`_determine_test_type`).

  Never invent a family or label; an unknown label matches no component, so only sanity runs.
- **Reuse a build**: `--reuse-build <run-id>` skips every build stage and copies the artifacts from that earlier run (it sets `baseline_run_id=<run-id>` and `prebuilt_stages=all`), so only the tests run. Use it to re-run tests with other labels or another tier without rebuilding. Only a workflow that declares both inputs accepts it (today `multi-arch`); do not pass either with `-f`. The script refuses a run of another workflow or repo, a run of a different commit than the branch head, and a run that has not completed. It does not check the baseline's jobs: every family requested now must have built successfully in that run, or its test jobs fail on missing artifacts. A cancelled run whose build stages all passed is a valid baseline. The workflow's own reuse policy never copies `compiler-runtime`; `all` does, which is untested here, so confirm the first such run gets past the copy step.

## Workflow

1. Locate this skill's helper directory. Skills are host-level, not tied to a repo checkout — **default to the script bundled with the skill you were invoked from** (`<skill-directory>/scripts/trigger_ci.py`), even when working inside a repo or worktree. Use the `projects/hipdnn/tools/ai/skills/hipdnn-trigger-ci/scripts` copy in a checkout only when actively developing this skill.

2. Run from anywhere inside a rocm-libraries checkout (branch detection uses `git`). `gh` must be installed and authenticated to github.com (`gh auth status --hostname github.com`); the script checks this before every subcommand, `--dry-run` included. Every `gh` call names github.com, so `GH_HOST` or a GitHub Enterprise login does not redirect it.

3. Dry-run the dispatch and show the user the printed command:
   ```bash
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch> | --pr <pr-number>] dispatch -w <workflow> [-f <input>=<value> ...] [--reuse-build <run-id>] --dry-run
   ```
   Here `<workflow>` is `multi-arch` or `hipdnn-superbuild`; inputs and values are described under Inputs.

4. After explicit approval, run the same command with `--yes` in place of `--dry-run`. If the dry-run listed active runs that the dispatch cancels, ask for approval to cancel them too, and add `--cancel-active`. If it warned that the local `HEAD` differs from the branch head on GitHub, get approval for that too and add `--allow-head-mismatch`. The script dispatches, waits up to about 15 seconds for the new run to appear and prints its run ID with `gh run watch` / `gh run view --log` commands.

5. Check status or watch:
   ```bash
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch>] status
   python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> status
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch>] watch [--run-id <run-id>]
   ```
   `status` with `--pr` shows `gh pr checks`; otherwise it lists the 10 most recent runs on the branch. `--branch` takes precedence over `--pr` in every subcommand. `status` exits non-zero when the query itself fails, including a PR with no checks yet; failing or pending checks are reported, not treated as errors. `watch` without `--run-id` follows the in-progress or queued run with the highest run ID on the branch and exits with the run's status, or exits non-zero when there is no such run; with `--run-id` on a finished run it prints the conclusion and exits with it. After a real dispatch, the reported run ID is the newest `workflow_dispatch` run of that workflow on the branch, so a run someone else dispatched on the same branch at the same moment can still be picked up.

## Test reference

Test labels are the value of the `pr_labels` workflow input, which stands in for the labels a pull request would carry; they are not GitHub PR labels, and the string is passed through as is. Never create or add `test:hipdnn` or any other hipDNN test label to a PR.

For which components, tiers and ctest labels a run selects, read the TheRock files listed under Inputs, then the test category YAMLs in this repo:

- `projects/hipdnn/test_categories.yaml`
- `dnn-providers/integration-tests/test_categories*.yaml`
- `dnn-providers/miopen-provider/test_categories*.yaml`, `dnn-providers/hipblaslt-provider/test_categories*.yaml`
- `dnn-providers/hip-kernel-provider/*_test_categories*.yaml`

## Examples

Always dry-run first and show the printed command. The values below are examples; check them against the sources under Inputs.

Multi-arch, Linux only, MIOpen provider and hipDNN tests:

```bash
python3 <skill-directory>/scripts/trigger_ci.py dispatch -w multi-arch -f linux_amdgpu_families=gfx94X,gfx950 -f windows_amdgpu_families=none -f pr_labels=test:hipdnn,test:miopenprovider --dry-run
```

Re-run the tests of a finished multi-arch run without rebuilding (the run must be of the branch's current head commit):

```bash
python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> dispatch -w multi-arch -f linux_amdgpu_families=gfx94X -f windows_amdgpu_families=none -f pr_labels=test:miopenprovider,test_filter:quick --reuse-build <run-id> --dry-run
```

The hipDNN superbuild CI (no inputs):

```bash
python3 <skill-directory>/scripts/trigger_ci.py dispatch -w hipdnn-superbuild --dry-run
```

Check or follow the run on a PR's branch:

```bash
python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> status
python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> watch
```

## Report

Summarize:

- The exact `gh workflow run` command (dry-run output) and whether it was dispatched
- Workflow, branch and the `-f` inputs (GPU families, test labels)
- Run ID and URL when dispatched, or the status/watch result

## Notes

- `scripts/trigger_ci.py` is bundled in this skill so linked and copied installs work independently.
- The script never adds or removes GitHub PR labels. Test labels here are workflow_dispatch input values.
