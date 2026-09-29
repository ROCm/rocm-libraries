---
name: hipdnn-trigger-ci
description: Dispatch TheRock CI, TheRock Multi-Arch CI or the hipDNN superbuild CI on a rocm-libraries branch with chosen GPU families, projects and test labels, then check status or watch the run. Always dry-runs first; a real dispatch needs explicit user approval.
argument-hint: "[therock-ci|multi-arch|hipdnn-superbuild|status|watch] [--gfx <families>] [--windows-gfx <families>] [--projects <paths>] [--test-labels <labels>] [--branch <branch>] [--pr <pr-number>] [--run-id <run-id>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Trigger CI

Use this skill when the user asks to trigger, re-run with different settings, check or watch GitHub Actions CI for a rocm-libraries branch or PR. It wraps `gh workflow run`, `gh run list`, `gh pr checks` and `gh run watch` for `ROCm/rocm-libraries`.

A dispatch starts real CI on shared runners. **Always run the `--dry-run` form first, show the printed `gh workflow run` command to the user, and dispatch for real only after they explicitly approve that exact command.** `status` and `watch` are read-only.

## Inputs

Infer options from the user request:

- **Workflow** (`dispatch -w`):
  - `multi-arch` → `.github/workflows/therock-multi-arch-ci.yml`. Builds ROCm with TheRock and runs component tests on the chosen GPU families. This is the one to use for hipDNN/provider test labels.
  - `therock-ci` → `.github/workflows/therock-ci.yml`. Single-arch TheRock CI for a set of rocm-libraries subtrees.
  - `hipdnn-superbuild` → `.github/workflows/hipdnn-superbuild-ci.yml`. Takes no inputs.
- **Branch**: `--branch <branch>` (a global option, placed before the subcommand). If omitted, `--pr <pr-number>` resolves the PR head branch; otherwise the current git branch is used. The branch must already be pushed to `ROCm/rocm-libraries` for a dispatch to find it.
- **GPU families**: `--gfx` (Linux) and `--windows-gfx`, comma-separated. When omitted the workflow defaults apply:
  - `multi-arch`: Linux `gfx94X,gfx950,gfx125X`, Windows `gfx110X` (`.github/workflows/therock-multi-arch-ci.yml`, `setup` job inputs). Pass `none` to skip a platform, for example `--windows-gfx none` for a Linux-only run.
  - `therock-ci`: Linux `gfx94X, gfx950, gfx125X`, Windows `gfx1151` (`.github/workflows/therock-ci.yml`, "Fetch Linux/Windows targets for build and test" steps).
- **Projects** (`therock-ci` only): `--projects`, space-separated subtree paths that are keys of `subtree_to_project_map` in `.github/scripts/therock_matrix.py` (for example `dnn-providers/integration-tests`, `projects/hipdnn`), or `all`.
- **Test labels** (`multi-arch` only): `--test-labels` (Linux) and `--windows-test-labels`, comma-separated. See the test reference below.

The accepted GPU family names come from TheRock's `build_tools/github_actions/amdgpu_family_matrix.py` at the TheRock ref pinned in `.github/actions/ci-env/action.yml` (`therock-ref`). `trigger_ci.py --help` prints the current list. Names are case-insensitive. A family that has no entry for the target platform is dropped for that platform (for example `gfx94X` and `gfx950` are Linux-only). `multi-arch` rejects an unknown name with an error listing the known families; `therock-ci` skips unknown names silently, so check the spelling. `multi-arch` also accepts `gfx1250-strict` (dispatch-only) and `all`.

Multi-arch has further dispatch inputs (`prebuilt_stages`, `baseline_run_id`, `baseline_repository`, `build_python_packages`, `build_pytorch`, `build_jax`, `build_native_linux`) that this script does not expose. Read `.github/workflows/therock-multi-arch-ci.yml` if the user needs one; pass it through `gh workflow run` directly only after showing the user the command.

## Workflow

1. Locate this skill's helper directory. Skills are host-level, not tied to a repo checkout — **default to the script bundled with the skill you were invoked from** (`<skill-directory>/scripts/trigger_ci.py`), even when working inside a repo or worktree. Use the `projects/hipdnn/tools/ai/skills/hipdnn-trigger-ci/scripts` copy in a checkout only when actively developing this skill.

2. Run from anywhere inside a rocm-libraries checkout (branch detection uses `git`). `gh` must be installed and authenticated (`gh auth status`); the script checks this before every subcommand, `--dry-run` included.

3. Dry-run the dispatch and show the user the printed command:
   ```bash
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch> | --pr <pr-number>] dispatch -w <workflow> [--gfx <families>] [--windows-gfx <families>] [--projects "<paths>"] [--test-labels <labels>] [--windows-test-labels <labels>] --dry-run
   ```
   Here `<workflow>` is `multi-arch`, `therock-ci` or `hipdnn-superbuild`; `<families>`, `<paths>` and `<labels>` are the values described under Inputs.

4. After explicit approval, run the same command without `--dry-run`. The script dispatches, waits up to about 15 seconds for the new run to appear and prints its run ID with `gh run watch` / `gh run view --log` commands.

5. Check status or watch:
   ```bash
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch>] status
   python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> status
   python3 <skill-directory>/scripts/trigger_ci.py [--branch <branch>] watch [--run-id <run-id>]
   ```
   `status` with `--pr` shows `gh pr checks`; otherwise it lists the 10 most recent runs on the branch. `watch` without `--run-id` follows the newest in-progress or queued run on the branch and exits with the run's status.

## Report

Summarize:

- The exact `gh workflow run` command (dry-run output) and whether it was dispatched
- Workflow, branch, GPU families per platform and test labels
- Run ID and URL when dispatched, or the status/watch result

## Notes

- `scripts/trigger_ci.py` is bundled in this skill so linked and copied installs work independently.
- The script never adds or removes GitHub PR labels. Test labels here are workflow_dispatch input values.
