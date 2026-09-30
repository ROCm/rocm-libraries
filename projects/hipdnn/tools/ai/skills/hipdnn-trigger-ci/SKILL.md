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
- **Branch**: `--branch <branch>` (a global option, placed before the subcommand). If omitted, `--pr <pr-number>` resolves the PR head branch; otherwise the current git branch is used, under its upstream name when it tracks one. The branch must already be pushed to `ROCm/rocm-libraries` for a dispatch to find it.
- **GPU families**: `--gfx` (Linux) and `--windows-gfx`, comma-separated. When omitted the workflow defaults apply:
  - `multi-arch`: Linux `gfx94X,gfx950,gfx125X`, Windows `gfx110X` (`.github/workflows/therock-multi-arch-ci.yml`, `setup` job inputs). Pass `none` to skip a platform, for example `--windows-gfx none` for a Linux-only run.
  - `therock-ci`: Linux `gfx94X, gfx950, gfx125X`, Windows `gfx1151` (`.github/workflows/therock-ci.yml`, "Fetch Linux/Windows targets for build and test" steps).
- **Projects** (`therock-ci` only): `--projects`, space-separated subtree paths that are keys of `subtree_to_project_map` in `.github/scripts/therock_matrix.py` (for example `dnn-providers/integration-tests`, `projects/hipdnn`), or `all`.
- **Test labels** (`multi-arch` only): `--test-labels` (Linux) and `--windows-test-labels`, comma-separated. See the test reference below.

`dispatch` rejects an option the chosen workflow does not take (for example `--test-labels` with `-w therock-ci`, or any option with `-w hipdnn-superbuild`) instead of dropping it.

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
   `status` with `--pr` shows `gh pr checks`; otherwise it lists the 10 most recent runs on the branch. `--branch` takes precedence over `--pr` in every subcommand. `status` exits non-zero when the query itself fails; failing or pending checks are reported, not treated as errors. `watch` without `--run-id` follows the in-progress or queued run with the highest run ID on the branch and exits with the run's status. After a real dispatch, the reported run ID is the newest `workflow_dispatch` run of that workflow on the branch, so a run someone else dispatched on the same branch at the same moment can still be picked up.

## Test reference

Sources, all at the pinned TheRock ref: [fetch_test_configurations.py](https://github.com/ROCm/TheRock/blob/7440cb8578f4daae0d85a428fadd6645dc5464a0/build_tools/github_actions/fetch_test_configurations.py) (component matrix and label matching), [configure_multi_arch_ci.py](https://github.com/ROCm/TheRock/blob/7440cb8578f4daae0d85a428fadd6645dc5464a0/build_tools/github_actions/configure_multi_arch_ci.py) (`_determine_test_type`), [test_runner.py](https://github.com/ROCm/TheRock/blob/7440cb8578f4daae0d85a428fadd6645dc5464a0/build_tools/github_actions/test_executable_scripts/test_runner.py) (ctest invocation), [amdgpu_family_matrix.py](https://github.com/ROCm/TheRock/blob/7440cb8578f4daae0d85a428fadd6645dc5464a0/build_tools/github_actions/amdgpu_family_matrix.py) and [test_artifacts.yml](https://github.com/ROCm/TheRock/blob/7440cb8578f4daae0d85a428fadd6645dc5464a0/.github/workflows/test_artifacts.yml) (sanity gate and component test jobs). Re-read them if the pin in `.github/actions/ci-env/action.yml` has moved.

### Component labels

`--test-labels` values are the `linux_test_labels` / `windows_test_labels` workflow_dispatch inputs, not GitHub PR labels. The string is passed through as is, so no matching label has to exist in the repo. Each label is a key of `test_matrix` in `fetch_test_configurations.py` and selects one TheRock test component. The `test:` prefix is optional (it is stripped, so `test:hipdnn` and `hipdnn` select the same component) and matching is exact. The `sanity` component always runs and gates the rest: component test jobs start only after it passes (`test_components` needs `test_sanity_check` in `test_artifacts.yml`). With no labels, every component that applies to the family and platform runs.

hipDNN components (all run on Linux and Windows, one shard, 30-minute timeout):

| Label | `fetch_artifact_args` | CTest directory in the test job |
|---|---|---|
| `test:hipdnn` | `--hipdnn --tests` | `hipdnn` |
| `test:hipdnn_install` | none | none (runs `test_hipdnn_install.py`) |
| `test:hipdnn-integration-tests` | `--hipdnn --hipdnn-integration-tests --tests` | `hipdnn_integration_tests_ctest` |
| `test:hipdnn-samples` | `--blas --miopen --hipdnn --miopenprovider --hipdnn-samples --tests` | `hipdnn_samples` |
| `test:miopenprovider` | `--blas --miopen --hipdnn --miopenprovider --hipdnn-integration-tests --tests` | `miopen_plugin` |
| `test:hipblasltprovider` | `--blas --hipdnn --hipblasltprovider --hipdnn-integration-tests --tests` | `hipblaslt_plugin` |
| `test:hipkernelprovider` | `--hipdnn --hipkernelprovider --hipdnn-integration-tests --tests` | `hip_kernel_provider` (installs the rocKE wheels first) |

`test:hipdnn_install` uses an underscore; `test:hipdnn-install` matches nothing. The three provider labels also fetch `--hipdnn-integration-tests` and exercise the integration-test harness, so for a change under `dnn-providers/integration-tests` select `test:hipdnn-integration-tests,test:miopenprovider,test:hipblasltprovider,test:hipkernelprovider`.

Other components use their own names (for example `test:rocblas`, `test:hipblaslt`); read the component matrix for the full list. Do not invent labels: an unknown label matches no component, so only sanity runs.

### Test tier

One tier (`test_type`) is chosen for the whole run:

1. A `test_filter:<tier>` entry in the test labels wins. `<tier>` is `quick`, `standard`, `comprehensive` or `full`; any other value fails the setup job.
2. Otherwise any component label selects `full`.
3. Otherwise a dispatch runs `quick`.

`gfx125X` is forced to `quick` for its own test jobs (`test_type_for_family` in the family matrix).

`test_filter:` entries are passed through to component matching unchanged and match no component, so **always pair `test_filter:<tier>` with at least one component label**; on its own it leaves only sanity. Because component labels imply `full`, add `test_filter:quick` for a fast signal on a provider.

Each test job runs `ctest -L ^<tier>$` in its CTest directory, plus the `ex_gpu_<arch>` label when the build defines it, and excludes the `<tier>_exclude` and `<tier>_therock_ci_exclude` labels.

### What each tier runs

CTest labels come from the test category YAMLs. A test in a lower tier also carries every higher tier's label (quick ⊂ standard ⊂ comprehensive ⊂ full).

- `hipdnn`: `projects/hipdnn/test_categories.yaml`. Quick covers all GTests; the `unit` and `integration` categories name the test executables (for example `hipdnn_backend_tests`, `hipdnn_frontend_tests`, `hipdnn_public_backend_tests`).
- Integration-test bundle sweeps are driven by each provider, not by `hipdnn-integration-tests` (see `dnn-providers/integration-tests/CMakeLists.txt`). Bundle GTest suites are named `<tier>_<Op>_<Variant>` after the bundle folder (for example `quick_SdpaFwd_bhsd_bf16_hd128_nomask_batch_Small`). `dnn-providers/miopen-provider/test_categories_integration.yaml` and `dnn-providers/hipblaslt-provider/test_categories_integration.yaml` select `quick_*` for quick, add `standard_*` for standard, add `full_*` for comprehensive, and `*` for full.
- `hipdnn-integration-tests`: `dnn-providers/integration-tests/test_categories.yaml` (quick is everything not prefixed `Standard`, `Comprehensive` or `Full`) and `dnn-providers/integration-tests/test_categories_external.yaml` (the pre-registered `hipdnn-integration-tests_test_name_validation`, `hipdnn_bundle_verifier_python_tests` and `hipdnn_support_claim_verifier_python_tests`).
- `hipkernelprovider`: the `*_test_categories*.yaml` files in `dnn-providers/hip-kernel-provider`, for example `dnn-providers/hip-kernel-provider/ROCKE_ENGINE_test_categories_external.yaml` (the rocKE quick checks).

To choose ctest names or reproduce a tier locally, prefer the `hipdnn-integration-testing` skill once it lands (ALMIOPEN-2578; not on develop yet). Until then, read the YAMLs above.

### Examples

Always dry-run first and show the printed command.

TheRock CI for the integration-tests subtree on one family (no TheRock branch needed):

```bash
python3 <skill-directory>/scripts/trigger_ci.py dispatch -w therock-ci --gfx gfx94X --projects "dnn-providers/integration-tests" --dry-run
```

Multi-arch, Linux-only, MIOpen provider and hipDNN tests at the full tier (implied by the labels):

```bash
python3 <skill-directory>/scripts/trigger_ci.py dispatch -w multi-arch --gfx gfx94X,gfx950 --windows-gfx none --test-labels test:hipdnn,test:miopenprovider --dry-run
```

Same, but at the quick tier:

```bash
python3 <skill-directory>/scripts/trigger_ci.py dispatch -w multi-arch --gfx gfx94X --windows-gfx none --test-labels test:miopenprovider,test_filter:quick --dry-run
```

Check or follow the run on a PR's branch:

```bash
python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> status
python3 <skill-directory>/scripts/trigger_ci.py --pr <pr-number> watch
```

## Report

Summarize:

- The exact `gh workflow run` command (dry-run output) and whether it was dispatched
- Workflow, branch, GPU families per platform and test labels
- Run ID and URL when dispatched, or the status/watch result

## Notes

- `scripts/trigger_ci.py` is bundled in this skill so linked and copied installs work independently.
- The script never adds or removes GitHub PR labels. Test labels here are workflow_dispatch input values.
