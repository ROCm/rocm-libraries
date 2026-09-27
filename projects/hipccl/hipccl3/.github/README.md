# hipCCL CI scaffold (not live)

GitHub Actions only reads `.github/workflows/` at a repository's *root*. Since
`hipccl3` currently lives nested inside `rocm-libraries` at
`projects/hipccl/hipccl3/`, nothing in this directory runs today - it is a
staged copy of `rocm-libraries`' own `.github/` (see
`../../../../docs/hipccl-repository-split-proposal.md`), adapted for hipCCL's
scope, ready to become the real root `.github/` on the day `hipccl3` becomes
the root of its own `ROCm/hipCCL` repository.

## What's here and where it came from

| Path | Source | Adapted? |
|---|---|---|
| `workflows/therock-ci.yml`, `therock-ci-linux.yml`, `therock-ci-windows.yml`, `therock-multi-arch-ci*.yml` | Copied verbatim from `rocm-libraries/.github/workflows/` | No - references `ROCm/TheRock` and generic scripts only |
| `workflows/pre-commit.yml`, `clang-tidy.yml`, `codeql.yml`, `labeler.yml`, `docs-pr-preview*.yml`, `update-docs.yml` | Copied verbatim | No |
| `actions/ci-env`, `setup-rocm-linux`, `setup-rocm-windows`, `pip-install-test` | Copied verbatim | No |
| `scripts/ci_utils.py`, `resolve_therock_ref.py`, `therock_configure_ci.py`, `pre_commit_helper.py`, `pr_detect_changed_subtrees.py`, `config_loader.py`, `repo_config_model.py`, `github_cli_client.py` | Copied verbatim | No |
| `scripts/therock_matrix.py` | Rewritten | **Yes** - trimmed to only the `"prim"` group (rocprim/hipcub/rocthrust); rocm-libraries' version covers every project in that monorepo |
| `repos-config.json`, `docs-config.json`, `labeler.yml`, `CODEOWNERS` | Rewritten | **Yes** - scoped to rocprim/hipcub/rocthrust(/libhipcxx), paths adjusted for hipccl3's flat layout (no `projects/` prefix) |

## Known gaps / not done

- **Math CI**: no in-repo config exists anywhere in `rocm-libraries` for Math
  CI (Jenkins) - it's entirely configured on the Jenkins side against
  `math-ci.amd.com`. There is nothing to scaffold here; new Jenkins jobs would
  need to be created once hipCCL is a real repository.
- **Azure Pipelines / "External CI"**: the pipeline *definitions* referenced
  by `rocm-libraries`' `docs/continuous-integration.md` live in a different
  repo (`ROCm/ROCm/.azuredevops`), not in `rocm-libraries` itself, so there is
  nothing here to copy either - this needs separate coordination with that
  repo's owners.
- **`repos-config.json` category/prefix mismatch**: the shared
  `pr_detect_changed_subtrees.py` builds subtree prefixes as
  `f"{category}/{name}"` (e.g. `"projects/rocprim"`). This scaffold's
  `repos-config.json` uses `"category": "projects"` to match that expectation,
  but hipccl3's actual on-disk layout is flat (`rocprim/`, not
  `projects/rocprim/`). Either hipCCL adopts a `projects/` subfolder
  convention before real extraction, or `get_valid_prefixes()` needs a small
  change to support an empty/no category. Not resolved in this pass.
- **`component-ci*.yml`, `hipblaslt-asan-ci.yml`, `hipdnn-superbuild-ci.yml`**
  intentionally not copied - they're scoped to Tensile/MIOpen/rocISA/hipBLASLt/
  hipDNN, none of which are part of hipCCL.
- None of this has been run. Treat it as a starting point for whoever does the
  real extraction, not as validated CI.
