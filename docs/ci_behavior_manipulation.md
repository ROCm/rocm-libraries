# CI Behavior Manipulation

rocm-libraries runs TheRock CI through
[`therock-multi-arch-ci.yml`](../.github/workflows/therock-multi-arch-ci.yml),
which calls TheRock's reusable multi-arch workflows. The labels and
`workflow_dispatch` inputs that control a run are therefore interpreted by
TheRock, and are documented there:

**[TheRock: CI Behavior Manipulation](https://github.com/ROCm/TheRock/blob/main/docs/development/ci_behavior_manipulation.md)**

That page is the authority for `ci:skip`, `ci:run-all-archs`, `ci:asan`,
`ci:gfx...`, `test:<project>`, `test_filter:<level>` and the rest. Keep it as
the single source of truth rather than restating the list here, which is how
this page went stale before.

## rocm-libraries specifics

- `ci:native-linux-packages` opts a pull request into native Linux package
  builds. This label is read by `therock-multi-arch-ci.yml` in this repo, not
  by TheRock, and is spelled differently from TheRock's own
  `ci:build-native-linux`.
- Changed projects and the resulting test set are resolved by TheRock's
  [`configure_external_repo_ci.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/configure_external_repo_ci.py)
  against this repo's [`repos-config.json`](../.github/repos-config.json).
  Paths listed in [`skip-ci-config.toml`](../.github/skip-ci-config.toml) do
  not trigger a build.
- The scheduled nightly run is
  [TheRock Multi-Arch Nightly CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci-nightly.yml);
  ASAN coverage runs in `therock-multi-arch-ci-asan*.yml`.
- Python package, PyTorch, JAX and native Linux package builds are off by
  default on pull requests and pushes, and are enabled through
  [workflow dispatch](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci.yml)
  inputs.

## Coming from math-ci

The math-ci (Jenkins) controls you may know have these TheRock CI
equivalents. See the TheRock page above for what each label does.

| math-ci | TheRock CI | Notes |
|---|---|---|
| `project: <name>` to test another library on a PR | Nothing to add | A PR tests the projects it changes and their direct consumers. `test:<project>` limits testing to the named projects, and `ci:no-downstream-test` skips the consumers. math-ci ran downstream libraries only after merge. |
| `ci:extended` | `test_filter:comprehensive`, usually with `test:<project>` | Each library's test categories file sets what a level runs. For rocBLAS, `comprehensive` runs the `*nightly*` tests that `extended` ran. |
| `ci:smoke` | `test_filter:quick` | rocFFT and hipFFT run `--smoketest` at the `quick` level. |
| `ci:multi-gpu` | `ci:run-multi-gpu` | Multi-GPU test jobs exist for rocFFT and hipFFT only. |
| (none) | `ci:asan` | Builds and tests with ASAN. Add `test:<project>` to limit the test jobs. |

A pull request that changes a library runs the `standard` level, and the
nightly run uses `comprehensive`. Pull request tests run on gfx94X by
default; to test other families, use the `ci:gfx...` and `ci:test:gfx...`
labels. math-ci jobs not listed here, such as `ci:weekly` and `ci:debug`, have
no TheRock CI equivalent yet.
