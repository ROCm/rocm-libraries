# CI Behavior Manipulation

TheRock CI is controlled by the workflows in [`.github/workflows`](../.github/workflows).
rocm-libraries uses
[`therock-multi-arch-ci.yml`](../.github/workflows/therock-multi-arch-ci.yml)
as the default TheRock CI path. The classic single-arch workflow,
[`therock-ci.yml`](../.github/workflows/therock-ci.yml), remains available as
an explicit fallback.

## Default behavior for push and pull request

TheRock Multi-Arch CI runs by default for pull requests and pushes to
`develop` and `release/therock-*`.

The multi-arch workflow uses
[`configure_external_repo_ci.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/configure_external_repo_ci.py)
with [`repos-config.json`](../.github/repos-config.json) to determine file
changes, changed projects, and tests to run.

Example: a change made to `projects/rocfft` will run the tests selected for
that project.

For CI changes or changes that otherwise require broader coverage, TheRock
Multi-Arch CI runs all tests.

The classic single-arch TheRock CI workflow is skipped by default for pull
requests and pushes.

## Pull request behavior

Here are additional labels that manipulate CI behavior. The labels we provide
are:

- `ci:skip`: skip TheRock Multi-Arch CI builds and tests.
- `ci:run-all-archs`: run TheRock Multi-Arch CI for all known GPU families.
- `ci:single-arch`: run the classic single-arch TheRock CI fallback workflow.
- `ci:asan` or `ci:host-asan`: enable the TheRock Multi-Arch CI ASAN
  workflow for the pull request.
- `gfx...`: add a GPU family to the TheRock Multi-Arch CI run, such as
  `gfx950`.
- `test:<project>`: request tests for a specific project, such as
  `test:rocprim` or `test:hipblaslt`.
- `test_filter:<level>`: override the multi-arch test level. Valid values are
  `quick`, `standard`, `comprehensive`, and `full`.
- `test_runner:<kernel>`: use a custom test runner where the selected GPU
  family supports one.

- `skip-therockci`: skip the classic single-arch TheRock CI fallback workflow.
- `test_type:<level>`: override the classic single-arch test level. Valid
  values are `quick`, `standard`, `comprehensive`, and `full`.

The classic single-arch labels affect only
[`therock-ci.yml`](../.github/workflows/therock-ci.yml) when that fallback
workflow is enabled.

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI from GitHub Actions:

- [TheRock Multi-Arch CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci.yml)
- [TheRock CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml)

Use workflow dispatch for one-off runs that need custom GPU families, test
labels, package build options, or the classic single-arch fallback without
adding a pull request label.

Python package, PyTorch, JAX, and native Linux package builds are not enabled
by default in the pull request and push multi-arch workflow. They can be
enabled through workflow dispatch inputs when needed.

## Schedule behavior

The scheduled nightly TheRock workflow is
[TheRock Multi-Arch Nightly CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci-nightly.yml).
The classic single-arch nightly workflow is available for workflow dispatch
only.
