# Continuous Integration

> [!IMPORTANT]
> This document is currently in **draft** and may be subject to change.

This document is to detail the various continuous integration (CI) systems that are run on the rocm-libraries monorepo.

## Table of Contents
- [Continuous Integration](#continuous-integration)
  - [Table of Contents](#table-of-contents)
  - [Math CI](#math-ci)
    - [Overview ](#overview--1)
  - [Windows CI](#windows-ci)
    - [Overview ](#overview--2)
  - [TheRock CI](#therock-ci)
    - [Overview ](#overview--3)

## Math CI

### Overview <a id="math-overview"></a>

## Windows CI

### Overview <a id="win-overview"></a>

## TheRock CI

### Overview <a id="rock-overview"></a>

TheRock CI runs Linux and Windows builds and tests for nightly jobs, pushes to develop, pull requests and workflow dispatches. It is implemented by [`therock-multi-arch-ci.yml`](../.github/workflows/therock-multi-arch-ci.yml), which calls TheRock's reusable multi-arch workflows to run a sharded, multi-stage build with per-architecture testing.

TheRock CI builds using `rocm-libraries` at HEAD with TheRock build system. For testing, we test using TheRock's [`fetch_test_configurations.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/fetch_test_configurations.py) along with [ROCm test machines](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/amdgpu_family_matrix.py).

For pull requests and pushes to `develop`, TheRock's [`configure_external_repo_ci.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/configure_external_repo_ci.py) reads [`repos-config.json`](../.github/repos-config.json) to determine which projects changed and which tests to run:
- If a change is made to CI configuration or otherwise requires broader coverage, all projects are built and all tests are run.
- If a change is made to `projects/*` or `shared/*`, only the affected projects are built and tested.
- Paths listed in [`skip-ci-config.toml`](../.github/skip-ci-config.toml) do not trigger a build.

For nightly jobs, [`therock-multi-arch-ci-nightly.yml`](../.github/workflows/therock-multi-arch-ci-nightly.yml) runs all builds and full tests for all projects. Dedicated ASAN coverage runs in `therock-multi-arch-ci-asan*.yml`.

For workflow dispatch triggers, TheRock CI will run builds and tests for whichever GPU families and test labels are specified. See [CI behavior manipulation](ci_behavior_manipulation.md) for the labels and dispatch inputs that control this.
