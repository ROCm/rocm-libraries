# ROCm Libraries

Welcome to the ROCm Libraries monorepo. This repository consolidates multiple ROCm-related libraries and shared components into a single repository to streamline development, CI, and integration. The first set of libraries focuses on components required for building PyTorch.

# Monorepo Status and CI Health

This table provides the current status of the migration of specific ROCm libraries as well as a pointer to their current CI health.

**Key:**
- **Completed**: Fully migrated and integrated. This monorepo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the monorepo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component           | Migration Status | CI Status                            |
|---------------------|------------------|---------------------------------------|
| `composablekernel`  | Pending     |  |
| `hipblas`           | Pending     |  |
| `hipblas-common`    | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipBLAS-common?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=300&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `hipblaslt`         | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipBLASLt?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=301&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `hipcub`            | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipCUB?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=277&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `hipfft`            | Pending     |  |
| `hiprand`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocRAND?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=274&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `hipsolver`         | Pending     |  |
| `hipsparse`         | Pending     |  |
| `hipsparselt`       | Pending     |  |
| `miopen`            | Pending     |  |
| `mxdatagenerator`   | Completed   |  |
| `rocblas`           | Pending     |  |
| `rocfft`            | Pending     |  |
| `rocprim`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocPRIM?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=273&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `rocrand`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocRAND?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=274&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `rocsolver`         | Pending     |  |
| `rocsparse`         | Pending     |  |
| `rocthrust`         | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocThrust?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=276&repoName=ROCm%2Frocm-libraries&branchName=develop) |
| `rocroller`         | Pending     |  |
| `tensile`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FTensile?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=305&repoName=ROCm%2Frocm-libraries&branchName=develop) |

# TheRock CI Status

Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![TheRock CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml/badge.svg?branch=develop&event=push)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml)

---

## Nomenclature

Project names have been standardized to match the casing and punctuation of released packages. This removes inconsistent camel-casing and underscores used in legacy repositories.

## Structure

The repository is organized as follows:

```
projects/
  composablekernel/
  hipblas/
  hipblas-common/
  hipblaslt/
  hipcub/
  hipfft/
  hiprand/
  hipsolver/
  hipsparse/
  hipsparselt/
  miopen/
  rocblas/
  rocfft/
  rocprim/
  rocrand/
  rocsolver/
  rocsparse/
  rocthrust/
shared/
  rocroller/
  tensile/
  mxdatagenerator/
```

- Each folder under `projects/` corresponds to a ROCm library that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple libraries, but does not produce its own distinct packages in previous ROCm releases.

## Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Getting Started

To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This monorepo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-libraries/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-libraries/issues)

We're happy to help!
