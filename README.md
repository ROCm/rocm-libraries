# ROCm Libraries

Welcome to the ROCm Libraries monorepo. This repository consolidates multiple ROCm-related libraries and shared components into a single repository to streamline development, CI, and integration. The first set of libraries focuses on components required for building PyTorch.

# Monorepo Status and CI Health

This table provides the current status of the migration of specific ROCm libraries as well as a pointer to their current CI health.

**Key:**
- **Completed**: Fully migrated and integrated. This monorepo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the monorepo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component           | Migration Status | Azure CI Status                       | Math CI Status                        |
|---------------------|------------------|---------------------------------------|---------------------------------------|
| `composablekernel`  | Pending     |  |  |
| `hipblas`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipBLAS?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=317&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas/job/develop/lastBuild/) |
| `hipblas-common`    | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipBLAS-common?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=300&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas-common/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas-common/job/develop/lastBuild/) |
| `hipblaslt`         | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipBLASLt?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=301&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![PreCheckin](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblaslt/develop&subject=PreCheckin)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblaslt/job/develop/lastBuild/)  [![Preliminary](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/preliminary/hipblaslt/develop&subject=Preliminary)](http://math-ci.amd.com/job/rocm-libraries/job/preliminary/job/hipblaslt/job/develop/lastBuild/)|
| `hipcub`            | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipCUB?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=277&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipcub/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipcub/job/develop/lastBuild/) |
| `hipfft`            | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipFFT?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=283&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipfft/job/develop/lastBuild/) |
| `hiprand`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipRAND?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=275&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hiprand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hiprand/job/develop/lastBuild/) |
| `hipsolver`         | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipSOLVER?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=322&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsolver/job/develop/lastBuild/) |
| `hipsparse`         | Completed   |  [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipSPARSE?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=315&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparse/job/develop/lastBuild/) |
| `hipsparselt`       | Completed | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FhipSPARSELt?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=309&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparselt/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparselt/job/develop/lastBuild/) |
| `miopen`            | Completed | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FMIOpen?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=320&repoName=ROCm%2Frocm-libraries&branchName=develop) | |
| `mxdatagenerator`   | Completed | | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/mxdatagenerator/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/mxdatagenerator/job/develop/lastBuild/) |
| `rocblas`           | Completed | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocBLAS?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=302&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocblas/job/develop/lastBuild/) |
| `rocfft`            | Completed | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocFFT?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=282&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocfft/job/develop/lastBuild/) |
| `rocprim`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocPRIM?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=273&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocprim/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocprim/job/develop/lastBuild/) |
| `rocrand`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocRAND?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=274&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocrand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocrand/job/develop/lastBuild/) |
| `rocsolver`         | Completed    | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocSOLVER?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=303&repoName=ROCm%2Frocm-libraries&branchName=develop) | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsolver/job/develop/lastBuild/) |
| `rocsparse`         | Completed   |  [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocSPARSE?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=314&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsparse/job/develop/lastBuild/) |
| `rocthrust`         | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FrocThrust?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=276&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocthrust/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocthrust/job/develop/lastBuild/) |
| `rocroller`         | Completed     |  |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocroller/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocroller/job/develop/lastBuild/) |
| `tensile`           | Completed   | [![Azure CI](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Fmonorepo%2FTensile?repoName=ROCm%2Frocm-libraries&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=305&repoName=ROCm%2Frocm-libraries&branchName=develop) |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/tensile/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/tensile/job/develop/lastBuild/) |


## Tentative migration schedule

| Component           | Tentative Date |
|---------------------|----------------|
| `composable_kernel` | TBD            |


*Remaining math libraries will be migrated in August*

# TheRock CI Status

Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![The Rock CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml/badge.svg?branch%3Adevelop+event%3Apush)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush)

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
📄 Refer to the header notice in individual files outside `projects/` or `shared/` folders for their specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-libraries/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-libraries/issues)

We're happy to help!
