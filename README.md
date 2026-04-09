# ROCm Libraries

Welcome to the ROCm Libraries super-repo. This repository consolidates multiple ROCm-related libraries and shared components into a single repository to streamline development, CI, and integration.

## Super-repo Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Super-repo Project Status

### TheRock CI Status

TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![TheRock CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml/badge.svg?branch=develop&event=push)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush) [![TheRock CI Nightly](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci-nightly.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci-nightly.yml?query=branch%3Adevelop)

### Component Migration and Legacy CI Status

This table provides the current status of the migration of specific components as well as a pointer to the health of their legacy CI systems.

**Key:**
- **Completed**: Fully migrated and integrated. This super-repo should be considered the source of truth for this project. The old repo may still be used for certain release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the super-repo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component           | Migration Status |  Math CI Status                        |
|---------------------|------------------|---------------------------------------|
| `composablekernel`  | Completed   |    |
| `hipblas`           | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas/job/develop/lastBuild/) |
| `hipblas-common`    | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas-common/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas-common/job/develop/lastBuild/) |
| `hipblaslt`         | Completed   |  [![Math-CI PreCheckin](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblaslt/develop&subject=Math-CI%20PreCheckin)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblaslt/job/develop/lastBuild/)  [![Math-CI Preliminary](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/preliminary/hipblaslt/develop&subject=Math-CI%20Preliminary)](http://math-ci.amd.com/job/rocm-libraries/job/preliminary/job/hipblaslt/job/develop/lastBuild/)|
| `hipcub`            | Completed   | [![Math CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipcub/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipcub/job/develop/lastBuild/) |
| `hipfft`            | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipfft/job/develop/lastBuild/) |
| `hiprand`           | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hiprand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hiprand/job/develop/lastBuild/) |
| `hipsolver`         | Completed   |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsolver/job/develop/lastBuild/) |
| `hipsparse`         | Completed   |   [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparse/job/develop/lastBuild/) |
| `hipsparselt`       | Completed | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparselt/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparselt/job/develop/lastBuild/) |
| `miopen`            | Completed |  [![MICI](https://pcue-math-rocm-ci-apim.azure-api.net/micibuildstatus?job=/rocm-libraries-folder/MIOpen/develop&subject=MICI)](http://micimaster.amd.com/job/rocm-libraries-folder/job/MIOpen/job/develop/lastBuild/) |
| `mxdatagenerator`   | Completed |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/mxdatagenerator/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/mxdatagenerator/job/develop/lastBuild/) |
| `origami`         | Completed     |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/origami/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/origami/job/develop/lastBuild/) |
| `rocblas`           | Completed |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocblas/job/develop/lastBuild/) |
| `rocfft`            | Completed |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocfft/job/develop/lastBuild/) |
| `rocprim`           | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocprim/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocprim/job/develop/lastBuild/) |
| `rocrand`           | Completed   |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocrand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocrand/job/develop/lastBuild/) |
| `rocsolver`         | Completed    |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsolver/job/develop/lastBuild/) |
| `rocsparse`         | Completed   |  [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsparse/job/develop/lastBuild/) |
| `rocthrust`         | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocthrust/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocthrust/job/develop/lastBuild/) |
| `rocroller`         | Completed       |[![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocroller/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocroller/job/develop/lastBuild/) |
| `tensile`           | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/tensile/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/tensile/job/develop/lastBuild/) |
| `rocwmma`           | Completed   | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocwmma/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocwmma/job/develop/lastBuild/) |
| `hiptensor`           | Completed  | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hiptensor/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hiptensor/job/develop/lastBuild/) |

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
  hipdnn/
  hipfft/
  hiprand/
  hipsolver/
  hipsparse/
  hipsparselt/
  hiptensor/
  miopen/
  rocblas/
  rocfft/
  rocprim/
  rocrand/
  rocsolver/
  rocsparse/
  rocthrust/
  rocwmma/
shared/
  rocroller/
  tensile/
  mxdatagenerator/
  origami/
```

- Each folder under `projects/` corresponds to a ROCm library that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple libraries, but does not produce its own distinct packages in previous ROCm releases.

## Getting Started

To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This super-repo contains multiple subprojects, each of which retains the license under which it was originally published.

- 📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.
- 📄 Refer to the header notice in individual files outside `projects/` or `shared/` folders for their specific license terms.

> [!NOTE]
> The root of this repository does not yet define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-libraries/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-libraries/issues)

We're happy to help!

### Filing an issue

When you open an issue in the rocm-libraries repo, please help us help you by being as clear and
reproducible as possible. Before creating a new issue, please search existing ones to avoid duplicates.
For bug reports, include a minimal reproducible example (or small test case) that triggers the error,
along with full environment details (ROCm version, GPU, compiler, OS, etc.). If relevant, try to reduce
the problem (e.g., smaller code snippet) to make diagnosis easier. Finally, if you have ideas for how
to fix or improve something, feel free to suggest them — maintainers appreciate actionable feedback.

Be sure to check out the [hipblaslt runtime error triage checklist](./docs/hipblaslt-runtime-triage-checklist.md)
for a detailed step-by-step process for identifying and reporting runtime errors so we have all of the
information necessary to help resolve the issue quickly. While this checklist is specific to hipblaslt
the same process is generally applicable to other rocm-libraries components.
