# ROCm Libraries

Welcome to the ROCm Libraries super-repo. This repository consolidates multiple ROCm-related libraries and shared components into a single repository to streamline development, CI, and integration.

## Super-repo Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Super-repo Project Status

### TheRock CI Status

TheRock CI performs multi-component testing on top of builds leveraging the [TheRock](https://github.com/ROCm/TheRock) build system.

[![TheRock Multi-Arch CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci.yml/badge.svg?branch=develop&event=push)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci.yml?query=branch%3Adevelop+event%3Apush) [![TheRock Multi-Arch Nightly CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci-nightly.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci-nightly.yml?query=branch%3Adevelop)

### Library Directory and Math CI Status


| Library | Description | Math CI Status | Documentation |
|---------|-------------|----------------|---------------|
| [`Composable Kernel`](./projects/composablekernel/) | Composable GPU kernel building blocks for machine learning workloads. | — | [AMD Docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/) |
| [`hipBLAS`](./projects/hipblas/) | Portable BLAS interface supporting rocBLAS and cuBLAS backends. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/) |
| [`hipBLAS-common`](./projects/hipblas-common/) | Shared headers and types used by hipBLAS and hipBLASLt. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblas-common/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblas-common/job/develop/lastBuild/) | [Project README](./projects/hipblas-common/) |
| [`hipBLASLt`](./projects/hipblaslt/) | Flexible, GPU-accelerated matrix multiplication (GEMM) operations. | [![Math-CI PreCheckin](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipblaslt/develop&subject=Math-CI%20PreCheckin)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipblaslt/job/develop/lastBuild/)  [![Math-CI Preliminary](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/preliminary/hipblaslt/develop&subject=Math-CI%20Preliminary)](http://math-ci.amd.com/job/rocm-libraries/job/preliminary/job/hipblaslt/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) |
| [`hipCUB`](./projects/hipcub/) | Portable CUB-compatible parallel primitives API. | [![Math CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipcub/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipcub/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipCUB/en/latest/) |
| [`hipDNN`](./projects/hipdnn/) | Graph-based deep learning API with pluggable execution backends. | — | [AMD Docs](https://rocm.docs.amd.com/projects/hipdnn/en/latest/) |
| [`hipFFT`](./projects/hipfft/) | Portable interface for GPU fast Fourier transforms. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipfft/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipFFT/en/latest/) |
| [`hipRAND`](./projects/hiprand/) | Portable interface for GPU random number generation. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hiprand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hiprand/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipRAND/en/latest/) |
| [`hipSOLVER`](./projects/hipsolver/) | Portable interface for GPU dense linear algebra solvers. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsolver/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/) |
| [`hipSPARSE`](./projects/hipsparse/) | Portable interface for GPU sparse linear algebra operations. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparse/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/) |
| [`hipSPARSELt`](./projects/hipsparselt/) | GPU-accelerated sparse matrix multiplication. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hipsparselt/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hipsparselt/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/) |
| [`hipTensor`](./projects/hiptensor/) | GPU tensor contraction, permutation, and reduction primitives. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/hiptensor/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/hiptensor/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/hipTensor/en/latest/) |
| [`MIOpen`](./projects/miopen/) | GPU-accelerated deep learning primitives and convolutions. | [![MICI](https://pcue-math-rocm-ci-apim.azure-api.net/micibuildstatus?job=/rocm-libraries-folder/MIOpen/develop&subject=MICI)](http://micimaster.amd.com/job/rocm-libraries-folder/job/MIOpen/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) |
| [`rocBLAS`](./projects/rocblas/) | AMD GPU-optimized BLAS and matrix multiplication routines. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocblas/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocblas/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/) |
| [`rocFFT`](./projects/rocfft/) | AMD GPU-optimized fast Fourier transforms. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocfft/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocfft/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) |
| [`rocPRIM`](./projects/rocprim/) | Low-level, GPU-accelerated parallel primitives. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocprim/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocprim/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/) |
| [`rocRAND`](./projects/rocrand/) | GPU random number generation and probability distributions. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocrand/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocrand/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocRAND/en/latest/) |
| [`rocSOLVER`](./projects/rocsolver/) | GPU-accelerated dense matrix factorization and solvers. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsolver/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsolver/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/) |
| [`rocSPARSE`](./projects/rocsparse/) | GPU-accelerated sparse linear algebra operations. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocsparse/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocsparse/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/) |
| [`rocThrust`](./projects/rocthrust/) | High-level parallel algorithms for HIP and CUDA. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocthrust/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocthrust/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocThrust/en/latest/) |
| [`rocWMMA`](./projects/rocwmma/) | Wavefront-level matrix multiply-accumulate primitives. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocwmma/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocwmma/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/) |
| [`RPP`](./projects/rpp/) | Image processing and computer vision primitives. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin-cv-nightly-therock/rpp/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin-cv-nightly-therock/job/rpp/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/rpp/en/latest/) |
| [`rocRoller`](./shared/rocroller/) | Generator for optimized AMD GPU assembly kernels. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/rocroller/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/rocroller/job/develop/lastBuild/) | [Project README](./shared/rocroller/) |
| [`Tensile`](./shared/tensile/) | Benchmark-driven generation and selection of GEMM kernels. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/tensile/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/tensile/job/develop/lastBuild/) | [AMD Docs](https://rocm.docs.amd.com/projects/Tensile/en/latest/) |
| [`mxDataGenerator`](./shared/mxdatagenerator/) | Test data generation for low-precision floating-point formats. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/mxdatagenerator/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/mxdatagenerator/job/develop/lastBuild/) | [Project README](./shared/mxdatagenerator/) |
| [`Origami`](./shared/origami/) | Analytical selection of GEMM kernel configurations. | [![Math-CI](https://pcue-math-rocm-ci-apim.azure-api.net/buildstatus?job=/rocm-libraries/precheckin/origami/develop&subject=MathCI)](http://math-ci.amd.com/job/rocm-libraries/job/precheckin/job/origami/job/develop/lastBuild/) | [Project README](./shared/origami/) |

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
  rpp/
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
