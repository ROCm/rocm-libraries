# ROCm Libraries

Welcome to the ROCm Libraries monorepo. This repository consolidates multiple ROCm-related libraries and shared components into a single repository to streamline development, CI, and integration. The first set of libraries focuses on components required for building PyTorch.

# Monorepo Migration Status

This table provides the current status of the migration of specific ROCm libraries.

**Key:**
- **Completed**: Fully migrated and integrated. This monorepo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the monorepo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component           | Migration Status | Notes                                 |
|---------------------|------------------|---------------------------------------|
| `composablekernel`  | Pending     |  |
| `hipblas`           | Pending     |  |
| `hipblas-common`    | Completed   |  |
| `hipblaslt`         | Completed   |  |
| `hipcub`            | Completed   |  |
| `hipfft`            | Pending     |  |
| `hiprand`           | Completed   |  |
| `hipsolver`         | Pending     |  |
| `hipsparse`         | Pending     |  |
| `hipsparselt`       | Pending     |  |
| `miopen`            | Pending     |  |
| `rocblas`           | Pending     |  |
| `rocfft`            | Pending     |  |
| `rocprim`           | Completed   |  |
| `rocrand`           | Completed   |  |
| `rocsolver`         | Pending     |  |
| `rocsparse`         | Pending     |  |
| `rocthrust`         | Completed   |  |
| `rocroller`         | Pending     |  |
| `tensile`           | Completed   |  |

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
