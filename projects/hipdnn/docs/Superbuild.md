# Superbuild: Building hipDNN with Providers

## Overview

The rocm-libraries superbuild allows building hipDNN and its provider plugins (miopen-provider, hipblaslt-provider) together from a single build directory. This eliminates the need to install hipDNN before building providers and ensures all components are built against the same source.

## Quick Start

From the rocm-libraries root:

```bash
# Using the hipdnn preset (recommended)
cmake --preset hipdnn

# Or manually specifying components
cmake -B build -GNinja -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider;hipblaslt-provider" .
cmake --build build
```

## How It Works

The superbuild uses `add_subdirectory()` to include each component sequentially. Components added earlier make their CMake targets visible to components added later, removing the need for `find_package()`.

### Dependency Flow

```
hipdnn (added first)
├── hipdnn_data_sdk    (INTERFACE target)
├── hipdnn_plugin_sdk  (INTERFACE target)
├── hipdnn_test_sdk    (INTERFACE target)
└── hipdnn_backend     (shared library)
         │
         ▼
miopen-provider / hipblaslt-provider (added after)
└── Uses SDK targets directly (skips find_package)
```

Each provider's `CMakeLists.txt` uses conditional target checks:

```cmake
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()
```

This means the same CMakeLists.txt works for both superbuild and standalone builds.

## Components

| Component | Directory | Key Target | External Dependencies |
|-----------|-----------|------------|----------------------|
| hipdnn | `projects/hipdnn/` | `hipdnn_backend` | ROCm/HIP |
| miopen-provider | `dnn-providers/miopen-provider/` | `miopen_plugin` | MIOpen |
| hipblaslt-provider | `dnn-providers/hipblaslt-provider/` | `hipblaslt_plugin` | hipBLASLt |

### Selecting Components

You can build any subset of components:

```bash
# hipDNN only (no plugins)
cmake -B build -GNinja -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn" .

# hipDNN + miopen-provider only
cmake -B build -GNinja -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider" .

# All DNN components
cmake -B build -GNinja -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider;hipblaslt-provider" .
```

> **Note:** Providers depend on hipDNN. If you include a provider, you must also include `hipdnn`.

## Build Targets

In the superbuild, targets are prefixed with the project name to avoid collisions between components.

### Code Quality Targets

Each component provides prefixed targets and hyphenated aliases:

| Alias Target | Description |
|-------------|-------------|
| `hipdnn-check-format` | Check hipDNN code formatting |
| `hipdnn-format` | Auto-format hipDNN sources |
| `hipdnn-tidy` | Run clang-tidy on hipDNN |
| `miopen-provider-check-format` | Check miopen-provider code formatting |
| `miopen-provider-format` | Auto-format miopen-provider sources |
| `miopen-provider-tidy` | Run clang-tidy on miopen-provider |
| `hipblaslt-provider-check-format` | Check hipblaslt-provider code formatting |
| `hipblaslt-provider-format` | Auto-format hipblaslt-provider sources |

### Test Targets

Each component provides prefixed test targets:

| Alias Target | Description |
|-------------|-------------|
| `hipdnn-check` | Run all hipDNN tests |
| `hipdnn-unit-check` | Run hipDNN unit tests only |
| `hipdnn-integration-check` | Run hipDNN integration tests only |
| `miopen-provider-check` | Run all miopen-provider tests |
| `miopen-provider-unit-check` | Run miopen-provider unit tests only |
| `miopen-provider-integration-check` | Run miopen-provider integration tests only |
| `hipblaslt-provider-check` | Run all hipblaslt-provider tests |
| `hipblaslt-provider-unit-check` | Run hipblaslt-provider unit tests only |


## Superbuild vs Standalone Builds

| | Superbuild | Standalone |
|-|-----------|-----------|
| **When to use** | Building multiple components together | Working on a single component |
| **Setup** | Single cmake configure from repo root | Configure from component directory |
| **Dependencies** | Automatic via target visibility | Requires `find_package()` (install deps first) |
| **Target names** | Prefixed (e.g., `hipdnn-check`) | Unprefixed (e.g., `check`) |
| **Build directory** | `<repo-root>/build/` | `<component>/build/` |

For standalone build instructions, see:
- [Building hipDNN](./Building.md)
- [miopen-provider README](../../../dnn-providers/miopen-provider/README.md)
- [hipblaslt-provider README](../../../dnn-providers/hipblaslt-provider/README.md)

## Troubleshooting

### Target name collisions

If you see CMake errors about duplicate target names, ensure you are building from the repository root (not from a component subdirectory) when using the superbuild. The superbuild sets `ROCM_LIBS_SUPERBUILD=ON`, which enables target name prefixing.
