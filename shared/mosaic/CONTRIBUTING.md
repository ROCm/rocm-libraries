# Contributing to Mosaic

Mosaic lives in `shared/mosaic` of
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries).
Follow the monorepo [CONTRIBUTING.md](../../CONTRIBUTING.md) and the
[ROCm pull-request guidelines](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests).

## Getting started

1. Branch from `develop` as `users/<github-username>/<short-description>`.
1. Make the change. The root [pre-commit](../../.pre-commit-config.yaml) hooks
   run on this tree.
1. Open the pull request against `develop`. Explain why the change was made
   and how you verified it.

## Code style

C and C++ are formatted with `clang-format`, Python with `black`, and CMake
is checked with `cmake-lint`.

New C++ and CMake sources start with:

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

New Python sources start with:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## Testing

From `shared/mosaic`:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure

pip install -e 'python/pymosaic[dev]'
python -m pytest python/tests
```

Host C++ tests do not require a GPU. HIP examples stay off unless you pass
`-DMOSAIC_BUILD_HIP_EXAMPLES=ON`.

To build mosaic through the monorepo superbuild, configure the repo root with
`-DROCM_LIBS_ENABLE_COMPONENTS=mosaic`. The component is not in the default
`all` set.

## Security

Do not report security vulnerabilities through a public issue or pull request.
See [SECURITY.md](SECURITY.md).
