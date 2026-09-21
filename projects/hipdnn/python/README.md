# hipDNN Python Bindings

> [!CAUTION]
> **This is a POC of python bindings for hipdnn.  It likely has bugs and features missing.  Making this not a POC has been planned for a future date**


This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

```
python/
├── README.md
├── download_third_party_deps.py             # Downloads pinned CI third-party source archives
├── frontend_bindings/
│   ├── CMakeLists.txt                     # CMake build; superbuild component or standalone
│   └── src/
│       ├── module.cpp                     # Main nanobind module entry point
│       ├── bindings.hpp                   # Shared binding declarations
│       ├── autotune_bindings.cpp
│       ├── graph_bindings.cpp
│       ├── handle_bindings.cpp
│       ├── memory_bindings.cpp
│       ├── tensor_bindings.cpp
│       ├── attributes_bindings.cpp
│       ├── hip_bindings.cpp
│       └── types_bindings.cpp
└── frontend_wheel_package/
    ├── src/
    │   └── hipdnn_frontend/
    │       └── __init__.py                # Runtime package initializer
    ├── samples/                           # Source-tree sample scripts
    ├── tests/                             # Source-tree tests
    ├── pyproject.toml                     # Wheel metadata and pytest config
    └── pack_frontend_wheel.py             # Stages and packs the wheel package
```

## Prerequisites

- CMake 3.26 or higher
- Ninja or another CMake generator
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.12 or higher, including development headers
- ROCm/HIP runtime and libraries
- Installed hipDNN development artifacts with `hipdnn_frontendConfig.cmake`,
  `hipdnn_backendConfig.cmake`, headers, and the backend shared library
- `nanobind` and `tsl-robin-map` CMake packages, or network access during
  configure so CMake can fetch the pinned source archives
- The `build` Python package when creating a wheel
- The `numpy` and `pytest` Python packages when running source-tree tests or samples

## Building

The recommended way to build the bindings is as a component of the
rocm-libraries superbuild, alongside hipDNN itself — no separate install step
or `CMAKE_PREFIX_PATH` needed since the extension links the in-tree
`hipdnn_frontend`/`hipdnn_backend` targets directly. From the repository root:

```bash
cmake --preset hipdnn-python -GNinja -DROCM_PATH=/path/to/rocm
cmake --build build --target hipdnn_frontend_bindings
```

See [Superbuild](../docs/Building.md#superbuild) for preset details; e.g.
`hipdnn-dev-all` builds the bindings alongside every provider.

The bindings can also be built standalone against an already-installed hipDNN
(e.g. a ROCm/hipDNN nightly artifact), which doesn't require a full source
checkout:

```bash
cmake -S projects/hipdnn/python/frontend_bindings -B build/hipdnn-python -GNinja \
    -DCMAKE_PREFIX_PATH=/path/to/hipdnn/install
cmake --build build/hipdnn-python
```

`CMAKE_PREFIX_PATH` is required for the standalone build: point it at the
installed hipDNN artifact prefix, or set the `CMAKE_PREFIX_PATH` environment
variable before configuring.

Either way, the build only produces the nanobind extension in the CMake build
tree. CMake does not know about wheel packaging, does not configure
`hipdnn_frontend/__init__.py`, and has no install rules.

Run the wheel packer to create the staged import package under
`<build-dir>/wheel_package/hipdnn_frontend`; downstream environment wiring
should use that staged package tree.

For a source-tree development import after staging the package, put the staged
wheel package root on `PYTHONPATH`:

```bash
PYTHONPATH=build/wheel_package python -c "import hipdnn_frontend"
```

The backend shared library must also be discoverable at runtime: use
`LD_LIBRARY_PATH=/path/to/hipdnn/install/lib` on Linux, or set `ROCM_PATH` to the
artifact prefix on Windows so `hipdnn_frontend/__init__.py` can register `bin/`.

## Creating a Wheel

After building the bindings, run the packer script:

```bash
python projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py \
    --build-dir <build-dir> \
    --wheel-dir <build-dir>/wheel_package
```

`<build-dir>` is `build` for the superbuild or `build/hipdnn-python` for the
standalone build; the packer searches the whole tree for the extension, so
the exact nesting doesn't matter. The wheel is written to
`<build-dir>/wheel_package/`, beside the `hipdnn_frontend/` package directory.
The script packs `wheel_package/hipdnn_frontend` into a temporary setuptools
project. The wheel contains only `hipdnn_frontend/__init__.py` and the native
extension. It does not include samples or tests, and it does not bundle
`libhipdnn_backend`; users still need ROCm and hipDNN runtime libraries
discoverable through ROCm wheels, `ROCM_PATH`, or the platform loader path.

## Testing the Wheel

The `hipDNN Superbuild CI` workflow validates the wheel end-to-end inside the
matching Linux and Windows superbuild jobs after installing the superbuild
outputs into the ROCm SDK path. The workflow calls
`projects/hipdnn/python/download_third_party_deps.py` to download and verify
pinned third-party source archives from `rocm-third-party-deps`, then passes
those source directories to CMake FetchContent as part of the single
superbuild configure. It then builds hipDNN and the bindings together, packs
the wheel, installs that wheel into the same venv, and runs:

```bash
python -m pytest -q projects/hipdnn/python/frontend_wheel_package/tests
```

The wheel package uses a `src/` layout, so running pytest from
`frontend_wheel_package/` does not accidentally import the source package.

## Running the Samples

Sample scripts are source-tree utilities and are not included in the wheel.

```bash
python projects/hipdnn/python/frontend_wheel_package/samples/conv_fprop.py
python projects/hipdnn/python/frontend_wheel_package/samples/conv_dgrad.py
python projects/hipdnn/python/frontend_wheel_package/samples/conv_wgrad.py
python projects/hipdnn/python/frontend_wheel_package/samples/matmul.py
python projects/hipdnn/python/frontend_wheel_package/samples/autotune.py
```
