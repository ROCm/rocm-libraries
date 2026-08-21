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
│   ├── CMakeLists.txt                     # Standalone CMake build for bindings
│   └── src/
│       ├── module.cpp                     # Main nanobind module entry point
│       ├── bindings.hpp                   # Shared binding declarations
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

The Python bindings are a standalone CMake project. Build and install hipDNN
first, then point `CMAKE_PREFIX_PATH` at that install or nightly artifact prefix.
The extension uses installed hipDNN package metadata for headers and compile
definitions, and links the installed `hipdnn_frontend` and `hipdnn_backend`
package targets directly.

From the repository root:

```bash
cmake -S projects/hipdnn/python/frontend_bindings -B build/hipdnn-python -GNinja \
    -DCMAKE_PREFIX_PATH=/path/to/hipdnn/install
cmake --build build/hipdnn-python
```

`CMAKE_PREFIX_PATH` is required. Point it at the installed hipDNN artifact prefix
or set the `CMAKE_PREFIX_PATH` environment variable before configuring.

The default build builds only the nanobind extension into the CMake build tree.
CMake does not know about wheel packaging, does not configure
`hipdnn_frontend/__init__.py`, and has no install rules.

Run the wheel packer to create the staged import package under
`build/hipdnn-python/wheel_package/hipdnn_frontend`; downstream environment
wiring should use that staged package tree.

For a source-tree development import after staging the package, put the staged
wheel package root on `PYTHONPATH`:

```bash
PYTHONPATH=build/hipdnn-python/wheel_package python -c "import hipdnn_frontend"
```

The backend shared library must also be discoverable at runtime: use
`LD_LIBRARY_PATH=/path/to/hipdnn/install/lib` on Linux, or set `ROCM_PATH` to the
artifact prefix on Windows so `hipdnn_frontend/__init__.py` can register `bin/`.

## Creating a Wheel

After building the bindings, run the packer script:

```bash
python projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py \
    --build-dir build/hipdnn-python \
    --wheel-dir build/hipdnn-python/wheel_package
```

The wheel is written to `build/hipdnn-python/wheel_package/`, beside the
`hipdnn_frontend/` package directory. The script packs
`wheel_package/hipdnn_frontend` into a temporary setuptools project. The wheel
contains only `hipdnn_frontend/__init__.py` and the native extension. It does
not include samples or tests, and it does not bundle `libhipdnn_backend`; users
still need ROCm and hipDNN runtime libraries discoverable through ROCm wheels,
`ROCM_PATH`, or the platform loader path.

## Testing the Wheel

The `hipDNN Superbuild CI` workflow validates the wheel end-to-end inside the
matching Linux and Windows superbuild jobs after installing the superbuild
outputs into the ROCm SDK path. The workflow calls
`projects/hipdnn/python/download_third_party_deps.py` to download and verify
pinned third-party source archives from `rocm-third-party-deps`, then passes
those source directories to CMake FetchContent. It then builds the nanobind
extension, packs the wheel, installs that wheel into the same venv, and runs:

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
```
