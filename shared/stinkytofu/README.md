# StinkyTofu

StinkyTofu is an LLVM-inspired pass-based IR optimizer for AMD GPU assembly kernels. It is used by hipBLASLt/TensileLite to schedule and optimize generated GPU code for gfx1250.

## Features

- **Two-level IR**: High-level Logical IR (architecture-agnostic) and low-level Asm IR (architecture-specific)
- **Pass pipeline**: DAG scheduling, wait count insertion, dead code elimination, redundant mov elimination, peephole optimization
- **TableGen-based instruction definitions**: New architectures require only `.def` files, no C++ changes
- **FileCheck-style testing**: LLVM-style lit tests with `stinkytofu-check`
- **Python bindings**: Full IR construction and optimization accessible from Python via nanobind

## Prerequisites

- [CMake](https://cmake.org/) >= 3.16
- [Ninja](https://ninja-build.org/)
- [ROCm SDK](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) (provides `amdclang++`)

### Windows additional setup

Enable long paths (run as Administrator):

```cmd
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Install Visual Studio Build Tools:

```cmd
winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --override "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

## Build

### Python environment

Python 3.12 with a virtual environment is recommended for both platforms:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Linux
.venv\Scripts\activate      # Windows
```

**Note:** Avoid installing Python from the Microsoft Store on Windows — its path contains spaces, which breaks Ninja. The build falls back to NMake (single-threaded). Install Python from [python.org](https://www.python.org/) instead.

When using ROCm SDK, install it into the venv:

```bash
pip install rocm-sdk
rocm-sdk init
```

### Standalone (Linux)

The quickest way is `invoke`, which wraps the CMake steps:

```bash
pip install invoke
invoke build
```

Or manually with CMake:

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build
```

### Standalone (Windows)

```bash
pip install invoke
invoke build
```

### As part of hipBLASLt

StinkyTofu is automatically built when hipBLASLt includes it via `add_subdirectory`. Tests and Python bindings are disabled in sub-project mode.

### Rebuilding Python bindings after C++ changes

If you modify C++ sources, the installed `.so` becomes stale. Importing `stinkytofu` will raise an `ImportError` listing the modified files.

**Standalone** — rebuild with invoke or CMake directly:

```bash
invoke build
# or:
cmake --build build --target stinkytofu_python
```

**As part of hipBLASLt** — rebuild the Python bindings target in your hipBLASLt build directory:

```bash
cmake --build <build_dir> --target stinkytofu_python
```

## Run clang-tidy

We use `clang-tidy-20`, so additional installation may required.

### 1. Download and Run the LLVM Setup Script
This script automatically detects your OS distribution, adds the correct LLVM stable/nightly APT repository, and updates your package lists[span_2](start_span)[span_2](end_span).

```bash
# Get the official script and make it executable
wget [https://apt.llvm.org/llvm.sh](https://apt.llvm.org/llvm.sh)
chmod +x llvm.sh

# Run the script to add repositories
sudo ./llvm.sh 20 reponly
```

### Install `clang-tidy-20`

```bash
sudo apt-get update
sudo apt-get install -y clang-tidy-20
```

### Run clang-tidy

```bash
invoke build
invoke tidy
```

## Test

```bash
# All tests
cd build && ctest

# Unit tests only
./build/tests/unit_tests

# Single test
./build/tests/unit_tests --gtest_filter="DAGSchedulerPassTest.*"

# FileCheck tests
cd build && ctest -R FileCheck
```

## Architecture

```
hardware/*.def  -->  TableGen  -->  generated .inc files
                                        |
                                    gfxisa library
                                        |
                                  stinkytofu library  -->  tools / Python bindings
```

- `hardware/` - Per-architecture instruction definitions (`.def` files)
- `include/stinkytofu/` - Public headers
- `src/` - Library implementation (IR, passes, serialization, pipeline)
- `tools/` - `stinkytofu-opt`, `stinkytofu-check`, `intrinsic-compiler`
- `tests/` - Unit tests and FileCheck tests
- `python_module/` - Python bindings

## Documentation

See [docs/](docs/README.md) for detailed documentation including user guides, developer guides, and design documents.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
