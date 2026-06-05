# hipTensor

> [!NOTE]
> The published documentation is available at [hipTensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `projects/hiptensor/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

Welcome! hiptensor is AMD's C++ library for accelerating tensor primitives using GPU matrix cores on AMD's latest discrete GPUs.

## Requirements

hipTensor currently supports the following AMDGPU architectures:

* CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx942, gfx950 as 'gfx9'
* RDNA class GPU featuring matrix core support: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153, gfx1200, gfx1201 and gfx1250.

> [!NOTE]
> Double precision FP64 datatype support requires gfx90a, gfx942, gfx950 or gfx1250.

Dependencies:

* Minimum ROCm version support is 7.0 (7.13 for Windows).
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum Composable Kernel version support is composable_kernel 1.2.0 for ROCm 7.13 (or ROCm package composablekernel-dev).
* Minimum HIP runtime version support is 4.3.0 (or ROCm package ROCm hip-runtime-amd).
* Minimum LLVM dev package version support is 7.0 (available as ROCm package rocm-llvm-dev).

Optional:

* doxygen (for building documentation)

### Building Composable Kernel

In case the Composable Kernel library isn't included in your ROCm installation, please refer to the
[Composable Kernel installation guide](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/install/Composable-Kernel-install.html)
in order to build and install the library.

> [!TIP]
> When building Composable Kernel, add `-DHIPTENSOR_BUILD_TESTS=ON` to the cmake configure command in order to only build the
> targets required by hipTensor and speed-up the build time.

## Build with CMake

For more detailed information, please refer to the [hipTensor installation guide](https://rocm.docs.amd.com/projects/hipTensor/en/latest/install/installation.html).

### Project options

| Option                              | Description                                                              | Default Value                                           |
|-------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------|
| GPU_TARGETS                         | Build code for specific GPU target(s)                                    | gfx908;gfx90a;gfx942;gfx950;gfx11-generic;gfx12-generic |
| HIPTENSOR_BUILD_TESTS               | Build the tests                                                          | ON                                                      |
| HIPTENSOR_BUILD_SAMPLES             | Build the samples                                                        | ON                                                      |
| HIPTENSOR_BUILD_COMPRESSED_DBG      | Enable compressed debug symbols                                          | ON                                                      |
| HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR | Set the hipTensor default data layout to column major                    | ON                                                      |
| HIPTENSOR_INLINE_UNARY_OPS          | Inline all unary ops for best runtime performance (slower compilation)   | OFF                                                     |
| CREATE_TEST_APP_LOCAL_DEPLOY        | Copy ROCm runtime DLLs next to test binaries so they take precedence over System32 (Windows only) | OFF                            |

### Building on Linux

By default, the project is configured as Release mode. Here are some example configurations:

| Configuration                    | Command                                                                                                     |
|----------------------------------|-------------------------------------------------------------------------------------------------------------|
| Basic                            | `CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> .`                              |
| Targeting gfx908                 | `CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DGPU_TARGETS=gfx908`         |
| Debug build                      | `CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug`     |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`.

Finally, install the built binaries with `cmake --install .`.

#### Docker

Dockerfiles are available for Ubuntu 24.04 with prebuilt or source-built ROCm (using TheRock). See [docker/README.md](docker/README.md) for instructions.

### Building on Windows

#### Prerequisites

- **Visual Studio 2026** (VS 18) or **Visual Studio 2022** — open a **"Command Prompt for VS 18"** (or "Command Prompt for VS 2022") terminal for all commands below.
- **CMake 4.2.3+** — the version bundled with Visual Studio 2026 (msvc3) is recommended.
- **vcpkg** — bundled with the VS command prompt. If `vcpkg --version` or `echo %VCPKG_ROOT%` returns nothing, install it following the [vcpkg getting-started guide](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started).
- **ROCm** — version 7.13 or later.

#### Install ROCm (TheRock)

1. If not already installed, [install ROCm from TheRock](https://github.com/ROCm/TheRock#installing-from-releases) and set the installation directory as a variable so you can reuse it in subsequent steps:
   ```bat
   set ROCM_PATH=C:\dist\TheRock
   ```
2. If you choose to [install from prebuilt tarball](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-from-tarballs), create the directory:
   ```bat
   mkdir %ROCM_PATH%
   ```
   Download and extract the tarball to `%ROCM_PATH%`.
3. Set the required environment variables:
   ```bat
   set HIP_PATH=%ROCM_PATH%
   set HIP_DEVICE_LIB_PATH=%ROCM_PATH%\lib\llvm\amdgcn\bitcode
   set HIP_PLATFORM=amd
   ```

#### Configure and build hipTensor

Change to the hipTensor source code directory, create a build directory and run the CMake configure command (e.g. Targeting gfx11-generic).

```bat
cd hiptensor
mkdir build
cd build

cmake -G Ninja ^
  -DCMAKE_INSTALL_PREFIX=%ROCM_PATH% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang++.exe" ^
  -DCMAKE_C_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang.exe" ^
  -DCMAKE_PREFIX_PATH="%ROCM_PATH%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
  -DGPU_TARGETS=gfx11-generic ^
  -DHIPTENSOR_BUILD_TESTS=ON ^
  -B. ..
```

> [!NOTE]
> If your system has a different version of ROCm installed alongside the build toolchain (for example, a system ROCm in `System32` and a development build under `%ROCM_PATH%`), add `-DCREATE_TEST_APP_LOCAL_DEPLOY=ON` to the CMake command. This copies the required ROCm runtime DLLs (`amdhip64`, `amd_comgr`, `rocm_kpack`, etc.) from `%ROCM_PATH%\bin` next to the test binaries at configure time, ensuring the correct runtime is loaded instead of the one found in `System32`.

Then build and install:

```bat
cmake --build . -- -j%NUMBER_OF_PROCESSORS%
cmake --install .
```

## Documentation

For more comprehensive documentation on installation, samples and test contents, API reference and programmer's guide you can build the documentation locally.

<details open>
<summary>Standard</summary>

```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

</details>

<details>
<summary>Ubuntu 24.04+</summary>

Ubuntu 24.04 and later enforce [PEP 668](https://peps.python.org/pep-0668/), which prevents system-wide pip installs. Use a virtual environment instead:

```bash
cd docs
python3 -m venv .venv
.venv/bin/pip install -r sphinx/requirements.txt
.venv/bin/python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

</details>

The HTML documentation can be viewed in your browser by opening the `docs/_build/html/index.html` result.

The latest official documentation for hipTensor is available at:
[https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## Contributing to the hipTensor Library

Community collaboration is encouraged! If you are considering contributing, please follow the [hipTensor Contribution Guide](https://rocm.docs.amd.com/projects/hipTensor/en/latest/contribution/contributors-guide.html) to get started.
