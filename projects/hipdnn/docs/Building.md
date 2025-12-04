# Building hipDNN

## Table of Contents
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Dependencies](#dependencies)
- [Quick Start Guide](#quick-start-guide)
- [Build Configurations](#build-configurations)
- [Build Targets](#build-targets)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Linux](#linux)
  - [Windows](#windows)
- [Troubleshooting](#troubleshooting)
- [Verifying Installation](#verifying-installation)

## Prerequisites

### System Requirements
- **GPU**: AMD GPU with ROCm support
- **Operating System**:
  - Linux: Any distribution supported by [TheRock](https://github.com/ROCm/TheRock), such as Ubuntu 24
  - Windows: Windows 11 (limited support, see [Windows section](#windows))

### Dependencies
> [!TIP]
> 💡 Prebuilt binaries and Docker files are available to provide a consistent development environment with all dependencies pre-installed. This is the recommended approach for most users. For more details about these Docker images, see the [Docker README](../dockerfiles/README.md). Dockerfile development environments are not available for Windows. Refer to the [Windows](#windows) section for details on building under Windows.

#### Required Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| ROCm | Matching TheRock (ROCm version 7.0+) | AMD GPU programming stack (see [TheRock releases](https://github.com/ROCm/TheRock/releases)) |
| CMake | 3.25.2+ | Build system generator |
| C++ Compiler | C++17 compatible | hipDNN requires C++17 compatible AMD Clang (plugins using device code may require C++20)|
| HIP | Matching TheRock | GPU programming interface (included with ROCm/TheRock) |
| clang-format | 18.x | Code formatting tool |
| clang-tidy | 20.x | Static analysis tool |
| LLVM Tools | 20.x | LLVM tools for code_coverage, and ASAN enabled builds |

#### Optional Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| Ninja | 1.12.1+ | Faster build system (recommended) |
| Docker | Latest | For containerized build environment |
| Python3 | Latest | For test name validation |

#### Third-Party Libraries
The following libraries are automatically managed by CMake (see [Dependencies.cmake](../cmake/Dependencies.cmake)):
- [FlatBuffers](https://github.com/google/flatbuffers) - Serialization library
- [Google Test](https://github.com/google/googletest) - Unit testing framework
- [spdlog](https://github.com/gabime/spdlog) - Logging library

## Quick Start Guide

Ensure the required dependencies are installed on your system as outlined in [Dependencies](#dependencies).

> [!TIP]
> 💡 See [Docker README](../dockerfiles/README.md) for details on using prebuilt binaries in Docker containers to ensure a consistent build environment.

Refer to the [Platform-Specific Instructions](#platform-specific-instructions) section for details on building under Windows.

#### 1. Clone the rocm-libraries Repository

As a faster alternative to cloning the entire git repository, you can do a fast sparse-checkout of just the hipDNN project.

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn
git checkout develop # or the branch you are starting from
```
Alternatively, a traditional `git clone` can also be used (though only the `projects/hipdnn` folder is needed):
```bash
git clone https://github.com/ROCm/rocm-libraries.git
```

#### 2. Build hipDNN
   ```bash
   cd rocm-libraries/projects/hipdnn
   mkdir build && cd build

   # Configure with Ninja (recommended)
   cmake -GNinja ..

   # Build and run all tests
   # Note that this may take several minutes to complete
   ninja check
   ```
   Refer to the [Build Targets](#build-targets) section below for additional build targets that can be used.

#### 3. Install hipDNN

   Refer to the [Build Configurations](#build-configurations) section below for details on using an install path other than the default `/opt/rocm`.
   ```bash
   sudo ninja install
   ```

## Build Configurations

### Release Build (Default)
```bash
cmake -GNinja ..
```

### Debug Build
```bash
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..
```

### Code Coverage Build
```bash
cmake -GNinja -DCODE_COVERAGE=ON ..
ninja code_coverage
# Coverage reports will be generated in build/hipdnn_code_coverage/
```

### Address Sanitizer Build
```bash
cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check
# Note: Some HIP-related tests may be skipped due to AddressSanitizer incompatibility
```

### Building Specific Components
```bash
# Build without plugins
cmake -GNinja -DHIP_DNN_BUILD_PLUGINS=OFF ..

# Build without frontend
cmake -GNinja -DHIP_DNN_BUILD_FRONTEND=OFF ..

# Build without backend
cmake -GNinja -DHIP_DNN_BUILD_BACKEND=OFF ..
```

### ROCM_PATH & CMAKE_INSTALL_PREFIX

- **`ROCM_PATH`**: Specifies where the build system looks for ROCm dependencies (default: `/opt/rocm` (Linux) / `C:/dist/therock` (Windows))
- **`CMAKE_INSTALL_PREFIX`**: Specifies where hipDNN will be installed (defaults to `ROCM_PATH`)

These variables can be set independently:

```bash
# Default: Both use /opt/rocm
cmake -GNinja ..

# Install hipDNN to custom location, find ROCm dependencies in /opt/rocm
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/custom/install/path ..

# Both custom (if ROCm is in a non-standard location)
cmake -GNinja -DROCM_PATH=/custom/rocm -DCMAKE_INSTALL_PREFIX=/another/path ..
```
## Build Targets

> [!NOTE]
> 📝 Make is supported for all targets. Configure with `cmake -G "Unix Makefiles" ..` if it is not the default generator in your environment. For parallel builds, use `make -j$(nproc)` on Linux. Unlike `ninja`, `make` does not build in parallel by default.

All targets support parallel builds with ninja.

| Target | Description |
|--------|-------------|
| \<no target\> | Build all components |
| `check` | Build and run all tests (see [Testing](./Testing.md)) |
| `unit-check` | Build and run exclusively the unit tests and API tests (minimal version of `check`) |
| `integration-check` | Build and run exclusively the E2E integration tests (this is the bulk of the testing time) |
| `install` | Install libraries and headers |
| `format` | Auto-format all C++ source files |
| `check_format` | Check code formatting compliance |
| `code_coverage` | Generate test coverage reports (requires `-DCODE_COVERAGE=ON`) |
| `clean` | Clean build artifacts |
| `validate_test_names` | Validates test names conform to naming rules |
| `generate_hipdnn_sdk_headers` | Generate C++ headers from schema (`.fbs`) files |

The following example build commands are equivalent (depending on which generator was used) and will build the `check` target to build and run all tests.

Using `cmake` to invoke build (regardless of which generator was used):
```bash
projects/hipdnn/build> cmake --build . --target check
```

If `Ninja` was used as the generator:
```bash
projects/hipdnn/build> ninja check
```

If a Makefile-type generator was used (not recommended):
```bash
projects/hipdnn/build> make check
```

## Platform-Specific Instructions

### Linux
The standard build instructions above work for all supported Linux distributions. Ensure ROCm is properly installed and configured for your distribution.

### Windows

Windows 10 and Windows 11 are supported. Windows 11 is recommended.

> [!WARNING]
> Some GPU functionality and HIP-related tests are not currently supported on Windows.

To do a standalone build of hipDNN, you will need to set up a number of pre-requisites.

> [!NOTE]
> The standalone build of hipDNN requires a subset of the full environment required for building TheRock. Refer to [TheRock Windows Support](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md) for a full Windows 11 build environment setup for TheRock (_but do not perform a build of TheRock_ as this is generally not necessary for building hipDNN standalone).

#### 1. Install Chocolatey Package Manager

Though dependencies can be installed _and configured_ manually, using [Chocolatey](https://community.chocolatey.org/) will streamline the environment setup. The Chocolatey client, `choco` is used in the instructions below.

#### 2. Install Utilities

The following third-party tools are needed for building hipDNN:
   - Git (installed with both git and unix tools available on the windows PATH)
   - Visual Studio 2022 with C++ workload (easy way to get Windows libraries)
   - CMake 3.25.2+
   - Ninja
   - Python 3

Using Chocolatey, install any of the missing required dependencies using an **⚠️Administrative Command Prompt (or PowerShell)**:

```bash
choco install visualstudio2022buildtools -y --params "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```
```bash
choco install git.install -y --params "'/GitAndUnixToolsOnPath'"
```
```bash
choco install cmake --version=3.31.0 -y
```
```bash
choco install ninja -y
```
```bash
choco install python -y
```

#### 3. Enable Windows 10 Long Paths

A detailed description and instructions for enabling long paths on Windows 10+ are available at https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later.

Abbreviated quotation:
>In the Windows API (with some exceptions discussed in the following paragraphs), the maximum length for a path is MAX_PATH, which is defined as 260 characters. A local path is structured in the following order: drive letter, colon, backslash, name components separated by backslashes, and a terminating null character. For example, the maximum path on drive D is `"D:\some 256-character path string<NUL>"` where `"<NUL>"` represents the invisible terminating null character for the current system codepage. (The characters `<` `>` are used here for visual clarity and cannot be part of a valid path string.)
>
>For example, you may hit this limitation if you are cloning a git repo that has long file names into a folder that itself has a long name.
>
>Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions.
>
>The registry value `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled` (Type: `REG_DWORD`) must exist and be set to `1`. The registry value will not be reloaded during the lifetime of the process. In order for all apps on the system to recognize the value, a reboot might be required because some processes may have started before the key was set.
>
> The following Administrative PowerShell command can be used to set this registry value:
>```PowerShell
>New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
>```

#### 4. Enable Windows 10 Symlinks

The instructions below are summarized from web content [here](https://portal.perforce.com/s/article/3472), [here](https://stackoverflow.com/questions/5917249/git-symbolic-links-in-windows/59761201#59761201), and [here](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links).


Verify ability to create symlinks. From a command window, run `mklink`:
```cmd
> echo "test" > mklinktest.txt
> mklink linkedfile.txt mklinktest.txt
symbolic link created for link.txt <<===>> ExistingFile.txt
```
If you do not have the ability to create symlinks you will see:
```cmd
> echo "test" > mklinktest.txt
> mklink linkedfile.txt mklinktest.txt
You do not have sufficient privilege to perform this operation.
```
If you do not have the ability to enable symlinks, the simplest way to enable this is to enable "[Developer Mode](https://www.wikihow.com/Enable-Developer-Mode-in-Windows-10)" in Windows 10/11.

**Windows 10**: Settings --> Update & Security --> For Developers --> Developer Mode --> toggle `On` --> confirm `Yes`.

**Windows 11**: Settings --> System --> For Developers --> Developer Mode --> toggle `On`.

Refer to the links at the beginning of this section for alternative methods to enable symlinks on your system.

You may need to restart your computer for the settings to take effect.

#### 5. Configure Git

With _long-paths and symlinks enabled_ as described in the above sections, enable symlink and long-path support in git:

```bash
git config --global core.symlinks true
git config --global core.longpaths true
```
Tip: you can use `git config --show-scope --show-origin core.symlinks` and `git config --show-scope --show-origin core.longpaths` to determine what the current active git configuration is and where that setting is configured.

#### 6. Install Clang Toolchain

Though TheRock toolchain is used to build hipDNN, utilities such as clang-format are currently provided by Clang.

Download and unzip a recent 20.x.x version of the Clang Toolchain: https://github.com/llvm/llvm-project/releases?q=20.

Unzip it to a path with no spaces. E.g. after being unzipped to `C:\dist\clang` the bin folder will be located at `C:\dist\clang\bin`.

#### 7. Install ROCm SDK

Run `amdgpu-arch.exe` from the clang release you just downloaded to find the GPU architecture you're using, and record the result (it’ll be something like gfx1103). E.g.
```cmd
> c:\dist\clang\bin\amdgpu-arch
gfx1103
```

You can use the table at https://github.com/ROCm/TheRock/blob/main/RELEASES.md#index-page-listing or the CMake source at https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L43 to determine the GFX Family. E.g. GPU architecture `gfx1103` is GFX Family `gfx110X-all`.

Download a recent nightly *Windows* build tarball of TheRock ROCm SDK for your GFX Family from https://therock-nightly-tarball.s3.amazonaws.com/index.html. E.g., for `gfx110X-all`, the most recent tarball available (at the time of this writing) is `therock-dist-windows-gfx110X-all-7.10.0a20251103.tar.gz`.

Complete instructions and alternate methods for installing TheRock are available at https://github.com/ROCm/TheRock/blob/main/RELEASES.md.

> [!NOTE]
> If a nightly tarball is not available for your GFX Family, you  may be able to [build TheRock from source](https://github.com/ROCm/TheRock/tree/main#building-from-source) or follow the [Roadmap for Support](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md) for more details.

Unzip the downloaded tarball to a path with no spaces. E.g. after unzipped to `C:\dist\therock` the bin folder will be located at `c:\dist\therock\bin`.

#### 8. Setup Environment Variables

* Add TheRock bin folder to the system PATH so that applications can find and load the DLL files. E.g.:
   ```cmd
   set PATH=C:\dist\therock\bin;%PATH%
   ```
   It isn't necessary to add the clang toolchain to your system PATH to perform the build as these can be specified using the `-D` option to cmake (see example below).
   * ROCM_PATH -- Used by the hipDNN CMake project to determine where TheRock was installed (defaults to `c:/dist/therock`).
   * CMAKE_PROGRAM_PATH -- Specifies the folder that CMake can use to find additional tools such as clang-format.

* Set the HIP_PLATFORM environment varilable:
   ```cmd
   set HIP_PLATFORM=amd
   ```

* If desired, set Ninja as the default generator so that `-g Ninja` doesn't need to be explicitly added to the `cmake` command line:
   ```cmd
   set CMAKE_GENERATOR=Ninja
   ```

Use `hipconfig` to check that TheRock is installed correctly and the PATH is setup correctly. The output from the command, as shown below, will show the detected ROCm path and ROCm clang path (`c:\dist\therock\` will be replaced with the folder TheRock was installed to on your system). This command requires that TheRock bin directory is in your system PATH.
```cmd
hipconfig -rocmpath -n --hipclangpath
c:\dist\therock
c:\dist\therock\lib\llvm\bin
```

Example CMake configure command (including `ROCM_PATH` for completeness even though it is not required when using the default value):
```
projects\hipdnn\build>set CMAKE_GENERATOR=Ninja
projects\hipdnn\build>cmake -DGPU_TARGETS=gfx1103 -DROCM_PATH=c:/dist/therock -DCMAKE_PROGRAM_PATH=c:/dist/clang/bin ..
-- Building for: Ninja
-- Using ROCm Clang compilers from c:/dist/therock/lib/llvm/bin
```

See the note on setting `GPU_TARGETS` in the following section.

If using custom paths, you may need to modify [ClangToolChain.cmake](../cmake/ClangToolChain.cmake).

#### 9. Clone Repository and Build hipDNN

From here, follow the instructions in the [Quick Start Guide](#quick-start-guide) section to clone the repository and build hipDNN, **keeping in mind the following notes**:
* Do **NOT** open the "x64 Native Tools Command Prompt for VS 2022" as this will interfere with the ROCm SDK and Clang toolchain.
* When generating the project, be sure to set GPU_TARGETS to your GPU as auto-detection is not currently supported on Windows, e.g. `cmake -DGPU_TARGETS=gfx1103 ..` (replacing gfx1103 with your GPU)
* When generating the project, CMake will warn about a clang-format or clang-tidy mismatch. That’s okay for now.
* Generating the project files may take longer than on Linux, but should complete within a few minutes.
* You may want to limit the number of threads used by Ninja when building hipDNN so that your computer is not bogged-down by the build. You can use the `ninja -j` option to set the number of threads to something smaller than the number of threads available on your CPU.

## Troubleshooting

### Common Build Issues

* **Out of memory during build**
   ```bash
   # Reduce parallel jobs
   ninja -j4  # or even -j2 for systems with limited RAM
   ```

* **Docker GPU access issues**
   - Ensure ROCm is installed on the host system
   - Verify GPU is visible: `rocm-smi` or `rocminfo`
   - Check user is in `video` and `render` groups:
     ```bash
     sudo usermod -a -G video,render $USER
     # Log out and back in for changes to take effect
     ```

## Verifying Installation

See [samples README](../samples/README.md) for detailed instructions on building test sample programs using hipDNN.
