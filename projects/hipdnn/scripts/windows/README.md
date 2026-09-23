# Windows Scripts

Scripts for setting up hipDNN development environments on Windows.

## Scripts

### `windows_build_setup.ps1`

Full environment setup script that installs prerequisites (Visual Studio Build
Tools, Git, CMake, Ninja, Python), downloads Clang and TheRock toolchains, and
configures system environment variables. Requires Administrator privileges.

See the script header for parameter details.

### `wheel_build_setup.ps1`

Lightweight setup using ROCm Python wheels. Creates a virtual environment,
installs ROCm SDK wheels, and prints the CMake variables needed to build hipDNN.
Does **not** require Administrator privileges.

#### Prerequisites

- Python 3.x on PATH
- A Clang toolchain already installed (see `windows_build_setup.ps1` or download
  manually from the [LLVM releases](https://github.com/llvm/llvm-project/releases))

#### Usage

Install from ROCm nightlies (default):

```powershell
.\wheel_build_setup.ps1
```

Install from S3 staging with a specific build SHA:

```powershell
.\wheel_build_setup.ps1 -SHA "abc123def"
```

Override default paths and GPU target:

```powershell
.\wheel_build_setup.ps1 -VenvPath "C:\my_venv" -ClangPath "C:\clang\bin" -GpuTarget "gfx1100"
```

#### Parameters

| Parameter    | Default                        | Description                                      |
|--------------|--------------------------------|--------------------------------------------------|
| `-SHA`       | *(empty)*                      | Build SHA for S3 staging. Omit to use nightlies. |
| `-VenvPath`  | `D:\develop\latest_wheels`     | Where to create the Python virtual environment.  |
| `-ClangPath` | `D:\develop\dist\clang\bin`    | Path to the Clang toolchain bin directory.        |
| `-GpuTarget` | `gfx1151`                      | GPU architecture for wheel selection and CMake output. |

For the full built-in help, run:

```powershell
Get-Help .\wheel_build_setup.ps1 -Detailed
```
