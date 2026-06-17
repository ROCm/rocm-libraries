# rock_dev_bootstrap.py — Windows Setup Guide

## Prerequisites

Before running the script, ensure you have:

- **Python 3.9+** on PATH
- **CMake < 4.0** (CMake 4 is not yet supported by TheRock on Windows)
- **Ninja** on PATH
- **Visual Studio 2022 Build Tools** (MSVC 19.42+ / VS 17.12+) **with Windows 11 SDK**
  - The Windows SDK (`Microsoft.VisualStudio.Component.Windows11SDK.22621`) must be
    installed — it provides `rc.exe`, `mt.exe`, and system libraries (`kernel32.lib`, etc.)
  - Run `scripts/windows/windows_build_setup.ps1` to install everything via winget
- **Git** configured with symlinks and long paths:
  ```
  git config --global core.symlinks true
  git config --global core.longpaths true
  ```
- **Long path support** enabled in Windows:
  ```powershell
  reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
  ```
- **GitHub CLI (`gh`)** installed and authenticated (for artifact auto-detection)
- A **TheRock checkout** (e.g. `D:\develop\claude_workspace\repos\TheRock`)

## Quick Start

```bash
# 1. Bootstrap (auto-detects latest nightly):
python rock_dev_bootstrap.py bootstrap \
    --therock-dir D:/develop/claude_workspace/repos/TheRock \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151

# 2. Configure components for source build:
python rock_dev_bootstrap.py configure \
    --therock-dir D:/develop/claude_workspace/repos/TheRock \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 \
    hipdnn miopenprovider

# 3. Build:
python rock_dev_bootstrap.py build \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 \
    -j24
```

You can also pass a specific run ID to bootstrap if needed:
```bash
python rock_dev_bootstrap.py bootstrap --therock-dir /path/to/TheRock 24814526637
```

Find run IDs at: https://github.com/ROCm/TheRock/actions/workflows/ci_nightly.yml

## Iterating on Code Changes

After the initial setup, use the `build` command for incremental builds:
```bash
python rock_dev_bootstrap.py build \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 -j24
```

For a clean rebuild of a component:
```bash
python rock_dev_bootstrap.py rebuild \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 hipdnn
```
