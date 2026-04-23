# rock_dev_bootstrap.py — Windows Setup Guide

This document covers the issues encountered when bootstrapping a TheRock-based
build of hipDNN and miopen-provider on Windows (gfx1151), and their solutions.

## Prerequisites

Before running the script, ensure you have:

- **Python 3.9+** on PATH
- **CMake < 4.0** (CMake 4 is not yet supported by TheRock on Windows)
- **Ninja** on PATH
- **Visual Studio 2022 Build Tools** (MSVC 19.42+ / VS 17.12+)
- **Git** configured with symlinks and long paths:
  ```
  git config --global core.symlinks true
  git config --global core.longpaths true
  ```
- **Long path support** enabled in Windows:
  ```powershell
  reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
  ```
- **GitHub CLI (`gh`)** installed and authenticated (see Issue #2 below)
- A **TheRock checkout** (e.g. `D:\develop\claude_workspace\repos\TheRock`)

## Quick Start (known working)

```bash
# 1. Bootstrap with a known run ID (skip artifact auto-detection)
python rock_dev_bootstrap.py bootstrap \
    --therock-dir D:/develop/claude_workspace/repos/TheRock \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 \
    24170213928

# 2. Configure components for source build
python rock_dev_bootstrap.py configure \
    --therock-dir D:/develop/claude_workspace/repos/TheRock \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 \
    hipdnn miopenprovider

# 3. Build
python rock_dev_bootstrap.py build \
    --build-dir D:/develop/claude_workspace/build/therock-gfx1151 \
    -j24
```

## Issues and Solutions

### Issue #1: DVC not found — `fetch_sources.py` fails on Windows

**Symptom:** `fetch_sources.py` exits with code 1 and prints:
```
Could not find `dvc` on PATH so large files could not be fetched
```

**Root cause:** On Windows, `fetch_sources.py` treats missing DVC as fatal
(unlike Linux where it's a warning). DVC is used to pull large binary files
for `rocm-libraries` and `rocm-systems`.

**Solution:** The script detects missing DVC and passes `--dvc-projects _skip`
to `fetch_sources.py`, bypassing the check. A warning is printed. DVC-managed
files are not needed when using prebuilt CI artifacts.

**If you need DVC:** Install from https://dvc.org/doc/install (version 3.62.0+).

---

### Issue #2: GitHub API rate limiting during artifact auto-detection

**Symptom:** `find_latest_artifacts.py` prints repeated warnings and then fails:
```
Warning: No GitHub auth available, requests may be rate limited
Error: GitHub API rate limit exceeded
```

**Root cause:** Unauthenticated GitHub API requests are limited to 60/hour.
The artifact search checks one API call per commit and exhausts the limit
before finding artifacts.

**Solution:** Install the GitHub CLI and authenticate:
```bash
# Install gh (via chocolatey, winget, or https://cli.github.com)
choco install gh

# Authenticate
gh auth login
```

TheRock's `find_latest_artifacts.py` will automatically detect `gh` and use
it for authenticated API requests (5000/hour).

---

### Issue #3: SAML/SSO enforcement for ROCm organization

**Symptom:** After authenticating `gh`, requests fail with:
```
Error: gh api request failed: gh: Resource protected by organization SAML enforcement.
You must grant your Personal Access token access to this organization.
```

**Root cause:** The ROCm GitHub organization requires SSO authorization for
personal access tokens.

**Solution:**
1. Go to GitHub Settings > Personal Access Tokens
2. Find your token
3. Click "Configure SSO"
4. Authorize the `ROCm` organization

---

### Issue #4: Artifact auto-detection finds no artifacts for gfx1151

**Symptom:** `find_latest_artifacts.py` checks 50 commits and finds nothing:
```
Searching 50 commits on ROCm/TheRock/main for 1 group(s): gfx1151...
  [1/50] Checking abc123...
    No artifacts found
  ...
No artifacts found in last 50 commits
```

**Root cause:** The tool checks for an S3 index file named
`index-gfx1151.html`, but the CI now uploads a single `index.html` (without
the per-group suffix). This is a naming convention mismatch between
`find_latest_artifacts.py` and the current CI upload scripts.

**Workaround:** Pass the run ID directly instead of relying on auto-detection:
```bash
python rock_dev_bootstrap.py bootstrap --therock-dir /path/to/TheRock 24170213928
```

**Finding a run ID:** Browse the TheRock CI Nightly workflow runs at:
https://github.com/ROCm/TheRock/actions/workflows/ci_nightly.yml

Pick a recent successful run — the number in the URL is the run ID.

You can also verify artifacts exist for a run by checking S3:
```
https://therock-ci-artifacts.s3.amazonaws.com/<run-id>-windows/index.html
```

---

### Issue #5: CMake cannot find MSVC compiler (cl.exe not on PATH)

**Symptom:** CMake configure fails with:
```
No CMAKE_C_COMPILER could be found.
No CMAKE_CXX_COMPILER could be found.
```

**Root cause:** MSVC requires environment activation via `vcvars64.bat` to
place `cl.exe` on PATH and set up include/lib paths. The script runs from
Git Bash where these variables are not set.

**Solution:** The script auto-detects and activates the MSVC environment by:
1. Searching `D:\develop` and `C:\Program Files\Microsoft Visual Studio` for
   `vcvars64.bat`
2. Running it in a temp `.bat` wrapper via `cmd.exe /c`
3. Capturing the resulting environment variables
4. Passing them to CMake/ninja subprocess calls

If `cl.exe` is already on PATH (e.g. running from a Developer Command Prompt),
this step is skipped.

**If auto-detection fails:** Run the script from an "x64 Native Tools Command
Prompt for VS 2022" instead.

---

### Issue #6: vcvarsall.bat activation fails from Python subprocess

**Symptom:** The script finds `vcvarsall.bat` but reports:
```
WARNING: vcvarsall.bat failed (exit 1)
```

**Root cause:** Passing `vcvarsall.bat x64` through `cmd.exe /c` with inline
quotes and redirects causes path escaping issues in Python's `subprocess.run`.

**Solution:** Changed to use `vcvars64.bat` (no arguments needed, equivalent
to `vcvarsall.bat x64`) and write a temporary `.bat` file to avoid all quoting
issues. The temp file runs `vcvars64.bat`, then dumps the environment with
`set`.

---

## Notes for Repeatability

### Workflow for the default `--workflow` flag

The script defaults to `ci_windows.yml` on Windows, but as noted in Issue #4,
artifact auto-detection is currently broken for gfx1151 due to the S3 index
naming change. Until this is fixed upstream in TheRock, always pass a run ID
for the bootstrap step.

### Re-running bootstrap

Running bootstrap again after a partial failure is safe. It will:
- Skip `fetch_sources.py` if submodule sources already exist
- Re-download and re-extract artifacts (overwrites existing)

### Re-running configure after a failure

If configure fails partway through, the `.prebuilt` markers may already be
removed. Re-running configure will report "no .prebuilt marker (will build
from source)" — this is expected and harmless.

### Iterating on code changes

After the initial setup, you only need the `build` command:
```bash
python rock_dev_bootstrap.py build --build-dir D:/develop/claude_workspace/build/therock-gfx1151 -j24
```

For a clean rebuild of a component:
```bash
python rock_dev_bootstrap.py rebuild --build-dir D:/develop/claude_workspace/build/therock-gfx1151 hipdnn
```

### Build artifacts location

- **hipDNN binaries:** `build/therock-gfx1151/ml-libs/hipDNN/build/bin/`
- **miopenprovider plugin:** `build/therock-gfx1151/ml-libs/miopenprovider/build/bin/hipdnn_plugins/engines/miopen_plugin.dll`
- **hipDNN staged install:** `build/therock-gfx1151/ml-libs/hipDNN/stage/`
- **miopenprovider staged install:** `build/therock-gfx1151/ml-libs/miopenprovider/stage/`

### Running tests

```bash
# hipDNN unit tests
build/therock-gfx1151/ml-libs/hipDNN/build/bin/hipdnn_backend_tests.exe

# miopen-provider unit tests
build/therock-gfx1151/ml-libs/miopenprovider/build/bin/miopen_plugin_tests.exe

# Filter specific tests
build/therock-gfx1151/ml-libs/hipDNN/build/bin/hipdnn_backend_tests.exe --gtest_filter="TestBackendDescriptor.*"
```
