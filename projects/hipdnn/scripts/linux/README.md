# TheRock Developer Bootstrap Scripts (Linux)

Scripts for building hipDNN and its DNN provider plugins from source using
[TheRock](https://github.com/ROCm/TheRock) build infrastructure. TheRock
downloads prebuilt CI artifacts for the full ROCm stack and lets you
selectively rebuild individual components from source.

## Prerequisites

- A clone of [TheRock](https://github.com/ROCm/TheRock)
- Python 3.9+
- CMake and Ninja
- `pip install meson` (or install it in TheRock's `.venv` — see below)

## Quick Start

Copy `rock_dev_bootstrap.sh` into your TheRock checkout, then:

```bash
cd /path/to/TheRock

# 1. Download prebuilt CI artifacts (one-time setup)
./rock_dev_bootstrap.sh bootstrap --gpu gfx90a

# 2. Activate the Python venv (meson must be on PATH for cmake)
source .venv/bin/activate

# 3. Configure hipDNN + providers for source build
./rock_dev_bootstrap.sh configure --gpu gfx90a hipdnn miopenprovider hipkernelprovider hipblasltprovider

# 4. Build
./rock_dev_bootstrap.sh build --gpu gfx90a
```

## Commands

| Command | Description |
|---------|-------------|
| `bootstrap [run-id]` | Download prebuilt CI artifacts. Auto-detects the latest nightly if no run ID is given. |
| `configure [components...]` | Remove `.prebuilt` markers for the listed components and run cmake. Defaults to all components if none specified. |
| `build [components...]` | Build configured components with ninja. |
| `rebuild [components...]` | Expunge (full clean) then rebuild components. |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu <family>` | `gfx94X-dcgpu` | GPU architecture family (e.g. `gfx90a`, `gfx94X-dcgpu`). |
| `--build-dir <dir>` | `~/therock-build-<gpu>` | Build output directory. |
| `--workflow <file>` | `ci_nightly.yml` | GitHub Actions workflow to search for artifacts. |

## Available Components

| Component | Description |
|-----------|-------------|
| `hipdnn` | Core hipDNN library |
| `miopenprovider` | MIOpen backend DNN provider plugin |
| `hipblasltprovider` | hipBLASLt backend DNN provider plugin |
| `hipkernelprovider` | HIP kernel backend DNN provider plugin |

## Notes

- **Python venv:** The `bootstrap` command creates a `.venv` in the TheRock
  repo and installs dependencies (including `meson`). You must activate this
  venv before running `configure` so that cmake can find `meson` on PATH.
- **GitHub auth:** Without `gh auth` or a `GITHUB_TOKEN`, artifact search
  requests are unauthenticated and may be rate-limited. Set up authentication
  for faster, more reliable bootstrapping.
- **Build directory:** Each GPU family gets its own build directory
  (`~/therock-build-gfx90a`, `~/therock-build-gfx94X`, etc.). You can
  override this with `--build-dir`.
- **Incremental builds:** After the initial build, re-running `build` only
  recompiles changed files. Use `rebuild` for a full clean rebuild.
