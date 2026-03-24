# hipDNN Linux Scripts

Developer scripts for building, testing, and working with hipDNN on Linux.

## Scripts

### `rock_dev_bootstrap.sh`

Automates building hipDNN and its DNN provider plugins from source using
[TheRock](https://github.com/ROCm/TheRock) build infrastructure. Downloads
prebuilt CI artifacts for the full ROCm stack and lets you selectively rebuild
individual components from source.

**Prerequisites:** A TheRock clone, Python 3.9+, CMake, Ninja.

**Quick start:**

```bash
# Copy the script into your TheRock checkout, then:
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

Run `./rock_dev_bootstrap.sh --help` for full usage details including
available commands (`bootstrap`, `configure`, `build`, `rebuild`), GPU family
options, and component list.
