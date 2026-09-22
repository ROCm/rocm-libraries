# TensileLite Reference

Supplementary reference for `AGENTS.md` — load this when you need test commands, custom builds, linting, CMake options, or supported targets.

## Running Tests

```bash
# Full test suite (builds client + runs all common tests)
tox -e py3 -- tensilelite/Tests -m common


# Python unit tests only (builds and configures the client in the tox environment)
tox -e unit -- tensilelite/Tests/unit

# Python unit tests only (builds and configures the client in the tox environment)
tox -e unit -- tensilelite/Tests/unit

# Run a specific test category
tox -e py3 -- tensilelite/Tests -m gemm

# Run a single test directly (after a prior `invoke build-client`)
python -m tensilelite run tensilelite/Tests/common/exception/<test>.yaml tensile-out
```

## Custom CMake Build

```bash

# Client-only builds use the selected SDK and do not create a private venv.
cmake --preset tensilelite -S .. -B my-custom-build
cmake --build my-custom-build --parallel

# Run test with custom client path
python -m tensilelite_configure_client \
    --client "$PWD/my-custom-build/tensilelite/client/tensilelite-client"
ROCM_PATH=/opt/rocm python -m tensilelite run \
    tensilelite/Tests/common/<test>.yaml tensile-out

# Build with custom args (e.g., Debug + specific GPU)
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- tensilelite/Tests -m common
```

Iterate on rocisa C++ without re-pip-installing:

```bash
cd rocisa && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
make -j8
```

`invoke build-client` accepts `--clean`, `--build-dir`, `--build-type`, `--gpu-targets`, `--rocm-path`, `--export-compile-commands`, `--enable-rocprof`, and `--cxx-flags-release`. See `tasks.py`.

## Linting and Formatting

```bash
tox -e lint          # flake8 (pyflakes errors only, E/W ignored)
tox -e format        # black (line-length=100) on Common/, tensilelite_create_library/, Utilities/Decorators/
tox -e isort         # isort (black profile) on same directories
```

## Rebuilding Assembly Without Full Rerun

After a TensileLite run creates `tensile-out/`, you can edit assembly and rebuild only object code:

```bash
make co TENSILE_OUT=tensile-out                          # auto-detect arch
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64   # gfx9 explicit
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32  # gfx11 explicit
```

## CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `TENSILELITE_ENABLE_HOST` | ON | Build C++ runtime library |
| `TENSILELITE_ENABLE_CLIENT` | ON | Build benchmark client |
| `TENSILELITE_BUILD_TESTING` | OFF | Build C++ host library tests |
| `ROCISA_BUILD_PYTHON` | OFF | Build only the in-tree rocisa extension without device libraries |
| `GPU_TARGETS` | (detected) | Semicolon-separated list of gfx targets |

## Supported Targets

GPU architectures (see `tensilelite/Common/Architectures.py`): gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1010/1011/1012, gfx1030, gfx1100/1101/1102, gfx1200/1201, gfx1250 (each with optional `:xnack+/-`).

Test markers for architectures (see `pytest.ini`): `gfx11`, `gfx12`, `gfx94x`, `gfx950`, `gfx1250`, plus per-arch `xfail-gfxNNN` / `skip-gfxNNN`. Data type markers: `Float`, `Double`, `Half`, `BFloat16`, `Int8`, `Float8`/`BFloat8` (OCP and `_fnuz` NANOO variants), mixed `Float8BFloat8`, `Float4`, `Float6`, `BFloat6`.
