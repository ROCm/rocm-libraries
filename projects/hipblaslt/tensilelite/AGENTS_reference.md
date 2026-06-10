# TensileLite Reference

Supplementary reference for `AGENTS.md` — load this when you need test commands, custom builds, linting, CMake options, or supported targets.

## Running Tests

```bash
# Full test suite (builds client + runs all common tests)
tox -e py3 -- Tensile/Tests -m common

# Python unit tests only (skips the long client build; requires a prior build)
tox -e unit -- Tensile/Tests/unit

# Run a specific test category
tox -e py3 -- Tensile/Tests -m gemm

# Run a single test directly (after a prior `invoke build-client`)
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

## Custom CMake Build

```bash
cmake --preset tensilelite -S .. -B my-custom-build
cmake --build my-custom-build --parallel

# Run test with custom client path
./my-custom-build/Tensile.sh Tensile/Tests/common/<test>.yaml tensile-out \
    --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client

# Build with custom args (e.g., Debug + specific GPU)
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common
```

Iterate on rocisa C++ without re-pip-installing:

```bash
cd rocisa && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
make -j8
```

`invoke build-client` accepts `--clean`, `--build-dir`, `--build-type`, `--gpu-targets`, `--rocm-path`, `--export-compile-commands`, `--bundle-python-deps`, `--enable-rocprof`. See `tasks.py`.

## Linting and Formatting

```bash
tox -e lint          # flake8 (pyflakes errors only, E/W ignored)
tox -e format        # black (line-length=100) on Common/, TensileCreateLibrary/, Utilities/Decorators/
tox -e isort         # isort (black profile) on same directories
```

## Rebuilding Assembly Without Full Rerun

After a Tensile run creates `tensile-out/`, you can edit assembly and rebuild only object code:

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
| `TENSILELITE_ENABLE_AUTOBUILD` | OFF | Auto-rebuild rocisa wrapper scripts |
| `TENSILELITE_BUILD_TESTING` | OFF | Build C++ host library tests |
| `GPU_TARGETS` | (detected) | Semicolon-separated list of gfx targets |

## Supported Targets

GPU architectures (see `Tensile/Common/Architectures.py`): gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1010/1011/1012, gfx1030, gfx1100/1101/1102, gfx1200/1201, gfx1250 (each with optional `:xnack+/-`).

Test markers for architectures (see `pytest.ini`): `gfx11`, `gfx12`, `gfx94x`, `gfx950`, `gfx1250`, plus per-arch `xfail-gfxNNN` / `skip-gfxNNN`. Data type markers: `Float`, `Double`, `Half`, `BFloat16`, `Int8`, `Float8`/`BFloat8` (OCP and `_fnuz` NANOO variants), mixed `Float8BFloat8`, `Float4`, `Float6`, `BFloat6`.
