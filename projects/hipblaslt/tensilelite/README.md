# Tensilelite

## Building and Running Tests

While full test suites can be run with a single `tox` command, developers may wish to
build the hipBLASLt tensilelite client executable (`tensilelite-client`) and run individual tests separately.
This is useful for debugging specific problems or isolating issues in a specific test.

### Run Test Suite with Tox

The standard workflow for running the entire test suite is to use `tox`. This command will build
`tensilelite-client` and execute all tests.

```
cd rocm-libraries/projects/hipblaslt/tensilelite
tox -e py3 -- Tensile/Tests -m common
```

Subsequently, you can run just the Tensile unit tests via:

```
tox -e unit -- Tensile/Tests/unit
```

### Generate coverage report with Tox

```
cd rocm-libraries/projects/hipblaslt/tensilelite
tox -e coverage
```

This will:
- Run all unit tests with coverage
- Run all common tests with coverage
- Generate HTML, XML, and JSON reports
- Display a summary in the terminal

```
cd rocm-libraries/projects/hipblaslt/tensilelite
tox -e coverage-unit
```

Runs only Python unit tests.

### Pre-commit hook (affected tests)

An opt-in git pre-commit hook runs the unit + characterization tests affected by
your staged TensileLite changes and blocks the commit on real failures (it falls
back to the full unit + characterization suite when it cannot narrow the set). It
runs `uv run pytest`, which builds rocisa (a HIP native extension), so install and
commit from inside a ROCm dev container (HIP at `/opt/rocm`, a Python with dev
headers). Mount the repo at the same absolute path inside the container as on the
host — git worktrees use an absolute gitdir pointer, so a different mount breaks
git.

```
cd rocm-libraries/projects/hipblaslt/tensilelite
uv sync                          # provisions deps, rocisa, and the pre-commit app
uv run invoke precommit-install  # writes the git hook (once per clone)
```

`git commit` from inside the container then runs the affected tests. Bypass once
with `git commit --no-verify`. On a snapshot mismatch the hook prints a
`--snapshot-update` command scoped to the failing file(s).

### Build client with invoke and Run a Test (Default Path)

This workflow uses `invoke` to build the C++ client into the default `build_tmp` directory.
Tensile will search for `tensilelite-client` in `tensilelite/build_tmp` if `--prebuilt-client`
is not specified.

```
cd rocm-libraries/projects/hipblaslt/tensilelite

# install invoke if you haven't already
pip3 install invoke

# install rocisa as an editable package (once after cloning, or after pyproject.toml changes)
invoke rocisa

# build the C++ client to the default location
invoke build-client

# override the default toolchain with a specific ROCm install
invoke build-client \
  --gpu-targets gfx950 \
  --rocm-path /opt/rocm-7.3.0 \
  --export-compile-commands

# run an individual test directly — no wrapper script needed
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

### Rebuilding after C++ changes

`invoke build-client` builds the tensilelite-client executable only — it does
**not** rebuild the rocisa Python module (`_rocisa.so`).  If you edit rocisa or
stinkytofu C++ sources you must re-run `invoke rocisa` for those changes to
take effect in Python:

| What you changed | Command to rebuild |
|---|---|
| rocisa C++ sources | `invoke rocisa` |
| stinkytofu C++ sources | `invoke rocisa` |
| tensilelite-client C++ sources | `invoke build-client` |
| rocisa `pyproject.toml` or `CMakeLists.txt` | `invoke rocisa` |

Example workflow after editing stinkytofu or rocisa code:

```bash
# 1. Rebuild the rocisa Python module (includes stinkytofu)
invoke rocisa

# 2. Rebuild the C++ client (if needed)
invoke build-client
```

If you forget to rebuild, importing rocisa will raise an `ImportError` listing
the stale source files:

```
ImportError: rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.
  Modified: .../shared/stinkytofu/src/ir/asm/Function.cpp
  Rebuild:  cmake --build <build_dir> --target _rocisa
```

**3. Build with CMake (Custom Location) and Run Test with Path Flag**

This workflow is for when you need to build the client in a location other than the default
`build_tmp` directory. The `--prebuilt-client` flag is then used to specify this custom path when
running a test. Be sure to pass the root directory of the hipblaslt project when configuring.

```
cd rocm-libraries/projects/hipblaslt/tensilelite

# install rocisa (once after cloning)
invoke rocisa

# configure in a custom directory (e.g., my-custom-build)
cmake --preset tensilelite -S .. -B my-custom-build

# build
cmake --build my-custom-build --parallel

# run a test directly
Tensile/bin/Tensile Tensile/Tests/pre_checkin/<test>.yaml tensile-out \
                           --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client
```

**4. Build with tox (Custom Build Args)**

This workflow uses `tox` with custom CMake arguments, which is useful for creating
specialized builds (e.g., Debug builds) and setting the architecture.

```
# build the client using tox with custom CMake flags
cd rocm-libraries/projects/hipblaslt/tensilelite
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common

# run tests with a single pytest worker (useful for debugging)
TENSILE_NUM_PYTEST_WORKERS=1 tox -e py3 -- Tensile/Tests -m common
```

`invoke build-client` follows the existing `tensilelite` CMake preset by default.
In this repo, that means `/opt/rocm` compiler settings come from the preset, and
`CMAKE_EXPORT_COMPILE_COMMANDS` and `HIPBLASLT_BUNDLE_PYTHON_DEPS` are already enabled
by default.

Use these flags when you want to override or make that behavior explicit:

* `--rocm-path <path>`: Override the compiler toolchain to use `<path>/bin/amdclang` and `<path>/bin/amdclang++`
* `--export-compile-commands`: Explicitly force `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
* `--bundle-python-deps`: Explicitly force `HIPBLASLT_BUNDLE_PYTHON_DEPS=ON`
* `--enable-rocprof`: Sets `TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=ON`

### Speeding Up Builds with ccache

Install ccache to cache compiled objects across rebuilds:

```bash
sudo apt install ccache    # Ubuntu/Debian
```

`invoke rocisa` and `invoke build-client` will detect ccache automatically
and use it as the compiler launcher. No additional configuration is needed.

### Environment Variables

* `TENSILE_NUM_PYTEST_WORKERS`: Number of parallel pytest workers used by tox (default: `4`)
* `TENSILELITE_CLIENT_ARGS`: Additional arguments passed to `invoke build-client` during tox runs

### Options

* `TENSILELITE_ENABLE_HOST`: Enables generation of tensilelite host (default: `ON`)
* `TENSILELITE_ENABLE_CLIENT`: Enables generation of tensilelite client application (default: `ON`)
* `TENSILELITE_ENABLE_AUTOBUILD`: Generate wrapper scripts (e.g. `Tensile.sh`) for the cmake build tree. **Deprecated** — run `Tensile/bin/Tensile` directly instead (default: `OFF`)
* `TENSILELITE_BUILD_TESTING`: Build tensilelite host library tests (default: `OFF`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build

## How to Rebuild Object Codes Directly from Assembly

During tuning it is often useful to edit one or more assembly kernels and rebuild the `.co` without re-running the full Tensile pipeline. The `invoke asm` task handles that incremental rebuild and works on Linux and Windows.

**You don't need to rerun CMake unless you delete the `tensile-out` folder.**

### Basic usage

```bash
# 1. Edit an assembly file under:
#    tensile-out/1_BenchmarkProblems/<problem>/00_Final/.../assembly/

# 2. Reassemble and relink
invoke asm --tensile-out tensile-out

# Default --tensile-out is build_tmp (matches invoke build-client), so if you
# used the default build directory you can just run:
invoke asm
```

The rebuilt `.co` is written back to the `library/` directory that
`ClientParameters.ini` already references, so you can re-run the benchmark
client immediately.

### Wavefront size

```bash
# gfx942 / gfx950 (64-wide, default)
invoke asm --tensile-out tensile-out --arch gfx942 --wave 64

# gfx1100 (32-wide)
invoke asm --tensile-out tensile-out --arch gfx1100 --wave 32
```

`--arch` is auto-detected from the `.co` filename (e.g. `TensileLibrary_gfx942.co` → `gfx942`).
If detection fails, pass it explicitly — including the xnack suffix if needed:

```bash
invoke asm --tensile-out tensile-out --arch "gfx942:xnack-"
```

### Targeting specific assembly files

```bash
ASM_DIR=tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly

# Rebuild a single kernel
invoke asm --tensile-out tensile-out --srcfiles "$ASM_DIR/kernel_0.s"

# Rebuild all kernels in a directory (glob)
invoke asm --tensile-out tensile-out --srcfiles "$ASM_DIR/*.s"
```

### Extra assembler / linker flags

```bash
invoke asm --tensile-out tensile-out --asm-args "-v" --link-args "--emit-relocs"
```

### Full option reference

| Option | Default | Description |
|--------|---------|-------------|
| `--tensile-out` | `build_tmp` | Path to the Tensile output directory |
| `--arch` | auto | GPU architecture (e.g. `gfx942`, `gfx1100`, `gfx942:xnack-`) |
| `--wave` | `64` | Wavefront size: `32` or `64` |
| `--srcfiles` | auto | Explicit `.s` file(s) or glob pattern to assemble |
| `--asm-args` | — | Extra flags forwarded to the assembler step |
| `--link-args` | — | Extra flags forwarded to the linker step |
| `--rocm-path` | auto | Path to ROCm installation (uses `ROCM_PATH` env or `rocm-sdk` if omitted) |
