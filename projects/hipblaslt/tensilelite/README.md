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

During tuning it is often useful to edit one or more assembly kernels and rebuild the `.co` without re-running the full Tensile pipeline. The `Makefile` in this directory handles that incremental rebuild.

### Directory layout

Running `Tensile/bin/Tensile <yaml> <output-dir>` produces the following structure under `<output-dir>` (referred to as `TENSILE_OUT` below):

```
<TENSILE_OUT>/                                  ← the root; pass this as TENSILE_OUT
└── 1_BenchmarkProblems/
    └── Cijk_Ailk_Bjlk_DB_UserArgs_00/         ← one directory per problem
        └── 00_Final/
            ├── source/
            │   ├── build_tmp/
            │   │   └── SOURCE/
            │   │       └── assembly/           ← .s files live here
            │   │           ├── kernel_0.s
            │   │           └── kernel_1.s
            │   └── ClientParameters.ini        ← read by the benchmark client
            └── caches/<hash>/source/
                └── library/
                    └── TensileLibrary_gfx942.co  ← rebuilt .co goes here
```

> **Note:** `TENSILE_OUT` is whatever name you gave the output directory.
> The default when running via `invoke build-client` is `build_tmp`, so in
> that case pass `TENSILE_OUT=build_tmp`.

**You do not need to rerun CMake unless you delete `TENSILE_OUT`.**

### How file discovery works

The Makefile searches for `.s` and `.co` files anywhere under `00_Final/` using `find`. This means it handles both the `caches/<hash>/source/` layout and a flat `source/` layout without any path hardcoding.

### Basic usage (auto-detect everything)

```bash
# 1. Edit an assembly file, for example:
#    tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/kernel_0.s

# 2. Reassemble and relink — make detects changed .s files automatically
make co TENSILE_OUT=tensile-out

# The rebuilt .co is written back to its original location:
#   tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/caches/<hash>/source/library/TensileLibrary_gfx942.co
#
# ClientParameters.ini already points at this path, so you can re-run the
# benchmark client immediately without any extra steps.
```

### Wavefront size

The default wavefront size is 64. Override with `WAVE`:

```bash
# gfx942 / gfx950 (64-wide)
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64

# gfx1100 (32-wide)
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
```

### Targeting a specific assembly file (SRCFILES override)

If auto-detection picks up the wrong files, or the assembly lives outside the standard tree, pass `SRCFILES` explicitly.

**Scenario:** you have a kernel at a custom path and want to rebuild just that file.

`SRCFILES` is shell-expanded before Make sees it, so wildcards work normally:

```bash
ASM_DIR=tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly

# Rebuild all kernels in the directory
make co TENSILE_OUT=tensile-out SRCFILES="$ASM_DIR/*.s"

# Rebuild a single kernel
make co TENSILE_OUT=tensile-out SRCFILES="$ASM_DIR/kernel_0.s"

# Rebuild a specific subset
make co TENSILE_OUT=tensile-out SRCFILES="$ASM_DIR/kernel_0.s $ASM_DIR/kernel_1.s"
```

The resulting `.co` is written to the `library/` directory alongside the existing `.co` file — the same location `ClientParameters.ini` already references. No copy is needed.

**If the `.co` ends up in an unexpected location** (e.g. you changed `TENSILE_OUT` to point at a scratch directory), the client will not find it automatically. You have two options:

1. **Copy the `.co`** to the `library/` directory the client expects:
   ```bash
   cp my-scratch/TensileLibrary_gfx942.co \
     tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/caches/<hash>/source/library/
   ```

2. **Edit `ClientParameters.ini`** to point `code-object` at the new path:
   ```
   # tensile-out/1_BenchmarkProblems/Cijk_.../00_Final/source/ClientParameters.ini
   code-object=/absolute/path/to/my-scratch/TensileLibrary_gfx942.co
   ```
   `ClientParameters.ini` is generated by `Tensile/ClientWriter.py` and is plain text — editing it directly is safe for local iteration.

### Architecture detection

`ARCH` is auto-detected from the `.co` filename (e.g. `TensileLibrary_gfx942.co` → `gfx942`). If detection fails, set it manually:

```bash
make co TENSILE_OUT=tensile-out ARCH="gfx942:xnack-"
```

### Extra assembler / linker flags

```bash
make co TENSILE_OUT=tensile-out ASM_ARGS="-g" LINK_ARGS="--some-linker-flag"
```

### Reference: all variables

| Variable | Default | Description |
|---|---|---|
| `TENSILE_OUT` | *(required)* | Root Tensile output directory (e.g. `tensile-out` or `build_tmp`) |
| `ARCH` | auto-detected from `.co` name | GPU target, e.g. `gfx942` or `gfx942:xnack-` |
| `WAVE` | `64` | Wavefront size (`32` or `64`) |
| `WAVEFRONTSIZE` | derived from `WAVE` | Raw clang flag, e.g. `-mwavefrontsize64` |
| `SRCFILES` | auto-detected under `00_Final/` | Space-separated list of `.s` files to assemble |
| `AS` | `$(ROCM_PATH)/bin/amdclang++` | Assembler binary |
| `LDD` | `$(ROCM_PATH)/bin/amdclang++` | Linker binary |
| `ASM_ARGS` | *(empty)* | Extra flags for the assemble step |
| `LINK_ARGS` | *(empty)* | Extra flags for the link step |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation root |
