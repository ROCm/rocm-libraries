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
tox -e py3 -- tensilelite/Tests -m common
```

Subsequently, you can run just the Tensile unit tests via:

```
tox -e unit -- tensilelite/Tests/unit
```

Tox uses [tox-uv](https://github.com/tox-dev/tox-uv/tree/1.36.0#uvlock-support)
to synchronize its environments from `uv.lock`; tox provisions the plugin and uv itself.
Most test environments use the `dev` dependency group and rebuild rocisa once before tests.
Lint/format/security commands use only `lint`; mutation wrapper self-tests and library
uniqueness checks use only `test-tools`. These source-only tools need no ROCm SDK.
Each environment that installs rocisa keeps a separate native build tree. If you delete
that tree or its environment and the rebuild reports a missing build directory, run
`tox --recreate -e <environment>` to rebuild the cached editable installation.

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

### Development environment with uv

From the TensileLite directory, with the ROCm SDK and Python development headers available:

```bash
uv sync
uv run --no-sync pytest rocisa/test
```

The default `dev` group includes the test and build tools. Both TensileLite and rocisa are
editable. After native source changes, rebuild rocisa before running Python directly:

```bash
uv run --no-sync python -c "import importlib.util; importlib.util.find_spec('rocisa').loader.rebuild()"
```

This uses scikit-build-core's [manual rebuild API](https://scikit-build-core.readthedocs.io/en/stable/configuration/editable.html#triggering-a-rebuild-manually)
to build and install changed native code without reinstalling the Python package. Tox and
the affected-tests hook run it once before tests. Ordinary imports do not rebuild;
rocisa's stale-source diagnostic rejects watched native sources newer than the extension.
Keep `rocisa/build/uv/<wheel-tag>` and the environment's build tools in place.
Changes to packaging metadata or the set of installed files require
`uv sync --reinstall-package rocisa` to regenerate the editable installation.
The compiled-in example plugin is enabled by TensileLite's uv settings; standalone
rocisa installs retain their existing defaults. The separate uv build directory keeps
these development settings out of standalone pip/Invoke CMake caches.
When migrating an older isolated uv editable build, remove `rocisa/build/uv` and run
`uv sync --reinstall-package rocisa` once to discard its temporary build-environment paths.

To build the client while retaining uv's rocisa installation, run the manual rebuild
above after native changes, then:

```bash
uv run --no-sync invoke build-client --no-rebuild-rocisa
```

The existing Invoke installation workflow uses pip. If you want to run `invoke rocisa` or
let `invoke build-client` refresh rocisa inside the uv environment, first seed it with pip:

```bash
uv venv --seed --allow-existing
uv sync
```

Invoke retains its own build settings. Use `uv sync --reinstall-package rocisa` to restore
the uv settings after an Invoke refresh. Cached CMake settings not explicitly overridden
remain until the native build tree is recreated.

### Updating Python dependencies

Declare dependency compatibility ranges in `pyproject.toml`; `uv.lock` records the resolved
versions. Groups let tox select the tools a task needs without installing unrelated packages:

| Group | Purpose |
|---|---|
| `test-tools` | Pytest and its shared plugins, usable by source-only checks without TensileLite or ROCm. |
| `test` | Includes `test-tools` and rocisa for tests that use the native extension. |
| `lint` | flake8, Bandit, Black, and isort for source inspection and formatting without package installation. |
| `build` | The rocisa backend, kept installed for manual editable rebuilds and existing no-isolation consumers. |
| `coverage` | Coverage report tools, independently installable to inspect existing results. |
| `dev` | Includes `test`, `lint`, `build`, and `coverage`, plus the remaining contributor tools; selected by default. |

`test` and `dev` include the smaller groups so dependency constraints are declared once.
The two source-only test commands share `test-tools`, including its pytest plugins, rather
than maintaining a separate pytest requirement. Tox uses `only_groups` for these limited
setups, excluding the project and its dependencies as well as other groups.
Keep compatibility requirements in `pyproject.toml` and selected versions in `uv.lock`.
After editing a declaration, run these commands from `projects/hipblaslt/tensilelite`:

```bash
uv lock
python scripts/generate_requirements.py
python scripts/generate_requirements.py --check
```

`uv lock` updates the lock while preferring existing locked versions that still satisfy
the declarations. It does not automatically upgrade packages or select the lowest versions.
For packages without a reusable lock entry, uv defaults to the highest compatible version.
Keep the existing lock when regenerating it; deleting it discards those version selections.

To deliberately change a locked version, use one of these commands in place of `uv lock`,
then regenerate and check the requirements files as above:

| Intent | Command |
|---|---|
| Upgrade one package | `uv lock --upgrade-package black` |
| Select a specific release | `uv lock --upgrade-package black==24.4.0` |
| Upgrade all packages | `uv lock --upgrade` |

Review the lock and generated requirements diffs for version changes. See uv's
[upgrade rules](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions)
and [resolution strategies](https://docs.astral.sh/uv/concepts/resolution/#resolution-strategy).

The generated, fully pinned requirements files support pip consumers: `requirements.txt`
contains project and build dependencies, `requirements-test.txt` adds test dependencies
without replacing an already installed rocisa artifact, and `requirements-dev.txt` contains
the full development profile including noneditable rocisa. Pip does not apply the
`tool.uv` editable build settings. Run installations from this directory
so the relative rocisa path resolves. Install the development profile, or tox separately,
when you need the `tox` command; the base requirements no longer install it.

The unit suite checks that the lock and generated requirements are current. This runs
in the existing `coverage-unit` CI job and the affected-tests hook for dependency changes.
It uses uv provided by tox or your development environment. For direct pytest runs in
a pip-only source environment, install uv separately or run through tox. Installed-artifact
tests skip this repository-only check.

### Pre-commit hook (affected tests)

An opt-in git pre-commit hook runs the unit + characterization tests affected by
your staged TensileLite changes and blocks the commit on real failures (it falls
back to the full unit + characterization suite when it cannot narrow the set). It
runs `uv run --no-sync pytest`, which imports rocisa (a HIP native extension), so install and
commit from inside a ROCm dev container (HIP at `/opt/rocm`, a Python with dev
headers). The hook uses the [uv development setup](#development-environment-with-uv) and
rebuilds changed native sources once before starting pytest. Mount the repo and its Git
metadata at their host paths inside the container; linked worktrees can point to Git
metadata outside the worktree.

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
tensilelite/bin/Tensile tensilelite/Tests/common/exception/<test>.yaml tensile-out
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
tensilelite/bin/Tensile tensilelite/Tests/pre_checkin/<test>.yaml tensile-out \
                           --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client
```

**4. Build with tox (Custom Build Args)**

This workflow uses `tox` with custom CMake arguments, which is useful for creating
specialized builds (e.g., Debug builds) and setting the architecture.

```
# build the client using tox with custom CMake flags
cd rocm-libraries/projects/hipblaslt/tensilelite
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- tensilelite/Tests -m common

# run tests with a single pytest worker (useful for debugging)
TENSILE_NUM_PYTEST_WORKERS=1 tox -e py3 -- tensilelite/Tests -m common
```

`invoke build-client` follows the existing `tensilelite` CMake preset by default.
In this repo, `/opt/rocm` compiler settings come from the preset,
`CMAKE_EXPORT_COMPILE_COMMANDS` is enabled by default, and the client build includes rocisa.

Use these flags when you want to override or make that behavior explicit:

* `--rocm-path <path>`: Override the compiler toolchain to use `<path>/bin/amdclang` and `<path>/bin/amdclang++`
* `--export-compile-commands`: Explicitly force `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
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
* `TENSILELITE_ENABLE_AUTOBUILD`: Generate wrapper scripts (e.g. `Tensile.sh`) for the cmake build tree. **Deprecated** — run `tensilelite/bin/Tensile` directly instead (default: `OFF`)
* `TENSILELITE_BUILD_TESTING`: Build tensilelite host library tests (default: `OFF`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build

## How to Rebuild Object Codes Directly from Assembly

During the tuning process, it is of interest to modify an assembly file/s and rebuild the corresponding object file/s and then relink the corresponding co file. Currently, we generate additional source files and a script to provide this workflow.

A new `Makefile` is added that manages rebuilding a co file during iterative development when tuning. One modifies an assembly file of interest, then runs `make` and make will detect what file/s changed and rebuild accordingly.

Assumptions:

- Each problem directory contains a library directory with one co file corresponding to one architecture

**Edit**(2025/3/31) ``rocisa`` use the CMake build system instead of the ``virtualenv``. The behavior of the TensileLite changed a bit with only one extra line.

Example:

```cmake -DTENSILE_BIN=Tensile -DDEVELOP_MODE=ON -S <path-to-tensilelite-root> -B <tensile-out>```

The script will be created in the build folder and will be named in Tensile.bat or Tensile.sh depending on the platform. Then you can then run the script under the ``tensile-out`` folder as usual:

> **Deprecated:** `Tensile.sh` / `Tensile.bat` will be removed in a future release.
> Run `tensilelite/bin/Tensile` directly instead.

```
Tensile.sh <abs-path>/tensilelite/Tests/gemm/fp16_use_e.yaml tensile-out
```

or

```
Tensile.bat <abs-path>/tensilelite/Tests/gemm/fp16_use_e.yaml tensile-out
```

**You don't need to rerun CMake unless you delete the ``tensile-out`` folder.**

To build asm only:

```
# modify an assembly file in
# tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/caches/<key>/source/build_tmp/SOURCE/assembly
# The scratch directory is SOURCE, except when a silicon stepping was asked for.
# A stepping shares its ISA with the architecture it steps, so both would otherwise
# claim the same directory; it gets SOURCE-<stepping> instead, and only it does
# (gfx1250 -> SOURCE, gfx1250-strict -> SOURCE-gfx1250-strict).
make co TENSILE_OUT=tensile-out
# re-run the client
```

The Makefile will set the target based on the name of the co file and sets a default wavefront flag but each of these can be customized as follows:

For 64 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64
```

For 32 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
```

In addition, we provide `ASM_ARGS` and `LINK_ARGS` as additional customization points for the assemble and link step respectively. If the architecture cannot be detect corectly, you may need to manually add ``ARCH="gfx942:xnack-"`` to the ``make`` command.
