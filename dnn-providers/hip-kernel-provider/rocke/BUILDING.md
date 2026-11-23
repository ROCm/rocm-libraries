# Building & running rocke

`rocke/` has two Python source roots:

| Root | Packages | Installed in product? |
|------|----------|-----------------------|
| `platform/python` | `rocke` (authoring SDK + engine frontend) | yes (wheel / CMake) |
| `library`         | `kernels`, `builders`, `dispatch` (SDPA/MHA, conv, ...) | **no** for the Python (build-time only); the C++ mirrors in `library/cpp` ship in the `ckc` package (see "C++ engine") |

The package import name is `rocke`; the SDPA product lives under separate
top-level packages. Because the two halves sit under different roots, every
process that touches both needs both roots resolvable.

We do **not** scatter `sys.path` surgery across scripts. Instead both roots are
**editable-installed** once into an environment, after which `rocke`, `kernels`,
`builders`, and `dispatch` import normally everywhere — scripts, `-m` module
runs, and `pytest` — from any working directory, with no `PYTHONPATH`.

The only remaining filesystem lookups in the tree are for genuine *non-package*
data (the `dsl_docs/` tree, the platform `tools/` scripts and the loose
`_ua_shape_utils` benchmark helper). Those resolve through `rocke.assets`
(`platform_root()`, `dsl_docs_dir()`, `platform_tools_dir()`,
`shape_utils_dir()`), with the roots overridable via `ROCKE_PLATFORM_ROOT` /
`ROCKE_DSL_DOCS` / `ROCKE_PLATFORM_TOOLS_DIR` (`shape_utils_dir()` derives from
`dsl_docs_dir()`) — never via per-file `parents[N]` math.

## CI-matched environment (perf, parity and AOT work)

CI compiles rocKE kernels with the COMGR of the TheRock revision pinned as
`therock-ref` in `.github/actions/ci-env/action.yml`. A local `/opt/rocm` or a
ROCm torch wheel usually brings a different LLVM, so register use and timings
measured there are not what CI ships. `tools/setup_dev_env.py` builds a venv
whose only COMGR is CI's:

```sh
VENV=$(python3 tools/setup_dev_env.py)        # Linux; stdout is the venv path
. "$VENV/bin/activate"
```

```powershell
$venv = py -3.12 tools\setup_dev_env.py        # Windows
& "$venv\Scripts\Activate.ps1"
```

It resolves CI's LLVM commit from the pin, installs ROCm, ROCm torch and the
device wheel for the visible GPU from the newest nightly built with that LLVM
into a shared base venv, gives this checkout its own small venv on top of it
with `platform[dev]` and `library` editable-installed, and pins
`ROCKE_COMGR_LIB` / `ROCKE_HIP_LIB` to the venv's runtime for every process that
runs the venv's Python. Venvs live under `$ROCKE_VENV_ROOT` (default
`~/rocke-venvs`) as `<arch>-llvm-<commit>/{base,<checkout>-<hash>}`; a re-run
reuses complete ones, and a pin with a new LLVM gets new ones. Run it where the
GPU is (inside the Slurm job, not on a login node). It fails rather than install
a nightly built with a different LLVM.

For AOT packaging builds pass the same COMGR to CMake:
`-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB=$ROCKE_COMGR_LIB`. A process that also
loads C++ libraries linked against `/opt/rocm` (e.g. a hipDNN you built there)
before rocKE cannot use the venv's COMGR; in that process point both variables
at `/opt/rocm/lib` (an explicit value wins over the venv's).

The base venv takes several GB (more on Windows, which also installs
`rocm-sdk-devel`, the `ROCM_PATH` for the hipDNN superbuild), so point
`ROCKE_VENV_ROOT` at shared storage with room when the home quota is small.
Some wheel files sit about 200 characters deep, so on Windows either enable long
paths (`projects/hipdnn/scripts/windows/windows_build_setup.ps1`, needs
Administrator) or use a short root such as `ROCKE_VENV_ROOT=C:\rv`; the script
checks this before downloading.

## Quick start (CMake-managed environment)

When the rocke features are enabled, the build creates and maintains the
editable environment for you:

```sh
cmake -S dnn-providers/hip-kernel-provider -B build \
      -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON
cmake --build build            # also builds the `rocke-pyenv` target
```

This produces `build/rocke-pyenv/`, a virtualenv with `platform` and `library`
editable-installed. Use it directly:

```sh
build/rocke-pyenv/bin/python -m builders.gfx950.attention.benchmark_prefill2d_live --help
build/rocke-pyenv/bin/python -m pytest rocke/library/tests
```

Properties:

- **Always current.** Editable installs point at live source, so editing a
  `.py` needs no reinstall. The install only re-runs when a `pyproject.toml`
  changes (the build gates it on a stamp), e.g. when a new top-level package is
  added.
- **Hermetic.** Everything lands in `build/rocke-pyenv/`, never your system or
  user site-packages — so the generic names `kernels` / `builders` / `dispatch`
  are confined to this venv.
- **Inherits system torch.** The venv is created with `--system-site-packages`,
  so the ROCm PyTorch you already have is reused, not reinstalled.

Toggle off with `-DROCKE_BUILD_PYENV=OFF` if you manage the environment yourself.

## Manual setup (your own venv)

Equivalent to what CMake does, for editing/running outside the build:

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip "setuptools>=61" wheel
# editable_mode=compat writes a .pth at the source root so newly added modules
# are importable without reinstalling.
pip install --config-settings editable_mode=compat -e rocke/platform
pip install --config-settings editable_mode=compat --no-deps -e rocke/library
```

Order matters: `platform` first (it provides the `rocke` distribution that
`rocke-library` depends on); `library` with `--no-deps` so pip resolves `rocke`
from the local editable install rather than an index.

After this, from any directory:

```sh
python -c "import rocke, kernels, builders, dispatch, benchmarks"   # all resolve
python -m benchmarks.common.attention_combo_sweep --help
rocke-attention-combo-sweep --list-only --arch gfx950
pytest rocke/library/tests
```

ROCm PyTorch must come from the ROCm wheel index for your system (see
`platform/requirements.txt` / `platform/BUILD.md`); it is not installed here.

## Daily loop

1. One-time: build (CMake makes the venv) or run the manual setup.
2. Edit any `.py` under `platform/python/rocke/` or `library/` — changes are
   live, no reinstall.
3. Run scripts/tests against the venv interpreter.
4. Re-run the install only after adding a **new top-level package** or changing
   a `pyproject.toml`. Under CMake this happens automatically (the stamp depends
   on both `pyproject.toml` files).

## Running the full test sweep

The sweep needs the test/heuristics extras plus a ROCm PyTorch:

```sh
# test runner + heuristics ML deps (pytest, pytest-subtests, pandas,
# scikit-learn, lightgbm) come from the platform `dev` extra:
pip install --config-settings editable_mode=compat -e "rocke/platform[dev]"
pip install --config-settings editable_mode=compat --no-deps -e rocke/library
# torch: ROCm build for your system (see platform/requirements.txt). A CPU
# torch lets the non-GPU suite import/run; GPU tests then skip without a device.

pytest rocke/platform/tests rocke/library/tests
```

Expected residual on a box whose `/opt/rocm` clang is a *different* LLVM vintage
than rocke's hardcoded datalayout constants: the 7
`TestDatalayoutDriftGuard::test_datalayout_matches_hipcc_emitted_ir` subtests
fail with an `e-m:` datalayout diff. This is a toolchain-vs-constant drift the
guard is designed to catch (regenerate `_DATALAYOUT_*` in
`core/lower_llvm.py` for the local toolchain) — it is independent of the
platform/library layout and reproduces identically on `develop`.

Everything else is green; GPU-only tests skip without a device.

## Data / asset paths

Non-package assets are located through `rocke.assets`, not path math:

```python
from rocke.assets import platform_root, dsl_docs_dir, platform_tools_dir, shape_utils_dir
```

Override the roots for out-of-tree consumption with `ROCKE_PLATFORM_ROOT`,
`ROCKE_DSL_DOCS`, `ROCKE_PLATFORM_TOOLS_DIR`.

## C++ engine

The platform C++ engine (`platform/cpp/` → `librocke_core.a`) and its
`rocke_engine` pybind binding are built by CMake, not pip, and are intentionally
not part of any wheel. See `platform/BUILD.md`. The same configure also builds
the library families' C++ mirrors (`library/cpp/` → `librocke_library.a`, beside
`librocke_core.a`) and, with `ROCKE_BUILD_PYBIND=ON`, their `rocke_library_engine`
binding beside `rocke_engine`; `-DROCKE_BUILD_LIBRARY_CPP=OFF` skips them.
`cmake --install` puts the library archive and headers into the same `ckc`
package: a C/C++ consumer of a library family (e.g. the conv API in
`rocke/instance_conv_*.h`) links `ckc::rocke_library`, which brings
`ckc::rocke_core` with it.
