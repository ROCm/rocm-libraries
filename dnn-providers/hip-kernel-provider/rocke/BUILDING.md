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
data (the `dsl_docs/` tree, the platform `tools/` scripts, the platform parity
emitters and the loose `_ua_shape_utils` benchmark helper). Those resolve
through `rocke.assets` (`platform_root()`, `dsl_docs_dir()`,
`platform_tools_dir()`, `platform_parity_dir()`, `shape_utils_dir()`), with the
roots overridable via `ROCKE_PLATFORM_ROOT` / `ROCKE_DSL_DOCS` /
`ROCKE_PLATFORM_TOOLS_DIR` / `ROCKE_PLATFORM_PARITY_DIR` (`shape_utils_dir()`
derives from `dsl_docs_dir()`) — never via per-file `parents[N]` math.

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
- **No package index in provider mode.** When rocKE is built as part of the
  hip-kernel-provider, the build runs `pip` with `--no-deps --no-build-isolation`
  and never upgrades pip/setuptools, so it works offline. `setuptools>=61` must
  already be installed for the Python interpreter CMake finds (configure fails
  with a message otherwise). The wheels are built from temporary copies, so no
  `build/` or `*.egg-info` is written into the source tree. Set
  `SOURCE_DATE_EPOCH` to stamp the engine version with a reproducible date;
  without it the version is just `1.0.0`.

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
# test runner + heuristics ML deps (pytest 9 or newer, pandas,
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
from rocke.assets import (
    dsl_docs_dir,
    platform_parity_dir,
    platform_root,
    platform_tools_dir,
    shape_utils_dir,
)
```

Override the roots for out-of-tree consumption with `ROCKE_PLATFORM_ROOT`,
`ROCKE_DSL_DOCS`, `ROCKE_PLATFORM_TOOLS_DIR`, `ROCKE_PLATFORM_PARITY_DIR`.

## C++ engine

The platform C++ engine (`platform/cpp/` → `librocke_core.a`), its shared
library `librocke` (`-DROCKE_BUILD_SHARED_ENGINE=ON`) and its `rocke_engine`
pybind binding are built by CMake, not pip, and are intentionally not part of
any wheel.

Python reaches the engine two ways. The default `cpp` backend, and every test
that compares the two engines on one kernel, lowers serialized IR through
`librocke`'s C ABI (`rocke.core.native_engine`, ctypes): point
`ROCKE_ONLINE_LIB` at the built library, or install it beside the `rocke`
package, as the provider's test build does. Only the tests of a kernel family's
C++ builder need the `rocke_engine` module (`-DROCKE_BUILD_PYBIND=ON`, which
needs pybind11 in the build's Python). See `platform/BUILD.md`. The same configure also builds
the library families' C++ mirrors (`library/cpp/` → `librocke_library.a`, beside
`librocke_core.a`) and, with `ROCKE_BUILD_PYBIND=ON`, their `rocke_library_engine`
binding beside `rocke_engine`; `-DROCKE_BUILD_LIBRARY_CPP=OFF` skips them.
`cmake --install` puts the library archive and headers into the same `ckc`
package: a C/C++ consumer of a library family (e.g. the conv API in
`rocke/instance_conv_*.h`) links `ckc::rocke_library`, which brings
`ckc::rocke_core` with it. `-DROCKE_INSTALL_DEVELOPMENT=OFF` skips that
developer install (both archives, all public headers and the `ckc` package,
which share one export set). The hip-kernel-provider sets it OFF, because no
TheRock artifact component ships those files, and ties every rocKE test install
switch and `BUILD_TESTING` to `HIPKERNELPROVIDER_ENABLE_TESTS`.
