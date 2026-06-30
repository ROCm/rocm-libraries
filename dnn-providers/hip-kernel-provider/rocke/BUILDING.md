# Building & running rocke

`rocke/` has two Python source roots:

| Root | Packages | Installed in product? |
|------|----------|-----------------------|
| `platform/Python` | `rocke` (authoring SDK + engine frontend) | yes (wheel / CMake) |
| `library`         | `kernels`, `builders`, `dispatch` (SDPA/MHA) | **no** — build-time only |

The package import name is `rocke`; the SDPA product lives under separate
top-level packages. Because the two halves sit under different roots, every
process that touches both needs both roots resolvable.

We do **not** scatter `sys.path` surgery across scripts. Instead both roots are
**editable-installed** once into an environment, after which `rocke`, `kernels`,
`builders`, and `dispatch` import normally everywhere — scripts, `-m` module
runs, and `pytest` — from any working directory, with no `PYTHONPATH`.

The only remaining filesystem lookups in the tree are for genuine *non-package*
data (the `dsl_docs/` tree and the loose `_ua_shape_utils` benchmark helper).
Those resolve through `rocke.assets` (`platform_root()`, `dsl_docs_dir()`,
`shape_utils_dir()`), each overridable by an env var — never via per-file
`parents[N]` math.

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
python -c "import rocke, kernels, builders, dispatch"   # all resolve
python -m builders.gfx942.attention.parity_unified_attention --help
pytest rocke/library/tests
```

ROCm PyTorch must come from the ROCm wheel index for your system (see
`platform/requirements.txt` / `platform/BUILD.md`); it is not installed here.

## Daily loop

1. One-time: build (CMake makes the venv) or run the manual setup.
2. Edit any `.py` under `platform/Python/rocke/` or `library/` — changes are
   live, no reinstall.
3. Run scripts/tests against the venv interpreter.
4. Re-run the install only after adding a **new top-level package** or changing
   a `pyproject.toml`. Under CMake this happens automatically (the stamp depends
   on both `pyproject.toml` files).

## Data / asset paths

Non-package assets are located through `rocke.assets`, not path math:

```python
from rocke.assets import platform_root, dsl_docs_dir, shape_utils_dir
```

Override the roots for out-of-tree consumption with `ROCKE_PLATFORM_ROOT`,
`ROCKE_DSL_DOCS`.

## C++ engine

The platform C++ engine (`platform/Cpp/` → `librocke_core.a`) and its
`rocke_engine` pybind binding are built by CMake, not pip, and are intentionally
not part of any wheel. See `platform/BUILD.md`.
