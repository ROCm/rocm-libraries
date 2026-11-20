# pymosaic

Host-side, JIT-backed mosaic layouts for Python.

`pymosaic` exposes mosaic's coordinate ↔ offset layout mapping as a pure
Python `Layout` class. Each distinct layout **structure** (nesting profile)
is compiled once into a small C++ extension; the shape/stride **values** are
supplied at call time, so one compiled module serves every layout that shares
a structure.

## The core distinction: structure in Python, math in C++

There are two kinds of layout operations, and they live in two different
places:

- **Structural** changes happen in **Python**. Rearranging a layout's *mode
  tree* — selecting, reordering, nesting, unnesting, slicing a mode away — is
  pure tuple manipulation on the `(shape, stride)` pair. It never reads a
  stride value and never computes an offset, so it runs in plain Python with
  no compiler involved: `get`, `take`, `select`, `group`, `flatten`,
  `append`/`prepend`, and `[...]` slicing.

- **Math** happens in **C++**. Once a *structure* exists, everything that
  actually evaluates the `coordinate → offset` map — `L[i]`, `crd2idx`,
  `idx2crd`, `cosize`, iteration — is handled by the JIT-compiled module for
  that structure. mosaic's C++ headers are the single source of truth for the
  arithmetic; Python just supplies the runtime values.

In other words: **Python decides the *shape* of the computation; C++ does the
*computation*.** A structural op produces a new `(shape, stride)` — possibly a
brand-new structure — and the *first* time that structure's math is evaluated,
its module is compiled and cached. So `L.group(0, 2)` is instant (pure Python),
but the first `L.group(0, 2)[i]` may trigger a sub-second compile for the
grouped structure.

This is also why some algebra (e.g. `coalesce`, `complement`) is **not** a
Python structural op: their *result structure* depends on the actual stride
values, so they can't be decided by tuple-shuffling alone and belong on the
C++ side. See `pymosaic/_structural.py` for the precise litmus test.

## Install

Straight from the repo (the build vendors the mosaic C++ headers into the
wheel, so nothing else from the repo is needed at runtime):

```bash
pip install -e python/pymosaic
```

### Runtime requirements

- **Python** 3.10–3.13
- **A C++20 compiler** on `PATH` — `g++` by default; override with `$CXX`.
  (This is a system package, e.g. `apt install g++`; it cannot be pip-installed.)
- `ninja` and `nanobind` — installed automatically as dependencies.

The first time a given structure is used it is compiled (sub-second to a few
seconds); afterwards it is loaded from an on-disk cache.

## Usage

```python
from pymosaic import Layout

L = Layout(shape=(4, 8))          # column-major by default (CuTe canonical)
L[2, 3]                           # 14  ==  2*1 + 3*4
L[0, 1]                           # 4
len(L)                            # 32
list(L)[:4]                       # [0, 1, 2, 3]  (flat iteration is column-major)
L.idx2crd(1)                      # (1, 0)  (flat index -> coordinate)

C = Layout.column_major(4, 8)     # leftmost dim varies fastest
C[1, 0], C[0, 1]                  # 1, 4
list(C) == list(range(32))        # True — compact column-major is the identity

H = Layout.from_modes(            # hierarchical / nested layout
    Layout.row_major(2, 2),
    Layout.column_major(3, 3),
)
H.shape, H.rank, H.depth          # ((2, 2), (3, 3)), 2, 2
```

## Testing

The pymosaic test suite lives at `python/tests/pymosaic/` (repo root). It is
pure Python — the `_render` / `_structural` layers run with no compiler, while
the JIT-backed tests compile small C++ modules on the fly (so a C++20 compiler
on `PATH` is needed for those).

`conftest.py` puts the working-copy `pymosaic/` at the front of `sys.path`, so
**no install is required** — tests always run against the source tree. Run from
the repo root so `pytest.ini` is picked up:

```bash
# Install the test deps once (pytest ships in the `dev` extra):
pip install -e 'python/pymosaic[dev]'

# Whole pymosaic suite:
pytest python/tests/pymosaic/

# A single file / test, with verbose output:
pytest python/tests/pymosaic/test_render.py -v

# Pure-Python layers only (no compiler needed):
pytest python/tests/pymosaic/test_render.py python/tests/pymosaic/test_structural.py
```

## Cache management

```python
from pymosaic import jit

jit.cache_info()      # {entries, size_bytes, path, version}
jit.cache_root()      # resolved cache directory
jit.clear_cache()     # wipe all compiled modules
```

The cache root is `$MOSAIC_JIT_CACHE`, else `$XDG_CACHE_HOME/mosaic/jit`, else
`~/.cache/mosaic/jit`. Set `MOSAIC_INCLUDE_DIR` to point the JIT at a specific
mosaic header tree (otherwise the vendored headers shipped in the wheel are
used). Set `MOSAIC_JIT_DEBUG=1` to also emit preprocessed source and assembly
next to each compiled module.
