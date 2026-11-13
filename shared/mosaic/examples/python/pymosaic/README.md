# `pymosaic` layout JIT — runnable examples

Short scripts that walk through mosaic's **host-side JIT-backed
`Layout` class** (the one in `python/pymosaic/pymosaic/layout.py`, backed
by the JIT orchestrator in `python/pymosaic/pymosaic/_jit.py`).

**New to layouts? Start with the notebook.**
[`00_layout_concepts.ipynb`](00_layout_concepts.ipynb) builds the whole
mental model from scratch — what a layout *is*, stride, the offset
formula, nesting, slicing, and composing your own tilings — with
intuitive prose and a live offset-grid visualization at every step. It's
the recommended on-ramp before the scripts below.

The scripts are then meant to be read **in order** — each builds on the
last and leans on `Layout.pretty_print()` so you can *see* what every
operation does to the offset grid:

1. **`01_basics.py`** — make a layout, evaluate it, visualize the grid.
2. **`02_hierarchical.py`** — nest layouts; the grid grows block borders.
3. **`03_slicing.py`** — drop modes to get lower-rank sub-layouts.
4. **`04_tiling.py`** — compose layouts into a tiling and *see* the tiles.
5. **`05_cache_inspection.py`** — peek under the hood at the JIT cache.

This is the **dynamic route**: the compiled `.so` is keyed on layout
*structure* (nesting profile), and shape/stride **values are passed in
at call time**. One module therefore serves every layout sharing a
structure (e.g. all flat rank-2 layouts). The module is rendered,
compiled with `ninja` / `g++`, and dlopen'd on first use of a given
structure, then cached for the lifetime of the install. This route
covers the core coordinate↔offset mapping plus the pure-Python layout
*algebra* built on it (`coalesce`, `complement`, `right_inverse`,
`blocked_product`); general layout *composition* (and the logical
products built on it) is left to future work.

> [!NOTE]
> These examples are **CPU-only** and have nothing to do with HIP/ROCm
> or Triton. They exercise mosaic's typed C++ layout algebra
> (`include/mosaic/`) from Python via [nanobind](https://nanobind.readthedocs.io/).

## Prerequisites

The JIT pipeline shells out to a C++ toolchain on first use, so you need:

| Tool       | Why                                          | Check                |
| ---------- | -------------------------------------------- | -------------------- |
| Python 3.10+ | mosaic package                             | `python --version`   |
| `g++`      | compiles the per-layout `module.cpp`         | `g++ --version`      |
| `ninja`    | build driver (called by `_jit.py`)           | `ninja --version`    |
| `nanobind` | Python ↔ C++ glue + runtime (`nb_combined.cpp`) | `python -c "import nanobind"` |

The mosaic C++ headers (`include/mosaic/`) ship in this repo and are
located automatically by `_jit._find_mosaic_include()` — no install
step required.

If `g++` is missing, set `$CXX` to your preferred C++20 compiler
before running:

```bash
export CXX=clang++-18
```

## How to run

Two options, both equivalent.

### Option A — directly from a source checkout (no install)

From the repo root:

```bash
# Concepts notebook (recommended starting point)
pip install jupyter   # if you don't have it
jupyter notebook examples/python/pymosaic/00_layout_concepts.ipynb
# (the notebook's first cell auto-adds python/pymosaic to sys.path)

# Tutorials
PYTHONPATH=python/pymosaic python examples/python/pymosaic/01_basics.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/02_hierarchical.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/03_slicing.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/04_tiling.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/05_cache_inspection.py

# C++-example ports
PYTHONPATH=python/pymosaic python examples/python/pymosaic/basic_tiling/check_tiling.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/slice/slice.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/mfma/mfma_layout.py
PYTHONPATH=python/pymosaic python examples/python/pymosaic/gemm_grid/grid_mapping.py
```

This is the fastest path if you just want to try the examples.

### Option B — editable install (recommended for ongoing work)

The JIT layout examples only need the standalone `pymosaic` package,
which pulls in `nanobind` and `ninja` automatically:

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate
pip install -e python/pymosaic
```

After that the examples can be run from anywhere:

```bash
python examples/python/pymosaic/01_basics.py
```

## Contents

### Tutorials (flat files)

| Script                       | Topic                                                       |
| ---------------------------- | ----------------------------------------------------------- |
| `00_layout_concepts.ipynb`   | **Concepts notebook.** Layouts from first principles: shape/stride, offset formula, size/cosize, coords↔indices, nesting, slicing, tiling — with prose + live visualizations. Read this first. |
| `01_basics.py`               | Construction, evaluation, iteration; cold vs hot timings; `pretty_print()` offset grid (row- vs column-major). |
| `02_hierarchical.py`         | Nested layouts; flat / scalar-on-tuple / nested coord forms; `idx2crd` / `crd2idx` round-trip; the grid grows double-line block boundaries. |
| `03_slicing.py`              | `L[:, k]` / `L[i, :]` / `L[..., j]`; base-offset semantics; range slicing rejection. |
| `04_tiling.py`               | `blocked_product` to compose `inner` + `outer` into a hierarchical tiling; `pretty_print()` shows the tile edges; bijectivity check. |
| `05_cache_inspection.py`     | Inspect `~/.cache/mosaic/jit/<hash>/` — manifest, source, log; `cache_info()` / `clear_cache()`; structure-keyed sharing. |

#### Visualizing layouts — `pretty_print()`

Every `Layout` can draw itself as a grid of offsets — the single most
useful tool for building intuition, and what the tutorials lean on:

```python
from pymosaic import Layout

Layout.row_major(4, 8).pretty_print()
```

Flat layouts render with single-line cell borders; **hierarchical**
layouts automatically grow **double-line block boundaries** that mark
the sub-mode (tile) edges, with nested `(i, j)` coordinate labels on the
axes. This is a terminal-native cousin of CuTe's `print_layout`. Use
`L.pretty()` to get the string instead of printing, and the `max_rows` /
`max_cols` / `nested` keyword args to tune large or ambiguous layouts.

## Ports of the C++ examples (`examples/cpp/<name>/`)

These mirror the cpp examples 1:1 where the API allows, with pure-Python
fallbacks (clearly labelled, ~10 lines each) for algebra ops that
mosaic's C++ side exposes but the Python JIT hasn't surfaced yet
(`blocked_product`). `gemm_grid/` also keeps local integer helpers for
the chiplet remap; those are not part of the C++ library.

| Directory               | Cpp source                            | What it shows |
| ----------------------- | ------------------------------------- | ------------- |
| `basic_tiling/`         | `examples/cpp/basic_tiling/`          | 2-level workgroup tiling via `blocked_product`. |
| `slice/`                | `examples/cpp/slice/`                 | CuTe-style `_`-slicing on flat and hierarchical layouts. |
| `mfma/`                 | `examples/cpp/mfma/`                  | TV layouts for `v_mfma_f32_32x32x8_f16`; bijectivity + ground-truth check vs. AMD's matrix calculator. |
| `gemm_grid/`            | (Python only) | 3-level hierarchical GEMM *schedule*; default vs mosaic visualisation. |

Each port directory has its own `README.md` with run instructions and
caveats, a script (e.g. `check_tiling.py`) that mirrors the cpp demo
1:1, and a `visualize.py` companion that renders a PNG into a
`figures/` subdirectory when you run it. Those images are not committed.

```
basic_tiling/
├── README.md
├── check_tiling.py    # text demo (matches cpp)
└── visualize.py       # matplotlib renderer
```

## Visualizations

The `visualize.py` scripts require `matplotlib`. Install with
`pip install matplotlib`.

### libstdc++ ABI note (conda envs)

If you run Python from a conda env whose bundled `libstdc++` is older
than your system g++, the JIT `.so` may fail to load when matplotlib
is imported first (matplotlib pins the older libstdc++ into the process
before mosaic's `.so` gets a chance). The `visualize.py` scripts work
around this by importing mosaic and triggering a trivial JIT compile
*before* importing matplotlib, which forces the system libstdc++ to
be the one that lands in the process.

If you ever hit `version GLIBCXX_3.4.XX not found` from a different
script, the universal workaround is:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python your_script.py
```

## Cold vs hot — what to expect

First invocation per *structure* builds a ~250 KB `.so` and takes
**1–5 s** depending on your compiler. Every subsequent call for any
layout of that structure (same process or fresh) is a dlopen + table
lookup and runs in **microseconds**.

Each example except `05_cache_inspection.py` will use your real cache
at `$XDG_CACHE_HOME/mosaic/jit` (or `~/.cache/mosaic/jit`), so the
second run of any script is ~1 s faster than the first. `01_basics.py`
and `05_cache_inspection.py` both override the cache to a fresh
`tempfile.mkdtemp()` so timings and disk-inspection output are
reproducible regardless of prior runs.

## Useful environment knobs

| Variable           | Effect                                                         |
| ------------------ | -------------------------------------------------------------- |
| `MOSAIC_JIT_CACHE` | Override the cache root. Useful for isolating CI runs.         |
| `MOSAIC_JIT_DEBUG` | If `1`, also emit `debug/module.E.cpp` (preprocessed source) and `debug/module.s` (assembly) per entry — roughly doubles on-disk size but invaluable when debugging template instantiation failures or verifying constant-folding. |
| `CXX`              | C++ compiler to invoke (default `g++`). Must be C++20-capable. |

## Verifying your install

If imports fail or the JIT compile errors out, the most common causes:

* **`No module named 'pymosaic'`** — you forgot `PYTHONPATH=python/pymosaic`
  (or `pip install -e python/pymosaic`).
* **`nanobind is required`** — `pip install nanobind` (or just
  `pip install -e python/pymosaic`, which lists it as a dependency).
* **`could not locate the mosaic C++ headers`** — you ran from outside
  the repo against a `pymosaic` that has no vendored headers; reinstall
  with `pip install python/pymosaic` (the build vendors the headers) or
  set `MOSAIC_INCLUDE_DIR` to a mosaic header tree.
* **`mosaic JIT compile failed`** — the `JitCompileError` exception
  message points you at the cache entry's `compile.log` and tells you
  the exact `ninja` command to re-run for diagnostics.

A 5-second smoke test:

```bash
PYTHONPATH=python/pymosaic python -c "
from pymosaic import Layout
L = Layout.row_major(4, 8)
print(L, '->', L[2, 3])
"
# Expected: (4,8):(8,1) -> 19
```

## Where to go next

* `python/tests/pymosaic/test_layout_jit.py` — the 800+ line pytest suite that
  defines the API surface in formal-spec form.
* `python/pymosaic/pymosaic/_jit.py` — the JIT orchestrator (cache
  layout, build pipeline, version key).
* `python/pymosaic/pymosaic/_template.cpp.in` — the C++ template that the
  JIT fills in for each layout structure (runtime `std::size_t` leaves).
* `python/pymosaic/pymosaic/pretty.py` — the `pretty_print()` renderer.
* `docs/python_bindings_design.md` — the design document the
  implementation tracks against.
