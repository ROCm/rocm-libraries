# rocke.helpers.tiling -- human-approachable tiling + MMA primitives

A human-approachable, N-D tiling and MMA primitives layer for rocKE. It lets an author express a
tiled GEMM/MMA kernel -- or any custom thread/register distribution -- without hand-writing the raw
`TileDistributionEncoding` integer sequences, and resolves the concrete hardware intrinsic for the
bound target so authorship stays target-agnostic.

```python
from rocke.helpers.tiling import (
    TileMma, make_tensor_desc, make_window, make_fragment,
    fill_fragment, load_fragment, store_fragment,
)

mma = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")  # IR-free resolve
acc = make_fragment(mma.c_desc, F32)
fill_fragment(b, acc, 0)
for k0 in range(0, K_LEN, 16):
    k = b.const_i32(k0)
    a_frag = load_fragment(b, a_ptr, make_window(a_td, (m, k)), mma.a_desc, lane)
    b_frag = load_fragment(b, b_ptr, make_window(b_td, (n, k)), mma.b_desc, lane)
    acc = mma(b, a_frag, b_frag, acc)          # TileMma owns the whole wave-tile subtile grid + order
store_fragment(b, c_ptr, make_window(c_td, (m, n)), acc, lane)
```

## Documentation
- `docs/tiling_api_surface.md` -- the how-to-use catalog: every surface, its default (MMA-driven)
  mode and its manual override, a composability matrix, and runnable examples. **Start here.**
- `docs/tiling_design_proposal.md` -- the why + architecture + status/decisions.

## Package layout
```
encoding.py         WarpDistributionEncoding -- the foundational coordinate-transform type
register_mapper.py  RegisterMapper / LaneRegister -- pure-int lane x register -> coordinate
descriptors.py      TensorDesc + TensorWindow (memory model) + make_tensor_desc / make_window
fragments.py        TileDesc + Fragment (register model) + make_fragment / fragment_length
emit.py             b-first IR verbs: load_fragment / store_fragment / fill_fragment + coordinates
layouts/            make_tile_desc -- quantity-major human-approachable distribution authoring
mma/                TileMma + Tiling; the A/B/C warp-encoding calculators
traits/             typed MMA traits loader/catalog + committed mma_traits.json
reflection/         describe() + text forward/inverse layout maps
kernels/            end-to-end demo kernels (spec -> build -> run), bit-exact on gfx90a
```
Tests mirror the package at `platform/python/rocke/tests/helpers/tiling/`.

## Running tests
```
PYTHONPATH=platform/python <repo>/.venv/bin/python -m pytest platform/python/rocke/tests/helpers/tiling/ -q
```
GPU-gated numeric tests run on a gfx90a host; the offline suite (encoding / calculator / mapper /
authoring / reflection) runs anywhere.

## Status
Dense MMA GEMM on gfx90a is BUILT and bit-exact: single-atom and wave-tile subtiling (M/N/K grid +
`order`), the human-approachable `make_tile_desc` authoring surface, clipping/bounds for ragged
tiles, and target-agnostic MMA resolution. `c_transpose`, interleaved layouts, sparse, and MX are
designed-in reserved seams.

## Engineering standards
- Descriptive names for every argument, variable, and class; no obfuscating short names.
- Composition over inheritance; small, focused classes; fail-fast validation at the API boundary
  (builtin exceptions, `__post_init__` checks, a message template
  `"{what_failed} -- {param}={bad_value}, expected {constraint}"`; every raise has a unit test).
- PYTHON_STYLE compliant: modern typing (`list[int]`, `X | None`, `collections.abc`),
  `from __future__ import annotations`, relative intra-package imports, `@dataclass(frozen=True)`
  value objects, black (88 cols), `__all__` + `__init__` re-exports. `M/N/K` are reserved for the
  MMA atom shape; tiling uses `TILE_*` / `WAVE_*`.
