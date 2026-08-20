# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling -- the public API surface for the tiling primitives layer.

A human-approachable tiling layer over rocke's static tile-distribution substrate. Authors
import everything they need from this one module (``from rocke.helpers.tiling import TileMma,
make_tile_desc, make_tensor_desc, ...``) -- never by reaching into submodules. The raw
``WarpDistributionEncoding`` integer sequences stay behind the named factories.

The surface, grouped by role (each name is re-exported from the module that owns it):

MMA operation -- target-aware MMA resolution + the atom-grid driver:
    ``TileMma``          resolve a concrete intrinsic + A/B/C layouts for a bound target.
    ``Tiling``           the tile-knobs policy (atom_shape override, iteration order).

Traits -- the MMA intrinsic SSOT:
    ``load_mma_traits``  load + validate the mma_traits.json catalog.
    ``MmaTraits`` / ``MmaTraitsCatalog`` / ``DEFAULT_TRAITS_PATH``.

Authoring -- the human-approachable, quantity-major layout factory:
    ``make_tile_desc``   author a ``TileDesc`` from axes-ordered geometric quantities.

Memory model -- where a tensor sits + which sub-box a tile covers (pure data, no IRBuilder):
    ``TensorDesc`` / ``make_tensor_desc``   ptr-free lengths + strides + dtype.
    ``TensorWindow`` / ``make_window``       a desc positioned at an origin (+ optional clip).

Register model -- where each element lives in lanes/registers (pure data, no IRBuilder):
    ``TileDesc``         a logical-matrix -> per-lane-register layout (shape + encoding).
    ``Fragment`` / ``make_fragment``         a TileDesc bound to a dtype + its SSA registers.
    ``fragment_length``  per-lane register count for an encoding.

Encoding substrate -- the foundational value type the above speak in:
    ``WarpDistributionEncoding``   the raw coordinate-transform encoding (rarely built by hand).

IR verbs -- the lowering layer (these thread the IRBuilder ``b``):
    ``load_fragment`` / ``store_fragment`` / ``fill_fragment`` / ``emit_tensor_coordinates``.

Reflection -- see a layout instead of decoding it:
    ``describe`` / ``render_forward_map`` / ``render_inverse_map``.

Internal machinery (``RegisterMapper``, the a/b/c warp-encoding calculators) is intentionally
NOT re-exported here; import it from its own module if you are extending the layer.
"""

from __future__ import annotations

from .descriptors import TensorDesc, TensorWindow, make_tensor_desc, make_window
from .emit import (
    emit_tensor_coordinates,
    fill_fragment,
    load_fragment,
    store_fragment,
)
from .encoding import WarpDistributionEncoding
from .fragments import Fragment, TileDesc, fragment_length, make_fragment
from .layouts import make_tile_desc
from .mma import TileMma, Tiling
from .visualization import describe, render_forward_map, render_inverse_map
from .traits import (
    DEFAULT_TRAITS_PATH,
    MmaTraits,
    MmaTraitsCatalog,
    load_mma_traits,
)
from .transforms import transform_fragment

__all__ = [
    # MMA operation
    "TileMma",
    "Tiling",
    # Traits
    "load_mma_traits",
    "MmaTraits",
    "MmaTraitsCatalog",
    "DEFAULT_TRAITS_PATH",
    # Authoring
    "make_tile_desc",
    # Memory model
    "TensorDesc",
    "make_tensor_desc",
    "TensorWindow",
    "make_window",
    # Register model
    "TileDesc",
    "Fragment",
    "make_fragment",
    "fragment_length",
    # Encoding substrate
    "WarpDistributionEncoding",
    # IR verbs
    "load_fragment",
    "store_fragment",
    "fill_fragment",
    "transform_fragment",
    "emit_tensor_coordinates",
    # Reflection
    "describe",
    "render_forward_map",
    "render_inverse_map",
]
