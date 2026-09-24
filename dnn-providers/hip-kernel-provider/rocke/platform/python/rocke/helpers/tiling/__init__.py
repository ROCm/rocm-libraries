# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling -- the public API surface for the tiling primitives layer.

A human-approachable tiling layer over rocke's static tile-distribution substrate. Authors
import everything they need from this one module (``from rocke.helpers.tiling import TileMma,
make_tile_desc, make_tensor_desc, ...``) -- never by reaching into submodules. The raw
``WarpDistributionEncoding`` integer sequences stay behind the named factories.

The surface is grouped by AUDIENCE tier. START HERE: to WRITE a kernel, read
``docs/tiling_api_surface.md`` (its sec 2 is a complete minimal ``load -> mma -> store`` body) and reach
for the FRONT DOOR names below. To understand WHY the surface is shaped this way -- and the rules for
adding to it -- read ``docs/tiling_api_contract.md``.

FRONT DOOR -- what you reach for by default:
    ``TileMma``   resolve a concrete intrinsic + A/B/C layouts and drive the atom grid.
    ``make_tensor_desc`` / ``make_window``   where a tensor sits + which sub-box a tile covers.
    ``make_tile_desc``   author a custom (non-MMA) tile layout; for MMA operands use
        ``mma.a_desc`` / ``b_desc`` / ``c_desc``.
    ``make_fragment``    a ``TileDesc`` bound to a dtype + its SSA registers.
    ``load_fragment`` / ``store_fragment`` / ``fill_fragment`` / ``transform_fragment``   the IR verbs
        (these thread the IRBuilder ``b``).

TOOLBOX -- the primitives the front door composes, callable directly for finer control:
    ``Tiling``   optional knobs for ``TileMma`` (pin the atom / the subtile order); not needed by default.
    ``TileMmaPlan`` / ``TileMmaDriver``   the design + iteration halves ``TileMma`` composes -- reach for
        these to hold the resolved layouts (plan) or drive the atom grid (driver) yourself.
    ``LayoutStyle`` / ``CanonicalStyle`` / ``InterleavedStyle``   the operand-layout STRATEGY -- pass
        ``style=`` to pick a profile (default canonical), or subclass to add one (docs/tiling_api_contract.md).
    ``cooperative_load_desc`` / ``cooperative_load_width`` / ``lds_tile_alloc`` / ``lds_store`` /
        ``lds_read``   the style-agnostic global->LDS memory bridge a style composes (full-tile alloc,
        unmasked LDS verbs, extent-vs-allocation guard).
    ``TensorDesc`` / ``TensorWindow`` / ``TileDesc`` / ``Fragment`` / ``fragment_length``   the value types.
    ``load_mma_traits`` / ``MmaTraits`` / ``MmaTraitsCatalog`` / ``DEFAULT_TRAITS_PATH``   the atom SSOT.
    ``describe`` / ``render_forward_map`` / ``render_inverse_map``   see a layout instead of decoding it.
    ``emit_tensor_coordinates``   lower a desc to per-lane tensor coordinates.
    ``classify_transform`` / ``describe_edge`` / ``diagnose_k_match`` / ``operand_soundness`` /
        ``mma_pair_compatible`` / ``reorder_between`` / ``derive_c_distribution``   the read-only
        transform observers -- classify an edge, check MMA soundness, derive C (never mutate/emit).
    ``Diagnostic`` / ``TransformPlan`` / ``ReorderPlan``   the observers' result types (the objects the
        observers above hand back).
    ``WarpDistributionEncoding``   the raw coordinate-transform encoding (extension substrate; rarely
        built by hand).

MACHINERY -- internal, NOT re-exported (``RegisterMapper``, the warp-encoding calculators, the transform
solver core): import from its own module only if you are extending the layer.
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
from .mma import (
    TileMma,
    Tiling,
    TileMmaPlan,
    TileMmaDriver,
    LayoutStyle,
    CanonicalStyle,
    InterleavedStyle,
)
from .memory import (
    cooperative_load_desc,
    cooperative_load_width,
    lds_tile_alloc,
    lds_store,
    lds_read,
)
from .visualization import describe, render_forward_map, render_inverse_map
from .traits import (
    DEFAULT_TRAITS_PATH,
    MmaTraits,
    MmaTraitsCatalog,
    load_mma_traits,
)
from .transforms import (
    transform_fragment,
    classify_transform,
    describe_edge,
    diagnose_k_match,
    operand_soundness,
    mma_pair_compatible,
    reorder_between,
    derive_c_distribution,
    Diagnostic,
    TransformPlan,
    ReorderPlan,
)

__all__ = [
    # ---- FRONT DOOR: reach for these by default -------------------------------------------------
    "TileMma",
    "make_tensor_desc",
    "make_window",
    "make_tile_desc",
    "make_fragment",
    "load_fragment",
    "store_fragment",
    "fill_fragment",
    "transform_fragment",
    # ---- TOOLBOX: the primitives the front door composes, callable directly ---------------------
    "Tiling",
    "TileMmaPlan",
    "TileMmaDriver",
    # layout styles (the operand-layout strategy seam) + the cooperative memory bridge they compose
    "LayoutStyle",
    "CanonicalStyle",
    "InterleavedStyle",
    "cooperative_load_desc",
    "cooperative_load_width",
    "lds_tile_alloc",
    "lds_store",
    "lds_read",
    "TensorDesc",
    "TensorWindow",
    "TileDesc",
    "Fragment",
    "fragment_length",
    "load_mma_traits",
    "MmaTraits",
    "MmaTraitsCatalog",
    "DEFAULT_TRAITS_PATH",
    "describe",
    "render_forward_map",
    "render_inverse_map",
    "emit_tensor_coordinates",
    # transform observers (read-only analysis): classify an edge, check MMA soundness, derive C
    "classify_transform",
    "describe_edge",
    "diagnose_k_match",
    "operand_soundness",
    "mma_pair_compatible",
    "reorder_between",
    "derive_c_distribution",
    # the observers' result types (so an exported function's return type is importable from the root)
    "Diagnostic",
    "TransformPlan",
    "ReorderPlan",
    "WarpDistributionEncoding",  # extension substrate (rarely built by hand)
    # ---- MACHINERY is intentionally NOT re-exported (see the module docstring) ------------------
]
