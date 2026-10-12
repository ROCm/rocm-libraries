# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate registration for GDN/KDA decode.

The registry owns the configured tile space, registered once per gate kind:
one tuned candidate per (gate kind, tile), whose knob space (``rules.py``,
axes in ``axes.py``) holds every performance knob of that tile. The kernel
validator remains the only authority that decides which configurations are
legal for a request. Production ``auto`` runs one static configuration per
case (``auto_default``: gate kind x fused x state width, a tile plus a knob
stack, data in ``axes.AUTO_DEFAULTS``) and never consults runtime
measurements.

The tile is the variant, not an axis: each tile keeps its own ``spec_id``,
and ``auto`` only resolves the knob space of the one tile it selects, so its
cost does not grow with the knob space.
"""

from __future__ import annotations

import dataclasses as dc
from functools import lru_cache
from itertools import product
from numbers import Integral
from typing import Tuple

from kernels.gfx950.gdn_decode import (
    GDN_DTYPES,
    GdnDecodeSpec,
    conv_once_applies,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.dispatch.core import Capability, KernelCandidate, OperatorRequest
from rocke.dispatch.tuning import make_tuned_candidate

from .axes import AUTO_DEFAULTS, AUTO_STATE_ROW, AutoDefault
from .common import (
    FAMILY,
    GDN_ABI_VERSION,
    GdnDecodeRequest,
    normalize_dtype,
    request_errors,
)
from .rules import GdnDecodeSpace

ARCH = "gfx950"

NUM_WARPS = (1, 2, 4, 8, 16)
WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
BLOCKS_PER_V_DIM = (1, 2, 4, 8, 16, 32)
_LEXICOGRAPHIC_TILES = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))


def _auto_tiles(gate_kind: str) -> Tuple[Tuple[int, int, int], ...]:
    """The gate kind's ``AUTO_DEFAULTS`` tiles, in table order, once each."""
    tiles = (d.tile for (gate, _, _), d in AUTO_DEFAULTS.items() if gate == gate_kind)
    return tuple(dict.fromkeys(tiles))


# Each gate kind registers the whole tile space. Its priority order is its auto
# tiles first, then every other tile in product order. ``auto`` falls back to
# that order only when the case's tile is illegal for the request: the first
# legal tile, at its base configuration. Which tile that is depends on head_k_dim
# and head_v_dim (ALGORITHM.md §4.6). Known limitation: at head_k_dim =
# head_v_dim = 64 or 192, which cannot split over 16 K lanes, it is (1,1,1),
# which is register-heavy; a footprint-based fallback order is not implemented.
CONFIGURED_TILES_BY_GATE = {
    gate: _auto_tiles(gate)
    + tuple(t for t in _LEXICOGRAPHIC_TILES if t not in _auto_tiles(gate))
    for gate in ("gdn", "kda")
}
CONFIGURED_TILES = CONFIGURED_TILES_BY_GATE["gdn"]
KDA_CONFIGURED_TILES = CONFIGURED_TILES_BY_GATE["kda"]
assert set(CONFIGURED_TILES) == set(KDA_CONFIGURED_TILES) == set(_LEXICOGRAPHIC_TILES)

# Spec ids carry the gate kind: GDN's are ``nw{}_wtk{}_bpv{}``, KDA's add a
# ``kda_`` prefix, so the two can never collide in the shared registry.
_KDA_SPEC_ID_PREFIX = "kda_"
ALGORITHM = "warp_tiled"


def spec_id_for(tile: Tuple[int, int, int], gate_kind: str) -> str:
    """Registry spec id of the ``gate_kind`` candidate that carries ``tile``."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    base = f"nw{num_warps}_wtk{warp_threads_k}_bpv{blocks_per_v_dim}"
    return _KDA_SPEC_ID_PREFIX + base if gate_kind == "kda" else base


def is_fused(req: GdnDecodeRequest) -> bool:
    return req.fuse_conv or req.fuse_out_norm


def auto_default(req: GdnDecodeRequest) -> AutoDefault:
    """The ``AUTO_DEFAULTS`` entry of ``req``'s case: its gate kind, whether it
    is fused (either flag), and its state width's row (f16 reads the bf16
    row). Never batch, head count or work."""
    row = AUTO_STATE_ROW[normalize_dtype(req.state_dtype)]
    return AUTO_DEFAULTS[(req.gate_kind, is_fused(req), row)]


def auto_tile(req: GdnDecodeRequest) -> Tuple[int, int, int]:
    """The tile ``auto`` prefers for ``req`` whenever it is legal."""
    return auto_default(req).tile


def make_spec(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec: the
    tile's base configuration (every knob at its dataclass default), which
    its knob space (``GdnDecodeSpace``) builds every configuration from.

    Fields that ``is_valid_spec`` type-checks pass through unconverted, so a
    malformed request (``use_qk_l2norm="false"``, ``num_k_heads=True``) is
    rejected by that validator instead of being coerced into a different valid
    spec. ``normalize_dtype`` stays: it maps documented dtype aliases. A copied
    dim is cast only when it is an Integral that is not a bool -- a numpy
    integer canonicalizes to plain ``int`` for the JSON-hashed identity
    payload, while a malformed one (``True``, ``4.0``) passes through raw so
    ``is_valid_spec`` still rejects it.
    """
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    cast_dim = lambda v: (
        int(v) if isinstance(v, Integral) and not isinstance(v, bool) else v
    )
    return dc.replace(
        GdnDecodeSpec(),
        num_k_heads=cast_dim(req.num_k_heads),
        num_v_heads=cast_dim(req.num_v_heads),
        head_k_dim=cast_dim(req.head_k_dim),
        head_v_dim=cast_dim(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        state_dtype=normalize_dtype(req.state_dtype),
        use_qk_l2norm=req.use_qk_l2norm,
        gate_kind=req.gate_kind,
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
        fuse_conv=req.fuse_conv,
        fuse_out_norm=req.fuse_out_norm,
    )


@lru_cache(maxsize=64)
def _tile_is_legal(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> bool:
    # Knobs never change tile legality, so the base spec answers for all of
    # the tile's configurations. Memoized: every candidate of the gate kind
    # asks the same question about the auto tile for one ``auto`` request.
    return is_valid_spec(make_spec(req, tile), arch=req.arch)[0]


def _make_candidate(*, tile: Tuple[int, int, int], priority: int, gate_kind: str):
    spec_id = spec_id_for(tile, gate_kind)
    name = f"gdn_decode_{ARCH}_{spec_id}"
    space = GdnDecodeSpace(
        abi=GDN_ABI_VERSION,
        arch=ARCH,
        path="decode",
        variant_id=spec_id,
        candidate_name=name,
    )

    def errors(req: OperatorRequest) -> list:
        found = request_errors(req)
        if found:
            return found
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != ARCH:
            return [f"candidate arch {ARCH} != request arch {req.arch!r}"]
        # A candidate serves exactly one gate kind: the two gates emit
        # different kernels, so a GDN candidate must never answer a KDA
        # request or the reverse.
        if req.gate_kind != gate_kind:
            return [
                f"candidate {spec_id!r} serves the {gate_kind!r} gate, request "
                f"asks for {req.gate_kind!r}"
            ]
        return []

    def route_auto(req: GdnDecodeRequest) -> Tuple[bool, str]:
        # ``auto`` is one preferred tile per case (auto_tile); when that tile
        # is illegal for the request, every legal candidate stays admitted and
        # priority order picks the first.
        wanted = auto_tile(req)
        if tile != wanted and _tile_is_legal(req, wanted):
            return False, (
                f"static {req.gate_kind.upper()} auto tile for this request "
                f"is {wanted!r}, not {tile!r}"
            )
        return True, "ok"

    def auto_knobs(req: GdnDecodeRequest):
        # ``auto`` on its case's tile carries the case's knob stack (a knob
        # the request cannot read drops; a stack the tile's space refuses
        # raises); on a fallback tile it runs the tile's base configuration.
        if tile != auto_tile(req):
            return {}
        knobs = dict(auto_default(req).knobs)
        # norm_gate_once needs conv_once until ALGORITHM.md section 8,
        # follow-up 8 (a NOT_YET_IMPLEMENTED rule of the validator, not
        # scope), so a row that carries it drops it wherever conv_once drops.
        if knobs.get("norm_gate_once") and not conv_once_applies(make_spec(req, tile)):
            knobs.pop("norm_gate_once")
        return knobs

    def scope_check(req: GdnDecodeRequest) -> Tuple[bool, str]:
        # A stored tuning_id names the knob set that was canonical where it
        # was recorded. Where scope drops one of its knobs the id names no
        # configuration (resolution would call it stale); say why instead.
        wanted = req.tuning_id.strip() or "auto"
        if wanted == "auto" or not req.tuning_knobs:
            return True, "ok"
        base = make_spec(req, tile)
        knobs = dict(req.tuning_knobs)
        knobs.pop(space.outer_knob, None)
        if space.refuse(base, knobs):
            return True, "ok"  # resolution refuses the malformed value by name
        dropped = space.out_of_scope(base, knobs)
        if not dropped:
            return True, "ok"
        return False, (
            f"stored tuning_id {wanted!r} was recorded with {list(dropped)}, "
            f"which this request does not read on {spec_id} (out of scope "
            "here), so the id names no configuration on this problem; pin the "
            "tuning_knobs alone (tuning_id 'auto') to carry them across shapes; "
            "no fallback was selected"
        )

    # The shared tuned candidate: pin resolution (spec_id + tuning_id /
    # tuning_knobs), the knob space walk, and the kernel validator as the
    # final authority. Not opt-in (``route_auto``): unlike attention, where
    # ``auto`` routes to untuned path candidates, GDN ``auto`` is itself one
    # configuration of a registered tile (``auto_knobs``: the case's stack).
    return make_tuned_candidate(
        name=name,
        family=FAMILY,
        algorithm=ALGORITHM,
        spec_id=spec_id,
        abi_version=GDN_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=(ARCH,), dtypes=GDN_DTYPES),
        space=space,
        base=lambda req: make_spec(req, tile),
        request_errors=errors,
        signature=lambda spec: gdn_decode_signature(spec.kernel_spec),
        build=lambda spec, arch: spec.build(arch),
        # No Torch binding (and no `bind`): this family is selectable but not
        # launchable through the generic runner, so today EVERY launch goes
        # through the driver's `prepare()` and therefore through
        # `_validate_decode_inputs`. That is the only thing standing between a
        # mis-shaped tensor and an out-of-bounds access -- the kernel emits no
        # buffer descriptor, so there is no `num_records` to clamp one. Whoever
        # adds a binding must route it through that validator, or the checks
        # stop covering the path callers actually use.
        bind_torch=None,
        grid=lambda spec, req: spec.launch_grid(req.batch),
        precheck=scope_check,
        route_auto=route_auto,
        auto_knobs=auto_knobs,
    )


def candidates() -> Tuple[KernelCandidate, ...]:
    """GDN then KDA candidates, one per configured tile of each gate kind,
    in each gate kind's fallback order."""
    out = []
    for gate_kind in ("gdn", "kda"):
        for tile in CONFIGURED_TILES_BY_GATE[gate_kind]:
            out.append(
                _make_candidate(tile=tile, priority=10 + len(out), gate_kind=gate_kind)
            )
    return tuple(out)


def register(registry) -> None:
    registry.extend(candidates())
