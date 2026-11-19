# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Descriptor for native scaled-MMA bytes in (free axis, K-group) order."""

from ....core.arch import ArchTarget, MmaOp
from ..fragments import TileDesc
from .tile_distribution import make_tile_desc

__all__ = ["make_mma_scale_desc"]


def make_mma_scale_desc(atom: MmaOp, *, target: str) -> TileDesc:
    """Describe both native scale operands, with the catalog's B axes swapped.

    gfx950 distributes one scale per lane across its K groups. gfx1250 gives
    each lane the complete scale row and broadcasts across both half-waves.
    Only the verified symmetric map profiles are supported.
    """
    if not isinstance(atom, MmaOp):
        raise TypeError("scale descriptor requires an MmaOp")
    catalog = ArchTarget.from_gfx(target).mma
    registered = catalog.by_op_id(atom.op_id)
    if registered is None or atom != registered:
        raise ValueError("scale descriptor requires the registered atom metadata")
    if target == "gfx950" and atom.family == "mfma_scaled":
        m, k, wave, blocks = 32, 64, 64, (32,)
    elif target == "gfx1250" and atom.family == "wmma_scaled":
        m, k, wave, blocks = 16, 128, 32, (16, 32)
    else:
        raise NotImplementedError(f"unmodelled scale layout on {target!r}")
    block = atom.scale_block_k
    if (atom.m, atom.n, atom.k, atom.wave_size) != (
        m,
        m,
        k,
        wave,
    ) or block not in blocks:
        raise NotImplementedError(f"unmodelled scale shape for {atom.op_id!r}")
    groups = atom.k // block
    per_lane = 1 if target == "gfx950" else groups
    # The canonical FP4 E8M0 row identifies the map profile independently of
    # the selected matrix/scale formats. Compare actual callbacks and counts.
    profile = catalog.op_for_shape(
        family=atom.family,
        a_dtype="fp4",
        b_dtype="fp4",
        c_dtype="fp32",
        m=m,
        n=m,
        k=k,
        scales=("e8m0", "e8m0", block),
    )
    for role in (0, 1):
        layout = atom.src_scale_layout(role)
        if (
            profile is None
            or layout != profile.src_scale_layout(role)
            or layout.frag_len != per_lane
        ):
            raise NotImplementedError(
                f"unmodelled scale_src{role} map for {atom.op_id!r}"
            )
    shape = (atom.m, groups)
    if per_lane == 1:
        return make_tile_desc(
            shape=shape,
            thread_dist=shape,
            thread_order=(1, 0),
            wave_size=atom.wave_size,
        )
    return make_tile_desc(
        shape=shape,
        thread_dist=(atom.m, 1),
        thread_tile=(1, per_lane),
        thread_broadcast=atom.wave_size // atom.m,
        wave_size=atom.wave_size,
    )
