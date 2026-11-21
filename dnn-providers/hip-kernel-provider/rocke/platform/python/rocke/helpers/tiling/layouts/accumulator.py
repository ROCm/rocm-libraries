# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Accumulator/result tile descriptors for verified matrix-instruction profiles."""

from __future__ import annotations

from ....core.arch import ArchTarget
from ....core.arch.target import MmaOp
from ....core.ir import F32, dtype_to_ir_type
from ...atoms import MfmaAtom, make_c_warp_dstr_encoding
from ..encoding import WarpDistributionEncoding
from ..fragments import TileDesc
from .tile_distribution import make_tile_desc

__all__ = ["make_mma_accumulator_desc"]


def make_mma_accumulator_desc(
    atom: MmaOp, *, target: str, result: bool = False
) -> TileDesc:
    """Describe accumulator input C or independent result D in its machine order.

    MFMA composes the existing C-warp calculator. The gfx1250 WMMA profile has
    contiguous per-lane rows; other WMMA targets are deliberately refused rather
    than borrowing that layout. This does not select or emit an instruction.
    Missing verified maps and unmodelled profiles raise explicitly.
    """
    if not isinstance(atom, MmaOp):
        raise TypeError("accumulator descriptor requires an MmaOp")
    if type(result) is not bool:
        raise TypeError("result must be a bool")
    registered = ArchTarget.from_gfx(target).mma.by_op_id(atom.op_id)
    if registered is None:
        raise ValueError(f"atom {atom.op_id!r} is not available on {target!r}")
    metadata = atom.dst if result else atom.srcs[2]
    expected = registered.dst if result else registered.srcs[2]
    same_map = metadata.layout is expected.layout or (
        metadata.layout is not None
        and expected.layout is not None
        and metadata.layout.fn is expected.layout.fn
        and metadata.layout.frag_len == expected.layout.frag_len
        and metadata.layout.wave_size == expected.layout.wave_size
    )
    if (
        atom.family != registered.family
        or atom.shape != registered.shape
        or atom.wave_size != registered.wave_size
        or metadata.dtype != expected.dtype
        or metadata.frag_len != expected.frag_len
        or not same_map
    ):
        raise ValueError("accumulator descriptor requires the registered atom metadata")
    # These accessors refuse unverified maps; shape/count alone are not authority.
    if result:
        atom.dst_layout()
    else:
        atom.src_layout(2)
    if dtype_to_ir_type(metadata.dtype) != F32:
        raise NotImplementedError(
            "accumulator descriptor currently supports fp32 metadata"
        )
    if atom.family in ("mma", "mfma_scaled"):
        legacy = MfmaAtom(
            atom.m,
            atom.n,
            atom.k,
            atom.srcs[0].frag_len,
            atom.srcs[1].frag_len,
            metadata.frag_len,
            atom.srcs[0].dtype,
            "f32",
            atom.op_id,
        )
        encoding = make_c_warp_dstr_encoding(legacy)
        return TileDesc(
            (atom.m, atom.n),
            WarpDistributionEncoding(
                replication_lengths=encoding.Rs,
                hierarchical_lengths=encoding.Hs,
                lane_to_rh_major=encoding.Ps2RHs_major,
                lane_to_rh_minor=encoding.Ps2RHs_minor,
                register_to_rh_major=encoding.Ys2RHs_major,
                register_to_rh_minor=encoding.Ys2RHs_minor,
            ),
        )
    if atom.family in ("wmma", "wmma_scaled") and target == "gfx1250":
        if (atom.m, atom.n, atom.wave_size, metadata.frag_len) != (16, 16, 32, 8):
            raise NotImplementedError("unmodelled gfx1250 accumulator layout")
        return make_tile_desc(
            shape=(atom.m, atom.n),
            thread_dist=(atom.m // metadata.frag_len, atom.n),
            thread_tile=(metadata.frag_len, 1),
            wave_size=atom.wave_size,
        )
    raise NotImplementedError(f"unmodelled accumulator layout on {target!r}")
