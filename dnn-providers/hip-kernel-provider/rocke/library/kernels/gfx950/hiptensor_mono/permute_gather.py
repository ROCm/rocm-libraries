# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unrolled gather permute body: ``U`` output vectors per thread, loads first.

``permute_nd``'s body moves one output element (or one vector) per thread and
relies on many resident workgroups for memory parallelism. Inside the
monokernel a workgroup is one of ``P`` resident ones and runs a permute ticket
with only its own waves, so each thread must keep several loads in flight:
this body computes ``U`` source offsets (out-of-range lanes clamp to offset
0), issues all ``U`` loads unconditionally, then stores the in-range ones.
Same semantics, signature and descriptor as ``permute_nd`` (``X``, ``Y``;
``block_id_x`` = block of ``block_size * U * vec`` output elements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import IRBuilder, KernelDef, PtrType
from rocke.helpers.tensor_view import make_global_view
from rocke.instances.common.permute_nd import (
    PermuteSpec,
    _build_offset_descriptor,
    io_ir_type,
)

__all__ = ["PermuteGatherSpec", "build_permute_gather", "permute_gather_grid"]


@dataclass(frozen=True)
class PermuteGatherSpec:
    """``permute``: the ``permute_nd`` problem; ``unroll``: output vectors per
    thread (all their loads are issued before the stores)."""

    permute: PermuteSpec
    unroll: int = 8


def permute_gather_grid(spec: PermuteSpec, unroll: int) -> Tuple[int, int, int]:
    per = spec.block_size * unroll * spec.vec_width
    return (-(-spec.total_elements // per), 1, 1)


def build_permute_gather(spec: PermuteGatherSpec, *, arch: str = "gfx950") -> KernelDef:
    unroll = spec.unroll
    spec = spec.permute
    if unroll < 1:
        raise ValueError("unroll >= 1")
    io_ty = io_ir_type(spec.dtype)
    vec = spec.vec_width
    total = spec.total_elements
    BS = spec.block_size
    desc = _build_offset_descriptor(spec)
    b = IRBuilder(f"{spec.kernel_name()}_gu{unroll}")
    b.kernel.attrs["max_workgroup_size"] = BS
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    Xv = make_global_view(X, shape=spec.x_shape, dtype=io_ty)
    Yv = make_global_view(Y, shape=spec.y_shape, dtype=io_ty)
    tid = b.thread_id_x()
    bid = b.block_id_x()
    base = b.mul(bid, b.const_i32(BS * unroll * vec))
    c_total = b.const_i32(total)
    c0 = b.const_i32(0)
    outs, oks, vals = [], [], []
    for j in range(unroll):
        o = b.add(base, b.mul(b.add(b.const_i32(j * BS), tid), b.const_i32(vec)))
        ok = b.cmp_lt(o, c_total)
        oc = b.select(ok, o, c0)
        src, _v = desc.offset(b, out_idx=oc)
        vals.append(
            Xv.load_vec_at(b, src, n=vec) if vec > 1 else Xv.load_scalar_at(b, src)
        )
        outs.append(o)
        oks.append(ok)
    for j in range(unroll):
        with b.scf_if(oks[j]):
            if vec > 1:
                Yv.store_vec_at(b, outs[j], vals[j], n=vec)
            else:
                Yv.store_scalar_at(b, outs[j], vals[j])
    return b.kernel
