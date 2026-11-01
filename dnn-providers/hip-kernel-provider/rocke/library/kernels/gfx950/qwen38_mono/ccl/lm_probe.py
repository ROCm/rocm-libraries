# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CCL-LM diagnostic: xGMI link probe (push vs pull bandwidth between ranks).

One kernel per (mode, cache bits, waves, vectors in flight): every CTA moves
its share of ``nbytes`` between this rank and peer ``(rank + 1) mod N``:

* ``mode="push"``: ``global_store_dwordx4`` of 1 KiB per wave instruction into
  the peer's arena (``sc0 sc1``, ``+ nt`` with ``nt``), ``vec`` instructions
  back to back, then ``vmcnt(0)`` once at the end (every store acknowledged);
* ``mode="pull"``: ``global_load_dwordx4 ... sc0 sc1`` of the peer's arena,
  ``vec`` loads in flight per batch, folded with ``xor`` into one register
  stored to ``sink`` (so the loads stay live).

``active`` (runtime) = 0 makes a rank idle (one-directional runs).
``xmask`` (runtime) = the XCDs whose CTAs move data (CTA ``c`` runs on XCD
``c mod 8``); the bytes are split over the active CTAs only (``xrank``: 4-bit
rank of each XCD inside the mask, ``nact`` its population). Stamps
(i64[nctas x 2], ``s_memrealtime``): CTA start, CTA end (after its waves
drained). Payload offset ``off0`` (bytes) inside the arenas: a region no
collective layout uses. Diagnostics only (not a collective, no tags); the
peer-address lint holds (system-scope bits on every peer access).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType, VectorType
from rocke.helpers import device_ops as D

__all__ = ["LinkBwSpec", "build_linkbw_kernel"]

V4 = VectorType(I32, 4)
MODES = ("push", "pull")


@dataclass(frozen=True)
class LinkBwSpec:
    ranks: int = 2
    mode: str = "push"
    nt: bool = False
    waves: int = 8
    vec: int = 4

    def __post_init__(self) -> None:
        if self.mode not in MODES or not 2 <= self.ranks <= 8 or self.vec not in (1, 2, 4) or self.waves not in (4, 8, 16):
            raise ValueError("bad LinkBwSpec")
        if self.nt and self.mode != "push":
            raise ValueError("nt applies to push")

    @property
    def threads(self) -> int:
        return 64 * self.waves

    def name(self) -> str:
        return f"q38ccl_lbw_n{self.ranks}_{self.mode}{'nt' if self.nt else ''}_w{self.waves}_v{self.vec}"


def build_linkbw_kernel(spec: LinkBwSpec, *, arch: str = "gfx950") -> KernelDef:
    """Params: ``peer0 .. peer{N-1}`` (i64), ``sink`` i32*, ``stamps`` i64*,
    i32 ``rank``, ``nctas`` (a multiple of 8), ``nbytes`` (a multiple of
    ``vec`` KiB), ``off0``, ``active``, ``xmask``, ``xrank``, ``nact``."""
    if arch != "gfx950":
        raise ValueError("gfx950 only")
    N, Wv, Vv = spec.ranks, spec.waves, spec.vec
    b = IRBuilder(spec.name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    peers = [b.param(f"peer{r}", I64) for r in range(N)]
    sink = b.param("sink", PtrType(I32, "global"))
    stamps = b.param("stamps", PtrType(I64, "global"))
    rd = b.readfirstlane
    rank = rd(b.param("rank", I32))
    nctas = rd(b.param("nctas", I32))
    nbytes = rd(b.param("nbytes", I32))
    off0 = rd(b.param("off0", I32))
    active = rd(b.param("active", I32))
    xmask = rd(b.param("xmask", I32))
    xrank = rd(b.param("xrank", I32))
    nact = rd(b.param("nact", I32))
    bid = rd(b.block_id_x())
    wave = D.wave_id(b)
    lane = b.lane_id()
    zero = b.const_i32(0)
    nxt = b.mod(b.add(rank, b.const_i32(1)), b.const_i32(N))
    peer = peers[0]
    for r in range(1, N):
        peer = b.select(b.cmp_eq(nxt, b.const_i32(r)), peers[r], peer)
    peer = rd(peer)
    xcd = b.land(bid, b.const_i32(7))
    on = b.land(b.lshr(xmask, xcd), b.const_i32(1))
    active = rd(b.select(b.cmp_ne(on, zero), active, zero))
    ax = b.land(b.lshr(xrank, b.shl(xcd, b.const_i32(2))), b.const_i32(15))
    aidx = b.add(b.mul(b.lshr(bid, b.const_i32(3)), nact), ax)
    gw = b.add(b.mul(aidx, b.const_i32(Wv)), wave)
    tw = b.mul(b.mul(b.lshr(nctas, b.const_i32(3)), nact), b.const_i32(Wv))
    nb = b.lshr(nbytes, b.const_i32(10 + {1: 0, 2: 1, 4: 2}[Vv]))  # batches of vec KiB
    with b.scf_if(D.is_thread0(b)):
        b.store(D.memrealtime(b), stamps, b.mul(bid, b.const_i32(2)))
    acc0 = b.const_i32(0)
    cp = "sc0 sc1 nt" if spec.nt else "sc0 sc1"
    with b.scf_if(b.cmp_ne(active, zero)):
        left = b.sub(nb, gw)
        nk = rd(b.select(b.cmp_gt(left, zero), b.div(b.add(left, b.sub(tw, b.const_i32(1))), tw), zero))
        loop = b.scf_for_iter(zero, nk, b.const_i32(1), [("lbwacc", acc0)], iv_name="lbwk")
        with loop as (k, carried):
            batch = b.add(gw, b.mul(k, tw))
            voff = b.add(off0, b.add(b.shl(batch, b.const_i32(10 + {1: 0, 2: 1, 4: 2}[Vv])), b.shl(lane, b.const_i32(4))))
            if spec.mode == "push":
                data = b.vec_pack([lane, batch, k, rank], I32)
                lines = [f"global_store_dwordx4 $0, $1, $2{(' offset:' + str(1024 * v)) if v else ''} {cp}" for v in range(Vv)] + ["s_nop 1"]
                b.inline_asm("\n".join(lines), "v,v,s,~{memory}", [voff, data, peer], None)
                b.scf_yield(carried[0])
            else:
                lines = [f"global_load_dwordx4 ${v}, ${Vv}, ${Vv + 1}{(' offset:' + str(1024 * v)) if v else ''} sc0 sc1" for v in range(Vv)]
                lines.append("s_waitcnt vmcnt(0)")
                cons = ",".join(["=&v"] * Vv + ["v", "s"]) + ",~{memory}"
                outs = b.inline_asm_multi("\n".join(lines), cons, [voff, peer], result_types=[V4] * Vv, result_name_hint="lbwl")
                a = carried[0]
                for o in outs:
                    for i in range(4):
                        a = b.xor(a, b.vec_extract(o, i))
                b.scf_yield(a)
        b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)
        if spec.mode == "pull":
            b.store(loop.results[0], sink, b.add(b.mul(gw, b.const_i32(64)), lane))
    b.sync()
    with b.scf_if(D.is_thread0(b)):
        b.store(D.memrealtime(b), stamps, b.add(b.mul(bid, b.const_i32(2)), b.const_i32(1)))
    b.ret()
    return b.kernel
