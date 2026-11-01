# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Persistent hipTensor contraction monokernel on the universal-GEMM body (gfx950).

EXPERIMENTAL: not reached by any dispatcher; build it explicitly.

ONE persistent launch runs a whole hipTensor contraction plan
(:mod:`.contraction_plan`):

1. ``pack`` stage: producer tasks gather every non-canonical operand into a
   canonical ``[rows, K]`` workspace (separable offset tables, vector width
   from the plan). Task ``t`` of ``R_t`` rows runs on workgroup ``t % P``.
   It publishes with a release fence, a barrier and one atomic increment of a
   global counter (``dep="all"``, default) or one ``flags[t] = epoch`` store
   (``dep="panel"``).
2. ``gemm`` stage: the persistent tile loop of the universal GEMM body
   (:func:`persistent_body_spec`: DTL ring, AGPR accumulators, merged MFMA
   interleave, LDS row interleave, persistent grid of 256). Before a tile's
   DTL prologue the workgroup spins on the global pack counter (``dep="all"``,
   once per workgroup) or on the flags of exactly the pack tasks its A / B
   panels need (``dep="panel"``), then issues one acquire fence.
3. Epilogue: ``D[drow[m] + dcol[n]] = alpha * acc (+ beta * C[same])`` in f32,
   one rounding to the output type, vector width ``vec_d``, rows and columns
   masked for ragged shapes. The output permutation costs no extra pass.
4. ``reduce`` stage (optional): the row max over every GEMM-column mode
   (``R[m] = max_n D[m, n]``) as per-tile partials; the LAST tile of an M panel
   (atomic arrival counter, target ``epoch * tiles_n``) folds the partials of
   that panel in a fixed order (deterministic) and writes R.

Producers never wait and every wait points to an earlier stage in each
workgroup's program order, so with all P workgroups resident the schedule
cannot deadlock; :func:`build_persistent_contraction` validates that on the
host with :class:`rocke.helpers.task_schedule.TaskTable`. ``epoch`` grows by
one per launch, so flags and counters are never reset.

The hooks ride the universal GEMM's ``spec._mono`` side channel (Python only;
absent -> byte-identical emission). Every lever is a knob
(:class:`PersistentContractionKnobs`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    TemporalHint,
    Value,
)

from .contraction_plan import ContractionPlan

__all__ = [
    "PersistentContractionKnobs",
    "PersistentContractionSpec",
    "OperandPackSpec",
    "PersistentContractionHooks",
    "persistent_body_spec",
    "persistent_contraction_hooks",
    "build_persistent_contraction",
    "build_operand_pack",
]

# the persistent body's tile (persistent_body_spec): 256 x 256, 2 x 2 warps
_TILE_M, _TILE_N, _WARP_N = 256, 256, 2
_GRID = 256


@dataclass(frozen=True)
class PersistentContractionKnobs:
    rows_per_task: int = (
        64  # pack task granularity (rows), halved until it divides the rows
    )
    pack_unroll: int = 4  # chunks in flight per thread in a pack task
    # "all": one counter of every pack task (one wait + acquire per workgroup);
    # "panel": per-task flags, each tile waits on the tasks of its panels;
    # "none": no wait (only with pre_tasks=False: packs run in an earlier launch)
    dep: str = "all"
    pre_tasks: bool = (
        True  # False: the packs run in a separate launch (build_operand_pack)
    )
    store_nt: bool = True  # streaming hint on D stores (only at vec_d >= nt_min_vec)
    bilinear: bool = False  # + beta * C
    reduce: str = ""  # "" | "max" (row max over every GEMM-column mode)
    store_d: bool = True  # False: only R (the reduction output) is written
    epi_group: int = 8  # epilogue rows whose offset / C loads are issued together
    nt_min_vec: int = 8  # streaming D stores only at this vector width or wider
    reg_transpose: bool = True  # vec_rows 8 transposing packs in registers (no LDS)

    def tag(self) -> str:
        t = f"r{self.rows_per_task}u{self.pack_unroll}{self.dep[0]}"
        if not self.pre_tasks:
            t += "_np"
        if not self.store_nt:
            t += "_t"
        if self.bilinear:
            t += "_bil"
        if self.reduce:
            t += f"_red{self.reduce}"
        if not self.store_d:
            t += "_nod"
        if not self.reg_transpose:
            t += "_lds"
        return t


@dataclass(frozen=True)
class PersistentContractionSpec:
    """One persistent contraction monokernel.

    ``plan``: the contraction plan (problem shape). ``body_traits``: universal
    GEMM ``TraitSpec`` overrides of :func:`persistent_body_spec` as
    ``(name, value)`` pairs (e.g. ``(("chiplet_chunk_size", 8),)``).
    """

    plan: ContractionPlan
    knobs: PersistentContractionKnobs = field(
        default_factory=PersistentContractionKnobs
    )
    dtype: str = "f16"
    body_traits: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class OperandPackSpec:
    """The stand-alone operand pack kernel (first launch of the two-launch
    pipeline whose second launch is the monokernel built with
    ``pre_tasks=False``)."""

    plan: ContractionPlan
    knobs: PersistentContractionKnobs = field(
        default_factory=PersistentContractionKnobs
    )
    dtype: str = "f16"


def _ty(dtype: str):
    return {"f16": F16, "fp16": F16, "bf16": BF16}[dtype]


class PersistentContractionHooks:
    """The ``spec._mono`` side channel consumed by ``gemm_universal``."""

    def __init__(
        self,
        plan: ContractionPlan,
        knobs: PersistentContractionKnobs,
        dtype: str = "f16",
        tile_m: int = _TILE_M,
        tile_n: int = _TILE_N,
        warp_n: int = _WARP_N,
    ):
        if knobs.dep not in ("all", "panel", "none"):
            raise ValueError(f"dep must be all | panel | none, got {knobs.dep!r}")
        if knobs.reduce not in ("", "max"):
            raise ValueError(f"reduce must be '' or 'max', got {knobs.reduce!r}")
        self.plan = plan
        self.k = knobs
        self.dt = _ty(dtype)
        self.tile_m, self.tile_n, self.warp_n = tile_m, tile_n, warp_n
        self.packs = plan.packs if knobs.pre_tasks or knobs.dep != "none" else []
        self.writes_operands = bool(plan.packs) and knobs.pre_tasks
        # K padded to the GEMM's K tile: the packs zero-fill k in [K, Kp)
        self.Kp = -(-plan.K // 64) * 64
        if self.Kp != plan.K:
            bad = [
                p.name
                for p in plan.ops.values()
                if p.canonical
                or not (p.mode == "t" and p.vec_rows == 8 and knobs.reg_transpose)
            ]
            if bad or plan.K % 8:
                raise ValueError(
                    f"K={plan.K} not a multiple of 64 needs register-transpose packs of every operand ({bad})"
                )
        self.ntask = {}
        self.flag_base = {}
        self.R = {}
        base = 0
        for p in plan.packs:
            # task rows: rows_per_task, halved until it divides the operand's rows
            # (>= 8 for the register transpose / K gather, >= 64 for the LDS transpose)
            R = knobs.rows_per_task
            lds = p.mode == "t" and not (p.vec_rows == 8 and knobs.reg_transpose)
            floor = 64 if lds else 8
            while R > floor and p.rows % R:
                R //= 2
            if p.rows % R:
                raise ValueError(
                    f"rows {p.rows} of {p.name}: no task size >= {floor} divides them"
                )
            self.R[p.gemm_role] = R
            self.ntask[p.gemm_role] = p.rows // R
            self.flag_base[p.gemm_role] = base
            base += p.rows // R
        self.n_pack_tasks = base
        self.tiles_m = -(-plan.M // tile_m)
        self.tiles_n = -(-plan.N // tile_n)
        self.cnt_all = base  # dep == "all": one counter
        self.red_base = base + 1
        # i32 words of the Flags buffer the launch must pass (zeroed once)
        self.n_flags = self.red_base + (self.tiles_m if knobs.reduce else 0)
        self.n_part = self.tiles_n * warp_n
        self.M_pad = self.tiles_m * tile_m
        self._first_wait = True

    # ------------------------------------------------------------ params
    def add_params(self, b: IRBuilder, *, A, B, C, M, N, K) -> None:
        self.A, self.B, self.D = A, B, C
        dt = self.dt
        self.p: Dict[str, Value] = {}
        for p in self.plan.packs:
            r = p.gemm_role
            self.p[f"Src{r}"] = b.param(
                f"Src{r}", PtrType(dt, "global"), readonly=True, align=16
            )
            self.p[f"Row{r}"] = b.param(
                f"Row{r}", PtrType(I32, "global"), readonly=True, align=4
            )
            self.p[f"Koff{r}"] = b.param(
                f"Koff{r}", PtrType(I32, "global"), readonly=True, align=4
            )
        self.p["DRow"] = b.param("DRow", PtrType(I32, "global"), readonly=True, align=4)
        self.p["DCol"] = b.param("DCol", PtrType(I32, "global"), readonly=True, align=4)
        if self.k.bilinear:
            self.p["Cin"] = b.param(
                "Cin", PtrType(dt, "global"), readonly=True, align=16
            )
            self.p["beta"] = b.param("beta", F32)
        self.p["alpha"] = b.param("alpha", F32)
        self.p["Flags"] = b.param("Flags", PtrType(I32, "global"), align=4)
        self.p["epoch"] = b.param("epoch", I32)
        self._alloc_tsm(b)
        if self.k.reduce:
            self.p["RedPart"] = b.param("RedPart", PtrType(F32, "global"), align=16)
            self.p["Red"] = b.param("Red", PtrType(dt, "global"), align=4)
            self.slot = b.smem_alloc(I32, [4], name_hint="mono_slot")

    def _alloc_tsm(self, b: IRBuilder) -> None:
        self.tsm = None
        if self.k.pre_tasks and any(
            p.mode == "t" and not (p.vec_rows == 8 and self.k.reg_transpose)
            for p in self.plan.packs
        ):
            self.tsm = b.smem_alloc(self.dt, [64, 72], name_hint="mono_tpose")

    def _emit_pack_t(self, b: IRBuilder, p, t, tid) -> None:
        """LDS-transposing pack of one task (row-contiguous source): per 64 x 64
        tile, vector loads along the rows (vec_rows), scalar LDS writes
        [row][k], 16-byte LDS reads along K, 16-byte workspace stores."""
        K = self.plan.K
        r_ = p.gemm_role
        R = self.R[r_]
        src, rowt, kt = self.p[f"Src{r_}"], self.p[f"Row{r_}"], self.p[f"Koff{r_}"]
        dst = self.A if r_ == "A" else self.B
        vr = p.vec_rows
        g = 64 // vr
        npairs = 64 * g
        row0 = b.mul(t, b.const_i32(R))
        nkt = K // 64
        U = 1
        for u in (8, 4, 2, 1):
            if u <= max(1, self.k.pack_unroll) and nkt % u == 0:
                U = u
                break
        for rb in range(R // 64):
            rbase = b.add(row0, b.const_i32(rb * 64))
            with b.scf_for(
                b.const_i32(0),
                b.const_i32(nkt // U),
                b.const_i32(1),
                iv_name=f"tk{r_}{rb}",
            ) as kbi:
                # U k-tiles of loads in flight, then U LDS transposes
                tiles = []
                for uu in range(U):
                    kb = b.add(b.mul(kbi, b.const_i32(64 * U)), b.const_i32(64 * uu))
                    vals = []
                    for i in range(npairs // 256):
                        pidx = b.add(tid, b.const_i32(i * 256))
                        rg = b.mod(pidx, b.const_i32(g))
                        kl = b.div(pidx, b.const_i32(g))
                        rr = b.mul(rg, b.const_i32(vr))
                        so = b.add(
                            b.global_load_i32(rowt, b.add(rbase, rr)),
                            b.global_load_i32(kt, b.add(kb, kl)),
                        )
                        vv = b.global_load_vN(src, so, self.dt, vr, align=2 * vr)
                        vals.append((rr, kl, vv))
                    tiles.append((kb, vals))
                for kb, vals in tiles:
                    for rr, kl, vv in vals:
                        for e in range(vr):
                            b.smem_store_vN(
                                self.tsm,
                                [b.add(rr, b.const_i32(e)), kl],
                                b.vec_extract(vv, e),
                                1,
                            )
                    b.sync()
                    for i in range(512 // 256):
                        q = b.add(tid, b.const_i32(i * 256))
                        rq = b.div(q, b.const_i32(8))
                        kc = b.mul(b.mod(q, b.const_i32(8)), b.const_i32(8))
                        v8 = b.smem_load_vN(self.tsm, rq, kc, dtype=self.dt, n=8)
                        off = b.add(
                            b.mul(b.add(rbase, rq), b.const_i32(K)), b.add(kb, kc)
                        )
                        b.global_store_vN(dst, off, v8, 8, align=16)
                    b.sync()

    def _emit_pack_r(self, b: IRBuilder, p, t, tid) -> None:
        """Register-transposing pack (row-contiguous source with vec_rows 8, no
        LDS, no barrier): a thread loads an 8 x 8 block (8 rows contiguous at
        each of 8 consecutive k), transposes it in registers and stores 8
        rows of 8 k (16 bytes each). Consecutive lanes take consecutive row
        groups, so the loads coalesce; the stores complete whole lines across
        the k-groups in L2."""
        K = self.plan.K
        r_ = p.gemm_role
        R = self.R[r_]
        src, rowt, kt = self.p[f"Src{r_}"], self.p[f"Row{r_}"], self.p[f"Koff{r_}"]
        dst = self.A if r_ == "A" else self.B
        Kp = self.Kp
        rg_n = R // 8
        units = rg_n * (Kp // 8)
        iters = -(-units // 256)
        u = max(1, min(self.k.pack_unroll, iters))
        row0 = b.mul(t, b.const_i32(R))
        with b.scf_for(
            b.const_i32(0),
            b.const_i32(-(-iters // u)),
            b.const_i32(1),
            iv_name=f"rk{r_}",
        ) as it:
            blocks = []
            for uu in range(u):
                q = b.add(
                    b.mul(
                        b.add(b.mul(it, b.const_i32(u)), b.const_i32(uu)),
                        b.const_i32(256),
                    ),
                    tid,
                )
                ok = b.cmp_lt(q, b.const_i32(units)) if units % (256 * u) else None
                qs = b.select(ok, q, b.const_i32(0)) if ok is not None else q
                rg = b.mod(qs, b.const_i32(rg_n))
                kg = b.div(qs, b.const_i32(rg_n))
                r0 = b.add(row0, b.mul(rg, b.const_i32(8)))
                k0 = b.mul(kg, b.const_i32(8))
                kin = b.cmp_lt(k0, b.const_i32(K)) if Kp != K else None
                ksel = b.select(kin, k0, b.const_i32(0)) if kin is not None else k0
                rb = b.global_load_i32(rowt, r0)
                ins = []
                for j in range(8):
                    so = b.add(rb, b.global_load_i32(kt, b.add(ksel, b.const_i32(j))))
                    ins.append(b.global_load_vN(src, so, self.dt, 8, align=16))
                blocks.append((r0, k0, ins, ok, kin))
            for r0, k0, ins, ok, kin in blocks:
                el = [[b.vec_extract(ins[j], i) for i in range(8)] for j in range(8)]
                if kin is not None:
                    z = b.cast_f32_to(b.const_f32(0.0), self.dt)
                    el = [[b.select(kin, x, z) for x in row] for row in el]

                def _st(r0=r0, k0=k0, el=el):
                    for i in range(8):
                        vec = b.vec_pack([el[j][i] for j in range(8)], self.dt)
                        off = b.add(
                            b.mul(b.add(r0, b.const_i32(i)), b.const_i32(Kp)), k0
                        )
                        b.global_store_vN(dst, off, vec, 8, align=16)

                if ok is None:
                    _st()
                else:
                    with b.scf_if(ok):
                        _st()

    # ------------------------------------------------------------ stage 1
    def emit_pack_tasks(self, b: IRBuilder, wg: Value, nwg: Value, tid: Value) -> None:
        K = self.plan.K
        kc8 = K // 8
        threads = 256
        c0, c1 = b.const_i32(0), b.const_i32(1)
        epoch = self.p["epoch"]
        for p in self.plan.packs:
            r = p.gemm_role
            R = self.R[r]
            chunks = R * kc8
            iters = -(-chunks // threads)
            u = max(1, min(self.k.pack_unroll, iters))
            src, rowt, kt = self.p[f"Src{r}"], self.p[f"Row{r}"], self.p[f"Koff{r}"]
            dst = self.A if r == "A" else self.B
            v = p.vec
            nt = b.const_i32(self.ntask[r])
            with b.scf_for(wg, nt, nwg, iv_name=f"pk{r}") as t:
                row0 = b.mul(t, b.const_i32(R))
                for i0 in (range(0, iters, u) if p.mode == "k" else ()):
                    loaded = []
                    for i in range(i0, min(i0 + u, iters)):
                        c = b.add(tid, b.const_i32(i * threads))
                        guard = (i + 1) * threads > chunks
                        rr = b.div(c, b.const_i32(kc8))
                        kc = b.mod(c, b.const_i32(kc8))
                        if guard:
                            ok = b.cmp_lt(c, b.const_i32(chunks))
                            rr = b.select(ok, rr, c0)
                            kc = b.select(ok, kc, c0)
                        grow = b.add(row0, rr)
                        k0 = b.mul(kc, b.const_i32(8))
                        base = b.global_load_i32(rowt, grow)
                        parts = []
                        for j in range(8 // v):
                            ko = b.global_load_i32(kt, b.add(k0, b.const_i32(j * v)))
                            off = b.add(base, ko)
                            if v == 1:
                                parts.append(
                                    [
                                        (
                                            b.global_load_f16(src, off)
                                            if self.dt == F16
                                            else b.global_load(
                                                src, off, self.dt, align=2
                                            )
                                        )
                                    ]
                                )
                            else:
                                vv = b.global_load_vN(src, off, self.dt, v, align=2 * v)
                                parts.append([b.vec_extract(vv, e) for e in range(v)])
                        flat = [x for ps in parts for x in ps]
                        vec = b.vec_pack(flat, self.dt)
                        dst_off = b.add(b.mul(grow, b.const_i32(K)), k0)
                        loaded.append((vec, dst_off, ok if guard else None))
                    for vec, dst_off, ok in loaded:
                        if ok is None:
                            b.global_store_vN(dst, dst_off, vec, 8, align=16)
                        else:
                            with b.scf_if(ok):
                                b.global_store_vN(dst, dst_off, vec, 8, align=16)
                if p.mode == "t" and p.vec_rows == 8 and self.k.reg_transpose:
                    self._emit_pack_r(b, p, t, tid)
                elif p.mode == "t":
                    self._emit_pack_t(b, p, t, tid)
                # publish: every thread's stores visible at agent scope, then one counter / flag
                b.fence(scope="agent", ordering="release")
                b.sync()
                with b.scf_if(b.cmp_eq(tid, c0)):
                    if self.k.dep == "all":
                        b.global_atomic_add(
                            self.p["Flags"], b.const_i32(self.cnt_all), c1
                        )
                    else:
                        b.global_flag_store(
                            self.p["Flags"],
                            b.add(b.const_i32(self.flag_base[r]), t),
                            epoch,
                            ordering="monotonic",
                        )

    def pre_tasks(self, b: IRBuilder, *, wg, nwg, tid) -> None:
        if self.k.pre_tasks and self.plan.packs:
            self.emit_pack_tasks(b, wg, nwg, tid)

    # ------------------------------------------------------------ stage 2 deps
    def wait_tile(
        self, b: IRBuilder, bm: Value, bn: Value, valid: Value, *, tid
    ) -> None:
        if not self.plan.packs or self.k.dep == "none":
            return
        Fl, epoch = self.p["Flags"], self.p["epoch"]
        if self.k.dep == "all":
            if not self._first_wait:
                return
            self._first_wait = False
            target = b.mul(epoch, b.const_i32(self.n_pack_tasks))
            b.global_flag_wait_eq(
                Fl, b.const_i32(self.cnt_all), target, ordering="monotonic"
            )
            b.fence(scope="agent", ordering="acquire")
            return
        with b.scf_if(valid):
            for p in self.plan.packs:
                r = p.gemm_role
                R = self.R[r]
                org = bm if r == "A" else bn
                tl = b.div(org, b.const_i32(R))
                hi = b.smin(
                    b.add(org, b.const_i32(self.tile_m if r == "A" else self.tile_n)),
                    b.const_i32(p.rows),
                )
                th = b.div(hi, b.const_i32(R))
                self._nw = getattr(self, "_nw", 0) + 1
                with b.scf_for(tl, th, b.const_i32(1), iv_name=f"wt{r}{self._nw}") as t:
                    b.global_flag_wait_eq(
                        Fl,
                        b.add(b.const_i32(self.flag_base[r]), t),
                        epoch,
                        ordering="monotonic",
                    )
            b.fence(scope="agent", ordering="acquire")

    # ------------------------------------------------------------ epilogue
    def epilogue(
        self,
        b: IRBuilder,
        *,
        accf,
        row0,
        col0,
        mfmas_m,
        mfmas_n,
        c_per_lane,
        C,
        M,
        N,
        block_m_off,
        block_n_off,
        warp_n_idx,
        lane,
    ) -> None:
        """Whole-tile store hook of the row-interleave vector epilogue: lane
        (q, cn) holds rows ``row0 + v * mfmas_m + mi`` x columns ``col0 + ni``
        as f32 accumulators ``accf[mi * mfmas_n + ni][v]``. Rows are handed
        to :meth:`store_row8` 8 consecutive columns at a time."""
        if mfmas_n % 8:
            raise ValueError(
                f"persistent contraction epilogue needs mfmas_n % 8 == 0, got {mfmas_n}"
            )
        self.begin_tile_store(
            b,
            block_m_off=block_m_off,
            block_n_off=block_n_off,
            warp_n_idx=warp_n_idx,
            lane=lane,
            col0=col0,
        )
        for mi in range(mfmas_m):
            for v in range(c_per_lane):
                c_m = b.add(row0, b.const_i32(v * mfmas_m + mi))
                for n0 in range(0, mfmas_n, 8):
                    vals = [
                        b.vec_extract(accf[mi * mfmas_n + ni], v)
                        for ni in range(n0, n0 + 8)
                    ]
                    self.store_row8(b, c_m, b.add(col0, b.const_i32(n0)), vals, C)
        self.end_tile_store(b)

    def begin_tile_store(
        self, b: IRBuilder, *, block_m_off, block_n_off, warp_n_idx, lane, col0
    ) -> None:
        vd = self.plan.vec_d
        self._vd = vd
        self._q = []
        self._bm, self._bn = block_m_off, block_n_off
        Nn = self.plan.N
        self._cols = []
        for j in range(8 // vd):
            cn = b.add(col0, b.const_i32(j * vd))
            ragged = Nn % self.tile_n != 0
            if ragged:
                ok = b.cmp_lt(cn, b.const_i32(Nn))
                off = b.global_load_i32(
                    self.p["DCol"], b.select(ok, cn, b.const_i32(0))
                )
            else:
                ok = None
                off = b.global_load_i32(self.p["DCol"], cn)
            self._cols.append((off, ok))
        if self.k.reduce:
            # partial slot of this (tile_n, warp_n) column block
            tn = b.div(block_n_off, b.const_i32(self.tile_n))
            self._part = b.add(b.mul(tn, b.const_i32(self.warp_n)), warp_n_idx)
            self._lane_cn = b.mod(lane, b.const_i32(16))

    def store_row8(
        self, b: IRBuilder, c_m: Value, c_n: Value, vals: Sequence[Value], C: Value
    ) -> None:
        """Queue one row (8 columns); rows are emitted in groups of
        ``epi_group``: all row-offset (and C) loads of a group first, then the
        math and the stores (the loads' latency overlaps)."""
        self._q.append((c_m, list(vals), C))
        if len(self._q) >= max(1, self.k.epi_group):
            self._flush_rows(b)

    def _flush_rows(self, b: IRBuilder) -> None:
        q, self._q = self._q, []
        pre = [self._row_loads(b, c_m) for (c_m, _v, _C) in q]
        for (c_m, vals, C), pl in zip(q, pre):
            self._row_math_store(b, c_m, vals, C, pl)

    def _row_loads(self, b: IRBuilder, c_m: Value):
        vd = self._vd
        Mm = self.plan.M
        row_ragged = Mm % self.tile_m != 0
        row_ok = b.cmp_lt(c_m, b.const_i32(Mm)) if row_ragged else None
        rsel = b.select(row_ok, c_m, b.const_i32(0)) if row_ragged else c_m
        roff = b.global_load_i32(self.p["DRow"], rsel)
        cvs = []
        if self.k.bilinear:
            for j, (coff, ok) in enumerate(self._cols):
                off = b.add(roff, coff)
                if ok is not None:
                    off = b.select(ok, off, roff)
                if vd > 1:
                    cv = b.global_load_vN(self.p["Cin"], off, self.dt, vd, align=2 * vd)
                    cvs.append([b.vec_extract(cv, e) for e in range(vd)])
                else:
                    cvs.append([b.global_load(self.p["Cin"], off, self.dt, align=2)])
        return row_ok, roff, cvs

    def _row_math_store(self, b: IRBuilder, c_m: Value, vals, C: Value, pl) -> None:
        row_ok, roff, cvs = pl
        vd = self._vd
        alpha = self.p["alpha"]
        out = [b.fmul(alpha, x) for x in vals]
        if self.k.bilinear:
            beta = self.p["beta"]
            for j in range(len(self._cols)):
                for e in range(vd):
                    out[j * vd + e] = b.fma(
                        beta, b.cast_to_f32(cvs[j][e]), out[j * vd + e]
                    )
        if self.k.store_d:
            # partial-line stores (vd < nt_min_vec) stay cached so L2 merges them
            th = (
                TemporalHint.STREAMING
                if (self.k.store_nt and vd >= self.k.nt_min_vec)
                else TemporalHint.DEFAULT
            )

            def _do_stores():
                for j, (coff, ok) in enumerate(self._cols):
                    off = b.add(roff, coff)
                    if vd == 1:
                        h = b.cast_f32_to(out[j], self.dt)
                        if ok is None:
                            b.global_store(C, off, h, align=2)
                        else:
                            with b.scf_if(ok):
                                b.global_store(C, off, h, align=2)
                        continue
                    vec = b.vec_pack(
                        [b.cast_f32_to(out[j * vd + e], self.dt) for e in range(vd)],
                        self.dt,
                    )
                    if ok is None:
                        b.global_store_vN(
                            C, off, vec, vd, align=2 * vd, temporal_hint=th
                        )
                    else:
                        with b.scf_if(ok):
                            b.global_store_vN(
                                C, off, vec, vd, align=2 * vd, temporal_hint=th
                            )

            if row_ok is None:
                _do_stores()
            else:
                with b.scf_if(row_ok):
                    _do_stores()
        if self.k.reduce == "max":
            ninf = b.const_f32(-3.0e38)
            m = None
            for j, (coff, ok) in enumerate(self._cols):
                for e in range(vd):
                    x = out[j * vd + e]
                    if ok is not None:
                        x = b.select(ok, x, ninf)
                    m = x if m is None else b.fmax(m, x)
            for sh in (1, 2, 4, 8):
                m = b.fmax(m, b.warp_shuffle_xor(m, sh))
            # lane cn == 0 of each 16-lane row holds the row's max over this warp's columns
            idx = b.add(b.mul(self._part, b.const_i32(self.M_pad)), c_m)
            with b.scf_if(b.cmp_eq(self._lane_cn, b.const_i32(0))):
                b.global_store(self.p["RedPart"], idx, m, align=4)

    def end_tile_store(self, b: IRBuilder) -> None:
        if self._q:
            self._flush_rows(b)

    # ------------------------------------------------------------ stage 3
    def post_tile(self, b: IRBuilder, bm: Value, bn: Value, *, tid) -> None:
        if not self.k.reduce:
            return
        c0, c1 = b.const_i32(0), b.const_i32(1)
        Fl, epoch = self.p["Flags"], self.p["epoch"]
        b.fence(scope="agent", ordering="release")
        b.sync()
        pm = b.div(bm, b.const_i32(self.tile_m))
        with b.scf_if(b.cmp_eq(tid, c0)):
            old = b.global_atomic_add(Fl, b.add(b.const_i32(self.red_base), pm), c1)
            last = b.cmp_eq(b.add(old, c1), b.mul(epoch, b.const_i32(self.tiles_n)))
            b.smem_store_vN(self.slot, [c0], b.select(last, c1, c0), 1)
        b.sync()
        flag = b.smem_load_vN(self.slot, c0, dtype=I32, n=1)
        if getattr(flag.type, "count", None) is not None:
            flag = b.vec_extract(flag, 0)
        with b.scf_if(b.cmp_eq(flag, c1)):
            b.fence(scope="agent", ordering="acquire")
            row = b.add(bm, tid)  # 256 threads = 256 rows of the panel
            ok = b.cmp_lt(row, b.const_i32(self.plan.M))
            with b.scf_if(ok):
                acc = b.const_f32(-3.0e38)
                for q in range(self.n_part):
                    x = b.global_load_f32(
                        self.p["RedPart"], b.add(b.const_i32(q * self.M_pad), row)
                    )
                    acc = b.fmax(acc, x)
                b.global_store(self.p["Red"], row, b.cast_f32_to(acc, self.dt), align=2)
        b.sync()

    def post_tasks(self, b: IRBuilder, *, wg, nwg, tid) -> None:
        pass

    # ------------------------------------------------------------ host model
    def task_table(self, grid: int = _GRID):
        from rocke.helpers.task_schedule import Dep, Stage, TaskTable

        st = []
        for p in self.plan.packs:
            st.append(
                Stage(
                    f"pack_{p.gemm_role}",
                    self.ntask[p.gemm_role],
                    reads=(f"src_{p.name}",),
                    writes=(f"ws_{p.gemm_role}",),
                )
            )
        gemm_reads = tuple(f"ws_{p.gemm_role}" for p in self.plan.packs)
        st.append(
            Stage(
                "gemm",
                self.tiles_m * self.tiles_n,
                deps=tuple(Dep(s.name, full=False, covers=True) for s in st),
                reads=gemm_reads,
                writes=("D", "redpart"),
            )
        )
        if self.k.reduce:
            st.append(
                Stage(
                    "reduce",
                    self.tiles_m,
                    deps=(Dep("gemm", full=False, covers=True),),
                    reads=("redpart",),
                    writes=("R",),
                )
            )
        ins = tuple(f"src_{p.name}" for p in self.plan.packs)
        return TaskTable(
            stages=tuple(st),
            grid=grid,
            inputs=ins,
            kind="hiptensor_persistent_contraction",
        )


# ---------------------------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------------------------


def persistent_body_spec(dtype: str = "f16", **trait_over):
    """The persistent universal-GEMM body the monokernel runs: 256 x 256 x 64
    tile, 2 x 2 warps of 16x16x32 MFMAs, DTL ring (2 stages, prefetch,
    fragment prefetch, hoisted addressing), AGPR accumulators, merged MFMA
    interleave, LDS row interleave with the vector epilogue (the hook point),
    chiplet swizzle (wgm 8, chunk 32) and a persistent grid of 256; plus
    ``TraitSpec`` overrides."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    tile = dict(
        tile_m=_TILE_M,
        tile_n=_TILE_N,
        tile_k=64,
        warp_m=2,
        warp_n=_WARP_N,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_num_xcds=8,
        chiplet_chunk_size=32,
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=False,
        acc_agpr=True,
        dtl_frag_prefetch=True,
        mfma_interleave=True,
        ilv_merge=True,
        dtl_hoist=True,
        dtl_stages=2,
        lds_row_interleave=True,
        rilv_vec_epilogue=True,
        rve_store_nt=True,
        persistent_ctas=_GRID,
    )
    trait.update(trait_over)
    dt = "fp16" if dtype in ("f16", "fp16") else dtype
    return UniversalGemmSpec(
        name="htmono",
        tile=TileSpec(**tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
    )


def _plan_hash(
    plan: ContractionPlan, knobs: PersistentContractionKnobs, extra: str = ""
) -> str:
    key = json.dumps(
        [
            plan.modes_a,
            plan.modes_b,
            plan.modes_d,
            plan.ext,
            dataclasses.asdict(knobs),
            extra,
        ],
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _body_traits(spec: PersistentContractionSpec) -> Dict[str, Any]:
    trait_over = dict(spec.body_traits)
    # ragged M / N: the DTL buffer resources cover exactly the operand rows
    trait_over.setdefault("dtl_tight_rsrc", True)
    return trait_over


def persistent_contraction_hooks(
    spec: PersistentContractionSpec,
) -> PersistentContractionHooks:
    """A fresh hooks object of ``spec`` for host-side launch sizes (``n_flags``,
    ``Kp``, ``n_part``, ``M_pad``); the builder uses its own instance."""
    return PersistentContractionHooks(spec.plan, spec.knobs, dtype=spec.dtype)


def build_persistent_contraction(
    spec: PersistentContractionSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """KernelDef of the monokernel (with ``knobs.pre_tasks=False``: the GEMM
    with the fused permuted epilogue, second launch after
    :func:`build_operand_pack`).

    Kernel params: ``A``, ``B`` (canonical workspaces, or the operands read in
    place), ``C`` (D), ``M``, ``N``, ``K`` (= hooks ``Kp``), then per pack
    ``Src*`` / ``Row*`` / ``Koff*``, ``DRow``, ``DCol``, [``Cin``, ``beta``],
    ``alpha``, ``Flags`` (``n_flags`` i32, zero before the first launch),
    ``epoch`` (1, 2, ... per launch), [``RedPart``, ``Red``]. Launch grid 256 x
    256 threads; all 256 workgroups must be co-resident.
    """
    from rocke.instances.common.gemm_universal import build_universal_gemm

    if arch != "gfx950":
        raise ValueError(
            f"the persistent contraction monokernel is gfx950-only, got {arch}"
        )
    k = spec.knobs
    if k.pre_tasks and k.dep == "none" and spec.plan.packs:
        raise ValueError(
            "dep='none' with in-launch pack tasks has no producer/consumer ordering"
        )
    trait_over = _body_traits(spec)
    body = persistent_body_spec(spec.dtype, **trait_over)
    body = dataclasses.replace(
        body,
        name=f"htmono_{_plan_hash(spec.plan, k, json.dumps(trait_over, sort_keys=True))}",
    )
    hooks = PersistentContractionHooks(spec.plan, k, dtype=spec.dtype)
    hooks.task_table().validate()
    object.__setattr__(body, "_mono", hooks)
    return build_universal_gemm(body, arch=arch)


def build_operand_pack(spec: OperandPackSpec, *, arch: str = "gfx950") -> KernelDef:
    """Stand-alone pack kernel (the first launch of the two-launch pipeline):
    the same task emitter, task ``t`` on workgroup ``t % nwg``.

    Kernel params: ``A``, ``B`` (workspaces), per pack ``Src*`` / ``Row*`` /
    ``Koff*``, ``Flags``, ``epoch``, ``nwg`` (= the launch grid)."""
    if arch != "gfx950":
        raise ValueError(f"the operand pack kernel is gfx950-only, got {arch}")
    plan, knobs = spec.plan, spec.knobs
    hooks = PersistentContractionHooks(
        plan, dataclasses.replace(knobs, dep="none", pre_tasks=True), dtype=spec.dtype
    )
    b = IRBuilder(f"htpack_{_plan_hash(plan, knobs)}")
    b.kernel.attrs["max_workgroup_size"] = 256
    dt = hooks.dt
    A = b.param("A", PtrType(dt, "global"), align=16)
    B = b.param("B", PtrType(dt, "global"), align=16)
    hooks.A, hooks.B = A, B
    hooks.p = {}
    for p in plan.packs:
        r = p.gemm_role
        hooks.p[f"Src{r}"] = b.param(
            f"Src{r}", PtrType(dt, "global"), readonly=True, align=16
        )
        hooks.p[f"Row{r}"] = b.param(
            f"Row{r}", PtrType(I32, "global"), readonly=True, align=4
        )
        hooks.p[f"Koff{r}"] = b.param(
            f"Koff{r}", PtrType(I32, "global"), readonly=True, align=4
        )
    hooks.p["Flags"] = b.param("Flags", PtrType(I32, "global"), align=4)
    hooks.p["epoch"] = b.param("epoch", I32)
    nwg = b.param("nwg", I32)
    hooks._alloc_tsm(b)
    tid = b.thread_id_x()
    wg = b.block_id_x()
    hooks.emit_pack_tasks(b, wg, nwg, tid)
    return b.kernel
