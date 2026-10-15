# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shell of the dense SDPA backward main kernel (family glue, Python-only).

The dense backward runs the shared backward body of the layout-general family
(:class:`~kernels.common._attention_bwd_body.BwdTileStep`, the dQ sink, the
stats view, the per-kv-tile driver and the dK / dV epilogues of
:mod:`kernels.common.attention_bwd`) with three dense-specific pieces:

* :class:`DenseFixedAddr`: compile-time element strides (contiguous BSHD /
  BHSD), the batch and head rebased in 64 bits, and, under ``oob =
  "buffer_oob"``, a buffer resource per (batch, head) slab whose
  ``num_records`` ends at the last element of the slab's last row: a row at
  or past the sequence end reads as zero (hardware range check) instead of
  being clamped, and the remaining voffset is a small i32 (checked at build
  time). ``oob = "clamp"`` clamps rows exactly like the general family.
  The Q / dO prefetch ring loads Q and dO through a global pointer (its DMA
  copies take a pointer and an i32 offset; its register loads select between
  the Q and dO pointers), so under ``buffer_oob`` those two tensors get the
  global-pointer view of :class:`DenseFixedAddr` (no row clamp in a
  non-ragged kernel, where no q row passes the sequence end; a ragged kernel
  clamps them as ``oob = "clamp"`` does); K and V keep their buffer
  resources.
* :class:`DiagonalBand` / :class:`DenseNoMask`: the mask policies of the
  general family over compile-time lengths and band bounds (every compare
  folds to constants), plus a compile-time classification over every kv tile:
  when no visited tile of any kv tile is an edge tile (unmasked, aligned
  lengths) the interior range is the whole visited range and the edge loops
  are not emitted at all; when every visited tile is an edge tile the
  interior loop is not emitted (``static_runs``, read by the per-kv-tile
  driver). Otherwise the edge / interior split of each kv tile is the
  general family's formula over constants.
* the Q / dO prefetch ring of the general shell
  (:class:`~kernels.common._attention_bwd_ring.QRing`: the register or DMA
  path at depth 2 or 3, and the DMA path at depth 1), built per kv tile from
  the same body geometry, LDS plan, stats view and dQ atomic count and handed
  to the shared per-kv-tile driver. The merged-barrier 2-slot ring
  (``barrier_merge = "on"``) stages inside the tile step, as in the general
  family.
* the CTA decode: one kv tile per CTA on the ``(n_kv_tiles, h_k * g_split,
  B)`` grid (with the runtime chiplet remap under ``xcd_remap``), or a
  persistent grid of ``num_persistent`` CTAs striding over the work items in
  kv-tile-major (kv tile, head unit, batch) or kv-head-major (kv tile, batch,
  head unit) order. The runtime load-balance order (natural / reverse, and
  interleave when the body's ``lb_interleave`` is set) permutes the kv tiles
  of one (unit, batch) in both. The chiplet remap is the general family's
  ``xcd_remap_id`` over the whole grid (the batch is a kernarg, so the grid
  size is known in the kernel).

The body knobs ``static_scale`` (multipliers that are 1.0 under the compiled
scale placement are not applied) and ``ptr_noalias`` (over the dense pointer
list) follow the general family's rules.

``dq_mode = "split"``: the main kernel computes dK / dV only (a ``NullSink``:
no G4, no dS tile, no dQ atomics, no ``WS_DQ`` access; the kernarg stays in
the ABI), as the general shell does; the prefetch ring's DMA wait then counts
no dQ atomics. dQ comes from the dense split dQ kernel
(:mod:`kernels.common.attention_dense_bwd_dq`).

Batched rows of dK / dV past the sequence (ragged kernels) are never stored
(the direct epilogue's store rule with ``S_kv_max = seqlen_kv``), and q rows
past the sequence are dead rows of an edge tile (``P = dS = 0``), as in the
general family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocke.core.codegen_policy import (
    SCHEDULER_STRATEGY_ATTR,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, IRBuilder, PtrType, Type, Value
from rocke.helpers.attention_band import AttnRuntimeBounds

from kernels.common._attention_bwd_addr import (
    StridedTensorAddr,
    WorkspaceAddr,
    as_i32,
    as_i64,
    decode_batched,
)
from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    static_kv_tile_plan,
)
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    StatsView,
)
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_ring import QRing
from kernels.common._attention_bwd_sinks import AtomicWorkspaceSink, NullSink
from kernels.common._attention_bwd_worklist import (
    decode_cta,
    head_range,
    xcd_remap_id,
)
from kernels.common.attention_bwd import (
    AGPR_ALLOC_BACKEND,
    BWD_BARRIER_FENCE,
    SPLIT_AGPR_ROUTES,
    _kv_tile,
    _merged_ring,
    attn_bwd_arch_facts,
    lds_plan,
    static_unit_mults,
)
from kernels.common.attention_dense_bwd_spec import (
    AttentionDenseBwdSpec,
    _not_built_all,
    attn_dense_bwd_params,
    validate_attn_dense_bwd_spec,
)

__all__ = [
    "BufferRowSlab",
    "DenseFixedAddr",
    "DenseNoMask",
    "DiagonalBand",
    "dense_static_classes",
    "emit_dense_bwd_main",
]

_ELEM_BYTES = 2
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64
_READONLY = frozenset("Q K V O dO LSE WS_LSE2 WS_DSUM".split())
_WORKSPACE_PTRS = frozenset("WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV".split())


# ---------------------------------------------------------------------------
# addressing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BufferRowSlab:
    """Rows of one (batch, head) slab read through a buffer resource.

    Rows are absolute sequence rows (the resource starts at row 0 of the
    head); ``load_vec`` of a row at or past the sequence end returns zeros
    (the resource's ``num_records`` ends at the last row). Only loads go
    through the resource; the dK / dV stores use the strided epilogue.
    """

    b: IRBuilder
    rsrc: Value
    dtype: Type
    t_stride: int

    def _voffset(self, row: Value, col) -> Value:
        b = self.b
        idx = b.mul(row, b.const_i32(self.t_stride))
        if not (isinstance(col, int) and col == 0):
            idx = b.add(idx, col if isinstance(col, Value) else b.const_i32(col))
        return b.mul(idx, b.const_i32(_ELEM_BYTES))

    def load_vec(self, row: Value, col0, n: int = 8) -> Value:
        """``n`` consecutive elements (``n`` a multiple of 8) of ``row``."""
        if n % 8:
            raise ValueError("buffer row loads move multiples of 8 elements")
        b = self.b
        zero = b.const_i32(0)
        parts = []
        for i in range(n // 8):
            col = col0 if i == 0 else b.add(as_i32(b, col0), b.const_i32(8 * i))
            parts.append(
                b.buffer_load_vN(
                    self.rsrc, self._voffset(row, col), zero, self.dtype, 8
                )
            )
        out = parts[0]
        for p in parts[1:]:
            out = b.vec_concat(out, p)
        return out

    def load(self, row: Value, col) -> Value:
        b = self.b
        return b.buffer_load(
            self.rsrc, self._voffset(row, col), b.const_i32(0), self.dtype
        )


class DenseFixedAddr:
    """A contiguous dense tensor of one batch: compile-time strides.

    ``strides`` are the ``(batch, head, token)`` element strides (ints);
    ``seq`` the batch's :class:`~kernels.common._attention_bwd_addr.SeqInfo`
    (compile-time length). ``slab(h, row0)`` returns a
    :class:`BufferRowSlab` (``oob = "buffer_oob"``) or the general family's
    clamped :class:`~kernels.common._attention_bwd_addr.RowSlab`
    (``oob = "clamp"``).

    ``global_view = True`` (Q and dO of a kernel with the prefetch ring,
    under ``buffer_oob``): ``slab`` returns a
    :class:`~kernels.common._attention_bwd_addr.RowSlab` over the head's
    global pointer instead of a buffer resource. Its rows are clamped only in
    a ``ragged`` kernel (exactly as ``oob = "clamp"``); in a non-ragged one
    every q tile lies inside the sequence.
    """

    def __init__(
        self,
        b: IRBuilder,
        ptr: Value,
        *,
        dtype: Type,
        strides: tuple[int, int, int],
        seq,
        seqlen: int,
        head_size: int,
        oob: str,
        global_view: bool = False,
        ragged: bool = True,
    ) -> None:
        self.b = b
        self.dtype = dtype
        self.strides = tuple(int(s) for s in strides)
        self.oob = oob
        sb, _sh, st = self.strides
        if oob not in ("buffer_oob", "clamp"):
            raise ValueError(f"oob={oob!r} not in ('buffer_oob', 'clamp')")
        # The global-pointer view of a buffer_oob tensor clamps only where a
        # row can pass the sequence end (a ragged kernel).
        self._clamp = oob == "clamp" or ragged
        if oob == "clamp" or global_view:
            self._strided = StridedTensorAddr(
                b, ptr, dtype=dtype, strides=self.strides, seq=seq, stage_vec=8
            )
            return
        self._strided = None
        off = b.mul(as_i64(b, seq.batch), b.const_i64(sb * _ELEM_BYTES))
        self._batch_ptr = b.global_ptr_add(ptr, off)
        # The slab's bytes under the compile-time shape: the hardware range
        # check reads zeros past them, so no load leaves the slab; the slab
        # itself must exist, which the host checks against the caller's
        # tensors (attention_bwd_run.check_attn_dense_bwd_tensors). The spec
        # validator bounds the in-slab offset below 2^31.
        self._num_records = ((seqlen - 1) * st + head_size) * _ELEM_BYTES
        if self._num_records >= 2**31:
            raise ValueError(f"num_records {self._num_records} does not fit i32")

    def slab(self, h, row0=0, *, clamp: bool = True):
        if self._strided is not None:
            return self._strided.slab(h, row0, clamp=clamp and self._clamp)
        b = self.b
        _sb, sh, st = self.strides
        ptr = self._batch_ptr
        if not (isinstance(h, int) and h == 0):
            off = b.mul(as_i64(b, h), b.const_i64(sh * _ELEM_BYTES))
            ptr = b.global_ptr_add(ptr, off)
        rsrc = b.buffer_rsrc(ptr, b.const_i32(self._num_records))
        return BufferRowSlab(b=b, rsrc=rsrc, dtype=self.dtype, t_stride=st)


# ---------------------------------------------------------------------------
# compile-time band classification
# ---------------------------------------------------------------------------


def dense_static_classes(
    *,
    s_q: int,
    s_kv: int,
    k_m0: int,
    k_n0: int,
    left: int,
    right: int,
    bottom_right: bool,
    masked: bool,
) -> tuple[bool, bool]:
    """``(any_edge, any_interior)`` over every visited (q tile, kv tile) pair."""
    any_edge = any_interior = False
    for kt in range(-(-s_kv // k_n0)):
        plan = static_kv_tile_plan(
            s_q=s_q, s_kv=s_kv, k0=kt * k_n0, k_m0=k_m0, k_n0=k_n0, left=left,
            right=right, bottom_right=bottom_right, masked=masked,
        )  # fmt: skip
        any_edge = any_edge or bool(plan.edge_tiles)
        any_interior = any_interior or bool(plan.interior_tiles)
        if any_edge and any_interior:
            break
    return any_edge, any_interior


class _StaticClassMixin:
    """``interior_q_range`` shortcuts decided at build time (see the module)."""

    def _set_static(self, any_edge: bool, any_interior: bool) -> None:
        self.any_edge = any_edge
        self.any_interior = any_interior

    def static_runs(self, edge: bool) -> bool:
        """Whether the edge (``edge``) or interior loop of a key tile can run
        (the per-kv-tile driver drops a loop proved empty at build time)."""
        return self.any_edge if edge else self.any_interior

    def interior_q_range(self, b, k0, qt_start, n_qt):
        if not self.any_edge:
            return qt_start, b.add(qt_start, n_qt)
        if not self.any_interior:
            return qt_start, qt_start
        return super().interior_q_range(b, k0, qt_start, n_qt)


class DiagonalBand(_StaticClassMixin, RuntimeBand):
    """The general band policy over compile-time lengths and bounds."""

    def __init__(
        self,
        b: IRBuilder,
        *,
        s_q: int,
        s_kv: int,
        left: int,
        right: int,
        bottom_right: bool,
        k_m0: int,
        k_n0: int,
    ) -> None:
        off = s_kv - s_q if bottom_right else 0
        bounds = AttnRuntimeBounds(s_q, s_kv, off, left, right)
        super().__init__(b, bounds, k_m0=k_m0, k_n0=k_n0)
        self._set_static(
            *dense_static_classes(
                s_q=s_q, s_kv=s_kv, k_m0=k_m0, k_n0=k_n0, left=left, right=right,
                bottom_right=bottom_right, masked=True,
            )
        )  # fmt: skip


class DenseNoMask(_StaticClassMixin, NoMaskKTail):
    """No mask over compile-time lengths: only the length tails are edges."""

    def __init__(
        self, b: IRBuilder, *, s_q: int, s_kv: int, k_m0: int, k_n0: int
    ) -> None:
        super().__init__(b, s_q, s_kv, k_m0=k_m0, k_n0=k_n0)
        self._set_static(
            *dense_static_classes(
                s_q=s_q, s_kv=s_kv, k_m0=k_m0, k_n0=k_n0, left=-1, right=-1,
                bottom_right=False, masked=False,
            )
        )  # fmt: skip


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------


def _noalias(ptr_noalias) -> frozenset:
    """Dense kernargs marked ``noalias`` under the body's ``ptr_noalias``
    (the general family's rule over the dense pointer list)."""
    if ptr_noalias is False:
        return frozenset()
    if ptr_noalias == "workspace":
        return _WORKSPACE_PTRS
    return frozenset(n for n, fmt in attn_dense_bwd_params() if fmt == "Q")


def _declare(b: IRBuilder, io, noalias: frozenset = frozenset()) -> dict:
    args = {}
    for name, fmt in attn_dense_bwd_params():
        if fmt == "Q":
            if name in ("Q", "K", "V", "O", "dO", "dK", "dV"):
                t = PtrType(io, "global")
            else:
                t = PtrType(F32, "global")
            attrs = {"readonly": True} if name in _READONLY else {}
            if name in noalias:
                attrs["noalias"] = True
            args[name] = b.param(name, t, **attrs)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


def emit_dense_bwd_main(spec: AttentionDenseBwdSpec, arch: str, *, sentinel=True):
    """Emit the dense backward main kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term of the dS select (negative
    controls only, never a spec field).
    """
    body = validate_attn_dense_bwd_spec(spec, arch)
    missing = _not_built_all(spec, body)
    if missing:
        raise NotImplementedError(
            "the dense attention backward main kernel is not built for "
            + ", ".join(missing)
        )
    facts = attn_bwd_arch_facts(arch)
    geom = BwdTileGeometry.from_spec(body, arch)
    plan = lds_plan(body, arch)
    io = ir_type(body.dtype)
    km, kn = body.block_m, body.block_n
    s_q, s_kv = spec.seqlen_q, spec.seqlen_kv
    h_q, h_k = spec.num_query_heads, spec.num_kv_heads
    group = h_q // h_k
    n_kv_tiles = math.ceil(s_kv / kn)
    atomic = spec.dkv_mode == "atomic"
    split = spec.dq_mode == "split"
    # The Q / dO prefetch ring, by the general shell's rule (_cta_setup):
    # every staging but the synchronous (vgpr, 1) one, unless the merged-
    # barrier 2-slot ring stages the q steps instead.
    staged = (body.global_path, body.ring_depth) != ("vgpr", 1)
    ringed = staged and not _merged_ring(body)

    b = IRBuilder(spec.kernel_name())
    # the general family's barrier fence (the same tile step, staging and
    # epilogue barriers)
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, io, _noalias(body.ptr_noalias))
    # Multipliers that are 1.0 under the compiled scale placement are not
    # applied (the general family's rule; the kernargs stay in the ABI).
    a.update(dict.fromkeys(static_unit_mults(body.static_scale, body.dkv_mode)))
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if facts.matrix_path == "mfma" and body.agpr_alloc not in (
        None,
        AGPR_ALLOC_BACKEND,
    ):
        b.kernel.attrs["agpr_alloc"] = tuple(body.agpr_alloc)
    if body.agpr_route in SPLIT_AGPR_ROUTES:
        b.kernel.attrs["mfma_vgpr_form"] = True
    if body.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(body.waves_per_eu)
    if body.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            body.scheduler_strategy
        )

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    c = b.const_i32
    batch = c(spec.batch) if spec.batch is not None else a["batch"]
    g_split = a["g_split"] if atomic else None
    gx = c(n_kv_tiles)
    gy = b.mul(c(h_k), g_split) if atomic else c(h_k)
    left, right, br = spec.band()
    if spec.masked:
        mask = DiagonalBand(
            b, s_q=s_q, s_kv=s_kv, left=left, right=right, bottom_right=br,
            k_m0=km, k_n0=kn,
        )  # fmt: skip
    else:
        mask = DenseNoMask(b, s_q=s_q, s_kv=s_kv, k_m0=km, k_n0=kn)
    lds = BwdLds(b, plan, body.dtype)
    rows_q = b.mul(batch, c(s_q))
    rows_kv = b.mul(batch, c(s_kv))
    # Arguments of the shared per-kv-tile driver and epilogues (compile-time
    # strides and lengths, runtime pointers and multipliers).
    sb = {}
    for t, prefix in (("dk", "dk"), ("dv", "dv")):
        sb.update(
            dict(zip((f"{prefix}_b", f"{prefix}_h", f"{prefix}_t"), spec.strides(t)))
        )
    shared = dict(
        sb,
        dK=a["dK"],
        dV=a["dV"],
        dk_mult=a["dk_mult"],
        S_kv_max=c(s_kv),
        WS_DK=a["WS_DK"],
        WS_DV=a["WS_DV"],
        ws_rows_kv=rows_kv,
        h_k=c(h_k),
        h_v=c(h_k),
    )

    def tile(x, y, z) -> None:
        work = decode_cta(
            b, x=x, y=y, z=z, n_kv_tiles=gx, lb_order=a["lb_order"], g_split=g_split,
            interleave=body.lb_interleave,
        )  # fmt: skip
        bz = work.batch
        seq_q = decode_batched(b, bz, s_max=s_q)
        seq_kv = decode_batched(b, bz, s_max=s_kv)
        heads = head_range(
            b, unit=work.unit, split=work.split, group=c(group), gk=c(group),
            gv=c(group), g_split=g_split,
        )  # fmt: skip

        def tensor(name, role, seq, seqlen, *, global_view=False):
            return DenseFixedAddr(
                b, a[name], dtype=io, strides=spec.strides(role), seq=seq,
                seqlen=seqlen, head_size=spec.head_size, oob=spec.oob,
                global_view=global_view, ragged=spec.ragged,
            )  # fmt: skip

        # the prefetch ring reads Q / dO through global pointers
        q_addr = tensor("Q", "q", seq_q, s_q, global_view=ringed)
        do_addr = tensor("dO", "do", seq_q, s_q, global_view=ringed)
        k_addr = tensor("K", "k", seq_kv, s_kv)
        v_addr = tensor("V", "v", seq_kv, s_kv)
        nh = c(h_q) if spec.ws_layout == "token_major" else None

        def ws(name, width):
            return WorkspaceAddr(
                b, a[name], ws_rows=rows_q, width=width, row0=seq_q.ws_row0,
                layout=spec.ws_layout, n_heads=nh,
            )  # fmt: skip

        # stats_vec > 1 reads <n x f32> windows kept inside the head-major
        # arrays (StatsView needs their query-head count, as in the general
        # shell)
        stats = StatsView(
            ws("WS_LSE2", 1), ws("WS_DSUM", 1),
            n_heads=c(h_q) if body.stats_vec != 1 else None,
        )  # fmt: skip
        if split:
            # dQ comes from the dense Q-parallel dQ kernel: no G4, no dS tile,
            # no WS_DQ access (the kernarg stays in the ABI, unused)
            sink = NullSink()
        else:
            sink = AtomicWorkspaceSink(
                b, ws("WS_DQ", spec.head_size), seq=seq_q, scale=a["dq_mult"]
            )
        step = BwdTileStep(
            b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid,
            lane=lane, wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
            sentinel=sentinel,
        )  # fmt: skip
        ring = None
        if ringed:
            # The general shell's ring on the same geometry, LDS plan and
            # stats view; the DMA wait counts the step's dQ atomics (none
            # under split dQ, whose main kernel emits no G4).
            n_atomics = km * spec.head_size // (geom.waves * geom.wave_size)
            ring = QRing(
                b, geom, lds, tid=tid, lane=lane, wave=wave, stats=stats,
                n_atomics=0 if split else n_atomics,
            )  # fmt: skip
        _kv_tile(
            b, body, shared, step, mask, kt=work.first_kt, hk=heads.hk, hv=heads.hv,
            h0=heads.h0, n_h=heads.n_heads, seq_q=seq_q, seq_kv=seq_kv,
            q_addr=q_addr, do_addr=do_addr, k_addr=k_addr, v_addr=v_addr, io=io,
            tid=tid, ring=ring,
        )  # fmt: skip

    def split_linear(lin, *, unit_major_last: bool):
        """``(x, y, z)`` of a linear work id, kv tile fastest."""
        t = b.mod(lin, gx)
        r = b.div(lin, gx)
        if unit_major_last:  # kv-head-major: (kv tile, batch, unit)
            return t, b.div(r, batch), b.mod(r, batch)
        return t, b.mod(r, gy), b.div(r, gy)

    if spec.persistent:
        total = b.readfirstlane(b.mul(b.mul(gx, gy), batch))
        stride = a["num_persistent"]
        start = b.block_id_x()
        one, zero = c(1), c(0)
        left_items = b.sub(total, start)
        trips = b.div(b.add(left_items, b.sub(stride, one)), stride)
        trips = b.readfirstlane(b.select(b.cmp_lt(start, total), trips, zero))
        loop = b.scf_for_iter(
            zero, trips, one, [("dense_item", start)], iv_name="dense_it",
            elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (_it, (item,)):
            b.sync()  # the previous item's LDS reads finish before staging
            hkv_major = spec.persist_decode == "hkv_major"
            tile(*split_linear(item, unit_major_last=hkv_major))
            b.scf_yield(b.add(item, stride))
    elif spec.xcd_remap:
        x, y, z = b.block_id_x(), b.block_id_y(), b.block_id_z()
        lin = b.add(x, b.mul(gx, b.add(y, b.mul(gy, z))))
        # whole-grid remap (the batch is a kernarg, so the grid size is
        # known); identity when xcd_n or xcd_chunk is 0
        lin = b.readfirstlane(
            xcd_remap_id(
                b, lin, num_wgs=b.mul(b.mul(gx, gy), batch), n=a["xcd_n"],
                chunk=a["xcd_chunk"],
            )
        )  # fmt: skip
        tile(*split_linear(lin, unit_major_last=False))
    else:
        tile(b.block_id_x(), b.block_id_y(), b.block_id_z())
    b.ret()
    return b.kernel
