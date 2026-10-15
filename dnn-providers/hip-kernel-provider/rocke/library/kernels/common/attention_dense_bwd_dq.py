# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split dQ kernel of the dense SDPA backward (``dq_mode = "split"`` on the
dense route).

Under ``dq_mode = "split"`` the dense main kernel computes dK / dV only (a
``NullSink``: no G4, no dS tile, no dQ atomics, no dQ workspace, no dQ zero
fill in prep and no dQ convert) and this Q-parallel kernel produces dQ. It is
the layout-general split dQ kernel of :mod:`kernels.common.attention_bwd_dq`
(the same body, :class:`~kernels.common._attention_bwd_dqbody.BwdDqStep`, so
the same per-element operations in the same key order) behind the dense
family's addressing:

* compile-time sequence lengths, head counts, head group and element strides
  (contiguous BSHD / BHSD), the batch a runtime kernarg (grid z) unless the
  spec pins it;
* ``oob = "buffer_oob"`` (the dense spec's default): K and V (and Q / dO of a
  one-head CTA) are read through a buffer resource per (batch, head) slab
  whose ``num_records`` ends at the slab's last row, so a row at or past the
  sequence end reads as zero; ``oob = "clamp"`` clamps rows as the general
  family does. Q / dO of a head-packed CTA (``head_loop = "group"``: a head
  per row) use the global-pointer view (rows clamped only where a q block can
  pass the sequence end). dQ is stored through the compile-time strided
  pointer;
* the mask policies of the dense shell (every band compare over constants)
  and a compile-time classification of every (q block, key tile) pair: with
  ``edge_tiles`` the edge key loops are not emitted when no visited tile of
  any q block is an edge tile (unmasked aligned lengths), and the interior
  loop is not emitted when none is interior; otherwise the edge / interior /
  edge loops of the general kernel run over constants.

Its kernargs are the ``rocke.attn_dense_bwd_dq.v1`` ABI
(:func:`attn_dense_bwd_dq_params`). The tuning values are the general dQ
kernel's (:class:`~kernels.common.attention_bwd_dq.AttnBwdDqSpec`, the
plan's ``dq_*`` knobs), carried in :attr:`AttentionDenseBwdDqSpec.dq`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from rocke.core.codegen_policy import (
    SCHEDULER_STRATEGY_ATTR,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.helpers.attention_band import AttnRuntimeBounds, BandEmitter
from rocke.helpers.spec import salted_kernel_name

from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    decode_batched,
)
from kernels.common._attention_bwd_band import HostInt
from kernels.common._attention_bwd_body import StatsView
from kernels.common._attention_bwd_dqbody import (
    BwdDqGeometry,
    BwdDqStep,
    interior_kv_range,
    kv_tile_range,
)
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_worklist import xcd_remap_id
from kernels.common._attention_dense_bwd_shell import (
    DenseFixedAddr,
    DenseNoMask,
    DiagonalBand,
)
from kernels.common.attention_bwd import BWD_BARRIER_FENCE, attn_bwd_arch_facts
from kernels.common.attention_bwd_dq import (
    DQ_AGPR_ALLOC_BACKEND,
    AttnBwdDqSpec,
    _lds_views,
    attn_bwd_dq_grid,
    dq_lds_plan,
    dq_rows_per_head,
    dq_static_unit_mults,
    validate_attn_bwd_dq_spec,
)
from kernels.common.attention_dense_bwd_spec import (
    DENSE_BWD_ARCHS,
    AttentionDenseBwdSpec,
    dense_bwd_band,
    dense_layout_strides,
)

__all__ = [
    "ATTN_DENSE_BWD_DQ_ABI",
    "AttentionDenseBwdDqSpec",
    "attn_dense_bwd_dq_grid",
    "attn_dense_bwd_dq_params",
    "build_attention_dense_bwd_dq",
    "dense_dq_static_classes",
    "validate_attn_dense_bwd_dq_spec",
]

ATTN_DENSE_BWD_DQ_ABI = "rocke.attn_dense_bwd_dq.v1"
_ELEM_BYTES = 2
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64
# Largest in-slab byte offset of a buffer voffset / an i32 row index: rows up
# to one tile past the sequence end (the key prefetch reads one tile ahead).
_VOFFSET_LIMIT = 2**31
_TILE_SLACK_ROWS = 256
# Problem fields of the dense dQ spec (the dense main spec's own names).
_PROBLEM = (
    "seqlen_q",
    "seqlen_kv",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "batch",
    "causal",
    "causal_bottom_right",
    "dtype",
    "layout",
    "sliding_window",
    "window_left",
    "window_right",
)
# dQ spec fields that are problem fields here (they follow the dense spec).
_DQ_PROBLEM = ("head_size", "dtype", "seq_mode", "stage_vec", "mask_class")


@dataclass(frozen=True)
class AttentionDenseBwdDqSpec:
    """Compile-time spec of the dense split dQ kernel.

    Problem fields (the dense main spec's: lengths, heads, head size, batch,
    mask, dtype, layout) are required where the dense spec requires them;
    ``oob`` and ``ws_layout`` follow the dense main spec of the plan; ``dq``
    holds the general dQ kernel's tuning values (``None``: its defaults). The
    dQ spec's problem fields (head size, dtype, sequence mode, ``stage_vec``,
    mask class, workspace layout) must match this problem.
    """

    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    batch: Optional[int] = None
    causal: bool = True
    causal_bottom_right: bool = False
    dtype: str = "bf16"
    layout: str = "bshd"
    sliding_window: int = 0
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    oob: str = "buffer_oob"
    ws_layout: str = "head_major"
    dq: Optional[AttnBwdDqSpec] = None
    name: str = "rocke_attn_dense_bwd"

    def __post_init__(self) -> None:
        # the dense spec's own problem rules (the tile is checked per kernel)
        self.problem_spec()
        want = dict(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode="batched",
            stage_vec=8,
            mask_class="band" if self.masked else "none",
            ws_layout=self.ws_layout,
        )
        if self.dq is None:
            object.__setattr__(self, "dq", AttnBwdDqSpec(**want))
        if not isinstance(self.dq, AttnBwdDqSpec):
            raise ValueError(f"dq={self.dq!r} must be an AttnBwdDqSpec")
        for k, v in want.items():
            if getattr(self.dq, k) != v:
                raise ValueError(
                    f"dq.{k}={getattr(self.dq, k)!r} does not match the dense "
                    f"problem ({v!r})"
                )

    @classmethod
    def of(cls, main: AttentionDenseBwdSpec, dq: AttnBwdDqSpec):
        """The dense dQ spec of dense main spec ``main`` with dQ spec ``dq``."""
        return cls(
            **{k: getattr(main, k) for k in _PROBLEM},
            oob=main.oob,
            ws_layout=main.ws_layout,
            dq=dq,
            name=main.name,
        )

    def problem_spec(self) -> AttentionDenseBwdSpec:
        """A dense main spec of the same problem (validation and the mask)."""
        return AttentionDenseBwdSpec(
            **{k: getattr(self, k) for k in _PROBLEM},
            ragged=True,
            oob=self.oob,
            ws_layout=self.ws_layout,
            name=self.name,
        )

    @property
    def group(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def band(self) -> tuple[int, int, bool]:
        return dense_bwd_band(self)

    @property
    def masked(self) -> bool:
        left, right, _ = self.band()
        return not (left == -1 and right == -1)

    def strides(self, tensor: str) -> tuple[int, int, int]:
        q_side = tensor in ("q", "o", "do", "dq")
        heads = self.num_query_heads if q_side else self.num_kv_heads
        seq = self.seqlen_q if q_side else self.seqlen_kv
        return dense_layout_strides(self.layout, heads, seq, self.head_size)

    def kernel_name(self) -> str:
        """``{name}_dq_{dtype}_d{D}_{layout}_sq.._sk.._hq.._hk.._{mask}``
        (``_b{batch}`` when pinned) plus a salt over every non-default
        configuration value (``oob``, ``ws_layout`` and the dQ tuning)."""
        parts = [
            self.name,
            "dq",
            self.dtype,
            f"d{self.head_size}",
            self.layout,
            f"sq{self.seqlen_q}",
            f"sk{self.seqlen_kv}",
            f"hq{self.num_query_heads}",
            f"hk{self.num_kv_heads}",
            AttentionDenseBwdSpec._mask_token(self),
        ]
        if self.batch is not None:
            parts.append(f"b{self.batch}")
        salted = []
        if self.oob != "buffer_oob":
            salted.append(("oob", self.oob))
        if self.ws_layout != "head_major":
            salted.append(("ws_layout", self.ws_layout))
        salted += [
            (f"dq_{k}", v)
            for k, v in self.dq.non_default_fields()
            if k not in _DQ_PROBLEM and k != "ws_layout"
        ]
        return salted_kernel_name("_".join(parts), salted)


def validate_attn_dense_bwd_dq_spec(
    spec: AttentionDenseBwdDqSpec, arch: str
) -> AttnBwdDqSpec:
    """Arch checks of a dense dQ spec; returns its resolved, validated dQ spec.

    Raises ``ValueError`` for a target outside the dense family's, every
    error of the general dQ validator, a head group that does not divide the
    row tile of ``head_loop = "group"``, and a slab past the 32-bit offset.
    """
    if arch not in DENSE_BWD_ARCHS:
        raise ValueError(f"arch={arch!r} not in {DENSE_BWD_ARCHS}")
    dq = validate_attn_bwd_dq_spec(spec.dq, arch)
    dq_rows_per_head(dq, spec.group)
    for tensor in ("q", "k"):
        _sb, _sh, st = spec.strides(tensor)
        seq = spec.seqlen_q if tensor == "q" else spec.seqlen_kv
        rows = seq + max(_TILE_SLACK_ROWS, dq.block_n, dq.block_m)
        if rows * st * _ELEM_BYTES >= _VOFFSET_LIMIT:
            raise ValueError(
                f"{tensor} slab of {seq} rows at token stride {st} passes the 32-bit "
                "buffer offset"
            )
    if spec.batch is not None:
        rows = spec.batch * max(spec.seqlen_q, spec.seqlen_kv)
        if rows * spec.num_query_heads * spec.head_size >= _VOFFSET_LIMIT:
            raise ValueError("batch * rows * heads * head_size passes 2^31 elements")
    return dq


def attn_dense_bwd_dq_grid(
    spec: AttentionDenseBwdDqSpec, arch: str, *, batch: int
) -> tuple[int, int, int]:
    """``(ceil(seqlen_q / R), h_q / H, B)`` of the general dQ kernel's rule."""
    dq = validate_attn_dense_bwd_dq_spec(spec, arch)
    return attn_bwd_dq_grid(
        dq, s_q_max=spec.seqlen_q, h_q=spec.num_query_heads, group=spec.group,
        batch=batch,
    )  # fmt: skip


def dense_dq_static_classes(
    *,
    s_q: int,
    s_kv: int,
    rows: int,
    k_n0: int,
    left: int,
    right: int,
    bottom_right: bool,
    masked: bool,
) -> tuple[bool, bool]:
    """``(any_edge, any_interior)`` over every visited (q block, key tile)
    pair of the dQ kernel (q blocks of ``rows`` rows), evaluated with the
    kernel's own range emitters on the host."""
    hb = HostInt()
    off = s_kv - s_q if bottom_right else 0
    bounds = AttnRuntimeBounds(
        s_q, s_kv, off, left if masked else -1, right if masked else -1
    )
    band = BandEmitter(hb, bounds)
    any_edge = any_interior = False
    for x in range(-(-s_q // rows)):
        q0 = x * rows
        kt0, n_kt = kv_tile_range(hb, band, q0, rows, k_n0=k_n0, uniform=False)
        i0, i1 = interior_kv_range(
            hb, band, q0, rows, kt0, n_kt, k_n0=k_n0, masked=masked, uniform=False
        )
        any_edge = any_edge or (i0 - kt0) + (kt0 + n_kt - i1) > 0
        any_interior = any_interior or i1 > i0
        if any_edge and any_interior:
            break
    return any_edge, any_interior


# ---------------------------------------------------------------------------
# kernarg ABI rocke.attn_dense_bwd_dq.v1
# ---------------------------------------------------------------------------
#
# Order: pointers, f32 scalars, i32 scalars (every pointer 8 bytes).

_PTRS = ("Q", "K", "V", "dO", "dQ", "WS_LSE2", "WS_DSUM")
_F32 = ("scale_log2", "ds_mult", "dq_mult")
_I32 = ("batch", "xcd_n", "xcd_chunk")
_READONLY = frozenset({"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM"})


def attn_dense_bwd_dq_params(
    spec: Optional[AttentionDenseBwdDqSpec] = None,
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, struct format)`` kernargs of the dense dQ kernel
    (``"Q"`` a 64-bit pointer, ``"f"`` an f32, ``"i"`` an i32); the list
    does not depend on the spec."""
    return (
        tuple((p, "Q") for p in _PTRS)
        + tuple((f, "f") for f in _F32)
        + tuple((i, "i") for i in _I32)
    )


def _declare(b: IRBuilder, io) -> dict:
    args = {}
    for name, fmt in attn_dense_bwd_dq_params():
        if fmt == "Q":
            t = (
                PtrType(F32, "global")
                if name.startswith("WS_")
                else PtrType(io, "global")
            )
            attrs = {"readonly": True} if name in _READONLY else {}
            args[name] = b.param(name, t, **attrs)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------


def _pinned(b, slab, hint):
    """The slab with its pointer made opaque (an empty asm, as in the general
    dQ kernel), so one 64-bit pointer stays live across the key loops. Buffer
    slabs (a resource per slab) are returned as they are."""
    if not hasattr(slab, "ptr"):
        return slab
    ptr = b.inline_asm(
        "", "=s,0", [slab.ptr], result_type=slab.ptr.type, sideeffect=False,
        result_name_hint=hint,
    )  # fmt: skip
    return dataclasses.replace(slab, ptr=ptr)


def emit_dense_bwd_dq(spec: AttentionDenseBwdDqSpec, arch: str, *, sentinel=True):
    """Emit the dense split dQ kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term ``lse2 < +inf`` from the dS
    select (negative controls only, never a spec field).
    """
    s = validate_attn_dense_bwd_dq_spec(spec, arch)
    facts = attn_bwd_arch_facts(arch)
    geom = BwdDqGeometry.from_spec(s, arch)
    plan = dq_lds_plan(s, arch)
    io = ir_type(spec.dtype)
    km, kn = s.block_m, s.block_n
    s_q, s_kv = spec.seqlen_q, spec.seqlen_kv
    h_q = spec.num_query_heads
    G = spec.group
    group = s.head_loop == "group"
    rph = dq_rows_per_head(s, G)  # rows of one head in the CTA
    gx_n = -(-s_q // rph)

    b = IRBuilder(spec.kernel_name())
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, io)
    # Multipliers that are 1.0 under the compiled placement are not applied.
    a.update(dict.fromkeys(dq_static_unit_mults(s.static_scale)))
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if s.agpr_alloc not in (None, DQ_AGPR_ALLOC_BACKEND):
        b.kernel.attrs["agpr_alloc"] = tuple(s.agpr_alloc)
    if s.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(s.waves_per_eu)
    if s.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            s.scheduler_strategy
        )

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    c = b.const_i32
    zero, one = c(0), c(1)
    batch = c(spec.batch) if spec.batch is not None else a["batch"]
    x, y, z = b.block_id_x(), b.block_id_y(), b.block_id_z()
    if s.xcd_remap:
        # per batch plane, as in the general dQ kernel (identity at xcd_n = 0)
        gx = c(gx_n)
        gy = c(h_q // G if group else h_q)
        plane = xcd_remap_id(
            b, b.add(x, b.mul(gx, y)), num_wgs=b.mul(gx, gy), n=a["xcd_n"],
            chunk=a["xcd_chunk"],
        )  # fmt: skip
        x, y = b.mod(plane, gx), b.div(plane, gx)
    seq_q = decode_batched(b, z, s_max=s_q)
    seq_kv = decode_batched(b, z, s_max=s_kv)
    h = b.mul(y, c(G)) if group else y  # the (first) query head of the CTA
    hk = b.div(h, c(G))
    q0 = b.mul(x, c(rph))
    rows = (
        PackedRows(head0=h, n_heads=c(G), rows_per_head=c(rph), row0=q0)
        if group
        else None
    )
    # a q block can pass the sequence end only when rph does not divide s_q
    q_tail = s_q % rph != 0

    def tensor(name, role, seq, seqlen, *, global_view=False, ragged=True):
        return DenseFixedAddr(
            b, a[name], dtype=io, strides=spec.strides(role), seq=seq,
            seqlen=seqlen, head_size=spec.head_size, oob=spec.oob,
            global_view=global_view, ragged=ragged,
        )  # fmt: skip

    # a packed CTA reads one head per row: Q / dO through the global pointer
    q_addr = tensor("Q", "q", seq_q, s_q, global_view=group, ragged=q_tail)
    do_addr = tensor("dO", "do", seq_q, s_q, global_view=group, ragged=q_tail)
    k_addr = tensor("K", "k", seq_kv, s_kv)
    v_addr = tensor("V", "v", seq_kv, s_kv)
    dq_addr = StridedTensorAddr(
        b, a["dQ"], dtype=io, strides=spec.strides("dq"), seq=seq_q, stage_vec=8
    )
    rows_q = b.mul(batch, c(s_q))
    nh = c(h_q) if spec.ws_layout == "token_major" else None

    def ws(name):
        return WorkspaceAddr(
            b, a[name], ws_rows=rows_q, width=1, row0=seq_q.ws_row0,
            layout=spec.ws_layout, n_heads=nh,
        )  # fmt: skip

    stats = StatsView(ws("WS_LSE2"), ws("WS_DSUM"))
    left, right, br = spec.band()
    if spec.masked:
        mask = DiagonalBand(
            b, s_q=s_q, s_kv=s_kv, left=left, right=right, bottom_right=br,
            k_m0=km, k_n0=kn,
        )  # fmt: skip
    else:
        mask = DenseNoMask(b, s_q=s_q, s_kv=s_kv, k_m0=km, k_n0=kn)
    step = BwdDqStep(
        b, geom, mask=mask, stats=stats, lds=_lds_views(b, plan, io), tid=tid,
        lane=lane, wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        prescale=s.stats_prescale, sentinel=sentinel,
    )  # fmt: skip

    # Stage 0 (only a CTA with a row < len_q reads Q / dO).
    live = b.cmp_lt(q0, seq_q.length)  # CTA-uniform
    if group:
        q_slab = do_slab = None
    else:
        q_slab = q_addr.slab(h, q0)
        do_slab = do_addr.slab(h, q0)
    step.stage_q(
        b, live=live, h=h, q0=q0, q_slab=q_slab, do_slab=do_slab, q_addr=q_addr,
        do_addr=do_addr, rows=rows,
    )  # fmt: skip

    # Key tiles of the rows; the edge / interior runs proved empty at build
    # time are not emitted (same key order, so the same dQ bits).
    kt_start, n_kt = kv_tile_range(b, mask.band, q0, rph, k_n0=kn)
    kt_end = b.add(kt_start, n_kt)
    if s.edge_tiles:
        any_edge, any_interior = dense_dq_static_classes(
            s_q=s_q, s_kv=s_kv, rows=rph, k_n0=kn, left=left, right=right,
            bottom_right=br, masked=spec.masked,
        )  # fmt: skip
        if not any_edge:
            ranges = ((kt_start, kt_end, False),)
        elif not any_interior:
            ranges = ((kt_start, kt_end, True),)
        else:
            i0, i1 = interior_kv_range(
                b, mask.band, q0, rph, kt_start, n_kt, k_n0=kn, masked=mask.masked
            )
            ranges = ((kt_start, i0, True), (i0, i1, False), (i1, kt_end, True))
    else:
        ranges = ((kt_start, kt_end, True),)
    trips = [b.readfirstlane(b.sub(e, s0)) for s0, e, _ in ranges]

    k_slab = _pinned(b, k_addr.slab(hk), "kptr")
    v_slab = _pinned(b, v_addr.slab(hk), "vptr")
    # dQ at head h, row 0 (rows are addressed as given; the store predicates)
    dq_slab = _pinned(b, dq_addr.slab(h, clamp=False), "dqptr")
    head_step = None
    if group:
        # the head stride in bytes (fp16 / bf16 elements)
        head_step = b.const_i64(spec.strides("dq")[1] * _ELEM_BYTES)
    if s.kv_prefetch:
        step.prologue(b, k_slab, v_slab, b.mul(kt_start, c(kn)), b.cmp_lt(zero, n_kt))
    dq = step.zero_dq(b)  # after the range: the accumulators start here
    fm, fn = geom.g4t.frags_m, geom.g4t.frags_n
    names = ("k", "ki_", "ke_") if len(ranges) == 3 else ("k",)
    for (s0, _e, edge), trip, name in zip(ranges, trips, names):
        flat = [v for row in dq for v in row]
        loop = b.scf_for_iter(
            zero, trip, one,
            [(f"dq_{name}{i}", v) for i, v in enumerate(flat)],
            iv_name=f"dq_{name}it", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (it, cur):
            k0 = b.mul(b.add(s0, it), c(kn))
            acc = [list(cur[r * fn : (r + 1) * fn]) for r in range(fm)]
            out = step.run_kv_step(
                b, acc, k_slab, v_slab, k0, edge=edge, prefetch=s.kv_prefetch
            )
            b.scf_yield(*[v for row in out for v in row])
        res = list(loop.results)
        dq = [res[r * fn : (r + 1) * fn] for r in range(fm)]

    step.store_dq(
        b, dq, mult=a["dq_mult"], dq_slab=dq_slab, head_step=head_step, h=h,
        q0=q0, rows=rows, len_q=seq_q.length, s_q_max=c(s_q), seq_mode="batched",
    )  # fmt: skip
    b.ret()
    return b.kernel


def build_attention_dense_bwd_dq(spec: AttentionDenseBwdDqSpec, *, arch: str):
    """Build the dense split dQ kernel (``rocke.attn_dense_bwd_dq.v1``).

    The spec is validated for ``arch`` first (an illegal geometry raises
    ``ValueError`` before any IR is emitted).
    """
    if not isinstance(spec, AttentionDenseBwdDqSpec):
        raise TypeError(
            f"build_attention_dense_bwd_dq takes an AttentionDenseBwdDqSpec, got "
            f"{type(spec).__name__}"
        )
    return emit_dense_bwd_dq(spec, arch)
