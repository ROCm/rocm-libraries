# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused, weight-stationary Winograd ``F(m x m, 3 x 3)`` forward conv (gfx1250).

One kernel replaces K2 (input transform), K3 (per-bin GEMM) and K4 (output
transform) of the unfused pipeline (:mod:`kernels.common.conv_winograd_fwd`);
K1 still produces the transformed filter ``U[bin][k][c]``. Nothing but ``x``,
``U`` and ``y`` touches global memory.

Work split. A workgroup owns one group ``gi`` and one block of ``kb`` output
channels, and loops (persistently) over spatial *tile blocks*: one tile row
of ``tw = 16 * fm`` tiles of one dilation sub-grid. It has ``n * ks`` waves;
wave ``(r, ksw)`` owns *bin row* ``r`` (the ``n`` bins ``(r, 0..n-1)``) and
``fn`` 16-channel fragments of the K block. Lane ``l`` owns the ``fm``
consecutive tiles ``(l % 16) * fm + f`` and the 16 input channels
``(l // 16) * 16 ..`` of each 32-channel k-step.

Weights stay in registers. Each wave loads the WMMA A fragments of its bin
row, ``U[(r, c)][k][0 : Cg]``, once (``n * fn * Cg / 32`` fragments of 8
VGPRs) and keeps them for the whole kernel; nothing reads ``U`` again.

Input. Per ``ck``-channel chunk the ``n`` input rows of the block's patch
(``m * tw + 2`` pixels each) are copied to an LDS ring slot by ``n`` TDM
transfers (wave ``u`` copies row ``u``). The TDM inserts 16 bytes of padding
after every group of ``G = m * fm`` pixels, which makes the lanes' fragment
reads (stride ``G`` pixels) bank-conflict free. Image borders: the transfer
clips at the right / bottom edge and zero-fills; rows outside the image are
transfers with no valid pixel (all zero); at the left edge the first
in-image pixel lands on a group boundary (the slot is shifted by ``shift``
pixels) and the previous row's zero-filled overrun covers the left padding
(row 0's is zeroed with ``ds_store``).

Input transform, in registers, straight into WMMA B fragments: a lane first
combines the patch rows of its bin row, ``t[v] = sum_u B^T[r][u] d[u][v]``
(the nonzero terms only; their rows and coefficients are wave-uniform
runtime values, so all waves run one code body), then applies ``B^T``
along W to each of its tiles (:func:`bt_1d`), which yields the ``n`` B
fragments of bins ``(r, 0..n-1)`` for its tile and 16 channels. Packed
operand-dtype arithmetic (``arith="pk"``) or fp32 (``"f32"``).

GEMM: ``acc[c][i][f] += U[(r, c)] frag i  x  V[(r, c)] frag f`` with
``wmma_f32_16x16x32``; an accumulator lane holds one tile and 8 consecutive
output channels.

Output transform. After the last chunk each wave applies ``A^T`` along W to
its bins (:func:`at_1d`) in registers and stages the ``m`` results per
(tile, channel) in LDS (fp32); after a barrier every thread takes one (tile,
8 channels) item, applies ``A^T`` along H over the ``n`` staged bin rows and
stores the ``m x m`` output pixels (16-byte stores, dropped past the image
or ``K``). The ring keeps streaming the next block's first chunk meanwhile.

Variants (spec knobs; see :class:`WinogradFusedSpec`):

* ``pipe="early"`` (the measured best): chunk ``g + 1`` is published (wait +
  barrier + refill) during chunk ``g``'s last k-step and its first k-step is
  loaded before chunk ``g``'s last math; the output staging is double
  buffered, so block ``i``'s cross-row combine and stores run at the end of
  block ``i + 1``. One barrier per chunk.
* ``cs``: waves per bin row along C (partial sums added in the combine).
* ``tdm``: ``rows`` (one 2-D transfer per patch row), ``3d`` (one 3-D
  transfer of all rows, issuer rotates over the waves) or ``producer`` (an
  extra wave that only issues the 3-D transfers).
* ``bh``: tile rows per block (3-D transfers only), for narrow images.
* ``dpp_probe``: perf probe only (wrong results).

Scheduling constraint: a wave keeps at most one ``tensor_load_to_lds`` in
flight, so the issuing wave / set rotates per chunk and only the issuer
of a chunk waits for it. WMMA and VALU issue share the SIMD, so the
transform VALU directly adds to the WMMA time.

The problem is baked into the kernel (compile-time extents); the only
kernarg besides the pointers is ``P``, the workgroups per (group, K block).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import at_1d, bt_1d, toom_cook

WMMA_K = 32
_OOB = 0x7FFF0000
_I32_MAX = (1 << 31) - 1
_CPOL_NT = 1
ARITHS = ("pk", "f32")


def _ir(dtype: str):
    return BF16 if dtype == "bf16" else F16


def _int_rows(mat) -> List[List[int]]:
    out = []
    for row in mat:
        r = []
        for v in row:
            if v.denominator != 1:
                raise ValueError("expected an integer matrix")
            r.append(int(v))
        out.append(r)
    return out


@dataclass(frozen=True)
class WinogradFusedSpec:
    """One fused Winograd forward kernel (see the module doc).

    Args:
        problem: 2-D conv, 3x3, stride 1, any pad / dilation / groups.
        m: output tile (2 or 4).
        io_dtype: x / w / y dtype.
        operand_dtype: U / V dtype of the WMMA (fp16; bf16 only for F2 with
            bf16 I/O). Accumulation is fp32.
        fm: tiles per lane along W (block = 16 * fm tiles).
        fn: 16-channel output fragments per wave.
        ks: waves per bin row along K (K block = ks * fn * 16).
        ck: input channels per LDS ring chunk (multiple of 32).
        stages: ring depth (>= 2).
        arith: input-transform arithmetic, ``pk`` (packed operand dtype)
            or ``f32``.
    """

    problem: ConvProblem
    m: int = 2
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    fm: int = 1
    # tile rows per block (the block is bh x (16 * fm / bh) tiles; bh > 1
    # needs the 3-D transfers): less padding waste on narrow images
    bh: int = 1
    fn: int = 2
    ks: int = 2
    # waves per bin row along C: each wave transforms and multiplies every
    # cs-th k-step, the partial accumulators are summed in the combine.
    cs: int = 1
    ck: int = 32
    stages: int = 2
    arith: str = "pk"
    # Load the next k-step's patch pixels before this k-step's math.
    prefetch: bool = True
    # "simple": publish (wait + barrier) at the top of each chunk, output
    # transform + barrier + stores at the end of each block. "early": chunk
    # g + 1 is published during chunk g's last k-step (its first k-step is
    # loaded before chunk g's last math), the output staging is double
    # buffered and block i's cross-row combine + stores run at the end of
    # block i + 1 (after its staging write): one barrier per chunk.
    pipe: str = "simple"
    # patch transfers: "rows" (wave set g % (ks*cs) moves chunk g, one 2-D
    # transfer per row), "3d" (one 3-D transfer of all rows by wave g % waves)
    # or "producer" (an extra wave that only issues the 3-D transfers; early
    # pipe only)
    tdm: str = "rows"
    # PERF PROBE ONLY (results are wrong): load only the lane's own pixels
    # and take the two shared neighbour columns from lane + 1 with DPP
    # (row_shl:1), without the lane-15 fix-up: an upper bound of what DPP
    # sharing of the patch columns could gain.
    dpp_probe: bool = False
    # simple pipe: only the waves that move no patch rows store y
    store_split: bool = False
    nt_stores: bool = True
    # False: skip the VGPR estimate (the compile must then be checked for
    # scratch; tuning only)
    vgpr_check: bool = True
    setprio: bool = True
    sched: Optional[str] = None
    name: str = "conv_wino_fused"

    def __post_init__(self) -> None:
        p = self.problem
        if self.m not in (2, 4):
            raise ValueError("m must be 2 or 4")
        if p.is_3d or (p.Y, p.X) != (3, 3) or (p.sH, p.sW) != (1, 1):
            raise ValueError("fused Winograd needs a 2-D 3x3 stride-1 conv")
        if self.io_dtype not in ("fp16", "bf16"):
            raise ValueError("io_dtype must be fp16 or bf16")
        if self.operand_dtype not in ("fp16", "bf16"):
            raise ValueError("operand_dtype must be fp16 or bf16")
        if self.operand_dtype == "bf16" and (self.m != 2 or self.io_dtype != "bf16"):
            raise ValueError("bf16 operands only for F(2,3) with bf16 I/O")
        if self.m == 4 and self.io_dtype == "bf16":
            raise ValueError(
                "F(4,3) with bf16 I/O needs the per-tile prescale (fp16 V can "
                "overflow); not implemented in the fused kernel"
            )
        if self.arith not in ARITHS:
            raise ValueError(f"arith must be one of {ARITHS}")
        if self.io_dtype != self.operand_dtype and self.arith != "f32":
            raise ValueError("I/O and operand dtypes differ: needs arith='f32'")
        if self.fm not in (1, 2, 4) or self.fn < 1 or self.ks < 1:
            raise ValueError("fm in (1, 2, 4), fn >= 1, ks >= 1")
        if (
            self.cs < 1
            or (self.cg32 // WMMA_K) % self.cs
            or (self.ck // WMMA_K) % self.cs
        ):
            raise ValueError("cs must divide the k-steps of C/g and of a chunk")
        if self.ck not in (32, 64, 128):
            raise ValueError("ck must be 32, 64 or 128")
        if self.pipe not in ("simple", "early"):
            raise ValueError("pipe must be simple or early")
        if self.tdm not in ("rows", "3d", "producer"):
            raise ValueError("tdm must be rows, 3d or producer")
        if self.bh not in (1, 2, 4, 8, 16) or (16 * self.fm) % self.bh:
            raise ValueError("bh must be a power of two dividing 16 * fm")
        if self.twr % self.fm:
            raise ValueError("a lane's fm tiles must lie in one tile row")
        if self.bh > 1 and self.tdm == "rows":
            raise ValueError("bh > 1 needs tdm='3d' or 'producer'")
        if self.tdm == "producer" and self.pipe != "early":
            raise ValueError("the producer wave needs pipe='early'")
        if self.tdm == "3d" and self.stages - 1 > self.compute_waves:
            raise ValueError("tdm=3d: stages - 1 must not exceed the waves")
        if self.stages < 2:
            raise ValueError("stages must be >= 2")
        if p.C % 8 or p.cpg % 8 or p.kpg % 8:
            raise ValueError("C, C/g and K/g must be multiples of 8 (16-byte rows)")
        if p.Ho <= 0 or p.Wo <= 0:
            raise ValueError("empty output")
        if self.waves * 32 > 1024:
            raise ValueError("too many waves")
        if self.group_px * self.ck // 2 > 256:
            raise ValueError("TDM pad interval > 256 dwords (lower fm or ck)")
        for nm, nbytes in self.buffer_bytes().items():
            if nbytes > _I32_MAX:
                raise ValueError(f"{nm} exceeds 2 GiB (32-bit offsets)")

    # -- geometry --
    @property
    def n(self) -> int:
        return self.m + 2

    @property
    def compute_waves(self) -> int:
        return self.n * self.ks * self.cs

    @property
    def waves(self) -> int:
        return self.compute_waves + (1 if self.tdm == "producer" else 0)

    @property
    def vgpr_limit(self) -> int:
        """VGPRs per lane when every SIMD hosts ceil(waves / 4) waves."""
        per = -(-self.waves // 4)
        return 1024 // per // 8 * 8

    @property
    def threads(self) -> int:
        return 32 * self.waves

    @property
    def kb(self) -> int:
        return self.ks * self.fn * 16

    @property
    def tw(self) -> int:
        return 16 * self.fm

    @property
    def group_px(self) -> int:
        return self.m * self.fm

    @property
    def twr(self) -> int:
        """Tiles per block row."""
        return self.tw // self.bh

    @property
    def npr(self) -> int:
        """Patch rows a block needs."""
        return self.m * self.bh + 2

    @property
    def npx(self) -> int:
        """Patch pixels per row a block needs."""
        return self.m * self.twr + 2

    @property
    def npxb(self) -> int:
        """Pixels per LDS patch row (a whole number of pad groups)."""
        return ceil_div(self.npx, self.group_px) * self.group_px

    @property
    def vmax(self) -> int:
        """Most left-padding pixels a block can have."""
        p = self.problem
        return ceil_div(p.pW, p.dW) if p.pW > 0 else 0

    @property
    def cg32(self) -> int:
        return ceil_div(self.problem.cpg, WMMA_K) * WMMA_K

    @property
    def nchunks(self) -> int:
        return ceil_div(self.cg32, self.ck)

    def ksteps(self, q: int) -> int:
        return (min(self.ck, self.cg32 - q * self.ck)) // WMMA_K

    @property
    def sub_h(self) -> int:
        return ceil_div(self.problem.Ho, self.problem.dH)

    @property
    def sub_w(self) -> int:
        return ceil_div(self.problem.Wo, self.problem.dW)

    @property
    def tiles_h(self) -> int:
        return ceil_div(self.sub_h, self.m)

    @property
    def tiles_w(self) -> int:
        return ceil_div(self.sub_w, self.m)

    @property
    def col_blocks(self) -> int:
        return ceil_div(self.tiles_w, self.twr)

    @property
    def row_blocks(self) -> int:
        return ceil_div(self.tiles_h, self.bh)

    @property
    def num_blocks(self) -> int:
        p = self.problem
        return p.N * p.dH * p.dW * self.row_blocks * self.col_blocks

    @property
    def num_kblocks(self) -> int:
        return ceil_div(self.problem.kpg, self.kb)

    # -- LDS --
    def px_byte(self, px: int) -> int:
        return px * self.ck * 2 + (px // self.group_px) * 16

    @property
    def slot_px(self) -> int:
        g = self.group_px
        return self.npr * self.npxb + ceil_div(self.vmax + g, g) * g

    @property
    def slot_bytes(self) -> int:
        return ceil_div(self.px_byte(self.slot_px), 256) * 256

    @property
    def stage_pitch(self) -> int:
        """fp32 elements per staged (tile) row: K block + 4 (bank skew)."""
        return self.kb + 4

    @property
    def stage_bytes(self) -> int:
        return self.cs * self.stage_set_bytes

    @property
    def stage_set_bytes(self) -> int:
        return self.n * self.m * self.tw * self.stage_pitch * 4

    @property
    def dump_bytes(self) -> int:
        """Per-wave landing area of the dummy transfers (keeps every issuing
        wave at exactly one TDM per chunk, so tensorcnt waits stay exact)."""
        return self.n * ((self.ck * 2 + 15) // 16 * 16 + 16)

    def lds_bytes(self) -> int:
        nst = 2 if self.pipe == "early" else 1
        return self.stages * self.slot_bytes + nst * self.stage_bytes + self.dump_bytes

    def vgpr_estimate(self) -> int:
        n, fm, fn = self.n, self.fm, self.fn
        u = n * fn * (self.cg32 // WMMA_K // self.cs) * 8
        acc = n * fn * fm * 8
        bfr = n * fm * 8
        npl = self.m * fm + 2
        t = npl * 8 * (2 if self.arith == "f32" else 1)
        nt = 4 if self.m == 4 else 2
        raw = nt * npl * 8 * (2 if self.prefetch else 1)
        return u + acc + bfr + t + raw - 16  # calibrated: fm1 fn2 Cg128 -> 486 est

    # -- sizes --
    def buffer_bytes(self) -> dict:
        p = self.problem
        return {
            "x": p.N * p.Hi * p.Wi * p.C * 2,
            "y": p.N * p.Ho * p.Wo * p.K * 2,
            "U": (self.n**2) * p.K * p.cpg * 2,
        }

    def label(self) -> str:
        return f"F{self.m}_fm{self.fm}_fn{self.fn}_ks{self.ks}" + (
            f"_cs{self.cs}" if self.cs > 1 else ""
        ) + f"_ck{self.ck}_s{self.stages}" f"_{self.arith}" + (
            "" if self.nt_stores else "_rt"
        ) + (
            "" if self.prefetch else "_nopf"
        ) + (
            "_early" if self.pipe == "early" else ""
        ) + (
            "" if self.tdm == "rows" else f"_{self.tdm}"
        ) + (
            f"_bh{self.bh}" if self.bh > 1 else ""
        ) + (
            "_DPPPROBE" if self.dpp_probe else ""
        ) + (
            "" if self.setprio else "_noprio"
        ) + (
            f"_{self.sched}" if self.sched else ""
        )

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.problem.short(),
            self.label(),
            self.io_dtype,
            f"op{self.operand_dtype}",
        )

    # -- launch --
    def grid(self, wgs_per_kblock: int) -> Tuple[int, int, int]:
        return (wgs_per_kblock * self.num_kblocks * self.problem.groups, 1, 1)

    def default_wgs(self, cus: int = 256) -> int:
        """Workgroups per (group, K block): about one per CU in total, a
        multiple of 8 (so the K blocks of a tile block share an XCD)."""
        q = self.num_kblocks * self.problem.groups
        per = max(1, cus // q)  # never more workgroups than CUs (1 per CU)
        if per >= 8:
            per = per // 8 * 8
        return max(1, min(per, self.num_blocks))


def fused_reason(spec: WinogradFusedSpec, arch: str = "gfx1250") -> Optional[str]:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    if target.wave_size != 32:
        return "wave32 kernel"
    if not target.memory.has_tdm:
        return "needs TDM"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B too large"
    if spec.vgpr_check and spec.vgpr_estimate() > spec.vgpr_limit:
        return f"~{spec.vgpr_estimate()} VGPRs"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    return None


def fused_signature(spec: WinogradFusedSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("u", spec.operand_dtype)
        .ptr("y", spec.io_dtype)
        .scalar("P", "i32")
        .build()
    )


def _tdm_desc_3d_pad(
    b: IRBuilder,
    global_ptr: Value,
    lds_addr: Value,
    *,
    dims: Sequence[Value],
    strides: Sequence[Value],
    tile0: int,
    tile1: int,
    tile2: Value,
    pad_interval: int,
    pad_amount: int,
):
    """Rank-3 TDM descriptor (2-byte elements) with LDS padding: the tile
    lands packed ``[tile2][tile1][tile0]`` with ``pad_amount + 1`` dwords
    inserted after every ``2 ** (pad_interval + 1)`` dwords. Same encoding
    as ``conv_direct_dgrad_wmma._tdm_descriptor_3d`` plus the pad fields of
    :meth:`IRBuilder.tdm_descriptor_2d`."""
    c16 = b.const_i32(16)
    d0, d1, d2 = dims
    s1, s2 = strides
    gaddr = b.global_ptr_to_i64(global_ptr)
    g_hi = b.trunc(b.lshr(gaddr, b.const_i64(32)), I32)
    group0 = (
        b.const_i32(1),
        lds_addr,
        b.trunc(gaddr, I32),
        b.lor(b.land(g_hi, b.const_i32(0x1FFFFFF)), b.const_i32(-(1 << 31))),
    )
    flags = 1 << 16 | 1 << 20 | pad_interval << 22 | pad_amount << 25
    group1 = (
        b.const_i32(flags - (1 << 32) if flags & 0x80000000 else flags),
        b.shl(d0, c16),
        b.lor(b.lshr(d0, c16), b.shl(d1, c16)),
        b.lor(b.lshr(d1, c16), b.const_i32(tile0 << 16)),
        b.lor(b.const_i32(tile1), b.shl(tile2, c16)),
        s1,
        b.shl(s2, c16),
        b.lshr(s2, c16),
    )
    group2 = (d2, b.const_i32(0), b.const_i32(0), b.const_i32(0))

    def pk(words):
        return b.vec_pack([b.readfirstlane(v) for v in words], I32)

    return pk(group0), pk(group1), pk(group2), b.zero_vec(I32, 4), b.zero_vec(I32, 8)


class _VecOps:
    """Element-wise vector arithmetic for :func:`bt_1d` / :func:`at_1d`."""

    def __init__(self, b: IRBuilder, elem, n: int):
        self.b = b
        self.elem = elem
        self.n = n
        self._splats = {}

    def splat(self, c: float) -> Value:
        if c not in self._splats:
            b = self.b
            s = b.const_f32(float(c))
            if self.elem is not F32:
                s = b.cast_f32_to(s, self.elem)
            self._splats[c] = b.vector_splat(s, self.n)
        return self._splats[c]

    def add(self, x, y):
        return self.b.vector_add(x, y)

    def sub(self, x, y):
        return self.b.vector_sub(x, y)

    def fma(self, c: float, x, y):
        return self.b.vector_fma(self.splat(c), x, y)


def build_winograd_fused(
    spec: WinogradFusedSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    why = fused_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid WinogradFusedSpec: {why}")
    p = spec.problem
    m, n = spec.m, spec.n
    FM, FN, KS = spec.fm, spec.fn, spec.ks
    CK = spec.ck
    NS = spec.stages
    NC = spec.nchunks
    G = spec.group_px
    NPXB = spec.npxb
    NV = spec.npx
    TB = spec.tw
    TWR = spec.twr
    NPR = spec.npr
    KB = spec.kb
    SP = spec.stage_pitch
    SLOT = spec.slot_bytes
    Cg, Kg, C, K = p.cpg, p.kpg, p.C, p.K
    io_t = _ir(spec.io_dtype)
    op_t = _ir(spec.operand_dtype)
    ab = "bf16" if spec.operand_dtype == "bf16" else "f16"
    wmma = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )
    at, _, bt = toom_cook(m)
    BT = _int_rows(bt)
    AT = _int_rows(at)
    # Nonzero terms of each B^T row, the +1 term last (it seeds the sum).
    terms = []
    for r in range(n):
        tr = [(u, c) for u, c in enumerate(BT[r]) if c != 0]
        tr.sort(key=lambda uc: uc[1] == 1)
        if tr[-1][1] != 1:
            raise AssertionError("every B^T row has a +1 term")
        terms.append(tr)
    NT = max(len(t) for t in terms)
    for tr in terms:  # pad with zero-coefficient terms (row 0)
        while len(tr) < NT:
            tr.insert(0, (tr[0][0], 0))

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    X = b.param("x", PtrType(io_t, "global"), noalias=True, readonly=True, align=16)
    U = b.param("u", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("y", PtrType(io_t, "global"), noalias=True, writeonly=True, align=16)
    c_P = b.param("P", I32)

    ci = b.const_i32
    c0, c1, c2 = ci(0), ci(1), ci(2)
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(32)))  # wave-uniform (SGPR)
    lane = b.mod(tid, ci(32))
    l16 = b.mod(lane, ci(16))
    half = b.div(lane, ci(16))
    r_w = b.mod(wave, ci(n))  # bin row of this wave
    set_w = b.div(wave, ci(n))  # transfer set: (K slice, C slice)
    ks_w = b.mod(set_w, ci(KS))  # K slice of this wave
    cs_w = b.div(set_w, ci(KS))  # C slice (k-steps cs_w, cs_w + cs, ..)
    CS = spec.cs
    NSET = KS * CS

    # ---- workgroup -> (group, K block, persistent index p)
    fid = b.block_id_x()
    p_idx = b.mod(fid, c_P)
    q_idx = b.div(fid, c_P)
    kblk = b.mod(q_idx, ci(spec.num_kblocks))
    gi = b.div(q_idx, ci(spec.num_kblocks))
    NB = spec.num_blocks
    # blocks of this workgroup: p, p + P, ...
    n_mine = b.smax(b.div(b.add(b.sub(ci(NB), p_idx), b.sub(c_P, c1)), c_P), c0)
    total_chunks = b.mul(n_mine, ci(NC))

    # ---- U fragments in registers: A of bin (r, c), k-fragment i, k-step s
    u_bytes = spec.buffer_bytes()["U"]
    u_rsrc = b.buffer_rsrc(U, ci(u_bytes))
    oob = ci(_OOB)
    kloc0 = b.add(b.mul(kblk, ci(KB)), b.mul(ks_w, ci(FN * 16)))
    NKS = spec.cg32 // WMMA_K // CS  # k-steps of this wave
    ufr = {}
    for c in range(n):
        binv = b.add(b.mul(r_w, ci(n)), ci(c))
        for i in range(FN):
            kloc = b.add(kloc0, b.add(ci(i * 16), l16))
            krow = b.add(b.mul(binv, ci(K)), b.add(b.mul(gi, ci(Kg)), kloc))
            kok = b.cmp_lt(kloc, ci(Kg))
            rowb = b.select(
                kok, b.mul(b.add(b.mul(krow, ci(Cg)), b.mul(half, ci(16))), c2), oob
            )
            for s in range(NKS):
                parts = []
                sg = b.add(ci(s * CS), cs_w)  # global k-step
                for h8 in range(2):
                    cst = b.add(b.mul(sg, ci(WMMA_K * 2)), ci(h8 * 16))
                    if Cg % WMMA_K == 0:
                        # every 8-channel group is in range
                        off = b.add(rowb, cst)
                    else:
                        ch = b.add(
                            b.add(b.mul(sg, ci(WMMA_K)), ci(h8 * 8)),
                            b.mul(half, ci(16)),
                        )
                        off = b.select(b.cmp_lt(ch, ci(Cg)), b.add(rowb, cst), oob)
                    parts.append(b.buffer_load_vN(u_rsrc, off, c0, op_t, 8))
                ufr[(c, i, s)] = b.vec_concat(parts[0], parts[1])

    # ---- LDS: ring of patch slots, then the output staging area
    lds = b.smem_alloc(F16, [1, spec.lds_bytes() // 2], name_hint="lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)
    STAGE_OFF = NS * SLOT  # bytes
    DUMP_OFF = STAGE_OFF + spec.stage_bytes * (2 if spec.pipe == "early" else 1)
    DUMP_W = spec.dump_bytes // n

    def px_byte(px: Value) -> Value:
        """Byte offset of patch pixel ``px`` inside a slot (pad groups)."""
        return b.add(b.mul(px, ci(CK * 2)), b.mul(b.div(px, ci(G)), ci(16)))

    @dataclass
    class _Blk:
        n_img: Value
        a: Value
        bs: Value
        ti: Value
        tj0: Value
        v0: Value
        shift: Value
        wstart: Value
        ncols: Value

    def _block(i_blk: Value) -> _Blk:
        blk = b.add(p_idx, b.mul(i_blk, c_P))
        tjb = b.mod(blk, ci(spec.col_blocks))
        rest = b.div(blk, ci(spec.col_blocks))
        ti = b.mul(b.mod(rest, ci(spec.row_blocks)), ci(spec.bh))
        rest = b.div(rest, ci(spec.row_blocks))
        bs = b.mod(rest, ci(p.dW))
        rest = b.div(rest, ci(p.dW))
        a = b.mod(rest, ci(p.dH))
        n_img = b.div(rest, ci(p.dH))
        tj0 = b.mul(tjb, ci(TWR))
        # first patch column wi0 = bs - pW + dW * tj0 * m; v0 = left-pad pixels
        neg = b.sub(ci(p.pW), b.add(bs, b.mul(tj0, ci(p.dW * m))))
        v0 = b.select(b.cmp_gt(neg, c0), b.div(b.add(neg, ci(p.dW - 1)), ci(p.dW)), c0)
        shift = b.mod(b.sub(ci(G), b.mod(v0, ci(G))), ci(G))
        wstart = b.add(
            b.sub(bs, ci(p.pW)), b.mul(ci(p.dW), b.add(b.mul(tj0, ci(m)), v0))
        )
        left = b.sub(ci(p.Wi), wstart)
        ncols = b.select(
            b.cmp_gt(left, c0),
            b.smin(b.div(b.add(left, ci(p.dW - 1)), ci(p.dW)), b.sub(ci(NV), v0)),
            c0,
        )
        return _Blk(n_img, a, bs, ti, tj0, v0, shift, wstart, ncols)

    pad_interval = (G * CK // 2).bit_length() - 2  # 2^(f+1) dwords per group
    # Most bytes a software-zeroed range covers: one row plus the next row's
    # left padding (a row with no in-image pixel).
    ZMAX = spec.px_byte(NPXB + spec.vmax + G) + 16
    z_iters = ceil_div(ZMAX // 16, 32)

    def _zero_bytes(slot_off: Value, lo: Value, hi: Value) -> None:
        """ds_store zeros over slot bytes [lo, hi) (multiples of 16)."""
        z = b.zero_vec(F16, 8)
        for it in range(z_iters):
            byte = b.add(lo, b.mul(b.add(lane, ci(it * 32)), ci(16)))
            with b.scf_if(b.cmp_lt(byte, hi)):
                b.smem_store_vN(lds, [c0, b.div(b.add(slot_off, byte), c2)], z, 8)

    def _issue_rows(i_blk: Value, q: int, gidx: Value) -> None:
        """TDM chunk ``q`` of block ``i_blk`` into slot ``gidx % NS``; wave
        ``u < n`` moves patch row ``u``. A row with no in-image pixel is
        zeroed with ds_store instead (no zero-extent transfers)."""
        blk = _block(i_blk)
        slot_off = b.mul(b.mod(gidx, ci(NS)), ci(SLOT))
        slot = b.add(lds_addr, slot_off)
        # Chunk g is moved by wave set g % ks (wave (s, u) moves row u): on
        # this part a TDM issue waits for every memory op of its wave (incl.
        # its earlier transfers), so a wave must not issue back to back.
        with b.scf_if(b.cmp_eq(set_w, b.mod(gidx, ci(NSET)))):
            u = r_w
            hi = b.add(
                b.sub(blk.a, ci(p.pH)),
                b.mul(ci(p.dH), b.add(b.mul(blk.ti, ci(m)), u)),
            )
            rok = b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, ci(p.Hi)))
            valid = b.land(rok, b.cmp_gt(blk.ncols, c0))
            row_px = b.mul(u, ci(NPXB))
            dst_px = b.add(row_px, b.add(blk.v0, blk.shift))
            with b.scf_if_else(valid) as (t_ctx, e_ctx):
                with t_ctx:
                    c_0 = q * CK
                    pix = b.add(
                        b.mul(b.add(b.mul(blk.n_img, ci(p.Hi)), hi), ci(p.Wi)),
                        blk.wstart,
                    )
                    eoff = b.add(b.mul(pix, ci(C)), b.add(b.mul(gi, ci(Cg)), ci(c_0)))
                    if spec.vmax > 0:
                        # row 0's left padding (no previous row overruns it)
                        with b.scf_if(b.land(b.cmp_eq(u, c0), b.cmp_gt(blk.v0, c0))):
                            _zero_bytes(slot_off, px_byte(blk.shift), px_byte(dst_px))
                    b.tensor_load_to_lds(
                        *b.tdm_descriptor_2d(
                            b.global_ptr_add(X, b.mul(eoff, c2)),
                            b.add(slot, px_byte(dst_px)),
                            elem_bytes=2,
                            tensor_dim0=ci(Cg - c_0),
                            tensor_dim1=blk.ncols,
                            row_stride=ci(p.dW * C),
                            tile_dim0=CK,
                            tile_dim1=NPXB,
                            pad_interval=pad_interval,
                            pad_amount=3,
                        )
                    )
                with e_ctx:
                    # this row (it starts at row_px + shift: the cells before
                    # belong to the previous row's data) and the next row's
                    # left padding
                    _zero_bytes(
                        slot_off,
                        px_byte(b.add(row_px, blk.shift)),
                        px_byte(b.add(dst_px, ci(NPXB))),
                    )
                    # dummy one-pixel transfer of x[0..] into this wave's dump
                    # area: one TDM per wave and chunk, whatever the row
                    b.tensor_load_to_lds(
                        *b.tdm_descriptor_2d(
                            X,
                            b.add(lds_addr, b.add(ci(DUMP_OFF), b.mul(u, ci(DUMP_W)))),
                            elem_bytes=2,
                            tensor_dim0=ci(8),
                            tensor_dim1=c1,
                            row_stride=ci(C),
                            tile_dim0=CK,
                            tile_dim1=1,
                        )
                    )

    NWC = spec.compute_waves
    is_prod = b.cmp_eq(wave, ci(NWC))

    _zl_count = [0]

    def _zero_loop(slot_off: Value, lo: Value, hi: Value) -> None:
        """ds_store zeros over slot bytes [lo, hi) with a runtime loop."""
        z = b.zero_vec(F16, 8)
        cnt = b.div(b.add(b.sub(hi, lo), ci(511)), ci(512))
        _zl_count[0] += 1
        zl = b.scf_for(c0, cnt, c1, iv_name=f"zi{_zl_count[0]}")
        with zl as it:
            byte = b.add(lo, b.mul(b.add(lane, b.mul(it, ci(32))), ci(16)))
            with b.scf_if(b.cmp_lt(byte, hi)):
                b.smem_store_vN(lds, [c0, b.div(b.add(slot_off, byte), c2)], z, 8)

    def _issue3d(i_blk: Value, q: int, gidx: Value) -> None:
        """One 3-D transfer of all patch rows of chunk ``q`` of block
        ``i_blk`` into slot ``gidx % NS`` (by wave gidx % waves, or by the
        producer wave). Rows above the image and the left padding of the
        first in-image row are zeroed with ds_store, rows below and columns
        right of the image are zero-filled by the transfer."""
        blk = _block(i_blk)
        slot_off = b.mul(b.mod(gidx, ci(NS)), ci(SLOT))
        slot = b.add(lds_addr, slot_off)
        if spec.tdm == "producer":
            cond = is_prod
        else:
            cond = b.cmp_eq(wave, b.mod(gidx, ci(NWC)))
        with b.scf_if(cond):
            hi0 = b.add(b.sub(blk.a, ci(p.pH)), b.mul(blk.ti, ci(p.dH * m)))
            u0 = b.select(
                b.cmp_lt(hi0, c0),
                b.div(b.add(b.sub(c0, hi0), ci(p.dH - 1)), ci(p.dH)),
                c0,
            )
            u0 = b.smin(u0, ci(NPR))
            hs = b.add(hi0, b.mul(u0, ci(p.dH)))
            left = b.sub(ci(p.Hi), hs)
            nrows = b.select(
                b.cmp_gt(left, c0),
                b.smin(b.div(b.add(left, ci(p.dH - 1)), ci(p.dH)), b.sub(ci(NPR), u0)),
                c0,
            )
            valid = b.land(b.cmp_gt(nrows, c0), b.cmp_gt(blk.ncols, c0))
            dst_px = b.add(b.mul(u0, ci(NPXB)), b.add(blk.v0, blk.shift))
            with b.scf_if_else(valid) as (t_ctx, e_ctx):
                with t_ctx:
                    c_0 = q * CK
                    # rows above the image + the first row's left padding
                    _zero_loop(slot_off, px_byte(blk.shift), px_byte(dst_px))
                    pix = b.add(
                        b.mul(b.add(b.mul(blk.n_img, ci(p.Hi)), hs), ci(p.Wi)),
                        blk.wstart,
                    )
                    eoff = b.add(b.mul(pix, ci(C)), b.add(b.mul(gi, ci(Cg)), ci(c_0)))
                    b.tensor_load_to_lds(
                        *_tdm_desc_3d_pad(
                            b,
                            b.global_ptr_add(X, b.mul(eoff, c2)),
                            b.add(slot, px_byte(dst_px)),
                            dims=(ci(Cg - c_0), blk.ncols, nrows),
                            strides=(ci(p.dW * C), ci(p.dH * p.Wi * C)),
                            tile0=CK,
                            tile1=NPXB,
                            tile2=b.sub(ci(NPR), u0),
                            pad_interval=pad_interval,
                            pad_amount=3,
                        )
                    )
                with e_ctx:
                    _zero_loop(
                        slot_off,
                        px_byte(blk.shift),
                        px_byte(b.add(blk.shift, ci(NPR * NPXB))),
                    )
                    b.tensor_load_to_lds(
                        *b.tdm_descriptor_2d(
                            X,
                            b.add(lds_addr, ci(DUMP_OFF)),
                            elem_bytes=2,
                            tensor_dim0=ci(8),
                            tensor_dim1=c1,
                            row_stride=ci(C),
                            tile_dim0=CK,
                            tile_dim1=1,
                        )
                    )

    _issue = _issue_rows if spec.tdm == "rows" else _issue3d

    def _issue_global(g_static_off: int, i_blk: Value) -> None:
        """Issue global chunk ``i_blk * NC + g_static_off`` if it exists."""
        boff, q = divmod(g_static_off, NC)
        tgt = b.add(i_blk, ci(boff))
        with b.scf_if(b.cmp_lt(tgt, n_mine)):
            _issue(tgt, q, b.add(b.mul(i_blk, ci(NC)), ci(g_static_off)))

    # Transfers of one wave set that can be younger than the oldest one.
    YOUNG = (NS - 2) // NSET

    def _wait_publish(gidx: Value, role: str = "all") -> None:
        """Wait for chunk ``gidx`` (only its issuing wave / set has to),
        then barrier."""
        if spec.tdm == "producer":
            if role == "producer":
                more = b.cmp_lt(b.add(gidx, ci(NS - 2)), total_chunks)
                with b.scf_if_else(more) as (t_ctx, e_ctx):
                    with t_ctx:
                        b.s_wait_tensorcnt(NS - 2)
                    with e_ctx:
                        b.s_wait_tensorcnt(0)
            b.sync_lds_only()
            return
        if spec.tdm == "3d":
            with b.scf_if(b.cmp_eq(wave, b.mod(gidx, ci(NWC)))):
                b.s_wait_tensorcnt(0)
            b.sync_lds_only()
            return
        with b.scf_if(b.cmp_eq(set_w, b.mod(gidx, ci(NSET)))):
            if YOUNG == 0:
                b.s_wait_tensorcnt(0)
            else:
                more = b.cmp_lt(b.add(gidx, ci(YOUNG * NSET)), total_chunks)
                with b.scf_if_else(more) as (t_ctx, e_ctx):
                    with t_ctx:
                        b.s_wait_tensorcnt(YOUNG)
                    with e_ctx:
                        b.s_wait_tensorcnt(0)
        b.sync_lds_only()

    # ---- per-wave H-combination terms (runtime row / coefficient)
    def _sel(vals: Sequence[int]) -> Value:
        v = ci(vals[0])
        for r in range(1, n):
            v = b.select(b.cmp_eq(r_w, ci(r)), ci(vals[r]), v)
        return v

    term_rows = [_sel([terms[r][t][0] for r in range(n)]) for t in range(NT)]
    term_coef_i = [_sel([terms[r][t][1] for r in range(n)]) for t in range(NT)]

    ar_t = op_t if spec.arith == "pk" else F32
    vops = _VecOps(b, ar_t, 8)
    term_coef = []
    for t in range(NT - 1):
        cf = b.sitofp_f32(term_coef_i[t])
        if ar_t is not F32:
            cf = b.cast_f32_to(cf, ar_t)
        term_coef.append(b.vector_splat(cf, 8))

    NPL = m * FM + 2  # patch pixels a lane reads per row

    def _to_arith(v: Value) -> Value:
        if ar_t is not F32:
            return v
        return b.vec_pack([b.cast_to_f32(b.vec_extract(v, e)) for e in range(8)], F32)

    def _to_op(v: Value) -> Value:
        if ar_t is not F32:
            return v
        return b.vec_cast_f32_to(v, op_t)

    def _bload(slot_byte: Value, px_base: List[Value], kstep: int) -> list:
        """Raw patch vectors raw[dv][h8][t] of one k-step (LDS loads only)."""
        ch_byte = b.mul(b.add(ci(kstep * CS), cs_w), ci(WMMA_K * 2))
        hb = b.add(b.mul(half, ci(32)), ch_byte)
        raw = []
        for dv in range(NPL - 2 if spec.dpp_probe else NPL):
            lohi = []
            for h8 in range(2):
                vals = []
                for t in range(NT):
                    byte = b.add(
                        b.add(slot_byte, px_base[t * NPL + dv]),
                        b.add(hb, ci(h8 * 16)),
                    )
                    vals.append(
                        b.smem_load_vN(lds, c0, b.div(byte, c2), dtype=io_t, n=8)
                    )
                lohi.append(vals)
            raw.append(lohi)
        return raw

    def _bcompute(raw: list) -> dict:
        """B fragments ``bf[(c, f)]`` of this wave's bins from raw patch."""
        tvals = []  # t[dv] as (lo, hi) 8-channel vectors
        for dv in range(len(raw)):
            lohi = []
            for h8 in range(2):
                vals = [_to_arith(v) for v in raw[dv][h8]]
                acc = vals[NT - 1]
                for t in range(NT - 1):
                    acc = b.vector_fma(term_coef[t], vals[t], acc)
                lohi.append(acc)
            tvals.append(lohi)
        if spec.dpp_probe:
            for dv in range(2):  # columns of lane + 1
                lohi = []
                for h8 in range(2):
                    v = tvals[dv][h8]
                    words = b.vec_bitcast(
                        v,
                        VectorType(
                            I32,
                            (
                                v.type.count // 2
                                if v.type.elem is not F32
                                else v.type.count
                            ),
                        ),
                    )
                    nw = words.type.count
                    sh = [
                        b.mov_dpp(b.vec_extract(words, e), row_shl=1) for e in range(nw)
                    ]
                    lohi.append(b.vec_bitcast(b.vec_pack(sh, I32), v.type))
                tvals.append(lohi)
        out = {}
        for f in range(FM):
            parts = [[None, None] for _ in range(n)]
            for h8 in range(2):
                col = [tvals[m * f + v][h8] for v in range(n)]
                res = bt_1d(vops, m, col)
                for c in range(n):
                    parts[c][h8] = _to_op(res[c])
            for c in range(n):
                out[(c, f)] = b.vec_concat(parts[c][0], parts[c][1])
        return out

    # ---- output side: staged S[r][j][tile][kb (+4)] fp32
    def _stage_write(accs: dict, soff) -> None:
        """A^T along W in registers; stage the m results per bin row."""
        ops32 = _VecOps(b, F32, 8)
        # W-direction A^T over the bins of this row
        for i in range(FN):
            for f in range(FM):
                s = at_1d(ops32, m, [accs[(c, i, f)] for c in range(n)])
                tile = b.add(b.mul(l16, ci(FM)), ci(f))
                kk = b.add(
                    b.mul(ks_w, ci(FN * 16)), b.add(ci(i * 16), b.mul(half, ci(8)))
                )
                for j in range(m):
                    row = b.add(b.mul(b.add(b.mul(r_w, ci(m)), ci(j)), ci(TB)), tile)
                    el = b.add(b.mul(row, ci(SP)), kk)
                    byte = b.add(
                        b.add(soff, b.mul(cs_w, ci(spec.stage_set_bytes))),
                        b.mul(el, ci(4)),
                    )
                    lo = b.vec_pack([b.vec_extract(s[j], e) for e in range(4)], F32)
                    hi = b.vec_pack([b.vec_extract(s[j], e) for e in range(4, 8)], F32)
                    idx = b.div(byte, c2)
                    b.smem_store_vN_f32(lds, [c0, idx], lo, n=4)
                    b.smem_store_vN_f32(lds, [c0, b.add(idx, ci(8))], hi, n=4)

    def _combine(blk: _Blk, soff) -> None:
        """A^T along H over the staged bin rows; store y."""
        ops32 = _VecOps(b, F32, 8)
        items = TB * (KB // 8)
        y_rsrc = b.buffer_rsrc(Y, ci(spec.buffer_bytes()["y"]))
        cpol = _CPOL_NT if spec.nt_stores else 0
        # On this part a TDM issue first waits for all of the issuing wave's
        # memory ops (an s_wait_idle the code-object translator inserts), so
        # the y stores go to the waves that issue no transfers when there are
        # any (waves n..).
        if spec.waves > n and spec.pipe == "simple" and spec.store_split:
            st_tid = b.sub(tid, ci(32 * n))
            st_threads = spec.threads - 32 * n
        else:
            st_tid = tid
            st_threads = 32 * spec.compute_waves  # not the producer wave
        st_ok = b.cmp_ge(st_tid, c0)
        for it in range(ceil_div(items, st_threads)):
            e = b.add(st_tid, ci(it * st_threads))
            eok = b.land(st_ok, b.cmp_lt(e, ci(items)))
            g8 = b.mod(e, ci(KB // 8))
            tile = b.div(e, ci(KB // 8))
            sv = {}
            for r in range(n):
                for j in range(m):
                    row = b.add(ci((r * m + j) * TB), tile)
                    el = b.add(b.mul(row, ci(SP)), b.mul(g8, ci(8)))
                    tot = None
                    for cz in range(CS):
                        idx = b.div(
                            b.add(
                                soff,
                                b.add(ci(cz * spec.stage_set_bytes), b.mul(el, ci(4))),
                            ),
                            c2,
                        )
                        lo = b.smem_load_vN(lds, c0, idx, dtype=F32, n=4)
                        hi = b.smem_load_vN(lds, c0, b.add(idx, ci(8)), dtype=F32, n=4)
                        v = b.vec_concat(lo, hi)
                        tot = v if tot is None else b.vector_add(tot, v)
                    sv[(r, j)] = tot
            kloc = b.add(b.mul(kblk, ci(KB)), b.mul(g8, ci(8)))
            ok = b.land(eok, b.cmp_lt(kloc, ci(Kg)))
            tj = b.add(blk.tj0, b.mod(tile, ci(TWR)))
            tir = b.add(blk.ti, b.div(tile, ci(TWR)))
            for i in range(m):
                ho = b.add(blk.a, b.mul(ci(p.dH), b.add(b.mul(tir, ci(m)), ci(i))))
                hok = b.cmp_lt(ho, ci(p.Ho))
                for j in range(m):
                    terms_y = [(AT[i][r], sv[(r, j)]) for r in range(n) if AT[i][r]]
                    yv = None
                    for cf, v in terms_y:
                        if yv is None:
                            yv = v if cf == 1 else b.vector_mul(ops32.splat(cf), v)
                        elif cf == 1:
                            yv = b.vector_add(yv, v)
                        elif cf == -1:
                            yv = b.vector_sub(yv, v)
                        else:
                            yv = b.vector_fma(ops32.splat(cf), v, yv)
                    wo = b.add(blk.bs, b.mul(ci(p.dW), b.add(b.mul(tj, ci(m)), ci(j))))
                    pok = b.land(ok, b.land(hok, b.cmp_lt(wo, ci(p.Wo))))
                    pix = b.add(
                        b.mul(b.add(b.mul(blk.n_img, ci(p.Ho)), ho), ci(p.Wo)), wo
                    )
                    off = b.mul(
                        b.add(b.mul(pix, ci(K)), b.add(b.mul(gi, ci(Kg)), kloc)), c2
                    )
                    off = b.select(pok, off, oob)
                    val = b.vec_cast_f32_to(yv, io_t)
                    if spec.io_dtype == "bf16":
                        b.buffer_store_vN_bf16(
                            y_rsrc, off, c0, val, 4, cachepolicy=cpol
                        )
                    else:
                        b.buffer_store_vN_f16(y_rsrc, off, c0, val, 4, cachepolicy=cpol)

    # ---- per-block lane pixel offsets of the H terms
    # the lane's first tile: row l_tr, column l_tc of the block
    l_tr = b.div(b.mul(l16, ci(FM)), ci(TWR))
    l_tc = b.mod(b.mul(l16, ci(FM)), ci(TWR))

    def _px_base(blk: _Blk) -> List[Value]:
        out = []
        for t in range(NT):
            row = b.add(term_rows[t], b.mul(l_tr, ci(m)))
            for dv in range(NPL):
                pxi = b.add(
                    b.add(b.mul(row, ci(NPXB)), blk.shift),
                    b.add(b.mul(l_tc, ci(m)), ci(dv)),
                )
                out.append(px_byte(pxi))
        return out

    def _mmas(accs: dict, bfr: dict, gs: int) -> None:
        for c in range(n):
            for i in range(FN):
                for f in range(FM):
                    accs[(c, i, f)] = b.mma(
                        wmma, ufr[(c, i, gs)], bfr[(c, f)], accs[(c, i, f)]
                    )

    def _flat(raw: list) -> list:
        return [v for dv in raw for lohi in dv for v in lohi]

    NPL_LD = NPL - 2 if spec.dpp_probe else NPL

    def _unflat(vals: list) -> list:
        it = iter(vals)
        return [
            [[next(it) for _ in range(NT)] for _ in range(2)] for _ in range(NPL_LD)
        ]

    if spec.pipe == "early":
        SB = spec.stage_bytes

        def _soff(i_blk: Value) -> Value:
            return b.add(ci(STAGE_OFF), b.mul(b.mod(i_blk, c2), ci(SB)))

        def _early(role: str) -> None:
            """The early pipeline. ``role``: "all" (every wave computes and
            the issuers move patches), "consumer" (compute only) or
            "producer" (transfers only). Every role runs the same barrier
            sequence: prologue publish, one publish (or plain barrier) per
            chunk, a final barrier."""
            compute = role != "producer"
            moves = role != "consumer"
            with b.scf_if(b.cmp_gt(n_mine, c0)):
                if moves:
                    for g in range(NS - 1):
                        _issue_global(g, c0)
                _wait_publish(c0, role)
                if moves:
                    with b.scf_if(b.cmp_lt(ci(NS - 1), total_chunks)):
                        boff, qq = divmod(NS - 1, NC)
                        _issue(ci(boff), qq, ci(NS - 1))
            zero8 = b.zero_vec_f32(8)
            if compute:
                raw0 = _bload(ci(0), _px_base(_block(c0)), 0)
                loop = b.scf_for_iter(
                    c0,
                    n_mine,
                    c1,
                    [(f"r{k}", v) for k, v in enumerate(_flat(raw0))],
                    iv_name=f"ib_{role}",
                )
            else:
                loop = b.scf_for(c0, n_mine, c1, iv_name=f"ib_{role}")
            with loop as lp:
                if compute:
                    ib, carried = lp
                    raw = _unflat(list(carried))
                    blk = _block(ib)
                    pxb = _px_base(blk)
                    accs = {
                        (c, i, f): zero8
                        for c in range(n)
                        for i in range(FN)
                        for f in range(FM)
                    }
                else:
                    ib = lp
                for q in range(NC):
                    g = b.add(b.mul(ib, ci(NC)), ci(q))
                    slot_byte = b.mul(b.mod(g, ci(NS)), ci(SLOT))
                    nks = spec.ksteps(q) // CS
                    for s_ in range(nks):
                        gs = q * (CK // WMMA_K // CS) + s_
                        if s_ + 1 < nks:
                            if compute:
                                nxt = _bload(slot_byte, pxb, s_ + 1)
                        else:
                            # publish chunk g + 1 now; refill this chunk's slot
                            h = b.add(g, c1)
                            with b.scf_if_else(b.cmp_lt(h, total_chunks)) as (
                                t_ctx,
                                e_ctx,
                            ):
                                with t_ctx:
                                    _wait_publish(h, role)
                                    if moves:
                                        boff, qq = divmod(q + NS, NC)
                                        tgt = b.add(ib, ci(boff))
                                        with b.scf_if(b.cmp_lt(tgt, n_mine)):
                                            _issue(tgt, qq, b.add(g, ci(NS)))
                                with e_ctx:
                                    b.sync_lds_only()
                            if compute:
                                nslot = b.mul(b.mod(h, ci(NS)), ci(SLOT))
                                if q + 1 < NC:
                                    nxt = _bload(nslot, pxb, 0)
                                else:
                                    nxt = _bload(
                                        nslot, _px_base(_block(b.add(ib, c1))), 0
                                    )
                        if compute:
                            if spec.setprio:
                                b.s_setprio(1)
                            bfr = _bcompute(raw)
                            _mmas(accs, bfr, gs)
                            if spec.setprio:
                                b.s_setprio(0)
                            raw = nxt
                if compute:
                    _stage_write(accs, _soff(ib))
                    # block ib - 1 was staged before this block's publish barriers
                    with b.scf_if(b.cmp_gt(ib, c0)):
                        prev = b.sub(ib, c1)
                        _combine(_block(prev), _soff(prev))
                    b.scf_yield(*_flat(raw))
            with b.scf_if(b.cmp_gt(n_mine, c0)):
                b.sync_lds_only()
                if compute:
                    last = b.sub(n_mine, c1)
                    _combine(_block(last), _soff(last))

        if spec.tdm == "producer":
            with b.scf_if_else(is_prod) as (p_ctx, c_ctx):
                with p_ctx:
                    _early("producer")
                with c_ctx:
                    _early("consumer")
        else:
            _early("all")
        return b.kernel

    # ---- main persistent loop
    with b.scf_if(b.cmp_gt(n_mine, c0)):
        for g in range(NS - 1):
            _issue_global(g, c0)
    zero8 = b.zero_vec_f32(8)
    loop = b.scf_for(c0, n_mine, c1, iv_name="ib")
    with loop as ib:
        blk = _block(ib)
        # per-lane patch pixel byte offsets (slot-relative) of the H terms
        px_base = _px_base(blk)
        accs = {
            (c, i, f): zero8 for c in range(n) for i in range(FN) for f in range(FM)
        }
        for q in range(NC):
            gidx = b.add(b.mul(ib, ci(NC)), ci(q))
            _wait_publish(gidx)
            _issue_global_from = q + NS - 1
            boff, qq = divmod(_issue_global_from, NC)
            tgt = b.add(ib, ci(boff))
            with b.scf_if(b.cmp_lt(tgt, n_mine)):
                _issue(tgt, qq, b.add(gidx, ci(NS - 1)))
            slot_byte = b.mul(b.mod(gidx, ci(NS)), ci(SLOT))
            if spec.setprio:
                b.s_setprio(1)
            nks = spec.ksteps(q) // CS
            raw = _bload(slot_byte, px_base, 0)
            for s in range(nks):
                gs = q * (CK // WMMA_K // CS) + s
                if spec.prefetch:
                    nxt = _bload(slot_byte, px_base, s + 1) if s + 1 < nks else None
                    bfr = _bcompute(raw)
                    raw = nxt
                else:
                    if s:
                        raw = _bload(slot_byte, px_base, s)
                    bfr = _bcompute(raw)
                for c in range(n):
                    for i in range(FN):
                        for f in range(FM):
                            accs[(c, i, f)] = b.mma(
                                wmma, ufr[(c, i, gs)], bfr[(c, f)], accs[(c, i, f)]
                            )
            if spec.setprio:
                b.s_setprio(0)
        _stage_write(accs, ci(STAGE_OFF))
        b.sync_lds_only()
        _combine(blk, ci(STAGE_OFF))
    return b.kernel


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def fused_launch_values(
    spec: WinogradFusedSpec, x: int, u: int, y: int, P: int
) -> dict:
    return dict(x=x, u=u, y=y, P=int(P))


def fused_flops_gemm(spec: WinogradFusedSpec) -> int:
    """MACs * 2 of the in-kernel per-bin GEMMs (over the padded tiles)."""
    p = spec.problem
    tiles = p.N * p.dH * p.dW * spec.tiles_h * spec.tiles_w
    return 2 * tiles * spec.n**2 * p.cpg * p.kpg * p.groups


def tuned_fused_options(problem: ConvProblem, m: int = 2) -> dict:
    """A reasonable default configuration (see the report)."""
    return dict(m=m, fm=1, fn=2, ks=2 if m == 2 else 1, ck=32, stages=2)


def fused_bounds_check(spec: WinogradFusedSpec, max_blocks: int = 200000) -> None:
    """Host-side simulation of every block's patch transfers: raises if a
    TDM could read outside ``x`` or write outside its LDS slot (mirrors the
    kernel's index math)."""
    p = spec.problem
    m, n, G = spec.m, spec.n, spec.group_px
    npr = spec.npr
    xs = p.N * p.Hi * p.Wi * p.C
    slot_end = spec.px_byte(spec.slot_px)
    if spec.slot_bytes < slot_end:
        raise AssertionError("slot too small")
    nb = spec.num_blocks
    for blk in range(min(nb, max_blocks)):
        tjb = blk % spec.col_blocks
        rest = blk // spec.col_blocks
        ti = (rest % spec.row_blocks) * spec.bh
        rest //= spec.row_blocks
        bs = rest % p.dW
        rest //= p.dW
        a = rest % p.dH
        n_img = rest // p.dH
        tj0 = tjb * spec.twr
        neg = p.pW - (bs + tj0 * p.dW * m)
        v0 = (neg + p.dW - 1) // p.dW if neg > 0 else 0
        shift = (G - v0 % G) % G
        wstart = bs - p.pW + p.dW * (tj0 * m + v0)
        left = p.Wi - wstart
        ncols = min((left + p.dW - 1) // p.dW, spec.npx - v0) if left > 0 else 0
        if wstart < 0:
            raise AssertionError(f"block {blk}: negative start column")
        # LDS write intervals of one chunk (pixel units of the slot). A
        # transfer writes image data on [dst, dst + ncols) and zeros on the
        # rest of its box; ds_store ranges write zeros. Data intervals must
        # not overlap any other write (zero on zero is harmless), and every
        # pixel the lanes read must be written.
        data, zeros = [], []
        if spec.tdm == "rows":
            for u in range(n):
                hi = a - p.pH + p.dH * (ti * m + u)
                dst = u * spec.npxb + v0 + shift
                if 0 <= hi < p.Hi and ncols > 0:
                    data.append((dst, dst + ncols, f"tdm{u}"))
                    zeros.append((dst + ncols, dst + spec.npxb, f"tdm{u}-tail"))
                    if u == 0 and v0 > 0:
                        zeros.append((shift, dst, "pad0"))
                else:
                    zeros.append((u * spec.npxb + shift, dst + spec.npxb, f"zero{u}"))
        else:
            hi0 = a - p.pH + p.dH * ti * m
            u0 = min(npr, -(hi0 // p.dH) if hi0 < 0 else 0)
            hs = hi0 + p.dH * u0
            left_h = p.Hi - hs
            nrows = min((left_h + p.dH - 1) // p.dH, npr - u0) if left_h > 0 else 0
            if nrows > 0 and ncols > 0:
                zeros.append((shift, u0 * spec.npxb + v0 + shift, "zero-top"))
                for u in range(u0, npr):
                    dst = u * spec.npxb + v0 + shift
                    if u < u0 + nrows:
                        if hs + p.dH * (u - u0) >= p.Hi:
                            raise AssertionError(f"block {blk}: 3-D row {u} OOB")
                        data.append((dst, dst + ncols, f"tdm{u}"))
                        zeros.append((dst + ncols, dst + spec.npxb, f"tdm{u}-tail"))
                    else:
                        zeros.append((dst, dst + spec.npxb, f"tdm{u}-tail"))
                if spec.px_byte(npr * spec.npxb + v0 + shift) > slot_end:
                    raise AssertionError(f"block {blk}: 3-D box overruns the slot")
            else:
                zeros.append((shift, shift + npr * spec.npxb, "zero-all"))
        for d0, d1, dn in data:
            for w0, w1, wn in data + zeros:
                if wn != dn and wn != dn + "-tail" and w0 < d1 and d0 < w1:
                    raise AssertionError(
                        f"block {blk}: LDS writes {dn} and {wn} overlap"
                    )
        allw = data + zeros
        for u in range(npr):
            for v in range(spec.npx):
                px = u * spec.npxb + shift + v
                if not any(w0 <= px < w1 for w0, w1, _ in allw):
                    raise AssertionError(f"block {blk}: pixel ({u},{v}) never written")
        for u in range(npr):
            hi = a - p.pH + p.dH * (ti * m + u)
            dst = u * spec.npxb + v0 + shift
            if spec.px_byte(dst + spec.npxb) > slot_end:
                raise AssertionError(f"block {blk} row {u}: LDS overrun")
            if not (0 <= hi < p.Hi and ncols > 0):
                continue
            for gi in range(p.groups):
                for q in range(spec.nchunks):
                    c0 = q * spec.ck
                    pix = (n_img * p.Hi + hi) * p.Wi + wstart
                    first = pix * p.C + gi * p.cpg + c0
                    last = (pix + (ncols - 1) * p.dW) * p.C + gi * p.cpg + p.cpg - 1
                    if first < 0 or last >= xs:
                        raise AssertionError(f"block {blk} row {u}: x read OOB")
                    if wstart + (ncols - 1) * p.dW >= p.Wi:
                        raise AssertionError(f"block {blk} row {u}: column OOB")


def fused_output_coverage_check(spec: WinogradFusedSpec) -> None:
    """Host simulation of the block -> output mapping: every output pixel
    must be stored by exactly one (block, tile, i, j) (small problems)."""
    p = spec.problem
    m = spec.m
    seen = {}
    for blk in range(spec.num_blocks):
        tjb = blk % spec.col_blocks
        rest = blk // spec.col_blocks
        ti0 = (rest % spec.row_blocks) * spec.bh
        rest //= spec.row_blocks
        bs = rest % p.dW
        rest //= p.dW
        a = rest % p.dH
        n_img = rest // p.dH
        for tile in range(spec.tw):
            tr, tc = divmod(tile, spec.twr)
            for i in range(m):
                ho = a + p.dH * ((ti0 + tr) * m + i)
                for j in range(m):
                    wo = bs + p.dW * ((tjb * spec.twr + tc) * m + j)
                    if ho < p.Ho and wo < p.Wo:
                        key = (n_img, ho, wo)
                        if key in seen:
                            raise AssertionError(f"output {key} stored twice")
                        seen[key] = blk
    if len(seen) != p.N * p.Ho * p.Wo:
        raise AssertionError(f"{p.N * p.Ho * p.Wo - len(seen)} outputs never stored")


__all__ = [
    "WinogradFusedSpec",
    "build_winograd_fused",
    "fused_reason",
    "fused_signature",
    "fused_launch_values",
    "fused_flops_gemm",
    "tuned_fused_options",
    "fused_bounds_check",
    "fused_output_coverage_check",
]
