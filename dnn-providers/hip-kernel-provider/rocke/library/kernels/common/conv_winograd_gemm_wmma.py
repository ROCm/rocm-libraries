# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched GEMM of the Winograd ``F(m x m, 3 x 3)`` forward conv (K3), gfx1250.

The middle stage of the unfused Winograd pipeline
(:mod:`kernels.common.conv_winograd_fwd`) is ``G = n2 * g`` independent GEMMs,
one per (bin, group)::

    M[t][G][Kg] = sum_c V[t][G][c] * U[G][Kg][c]          c over Cg

``V`` and ``M`` keep the conv's absolute channel numbering (row ``t`` of
batch ``G`` starts at ``(t * G + G) * Cg``), ``U`` is GKYXC with Y = X = 1.
Both operands are contiguous along the reduction axis, so neither needs a
transposed LDS read. Operands are fp16 or bf16 with fp32 accumulation; the
output has the operand dtype, or is an fp32 split-K slab.

Per workgroup, ``tile_m`` tiles (rows of V / M) by ``tile_n`` output channels
(rows of U), reduced over ``tile_k``-channel chunks::

    for chunk in range(chunks of this split):          # runtime scf.for
        TDM V[m0 : m0 + tile_m][c0 : c0 + tile_k] -> LDS ring slot (wave 0)
        TDM U[n0 : n0 + tile_n][c0 : c0 + tile_k] -> LDS ring slot (wave 1)
        for k-step of 32:                              # Python-unrolled
            A = U fragment (16 channels x 32 c), B = V fragment (16 tiles x 32 c)
            acc[n-frag, m-frag] = wmma_16x16x32(A, B, acc)

The WMMA computes the transposed product ``M^T = U V^T``: an accumulator lane
then holds one tile and 8 consecutive output channels, i.e. 16 contiguous
bytes of ``M``. The transfers clip their contiguous extent to the chunk, so
the hardware zero-fills every row or channel past ``T`` / ``Kg`` / ``Cg``:
tails need no masking in the main loop. Each LDS row carries 16 bytes of
skew so the 16 rows of a fragment read land on different banks; the TDM
writes the skew itself (``lds_pad="tdm"``, power-of-two ``tile_k``) or writes
``tile_k + 8`` columns whose last 8 it clips to zero (``"clip"``, measured
slower).

Pipeline: the slots form a ring of ``stages``; one barrier per chunk
publishes the oldest slot, after which the slot the previous chunk read is
refilled ``stages - 1`` chunks ahead (the scheme of
``conv_direct_dgrad_wmma``). With ``prefetch`` the fragments of the next
k-step are read before the WMMAs of the current one. (Carrying the next
chunk's first fragments across the barrier was measured slower: it shortens
the time the transfer has to land, and the loop is transfer-bound.)

Blocks run on a flat grid; ``xcd_remap`` makes the blocks of one XCD (dealt
round-robin by the hardware) consecutive tiles, which share V and U rows.

Epilogue ``tdm``: the tile is staged in LDS as [tile][channel] and stored with
one TDM transfer, which clips it to ``T`` and ``Kg``. Epilogue ``direct``:
every lane stores its 16 bytes through a buffer resource (out-of-range rows
are dropped).

Split-K (``split_k=True``): ``G * splits`` batches of blocks; split ``s``
reduces chunks ``[s * cps, (s + 1) * cps)`` and writes an fp32 slab
``W[s][t][G][Kg]``; :func:`build_winograd_gemm_reduce` sums the slabs into
``M``. Meant for the small-T rows (few output tiles per batch).

``T``, ``G``, ``Cg``, ``Kg``, the split count, chunks per split and the
batch range of the launch (``g0``, ``nb``) are kernargs; the tile geometry
and dtype are build-time. A batch range lets a host plan run most batches
with one tile config and the last partial round of blocks with another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join

WMMA_K = 32  # K of the 16x16x32 atom
CH_PAD = 8  # elements of row skew in the operand ring (16 bytes)
# Estimated accumulator + fragment VGPRs per lane. Past 256 the backend
# addresses the upper VGPR banks (s_set_vgpr_msb); a SIMD holds 1024 per lane,
# so two waves per SIMD (8-wave blocks) get 512 each, of which the estimate
# leaves ~30 for addresses (measured: 8x5 fragments with prefetch spill).
VGPR_BUDGET = 480
_I32_MAX = (1 << 31) - 1
_OOB = 0x7FFFFF00  # buffer offset that loads 0 / drops the store
REDUCE_BLOCK = 256
EPILOGUES = ("tdm", "direct")
LDS_PADS = ("tdm", "clip")
# LLVM flavors the GEMM lowers at: every configuration stages its operands
# with ``tensor_load_to_lds``, which only the llvm23 flavor lowers.
LLVM_FLAVORS = ("llvm23",)


def _io_type(dtype: str):
    return BF16 if dtype == "bf16" else F16


@dataclass(frozen=True)
class WinogradGemmProblem:
    """``G`` GEMMs ``[T x Cg] x [Cg x Kg]`` in the Winograd workspace layouts."""

    T: int
    G: int
    Cg: int
    Kg: int

    @classmethod
    def from_conv(cls, problem, m: int) -> "WinogradGemmProblem":
        from kernels.common.conv_winograd_fwd import winograd_gemm_problem

        gp = winograd_gemm_problem(problem, m)
        return cls(T=gp.N, G=gp.groups, Cg=gp.C // gp.groups, Kg=gp.K // gp.groups)

    @property
    def flops(self) -> int:
        return 2 * self.T * self.G * self.Cg * self.Kg

    def bytes(self, elem: int = 2) -> int:
        """Unique bytes moved: V + U + M."""
        return elem * self.G * (self.T * self.Cg + self.Kg * self.Cg + self.T * self.Kg)

    def short(self) -> str:
        return f"T{self.T}_G{self.G}_C{self.Cg}_K{self.Kg}"


@dataclass(frozen=True)
class WinogradGemmWmmaSpec:
    """One configuration of :func:`build_winograd_gemm_wmma`.

    GEMM naming: ``m`` runs over tiles (rows of V and M), ``n`` over output
    channels (rows of U), ``k`` over input channels.
    """

    dtype: str = "fp16"  # operands and M (fp16 | bf16)
    name: str = "wino_gemm_wmma"
    waves_m: int = 2  # waves along tiles
    waves_n: int = 4  # waves along output channels
    frags_m: int = 8  # 16-tile fragments per wave
    frags_n: int = 4  # 16-channel fragments per wave
    tile_k: int = 128  # channels per LDS stage (multiple of 32)
    stages: int = 2  # LDS ring depth
    lds_pad: str = "tdm"  # "tdm" | "clip" (see the module docstring)
    epilogue: str = "tdm"  # "tdm" | "direct"
    split_k: bool = False  # fp32 slab output + reduce kernel
    # Read the next k-step's fragments before this step's WMMAs.
    prefetch: bool = True
    xcd_remap: bool = True
    num_xcds: int = 8
    # AMDGPU machine-scheduler strategy (rocke.core.codegen_policy), or None.
    sched: Optional[str] = "iterative-ilp"
    wave_size: int = 32
    # Segmented (long-filter) Winograd: M[t] = sum_s V[t + s] U[s]. The
    # reduction runs over (tap, channel chunk); V needs T + taps - 1 rows and
    # U is laid out [G][taps][Kg][Cg]. 1 = the plain batched GEMM.
    taps: int = 1

    @property
    def tile_m(self) -> int:
        return self.waves_m * self.frags_m * 16

    @property
    def tile_n(self) -> int:
        return self.waves_n * self.frags_n * 16

    @property
    def waves(self) -> int:
        return self.waves_m * self.waves_n

    @property
    def threads_per_block(self) -> int:
        return self.waves * self.wave_size

    @property
    def pitch(self) -> int:
        return self.tile_k + CH_PAD

    @property
    def stage_elems(self) -> int:
        return (self.tile_m + self.tile_n) * self.pitch

    @property
    def out_elem_bytes(self) -> int:
        return 4 if self.split_k else 2

    @property
    def out_pitch(self) -> int:
        # 16 bytes of skew per staged row.
        return self.tile_n + 16 // self.out_elem_bytes

    def lds_bytes(self) -> int:
        ring = 2 * self.stages * self.stage_elems
        epi = self.tile_m * self.out_pitch * self.out_elem_bytes
        return max(ring, epi if self.epilogue == "tdm" else 0)

    def vgpr_estimate(self) -> int:
        frags = 8 * (self.frags_m + self.frags_n)
        return 8 * self.frags_m * self.frags_n + frags * (2 if self.prefetch else 1)

    def label(self) -> str:
        return (
            f"{self.tile_m}x{self.tile_n}x{self.tile_k}_w{self.waves_m}x{self.waves_n}"
            f"_f{self.frags_m}x{self.frags_n}_s{self.stages}_{self.epilogue}"
            + ("_sk" if self.split_k else "")
            + ("" if self.prefetch else "_nopf")
            + ("" if self.lds_pad == "tdm" else f"_p{self.lds_pad}")
            + ("" if self.xcd_remap else "_noxcd")
            + ("" if self.sched else "_nosched")
            + (f"_t{self.taps}" if self.taps > 1 else "")
        )

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"wm{self.waves_m}",
            f"wn{self.waves_n}",
            f"fm{self.frags_m}",
            f"fn{self.frags_n}",
            f"tk{self.tile_k}",
            f"st{self.stages}",
            f"p{self.lds_pad}",
            self.epilogue,
            "sk" if self.split_k else "",
            "pf" if self.prefetch else "",
            f"x{self.num_xcds}" if self.xcd_remap else "",
            f"s{self.sched}" if self.sched else "",
            f"taps{self.taps}" if self.taps > 1 else "",
            flags={"bf16": self.dtype == "bf16"},
        )

    # -- launch --

    def chunks(self, prob: WinogradGemmProblem) -> int:
        return self.taps * ceil_div(prob.Cg, self.tile_k)

    def resolve_splits(self, prob: WinogradGemmProblem, splits: int) -> Tuple[int, int]:
        """``(splits, chunks per split)`` with no empty split (``splits`` is
        clamped to ``1 .. chunks``; 1 when ``split_k`` is off)."""
        n = self.chunks(prob)
        if not self.split_k:
            return 1, n
        s = max(1, min(int(splits), n))
        cps = ceil_div(n, s)
        return ceil_div(n, cps), cps

    def tiles(
        self,
        prob: WinogradGemmProblem,
        splits: int = 1,
        batches: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, int]:
        """Blocks along (tiles, channels, batch * split); ``batches`` =
        ``(first, count)`` restricts the launch to a range of the G batches."""
        s, _ = self.resolve_splits(prob, splits)
        nb = prob.G if batches is None else batches[1]
        return (ceil_div(prob.T, self.tile_m), ceil_div(prob.Kg, self.tile_n), nb * s)

    def grid(
        self,
        prob: WinogradGemmProblem,
        splits: int = 1,
        batches: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, int]:
        """Flat grid of one block per tile."""
        gx, gy, gz = self.tiles(prob, splits, batches)
        return (gx * gy * gz, 1, 1)

    def workspace_bytes(self, prob: WinogradGemmProblem, splits: int = 1) -> int:
        """fp32 slab bytes of the split-K mode (0 otherwise)."""
        if not self.split_k:
            return 0
        s, _ = self.resolve_splits(prob, splits)
        return 4 * s * prob.T * prob.G * prob.Kg

    def launch_values(
        self,
        prob: WinogradGemmProblem,
        v: int,
        u: int,
        d: int,
        splits: int = 1,
        batches: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """Kernarg values; ``d`` is M (or the fp32 slab with ``split_k``)."""
        s, cps = self.resolve_splits(prob, splits)
        g0, nb = (0, prob.G) if batches is None else batches
        return dict(
            v=v,
            u=u,
            d=d,
            T=prob.T,
            G=prob.G,
            Cg=prob.Cg,
            Kg=prob.Kg,
            splits=s,
            cps=cps,
            g0=g0,
            nb=nb,
        )


def winograd_gemm_signature(spec: WinogradGemmWmmaSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("v", spec.dtype)
        .ptr("u", spec.dtype)
        .ptr("d", "f32" if spec.split_k else spec.dtype)
        .scalar("T", "i32")
        .scalar("G", "i32")
        .scalar("Cg", "i32")
        .scalar("Kg", "i32")
        .scalar("splits", "i32")
        .scalar("cps", "i32")
        .scalar("g0", "i32")
        .scalar("nb", "i32")
        .build()
    )


def winograd_gemm_reason(
    spec: WinogradGemmWmmaSpec, arch: str = "gfx1250"
) -> Optional[str]:
    """Why ``spec`` cannot be built for ``arch`` (None when it can)."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    if target.wave_size != 32 or spec.wave_size != 32:
        return "wino_gemm_wmma is a wave32 kernel"
    if not target.memory.has_tdm:
        return f"no TDM on {arch}"
    if spec.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {spec.dtype!r}"
    ab = "bf16" if spec.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing wmma_f32_16x16x32_{ab} on {arch}"
    if min(spec.waves_m, spec.waves_n, spec.frags_m, spec.frags_n) < 1:
        return "waves and fragments must be >= 1"
    if spec.tile_k <= 0 or spec.tile_k % WMMA_K:
        return f"tile_k {spec.tile_k} must be a positive multiple of {WMMA_K}"
    if spec.stages < 2:
        return "stages must be >= 2"
    if spec.taps < 1:
        return "taps must be >= 1"
    if spec.epilogue not in EPILOGUES:
        return f"epilogue {spec.epilogue!r} not in {EPILOGUES}"
    if spec.lds_pad not in LDS_PADS:
        return f"lds_pad {spec.lds_pad!r} not in {LDS_PADS}"
    if spec.lds_pad == "tdm" and (
        spec.tile_k & (spec.tile_k - 1) or not 32 <= spec.tile_k <= 512
    ):
        return "lds_pad tdm needs a power-of-two tile_k in 32..512"
    if spec.num_xcds < 1:
        return f"num_xcds {spec.num_xcds} < 1"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if max(spec.tile_m, spec.tile_n, spec.pitch, spec.out_pitch) > 0xFFFF:
        return "tile too large for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    if spec.vgpr_estimate() > VGPR_BUDGET * (2 if spec.waves <= 4 else 1):
        return f"~{spec.vgpr_estimate()} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    return None


def winograd_gemm_problem_reason(
    spec: WinogradGemmWmmaSpec,
    prob: WinogradGemmProblem,
    splits: int = 1,
    batches: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    """Why ``spec`` cannot run ``prob`` (None when it can)."""
    if min(prob.T, prob.G, prob.Cg, prob.Kg) < 1:
        return f"empty problem {prob}"
    if batches is not None:
        g0, nb = batches
        if g0 < 0 or nb < 1 or g0 + nb > prob.G:
            return f"batch range {batches} outside 0..{prob.G}"
    if prob.Cg % 8 or prob.Kg % 8:
        return f"Cg {prob.Cg} and Kg {prob.Kg} must be multiples of 8 (16-byte rows)"
    if spec.split_k and splits < 1:
        return f"split count {splits} < 1"
    elems = (
        (prob.T + spec.taps - 1) * prob.G * prob.Cg,
        prob.G * spec.taps * prob.Kg * prob.Cg,
        prob.T * prob.G * prob.Kg,
    )
    if 2 * max(elems) > _I32_MAX:
        return "an operand exceeds the i32 byte offsets"
    if spec.workspace_bytes(prob, splits) > _I32_MAX:
        return "the split-K slab exceeds the i32 byte offsets"
    gx, gy, gz = spec.grid(prob, splits, batches)
    if min(gx, gy, gz) < 1 or max(gy, gz) > 65535 or gx >= _I32_MAX:
        return f"grid {(gx, gy, gz)} not launchable"
    return None


def build_winograd_gemm_wmma(
    spec: WinogradGemmWmmaSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    why = winograd_gemm_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid WinogradGemmWmmaSpec for {arch}: {why}")

    io_type = _io_type(spec.dtype)
    ab = "bf16" if spec.dtype == "bf16" else "f16"
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )
    WAVE = spec.wave_size
    WN = spec.waves_n
    FM, FN = spec.frags_m, spec.frags_n
    BM, BN, CK = spec.tile_m, spec.tile_n, spec.tile_k
    NS = spec.stages
    P = spec.pitch
    STAGE = spec.stage_elems
    U_OFF = BM * P  # the U tile follows the V tile inside a slot
    KSTEPS = CK // WMMA_K
    U_WAVE = 1 if spec.waves > 1 else 0
    OPS_PER_CHUNK = 1 if spec.waves > 1 else 2  # TDM transfers per issuing wave

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    out_ty = F32 if spec.split_k else io_type
    V = b.param("v", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    U = b.param("u", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("d", PtrType(out_ty, "global"), noalias=True, writeonly=True, align=16)
    c_T = b.param("T", I32)
    c_G = b.param("G", I32)
    c_Cg = b.param("Cg", I32)
    c_Kg = b.param("Kg", I32)
    c_splits = b.param("splits", I32)
    c_cps = b.param("cps", I32)
    c_g0 = b.param("g0", I32)  # first batch of this launch
    c_nb = b.param("nb", I32)  # batches in this launch

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    tid = b.thread_id_x()
    wave_id = b.div(tid, b.const_i32(WAVE))
    lane = b.mod(tid, b.const_i32(WAVE))
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_n = b.mod(wave_id, b.const_i32(WN))
    wave_m = b.div(wave_id, b.const_i32(WN))

    nx = b.div(b.add(c_T, b.const_i32(BM - 1)), b.const_i32(BM))
    ny = b.div(b.add(c_Kg, b.const_i32(BN - 1)), b.const_i32(BN))
    nxy = b.mul(nx, ny)
    n_tiles = b.mul(nxy, b.mul(c_nb, c_splits))
    TAPS = spec.taps
    n_cc = b.div(b.add(c_Cg, b.const_i32(CK - 1)), b.const_i32(CK))
    n_chunks = b.mul(n_cc, b.const_i32(TAPS)) if TAPS > 1 else n_cc
    row_v = b.mul(c_G, c_Cg)  # element stride between V rows
    out_row = b.mul(c_G, c_Kg)  # element stride between M rows

    def _remap(f: Value, n: Value) -> Value:
        if not spec.xcd_remap:
            return f
        from rocke.helpers.grid import chiplet_transform_chunked_runtime

        return chiplet_transform_chunked_runtime(
            b, f, num_wgs=n, num_xcds=spec.num_xcds
        )

    @dataclass
    class _Tile:
        m0: Value
        n0: Value
        g: Value
        split: Value
        m_ext: Value
        n_ext: Value
        v_base: Value
        u_base: Value
        d_base: Value

    def _tile(f: Value) -> _Tile:
        """Coordinates of flat tile ``f`` (x fastest, then channels, batch,
        split): consecutive tiles share V or U rows."""
        bx = b.mod(f, nx)
        by = b.div(b.mod(f, nxy), nx)
        z = b.div(f, nxy)
        m0 = b.mul(bx, b.const_i32(BM))
        n0 = b.mul(by, b.const_i32(BN))
        g = b.add(c_g0, b.mod(z, c_nb))
        split = b.div(z, c_nb)
        # M[t][g][k] (+ the slab offset split * T * G * Kg for split-K)
        d_base = b.add(b.mul(m0, out_row), b.add(b.mul(g, c_Kg), n0))
        if spec.split_k:
            d_base = b.add(d_base, b.mul(split, b.mul(c_T, out_row)))
        return _Tile(
            m0=m0,
            n0=n0,
            g=g,
            split=split,
            m_ext=b.sub(c_T, m0),  # > 0 by the grid
            n_ext=b.sub(c_Kg, n0),
            v_base=b.add(b.mul(m0, row_v), b.mul(g, c_Cg)),
            u_base=b.mul(b.add(b.mul(g, c_Kg), n0), c_Cg),
            d_base=d_base,
        )

    cur = _tile(_remap(b.block_id_x(), n_tiles))
    chunk0 = b.mul(cur.split, c_cps)
    n_loc = b.smax(b.smin(b.sub(n_chunks, chunk0), c_cps), c0)

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="wg_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)

    pad_kw = {}
    tile0 = P
    if spec.lds_pad == "tdm":
        # The TDM pads 4 dwords after every tile_k / 2 dwords; the fields
        # encode 2^(interval + 1) and amount + 1 dwords.
        pad_kw = dict(pad_interval=(CK // 2).bit_length() - 2, pad_amount=3)
        tile0 = CK

    def _issue(q: Value) -> None:
        """TDM local chunk ``q`` into ring slot q % NS."""
        slot = b.mod(q, b.const_i32(NS))
        slot_addr = b.add(lds_addr, b.mul(slot, b.const_i32(STAGE * 2)))
        for wave, opnd in ((0, 0), (U_WAVE, 1)):
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(wave))):
                t = cur
                if TAPS > 1:
                    # chunk -> (tap, channel chunk). V rows shift by the tap
                    # (V has T + taps - 1 rows, so the extent stays >= 1);
                    # U moves to slab ((g * taps + tap) * Kg + n0) * Cg.
                    gq = b.add(chunk0, q)
                    tap = b.div(gq, n_cc)
                    c_k0 = b.mul(b.sub(gq, b.mul(tap, n_cc)), b.const_i32(CK))
                    v_base = b.add(t.v_base, b.mul(tap, row_v))
                    v_ext = b.sub(b.add(t.m_ext, b.const_i32(TAPS - 1)), tap)
                    slab = b.add(b.mul(t.g, b.const_i32(TAPS - 1)), tap)
                    u_base = b.add(b.mul(slab, b.mul(c_Kg, c_Cg)), t.u_base)
                else:
                    c_k0 = b.mul(b.add(chunk0, q), b.const_i32(CK))
                    v_base, v_ext, u_base = t.v_base, t.m_ext, t.u_base
                k_ext = b.smin(b.sub(c_Cg, c_k0), b.const_i32(CK))
                ptr, base, ext, stride, rows, lds_off = (
                    (V, v_base, v_ext, row_v, BM, 0)
                    if opnd == 0
                    else (U, u_base, t.n_ext, c_Cg, BN, U_OFF)
                )
                b.tensor_load_to_lds(
                    *b.tdm_descriptor_2d(
                        b.global_ptr_add(ptr, b.mul(b.add(base, c_k0), c2)),
                        b.add(slot_addr, b.const_i32(lds_off * 2)),
                        elem_bytes=2,
                        tensor_dim0=k_ext,
                        tensor_dim1=ext,
                        row_stride=stride,
                        tile_dim0=tile0,
                        tile_dim1=rows,
                        **pad_kw,
                    )
                )

    def _wait_for(q: Value) -> None:
        """Wait until chunk ``q`` (the oldest transfer in flight) landed;
        the younger ones exist only while chunks remain to be issued."""
        if NS == 2:
            b.s_wait_tensorcnt(0)
            return
        more = b.cmp_lt(b.add(q, b.const_i32(NS - 2)), n_loc)
        with b.scf_if_else(more) as (then_ctx, else_ctx):
            with then_ctx:
                b.s_wait_tensorcnt((NS - 2) * OPS_PER_CHUNK)
            with else_ctx:
                b.s_wait_tensorcnt(0)

    # ---- per-lane fragment offsets: row l % 16 of the fragment, channels
    # (l // 16) * 16 .. +15 of the 32-wide k-step (same split for A and B).
    col = b.mul(lhalf, c16)

    def _frag_off(wave_pos: Value, nfrag: int, f: int, extra: int) -> Value:
        row = b.add(
            b.mul(b.add(b.mul(wave_pos, b.const_i32(nfrag)), b.const_i32(f)), c16), l16
        )
        return b.add(b.mul(row, b.const_i32(P)), b.add(col, b.const_i32(extra)))

    v_off = [_frag_off(wave_m, FM, j, 0) for j in range(FM)]
    u_off = [_frag_off(wave_n, FN, i, U_OFF) for i in range(FN)]

    def _read(base: Value, off: Value, ks: int) -> Value:
        addr = b.add(base, b.add(off, b.const_i32(ks * WMMA_K)))
        lo = b.smem_load_vN(lds, c0, addr, dtype=io_type, n=8)
        hi = b.smem_load_vN(lds, c0, b.add(addr, c8), dtype=io_type, n=8)
        return b.vec_concat(lo, hi)

    def _frags(q: Value, ks: int) -> list:
        base = b.mul(b.mod(q, b.const_i32(NS)), b.const_i32(STAGE))
        return [_read(base, u_off[i], ks) for i in range(FN)] + [
            _read(base, v_off[j], ks) for j in range(FM)
        ]

    def _publish(q: Value) -> None:
        """Make chunk ``q`` visible to every wave and refill the slot of
        chunk ``q - 1`` (all its fragments are read) with chunk
        ``q + NS - 1``."""
        _wait_for(q)
        b.sync_lds_only()
        nxt = b.add(q, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_loc)):
            _issue(nxt)

    def _mmas(accs: list, frags: list) -> list:
        af, bf = frags[:FN], frags[FN:]
        for i in range(FN):
            for j in range(FM):
                k = i * FM + j
                accs[k] = b.mma(wmma_op, af[i], bf[j], accs[k])
        return accs

    # ---- epilogue. Lane l of fragment (i, j): tile row (.. j) * 16 + l % 16,
    # channels (.. i) * 16 + (l // 16) * 8 .. +7.
    def _rows_cols(i: int, j: int) -> Tuple[Value, Value]:
        r = b.add(
            b.mul(b.add(b.mul(wave_m, b.const_i32(FM)), b.const_i32(j)), c16), l16
        )
        c = b.add(
            b.mul(b.add(b.mul(wave_n, b.const_i32(FN)), b.const_i32(i)), c16),
            b.mul(lhalf, c8),
        )
        return r, c

    EB = spec.out_elem_bytes

    def _halves(acc: Value) -> Tuple[Value, Value]:
        lo = b.vec_pack([b.vec_extract(acc, e) for e in range(4)], F32)
        hi = b.vec_pack([b.vec_extract(acc, e) for e in range(4, 8)], F32)
        return lo, hi

    def _epilogue_tdm(accs: list, t: _Tile) -> None:
        """Stage the tile in LDS as [tile][channel] (reusing the ring) and
        store it with one TDM transfer that clips it to T and Kg."""
        OP = spec.out_pitch
        HPE = EB // 2  # staged index unit: halves per output element
        # Every wave is past its last ring read (and no transfer is in
        # flight) once all waves reach this barrier.
        b.sync_lds_only()
        for i in range(FN):
            for j in range(FM):
                r, c = _rows_cols(i, j)
                acc = accs[i * FM + j]
                idx = b.mul(b.add(b.mul(r, b.const_i32(OP)), c), b.const_i32(HPE))
                if spec.split_k:
                    lo, hi = _halves(acc)
                    b.smem_store_vN_f32(lds, [c0, idx], lo, n=4)
                    b.smem_store_vN_f32(lds, [c0, b.add(idx, c8)], hi, n=4)
                else:
                    val = b.vec_cast_f32_to(acc, io_type)
                    b.smem_store_vN(lds, [c0, idx], val, n=8)
        b.sync_lds_only()
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            dst = b.global_ptr_add(D, b.mul(t.d_base, b.const_i32(EB)))
            b.tensor_store_from_lds(
                *b.tdm_descriptor_2d(
                    dst,
                    lds_addr,
                    elem_bytes=EB,
                    tensor_dim0=b.smin(t.n_ext, b.const_i32(BN)),
                    tensor_dim1=t.m_ext,
                    row_stride=out_row,
                    tile_dim0=OP,
                    tile_dim1=BM,
                )
            )
            b.s_wait_tensorcnt(0)

    def _epilogue_direct(accs: list, t: _Tile) -> None:
        """Every lane stores its 16 bytes (rows past T / Kg are dropped)."""
        d_bytes = b.mul(b.mul(c_T, out_row), b.const_i32(EB))
        if spec.split_k:
            d_bytes = b.mul(d_bytes, c_splits)
        rsrc = b.buffer_rsrc(D, d_bytes)
        oob = b.const_i32(_OOB)
        for i in range(FN):
            for j in range(FM):
                r, c = _rows_cols(i, j)
                ok = b.land(
                    b.cmp_lt(b.add(t.m0, r), c_T), b.cmp_lt(b.add(t.n0, c), c_Kg)
                )
                off = b.add(t.d_base, b.add(b.mul(r, out_row), c))
                off = b.select(ok, b.mul(off, b.const_i32(EB)), oob)
                acc = accs[i * FM + j]
                if spec.split_k:
                    lo, hi = _halves(acc)
                    b.buffer_store_vN_f32(rsrc, off, c0, lo, 4)
                    b.buffer_store_vN_f32(rsrc, off, b.const_i32(16), hi, 4)
                elif spec.dtype == "bf16":
                    val = b.vec_cast_f32_to(acc, io_type)
                    b.buffer_store_vN_bf16(rsrc, off, c0, val, 4)
                else:
                    val = b.vec_cast_f32_to(acc, io_type)
                    b.buffer_store_vN_f16(rsrc, off, c0, val, 4)

    _epilogue = _epilogue_tdm if spec.epilogue == "tdm" else _epilogue_direct

    # ---- main loop
    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_loc)):
            _issue(b.const_i32(st))
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0, n_loc, c1, [(f"acc{k}", zero_acc) for k in range(FN * FM)], iv_name="q"
    )
    with loop as (q, vals):
        accs = list(vals)
        _publish(q)
        b.s_setprio(1)
        frags = _frags(q, 0)
        for ks in range(KSTEPS):
            if spec.prefetch:
                # The next k-step's reads go out before this step's WMMAs.
                nxt = _frags(q, ks + 1) if ks + 1 < KSTEPS else None
            else:
                nxt = None
                if ks:
                    frags = _frags(q, ks)
            accs = _mmas(accs, frags)
            frags = nxt
        b.s_setprio(0)
        b.scf_yield(*accs)
    _epilogue(list(loop.results), cur)
    return b.kernel


# ---------------------------------------------------------------------------
# Split-K reduce
# ---------------------------------------------------------------------------


def winograd_gemm_reduce_signature(dtype: str) -> list:
    return (
        SignatureBuilder()
        .ptr("w", "f32")
        .ptr("d", dtype)
        .scalar("n_vec", "i32")
        .scalar("splits", "i32")
        .build()
    )


def winograd_gemm_reduce_grid(prob: WinogradGemmProblem) -> Tuple[int, int, int]:
    return (ceil_div(prob.T * prob.G * prob.Kg // 8, REDUCE_BLOCK), 1, 1)


def winograd_gemm_reduce_values(
    prob: WinogradGemmProblem, w: int, d: int, splits: int
) -> dict:
    return dict(w=w, d=d, n_vec=prob.T * prob.G * prob.Kg // 8, splits=splits)


@dataclass(frozen=True)
class WinogradGemmReduceSpec:
    """The split-K slab reduce of the batched GEMM: ``dtype`` is the output
    (operand) dtype."""

    dtype: str = "fp16"


def build_winograd_gemm_reduce(
    spec: WinogradGemmReduceSpec, *, arch: str = "gfx1250"  # noqa: ARG001
) -> KernelDef:
    """``d[i] = sum_s w[s][i]`` in fp32, stored as ``spec.dtype``; 8 elements
    per thread. ``n_vec`` is the element count / 8 (Kg is a multiple of 8)."""
    dtype = spec.dtype
    io_type = _io_type(dtype)
    b = IRBuilder(kernel_name_join("wino_gemm_reduce", flags={"bf16": dtype == "bf16"}))
    b.kernel.attrs["max_workgroup_size"] = REDUCE_BLOCK
    W = b.param("w", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    D = b.param("d", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    n_vec = b.param("n_vec", I32)
    splits = b.param("splits", I32)
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    gid = b.add(b.mul(b.block_id_x(), b.const_i32(REDUCE_BLOCK)), b.thread_id_x())
    ok = b.cmp_lt(gid, n_vec)
    oob = b.const_i32(_OOB)
    slab = b.mul(n_vec, b.const_i32(32))  # bytes of one split slab
    w_rsrc = b.buffer_rsrc(W, b.mul(slab, splits))
    d_rsrc = b.buffer_rsrc(D, b.mul(n_vec, b.const_i32(16)))
    w_off = b.select(ok, b.mul(gid, b.const_i32(32)), oob)
    zero4 = b.zero_vec_f32(4)
    loop = b.scf_for_iter(c0, splits, c1, [("lo", zero4), ("hi", zero4)], iv_name="s")
    with loop as (s, carried):
        lo, hi = carried
        off = b.select(ok, b.add(w_off, b.mul(s, slab)), oob)
        lo = b.fadd(lo, b.buffer_load_vN(w_rsrc, off, c0, F32, 4))
        hi = b.fadd(hi, b.buffer_load_vN(w_rsrc, off, b.const_i32(16), F32, 4))
        b.scf_yield(lo, hi)
    lo, hi = loop.results
    val = b.vec_cast_f32_to(b.vec_concat(lo, hi), io_type)
    d_off = b.select(ok, b.mul(gid, b.const_i32(16)), oob)
    if dtype == "bf16":
        b.buffer_store_vN_bf16(d_rsrc, d_off, c0, val, 4)
    else:
        b.buffer_store_vN_f16(d_rsrc, d_off, c0, val, 4)
    return b.kernel
