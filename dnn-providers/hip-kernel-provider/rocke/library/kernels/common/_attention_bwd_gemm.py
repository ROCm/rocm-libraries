# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multi-wave block GEMM for the attention backward tile step.

One :class:`BlockGemm` describes ``C[M x N] += A[M x K] * B[K x N]`` computed by
the ``W`` waves of a CTA laid out as a warp grid ``(rows, cols)``:

* wave ``w`` owns the C sub-block at rows ``(w // cols) * M / rows`` and columns
  ``(w % cols) * N / cols``;
* inside a wave the sub-block is a static ``frags_m x frags_n`` array of 16x16
  atom tiles, and K advances in ``k_steps`` atom-K steps.

Supported warp grids are ``(1, W)``, ``(W, 1)``, ``(2, W / 2)`` and ``(1, 1)``.
The atom is the per-arch 16x16 atom chosen by :mod:`._attention_bwd_caps`:
``16x16x16`` everywhere, native ``16x16x32`` on gfx950 (a gfx942 K = 32 step is
two 16x16x16 instructions, i.e. ``atom_k = 16`` with twice the ``k_steps``),
wave32 WMMA on RDNA.

Operands are either already in registers (a ``[frag][k_step]`` list of lists,
e.g. a C-to-A relabel from :mod:`._attention_bwd_frag`) or an
:class:`LdsOperand` read fragment by fragment inside the K loop. Every
fragment index is a Python int (static fragment indexing): no dynamic register
indexing, so no scratch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

from rocke.core.arch.target import MmaOp
from rocke.core.ir import IRBuilder, Value

from ._attention_bwd_caps import (
    ASM_MFMA_VALU_LEAD_WAIT_STATES,
    bwd_arch_caps,
    check_waves,
    select_bwd_atom,
)
from ._attention_bwd_frag import (
    FragCoords,
    RelabelPlan,
    emit_c_to_a,
    kperm_b_coords,
    load_frag_lds,
    load_frag_lds_tr16,
    operand_coords,
    plan_c_to_a,
)

__all__ = [
    "BlockGemm",
    "LdsOperand",
    "check_relabel_grids",
    "kperm_b_operand_coords",
    "lane_and_wave",
    "relabel_acc_to_a",
    "mma_agpr_asm",
    "mma_agpr_asm_steps",
]

Frags = Sequence[Sequence[Value]]


@dataclass(frozen=True)
class LdsOperand:
    """An operand tile in LDS, read one fragment per K step inside the loop.

    ``storage`` is ``"k_inner"`` (``smem[mn][k]``) or ``"k_outer"``
    (``smem[k][mn]``). ``mn0``/``k0`` are the block origin inside the LDS tile
    (the wave origin is added by the GEMM). ``coords`` overrides the catalog
    operand coordinates (e.g. a K-permuted B map). ``reader`` selects the LDS
    read: ``"plain"`` (vector or element reads, any storage) or ``"tr16"``
    (``ds_read_tr16_b64`` of a ``"k_outer"`` tile, gfx950 only).
    """

    smem: Value
    storage: str = "k_inner"
    mn0: int = 0
    k0: int = 0
    coords: FragCoords | None = None
    align: int = 8
    reader: str = "plain"

    def __post_init__(self) -> None:
        if self.reader not in ("plain", "tr16"):
            raise ValueError(f"reader must be 'plain' or 'tr16', got {self.reader!r}")
        if self.reader == "tr16" and self.storage != "k_outer":
            raise ValueError("the tr16 reader reads K-outer tiles only")


def mma_agpr_asm(
    g: "BlockGemm",
    b: IRBuilder,
    acc: Sequence[Sequence[Value]],
    a_ks: Sequence[Value],
    b_ks: Sequence[Value],
    *,
    tail_nop: int | None = None,
    agpr_base: int | None = None,
    lead_wait: int = ASM_MFMA_VALU_LEAD_WAIT_STATES,
) -> list[list[Value]]:
    """One K step of a block GEMM as a single inline-asm MFMA cluster.

    Every accumulator ``acc[fm][fn]`` is an ``=a`` output tied to its input, so
    the block's accumulators live in AGPRs across steps with no copy; the A and
    B fragments of the step are ``v`` inputs, each passed once (bitcast to an
    i32 vector of the same width). The cluster ends with ``s_nop tail_nop``
    (default :data:`AGPR_ASM_TAIL_NOP` of the atom): the asm hides the MFMAs
    from the backend's hazard recognizer, so the cluster itself covers its
    result latency and its source read window. ``lead_wait`` as in
    :func:`mma_agpr_asm_steps`.
    """
    return mma_agpr_asm_steps(
        g, b, acc, [(a_ks, b_ks)], tail_nop=tail_nop, agpr_base=agpr_base,
        lead_wait=lead_wait,
    )  # fmt: skip


def mma_agpr_asm_steps(
    g: "BlockGemm",
    b: IRBuilder,
    acc: Sequence[Sequence[Value]] | None,
    steps: Sequence[tuple[Sequence[Value], Sequence[Value]]],
    *,
    tail_nop: int | None = None,
    agpr_base: int | None = None,
    lead_wait: int = ASM_MFMA_VALU_LEAD_WAIT_STATES,
) -> list[list[Value]]:
    """Several K steps of a block GEMM as one inline-asm MFMA cluster.

    ``steps`` is ``[(a_ks, b_ks), ...]`` in K order. The MFMAs are issued K
    outer, fragment inner (the accumulation order of the intrinsic route), so
    two MFMAs on the same accumulator are ``frags_m * frags_n - 1`` MFMAs
    apart; with a single fragment they would be adjacent, and ``s_nop
    tail_nop`` separates the K steps. The cluster ends with ``s_nop tail_nop``
    as in :func:`mma_agpr_asm`, which is this function with one step.

    ``lead_wait`` (0 to 16; default the ISA's
    ``ASM_MFMA_VALU_LEAD_WAIT_STATES``) opens the cluster with ``s_nop
    lead_wait - 1``, so its first MFMA issues at least that many wait states
    after the instruction before the block: the backend does not see the
    MFMAs inside the asm, so it does not separate a VALU write of a source
    register (an A / B fragment, or a register copy the allocator puts in
    front of the block) from the MFMA that reads it. ``0`` emits no lead
    (the control of the hazard audit; the backward body always passes the
    ISA's value).

    ``acc = None`` starts from zero accumulators: the first K step reads the
    inline constant ``0`` as srcC (as the intrinsic of a zero accumulator
    does), and the ``=a`` outputs have no tied inputs, so no AGPR is written
    with a zero first (no ``v_accvgpr_write`` per call).

    ``agpr_base`` pins output ``i`` to the physical AGPR tuple starting at
    ``agpr_base + i * c_frag_len`` (``={a[lo:hi]}``) instead of letting the
    allocator choose. Without it the production register allocator was seen
    to make room for short-lived asm outputs (S / dP) by moving part of the
    long-lived dK / dV accumulators to arch VGPRs and back every step, though
    the reservation had free AGPRs; pinned to the top of the reservation,
    they take those free AGPRs instead.
    """
    from rocke.core.ir import F32, I32, VectorType

    from kernels.common._attention_bwd_caps import (
        AGPR_ASM_TAIL_NOP,
        mfma_asm_mnemonic,
    )

    if not steps:
        raise ValueError("an asm MFMA cluster needs at least one K step")
    mnem = mfma_asm_mnemonic(g.arch, g.dtype, g.atom_k)
    nop = AGPR_ASM_TAIL_NOP[g.atom_k] if tail_nop is None else int(tail_nop)
    if not 0 <= nop <= 15:
        raise ValueError(f"s_nop count {nop} outside [0, 15]")
    if (
        isinstance(lead_wait, bool)
        or not isinstance(lead_wait, int)
        or not 0 <= lead_wait <= 16
    ):
        raise ValueError(f"lead wait states {lead_wait!r} outside [0, 16]")
    words = g.atom_k // 8  # 16-bit elements per lane / 2 per dword: K=16 -> 2
    src_ty = VectorType(I32, words)
    outs = [(fm, fn) for fm in range(g.frags_m) for fn in range(g.frags_n)]
    n = len(outs)
    zero = acc is None
    tied = 0 if zero else n  # tied accumulator inputs after the n outputs
    per_step = g.frags_m + g.frags_n
    srcs = []
    # The backend does not see the asm's source reads, so a VALU write of a
    # source VGPR may be scheduled right before the block: the leading s_nop
    # gives the ISA's VALU-write-to-MFMA-read wait states.
    lines = [f"s_nop {lead_wait - 1}"] if lead_wait else []
    for ks, (a_ks, b_ks) in enumerate(steps):
        if len(a_ks) != g.frags_m or len(b_ks) != g.frags_n:
            raise ValueError("K step operands do not match the block geometry")
        srcs += [b.vec_bitcast(x, src_ty) for x in list(a_ks) + list(b_ks)]
        a0 = n + tied + ks * per_step  # first A source operand of this step
        b0 = a0 + g.frags_m  # first B source operand of this step
        if ks and n == 1 and nop:
            lines.append(f"s_nop {nop}")
        # srcC: the tied input (or the constant 0) on the first step, the
        # running output after it
        if ks:
            src_c = [f"${i}" for i in range(n)]
        elif zero:
            src_c = ["0"] * n
        else:
            src_c = [f"${n + i}" for i in range(n)]
        lines += [
            f"{mnem} ${i}, ${a0 + fm}, ${b0 + fn}, {src_c[i]}"
            for i, (fm, fn) in enumerate(outs)
        ]
    if nop:
        lines.append(f"s_nop {nop}")
    if agpr_base is None:
        outs_c = ["=a"] * n
    else:
        _check_agpr_base(g, agpr_base)
        w = g.op.c_frag_len
        outs_c = [
            "={a[%d:%d]}" % (agpr_base + i * w, agpr_base + (i + 1) * w - 1)
            for i in range(n)
        ]
    constraints = ",".join(outs_c + [str(i) for i in range(tied)] + ["v"] * len(srcs))
    if zero:
        accs = []
        result_types = [VectorType(F32, g.op.c_frag_len)] * n
    else:
        accs = [acc[fm][fn] for fm, fn in outs]
        result_types = [x.type for x in accs]
    res = b.inline_asm_multi(
        "\n\t".join(lines),
        constraints,
        accs + srcs,
        result_types=result_types,
        sideeffect=True,
        convergent=True,
        result_name_hint="g13asm" if not zero else "g02asm",
    )
    if n == 1 and not isinstance(res, (list, tuple)):
        res = [res]
    new = [[None] * g.frags_n for _ in range(g.frags_m)]
    if not zero:
        new = [list(row) for row in acc]
    for (fm, fn), v in zip(outs, res):
        new[fm][fn] = v
    return new


def _check_agpr_base(g: "BlockGemm", agpr_base: int) -> None:
    """Raise unless the wave's accumulators fit a0..a255 from ``agpr_base``."""
    if isinstance(agpr_base, bool) or not isinstance(agpr_base, int):
        raise ValueError(f"agpr_base must be an int, got {agpr_base!r}")
    last = agpr_base + g.frags_m * g.frags_n * g.op.c_frag_len - 1
    if agpr_base < 0 or last > 255:
        raise ValueError(f"pinned AGPRs a[{agpr_base}:{last}] outside a0..a255")


@dataclass(frozen=True)
class BlockGemm:
    """Validated geometry of one multi-wave block GEMM (hashable, no IR)."""

    arch: str
    dtype: str
    m: int
    n: int
    k: int
    grid: tuple[int, int]
    atom_k: int = 16

    def __post_init__(self) -> None:
        rows, cols = self.grid
        waves = rows * cols
        check_waves(self.arch, waves)
        if self.grid not in ((1, waves), (waves, 1), (2, waves // 2)):
            raise ValueError(
                f"warp grid {self.grid} is not one of (1, W), (W, 1), (2, W/2)"
            )
        op = select_bwd_atom(self.arch, self.dtype, self.atom_k)
        if self.m % (op.m * rows) or self.n % (op.n * cols):
            raise ValueError(
                f"block {self.m}x{self.n} does not split into {op.m}x{op.n} atoms "
                f"over the warp grid {self.grid}"
            )
        if self.k <= 0 or self.k % self.atom_k:
            raise ValueError(
                f"K={self.k} is not a multiple of the atom K {self.atom_k}"
            )

    # --- derived geometry ------------------------------------------------
    @cached_property
    def op(self) -> MmaOp:
        return select_bwd_atom(self.arch, self.dtype, self.atom_k)

    @property
    def waves(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def wave_size(self) -> int:
        return bwd_arch_caps(self.arch).wave_size

    @property
    def threads(self) -> int:
        return self.waves * self.wave_size

    @property
    def wave_m(self) -> int:
        return self.m // self.grid[0]

    @property
    def wave_n(self) -> int:
        return self.n // self.grid[1]

    @property
    def frags_m(self) -> int:
        return self.wave_m // self.op.m

    @property
    def frags_n(self) -> int:
        return self.wave_n // self.op.n

    @property
    def k_steps(self) -> int:
        return self.k // self.atom_k

    @property
    def mma_per_wave(self) -> int:
        """MMA instructions issued by each wave for one call of :meth:`mma`."""
        return self.frags_m * self.frags_n * self.k_steps

    def wave_rows(self, w: int) -> tuple[int, int]:
        r = w // self.grid[1]
        return r * self.wave_m, (r + 1) * self.wave_m

    def wave_cols(self, w: int) -> tuple[int, int]:
        c = w % self.grid[1]
        return c * self.wave_n, (c + 1) * self.wave_n

    # --- emission ----------------------------------------------------------
    def wave_origin(self, b: IRBuilder, wave: Value) -> tuple[Value, Value]:
        """Row and column origin of ``wave``'s C sub-block (wave-uniform i32)."""
        rows, cols = self.grid
        if cols == 1:
            r, c = wave, None
        elif rows == 1:
            r, c = None, wave
        else:
            cc = b.const_i32(cols)
            r, c = b.div(wave, cc), b.mod(wave, cc)
        row0 = b.mul(r, b.const_i32(self.wave_m)) if r is not None else b.const_i32(0)
        col0 = b.mul(c, b.const_i32(self.wave_n)) if c is not None else b.const_i32(0)
        return row0, col0

    def zero_acc(self, b: IRBuilder) -> list[list[Value]]:
        return [
            [b.zero_vec_f32(self.op.c_frag_len) for _ in range(self.frags_n)]
            for _ in range(self.frags_m)
        ]

    def load_a(self, b, src: LdsOperand, lane: Value, row0: Value, ks: int):
        """A fragments ``[fm]`` of K step ``ks`` for the wave at ``row0``."""
        coords = src.coords or operand_coords(self.op, "a")
        return [
            self._load(
                b, src, coords, lane, _off(b, row0, src.mn0 + fm * self.op.m), ks
            )
            for fm in range(self.frags_m)
        ]

    def load_b(self, b, src: LdsOperand, lane: Value, col0: Value, ks: int):
        """B fragments ``[fn]`` of K step ``ks`` for the wave at ``col0``."""
        coords = src.coords or operand_coords(self.op, "b")
        return [
            self._load(
                b, src, coords, lane, _off(b, col0, src.mn0 + fn * self.op.n), ks
            )
            for fn in range(self.frags_n)
        ]

    def _load(self, b, src: LdsOperand, coords, lane, mn0, ks: int):
        k0 = src.k0 + ks * self.atom_k
        if src.reader == "tr16":
            return load_frag_lds_tr16(
                b, coords, src.smem, lane, dtype=self.dtype, arch=self.arch,
                mn0=mn0, k0=k0,
            )  # fmt: skip
        return load_frag_lds(
            b, coords, src.smem, lane, dtype=self.dtype, mn0=mn0, k0=k0,
            storage=src.storage, align=src.align,
        )  # fmt: skip

    def load_all_a(self, b, src: LdsOperand, lane, row0) -> list[list[Value]]:
        """Register-resident A: ``[fm][ks]``."""
        per_k = [self.load_a(b, src, lane, row0, ks) for ks in range(self.k_steps)]
        return [
            [per_k[ks][fm] for ks in range(self.k_steps)] for fm in range(self.frags_m)
        ]

    def load_all_b(self, b, src: LdsOperand, lane, col0) -> list[list[Value]]:
        """Register-resident B: ``[fn][ks]``."""
        per_k = [self.load_b(b, src, lane, col0, ks) for ks in range(self.k_steps)]
        return [
            [per_k[ks][fn] for ks in range(self.k_steps)] for fn in range(self.frags_n)
        ]

    def load_operands(
        self,
        b: IRBuilder,
        a: Frags | LdsOperand,
        bop: Frags | LdsOperand,
        *,
        lane: Value | None = None,
        origin: tuple[Value, Value] | None = None,
    ) -> tuple[list[list[Value]], list[list[Value]]]:
        """Both operands of :meth:`mma` in register form ``([fm][ks], [fn][ks])``.

        An :class:`LdsOperand` is read here, K step by K step, A fragments
        before B fragments, which is the order :meth:`mma` reads them in; a
        register operand is passed through. Feeding the result to :meth:`mma`
        issues the same MMAs on the same values with no LDS read left in it
        (the operand-load hoist of the asm route).
        """
        steps = self.load_k_slice(b, a, bop, 0, self.k, lane=lane, origin=origin)
        a_regs = [[a_ks[fm] for a_ks, _ in steps] for fm in range(self.frags_m)]
        b_regs = [[b_ks[fn] for _, b_ks in steps] for fn in range(self.frags_n)]
        return a_regs, b_regs

    def mma(
        self,
        b: IRBuilder,
        acc: Sequence[Sequence[Value]] | None,
        a: Frags | LdsOperand,
        bop: Frags | LdsOperand,
        *,
        lane: Value | None = None,
        origin: tuple[Value, Value] | None = None,
        route: str = "intrinsic",
        asm_tail_nop: int | None = None,
        asm_cluster: str = "k_step",
        asm_agpr_base: int | None = None,
        asm_lead_wait: int | None = None,
        pre_issue: Callable[[], None] | None = None,
    ) -> list[list[Value]]:
        """Issue ``frags_m * frags_n * k_steps`` MMAs; returns the new accumulators.

        Register operands are ``a[fm][ks]`` and ``bop[fn][ks]``. An
        :class:`LdsOperand` is read per K step and needs ``lane`` and the wave
        ``origin`` (from :meth:`wave_origin`).

        ``route = "agpr_asm"`` issues each K step's ``frags_m * frags_n`` MMAs
        as one inline-asm cluster whose accumulators are AGPR outputs tied to
        their inputs (:func:`mma_agpr_asm`); the operands are read exactly as
        on the intrinsic route, in the same order. ``asm_cluster = "gemm"``
        reads every K step's operands first (same order) and issues all
        ``k_steps`` steps as one cluster (:func:`mma_agpr_asm_steps`).
        ``asm_tail_nop`` overrides the cluster's s_nop count (``None``: the
        atom's :data:`AGPR_ASM_TAIL_NOP`). Both are legal on the asm route only.

        ``acc = None`` accumulates from zero (asm route only): the first K step
        reads srcC 0, so no zero is written to the AGPR accumulators.
        ``asm_agpr_base`` pins the accumulators to the physical AGPRs from that
        index on (every cluster of the call; :func:`mma_agpr_asm_steps`).
        ``asm_lead_wait`` opens every cluster with that many wait states
        (:func:`mma_agpr_asm_steps`; ``None``: the ISA's
        ``ASM_MFMA_VALU_LEAD_WAIT_STATES``, so a VALU write of a source
        register right before the block, which the backend cannot see as an
        MFMA hazard, is covered). These are legal on the asm route only.

        ``pre_issue`` (any route) is called once, after the operands of the
        first issue are read and before its first MMA or asm cluster is
        emitted: code it emits sits between them (another GEMM's operand
        reads, hoisted above this GEMM's cluster). ``None`` emits nothing.
        """
        if route not in ("intrinsic", "agpr_asm"):
            raise ValueError(f"unknown MMA route {route!r}")
        if asm_cluster not in ("k_step", "gemm"):
            raise ValueError(f"unknown asm cluster granularity {asm_cluster!r}")
        if route != "agpr_asm" and (
            asm_tail_nop is not None
            or asm_cluster != "k_step"
            or asm_agpr_base is not None
            or asm_lead_wait is not None
        ):
            raise ValueError(
                "asm_tail_nop / asm_cluster / asm_agpr_base / asm_lead_wait need "
                "route='agpr_asm'"
            )
        if asm_lead_wait is None:
            asm_lead_wait = ASM_MFMA_VALU_LEAD_WAIT_STATES
        if asm_agpr_base is not None:
            _check_agpr_base(self, asm_agpr_base)
        if acc is None:
            if route != "agpr_asm":
                raise ValueError("a zero accumulator (acc=None) needs route='agpr_asm'")
        else:
            acc = [list(row) for row in acc]
            if len(acc) != self.frags_m or any(len(r) != self.frags_n for r in acc):
                raise ValueError("accumulator shape does not match the block geometry")
        self._check_operands(a, bop, lane, origin)
        pending = [pre_issue] if pre_issue is not None else []

        def issue_point() -> None:
            while pending:
                pending.pop()()

        steps = []
        for ks in range(self.k_steps):
            a_ks, b_ks = self._step_operands(b, a, bop, lane, origin, ks)
            if route == "agpr_asm" and asm_cluster == "gemm":
                steps.append((a_ks, b_ks))
                continue
            issue_point()
            if route == "agpr_asm":
                acc = mma_agpr_asm(
                    self, b, acc, a_ks, b_ks, tail_nop=asm_tail_nop,
                    agpr_base=asm_agpr_base, lead_wait=asm_lead_wait,
                )  # fmt: skip
                continue
            for fm in range(self.frags_m):
                for fn in range(self.frags_n):
                    acc[fm][fn] = b.mma(self.op, a_ks[fm], b_ks[fn], acc[fm][fn])
        if steps:
            issue_point()
            acc = mma_agpr_asm_steps(
                self, b, acc, steps, tail_nop=asm_tail_nop, agpr_base=asm_agpr_base,
                lead_wait=asm_lead_wait,
            )  # fmt: skip
        return acc

    def _check_operands(self, a, bop, lane, origin) -> None:
        for which, opnd, count in (("A", a, self.frags_m), ("B", bop, self.frags_n)):
            if isinstance(opnd, LdsOperand):
                if lane is None or origin is None:
                    raise ValueError(f"LDS operand {which} needs lane and origin")
            elif len(opnd) != count or any(len(f) != self.k_steps for f in opnd):
                raise ValueError(
                    f"register operand {which} must be [{count}][{self.k_steps}]"
                )

    def _step_operands(self, b, a, bop, lane, origin, ks: int):
        """``(a_ks, b_ks)``: the A fragments ``[fm]`` and B fragments ``[fn]``
        of K step ``ks`` (an LDS operand is read here, A first; a register
        operand is indexed)."""
        if isinstance(a, LdsOperand):
            a_ks = self.load_a(b, a, lane, origin[0], ks)
        else:
            a_ks = [a[fm][ks] for fm in range(self.frags_m)]
        if isinstance(bop, LdsOperand):
            b_ks = self.load_b(b, bop, lane, origin[1], ks)
        else:
            b_ks = [bop[fn][ks] for fn in range(self.frags_n)]
        return a_ks, b_ks

    def _check_k_slice(self, k_slice: int) -> int:
        """K steps per slice of ``k_slice`` keys (raises before any IR)."""
        if (
            isinstance(k_slice, bool)
            or not isinstance(k_slice, int)
            or k_slice <= 0
            or k_slice % self.atom_k
            or self.k % k_slice
        ):
            raise ValueError(
                f"K slice {k_slice!r} must be a positive multiple of the atom K "
                f"{self.atom_k} that divides K={self.k}"
            )
        return k_slice // self.atom_k

    def load_k_slice(
        self,
        b: IRBuilder,
        a: Frags | LdsOperand,
        bop: Frags | LdsOperand,
        s: int,
        k_slice: int,
        *,
        lane: Value | None = None,
        origin: tuple[Value, Value] | None = None,
    ) -> list[tuple[list[Value], list[Value]]]:
        """Operand fragments of K slice ``s`` (``k_slice`` keys wide).

        Returns ``[(a_ks, b_ks), ...]`` for the slice's K steps ``ks = s *
        k_slice / atom_k + j`` in ascending order: an :class:`LdsOperand` is
        read here (per step A, then B, as :meth:`mma` reads them; the LDS
        offsets are those of the global step ``ks``), a register operand is
        indexed.
        """
        per = self._check_k_slice(k_slice)
        n = self.k // k_slice
        if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < n:
            raise ValueError(f"slice {s!r} outside the {n} slices of K={self.k}")
        self._check_operands(a, bop, lane, origin)
        return [
            self._step_operands(b, a, bop, lane, origin, s * per + j)
            for j in range(per)
        ]

    def mma_k_sliced(
        self,
        b: IRBuilder,
        acc: Sequence[Sequence[Value]],
        a: Frags | LdsOperand,
        bop: Frags | LdsOperand,
        *,
        k_slice: int,
        buffers: int,
        lane: Value | None = None,
        origin: tuple[Value, Value] | None = None,
        first: Sequence[tuple[Sequence[Value], Sequence[Value]]] | None = None,
    ) -> list[list[Value]]:
        """The MMAs of :meth:`mma` (intrinsics), K steps grouped in slices.

        The ``k / k_slice`` slices of ``k_slice`` keys run in K order; a slice
        reads every operand of its K steps (:meth:`load_k_slice`) before any
        of its MMAs. ``buffers = 1``: slice ``s`` is read after the MMAs of
        slice ``s - 1``. ``buffers = 2``: slice ``s + 1`` is read before the
        MMAs of slice ``s`` (two register buffers). ``first``: slice 0 as
        already read by the caller (for example ahead of another GEMM's MMAs).
        Every accumulator takes its K steps in ascending order, as in
        :meth:`mma`, so the result is bit for bit the same.
        """
        if buffers not in (1, 2):
            raise ValueError(f"buffers must be 1 or 2, got {buffers!r}")
        per = self._check_k_slice(k_slice)
        n = self.k // k_slice
        acc = [list(row) for row in acc]
        if len(acc) != self.frags_m or any(len(r) != self.frags_n for r in acc):
            raise ValueError("accumulator shape does not match the block geometry")
        self._check_operands(a, bop, lane, origin)
        if first is not None and len(first) != per:
            raise ValueError(f"first slice must hold {per} K steps, got {len(first)}")

        def load(s):
            return self.load_k_slice(b, a, bop, s, k_slice, lane=lane, origin=origin)

        cur = list(first) if first is not None else load(0)
        for s in range(n):
            nxt = load(s + 1) if buffers == 2 and s + 1 < n else None
            for a_ks, b_ks in cur:
                for fm in range(self.frags_m):
                    for fn in range(self.frags_n):
                        acc[fm][fn] = b.mma(self.op, a_ks[fm], b_ks[fn], acc[fm][fn])
            if s + 1 < n:
                cur = nxt if nxt is not None else load(s + 1)
        return acc

    def mma_frag(
        self,
        b: IRBuilder,
        acc: Value,
        a: Frags | LdsOperand,
        bop: Frags | LdsOperand,
        fm: int,
        fn: int,
        *,
        lane: Value | None = None,
        origin: tuple[Value, Value] | None = None,
    ) -> Value:
        """Accumulate fragment ``[fm][fn]`` over every K step (``k_steps`` MMAs).

        The same products as :meth:`mma` for that fragment, with only its own
        A row and B column read (an :class:`LdsOperand` is read per K step).
        """
        if not (0 <= fm < self.frags_m and 0 <= fn < self.frags_n):
            raise ValueError("fragment index outside the block geometry")
        for opnd in (a, bop):
            if isinstance(opnd, LdsOperand) and (lane is None or origin is None):
                raise ValueError("LDS operands need lane and origin")
        for ks in range(self.k_steps):
            if isinstance(a, LdsOperand):
                coords = a.coords or operand_coords(self.op, "a")
                mn0 = _off(b, origin[0], a.mn0 + fm * self.op.m)
                a_f = self._load(b, a, coords, lane, mn0, ks)
            else:
                a_f = a[fm][ks]
            if isinstance(bop, LdsOperand):
                coords = bop.coords or operand_coords(self.op, "b")
                mn0 = _off(b, origin[1], bop.mn0 + fn * self.op.n)
                b_f = self._load(b, bop, coords, lane, mn0, ks)
            else:
                b_f = bop[fn][ks]
            acc = b.mma(self.op, a_f, b_f, acc)
        return acc

    def acc_coords(
        self, b: IRBuilder, lane: Value, origin: tuple[Value, Value], fm: int, fn: int
    ) -> list[tuple[Value, Value]]:
        """Block coordinates ``(row, col)`` of every slot of accumulator ``[fm][fn]``."""
        layout = self.op.c_layout()
        out = []
        for s in range(layout.frag_len):
            r, c = layout.coord(b, lane, s)
            out.append(
                (
                    _off(b, b.add(origin[0], r), fm * self.op.m),
                    _off(b, b.add(origin[1], c), fn * self.op.n),
                )
            )
        return out


def _off(b: IRBuilder, base: Value, k: int) -> Value:
    return b.add(base, b.const_i32(k)) if k else base


def lane_and_wave(
    b: IRBuilder, wave_size: int, *, tid: Value | None = None
) -> tuple[Value, Value, Value]:
    """``(tid, lane, wave)`` with ``wave`` made wave-uniform by ``readfirstlane``.

    ``tid`` is the thread index (``thread_id_x`` read here when ``None``).
    """
    if tid is None:
        tid = b.thread_id_x()
    ws = b.const_i32(wave_size)
    lane = b.mod(tid, ws)
    wave = b.readfirstlane(b.div(tid, ws))
    return tid, lane, wave


def check_relabel_grids(producer: BlockGemm, consumer: BlockGemm) -> RelabelPlan:
    """Prove that ``producer``'s C can feed ``consumer``'s A by register relabel.

    Conditions: the consumer computes ``C_p^T * X`` (``consumer.m == producer.n``
    and ``consumer.k == producer.m``), the same arch and dtype, every wave owns
    the same producer columns as consumer rows and the full producer row range,
    and the catalog lane maps of the atom pair relabel (:func:`plan_c_to_a`).
    Raises :class:`ValueError` otherwise; returns the plan.
    """
    if (producer.arch, producer.dtype) != (consumer.arch, consumer.dtype):
        raise ValueError("producer and consumer differ in arch or dtype")
    if consumer.m != producer.n or consumer.k != producer.m:
        raise ValueError(
            f"consumer {consumer.m}x{consumer.n}x{consumer.k} is not the transpose "
            f"product of producer {producer.m}x{producer.n}"
        )
    if consumer.waves != producer.waves:
        raise ValueError("producer and consumer use different wave counts")
    for w in range(producer.waves):
        if producer.wave_cols(w) != consumer.wave_rows(w):
            raise ValueError(
                f"wave {w}: producer columns {producer.wave_cols(w)} differ from "
                f"consumer rows {consumer.wave_rows(w)} (grids {producer.grid} -> "
                f"{consumer.grid})"
            )
        if producer.wave_rows(w) != (0, producer.m):
            raise ValueError(
                f"wave {w} does not own every producer row (grid {producer.grid})"
            )
    plan = plan_c_to_a(producer.op, consumer.op)
    if (
        consumer.atom_k % producer.op.m
        or plan.c_tiles * producer.op.m != consumer.atom_k
    ):
        raise ValueError("consumer atom K does not stack whole producer tiles")
    return plan


def relabel_acc_to_a(
    b: IRBuilder,
    producer: BlockGemm,
    consumer: BlockGemm,
    acc: Sequence[Sequence[Value]],
    lane: Value,
    *,
    bf16_cast: str = "generic",
) -> tuple[list[list[Value]], RelabelPlan]:
    """Turn the producer accumulators into consumer A fragments ``[fm][ks]``.

    Consumer A fragment ``(fm, ks)`` is built from producer C tiles
    ``(ks * tiles + t, fm)``; a plan with ``needs_k_perm`` requires the
    consumer B operand coordinates of :func:`kperm_b_operand_coords`.
    ``bf16_cast`` selects the bf16 rounding (:func:`emit_c_to_a`).
    """
    plan = check_relabel_grids(producer, consumer)
    t = plan.c_tiles
    frags = [
        [
            emit_c_to_a(
                b,
                plan,
                [acc[ks * t + i][fm] for i in range(t)],
                dtype=consumer.dtype,
                lane=lane,
                bf16_cast=bf16_cast,
            )
            for ks in range(consumer.k_steps)
        ]
        for fm in range(consumer.frags_m)
    ]
    return frags, plan


def kperm_b_operand_coords(plan: RelabelPlan, producer: BlockGemm, consumer: BlockGemm):
    """B coordinates matching ``plan``'s K permutation (identity plans: catalog).

    A K-permuted map is read either by plain element reads (``reader =
    "plain"``) or by the transpose read's stacked plan (``reader = "tr16"``,
    :func:`._attention_bwd_frag.tr16_read_plan`).
    """
    if not plan.needs_k_perm:
        return operand_coords(consumer.op, "b")
    return kperm_b_coords(plan, producer.op, consumer.op)
