# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``TileMmaDriver`` -- the MMA ITERATION object (the atom-grid walk + issue).

Given the layouts a :class:`~rocke.helpers.tiling.mma.plan.TileMmaPlan` resolved, the driver walks the
M x N x K atom grid for the wave tile (in ``tiling.order``), issuing one ``b.mma`` per atom and
accumulating each C subtile. It is stateless per call -- the accumulator is a loop-carried SSA value,
never instance state -- so the same driver drives every wave-tile call. The front-door
:class:`~rocke.helpers.tiling.mma.mma_operation.TileMma` composes a plan + a driver.
"""

from __future__ import annotations

from ....core.ir import F32, TF32
from ....core.storage import packed_word_range, register_packing
from ...mma_io import mma_operand_ir_type
from ..emit import _move_registers
from ..fragments import Fragment, TileDesc, _fragment_storage_type, fragment_length
from ..register_mapper import RegisterMapper
from ..transforms import (
    mma_accumulator_flow_consistent,
    mma_operand_layout_sound,
    mma_pair_k_aligned,
)
from .plan import TileMmaPlan
from .warp_encoding import canonical_layouts

# The catalog row families whose operands the driver supplies: it feeds A, B and C, so a scaled row
# (which also takes scale operands) or a sparse row (which also takes a sparsity index) is not driven.
_DRIVEN_FAMILIES = frozenset({"dense"})

# Math dtypes whose instruction also reads a wider fragment dtype, dropping its low mantissa bits: a
# TF32 MMA reads F32 registers as TF32 (truncation). An author who wants round-to-nearest-even
# converts the fragment to TF32 first (the "why pre-convert?" note in the gfx942 arch doc, 21.1).
_MMA_TRUNCATING_SOURCES = {TF32: F32}


def _assert_atom_contiguous(
    tile_desc: TileDesc,
    *,
    atom_k: int,
    free_sub: int,
    k_sub: int,
    free_blocks: int,
    role: str,
    op_id: str,
) -> None:
    """The SOA slice contract: the driver slices each atom as the CONTIGUOUS register block
    ``[i*atom_len : +atom_len]`` and issues one ``b.mma`` per block, so every register in a block must
    belong to ONE atom. A style/custom fragment owns the register order, so an AOS (atom-interleaved) or
    scattered layout can be K-sound yet mis-sliced here -- reject it fail-fast instead of miscompiling
    silently (correctness SOT: docs/mma_is_machinery.md).

    An operand atom is identified, at a fixed lane, by ``(free_coordinate, K-atom = K // atom_k)``: an
    A/B lane holds its ``k_per_lane`` K within ONE free row, so a single atom's registers share that
    pair and vary only in within-atom K. Each ``atom_len``-register block must be ONE atom -- i.e. at
    EVERY lane the block's registers share one ``(free, K-atom)``. The check assumes no free-coordinate
    stride (canonical strides atoms by ``atom_free``, interleaved by 1) and does not assume lane 0
    distinguishes atoms (some interleavings share a lane-0 coordinate across atoms), so it must sweep
    all lanes -- AOS (atom-iteration inner) then shows a block whose atom identity varies at some lane.

    ``free_blocks`` is the number of free rows one lane holds per atom (``a_m_register_blocks`` for A, 1
    for B). Such a lane's atom block is ``free_blocks`` equal runs, one free row each, all in one K-atom::

        gfx1250 32x16x128 fp4 A, lane 0, one atom of 128 registers:
          registers 0-63   (M 0,  K 0-31 and 64-95)
          registers 64-127 (M 16, K 0-31 and 64-95)
    """
    rm = RegisterMapper(tile_desc.layout)
    nregs = rm.num_vector_items
    denom = free_sub * k_sub
    if denom == 0 or nregs % denom:
        raise ValueError(
            f"MMA {role} operand register count {nregs} is not divisible by "
            f"free_sub*k_sub = {free_sub}*{k_sub} for {op_id!r}"
        )
    atom_len = nregs // denom
    if atom_len % free_blocks:
        raise ValueError(
            f"MMA {role} operand atom of {atom_len} registers does not split into the "
            f"{free_blocks} free rows a lane holds per atom for {op_id!r}"
        )
    run_len = atom_len // free_blocks
    for blk in range(denom):
        for lane in range(rm.num_lanes):
            key: tuple[int, int] | None = None
            for j in range(atom_len):
                free_c, k_c = rm.matrix_coordinates(lane, blk * atom_len + j)
                here = (free_c, k_c // atom_k)
                if j % run_len == 0 and key is not None and here[1] == key[1]:
                    key = here  # the next free row of this atom, in the same K-atom
                if key is None:
                    key = here
                elif here != key:
                    raise ValueError(
                        f"MMA {role} operand is not atom-contiguous (SOA) for {op_id!r}: register block "
                        f"{blk} mixes atoms {key} and {here} at lane {lane} -- an AOS/scattered register "
                        f"layout is not sliceable. Reorder to atom-major (SOA) before the MMA "
                        f"(TileDesc.reorder_registers)."
                    )


def _validate_mma_issue(a_fragment, b_fragment, accumulator, plan: TileMmaPlan) -> None:
    """The one validation run before the driver issues an MMA, kept as a single call so the issue point
    can never check a subset and silently let a bad operand or accumulator through. On the PASSED
    fragments, in order:

    - ``mma_operand_layout_sound`` on A and B -- each a well-formed MMA operand (one M/N per output row,
      well-formed K). Run FIRST and short-circuited, because the accumulator check below flows the operands
      through the machine and is undefined on an unsound operand.
    - ``mma_pair_k_aligned`` -- A and B agree on which K sits at each pair of positions the instruction
      couples (read off the canonical layouts; slot for slot on a same-format instruction), compared per
      atom (the free-dim atom counts come from ``plan.subtiles``).
    - ``mma_accumulator_flow_consistent`` -- the passed accumulator C carries the exact labels the machine
      produces from these operands, so the store writes the right coordinates. The absence of this check is
      what let a mislabeled C store to the wrong place.
    - ``_assert_atom_contiguous`` on A and B -- each atom's registers form the contiguous block the driver
      slices. C needs no equivalent: the accumulator check above is exact down to register order.

    The canonical reference comes from the ``canonical_layouts`` helper. A ``warning`` is rejected like an
    error: the driver issues no reorder, so a reorder-fixable K-misorder must not go out. Operand dtype
    agreement and backend-op resolution are separate issue-time checks the driver runs directly.
    """
    m_sub, n_sub, k_sub = plan.subtiles
    a_canon, b_canon, c_canon = canonical_layouts(plan.traits, plan.subtiles)
    a_lay = a_fragment.tile_desc.layout
    b_lay = b_fragment.tile_desc.layout

    for role, lay, canon in (("A", a_lay, a_canon), ("B", b_lay, b_canon)):
        d = mma_operand_layout_sound(lay, canon, role=role)
        if d.severity != "ok":
            raise ValueError(
                f"MMA {role} operand not sound for {plan.op_id!r} -- {d.message}"
            )

    d = mma_pair_k_aligned(
        a_lay,
        b_lay,
        a_canon=a_canon,
        b_canon=b_canon,
        a_free_atoms=m_sub,
        b_free_atoms=n_sub,
    )
    if d.severity != "ok":
        raise ValueError(
            f"MMA operands not K-aligned for {plan.op_id!r} -- {d.message}"
        )

    d = mma_accumulator_flow_consistent(
        accumulator.tile_desc.layout,
        a_lay,
        b_lay,
        a_canon=a_canon,
        b_canon=b_canon,
        c_canon=c_canon,
    )
    if d.severity != "ok":
        raise ValueError(
            f"MMA accumulator (C) not consistent for {plan.op_id!r} -- {d.message}"
        )

    atom_k = plan.atom_shape[2]
    _assert_atom_contiguous(
        a_fragment.tile_desc,
        atom_k=atom_k,
        free_sub=m_sub,
        k_sub=k_sub,
        free_blocks=plan.traits.layout_params.a_m_register_blocks,
        role="A",
        op_id=plan.op_id,
    )
    _assert_atom_contiguous(
        b_fragment.tile_desc,
        atom_k=atom_k,
        free_sub=n_sub,
        k_sub=k_sub,
        free_blocks=1,
        role="B",
        op_id=plan.op_id,
    )


class TileMmaDriver:
    """Walk the wave-tile atom grid and issue the MMAs. Consumes a :class:`TileMmaPlan`; holds no
    mutable state (the accumulator is SSA-carried)."""

    def __init__(self, plan: TileMmaPlan) -> None:
        self._plan = plan

    @property
    def plan(self) -> TileMmaPlan:
        return self._plan

    @staticmethod
    def _read_subvector(b, vec, start: int, length: int):
        """Extract one atom's contiguous register slice ``[start:start+length]`` of ``vec`` into a
        fresh ``length``-element vector of its element type (u8 included, through
        :func:`~rocke.helpers.tiling.emit._move_registers`) for ``b.mma``."""
        return _move_registers(b, vec, [(start + i, i) for i in range(length)], length)

    @staticmethod
    def _as_operand(b, value, math_dtype, count: int):
        """Return ``value`` (``count`` logical elements' registers) as the vector type ``b.mma``
        takes for an operand of ``math_dtype`` (:func:`~rocke.helpers.mma_io.mma_operand_ir_type`),
        bitcasting only when the types differ: TF32 storage ``i32`` and truncated F32 registers
        become ``vec<tf32>``; every other operand, packed words included, is already that type.
        """
        operand_type = mma_operand_ir_type(math_dtype.name, count)
        if value.type == operand_type:
            return value
        return b.bitcast(value, operand_type)

    def _read_operand_atom(
        self, b, fragment: Fragment, start: int, length: int, math_dtype
    ):
        """Return one instruction's A or B operand: logical elements ``[start, start+length)`` of
        ``fragment``, sliced from its storage registers -- the 32-bit words that hold them
        (:func:`~rocke.core.storage.packed_word_range`) for a packed dtype, the elements themselves
        otherwise -- then given the operand type of ``math_dtype`` (:meth:`_as_operand`).
        """
        packing = register_packing(
            fragment.dtype.name, fragment.tile_desc.register_count
        )
        if packing is not None:
            words = packed_word_range(packing, start, length)
            sub = self._read_subvector(b, fragment.value, words.start, len(words))
        else:
            sub = self._read_subvector(b, fragment.value, start, length)
        return self._as_operand(b, sub, math_dtype, length)

    @staticmethod
    def _write_subvector(b, vec, sub, start: int, length: int):
        """Write ``sub`` back into ``vec`` at ``[start:start+length]``, returning the new SSA
        vector (accumulators are loop-carried SSA values, so this rebuilds the tile C).
        """
        out = vec
        for i in range(length):
            out = b.vec_insert(out, b.vec_extract(sub, i), start + i)
        return out

    def _subtile_triples(self):
        """The (mi, nj, ki) atom visitation order, per ``tiling.order`` (right-most fastest)."""
        plan = self._plan
        ranges = {
            "M": range(plan._m_subtiles),
            "N": range(plan._n_subtiles),
            "K": range(plan._k_subtiles),
        }
        order = plan.tiling.order
        triples = []
        for x0 in ranges[order[0]]:
            for x1 in ranges[order[1]]:
                for x2 in ranges[order[2]]:
                    axis = {order[0]: x0, order[1]: x1, order[2]: x2}
                    triples.append((axis["M"], axis["N"], axis["K"]))
        return triples

    @staticmethod
    def _result_base(b, accumulator: Fragment, d_dtype):
        """The vector the grid walk assembles the per-subtile results into: the accumulator's own
        vector when the result has its dtype (each result replaces the subtile it was computed
        from), else a zero vector of the result dtype's storage type, which every subtile then
        fills."""
        if d_dtype == accumulator.dtype:
            return accumulator.value
        storage = _fragment_storage_type(d_dtype, accumulator.tile_desc.register_count)
        return b.zero_vec(storage.elem, storage.count)

    def __call__(self, b, a_fragment, b_fragment, accumulator):
        """Walk the M x N x K atom grid for the wave tile (in ``tiling.order``), issuing one
        ``b.mma`` per atom and accumulating each C subtile. The fragments are
        subtile-contiguous (from the wave layouts), so every atom is a register slice.

        Each A and B operand is fed from the fragment's storage registers: an atom's slice is the
        32-bit words that hold it for a packed dtype, its logical elements otherwise, then given
        the vector type ``b.mma`` takes for the plan's A or B dtype. A TF32 instruction reads a
        TF32 fragment (held as ``i32``) or an F32 fragment, whose low mantissa bits it drops.

        Before any IR it checks, in order: the operand dtypes; that the instruction's family is
        one the driver feeds (A, B and C only); that a result of another dtype than the
        accumulator is never fed back as the next K step's accumulator; that rocke issues the
        instruction on the target (:meth:`TileMmaPlan.emit_op`); then the single layout validation
        (:func:`_validate_mma_issue` -- operand soundness, pairwise K-match, accumulator
        consistency, atom contiguity). A reason the instruction cannot be issued is thus never
        hidden behind a layout error. Each fragment's register vector type is guaranteed by
        :class:`Fragment` itself, not re-checked here.

        Returns the result as a new :class:`Fragment` on the accumulator's tile, of the
        instruction's result dtype (``plan.d_dtype``, the accumulator dtype for every
        accumulate-in-place instruction); the `accumulator` argument is not mutated, so the result
        must be assigned (``acc = driver(b, a_frag, b_frag, acc)``).
        """
        plan = self._plan
        for name, fragment, want in (
            ("A", a_fragment, plan.a_dtype),
            ("B", b_fragment, plan.b_dtype),
            ("C", accumulator, plan.c_dtype),
        ):
            source = _MMA_TRUNCATING_SOURCES.get(want) if name != "C" else None
            if fragment.dtype == want or fragment.dtype == source:
                continue
            expected = want.name
            if source is not None:
                expected += (
                    f" or {source.name} ({want.name} math also reads {source.name} "
                    f"fragments, dropping their low mantissa bits)"
                )
            raise ValueError(
                f"MMA operand dtype mismatch -- operand={name}, "
                f"fragment={fragment.dtype.name}, expected {expected}"
            )

        traits = plan.traits
        if traits.family not in _DRIVEN_FAMILIES:
            raise NotImplementedError(
                f"{plan.op_id!r} is a {traits.family} instruction; the TileMma driver supplies "
                f"only A, B and C operands -- the plan can be visualised, and emit_op() returns "
                f"the backend op for a hand-written b.mma call wherever rocke issues it -- TODO: "
                f"not implemented yet"
            )
        m_sub, n_sub, k_sub = plan.subtiles
        if plan.d_dtype != plan.c_dtype and k_sub > 1:
            raise ValueError(
                f"{plan.op_id!r} returns {plan.d_dtype.name} from an {plan.c_dtype.name} "
                f"accumulator, so its result cannot be the next K step's accumulator -- this "
                f"tile issues {k_sub} instructions along K per C subtile; use a tile whose K is "
                f"the instruction's K ({plan.atom_shape[2]}), or the {plan.c_dtype.name}-result "
                f"instruction (omit d) and convert the finished tile"
            )

        # Issue is decided before the layouts are validated (emit_op emits no IR), then ONE
        # validation call (operand soundness, pairwise K-match, accumulator consistency, atom
        # contiguity), with atom counts from plan.subtiles and the canonical reference from the
        # canonical_layouts helper. Kept as ONE call so the issue point can't check a subset -- a
        # missing accumulator check is exactly what let a mislabeled C through before.
        op = plan.emit_op()
        _validate_mma_issue(a_fragment, b_fragment, accumulator, plan)

        # Single C subtile: accumulate in-register over K (byte-identical to the atom path).
        if m_sub == 1 and n_sub == 1:
            acc_value = accumulator.value
            if k_sub == 1:
                a_op = self._as_operand(
                    b,
                    a_fragment.value,
                    plan.a_dtype,
                    a_fragment.tile_desc.register_count,
                )
                b_op = self._as_operand(
                    b,
                    b_fragment.value,
                    plan.b_dtype,
                    b_fragment.tile_desc.register_count,
                )
                return Fragment(
                    accumulator.tile_desc,
                    plan.d_dtype,
                    b.mma(op, a_op, b_op, acc_value),
                )
            a_atom = fragment_length(a_fragment.tile_desc.layout) // k_sub
            b_atom = fragment_length(b_fragment.tile_desc.layout) // k_sub
            mac_prio = (
                plan.tiling.mac_prio
            )  # raise AFTER the first atom (see grid-branch note below)
            for ki in range(k_sub):
                a_sub = self._read_operand_atom(
                    b, a_fragment, ki * a_atom, a_atom, plan.a_dtype
                )
                b_sub = self._read_operand_atom(
                    b, b_fragment, ki * b_atom, b_atom, plan.b_dtype
                )
                acc_value = b.mma(op, a_sub, b_sub, acc_value)
                if mac_prio and ki == 0:
                    b.s_setprio(mac_prio)
            if mac_prio:
                b.s_setprio(0)
            return Fragment(accumulator.tile_desc, plan.d_dtype, acc_value)

        # Subtiled M/N grid. Carry a PER-ATOM accumulator SSA for each (mi, nj) C subtile so
        # that across K every C subtile is touched ONLY by `b.mma` (an MFMA def->use chain) --
        # no `vec_extract`/`vec_insert` on C inside the K-loop. LLVM then keeps each atom's
        # accumulator in an AGPR (the MFMA writes acc natively and reads Cin from acc), instead
        # of spilling the whole C tile into arch VGPRs (which a monolithic extract/insert-per-K
        # forces). The incoming C is split into per-atom SSAs ONCE (prologue) and packed back
        # ONCE (epilogue), off the K-loop. Any loop-nest order is correct (C accum is commutative).
        a_atom = fragment_length(a_fragment.tile_desc.layout) // (m_sub * k_sub)
        b_atom = fragment_length(b_fragment.tile_desc.layout) // (n_sub * k_sub)
        c_atom = fragment_length(accumulator.tile_desc.layout) // (m_sub * n_sub)
        accs = [
            self._read_subvector(b, accumulator.value, idx * c_atom, c_atom)
            for idx in range(m_sub * n_sub)
        ]
        # `mac_prio` raises wave issue priority for the matrix-dense body (MFMA or WMMA -- the driver is
        # instruction-agnostic): the FIRST atom issues at normal priority, then `s_setprio(mac_prio)` for
        # the remaining atoms, dropping to 0 after the last. Raising BEFORE the first atom instead
        # perturbs regalloc and can cost a wave of occupancy; raising after the first is the stable
        # placement. Reorders issue, not the math -- bit-exact. This branch always has >= 2 atoms (the
        # 1-atom case returns above), so the elevated window is non-empty; `mac_prio=0` emits nothing.
        mac_prio = plan.tiling.mac_prio
        for i, (mi, nj, ki) in enumerate(self._subtile_triples()):
            idx = mi * n_sub + nj
            a_sub = self._read_operand_atom(
                b, a_fragment, (mi * k_sub + ki) * a_atom, a_atom, plan.a_dtype
            )
            b_sub = self._read_operand_atom(
                b, b_fragment, (nj * k_sub + ki) * b_atom, b_atom, plan.b_dtype
            )
            accs[idx] = b.mma(op, a_sub, b_sub, accs[idx])
            if mac_prio and i == 0:
                b.s_setprio(mac_prio)
        if mac_prio:
            b.s_setprio(0)
        result = self._result_base(b, accumulator, plan.d_dtype)
        for idx in range(m_sub * n_sub):
            result = self._write_subvector(b, result, accs[idx], idx * c_atom, c_atom)
        return Fragment(accumulator.tile_desc, plan.d_dtype, result)

    def __repr__(self) -> str:
        return f"TileMmaDriver(plan={self._plan!r})"
