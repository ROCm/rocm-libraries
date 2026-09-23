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

from ..fragments import Fragment, TileDesc, fragment_length
from ..register_mapper import RegisterMapper
from ..transforms import operand_soundness, validate_operands
from .plan import TileMmaPlan


def _assert_atom_contiguous(
    tile_desc: TileDesc, *, atom_k: int, free_sub: int, k_sub: int, role: str, op_id: str,
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
    for blk in range(denom):
        for lane in range(rm.num_lanes):
            key: tuple[int, int] | None = None
            for j in range(atom_len):
                free_c, k_c = rm.matrix_coordinates(lane, blk * atom_len + j)
                here = (free_c, k_c // atom_k)
                if key is None:
                    key = here
                elif here != key:
                    raise ValueError(
                        f"MMA {role} operand is not atom-contiguous (SOA) for {op_id!r}: register block "
                        f"{blk} mixes atoms {key} and {here} at lane {lane} -- an AOS/scattered register "
                        f"layout is not sliceable. Reorder to atom-major (SOA) before the MMA "
                        f"(TileDesc.reorder_registers)."
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
    def _read_subvector(b, vec, start: int, length: int, dtype):
        """Extract one atom's contiguous register slice ``[start:start+length]`` into a fresh
        ``<length x dtype>`` vector for ``b.mma``."""
        out = b.zero_vec(dtype, length)
        for i in range(length):
            out = b.vec_insert(out, b.vec_extract(vec, start + i), i)
        return out

    @staticmethod
    def _write_subvector(b, vec, sub, start: int, length: int):
        """Write ``sub`` back into ``vec`` at ``[start:start+length]``, returning the new SSA
        vector (accumulators are loop-carried SSA values, so this rebuilds the tile C)."""
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

    def __call__(self, b, a_fragment, b_fragment, accumulator):
        """Walk the M x N x K atom grid for the wave tile (in ``tiling.order``), issuing one
        ``b.mma`` per atom and accumulating each C subtile. The fragments are
        subtile-contiguous (from the wave layouts), so every atom is a register slice.
        Validates operand dtypes AND K-alignment first."""
        plan = self._plan
        for name, fragment in (("A", a_fragment), ("B", b_fragment), ("C", accumulator)):
            want = plan._ir_type({"A": plan._a_dtype, "B": plan._b_dtype,
                                  "C": plan._c_dtype}[name])
            if fragment.dtype.name != want.name:
                raise ValueError(
                    f"MMA operand dtype mismatch -- operand={name}, "
                    f"fragment={fragment.dtype.name!r}, expected {want.name!r}"
                )

        # MMA safety (pairwise half of the sound MAC; correctness SOT: docs/mma_is_machinery.md): the
        # hardware pairs A-slot-s with B-slot-s and sums over K, so A and B must share the same positional
        # K-distribution (M/N register order is free -- you choose the constant; K order need not be
        # canonical). Per-operand soundness (M/N fixed per output) holds by construction here (fragments
        # are atom register-reorders). A mismatched pair is rejected with a fix hint.
        # Validate K PER ATOM: the driver pairs (mi,ki)*(nj,ki), so A's m_sub M-atoms and B's n_sub
        # N-atoms each only need their atom-K to match -- comparing the whole (multi-atom) fragments
        # would falsely reject rectangular wave tiles where m_sub != n_sub (register counts differ).
        ok, why = validate_operands(
            a_fragment.tile_desc.layout, b_fragment.tile_desc.layout,
            a_free_atoms=plan._m_subtiles, b_free_atoms=plan._n_subtiles,
        )
        if not ok:
            raise ValueError(f"MMA operands not K-aligned for {plan.op_id!r} -- {why}")

        # Per-operand soundness (the OTHER half of the sound MAC). The driver is stateless -- it cannot
        # tell a derived fragment from a custom one -- so it checks BOTH operands unconditionally against
        # the canonical machine (plan.a_layout/b_layout). A DERIVED fragment passes trivially (its layout
        # IS the canonical); a CUSTOM fragment that is K-aligned yet per-operand-unsound (a wandering M/N)
        # is caught here instead of miscompiling silently. Correctness SOT: docs/mma_is_machinery.md.
        for role, frag_layout, canon in (("A", a_fragment.tile_desc.layout, plan.a_layout),
                                         ("B", b_fragment.tile_desc.layout, plan.b_layout)):
            d = operand_soundness(frag_layout, canon, role=role)
            if d.severity != "ok":
                raise ValueError(f"MMA {role} operand not sound for {plan.op_id!r} -- {d.message}")

        op = plan.emit_op()
        m_sub, n_sub, k_sub = plan._m_subtiles, plan._n_subtiles, plan._k_subtiles

        # ATOM-CONTIGUITY (SOA) GUARD: the slicing below assumes each atom's registers are a contiguous
        # block ([i*atom_len : +atom_len]). `operand_soundness` polices LABELS, not register CONTIGUITY,
        # so a style/custom fragment that is K-sound yet AOS-packed would be mis-sliced. Fail-fast here
        # instead of miscompiling silently (correctness SOT: docs/mma_is_machinery.md). Derived and
        # interleaved-style operands (both SOA) pass; an AOS/scattered custom layout is rejected.
        atom_k = plan.atom_shape[2]
        _assert_atom_contiguous(
            a_fragment.tile_desc, atom_k=atom_k, free_sub=m_sub, k_sub=k_sub,
            role="A", op_id=plan.op_id,
        )
        _assert_atom_contiguous(
            b_fragment.tile_desc, atom_k=atom_k, free_sub=n_sub, k_sub=k_sub,
            role="B", op_id=plan.op_id,
        )

        # Single C subtile: accumulate in-register over K (byte-identical to the atom path).
        if m_sub == 1 and n_sub == 1:
            acc_value = accumulator.value
            if k_sub == 1:
                return Fragment(
                    accumulator.tile_desc, accumulator.dtype,
                    b.mma(op, a_fragment.value, b_fragment.value, acc_value),
                )
            a_atom = fragment_length(a_fragment.tile_desc.layout) // k_sub
            b_atom = fragment_length(b_fragment.tile_desc.layout) // k_sub
            for ki in range(k_sub):
                a_sub = self._read_subvector(b, a_fragment.value, ki * a_atom, a_atom, a_fragment.dtype)
                b_sub = self._read_subvector(b, b_fragment.value, ki * b_atom, b_atom, b_fragment.dtype)
                acc_value = b.mma(op, a_sub, b_sub, acc_value)
            return Fragment(accumulator.tile_desc, accumulator.dtype, acc_value)

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
            self._read_subvector(b, accumulator.value, idx * c_atom, c_atom, accumulator.dtype)
            for idx in range(m_sub * n_sub)
        ]
        for mi, nj, ki in self._subtile_triples():
            idx = mi * n_sub + nj
            a_sub = self._read_subvector(
                b, a_fragment.value, (mi * k_sub + ki) * a_atom, a_atom, a_fragment.dtype
            )
            b_sub = self._read_subvector(
                b, b_fragment.value, (nj * k_sub + ki) * b_atom, b_atom, b_fragment.dtype
            )
            accs[idx] = b.mma(op, a_sub, b_sub, accs[idx])
        result = accumulator.value
        for idx in range(m_sub * n_sub):
            result = self._write_subvector(b, result, accs[idx], idx * c_atom, c_atom)
        return Fragment(accumulator.tile_desc, accumulator.dtype, result)

    def __repr__(self) -> str:
        return f"TileMmaDriver(plan={self._plan!r})"
