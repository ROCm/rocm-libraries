# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host planners for the quantum-circuit tensor-network monokernel.

A circuit is a list of gates ``(qubits, U)`` (``U`` complex128, index bit ``i`` <->
``qubits[i]``) on an ``n``-qubit complex64 state vector stored as interleaved
(re, im) f32 pairs: float bit 0 = re/im, float bit ``q + 1`` = qubit ``q``.

Every gate is one contraction of the network ``psi' = U . psi`` over extent-2
modes (hipTensor notation: A = psi with n modes ``:2``, B = U with 2k modes,
D = psi' with n modes). Two planners:

* :func:`fuse_blocks` - dense-block gate fusion as state-vector simulators do it
  (qsim-style): merge consecutive gates into dense ``k``-qubit blocks, each block
  one TTGT contraction (permute + GEMM); the multi-kernel reference path.
* :func:`plan_phases` - the monokernel plan: group gates into *phases* whose
  qubits fit one on-chip tile (``t`` float bits = re/im + the ``c-1`` lowest
  qubits + ``t-c`` "extra" qubits). A phase reads the state once, applies all of
  its gates as ``k``-qubit MFMA blocks on the LDS tile, writes it once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Gate = Tuple[Tuple[int, ...], np.ndarray]


# --------------------------------------------------------------------------- matrices
def embed(U: np.ndarray, gq: Sequence[int], bq: Sequence[int]) -> np.ndarray:
    """Matrix of gate ``U`` (on qubits ``gq``) acting on the block qubits ``bq``.

    Index convention for both: bit ``i`` of the row/col index <-> qubit ``q[i]``.
    """
    k = len(bq)
    pos = [bq.index(q) for q in gq]
    dim = 1 << k
    M = np.zeros((dim, dim), dtype=np.complex128)
    g = len(gq)
    for col in range(dim):
        sub = 0
        for i, p in enumerate(pos):
            sub |= ((col >> p) & 1) << i
        rest = col
        for p in pos:
            rest &= ~(1 << p)
        for gs in range(1 << g):
            amp = U[gs, sub]
            if amp == 0:
                continue
            row = rest
            for i, p in enumerate(pos):
                row |= ((gs >> i) & 1) << p
            M[row, col] += amp
    return M


def real_embedding(U: np.ndarray) -> np.ndarray:
    """Real ``2^(k+1)`` matrix acting on interleaved (re, im) floats.

    ``R[(u' << 1) | c', (u << 1) | c]``: c = c' -> Re U, (c'=0, c=1) -> -Im U,
    (c'=1, c=0) -> +Im U.
    """
    d = U.shape[0]
    R = np.zeros((2 * d, 2 * d), dtype=np.float64)
    R[0::2, 0::2] = U.real
    R[1::2, 1::2] = U.real
    R[0::2, 1::2] = -U.imag
    R[1::2, 0::2] = U.imag
    return R


# --------------------------------------------------------------------------- circuits
def _rand_unitary(rng: np.random.Generator, d: int) -> np.ndarray:
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph


def sycamore_like(n_rows: int, n_cols: int, depth: int, seed: int = 0) -> List[Gate]:
    """Random 2-D lattice circuit (the public Sycamore-style random-circuit pattern).

    Qubit ``q = r * n_cols + c``. Each cycle: a random single-qubit unitary on
    every qubit, then random two-qubit unitaries on one of four coupler patterns
    (ABCD...) of the lattice (horizontal even/odd, vertical even/odd).
    """
    rng = np.random.default_rng(seed)
    gates: List[Gate] = []
    pats = []
    for par in (0, 1):  # horizontal couplers
        pats.append(
            [
                (r * n_cols + c, r * n_cols + c + 1)
                for r in range(n_rows)
                for c in range(par, n_cols - 1, 2)
            ]
        )
    for par in (0, 1):  # vertical couplers
        pats.append(
            [
                (r * n_cols + c, (r + 1) * n_cols + c)
                for r in range(par, n_rows - 1, 2)
                for c in range(n_cols)
            ]
        )
    order = [0, 2, 1, 3]
    for cyc in range(depth):
        for q in range(n_rows * n_cols):
            gates.append(((q,), _rand_unitary(rng, 2)))
        for a, bq in pats[order[cyc % 4]]:
            gates.append(((a, bq), _rand_unitary(rng, 4)))
    return gates


# --------------------------------------------------------------------------- fusion
@dataclass
class Block:
    qubits: Tuple[int, ...]
    U: np.ndarray  # complex128, bit i <-> qubits[i]
    ngates: int = 0


def _merge(blocks: List[Block]) -> Block:
    qs: Tuple[int, ...] = ()
    U = np.ones((1, 1), dtype=np.complex128)
    ng = 0
    for bl in blocks:
        # disjoint blocks: new index = old | (bl index << len(qs))
        U = np.kron(bl.U, U)
        qs = qs + bl.qubits
        ng += bl.ngates
    return Block(qs, U, ng)


def fuse_blocks(gates: Sequence[Gate], kmax: int) -> List[Block]:
    """Greedy qsim-style fusion into dense blocks of at most ``kmax`` qubits.

    Open blocks are qubit-disjoint (so they commute); a gate joins the open
    blocks it touches when the union fits, else those blocks are emitted and the
    gate opens a new block. Emission order respects every gate dependency.
    """
    open_: List[Block] = []
    out: List[Block] = []
    for gq, G in gates:
        touched = [bl for bl in open_ if set(bl.qubits) & set(gq)]
        union = (
            set(gq).union(*[set(bl.qubits) for bl in touched]) if touched else set(gq)
        )
        if len(union) <= kmax:
            for bl in touched:
                open_.remove(bl)
            base = (
                _merge(touched)
                if touched
                else Block((), np.ones((1, 1), np.complex128))
            )
            qs = base.qubits + tuple(q for q in gq if q not in base.qubits)
            U = (
                embed(base.U, base.qubits, qs)
                if base.qubits
                else np.eye(1 << len(qs), dtype=np.complex128)
            )
            U = embed(G, gq, qs) @ U
            open_.append(Block(qs, U, base.ngates + 1))
        else:
            for bl in touched:
                open_.remove(bl)
                out.append(bl)
            open_.append(Block(tuple(gq), G.astype(np.complex128), 1))
    out.extend(open_)
    return out


def pad_block(bl: Block, k: int, pool: Sequence[int]) -> Block:
    """Extend a block to exactly ``k`` qubits with identity on qubits from ``pool``."""
    qs = list(bl.qubits)
    for q in pool:
        if len(qs) >= k:
            break
        if q not in qs:
            qs.append(q)
    if len(qs) < k:
        raise ValueError(
            f"cannot pad block {bl.qubits} to {k} qubits from {list(pool)}"
        )
    return Block(tuple(qs), embed(bl.U, bl.qubits, qs), bl.ngates)


@dataclass
class Phase:
    extra: Tuple[int, ...]  # local qubits beyond the low ones (sorted, exactly t-c)
    blocks: List[Block]


def plan_phases(
    gates: Sequence[Gate], n: int, t: int, c: int, kf: int, kmin: int = 4
) -> List[Phase]:
    """Group gates into tile phases, then fuse each phase into kf-qubit blocks.

    Low qubits ``0 .. c-2`` are local in every phase (they form the contiguous
    ``2^c``-float run of each tile row). A phase may hold ``t - c`` more qubits.
    Gates whose qubits do not fit are deferred, and later gates that touch a
    deferred qubit are deferred too (dependency order kept).
    """
    low = set(range(c - 1))
    cap = t - c
    pending = list(gates)
    phases: List[Phase] = []
    while pending:
        P: set = set()
        D: set = set()
        take, rest = [], []
        for g in pending:
            gq = set(g[0])
            if not (gq & D) and len((P | gq) - low) <= cap:
                P |= gq
                take.append(g)
            else:
                D |= gq
                rest.append(g)
        if not take:
            raise RuntimeError("phase planner made no progress")
        extra = sorted(P - low)
        for q in range(
            n
        ):  # fill the tile with idle qubits (lowest first: better coalescing)
            if len(extra) >= cap:
                break
            if q not in low and q not in extra:
                extra.append(q)
        extra = sorted(extra)
        local_pool = sorted(low) + extra
        blocks = fuse_blocks(take, kf)
        blocks = [pad_block(bl, max(kmin, len(bl.qubits)), local_pool) for bl in blocks]
        phases.append(Phase(tuple(extra), blocks))
        pending = rest
    return phases


def merged_permute(old_order, new_order, tail: int = 0):
    """(old merged shape, permutation, new merged shape) moving qubits old -> new order.

    Orders are memory orders (slowest first) of qubit ids; ``tail`` > 0 appends a
    trailing axis of that extent (the re/im floats) that stays last. Adjacent
    axes that stay adjacent are merged (torch caps tensors at 16 dims).
    """
    n = len(old_order)
    pos = {q: i for i, q in enumerate(old_order)}
    seq = [pos[q] for q in new_order] + ([n] if tail else [])
    groups = [[seq[0]]]
    for a in seq[1:]:
        if a == groups[-1][-1] + 1:
            groups[-1].append(a)
        else:
            groups.append([a])
    ext = lambda g: (2 ** len([x for x in g if x < n])) * (tail if n in g else 1)
    by_old = sorted(groups, key=lambda g: g[0])
    old_shape = [ext(g) for g in by_old]
    perm = [by_old.index(g) for g in groups]
    new_shape = [ext(g) for g in groups]
    return old_shape, perm, new_shape


def apply_reference(psi: "object", gates: Sequence[Gate], n: int):
    """f64 gate-by-gate reference on the GPU (torch complex128).

    Per gate: permute the gate qubits to the fastest modes, one complex128 GEMM,
    permute back (canonical layout between gates). Independent of every fusion
    plan and of every kernel under test.
    """
    import torch

    canon = list(range(n - 1, -1, -1))
    st = psi.reshape(-1)
    for gq, U in gates:
        g = len(gq)
        new = [q for q in canon if q not in gq] + [gq[g - 1 - i] for i in range(g)]
        o, p, nw = merged_permute(canon, new)
        x = st.view(o).permute(p).contiguous().view(-1, 1 << g)
        Ut = torch.as_tensor(U.T.copy(), dtype=torch.complex128, device=psi.device)
        y = (x @ Ut).view(nw)
        o2, p2, nw2 = merged_permute(new, canon)
        st = y.view(o2).permute(p2).contiguous().view(-1)
    return st
