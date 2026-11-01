# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lane-exact NumPy emulation of the circuit kernel's index math (CPU check).

Mirrors ``circuit_kernel._emit_phase``: chunk gather, swizzle, MFMA 32x32x2 operand /
accumulator lane maps, fragment-packed matrices, output scatter. Use on small
``n`` to validate the address logic before any GPU run.
"""

from __future__ import annotations

import numpy as np

from .circuit_kernel import CircuitSpec, _deposit_const, local_bits


def _swz(a: int, mode: str) -> int:
    if mode == "none":
        return a
    return a ^ (((a >> 6) & 15) << 2)


def emulate(spec: CircuitSpec, mats: np.ndarray, state: np.ndarray) -> np.ndarray:
    st = state.astype(np.float32).copy()
    t = spec.t
    NT = spec.nthreads
    nwaves = NT // 64
    for ph in spec.phases:
        L = local_bits(spec, ph)
        rest = [p for p in range(spec.n + 1) if p not in set(L)]
        for tau in range(spec.tiles):
            tb = _deposit_const(tau, rest)
            lds = np.zeros(1 << t, np.float32)
            for ch in range(1 << (t - 2)):
                go = tb + _deposit_const(ch, [L[j] for j in range(2, t)])
                lo = _swz(ch << 2, spec.swizzle)
                lds[lo : lo + 4] = st[go : go + 4]
            for bl in ph.blocks:
                k = len(bl.qubits)
                NK = 1 << (k + 1)
                MT, S = NK // 32, NK // 2
                kpos = [0] + [L.index(q + 1) for q in bl.qubits]
                gpos = [j for j in range(t) if j not in set(kpos)]
                new = lds.copy()
                for w in range(nwaves):
                    mt = w % MT
                    gt = w // MT
                    acc = np.zeros((64, 16), np.float64)
                    # vectorised over lanes
                    lanes = np.arange(64)
                    g = gt * 32 + (lanes & 31)
                    kin = lanes >> 5
                    gdep = np.array([_deposit_const(int(x), gpos) for x in g])
                    rbase = gdep | (kin << kpos[1])
                    obase = gdep | (kin << kpos[2])
                    if MT > 1:
                        obase = obase | (mt << kpos[5])
                    frag = mats.reshape(-1, 64, 4)
                    A = np.zeros(
                        (S, 64), np.float64
                    )  # lane value of R operand per step
                    Bv = np.zeros((S, 64), np.float64)
                    for s in range(S):
                        q, e = s // 4, s % 4
                        A[s] = frag[bl.mat_off + mt * (S // 4) + q, :, e]
                        j, cc = s // 2, s % 2
                        off = _deposit_const(j << 2, kpos)
                        addr = np.array(
                            [_swz(int(a + off), spec.swizzle) for a in rbase]
                        )
                        Bv[s] = lds[addr + cc]
                    # MFMA 32x32x2: D[m][n] += sum_k A[m][k] B[k][n]
                    # A lane l: row l%32, k l//32 ; B lane l: col l%32, k l//32
                    for s in range(S):
                        Am = np.zeros((32, 2))
                        Bm = np.zeros((2, 32))
                        Am[lanes & 31, lanes >> 5] = A[s]
                        Bm[lanes >> 5, lanes & 31] = Bv[s]
                        D = Am @ Bm
                        for r in range(16):
                            row = 8 * (r // 4) + 4 * (lanes >> 5) + (r % 4)
                            acc[:, r] += D[row, lanes & 31]
                    for r in range(0, 16, 2):
                        noc = 8 * (r // 4) + (r % 4)
                        off = _deposit_const(noc, kpos)
                        for lane in range(64):
                            a2 = _swz(int(obase[lane] + off), spec.swizzle)
                            new[a2] = acc[lane, r]
                            new[a2 + 1] = acc[lane, r + 1]
                lds = new
            for ch in range(1 << (t - 2)):
                go = tb + _deposit_const(ch, [L[j] for j in range(2, t)])
                lo = _swz(ch << 2, spec.swizzle)
                st[go : go + 4] = lds[lo : lo + 4]
    return st
