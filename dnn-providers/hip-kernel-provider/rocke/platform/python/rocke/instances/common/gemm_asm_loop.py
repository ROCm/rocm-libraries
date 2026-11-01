"""Hand-ordered K-tile schedules for the universal GEMM (``TraitSpec.asm_loop``).

The K-tile body of the 256x256x64, 2x2-wave, 16x16x32 (8 x 8 MFMA atoms per
wave, 2 K atoms per tile), 2-stage DTL ring GEMM is emitted as ONE ordered
instruction list: 128 MFMAs with every fragment ds_read, DTL issue,
``s_waitcnt`` and ``s_barrier`` pinned after a given number of MFMAs
(``sched_barrier(0)`` after every MFMA keeps the order). The order is a table
of events; the built-in tables follow the main-loop orders of the public
hipBLASLt gfx950 256x256x64 (16x16 MFMA) kernels:

* ``tensile``: the Tensile SIA3 / PGR2 / PLR1 main loop (the f16 and bf16
  Tensile kernels share it). Four barriers per K tile: release of the inner (B)
  operand's stage, release of the outer (A) operand's stage, then an acquire
  per operand (``s_waitcnt vmcnt(n)`` that leaves the later-issued operand's
  loads in flight), each followed by that operand's next-tile fragment reads.
* ``custom``: the hand-written ``Custom_Cijk_..._MT256x256x64_MI16x16x1`` f16
  kernel: the same two releases, ONE acquire for both operands and
  the 16 next-tile reads packed right after it, ``lgkmcnt(0)`` at the tile end.
* ``table``: ``TraitSpec.asm_loop_table`` (same token syntax as below).

hipBLASLt's MFMA takes the B fragment as src A (Tensile ``SourceSwap``), so its
inner loop runs over the A-matrix fragments; rocKE's MFMA loop is ``for mi
(A, outer): for ni (B, inner)``, so the tables name the operands by role: the
INNER operand is rocKE's B, the OUTER one rocKE's A.

Token syntax (whitespace separated, ``K/EVENT``; K = MFMAs of the tile issued
before the event, 0..128):

* ``rB1.j`` / ``rA1.j``  ds_read of this tile's second-atom fragment j
* ``rB0.j`` / ``rA0.j``  ds_read of the NEXT tile's first-atom fragment j
  (from the next stage; only after that operand's acquire)
* ``dB.j`` / ``dA.j``    DTL issue j of this operand (tile i+2 into this stage)
* ``wl``                 ``s_waitcnt lgkmcnt(0)``
* ``vB`` / ``vA`` / ``vAB``  acquire wait: ``s_waitcnt vmcnt(n)`` with n = the
  loads allowed in flight once the previous tile's loads of that operand
  (both for ``vAB``) have landed
* ``bar``                ``s_barrier``
* ``p<N>``               ``s_setprio N``
* ``l<N>``               ``s_waitcnt lgkmcnt(N)`` (for ``asm_loop_rawrd``)

Every fragment read and DTL issue appears exactly once; :func:`check_table`
verifies the LDS hazards (a release = ``wl`` then ``bar`` after all of the
operand's ``r?1`` reads; a ``d?`` only after its operand's release; an
``r?0`` only after its operand's acquire = ``v?`` then ``bar``).
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Tuple

# geometry the tables are written for
N_ATOM = 64  # MFMAs per K atom (8 x 8)
N_TILE = 128  # MFMAs per K tile (2 K atoms)
N_FRAG = 8  # fragments per operand per K atom
N_PASS = 8  # DTL instructions per operand per K tile

ASM_LOOP_MODES = ("", "tensile", "custom", "hbr", "table")


def _seq(k0: int, step: int, ev: str, n: int = N_FRAG) -> List[str]:
    return [f"{k0 + j * step}/{ev}.{j}" for j in range(n)]


def _tensile() -> str:
    t: List[str] = []
    t += _seq(1, 2, "rB1")  # inner operand, second atom: MFMAs 1..15
    t += ["20/wl", "21/bar"]  # release the inner operand's stage
    t += ["23/dB.0", "25/rA1.0", "26/dB.1", "28/rA1.1", "29/dB.2", "31/rA1.2"]
    t += ["32/dB.3", "34/rA1.3", "35/dB.4", "37/rA1.4", "39/rA1.5", "41/rA1.6"]
    t += ["43/rA1.7"]
    t += ["51/wl", "52/bar"]  # release the outer operand's stage
    t += ["53/dB.5", "56/dB.6", "59/dB.7", "62/dA.0", "65/dA.1"]
    t += ["68/vB", "69/bar"]  # acquire the inner operand of tile i+1
    t += _seq(70, 2, "rB0")
    t += ["86/dA.2", "88/dA.3", "90/dA.4", "97/dA.5", "101/dA.6"]
    t += ["105/vA", "106/bar"]  # acquire the outer operand of tile i+1
    t += _seq(107, 2, "rA0")
    t += ["125/dA.7"]
    return " ".join(t)


def _custom() -> str:
    t: List[str] = []
    t += _seq(1, 2, "rB1")
    t += ["21/wl", "22/bar"]
    t += ["23/dB.0", "25/rA1.0", "26/dB.1", "28/rA1.1", "29/dB.2", "31/rA1.2"]
    t += ["32/dB.3", "34/rA1.3", "35/dB.4", "37/rA1.4", "39/rA1.5", "41/rA1.6"]
    t += ["43/rA1.7"]
    t += ["51/wl", "52/bar"]
    t += ["53/dB.5", "56/dB.6", "59/dB.7", "62/dA.0", "65/dA.1"]
    t += ["86/dA.2", "88/dA.3", "90/dA.4"]
    t += ["92/vAB", "93/bar"]  # one acquire for both operands
    t += ["94/rB0.0", "95/rB0.1", "96/rB0.2", "97/dA.5", "98/rB0.3", "99/rB0.4"]
    t += ["101/dA.6", "103/rB0.5", "104/rB0.6", "105/rB0.7"]
    t += ["106/rA0.0", "107/rA0.1", "110/rA0.2", "113/rA0.3", "115/rA0.4"]
    t += ["118/rA0.5", "121/rA0.6", "124/rA0.7", "125/dA.7", "127/wl"]
    return " ".join(t)


def _tensile_odd() -> str:
    """The Tensile loop copy run by waves on odd SIMDs (``hwreg(HW_ID, 4, 1)``):
    the fragment reads one MFMA later, the DTL issues around them moved."""
    t: List[str] = []
    t += _seq(2, 2, "rB1")
    t += ["20/wl", "21/bar"]
    t += ["23/rA1.0", "24/dB.0", "26/rA1.1", "27/dB.1", "29/rA1.2", "30/dB.2"]
    t += ["32/rA1.3", "33/dB.3", "35/rA1.4", "36/dB.4", "38/rA1.5", "40/rA1.6"]
    t += ["42/rA1.7"]
    t += ["51/wl", "52/bar"]
    t += ["54/dB.5", "57/dB.6", "60/dB.7", "63/dA.0", "66/dA.1"]
    t += ["68/vB", "69/bar"]
    t += _seq(71, 2, "rB0", 7) + ["91/rB0.7"]
    t += ["85/dA.2", "87/dA.3", "89/dA.4", "96/dA.5", "100/dA.6"]
    t += ["105/vA", "106/bar"]
    t += _seq(108, 2, "rA0")
    t += ["124/dA.7"]
    return " ".join(t)


def _custom_odd() -> str:
    """The hand-written kernel's odd-SIMD loop copy: DTL issues one MFMA later."""
    t: List[str] = []
    t += _seq(1, 2, "rB1")
    t += ["21/wl", "22/bar"]
    t += ["23/rA1.0", "24/dB.0", "26/rA1.1", "27/dB.1", "29/rA1.2", "30/dB.2"]
    t += ["32/rA1.3", "33/dB.3", "35/rA1.4", "36/dB.4", "39/rA1.5", "41/rA1.6"]
    t += ["43/rA1.7"]
    t += ["51/wl", "52/bar"]
    t += ["54/dB.5", "57/dB.6", "60/dB.7", "63/dA.0", "66/dA.1"]
    t += ["85/dA.2", "87/dA.3", "89/dA.4"]
    t += ["92/vAB", "93/bar"]
    t += ["94/rB0.0", "95/rB0.1", "96/dA.5", "97/rB0.2", "98/rB0.3", "99/rB0.4"]
    t += ["100/dA.6", "103/rB0.5", "104/rB0.6", "105/rB0.7"]
    t += ["106/rA0.0", "107/rA0.1", "110/rA0.2", "113/rA0.3", "115/rA0.4"]
    t += ["118/rA0.5", "121/rA0.6", "123/rA0.7", "124/dA.7", "127/wl"]
    return " ".join(t)


def _hbr(shift: int = 0) -> str:
    """rocKE's own ``hb_sched`` body (read every 2, release 4 MFMAs after the
    last second-atom read, the 16 DTL issues spread from there, one acquire
    (lgkmcnt(0) + vmcnt) after MFMA 68, then the next tile's first-atom reads
    every 2) as a table; ``shift`` moves every read / DTL issue (not the
    waits and barriers) that many MFMAs later (the odd copy)."""
    order = (
        [("A", 0)]
        + [("B", j) for j in range(N_FRAG)]
        + [("A", j) for j in range(1, N_FRAG)]
    )
    t: List[str] = []
    for j, (op, i) in enumerate(order):
        t.append(f"{1 + 2 * j + shift}/r{op}1.{i}")
    rel = 1 + 2 * (len(order) - 1) + 4  # 35
    t += [f"{rel}/wl", f"{rel}/bar"]
    span = N_TILE - (rel - 1)
    dl = [("A", j) for j in range(N_PASS)] + [("B", j) for j in range(N_PASS)]
    for j, (op, i) in enumerate(dl):
        t.append(f"{rel + (j * span) // len(dl) + shift}/d{op}.{i}")
    acq = N_ATOM + 4  # 68
    t += [f"{acq}/wl", f"{acq}/vAB", f"{acq}/bar"]
    for j, (op, i) in enumerate(order):
        t.append(f"{min(N_TILE, acq + 1 + 2 * j + shift)}/r{op}0.{i}")
    return " ".join(t)


BUILTIN_TABLES: Dict[str, str] = {
    "hbr": _hbr(),
    "hbr_odd": _hbr(1),
    "tensile": _tensile(),
    "custom": _custom(),
    "tensile_odd": _tensile_odd(),
    "custom_odd": _custom_odd(),
}

Event = Tuple[int, str, str, int]  # (k, kind, operand, index)


def parse_table(text: str) -> List[Event]:
    """Tokens -> ``(k, kind, operand, index)`` in table order (stable by k).

    kinds: ``r1`` / ``r0`` (reads), ``d`` (DTL), ``wl``, ``v`` (operand ``A``,
    ``B`` or ``AB``), ``bar``, ``p`` (index = priority)."""
    out: List[Tuple[int, int, Event]] = []
    for pos, tok in enumerate(text.split()):
        if "/" not in tok:
            raise ValueError(f"asm_loop token {tok!r}: expected K/EVENT")
        ks, ev = tok.split("/", 1)
        k = int(ks)
        if ev in ("wl", "bar"):
            e: Event = (k, ev, "", 0)
        elif ev in ("vA", "vB", "vAB"):
            e = (k, "v", ev[1:], 0)
        elif ev.startswith("p") and ev[1:].isdigit():
            e = (k, "p", "", int(ev[1:]))
        elif ev.startswith("l") and ev[1:].isdigit():
            e = (k, "l", "", int(ev[1:]))
        elif ev[:1] in ("r", "d") and "." in ev:
            head, idx = ev.split(".", 1)
            if head in ("rA1", "rB1", "rA0", "rB0"):
                e = (k, "r" + head[2], head[1], int(idx))
            elif head in ("dA", "dB"):
                e = (k, "d", head[1], int(idx))
            else:
                raise ValueError(f"asm_loop token {tok!r}: unknown event")
        else:
            raise ValueError(f"asm_loop token {tok!r}: unknown event")
        out.append((k, pos, e))
    out.sort(key=lambda x: (x[0], x[1]))
    return [e for _k, _p, e in out]


def check_table(evs: List[Event]) -> Optional[str]:
    """Why ``evs`` is not a hazard-free K-tile order, or ``None``."""
    seen = set()
    for k, kind, op, idx in evs:
        if not 0 <= k <= N_TILE:
            return f"event slot {k} outside 0..{N_TILE}"
        if kind in ("r1", "r0", "d"):
            lim = N_PASS if kind == "d" else N_FRAG
            if not 0 <= idx < lim:
                return f"{kind}{op}.{idx}: index outside 0..{lim - 1}"
            key = (kind, op, idx)
            if key in seen:
                return f"{kind}{op}.{idx} appears twice"
            seen.add(key)
        if kind == "p" and not 0 <= idx <= 3:
            return f"s_setprio {idx} outside 0..3"
        if kind == "l" and not 0 <= idx <= 15:
            return f"lgkmcnt({idx}) outside 0..15"
    for kind, lim in (("r1", N_FRAG), ("r0", N_FRAG), ("d", N_PASS)):
        for op in ("A", "B"):
            for j in range(lim):
                if (kind, op, j) not in seen:
                    return f"missing {kind}{op}.{j}"
    # second-atom reads must land before the atom's first MFMA ...
    for k, kind, op, idx in evs:
        if kind == "r1" and k > N_ATOM:
            return f"r{op}1.{idx} at {k}: after the second atom started (MFMA {N_ATOM})"
    # ... hazards: simulate the release / acquire protocol
    n_r1 = {"A": 0, "B": 0}
    released = {"A": False, "B": False}
    acquired = {"A": False, "B": False}
    wl_ok: Dict[str, bool] = {"A": False, "B": False}  # wl after all r1 of op
    pend_v = {"A": False, "B": False}
    for k, kind, op, idx in evs:
        if kind == "r1":
            if released[op]:
                return f"r{op}1.{idx} after its stage was released"
            n_r1[op] += 1
        elif kind == "wl":
            for o in ("A", "B"):
                if n_r1[o] == N_FRAG:
                    wl_ok[o] = True
        elif kind == "bar":
            for o in ("A", "B"):
                if wl_ok[o]:
                    released[o] = True
                if pend_v[o]:
                    acquired[o] = True
                    pend_v[o] = False
        elif kind == "v":
            for o in op:
                pend_v[o] = True
        elif kind == "d":
            if not released[op]:
                return f"d{op}.{idx} at {k}: before the {op} stage release (wl + bar)"
        elif kind == "r0":
            if not acquired[op]:
                return f"r{op}0.{idx} at {k}: before the {op} acquire (v{op} + bar)"
    for o in ("A", "B"):
        if not (released[o] and acquired[o]):
            return f"operand {o}: no release / acquire in the table"
    return None


def table_text(trait, odd: bool = False) -> str:
    """The event table of ``trait`` (``odd``: the copy for the odd wave group
    of ``asm_loop_skew``; a custom table gives it after ``||``)."""
    mode = trait.asm_loop
    if mode == "table":
        parts = trait.asm_loop_table.split("||")
        if len(parts) > 2:
            raise ValueError("asm_loop_table: at most one '||' (even || odd)")
        return parts[1 if odd and len(parts) == 2 else 0]
    return BUILTIN_TABLES[mode + ("_odd" if odd else "")]


def table_events(trait, odd: bool = False) -> List[Event]:
    return parse_table(table_text(trait, odd))


def barrier_signature(evs: List[Event]) -> List[str]:
    """The barrier sequence of a table (what each s_barrier completes): the
    two wave groups of ``asm_loop_skew`` must meet at matching barriers."""
    return [kind for (_k, kind, _o, _i) in evs if kind == "bar"]


def check_skew(trait) -> Optional[str]:
    """Why the even / odd tables of ``asm_loop_skew`` cannot run side by side."""
    e, o = table_events(trait), table_events(trait, odd=True)
    for nm, evs in (("even", e), ("odd", o)):
        why = check_table(evs)
        if why:
            return f"{nm} table: {why}"
    if len(barrier_signature(e)) != len(barrier_signature(o)):
        return "even / odd tables have different barrier counts"

    # the n-th barrier must have the same meaning: same set of events before
    # it (operand releases / acquires), so compare the protocol state
    def _proto(evs):
        st, out = [], []
        for _k, kind, op, _i in evs:
            if kind in ("wl", "v"):
                st.append(kind + op)
            elif kind == "bar":
                out.append(tuple(st))
                st = []
        return out

    if _proto(e) != _proto(o):
        return "even / odd tables: the barriers release / acquire different operands"
    return None


def name_tag(trait) -> str:
    """Kernel-name flag of the ``asm_loop`` mode (a custom table: its crc32)."""
    if trait.asm_loop == "table":
        return "asmt%08x" % (zlib.crc32(trait.asm_loop_table.encode()) & 0xFFFFFFFF)
    return "asm" + trait.asm_loop
