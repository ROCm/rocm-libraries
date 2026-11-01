# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device emitters of the CCL wire forms (inline asm, gfx950).

Every peer access is a ``global_*`` instruction on a 64-bit global address:
the ``saddr`` form (uniform SGPR-pair base = a peer arena kernarg, 32-bit VGPR
byte offset) for bulk payload, or the per-lane 64-bit VGPR address form for
flags (one instruction stores a line's flag to every rank: lane ``k`` writes
rank ``k mod N``, copy ``k div N``). Never buffer descriptors (``H7`` of
DESIGN-CCL), never scalar loads, never atomics on peer memory.

Cache policy: stores ``sc0 sc1`` (system scope; ``+ nt`` knob), loads of
mailbox words ``sc0 sc1``. The asm blocks own their ``s_waitcnt``: the
compiler's waitcnt pass does not see inside them.

Polls are bounded: ``s_memrealtime`` (100 MHz) against an absolute deadline
every 16 missed rounds. Every round also loads the local ``abort`` word (in
flight with the mailbox loads) and leaves on it before accepting data (for
responsiveness: correctness does not rest on it, because a wave that recorded
an error never publishes again, see ``collectives._Ctx.alive``).
Status (uniform, SGPR): 0 ready, 1 timeout, 2 abort seen, 3 tag ahead.

Must-fail acceptance mutants (tests only): ``ge`` accepts any tag >= the
expected one and has no tag-ahead check; ``any`` accepts any non-zero tag.

Inline asm is a mirrored op, so these emitters keep both engines; the CCL
family still compiles with an explicit backend (``collectives.backend_for``).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from rocke.core.ir import I32, I64, IRBuilder, Value, VectorType

__all__ = [
    "ST_ABORT",
    "ST_AHEAD",
    "ST_OK",
    "ST_TIMEOUT",
    "counted_publish",
    "load_lines",
    "load_lines_peers",
    "load_lines_v",
    "load_vecs_local",
    "poll_lamport",
    "poll_stripes",
    "push_vecs",
    "store_zero_lines",
    "load_word_sys",
    "poll_count_ge",
    "poll_flags",
    "poll_pairs",
    "push_v4",
    "sleep_units",
    "store_flags_lanes",
    "store_word_sys",
    "tag_add",
]

ST_OK, ST_TIMEOUT, ST_ABORT, ST_AHEAD = 0, 1, 2, 3
CHECK_EVERY = 16  # missed rounds between deadline / abort checks (power of two)
_PAUSE = {"s_nop": "s_nop 0", "s_sleep1": "s_sleep 1", "none": "", "s_sleep4": "s_sleep 4", "s_sleep16": "s_sleep 16"}
V4 = VectorType(I32, 4)


def cpol(push_cpol: str) -> str:
    return {"sc0sc1": "sc0 sc1", "sc0sc1nt": "sc0 sc1 nt"}[push_cpol]


POLL_CPOLS = {"sc0sc1": "sc0 sc1", "sc1": "sc1", "none": ""}


def lcpol(poll_cpol: str) -> str:
    """Cache bits of mailbox / payload loads (``sc0sc1`` default; ``sc1`` and
    ``none`` are levers / the coherence negative control)."""
    return POLL_CPOLS[poll_cpol]


def _lbl(stem: str, part: str) -> str:
    return "${:private}ccl" + stem + "${:uid}_" + part


def tag_add(b: IRBuilder, epoch: Value, use: Value) -> Value:
    """Uniform u32 ``epoch (+) use`` in the tag sequence 1 .. 2**32 - 1 (0 is
    skipped: the carry of the wrap adds one). Host twin:
    ``rocke.runtime.peer.tag_add``."""
    return b.inline_asm(
        "s_add_u32 $0, $1, $2\ns_addc_u32 $0, $0, 0",
        "=&s,s,s,~{scc}",
        [epoch, use],
        I32,
        result_name_hint="tag",
    )


def load_word_sys(b: IRBuilder, base: Value, voff: Value) -> Value:
    """One u32 of ``base + voff`` with system-scope bits, waited."""
    return b.inline_asm(
        "global_load_dword $0, $1, $2 sc0 sc1\ns_waitcnt vmcnt(0)",
        "=&v,v,s,~{memory}",
        [voff, base],
        I32,
        result_name_hint="ldsys",
    )


def store_word_sys(b: IRBuilder, base: Value, voff: Value, v: Value) -> None:
    """``global_store_dword ... sc0 sc1`` (not waited)."""
    b.inline_asm("global_store_dword $0, $1, $2 sc0 sc1", "v,v,s,~{memory}", [voff, v, base], None)


def push_v4(b: IRBuilder, bases: Sequence[Value], voff: Value, data: Value, push_cpol: str = "sc0sc1") -> None:
    """Store this lane's 16 bytes ``data`` at ``voff`` of every arena in
    ``bases`` (one asm: the stores issue back to back, not waited).

    Ends with ``s_nop 1``: a VALU write to the data VGPRs of a preceding store
    wider than 8 bytes needs a wait state (the GCN "12-dword store" hazard),
    and LLVM's hazard recognizer cannot see a store inside inline asm, so it
    does not pad the compiler's next VALU write. Seen on gfx950 without it: a
    ``v_mov_b64`` right after the asm overwrote the high pair of the line in
    flight (two-shot pair form, valid low pair, garbage high pair)."""
    cp = cpol(push_cpol)
    lines = [f"global_store_dwordx4 $0, $1, ${2 + i} {cp}" for i in range(len(bases))]
    lines.append("s_nop 1")
    b.inline_asm("\n".join(lines), "v,v," + ",".join(["s"] * len(bases)) + ",~{memory}", [voff, data, *bases], None)


def store_flags_lanes(b: IRBuilder, addr: Value, tag: Value) -> None:
    """Per-lane 64-bit address store of ``tag`` (``sc0 sc1``); call under the
    lane guard that selects the flag-writing lanes."""
    b.inline_asm("global_store_dword $0, $1, off sc0 sc1", "v,v,~{memory}", [addr, tag], None)


def drain(b: IRBuilder) -> None:
    """``s_waitcnt vmcnt(0)`` as asm (orders the payload stores before a flag)."""
    b.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], None)


def sleep_units(b: IRBuilder, n: Value, iv_name: str = "cclsl") -> None:
    """``n`` (uniform) x ``s_sleep 1`` (``iv_name`` unique per kernel)."""
    with b.scf_for(b.const_i32(0), n, b.const_i32(1), iv_name=iv_name):
        b.inline_asm("s_sleep 1", "", [], None)


def _abort_check(stem: str, abort_val: str, mask: str) -> List[str]:
    """Leave via ``abort`` when the local abort word (loaded in this round,
    in flight with the round's mailbox loads) is set: a peer that failed may
    keep pushing valid-tagged garbage, so the abort word wins over ready data."""
    return [f"v_cmp_ne_u32_e64 {mask}, 0, {abort_val}", f"s_and_b64 {mask}, {mask}, exec", "s_cbranch_scc1 " + _lbl(stem, "abort")]


def _tail(stem: str, pause: str, deadline_op: str, mask: str, now: str, it: str, st: str) -> List[str]:
    """Shared tail of a missed poll round: counter, deadline check every
    ``CHECK_EVERY`` rounds, pause, branch to top (leaves via ``timeout``)."""
    return [
        f"s_add_u32 {it}, {it}, 1",
        f"s_and_b32 {st}, {it}, {CHECK_EVERY - 1}",
        "s_cbranch_scc1 " + _lbl(stem, "pause"),
        f"s_memrealtime {now}",
        "s_waitcnt lgkmcnt(0)",
        f"v_cmp_lt_u64_e64 {mask}, {deadline_op}, {now}",
        f"s_and_b64 {mask}, {mask}, exec",
        "s_cbranch_scc1 " + _lbl(stem, "timeout"),
        _lbl(stem, "pause") + ":",
        _PAUSE[pause],
        "s_branch " + _lbl(stem, "top"),
    ]


def poll_flags(
    b: IRBuilder,
    addr: Value,
    tag: Value,
    deadline: Value,
    abort_addr: Value,
    win: Value,
    *,
    pause: str = "s_nop",
    mutant: str = "none",
    poll_cpol: str = "sc0sc1",
) -> Tuple[Value, Value, Value]:
    """Spin until every active lane's u32 at the 64-bit ``addr`` equals the
    uniform ``tag`` (must-fail mutants: ``ge`` is >= ``tag``, ``any`` is
    non-zero; neither checks tag-ahead),
    a lane's word is 1 .. ``win`` ahead of ``tag`` (status 3), the absolute
    ``deadline`` (i64, 100 MHz ticks) passes (1) or the local abort word at
    ``abort_addr`` is set (2). Returns ``(seen, lane mask, status)``: on a
    failure the mask holds the lanes still not at ``tag``."""
    s = "f"
    miss = {"ge": "v_cmp_lt_u32_e64 $1, $0, $7", "any": "v_cmp_eq_u32_e64 $1, 0, $0"}.get(mutant, "v_cmp_ne_u32_e64 $1, $0, $7")
    ahead_chk = [] if mutant in ("ge", "any") else [
        "v_sub_u32_e64 $4, $0, $7",
        "v_add_u32_e32 $4, -1, $4",
        "v_cmp_gt_u32_e64 $1, $10, $4",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc1 " + _lbl(s, "ahead"),
    ]  # fmt: skip
    lines = [
        "s_mov_b32 $2, 0",
        _lbl(s, "top") + ":",
        "global_load_dword $4, $9, off sc0 sc1",
        f"global_load_dword $0, $6, off {lcpol(poll_cpol)}",
        "s_waitcnt vmcnt(0)",
        *_abort_check(s, "$4", "$1"),
        miss,
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 " + _lbl(s, "ok"),
        *ahead_chk,
        *_tail(s, pause, "$8", "$1", "$5", "$2", "$3"),
        _lbl(s, "ok") + ":",
        "s_mov_b32 $3, 0",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "ahead") + ":",
        "s_mov_b32 $3, 3",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "timeout") + ":",
        "s_mov_b32 $3, 1",
        "s_branch " + _lbl(s, "miss"),
        _lbl(s, "abort") + ":",
        "s_mov_b32 $3, 2",
        _lbl(s, "miss") + ":",
        miss,
        "s_and_b64 $1, $1, exec",
        _lbl(s, "end") + ":",
    ]
    seen, mask, _it, st, _tv, _now = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,=&v,=&s,v,s,v,v,s,~{scc},~{memory}",
        [addr, tag, deadline, abort_addr, win],
        result_types=[I32, I64, I32, I32, I32, I64],
        result_name_hint="pollf",
    )
    return seen, mask, st


def load_lines(b: IRBuilder, base: Value, voffs: Sequence[Value], poll_cpol: str = "sc0sc1") -> List[Value]:
    """One ``global_load_dwordx4 ... sc0 sc1`` per offset (all in flight),
    then ``vmcnt(0)``: a lane's 16 bytes of every source's line."""
    n = len(voffs)
    lines = [f"global_load_dwordx4 ${i}, ${n + i}, ${2 * n} {lcpol(poll_cpol)}" for i in range(n)]
    lines.append("s_waitcnt vmcnt(0)")
    cons = ",".join(["=&v"] * n + ["v"] * n + ["s"]) + ",~{memory}"
    outs = b.inline_asm_multi("\n".join(lines), cons, [*voffs, base], result_types=[V4] * n, result_name_hint="ldl")
    return list(outs)


def load_v4_local(b: IRBuilder, base: Value, voff: Value) -> Value:
    """Plain (cached) ``global_load_dwordx4`` of local data, waited."""
    return b.inline_asm(
        "global_load_dwordx4 $0, $1, $2\ns_waitcnt vmcnt(0)", "=&v,v,s,~{memory}", [voff, base], V4,
        result_name_hint="ldx",
    )  # fmt: skip


def load_v2_local(b: IRBuilder, base: Value, voff: Value) -> Value:
    return b.inline_asm(
        "global_load_dwordx2 $0, $1, $2\ns_waitcnt vmcnt(0)", "=&v,v,s,~{memory}", [voff, base],
        VectorType(I32, 2), result_name_hint="ldx2",
    )  # fmt: skip


def poll_pairs(
    b: IRBuilder,
    base: Value,
    voffs: Sequence[Value],
    tag64: Value,
    neg64: Value,
    win64: Value,
    deadline: Value,
    abort_addr: Value,
    *,
    pause: str = "s_nop",
    mutant: str = "none",
    poll_cpol: str = "sc0sc1",
) -> Tuple[List[Value], Value, Value]:
    """Tagged-pair poll: each lane loads two pairs (16 bytes) at every
    ``voffs[r]`` of ``base`` per round (``2 N`` loads in flight) until every
    pair's tag (high dword) equals ``tag64`` (i64 zero-extended tag; must-fail
    mutants as :func:`poll_flags`). Tag
    ahead: ``0 <= tag' - tag - 1 < win`` with ``neg64 = -(tag + 1)``.
    Returns (``2 N`` i64 pairs in ``[src0 lo, src0 hi, src1 lo, ...]``
    order, mask, status)."""
    n = len(voffs)
    P = 2 * n
    # outputs: $0..$P-1 pairs, $P mask, $P+1 m2, $P+2 it, $P+3 st, $P+4 t64, $P+5 tmpv, $P+6 now
    mk, m2, it, st, t64, tv, now = (f"${P + k}" for k in range(7))
    i0 = P + 7  # inputs: voffs, base, tag64, neg64, win64, deadline, abort
    vo = [f"${i0 + r}" for r in range(n)]
    bs, tg, ng, wn, dl, ab = (f"${i0 + n + k}" for k in range(6))
    s = "p"
    cmp = {"ge": "v_cmp_lt_u64_e64", "any": "v_cmp_eq_u64_e64"}.get(mutant, "v_cmp_ne_u64_e64")

    def loads():
        out = [f"global_load_dword {tv}, {ab}, off sc0 sc1"]
        for r in range(n):
            cp = lcpol(poll_cpol)
            out.append(f"global_load_dwordx2 ${2 * r}, {vo[r]}, {bs} {cp}")
            out.append(f"global_load_dwordx2 ${2 * r + 1}, {vo[r]}, {bs} offset:8 {cp}")
        out.append("s_waitcnt vmcnt(0)")
        return out

    def missing():
        out = [f"s_mov_b64 {mk}, 0"]
        for p in range(P):
            ref = "0" if mutant == "any" else tg
            out += [f"v_lshrrev_b64 {t64}, 32, ${p}", f"{cmp} {m2}, {t64}, {ref}", f"s_or_b64 {mk}, {mk}, {m2}"]
        out.append(f"s_and_b64 {mk}, {mk}, exec")
        return out

    ahead = [f"s_mov_b64 {mk}, 0"]
    for p in range(P):
        ahead += [
            f"v_lshrrev_b64 {t64}, 32, ${p}",
            f"v_lshl_add_u64 {t64}, {t64}, 0, {ng}",
            f"v_cmp_gt_u64_e64 {m2}, {wn}, {t64}",
            f"s_or_b64 {mk}, {mk}, {m2}",
        ]
    ahead.append(f"s_and_b64 {mk}, {mk}, exec")
    ahead.append("s_cbranch_scc1 " + _lbl(s, "ahead"))
    if mutant in ("ge", "any"):
        ahead = []  # the acceptance mutants have no tag-ahead check
    lines = [
        f"s_mov_b32 {it}, 0",
        _lbl(s, "top") + ":",
        *loads(),
        *_abort_check(s, tv, mk),
        *missing(),
        "s_cbranch_scc0 " + _lbl(s, "ok"),
        *ahead,
        *_tail(s, pause, dl, mk, now, it, st),
        _lbl(s, "ok") + ":",
        f"s_mov_b32 {st}, 0",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "ahead") + ":",
        f"s_mov_b32 {st}, 3",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "timeout") + ":",
        f"s_mov_b32 {st}, 1",
        "s_branch " + _lbl(s, "miss"),
        _lbl(s, "abort") + ":",
        f"s_mov_b32 {st}, 2",
        _lbl(s, "miss") + ":",
        *missing(),
        _lbl(s, "end") + ":",
    ]
    cons = ",".join(["=&v"] * P + ["=&s", "=&s", "=&s", "=&s", "=&v", "=&v", "=&s"] + ["v"] * n + ["s", "s", "s", "s", "v", "v"])
    outs = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        cons + ",~{scc},~{memory}",
        [*voffs, base, tag64, neg64, win64, deadline, abort_addr],
        result_types=[I64] * P + [I64, I64, I32, I32, I64, I32, I64],
        result_name_hint="pollp",
    )
    return list(outs[:P]), outs[P], outs[P + 3]


def poll_count_ge(b: IRBuilder, base: Value, voff: Value, target: Value, deadline: Value, *, pause: str = "s_nop") -> Value:
    """Single-lane spin (call under a one-lane guard) until the u32 at
    ``base + voff`` (system-scope loads) is >= ``target`` (u32 compare) or the
    absolute ``deadline`` (i64, 100 MHz) passes. Returns the status (uniform):
    0 reached, 1 timeout. The grid-sync counter wait of ``sync="grid"``."""
    s = "g"
    lines = [
        "s_mov_b32 $2, 0",
        _lbl(s, "top") + ":",
        "global_load_dword $0, $5, $6 sc0 sc1",
        "s_waitcnt vmcnt(0)",
        "v_cmp_lt_u32_e64 $1, $0, $7",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 " + _lbl(s, "ok"),
        *_tail(s, pause, "$8", "$1", "$4", "$2", "$3"),
        _lbl(s, "ok") + ":",
        "s_mov_b32 $3, 0",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "timeout") + ":",
        "s_mov_b32 $3, 1",
        _lbl(s, "end") + ":",
    ]
    _v, _m, _it, st, _now = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,=&s,v,s,s,v,~{scc},~{memory}",
        [voff, base, target, deadline],
        result_types=[I32, I64, I32, I32, I64],
        result_name_hint="pollg",
    )
    return st


# ------------------------------------------------------------------ CCL-LM emitters
#
# Large-message levers (``layout.CclKnobs`` docstring): multi-vector pushes and
# loads (``unit_kib``), the lamport payload poll and its reset, the striped
# agent-scope grid-sync poll and the counted publish. Every block keeps its own
# ``s_waitcnt`` (the compiler's waitcnt pass does not see inside inline asm),
# and every store wider than 8 bytes ends with ``s_nop 1`` (see :func:`push_v4`).

LAMPORT_VBASE = 96  # first fixed VGPR of the lamport poll's vectors (up to 8 vectors: v96..v127; 16: v64..v127)
LAMPORT_VMAX = 64


def push_vecs(b: IRBuilder, bases: Sequence[Value], voff: Value, datas: Sequence[Value], push_cpol: str = "sc0sc1", *, drain_first: bool = False) -> None:
    """Store this lane's ``len(datas)`` 16-byte vectors at ``voff + 1024 v``
    of every arena in ``bases`` (vector ``v`` of a unit: one store
    instruction writes 1 KiB contiguous). ``drain_first`` prefixes
    ``s_waitcnt vmcnt(0)`` (lamport: this wave's earlier reset stores are
    performed before its next push)."""
    cp = cpol(push_cpol)
    nv = len(datas)
    lines = ["s_waitcnt vmcnt(0)"] if drain_first else []
    for i in range(len(bases)):
        for v in range(nv):
            off = f" offset:{1024 * v}" if v else ""
            lines.append(f"global_store_dwordx4 $0, ${1 + v}, ${1 + nv + i}{off} {cp}")
    lines.append("s_nop 1")
    cons = "v," + ",".join(["v"] * nv) + "," + ",".join(["s"] * len(bases)) + ",~{memory}"
    b.inline_asm("\n".join(lines), cons, [voff, *datas, *bases], None)


def load_vecs_local(b: IRBuilder, base: Value, voffs: Sequence[Value]) -> List[Value]:
    """Plain (cached) ``global_load_dwordx4`` of local data at every offset
    (all in flight), then ``vmcnt(0)``."""
    n = len(voffs)
    lines = [f"global_load_dwordx4 ${i}, ${n + i}, ${2 * n}" for i in range(n)] + ["s_waitcnt vmcnt(0)"]
    cons = ",".join(["=&v"] * n + ["v"] * n + ["s"]) + ",~{memory}"
    return list(b.inline_asm_multi("\n".join(lines), cons, [*voffs, base], result_types=[V4] * n, result_name_hint="ldv"))


def load_lines_v(b: IRBuilder, base: Value, voffs: Sequence[Value], nv: int, poll_cpol: str = "sc0sc1") -> List[Value]:
    """:func:`load_lines` of ``nv`` vectors per offset (``+ 1024 v``):
    ``len(voffs) x nv`` vectors in ``[off0 v0, off0 v1, .., off1 v0, ..]`` order."""
    n = len(voffs)
    cp = lcpol(poll_cpol)
    lines = []
    for i in range(n):
        for v in range(nv):
            off = f" offset:{1024 * v}" if v else ""
            lines.append(f"global_load_dwordx4 ${i * nv + v}, ${n * nv + i}, ${n * nv + n}{off} {cp}")
    lines.append("s_waitcnt vmcnt(0)")
    cons = ",".join(["=&v"] * (n * nv) + ["v"] * n + ["s"]) + ",~{memory}"
    return list(b.inline_asm_multi("\n".join(lines), cons, [*voffs, base], result_types=[V4] * (n * nv), result_name_hint="ldlv"))


def poll_lamport(
    b: IRBuilder,
    base: Value,
    voffs: Sequence[Value],
    nv: int,
    dtype: str,
    deadline: Value,
    abort_addr: Value,
    *,
    pause: str = "s_nop",
) -> Tuple[List[Value], Value, Value]:
    """Lamport payload poll: each round every lane loads its ``nv`` 16-byte
    vectors of every source (``voffs[r] + 1024 v``, system scope; ``N nv``
    loads in flight) plus the local abort word, until no wire element of any
    of them is empty (zero): bf16 halves (``(d - 0x10001) & ~d & 0x80008000``
    is non-zero exactly when some half is zero) or f32 words. Every element
    is checked, so readiness does not rest on any store width being
    single-copy atomic. The vectors land in fixed VGPRs (``{v[a:a+3]}``
    output constraints: inline asm cannot name the dwords of a tuple
    operand), from :data:`LAMPORT_VBASE` up. Returns (the ``N nv`` raw
    vectors in ``[src0 v0, src0 v1, .., src1 v0, ..]`` order, the mask of the
    lanes still missing, status 0 ok / 1 timeout / 2 abort)."""
    n = len(voffs)
    P = n * nv
    if 4 * P > LAMPORT_VMAX:
        raise ValueError(f"lamport poll: {P} vectors per lane > {LAMPORT_VMAX // 4}")
    vb = LAMPORT_VBASE if 4 * P <= 32 else LAMPORT_VBASE - 32
    regs = [f"v[{vb + 4 * p}:{vb + 4 * p + 3}]" for p in range(P)]
    # outputs: $0..$P-1 vectors (fixed), $P mask, $P+1 it, $P+2 st, $P+3 acc, $P+4 t, $P+5 abort word, $P+6 now
    mk, it, st, acc, tv, ab, now = (f"${P + k}" for k in range(7))
    i0 = P + 7  # inputs: voffs, base, deadline, abort addr
    vo = [f"${i0 + r}" for r in range(n)]
    bs, dl, aa = (f"${i0 + n + k}" for k in range(3))
    s = "l"

    def loads():
        out = [f"global_load_dword {ab}, {aa}, off sc0 sc1"]
        for r in range(n):
            for v in range(nv):
                off = f" offset:{1024 * v}" if v else ""
                out.append(f"global_load_dwordx4 {regs[r * nv + v]}, {vo[r]}, {bs}{off} sc0 sc1")
        out.append("s_waitcnt vmcnt(0)")
        return out

    def missing():
        out = []
        if dtype == "bf16":
            out.append(f"v_mov_b32 {acc}, 0")
            for p in range(P):
                for d in range(4):
                    w = f"v{vb + 4 * p + d}"
                    out += [f"v_subrev_u32_e32 {tv}, 0x10001, {w}", f"v_bfi_b32 {tv}, {w}, 0, {tv}", f"v_or_b32_e32 {acc}, {acc}, {tv}"]
            out += [f"v_and_b32_e32 {acc}, 0x80008000, {acc}", f"v_cmp_ne_u32_e64 {mk}, 0, {acc}"]
        else:
            out.append(f"v_mov_b32 {acc}, -1")
            for p in range(P):
                for d in range(4):
                    out.append(f"v_min_u32_e32 {acc}, {acc}, v{vb + 4 * p + d}")
            out.append(f"v_cmp_eq_u32_e64 {mk}, 0, {acc}")
        out.append(f"s_and_b64 {mk}, {mk}, exec")
        return out

    lines = [
        f"s_mov_b32 {it}, 0",
        _lbl(s, "top") + ":",
        *loads(),
        *_abort_check(s, ab, mk),
        *missing(),
        "s_cbranch_scc0 " + _lbl(s, "ok"),
        *_tail(s, pause, dl, mk, now, it, st),
        _lbl(s, "ok") + ":",
        f"s_mov_b32 {st}, 0",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "timeout") + ":",
        f"s_mov_b32 {st}, 1",
        "s_branch " + _lbl(s, "miss"),
        _lbl(s, "abort") + ":",
        f"s_mov_b32 {st}, 2",
        _lbl(s, "miss") + ":",
        *missing(),
        _lbl(s, "end") + ":",
    ]
    cons = ",".join([f"=&{{{r}}}" for r in regs] + ["=&s", "=&s", "=&s", "=&v", "=&v", "=&v", "=&s"] + ["v"] * n + ["s", "v", "v"])
    outs = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        cons + ",~{scc},~{memory}",
        [*voffs, base, deadline, abort_addr],
        result_types=[V4] * P + [I64, I32, I32, I32, I32, I32, I64],
        result_name_hint="polll",
    )
    return list(outs[:P]), outs[P], outs[P + 2]


def store_zero_lines(b: IRBuilder, base: Value, voffs: Sequence[Value], nv: int, zero: Value) -> None:
    """Lamport reset: store the zero vector ``zero`` (V4) over ``nv``
    vectors at every offset (``+ 1024 v``) of the local arena, system scope,
    not waited (the wave drains before its next push)."""
    n = len(voffs)
    lines = []
    for i in range(n):
        for v in range(nv):
            off = f" offset:{1024 * v}" if v else ""
            lines.append(f"global_store_dwordx4 ${i}, ${n}, ${n + 1}{off} sc0 sc1")
    lines.append("s_nop 1")
    cons = ",".join(["v"] * n + ["v", "s"]) + ",~{memory}"
    b.inline_asm("\n".join(lines), cons, [*voffs, zero, base], None)


def poll_stripes(b: IRBuilder, base: Value, voff: Value, target: Value, deadline: Value, *, pause: str = "s_nop") -> Value:
    """Wave-wide spin (call from one whole wave) until every active lane's
    u32 at ``base + voff`` (``sc1``: device scope; the stripes are local
    device memory) is ``>= target`` (per lane; a lane with target 0 is done
    at once) or the absolute ``deadline`` passes. Returns the uniform status:
    0 reached, 1 timeout. The striped grid sync of ``gsync="stripe"``."""
    s = "s"
    lines = [
        "s_mov_b32 $2, 0",
        _lbl(s, "top") + ":",
        "global_load_dword $0, $5, $6 sc1",
        "s_waitcnt vmcnt(0)",
        "v_cmp_lt_u32_e64 $1, $0, $7",
        "s_and_b64 $1, $1, exec",
        "s_cbranch_scc0 " + _lbl(s, "ok"),
        *_tail(s, pause, "$8", "$1", "$4", "$2", "$3"),
        _lbl(s, "ok") + ":",
        "s_mov_b32 $3, 0",
        "s_branch " + _lbl(s, "end"),
        _lbl(s, "timeout") + ":",
        "s_mov_b32 $3, 1",
        _lbl(s, "end") + ":",
    ]
    _v, _m, _it, st, _now = b.inline_asm_multi(
        "\n".join(x for x in lines if x),
        "=&v,=&s,=&s,=&s,=&s,v,s,v,v,~{scc},~{memory}",
        [voff, base, target, deadline],
        result_types=[I32, I64, I32, I32, I64],
        result_name_hint="polls",
    )
    return st


def counted_publish(b: IRBuilder, xin: Value, nxt_voffs: Sequence[Value], faddr: Value, tag: Value) -> List[Value]:
    """``publish="counted"``: issue the next unit's input loads (system-scope
    bits: this block also stores a peer flag, and the lint checks every
    memory line of a block with a peer operand), wait until only those loads
    are outstanding (``vmcnt(len)``: vector memory operations retire in issue
    order, so this wave's payload stores are performed), store ``tag`` at the
    per-lane 64-bit flag address ``faddr`` (lanes that publish nothing carry
    a local junk address), then wait for the loads only (``vmcnt(1)``: the
    flag store may stay in flight). Returns the loaded vectors."""
    n = len(nxt_voffs)
    lines = [f"global_load_dwordx4 ${i}, ${n + i}, ${2 * n} sc0 sc1" for i in range(n)]
    lines += [f"s_waitcnt vmcnt({n})", f"global_store_dword ${2 * n + 1}, ${2 * n + 2}, off sc0 sc1", "s_waitcnt vmcnt(1)"]
    cons = ",".join(["=&v"] * n + ["v"] * n + ["s", "v", "v"]) + ",~{memory}"
    return list(b.inline_asm_multi("\n".join(lines), cons, [*nxt_voffs, xin, faddr, tag], result_types=[V4] * n, result_name_hint="ldcnt"))


def load_lines_peers(b: IRBuilder, bases: Sequence[Value], voffs: Sequence[Value], nv: int) -> List[Value]:
    """``form="pull"``: this lane's ``nv`` 16-byte vectors (``+ 1024 v``) of
    unit ``voffs[r]`` from arena ``bases[r]`` (peer or own) for every source,
    system scope (``sc0 sc1``), all in flight, then ``vmcnt(0)``. Returns
    ``len(bases) x nv`` vectors in ``[src0 v0, src0 v1, .., src1 v0, ..]`` order."""
    n = len(bases)
    lines = []
    for r in range(n):
        for v in range(nv):
            off = f" offset:{1024 * v}" if v else ""
            lines.append(f"global_load_dwordx4 ${r * nv + v}, ${n * nv + r}, ${n * nv + n + r}{off} sc0 sc1")
    lines.append("s_waitcnt vmcnt(0)")
    cons = ",".join(["=&v"] * (n * nv) + ["v"] * n + ["s"] * n) + ",~{memory}"
    return list(b.inline_asm_multi("\n".join(lines), cons, [*voffs, *bases], result_types=[V4] * (n * nv), result_name_hint="ldpl"))
