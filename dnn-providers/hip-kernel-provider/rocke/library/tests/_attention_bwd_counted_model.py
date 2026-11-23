# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Model checks of the counted in-loop waits (``loop_sync = "counted"``).

The kernel's own count (``_attention_bwd_counted.CountedWaits``) follows the
emitter's events; the checks here recount from what was emitted, so a wrong
count, a moved load or a moved atomic is caught.

* :func:`counted_ir_problems`: on the IR of one q loop, every in-loop barrier
  is ``s_waitcnt(vmcnt=N, lgkmcnt=0)`` + ``s_barrier`` with ``N`` equal to the
  dQ atomics between the last global load and the barrier, walking the loop
  body twice (the back-edge), no full or LDS-only barrier is left in the loop,
  every global load of the body precedes its first atomic, and the first LDS
  stores of the body consume the staged (carried) values, so on the first trip
  the prologue's loads have retired before the first barrier.
* :func:`counted_isa_problems`: on the disassembly of a compiled kernel, a
  simulation of the in-order ``vmcnt`` queue through every matrix loop (entered
  through the instructions before it, then three trips): at every
  ``s_barrier`` no global load is in flight (safety), and no ``s_waitcnt
  vmcnt(k)`` in a steady trip retires an atomic issued after the wave's
  youngest load (no drain of the atomics). This checks the code the backend
  emitted, with its own waits and its own instruction order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_IR_LOADS = ("memref.global_load_typed", "memref.global_load_vN", "memref.global_load")
_IR_ATOMIC = "memref.global_atomic_add_f32"
_IR_FULL = ("tile.sync", "tile.sync_lds_only")


def _flat(ops):
    """Program-order ops of a region with every nested region inlined."""
    for op in ops:
        yield op
        for r in op.regions:
            yield from _flat(r.ops)


def ir_events(body_ops) -> list[tuple[str, object]]:
    """``(kind, payload)`` of a loop body: ``L`` load, ``A`` atomic, ``B``
    counted barrier (payload: the vmcnt of the wait right before it), ``S`` a
    full / LDS-only barrier, ``F`` a nested loop, ``M`` a matrix op."""
    out = []
    pending = None
    for op in _flat(body_ops):
        if op.name in _IR_LOADS:
            out.append(("L", op))
        elif op.name == _IR_ATOMIC:
            out.append(("A", op))
        elif op.name == "tile.s_waitcnt":
            pending = op
        elif op.name == "tile.s_barrier_bare":
            out.append(("B", pending))
            pending = None
        elif op.name in _IR_FULL:
            out.append(("S", op))
        elif op.name == "scf.for":
            out.append(("F", op))
        elif op.name == "tile.mma":
            out.append(("M", op))
        elif op.name == "tile.sched_barrier":
            out.append(("SB", op))
    return out


def atomics_since_load(events, j) -> int:
    """Atomics between the last load before position ``j`` and ``j``."""
    n = 0
    for kind, _p in reversed(events[:j]):
        if kind == "L":
            return n
        if kind == "A":
            n += 1
    return n


def counted_ir_problems(loop) -> tuple[list[str], list[int]]:
    """Problems of one counted q loop (an ``scf.for`` op) and its barrier
    vmcnt values in body order."""
    body = loop.regions[0].ops
    ev = ir_events(body)
    probs = []
    if any(k == "S" for k, _ in ev):
        probs.append("a full or LDS-only barrier inside the counted q loop")
    if any(k == "F" for k, _ in ev):
        probs.append("a loop nested in the counted q step")
    waits = []
    for k, op in ev:
        if k != "B":
            continue
        if op is None:
            probs.append("a barrier without an s_waitcnt right before it")
            continue
        vm, lk = op.attrs.get("vmcnt", -1), op.attrs.get("lgkmcnt", -1)
        if lk != 0:
            probs.append(f"counted barrier with lgkmcnt {lk} (not 0)")
        waits.append(vm)
    loads = [i for i, (k, _) in enumerate(ev) if k == "L"]
    atomics = [i for i, (k, _) in enumerate(ev) if k == "A"]
    if not loads or not atomics:
        probs.append("the counted q step has no next-step loads or no atomics")
    elif loads[-1] > atomics[0]:
        probs.append("a global load after the first dQ atomic of the step")
    else:
        # the loads sit between two sched_barrier(0) fences, with nothing else
        fences = [
            i
            for i, (k, op) in enumerate(ev)
            if k == "SB" and op.attrs.get("mask", -1) == 0
        ]
        before = [i for i in fences if i < loads[0]]
        after = [i for i in fences if i > loads[-1]]
        if not before or not after:
            probs.append("the next-step loads are not fenced by sched_barrier(0)")
        else:
            inside = {k for k, _ in ev[before[-1] + 1 : after[0]]}
            if inside != {"L"}:
                probs.append(f"the fenced next-step loads hold {sorted(inside)}")
    twice = ev + ev
    n = len(ev)
    bi = 0
    for j in range(n, 2 * n):
        if twice[j][0] != "B":
            continue
        want = atomics_since_load(twice, j)
        got = waits[bi] if bi < len(waits) else None
        if got != want:
            probs.append(
                f"barrier {bi}: vmcnt({got}), {want} atomics since the last load"
            )
        bi += 1
    # first trip: the step-top LDS stores consume the carried (staged) values
    carried = {m["name"] for m in loop.attrs["iter_args"]}
    first_b = next((i for i, (k, _) in enumerate(ev) if k == "B"), None)
    stores = []
    for op in _flat(body):
        if op.name == "tile.s_barrier_bare":
            break
        if op.name.startswith("tile.smem_store"):
            stores.append(op)
    if first_b is None or not stores:
        probs.append("no LDS store of the staged tile before the first barrier")
    else:
        fed = _fed_by(body, carried)
        if not all(any(v.name in fed for v in s.operands) for s in stores):
            probs.append("a step-top LDS store not fed by the carried staged values")
    return probs, waits


def _fed_by(body_ops, roots: set) -> set:
    """Value names derived from ``roots`` (vector element moves only)."""
    out = set(roots)
    for op in _flat(body_ops):
        if op.name in ("vector.extract", "vector.pack", "vector.concat") and any(
            v.name in out for v in op.operands
        ):
            out.update(r.name for r in op.results)
    return out


# --------------------------------------------------------------------------- ISA
_VM_RE = re.compile(r"vmcnt\((\d+)\)")
# Every VMEM op counts in vmcnt (scratch spill reloads and stores too).
_LOAD_PREFIX = ("global_load", "buffer_load", "flat_load", "scratch_load")
_OTHER_VMEM = (
    "global_atomic",
    "buffer_atomic",
    "flat_atomic",
    "global_store",
    "buffer_store",
    "flat_store",
    "scratch_store",
)


def _kind(ins) -> str | None:
    mn = ins.mnemonic
    if mn.startswith(_LOAD_PREFIX):
        return "L"
    if mn.startswith(_OTHER_VMEM):
        return "A"
    if mn == "s_waitcnt":
        return "W"
    if mn == "s_barrier":
        return "B"
    return None


class _Queue:
    """The in-order vmcnt queue of one wave: ``(kind, trip)`` entries."""

    def __init__(self) -> None:
        self.q: list[tuple[str, int]] = []
        self.since_load = 0

    def issue(self, kind: str, trip: int = -1) -> None:
        self.q.append((kind, trip))
        self.since_load = 0 if kind == "L" else self.since_load + 1

    def wait(self, k: int) -> list[tuple[str, int]]:
        """Retire down to ``k`` entries; the retired entries."""
        out = []
        while len(self.q) > k:
            out.append(self.q.pop(0))
        return out

    def loads(self) -> bool:
        return any(kind == "L" for kind, _t in self.q)


_TERMINATORS = ("s_branch", "s_endpgm", "s_setpc_b64")


def _dominators(succ, pred) -> dict:
    """Immediate dominators of the blocks reachable from block 0 (iterative
    Cooper-Harvey-Kennedy over the reverse postorder); ``idom[0] = 0``."""
    order, seen = [], set()
    stack = [(0, iter(succ[0]))]
    seen.add(0)
    while stack:
        node, it = stack[-1]
        nxt = next((s for s in it if s not in seen), None)
        if nxt is None:
            order.append(node)
            stack.pop()
        else:
            seen.add(nxt)
            stack.append((nxt, iter(succ[nxt])))
    rpo = order[::-1]
    pos = {b: i for i, b in enumerate(rpo)}
    idom = {0: 0}

    def intersect(a, b):
        while a != b:
            while pos[a] > pos[b]:
                a = idom[a]
            while pos[b] > pos[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            ps = [p for p in pred[b] if p in idom]
            if not ps:
                continue
            new = ps[0]
            for p in ps[1:]:
                new = intersect(p, new)
            if idom.get(b) != new:
                idom[b] = new
                changed = True
    return idom


def natural_loops(instrs) -> list[tuple[int, list]]:
    """Innermost natural loops of a disassembly: ``(header address, the
    loop's instructions in address order)``.

    The basic blocks come from the branches; a back-edge is an edge to a
    block that dominates its source; a loop is its header plus every block
    that reaches a back-edge source without passing the header (all
    back-edges to one header form one loop). A loop that holds another
    loop's header is not innermost. The instructions are the loop's blocks in
    layout order from the header on (a loop whose blocks the backend laid out
    around another loop's is still exactly its own blocks).
    """
    n = len(instrs)
    if not n:
        return []
    index = {ins.addr: i for i, ins in enumerate(instrs)}
    leaders = {0}
    for i, ins in enumerate(instrs):
        mn = ins.mnemonic
        if ins.target is not None and ins.target in index:
            leaders.add(index[ins.target])
        if (mn.startswith("s_cbranch") or mn in _TERMINATORS) and i + 1 < n:
            leaders.add(i + 1)
    starts = sorted(leaders)
    blocks = [
        (s, (starts[k + 1] if k + 1 < len(starts) else n)) for k, s in enumerate(starts)
    ]
    bstart = {s: k for k, (s, _e) in enumerate(blocks)}
    succ = []
    for s, e in blocks:
        last = instrs[e - 1]
        out = []
        tgt = (
            bstart.get(index.get(last.target, -1)) if last.target is not None else None
        )
        if last.mnemonic == "s_branch":
            out = [tgt] if tgt is not None else []
        elif last.mnemonic in ("s_endpgm", "s_setpc_b64"):
            out = []
        else:
            if last.mnemonic.startswith("s_cbranch") and tgt is not None:
                out.append(tgt)
            if e < n:
                out.append(bstart[e])
        succ.append(out)
    pred = [[] for _ in blocks]
    for k, ss in enumerate(succ):
        for t in ss:
            pred[t].append(k)
    idom = _dominators(succ, pred)

    def dominates(h, u):
        while u is not None:
            if u == h:
                return True
            nxt = idom.get(u)
            u = None if nxt == u else nxt
        return False

    loops: dict[int, set] = {}
    for u, ss in enumerate(succ):
        for h in ss:
            if not dominates(h, u):
                continue  # not a back-edge (the target does not dominate it)
            body = loops.setdefault(h, {h})
            stack = [u]
            while stack:
                x = stack.pop()
                if x in body:
                    continue
                body.add(x)
                stack.extend(pred[x])
    out = []
    for h, body in loops.items():
        if any(o != h and o in body for o in loops):
            continue  # holds another loop: not innermost
        ks = sorted(body)
        ks = ks[ks.index(h) :] + ks[: ks.index(h)]  # from the header on
        ins = [instrs[i] for k in ks for i in range(*blocks[k])]
        out.append((instrs[blocks[h][0]].addr, ins))
    return sorted(out, key=lambda x: x[0])


def counted_isa_problems(
    isa_text: str, *, barriers_per_step: int = 3, single_wave: bool = False
) -> dict:
    """Simulate every matrix loop of a counted kernel's disassembly.

    A loop trip holds a multiple of ``barriers_per_step`` barriers (the
    backend may unroll the q loop); in a single-wave workgroup a backend may
    drop every ``s_barrier`` and keep the waits, which the drain check still
    covers. Returns ``{"problems": [...], "loops": [{"header":
    addr, "waits": [vmcnt at each barrier of a steady trip], "vmcnt0": bool,
    "barriers": n}, ...]}``.
    """
    from ._attention_bwd_resources import _is_mfma, parse_disassembly

    instrs = parse_disassembly(isa_text)
    loops = [
        (h, own)
        for h, own in natural_loops(instrs)
        if any(_is_mfma(x.mnemonic) for x in own)
    ]
    probs, recs = [], []
    if not loops:
        probs.append("no matrix loop in the disassembly")
    for header, own in loops:
        lp = _Hdr(header)
        qu = _Queue()
        for ins in instrs:  # the instructions laid out before the loop
            if ins.addr >= header:
                break
            _step(qu, ins, None, None)
        rec = {"header": lp.header_addr, "waits": [], "vmcnt0": False}
        # the wait right before each barrier (no memory op in between)
        kinds = [_kind(x) for x in own]
        bar_wait = set()
        for i, k in enumerate(kinds):
            if k != "W":
                continue
            nxt = next((kk for kk in kinds[i + 1 :] if kk is not None), None)
            if nxt == "B":
                bar_wait.add(i)
        has_bar = "B" in kinds
        for trip in range(3):
            n_bar, waits = 0, []
            last_wait = None
            for i, ins in enumerate(own):
                kind = kinds[i]
                if kind == "W":
                    m = _VM_RE.search(ins.operands)
                    if m:
                        k = int(m.group(1))
                        last_wait = k
                        if k == 0:
                            rec["vmcnt0"] = True
                        if trip and k < qu.since_load:
                            probs.append(
                                f"loop {lp.header_addr:#x} trip {trip}: s_waitcnt "
                                f"vmcnt({k}) at {ins.addr:#x} drains "
                                f"{qu.since_load - k} op(s) issued after the "
                                "youngest load"
                            )
                        retired = qu.wait(k)
                        # A next-step load waited on before the barrier that
                        # retires it: its latency is not hidden, and the wait
                        # retires the previous step's atomics with it.
                        early = [t for kk, t in retired if kk == "L" and t == trip]
                        if trip and has_bar and i not in bar_wait and early:
                            probs.append(
                                f"loop {lp.header_addr:#x} trip {trip}: s_waitcnt "
                                f"vmcnt({k}) at {ins.addr:#x} waits for a load of "
                                "this trip before a barrier"
                            )
                    else:
                        last_wait = None
                    continue
                if kind == "B":
                    n_bar += 1
                    waits.append(last_wait)
                    if qu.loads():
                        probs.append(
                            f"loop {lp.header_addr:#x} trip {trip}: a global load "
                            f"in flight at the s_barrier at {ins.addr:#x}"
                        )
                    continue
                if kind in ("L", "A"):
                    qu.issue(kind, trip)
                    last_wait = None
            ok_bar = n_bar % barriers_per_step == 0 and (n_bar > 0 or single_wave)
            if not ok_bar:
                probs.append(
                    f"loop {lp.header_addr:#x}: {n_bar} barriers per trip, "
                    f"expected a multiple of {barriers_per_step}"
                    + (" (or 0: single-wave workgroup)" if single_wave else "")
                )
            if trip == 2:
                rec["waits"] = waits
                rec["barriers"] = n_bar
        recs.append(rec)
    return {"problems": probs, "loops": recs}


class _Hdr:
    def __init__(self, addr: int) -> None:
        self.header_addr = addr


def _step(qu: _Queue, ins, _a, _b) -> None:
    kind = _kind(ins)
    if kind in ("L", "A"):
        qu.issue(kind)
    elif kind == "W":
        m = _VM_RE.search(ins.operands)
        if m:
            qu.wait(int(m.group(1)))


def expected_waits(n_atomics: int, issue: str) -> list:
    """The vmcnt of the three in-loop barriers (staging, dS or P / dS, step
    end) of the model in ``_attention_bwd_counted`` (every P / dS route and
    stage order: the middle barrier precedes G4)."""
    middle = 0 if issue == "early" else n_atomics
    return [n_atomics, middle, n_atomics]


def summarize(waits: Sequence) -> str:
    return ",".join("-" if w is None else str(w) for w in waits)
