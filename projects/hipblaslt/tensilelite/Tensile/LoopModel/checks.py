# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ordering obligations, and whether the emitted tree honours them."""

from __future__ import annotations

import dataclasses
from itertools import product as _iproduct
from dataclasses import dataclass

from .ir import (Bind, Branch, Cond, Inst, Load, Loop, Mma, Peel, Space,
                 _movement, binds_axis, child_bodies)
from .schedule import Schedule, build_S, loads_in_place, peel_depths, preloaded_tiles
from .traversal import requested_read_ahead
from .theta import AgentAssignment
from .ir import COPY, READ
from .traversal import chunks_crossed, prefetch_axis_name, varying_axes


# --- ledger ----------------------------------------------------------------

@dataclass(frozen=True)
class Endpoint:
    """One end of an obligation: which operand, at which site, at which coordinate."""
    op: object
    at: str
    role: str = ""
    coord: tuple = ()
    gen: tuple = ()

    def qualified(self, *names):
        """This endpoint with additional generation qualifiers appended (the `Await` dep form)."""
        return Endpoint(self.op, self.at, self.role, self.coord,
                        self.gen + tuple(name for name in names if name))

    def render(self) -> str:
        """Display form."""
        operand = "+".join(self.op) if isinstance(self.op, tuple) else self.op
        step = f"{operand}@{self.at}" if not self.role else f"{operand}:{self.at}:{self.role}"
        if self.coord:
            step += ":" + ",".join(f"{axis}{v}" for axis, v in self.coord)
        if self.gen:
            step += ":" + ":".join(self.gen)
        return step

    def __str__(self):
        return self.render()


@dataclass(frozen=True)
class Obligation:
    producer: Endpoint; consumer: Endpoint; kind: str; counter: str


def movement_name(theta, opname):
    """The producer identity of this operand's global-to-shared movement."""
    for key, _members, _name in theta.movement_units():
        if opname in key:
            return key[0] if len(key) == 1 else key
    return opname


def _raw_residency(theta):
    """Every transfer owes its destination: the value must have landed before it is used."""
    out = set()
    for operand in theta.operands:
        for hop in operand.movements:
            name = movement_name(theta, operand.name) if hop.dst == Space.SHARED else operand.name
            out.add(Obligation(Endpoint(name, hop.src), Endpoint(name, hop.dst),
                               "RAW-residency", hop.counter))
    return out


def _refill_war(theta, depths, requested_steps):
    """Require each refill to follow the prior value's last reader."""
    out = set()
    chunk = theta.summation_chunk_name()
    for operand in theta.operands:
        read = operand.trajectory.fragment_fill
        if read is None:
            continue
        for group, depth in depths.groups_of(operand.name).items():
            inplace = depth < 2 or loads_in_place(
                theta, operand, requested_steps, depths, groups=(group,))
            out.add(Obligation(
                Endpoint(operand.name, group, "lastread"),
                Endpoint(operand.name, group, "refill"),
                "inplace-WAR" if inplace else "rotation-WAR", read.counter))

        shared = operand.trajectory.shared
        if shared is None:
            continue
        delta = theta.off_at(operand, COPY, chunk) if chunk is not None else 0
        inplace = shared.ring_depth < 2 or shared.ring_depth <= delta
        out.add(Obligation(
            Endpoint(operand.name, Space.SHARED, "lastread"),
            Endpoint(operand.name, Space.SHARED, "refill"),
            "inplace-WAR" if inplace else "rotation-WAR", read.counter))
    return out


def _chunk_crossing_raw(theta, depths, plans):
    """A read that runs into the next chunk owes the copy that fills it, in this trip."""
    chunk = theta.summation_chunk_name()
    if chunk is None:
        return set()
    out = set()
    for operand in theta.operands:
        if operand.trajectory.shared is None:
            continue
        steps = plans.want(operand)
        crossed = max((chunks_crossed(theta, operand, plans.steps(operand, group, steps))
                       for group in operand.fragment.groups()), default=0)
        if crossed < 1:
            continue  # the read stays inside its own chunk
        if theta.off_at(operand.name, COPY, chunk) > crossed:
            continue  # an earlier trip already copied it
        movement = movement_name(theta, operand.name)
        out.add(Obligation(Endpoint(movement, Space.SHARED, "copy"),
                           Endpoint(movement, Space.SHARED, "crossing-read"),
                           "crossing-RAW", operand.trajectory.shared_fill.counter))
    return out


def _readahead_residency(theta, depths, requested_steps, plans):
    """Each tile the prologue pre-issues owes the wmma that consumes it on the first trip."""
    if not requested_steps:
        return set()
    out = set()
    for operand in theta.operands:
        read = operand.trajectory.shared_read
        if read is None:
            continue
        counter = read.counter
        # the OPERAND's own depth, matching what `fill_reads` primes
        varying = set(varying_axes(theta, operand))
        for coord in preloaded_tiles(theta, operand,
                                     plans.want(operand),
                                     depths, plans):
            key = tuple(sorted((axis, value) for axis, value in coord.items()
                               if axis in varying))
            out.add(Obligation(Endpoint(operand.name, "prologue", coord=key),
                               Endpoint(operand.name, "wmma", coord=key),
                               "readahead-residency", counter))
    return out


def build_ledger(theta, depths, plans=None):
    """The orderings the kernel must respect, in four families, built before any instruction."""
    plans = Schedule(theta, depths) if plans is None else plans
    steps = peel_depths(theta, depths, plans).requested_steps
    ledger = (_raw_residency(theta)
              | _refill_war(theta, depths, steps)
              | _chunk_crossing_raw(theta, depths, plans)
              | _readahead_residency(theta, depths, steps, plans))
    return sorted(ledger, key=lambda o: (o.kind, o.producer.render(), o.consumer.render(),
                                         o.counter))


_MERGEABLE = ("RAW-residency", "crossing-RAW")


def _reissues(inst, dep):
    """Does this instruction re-issue the producer `dep` names, invalidating its boundary?"""
    load = getattr(inst, "op", None)
    if not isinstance(load, Load) or load.dst != Space.SHARED:
        return False  # only a copy produces a shared residency
    return dep.op in (tuple(load.tokens), load.tokens[0] if load.tokens else None)


def _drop_established_awaits(inst, live):
    """Drop the awaits an earlier one on the same straight line already made true."""
    for dep in list(live):
        if _reissues(inst, dep):
            live.discard(dep)  # re-issued, so the earlier wait no longer covers it
    keep = []
    for await_node in (inst.awaits or ()):
        if await_node.kind in _MERGEABLE:
            if await_node.dep in live:
                continue  # established above, in this same line
            live.add(await_node.dep)
        keep.append(await_node)
    return _replace_awaits(inst, tuple(keep))


def _discharge(nodes, live):
    """`live` is what is known true here; only a straight line carries it forward."""
    out = []
    for node in nodes:
        if isinstance(node, Loop):
            out.append(_rebuild(node, [_discharge(body, set()) for body in node.bodies]))
            live.clear()
        elif isinstance(node, Branch):
            out.append(_rebuild(node, {role: _discharge(arm, set(live))
                                       for role, arm in node.arms.items()}))
            live.clear()  # arms may differ; nothing survives the join
        elif isinstance(node, Cond):
            out.append(_rebuild(node, (_discharge(node.then, set(live)),
                                       _discharge(node.els or [], set(live)))))
            live.clear()
        elif isinstance(node, (Peel, Bind)):
            out.append(_rebuild(node, _discharge(node.body, live)))  # runs once: `live` flows on
        elif isinstance(node, Inst):
            out.append(_drop_established_awaits(node, live))
        else:
            out.append(node)
    return out


def discharge_once(tree):
    """Wait for each shared residency once per straight line, not at every instruction."""
    return _discharge(list(tree), set())


def _rebuild(node, body):
    """`node` with its child body/bodies/arms replaced -- dataclasses.replace on the right field."""
    if isinstance(node, Loop):
        return dataclasses.replace(node, bodies=body)
    if isinstance(node, Branch):
        return dataclasses.replace(node, arms=body)
    if isinstance(node, Cond):
        return dataclasses.replace(node, then=body[0], els=body[1])
    return dataclasses.replace(node, body=body)


def _replace_awaits(inst, awaits):
    return inst if awaits == tuple(inst.awaits or ()) else dataclasses.replace(inst, awaits=awaits)


def walk_insts(tree):
    """Every Inst in the tree, in emission order."""
    for node in tree:
        if isinstance(node, Inst):
            yield node
        for body in child_bodies(node):
            yield from walk_insts(body)


def all_awaits(tree):
    """Every Await in the tree."""
    return [await_node for inst in walk_insts(tree) for await_node in inst.awaits]


def _flatten_trip(nodes, inline_outer=False):
    """The instructions of one trip, with every inner loop and branch inlined."""
    out = []
    for node in nodes:
        if isinstance(node, Inst):
            out.append(node)
        elif isinstance(node, Loop) and (inline_outer or not node.outer):
            for body in node.bodies:
                out += _flatten_trip(body, inline_outer)
        elif isinstance(node, Branch) and (inline_outer or not node.outer):
            for arm in node.arms.values():
                out += _flatten_trip(arm, inline_outer)
        elif isinstance(node, Bind):  # a peel step's iter binding
            out += _flatten_trip(node.body, inline_outer)
        elif isinstance(node, Cond):
            out += _flatten_trip(node.then, inline_outer)  # the steady arm
    return out


def _collect_regions(nodes, label, regions):
    for node in nodes:
        if isinstance(node, Loop) and node.outer:  # steady trip: one region per body
            for body in node.bodies:
                regions.append((label + "/steady", _flatten_trip(body)))
        elif isinstance(node, Peel):
            if node.kind == "drain":  # each drain step (a Bind) is its own trip
                for step in node.body:
                    regions.append((label + "/drain", _flatten_trip([step])))
            else:
                regions.append((label + "/" + (node.kind or "peel"), _flatten_trip(node.body)))
        elif isinstance(node, Cond):
            _collect_regions(node.then, label + "/cond", regions)
            if node.els:
                _collect_regions(node.els, label + "/else", regions)


def _region_bodies(tree):
    """The instruction lists that form ONE emitted trip each -- the reload-ordering unit."""
    regions = []
    _collect_regions(tree, "root", regions)
    return regions


def _flag(ledger, violated, kind, is_undischarged):
    """Add every obligation of `kind` the walk found undischarged, without duplicating."""
    for obligation in ledger:
        if obligation.kind == kind and is_undischarged(obligation) and obligation not in violated:
            violated.append(obligation)


def _loads(body):
    """(index, inst, load) for every Load in one flat trip body."""
    for i, inst in enumerate(body):
        if isinstance(inst.op, Load):
            yield i, inst, inst.op


def _reads_registers(load):
    return load.src == Space.SHARED and load.dst == Space.REGISTER


def _refills_before_last_read(body):
    """Tokens whose shared refill is emitted before a read of the value it overwrites."""
    last_read, first_refill = {}, {}
    for i, inst, load in _loads(body):
        if _reads_registers(load):
            for token in getattr(load, "tokens", ()):
                last_read[token] = i
        elif load.dst == Space.SHARED and any(a.kind == "inplace-WAR" for a in inst.awaits):
            for token in getattr(load, "tokens", ()):
                first_refill.setdefault(token, i)
    return {token for token, at in first_refill.items()
            if last_read.get(token, len(body)) > at}


def _undischarged_war(ledger, tree, violated):
    """A rotation WAR is discharged only where the refill actually waits its last reader."""
    war_ops = {o.consumer.op for o in ledger
               if o.kind == "inplace-WAR" and o.consumer.at == "shared"}
    bad = set()
    for _label, body in _region_bodies(tree):
        bad |= _refills_before_last_read(body) & war_ops
    _flag(ledger, violated, "inplace-WAR",
          lambda o: o.consumer.op in bad and o.consumer.at == "shared")


def _read_before_its_copy(body, tokens):
    """Of `tokens`, those read out of LDS before the copy that fills them, within one trip."""
    read_first, bad = set(), set()
    for _i, _inst, load in _loads(body):
        if _reads_registers(load):
            read_first |= {t for t in getattr(load, "tokens", ()) if t in tokens}
        elif load.dst == Space.SHARED:
            bad |= {t for t in getattr(load, "tokens", ()) if t in read_first}
    return bad


def _undischarged_crossing(ledger, tree, violated):
    """A chunk-crossing RAW is discharged only inside the block that crosses."""
    crossing = {o.consumer.op for o in ledger if o.kind == "crossing-RAW"}
    bad = set()
    if crossing:
        for label, body in _region_bodies(tree):
            if "/steady" in label:  # the crossing is a steady-body property
                bad |= _read_before_its_copy(body, crossing)
    _flag(ledger, violated, "crossing-RAW", lambda o: o.consumer.op in bad)


def _slots_read_by_wmma(inst, slot_key):
    """The (operand, group) register slots one wmma consumes."""
    return [slot_key(opname, label)
            for opname, placement in (inst.placement or {}).items()
            for label, _expr in getattr(placement, "slots", ())]


def _slots_refilled_in_place(inst, slot_key):
    """The slots an in-place refill overwrites; empty if this instruction is not one."""
    if not (isinstance(inst.op, Load) and inst.op.dst == Space.REGISTER
            and getattr(inst.op, "advance", 0)
            and any(await_node.kind == "inplace-WAR" for await_node in inst.awaits)):
        return []
    return [slot_key(token, label)
            for token in getattr(inst.op, "tokens", ())
            for label, _expr in getattr(inst.placement, "slots", ())]


def _refills_above_their_consumer(body, inplace, slot_key):
    """Slots refilled while the wmma that still needs the old value has not run yet."""
    bad, consumed = set(), set()
    for inst in body:
        if isinstance(inst.op, Mma):
            consumed.update(_slots_read_by_wmma(inst, slot_key))
            continue
        for slot in _slots_refilled_in_place(inst, slot_key):
            if slot in inplace and slot not in consumed:
                bad.add(slot)
            consumed.discard(slot)
    return bad


def _undischarged_inplace(ledger, tree, violated):
    """An in-place refill WAR is discharged at the group's own read."""
    inplace = {(o.consumer.op, o.consumer.at) for o in ledger if o.kind == "inplace-WAR"}
    if not inplace:
        return
    groups = {}
    for obligation in ledger:
        if obligation.kind in ("inplace-WAR", "rotation-WAR") \
                and obligation.consumer.at != Space.SHARED:
            groups.setdefault(obligation.consumer.op, set()).add(obligation.consumer.at)

    def slot_key(opname, label):
        """An unlabelled slot names the operand's only group, when it has exactly one."""
        if label == "" and len(groups.get(opname, ())) == 1:
            return (opname, next(iter(groups[opname])))
        return (opname, label)

    bad = set()
    for _label, body in _region_bodies(tree):
        bad |= _refills_above_their_consumer(body, inplace, slot_key)
    _flag(ledger, violated, "inplace-WAR", lambda o: (o.consumer.op, o.consumer.at) in bad)


def _undischarged_readahead(ledger, tree, violated):
    """A read-ahead residency is discharged by the prologue read that pre-issues it."""
    ra_obls = [obligation for obligation in ledger if obligation.kind == "readahead-residency"]
    if ra_obls:
        prologue_reads = set()  # {(opname, ckey)} actually pre-issued
        for label, body in _region_bodies(tree):
            if not label.endswith("/prologue"):
                continue
            for inst in body:
                operand = inst.op
                if isinstance(operand, Load) and operand.src == Space.SHARED and operand.dst == Space.REGISTER:
                    prologue_reads.add((operand.tokens[0], tuple(sorted(operand.coord))))
        _flag(ledger, violated, "readahead-residency",
              lambda o: (o.producer.op, o.producer.coord) not in prologue_reads)


def check_ledger_discharged(ledger, tree):
    counters_awaited = {await_node.counter for await_node in all_awaits(tree)}
    violated = [obligation for obligation in ledger if obligation.counter not in counters_awaited]  # (a) coverage
    _undischarged_war(ledger, tree, violated)
    _undischarged_crossing(ledger, tree, violated)
    _undischarged_inplace(ledger, tree, violated)
    _undischarged_readahead(ledger, tree, violated)
    return violated


# --- validate --------------------------------------------------------------

#: `reg_group` restricts a read to the grouping-mode values its register group owns
#: owns move, -- a structural fact about the register partition, in the same class as:
_GENERIC_KINDS = {"peel_validity", "first_touch", "readahead_suppress", "short_step_validity",
                  "reg_group", "reload_wrap"}

_TRIP_SYMBOL = "T"


def _find(nodes, pred):
    """Depth-first: first node satisfying `pred`, descending Cond.then/els, Peel, Bind, Loop."""
    for node in nodes:
        if pred(node):
            return node
        for body in child_bodies(node):
            found = _find(body, pred)
            if found is not None:
                return found  # first match wins; the rest of the tree is not visited
    return None


def _placement_exprs(inst):
    pls = inst.placement
    if pls is None:
        return
    for placement in (pls.values() if isinstance(pls, dict) else [pls]):
        if placement is None:
            continue
        for _label, e in (placement.slots or ()):
            if hasattr(e, "free_vars"):
                yield e
        if getattr(placement, "src_slot", None) is not None and hasattr(placement.src_slot, "free_vars"):
            yield placement.src_slot


def _unbound_placement_refs(nodes, bound=frozenset(), out=None):
    """(inst, name) for every placement variable no enclosing node put in scope."""
    out = [] if out is None else out
    for node in nodes:
        if isinstance(node, Inst):
            for expr in _placement_exprs(node):
                for name in sorted(expr.free_vars() - set(bound) - {_TRIP_SYMBOL}):
                    out.append((node, name))
            continue
        axis = binds_axis(node)
        inner = bound | {axis} if axis else bound
        for body in child_bodies(node):
            _unbound_placement_refs(body, inner, out)
    return out


def _is_prologue(node):
    return isinstance(node, Peel) and node.kind == "prologue"


def _is_drain(node):
    return isinstance(node, Peel) and node.kind == "drain"


def _is_steady(node):
    return isinstance(node, Loop) and node.outer


def _check_pipelined_root(errs, root):
    """P1, when the root IS the peel-validity Cond: its guard and the order of its three arms."""
    if root.kind != "peel_validity":
        errs.append(f"P1: root Cond kind is {root.kind!r}, expected 'peel_validity'")
    if not (root.pred.lhs.var == "T" and root.pred.op == ">"):
        errs.append(f"P1: root guard is {root.pred.render()!r}, expected 'T > M' (strict: the "
                    f"steady region is post-tested and cannot express zero trips)")
    arms = [next((node for node in root.then if is_arm(node)), None)
            for is_arm in (_is_prologue, _is_steady, _is_drain)]
    if any(arm is None for arm in arms):
        errs.append("P1: root.then must contain [prologue Peel, steady Loop, drain Peel]")
    else:
        order = [root.then.index(arm) for arm in arms]
        if order != sorted(order):
            errs.append(f"P1: then order is {order}, expected prologue<steady<drain")
    if not root.els:
        errs.append("P1: root Cond has no `els` short-loop arm")


def _check_root_shape(errs, ir):
    """P1/P2: the root is the pipelined skeleton, and nothing runs outside its arms."""
    steady = _find(ir, _is_steady)
    if len(ir) == 1 and isinstance(ir[0], Cond):
        _check_pipelined_root(errs, ir[0])
    elif steady is not None and not any(isinstance(node, Cond) for node in ir):
        if any(_is_prologue(node) or _is_drain(node) for node in ir):
            errs.append("P1: prologue/drain at top level -- the pipeline must be wrapped in a "
                        "peel-validity Cond (nothing runs before the guard)")
    else:
        errs.append("P1: root is neither one peel-validity Cond nor a bare steady loop")

    for node in ir:                       # one error per stray Inst, as before
        if isinstance(node, Inst):
            errs.append("P2: a top-level Inst runs before the peel-validity guard")
    return steady


def _check_drain(errs, ir, iter_name):
    """P3/P4: the drain is M straight-line steps, bound to iter, with no residue pin."""
    # --- Property 3 & 4: drain = M straight-line Bind steps, iter bound, no residue pin -------
    drain = _find(ir, _is_drain)
    if drain is not None:
        if not all(isinstance(step, Bind) for step in drain.body):
            errs.append("P4: drain steps must each be a Bind(iter=T-M+t), not a Cond/Branch pin")
        for step in drain.body:
            if isinstance(step, Bind) and step.axis != iter_name:
                errs.append(f"P3: drain Bind pins {step.axis!r}, expected {iter_name!r}")
        if _find(drain.body, lambda node: isinstance(node, Cond) and getattr(node.pred.lhs, "mod", 0)
                 and node.pred.lhs.var == iter_name):
            errs.append("P4: drain contains an iter%d residue-pin Cond (should be a Bind)")
    prologue = _find(ir, _is_prologue)
    if prologue is not None:
        for node in prologue.body:
            if isinstance(node, Bind) and node.axis != iter_name:
                errs.append(f"P3: prologue Bind pins {node.axis!r}, expected {iter_name!r}")

    for inst, name in _unbound_placement_refs(ir):
        tok = getattr(inst.op, "tokens", None) or "wmma"
        why = f" (the peel's induction -- needs a Bind)" if name == iter_name else ""
        errs.append(f"P3/P5: {tok} placement references UNBOUND mode {name!r}{why}")


def _check_slots(errs, ir):
    """P5: every register instruction names a group and a concrete pinned slot."""
    # --- Property 5: register group + concrete pinned slot -----------------------------------
    for inst in walk_insts(ir):
        operand = inst.op
        if isinstance(operand, Load) and operand.dst == Space.REGISTER:
            placement = inst.placement
            if placement is None or not placement.slots:
                errs.append(f"P5: register read {operand.tokens} carries no placement slot")
            elif any(lbl is None for lbl, _expr in placement.slots):
                errs.append(f"P5: register read {operand.tokens} has a slot with no register GROUP")
        elif isinstance(operand, Mma):
            pls = inst.placement or {}
            if not pls:
                errs.append("P5: wmma carries no per-operand register placement")


def _check_readahead_order(errs, steady):
    """P6: a read-ahead leads its wmma, and sigma_c puts refill copies last."""
    # --- Property 6: read-ahead leads wmma; sigma_c refill copies last (steady body) --------------
    if steady is not None:
        for body in steady.bodies:
            def _is_copy(node):
                axis = _movement(node)
                return (axis is not None and isinstance(axis.op, Load) and axis.op.dst == Space.SHARED
                        and any(await_node.kind == "inplace-WAR" for await_node in axis.awaits))
            seen_copy = False
            for node in body:
                if _is_copy(node):
                    seen_copy = True
                elif seen_copy and isinstance(node, (Loop, Cond, Bind, Inst)):
                    def _is_read_or_wmma(item):
                        return isinstance(item, Inst) and (
                            isinstance(item.op, Mma)
                            or (isinstance(item.op, Load) and item.op.dst == Space.REGISTER))
                    if _find([node], _is_read_or_wmma):
                        errs.append("P6: a read/wmma follows a sigma_c refill copy in the steady body")
                        break


def _check_ledger(errs, ir, theta, depths, undischarged):
    """P7: the ledger is discharged."""
    # --- Property 7: ledger empty ------------------------------------------------------------
    if undischarged is None:
        if depths is None:
            depths, _ = build_S(theta)
        undischarged = check_ledger_discharged(build_ledger(theta, depths), ir)
    if undischarged:
        def _end(e):
            step = "%s@%s" % (getattr(e, "op", "?"), getattr(e, "at", "?"))
            if getattr(e, "role", ""):
                step += ":" + e.role
            if getattr(e, "coord", ()):
                step += " " + ",".join("%s%s" % (axis, value) for axis, value in e.coord)
            return step

        def _render_obligation(obligation):
            return "%-14s %s  ->  %s   [counter %s]" % (
                getattr(obligation, "kind", "?"), _end(obligation.producer), _end(obligation.consumer),
                getattr(obligation, "counter", "?"))

        errs.append("P7: obligation ledger not empty: %d undischarged\n       %s"
                    % (len(undischarged),
                       "\n       ".join(_render_obligation(obligation) for obligation in undischarged)))


def _check_cond_kinds(errs, ir):
    """P8: Cond kinds are generic -- no scaffold labels in the core."""
    # --- Property 8: generic Cond kinds, no scaffold labels in the core ----------------------
    def _check_kinds(nodes):
        for node in nodes:
            if isinstance(node, Cond):
                if node.kind and node.kind not in _GENERIC_KINDS:
                    errs.append(f"P8: Cond has non-generic kind {node.kind!r}")
                if node.label:
                    errs.append(f"P8: core Cond carries a scaffold label {node.label!r} (should be set in GIR)")
                _check_kinds(node.then)
                _check_kinds(node.els)
            elif isinstance(node, (Peel, Bind)):
                _check_kinds(node.body)
            elif isinstance(node, Loop):
                for body in node.bodies:
                    _check_kinds(body)
    _check_kinds(ir)


def _check_short_loop(errs, ir):
    """P9: every short-loop step past the first is guarded by a trip compare."""
    # --- Property 9: every short-loop step past the first is guarded by `T > t` ---------------
    if len(ir) == 1 and isinstance(ir[0], Cond) and ir[0].els:
        for t, step in enumerate(ir[0].els):
            if not t:
                continue
            if not (isinstance(step, Cond) and step.kind == "short_step_validity"
                    and step.pred.lhs.var == _TRIP_SYMBOL and step.pred.op == ">"
                    and step.pred.rhs == t):
                errs.append(f"P9: short-loop step {t} is not guarded by "
                            f"'{_TRIP_SYMBOL} > {t}' -- it would read an out-of-bounds chunk")


def _slot_domain(theta, samples=8):
    """`{name: count}` -- how many values each name takes when probing a slot expression."""
    domain = {axis.name: min(samples, max(1, int(axis.extent))) for axis in theta.inner_axes()}
    chunk = theta.summation_chunk_name()
    if chunk:
        domain[chunk] = samples       # a few chunks, whatever the axis itself declares
    return domain


def _slot_values(expression, domain):
    """Every slot `expression` resolves to, over ITS OWN names.

    A name the expression does not reference cannot change its value, so walking the whole inner
    product would ask the same question once per combination of names it ignores.
    """
    names = [name for name in expression.free_vars() if name in domain]
    out = set()
    for point in _iproduct(*[range(domain[name]) for name in names]):
        try:
            out.add(int(expression.eval(dict(zip(names, point)))))
        except Exception:
            pass  # a slot that will not settle here says nothing about the ring
    return out


def _collect_slots(into, name, placement):
    """File this placement's slot EXPRESSIONS under `(name, group)`, deduplicated."""
    for label, expression in getattr(placement, "slots", ()) or ():
        into.setdefault((name, label or "g0"), set()).add(expression)


def _check_dead_slots(_errs, ir, theta):
    """Every slot the ring declares is written by a read and sourced by a wmma.

    NOT A REJECTION -- this RAISES.  A request past an operand's own outer axis is refused at the
    solution, so by the time a tree exists every declared slot must be reachable.  One that is
    not means the depth and the traversal were derived from different things, which is a defect
    in the model, not a configuration a caller can choose differently.
    """
    # Gather the EXPRESSIONS first: a tree names a handful of distinct ones over hundreds of
    # instructions, and resolving per instruction asks each the same question again.
    domain = _slot_domain(theta)
    written, consumed = {}, {}
    for inst in walk_insts(ir):
        operand = inst.op
        if isinstance(operand, Load) and operand.dst == Space.REGISTER:
            _collect_slots(written, operand.tokens[0], inst.placement)
        elif isinstance(operand, Mma) and isinstance(inst.placement, dict):
            for name, placement in inst.placement.items():
                _collect_slots(consumed, name, placement)
    written = {key: set().union(*(_slot_values(e, domain) for e in exprs)) if exprs else set()
               for key, exprs in written.items()}
    consumed = {key: set().union(*(_slot_values(e, domain) for e in exprs)) if exprs else set()
                for key, exprs in consumed.items()}
    from .traversal import _region_count
    for operand in theta.operands:
        depths = getattr(getattr(operand, "fragment", None), "ring_depths", None) or {}
        # Slots the emitter addresses: `group_ring_depth` gives each region its own turn.
        regions = max(1, int(_region_count(theta, operand)))
        for label, depth in depths.items():
            declared = set(range(int(depth) * regions))
            key = (operand.name, label)
            for what, seen in (("written by a read", written.get(key, set())),
                               ("read by a wmma", consumed.get(key, set()))):
                missing = sorted(declared - seen)
                if missing:
                    from . import traversal as _g
                    def _safe(fn, *a):
                        try:
                            return fn(*a)
                        except Exception as error:
                            return "!%s" % type(error).__name__
                    # The depth decision's own inputs, not a summary of them.
                    facts = ("requested_read_ahead=%s reloads_whole=%s outer_role=%s varying=%s "
                             "read_coverage=%s nest=%s ord=%s"
                             % (_safe(_g.requested_read_ahead, theta, operand),
                                _safe(_g.reloads_whole_set, theta, operand),
                                _safe(_g.operand_outer_role, theta, operand),
                                _safe(_g.varying_axes, theta, operand),
                                _safe(_g.read_coverage, theta, operand),
                                [(a.name, int(a.extent)) for a in theta.inner_axes()],
                                [a.name for a in theta.ord]))
                    raise ValueError(
                        "register ring over-allocated: %s:%s declares depth %d but slot(s) %s "
                        "are never %s.  The solution refuses a read-ahead past this operand's "
                        "own outer axis, so a slot nothing reaches means the ring depth and the "
                        "traversal disagree -- fix the derivation, not the kernel.  [%s]"
                        % (operand.name, label, depth, missing, what, facts))


def validate_loopir(theta, ir, depths=None, undischarged=None):
    errs = []
    iter_name = theta.summation_chunk_name("iter")
    steady = _check_root_shape(errs, ir)
    _check_drain(errs, ir, iter_name)
    _check_slots(errs, ir)
    _check_readahead_order(errs, steady)
    _check_ledger(errs, ir, theta, depths, undischarged)
    _check_cond_kinds(errs, ir)
    _check_short_loop(errs, ir)
    _check_dead_slots(errs, ir, theta)
    return errs


def agent_assignment_consistency(theta) -> list:
    """Validate wave-level assignment extents."""
    out = []
    assignment = theta.agent_assignment
    if not isinstance(assignment, AgentAssignment):
        return ["agent_assignment has type %s" % type(assignment).__name__]

    wave = assignment.coarser_than("wave")
    if wave:
        prod = 1
        for role in wave:
            prod *= max(1, int(role.extent))
        n_agents = max(1, int(theta.wave_count))
        if prod > n_agents or n_agents % prod:
            out.append("wave-level agent extents %d do not divide wave_count %d"
                       % (prod, n_agents))
    return out
