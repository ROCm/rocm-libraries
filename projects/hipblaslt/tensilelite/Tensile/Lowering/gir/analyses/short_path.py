# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ShortPathFold -- may the `T < M` arm be FOLDED into the shared prologue/drain path?
THE TWO SHAPES. TensileLite's scaffold and the LoopIR describe the same kernel differently.

"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..nodes import Move, Mma, Mark
from ..analysis import Analysis


FOLD, SPLIT, UNSOUND, NA, VACUOUS = "fold", "split", "unsound", "n/a", "vacuous"

# Why a def-use pairing changed.  Both are breaks; they have different causes and different fixes,
# and collapsing them was what let "the coverage is legal iff S >= M" look like the whole story.
UNWRITTEN = "unwritten"   # the folded path reads a generation NOTHING on it wrote -- the prologue's
                          # generations are ABSOLUTE while the drain's are relative to a loop that
                          # ran zero times, so the two frames do not meet.
OVERWRITTEN = "overwritten"  # a different producer got there first -- the ring is too shallow for
                             # the separated order to keep every peeled chunk live.
ARM_UNWRITTEN = "arm-unwritten"  # the SHORT ARM reads something IT never wrote; the folded path is
                                 # not at fault, the arm is (see FoldVerdict.unbound).


# ===========================================================================
# the two sequences
def _inst_kind(inst):
    """'copy' | 'read' | 'mma' -- from the destination SPACE, no name test."""
    if isinstance(inst, Mma):
        return "mma"
    if isinstance(inst, Move):
        return "copy" if any(d.tile.space == "shared" for d in inst.dsts) else "read"
    return None


def _operands(inst):
    """The operand tuple an instruction names on its destination -- a Phi-fused copy names several."""
    if isinstance(inst, Mma):
        return tuple(s.tile.operand for s in inst.srcs)
    return tuple(d.tile.operand for d in inst.dsts)


def _coord(inst):
    if isinstance(inst, Mma):
        return inst.coord
    return inst.dsts[0].tile.coord + tuple(("@issue:" + name, value)
                                           for name, value in getattr(inst, "issue", ()))


def _groups(inst):
    """The register groups an instruction touches -- part of its identity, not decoration.

 A SPLIT read (`emit._read_groups`: one Inst per register-width class, -- mxfp8's `lo`
 rotates a 2-ring while `hi` is in-place) puts two instructions at the SAME (operand, coord),
 distinguished only by which groups they fill. Without this the correspondence is ambiguous and
 `_trace` raises -- which is how this was found, rather than by silently pairing the wrong two."""
    refs = inst.srcs if isinstance(inst, Mma) else inst.dsts
    return tuple(r.group for r in refs if r.tile.space == "register")


def _entry_gens(prog):
    """{gen.id: entry_val} from the loop headers' phis.

    On the folded path the steady loop takes ZERO trips (that is what `T < M` means), so every
    loop-carried generation still holds its phi's entry value and a drain Ref's `gdelta` resolves
    against that.  This is the one place the folded path's semantics enter the resolution, and it
    is a consequence of the branch being taken, not a convention."""
    out = {}
    for blk in prog.blocks.values():
        if not blk.loop:
            continue
        for phi in blk.phis:
            if phi.entry_val is not None:
                out[phi.gen.id] = phi.entry_val
    return out


def _frame_base(blk):
    """The CHUNK POSITION a block's `gdelta = 0` is stated against, ON THE FOLDED PATH."""
    if blk.gen_rel is None:
        return 0                       # absolute-generation block (prologue, short arm): no frame
    return blk.chunk_base - 1 - int(blk.gen_rel)


def _shared_generation(ref, entry, base=0):
    """The concrete shared generation a Ref names, or None if it names no shared residence.

    `base` is the block's folded-path chunk position (`_frame_base`); `gdelta` is an offset from
    it, so the chunk named is `base + gdelta` and the generation is that chunk mod the ring."""
    if ref.abs_gen is not None:
        return int(ref.abs_gen)
    if ref.gen is not None:
        return (entry.get(ref.gen.id, 0) + base + ref.gdelta) % max(1, ref.gen.ring)
    return None


from ..nodes import CondGoto


def _defs_uses(inst, entry, free_axes, base=0):
    """(defs, uses) as LOCATION keys -- where a value is handed over:

        ('shared',   operand, generation)
        ('register', operand, group, slot, unit index)
    """
    defs, uses = [], []
    if isinstance(inst, Mma):
        for s in inst.srcs:
            if s.tile.space == "register":
                uses.append(("register", s.tile.operand, s.group, s.slot,
                             s.unit_index))
        return defs, uses
    for s in inst.srcs:
        if s.tile.space == "shared":
            g = _shared_generation(s, entry, base)
            if g is not None:
                uses.append(("shared", s.tile.operand, g))
    for d in inst.dsts:
        if d.tile.space == "shared":
            g = _shared_generation(d, entry, base)
            if g is not None:
                defs.append(("shared", d.tile.operand, g))
        elif d.tile.space == "register":
            units = tuple(d.unit_indexes or (d.unit_index,))
            defs.extend(("register", d.tile.operand, d.group, d.slot, unit)
                        for unit in dict.fromkeys(units))
    return defs, uses


ENTRY = ("<entry>",)          # a location this arm never writes

_UNSERVED = (ENTRY, ("<absent>",), None)   # the three spellings of "no producer reached here"


def pairing_cause(f_src, s_src):
    """Classify one corresponded consumer's producer pair, or `None` when the two frames agree."""
    if f_src == s_src and s_src not in _UNSERVED:
        return None
    if s_src in _UNSERVED and f_src in _UNSERVED:
        return UNWRITTEN          # neither frame serves it -- folding is not a rescue
    if s_src in _UNSERVED:
        return ARM_UNWRITTEN      # the arm's own defect; the folded path is not at fault
    if f_src in _UNSERVED:
        return UNWRITTEN          # the two frames do not meet
    return OVERWRITTEN            # ring too shallow for the separated order


def _trace(prog, labels, entry, step_of, free_axes):
    """One forward walk of an arm.

    `reach`  {consumer key: {location: the VALUE that reaches it}}
    `keys`   the consumer keys this arm defines
    `extras` instructions with no consumer key (a prologue fill; the drain's next-tile prefetch)
    `providers` names the concrete instruction providing each reached value.  It is deliberately
    separate from `reach`: values prove equivalence; providers tell the fold transformation which
    early write caused a failed proof.
    """
    last, last_provider, reach, providers, nodes, keys = {}, {}, {}, {}, {}, set()
    copy_seen, extras = {}, 0
    for lab in labels:
        blk = prog.blocks[lab]
        step = step_of(lab)
        base = _frame_base(blk)
        for inst in blk.body:
            if isinstance(inst, Mark):
                continue
            k = _inst_kind(inst)
            if k is None:
                continue
            ops, crd = _operands(inst), _coord(inst)
            defs, uses = _defs_uses(inst, entry, free_axes, base)
            key = None if (k == "copy" or step is None) else (k, step, ops, crd, _groups(inst))
            if key is not None:
                if key in keys:
                    raise RuntimeError(
                        f"short-path correspondence is ambiguous: two instructions share the key "
                        f"{key!r}.  It must name one instruction per arm.")
                keys.add(key)
                if uses:
                    reach[key] = {u: last.get(u, ENTRY) for u in uses}
                    providers[key] = {u: last_provider.get(u) for u in uses}
                nodes[key] = (lab, inst)
            else:
                extras += 1
            if k == "copy":
                n = copy_seen.get((ops, crd), 0)
                copy_seen[(ops, crd)] = n + 1
                value = ("copy", ops, crd, n)
            elif k == "read":
                src = next((u for u in uses if u[0] == "shared"), None)
                value = ("read", ops, crd, last.get(src, ENTRY) if src else None)
            else:
                value = None                      # an Mma defines nothing this analysis tracks
            for d in defs:
                last[d] = value
                last_provider[d] = (lab, inst)
    return reach, keys, extras, providers, nodes


# ===========================================================================
# the verdict
@dataclass(frozen=True)
class Break:
    """One consumer whose value changed: it reads `location`, produced by `folded` on the folded
    path but by `short` in the short arm.  `cause` is one of UNWRITTEN / OVERWRITTEN /
    ARM_UNWRITTEN -- the three have different root causes and different owners."""
    consumer: object
    location: object
    folded:   object
    short:    object
    cause:    str = OVERWRITTEN


@dataclass(frozen=True)
class Ordering:
    """One early folded-path register write and the last old-value consumer it must follow."""
    block:    str
    producer: object
    consumer: object


@dataclass(frozen=True)
class FoldVerdict:
    verdict:  str                                  # FOLD | SPLIT | UNSOUND | NA | VACUOUS
    reason:   str = ""
    breaks:   tuple = ()                           # (Break, ...)  -- F2
    missing:  tuple = ()                           # short-arm keys absent from the folded path -- F1
    unbound:  tuple = ()                           # short-arm consumers with no def in the arm -- F0
    extras:   int = 0                              # folded-path instructions with no counterpart
    obligations: tuple = ()                        # preconditions the CONSUMER must satisfy
    folded_blocks: tuple = ()
    short_blocks:  tuple = ()
    ordering: tuple = ()                           # body ordering needed to realize this fold

    def causes(self) -> dict:
        out = {}
        for b in self.breaks:
            out[b.cause] = out.get(b.cause, 0) + 1
        return out


def _compare_shapes(f_reach, f_keys, s_reach, s_keys):
    """What the short arm needs that the folded path cannot supply.

    `missing` -- a key the arm reads and the folded path never writes.
    `unbound` -- a consumer with no producer at all.
    `breaks`  -- a key both write, but from different sources.
    """
    missing = tuple(sorted((k for k in s_keys
                            if k[0] == "mma" and k not in f_keys), key=repr))

    unbound = tuple(sorted(
        ((k, loc) for k, m in s_reach.items() for loc, src in m.items()
         if src == ENTRY), key=repr))

    # F2 -- pairing: every corresponded consumer must receive the same producer's value.
    breaks = []
    for k, s_map in s_reach.items():
        # F2 IS ASKED AT THE `wmma` ONLY.  A `wmma` is the sole instruction whose semantics the
        # two shapes must agree on; a read is a MEANS, and its identity is not stable across
        if not (isinstance(k, tuple) and k and k[0] == "mma"):
            continue
        f_map = f_reach.get(k)
        if f_map is None:
            continue                              # already reported by F1
        for loc, s_src in s_map.items():
            # the LOCATION differs between the frames, so compare the PRODUCERS, matching the
            # short arm's location to whatever the folded path used for the same consumer.
            f_src = _folded_producer(f_map, loc, s_src)
            cause = pairing_cause(f_src, s_src)
            if cause is None:
                continue
            breaks.append(Break(k, loc, f_src, s_src, cause))
    breaks = tuple(breaks)
    fold_breaks = tuple(b for b in breaks if b.cause != ARM_UNWRITTEN)
    return missing, unbound, breaks, fold_breaks


def _provider_for(f_map, loc):
    """The concrete folded provider corresponding to the short arm's operand location."""
    if loc in f_map:
        return f_map[loc]
    same = [v for k, v in f_map.items() if k[0] == loc[0] and k[1] == loc[1]]
    return same[0] if len(same) == 1 else None


def _ordering_constraints(prog, fold_breaks, providers, nodes):
    """Translate failed value pairs into exact in-block ordering constraints.

    A failed pair already proves which folded write reached an MMA in place of the short arm's
    value.  When that write and consumer are in one block, delaying the write is the structural
    merge required by the emitted scaffold.  Ring-depth failures cross blocks and intentionally
    produce no such constraint.
    """
    latest = {}
    for brk in fold_breaks:
        provider = _provider_for(providers.get(brk.consumer, {}), brk.location)
        consumer = nodes.get(brk.consumer)
        if provider is None or consumer is None or provider[0] != consumer[0]:
            continue
        label, producer = provider
        _consumer_label, consumer_inst = consumer
        if _inst_kind(producer) != "read":
            continue
        body = prog.blocks[label].body
        positions = {id(node): pos for pos, node in enumerate(body)}
        producer_pos = positions.get(id(producer))
        consumer_pos = positions.get(id(consumer_inst))
        if producer_pos is None or consumer_pos is None:
            continue
        if producer_pos >= consumer_pos:
            continue
        prior = latest.get(id(producer))
        if prior is None or positions[id(prior.consumer)] < consumer_pos:
            latest[id(producer)] = Ordering(label, producer, consumer_inst)

    def _position(item):
        body = prog.blocks[item.block].body
        return item.block, next(pos for pos, node in enumerate(body) if node is item.producer)

    return tuple(sorted(
        latest.values(),
        key=_position))


def _fold_verdict(prog, folded_labels, short_labels, s_keys, extras,
                  missing, unbound, breaks, fold_breaks):
    """FOLD, SPLIT, VACUOUS or UNSOUND -- and the reason, reported verbatim by the caller."""
    obligations = (
        ("the folded drain chain must be enterable at step M-T, not only at step 0 -- the short "
         "arm's `T > t` guards have no counterpart on a single-entry chain (G3/; "
         "TensileLite's NoGlobalLoadLoop_k selection supplies this today)"),)

    # F0 first: an arm that cannot stand alone is a DEFECT, not a shape to choose between.
    if unbound:
        return FoldVerdict(
            UNSOUND,
            f"the short arm cannot stand alone: {len(unbound)} consumer(s) in it read a "
            f"location the arm never writes (its read-ahead has no warm-up).  The folded path "
            f"{'also breaks' if fold_breaks else 'supplies those values'}, so "
            f"{'NEITHER shape' if fold_breaks else 'only the folded shape'} serves T < M "
            f"as emitted",
            breaks, missing, unbound, extras, obligations,
            folded_labels, short_labels)

    if _arm_is_unreachable(prog):
        # The arm is dead code: no legal trip count satisfies its guard.  Do not grow the
        # kernel for it, and do not let a (real, but moot) pairing difference read as a live
        return FoldVerdict(
            VACUOUS,
            f"no legal trip count reaches the arm; {len(fold_breaks)} pairing(s) differ but "
            f"on a path that cannot execute",
            breaks, missing, unbound, extras, obligations,
            folded_labels, short_labels)

    if missing or fold_breaks:
        return FoldVerdict(
            SPLIT,
            f"{len(fold_breaks)} def-use pairing(s) change and {len(missing)} instruction(s) "
            f"are absent under the folded order "
            f"({sorted((c, n) for c, n in FoldVerdict(SPLIT, breaks=fold_breaks).causes().items())}) "
            f"-- merge the short arm's value order into the emitted scaffold",
            breaks, missing, unbound, extras, obligations,
            folded_labels, short_labels)
    return FoldVerdict(FOLD,
                       f"every def-use pairing survives the reordering "
                       f"({len(s_keys)} corresponded instructions, {extras} extra on the "
                       f"folded path)",
                       (), (), unbound, extras, obligations,
                       folded_labels, short_labels)


class ShortPathFold(Analysis):
    """See module docstring.  Pure; returns a `FoldVerdict`."""

    def run(self, prog, am):
        short_labels = tuple(lab for lab, b in prog.blocks.items()
                             if str(b.phase).startswith("short"))
        if not short_labels:
            return FoldVerdict(NA, "no short arm in this Program (M == 0, or already folded)")

        # the folded path: the prologue, then the drain chain in peel order.  Derived from the
        # blocks' own frame (`chunk_base`), not from a name sort -- 'drain10' must not sort before
        pro = prog.entry
        drains = sorted((lab for lab, b in prog.blocks.items()
                         if str(b.phase).startswith("drain")),
                        key=lambda l: prog.blocks[l].chunk_base)
        if not drains:
            return FoldVerdict(SPLIT, "no drain chain to fold into",
                               short_blocks=short_labels)
        folded_labels = (pro,) + tuple(drains)
        short_labels = tuple(sorted(short_labels, key=lambda l: prog.blocks[l].chunk_base))

        drain_step = {lab: i for i, lab in enumerate(drains)}
        short_step = {lab: i for i, lab in enumerate(short_labels)}
        entry = _entry_gens(prog)

        free_axes = prog.meta.get("free_axes", {})
        # An "extra" is an instruction with no CONSUMER key.  It is not thereby uncorresponded.
        f_reach, f_keys, extras, f_providers, f_nodes = _trace(
            prog, folded_labels, entry, drain_step.get, free_axes)
        s_reach, s_keys, _, _s_providers, _s_nodes = _trace(
            prog, short_labels, entry, short_step.get, free_axes)

        # F1 -- coverage: every COMPUTE the short arm performs must also happen on the folded path.
        # Restricted to `wmma` for the same reason F2 is (below): the two shapes deliberately read
        missing, unbound, breaks, fold_breaks = _compare_shapes(f_reach, f_keys,
                                                               s_reach, s_keys)
        verdict = _fold_verdict(prog, folded_labels, short_labels, s_keys, extras,
                                missing, unbound, breaks, fold_breaks)
        if verdict.verdict == SPLIT:
            verdict = replace(
                verdict,
                ordering=_ordering_constraints(prog, fold_breaks, f_providers, f_nodes))
        return verdict


def _folded_producer(f_map, loc, s_src):
    """The VALUE this consumer receives at `loc` on the FOLDED path (see `_trace`)."""
    if loc in f_map:
        return f_map[loc]
    same = [v for k, v in f_map.items() if k[0] == loc[0] and k[1] == loc[1]]
    if len(same) == 1:
        return same[0]
    if not same:
        return ("<absent>",)
    raise RuntimeError(
        f"consumer reads {len(same)} locations on space {loc[0]!r} for operand {loc[1]!r}; the "
        f"short-arm location {loc!r} cannot be matched to one of them without a role")


def _arm_is_unreachable(prog):
    """Is the `T < ...` arm dead code?  DERIVED from the peel-validity guard, not from M."""
    entry = prog.blocks.get(prog.entry)
    term = entry.term if entry is not None else None
    if not isinstance(term, CondGoto):
        return False
    p = term.pred
    if p.lhs != "T" or p.rhs.var:
        return False
    m = p.rhs.const
    # loop entered iff `T > m` (strict) or `T >= m`; the arm takes the complement.
    highest_T_reaching_arm = m if p.op == ">" else m - 1
    return highest_T_reaching_arm < 1          # no T >= 1 reaches the arm
