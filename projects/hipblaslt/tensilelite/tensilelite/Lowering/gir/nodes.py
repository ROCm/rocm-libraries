# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GIR nodes -- the GEMM-dataflow IR (layer 2): what a program is made of.

Four sections, in dependency order: the nouns (Tile/Gen/Ref), the verbs (Move/Mma/Mark), the
control flow (Pred/terminators/Block/Program), and the two things layered directly on those --
the Ref accessors every hop-shaped question goes through, and the Region/PendingMark model the
analyses use to propose a Mark before a pass places it.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ===========================================================================
# 4.2  Tile, Gen, Ref -- the nouns
@dataclass(frozen=True)
class Tile:
    """A logical operand fragment -- 'what data', no color / no address.

 operand -- 'A'|'B'|'C'|'acc'|'scaleA'|'scaleB'|'meta'|... (open set; identity of an Mma
 operand is read from here, -- there is no separate `role` map).
 space -- 'global'|'shared'|'register'.
    """
    operand: str
    space:   str
    coord:   tuple = ()          # ((axis, val|None), ...) -- hashable form of the coord dict
    shape:   tuple = ()          # (("regs", n), ("bytes", n), ...) -- hashable size record


@dataclass(frozen=True)
class Gen:
    """A loop-carried GENERATION -- the ONE SSA value class in GIR.

 Defined by a Phi in a loop header; the back-edge transfer is v' = (v + adv) % ring.
 `ring` is the buffer ring size S (1 = in-place, the generation never changes).
    """
    id:   int
    ring: int


@dataclass(frozen=True)
class Ref:
    """A USE of a Tile inside an instruction.

    A `Ref` names EITHER a shared-staged residence (a `gen`(+`gdelta`), loop-carried) OR a
    register residence (a concrete (group, slot, unit index)), plus its size.
    """
    tile:     Tile
    # -- shared-staged residence (loop-carried) --
    gen:      object = None      # a Gen, or None
    gdelta:   int = 0            # concrete read-ahead offset from `gen` (0, 1, ...)
    # -- absolute (non-loop-carried) generation --
    # prologue fills and drain reads have CONCRETE generations pinned by the LoopIR peel
    abs_gen:  object = None      # concrete generation int for a straight-line (peel) shared ref
    # -- register residence (concrete after enumeration;)
    group:    object = None      # register group INDEX (0..VgprPartition-1); None if not a reg use
    slot:     object = None      # concrete rotation slot within that group
    reg_ring: object = None      # register rotation WIDTH W for this group -- a LoopIR fact GIR
                                 # CONSUMES : read from the LoopIR slot Expr.mod, never
                                 # decided in GIR; reg_band only validates it.
    unit_index: int = 0          # dense tile offset inside this group's rotation unit
    unit_indexes: tuple = ()     # every scalar unit a covered register write defines
    issue_axes: tuple = ()       # invariant axes that delimit re-read residency events
    covers:   tuple = ()
    # -- size --
    size_regs:  int = 0
    size_bytes: int = 0


# ===========================================================================
# 4.3  Move and Mma -- the verbs (unified srcs/dsts; semantics from Tile.operand)
@dataclass
class Move:
    """THE data-movement verb: one hop down the memory hierarchy.

 srcs -- (Ref,) e.g. a global tile.
 dsts -- (Ref,) e.g. a shared tile at a Gen (a copy), or a register (a read).
 deps -- RAW + WAR edges carried from the LoopIR awaits (kind on each dep, / analysis).
    """
    srcs: tuple
    dsts: tuple
    deps: tuple = ()
    advance: int = 0
    issue: tuple = ()            # invariant-axis pass that owns this residency event
    # The movement's CANONICAL members.  A copy may carry refs for only the members present on
    # this region, but it is still one movement; readers key on identity, so it is stated here
    # rather than re-derived from whichever refs survived.
    unit: tuple = ()


@dataclass
class Mma:
    """THE compute verb -- OPEN operand set via srcs/dsts.

 Operand identity (A vs B vs scaleA vs meta vs C) is read from each `Ref.tile.operand`;
 there is no `role` vocabulary. plain=(A,B); MX=(A,B,scaleA,scaleB); sparse=(A,B,meta);
 fused=(A,B,C). `dsts` is the accumulator ('acc').
 """
    srcs:  tuple
    dsts:  tuple
    coord: tuple = ()            # ((axis, val), ...) over the 6 axes
    block: str = ""             # hardware wmma block group
    deps:  tuple = ()


# ===========================================================================
# 4.4  Mark -- a semantic point (+ the region-based insertion model)
MARK_KINDS = (
    "phase_boundary", "fence", "swap", "gr_increment", "region_increment",
    "descriptor_enable", "gsu_guard", "gl2_prefetch", "chunk_pin", "buffer_pin",
)


@dataclass
class Mark:
    """A body element carrying NO data movement or compute -- only a GEMM-semantic FACT at a
    program point for L3 to realize (R-SEMANTIC).  `Mark('swap', {hop, gen_from, gen_to})` is
    the FACT "the read generation changes here", NOT a v_xor.

    kind -- one of MARK_KINDS (native from LoopIR structure, or produced by an analysis).
    """
    kind: str
    at:   dict = field(default_factory=dict)


# ===========================================================================
# 4.5  Control-flow primitives (the CFG terminators)
_PRED_OPS = (">=", ">", "==", "<", "<=", "!=")


@dataclass(frozen=True)
class Bound:
    """The right-hand side of a terminator predicate: the constant `const` when `var` is empty,
    else the trip-count symbol `var` offset by `const` (negative for `T - M`).
    """
    var:   str = ""
    const: int = 0

    def render(self) -> str:
        if not self.var:
            return str(self.const)
        if self.const:
            return f"{self.var} {'-' if self.const < 0 else '+'} {abs(self.const)}"
        return self.var


@dataclass(frozen=True)
class Pred:
    """A STRUCTURED symbolic predicate over trip-count symbols (no hardware): `lhs op rhs`."""
    lhs:   str
    op:    str
    rhs:   Bound = field(default_factory=Bound)
    label: str = ""

    def __post_init__(self):
        if self.op not in _PRED_OPS:
            raise ValueError(f"Pred.op {self.op!r} not in {list(_PRED_OPS)}")

    def render(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs.render()}"


@dataclass(frozen=True)
class Goto:
    """Unconditional terminator."""
    target: str


@dataclass(frozen=True)
class CondGoto:
    """Conditional terminator -- the peel-validity guard AND the loop back-edge.

 There is no separate loop-back primitive : the steady back-edge is just a CondGoto
 whose taken target dominates it. "Is this a back-edge?" is a graph property answered by
 back_edges(prog) over the CFG, not by a distinct node type.
 """
    pred:     Pred
    t_target: str                # taken block
    f_target: str                # fall-through block


@dataclass(frozen=True)
class Trips:
    """A loop's DYNAMIC TRIP COUNT: `(var - sub) // div`, or the constant `-sub` when `var` is ""."""
    var: str = "T"
    sub: int = 0
    div: int = 1

    def render(self) -> str:
        base = self.var if not self.sub else (
            f"{self.var} - {self.sub}" if self.var else str(-self.sub))
        return base if self.div == 1 else f"({base}) // {self.div}"


@dataclass(frozen=True)
class LoopBack:
    """Loop terminator: run `body` for `trips` iterations, then go to `exit_target`."""
    trips:       Trips
    body:        str             # the block the back edge targets (the loop header)
    exit_target: str
    label:       str = ""


@dataclass(frozen=True)
class LoopCopy:
    """One copy in a statically unrolled counted loop.

    The emitted scaffold decrements/tests after every copy. `index` is zero-based, `copies` is
    the unroll width, `next_target` is the next copy (or the header on the last copy), and
    `exit_target` is the common drain/exit. `trips` counts chunks, not groups of `copies`.
    """
    trips:       Trips
    index:       int
    copies:      int
    next_target: str
    exit_target: str
    label:       str = ""


@dataclass(frozen=True)
class Return:
    """Function-early-exit terminator -- a SINK with no successors."""


@dataclass(frozen=True)
class CondChain:
    """Multi-exit terminator -- an ORDERED chain of guarded exits, then a default."""
    arms:    tuple               # ((Pred, target), ...) -- ORDERED; first satisfied arm is taken
    default: str                 # fall-through when no arm matches
    default_label: str = ""      # scaffold label for the default edge, when it is a real branch

    def __post_init__(self):
        if not self.arms:
            raise ValueError("CondChain needs at least one guarded exit (use Goto for none)")
        for a in self.arms:
            if not (isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], Pred)):
                raise ValueError(f"CondChain arm {a!r} is not a (Pred, target) pair")


# ===========================================================================
# 4.6  Block and Program (the minimal CFG)
def successor_labels(blk):
    """THE successor list of a block -- the single authority every CFG walk uses."""
    return terminator_targets(blk.term)


def terminator_targets(term):
    """Where a terminator can go. THE definition -- every caller delegates here.

    Missing a case here is silent -- the caller just sees fewer successors -- so there is exactly
    one copy of it.
    """
    if isinstance(term, Goto):
        return [term.target]
    if isinstance(term, CondGoto):
        return [term.t_target, term.f_target]
    if isinstance(term, LoopBack):
        return [term.body, term.exit_target]
    if isinstance(term, LoopCopy):
        return [term.next_target, term.exit_target]
    if isinstance(term, Return):
        return []                       # a sink: no successors, no dominance/back-edge role
    if isinstance(term, CondChain):
        return [tgt for _p, tgt in term.arms] + [term.default]
    return []


@dataclass
class Block:
    """One software-pipeline Phase as a basic block in the GIR CFG.

 phase -- 'prologue' | 'steady{n}' | 'drain{n}' | 'short{n}' | 'tail' (also the block LABEL).
 'short{n}' is the `T < M` degenerate arm: present only until
 FoldShortPathPass has ruled, then either folded away or kept as a real second path.
    """
    phase: str
    loop:  bool = False
    preds: tuple = ()
    succs: tuple = ()
    phis:  list = field(default_factory=list)
    xfers: list = field(default_factory=list)
    edge_xfers: dict = field(default_factory=dict)  # {successor: [GenXfer]} for non-back exits
    body:  list = field(default_factory=list)
    term:  object = None
    role:  str = "all"
    gen_rel:    object = None    # frame of this block's gdeltas; None = absolute generations
    chunk_base: int = 0          # position on the global summation-chunk timeline
    model_only: bool = False     # GIR reasons about this block; NO backend emits it (see below)
    path_chunk_base: dict = field(default_factory=dict)   # {pred: chunk_base ON THAT PATH}

    @property
    def label(self) -> str:
        return self.phase


@dataclass(frozen=True)
class GenPhi:
    """A general frame phi: `gen = phi(pred0:value0, pred1:value1, ...)`.

 gen -- the Gen this phi defines.
 entry_val -- legacy loop-entry value before EntranceFramePhisPass makes inputs explicit.
 incomings -- `(predecessor, value)`; `None` forwards that predecessor's transferred value.
 """
    gen:       Gen
    entry_val: object = None
    incomings: tuple = ()


@dataclass(frozen=True)
class GenXfer:
    """A back-edge transfer: `v_next = (v + adv) % ring` for a Gen.

 Attached to the block whose terminator carries the back-edge; `gen` names which Gen it
 advances so frame_map keys the transfer per back-edge (not a single xfers[0]).
 """
    gen: Gen
    adv: int
    ring: int


@dataclass
class Program:
    """The whole mainloop as one minimal CFG.

 blocks -- {label: Block}, insertion-ordered.
 entry -- entry block label.
 tiles -- interned Tile table (optional; may be empty).
    """
    blocks:  dict = field(default_factory=dict)
    entry:   str = "prologue"
    tiles:   dict = field(default_factory=dict)
    params:  dict = field(default_factory=dict)
    meta:    dict = field(default_factory=dict)
    version: int = 0
    pending: list = field(default_factory=list)   # PendingMarks from CollectPendingMarksPass

    def add_block(self, blk: Block):
        self.blocks[blk.label] = blk
        return blk

    def block(self, label: str) -> Block:
        return self.blocks[label]

    def bump(self):
        self.version += 1

    def walk_rpo(self):
        """Reverse-postorder over the minimal CFG (follows terminators), so a block is yielded
        after its non-back-edge predecessors.  Back-edges (target dominates source) do not
        gate ordering."""
        order, seen, stack = [], set(), []
        succ_labels = successor_labels

        # iterative post-order, skipping edges that revisit an on-stack node (back-edges)
        def visit(label):
            stack.append(label)
            seen.add(label)
            for s in succ_labels(self.blocks[label]):
                if s in self.blocks and s not in seen:
                    visit(s)
            stack.pop()
            order.append(label)

        if self.entry in self.blocks:
            visit(self.entry)
        # any block unreachable from entry still appears (defensive), in insertion order
        for label in self.blocks:
            if label not in seen:
                order.append(label)
        order.reverse()
        return [self.blocks[l] for l in order]


# ===========================================================================
# Ref accessors -- the one answer to "which operand(s) does this hop name?"
SHARED = "shared"
REGISTER = "register"


def _shared_only(refs):
    """Every Ref in `refs` that resides in shared space, in order."""
    return tuple(r for r in refs if r.tile.space == SHARED)


def first_shared_ref(refs):
    """The single shared Ref in `refs`, or None.

    For the read hop, whose source is one operand's shared tile; a second shared src would be a
    different movement, not a member of this one.
    """
    for r in refs:
        if r.tile.space == SHARED:
            return r
    return None


def _rotation_state(ref):
    """The rotation state a shared Ref names -- what a pointer register must hold to serve it."""
    gen = getattr(ref, "gen", None)
    return (None if gen is None else gen.ring, ref.gdelta, ref.abs_gen)


def copy_unit(inst):
    """`(members, refs)` for a global->shared copy Move, or `(None, None)` if `inst` is not one.

    `members` is the tuple of operand names the movement carries -- `('A',)` unfused, `('A','B')`
    for the Phi-fused AB group multi-wave TDM realizes as one aliased descriptor.
    """
    if not isinstance(inst, Move):
        return (None, None)
    refs = _shared_only(inst.dsts)
    if not refs:
        return (None, None)
    states = {_rotation_state(r) for r in refs}
    if len(states) > 1:
        raise RuntimeError(
            "GIR: fused copy movement %s names %d shared destinations whose pointers rotate "
            "DIFFERENTLY %s -- a Phi group is one cooperative instruction, so its members must "
            "share a ring depth and an offset for one register to serve them all. Differing "
            "rotation needs separate pointers, which the swap/increment analyses cannot express "
            "as a single def."
            % (tuple(r.tile.operand for r in refs), len(refs), sorted(map(str, states))))
    return (tuple(inst.unit) if inst.unit else tuple(r.tile.operand for r in refs), refs)


def descriptor_unit(prog, members):
    """The DESCRIPTOR a copy movement issues on: the Phi group it belongs to, not the members it
 happens to carry.  An asymmetric split gives region 0 the whole group and region 1 only the split
 member, and every pointer fact -- the region walk, the buffer swap -- belongs to the ONE physical
 descriptor both of them use."""
    keys = prog.meta.get("unit_regions", {}) or {}
    if tuple(members) in keys:
        return tuple(members)
    for key in keys:
        if set(members) <= set(key):
            return key
    return tuple(members)


def read_operand(inst):
    """The operand of a shared->register read Move, or None if `inst` is not one."""
    if not isinstance(inst, Move):
        return None
    src = first_shared_ref(inst.srcs)
    if src is None or not any(d.tile.space == REGISTER for d in inst.dsts):
        return None
    return src.tile.operand


def covered_coords(ref) -> tuple:
    """Every coordinate ONE reference fills -- its absorbed axes, re-inserted.

    `ref.covers` is `((axis, extent), ...)`: the inner axes this one instruction spans, which the
    hop's varying-axis set has already had removed. Empty means the reference fills exactly
    `ref.tile.coord`, which is every ordinary read.
    """
    axes = getattr(ref, "covers", None) or ()
    coord = ref.tile.coord
    if not axes:
        return (coord,)
    named = {a for a, _v in coord}
    out = [coord]
    for name, ext in axes:
        n = max(1, int(ext))
        if n <= 1:
            continue                  # degenerate: the cross product is the identity
        if name in named:             # partial coverage -- the coord holds the group index
            out = [tuple((a, v + j) if a == name else (a, v) for a, v in c)
                   for c in out for j in range(n)]
        else:                         # full absorption -- the axis left the varying set entirely
            out = [c + ((name, v),) for c in out for v in range(n)]
    return tuple(out)


BLOCK_ENTRY = "<entry>"
BLOCK_EXIT  = "<exit>"

# Where in its legal window a Mark lands.  The choice is per Mark kind, not global: a descriptor
# change wants to be as early as legal, a pointer change wants to sit between its two usages.
EARLIEST = "earliest"   # every TDM descriptor change: gr_increment and the copy-hop swap
MIDPOINT = "midpoint"   # the LDS read pointer, and the GL2 prefetch


@dataclass(frozen=True)
class Region:
    """A legal placement window over one block's body (opaque ordinals).

    block  -- the block LABEL the placement point lands in.
    after  -- place strictly AFTER this anchor (a body node, or BLOCK_ENTRY).
    before -- place strictly BEFORE this anchor (a body node, or BLOCK_EXIT).
    """
    block:  str
    after:  object = BLOCK_ENTRY
    before: object = BLOCK_EXIT
    policy: str = EARLIEST


def region_slot(body, anchor, opening):
    """A Region anchor as a slot index: `after` -> just past it, `before` -> the slot it occupies.

    Identity, never equality: Move/Mma/Mark are value-comparing dataclasses, so two structurally
    identical instructions in one body would otherwise resolve to the same slot."""
    if anchor is BLOCK_ENTRY or anchor == BLOCK_ENTRY:
        return 0
    if anchor is BLOCK_EXIT or anchor == BLOCK_EXIT:
        return len(body)
    for i, node in enumerate(body):
        if node is anchor:
            return i + 1 if opening else i
    return 0 if opening else len(body)


@dataclass
class PendingMark:
    """An un-placed Mark: the FACT (a Mark) plus WHERE it may legally go (a Region).

    `anchor` is filled by PlacementPass (a concrete body node or a boundary sentinel); apply
    inserts `mark` immediately before `anchor`, or at block entry/exit for a sentinel.
    """
    mark:   object            # a Mark
    region: Region
    anchor: object = None     # resolved by PlacementPass; None until then
    edges:  tuple = ()        # what this Mark is CHARGED with, for the layer that names it
    options: tuple = ()       # every legal Region, ACROSS BLOCKS; the naming layer picks one
