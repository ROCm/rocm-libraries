# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""IR vocabulary -- the totally-ordered instruction stream the decoder emits."""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass


# The three operand roles used by an `off` entry.
COPY, READ, STORE = "copy", "read", "store"

# Path direction: toward the compute, or away from it (the accumulator).
FORWARD, REVERSE = "forward", "reverse"


class Space:
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"


class Counter:
    COPY = "C_copy"  # global -> shared (the bulk staging movement)
    READ = "C_read"  # shared -> register (the fragment read)
    LOAD = "C_load"  # global -> register (a one-hop direct transfer)


HOP_COUNTER = {
    (Space.GLOBAL, Space.SHARED): Counter.COPY,
    (Space.SHARED, Space.REGISTER): Counter.READ,
    (Space.GLOBAL, Space.REGISTER): Counter.LOAD,
}


# Logical tile variable used by transfer-coverage expressions.
COVERAGE_VAR = "tile"


@dataclass(frozen=True)
class TransferCoverage:
    """Map logical tiles to an instruction and its destination slot."""
    carrier: object  # Expr over the operand's tile coordinate -> instruction index
    slot: object  # Expr over the operand's tile coordinate -> position within it

    def carrier_of(self, term: int, env=None) -> int:
        e = dict(env or {}); e[COVERAGE_VAR] = term
        return self.carrier.eval(e) if hasattr(self.carrier, "eval") else int(self.carrier)

    def slot_of(self, term: int, env=None) -> int:
        e = dict(env or {}); e[COVERAGE_VAR] = term
        return self.slot.eval(e) if hasattr(self.slot, "eval") else int(self.slot)

    def group(self, term: int, extent: int, env=None) -> tuple:
        count = self.carrier_of(term, env)
        return tuple(unit for unit in range(max(1, extent)) if self.carrier_of(unit, env) == count)

    def regs(self, term: int, extent: int, env=None) -> int:
        return len({self.slot_of(unit, env) for unit in self.group(term, extent, env)})


CoverageMap = TransferCoverage


@dataclass(frozen=True)
class Load:
    """A named transfer op: move `tokens` from `src` space to `dst` space."""
    tokens: tuple  # operand tokens moved together (len>1 => fused)
    src: str  # Space.*
    dst: str  # Space.*
    kiter: int  # K-loop iteration this load feeds (issue coordinate)
    part: int = 0  # which transfer of a multi-instruction load this is
    n_parts: int = 1  # total transfers this tile splits into (1 = not split)
    coord: tuple = ()  # semantic ((axis,val),...) this transfer fills
    size_regs: int = 0  # registers this ONE instruction moves (the load size)
    size_bytes: int = 0  # bytes this ONE instruction moves (for shared dest)
    advance: int = 0
    coverage: object = None
    ranges: tuple = ()  # display-only per-token axis ranges for a bulk/fused copy

    @property
    def counter(self) -> str:
        return HOP_COUNTER[(self.src, self.dst)]

    @property
    def semantic(self):
        return ("load", self.tokens, self.src, self.dst, self.size_regs, self.size_bytes)


@dataclass(frozen=True)
class Mma:
    """One matrix-multiply-accumulate: acc += a * b, optionally scaled."""
    a: str; b: str; acc: str
    kiter: int
    coord: tuple = ()
    scales: tuple = ()  # e.g. ("MXSA0","MXSB0")
    block: str = ""  # the hardware wmma block this lane-wmma belongs to, e.g.

    @property
    def semantic(self):
        return ("mma", self.block or tuple(ax for ax, _ in self.coord), bool(self.scales))


def _term(term):
    return (term[0], term[1], term[2]) if len(term) > 2 else (term[0], term[1], 1)


def _lin(terms, env) -> int:
    return sum(count * (env.get(axis, 0) // max(1, divisor))
               for axis, count, divisor in map(_term, terms))


def _term_str(term) -> str:
    axis, count, divisor = _term(term)
    slot = axis if divisor == 1 else f"{axis}//{divisor}"
    return slot if count == 1 else (f"{slot}*{count}" if divisor == 1 else f"({slot})*{count}")


@dataclass(frozen=True)
class Expr:
    """A tiny symbolic index expression over loop-mode names."""
    var: str = ""  # loop-mode name, or "" for a constant (single-var form)
    mod: int = 0  # modulus (0 = none)
    add: int = 0  # constant offset
    terms: tuple = ()  # mixed-radix: ((mode, coef), ...) -- supersedes `var` when non-empty
    digits: tuple = ()  # mixed-radix digits ((coef, div, extents), ...); extents 0 = no wrap
    carry: tuple = ()  # mixed-radix chunk-carry floor-div: (carry_terms, add, div) ->

    def free_vars(self) -> set:
        """The mode names this expression references -- its free variables under an env."""
        names = set()
        if self.var:
            names.add(self.var)
        names.update(axis for axis, _count, _divisor in map(_term, self.terms))
        if self.carry:
            names.update(axis for axis, _count, _divisor in map(_term, self.carry[0]))
        if self.digits:
            names.update(axis for axis, _count, _divisor in map(_term, self.digits[0]))
        return names

    def eval(self, env) -> int:
        if self.terms:
            value = self.add + _lin(self.terms, env)
        else:
            value = (env.get(self.var, 0) if self.var else 0) + self.add
        if self.digits:
            dterms, left, fields = self.digits
            pos = left + _lin(dterms, env)
            for coef, div, extents in fields:
                digit = pos // max(1, div)
                value += coef * (digit % extents if extents else digit)
        if self.carry:
            cterms, left, d = self.carry
            value += (left + _lin(cterms, env)) // max(1, d)
        return value % self.mod if self.mod else value

    def render(self) -> str:
        if self.terms:
            body = "+".join(_term_str(term) for term in self.terms)
            if self.add:
                body += f"+{self.add}"
        elif not self.var:
            body = str(self.add)
        else:
            body = self.var
            if self.add:
                body = f"{body}{self.add:+d}"  # +n / -n (no `+-` for a negative offset)
        if self.carry:
            cterms, left, d = self.carry
            inner = "+".join(_term_str(term) for term in cterms)
            if left:
                inner = f"{inner}+{left}" if inner else str(left)
            body += f"+({inner})//{d}"
        if self.digits:
            dterms, left, fields = self.digits
            inner = "+".join(_term_str(term) for term in dterms)
            if left:
                inner = f"{inner}+{left}" if inner else str(left)
            body += "+" + "+".join(_digit_str(inner, *field) for field in fields)
        if self.mod and (self.terms or self.var or self.carry or self.digits):
            return f"({body})%{self.mod}"
        return body


def _digit_str(inner, coef, div, extents) -> str:
    """One mixed-radix digit; `extents` of 0 means the digit does not wrap."""
    text = "(%s)//%d" % (inner, div)
    if extents:
        text += "%%%d" % extents
    return text if coef == 1 else "%d*(%s)" % (coef, text)


def cst(n: int) -> Expr:
    return Expr(var="", add=int(n))


_PRED_OPS = {">=": _operator.ge, ">": _operator.gt, "==": _operator.eq,
             "<": _operator.lt, "<=": _operator.le, "!=": _operator.ne}


@dataclass(frozen=True)
class Pred:
    """A structured predicate: compare an index `Expr` against an integer."""
    lhs: Expr
    op: str
    rhs: object = 0  # int OR Expr (a symbolic bound, e.g. `T - M`)

    def __post_init__(self):
        if self.op not in _PRED_OPS:
            raise RuntimeError(f"Pred.op {self.op!r} not in {sorted(_PRED_OPS)}")

    @staticmethod
    def _runtime(e, env):
        """Return whether an expression references a symbol absent from `env`."""
        if not hasattr(e, "eval"):
            return False  # a plain int is always known
        return any(slot and slot not in env for slot in e.free_vars())

    def eval(self, env):
        """Concrete bool if both sides are statically known in `env`, else None (runtime)."""
        if self._runtime(self.lhs, env) or self._runtime(self.rhs, env):
            return None
        rv = self.rhs.eval(env) if hasattr(self.rhs, "eval") else self.rhs
        return _PRED_OPS[self.op](self.lhs.eval(env), rv)

    def render(self) -> str:
        rs = self.rhs.render() if hasattr(self.rhs, "render") else str(self.rhs)
        return f"{self.lhs.render()} {self.op} {rs}"


@dataclass(frozen=True)
class Placement:
    """The logical storage an instance reads/writes -- NOT the physical register color."""
    space: str
    slots: tuple = (("", Expr()),)
    reg_off: int = 0
    byte_off: int = 0
    src_slot: object = None  # Expr: shared source-buffer generation for a shared->register read

    def render(self) -> str:
        if len(self.slots) == 1 and not self.slots[0][0]:
            body = f"{self.space}[buf{self.slots[0][1].render()}]"
        else:
            body = f"{self.space}[" + ",".join(f"{group},buf{e.render()}"
                                               for group, e in self.slots) + "]"
        if self.src_slot is not None:
            body += f"<-lds[buf{self.src_slot.render()}]"
        if self.reg_off:
            body += f"@r{self.reg_off}"
        elif self.byte_off:
            body += f"@b{self.byte_off}"
        return body

    def at(self, env, keep_unresolved=False):
        """Concrete Placement with each slot Expr evaluated to its buffer value but preserving"""
        def _fix(e):
            if keep_unresolved and (e.free_vars() - set(env or {})):
                return e
            return Expr(var="", add=e.eval(env), mod=e.mod)  # concrete value, keep the modulus
        return Placement(self.space,
                         tuple((group, _fix(e)) for group, e in self.slots),
                         self.reg_off, self.byte_off,
                         src_slot=(_fix(self.src_slot) if self.src_slot is not None else None))


# Obligation kind vocabulary.
RAW_KINDS = frozenset({"RAW-residency", "crossing-RAW"})
WAR_KINDS = frozenset({"rotation-WAR", "inplace-WAR"})
OBLIGATION_KINDS = RAW_KINDS | WAR_KINDS


def is_war(kind: str) -> bool:
    """Return whether `kind` is an anti-dependency."""
    if kind not in OBLIGATION_KINDS:
        raise RuntimeError(
            "unknown obligation kind %r -- the ledger hazard classes are a CLOSED set %s.  Add the "
            "new kind to RAW_KINDS or WAR_KINDS in LoopModel/ir.py; do not let it default, because "
            "every issue-order and await decision downstream branches on this answer."
            % (kind, sorted(OBLIGATION_KINDS)))
    return kind in WAR_KINDS


def is_raw(kind: str) -> bool:
    return not is_war(kind)


@dataclass(frozen=True)
class Await:
    """A named-dependency discharge point for a completion class."""
    dep: str
    counter: str
    scope: str = "wave"
    count: int = -1
    note: str = ""
    kind: str = "RAW-residency"


@dataclass
class Inst:
    """One operand instance at the axes it varies over."""
    op: object  # Load or Mma payload (operand, coord math)
    placement: object = None  # Placement with symbolic Expr slots
    awaits: tuple = ()
    anchor: tuple = ()  # loop order-coordinate this Inst must follow; () = emit at my own leaf

    @property
    def semantic(self):
        return self.op.semantic


@dataclass
class Loop:
    """A real loop node of the rolled nest: `for <mode> in range(extent)`."""
    axis: str  # loop index name (e.g. "iter", "substep", "min")
    trip: object = ""  # loop trip or continuation bound
    bodies: list = None  # list of body-sequences; body = [Loop|Branch|Inst|Peel]
    body_ranges: tuple = ()  # per-body half-open (lo, hi) over the trip; () = each spans it all
    outer: bool = False  # True for the pipelined outer loop.

    def __post_init__(self):
        if self.bodies is None:
            self.bodies = [[]]
        if self.body_ranges and len(self.body_ranges) != len(self.bodies):
            raise RuntimeError(
                f"Loop({self.axis}): {len(self.body_ranges)} body_ranges for "
                f"{len(self.bodies)} bodies -- one range per sub-body")

    def ranged_bodies(self):
        """`[(lo, hi, nodes)]` -- each sub-body with the half-open iteration range it runs"""
        end = self.trip if isinstance(self.trip, int) else None
        if not self.body_ranges:
            return [(0, end, body) for body in self.bodies]
        return [(lo, hi, body) for (lo, hi), body in zip(self.body_ranges, self.bodies)]

    @property
    def body(self):
        """The single body -- valid only on a single-body loop."""
        if len(self.bodies) != 1:
            raise RuntimeError(
                f"Loop({self.axis}) has {len(self.bodies)} sub-bodies; read them with `bodies` "
                f"or `ranged_bodies()`. `.body` only works when there is exactly one.")
        return self.bodies[0]


@dataclass
class Branch:
    """Explicit register-rotation control (user decision): a switch on `selector = mode % S`"""
    axis: str  # the loop mode the residue is taken over
    modulus: int  # S (number of residues / arms)
    arms: dict = None  # {residue: [Loop|Branch|Inst]}
    outer: bool = False  # True iff this Branch pins an outer-level residue (a drain step's

    def __post_init__(self):
        if self.arms is None:
            self.arms = {}


@dataclass
class Bind:
    """Bind an outer loop induction (`mode`, e.g."""
    axis: str  # the outer induction being pinned (iter)
    value: object  # an Expr giving the bound value (cst(j-M) prologue, Expr(T)+(t-M) drain)
    body: list = None

    def __post_init__(self):
        if self.body is None:
            self.body = []

    def bound_env(self, env):
        """`env` with `mode` bound to `value` iff `value` is statically resolvable in `env`"""
        value = self.value
        if all(f in env for f in value.free_vars()):
            return {**env, self.axis: value.eval(env)}
        return env

    def render(self) -> str:
        return f"bind {self.axis} = {self.value.render()}"


@dataclass
class Peel:
    """A prologue/drain peel node: the hoisted first-`k` (`kind='prologue'`) or last-`k`"""
    kind: str  # 'prologue' | 'drain'
    axis: str = ""
    k: object = 0
    body: list = None
    outer: bool = True  # a peel is over an outer level (the reduction chunk, or a persist level) by

    def __post_init__(self):
        if self.body is None:
            self.body = []


@dataclass
class Cond:
    """A first-class conditional over the reduction trip count -- the peel's validity"""
    pred: object  # a Pred (structured); the peel-validity guard / step pin
    then: list = None
    els: list = None
    kind: str = ""  # generic structural meaning (core sets this; see above)
    label: str = ""  # scaffold hint (GIR scaffold pass sets this from `kind`; core leaves "")
    outer: bool = False  # True iff this Cond pins an outer-level step -- the structural marker a

    def __post_init__(self):
        if self.then is None:
            self.then = []
        if self.els is None:
            self.els = []


# --- reorder: a reload follows the reads it overwrites --------------------------------------

def _movement(node):
    while True:
        if isinstance(node, Loop) and not node.outer:
            kids = [x for body in (node.bodies or []) for x in body]
            if len(kids) != 1:
                return None  # not a single rolled movement
            node = kids[0]
        elif isinstance(node, Cond) and not node.els and len(node.then) == 1:
            node = node.then[0]
        else:
            break
    return node if isinstance(node, Inst) else None


def _rebuilt(node, copy_first=frozenset()):
    if isinstance(node, Loop):
        return Loop(axis=node.axis, trip=node.trip, outer=node.outer,
                    bodies=[move_reloads_after_last_use(body, copy_first) for body in node.bodies], body_ranges=node.body_ranges)
    if isinstance(node, Cond):
        return Cond(pred=node.pred, kind=node.kind, label=node.label,
                    then=move_reloads_after_last_use(node.then, copy_first),
                    els=move_reloads_after_last_use(node.els, copy_first))
    if isinstance(node, Bind):
        return Bind(axis=node.axis, value=node.value, body=move_reloads_after_last_use(node.body, copy_first))
    if isinstance(node, Peel):
        return Peel(kind=node.kind, axis=node.axis, k=node.k, outer=node.outer, body=move_reloads_after_last_use(node.body, copy_first))
    return node


def _is_deferrable_read(node, _copy_first):
    """A read that runs ahead and refills a slot in place -- it must follow its last reader."""
    move = _movement(node)
    if move is None or not isinstance(move.op, Load) or move.op.dst != Space.REGISTER:
        return False
    if not getattr(move.op, "advance", 0):
        return False  # loads in place already, so it is already where it belongs
    if getattr(node, "anchor", ()) or getattr(move, "anchor", ()):
        return True
    return any(await_node.kind == "inplace-WAR" for await_node in move.awaits)


def _is_deferrable_copy(node, copy_first):
    """A shared copy that overwrites a buffer, and is not one this trip must issue first."""
    move = _movement(node)
    if move is None or not isinstance(move.op, Load) or move.op.dst != Space.SHARED:
        return False
    if any(token in copy_first for token in getattr(move.op, "tokens", ())):
        return False
    return any(is_war(await_node.kind) for await_node in move.awaits)


def child_bodies(node):
    """Every list of child nodes this node holds, in emission order; [] for a leaf.

    The one place that knows the tree's shape, so a walker only has to say what it does at a
    node, not re-derive where that node's children live.
    """
    if isinstance(node, Loop):
        return list(node.bodies or [])
    if isinstance(node, Branch):
        return list(node.arms.values())
    if isinstance(node, Cond):
        return [node.then, node.els or []]
    if isinstance(node, (Peel, Bind)):
        return [node.body]
    return []


def binds_axis(node):
    """The loop axis this node puts in scope for its children, or None."""
    return node.axis if isinstance(node, (Loop, Bind, Branch)) else None


def move_reloads_after_last_use(body, copy_first=frozenset()):
    """Move each refill after the instructions that still read the value it overwrites."""
    body = [_rebuilt(node, copy_first) for node in body]
    for is_deferrable in (_is_deferrable_read, _is_deferrable_copy):
        deferred = [node for node in body if is_deferrable(node, copy_first)]
        if deferred:
            body = [node for node in body if not is_deferrable(node, copy_first)] + deferred
    return body
