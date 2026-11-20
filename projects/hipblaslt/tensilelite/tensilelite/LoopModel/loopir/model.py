# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Address-agnostic LoopIR declarations.

The tree stores schedule structure and semantic tile expressions.  Operation
instances, effects, effect domains, and dependences are analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping

from .expr import (
    Axis,
    Expr,
    Param,
    Pred,
    TRUE,
    as_expr,
    as_pred,
)
from .values import (
    IMMEDIATE,
    INPUT,
    OUTPUT,
    CompletionClass,
    IRTypeError,
    OperandRole,
    Scope,
    TransformRefused,
)


def _tuple(value, field_name: str) -> tuple:
    if not isinstance(value, tuple):
        raise IRTypeError(f"{field_name} must be a tuple")
    return value


def _axes(value, field_name: str) -> tuple[Axis, ...]:
    result = _tuple(value, field_name)
    if not all(isinstance(axis, Axis) for axis in result):
        raise IRTypeError(f"{field_name} must contain only Axis values")
    if len({id(axis) for axis in result}) != len(result):
        raise IRTypeError(f"{field_name} must not repeat an Axis")
    return result


class Level(Enum):
    GLOBAL = auto()
    SHARED = auto()
    REGISTER = auto()


GLOBAL = Level.GLOBAL
SHARED = Level.SHARED
REGISTER = Level.REGISTER


class MovementOp:
    """Checked operation identity for one logical memory transfer instruction."""

    source: Level
    destination: Level
    source_effect_tile = True
    destination_effect_tile = True

    @classmethod
    def check_transfers(cls, transfers: tuple["Transfer", ...]):
        for transfer in transfers:
            if transfer.source is not cls.source \
                    or transfer.destination is not cls.destination:
                raise IRTypeError(
                    f"{cls.__name__} requires "
                    f"{cls.source.name}->{cls.destination.name} transfers"
                )


class TENSOR_LOAD(MovementOp):
    source = GLOBAL
    destination = SHARED


class DS_LOAD(MovementOp):
    source = SHARED
    destination = REGISTER
    destination_effect_tile = False


class GLOBAL_LOAD(MovementOp):
    source = GLOBAL
    destination = REGISTER
    destination_effect_tile = False


class STORE(MovementOp):
    source = REGISTER
    destination = GLOBAL
    source_effect_tile = False


class ComputeOp:
    """Checked operation identity for one logical compute instruction."""

    @classmethod
    def check_accesses(
        cls,
        reads: tuple["Tile", ...],
        writes: tuple["Tile", ...],
    ):
        if not reads or not writes:
            raise IRTypeError(
                f"{cls.__name__} requires reads and writes"
            )
        for tile in reads + writes:
            if len(tile.indices) != len(tile.operand.axes):
                raise IRTypeError(
                    f"{cls.__name__} local tiles must match operand rank"
                )


class WMMA(ComputeOp):
    pass


class BlockTag(Enum):
    PROLOGUE = auto()
    STEADY = auto()
    DRAIN = auto()
    SHORT = auto()


PROLOGUE = BlockTag.PROLOGUE
STEADY = BlockTag.STEADY
DRAIN = BlockTag.DRAIN
SHORT = BlockTag.SHORT


class LoopIR:
    """Common base type for values stored in Program bodies."""

    __slots__ = ()


@dataclass(frozen=True, eq=False, init=False)
class Operand:
    name: str
    axes: tuple[Axis, ...]
    role: OperandRole
    _identity: object = field(repr=False)

    def __init__(
        self,
        name: str,
        axes: tuple[Axis, ...],
        role: OperandRole = INPUT,
        *,
        _identity=None,
    ):
        if not isinstance(name, str) or not name:
            raise IRTypeError("Operand.name must be a non-empty string")
        if role not in (INPUT, OUTPUT):
            raise IRTypeError("Operand.role must be INPUT or OUTPUT")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "axes", _axes(axes, "Operand.axes"))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "_identity", _identity or object())

    def __getitem__(self, indices) -> "Tile":
        values = indices if isinstance(indices, tuple) else (indices,)
        return Tile(self, values)


@dataclass(frozen=True, eq=False, init=False)
class Tile:
    """One semantic operand value indexed by logical Axis expressions."""

    operand: Operand
    indices: tuple[Expr, ...]

    def __init__(self, operand: Operand, indices: tuple):
        if not isinstance(operand, Operand):
            raise IRTypeError("Tile.operand must be an Operand")
        indices = _tuple(indices, "Tile.indices")
        if len(indices) < len(operand.axes):
            raise IRTypeError(
                f"{operand.name} expects at least {len(operand.axes)} indices, "
                f"got {len(indices)}"
            )
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "indices", tuple(as_expr(value) for value in indices))

    @property
    def local_indices(self) -> tuple[Expr, ...]:
        """Operand-coordinate suffix, excluding explicit effect-only axes."""

        count = len(self.operand.axes)
        return self.indices[-count:] if count else ()


@dataclass(frozen=True)
class Effect:
    """Explicit semantic tile relation attached to a Movement or Transfer."""

    tile: Tile
    where: Pred = TRUE

    def __post_init__(self):
        if not isinstance(self.tile, Tile):
            raise IRTypeError("Effect.tile must be a Tile")
        object.__setattr__(self, "where", as_pred(self.where))


@dataclass(frozen=True)
class Transfer:
    """One instruction-local member of a Movement.

    ``local_tile`` describes the wave-local tile carried by the instruction.
    ``effect`` optionally supplies this member's semantic dependency
    identity; otherwise the containing Movement supplies one common effect.
    """

    local_tile: Tile
    source: Level
    destination: Level
    effect: Effect | None = None

    def __post_init__(self):
        if not isinstance(self.local_tile, Tile):
            raise IRTypeError("Transfer.local_tile must be a Tile")
        if len(self.local_tile.indices) != len(self.local_tile.operand.axes):
            raise IRTypeError("Transfer.local_tile must match operand rank")
        if self.effect is not None:
            if not isinstance(self.effect, Effect):
                raise IRTypeError("Transfer.effect must be an Effect")
            if self.effect.tile.operand is not self.local_tile.operand:
                raise IRTypeError(
                    "Transfer effect and local tiles must use the same operand"
                )
        if not isinstance(self.source, Level) \
                or not isinstance(self.destination, Level):
            raise IRTypeError("Transfer levels must be Level values")
        if self.source is self.destination:
            raise IRTypeError("Transfer source and destination levels must differ")


class _LogicalIdentity:
    __slots__ = ()


def _completion(value):
    if value is IMMEDIATE or isinstance(value, CompletionClass):
        return value
    raise IRTypeError("completion must be a CompletionClass or IMMEDIATE")


@dataclass(frozen=True, eq=False, init=False)
class Movement(LoopIR):
    op: type[MovementOp]
    transfers: tuple[Transfer, ...]
    completion: CompletionClass | object
    effect: Effect | None
    _identity: _LogicalIdentity = field(repr=False)

    def __init__(
        self,
        op: type[MovementOp],
        transfers: tuple[Transfer, ...],
        completion,
        *,
        effect=None,
        _identity=None,
    ):
        if not isinstance(op, type) or not issubclass(op, MovementOp):
            raise IRTypeError("Movement.op must inherit MovementOp")
        transfers = _tuple(transfers, "Movement.transfers")
        if not transfers or not all(isinstance(value, Transfer) for value in transfers):
            raise IRTypeError("Movement.transfers must contain Transfer values")
        if effect is not None and not isinstance(effect, Effect):
            raise IRTypeError("Movement.effect must be an Effect")
        if effect is None:
            if any(transfer.effect is None for transfer in transfers):
                raise IRTypeError(
                    "each Transfer needs an Effect when Movement has no common Effect"
                )
        else:
            if any(transfer.effect is not None for transfer in transfers):
                raise IRTypeError(
                    "Effect must belong to Movement or Transfers, not both"
                )
            if any(
                transfer.local_tile.operand is not effect.tile.operand
                for transfer in transfers
            ):
                raise IRTypeError(
                    "common Movement Effect requires one operand"
                )
        op.check_transfers(transfers)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "transfers", transfers)
        object.__setattr__(self, "completion", _completion(completion))
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "_identity", _identity or _LogicalIdentity())

    def effect_for(self, transfer: int) -> Effect:
        member = self.transfers[transfer]
        return self.effect if self.effect is not None else member.effect

    def tile_for_endpoint(self, transfer: int, *, source: bool) -> Tile:
        member = self.transfers[transfer]
        uses_effect = (
            self.op.source_effect_tile
            if source else self.op.destination_effect_tile
        )
        return self.effect_for(transfer).tile if uses_effect else member.local_tile


@dataclass(frozen=True, eq=False, init=False)
class Compute(LoopIR):
    op: type[ComputeOp]
    reads: tuple[Tile, ...]
    writes: tuple[Tile, ...]
    completion: CompletionClass | object
    _identity: _LogicalIdentity = field(repr=False)

    def __init__(
        self,
        op: type[ComputeOp],
        reads: tuple[Tile, ...],
        writes: tuple[Tile, ...],
        completion=IMMEDIATE,
        *,
        _identity=None,
    ):
        if not isinstance(op, type) or not issubclass(op, ComputeOp):
            raise IRTypeError("Compute.op must inherit ComputeOp")
        reads = _tuple(reads, "Compute.reads")
        writes = _tuple(writes, "Compute.writes")
        if not reads and not writes:
            raise IRTypeError("Compute needs at least one read or write")
        if not all(isinstance(value, Tile) for value in reads + writes):
            raise IRTypeError("Compute reads/writes must contain Tile values")
        op.check_accesses(reads, writes)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "completion", _completion(completion))
        object.__setattr__(self, "_identity", _identity or _LogicalIdentity())


def same_operation(left, right) -> bool:
    return isinstance(left, (Movement, Compute)) \
        and isinstance(right, (Movement, Compute)) \
        and left._identity is right._identity


def _body(value, field_name: str) -> tuple[LoopIR, ...]:
    value = _tuple(value, field_name)
    if not all(isinstance(node, LoopIR) for node in value):
        raise IRTypeError(f"{field_name} must contain only LoopIR values")
    return value


@dataclass(eq=False, init=False)
class Loop(LoopIR):
    axis: Axis
    lo: Expr
    hi: Expr
    body: tuple[LoopIR, ...]

    def __init__(
        self,
        axis: Axis,
        lo=0,
        hi=None,
        body=None,
    ):
        if not isinstance(axis, Axis):
            raise IRTypeError("Loop.axis must be an Axis")
        if body is None and isinstance(lo, tuple):
            body = lo
            lo = 0
            hi = axis.extent
        if body is None:
            raise IRTypeError("Loop.body is required")
        if hi is None:
            hi = axis.extent
        lower = as_expr(lo)
        upper = as_expr(hi)
        if lower.constant_value is not None \
                and upper.constant_value is not None \
                and upper.constant_value < lower.constant_value:
            raise IRTypeError("Loop.hi must not precede Loop.lo")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "lo", lower)
        object.__setattr__(self, "hi", upper)
        object.__setattr__(self, "body", _body(body, "Loop.body"))


@dataclass(eq=False, init=False)
class Cond(LoopIR):
    pred: Pred
    then_body: tuple[LoopIR, ...]
    else_body: tuple[LoopIR, ...]

    def __init__(self, pred, then_body: tuple[LoopIR, ...], else_body=()):
        object.__setattr__(self, "pred", as_pred(pred))
        object.__setattr__(self, "then_body", _body(then_body, "Cond.then_body"))
        object.__setattr__(self, "else_body", _body(else_body, "Cond.else_body"))


@dataclass(eq=False, init=False)
class Let(LoopIR):
    axis: Axis
    value: Expr
    body: tuple[LoopIR, ...]

    def __init__(self, axis: Axis, value, body: tuple[LoopIR, ...]):
        if not isinstance(axis, Axis):
            raise IRTypeError("Let.axis must be an Axis")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "value", as_expr(value))
        object.__setattr__(self, "body", _body(body, "Let.body"))


@dataclass(eq=False, init=False)
class Switch(LoopIR):
    expression: Expr
    cases: tuple[tuple[int, tuple[LoopIR, ...]], ...]
    otherwise: tuple[LoopIR, ...]

    def __init__(self, expression, cases, otherwise=()):
        values = cases.items() if isinstance(cases, Mapping) else cases
        normalized = []
        seen = set()
        for key, body in values:
            if not isinstance(key, int) or isinstance(key, bool):
                raise IRTypeError("Switch case values must be integers")
            if key in seen:
                raise IRTypeError(f"Switch repeats case {key}")
            seen.add(key)
            normalized.append((key, _body(body, f"Switch case {key}")))
        normalized.sort(key=lambda item: item[0])
        object.__setattr__(self, "expression", as_expr(expression))
        object.__setattr__(self, "cases", tuple(normalized))
        object.__setattr__(self, "otherwise", _body(otherwise, "Switch.otherwise"))

    def case(self, value: int) -> tuple[LoopIR, ...]:
        try:
            return next(body for key, body in self.cases if key == value)
        except StopIteration as error:
            raise TransformRefused(f"Switch has no case {value}") from error


@dataclass
class Region(LoopIR):
    """Zero-runtime marker starting a tagged block in its containing body."""

    tag: BlockTag

    def __post_init__(self):
        if not isinstance(self.tag, BlockTag):
            raise IRTypeError("Region.tag must be a BlockTag")


Node = LoopIR


def child_bodies(node: LoopIR) -> tuple[tuple[LoopIR, ...], ...]:
    if isinstance(node, Loop):
        return (node.body,)
    if isinstance(node, Cond):
        return node.then_body, node.else_body
    if isinstance(node, Let):
        return (node.body,)
    if isinstance(node, Switch):
        return tuple(body for _value, body in node.cases) + (node.otherwise,)
    return ()


def operation_tiles(operation: Movement | Compute) -> tuple[Tile, ...]:
    if isinstance(operation, Movement):
        return tuple(
            tile
            for index, transfer in enumerate(operation.transfers)
            for tile in (
                operation.effect_for(index).tile,
                transfer.local_tile,
            )
        )
    return operation.reads + operation.writes


def _check_program(
    params: tuple[Param, ...],
    axes: tuple[Axis, ...],
    operands: tuple[Operand, ...],
    body: tuple[LoopIR, ...],
    agents: tuple[tuple[Axis, Scope], ...],
):
    if not all(isinstance(value, Param) for value in params):
        raise IRTypeError("Program.params must contain Param values")
    if not all(isinstance(value, Operand) for value in operands):
        raise IRTypeError("Program.operands must contain Operand values")
    if not all(
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], Axis)
        and isinstance(value[1], Scope)
        for value in agents
    ):
        raise IRTypeError(
            "Program.agents must contain (Axis, Scope) pairs"
        )
    for values, name in (
        (params, "params"),
        (axes, "axes"),
        (operands, "operands"),
    ):
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise IRTypeError(f"Program.{name} contains duplicate names")
    agent_axes = tuple(axis for axis, _scope in agents)
    if len({id(axis) for axis in agent_axes}) != len(agent_axes):
        raise IRTypeError("Program.agents must not repeat an Axis")
    declaration_names = (
        [value.name for value in params]
        + [value.name for value in axes]
        + [value.name for value in agent_axes]
        + [value.name for value in operands]
    )
    if len(declaration_names) != len(set(declaration_names)):
        raise IRTypeError("Program declaration names must be globally unique")
    declared_axes = {
        id(value) for value in (*axes, *agent_axes)
    }
    declared_operands = {id(value) for value in operands}
    for operand in operands:
        if any(id(axis) not in declared_axes for axis in operand.axes):
            raise IRTypeError(f"Operand {operand.name!r} uses undeclared axes")
    for operation in _operations_in(body):
        for tile in operation_tiles(operation):
            if id(tile.operand) not in declared_operands:
                raise IRTypeError(
                    f"operation uses undeclared operand {tile.operand.name!r}"
                )
            free = {
                axis
                for expression in tile.indices
                for axis in expression.free_axes()
            }
            if {id(axis) for axis in free} - declared_axes:
                raise IRTypeError("operation tile uses undeclared axes")
    agent_ids = {id(axis) for axis in agent_axes}
    for node in _walk_nodes(body):
        if isinstance(node, Loop) and id(node.axis) in agent_ids:
            raise IRTypeError("Program agent axes cannot be bound by Loop")


@dataclass(eq=False, init=False)
class Program:
    params: tuple[Param, ...]
    axes: tuple[Axis, ...]
    operands: tuple[Operand, ...]
    body: tuple[LoopIR, ...]
    agents: tuple[tuple[Axis, Scope], ...]
    _identity: object = field(repr=False)

    def __init__(
        self,
        params: tuple[Param, ...],
        axes: tuple[Axis, ...],
        operands: tuple[Operand, ...],
        body: tuple[LoopIR, ...],
        agents=(),
        *,
        _identity=None,
    ):
        params = _tuple(params, "Program.params")
        axes = _axes(axes, "Program.axes")
        operands = _tuple(operands, "Program.operands")
        agents = _tuple(agents, "Program.agents")
        body = _body(body, "Program.body")
        _check_program(params, axes, operands, body, agents)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "_identity", _identity or object())

    def check(self) -> "Program":
        """Check declarations and every stored node, returning this Program."""

        _check_program(
            self.params,
            self.axes,
            self.operands,
            self.body,
            self.agents,
        )
        return self

    def with_body(
        self,
        body: tuple[LoopIR, ...],
        *,
        check: bool = True,
    ) -> "Program":
        """Return a body replacement, optionally deferring the whole-tree check."""

        body = _body(body, "Program.body")
        if not isinstance(check, bool):
            raise IRTypeError("Program.with_body.check must be a bool")
        if check:
            _check_program(
                self.params,
                self.axes,
                self.operands,
                body,
                self.agents,
            )
        self.body = body
        return self


def _operations_in(body):
    for node in body:
        if isinstance(node, (Movement, Compute)):
            yield node
        for child in child_bodies(node):
            yield from _operations_in(child)


def _walk_nodes(body):
    for node in body:
        yield node
        for child in child_bodies(node):
            yield from _walk_nodes(child)


__all__ = [
    "Level", "GLOBAL", "SHARED", "REGISTER",
    "MovementOp", "TENSOR_LOAD", "DS_LOAD", "GLOBAL_LOAD", "STORE",
    "ComputeOp", "WMMA",
    "BlockTag", "PROLOGUE", "STEADY", "DRAIN", "SHORT",
    "LoopIR", "Operand", "Tile", "Effect", "Transfer",
    "Movement", "Compute", "Loop", "Cond", "Let", "Switch", "Region",
    "Node", "Program", "same_operation", "child_bodies", "operation_tiles",
]
