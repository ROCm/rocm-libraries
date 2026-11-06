# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived logical effects and dependences for LoopIR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Mapping

from .values import IRTypeError
from .expr import Axis, Compare, Expr, Pred, TRUE, expr_equivalent
from .model import (
    PROLOGUE,
    REGISTER,
    SHARED,
    Compute,
    Cond,
    Let,
    Loop,
    Movement,
    Program,
    Region,
    Switch,
    Tile,
)
from .query import occurrences
from .query import Occurrence
from .values import IMMEDIATE


class AccessKind(Enum):
    READ = auto()
    WRITE = auto()


READ = AccessKind.READ
WRITE = AccessKind.WRITE


@dataclass(frozen=True)
class MovementSource:
    transfer: int


@dataclass(frozen=True)
class MovementDestination:
    transfer: int


@dataclass(frozen=True)
class ComputeRead:
    index: int


@dataclass(frozen=True)
class ComputeWrite:
    index: int


Endpoint = MovementSource | MovementDestination | ComputeRead | ComputeWrite


@dataclass(frozen=True)
class EffectInstance:
    """Minimal view of one operation endpoint at one structural occurrence."""

    occurrence: Occurrence
    endpoint: Endpoint

    def __post_init__(self):
        operation = self.occurrence.node
        if isinstance(self.endpoint, (MovementSource, MovementDestination)):
            if not isinstance(operation, Movement):
                raise IRTypeError("Movement endpoint requires a Movement occurrence")
            index = self.endpoint.transfer
            if not 0 <= index < len(operation.transfers):
                raise IRTypeError("Movement endpoint transfer is out of range")
        elif isinstance(self.endpoint, ComputeRead):
            if not isinstance(operation, Compute) \
                    or not 0 <= self.endpoint.index < len(operation.reads):
                raise IRTypeError("Compute read endpoint is out of range")
        elif isinstance(self.endpoint, ComputeWrite):
            if not isinstance(operation, Compute) \
                    or not 0 <= self.endpoint.index < len(operation.writes):
                raise IRTypeError("Compute write endpoint is out of range")
        else:
            raise IRTypeError("unknown effect endpoint")

    @property
    def operation(self):
        return self.occurrence.node

    @property
    def kind(self) -> AccessKind:
        return (
            READ
            if isinstance(self.endpoint, (MovementSource, ComputeRead))
            else WRITE
        )

    @property
    def tile(self) -> Tile:
        operation = self.operation
        if isinstance(self.endpoint, MovementSource):
            return operation.tile_for_endpoint(
                self.endpoint.transfer,
                source=True,
            )
        if isinstance(self.endpoint, MovementDestination):
            return operation.tile_for_endpoint(
                self.endpoint.transfer,
                source=False,
            )
        if isinstance(self.endpoint, ComputeRead):
            return operation.reads[self.endpoint.index]
        return operation.writes[self.endpoint.index]

    @property
    def local_tile(self) -> Tile:
        operation = self.operation
        if isinstance(self.endpoint, (MovementSource, MovementDestination)):
            return operation.transfers[self.endpoint.transfer].local_tile
        return self.tile

    @property
    def level(self):
        operation = self.operation
        if isinstance(self.endpoint, MovementSource):
            return operation.transfers[self.endpoint.transfer].source
        if isinstance(self.endpoint, MovementDestination):
            return operation.transfers[self.endpoint.transfer].destination
        return REGISTER

    @property
    def predicate(self) -> Pred:
        operation = self.operation
        if isinstance(self.endpoint, (MovementSource, MovementDestination)):
            return operation.effect_for(self.endpoint.transfer).where
        return TRUE


@dataclass(frozen=True)
class _EffectContext:
    """Private result backing the public effect-domain query functions."""

    loops: tuple[tuple[Axis, Expr, Expr], ...]
    bindings: tuple[tuple[Axis, Expr], ...]
    predicates: tuple[Pred, ...]
    local_axes: tuple[Axis, ...]
    agent_axes: tuple[tuple[Axis, object], ...]
    fixed_equalities: tuple


def effect_domain(effect: EffectInstance) -> _EffectContext:
    occurrence = effect.occurrence
    bound = {id(axis) for axis, _lo, _hi in occurrence.domains}
    bound.update(id(axis) for axis, _value in occurrence.bindings)
    free = set()
    for expression in effect.tile.indices:
        free.update(expression.free_axes())
    free.update(effect.predicate.free_axes())
    for predicate in occurrence.predicates:
        free.update(predicate.free_axes())
    program = occurrence._program
    free_ids = {id(axis) for axis in free}
    local = tuple(
        axis for axis in program.axes
        if id(axis) in free_ids and id(axis) not in bound
    )
    agent = tuple(
        (axis, scope)
        for axis, scope in occurrence._program.agents
        if id(axis) in free_ids
    )
    predicates = occurrence.predicates + (effect.predicate,)
    fixed_equalities = tuple(
        (
            expression,
            value,
            frozenset(id(axis) for axis in expression.free_axes()),
        )
        for predicate in predicates
        for fixed in (_fixed_equality(predicate),)
        if fixed is not None
        for expression, value in (fixed,)
    )
    return _EffectContext(
        occurrence.domains,
        occurrence.bindings,
        predicates,
        local,
        agent,
        fixed_equalities,
    )


def effects_of_occurrence(occurrence: Occurrence) -> tuple[EffectInstance, ...]:
    operation = occurrence.node
    if isinstance(operation, Movement):
        return tuple(
            effect
            for index in range(len(operation.transfers))
            for effect in (
                EffectInstance(occurrence, MovementSource(index)),
                EffectInstance(occurrence, MovementDestination(index)),
            )
        )
    if isinstance(operation, Compute):
        return tuple(
            EffectInstance(occurrence, ComputeRead(index))
            for index in range(len(operation.reads))
        ) + tuple(
            EffectInstance(occurrence, ComputeWrite(index))
            for index in range(len(operation.writes))
        )
    raise IRTypeError("effects require a Movement or Compute occurrence")


def effects(program: Program) -> tuple[EffectInstance, ...]:
    if not isinstance(program, Program):
        raise IRTypeError("effects expects a Program")
    return tuple(
        effect
        for occurrence in occurrences(program)
        if isinstance(occurrence.node, (Movement, Compute))
        for effect in effects_of_occurrence(occurrence)
    )


@dataclass(frozen=True)
class _ConcreteEffect:
    """Concrete oracle point for tests of an EffectInstance."""

    effect: EffectInstance
    bindings: tuple[tuple[Axis, int], ...]
    agents: tuple[tuple[Axis, int], ...]
    tile: tuple[int, ...]


def effect_points(
    effect: EffectInstance,
    bindings: Mapping[object, int] = {},
) -> tuple[_ConcreteEffect, ...]:
    """Enumerate one EffectInstance for concrete parameter bindings.

    This is a test/debug oracle.  Production dependence derivation remains
    symbolic and never relies on finite runtime-loop sampling.
    """

    domain = effect_domain(effect)
    initial = dict(bindings)
    points = []

    def enumerate_ranges(ranges, index, env, done):
        if index == len(ranges):
            done(env)
            return
        axis, lower, upper = ranges[index]
        lo = lower.evaluate(env)
        hi = upper.evaluate(env)
        for value in range(lo, hi):
            nested = dict(env)
            nested[axis] = value
            enumerate_ranges(ranges, index + 1, nested, done)

    def with_bound_axes(env):
        env = dict(env)
        for axis, value in domain.bindings:
            env[axis] = value.evaluate(env)
        local_ranges = tuple(
            (axis, Expr.constant(0), axis.extent)
            for axis in domain.local_axes
        )
        agent_ranges = tuple(
            (axis, Expr.constant(0), axis.extent)
            for axis, _scope in domain.agent_axes
            if axis not in env and axis.name not in env
        )

        def collect(local_env):
            values = tuple(predicate.evaluate(local_env) for predicate in domain.predicates)
            if any(value is None for value in values):
                raise IRTypeError("concrete effect oracle has an unbound predicate")
            if not all(values):
                return
            tile = tuple(expression.evaluate(local_env) for expression in effect.tile.indices)
            axes = tuple(
                (axis, int(local_env[axis]))
                for axis, _lo, _hi in domain.loops
            ) + tuple(
                (axis, int(local_env[axis])) for axis in domain.local_axes
            )
            agents = tuple(
                (axis, int(local_env[axis]))
                for axis, _binding in domain.agent_axes
            )
            points.append(_ConcreteEffect(effect, axes, agents, tile))

        enumerate_ranges(agent_ranges + local_ranges, 0, env, collect)

    enumerate_ranges(domain.loops, 0, initial, with_bound_axes)
    return tuple(points)


class DependenceKind(Enum):
    RAW = auto()
    WAR = auto()


RAW = DependenceKind.RAW
WAR = DependenceKind.WAR


@dataclass(frozen=True)
class EffectEquality:
    complete: Expr
    issue: Expr


@dataclass(frozen=True)
class IterationConstraint:
    axis: Axis
    distance: int


@dataclass(frozen=True)
class SymbolicRelation:
    complete_domain: _EffectContext
    issue_domain: _EffectContext
    equalities: tuple[EffectEquality, ...]
    iteration_constraints: tuple[IterationConstraint, ...]
    conservative: bool = False


@dataclass(frozen=True)
class Dependence:
    """Completion-to-issue ordering over one symbolic relation."""

    complete: EffectInstance
    issue: EffectInstance
    domain: SymbolicRelation

    @property
    def kind(self) -> DependenceKind:
        if self.complete.kind is WRITE and self.issue.kind is READ:
            return RAW
        if self.complete.kind is READ and self.issue.kind is WRITE:
            return WAR
        raise IRTypeError("a dependence requires exactly one write")

    @property
    def crossing_raw(self) -> bool:
        """Whether a copy completes the same runtime iteration this read issues."""

        if not (
            self.kind is RAW
            and isinstance(self.complete.operation, Movement)
            and isinstance(self.issue.operation, Movement)
            and self.complete.level is SHARED
            and self.issue.level is SHARED
            and self.complete.operation.completion is not IMMEDIATE
        ):
            return False
        if self.domain.conservative:
            return False
        runtime_shifted = any(
            not expr_equivalent(equality.complete, equality.issue)
            and any(
                axis.extent.free_params()
                for axis in (
                    set(equality.complete.free_axes())
                    | set(equality.issue.free_axes())
                )
            )
            for equality in self.domain.equalities
        )
        if not runtime_shifted:
            return False
        if self.domain.iteration_constraints:
            return all(
                constraint.distance == 0
                for constraint in self.domain.iteration_constraints
            )
        complete_runtime = {
            id(node)
            for node in self.complete.occurrence.ancestors
            if isinstance(node, Loop)
            and node.axis.extent.free_params()
        }
        return any(
            isinstance(node, Loop)
            and node.axis.extent.free_params()
            and id(node) in complete_runtime
            for node in self.issue.occurrence.ancestors
        )


def dependence_points(
    dependence: Dependence,
    bindings: Mapping[object, int] = {},
) -> tuple[tuple[_ConcreteEffect, _ConcreteEffect], ...]:
    """Concrete oracle pairs satisfying one symbolic Dependence."""

    complete_agent_axes = {
        id(axis) for axis, _binding in dependence.domain.complete_domain.agent_axes
    }
    issue_agent_axes = {
        id(axis) for axis, _binding in dependence.domain.issue_domain.agent_axes
    }
    common_agents = complete_agent_axes & issue_agent_axes

    def agents_match(complete, issue):
        left = {id(axis): value for axis, value in complete.agents}
        right = {id(axis): value for axis, value in issue.agents}
        return all(left[axis] == right[axis] for axis in common_agents)

    return tuple(
        (complete, issue)
        for complete in effect_points(dependence.complete, bindings)
        for issue in effect_points(dependence.issue, bindings)
        if complete.tile == issue.tile and agents_match(complete, issue)
    )


def _affine(expression: Expr):
    return expression._affine_form()


def _path_order(first, second):
    for left, right in zip(first, second):
        if left == right:
            continue
        left_field, left_index = left
        right_field, right_index = right
        if left_field == right_field == "body":
            return -1 if left_index < right_index else 1
        if left_field == right_field == "case" and left_index != right_index:
            return None
        if {left_field, right_field} <= {"then", "else"}:
            return None
        return None
    if len(first) == len(second):
        return 0
    return -1 if len(first) < len(second) else 1


def _switch_case_precedes(first, second):
    """Whether serial finite iteration makes one sibling Switch case earlier."""

    divergence = next(
        (
            index
            for index, (left, right) in enumerate(zip(first.path, second.path))
            if left != right
        ),
        None,
    )
    if divergence is None:
        return False
    left_field, left_case = first.path[divergence]
    right_field, right_case = second.path[divergence]
    if left_field != "case" or right_field != "case":
        return False
    case_ordinal = sum(
        field in ("case", "otherwise")
        for field, _index in first.path[:divergence + 1]
    ) - 1
    first_switches = tuple(
        node for node in first.ancestors if isinstance(node, Switch)
    )
    second_switches = tuple(
        node for node in second.ancestors if isinstance(node, Switch)
    )
    if not (
        0 <= case_ordinal < len(first_switches)
        and case_ordinal < len(second_switches)
        and first_switches[case_ordinal] is second_switches[case_ordinal]
    ):
        return False
    switch = first_switches[case_ordinal]
    if not (
        0 <= left_case < len(switch.cases)
        and 0 <= right_case < len(switch.cases)
    ):
        return False

    selector_axes = switch.expression.free_axes()
    selector_ids = {id(axis) for axis in selector_axes}
    loops = tuple(
        node for node in first.ancestors
        if isinstance(node, Loop) and id(node.axis) in selector_ids
    )
    if (
        not selector_axes
        or {id(loop.axis) for loop in loops} != selector_ids
        or any(
            loop.lo.constant_value is None or loop.hi.constant_value is None
            for loop in loops
        )
    ):
        return False

    values = []

    def enumerate_selector(index, environment):
        if index == len(loops):
            values.append(switch.expression.evaluate(environment))
            return
        loop = loops[index]
        for value in range(
            int(loop.lo.constant_value),
            int(loop.hi.constant_value),
        ):
            enumerate_selector(
                index + 1,
                {**environment, loop.axis: value},
            )

    enumerate_selector(0, {})
    if values != sorted(values):
        return False
    left_value = switch.cases[left_case][0]
    right_value = switch.cases[right_case][0]
    left_positions = [
        index for index, value in enumerate(values) if value == left_value
    ]
    right_positions = [
        index for index, value in enumerate(values) if value == right_value
    ]
    return bool(left_positions and right_positions) \
        and max(left_positions) < min(right_positions)


def _serial_axes(
    first: EffectInstance,
    second: EffectInstance,
) -> tuple[Axis, ...]:
    first_loops = {
        id(node): node
        for node in first.occurrence.ancestors
        if isinstance(node, Loop)
    }
    return tuple(
        node.axis
        for node in second.occurrence.ancestors
        if isinstance(node, Loop)
        and id(node) in first_loops
    )


def _fixed_equality(predicate):
    if not isinstance(predicate, Compare) or predicate.op != "==":
        return None
    if predicate.rhs.constant_value is not None:
        return predicate.lhs, int(predicate.rhs.constant_value)
    if predicate.lhs.constant_value is not None:
        return predicate.rhs, int(predicate.lhs.constant_value)
    return None


def _domains_are_disjoint(complete_domain, issue_domain, loop_axes):
    loop_ids = {id(axis) for axis in loop_axes}
    complete_equalities = tuple(
        (expression, value)
        for expression, value, free_ids in complete_domain.fixed_equalities
        if not loop_ids & free_ids
    )
    issue_equalities = tuple(
        (expression, value)
        for expression, value, free_ids in issue_domain.fixed_equalities
        if not loop_ids & free_ids
    )
    return any(
        left_value != right_value
        and expr_equivalent(left, right)
        for left, left_value in complete_equalities
        for right, right_value in issue_equalities
    )


def _finite_relation_may_overlap(relation: SymbolicRelation) -> bool:
    """Prove emptiness over finite local/agent coordinates.

    Runtime loop/parameter terms remain symbolic.  A finite assignment is
    viable unless a predicate is false or an affine equality is contradictory
    after substituting that assignment.
    """

    axes = []
    seen_axes = set()
    for domain in (relation.complete_domain, relation.issue_domain):
        for axis in domain.local_axes:
            if id(axis) not in seen_axes:
                seen_axes.add(id(axis))
                axes.append(axis)
        for axis, _scope in domain.agent_axes:
            if id(axis) not in seen_axes:
                seen_axes.add(id(axis))
                axes.append(axis)
    if any(axis.extent.constant_value is None for axis in axes):
        return True

    ranges = tuple(range(int(axis.extent.constant_value)) for axis in axes)
    for values in product(*ranges):
        environment = dict(zip(axes, values))
        viable = True
        for domain in (relation.complete_domain, relation.issue_domain):
            for predicate in domain.predicates:
                try:
                    verdict = predicate.evaluate(environment)
                except KeyError:
                    verdict = None
                if verdict is False:
                    viable = False
                    break
            if not viable:
                break
        if not viable:
            continue
        for equality in relation.equalities:
            left = _affine(equality.complete.substitute(environment))
            right = _affine(equality.issue.substitute(environment))
            if left is not None and right is not None \
                    and left[0] == right[0] and left[1] != right[1]:
                viable = False
                break
        if viable:
            return True
    return False


def _relation(
    complete: EffectInstance,
    issue: EffectInstance,
    complete_domain,
    issue_domain,
    structural,
    loop_axes,
    domains_disjoint,
):
    if complete.tile.operand is not issue.tile.operand \
            or complete.level is not issue.level \
            or len(complete.tile.indices) != len(issue.tile.indices):
        return None
    if structural is None:
        return None
    if domains_disjoint:
        return None
    loop_axis_ids = {id(axis) for axis in loop_axes}
    distances = {}
    equalities = []
    for left, right in zip(complete.tile.indices, issue.tile.indices):
        equalities.append(EffectEquality(left, right))
        lhs, rhs = _affine(left), _affine(right)
        if lhs is None or rhs is None or lhs[0] != rhs[0]:
            continue
        active = [
            (axis, coefficient)
            for axis, coefficient in lhs[0]
            if id(axis) in loop_axis_ids and coefficient
        ]
        if not active:
            if lhs[1] != rhs[1]:
                return None
            continue
        if len(active) != 1:
            continue
        axis, coefficient = active[0]
        difference = lhs[1] - rhs[1]
        if difference % coefficient:
            # The affine equality is an impossible integer congruence.
            return None
        distance = difference // coefficient
        prior = distances.setdefault(axis, distance)
        if prior != distance:
            # Two coordinates require contradictory distances on one axis.
            return None
    conservative = False
    if structural != -1 and not any(distances.values()):
        if not loop_axes:
            return None
        conservative = True
    if distances and next(
        (value for value in distances.values() if value),
        0,
    ) < 0:
        conservative = True
    complete_agents = {id(axis): axis for axis, _binding in complete_domain.agent_axes}
    issue_agents = {id(axis): axis for axis, _binding in issue_domain.agent_axes}
    for axis_id in complete_agents.keys() & issue_agents.keys():
        equalities.append(EffectEquality(
            Expr.axis_value(complete_agents[axis_id]),
            Expr.axis_value(issue_agents[axis_id]),
        ))
    relation = SymbolicRelation(
        complete_domain,
        issue_domain,
        tuple(equalities),
        tuple(
            IterationConstraint(axis, distance)
            for axis, distance in distances.items()
        ),
        conservative,
    )
    if not _finite_relation_may_overlap(relation):
        return None
    if not complete.occurrence._program.params:
        candidate = Dependence(complete, issue, relation)
        if not dependence_points(candidate):
            return None
    return relation


def _constant_effect_index(grouped, constants):
    index = {}
    for effect in grouped:
        values = constants[id(effect)]
        bucket = index.setdefault(
            len(values),
            {
                "effects": [],
                "wildcards": [set() for _value in values],
                "values": [{} for _value in values],
            },
        )
        position = len(bucket["effects"])
        bucket["effects"].append(effect)
        for coordinate, value in enumerate(values):
            if value is None:
                bucket["wildcards"][coordinate].add(position)
            else:
                bucket["values"][coordinate].setdefault(
                    value,
                    set(),
                ).add(position)
    return index


def _constant_candidates(effect, index, constants):
    values = constants[id(effect)]
    bucket = index.get(len(values))
    if bucket is None:
        return ()
    selected = None
    for coordinate, value in enumerate(values):
        if value is None:
            continue
        matching = set(bucket["wildcards"][coordinate])
        matching.update(
            bucket["values"][coordinate].get(value, ())
        )
        selected = matching if selected is None else selected & matching
        if not selected:
            return ()
    if selected is None:
        return tuple(bucket["effects"])
    return tuple(
        bucket["effects"][position]
        for position in sorted(selected)
    )


def dependences(program: Program) -> tuple[Dependence, ...]:
    """Derive completion-to-issue RAW/WAR relations from Program effects."""

    all_effects = effects(program)
    domains = {
        id(effect): effect_domain(effect)
        for effect in all_effects
    }
    constants = {
        id(effect): tuple(
            expression.constant_value
            for expression in effect.tile.indices
        )
        for effect in all_effects
    }
    groups = {}
    for effect in all_effects:
        groups.setdefault(
            (id(effect.tile.operand), effect.level),
            [],
        ).append(effect)
    result = []
    contexts = {}
    disjoint = {}
    for grouped in groups.values():
        reads = tuple(
            effect for effect in grouped
            if effect.kind is READ
        )
        writes = tuple(
            effect for effect in grouped
            if effect.kind is WRITE
        )
        read_index = _constant_effect_index(reads, constants)
        write_index = _constant_effect_index(writes, constants)
        for complete in grouped:
            issues = _constant_candidates(
                complete,
                write_index if complete.kind is READ else read_index,
                constants,
            )
            for issue in issues:
                if complete.occurrence.path == issue.occurrence.path:
                    continue
                context_key = (
                    id(complete.occurrence),
                    id(issue.occurrence),
                )
                if context_key not in contexts:
                    contexts[context_key] = (
                        _path_order(
                            complete.occurrence.path,
                            issue.occurrence.path,
                        ),
                        _serial_axes(complete, issue),
                    )
                structural, loop_axes = contexts[context_key]
                if structural is None:
                    continue
                disjoint_key = (
                    id(domains[id(complete)]),
                    id(domains[id(issue)]),
                    tuple(id(axis) for axis in loop_axes),
                )
                if disjoint_key not in disjoint:
                    disjoint[disjoint_key] = _domains_are_disjoint(
                        domains[id(complete)],
                        domains[id(issue)],
                        loop_axes,
                    )
                domain = _relation(
                    complete,
                    issue,
                    domains[id(complete)],
                    domains[id(issue)],
                    structural,
                    loop_axes,
                    disjoint[disjoint_key],
                )
                if domain is not None:
                    result.append(Dependence(complete, issue, domain))
    result = tuple(result)
    return result


def _dependence_replacements(dependence):
    replacements = {}
    for predicate in dependence.domain.issue_domain.predicates:
        if not isinstance(predicate, Compare) or predicate.op != "==":
            continue
        if predicate.lhs.op == "axis" \
                and predicate.rhs.constant_value is not None:
            replacements[predicate.lhs.args[0]] = \
                predicate.rhs.constant_value
        elif predicate.rhs.op == "axis" \
                and predicate.lhs.constant_value is not None:
            replacements[predicate.rhs.args[0]] = \
                predicate.lhs.constant_value
    for axis, value in dependence.domain.issue_domain.bindings:
        replacements[axis] = value.substitute(replacements)
    replacements.update({
        axis: 0
        for axis in dependence.domain.issue_domain.local_axes
    })
    return replacements


def dependence_logical_tile(
    dependence: Dependence,
) -> tuple[Expr, ...]:
    """Logical issue tile selected by one dependence occurrence."""

    replacements = _dependence_replacements(dependence)
    return tuple(
        expression.substitute(replacements)
        for expression in dependence.issue.tile.local_indices
    )


def dependence_runtime_generation(
    dependence: Dependence,
) -> tuple[Expr, ...]:
    """Logical runtime-tile offsets selected by one dependence occurrence."""

    replacements = _dependence_replacements(dependence)
    runtime = tuple(
        axis
        for axis in dependence.issue.tile.operand.axes
        if axis.extent.free_params()
    )
    replacements.update({axis: 0 for axis in runtime})
    return tuple(
        expression.substitute(replacements)
        for axis, expression in zip(
            dependence.issue.tile.operand.axes,
            dependence.issue.tile.local_indices,
        )
        if axis.extent.free_params()
    )


def required_dependences(
    program: Program,
    derived: tuple[Dependence, ...] | None = None,
) -> tuple[Dependence, ...]:
    """Derived completion-to-issue waits after logical completion coalescing."""

    if derived is None:
        derived = dependences(program)
    waiters = {}
    result = []
    operation_regions = {}

    def collect_regions(body, active=None, path=()):
        current = active
        for index, node in enumerate(body):
            node_path = path + (("body", index),)
            if isinstance(node, Region):
                current = node.tag
                continue
            if isinstance(node, (Movement, Compute)):
                operation_regions[node_path] = current
            if isinstance(node, Loop):
                collect_regions(
                    node.body,
                    current,
                    node_path + (("loop", 0),),
                )
            elif isinstance(node, Let):
                collect_regions(
                    node.body,
                    current,
                    node_path + (("let", 0),),
                )
            elif isinstance(node, Cond):
                collect_regions(
                    node.then_body,
                    current,
                    node_path + (("then", 0),),
                )
                collect_regions(
                    node.else_body,
                    current,
                    node_path + (("else", 0),),
                )
            elif isinstance(node, Switch):
                for case_index, (_value, case) in enumerate(node.cases):
                    collect_regions(
                        case,
                        current,
                        node_path + (("case", case_index),),
                    )
                collect_regions(
                    node.otherwise,
                    current,
                    node_path + (("otherwise", 0),),
                )

    collect_regions(program.body)

    def directly_dominates(first, second):
        return _path_order(first.path, second.path) == -1 \
            or _switch_case_precedes(first, second)

    def same_runtime_generation(first, second):
        left = dependence_runtime_generation(first)
        right = dependence_runtime_generation(second)
        return len(left) == len(right) and all(
            expr_equivalent(a, b) for a, b in zip(left, right)
        )

    def same_completion_value(first, second):
        if first.issue.tile.operand is not second.issue.tile.operand:
            return same_runtime_generation(first, second)
        left = dependence_logical_tile(first)
        right = dependence_logical_tile(second)
        return len(left) == len(right) and all(
            expr_equivalent(a, b) for a, b in zip(left, right)
        )

    for dependence in sorted(
        derived,
        key=lambda item: item.issue.occurrence.path,
    ):
        if not (
            dependence.kind is RAW
            and dependence.complete.level is SHARED
            and dependence.issue.level is SHARED
            and dependence.complete.operation.completion is not IMMEDIATE
        ):
            result.append(dependence)
            continue
        key = (
            dependence.complete.operation.completion,
            operation_regions.get(dependence.issue.occurrence.path),
        )
        prior = waiters.setdefault(key, [])
        region = operation_regions.get(dependence.issue.occurrence.path)
        redundant = bool(prior) if region is PROLOGUE else any(
            directly_dominates(other.issue.occurrence, dependence.issue.occurrence)
            and same_completion_value(other, dependence)
            for other in prior
        )
        if not redundant:
            result.append(dependence)
            prior.append(dependence)
    result = tuple(result)
    return result


__all__ = [
    "AccessKind", "READ", "WRITE",
    "MovementSource", "MovementDestination", "ComputeRead", "ComputeWrite",
    "Endpoint", "EffectInstance", "effect_domain",
    "effects_of_occurrence", "effects", "effect_points",
    "DependenceKind", "RAW", "WAR",
    "EffectEquality", "IterationConstraint", "SymbolicRelation",
    "Dependence", "dependences", "dependence_points",
    "dependence_logical_tile",
    "dependence_runtime_generation",
    "required_dependences",
]
