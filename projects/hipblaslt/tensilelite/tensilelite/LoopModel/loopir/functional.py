# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Named in-place Program transformations for LoopIR generation."""

from __future__ import annotations

from collections.abc import Callable

from .values import IRTypeError, TransformRefused
from .expr import (
    And,
    Axis,
    FALSE,
    Not,
    Or,
    Pred,
    TRUE,
    as_pred,
)
from .model import (
    DRAIN,
    PROLOGUE,
    SHORT,
    STEADY,
    Compute,
    Cond,
    Effect,
    Let,
    Loop,
    LoopIR,
    Movement,
    Program,
    Region,
    Switch,
    Tile,
    Transfer,
    child_bodies,
    same_operation,
)
from .rewrite import _rebuild_children, insert, replace, rewrite
Transform = Callable[[Program], Program]


def _in_regions(active, regions) -> bool:
    if regions is None:
        return True
    if isinstance(regions, (tuple, set, frozenset)):
        return any(active is item for item in regions)
    return active is regions


def compose(*transforms: Transform) -> Transform:
    """Compose named ``Program -> Program`` functions left to right."""

    if not all(callable(transform) for transform in transforms):
        raise IRTypeError("compose expects only callable Program transformations")

    def composed(program: Program) -> Program:
        current = program
        for transform in transforms:
            current = transform(current)
            if not isinstance(current, Program):
                raise IRTypeError("a Program transformation returned a non-Program")
        return current

    return composed


def append(program: Program, nodes: tuple[LoopIR, ...]) -> Program:
    """Append nodes to the root body."""

    return insert(program, program.body, len(program.body), nodes)


def _reorder(program, parent_body, nodes, *, first):
    if not isinstance(parent_body, tuple):
        raise IRTypeError("the selected body must be a Program body tuple")
    if not isinstance(nodes, tuple) or not nodes:
        raise IRTypeError("nodes must be a non-empty tuple")
    selected = {id(node) for node in nodes}
    if len(selected) != len(nodes) or any(
        not any(candidate is node for candidate in parent_body)
        for node in nodes
    ):
        raise TransformRefused(
            "defer/hoist nodes must be distinct direct children of the body"
        )
    hits = 0

    def edit_body(body):
        nonlocal hits
        if body is parent_body:
            hits += 1
            chosen = tuple(node for node in body if id(node) in selected)
            rest = tuple(node for node in body if id(node) not in selected)
            return chosen + rest if first else rest + chosen
        return tuple(
            _rebuild_children(node, edit_body)[0]
            for node in body
        )

    body = edit_body(program.body)
    if hits != 1:
        raise TransformRefused(
            f"selected body must occur exactly once, found {hits}"
        )
    return program.with_body(body, check=False)


def defer(program: Program, body, nodes: tuple[LoopIR, ...]) -> Program:
    """Move selected direct children to the end of one AST body."""

    return _reorder(program, body, nodes, first=False)


def hoist(program: Program, body, nodes: tuple[LoopIR, ...]) -> Program:
    """Move selected direct children to the start of one AST body."""

    return _reorder(program, body, nodes, first=True)


def unroll(program: Program, loop: Loop) -> Program:
    """Replace one constant-bounded Loop with value-binding Let bodies."""

    if not isinstance(loop, Loop):
        raise IRTypeError("unroll expects a Loop")
    lo = loop.lo.constant_value
    hi = loop.hi.constant_value
    if lo is None or hi is None:
        raise TransformRefused("unroll requires constant loop bounds")
    bodies = []
    for value in range(lo, hi):
        body = _specialize_body(loop.body, {loop.axis: value})
        bodies.append(Let(loop.axis, value, body))
    return replace(program, loop, tuple(bodies))


def _repeat_steady(program: Program, copies: int) -> Program:
    """Materialize the target-derived steady ring period in Program structure."""

    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise IRTypeError("steady copy count must be a positive integer")
    if copies == 1:
        return program
    copy_axis = Axis("_steady_copy", copies)
    changed = 0

    def edit_body(body, active=None):
        nonlocal changed
        current = active
        output = []
        for node in body:
            if isinstance(node, Region):
                current = node.tag
                output.append(node)
                continue
            if isinstance(node, Loop):
                child = edit_body(node.body, current)
                if current is STEADY and node.hi.free_params():
                    child = tuple(
                        Let(copy_axis, index, child)
                        for index in range(copies)
                    )
                    changed += 1
                node = Loop(node.axis, node.lo, node.hi, child)
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current),
                    edit_body(node.else_body, current),
                )
            elif isinstance(node, Let):
                node = Let(node.axis, node.value, edit_body(node.body, current))
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current),
                )
            output.append(node)
        return tuple(output)

    body = edit_body(program.body)
    if changed != 1:
        raise TransformRefused(
            f"expected one symbolic steady Loop, found {changed}"
        )
    axes = (*program.axes, copy_axis)
    program.axes = axes
    return program.with_body(body, check=False)


def _replace_predicate(original: Pred, extra: Pred) -> Pred:
    return extra if original is TRUE else And(original, extra)


def _restrict_effect(effect: Effect, predicate: Pred) -> Effect:
    return Effect(
        effect.tile,
        _replace_predicate(effect.where, predicate),
    )


def partition_movement(
    program: Program,
    movement: Movement,
    predicates: tuple[Pred, ...],
    completions: tuple,
) -> Program:
    """Replace one Movement with independently-issued domain partitions.

    This is a generic checked structural primitive.  TensorLoad-side
    generation supplies TDMSplit predicates and completion values.
    """

    if not isinstance(movement, Movement):
        raise IRTypeError("partition_movement expects a Movement")
    if not isinstance(predicates, tuple) or not predicates:
        raise IRTypeError("partition predicates must be a non-empty tuple")
    if not isinstance(completions, tuple) or len(completions) != len(predicates):
        raise IRTypeError("partition completions must match predicates")
    parts = tuple(
        Movement(
            movement.op,
            tuple(
                Transfer(
                    transfer.local_tile,
                    transfer.source,
                    transfer.destination,
                    (
                        _restrict_effect(transfer.effect, as_pred(predicate))
                        if transfer.effect is not None else None
                    ),
                )
                for transfer in movement.transfers
            ),
            completion,
            effect=(
                _restrict_effect(movement.effect, as_pred(predicate))
                if movement.effect is not None else None
            ),
        )
        for predicate, completion in zip(predicates, completions)
    )
    return replace(program, movement, parts)


def fuse_movements(
    program: Program,
    movements: tuple[Movement, ...],
    completion,
) -> Program:
    """Fuse adjacent compatible Movements into one issue/completion."""

    if not isinstance(movements, tuple) or len(movements) < 2 \
            or not all(isinstance(value, Movement) for value in movements):
        raise IRTypeError("fuse_movements requires at least two Movements")
    if len({movement.op for movement in movements}) != 1:
        raise TransformRefused("fused Movements must have the same kind")
    transfers = []
    for movement in movements:
        for index, transfer in enumerate(movement.transfers):
            transfers.append(Transfer(
                transfer.local_tile,
                transfer.source,
                transfer.destination,
                movement.effect_for(index),
            ))
    fused = Movement(movements[0].op, tuple(transfers), completion)
    return replace(program, movements, (fused,))


def _substitute_tile(tile: Tile, replacements) -> Tile:
    return Tile(
        tile.operand,
        tuple(expression.substitute(replacements) for expression in tile.indices),
    )


def _substitute_operation(operation, replacements):
    if isinstance(operation, Movement):
        return Movement(
            operation.op,
            tuple(
                Transfer(
                    _substitute_tile(transfer.local_tile, replacements),
                    transfer.source,
                    transfer.destination,
                    (
                        Effect(
                            _substitute_tile(
                                transfer.effect.tile,
                                replacements,
                            ),
                            transfer.effect.where.substitute(replacements),
                        )
                        if transfer.effect is not None else None
                    ),
                )
                for transfer in operation.transfers
            ),
            operation.completion,
            effect=(
                Effect(
                    _substitute_tile(operation.effect.tile, replacements),
                    operation.effect.where.substitute(replacements),
                )
                if operation.effect is not None else None
            ),
            _identity=operation._identity,
        )
    if isinstance(operation, Compute):
        return Compute(
            operation.op,
            tuple(_substitute_tile(value, replacements) for value in operation.reads),
            tuple(_substitute_tile(value, replacements) for value in operation.writes),
            operation.completion,
            _identity=operation._identity,
        )
    raise IRTypeError("operation substitution expects Movement or Compute")


def _specialize_predicate(predicate, bindings):
    predicate = predicate.substitute(bindings)
    if isinstance(predicate, And):
        operands = []
        for operand in predicate.operands:
            simplified, verdict = _specialize_predicate(operand, {})
            if verdict is False:
                return FALSE, False
            if verdict is None:
                operands.append(simplified)
        if not operands:
            return TRUE, True
        if len(operands) == 1:
            return operands[0], None
        return And(*operands), None
    if isinstance(predicate, Or):
        operands = []
        for operand in predicate.operands:
            simplified, verdict = _specialize_predicate(operand, {})
            if verdict is True:
                return TRUE, True
            if verdict is None:
                operands.append(simplified)
        if not operands:
            return FALSE, False
        if len(operands) == 1:
            return operands[0], None
        return Or(*operands), None
    if isinstance(predicate, Not):
        operand, verdict = _specialize_predicate(predicate.operand, {})
        if verdict is not None:
            return (FALSE, False) if verdict else (TRUE, True)
        return ~operand, None
    verdict = predicate.evaluate({})
    if verdict is True:
        return TRUE, True
    if verdict is False:
        return FALSE, False
    return predicate, None


def _specialize_body(nodes, bindings):
    """Apply known finite-axis values and remove control that they resolve."""

    output = []
    for node in nodes:
        if isinstance(node, (Movement, Compute)):
            output.append(_substitute_operation(node, bindings))
        elif isinstance(node, Let):
            value = node.value.substitute(bindings)
            if value.constant_value is None:
                raise TransformRefused("specialized Let is not constant")
            body = _specialize_body(
                node.body,
                {**bindings, node.axis: value.constant_value},
            )
            output.append(Let(node.axis, value, body))
        elif isinstance(node, Loop):
            lower = node.lo.substitute(bindings)
            upper = node.hi.substitute(bindings)
            if lower.constant_value is not None \
                    and upper.constant_value == lower.constant_value + 1:
                output.append(Let(
                    node.axis,
                    lower.constant_value,
                    _specialize_body(
                        node.body,
                        {**bindings, node.axis: lower.constant_value},
                    ),
                ))
            else:
                output.append(Loop(
                    node.axis,
                    lower,
                    upper,
                    _specialize_body(node.body, bindings),
                ))
        elif isinstance(node, Cond):
            predicate, verdict = _specialize_predicate(node.pred, bindings)
            if verdict is not None:
                output.extend(_specialize_body(
                    node.then_body if verdict else node.else_body,
                    bindings,
                ))
                continue
            then_body = _specialize_body(node.then_body, bindings)
            else_body = _specialize_body(node.else_body, bindings)
            if not else_body and len(then_body) == 1 \
                    and isinstance(then_body[0], Cond) \
                    and not then_body[0].else_body:
                inner = then_body[0]
                predicate, _verdict = _specialize_predicate(
                    predicate & inner.pred,
                    {},
                )
                output.append(Cond(
                    predicate,
                    inner.then_body,
                ))
            elif then_body or else_body:
                output.append(Cond(predicate, then_body, else_body))
        elif isinstance(node, Switch):
            expression = node.expression.substitute(bindings)
            if expression.constant_value is not None:
                output.extend(_specialize_body(
                    dict(node.cases).get(
                        expression.constant_value,
                        node.otherwise,
                    ),
                    bindings,
                ))
            else:
                output.append(Switch(
                    expression,
                    tuple(
                        (value, _specialize_body(case, bindings))
                        for value, case in node.cases
                    ),
                    _specialize_body(node.otherwise, bindings),
                ))
        else:
            output.append(node)
    return tuple(output)


def substitute_operation(operation, replacements):
    """Return one operation with Axis expressions substituted, preserving identity."""

    return _substitute_operation(operation, dict(replacements))


def _operations_in(nodes):
    return tuple(
        operation
        for node in nodes
        for operation in (
            (node,)
            if isinstance(node, (Movement, Compute))
            else _operations_in(
                tuple(
                    item
                    for body in child_bodies(node)
                    for item in body
                )
            )
        )
    )


def _operation_identities(node):
    return {
        operation._identity
        for operation in _operations_in((node,))
    }


def _flatten_finite(nodes, bindings):
    output = []
    for node in nodes:
        if isinstance(node, (Movement, Compute)):
            output.append(_substitute_operation(node, bindings))
        elif isinstance(node, Let):
            value = node.value.substitute(bindings)
            if value.constant_value is None:
                raise TransformRefused("materialized Let is not constant")
            output.extend(_flatten_finite(
                node.body,
                {**bindings, node.axis: value.constant_value},
            ))
        elif isinstance(node, Loop):
            lower = node.lo.substitute(bindings)
            upper = node.hi.substitute(bindings)
            if lower.constant_value is None or upper.constant_value is None:
                raise TransformRefused("materialized Loop is not finite")
            for value in range(lower.constant_value, upper.constant_value):
                output.extend(_flatten_finite(
                    node.body,
                    {**bindings, node.axis: value},
                ))
        elif isinstance(node, Cond):
            predicate = node.pred.substitute(bindings)
            verdict = predicate.evaluate({})
            if verdict is None:
                raise TransformRefused("materialized Cond is unresolved")
            output.extend(_flatten_finite(
                node.then_body if verdict else node.else_body,
                bindings,
            ))
        elif isinstance(node, Switch):
            expression = node.expression.substitute(bindings)
            if expression.constant_value is None:
                raise TransformRefused("materialized Switch is unresolved")
            output.extend(_flatten_finite(
                dict(node.cases).get(
                    expression.constant_value,
                    node.otherwise,
                ),
                bindings,
            ))
    return tuple(output)


def _materialize_finite_units(nodes, bindings):
    """Specialize a finite body into one binding-preserving unit per operation."""
    output = []

    def bound(operation, environment):
        node = _substitute_operation(operation, environment)
        for axis, value in reversed(tuple(environment.items())):
            node = Let(axis, value, (node,))
        return node

    for node in nodes:
        if isinstance(node, (Movement, Compute)):
            output.append(bound(node, bindings))
        elif isinstance(node, Let):
            value = node.value.substitute(bindings)
            if value.constant_value is None:
                raise TransformRefused("materialized Let is not constant")
            output.extend(_materialize_finite_units(
                node.body,
                {**bindings, node.axis: value.constant_value},
            ))
        elif isinstance(node, Loop):
            lower = node.lo.substitute(bindings)
            upper = node.hi.substitute(bindings)
            if lower.constant_value is None or upper.constant_value is None:
                raise TransformRefused("materialized Loop is not finite")
            for value in range(lower.constant_value, upper.constant_value):
                output.extend(_materialize_finite_units(
                    node.body,
                    {**bindings, node.axis: value},
                ))
        elif isinstance(node, Cond):
            predicate = node.pred.substitute(bindings)
            verdict = predicate.evaluate({})
            if verdict is None:
                raise TransformRefused("materialized Cond is unresolved")
            output.extend(_materialize_finite_units(
                node.then_body if verdict else node.else_body,
                bindings,
            ))
        elif isinstance(node, Switch):
            expression = node.expression.substitute(bindings)
            if expression.constant_value is None:
                raise TransformRefused("materialized Switch is unresolved")
            output.extend(_materialize_finite_units(
                dict(node.cases).get(
                    expression.constant_value,
                    node.otherwise,
                ),
                bindings,
            ))
    return tuple(output)


def rewrite_operation(
    program: Program,
    operation: Movement | Compute,
    rule: Callable[[Movement | Compute], Movement | Compute],
) -> Program:
    """Apply a typed semantic operation rewrite to every logical occurrence."""

    if not callable(rule):
        raise IRTypeError("rewrite_operation.rule must be callable")

    def apply(node):
        if isinstance(node, (Movement, Compute)) and same_operation(node, operation):
            replacement = rule(node)
            if not isinstance(replacement, (Movement, Compute)):
                raise IRTypeError("operation rewrite must return an operation")
            return replacement
        return node

    return rewrite(program, apply)


def _operation_map(body):
    result = {}

    def visit(nodes):
        for node in nodes:
            if isinstance(node, (Movement, Compute)):
                result.setdefault(node._identity, node)
            if isinstance(node, Loop):
                visit(node.body)
            elif isinstance(node, Cond):
                visit(node.then_body)
                visit(node.else_body)
            elif isinstance(node, Let):
                visit(node.body)
            elif isinstance(node, Switch):
                for _value, case in node.cases:
                    visit(case)
                visit(node.otherwise)

    visit(body)
    return result


def _pipeline_body(body, include, variants):
    output = []
    for node in body:
        if isinstance(node, (Movement, Compute)):
            if include(node._identity):
                output.append(variants.get(node._identity, node))
            continue
        if isinstance(node, Region):
            output.append(node)
        elif isinstance(node, Loop):
            child = _pipeline_body(node.body, include, variants)
            if child:
                output.append(Loop(
                    node.axis, node.lo, node.hi, child,
                ))
        elif isinstance(node, Let):
            child = _pipeline_body(node.body, include, variants)
            if child:
                output.append(Let(node.axis, node.value, child))
        elif isinstance(node, Cond):
            then_body = _pipeline_body(node.then_body, include, variants)
            else_body = _pipeline_body(node.else_body, include, variants)
            if then_body or else_body:
                output.append(Cond(node.pred, then_body, else_body))
        elif isinstance(node, Switch):
            cases = tuple(
                (value, _pipeline_body(case, include, variants))
                for value, case in node.cases
            )
            otherwise = _pipeline_body(node.otherwise, include, variants)
            if any(case for _value, case in cases) or otherwise:
                output.append(Switch(node.expression, cases, otherwise))
    return tuple(output)


def pipeline(
    program: Program,
    loop: Loop,
    specifications: tuple[tuple[Movement | Compute, int], ...],
) -> Program:
    """Build prologue/steady/drain/short structure from operation advances.

    Each specification names a logical operation and a non-negative advance
    along ``loop.axis``.  The result contains only ordinary LoopIR nodes and
    Region markers.
    """

    if not isinstance(loop, Loop):
        raise TransformRefused("pipeline requires one serial Loop")
    if not isinstance(specifications, tuple):
        raise IRTypeError("pipeline specifications must be a tuple")
    originals = _operation_map(loop.body)
    depths = {}
    variants = {}
    for operation, depth in specifications:
        if not isinstance(operation, (Movement, Compute)):
            raise IRTypeError("pipeline specifications require operations")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise IRTypeError("pipeline depth must be a non-negative integer")
        if operation._identity not in originals:
            raise TransformRefused("pipeline operation is not inside the selected Loop")
        if depth:
            depths[operation._identity] = depth
            variants[operation._identity] = _substitute_operation(
                originals[operation._identity],
                {loop.axis: loop.axis + depth},
            )
    maximum = max(depths.values(), default=0)
    if maximum == 0:
        return program

    prologue = []
    for step in range(maximum):
        threshold = maximum - 1 - step
        included = {
            identity for identity, depth in depths.items()
            if depth > threshold
        }
        body = _pipeline_body(
            loop.body,
            lambda identity: identity in included,
            variants,
        )
        if body:
            prologue.append(Let(loop.axis, loop.lo + step - maximum, body))

    steady = Loop(
        loop.axis,
        loop.lo,
        loop.hi - maximum,
        _pipeline_body(loop.body, lambda _identity: True, variants),
    )

    selected = set(depths)
    drain = []
    for step in range(maximum):
        limit = maximum - 1 - step
        included = {
            identity for identity, depth in depths.items()
            if depth <= limit
        }
        body = _pipeline_body(
            loop.body,
            lambda identity: identity not in selected or identity in included,
            {**originals, **{
                identity: variants[identity]
                for identity in included
            }},
        )
        if body:
            drain.append(Let(loop.axis, loop.hi - maximum + step, body))

    replacement = Cond(
        loop.hi - loop.lo >= maximum,
        (
            Region(PROLOGUE),
            *prologue,
            Region(STEADY),
            steady,
            Region(DRAIN),
            *drain,
        ),
        (
            Region(SHORT),
            Loop(
                loop.axis, loop.lo, loop.hi, loop.body,
            ),
        ),
    )
    return replace(program, loop, (replacement,))


def _drop_drain(program: Program) -> Program:
    """Remove pipeline drain/short arms while retaining prologue and steady.

    The result contains the same ordinary Region markers and Loop nodes; this
    transform adds no special no-load-loop node to LoopIR.
    """

    def rule(node):
        if not isinstance(node, Cond):
            return node
        tags = {
            candidate.tag
            for candidate in node.then_body
            if isinstance(candidate, Region)
        }
        if not {PROLOGUE, STEADY, DRAIN} <= tags:
            return node
        original_loop = next(
            (
                candidate for candidate in node.else_body
                if isinstance(candidate, Loop)
            ),
            None,
        )
        kept = []
        for candidate in node.then_body:
            if isinstance(candidate, Region) and candidate.tag is DRAIN:
                break
            if (
                original_loop is not None
                and isinstance(candidate, Loop)
                and candidate.axis is original_loop.axis
            ):
                candidate = Loop(
                    candidate.axis,
                    candidate.lo,
                    original_loop.hi,
                    candidate.body,
                )
            kept.append(candidate)
        return tuple(kept)

    return rewrite(program, rule)


def _defer_paths(
    program: Program,
    *,
    regions,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Move selected operations to body ends through one-branch wrappers."""

    identities = {operation._identity for operation in operations}
    selected = {}

    def selected_identity(node):
        original = node
        cached = selected.get(id(original), ...)
        if cached is not ...:
            return cached
        while True:
            if isinstance(node, (Movement, Compute)):
                result = (
                    node._identity
                    if node._identity in identities else None
                )
                selected[id(original)] = result
                return result
            if isinstance(node, Loop) and len(node.body) == 1:
                node = node.body[0]
                continue
            if (
                isinstance(node, Cond)
                and not node.else_body
                and len(node.then_body) == 1
            ):
                node = node.then_body[0]
                continue
            selected[id(original)] = None
            return None

    def edit_body(body, active_region=None):
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
                continue
            node = _rebuild_children(
                node,
                lambda nested: edit_body(nested, current_region),
            )[0]
            rebuilt.append(node)
        def reordered(nodes):
            return (
                tuple(
                    node for node in nodes
                    if selected_identity(node) is None
                )
                + tuple(
                    node for node in nodes
                    if selected_identity(node) is not None
                )
            )

        if not any(isinstance(node, Region) for node in rebuilt):
            return (
                reordered(tuple(rebuilt))
                if _in_regions(active_region, regions) else tuple(rebuilt)
            )
        output = []
        index = 0
        segment_region = active_region
        while index < len(rebuilt):
            node = rebuilt[index]
            if isinstance(node, Region):
                segment_region = node.tag
                output.append(node)
                index += 1
                continue
            stop = index
            while stop < len(rebuilt) \
                    and not isinstance(rebuilt[stop], Region):
                stop += 1
            segment = tuple(rebuilt[index:stop])
            output.extend(
                reordered(segment)
                if _in_regions(segment_region, regions) else segment
            )
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _defer_loops(
    program: Program,
    *,
    regions,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Move selected operation-only children to each selected Loop's end."""

    identities = {
        operation._identity
        for operation in operations
        if isinstance(operation, (Movement, Compute))
    }
    if not identities:
        return program

    selected = {}

    def selected_unit(node):
        key = id(node)
        if key not in selected:
            present = _operation_identities(node)
            selected[key] = bool(present) and present <= identities
        return selected[key]

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            rebuilt = node
            if isinstance(node, Loop):
                child = edit_body(node.body, current_region)
                if _in_regions(current_region, regions):
                    leading = tuple(
                        item for item in child
                        if not selected_unit(item)
                    )
                    trailing = tuple(
                        item for item in child
                        if selected_unit(item)
                    )
                    child = leading + trailing
                node.body = child
                rebuilt = node
            else:
                rebuilt = _rebuild_children(
                    node,
                    lambda nested: edit_body(nested, current_region),
                )[0]
            output.append(rebuilt)
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _hoist_region(
    program: Program,
    *,
    regions,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Move selected operation-only children to selected region body starts."""

    identities = {operation._identity for operation in operations}
    if not identities:
        return program

    selected = {}

    def selected_unit(node):
        key = id(node)
        if key not in selected:
            present = _operation_identities(node)
            selected[key] = bool(present) and present <= identities
        return selected[key]

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            node = _rebuild_children(
                node,
                lambda nested: edit_body(nested, current_region),
            )[0]
            output.append(node)
        output = tuple(output)
        if not _in_regions(current_region, regions):
            return output
        return (
            tuple(item for item in output if selected_unit(item))
            + tuple(item for item in output if not selected_unit(item))
        )

    return program.with_body(edit_body(program.body), check=False)


def _unroll_region(
    program: Program,
    *,
    region,
    selector: Callable[[Movement | Compute], bool],
    key: Callable[[Movement | Compute], object],
) -> Program:
    """Unroll and sort an operation-only region using caller-owned policy.

    The structural rewrite is generic.  Theta generation supplies the
    operation selector and ordering key; lowering never reconstructs them.
    """

    if not callable(selector) or not callable(key):
        raise IRTypeError("unroll_region expects callables")

    def edit_body(body):
        body = tuple(
            _rebuild_children(node, edit_body)[0]
            for node in body
        )
        output = []
        index = 0
        while index < len(body):
            node = body[index]
            output.append(node)
            if not isinstance(node, Region) or node.tag is not region:
                index += 1
                continue
            stop = index + 1
            while stop < len(body) and not isinstance(body[stop], Region):
                stop += 1
            segment = body[index + 1:stop]
            found = _operations_in(segment)
            if found and all(selector(operation) for operation in found):
                materialized = _flatten_finite(segment, {})
                output.extend(sorted(materialized, key=key))
            else:
                output.extend(segment)
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _unroll_runs(
    program: Program,
    *,
    regions,
    selector: Callable[[Movement | Compute], bool],
    key: Callable[[Movement | Compute], object],
) -> Program:
    """Unroll and sort consecutive structural runs of selected operations."""

    selected = {}

    def selection_status(node):
        key = id(node)
        cached = selected.get(key, ...)
        if cached is not ...:
            return cached
        if isinstance(node, (Movement, Compute)):
            result = bool(selector(node))
        else:
            found = False
            result = None
            for body in child_bodies(node):
                for child in body:
                    status = selection_status(child)
                    if status is False:
                        result = False
                        break
                    if status is True:
                        found = True
                if result is False:
                    break
            if result is not False:
                result = True if found else None
        selected[key] = result
        return result

    def selected_unit(node):
        return selection_status(node) is True

    def edit_body(body, active_region=None):
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
                continue
            node = _rebuild_children(
                node,
                lambda child: edit_body(child, current_region),
            )[0]
            rebuilt.append(node)
        body = tuple(rebuilt)
        if not _in_regions(current_region, regions):
            return body
        output = []
        index = 0
        while index < len(body):
            if not selected_unit(body[index]):
                output.append(body[index])
                index += 1
                continue
            stop = index
            while stop < len(body) and selected_unit(body[stop]):
                stop += 1
            try:
                materialized = _flatten_finite(body[index:stop], {})
            except TransformRefused:
                output.extend(body[index:stop])
            else:
                output.extend(sorted(materialized, key=key))
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _schedule_runs(
    program: Program,
    *,
    regions,
    selector: Callable[[Movement | Compute], bool],
    schedule: Callable[
        [tuple[Movement | Compute, ...]],
        tuple[Movement | Compute, ...] | None,
    ],
) -> Program:
    """Materialize and schedule finite runs; ``None`` preserves an unchanged run."""

    if not callable(selector) or not callable(schedule):
        raise IRTypeError("_schedule_runs expects callable selector and schedule")
    selected = {}

    def selection_status(node):
        key = id(node)
        cached = selected.get(key, ...)
        if cached is not ...:
            return cached
        if isinstance(node, (Movement, Compute)):
            result = bool(selector(node))
        else:
            found = False
            result = None
            for body in child_bodies(node):
                for child in body:
                    status = selection_status(child)
                    if status is False:
                        result = False
                        break
                    if status is True:
                        found = True
                if result is False:
                    break
            if result is not False:
                result = True if found else None
        selected[key] = result
        return result

    def selected_unit(node):
        return selection_status(node) is True

    def edit_body(body, active_region=None, bindings=None):
        bindings = dict(bindings or {})
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
                continue
            if selected_unit(node):
                rebuilt.append(node)
                continue
            if isinstance(node, Let):
                value = node.value.constant_value
                child_bindings = bindings if value is None \
                    else {**bindings, node.axis: value}
                node = Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region, child_bindings),
                )
            else:
                node = _rebuild_children(
                    node,
                    lambda child: edit_body(child, current_region, bindings),
                )[0]
            rebuilt.append(node)
        body = tuple(rebuilt)
        if not _in_regions(current_region, regions):
            return body
        output = []
        index = 0
        while index < len(body):
            if not selected_unit(body[index]):
                output.append(body[index])
                index += 1
                continue
            stop = index
            while stop < len(body) and selected_unit(body[stop]):
                stop += 1
            try:
                materialized = _materialize_finite_units(
                    body[index:stop], bindings)
            except TransformRefused:
                output.extend(body[index:stop])
            else:
                scheduled = schedule(materialized)
                output.extend(
                    body[index:stop]
                    if scheduled is None
                    else scheduled
                )
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _append_region(
    program: Program,
    region,
    nodes: tuple[LoopIR, ...],
) -> Program:
    """Append nodes to every matching marker-delimited region in the Program."""

    if not isinstance(nodes, tuple) \
            or not all(isinstance(node, LoopIR) for node in nodes):
        raise IRTypeError("append_region.nodes must be LoopIR values")

    def edit_body(body):
        body = tuple(
            _rebuild_children(node, edit_body)[0]
            for node in body
        )
        output = []
        index = 0
        while index < len(body):
            marker = body[index]
            output.append(marker)
            index += 1
            if not isinstance(marker, Region) or marker.tag is not region:
                continue
            while index < len(body) and not isinstance(body[index], Region):
                output.append(body[index])
                index += 1
            output.extend(nodes)
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


def _rewrite_region(
    program: Program,
    region,
    replacements: dict,
) -> Program:
    """Replace logical operations only while a marker-delimited region is active."""

    replacements = dict(replacements)

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            if isinstance(node, (Movement, Compute)):
                replacement = (
                    replacements.get(node._identity)
                    if current_region is region else None
                )
                output.append(replacement or node)
            elif isinstance(node, Loop):
                output.append(Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Cond):
                output.append(Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                ))
            elif isinstance(node, Let):
                output.append(Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Switch):
                output.append(Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                ))
        return tuple(output)

    return program.with_body(edit_body(program.body), check=False)


__all__ = [
    "Transform", "compose", "append", "partition_movement",
    "fuse_movements",
    "substitute_operation", "rewrite_operation", "pipeline",
    "defer", "hoist", "unroll",
]
