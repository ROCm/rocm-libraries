# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The LoopIR walker and four structural edit primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .values import IRTypeError, TransformRefused
from .model import (
    Compute,
    Cond,
    Let,
    Loop,
    LoopIR,
    Movement,
    Node,
    Program,
    Switch,
    child_bodies,
    same_operation,
)


def walk(value: Program | tuple[Node, ...]) -> Iterator[Node]:
    """Yield every node in deterministic pre-order."""

    body = value.body if isinstance(value, Program) else value
    if not isinstance(body, tuple) or not all(isinstance(node, LoopIR) for node in body):
        raise IRTypeError("walk expects a Program or a tuple of nodes")
    for node in body:
        yield node
        for child in child_bodies(node):
            yield from walk(child)


def _normalize_rule_result(value) -> tuple[Node, ...]:
    if isinstance(value, LoopIR):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(node, LoopIR) for node in value):
        return value
    raise IRTypeError("a rewrite rule must return a node or a node tuple")


def _rebuild_children(node: Node, edit_body) -> tuple[Node, bool]:
    changed = False
    if isinstance(node, Loop):
        body = edit_body(node.body)
        changed = body is not node.body
        if changed:
            node.body = body
        return node, changed
    if isinstance(node, Cond):
        then_body = edit_body(node.then_body)
        else_body = edit_body(node.else_body)
        changed = then_body is not node.then_body or else_body is not node.else_body
        if changed:
            node.then_body = then_body
            node.else_body = else_body
        return node, changed
    if isinstance(node, Let):
        body = edit_body(node.body)
        changed = body is not node.body
        if changed:
            node.body = body
        return node, changed
    if isinstance(node, Switch):
        cases = tuple((value, edit_body(body)) for value, body in node.cases)
        otherwise = edit_body(node.otherwise)
        changed = otherwise is not node.otherwise or any(
            new_body is not old_body
            for (_value, new_body), (_old_value, old_body) in zip(cases, node.cases)
        )
        if changed:
            node.cases = cases
            node.otherwise = otherwise
        return node, changed
    return node, False


def rewrite(
    program: Program,
    rule: Callable[[Node], Node | tuple[Node, ...]],
) -> Program:
    """Apply one post-order rule recursively, mutating and returning Program."""

    if not isinstance(program, Program):
        raise IRTypeError("rewrite expects a Program")
    if not callable(rule):
        raise IRTypeError("rewrite.rule must be callable")

    def edit_body(body: tuple[Node, ...]) -> tuple[Node, ...]:
        output = []
        changed = False
        for original in body:
            rebuilt, child_changed = _rebuild_children(original, edit_body)
            replacement = _normalize_rule_result(rule(rebuilt))
            output.extend(replacement)
            if child_changed or len(replacement) != 1 or replacement[0] is not original:
                changed = True
        return tuple(output) if changed else body

    body = edit_body(program.body)
    if body is program.body:
        return program
    return program.with_body(body, check=False)


def _matches(candidate: Node, target: Node) -> bool:
    if isinstance(target, (Movement, Compute)):
        return same_operation(candidate, target)
    return candidate is target


def _target_sequence(old) -> tuple[Node, ...]:
    if isinstance(old, LoopIR):
        return (old,)
    if isinstance(old, tuple) and old and all(isinstance(node, LoopIR) for node in old):
        return old
    raise IRTypeError("replace.old must be a node or a non-empty tuple of nodes")


def _new_nodes(value) -> tuple[Node, ...]:
    if not isinstance(value, tuple) \
            or not all(isinstance(node, LoopIR) for node in value):
        raise IRTypeError("new_nodes must be a tuple containing only LoopIR nodes")
    return value


def replace(
    program: Program,
    old: Node | tuple[Node, ...],
    new_nodes: tuple[Node, ...],
) -> Program:
    """Replace matching nodes or one exact structural sequence."""

    if not isinstance(program, Program):
        raise IRTypeError("replace expects a Program")
    targets = _target_sequence(old)
    replacements = _new_nodes(new_nodes)
    target_count = len(targets)
    matches = 0

    def edit_body(body: tuple[Node, ...]) -> tuple[Node, ...]:
        nonlocal matches
        output = []
        changed = False
        index = 0
        while index < len(body):
            if index + target_count <= len(body) and all(
                _matches(body[index + offset], target)
                for offset, target in enumerate(targets)
            ):
                output.extend(replacements)
                index += target_count
                matches += 1
                changed = True
                continue
            original = body[index]
            rebuilt, child_changed = _rebuild_children(original, edit_body)
            output.append(rebuilt)
            changed = changed or child_changed
            index += 1
        return tuple(output) if changed else body

    body = edit_body(program.body)
    if matches == 0:
        raise TransformRefused("replace matched no nodes")
    return program.with_body(body, check=False)


def insert(
    program: Program,
    parent_body: tuple[Node, ...],
    index: int,
    new_nodes: tuple[Node, ...],
) -> Program:
    """Insert nodes into one body tuple identified by object identity."""

    if not isinstance(program, Program):
        raise IRTypeError("insert expects a Program")
    if not isinstance(parent_body, tuple):
        raise IRTypeError("insert.parent_body must be a body tuple from the Program")
    if not isinstance(index, int) or isinstance(index, bool):
        raise IRTypeError("insert.index must be an integer")
    replacements = _new_nodes(new_nodes)
    hits = 0

    def edit_body(body: tuple[Node, ...]) -> tuple[Node, ...]:
        nonlocal hits
        if body is parent_body:
            if index < 0 or index > len(body):
                raise TransformRefused(
                    f"insert index {index} is outside body bounds [0, {len(body)}]"
                )
            hits += 1
            return body[:index] + replacements + body[index:]
        output = []
        changed = False
        for original in body:
            rebuilt, child_changed = _rebuild_children(original, edit_body)
            output.append(rebuilt)
            changed = changed or child_changed
        return tuple(output) if changed else body

    body = edit_body(program.body)
    if hits == 0:
        raise TransformRefused("insert parent_body does not belong to this Program")
    if hits > 1:
        raise TransformRefused("insert parent_body is shared by several tree positions")
    return program.with_body(body, check=False)
