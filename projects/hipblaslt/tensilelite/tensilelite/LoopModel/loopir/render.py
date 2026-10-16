# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Total human-readable rendering for LoopIR."""

from __future__ import annotations

from .expr import TRUE
from .model import (
    REGISTER,
    Compute,
    Cond,
    Let,
    Loop,
    Movement,
    Program,
    Region,
    Switch,
)
from .values import IMMEDIATE


def _tile(tile):
    extra_count = len(tile.indices) - len(tile.operand.axes)
    extra = []
    for index, value in enumerate(tile.indices[:extra_count]):
        free_axes = tuple(value.free_axes())
        name = free_axes[0].name if len(free_axes) == 1 else f"extra{index}"
        extra.append((name, value))
    local = (
        (axis.name, value)
        for axis, value in zip(
            tile.operand.axes,
            tile.indices[extra_count:],
        )
    )
    coordinates = ", ".join(
        f"{name}={value.render()}"
        for name, value in (*extra, *local)
    )
    return f"{tile.operand.name}[{coordinates}]"


def _where(predicate):
    return (
        ""
        if predicate is TRUE or predicate.evaluate({}) is True
        else f" where {predicate.render()}"
    )


def _render_completion(value):
    if value is IMMEDIATE:
        return "immediate"
    return f"{value.name}@{value.counter_scope.name.lower()}"


def _transfer_label(transfer):
    return (
        "register"
        if transfer.source is REGISTER or transfer.destination is REGISTER
        else "local"
    )


def _render_movement(node, indent, out):
    pad = "  " * indent
    completion = _render_completion(node.completion)
    if len(node.transfers) == 1:
        transfer = node.transfers[0]
        out.append(
            f"{pad}{node.op.__name__.lower()} "
            f"{_transfer_label(transfer)}={_tile(transfer.local_tile)}  "
            f"{transfer.source.name.lower()} -> "
            f"{transfer.destination.name.lower()}  "
            f"completion={completion}"
        )
    else:
        out.append(
            f"{pad}{node.op.__name__.lower()} completion={completion}"
        )
        for transfer in node.transfers:
            out.append(
                f"{pad}  {_transfer_label(transfer)}="
                f"{_tile(transfer.local_tile)}  "
                f"{transfer.source.name.lower()} -> "
                f"{transfer.destination.name.lower()}"
            )
    for index in range(len(node.transfers)):
        effect = node.effect_for(index)
        out.append(
            f"{pad}  effect={_tile(effect.tile)}"
            f"{_where(effect.where)}"
        )


def _render_loop(node, indent, out):
    pad = "  " * indent
    loops = [node]
    body = node.body
    while len(body) == 1 and isinstance(body[0], Loop):
        loops.append(body[0])
        body = body[0].body
    clauses = ", ".join(
        f"{loop.axis.name} in "
        f"[{loop.lo.render()}, {loop.hi.render()})"
        for loop in loops
    )
    out.append(f"{pad}for {clauses}")
    _render_body(body, indent + 1, out)


def _render_switch(node, indent, out):
    pad = "  " * indent
    rendered = []
    for value, body in node.cases:
        lines = []
        _render_body(body, indent + 1, lines)
        rendered.append((value, body, tuple(lines)))
    if rendered and not node.otherwise \
            and len({lines for _value, _body, lines in rendered}) == 1:
        values = ", ".join(str(value) for value, _body, _lines in rendered)
        out.append(
            f"{pad}switch {node.expression.render()} over {{{values}}}"
        )
        _render_body(rendered[0][1], indent + 1, out)
        return
    out.append(f"{pad}switch {node.expression.render()}")
    for value, body, _lines in rendered:
        out.append(f"{pad}  case {value}")
        _render_body(body, indent + 2, out)
    if node.otherwise:
        out.append(f"{pad}  otherwise")
        _render_body(node.otherwise, indent + 2, out)


def _render_let(node, indent, out):
    pad = "  " * indent
    bindings = []
    if node.axis.extent.constant_value != 1:
        bindings.append((node.axis.name, node.value.render()))
    body = node.body
    while len(body) == 1 and isinstance(body[0], Let):
        nested = body[0]
        if nested.axis.extent.constant_value != 1:
            bindings.append((nested.axis.name, nested.value.render()))
        body = nested.body
    if not bindings:
        _render_body(body, indent, out)
        return
    out.append(
        f"{pad}at "
        + ", ".join(f"{axis}={value}" for axis, value in bindings)
    )
    _render_body(body, indent + 1, out)


def _render_body(nodes, indent, out):
    pad = "  " * indent
    for node in nodes:
        if isinstance(node, Region):
            if out and out[-1]:
                out.append("")
            out.append(f"{pad}--- {node.tag.name.lower()} ---")
        elif isinstance(node, Movement):
            _render_movement(node, indent, out)
        elif isinstance(node, Compute):
            reads = ", ".join(
                f"{_tile(value)}@register"
                for value in node.reads
            )
            writes = ", ".join(
                f"{_tile(value)}@register"
                for value in node.writes
            )
            out.append(
                f"{pad}{node.op.__name__.lower()} registers "
                f"read({reads}) -> write({writes})"
            )
        elif isinstance(node, Loop):
            _render_loop(node, indent, out)
        elif isinstance(node, Cond):
            if not node.then_body and not node.else_body:
                continue
            if not node.then_body and node.else_body:
                out.append(f"{pad}if not ({node.pred.render()})")
                _render_body(node.else_body, indent + 1, out)
            else:
                out.append(f"{pad}if {node.pred.render()}")
                _render_body(node.then_body, indent + 1, out)
            if node.then_body and node.else_body:
                out.append(f"{pad}else")
                _render_body(node.else_body, indent + 1, out)
        elif isinstance(node, Let):
            _render_let(node, indent, out)
        elif isinstance(node, Switch):
            _render_switch(node, indent, out)
        else:
            out.append(f"{pad}<{type(node).__name__}>")


def render(program: Program) -> str:
    params = ", ".join(param.name for param in program.params) or "-"
    axes = ", ".join(
        f"{axis.name}:{axis.extent.render()}"
        for axis in program.axes
    ) or "-"
    agents = ", ".join(
        f"{axis.name}:{axis.extent.render()}@{scope.name.lower()}"
        for axis, scope in program.agents
    ) or "-"
    out = [
        f"program params({params}) agents({agents})",
        f"  axes: {axes}",
        "  operands:",
        *(
            f"    {operand.name}[{', '.join(axis.name for axis in operand.axes)}] "
            f"{operand.role.name.lower()}"
            for operand in program.operands
        ),
        "  schedule:",
    ]
    _render_body(program.body, 2, out)
    return "\n".join(out)


__all__ = ["render"]
