# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validation for finalized LoopIR and target-agent assignments."""

from __future__ import annotations

from collections import Counter
from itertools import product

from .loopir import (
    DRAIN,
    PROLOGUE,
    SHORT,
    STEADY,
    TENSOR_LOAD,
    Compute,
    Cond,
    Let,
    Loop,
    Movement,
    Program,
    Region,
    Switch,
    dependences,
    operations,
    required_dependences,
    walk,
)
from .theta import AgentAssignment


def _check_phases(program, errors):
    order = {
        PROLOGUE: 0,
        STEADY: 1,
        DRAIN: 2,
        SHORT: 3,
    }
    has_regions = any(isinstance(node, Region) for node in walk(program))

    def check_body(body, inherited=None):
        active = inherited
        last = -1
        for node in body:
            if isinstance(node, Region):
                rank = order[node.tag]
                if rank < last:
                    errors.append(
                        "phase markers are out of order: %s after rank %d"
                        % (node.tag.name, last)
                    )
                active = node.tag
                last = rank
                continue
            if has_regions and active is None \
                    and isinstance(node, (Movement, Compute)):
                errors.append(
                    "%s occurs before a phase marker"
                    % type(node).__name__
                )
            if isinstance(node, (Loop, Let)):
                check_body(node.body, active)
            elif isinstance(node, Cond):
                check_body(node.then_body, active)
                check_body(node.else_body, active)
            elif isinstance(node, Switch):
                for _value, case in node.cases:
                    check_body(case, active)
                check_body(node.otherwise, active)

    check_body(program.body)


def _check_effect_partitions(program, errors):
    by_operand = {}
    for operation in operations(program):
        if not isinstance(operation, Movement) \
                or operation.op is not TENSOR_LOAD:
            continue
        for index in range(len(operation.transfers)):
            effect = operation.effect_for(index)
            by_operand.setdefault(effect.tile.operand, []).append(effect)

    declared = tuple(program.axes) + tuple(
        axis for axis, _scope in program.agents
    )
    params = {param: 0 for param in program.params}
    for operand, assigned in by_operand.items():
        expected = set()
        actual = Counter()
        for effect in assigned:
            free = set(effect.where.free_axes())
            for expression in effect.tile.indices:
                free.update(expression.free_axes())
            axes = tuple(axis for axis in declared if axis in free)
            ranges = tuple(
                range(max(1, int(axis.extent.constant_value)))
                if axis.extent.constant_value is not None else (0,)
                for axis in axes
            )
            for values in product(*ranges):
                environment = dict(params)
                environment.update(zip(axes, values))
                tile = tuple(
                    expression.evaluate(environment)
                    for expression in effect.tile.indices
                )
                expected.add(tile)
                if effect.where.evaluate(environment) is not False:
                    actual[tile] += 1
        bad = tuple(
            (tile, actual[tile])
            for tile in sorted(expected)
            if actual[tile] != 1
        )
        if bad:
            errors.append(
                "%s TensorLoad Effects are not a disjoint cover: %s"
                % (operand.name, bad[:8])
            )


def validate_loopir(program, profile=None):
    """Return structural/effect violations and finalized-profile mismatches.

    Profile-less programs also run the exhaustive symbolic dependence oracle.
    Target-decoded programs use the direct ledger verified after GIR lowering.
    """

    if not isinstance(program, Program):
        raise ValueError("validate_loopir expects a LoopIR Program")
    errors = []
    try:
        program.check()
        _check_phases(program, errors)
        _check_effect_partitions(program, errors)
        if profile is not None:
            profile.validate(program)
        else:
            derived = dependences(program)
            required_dependences(program, derived)
    except Exception as error:
        errors.append("%s: %s" % (type(error).__name__, error))
    return errors


def agent_assignment_consistency(theta) -> list:
    """Validate wave-level assignment extents."""

    assignment = theta.agent_assignment
    if not isinstance(assignment, AgentAssignment):
        return [
            "agent_assignment has type %s"
            % type(assignment).__name__
        ]
    errors = []
    wave = assignment.coarser_than("wave")
    if wave:
        product = 1
        for role in wave:
            product *= max(1, int(role.extent))
        agents = max(1, int(theta.wave_count))
        if product > agents or agents % product:
            errors.append(
                "wave-level agent extents %d do not divide wave_count %d"
                % (product, agents)
            )
    return errors


__all__ = ["agent_assignment_consistency", "validate_loopir"]
