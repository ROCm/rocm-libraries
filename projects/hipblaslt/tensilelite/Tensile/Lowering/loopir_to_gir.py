# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Direct LoopIR Program to existing GIR conversion.

This module performs representation conversion and CFG construction only.
Scheduling and logical dependence discovery are completed before conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..LoopModel.loopir import (
    DRAIN,
    DS_LOAD,
    IMMEDIATE,
    PROLOGUE,
    REGISTER,
    SHORT,
    STEADY,
    TENSOR_LOAD,
    Compute,
    Cond,
    Let,
    Loop,
    Movement,
    Program as LoopProgram,
    Region,
    Switch,
    dependence_logical_tile,
    dependences,
    expr_equivalent,
    occurrences,
    operations,
    required_dependences,
)
from .gir.nodes import (
    Block,
    Bound,
    CondGoto,
    Gen,
    GenPhi,
    GenXfer,
    Goto,
    LoopBack,
    LoopCopy,
    Mark,
    Mma,
    Move,
    Program,
    Pred,
    Ref,
    Return,
    Tile,
    Trips,
    covered_coords,
)
from .gir.render import gir_counts, render_gir


@dataclass(frozen=True)
class _FlatOperation:
    operation: Movement | Compute
    environment: tuple


@dataclass(frozen=True)
class _DependencyEndpoint:
    """Existing GIR dependency payload spelling, built from a derived Effect."""

    op: object
    at: str
    role: str = ""
    gen: tuple = ()

    def render(self):
        operand = "+".join(self.op) if isinstance(self.op, tuple) else self.op
        result = f"{operand}@{self.at}" if not self.role \
            else f"{operand}:{self.at}:{self.role}"
        if self.gen:
            result += ":" + ":".join(self.gen)
        return result

    __str__ = render


def _evaluate(expression, environment):
    try:
        return expression.evaluate(environment)
    except KeyError:
        return None


def _flatten(nodes, environment, output, symbolic_loops):
    for node in nodes:
        if isinstance(node, Region):
            continue
        if isinstance(node, (Movement, Compute)):
            output.append(_FlatOperation(node, tuple(environment.items())))
            continue
        if isinstance(node, Let):
            value = _evaluate(node.value, environment)
            nested = dict(environment)
            nested[node.axis] = 0 if value is None else value
            _flatten(node.body, nested, output, symbolic_loops)
            continue
        if isinstance(node, Loop):
            lower = _evaluate(node.lo, environment)
            upper = _evaluate(node.hi, environment)
            if lower is not None and upper is not None:
                for value in range(lower, upper):
                    nested = dict(environment)
                    nested[node.axis] = value
                    _flatten(node.body, nested, output, symbolic_loops)
            else:
                symbolic_loops.append(node)
                nested = dict(environment)
                nested[node.axis] = 0
                _flatten(node.body, nested, output, symbolic_loops)
            continue
        if isinstance(node, Cond):
            verdict = node.pred.evaluate(environment)
            if verdict is None:
                raise RuntimeError(
                    "lowering requires unresolved Cond to be CFG-structured "
                    "before operation flattening"
                )
            _flatten(
                node.then_body if verdict else node.else_body,
                dict(environment),
                output,
                symbolic_loops,
            )
            continue
        if isinstance(node, Switch):
            value = _evaluate(node.expression, environment)
            if value is None:
                raise RuntimeError(
                    "lowering requires unresolved Switch to be CFG-structured"
                )
            _flatten(
                dict(node.cases).get(value, node.otherwise),
                dict(environment),
                output,
                symbolic_loops,
            )
            continue
        raise TypeError(f"unsupported LoopIR node {type(node).__name__}")


def _runtime_offset(expression, axis, environment):
    """Return ``expression - axis`` for one affine runtime-axis expression."""

    zero = dict(environment)
    one = dict(environment)
    zero[axis] = 0
    one[axis] = 1
    try:
        at_zero = expression.evaluate(zero)
        at_one = expression.evaluate(one)
    except KeyError as error:
        raise RuntimeError(
            f"runtime tile expression for {axis.name!r} has an unresolved symbol"
        ) from error
    if at_one - at_zero == 0:
        return 0
    if at_one - at_zero != 1:
        raise RuntimeError(
            f"runtime tile expression for {axis.name!r} is not axis+constant"
        )
    return int(at_zero)


def _project_tile(tile, environment):
    coord = []
    runtime = []
    for axis, expression in zip(tile.operand.axes, tile.indices):
        if axis.extent.free_params():
            runtime.append((
                axis,
                _runtime_offset(expression, axis, environment),
                _evaluate(expression, environment),
            ))
        else:
            value = _evaluate(expression, environment)
            if value is not None:
                coord.append((axis.name, value))
    return tuple(coord), tuple(runtime)


def _project_reference(tile, environment, covers=(), omit_axes=()):
    """Project one logical tile once for both sides of a Movement."""

    projection_environment = environment
    if covers:
        projection_environment = dict(environment)
        for expression in tile.indices:
            for axis in expression.free_axes():
                projection_environment.setdefault(axis, 0)
    coordinate, runtime = _project_tile(tile, projection_environment)
    omitted = {name for name, _extent in omit_axes}
    if covers:
        operand_extents = {
            axis.name: axis.extent.constant_value
            for axis in tile.operand.axes
        }
        omitted.update(
            name for name, extent in covers
            if operand_extents.get(name) == int(extent)
        )
    if omitted:
        coordinate = tuple(
            item for item in coordinate if item[0] not in omitted
        )
    return coordinate, runtime


def _logical_coverage(tile, environment):
    bound = {
        id(key)
        for key in environment
        if hasattr(key, "extent")
    }
    result = []
    for operand_axis, expression in zip(tile.operand.axes, tile.indices):
        local = tuple(
            axis for axis in expression.free_axes()
            if id(axis) not in bound
            and axis is not operand_axis
            and axis.extent.constant_value is not None
        )
        if not local:
            continue
        extent = 1
        for axis in local:
            extent *= int(axis.extent.constant_value)
        result.append((operand_axis.name, extent))
    return tuple(result)


def _unbound_operand_axes(tile, environment):
    bound = {
        id(key)
        for key in environment
        if hasattr(key, "extent")
    }
    return tuple(
        (operand_axis.name, int(operand_axis.extent.constant_value))
        for operand_axis, expression in zip(tile.operand.axes, tile.indices)
        if id(operand_axis) not in bound
        and operand_axis.extent.constant_value is not None
        and expr_equivalent(expression, operand_axis)
        and int(operand_axis.extent.constant_value) > 1
    )


def _copy_omit_axes(tile, environment):
    result = {}
    for name, extent in (
        *_unbound_operand_axes(tile, environment),
        *_logical_coverage(tile, environment),
    ):
        result[name] = max(int(extent), result.get(name, 1))
    bound = {
        id(key)
        for key in environment
        if hasattr(key, "extent")
    }
    for operand_axis, expression in zip(tile.operand.axes, tile.indices):
        if (
            id(operand_axis) in bound
            and operand_axis.extent.constant_value is not None
            and int(operand_axis.extent.constant_value) > 1
            and operand_axis not in expression.free_axes()
        ):
            result[operand_axis.name] = max(
                int(operand_axis.extent.constant_value),
                result.get(operand_axis.name, 1),
            )
    return tuple(result.items())


def _read_issue_axes(program, profile):
    """Finalized prefetch-pass axes used only for target register projection."""

    axes_by_name = {axis.name: axis for axis in program.axes}
    by_identity = {}
    by_operand = {}
    for operation in operations(program):
        if not isinstance(operation, Movement) \
                    or operation.op is not DS_LOAD:
            continue
        axes = tuple(
            axes_by_name[name]
            for name in (
                profile.read_issue_axes(operation)
                if profile is not None else ()
            )
            if name in axes_by_name
        )
        by_identity[operation._identity] = axes
        for transfer in operation.transfers:
            present = {
                id(axis): axis
                for axis in by_operand.get(transfer.local_tile.operand.name, ())
            }
            present.update({id(axis): axis for axis in axes})
            by_operand[transfer.local_tile.operand.name] = tuple(present.values())
    return by_identity, by_operand


def _ref(
    tile,
    level,
    environment,
    gens,
    *,
    destination=False,
    shared_delta=0,
    absolute=None,
    profile=None,
    size_bytes_override=None,
    size_regs_override=None,
    covers=(),
    prefetch_steps=0,
    issue_axes=(),
    omit_axes=(),
    projection=None,
):
    space = level.name.lower()
    size_bytes = int(size_bytes_override or 0)
    size_regs = int(size_regs_override or 0)
    coordinate, runtime = (
        _project_reference(tile, environment, covers, omit_axes)
        if projection is None else projection
    )
    gir_tile = Tile(
        tile.operand.name,
        space,
        coordinate,
        tuple(
            item for item in (
                ("regs", size_regs) if size_regs else None,
                ("bytes", size_bytes) if size_bytes else None,
            )
            if item is not None
        ),
    )
    if space == "shared":
        region = (
            profile.region_index(tile.operand.name, coordinate)
            if profile is not None else 0
        )
        key = (tile.operand.name, region)
        gen = gens.setdefault(key, Gen(len(gens), 1))
        runtime_offset = runtime[0][1] if runtime else 0
        if absolute:
            runtime_value = next(
                (value for _axis, _offset, value in runtime if value is not None),
                0,
            )
            return Ref(
                gir_tile,
                abs_gen=int(runtime_value) % max(1, gen.ring),
                covers=covers,
                size_bytes=size_bytes,
            )
        return Ref(
            gir_tile,
            gen=gen,
            gdelta=int(shared_delta) + int(runtime_offset),
            covers=covers,
            size_bytes=size_bytes,
        )
    if space == "register":
        logical_coordinate = dict(coordinate)
        logical_coordinate.update({
            key.name: value
            for key, value in environment.items()
            if hasattr(key, "extent")
            and not key.extent.free_params()
            and key.name not in logical_coordinate
            and value is not None
        })
        logical_coordinate.update({
            axis.name: value
            for axis, _offset, value in runtime
            if value is not None
        })
        copy_phase = next(
            (
                int(value)
                for axis, value in environment.items()
                if hasattr(axis, "extent")
                and axis.extent.free_params()
                and value is not None
            ),
            0,
        )
        placement_environment = (
            {
                getattr(key, "name", key): value
                for key, value in environment.items()
                if value is not None
            }
            if prefetch_steps else logical_coordinate
        )
        group, slot, ring = (
            profile.register_location(
                tile.operand.name,
                logical_coordinate,
                prefetch_steps=prefetch_steps,
                copy_phase=copy_phase,
                placement_environment=placement_environment,
                issue_axes=issue_axes,
            )
            if profile is not None else (0, 0, 1)
        )
        unit_index = (
            profile.unit_index(tile.operand.name, coordinate)
            if profile is not None else 0
        )
        reference = Ref(
            gir_tile,
            group=group,
            slot=slot,
            reg_ring=ring,
            unit_index=unit_index,
            unit_indexes=(),
            issue_axes=issue_axes,
            covers=covers,
            size_regs=size_regs,
        )
        if destination:
            indexes = tuple(
                profile.unit_index(tile.operand.name, dict(coordinate))
                if profile is not None else unit_index
                for coordinate in covered_coords(reference)
            )
            reference = replace(reference, unit_indexes=indexes)
        return reference
    return Ref(gir_tile, covers=covers, size_bytes=size_bytes)


def _dependency_map(program, profile=None):
    if profile is not None:
        return _build_dependency_ledger(program, profile)

    result = {}
    derived = dependences(program)
    required = required_dependences(program, derived)
    distributed = (
        profile.as_meta().get("agent_distributed", {})
        if profile is not None else {}
    )
    operations_by_identity = {}
    for operation in operations(program):
        operations_by_identity[operation._identity] = operation
    movement_units = {}
    for operation in operations_by_identity.values():
        if not isinstance(operation, Movement) \
                or operation.op is not TENSOR_LOAD:
            continue
        unit = tuple(dict.fromkeys(
            transfer.local_tile.operand.name
            for transfer in operation.transfers
        ))
        for name in unit:
            movement_units.setdefault(name, unit)
    seen = set()
    for dependence in derived:
        if not dependence.crossing_raw:
            continue
        completion = dependence.complete.operation.completion
        counter = (
            f"C_{completion.name}"
            if completion is not IMMEDIATE else "immediate"
        )
        present = tuple(dict.fromkeys(
            transfer.local_tile.operand.name
            for transfer in dependence.complete.operation.transfers
        ))
        crossing_unit = next(
            (
                movement_units[name]
                for name in present
                if name in movement_units
            ),
            present,
        )
        complete_id = dependence.complete.operation._identity
        crossing_key = complete_id, crossing_unit, counter
        if crossing_key in seen:
            continue
        seen.add(crossing_key)
        result.setdefault(complete_id, []).append((
            None,
            (
                _DependencyEndpoint(crossing_unit, "shared", "copy"),
                counter,
                "crossing-RAW",
                "wave",
            ),
        ))

    for dependence in required:
        kind = dependence.kind.name
        operand = dependence.complete.tile.operand.name
        completion = dependence.complete.operation.completion
        counter = (
            f"C_{completion.name}"
            if completion is not IMMEDIATE
            else "immediate"
        )
        if kind == "RAW":
            legacy_kind = "RAW-residency"
        elif (
            isinstance(dependence.issue.operation, Movement)
            and dependence.issue.operation.op is DS_LOAD
            and dependence.issue.level.name == "REGISTER"
        ):
            legacy_kind = (
                "inplace-WAR"
                if profile is not None
                and profile.register_refill_in_place(operand)
                else "rotation-WAR"
            )
            issue_completion = dependence.issue.operation.completion
            counter = (
                f"C_{issue_completion.name}"
                if issue_completion is not IMMEDIATE
                else "immediate"
            )
        else:
            legacy_kind = (
                "inplace-WAR"
                if profile is not None
                and profile.shared_refill_in_place(operand)
                else "rotation-WAR"
            )
        scope = (
            "wave"
            if legacy_kind == "rotation-WAR"
            else "block"
            if distributed.get(dependence.complete.tile.operand.name, False)
            else "wave"
        )
        key = (
            dependence.issue.occurrence.path,
            dependence.complete.tile.operand.name,
            completion,
            counter,
            legacy_kind,
            scope,
        )
        if key in seen:
            continue
        seen.add(key)
        unit = (
            tuple(dict.fromkeys(
                transfer.local_tile.operand.name
                for transfer in dependence.complete.operation.transfers
            ))
            if isinstance(dependence.complete.operation, Movement)
            and dependence.complete.operation.op is TENSOR_LOAD
            else operand
        )
        if kind == "RAW":
            endpoint = (
                _DependencyEndpoint(unit, "global", gen=("iter",))
                if isinstance(dependence.complete.operation, Movement)
                and dependence.complete.operation.op is TENSOR_LOAD
                else _DependencyEndpoint(operand, "shared")
            )
        elif (
            isinstance(dependence.issue.operation, Movement)
            and dependence.issue.operation.op is DS_LOAD
        ):
            endpoint = _DependencyEndpoint(operand, "g0", "lastread")
        else:
            endpoint = _DependencyEndpoint(operand, "shared", "lastread")
        entry = (
            endpoint,
            counter,
            legacy_kind,
            scope,
        )
        issue_id = dependence.issue.operation._identity
        generation = None
        if (
            kind == "RAW"
            and isinstance(dependence.complete.operation, Movement)
            and dependence.complete.operation.op is TENSOR_LOAD
            and isinstance(dependence.issue.operation, Movement)
            and dependence.issue.operation.op is DS_LOAD
        ):
            runtime_axes = tuple(
                axis
                for axis in dependence.issue.tile.operand.axes
                if axis.extent.free_params()
            )
            expressions = dependence_logical_tile(dependence)
            base_point = {
                axis: 0
                for expression in expressions
                for axis in expression.free_axes()
            }
            base_point.update({axis: 0 for axis in runtime_axes})
            expressions = tuple(
                expression.substitute(base_point)
                for expression in expressions
            )
            generation = expressions
        result.setdefault(issue_id, []).append((generation, entry))
    return {
        identity: tuple(dict.fromkeys(records))
        for identity, records in result.items()
    }


def _completion_counter(completion):
    return (
        f"C_{completion.name}"
        if completion is not IMMEDIATE else "immediate"
    )


def _build_dependency_ledger(program, profile):
    """Build the finalized dependency ledger from scheduled operation roles."""

    ordered = operations(program)
    distributed = profile.as_meta().get("agent_distributed", {})
    reads = {}
    copies = []
    computes = []
    for operation in ordered:
        if isinstance(operation, Movement):
            if operation.op is TENSOR_LOAD:
                copies.append(operation)
            elif operation.op is DS_LOAD:
                for transfer in operation.transfers:
                    reads.setdefault(
                        transfer.local_tile.operand.name,
                        [],
                    ).append(operation)
        elif isinstance(operation, Compute):
            computes.append(operation)

    result = {}

    def add(operation, generation, endpoint, counter, kind, scope):
        record = generation, (endpoint, counter, kind, scope)
        records = result.setdefault(operation._identity, [])
        if record not in records:
            records.append(record)

    copy_groups = {}
    for operation in copies:
        unit = tuple(dict.fromkeys(
            transfer.local_tile.operand.name
            for transfer in operation.transfers
        ))
        completion = operation.completion
        copy_groups.setdefault((completion, unit), operation)

        for name in unit:
            operand_reads = reads.get(name, ())
            if not operand_reads:
                continue
            read = operand_reads[0]
            kind = (
                "inplace-WAR"
                if profile.shared_refill_in_place(name)
                else "rotation-WAR"
            )
            scope = (
                "wave"
                if kind == "rotation-WAR"
                else "block" if distributed.get(name, False)
                else "wave"
            )
            add(
                operation,
                None,
                _DependencyEndpoint(name, "shared", "lastread"),
                _completion_counter(read.completion),
                kind,
                scope,
            )

        if any(name in profile.copy_first for name in unit):
            add(
                operation,
                None,
                _DependencyEndpoint(unit, "shared", "copy"),
                _completion_counter(completion),
                "crossing-RAW",
                "wave",
            )

    for (completion, unit), _copy in copy_groups.items():
        waiter_name = next(
            (name for name in unit if name in reads),
            None,
        )
        if waiter_name is None:
            continue
        waiter = reads[waiter_name][0]
        add(
            waiter,
            None,
            _DependencyEndpoint(unit, "global", gen=("iter",)),
            _completion_counter(completion),
            "RAW-residency",
            "block" if distributed.get(waiter_name, False) else "wave",
        )

    consumed = {
        tile.operand.name
        for compute in computes
        for tile in compute.reads
    }
    for name, operand_reads in reads.items():
        if name not in consumed:
            continue
        for operation in operand_reads:
            kind = (
                "inplace-WAR"
                if profile.register_refill_in_place(name)
                else "rotation-WAR"
            )
            scope = (
                "wave"
                if kind == "rotation-WAR"
                else "block" if distributed.get(name, False)
                else "wave"
            )
            add(
                operation,
                None,
                _DependencyEndpoint(name, "g0", "lastread"),
                _completion_counter(operation.completion),
                kind,
                scope,
            )

    for compute in computes:
        for tile in compute.reads:
            name = tile.operand.name
            operand_reads = reads.get(name, ())
            if not operand_reads:
                continue
            read = operand_reads[0]
            add(
                compute,
                None,
                _DependencyEndpoint(name, "shared"),
                _completion_counter(read.completion),
                "RAW-residency",
                "block" if distributed.get(name, False) else "wave",
            )

    return {
        identity: tuple(records)
        for identity, records in result.items()
    }


def _operation_dependencies(dependencies, operation, environment):
    projected = []
    actual_generation = None
    for generation, entry in dependencies.get(operation._identity, ()):
        if generation is not None and actual_generation is None:
            tiles = (
                tuple(transfer.local_tile for transfer in operation.transfers)
                if isinstance(operation, Movement)
                else tuple(
                    tile
                    for tile in operation.reads + operation.writes
                )
            )
            tile = next(
                (
                    candidate
                    for candidate in tiles
                    if any(
                        axis.extent.free_params()
                        for axis in candidate.operand.axes
                    )
                ),
                tiles[0] if tiles else None,
            )
            if tile is not None:
                projection_environment = dict(environment)
                runtime_axes = tuple(
                    axis
                    for axis in tile.operand.axes
                    if axis.extent.free_params()
                )
                projection_environment.update({
                    axis: 0 for axis in runtime_axes
                })
                for expression in tile.indices:
                    for axis in expression.free_axes():
                        if (
                            axis not in projection_environment
                            and axis.name not in projection_environment
                        ):
                            projection_environment[axis] = 0
                actual_generation = tuple(
                    expression.substitute(projection_environment)
                    for expression in tile.indices
                )
        if generation is not None and (
            actual_generation is None
            or len(generation) != len(actual_generation)
            or not all(
                expr_equivalent(expected, actual)
                for expected, actual in zip(generation, actual_generation)
            )
        ):
            continue
        if entry not in projected:
            projected.append(entry)
    return tuple(projected)


def _project_explicit_transfers(operation, environment):
    """Collapse only coverage proven by explicit local tiles on one issue."""

    groups = {}
    for index, transfer in enumerate(operation.transfers):
        groups.setdefault(transfer.local_tile.operand, []).append((index, transfer))
    projected = []
    coverage = {}
    for operand, members in groups.items():
        if len(members) == 1:
            index, transfer = members[0]
            projected.append((index, transfer))
            coverage[index] = _logical_coverage(transfer.local_tile, environment)
            continue
        coordinates = []
        for _index, transfer in members:
            probe = dict(environment)
            for expression in transfer.local_tile.indices:
                probe.update({
                    axis: 0 for axis in expression.free_axes()
                    if axis not in probe
                })
            values = tuple(
                _evaluate(expression, probe)
                for expression in transfer.local_tile.indices
            )
            if any(value is None for value in values):
                coordinates = []
                break
            coordinates.append(values)
        varying = []
        if coordinates:
            for axis_index, axis in enumerate(operand.axes):
                values = sorted({
                    coordinate[axis_index] for coordinate in coordinates
                })
                if len(values) <= 1:
                    continue
                if values != list(range(values[0], values[0] + len(values))):
                    varying = []
                    break
                varying.append((axis.name, len(values)))
        volume = 1
        for _axis, extent in varying:
            volume *= extent
        if varying and volume == len(members):
            representative = min(
                range(len(members)),
                key=lambda member: coordinates[member],
            )
            index, transfer = members[representative]
            projected.append((index, transfer))
            coverage[index] = tuple(varying)
        else:
            for index, transfer in members:
                projected.append((index, transfer))
                coverage[index] = _logical_coverage(
                    transfer.local_tile,
                    environment,
                )
    projected.sort(key=lambda item: item[0])
    return tuple(projected), coverage


def _convert_compute(
    operation,
    environment,
    gens,
    deps,
    profile,
    source_issue_axes,
):
    coord = {}
    for tile in operation.reads + operation.writes:
        for axis, expression in zip(tile.operand.axes, tile.indices):
            if axis.extent.free_params():
                continue
            value = _evaluate(expression, environment)
            if value is not None:
                coord.setdefault(axis.name, value)
    if profile is not None:
        order = tuple(profile.as_meta().get("inner_axis_order", ()))
        coord = {
            name: coord[name]
            for name in order
            if name in coord
        } | {
            name: value
            for name, value in coord.items()
            if name not in order
        }
    srcs = tuple(
        _ref(
            tile,
            REGISTER,
            environment,
            gens,
            profile=profile,
            size_bytes_override=0,
            size_regs_override=(
                profile.fragment_regs(tile.operand.name)
                if profile is not None else 0
            ),
            issue_axes=tuple(
                axis.name
                for axis in (source_issue_axes or {}).get(
                    tile.operand.name,
                    (),
                )
            ),
        )
        for tile in operation.reads
    )
    dsts = tuple(
        (
            Ref(Tile("acc", "register", tuple(coord.items())))
            if tile.operand.role.name == "OUTPUT"
            else _ref(
                tile,
                REGISTER,
                environment,
                gens,
                destination=True,
                profile=profile,
                size_bytes_override=0,
                size_regs_override=(
                    profile.fragment_regs(tile.operand.name)
                    if profile is not None else 0
                ),
            )
        )
        for tile in operation.writes
    )
    return Mma(
        srcs,
        dsts,
        coord=tuple(coord.items()),
        block=(
            "*".join(
                f"{name}(k)"
                for name in (
                    profile.as_meta().get("mma_inputs", ())
                    if profile is not None else ()
                )
            )
        ),
        deps=_operation_dependencies(deps, operation, environment),
    )


def _convert(
    operation,
    environment,
    gens,
    deps,
    *,
    source_delta=0,
    destination_delta=0,
    absolute=None,
    profile=None,
    phase="",
    issue_axes=(),
    source_issue_axes=None,
):
    if isinstance(operation, Movement):
        tensor_load = operation.op is TENSOR_LOAD
        local_read = operation.op is DS_LOAD
        indexed_transfers, explicit_coverage = _project_explicit_transfers(
            operation,
            environment,
        )
        prefetch_steps = (
            profile.read_prefetch_steps(operation)
            if local_read
            and phase in ("steady", "drain")
            and profile is not None
            else 0
        )
        issue_axis_names = tuple(axis.name for axis in issue_axes)
        projected_transfers = []
        for index, transfer in indexed_transfers:
            covers = (
                explicit_coverage.get(index, ())
                if local_read
                else _logical_coverage(transfer.local_tile, environment)
                if not tensor_load else ()
            )
            omit_axes = (
                tuple(dict.fromkeys((
                    *explicit_coverage.get(index, ()),
                    *_copy_omit_axes(transfer.local_tile, environment),
                )))
                if tensor_load else ()
            )
            size_bytes = (
                profile.bulk_bytes(transfer.local_tile.operand.name)
                if tensor_load and profile is not None else 0
            )
            size_regs = (
                profile.fragment_regs(transfer.local_tile.operand.name)
                if not tensor_load and profile is not None else 0
            )
            projected_transfers.append((
                transfer,
                covers,
                size_bytes,
                size_regs,
                _project_reference(
                    transfer.local_tile,
                    environment,
                    covers,
                    omit_axes,
                ),
            ))
        srcs = tuple(
            _ref(
                transfer.local_tile,
                transfer.source,
                environment,
                gens,
                shared_delta=source_delta,
                absolute=absolute,
                profile=profile,
                size_bytes_override=size_bytes,
                size_regs_override=size_regs,
                covers=covers,
                prefetch_steps=prefetch_steps,
                issue_axes=issue_axis_names,
                projection=projection,
            )
            for transfer, covers, size_bytes, size_regs, projection
            in projected_transfers
        )
        dsts = tuple(
            _ref(
                transfer.local_tile,
                transfer.destination,
                environment,
                gens,
                destination=True,
                shared_delta=destination_delta,
                absolute=absolute,
                profile=profile,
                size_bytes_override=size_bytes,
                size_regs_override=size_regs,
                covers=covers,
                prefetch_steps=prefetch_steps,
                issue_axes=issue_axis_names,
                projection=projection,
            )
            for transfer, covers, size_bytes, size_regs, projection
            in projected_transfers
        )
        operation_deps = _operation_dependencies(
            deps,
            operation,
            environment,
        )
        if phase == "prologue" and operation.op is TENSOR_LOAD:
            operation_deps = tuple(
                item for item in operation_deps
                if item[2] == "crossing-RAW"
            )
        elif phase == "prologue":
            operation_deps = tuple(
                item for item in operation_deps
                if item[2] == "RAW-residency"
            )
        advance = (
            profile.read_advance(operation)
            if local_read and prefetch_steps and profile is not None
            else 0
        )
        return Move(
            srcs,
            dsts,
            deps=operation_deps,
            unit=tuple(dict.fromkeys(
                transfer.local_tile.operand.name
                for transfer in operation.transfers
            )) if tensor_load else (),
            advance=advance,
            issue=(
                profile.read_issue(operation, environment)
                if profile is not None and phase in ("steady", "drain")
                else tuple(
                    (axis.name, int(environment.get(axis, 0)))
                    for axis in issue_axes
                )
            ) if local_read else (),
        )
    return _convert_compute(
        operation,
        environment,
        gens,
        deps,
        profile,
        source_issue_axes,
    )


def _region_bodies(program):
    regions = []
    current_tag = None
    current = []
    for node in program.body:
        if isinstance(node, Region):
            if current_tag is not None:
                regions.append((current_tag, tuple(current)))
            current_tag = node.tag
            current = []
        else:
            current.append(node)
    if current_tag is None:
        return (("entry", tuple(current)),)
    regions.append((current_tag.name.lower(), tuple(current)))
    return tuple(regions)


def _marked_segments(body):
    result = {}
    current = None
    nodes = []
    for node in body:
        if isinstance(node, Region):
            if current is not None:
                result[current] = tuple(nodes)
            current = node.tag
            nodes = []
        else:
            nodes.append(node)
    if current is not None:
        result[current] = tuple(nodes)
    return result


def _pipeline_shape(program):
    body = program.body
    if len(body) == 1 and isinstance(body[0], Cond):
        guard = body[0]
        full = _marked_segments(guard.then_body)
        short = _marked_segments(guard.else_body)
    else:
        guard = None
        full = _marked_segments(body)
        short = {}
    required = {PROLOGUE, STEADY}
    if not required <= set(full):
        return None
    return guard, full, short


def _nested_pipeline(body):
    for node in body:
        if not isinstance(node, Cond):
            continue
        full = _marked_segments(node.then_body)
        short = _marked_segments(node.else_body)
        if {PROLOGUE, STEADY, DRAIN} <= set(full):
            return node, full, short
    return None


def _parameter_minus_constant(expression):
    if expression.op == "param":
        return expression.args[0].name, 0
    if expression.op != "add":
        return "", 0
    variable = ""
    constant = 0
    for argument in expression.args:
        if argument.op == "param":
            variable = argument.args[0].name
        elif argument.op == "const":
            constant += argument.args[0]
        else:
            return "", 0
    return variable, -constant


def _explicit_pipeline_depth(program):
    depth = 0
    for occurrence in occurrences(program):
        operation = occurrence.node
        if not isinstance(operation, Movement) \
                or operation.op is not TENSOR_LOAD:
            continue
        for transfer in operation.transfers:
            for axis, expression in zip(
                transfer.local_tile.operand.axes,
                transfer.local_tile.indices,
            ):
                if not axis.extent.free_params():
                    continue
                try:
                    depth = max(
                        depth,
                        _runtime_offset(expression, axis, {}),
                    )
                except RuntimeError:
                    continue
    return depth


def _convert_body(
    body,
    gens,
    deps,
    environment=None,
    *,
    source_delta=0,
    destination_delta=0,
    absolute=None,
    profile=None,
    phase="",
    issue_axes=None,
    source_issue_axes=None,
):
    flattened = []
    symbolic = []
    _flatten(body, dict(environment or {}), flattened, symbolic)
    converted = []
    for item in flattened:
        converted.append(
            _convert(
                item.operation,
                dict(item.environment),
                gens,
                deps,
                source_delta=source_delta,
                destination_delta=destination_delta,
                absolute=absolute,
                profile=profile,
                phase=phase,
                issue_axes=(issue_axes or {}).get(
                    item.operation._identity,
                    (),
                ),
                source_issue_axes=source_issue_axes,
            )
        )
    return converted, symbolic


def _finish_pipeline(
    program,
    gir,
    gens,
    profile,
    guard,
    prologue,
    steady_bodies,
    drain_bodies,
    short_bodies,
    variable,
    sub,
    has_drain,
    has_short_loop,
):
    copy_count = len(steady_bodies)
    drain_labels = [
        f"drain{index}" for index in range(len(drain_bodies))
    ]
    short_labels = [
        f"short{index}" for index in range(len(short_bodies))
    ]
    exit_target = drain_labels[0] if drain_labels else "end"
    false_target = short_labels[0] if short_labels else exit_target
    prologue_target = "steady" if variable else false_target
    if guard is None:
        prologue_term = Goto(prologue_target)
        prologue_succs = (prologue_target,)
    elif variable:
        prologue_term = CondGoto(
            Pred(variable, ">", Bound(const=sub)),
            "steady",
            false_target,
        )
        prologue_succs = (prologue_target, false_target)
    else:
        prologue_term = Goto(false_target)
        prologue_succs = (false_target,)
    gir.add_block(Block(
        phase="prologue",
        body=[Mark("phase_boundary", {"phase": "prologue"}), *prologue],
        term=prologue_term,
        succs=prologue_succs,
    ))

    steady_labels = ["steady"] + [
        f"steady{index}" for index in range(1, copy_count)
    ]
    for index, (label, body) in enumerate(
        zip(steady_labels, steady_bodies)
    ):
        last = index == copy_count - 1
        next_target = "steady" if last else steady_labels[index + 1]
        if copy_count > 1:
            term = LoopCopy(
                Trips(variable, sub),
                index,
                copy_count,
                next_target,
                exit_target,
            )
            edge_xfers = {
                exit_target: [
                    GenXfer(gen, index + 1, gen.ring)
                    for gen in gens.values()
                ]
            }
        else:
            term = LoopBack(
                Trips(variable, sub), "steady", exit_target
            )
            edge_xfers = {}
        gir.add_block(Block(
            phase=label,
            loop=index == 0,
            preds=(
                ("prologue", steady_labels[-1])
                if index == 0 else (steady_labels[index - 1],)
            ),
            succs=(next_target, exit_target),
            phis=[
                GenPhi(gen, entry_val=0)
                for gen in gens.values()
            ] if index == 0 else [],
            xfers=[
                GenXfer(gen, copy_count, gen.ring)
                for gen in gens.values()
            ] if last else [],
            edge_xfers=edge_xfers,
            body=[Mark("phase_boundary", {"phase": label}), *body],
            term=term,
            gen_rel=index,
            chunk_base=index,
        ))

    for index, (label, body) in enumerate(
        zip(drain_labels, drain_bodies)
    ):
        target = (
            drain_labels[index + 1]
            if index + 1 < len(drain_labels)
            else "end"
        )
        gir.add_block(Block(
            phase=label,
            preds=tuple(steady_labels)
            if index == 0 else (drain_labels[index - 1],),
            succs=(target,),
            phis=[
                GenPhi(gen) for gen in gens.values()
            ] if index == 0 and copy_count > 1 else [],
            body=[Mark("phase_boundary", {"phase": label}), *body],
            term=Goto(target),
            gen_rel=-len(drain_labels) + index,
            chunk_base=copy_count + index,
            path_chunk_base={},
        ))

    for index, (label, body) in enumerate(
        zip(short_labels, short_bodies)
    ):
        target = (
            short_labels[index + 1]
            if index + 1 < len(short_labels)
            else "end"
        )
        gir.add_block(Block(
            phase=label,
            preds=("prologue",)
            if index == 0 else (short_labels[index - 1],),
            succs=(target,),
            body=[Mark("phase_boundary", {"phase": label}), *body],
            term=Goto(target),
            chunk_base=index,
            model_only=True,
        ))

    gir.meta.update({
        "peel_depth": (
            len(drain_bodies)
            if has_drain else _explicit_pipeline_depth(program)
        ),
        "drain_steps": len(drain_bodies),
        "buffer_depths": getattr(profile, "depths", None),
        "short_loop": {
            "steps": len(drain_bodies),
            "guards": tuple(
                None if index == 0
                else Pred("T", ">", Bound(const=index))
                for index in range(len(short_bodies))
            ),
            "verdict": "unfolded",
            "model_only": {
                "blocks": tuple(short_labels),
                "marks_dropped": 0,
                "reason": "LoopIR short arm",
            },
        } if has_short_loop else None,
    })
    if profile is not None:
        gir.meta.update(profile.as_meta())
    gir.meta["generation_regions"] = {
        gen.id: {"operand": operand, "region": region}
        for (operand, region), gen in gens.items()
    }
    return gir


def _lower_pipeline(
    program,
    shape,
    profile=None,
    include_dependencies=True,
):
    _guard, regions, short_regions = shape
    has_drain = DRAIN in regions
    agent_environment = {
        key: 0
        for axis, _scope in program.agents
        for key in (axis, axis.name)
    }
    gir = Program()
    gir.entry = "prologue"
    gens = {}
    if profile is not None:
        for operand in program.operands:
            ring = profile.shared_ring(operand.name)
            if ring <= 0:
                continue
            region_count = max(
                1,
                int(profile.as_meta().get("operand_regions", {}).get(
                    operand.name, 1
                )),
            )
            for region in range(region_count):
                gens[(operand.name, region)] = Gen(len(gens), ring)
    deps = (
        _dependency_map(program, profile)
        if include_dependencies else {}
    )
    issue_axes, source_issue_axes = _read_issue_axes(program, profile)
    convert_body = _convert_body

    prologue, _ = convert_body(
        regions[PROLOGUE],
        gens,
        deps,
        agent_environment,
        absolute=True,
        profile=profile,
        phase="prologue",
        issue_axes=issue_axes,
        source_issue_axes=source_issue_axes,
    )
    steady_loops = [
        node for node in regions[STEADY] if isinstance(node, Loop)
    ]
    if len(steady_loops) != 1:
        raise RuntimeError("steady region requires one runtime Loop")
    steady_loop = steady_loops[0]
    copy_steps = tuple(
        node for node in steady_loop.body
        if isinstance(node, Let) and node.axis.name == "_steady_copy"
    )
    if copy_steps:
        if len(copy_steps) != len(steady_loop.body):
            raise RuntimeError(
                "materialized steady copies must cover the whole steady body"
            )
        copy_axis = copy_steps[0].axis
        if any(step.axis is not copy_axis for step in copy_steps):
            raise RuntimeError("steady copy Lets must share one Axis")
        steady_variants = tuple(step.body for step in copy_steps)
        copy_bindings = tuple(
            {copy_axis: step.value.evaluate({})}
            for step in copy_steps
        )
    else:
        steady_variants = (steady_loop.body,)
        copy_bindings = ({},)
    copy_count = len(steady_variants)
    nested_pipeline = _nested_pipeline(steady_variants[0])

    def convert_steady(copy_index):
        environment = {
            **agent_environment,
            steady_loop.axis: copy_index,
            **copy_bindings[copy_index],
        }
        body = steady_variants[copy_index]
        current_nested = _nested_pipeline(body)
        if current_nested is None:
            steady, nested = convert_body(
                body,
                gens,
                deps,
                environment,
                source_delta=copy_index,
                destination_delta=copy_index,
                profile=profile,
                phase="steady",
                issue_axes=issue_axes,
                source_issue_axes=source_issue_axes,
            )
            if nested:
                raise RuntimeError(
                    "steady body contains a second symbolic Loop"
                )
            return steady
        nested_guard, nested_regions, _nested_short = current_nested
        nested_steady = next(
            node for node in nested_regions[STEADY]
            if isinstance(node, Loop)
        )
        outer_remainder = tuple(
            node for node in body if node is not nested_guard
        )
        core, _ = convert_body(
            (nested_steady, *nested_regions[DRAIN]),
            gens,
            deps,
            environment,
            source_delta=copy_index,
            destination_delta=copy_index,
            profile=profile,
            phase="steady",
            issue_axes=issue_axes,
            source_issue_axes=source_issue_axes,
        )
        next_prefetch, _ = convert_body(
            nested_regions[PROLOGUE],
            gens,
            deps,
            {**environment, steady_loop.axis: copy_index + 1},
            source_delta=copy_index + 1,
            destination_delta=copy_index + 1,
            profile=profile,
            phase="steady",
            issue_axes=issue_axes,
            source_issue_axes=source_issue_axes,
        )
        remainder, nested = convert_body(
            outer_remainder,
            gens,
            deps,
            environment,
            source_delta=copy_index,
            destination_delta=copy_index,
            profile=profile,
            phase="steady",
            issue_axes=issue_axes,
            source_issue_axes=source_issue_axes,
        )
        if nested:
            raise RuntimeError("steady remainder contains a symbolic Loop")
        return core + next_prefetch + remainder

    steady_bodies = [
        convert_steady(copy_index)
        for copy_index in range(copy_count)
    ]
    if nested_pipeline is not None:
        _nested_guard, nested_regions, _nested_short = nested_pipeline
        initial_reads, _ = convert_body(
            nested_regions[PROLOGUE],
            gens,
            deps,
            {
                **agent_environment,
                steady_loop.axis: 0,
                **copy_bindings[0],
            },
            absolute=True,
            profile=profile,
            phase="prologue",
            issue_axes=issue_axes,
            source_issue_axes=source_issue_axes,
        )
        prologue += initial_reads
    variable, sub = _parameter_minus_constant(steady_loop.hi)
    drain_steps = tuple(
        node for node in regions.get(DRAIN, ()) if isinstance(node, Let)
    )
    drain_bodies = []
    for index, step in enumerate(drain_steps):
        relative = -len(drain_steps) + index
        register_phase = index + 1 if copy_count > 1 else 0
        if nested_pipeline is None:
            body = convert_body(
                step.body,
                gens,
                deps,
                {**agent_environment, step.axis: register_phase},
                source_delta=relative,
                destination_delta=0,
                profile=profile,
                phase="drain",
                issue_axes=issue_axes,
                source_issue_axes=source_issue_axes,
            )[0]
        else:
            step_nested = _nested_pipeline(step.body)
            if step_nested is None:
                raise RuntimeError("drain lost the nested LoopIR pipeline")
            step_guard, step_regions, _step_short = step_nested
            nested_steady = next(
                node for node in step_regions[STEADY]
                if isinstance(node, Loop)
            )
            core, _ = convert_body(
                (nested_steady, *step_regions[DRAIN]),
                gens,
                deps,
                {**agent_environment, step.axis: register_phase},
                source_delta=relative,
                profile=profile,
                phase="drain",
                issue_axes=issue_axes,
                source_issue_axes=source_issue_axes,
            )
            next_prefetch = []
            if index + 1 < len(drain_steps):
                next_prefetch, _ = convert_body(
                    step_regions[PROLOGUE],
                    gens,
                    deps,
                    {**agent_environment, step.axis: register_phase},
                    source_delta=relative + 1,
                    profile=profile,
                    phase="drain",
                    issue_axes=issue_axes,
                    source_issue_axes=source_issue_axes,
                )
            remainder_nodes = tuple(
                node for node in step.body if node is not step_guard
            )
            remainder, _ = convert_body(
                remainder_nodes,
                gens,
                deps,
                {**agent_environment, step.axis: register_phase},
                source_delta=relative,
                profile=profile,
                phase="drain",
                issue_axes=issue_axes,
                source_issue_axes=source_issue_axes,
            )
            body = core + next_prefetch + remainder
        drain_bodies.append(body)
    if has_drain and not drain_bodies:
        drain_bodies = [[]]
    short_body = short_regions.get(SHORT, ())
    short_loops = [node for node in short_body if isinstance(node, Loop)]
    short_bodies = []
    if short_loops:
        short_loop = short_loops[0]
        for index in range(len(drain_bodies)):
            body, _nested = convert_body(
                short_loop.body,
                gens,
                deps,
                {**agent_environment, short_loop.axis: index},
                absolute=True,
                profile=profile,
                phase="short",
                issue_axes=issue_axes,
                source_issue_axes=source_issue_axes,
            )
            short_bodies.append(body)
    return _finish_pipeline(
        program,
        gir,
        gens,
        profile,
        _guard,
        prologue,
        steady_bodies,
        drain_bodies,
        short_bodies,
        variable,
        sub,
        has_drain,
        bool(short_loops),
    )


def lower_to_gir(
    program: LoopProgram,
    profile=None,
    *,
    include_dependencies=True,
    verify=True,
) -> Program:
    """Convert finalized LoopIR to GIR, optionally checking its preconditions."""

    if not isinstance(program, LoopProgram):
        raise TypeError("lower_to_gir expects a LoopIR Program")
    if not isinstance(include_dependencies, bool):
        raise TypeError("include_dependencies must be a bool")
    if not isinstance(verify, bool):
        raise TypeError("lower_to_gir.verify must be a bool")
    if verify:
        program.check()
        if profile is not None:
            profile.validate(program)
    shape = _pipeline_shape(program)
    if shape is not None:
        return _lower_pipeline(
            program,
            shape,
            profile,
            include_dependencies,
        )
    gir = Program()
    regions = _region_bodies(program)
    labels = [
        tag if isinstance(tag, str) else tag.name.lower()
        for tag, _body in regions
    ]
    gir.entry = labels[0]
    gens = {}
    deps = (
        _dependency_map(program, profile)
        if include_dependencies else {}
    )
    issue_axes, source_issue_axes = _read_issue_axes(program, profile)
    agent_environment = {
        key: 0
        for axis, _scope in program.agents
        for key in (axis, axis.name)
    }
    for index, ((tag, body), label) in enumerate(zip(regions, labels)):
        flattened = []
        symbolic = []
        _flatten(body, agent_environment, flattened, symbolic)
        converted = [
            _convert(
                item.operation,
                dict(item.environment),
                gens,
                deps,
                profile=profile,
                issue_axes=issue_axes.get(
                    item.operation._identity,
                    (),
                ),
                source_issue_axes=source_issue_axes,
            )
            for item in flattened
        ]
        next_label = labels[index + 1] if index + 1 < len(labels) else None
        if symbolic:
            if len(symbolic) != 1:
                raise RuntimeError("lowering supports one symbolic loop per block")
            loop = symbolic[0]
            variable = next(
                (
                    param.name
                    for param in loop.hi.free_params()
                ),
                "",
            )
            exit_label = next_label or f"{label}_exit"
            term = LoopBack(Trips(variable), label, exit_label)
            loop_flag = True
        else:
            term = Goto(next_label) if next_label else Return()
            loop_flag = False
        gir.add_block(Block(
            phase=label,
            loop=loop_flag,
            body=converted,
            term=term,
            succs=tuple(
                value for value in (
                    getattr(term, "target", None),
                    getattr(term, "body", None),
                    getattr(term, "exit_target", None),
                )
                if value is not None
            ),
        ))
        if symbolic and next_label is None:
            gir.add_block(Block(
                phase=f"{label}_exit",
                body=[],
                term=Return(),
            ))
    if profile is not None:
        gir.meta.update(profile.as_meta())
    gir.meta["generation_regions"] = {
        gen.id: {"operand": operand, "region": region}
        for (operand, region), gen in gens.items()
    }
    return gir


def steady_chain(program) -> list:
    """Return finalized steady GIR blocks in execution order."""
    if "steady" not in program.blocks:
        return []
    chain = ["steady"]
    while f"steady{len(chain)}" in program.blocks:
        chain.append(f"steady{len(chain)}")
    return chain


def gir_text(program) -> str:
    """Render an already-built GIR Program for diagnostics."""
    if program is None:
        return "# GIR unavailable: this kernel has no GIR program\n"
    try:
        return (
            "# GIR dump (observability; finalized layer-2 program)\n"
            f"# counts = {gir_counts(program)}\n"
            + render_gir(program)
            + "\n"
        )
    except Exception as error:
        return f"# GIR render failed: {error}\n"


__all__ = ["gir_text", "lower_to_gir", "steady_chain"]
