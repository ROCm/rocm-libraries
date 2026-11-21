# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The builder surface rocKE publishes to the projects that build its kernels.

A consumer names a kernel by module and builder (``kernels.gfx950.attention_dense``
and ``build_attention_dense``), calls it as ``builder(spec, *, arch)``, hydrates the
spec dataclass from stored field values (``Spec(**fields)``) and asks the module's
``supports_*`` predicates which targets a spec serves. None of that is visible to
rocKE's own tests, which import the same code under the same names: renaming a
builder, dropping or retyping a spec field, or making a field required passes every
in-tree test and breaks the consumer.

``builder_surface.json`` beside this file pins that surface for every builder a
consumer names. This module captures the surface from live code and reports what
changed against the pinned copy. A change that breaks a consumer fails
``test_builder_surface.py``; additions (a new defaulted field, a new predicate, a
field that stops being required) do not.

Changing the published surface on purpose: make the change, tell the consumers,
then rewrite the snapshot from the live code and commit it with the change::

    python library/tests/contracts/builder_surface.py --update

Adding or renaming a pinned builder: edit its ``module``/``builder`` pair in the
snapshot, then run ``--update``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
import types
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

# Run as a script, this module needs the library packages and the rocke package
# it reads; under pytest the library tests' conftest puts both on the path.
_LIBRARY_ROOT = Path(__file__).resolve().parents[2]
for _package_root in (_LIBRARY_ROOT, _LIBRARY_ROOT.parent / "platform" / "python"):
    if _package_root.is_dir() and str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

__all__ = [
    "OPTIONAL",
    "REQUIRED",
    "SNAPSHOT_PATH",
    "SPEC_ARCH_CALL_SHAPE",
    "UPDATE_COMMAND",
    "BuilderRef",
    "capture_surface",
    "find_breaking_changes",
    "load_snapshot",
    "pinned_builders",
    "write_snapshot",
]

SNAPSHOT_PATH = Path(__file__).with_name("builder_surface.json")
UPDATE_COMMAND = "python library/tests/contracts/builder_surface.py --update"

# Whether a call or a stored spec may omit a parameter or field.
REQUIRED, OPTIONAL = "required", "optional"

# The one call shape a consumer can drive without calling the builder by hand,
# as (name, kind) pairs: the spec positionally, the target by keyword, nothing else.
SPEC_ARCH_CALL_SHAPE = [["spec", "POSITIONAL_OR_KEYWORD"], ["arch", "KEYWORD_ONLY"]]


@dataclasses.dataclass(frozen=True, order=True)
class BuilderRef:
    """A builder as a consumer names it: its module's dotted path and its name."""

    module: str
    builder: str

    def __str__(self) -> str:
        return f"{self.module}.{self.builder}"


def _qualified_name(cls: type) -> str:
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _type_label(annotation: object) -> str:
    """A text form of a resolved annotation that names the type, not its spelling.

    ``Optional[int]`` and ``int | None`` are one type and get one label, so
    modernising an annotation's spelling is not reported as a retyped field.
    Classes are module-qualified, so two same-named specs stay distinct.
    """
    if annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(_type_label(argument) for argument in arguments)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(value) for value in arguments)}]"
    if origin is not None:
        inner = ", ".join(_type_label(argument) for argument in arguments)
        return f"{_type_label(origin)}[{inner}]" if arguments else _type_label(origin)
    if isinstance(annotation, type):
        return _qualified_name(annotation)
    return repr(annotation)


def _requirement(has_default: bool) -> str:
    return OPTIONAL if has_default else REQUIRED


def _parameters(function: Callable) -> list[list[str]]:
    """``[name, kind, requirement]`` per parameter, in order."""
    return [
        [
            name,
            parameter.kind.name,
            _requirement(parameter.default is not inspect.Parameter.empty),
        ]
        for name, parameter in inspect.signature(function).parameters.items()
    ]


def _dataclasses_in(annotation: object) -> list[type]:
    """Dataclass types an annotation names, through unions and containers."""
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return [annotation]
    found = []
    for argument in typing.get_args(annotation):
        found.extend(_dataclasses_in(argument))
    return found


def _spec_fields(spec_class: type) -> dict[str, list[str]]:
    """``{field: [type, requirement]}`` for one spec dataclass."""
    hints = typing.get_type_hints(spec_class)
    return {
        field.name: [
            _type_label(hints[field.name]),
            _requirement(
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
            ),
        ]
        for field in dataclasses.fields(spec_class)
    }


def _capture_specs(roots: Iterable[type]) -> dict[str, dict[str, list[str]]]:
    """Fields of every spec class a consumer must express, nested ones included.

    A nested spec (``problem: UnifiedAttentionProblem``) is stored and hydrated
    like the outer one, so its fields are as much a part of the surface.
    """
    specs: dict[str, dict[str, list[str]]] = {}
    pending = list(roots)
    while pending:
        spec_class = pending.pop()
        name = _qualified_name(spec_class)
        if name in specs:
            continue
        specs[name] = _spec_fields(spec_class)
        for annotation in typing.get_type_hints(spec_class).values():
            pending.extend(_dataclasses_in(annotation))
    return dict(sorted(specs.items()))


def _capture_predicates(module: types.ModuleType) -> dict[str, list[list[str]]]:
    """The module's own ``supports_*`` functions; imported ones belong to the
    module that defines them."""
    return {
        name: _parameters(function)
        for name, function in sorted(vars(module).items())
        if name.startswith("supports_")
        and inspect.isfunction(function)
        and function.__module__ == module.__name__
    }


def capture_surface(refs: Iterable[BuilderRef]) -> dict[str, object]:
    """The published surface of ``refs``, read from the live code.

    A builder that cannot be resolved is recorded with the reason in ``missing``
    rather than raised, so one comparison reports every break at once.
    """
    builders = []
    spec_roots = []
    for ref in sorted(set(refs)):
        entry: dict[str, object] = {"module": ref.module, "builder": ref.builder}
        try:
            module = importlib.import_module(ref.module)
        except ImportError as error:
            entry["missing"] = f"module {ref.module!r} does not import: {error}"
            builders.append(entry)
            continue
        function = getattr(module, ref.builder, None)
        if not callable(function):
            entry["missing"] = f"{ref.module!r} has no builder {ref.builder!r}"
            builders.append(entry)
            continue
        hints = typing.get_type_hints(function)
        spec_class = hints.get(next(iter(inspect.signature(function).parameters), ""))
        entry["parameters"] = _parameters(function)
        entry["returns"] = (
            _type_label(hints["return"]) if "return" in hints else "unannotated"
        )
        if isinstance(spec_class, type) and dataclasses.is_dataclass(spec_class):
            entry["spec"] = _qualified_name(spec_class)
            spec_roots.append(spec_class)
        else:
            entry["spec"] = f"not a dataclass: {spec_class!r}"
        entry["predicates"] = _capture_predicates(module)
        builders.append(entry)
    return {"builders": builders, "specs": _capture_specs(spec_roots)}


def _builder_changes(recorded: dict, live: dict | None) -> list[str]:
    name = f"{recorded['module']}.{recorded['builder']}"
    if live is None or "missing" in live:
        reason = live["missing"] if live else "not captured"
        return [f"{name}: builder is gone ({reason})"]
    changes = [
        f"{name}: {change}"
        for change in _call_changes(recorded["parameters"], live["parameters"])
    ]
    for key in ("returns", "spec"):
        if live[key] != recorded[key]:
            changes.append(f"{name}: {key} changed from {recorded[key]} to {live[key]}")
    for predicate, parameters in recorded["predicates"].items():
        if predicate not in live["predicates"]:
            changes.append(f"{name}: predicate {predicate} is gone")
            continue
        changes += [
            f"{name}: predicate {predicate} {change}"
            for change in _call_changes(parameters, live["predicates"][predicate])
        ]
    return changes


def _call_changes(recorded: list[list[str]], live: list[list[str]]) -> list[str]:
    """How a call written against ``recorded`` parameters breaks against ``live``.

    A new parameter with a default does not break a call; a new required one, a
    dropped or re-kinded one, one that lost its default, or a reordered
    positional one does.
    """
    live_by_name = {parameter[0]: parameter for parameter in live}
    changes = []
    for parameter_name, kind, requirement in recorded:
        live_parameter = live_by_name.get(parameter_name)
        if live_parameter is None:
            changes.append(f"parameter {parameter_name!r} is gone")
        elif live_parameter[1] != kind:
            changes.append(
                f"parameter {parameter_name!r} kind changed from {kind} to "
                f"{live_parameter[1]}"
            )
        elif live_parameter[2] == REQUIRED and requirement == OPTIONAL:
            changes.append(f"parameter {parameter_name!r} lost its default")
    recorded_names = {parameter[0] for parameter in recorded}
    changes += [
        f"new parameter {parameter[0]!r} has no default"
        for parameter in live
        if parameter[0] not in recorded_names and parameter[2] == REQUIRED
    ]
    positional = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
    recorded_order = [p[0] for p in recorded if p[1] in positional]
    live_order = [p[0] for p in live if p[1] in positional]
    if live_order[: len(recorded_order)] != recorded_order:
        changes.append(
            f"positional parameters reordered from {recorded_order} to {live_order}"
        )
    return changes


def _spec_changes(spec: str, recorded: dict, live: dict | None) -> list[str]:
    if live is None:
        return []  # reported through the builder whose spec annotation changed
    changes = []
    for field, (recorded_type, recorded_requirement) in recorded.items():
        if field not in live:
            changes.append(f"{spec}: field {field!r} was dropped")
            continue
        live_type, live_requirement = live[field]
        if live_type != recorded_type:
            changes.append(
                f"{spec}: field {field!r} type changed from {recorded_type} to "
                f"{live_type}"
            )
        if live_requirement == REQUIRED and recorded_requirement == OPTIONAL:
            changes.append(f"{spec}: field {field!r} lost its default")
    for field in sorted(set(live) - set(recorded)):
        if live[field][1] == REQUIRED:
            changes.append(f"{spec}: new field {field!r} has no default")
    return changes


def find_breaking_changes(recorded: dict, live: dict) -> list[str]:
    """Every difference between two surfaces that breaks an existing consumer.

    Breaking: a builder, predicate or spec field that is gone; a builder or
    predicate call that no longer binds (see `_call_changes`); a changed spec class
    or return type; a retyped field; a field a stored spec cannot omit that it
    could omit before. Additions a consumer can ignore (a new defaulted field or
    parameter, a new predicate, a field that gained a default) are not reported.
    """
    live_builders = {
        (entry["module"], entry["builder"]): entry for entry in live["builders"]
    }
    changes = []
    for entry in recorded["builders"]:
        changes += _builder_changes(
            entry, live_builders.get((entry["module"], entry["builder"]))
        )
    for spec, fields in recorded["specs"].items():
        changes += _spec_changes(spec, fields, live["specs"].get(spec))
    return changes


def _to_json(value: object, depth: int = 0) -> str:
    """Indented JSON with each parameter (a list of scalars) on one line, so a
    snapshot diff shows one changed parameter as one changed line."""
    if isinstance(value, list) and all(not isinstance(v, (list, dict)) for v in value):
        return json.dumps(value)
    pad, inner = "  " * depth, "  " * (depth + 1)
    if isinstance(value, dict):
        items = [
            f"{inner}{json.dumps(k)}: {_to_json(v, depth + 1)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}" if items else "{}"
    if isinstance(value, list):
        items = [f"{inner}{_to_json(v, depth + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]" if items else "[]"
    return json.dumps(value)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pinned_builders(snapshot: dict) -> list[BuilderRef]:
    return [
        BuilderRef(entry["module"], entry["builder"]) for entry in snapshot["builders"]
    ]


def write_snapshot(surface: dict, path: Path = SNAPSHOT_PATH) -> None:
    """Write ``surface`` as the pinned snapshot; refuses one with a missing builder."""
    missing = [entry["missing"] for entry in surface["builders"] if "missing" in entry]
    if missing:
        raise ValueError(
            "write_snapshot failed - surface has unresolved builders, expected every "
            "pinned builder to import: " + "; ".join(missing)
        )
    path.write_text(_to_json(surface) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the snapshot from the live code for the builders it pins",
    )
    arguments = parser.parse_args(argv)
    live = capture_surface(pinned_builders(load_snapshot()))
    if arguments.update:
        write_snapshot(live)
        print(f"wrote {SNAPSHOT_PATH.name}: {len(live['builders'])} builders")
        return 0
    changes = find_breaking_changes(load_snapshot(), live)
    print("\n".join(changes) if changes else "builder surface matches the snapshot")
    return 1 if changes else 0


if __name__ == "__main__":
    sys.exit(main())
