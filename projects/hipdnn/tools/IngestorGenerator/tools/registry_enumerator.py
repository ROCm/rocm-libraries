# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Enumerate an engine's kernel catalog from a rocKE dispatcher registry.

    registry_enumerator.py --manifest configs/<slug>.engine.yaml \\
                           --out <scratch>/config.yaml [--shapes <shapes.json>]
    registry_enumerator.py --manifest configs/<slug>.engine.yaml --plan-space

The manifest's ``enumerate`` block names the family's candidate list and request
class; ``pin`` selects exactly one candidate, and each cell of the ``cells`` cross
product (first group outermost) is a request whose ``select_spec``/``sweep_space``
specs become kernels. ``pin`` may also be a list of such selectors: the first
candidate gives every cell its default spec and must admit every cell, and each
later one adds its ``sweep_space`` specs to the cells it admits. A kernel's
``provenance`` names the first pin that supplied its spec. ``space: slice`` keeps
the specs differing from the cell's default only in ``free_fields``.
``space: full`` keeps those matching the default in ``reduction.fields.feature``,
projects varying ``performance`` knobs into the spec, KMD, metadata, bindings and
name, and holds any other varying field at the registry default (with a WARN
unless ``ignored``). The default spec carries ``default: true``.

Two opt-in ``enumerate`` keys let a rule-driven candidate (one spec per request,
picked from request fields) yield one cell per kernel family rather than one per
request: ``selector_fields`` merges the combos that differ only in those cell fields
into one cell holding every combo's pick; ``neutral_fields`` first sets the named
spec fields, which the manifest author asserts are IR-neutral, to the default
spec's value. See `cell_spaces`.

One pack may draw kernels from more than one registry candidate: ``enumerate.also`` lists
further sections, each with its own ``pin``, ``fixed``, ``cells``, ``free_fields``,
``spec_fields`` and ``max_kernels`` (the candidate list, request class, sweep level and
space are inherited from ``enumerate``, and a section may set its own ``space``). Kernels
are emitted primary section first, then each ``also`` section in order; a cell listed by
more than one section keeps the first section's default as its one ``default: true``
floor. A section may also set ``spec_overrides``, spec fields replaced on every spec it
yields (e.g. an output the registry never selects, such as ``emit_lse``), and its own
``kernel_name_template``, so its kernels are new names beside the other sections'; its
default is the ``default: true`` floor of its own (cell, overrides) group. A ``space: full``
section projects its own varying performance knobs; the catalog's projected knobs are the
union over sections, in ``reduction.fields.performance`` order.

``launches`` (optional) maps launch name to a ``kernel_source`` launch override and is
copied onto every kernel: a rocKE builder returning a ``KernelModule`` names its primary
kernel by ``{}`` and each companion by ``{role: <role>}``. A registry candidate that pairs
its kernel with a reduce kernel (a ``reduce_spec`` beside the ``kernel_spec``) needs one,
or the catalog would ship the kernel without the reduce the dispatcher launches after it.

Exit 0 on success; 2 on unusable input. ``--strict`` makes exit 1 mean a gate refusal:
kernels over ``aot_budget_per_arch`` (``--plan-space``: the current classification) or a
``--shapes`` GAP.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import itertools
import json
import os
import re
import sys
import typing
from collections import Counter, OrderedDict
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[5]
_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
if str(_GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_ROOT))

from codegen.config_loader import (  # noqa: E402
    DEFAULT_AOT_BUDGET_PER_ARCH,
    ConfigError,
    check_reduction,
    read_yaml,
)

#: A projected value's Python type -> its KMD field type.
_KMD_TYPES = {bool: "int", int: "int", float: "float", str: "string"}

#: A launch name, as the generator and the descriptor loader accept it.
_LAUNCH_NAME = re.compile(r"[a-z][a-z0-9_]*")

#: The last ``{field}`` slot of a name template; projected knobs render after it.
_LAST_SLOT = re.compile(r"(?<!\{)\{[^{}]+\}(?!\})")


class EnumerationError(RuntimeError):
    """The registry could not be reached, or disagrees with the manifest."""


class OverBudget(EnumerationError):
    """``build_config(strict=True)`` found more kernels than the AOT budget allows."""


def _require(ok: bool, path: str, field: str, shape: str) -> None:
    if not ok:
        raise EnumerationError(f"manifest {path}: '{field}' must be {shape}.")


def _is_names(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _validate_manifest(path: str, manifest: dict) -> None:
    """Check the structure and types of every manifest field `Registry`,
    `plan_space` and `build_config` read, so a malformed manifest is an
    `EnumerationError` naming the field rather than a crash in a consumer."""
    for key in ("source", "builder", "arch"):
        _require(
            isinstance(manifest[key], str) and manifest[key], path, key, "a string"
        )
    for key in (
        "authored_subpath",
        "pack",
        "provider_root",
        "dialect",
        "kernel_source_kind",
        "workspace_policy",
        "kernel_name_template",
    ):
        if key in manifest:
            _require(isinstance(manifest[key], str), path, key, "a string")
    engine = manifest["engine"]
    _require(isinstance(engine, dict), path, "engine", "a mapping")
    _require(isinstance(engine.get("name"), str), path, "engine.name", "a string")
    kmd_fields = manifest["kmd_fields"]
    _require(
        isinstance(kmd_fields, list)
        and all(
            isinstance(f, dict) and isinstance(f.get("name"), str) for f in kmd_fields
        ),
        path,
        "kmd_fields",
        "a list of mappings, each with a 'name'",
    )
    specialization = manifest["specialization"]
    _require(isinstance(specialization, dict), path, "specialization", "a mapping")
    _require(
        _is_names(specialization.get("metadata_fields")),
        path,
        "specialization.metadata_fields",
        "a list of field names",
    )
    for key in ("vocabulary", "bindings"):
        _require(
            key not in specialization or isinstance(specialization[key], dict),
            path,
            f"specialization.{key}",
            "a mapping",
        )
    launches = manifest.get("launches")
    _require(
        launches is None
        or (
            isinstance(launches, dict)
            and len(launches) >= 2
            and all(isinstance(n, str) and _LAUNCH_NAME.fullmatch(n) for n in launches)
            and all(isinstance(v, dict) for v in launches.values())
        ),
        path,
        "launches",
        "a mapping of two or more launch names ([a-z][a-z0-9_]*) to override mappings",
    )
    decl = manifest["enumerate"]
    _require(isinstance(decl, dict), path, "enumerate", "a mapping")
    also = decl.get("also")
    _require(
        also is None
        or (isinstance(also, list) and all(isinstance(a, dict) for a in also)),
        path,
        "enumerate.also",
        "a list of mappings",
    )
    for where, section in [("enumerate", decl)] + [
        (f"enumerate.also[{i}]", a) for i, a in enumerate(also or [])
    ]:
        for key in (
            "free_fields",
            "spec_fields",
            "selector_fields",
            "neutral_fields",
            "cell_keyed_by",
        ):
            _require(
                section.get(key) is None or _is_names(section[key]),
                path,
                f"{where}.{key}",
                "a list of field names",
            )
        overrides = section.get("spec_overrides")
        _require(
            overrides is None
            or (
                isinstance(overrides, dict)
                and overrides
                and _is_names(list(overrides))
                and all(
                    isinstance(v, (bool, int, float, str)) for v in overrides.values()
                )
            ),
            path,
            f"{where}.spec_overrides",
            "a non-empty mapping of spec field names to scalar values",
        )
        _require(
            isinstance(section.get("kernel_name_template", ""), str),
            path,
            f"{where}.kernel_name_template",
            "a string",
        )
        limit = section.get("max_kernels")
        _require(
            limit is None
            or (isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0),
            path,
            f"{where}.max_kernels",
            "a non-negative integer",
        )
        _require(
            isinstance(section.get("sweep_level", ""), str),
            path,
            f"{where}.sweep_level",
            "a string",
        )
    _require(
        manifest.get("ground_truth") is None
        or isinstance(manifest["ground_truth"], dict),
        path,
        "ground_truth",
        "a mapping",
    )
    _require(
        manifest.get("reduction") is None or isinstance(manifest["reduction"], dict),
        path,
        "reduction",
        "a mapping",
    )


def load_manifest(path: str) -> dict:
    """The manifest at ``path``, parsed with `read_yaml` (a repeated key is an
    error) and checked by `_validate_manifest`."""
    try:
        loaded = read_yaml(Path(path))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EnumerationError(f"manifest {path} is not readable YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise EnumerationError(f"manifest {path} must be a mapping")
    for key in (
        "engine",
        "kmd_fields",
        "specialization",
        "arch",
        "source",
        "builder",
        "enumerate",
    ):
        if key not in loaded:
            raise EnumerationError(f"manifest {path} is missing '{key}'")
    _validate_manifest(path, loaded)
    return loaded


def bind_provider(provider_root: str | None) -> None:
    """Put the rocKE library on ``sys.path``. ``ROCKE_PROVIDER_ROOT`` is the
    fallback; a relative root is repository-relative."""
    root = provider_root or os.environ.get("ROCKE_PROVIDER_ROOT")
    if not root:
        raise EnumerationError(
            "no provider_root in the manifest and no ROCKE_PROVIDER_ROOT set; the "
            "dispatcher registry cannot be imported without the rocKE library."
        )
    expanded = Path(os.path.expanduser(root))
    base = expanded if expanded.is_absolute() else _REPO / expanded
    for sub in ("rocke/library", "rocke/platform/python"):
        candidate = base / sub
        if not candidate.is_dir():
            raise EnumerationError(
                f"{candidate} does not exist; is {base} a provider root?"
            )
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _plain(value):
    """A spec or request value as it is written into a descriptor."""
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _rendered(value):
    """A value as a metadata entry or a name token reads it: bools are 0/1."""
    value = _plain(value)
    return int(value) if isinstance(value, bool) else value


def _order_key(value) -> tuple:
    """A total order over mixed ``None``/bool/int/float/str knob values; numbers
    sort numerically."""
    value = _plain(value)
    native = isinstance(value, (int, float, str))
    return (value is None, type(value).__name__, value if native else repr(value))


def _unwrap(spec):
    return getattr(spec, "kernel_spec", spec)


def _load_callable(reference, what: str):
    if not isinstance(reference, str):
        raise EnumerationError(
            f"enumerate.{what} must be a 'module:attribute' string, got {reference!r}."
        )
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise EnumerationError(
            f"enumerate.{what} {reference!r} must be written 'module:attribute'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EnumerationError(
            f"cannot import the rocKE dispatcher registry module '{module_name}' "
            f"(enumerate.{what}): {exc}"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise EnumerationError(
            f"enumerate.{what}: module '{module_name}' has no '{attribute}'."
        ) from exc


def _mapping(decl: dict, key: str) -> dict:
    value = decl.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EnumerationError(f"enumerate.{key} must be a mapping, got {value!r}.")
    return dict(value)


def cell_groups(raw) -> list:
    """``[(fields, rows), ...]`` from ``enumerate.cells``: each row one raw value per
    field, as the manifest writes it."""
    if not isinstance(raw, list) or not raw:
        raise EnumerationError("enumerate.cells must be a non-empty list of groups.")
    groups = []
    seen: set = set()
    for index, group in enumerate(raw):
        where = f"enumerate.cells[{index}]"
        if not isinstance(group, dict) or set(group) != {"fields", "values"}:
            raise EnumerationError(
                f"{where} must be a mapping with exactly 'fields' and 'values'."
            )
        fields = group["fields"]
        if not _is_names(fields) or not fields:
            raise EnumerationError(f"{where}.fields must be a list of field names.")
        repeated = sorted(seen & set(fields))
        if repeated or len(set(fields)) != len(fields):
            raise EnumerationError(
                f"{where} repeats cell field(s) {repeated or fields}."
            )
        seen |= set(fields)
        values = group["values"]
        if not isinstance(values, list) or not values:
            raise EnumerationError(f"{where}.values must be a non-empty list.")
        rows = []
        for value in values:
            row = value if isinstance(value, list) else [value]
            if len(row) != len(fields):
                raise EnumerationError(
                    f"{where}: value {value!r} does not have one entry per field "
                    f"{fields}."
                )
            rows.append(row)
        groups.append((list(fields), rows))
    return groups


def check_value_map_cells(groups: list, value_map) -> None:
    """Each ``ground_truth.value_map`` field that is also a cell field maps exactly
    the values its cells declare (an enum cell by its member name), so no declared
    value is unmapped and no key is dead."""
    if not isinstance(value_map, dict):
        return
    declared: dict = {}
    for fields, rows in groups:
        for position, field in enumerate(fields):
            for row in rows:
                value = row[position]
                if isinstance(value, (list, dict)):
                    raise EnumerationError(
                        f"enumerate.cells value {value!r} of '{field}' must be a scalar."
                    )
                declared.setdefault(field, set()).add(
                    value.name if isinstance(value, enum.Enum) else value
                )
    for field, table in value_map.items():
        if field not in declared or not isinstance(table, dict):
            continue
        missing = sorted(map(str, declared[field] - set(table)))
        extra = sorted(map(str, set(table) - declared[field]))
        if missing or extra:
            raise EnumerationError(
                f"ground_truth.value_map.{field} must have exactly the keys of the "
                f"enumerate.cells values of '{field}': missing {missing}, extra "
                f"{extra}."
            )


class Registry:
    """One manifest's view of the registry: its candidate, request class and cells."""

    #: What an ``enumerate.also`` section takes from ``enumerate`` unless it sets its own.
    _INHERITED = (
        "candidates",
        "request",
        "sweep_level",
        "space",
        "selector_fields",
        "cell_keyed_by",
    )
    #: The ``enumerate`` keys under which `cell_spaces` rewrites or merges the
    #: registry's specs, so a request combo's ``select_spec`` is no longer a cell's
    #: default kernel.
    _RESHAPING = (
        "spec_overrides",
        "neutral_fields",
        "selector_fields",
        "cell_keyed_by",
    )

    def __init__(self, manifest: dict, decl: dict | None = None):
        primary = decl is None
        decl = manifest["enumerate"] if primary else decl
        where = "enumerate" if primary else "an enumerate.also section"
        for key in ("candidates", "request", "pin", "cells", "sweep_level"):
            if key not in decl:
                raise EnumerationError(f"the manifest's {where} lacks '{key}'.")
        self.decl = decl
        #: ``(cell_key, kernel spec with spec_overrides)`` -> ``(pin, tuning_id)`` of the pin that supplied
        #: it, filled by `space`. ``tuning_id`` is None for a spec the registry hands
        #: back unwrapped.
        self.origins: dict = {}
        self.space_kind = decl.get("space", "slice")
        if self.space_kind not in ("slice", "full"):
            raise EnumerationError(
                f"enumerate.space must be 'slice' or 'full', got {self.space_kind!r}."
            )
        candidates_fn = _load_callable(decl["candidates"], "candidates")
        self.request_cls = _load_callable(decl["request"], "request")
        if not dataclasses.is_dataclass(self.request_cls):
            raise EnumerationError(
                f"enumerate.request {decl['request']!r} is not a dataclass request."
            )
        self.request_fields = {f.name: f for f in dataclasses.fields(self.request_cls)}
        try:
            self.hints = typing.get_type_hints(self.request_cls)
        except Exception as exc:  # an unresolvable annotation names nothing useful
            raise EnumerationError(
                f"cannot resolve the type hints of {decl['request']!r}: {exc}"
            ) from exc

        raw_pins = decl["pin"]
        if isinstance(raw_pins, list):
            if not raw_pins or not all(isinstance(p, dict) for p in raw_pins):
                raise EnumerationError(
                    f"enumerate.pin must be a mapping or a non-empty list of mappings, "
                    f"got {raw_pins!r}."
                )
            pins = [dict(p) for p in raw_pins]
        else:
            pins = [_mapping(decl, "pin")]
        self.fixed = _mapping(decl, "fixed")
        arch = manifest.get("arch")
        if "arch" in self.fixed and self.fixed["arch"] != arch:
            raise EnumerationError(
                f"enumerate.fixed.arch {self.fixed['arch']!r} disagrees with the "
                f"manifest's arch {arch!r}."
            )
        clash = sorted(set().union(*pins) & set(self.fixed))
        if clash:
            raise EnumerationError(
                f"enumerate.pin and enumerate.fixed both set {clash}."
            )
        self.groups = self._parse_groups(decl["cells"])
        self.cell_fields = [f for fields, _ in self.groups for f in fields]
        check_value_map_cells(
            self.groups, (manifest.get("ground_truth") or {}).get("value_map")
        )
        self.selector_fields = list(decl.get("selector_fields") or [])
        unknown = [n for n in self.selector_fields if n not in self.cell_fields]
        if unknown:
            raise EnumerationError(
                f"enumerate.selector_fields names {unknown}, which are not "
                f"enumerate.cells fields {self.cell_fields}."
            )
        self.neutral_fields = list(decl.get("neutral_fields") or [])
        #: Spec fields replaced on every spec this section yields.
        self.spec_overrides = _mapping(decl, "spec_overrides")
        self.cell_keyed_by = list(decl.get("cell_keyed_by") or [])
        for name in itertools.chain(*pins, self.fixed, self.cell_fields):
            if name not in self.request_fields:
                raise EnumerationError(
                    f"enumerate names request field '{name}', which "
                    f"{self.request_cls.__name__} does not define."
                )
        pins = [{k: self.coerce(k, v, strict=True) for k, v in p.items()} for p in pins]
        self.fixed = {k: self.coerce(k, v, strict=True) for k, v in self.fixed.items()}

        try:
            candidates = list(candidates_fn())
        except Exception as exc:  # the registry's own failure, not ours to classify
            raise EnumerationError(
                f"the registry's candidate list could not be built "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        pinned = [_pinned_candidate(candidates, pin) for pin in pins]
        # The first pin: every cell's default spec, and the request shapes_report
        # judges. Each later pin only adds specs (`space`).
        self.pin, self.candidate = pins[0], pinned[0]
        self.extra_pins = list(zip(pins[1:], pinned[1:]))

        try:
            from rocke.dispatch.tuning import walk
        except ImportError as exc:
            raise EnumerationError(
                f"rocKE dispatch is not importable (is the provider root bound?): {exc}"
            ) from exc

        self._sweep_level = walk.sweep_level

        #: Whether the registry paired a kernel with a reduce kernel (``reduce_spec``),
        #: which the catalog must then launch too (``launches``).
        self.pairs_reduce = False
        #: The ``enumerate.also`` sections, each a `Registry` of its own.
        self.also: list = []
        #: The ``spec_overrides`` fields another section does not pin to the same
        #: value: across the engine they take more than one value, so they may be a
        #: ``reduction.fields.feature`` axis. Filled by the primary section.
        self.axis_overrides: frozenset = frozenset()
        if primary and decl.get("also"):
            for index, extra in enumerate(decl["also"]):
                merged = {
                    **{k: decl[k] for k in self._INHERITED if k in decl},
                    **extra,
                }
                section = Registry(manifest, merged)
                if section.cell_fields != self.cell_fields:
                    raise EnumerationError(
                        f"enumerate.also[{index}] crosses {section.cell_fields}, not "
                        f"{self.cell_fields}: every section's cells are the same "
                        f"fields, so a shape's coverage is one key."
                    )
                for key in ("selector_fields", "cell_keyed_by"):
                    if getattr(section, key) != getattr(self, key):
                        raise EnumerationError(
                            f"enumerate.also[{index}] sets {key} "
                            f"{getattr(section, key)}, not {getattr(self, key)}: every "
                            f"section keys a shape's cell the same way (omit {key} "
                            f"to inherit it)."
                        )
                self.also.append(section)
            sections = self.sections()
            for section in sections:
                section.axis_overrides = frozenset(
                    name
                    for name, value in section.spec_overrides.items()
                    if any(
                        other.spec_overrides.get(name, _Missing) != value
                        for other in sections
                        if other is not section
                    )
                )

    def sections(self) -> list:
        """This section, then each ``enumerate.also`` section."""
        return [self, *self.also]

    def cell_key(self, cell: dict) -> tuple:
        return tuple(cell.get(f) for f in self.cell_fields)

    def _parse_groups(self, raw) -> list:
        return [
            (
                fields,
                [
                    tuple(
                        self.coerce(f, v, strict=True, by_name=True)
                        for f, v in zip(fields, row)
                    )
                    for row in rows
                ],
            )
            for fields, rows in cell_groups(raw)
        ]

    def coerce(self, field: str, value, strict: bool = False, by_name: bool = False):
        """``value`` as the request's own type: a member, by name or value, of the
        first enum in the field's type that has one. ``strict`` refuses an unknown
        string for an enum field; ``by_name`` also refuses a value that matches a
        member only by value, since the README asks for member names and
        ``ground_truth graphs`` reads them as such."""
        hint = self.hints.get(field)
        args = typing.get_args(hint) or (hint,)
        enums = [a for a in args if isinstance(a, type) and issubclass(a, enum.Enum)]
        for arg in enums:
            if isinstance(value, arg):
                return value
            if isinstance(value, str) and value in arg.__members__:
                return arg[value]
            try:
                member = arg(value)
            except (ValueError, TypeError):
                continue
            if by_name:
                raise EnumerationError(
                    f"{value!r} is the value of {arg.__name__}.{member.name}, not a "
                    f"member name (request field '{field}'); enumerate.cells takes "
                    f"member names. Members: {[m.name for m in arg]}."
                )
            return member
        if strict and enums and isinstance(value, str) and str not in args:
            raise EnumerationError(
                f"'{value}' is not a member of {' or '.join(a.__name__ for a in enums)} "
                f"(request field '{field}'). Members: "
                f"{[m.name for a in enums for m in a]}."
            )
        return value

    def cells(self) -> list:
        """Every cell the manifest lists, in catalog order."""
        out = []
        for combo in itertools.product(*(rows for _, rows in self.groups)):
            cell = {}
            for (fields, _), row in zip(self.groups, combo):
                cell.update(zip(fields, row))
            out.append(cell)
        return out

    def request(self, cell: dict, pin: dict | None = None):
        try:
            return self.request_cls(**self.fixed, **(pin or self.pin), **cell)
        except (TypeError, ValueError) as exc:
            raise EnumerationError(
                f"cannot build a {self.request_cls.__name__} for cell "
                f"{_show(cell)}: {exc}"
            ) from exc

    def space(self, cell: dict) -> tuple:
        """``(select_spec spec, sweep specs)`` for one cell, both unwrapped: the first
        pinned candidate's, then the sweep specs of every later pinned candidate that
        admits the cell, each spec once. Each spec is traced to the first pin that
        supplied it."""
        key = self.cell_key(cell)
        base, specs, ids = self._candidate_space(cell, self.pin, self.candidate)
        for spec in (base, *specs):
            self.origins.setdefault(
                (key, self.overridden(spec, cell)), (self.pin, ids.get(spec))
            )
        for pin, candidate in self.extra_pins:
            request = self.request(cell, pin)
            try:
                ok, _why = candidate.admits(request)
            except (ValueError, TypeError, AttributeError) as exc:
                raise EnumerationError(
                    f"the registry cannot judge cell {_show(cell)} for pin {pin} "
                    f"({type(exc).__name__}): {exc}"
                ) from exc
            if ok:
                _, more, ids = self._candidate_space(cell, pin, candidate)
                added = [s for s in more if s not in specs]
                for spec in added:
                    self.origins.setdefault(
                        (key, self.overridden(spec, cell)), (pin, ids.get(spec))
                    )
                specs += added
        return base, specs

    def reshaped_by(self) -> list:
        """The `_RESHAPING` keys this section sets."""
        return [key for key in self._RESHAPING if getattr(self, key)]

    def base_spec(self, cell: dict, pin: dict | None = None):
        """The unwrapped ``select_spec`` spec for one request combo under one of the
        manifest's pins (default the first, whose base spec is the cell's default
        kernel). Raises for a section that sets any `_RESHAPING` key: `cell_spaces`
        rewrites or merges its specs, so a raw combo's pick is no cell's default."""
        reshaped = self.reshaped_by()
        if reshaped:
            raise EnumerationError(
                f"base_spec answers a raw request combo's select_spec, which is not a "
                f"cell's default kernel when enumerate sets {reshaped}."
            )
        pin = self.pin if pin is None else pin
        pairs = [(self.pin, self.candidate), *self.extra_pins]
        candidate = next((c for p, c in pairs if p == pin), None)
        if candidate is None:
            raise EnumerationError(f"{pin} is not one of the manifest's enumerate.pin.")
        base, _, _ = self._candidate_space(cell, pin, candidate)
        return base

    def _candidate_space(self, cell: dict, pin: dict, candidate) -> tuple:
        """``(base, specs, {spec: tuning_id})`` for one pinned candidate."""
        request = self.request(cell, pin)
        try:
            ok, why = candidate.admits(request)
        except (ValueError, TypeError, AttributeError) as exc:
            raise EnumerationError(
                f"the registry cannot judge cell {_show(cell)} "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        if not ok:
            raise EnumerationError(
                f"the registry declines cell {_show(cell)}, which the manifest's "
                f"cells promise: {why}"
            )
        try:
            base_wrapped = candidate.select_spec(request)
            base = _unwrap(base_wrapped)
            with self._sweep_level(self.decl["sweep_level"]):
                wrapped = list(candidate.sweep_space(request))
            specs = [_unwrap(s) for s in wrapped]
        except (ValueError, TypeError, AttributeError) as exc:
            raise EnumerationError(
                f"the registry rejects cell {_show(cell)}, which the manifest's cells "
                f"promise ({type(exc).__name__}): {exc}"
            ) from exc
        if base is None or any(spec is None for spec in specs):
            raise EnumerationError(
                f"the registry routes cell {_show(cell)} without a kernel_spec; the "
                f"pinned candidate must resolve to a kernel."
            )
        if any(
            getattr(item, "reduce_spec", None) is not None
            for item in (base_wrapped, *wrapped)
        ):
            self.pairs_reduce = True
        ids: dict = {}
        for item in (base_wrapped, *wrapped):
            tuning_id = getattr(item, "tuning_id", None)
            if tuning_id:
                ids.setdefault(_unwrap(item), str(tuning_id))
        return base, specs, ids

    def overridden(self, spec, cell: dict):
        """``spec`` with this section's ``spec_overrides`` applied (``spec`` itself
        when there are none). The spec class validates the result."""
        if spec is None or not self.spec_overrides:
            return spec
        unknown = sorted(set(self.spec_overrides) - set(_spec_fields(spec)))
        if unknown:
            raise EnumerationError(
                f"enumerate.spec_overrides names {unknown}, which the registry spec "
                f"{type(spec).__name__} does not have."
            )
        return _overridden(spec, self.spec_overrides, cell)

    def overrides_key(self) -> tuple:
        """What, beside the cell, groups this section's kernels for the
        ``default: true`` floor."""
        return tuple(sorted(self.spec_overrides.items()))

    def provenance(self, cell: dict, spec) -> dict:
        """The supplying pin's ``spec_id`` and, when the registry names the kernel,
        that pin's ``tuning_id`` for it: what lets a shipped descriptor be traced back
        to the registry entry that produced it. Informational; never compared."""
        pin, tuning_id = self.origins.get((self.cell_key(cell), spec), (self.pin, None))
        out = {}
        if pin.get("spec_id"):
            out["rocke_spec_id"] = str(pin["spec_id"])
        if tuning_id and tuning_id != "auto":
            out["rocke_tuning_id"] = tuning_id
        return out

    def slice_of(self, base, specs: list) -> list:
        free = set(self.decl.get("free_fields") or [])
        unknown = sorted(free - set(_spec_fields(base)))
        if unknown:
            raise EnumerationError(
                f"enumerate.free_fields names {unknown}, which the registry spec "
                f"{type(base).__name__} does not have."
            )
        out = [
            ks
            for ks in specs
            if all(
                getattr(ks, f.name) == getattr(base, f.name)
                for f in dataclasses.fields(ks)
                if f.name not in free
            )
        ]
        # Default first, then by the free fields, independent of sweep order.
        free_order = sorted(free)
        key = lambda ks: tuple(
            _order_key(getattr(ks, n)) for n in free_order
        )  # noqa: E731
        base_key = key(base)
        return sorted(out, key=lambda ks: (key(ks) != base_key, key(ks)))


class _Missing:
    """No such candidate attribute; equal to nothing a pin can name."""


def _pinned_candidate(candidates: list, pin: dict):
    """The one registry candidate whose attributes equal every value of ``pin``."""
    matched = [
        c
        for c in candidates
        if all(getattr(c, k, _Missing) == v for k, v in pin.items())
    ]
    if len(matched) != 1:
        shown = (
            [getattr(c, "name", repr(c)) for c in matched]
            if matched
            else sorted({getattr(c, "name", repr(c)) for c in candidates})
        )
        raise EnumerationError(
            f"enumerate.pin {pin} matches {len(matched)} registry candidates, "
            f"not one. {'Matched' if matched else 'Available'}: {shown}."
        )
    return matched[0]


def _show(cell: dict) -> str:
    return "{" + ", ".join(f"{k}: {_shown(v)}" for k, v in cell.items()) + "}"


def _shown(value) -> str:
    return value.name if isinstance(value, enum.Enum) else repr(value)


@dataclasses.dataclass
class Classification:
    """``reduction.fields`` as the enumerator reads it."""

    feature: list
    #: name -> unset accessor (or None), in priority order.
    performance: "OrderedDict[str, str | None]"
    ignored: list
    budget: int

    @classmethod
    def from_manifest(cls, manifest: dict, required: bool) -> "Classification":
        reduction = manifest.get("reduction")
        if reduction is None:
            if required:
                raise EnumerationError(
                    "space: full and --plan-space need a 'reduction' block: its "
                    "feature fields define a cell and its performance knobs the arms."
                )
            return cls([], OrderedDict(), [], DEFAULT_AOT_BUDGET_PER_ARCH)
        try:
            check_reduction(
                reduction, kmd_fields={f["name"] for f in manifest["kmd_fields"]}
            )
        except ConfigError as exc:
            raise EnumerationError(str(exc)) from exc
        fields = reduction.get("fields") or {}
        return cls(
            feature=list(fields.get("feature") or []),
            performance=OrderedDict(
                (e["name"], e.get("unset_resolves_by"))
                for e in fields.get("performance") or []
            ),
            ignored=list(fields.get("ignored") or []),
            budget=reduction.get("aot_budget_per_arch", DEFAULT_AOT_BUDGET_PER_ARCH),
        )

    def with_performance(self, performance: "OrderedDict") -> "Classification":
        return dataclasses.replace(self, performance=performance)


@dataclasses.dataclass
class FullSpace:
    """The ``space: full`` selection over a list of cell spaces."""

    #: Per cell: the kept specs, default first.
    kept: list
    #: Performance knobs that vary in some cell, in priority order.
    projected: list
    #: ``(field, held value, reason)`` for every warned hold, in first-seen order.
    held: list


def _spec_fields(spec) -> list:
    return [f.name for f in dataclasses.fields(spec)]


def _feature_matched(spaces: list, classification: Classification) -> list:
    """``[(base, specs equal to base in every feature field), ...]``, after checking
    that ``reduction.fields`` names only spec fields."""
    out = []
    for base, specs in spaces:
        names = set(_spec_fields(base))
        for kind in ("feature", "performance", "ignored"):
            # A feature may be a derived attribute (a property): it only filters.
            unknown = [
                n
                for n in getattr(classification, kind)
                if n not in names and not (kind == "feature" and hasattr(base, n))
            ]
            if unknown:
                raise EnumerationError(
                    f"reduction.fields.{kind} names {unknown}, which the registry "
                    f"spec {type(base).__name__} does not have."
                )
        want = {f: getattr(base, f) for f in classification.feature}
        out.append(
            (
                base,
                [s for s in specs if all(getattr(s, f) == v for f, v in want.items())],
            )
        )
    return out


def _unresolvable(sames: list, performance: "OrderedDict") -> set:
    """Performance knobs with no ``unset_resolves_by`` left ``None`` in some cell:
    a KMD field cannot hold ``None``, so each is held wherever it varies."""
    return {
        name
        for name, accessor in performance.items()
        if accessor is None
        and any(getattr(s, name) is None for _, same in sames for s in same)
    }


def select_full(spaces: list, classification: Classification) -> FullSpace:
    """Apply the ``space: full`` rules to ``[(base, specs), ...]``."""
    performance = classification.performance
    ignored = set(classification.ignored)
    held_notes: "OrderedDict[tuple, None]" = OrderedDict()
    projected: set = set()
    sames = _feature_matched(spaces, classification)
    unresolvable = _unresolvable(sames, performance)
    filtered = []
    for base, same in sames:
        varying = [
            n
            for n in _spec_fields(base)
            if len({repr(getattr(s, n)) for s in same}) > 1
        ]
        hold = []
        for name in varying:
            if name in unresolvable:
                hold.append(name)
                held_notes[(name, repr(getattr(base, name)), "no unset accessor")] = (
                    None
                )
            elif name in performance:
                projected.add(name)
            elif name in ignored:
                hold.append(name)
            else:
                hold.append(name)
                held_notes[(name, repr(getattr(base, name)), "unclassified")] = None
        kept = [s for s in same if all(getattr(s, n) == getattr(base, n) for n in hold)]
        filtered.append((base, kept))

    order = [n for n in performance if n in projected]
    accessors = {n: performance[n] for n in order if performance[n]}
    out = []
    for base, kept in filtered:
        kept = _collapse_unset(kept, accessors, base)
        key = lambda s: tuple(_order_key(getattr(s, n)) for n in order)  # noqa: E731
        base_key = key(base)
        out.append(sorted(kept, key=lambda s: (key(s) != base_key, key(s))))
    return FullSpace(
        kept=out,
        projected=order,
        held=[(f, v, why) for f, v, why in held_notes],
    )


def _collapse_unset(kept: list, accessors: dict, base) -> list:
    """Keep one spec per fully resolved value: an unset knob and its resolved
    value are the same binary. The default wins, then the most-unset spec."""
    if not accessors:
        return kept
    best: dict = {}
    for spec in kept:
        key = tuple(
            repr(_resolved(spec, n, accessors.get(n))) for n in _spec_fields(spec)
        )
        rank = (spec != base, -sum(getattr(spec, n) is None for n in accessors))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, spec)
    survivors = {id(spec) for _, spec in best.values()}
    return [s for s in kept if id(s) in survivors]


def _resolved(spec, knob: str, accessor: str | None):
    value = getattr(spec, knob)
    if value is None and accessor:
        try:
            return getattr(spec, accessor)()
        except AttributeError:
            raise EnumerationError(
                f"reduction names unset accessor '{accessor}' for '{knob}', which the "
                f"registry spec {type(spec).__name__} does not have."
            ) from None
    return value


@dataclasses.dataclass
class Enumeration:
    """Every kernel the manifest names, in catalog order."""

    #: ``(cell, spec, is_default)`` per kernel.
    rows: list
    #: Projected performance knobs (``space: full`` only), in priority order.
    projected: list
    #: Projected knob -> its unset accessor, for knobs declaring one.
    accessors: dict
    held: list
    classification: Classification
    #: The `Registry` section that produced each row, parallel to ``rows``.
    sections: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CellSpaces:
    """The catalog's cells and each one's ``(default spec, specs)``, in catalog order."""

    cells: list
    spaces: list
    #: Per cell, ``(request combo, its normalised select_spec)`` for every merged
    #: combo; empty when the manifest merges nothing.
    picks: list
    #: Request combos before merging.
    combos: int


def _overridden(spec, overrides: dict, combo: dict):
    """``spec`` with every ``enumerate.spec_overrides`` field set; the spec's own
    validation decides whether the value builds."""
    moved = {n: v for n, v in overrides.items() if getattr(spec, n) != v}
    if not moved:
        return spec
    try:
        return dataclasses.replace(spec, **moved)
    except (TypeError, ValueError) as exc:
        raise EnumerationError(
            f"enumerate.spec_overrides {moved} does not build for request "
            f"{_show(combo)}: {exc}"
        ) from exc


def cell_spaces(registry: Registry, classification: Classification) -> CellSpaces:
    """The registry's space per catalog cell.

    With none of ``spec_overrides``, ``selector_fields``, ``neutral_fields`` or
    ``cell_keyed_by`` every request combo is a cell and its space is the
    registry's, unchanged. Otherwise:

    0. every spec takes the ``spec_overrides`` values (the manifest's deliberate
       departure from the candidate's own pick);
    1. each spec's ``neutral_fields`` take the value of the cell's default spec;
    2. combos equal outside ``selector_fields`` and in every ``cell_keyed_by``
       attribute of their ``select_spec`` merge into one cell, ordered by its first
       combo in catalog order, without the selector fields and with the keyed
       attributes;
    3. that cell's default spec is its first combo's ``select_spec``, and its specs
       are the union, first occurrence kept, of each combo's ``select_spec`` then
       its ``sweep_space`` specs, combo by combo in catalog order. Every one of them
       must keep the cell's keyed values.
    """
    combos = registry.cells()
    raw = [registry.space(cell) for cell in combos]
    selectors, neutral = registry.selector_fields, registry.neutral_fields
    overrides, keyed = registry.spec_overrides, registry.cell_keyed_by
    if not combos or not registry.reshaped_by():
        return CellSpaces(combos, raw, [], len(combos))
    absent = [a for a in keyed if not hasattr(raw[0][0], a)]
    if absent:
        raise EnumerationError(
            f"enumerate.cell_keyed_by names {absent}, which the registry spec "
            f"{type(raw[0][0]).__name__} does not have."
        )
    taken = [a for a in keyed if a in registry.cell_fields]
    if taken:
        raise EnumerationError(
            f"enumerate.cell_keyed_by names {taken}, which are enumerate.cells "
            f"fields: a cell field already keys the cell."
        )
    names = _spec_fields(raw[0][0])
    for key, declared in (
        ("neutral_fields", neutral),
        ("spec_overrides", list(overrides)),
    ):
        unknown = [n for n in declared if n not in names]
        if unknown:
            raise EnumerationError(
                f"enumerate.{key} names {unknown}, which the registry spec "
                f"{type(raw[0][0]).__name__} does not have."
            )
    for key, declared in (("selector_fields", selectors), ("neutral_fields", neutral)):
        for kind in ("feature", "performance"):
            clash = [n for n in declared if n in getattr(classification, kind)]
            if clash:
                raise EnumerationError(
                    f"enumerate.{key} names {clash}, which reduction.fields.{kind} "
                    f"also names: a {kind} field defines a kernel, so it cannot "
                    f"only steer the selector or leave the IR unchanged."
                )
    for kind in ("feature", "performance"):
        # A field another `also` section pins differently is an axis across the
        # engine (e.g. `emit_lse`), so it may still be a feature field.
        allowed = registry.axis_overrides if kind == "feature" else frozenset()
        clash = [
            n
            for n in overrides
            if n in getattr(classification, kind) and n not in allowed
        ]
        if clash:
            raise EnumerationError(
                f"enumerate.spec_overrides names {clash}, which reduction.fields.{kind} "
                f"also names: a pinned field takes one value, so it is no {kind} axis."
            )
    both = sorted(set(overrides) & set(neutral))
    if both:
        raise EnumerationError(
            f"enumerate.spec_overrides and enumerate.neutral_fields both name {both}."
        )
    if overrides:
        raw = [
            (
                _overridden(base, overrides, combo),
                list(
                    OrderedDict(
                        (_identity(s), s)
                        for s in (_overridden(s, overrides, combo) for s in specs)
                    ).values()
                ),
            )
            for combo, (base, specs) in zip(combos, raw)
        ]
    if not selectors and not neutral and not keyed:
        return CellSpaces(combos, raw, [], len(combos))
    merged: "OrderedDict[tuple, list]" = OrderedDict()
    for cell, space in zip(combos, raw):
        key = tuple((f, v) for f, v in cell.items() if f not in selectors)
        key += tuple((a, getattr(space[0], a)) for a in keyed)
        merged.setdefault(key, []).append((cell, space))
    cells, spaces, picks = [], [], []
    for key, members in merged.items():
        default = members[0][1][0]
        keyed_values = {a: getattr(default, a) for a in keyed}

        def normalised(spec, combo):
            moved = {
                n: getattr(default, n)
                for n in neutral
                if getattr(spec, n) != getattr(default, n)
            }
            if not moved:
                return spec
            try:
                return dataclasses.replace(spec, **moved)
            except (TypeError, ValueError) as exc:
                raise EnumerationError(
                    f"enumerate.neutral_fields {sorted(moved)} cannot take the "
                    f"default spec's values in cell {_show(combo)}: {exc}"
                ) from exc

        union: "OrderedDict[object, object]" = OrderedDict()
        cell_picks = []
        for combo, (base, specs) in members:
            base = normalised(base, combo)
            cell_picks.append((combo, base))
            for spec in (base, *(normalised(s, combo) for s in specs)):
                drift = {
                    a: getattr(spec, a)
                    for a, v in keyed_values.items()
                    if getattr(spec, a) != v
                }
                if drift:
                    raise EnumerationError(
                        f"enumerate.cell_keyed_by: request {_show(combo)} puts "
                        f"{spec} in the cell keyed {keyed_values}, but its {drift} "
                        f"differ (a neutral field or a sweep spec changes a keyed "
                        f"attribute)."
                    )
                union.setdefault(_identity(spec), spec)
        cells.append(dict(key))
        spaces.append((default, list(union.values())))
        picks.append(cell_picks)
    return CellSpaces(cells, spaces, picks, len(combos))


def _identity(spec):
    try:
        hash(spec)
    except TypeError:
        return repr(spec)
    return spec


def _check_picks(picks: list, kept_per_cell: list) -> None:
    """Every merged combo's own ``select_spec`` ships: a selector field that changes
    anything but the kept knobs would otherwise drop it silently."""
    for cell_picks, kept in zip(picks, kept_per_cell):
        for combo, pick in cell_picks:
            if pick not in kept:
                raise EnumerationError(
                    f"the registry selects {pick} for request {_show(combo)}, but its "
                    f"merged cell keeps no such kernel: enumerate.selector_fields "
                    f"changes a field the cell holds or filters (declare it "
                    f"neutral only if it is proven IR-neutral)."
                )


def _section_rows(registry: Registry, classification: Classification) -> tuple:
    """``(rows, projected, held)`` for one `Registry` section."""
    full = registry.space_kind == "full"
    merged = cell_spaces(registry, classification)
    cells, spaces = merged.cells, merged.spaces
    if full:
        selected = select_full(spaces, classification)
        kept_per_cell, projected, held = (
            selected.kept,
            selected.projected,
            selected.held,
        )
    else:
        kept_per_cell = [registry.slice_of(base, specs) for base, specs in spaces]
        projected, held = [], []
    _check_picks(merged.picks, kept_per_cell)
    rows = []
    for cell, (base, _), kept in zip(cells, spaces, kept_per_cell):
        if not kept:
            raise EnumerationError(
                f"the registry yields no spec for cell {_show(cell)}, which the "
                f"manifest's cells promise."
            )
        rows.extend((cell, ks, ks == base) for ks in kept)
    limit = registry.decl.get("max_kernels")
    if not full and limit is not None and len(rows) > limit:
        raise EnumerationError(
            f"the registry yields {len(rows)} kernels, over the manifest's "
            f"max_kernels {limit}. Raise it deliberately: every kernel is a comgr "
            f"build in the package."
        )
    return rows, projected, held


def enumerate_space(manifest: dict, registry: Registry | None = None) -> Enumeration:
    registry = registry or Registry(manifest)
    classification = Classification.from_manifest(
        manifest,
        required=any(s.space_kind == "full" for s in registry.sections()),
    )
    rows, sections, varied, held_notes = [], [], set(), OrderedDict()
    floors: set = set()
    for section in registry.sections():
        extra, section_projected, section_held = _section_rows(section, classification)
        # One floor per cell: a later section listing a cell an earlier one already
        # has keeps its kernels, but not a second `default: true`. A merged cell is
        # keyed by all its items, the keyed attributes `cell_spaces` adds included;
        # a section's own `spec_overrides` group keeps its floor apart from another
        # section's, since their kernels differ.
        group = section.overrides_key()
        extra = [
            (cell, ks, is_default and (tuple(cell.items()), group) not in floors)
            for cell, ks, is_default in extra
        ]
        floors.update((tuple(cell.items()), group) for cell, _, d in extra if d)
        rows.extend(extra)
        sections.extend([section] * len(extra))
        varied.update(section_projected)
        held_notes.update((h, None) for h in section_held)
    performance = classification.performance
    projected = [n for n in performance if n in varied]
    held = list(held_notes)
    accessors = {n: performance[n] for n in projected if performance[n]}
    return Enumeration(rows, projected, accessors, held, classification, sections)


def _kmd_type(knob: str, values: list) -> str:
    types = set()
    for value in values:
        kmd = _KMD_TYPES.get(type(value))
        if kmd is None:
            raise EnumerationError(
                f"performance knob '{knob}' resolves to {value!r} "
                f"({type(value).__name__}), which no KMD field type represents."
            )
        types.add(kmd)
    if len(types) != 1:
        raise EnumerationError(
            f"performance knob '{knob}' projects to more than one KMD type across "
            f"cells: {sorted(types)}. A KMD field has one type."
        )
    return types.pop()


def _template_slots(template: str) -> set:
    import string

    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in string.Formatter().parse(template)
        if field
    }


def build_config(
    manifest: dict, registry: Registry | None = None, warn=None, strict: bool = False
) -> dict:
    """The generator config for ``manifest``. Holds and a budget overrun are
    reported through ``warn`` (stderr by default); with ``strict`` an overrun raises
    `OverBudget` instead."""
    warn = warn or (lambda message: print(message, file=sys.stderr))
    template = manifest.get("kernel_name_template")
    if not template:
        raise EnumerationError("manifest declares no kernel_name_template")
    if not manifest.get("authored_subpath"):
        raise EnumerationError("manifest declares no authored_subpath")
    registry = registry or Registry(manifest)
    enumeration = enumerate_space(manifest, registry)
    for name, value, why in enumeration.held:
        warn(f"WARN: field {name} held at default {value} ({why})")

    projected = enumeration.projected
    accessors = enumeration.accessors
    kmd_fields = [dict(f) for f in manifest["kmd_fields"]]
    specialization = json.loads(json.dumps(manifest["specialization"]))
    if projected:
        by_knob: dict = {n: [] for n in projected}
        for _, ks, _ in enumeration.rows:
            for n in projected:
                by_knob[n].append(_resolved(ks, n, accessors.get(n)))
        declared = {f["name"] for f in kmd_fields}
        metadata_fields = specialization.setdefault("metadata_fields", [])
        bindings = specialization.setdefault("bindings", {})
        for n in projected:
            kmd_type = _kmd_type(n, by_knob[n])
            if n not in declared:
                kmd_fields.append({"name": n, "type": kmd_type})
            if n not in metadata_fields:
                metadata_fields.append(n)
            if n not in bindings:
                bindings[n] = (
                    {"method": accessors[n]} if n in accessors else {"field": n}
                )

    def name_layout(template: str) -> tuple:
        """``(named, split)``: the projected knobs ``template`` has no slot for,
        and where their ``_<knob><value>`` suffix goes (after the last slot)."""
        slots = _template_slots(template)
        named = [n for n in projected if n not in slots]
        split = None
        if named:
            last = None
            for last in _LAST_SLOT.finditer(template):
                pass
            split = last.end() if last else len(template)
        return named, split

    kernels = []
    for (cell, ks, is_default), section in zip(enumeration.rows, enumeration.sections):
        section_template = section.decl.get("kernel_name_template") or template
        named, split = name_layout(section_template)
        names = list(section.decl.get("spec_fields") or _spec_fields(ks))
        names += [n for n in projected if n not in names]
        # An override is what sets the section's kernels apart: always ship it.
        names += [n for n in section.spec_overrides if n not in names]
        missing = [n for n in names if not hasattr(ks, n)]
        if missing:
            raise EnumerationError(
                f"enumerate.spec_fields names {missing}, which the registry spec "
                f"{type(ks).__name__} does not have."
            )
        spec = {
            n: _plain(getattr(ks, n))
            for n in names
            if not (n in accessors and getattr(ks, n) is None)
        }
        request = {n: _rendered(v) for n, v in {**section.fixed, **cell}.items()}
        metadata = _metadata(spec, ks, request, accessors, specialization)
        # Spec wins, then request, then metadata; bools render 1/0 (`c{causal}`).
        fields = {
            **metadata,
            **request,
            **{k: _rendered(v) for k, v in spec.items()},
        }
        try:
            if split is None:
                name = section_template.format(**fields)
            else:
                suffix = "".join(
                    (
                        f"_{n}{_rendered(spec[n])}"
                        if n in spec
                        else f"_{n}u{_rendered(_resolved(ks, n, accessors[n]))}"
                    )
                    for n in named
                )
                name = (
                    section_template[:split].format(**fields)
                    + suffix
                    + section_template[split:].format(**fields)
                )
        except (KeyError, IndexError, ValueError) as exc:
            raise EnumerationError(
                f"kernel_name_template {section_template!r} cannot be rendered: {exc!r}. "
                f"Available: {sorted(fields)}."
            ) from exc
        kernel = {
            "name": name,
            "kernel_source": {
                "kind": manifest.get("kernel_source_kind", "rocke"),
                "source": manifest["source"],
                "builder": manifest["builder"],
                "spec": spec,
            },
            "metadata": metadata,
        }
        if manifest.get("launches"):
            kernel["kernel_source"]["launches"] = json.loads(
                json.dumps(manifest["launches"])
            )
        if is_default:
            kernel["default"] = True
        provenance = section.provenance(cell, ks)
        if provenance:
            kernel["provenance"] = provenance
        kernels.append(kernel)

    unlaunched = [
        s.pin.get("spec_id", "enumerate")
        for s in registry.sections()
        if s.pairs_reduce and not manifest.get("launches")
    ]
    if unlaunched:
        raise EnumerationError(
            f"the registry pairs {unlaunched[:3]} with a reduce kernel, but the manifest "
            f"declares no 'launches': the catalog would ship the kernel without the "
            f"reduce the dispatcher launches after it. Build both from one module "
            f"builder and declare its launches."
        )
    merges = any(s.selector_fields or s.neutral_fields for s in registry.sections())
    if (
        any(s.space_kind == "full" for s in registry.sections())
        or registry.also
        or merges
    ):
        repeated = [n for n, c in Counter(k["name"] for k in kernels).items() if c > 1]
        if repeated and registry.also:
            raise EnumerationError(
                f"{len(repeated)} kernel name(s) repeat across enumerate sections, "
                f"e.g. {repeated[:3]}: a section's kernels must differ from every "
                f"other section's in a field the name template renders."
            )
        if repeated:
            raise EnumerationError(
                f"space: {registry.space_kind} renders {len(repeated)} duplicated "
                f"kernel name(s), e.g. "
                f"{repeated[:3]}: a varying field reaches neither the name template "
                f"nor a projected knob."
            )
    budget = enumeration.classification.budget
    if len(kernels) > budget:
        message = (
            f"{len(kernels)} instances for {manifest['arch']} exceed the AOT "
            f"budget {budget} (red flag)"
        )
        if strict:
            raise OverBudget(message)
        warn(f"WARN: {message}")

    slug = manifest.get("pack") or Path(manifest["authored_subpath"]).name
    config = {
        "dialect": manifest.get("dialect", "packaged"),
        "authored_subpath": manifest["authored_subpath"],
        "engine": manifest["engine"],
        "kmd_fields": kmd_fields,
        "specialization": specialization,
        "kernel_source_kind": manifest.get("kernel_source_kind", "rocke"),
        "workspace_policy": manifest.get("workspace_policy", "none"),
    }
    if "reduction" in manifest:
        config["reduction"] = manifest["reduction"]
    config["packs"] = [{"name": slug, "arch": [manifest["arch"]], "kernels": kernels}]
    return config


def _held_at_dataclass_default(ks, name: str) -> bool:
    """Whether registry spec ``ks`` has field ``name`` at its dataclass default, which
    a kernel spec that omits the field also builds with."""
    if not dataclasses.is_dataclass(ks):
        return False
    field = next((f for f in dataclasses.fields(ks) if f.name == name), None)
    if field is None or field.default is dataclasses.MISSING:
        return False
    return getattr(ks, name) == field.default


def _metadata(spec: dict, ks, request: dict, accessors: dict, specialization) -> dict:
    vocabulary = specialization.get("vocabulary") or {}
    metadata = {}
    for name in specialization["metadata_fields"]:
        if name in spec:
            value = spec[name]
        elif name in accessors:
            value = _resolved(ks, name, accessors[name])
        elif name in request:
            value = request[name]
        elif _held_at_dataclass_default(ks, name):
            # The spec omits it, so the builder builds this default: metadata agrees.
            value = getattr(ks, name)
        else:
            raise EnumerationError(
                f"metadata field '{name}' is neither a field of the projected spec "
                f"({sorted(spec)}) nor a request field of the cell; add it to "
                f"enumerate.spec_fields."
            )
        value = _rendered(value)
        mapping = vocabulary.get(name)
        if isinstance(mapping, dict) and isinstance(value, str):
            if value not in mapping:
                raise EnumerationError(
                    f"metadata field '{name}' has value {value!r}, which "
                    f"specialization.vocabulary.{name} does not map. Mapped: "
                    f"{sorted(map(str, mapping))}."
                )
            value = mapping[value]
        metadata[name] = value
    return metadata


def plan_space(manifest: dict, out=None, strict: bool = False) -> int:
    """Print the full space's varying fields and exact instance counts, over every
    ``enumerate`` section. With ``strict``, return 1 when the current classification
    is over budget."""
    out = out or sys.stdout
    registry = Registry(manifest)
    classification = Classification.from_manifest(manifest, required=True)
    sections = []
    for section in registry.sections():
        merged = cell_spaces(section, classification)
        sections.append((section, merged.cells, merged.spaces, merged.combos))
    spaces = [space for _, _, section_spaces, _ in sections for space in section_spaces]
    budget = classification.budget

    def count(performance) -> int:
        total = 0
        for section, _, section_spaces, _ in sections:
            if section.space_kind == "full":
                selected = select_full(
                    section_spaces, classification.with_performance(performance)
                )
                total += sum(len(kept) for kept in selected.kept)
            else:
                total += sum(
                    len(section.slice_of(base, specs)) for base, specs in section_spaces
                )
        return total

    def against(n: int) -> str:
        verdict = "OVER" if n > budget else "within"
        return f"{n} ({verdict} budget {budget})"

    performance = classification.performance
    ignored = set(classification.ignored)
    values: "OrderedDict[str, dict]" = OrderedDict()
    defaults: "OrderedDict[str, dict]" = OrderedDict()
    sames = _feature_matched(spaces, classification)
    for base, same in sames:
        for name in _spec_fields(base):
            seen = {repr(getattr(s, name)): getattr(s, name) for s in same}
            if len(seen) > 1:
                values.setdefault(name, {}).update(seen)
    unresolvable = _unresolvable(sames, performance)
    for name in values:
        accessor = performance.get(name)
        for base, _ in spaces:
            value = _resolved(base, name, accessor)
            defaults.setdefault(name, {})[repr(value)] = value

    def listed(shown: dict) -> str:
        return ", ".join(sorted(shown, key=lambda r: _order_key(shown[r])))

    def label(name: str) -> str:
        if name in unresolvable:
            return "performance, held (no unset accessor)"
        if name in performance:
            return "covered (performance)"
        if name in ignored:
            return "ignored (held at default)"
        return "unclassified-held"

    arch = manifest["arch"]
    print(f"plan space: {manifest['engine']['name']} @ {arch}", file=out)
    for section, cells, _, combos in sections:
        if len(sections) > 1:
            print(
                f"  section {section.pin.get('spec_id', 'enumerate')} "
                f"(space: {section.space_kind}):",
                file=out,
            )
        for fields, rows in section.groups:
            print(f"  cell group {fields}: {len(rows)} values", file=out)
        if combos > len(cells):
            print(
                f"  cells: {len(cells)} ({combos} request combos merged on "
                f"selector_fields {section.selector_fields}, neutral_fields "
                f"{section.neutral_fields}, cell_keyed_by {section.cell_keyed_by})",
                file=out,
            )
        else:
            print(f"  cells: {len(cells)}", file=out)
    print("  varying spec fields (value set; registry default):", file=out)
    if not values:
        print("    none", file=out)
    for name in values:
        shown, default = listed(values[name]), listed(defaults[name])
        print(f"    {name}: {{{shown}}}; default {default}; {label(name)}", file=out)
    current = count(performance)
    print(f"  instances (current classification): {against(current)}", file=out)
    unclassified = [n for n in values if n not in performance and n not in ignored]
    held_ignored = [n for n in values if n in ignored]
    if held_ignored:
        print(f"  ignored fields: {held_ignored}", file=out)
    for name in unclassified:
        added = OrderedDict(performance)
        added[name] = None
        print(f"  if added {name}: {against(count(added))}", file=out)
    for name in [n for n in performance if n in values]:
        dropped = OrderedDict((k, v) for k, v in performance.items() if k != name)
        print(f"  if dropped {name}: {against(count(dropped))}", file=out)
    return 1 if strict and current > budget else 0


def load_shapes(path: str) -> list:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnumerationError(
            f"shapes file {path} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(loaded, list) or any(not isinstance(s, dict) for s in loaded):
        raise EnumerationError(
            f"shapes file {path} must be a JSON list of request-field mappings."
        )
    return loaded


def _field_default(field: dataclasses.Field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def shapes_report(registry: Registry, shapes: list) -> tuple:
    """``(report lines, GAP shape count, covered shape count)``: each shape in its
    first matching category of outside cell axes, declined by registry, covered,
    GAP. "Covered" is coverage by the manifest's cell axes: a shape's sequence
    lengths are not checked against what the catalog's kernels can serve.

    A shape's cell is keyed without ``enumerate.selector_fields``: they only steer
    the registry's rule, a merged cell carries none of them (`cell_spaces`), and a
    graph need not set them or take a listed value. It is keyed with every
    ``enumerate.cell_keyed_by`` attribute of the registry's pick, as the cells are.
    Every section declares the same ``selector_fields`` and ``cell_keyed_by``
    (`Registry`), so one key serves them all."""
    sections = registry.sections()
    fields = [f for f in registry.cell_fields if f not in registry.selector_fields]
    keyed = registry.cell_keyed_by

    def keyed_values(spec) -> tuple:
        return tuple(getattr(_unwrap(spec), a) for a in keyed)

    listed = {
        tuple(cell.get(f) for f in fields)
        + (keyed_values(s.space(cell)[0]) if keyed else ())
        for s in sections
        for cell in s.cells()
    }
    controlled = set(registry.fixed) | set(registry.cell_fields)
    categories = OrderedDict(
        (name, {"shapes": 0, "cells": set(), "detail": Counter()})
        for name in ("outside cell axes", "declined by registry", "covered", "GAP")
    )
    distinct = set()
    for raw in shapes:
        shape = {k: v for k, v in raw.items() if not k.startswith("_")}
        key = tuple(registry.coerce(f, shape.get(f)) for f in fields)
        distinct.add(key)
        outside = []
        for name, value in shape.items():
            field = registry.request_fields.get(name)
            if field is None:
                outside.append(f"{name} (not a request field)")
            elif name not in controlled:
                default = _field_default(field)
                if registry.coerce(name, value) != registry.coerce(name, default):
                    outside.append(f"{name}={_shown(registry.coerce(name, value))}")
        if outside:
            category, detail = "outside cell axes", ", ".join(outside)
        else:
            kwargs = {
                name: registry.coerce(name, value) for name, value in shape.items()
            }
            # The shape's own values win; `fixed` fills only the fields it omits.
            # A shape the request class or the registry cannot take (a null or
            # mistyped value raises TypeError or ValueError) is that shape's decline,
            # as it always was; only a missing attribute is the registry itself
            # being unusable.
            ok, why, declined, admitting, request = False, "", None, None, None
            for section in sections:
                try:
                    request = section.request_cls(
                        **{**section.fixed, **kwargs, **section.pin}
                    )
                    ok, why = section.candidate.admits(request)
                except (TypeError, ValueError) as exc:
                    ok, why = False, f"request rejected: {exc}"
                except AttributeError as exc:
                    raise EnumerationError(
                        f"the registry cannot judge shape {shape!r} "
                        f"(AttributeError): {exc}"
                    ) from exc
                if ok:
                    admitting = section
                    break
                declined = declined or why
            why = why if ok else declined
            cell_text = _show(dict(zip(fields, key)))
            if ok and keyed:
                picked = keyed_values(admitting.candidate.select_spec(request))
                key += picked
                cell_text = _show(
                    {**dict(zip(fields, key)), **dict(zip(keyed, picked))}
                )
            if not ok:
                category, detail = "declined by registry", f"{cell_text}: {why}"
            elif key in listed:
                category, detail = "covered", None
            else:
                category, detail = "GAP", cell_text
        entry = categories[category]
        entry["shapes"] += 1
        entry["cells"].add(key)
        if detail:
            entry["detail"][detail] += 1
    lines = [f"shape coverage: {len(shapes)} shapes, {len(distinct)} distinct cells"]
    lines.append(
        "  (by cell axes; sequence lengths are not checked against the kernels)"
    )
    for name, entry in categories.items():
        lines.append(f"  {name}: {entry['shapes']} shapes, {len(entry['cells'])} cells")
        for detail, n in sorted(entry["detail"].items()):
            lines.append(f"    {n} x {detail}")
    return lines, categories["GAP"]["shapes"], categories["covered"]["shapes"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate an engine's kernels from the rocKE dispatcher registry."
    )
    parser.add_argument("--manifest", required=True, help="Engine manifest (YAML).")
    parser.add_argument("--out", help="Write the generator config here.")
    parser.add_argument(
        "--shapes",
        help="Report how this JSON list of request-field mappings falls on the "
        "manifest's cells (coverage by cell axes; sequence lengths are not checked).",
    )
    parser.add_argument(
        "--plan-space",
        action="store_true",
        help="Print the full space's varying fields and exact instance counts per "
        "classification change; builds and writes nothing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when the run finds what a gate must refuse: an instance count "
        "over reduction.aot_budget_per_arch (the build, or --plan-space's current "
        "classification) or a --shapes GAP. Exit 2 stays for unusable input.",
    )
    args = parser.parse_args(argv)
    if not args.plan_space and not args.out:
        parser.error("--out is required unless --plan-space is given")
    if args.plan_space and (args.out or args.shapes):
        parser.error("--plan-space builds and writes nothing; drop --out and --shapes")
    try:
        manifest = load_manifest(args.manifest)
        bind_provider(manifest.get("provider_root"))
        if args.plan_space:
            return plan_space(manifest, strict=args.strict)
        shapes = load_shapes(args.shapes) if args.shapes else None
        registry = Registry(manifest)
        config = build_config(manifest, registry, strict=args.strict)
        report, gaps, covered = (
            shapes_report(registry, shapes) if shapes is not None else ([], 0, 0)
        )
        out = Path(args.out)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
            )
        except OSError as exc:
            raise EnumerationError(f"cannot write {out}: {exc}") from exc
    except OverBudget as exc:
        print(f"FAIL: --strict: {exc}", file=sys.stderr)
        return 1
    except EnumerationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"{len(config['packs'][0]['kernels'])} kernels -> {args.out}")
    for line in report:
        print(line)
    if args.strict and shapes is not None and (gaps or not covered):
        why = f"{gaps} GAP shapes" if gaps else "no shape is covered (a wrong pin?)"
        print(f"FAIL: --strict: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
