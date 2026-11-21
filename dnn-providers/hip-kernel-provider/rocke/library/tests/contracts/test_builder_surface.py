# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The builder surface consumers depend on matches its pinned snapshot.

The first two tests guard the live code. The rest prove the comparison reports
each kind of break, and stays quiet on additions, by applying the change to the
snapshot, a real capture of the surface, so they hold whatever state the code is in.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib

import pytest

from .builder_surface import (
    OPTIONAL,
    REQUIRED,
    SPEC_ARCH_CALL_SHAPE,
    UPDATE_COMMAND,
    BuilderRef,
    capture_surface,
    find_breaking_changes,
    load_snapshot,
    pinned_builders,
    required_spec_fields,
    write_snapshot,
)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return load_snapshot()


@pytest.fixture(scope="module")
def live_surface(snapshot: dict) -> dict:
    return capture_surface(pinned_builders(snapshot))


def test_pinned_builders_keep_their_published_surface(
    snapshot: dict, live_surface: dict
) -> None:
    changes = find_breaking_changes(snapshot, live_surface)
    assert not changes, (
        "a builder that other projects build kernels from changed in a way that "
        "breaks them:\n  "
        + "\n  ".join(changes)
        + "\nIf the change is wrong, revert it. If it is deliberate, tell the "
        "projects that name this builder, then refresh the snapshot from rocke/ "
        f"with `{UPDATE_COMMAND}` and commit it with the change."
    )


def test_every_pinned_builder_takes_spec_and_keyword_arch(live_surface: dict) -> None:
    wrong = [
        f"{entry['module']}.{entry['builder']}: "
        f"{entry.get('missing') or entry['parameters']} spec={entry.get('spec')}"
        for entry in live_surface["builders"]
        if "missing" in entry
        or [parameter[:2] for parameter in entry["parameters"]] != SPEC_ARCH_CALL_SHAPE
        or entry["spec"] not in live_surface["specs"]
    ]
    assert not wrong, (
        f"a pinned builder must take exactly {SPEC_ARCH_CALL_SHAPE} with a spec "
        "dataclass; a consumer cannot supply any other parameter:\n  "
        + "\n  ".join(wrong)
    )


def _first_builder(surface: dict) -> dict:
    return surface["builders"][0]


def _first_spec_fields(surface: dict) -> dict:
    return surface["specs"][_first_builder(surface)["spec"]]


def test_a_renamed_builder_is_reported(snapshot: dict) -> None:
    """The snapshot pins the name consumers use; the code no longer has it."""
    entry = copy.deepcopy(_first_builder(snapshot))
    entry["builder"] += "_before_rename"
    recorded = {"builders": [entry], "specs": {}}
    live = capture_surface(pinned_builders(recorded))
    expected = (
        f"{entry['module']}.{entry['builder']}: builder is gone "
        f"({entry['module']!r} has no builder {entry['builder']!r})"
    )
    assert find_breaking_changes(recorded, live) == [expected]


@pytest.mark.parametrize(
    "edit, expected",
    [
        (lambda fields, name: fields.pop(name), "was dropped"),
        (
            lambda fields, name: fields[name].__setitem__(0, "float"),
            "type changed from",
        ),
    ],
    ids=["dropped", "retyped"],
)
def test_a_dropped_or_retyped_field_is_reported(
    snapshot: dict, edit, expected: str
) -> None:
    live = copy.deepcopy(snapshot)
    field = next(iter(_first_spec_fields(live)))
    edit(_first_spec_fields(live), field)
    [change] = find_breaking_changes(snapshot, live)
    assert f"field {field!r} {expected}" in change


def test_a_field_that_loses_its_default_is_reported(snapshot: dict) -> None:
    live = copy.deepcopy(snapshot)
    fields = _first_spec_fields(live)
    field = next(name for name, info in fields.items() if info[1] == OPTIONAL)
    fields[field][1] = REQUIRED
    assert find_breaking_changes(snapshot, live) == [
        f"{_first_builder(live)['spec']}: field {field!r} lost its default"
    ]


def test_a_new_required_field_is_reported_and_a_defaulted_one_is_not(
    snapshot: dict,
) -> None:
    live = copy.deepcopy(snapshot)
    fields = _first_spec_fields(live)
    fields["new_defaulted_knob"] = ["bool", OPTIONAL]
    assert find_breaking_changes(snapshot, live) == []
    fields["new_problem_axis"] = ["int", REQUIRED]
    assert find_breaking_changes(snapshot, live) == [
        f"{_first_builder(live)['spec']}: new field 'new_problem_axis' has no default"
    ]


def test_predicate_changes_break_only_existing_calls(snapshot: dict) -> None:
    live = copy.deepcopy(snapshot)
    entry = next(e for e in live["builders"] if e["predicates"])
    name = f"{entry['module']}.{entry['builder']}"
    predicate, parameters = next(iter(entry["predicates"].items()))
    parameters.append(["new_option", "KEYWORD_ONLY", OPTIONAL])
    assert find_breaking_changes(snapshot, live) == []
    first_name, first_kind, _ = parameters[0]
    parameters[0][1] = "VAR_KEYWORD"
    assert (
        f"{name}: predicate {predicate} parameter {first_name!r} kind changed from "
        f"{first_kind} to VAR_KEYWORD"
    ) in find_breaking_changes(snapshot, live)
    del entry["predicates"][predicate]
    assert f"{name}: predicate {predicate} is gone" in find_breaking_changes(
        snapshot, live
    )


def test_a_builder_call_shape_change_is_reported(snapshot: dict) -> None:
    live = copy.deepcopy(snapshot)
    _first_builder(live)["parameters"].append(["tuning", "KEYWORD_ONLY", REQUIRED])
    [change] = find_breaking_changes(snapshot, live)
    assert change.endswith("new parameter 'tuning' has no default")


def test_write_snapshot_refuses_an_unresolved_builder(tmp_path) -> None:
    unresolved = capture_surface([BuilderRef("kernels", "build_no_such_kernel")])
    target = tmp_path / "builder_surface.json"
    with pytest.raises(ValueError, match=r"write_snapshot failed - .*no builder"):
        write_snapshot(unresolved, target)
    assert not target.exists()


def test_required_spec_fields_are_the_fields_without_a_default(
    live_surface: dict,
) -> None:
    """Each captured spec's required fields are its dataclass fields that carry
    neither a default nor a default factory."""
    for spec, required in required_spec_fields(live_surface).items():
        module_name, _, class_name = spec.rpartition(".")
        spec_class = getattr(importlib.import_module(module_name), class_name)
        no_default = tuple(
            sorted(
                field.name
                for field in dataclasses.fields(spec_class)
                if field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            )
        )
        assert required == no_default, spec
