#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gauntlet fuzz tool (RFC 0023): spaces, sampler, driver, fault artifacts.

No GPU and no C++ binary: the driver runs against ``stub_binary.py``, a script that honors
the same command line and writes the same outcome-report JSON Lines. Run with::

    python -m pytest -q -p no:cacheprovider dnn-providers/integration-tests/gauntlet/test_gauntlet.py
"""

from __future__ import annotations

import ast
import copy
import fnmatch
import hashlib
import importlib.util
import itertools
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import pairwise
import run_gauntlet_fuzz as fuzz
import sampler
import space as space_lib

HERE = Path(__file__).resolve().parent
TESTS_ROOT = HERE.parent
MIGRATION_DIR = TESTS_ROOT / "migration-scripts"
STUB = HERE / "stub_binary.py"
BUNDLES = TESTS_ROOT / "integration-test-bundles"
SPACES = space_lib.load_spaces()
SPACE_IDS = [s.space_id for s in SPACES]
bu = fuzz.load_bundle_utils()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle_discovery = _load_by_path(
    "gauntlet_test_bundle_discovery", TESTS_ROOT / "scripts" / "bundle_discovery.py"
)


def space_by_id(space_id):
    return next(s for s in SPACES if s.space_id == space_id)


TAG_OF_DTYPE = {"float": "fp32", "half": "fp16", "bfloat16": "bfp16"}


def input_dtype_tag(space_id, seed):
    """The tag of the sample's input dtype, as the quick bundle case ids spell it; None for a
    space with no dtype axis (Matmul/Align), whose suite has no dtype directory."""
    axes = sampler.sample(space_by_id(space_id), seed).axis_values
    for name in ("x_dtype", "dtype"):
        if name in axes:
            return TAG_OF_DTYPE[axes[name]]
    return None


def gtest(space_id, seed):
    """The GTest name the harness registers for a sample: the suite is the relative path of its
    sweep directory, sanitized and joined by ``_`` (``BundleDiscovery.hpp``)."""
    sp = space_by_id(space_id)
    tag = input_dtype_tag(space_id, seed)
    suite = f"gauntlet_{sp.operation}_{sp.name}" + (
        "" if tag is None else f"_{tag}_cases"
    )
    return f"{suite}.s{seed}"


def sweep_dir(work, space_id, seed):
    """The directory under the bundle root that the driver puts the sample's sweep in."""
    sp = space_by_id(space_id)
    tag = input_dtype_tag(space_id, seed)
    base = work / "bundles" / "gauntlet" / sp.operation / sp.name
    return base if tag is None else base / f"{tag}_cases"


# --------------------------------------------------------------------------
# Toy spaces
# --------------------------------------------------------------------------


def write_space(tmp_path, file_name="Toy.space.json", **overrides):
    """A small valid space (and its template) in ``tmp_path``; ``overrides`` replace keys."""
    (tmp_path / "toy.template.json").write_text('{"nodes": [], "tensors": []}')
    decl = {
        "version": 1,
        "operation": file_name.split(".")[0],
        "template": "toy.template.json",
        "axes": {
            "a": {"values": [1, 2, 3]},
            "n": {"range": [1, 100], "boundaries": [1, 100]},
        },
        "derive": {"twice": "a * 2"},
        "constraints": [{"rule": "n >= a", "reason": "n must reach a"}],
        "case": {
            "values": {"a": "${a}", "n": "${n}"},
            "metadata": {"purpose": "probe"},
        },
    }
    decl.update(overrides)
    path = tmp_path / file_name
    path.write_text(json.dumps(decl))
    return path


def derived(tmp_path, expression, a=2, n=7):
    sp = space_lib.load_space(write_space(tmp_path, derive={"r": expression}))
    return sp.derive_env({"a": a, "n": n})["r"]


# --------------------------------------------------------------------------
# R5: expression evaluator
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("a + 1", 3),
        ("a * n // 2", 7),
        ("n % 4", 3),
        ("2 ** a", 4),
        ("n / 2", 3.5),
        ("-a", -2),
        ("a == 2 and n > 5", True),
        ("a == 3 or n == 7", True),
        ("not a == 2", False),
        ("[a, n][1]", 7),
        ("[1, [2, 3]][1][0]", 2),
        ("min(a, n)", 2),
        ("max([a, n])", 7),
        ("len([1, 2, 3])", 3),
        ("abs(-n)", 7),
        ("'x' if a == 2 else 'y'", "x"),
        ("None", None),
        ("True", True),
        ("1 < a < 3", True),
        ("a in [1, 2]", True),
        ("1.5e-3 * 2", 0.003),
    ],
)
def test_evaluator_accepts_the_safe_subset(tmp_path, expression, expected):
    assert derived(tmp_path, expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "foo(1)",
        "sum([1, 2])",
        "open('x')",
        "__import__('os')",
        "a.real",
        "'x'.upper()",
        "[x for x in [1, 2]]",
        "{x for x in [1]}",
        "(x for x in [1])",
        "{x: x for x in [1]}",
        "(lambda: 1)()",
        "lambda: 1",
        "a[1:2]",
        "f'{a}'",
        "{1: 2}",
        "(1, 2)",
        "(b := 3)",
        "min(key=abs)",
        "min(*[1, 2])",
        "unknown_name + 1",
        "a +",
        "",
        "2 ** 100000",
        "a @ a",
        "a is None",
        "b'bytes'",
        "1j",
    ],
)
def test_evaluator_rejects_everything_else_as_a_load_error_naming_the_space(
    tmp_path, expression
):
    path = write_space(tmp_path, derive={"r": expression})
    if expression == "2 ** 100000":
        sp = space_lib.load_space(path)  # loadable; the guard fires on evaluation
        with pytest.raises(space_lib.SpaceError, match="Toy/Default"):
            sp.derive_env({"a": 1, "n": 1})
        return
    with pytest.raises(space_lib.SpaceError, match="Toy/Default"):
        space_lib.load_space(path)


@pytest.mark.parametrize("section", ["derive", "constraints", "case"])
def test_every_expression_site_is_checked(tmp_path, section):
    bad = "[x for x in [1]]"
    if section == "derive":
        path = write_space(tmp_path, derive={"r": bad})
    elif section == "constraints":
        path = write_space(tmp_path, constraints=[{"rule": bad, "reason": "r"}])
    else:
        path = write_space(tmp_path, case={"values": {"v": "${" + bad + "}"}})
    with pytest.raises(space_lib.SpaceError, match="Toy/Default.*comprehension"):
        space_lib.load_space(path)


def test_derived_values_see_only_earlier_ones(tmp_path):
    ok = write_space(tmp_path, derive={"p": "a + 1", "q": "p * 2"})
    sp = space_lib.load_space(ok)
    assert sp.derive_env({"a": 2, "n": 5})["q"] == 6
    forward = write_space(tmp_path, derive={"q": "p * 2", "p": "a + 1"})
    with pytest.raises(space_lib.SpaceError, match="unknown name 'p'"):
        space_lib.load_space(forward)


def test_division_by_zero_is_an_evaluation_error_naming_the_space(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path, derive={"r": "a // (n - n)"}))
    with pytest.raises(space_lib.SpaceError, match="Toy/Default.*ZeroDivisionError"):
        sp.derive_env({"a": 1, "n": 1})


def test_missing_template_is_a_load_error_naming_the_space(tmp_path):
    path = write_space(tmp_path, template="no/such/graph.template.json")
    with pytest.raises(space_lib.SpaceError, match=r"Toy/Default.*not found"):
        space_lib.load_space(path)


def test_template_is_resolved_relative_to_the_space_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "graph.template.json").write_text('{"nodes": []}')
    path = write_space(tmp_path, template="sub/graph.template.json")
    assert (
        space_lib.load_space(path).template_path
        == (tmp_path / "sub" / "graph.template.json").resolve()
    )


@pytest.mark.parametrize(
    "axes, message",
    [
        ({"a": {"values": []}}, "non-empty list"),
        ({"a": {"range": [5, 1]}}, "lo <= hi"),
        ({"a": {"range": [1.5, 2]}}, "integers"),
        ({"a": {"range": [1, 10], "boundaries": [11]}}, "inside the range"),
        ({"a": {"values": [1], "range": [1, 2]}}, "exactly one"),
        ({"a": {"values": [1], "boundaries": [1]}}, "needs a 'range'"),
        ({"a": {"values": [1], "extra": 1}}, "unknown keys"),
        ({"min": {"values": [1]}}, "reserved"),
        ({"1a": {"values": [1]}}, "not a valid name"),
        ({}, "non-empty object"),
    ],
)
def test_axis_declarations_are_validated(tmp_path, axes, message):
    with pytest.raises(space_lib.SpaceError, match=f"Toy/Default.*{message}"):
        space_lib.load_space(
            write_space(
                tmp_path, axes=axes, derive={}, constraints=[], case={"values": {}}
            )
        )


def test_space_file_shape_is_validated(tmp_path):
    with pytest.raises(space_lib.SpaceError, match="'version' must be 1"):
        space_lib.load_space(write_space(tmp_path, version=2))
    with pytest.raises(space_lib.SpaceError, match="'operation'"):
        space_lib.load_space(write_space(tmp_path, operation="Other"))
    with pytest.raises(space_lib.SpaceError, match="driver-owned"):
        space_lib.load_space(
            write_space(tmp_path, case={"values": {}, "metadata": {"seed": 1}})
        )
    with pytest.raises(space_lib.SpaceError, match="reason"):
        space_lib.load_space(write_space(tmp_path, constraints=[{"rule": "a > 0"}]))
    with pytest.raises(space_lib.SpaceError, match="unknown top-level"):
        space_lib.load_space(write_space(tmp_path, bogus=1))


def test_space_file_name_gives_operation_and_space_name(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path, file_name="Toy.Odd.space.json"))
    assert (sp.operation, sp.name, sp.space_id, sp.suite_name) == (
        "Toy", "Odd", "Toy/Odd", "gauntlet_Toy_Odd",
    )  # fmt: skip
    sp = space_lib.load_space(write_space(tmp_path))
    assert (sp.space_id, sp.suite_name) == ("Toy/Default", "gauntlet_Toy_Default")


# --------------------------------------------------------------------------
# R5: typed substitution
# --------------------------------------------------------------------------


def test_substitution_keeps_json_types_and_copies_other_strings(tmp_path):
    case = {
        "values": {
            "int": "${a}",
            "float": "${n / 4}",
            "bool": "${a > 1}",
            "none": "${None}",
            "list": "${[a, 2.5, 'x']}",
            "str": "${'p' + 'q' if a == 2 else 'z'}",
            "nested": {"deep": ["${a * 10}", "literal"]},
            "plain": "no substitution",
            "open": "${a",
            "prefix": "x ${a}",
            "close": "a}",
            "number": 7,
        },
        "metadata": {"purpose": "probe"},
    }
    sp = space_lib.load_space(write_space(tmp_path, case=case))
    rendered = sp.render(sp.derive_env({"a": 2, "n": 10}))["values"]
    assert rendered == {
        "int": 2,
        "float": 2.5,
        "bool": True,
        "none": None,
        "list": [2, 2.5, "x"],
        "str": "pq",
        "nested": {"deep": [20, "literal"]},
        "plain": "no substitution",
        "open": "${a",
        "prefix": "x ${a}",
        "close": "a}",
        "number": 7,
    }
    assert type(rendered["int"]) is int and type(rendered["bool"]) is bool
    assert json.loads(json.dumps(rendered)) == rendered


def test_rendered_lists_are_not_shared_between_samples(tmp_path):
    sp = space_lib.load_space(
        write_space(tmp_path, derive={"v": "[a]"}, case={"values": {"v": "${v}"}})
    )
    first = sp.render(sp.derive_env({"a": 1, "n": 1}))
    first["values"]["v"].append(99)
    assert sp.render(sp.derive_env({"a": 1, "n": 1}))["values"]["v"] == [1]


# --------------------------------------------------------------------------
# R5: sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_sample_is_determined_by_its_shape_seed(space_id):
    sp = space_by_id(space_id)
    first = [sampler.sample(sp, s) for s in (7, 8, 9)]
    again = [sampler.sample(sp, s) for s in (9, 8, 7)][
        ::-1
    ]  # call order does not matter
    for a, b in zip(first, again):
        assert json.dumps(a.row) == json.dumps(b.row)
        assert a.axis_values == b.axis_values
    if space_id in FIXED_SHAPE_SPACES:
        # the shape is fixed; what a seed draws is the alignment and the placement
        assert (
            len(
                {
                    json.dumps(s.row["tensor_patches"]) + s.row["metadata"]["placement"]
                    for s in first
                }
            )
            > 1
        )
    else:
        assert len({json.dumps(s.row["values"]) for s in first}) == 3


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_constraints_hold_over_1000_samples(space_id):
    sp = space_by_id(space_id)
    assert sp.constraints or space_id in FIXED_SHAPE_SPACES
    for seed in range(1000):
        s = sampler.sample(sp, seed)
        env = sp.derive_env(s.axis_values)
        for c in sp.constraints:
            assert c.expr.evaluate(env), f"seed {seed}: {c.rule} ({c.reason})"


def test_constraints_hold_for_a_toy_space_and_redraws_happen(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path))
    attempts = []
    for seed in range(1000):
        s = sampler.sample(sp, seed)
        assert s.axis_values["n"] >= s.axis_values["a"]
        attempts.append(s.attempts)
    assert max(attempts) >= 1


def test_boundary_values_are_drawn_about_half_the_time(tmp_path):
    sp = space_lib.load_space(
        write_space(
            tmp_path,
            axes={"n": {"range": [1, 100000], "boundaries": [1, 2]}},
            derive={},
            constraints=[],
            case={"values": {}},
        )
    )
    rng = random.Random(0)
    draws = [sp.axes["n"].draw(rng) for _ in range(4000)]
    share = sum(d in (1, 2) for d in draws) / len(draws)
    assert 0.45 < share < 0.56
    assert all(1 <= d <= 100000 for d in draws)
    assert sum(d not in (1, 2) for d in draws) > 1500  # and the rest is uniform


def test_infeasible_space_is_an_explicit_error_with_the_reasons(tmp_path):
    sp = space_lib.load_space(
        write_space(
            tmp_path, constraints=[{"rule": "n > 1000", "reason": "n is capped at 100"}]
        )
    )
    with pytest.raises(sampler.InfeasibleSpaceError) as info:
        sampler.sample(sp, 1, max_attempts=50)
    text = str(info.value)
    assert "Toy/Default" in text and "infeasible" in text
    assert "n > 1000" in text and "n is capped at 100" in text and "50" in text


def test_data_seed_is_derived_and_always_in_range():
    seen = set()
    for seed in range(5000):
        d = sampler.derive_data_seed(seed)
        assert 0 <= d < 2**32
        assert d == sampler.derive_data_seed(seed)
        seen.add(d)
    assert len(seen) > 4990
    assert 0 <= sampler.derive_data_seed(2**62) < 2**32


def test_data_seed_can_be_fixed_and_is_validated():
    sp = space_by_id("SdpaFwd/Default")
    assert sampler.sample(sp, 5, data_seed=4021).row["metadata"]["seed"] == 4021
    assert sampler.sample(sp, 5, data_seed=2**32 - 1).data_seed == 2**32 - 1
    assert sampler.sample(sp, 5, data_seed=0).data_seed == 0
    for bad in (-1, 2**32, 1.5, True, "7"):
        with pytest.raises(ValueError):
            sampler.sample(sp, 5, data_seed=bad)
    for bad in (-1, 2**63, 1.5):
        with pytest.raises(ValueError):
            sampler.sample(sp, bad)
    # a new data seed changes the data, never the geometry
    a, b = sampler.sample(sp, 5, data_seed=1), sampler.sample(sp, 5, data_seed=2)
    assert (
        a.row["values"] == b.row["values"]
        and a.row["metadata"]["seed"] != b.row["metadata"]["seed"]
    )


# the sweep-case keys a space declares to switch a feature per sample (add_tensors adds the
# tensor of an optional input or output, node_patches edits the node)
FEATURE_KEYS = {
    "SdpaBackward/Default": {"node_patches"},
    "SdpaFwd/Default": {"add_tensors", "node_patches"},
    "SdpaFwd/Gfx1151": {"add_tensors", "node_patches"},
    "SdpaFwd/Gfx1151Decode": {"add_tensors", "node_patches"},
}


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_row_has_sweep_json_format(space_id):
    s = sampler.sample(space_by_id(space_id), 913377)
    row = s.row
    assert list(row)[:3] == ["id", "values", "metadata"]
    assert set(row) - {"id", "values", "metadata"} <= {
        "add_tensors",
        "node_patches",
        "tensor_patches",
    }
    # only a space that switches a feature declares these keys
    assert {"add_tensors", "node_patches"} & set(row) == (
        FEATURE_KEYS.get(space_id, set())
    )
    assert ("tensor_patches" in row) == (space_id in PATCHED_OPERANDS)
    assert row["id"] == "s913377"
    meta = row["metadata"]
    assert meta["format_version"] == 1 and isinstance(meta["format_version"], int)
    assert meta["generator"] == "gauntlet-fuzz"
    assert meta["purpose"] == "probe"
    assert (
        meta["seed"] == sampler.derive_data_seed(913377) and 0 <= meta["seed"] < 2**32
    )
    assert space_id in meta["notes"] and "913377" in meta["notes"]
    assert json.loads(json.dumps(row)) == row


# --------------------------------------------------------------------------
# R7: shipped spaces
# --------------------------------------------------------------------------


def test_spaces_ship_with_the_tool_not_in_the_bundle_root():
    assert space_lib.DEFAULT_SPACES_DIR == HERE / "spaces"
    assert SPACE_IDS == [
        "Batchnorm/Default",
        "BatchnormBackward/Default",
        "BatchnormInference/Default",
        "Layernorm/Default",
        "LayernormBackward/Default",
        "Matmul/Align",
        "RMSNorm/Default",
        "RMSNormBackward/Default",
        "ResampleFwd/Default",
        "SdpaBackward/Default",
        "SdpaFwd/Default",
        "SdpaFwd/Flash2",
        "SdpaFwd/Flash2Region",
        "SdpaFwd/Gfx1151",
        "SdpaFwd/Gfx1151Decode",
        "SdpaFwd/Gfx950Dense",
    ]
    for sp in SPACES:
        assert BUNDLES not in sp.path.parents
        assert sp.template_path.name == "graph.template.json"
        assert BUNDLES in sp.template_path.parents
        assert (
            sp.path.parent / "graph.template.json"
        ).exists() is False  # never a copy
        decl = json.loads(sp.path.read_text())
        assert decl["template"].startswith("../../integration-test-bundles/quick/")
        assert decl["case"]["metadata"]["purpose"] == "probe"


def test_shipped_spaces_declare_cost_ceilings():
    rules = {
        s.space_id: [c.rule.replace(" ", "") for c in s.constraints] for s in SPACES
    }
    assert any("2**31" in r for r in rules["SdpaFwd/Default"])
    assert any("2**31" in r for r in rules["SdpaFwd/Gfx950Dense"])
    assert any("2**31" in r for r in rules["SdpaFwd/Flash2Region"])
    assert any("2**31" in r for r in rules["SdpaFwd/Gfx1151"])
    assert any("2**31" in r for r in rules["SdpaFwd/Gfx1151Decode"])
    assert any("2**29" in r for r in rules["SdpaBackward/Default"])
    assert any("2**30" in r for r in rules["RMSNorm/Default"])
    assert any("2**30" in r for r in rules["ResampleFwd/Default"])
    assert any("sq==skv" in r for r in rules["SdpaFwd/Gfx950Dense"])


def _packed(dims, strides):
    """True when ``strides`` address ``dims`` densely (some permutation of a contiguous layout)."""
    if len(dims) != len(strides) or any(d < 1 for d in dims):
        return False
    count = 1
    for d in dims:
        count *= d
    return sum((d - 1) * s for d, s in zip(dims, strides)) + 1 == count


def expanded_graph(template, row):
    """The graph the harness runs for a sweep row: the template expanded with the row's
    values, then its ``tensor_patches`` applied."""
    graph = bu.expand(template, row["values"])
    fuzz.load_support_matrix_units().apply_case_patches(graph, row)
    return graph


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_shipped_space_row_appended_to_the_template_sweep_expands(tmp_path, space_id):
    sp = space_by_id(space_id)
    topology = tmp_path / "quick" / sp.operation / sp.template_path.parent.name
    topology.mkdir(parents=True)
    for name in ("graph.template.json", "sweep.json", "support.json"):
        if (
            sp.template_path.parent / name
        ).is_file():  # SdpaBackward ships no support.json
            shutil.copyfile(sp.template_path.parent / name, topology / name)
    sweep = json.loads((topology / "sweep.json").read_text())
    existing = len(sweep["cases"])
    samples = [sampler.sample(sp, seed) for seed in range(200, 240)]
    sweep["cases"] += [s.row for s in samples]
    (topology / "sweep.json").write_text(json.dumps(sweep, indent=2))

    cases, error = bundle_discovery.read_sweep_cases(topology / "sweep.json")
    assert error is None and len(cases) == existing + len(samples)
    template = json.loads((topology / "graph.template.json").read_text())
    for s in samples:
        graph = expanded_graph(template, s.row)
        assert "${" not in json.dumps(graph), "an unresolved placeholder is left"
        assert bu.canonical_uid_map_by_name(graph) == {
            t["uid"]: t["uid"] for t in graph["tensors"]
        }
        for tensor in graph["tensors"]:
            if tensor.get("value") is None:
                assert _packed(tensor["dims"], tensor["strides"]), (s.case_id, tensor)
    expanded = {
        json.dumps(expanded_graph(template, s.row), sort_keys=True) for s in samples
    }
    # the samples are different graphs (a fixed-shape space differs by its alignments only)
    assert len(expanded) > (5 if space_id in FIXED_SHAPE_SPACES else 30)


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_shipped_space_samples_cover_its_axes(space_id):
    sp = space_by_id(space_id)
    drawn = [sampler.sample(sp, seed).axis_values for seed in range(400)]
    for name, axis in sp.axes.items():
        seen = {json.dumps(d[name]) for d in drawn}
        if not axis.is_range:
            assert seen == {json.dumps(v) for v in axis.values}
        else:
            assert {json.dumps(b) for b in axis.boundaries} <= seen
            assert len(seen) > len(axis.boundaries)


def test_gfx950_dense_space_obeys_the_rocke_rules():
    sp = space_by_id("SdpaFwd/Gfx950Dense")
    for seed in range(500):
        env = sp.derive_env(sampler.sample(sp, seed).axis_values)
        assert env["d"] in (64, 128) and env["dtype"] in ("bfloat16", "half")
        assert env["sq"] % 128 == 0 and env["skv"] % 32 == 0
        assert env["hq"] % env["hkv"] == 0
        assert not env["causal"] or env["sq"] == env["skv"]
        assert 2**-64 <= env["scale"] <= 16
        assert env["stride_q"] == [
            env["sq"] * env["hq"] * env["d"],
            env["d"],
            env["hq"] * env["d"],
            1,
        ]  # BSHD


def _row_has_a_key(left, right, top_left, sq, skv):
    """Whether every query row keeps a key under a band, by the CPU reference's mask
    (``CpuFpReferenceSdpa::isMasked``: left_bound L keeps the L keys ending at the diagonal,
    right_bound R the R keys after it): row i keeps keys j with i + offset - L < j <= i + offset + R.
    """
    offset = 0 if top_left else skv - sq
    for i in range(sq):
        lo = 0 if left is None else max(0, i + offset - left + 1)
        hi = skv - 1 if right is None else min(skv - 1, i + offset + right)
        if lo > hi:
            return False
    return True


def test_row_has_a_key_agrees_with_the_reference_mask_cell_by_cell():
    # bug caught: an oracle off by one from CpuFpReferenceSdpa::isMasked would bless a window
    # that leaves a row with no key (or reject a legal one) in every test that uses it
    for sq in range(1, 7):
        for skv in range(1, 7):
            for top_left in (True, False):
                for left in (None, 1, 2, 5):
                    for right in (None, 0, 1, 3):
                        offset = 0 if top_left else skv - sq

                        def masked(i, j):
                            return (
                                right is not None and j >= i + 1 + offset + right
                            ) or (left is not None and j <= i + offset - left)

                        brute = all(
                            any(not masked(i, j) for j in range(skv)) for i in range(sq)
                        )
                        assert _row_has_a_key(left, right, top_left, sq, skv) == brute


def test_sdpa_forward_space_renders_the_stats_output_and_the_window_masks():
    # bug caught: a stats axis that never adds the LSE tensor or leaves generate_stats or
    # stats_tensor_uid unset (the case would pass without the feature), an LSE tensor whose shape
    # or alignment differs from the sample's (the engine reads it at 16 bytes), a window value
    # that does not reach left_bound, right_bound or diagonal_alignment, a left bound below 1
    # (the builder throws) or one that leaves a query row with no key, and a feature
    # combination the sampler never draws
    sp = space_by_id("SdpaFwd/Default")
    template = json.loads(sp.template_path.read_text())
    seen, raised_left, passable_stats = set(), 0, 0
    for seed in range(600):
        s = sampler.sample(sp, seed)
        axes = s.axis_values
        env = sp.derive_env(axes)
        graph = expanded_graph(template, s.row)
        node = graph["nodes"][0]
        attrs, outs = node["attributes"], node["outputs"]
        by_uid = {t["uid"]: t for t in graph["tensors"]}
        b, hq, sq, skv = axes["b"], env["hq"], axes["sq"], axes["skv"]
        stats, window, causal = axes["stats"], axes["window"], axes["causal"]

        if stats == "on":
            assert sorted(by_uid) == [0, 1, 2, 3, 4], seed
            lse = by_uid[4]
            assert lse["dims"] == [b, hq, sq, 1] and lse["data_type"] == "float", seed
            assert lse["strides"] == [hq * sq, sq, 1, 1], seed
            assert lse["alignment"] == axes["alignment"], seed
            assert not lse["virtual"] and not lse["is_runtime_pass_by_value"], seed
            assert attrs["generate_stats"] is True and outs["stats_tensor_uid"] == 4
        else:
            assert sorted(by_uid) == [0, 1, 2, 3], seed
            assert attrs["generate_stats"] is None and outs["stats_tensor_uid"] is None
        assert {by_uid[u]["alignment"] for u in range(4)} == {axes["alignment"]}
        assert (outs["o_tensor_uid"], attrs["attn_scale_value"]) == (2, axes["scale"])

        assert not (attrs["causal_mask"] or attrs["causal_mask_bottom_right"]), seed
        align = "BOTTOM_RIGHT" if causal else "TOP_LEFT"
        if window == "none":
            want = (align, None, 0 if causal else None)
        elif window == "top_left_causal":
            want = ("TOP_LEFT", None, 0)
        else:
            want = (align, attrs["left_bound"], axes["wr"])
            left = attrs["left_bound"]
            assert isinstance(left, int) and left >= max(1, axes["wl"]), seed
            assert _row_has_a_key(left, axes["wr"], align == "TOP_LEFT", sq, skv), seed
            # raised only as far as a key per row needs
            if left > axes["wl"]:
                raised_left += 1
                assert not _row_has_a_key(
                    left - 1, axes["wr"], align == "TOP_LEFT", sq, skv
                ), seed
        assert (
            attrs["diagonal_alignment"],
            attrs["left_bound"],
            attrs["right_bound"],
        ) == want
        if window != "sliding":
            assert attrs["left_bound"] is None
        assert _row_has_a_key(
            attrs["left_bound"],
            attrs["right_bound"],
            attrs["diagonal_alignment"] == "TOP_LEFT",
            sq,
            skv,
        ), seed

        seen.add((stats, window, causal))
        passable_stats += (
            stats == "on"
            and window == "none"
            and not causal
            and axes["alignment"] >= 16
        )
    assert seen == {
        (stats, window, causal)
        for stats in ("off", "on")
        for window in ("none", "top_left_causal", "sliding")
        for causal in (False, True)
    }
    assert raised_left > 0, "no sample needed a left bound above the drawn window"
    # the only region ASM_SDPA fwd can answer with stats (it declines stats under any mask and
    # below 16-byte alignment) stays drawn often enough to check on a device
    assert passable_stats > 10


def test_sdpa_forward_constraints_ignore_the_late_stats_and_window_axes():
    # bug caught: a constraint that reads a late axis rejects the draw, so the shared generator
    # advances and a recorded seed fixes other shapes (the digest test would catch the shapes
    # only where a seed happens to hit it); the window is legal by its derive chain instead
    sp = space_by_id("SdpaFwd/Default")
    rng = random.Random(3)
    for _ in range(3000):
        axes = {
            name: (
                axis.values[rng.randrange(len(axis.values))]
                if not axis.is_range
                else rng.randint(axis.lo, min(axis.hi, 600))
            )
            for name, axis in sp.axes.items()
        }
        base = dict(axes, stats="off", window="none")
        for stats in ("off", "on"):
            for window in sp.axes["window"].values:
                varied = dict(axes, stats=stats, window=window)
                assert (sp.first_violation(sp.derive_env(varied)) is None) == (
                    sp.first_violation(sp.derive_env(base)) is None
                ), varied


def _flash2_crossover(sq, skv):
    """``useFlash2ForShape`` (HipFlash2FwdPlanBuilder.hpp), written out again."""
    return sq > 1 and sq * skv > -(-sq // 64) * 6000


def test_flash2_region_space_draws_the_engine_region_and_two_probe_slices():
    # bug caught: a sample inside the region that the engine's gates would decline (the on-device
    # PASS count would then say nothing), a probe slice that fails more than the one gate it is
    # named for, a layout or mask that never reaches the graph, a feature the builder declines
    # outright (stats, attn_mask), and a slice or edge of the crossover the sampler never draws
    sp = space_by_id("SdpaFwd/Flash2Region")
    template = json.loads(sp.template_path.read_text())
    seen, edges, claimable = set(), set(), 0
    for seed in range(800):
        s = sampler.sample(sp, seed)
        axes = s.axis_values
        env = sp.derive_env(axes)
        assert sp.first_violation(env) is None
        assert not {"add_tensors", "node_patches"} & set(s.row)
        graph = expanded_graph(template, s.row)
        assert len(graph["tensors"]) == 4
        node = graph["nodes"][0]
        attrs, ins, outs = node["attributes"], node["inputs"], node["outputs"]
        by_name = {t["name"]: t for t in graph["tensors"]}
        b, hkv, hq, sq, skv, d = (
            axes["b"], axes["hkv"], env["hq"], env["sq"], axes["skv"], axes["d"]
        )  # fmt: skip
        assert hq % hkv == 0 and d in (64, 128)
        assert {t["data_type"] for t in graph["tensors"]} == {"half"}
        assert by_name["Q"]["dims"] == [b, hq, sq, d]
        assert by_name["SdpaFwd_0::O"]["dims"] == [b, hq, sq, d]
        assert by_name["K"]["dims"] == by_name["V"]["dims"] == [b, hkv, skv, d]
        for t in graph["tensors"]:
            assert t["strides"][3] == 1, (
                seed,
                t["name"],
            )  # the builder's only stride gate
            assert _packed(t["dims"], t["strides"])
        bshd = axes["layout"] == "bshd"
        assert (by_name["Q"]["strides"][1] == d) == bshd, seed
        # no feature the builder declines outright
        assert attrs["generate_stats"] is None and outs["stats_tensor_uid"] is None
        assert ins["attn_mask_tensor_uid"] is None and attrs["left_bound"] is None
        assert not (attrs["causal_mask"] or attrs["causal_mask_bottom_right"])
        assert attrs["diagonal_alignment"] == "TOP_LEFT"
        assert attrs["right_bound"] == (0 if axes["causal"] else None)

        slice_ = axes["slice"]
        assert _flash2_crossover(sq, skv) == (slice_ != "below_crossover"), seed
        assert (sq % 64 == 0) == (slice_ != "partial_q"), seed  # K3
        assert b * hq * sq * skv * d <= 2**31
        seen.add((slice_, axes["layout"], axes["causal"], d))
        if skv in (93, 94):
            edges.add((skv, slice_))
        claimable += slice_ == "region" and axes["alignment"] >= 8
    assert seen == {
        (slice_, layout, causal, d)
        for slice_ in ("region", "below_crossover", "partial_q")
        for layout in ("bshd", "bhsd")
        for causal in (False, True)
        for d in (64, 128)
    }
    # seqlen_kv 94 is the first one over the crossover for a multiple of 64, and 93 the last under it
    assert (93, "below_crossover") in edges and (94, "region") in edges
    assert (93, "region") not in edges and (94, "below_crossover") not in edges
    # the region the engine claims at an alignment it accepts (Q and O need 8) stays the bulk
    assert claimable > 200


def test_flash2_region_crossover_constraint_follows_use_flash2_for_shape():
    # bug caught: a constraint off by one from useFlash2ForShape (the crossover is strict, and
    # the ceil is over 64-row query tiles), which would draw a region sample the engine declines
    # or reject an edge it claims
    sp = space_by_id("SdpaFwd/Flash2Region")
    rng = random.Random(5)
    legal = {True: 0, False: 0}
    for _ in range(4000):
        axes = {
            name: (
                axis.values[rng.randrange(len(axis.values))]
                if not axis.is_range
                else rng.randint(axis.lo, rng.choice([axis.hi, 200]))
            )
            for name, axis in sp.axes.items()
        }
        env = sp.derive_env(axes)
        cost = axes["b"] * env["hq"] * env["sq"] * axes["skv"] * axes["d"] <= 2**31
        crossover = _flash2_crossover(env["sq"], axes["skv"])
        expected = cost and crossover == (axes["slice"] != "below_crossover")
        got = sp.first_violation(env) is None
        assert got == expected, (axes, env["sq"])
        legal[got] += 1
    assert legal[True] > 100 and legal[False] > 100


GFX1151_SPACES = ["SdpaFwd/Gfx1151", "SdpaFwd/Gfx1151Decode"]
GFX1151_CATALOGS = (
    TESTS_ROOT.parent
    / "hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE"
)


def _gfx1151_catalog(space_id):
    """The metadata of every kernel descriptor the shipped catalog of the space's engine holds."""
    name = "gfx1151_wmma_attention" + ("_decode" if space_id.endswith("Decode") else "")
    path = GFX1151_CATALOGS / name / f"{name}.kdp.json"
    if not path.is_file():
        pytest.skip("the hip-kernel-provider descriptors are not in this checkout")
    return [d["metadata"] for d in json.loads(path.read_text())["kernelDescriptors"]]


def _contiguous(dims):
    strides, running = [], 1
    for d in reversed(dims):
        strides.append(running)
        running *= d
    return strides[::-1]


@pytest.mark.parametrize("space_id", GFX1151_SPACES)
def test_gfx1151_space_draws_the_shipped_buckets_and_renders_every_feature(space_id):
    # bug caught: a head-size pair, dtype, bias kind or mask the shipped descriptors do not hold
    # (it would decline and say nothing), a catalog row the space never draws, a bias or LSE
    # tensor with other dims, strides, dtype or alignment than the sample's, a wiring key that
    # misses (attn_mask_tensor_uid, stats_tensor_uid, generate_stats, left_bound: the case would
    # pass without the feature), uids a freeze renumbers, and a row with no key under its mask
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    catalog = _gfx1151_catalog(space_id)
    shipped_heads = {(m["head_size"], m["head_size_v"]) for m in catalog}
    probe = {(100, 100)}  # not a multiple of 8: both engines decline it
    assert {tuple(v) for v in sp.axes["hd"].values} == shipped_heads | probe
    assert {m["dtype"] for m in catalog} == {"FP16", "BF16"}
    assert {m["bias"] for m in catalog} == set(sp.axes["bias"].values)
    # a 'causal' prefill object and a 'runtime' decode object both serve every mask drawn
    assert {m["mask_mode"] for m in catalog} == {
        "runtime" if space_id.endswith("Decode") else "causal"
    }
    dtypes = {"half": "FP16", "bfloat16": "BF16"}

    seen = {name: set() for name in ("dtype", "layout", "mask", "bias", "stats", "hd")}
    bias_shapes, claimable = set(), 0
    for seed in range(700):
        s = sampler.sample(sp, seed)
        axes = s.axis_values
        env = sp.derive_env(axes)
        graph = expanded_graph(template, s.row)
        node = graph["nodes"][0]
        attrs, ins, outs = node["attributes"], node["inputs"], node["outputs"]
        by_name = {t["name"]: t for t in graph["tensors"]}
        b, hkv, hq, sq, skv = axes["b"], axes["hkv"], env["hq"], axes["sq"], axes["skv"]
        dq, dv = axes["hd"]
        dtype, bias, stats = axes["dtype"], axes["bias"], axes["stats"]
        assert dtype in dtypes and hq % hkv == 0
        assert len(graph["tensors"]) == 4 + (bias != "none") + (stats == "on"), seed
        assert bu.canonical_uid_map_by_name(graph) == {
            t["uid"]: t["uid"] for t in graph["tensors"]
        }, seed  # a freeze (import_graph.py) renumbers by sorted name and must keep these

        assert by_name["Q"]["dims"] == [b, hq, sq, dq]
        assert by_name["K"]["dims"] == [b, hkv, skv, dq]
        assert by_name["V"]["dims"] == [b, hkv, skv, dv]
        assert by_name["SdpaFwd_0::O"]["dims"] == [b, hq, sq, dv]
        base = [by_name[n] for n in ("Q", "K", "V", "SdpaFwd_0::O")]
        assert {t["data_type"] for t in base} == {dtype}
        for t in base:
            assert t["strides"][3] == 1 and _packed(t["dims"], t["strides"])
            if dq % 8 == 0 and dv % 8 == 0:  # hasAlignedRows
                assert all(
                    t["strides"][i] % 8 == 0 for i in range(3) if t["dims"][i] > 1
                ), (seed, t["name"])
        assert by_name["Q"]["strides"] == (
            [sq * hq * dq, dq, hq * dq, 1]
            if axes["layout"] == "bshd"
            else [hq * sq * dq, sq * dq, dq, 1]
        )

        # the mask, by the cuDNN semantics of left_bound
        mask = axes["mask"]
        want = {
            "none": ("TOP_LEFT", None, None),
            "causal_top_left": ("TOP_LEFT", None, 0),
            "causal_bottom_right": ("BOTTOM_RIGHT", None, 0),
            "window": (axes["walign"].upper(), axes["wl"], axes["wr"]),
        }[mask]
        assert (attrs["diagonal_alignment"], attrs["left_bound"], attrs["right_bound"]) == want, seed  # fmt: skip
        assert not (attrs["causal_mask"] or attrs["causal_mask_bottom_right"])
        assert _row_has_a_key(
            attrs["left_bound"],
            attrs["right_bound"],
            attrs["diagonal_alignment"] == "TOP_LEFT",
            sq,
            skv,
        ), seed

        # the additive mask: [B|1, Hq|1, Sq|1, Sk], packed, FP32 or the Q dtype, wired by uid
        if bias == "none":
            assert "bias" not in by_name and ins["attn_mask_tensor_uid"] is None
        else:
            dims = [
                1 if axes["bcast_b"] else b,
                1 if axes["bcast_h"] else hq,
                1 if axes["bcast_q"] else sq,
                skv,
            ]
            t = by_name["bias"]
            assert (t["uid"], t["dims"], t["strides"]) == (4, dims, _contiguous(dims))
            assert t["data_type"] == ("float" if bias == "f32" else dtype), seed
            assert t["alignment"] == axes["alignment"]
            assert not t["virtual"] and not t["is_runtime_pass_by_value"]
            assert ins["attn_mask_tensor_uid"] == 4
            bias_shapes.add((axes["bcast_b"], axes["bcast_h"], axes["bcast_q"]))
        # the log-sum-exp output
        if stats == "off":
            assert "stats" not in by_name
            assert attrs["generate_stats"] is None and outs["stats_tensor_uid"] is None
        else:
            t = by_name["stats"]
            uid = 4 if bias == "none" else 5
            assert (t["uid"], t["dims"], t["strides"]) == (
                uid, [b, hq, sq, 1], [hq * sq, sq, 1, 1],
            ), seed  # fmt: skip
            assert t["data_type"] == "float" and t["alignment"] == axes["alignment"]
            assert attrs["generate_stats"] is True and outs["stats_tensor_uid"] == uid
        assert {by_name[n]["alignment"] for n in ("Q", "K", "V", "SdpaFwd_0::O")} == {
            axes["alignment"]
        }

        for name in seen:
            seen[name].add(json.dumps(axes[name]))
        claimable += (dq, dv) != (100, 100) and axes["alignment"] >= 16
    for name in seen:
        assert seen[name] == {json.dumps(v) for v in sp.axes[name].values}, name
    assert bias_shapes == {
        (b, h, q) for b in (False, True) for h in (False, True) for q in (False, True)
    }
    # the part the engine can answer (heads it ships, 16-byte alignment) stays the bulk
    assert claimable > 200


def _decode_kv_range(sq, skv, bottom_right, window_left, window_right):
    """``decodeKvRange`` (Gfx1151WmmaAttentionNative.cpp), written out again: the keys any query
    row sees, [lo, hi); the window bounds are the kernel's (window_left = left_bound - 1).
    """
    ctx = skv - sq if bottom_right else 0
    lo = max(ctx - window_left, 0) if window_left >= 0 else 0
    hi = max(min(ctx + window_right + sq, skv), 0) if window_right >= 0 else skv
    return lo, hi


def _rdna_use_decode_gfx11(sq, visible, hq, hkv, batch, num_cus):
    """``rdnaAttentionUseDecode`` for a gfx11 device, written out again."""
    groups = max(1, hq // max(hkv, 1))
    if sq > 64:
        return 4 * batch * hq * ((sq + 127) // 128) < num_cus
    return (
        groups * sq < 512 or visible < 64 * num_cus or batch * visible < 512 * num_cus
    )


def _routed_to_decode(attrs, b, hq, hkv, sq, skv, num_cus):
    left, right = attrs["left_bound"], attrs["right_bound"]
    lo, hi = _decode_kv_range(
        sq,
        skv,
        attrs["diagonal_alignment"] == "BOTTOM_RIGHT",
        -1 if left is None else left - 1,
        -1 if right is None else right,
    )
    return _rdna_use_decode_gfx11(sq, hi - lo, hq, hkv, b, num_cus)


@pytest.mark.parametrize("space_id", GFX1151_SPACES)
def test_gfx1151_spaces_split_prefill_and_decode_by_the_dispatch_rule(space_id):
    # bug caught: a prefill sample the decode engine would take (the prefill engine declines
    # it), a decode sample the dispatcher sends to prefill, and a split that holds at one
    # compute-unit count only (the rule reads the device: the spaces keep to shapes whose route
    # is the same at 32 and 40 compute units). The mask moves the route through the visible keys.
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    decode_space = space_id.endswith("Decode")
    short, long, routed_by_window = 0, 0, 0
    for seed in range(700):
        s = sampler.sample(sp, seed)
        axes = s.axis_values
        env = sp.derive_env(axes)
        attrs = expanded_graph(template, s.row)["nodes"][0]["attributes"]
        args = (axes["b"], env["hq"], axes["hkv"], axes["sq"], axes["skv"])
        d32, d40 = (_routed_to_decode(attrs, *args, n) for n in (32, 40))
        assert not d32 or d40  # more compute units never move a shape to prefill
        short += axes["sq"] <= 64
        long += axes["sq"] > 64
        if axes["hd"] == [100, 100]:
            continue  # the head-size probe has no route: the matcher declines it first
        if decode_space:
            assert d32 and d40, (seed, axes["sq"], axes["skv"], axes["mask"])
        else:
            assert not d32 and not d40, (seed, axes["sq"], axes["skv"], axes["mask"])
        routed_by_window += decode_space and axes["mask"] != "none" and axes["sq"] <= 64
    if decode_space:
        assert short > 100 and long > 20 and routed_by_window > 20
    else:
        assert (
            short == 0 and long == 700
        )  # gfx11: seqlen_q <= 64 keeps decode below 2560 keys


@pytest.mark.parametrize("space_id", GFX1151_SPACES)
def test_gfx1151_space_constraints_follow_the_dispatch_rule_and_the_reference_mask(
    space_id,
):
    # bug caught: a constraint off by one from rdnaAttentionUseDecode or the mask (a window that
    # leaves a row with no key, or the wrong side of the 32/40 compute-unit bounds), checked on
    # axes drawn independently of the sampler so that rejected draws are exercised too
    sp = space_by_id(space_id)
    decode_space = space_id.endswith("Decode")
    rng = random.Random(9)
    legal = {True: 0, False: 0}
    for _ in range(3000):
        axes = {
            name: (
                axis.values[rng.randrange(len(axis.values))]
                if not axis.is_range
                else rng.randint(axis.lo, rng.choice([axis.hi, min(axis.hi, 300)]))
            )
            for name, axis in sp.axes.items()
        }
        env = sp.derive_env(axes)
        attrs = {
            "left_bound": env["left"],
            "right_bound": env["right"],
            "diagonal_alignment": env["align"],
        }
        dq, dv = axes["hd"]
        sq, skv, hq = axes["sq"], axes["skv"], env["hq"]
        sees_keys = _row_has_a_key(
            env["left"], env["right"], env["align"] == "TOP_LEFT", sq, skv
        )
        cost = axes["b"] * hq * sq * skv * dq <= 2**31
        args = (axes["b"], hq, axes["hkv"], sq, skv)
        n = 32 if decode_space else 40
        routed = _routed_to_decode(attrs, *args, n)
        route_ok = (routed if decode_space else not routed) or dq % 8 != 0
        got = sp.first_violation(env) is None
        assert got == (sees_keys and cost and route_ok), (
            axes,
            env["left"],
            env["right"],
        )
        legal[got] += 1
    assert legal[True] > 100 and legal[False] > 100


def test_sdpa_backward_space_draws_every_feature_and_renders_it():
    # bug caught: a mask axis that never reaches the node's left_bound, right_bound or
    # diagonal_alignment, a scale patch that leaves the template's literal 1/sqrt(128), an
    # inputs recipe for the derived O or LSE (which the fill overwrites), a head size or mask the
    # space cannot draw, and a cost ceiling that a sample exceeds
    sp = space_by_id("SdpaBackward/Default")
    template = json.loads(sp.template_path.read_text())
    node = template["nodes"][0]
    uid = {t["name"]: t["uid"] for t in template["tensors"]}
    literal_scale = node["parameters"]["attn_scale_value"]
    bands = {
        "none": ("TOP_LEFT", None, None),
        "causal_top_left": ("TOP_LEFT", None, 0),
        "causal_bottom_right": ("BOTTOM_RIGHT", None, 0),
    }
    seen = {name: set() for name in sp.axes}
    seen_heads_masks, groups_above_one, scale_moved = set(), 0, 0
    for seed in range(800):
        s = sampler.sample(sp, seed)
        axes = s.axis_values
        env = sp.derive_env(axes)
        assert sp.first_violation(env) is None
        for name, value in axes.items():
            seen[name].add(json.dumps(value))
        seen_heads_masks.add((axes["d"], axes["mask"]))
        groups_above_one += axes["group"] > 1

        graph = expanded_graph(template, s.row)
        params = graph["nodes"][0]["parameters"]
        assert params["attn_scale_value"] == axes["scale"]  # the patch, not the literal
        scale_moved += (
            abs(axes["scale"] - literal_scale) > 1e-6
        )  # the template holds a float32
        want = bands.get(axes["mask"]) or (
            axes["walign"].upper(),
            axes["wl"],
            axes["wr"],
        )
        assert (
            params["diagonal_alignment"], params["left_bound"], params["right_bound"]
        ) == want, (seed, axes["mask"])  # fmt: skip
        assert not (
            params["causal_mask"] or params["causal_mask_bottom_right"]
        ), "the deprecated flags stay off: the bounds carry the mask"

        by_name = {t["name"]: t for t in graph["tensors"]}
        d, hq, hkv = axes["d"], env["hq"], axes["hkv"]
        for name in ("Q", "O", "dO", "SdpaBwd_0::dQ"):
            assert by_name[name]["dims"] == [axes["b"], hq, axes["sq"], d], name
        for name in ("K", "V", "SdpaBwd_0::dK", "SdpaBwd_0::dV"):
            assert by_name[name]["dims"] == [axes["b"], hkv, axes["skv"], d], name
        assert by_name["stats"]["dims"] == [axes["b"], hq, axes["sq"], 1]
        assert by_name["stats"]["data_type"] == "float"
        assert {by_name[n]["data_type"] for n in by_name if n != "stats"} == {
            axes["dtype"]
        }
        # the ceiling the description states: dqk == dv == d
        assert axes["b"] * hq * axes["sq"] * axes["skv"] * (d + d) <= 2**29

        # O and the LSE are derived by the fill from Q, K and V, so they carry no recipe
        recipes = {int(u) for u in s.row["metadata"]["inputs"]}
        assert recipes == {uid[n] for n in ("Q", "K", "V", "dO")}, recipes

    for name, axis in sp.axes.items():
        if not axis.is_range:
            assert seen[name] == {json.dumps(v) for v in axis.values}, name
    assert seen_heads_masks == {
        (d, m) for d in (64, 128, 192) for m in sp.axes["mask"].values
    }
    assert (
        0 < groups_above_one < 400
    ), "a small grouped-query slice, kept as decline probes"
    assert scale_moved > 400


def test_sdpa_backward_rows_see_a_key_follows_the_reference_mask():
    # bug caught: a constraint that disagrees with the CPU reference's mask (left_bound L keeps
    # the L keys ending at the diagonal), so a sample leaves a query row with no key (an LSE of
    # -inf the kernels do not define) or the space drops legal windows
    sp = space_by_id("SdpaBackward/Default")
    rng = random.Random(7)
    seen = set()
    for _ in range(6000):
        axes = {
            name: (
                axis.values[rng.randrange(len(axis.values))]
                if not axis.is_range
                else rng.randint(axis.lo, min(axis.hi, 24))
            )
            for name, axis in sp.axes.items()
        }
        env = sp.derive_env(axes)
        attrs = sp.render(env)["values"]["attributes"]
        left, right = attrs["parameters__left_bound"], attrs["parameters__right_bound"]
        top_left = attrs["parameters__diagonal_alignment"] == "TOP_LEFT"
        sq, skv = axes["sq"], axes["skv"]
        offset = 0 if top_left else skv - sq

        def masked(i, j):  # CpuFpReferenceSdpa::isMasked
            return (right is not None and j >= i + 1 + offset + right) or (
                left is not None and j <= i + offset - left
            )

        every_row_sees_a_key = all(
            any(not masked(i, j) for j in range(skv)) for i in range(sq)
        )
        assert env["rows_see_a_key"] == every_row_sees_a_key, (axes, left, right)
        seen.add((axes["mask"], every_row_sees_a_key))
    assert seen == {
        (m, ok) for m in ("none", "causal_top_left", "causal_bottom_right", "window")
        for ok in (True, False)
    } - {("none", False), ("causal_top_left", False)}  # fmt: skip


# --------------------------------------------------------------------------
# S13: the domain axis (phase 1 ranges, phase 2 fill kinds) and the late-axis mechanism
# --------------------------------------------------------------------------

DOMAINS = [
    "default", "narrow", "wide", "positive", "constant",
    "nonzero", "zeros", "large", "subnormal", "nan", "inf",
]  # fmt: skip
POISONING_DOMAINS = ("nan", "inf")
# What a domain renders for a primary data operand:
# kind, lo, hi, value, scale, special, fraction, case validator
DOMAIN_TABLE = {
    "default": ("free", -1, 1, 0, 1, "nan", 0.05, "default"),
    "narrow": ("free", -0.01, 0.01, 0, 1, "nan", 0.05, "default"),
    "wide": ("free", -8, 8, 0, 1, "nan", 0.05, "default"),
    "positive": ("free", 0.25, 4, 0, 1, "nan", 0.05, "default"),
    "constant": ("fixed", 0.5, 0.5, 0.5, 1, "nan", 0.05, "default"),
    "nonzero": ("nonzero", -1, 1, 0, 1, "nan", 0.05, "default"),
    "zeros": ("zeros", -1, 1, 0, 1, "nan", 0.05, "default"),
    "large": ("large", -1, 1, 0, 32, "nan", 0.05, "default"),
    "subnormal": ("subnormal", -1, 1, 0, 1, "nan", 0.05, "default"),
    "nan": ("special", -1, 1, 0, 1, "nan", 0.05, "match_nonfinite"),
    "inf": ("special", -1, 1, 0, 1, "inf", 0.05, "match_nonfinite"),
}
LEAF_KEYS = ("kind", "lo", "hi", "value", "scale", "special", "fraction")
# the `large` scale of the spaces whose arithmetic would leave the half range at 32
LARGE_SCALE = {
    "LayernormBackward/Default": 4,
    "RMSNormBackward/Default": 4,
    "SdpaFwd/Default": 8,
    "SdpaBackward/Default": 8,
    "SdpaFwd/Gfx950Dense": 8,
    "SdpaFwd/Flash2Region": 8,
    "SdpaFwd/Gfx1151": 8,
    "SdpaFwd/Gfx1151Decode": 8,
}
# The poisoned share of V: a non-finite element of V reaches only its own head-dim column of the
# output, so the share keeps part of the columns finite for the non-finite mask to grade
# (NonFiniteMask.hpp fails a reference with no finite element). At the 5% of the other spaces
# nearly every column of a long kv sequence holds one and the reference has none.
POISON_FRACTION = {
    "SdpaFwd/Default": 2**-12,
    "SdpaFwd/Gfx950Dense": 2**-12,
    "SdpaFwd/Flash2Region": 2**-12,
    "SdpaFwd/Gfx1151": 2**-12,
    "SdpaFwd/Gfx1151Decode": 2**-12,
    # dscale and dbias sum over every normalization group, so one poisoned group (or stat) makes
    # every dscale element non-finite; one poisoned dy element (the fill plants at least one,
    # which this fraction rounds down to) leaves the other columns and rows finite
    "LayernormBackward/Default": 1e-12,
    "RMSNormBackward/Default": 1e-12,
}
# What a domain renders for one operand of a space instead of the table's row: (lo, hi, scale).
# The kernels round P to bf16 or half for the P.V product, which errs by about 2**-9 * |V| per
# term, and a cancelling sum leaves an output below that error, so atol 0.01 only grades a V of
# magnitude up to about 2. Probed on gfx950 with Q and K wide or large (ASM_SDPA_ENGINE, then
# Gfx950AttentionDense): V in +-8 or large scale 8 fails 82 of 100 and 21 of 60 samples, V in
# +-4 fails 41 of 150 and 14 of 60, V in +-2 fails none of 150 and 60.
OPERAND_DOMAIN = {
    ("SdpaFwd/Default", "V"): {"wide": (-2, 2, 1), "large": (-1, 1, 2)},
    ("SdpaFwd/Gfx950Dense", "V"): {"wide": (-2, 2, 1), "large": (-1, 1, 2)},
    ("SdpaFwd/Flash2Region", "V"): {"wide": (-2, 2, 1), "large": (-1, 1, 2)},
    ("SdpaFwd/Gfx1151", "V"): {"wide": (-2, 2, 1), "large": (-1, 1, 2)},
    ("SdpaFwd/Gfx1151Decode", "V"): {"wide": (-2, 2, 1), "large": (-1, 1, 2)},
    # V and dO of the backward keep magnitude 1 where wide and large widen Q and K (the
    # outputs grow with |dO| * |V| and the engine tolerances are calibrated at magnitude 1)
    **{
        ("SdpaBackward/Default", name): {
            "wide": (-1, 1, 1),
            "positive": (0.25, 1, 1),
            "large": (-1, 1, 1),
        }
        for name in ("V", "dO")
    },
}
# The sample the domain test renders when seed 11 is not one the nan and inf domains poison (a
# causal or window-masked sample, half, or a softmax scale above 1 falls back to the default
# fill, see test_sdpa_poisons_v_only_where_the_reference_and_the_kernels_agree): the first that is.
POISONABLE_SEED = {
    "SdpaFwd/Default": 14,
    "SdpaFwd/Flash2Region": 13,
    "SdpaFwd/Gfx1151Decode": 13,
    "SdpaFwd/Gfx950Dense": 12,
}
# the tensors (by template name) whose fill the domain axis drives, per space
PRIMARY_OPERANDS = {
    "Batchnorm/Default": ["X"],
    "BatchnormBackward/Default": ["x", "dy"],
    "BatchnormInference/Default": ["X"],
    "Layernorm/Default": ["x"],
    "LayernormBackward/Default": ["X", "dY"],
    "RMSNorm/Default": ["x"],
    "RMSNormBackward/Default": ["x", "dy"],
    "ResampleFwd/Default": ["X"],
    "SdpaFwd/Default": ["Q", "K", "V"],
    "SdpaFwd/Gfx950Dense": ["Q", "K", "V"],
    "SdpaFwd/Flash2Region": ["Q", "K", "V"],
    "SdpaFwd/Gfx1151": ["Q", "K", "V"],
    "SdpaFwd/Gfx1151Decode": ["Q", "K", "V"],
    "SdpaBackward/Default": ["Q", "K", "V", "dO"],
}
# The operands that carry the `special` fill in the nan and inf domains, per space. The other
# primary operands keep the default range there. Each choice keeps the non-finite positions
# of the output local to the operand (the description of each space says why).
POISONED_OPERANDS = {
    "Batchnorm/Default": ["prev_running_mean", "prev_running_variance", "scale"],
    "BatchnormBackward/Default": ["scale"],
    "BatchnormInference/Default": ["X"],
    "Layernorm/Default": ["bias", "scale"],
    "LayernormBackward/Default": ["dY"],
    "RMSNorm/Default": ["bias", "scale"],
    "RMSNormBackward/Default": ["dy"],
    "ResampleFwd/Default": ["X"],
    "SdpaFwd/Default": ["V"],
    "SdpaBackward/Default": [],  # nan and inf are not drawn: see that space's description
    "SdpaFwd/Gfx950Dense": ["V"],
    "SdpaFwd/Flash2Region": ["V"],
    "SdpaFwd/Gfx1151": ["V"],
    "SdpaFwd/Gfx1151Decode": ["V"],
}
# sha256 over seeds 0..299 of the pre-existing axes' draws, taken before the domain axis existed
OLD_SHAPE_DIGESTS = {
    "RMSNorm/Default": "7ff3fe8bdf5afd4845a1bf5e631466fe1cb0bfd55a4056059dae694deed4b8f3",
    "SdpaFwd/Default": "13171199efa5647432d6ed13745de2cdcb3925fb5b0589e15f457a4fb6b95302",
    "SdpaFwd/Gfx950Dense": "69623b2107c84d79090cab4630861813954bf838689cb9ae32950835d7e8340c",
}
# the spaces without a domain axis: one fixed shape, only alignment and placement are drawn
FIXED_SHAPE_SPACES = ["Matmul/Align", "SdpaFwd/Flash2"]
# the late axes a space gained after its domain axis, which sit before `domain` in the axis order
EXTRA_LATE_AXES = {"SdpaFwd/Default": ["stats", "window", "wl", "wr"]}
DOMAIN_SPACE_IDS = [s for s in SPACE_IDS if s not in FIXED_SHAPE_SPACES]
ALIGNMENTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PLACEMENTS = ["default", "exact", "under"]
# The tensor (by template name) whose pointer `under` misaligns: the op's primary data
# operand. Every space renders it as metadata.placement_uid, whatever the placement.
PLACEMENT_OPERAND = {
    "Batchnorm/Default": "X",
    "BatchnormBackward/Default": "x",
    "BatchnormInference/Default": "X",
    "Layernorm/Default": "x",
    "LayernormBackward/Default": "X",
    "Matmul/Align": "a",
    "RMSNorm/Default": "x",
    "RMSNormBackward/Default": "x",
    "ResampleFwd/Default": "X",
    "SdpaBackward/Default": "Q",
    "SdpaFwd/Default": "Q",
    "SdpaFwd/Flash2": "Q",
    "SdpaFwd/Flash2Region": "Q",
    "SdpaFwd/Gfx1151": "Q",
    "SdpaFwd/Gfx1151Decode": "Q",
    "SdpaFwd/Gfx950Dense": "Q",
}
PATCHED_OPERANDS = (
    PLACEMENT_OPERAND  # every space declares alignments through tensor_patches
)

# the spaces whose domain axis leaves out nan and inf (each space's description says why)
NONFINITE_EXCLUDED = {"SdpaBackward/Default"}
# the spaces that draw half only: an infinity never poisons there (P underflows), see
# test_sdpa_poisons_v_only_where_the_reference_and_the_kernels_agree
HALF_ONLY_SPACES = {"SdpaFwd/Flash2Region"}


def domains_of(space_id):
    return [
        d
        for d in DOMAINS
        if space_id not in NONFINITE_EXCLUDED or d not in POISONING_DOMAINS
    ]


def test_every_shipped_space_has_its_operands_listed():
    assert sorted(PRIMARY_OPERANDS) == DOMAIN_SPACE_IDS
    assert sorted(POISONED_OPERANDS) == DOMAIN_SPACE_IDS
    assert sorted(PLACEMENT_OPERAND) == SPACE_IDS
    assert sorted(OLD_SHAPE_DIGESTS) == [
        "RMSNorm/Default", "SdpaFwd/Default", "SdpaFwd/Gfx950Dense",
    ]  # fmt: skip


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_the_last_axes_are_domain_alignment_placement_in_that_order(space_id):
    # bug caught: an axis in the middle shifts the draws of the axes after it, a reordered or
    # misspelled value is never rendered by the derive chain, and an old value that moves
    # changes what recorded seeds mean; `late` marks exactly the spaces that had seeds recorded
    sp = space_by_id(space_id)
    tail = ["alignment", "placement"] if space_id in FIXED_SHAPE_SPACES else [
        "domain", "alignment", "placement",
    ]  # fmt: skip
    assert list(sp.axes)[-len(tail) :] == tail
    assert sp.axes["alignment"].values == ALIGNMENTS
    assert sp.axes["placement"].values == PLACEMENTS
    if "domain" in sp.axes:
        assert sp.axes["domain"].values == domains_of(space_id)
    assert [name for name, axis in sp.axes.items() if axis.late] == (
        EXTRA_LATE_AXES.get(space_id, []) + tail
        if space_id in OLD_SHAPE_DIGESTS
        else []
    )
    drawn = [sampler.sample(sp, seed).axis_values for seed in range(300)]
    for name in tail:
        assert {d[name] for d in drawn} == set(sp.axes[name].values)


def _rendered_by_domain(sp, seed=11):
    """The case metadata a legal sample renders under every domain value."""
    sample = sampler.sample(sp, seed)
    return {
        domain: sp.render(sp.derive_env(dict(sample.axis_values, domain=domain)))[
            "metadata"
        ]
        for domain in sp.axes["domain"].values
    }


@pytest.mark.parametrize("space_id", DOMAIN_SPACE_IDS)
def test_domain_renders_the_fill_recipe_of_every_primary_operand(space_id):
    # bug caught: a domain that reaches the wrong uid, drops a leaf, poisons an operand the
    # description does not name, sets the validator for a finite domain, or overrides the
    # tuned ranges of the saved statistics or of the default domain
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    uid_of = {t["name"]: str(t["uid"]) for t in template["tensors"]}
    name_of = {v: k for k, v in uid_of.items()}
    primary = [uid_of[name] for name in PRIMARY_OPERANDS[space_id]]
    poisoned = [uid_of[name] for name in POISONED_OPERANDS[space_id]]
    quick = json.loads((sp.template_path.parent / "sweep.json").read_text())["cases"]
    metas = _rendered_by_domain(sp, seed=POISONABLE_SEED.get(space_id, 11))
    assert set(metas) == set(domains_of(space_id))
    default = metas["default"]["inputs"]
    # a space that poisons nothing carries no special and fraction keys
    leaf_keys = LEAF_KEYS if POISONED_OPERANDS[space_id] else LEAF_KEYS[:5]
    for domain, (
        kind,
        lo,
        hi,
        value,
        scale,
        special,
        fraction,
        validator,
    ) in DOMAIN_TABLE.items():
        if domain not in metas:
            continue
        meta = metas[domain]
        inputs = meta["inputs"]
        assert meta["purpose"] == "probe"
        if domain == "inf" and space_id in HALF_ONLY_SPACES:
            # P underflows in half, so an infinity falls back to the default fill (the poison
            # rule of test_sdpa_poisons_v_only_where_the_reference_and_the_kernels_agree)
            assert meta["validator"] == "default"
            for uid in primary:
                keep = ("kind", "lo", "hi", "value")
                assert {k: inputs[uid][k] for k in keep} == {
                    k: default[uid][k] for k in keep
                }
            continue
        assert meta["validator"] == validator
        if domain == "large":
            scale = LARGE_SCALE.get(space_id, scale)
        fraction = POISON_FRACTION.get(space_id, fraction)
        for uid in primary:
            leaf = inputs[uid]
            if domain == "default":
                tuned = {
                    (e["lo"], e["hi"])
                    for c in quick
                    if (e := (c["metadata"].get("inputs") or {}).get(uid))
                    and e["kind"] == "free"
                }
                if tuned:
                    # the default domain keeps the range the quick sweep tunes for this uid
                    # (the quick files hold the float32 spelling of 0.1 and the like)
                    assert any(
                        leaf["lo"] == pytest.approx(t_lo, rel=1e-6)
                        and leaf["hi"] == pytest.approx(t_hi, rel=1e-6)
                        for t_lo, t_hi in tuned
                    ), (uid, leaf, tuned)
                    lo, hi = leaf["lo"], leaf["hi"]
            if domain in POISONING_DOMAINS and uid not in poisoned:
                # a primary operand that is not poisoned keeps the default range
                keep = ("kind", "lo", "hi", "value")
                assert {k: leaf[k] for k in keep} == {k: default[uid][k] for k in keep}, (
                    space_id, domain, uid, leaf,
                )  # fmt: skip
                continue
            lo_u, hi_u, scale_u = OPERAND_DOMAIN.get((space_id, name_of[uid]), {}).get(
                domain, (lo, hi, scale)
            )
            expected = dict(
                zip(leaf_keys, (kind, lo_u, hi_u, value, scale_u, special, fraction))
            )
            assert {k: leaf[k] for k in leaf_keys} == expected, (
                space_id,
                domain,
                uid,
                leaf,
            )
            assert set(leaf) == set(leaf_keys), (space_id, domain, uid, leaf)
        for uid in poisoned:
            if uid in primary:
                continue
            leaf = inputs[uid]
            if domain in POISONING_DOMAINS:
                assert (leaf["kind"], leaf["special"], leaf["fraction"]) == (
                    "special", special, fraction,
                ), (space_id, domain, uid, leaf)  # fmt: skip
                # the finite share keeps the range the operand has without poison
                assert (leaf["lo"], leaf["hi"]) == (
                    default[uid]["lo"],
                    default[uid]["hi"],
                )
            else:
                keep = ("kind", "lo", "hi")
                assert {k: leaf[k] for k in keep} == {k: default[uid][k] for k in keep}
    # every other entry is literal and does not move with the domain
    others = {
        domain: {
            u: e
            for u, e in meta["inputs"].items()
            if u not in primary and u not in poisoned
        }
        for domain, meta in metas.items()
    }
    assert all(o == others["default"] for o in others.values()), others
    assert all(set(primary + poisoned) <= set(m["inputs"]) for m in metas.values())


@pytest.mark.parametrize("space_id", ["Layernorm/Default", "RMSNorm/Default"])
def test_norm_poisons_x_when_scale_and_bias_are_too_small_to_leave_a_finite_column(
    space_id,
):
    # bug caught: the one element a special fill always plants covers a scale and bias of a single
    # element whole, y has no finite element and the non-finite mask fails the reference by design
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    uid = {t["name"]: str(t["uid"]) for t in template["tensors"]}
    seen = set()
    for seed in range(400):
        axes = sampler.sample(sp, seed).axis_values
        for domain in POISONING_DOMAINS:
            env = sp.derive_env(dict(axes, domain=domain))
            inputs = sp.render(env)["metadata"]["inputs"]
            small = env["poi_m"] < 4
            kinds = {name: inputs[uid[name]]["kind"] for name in ("x", "scale", "bias")}
            assert kinds == (
                {"x": "special", "scale": "free", "bias": "free"}
                if small
                else {"x": "free", "scale": "special", "bias": "special"}
            ), (seed, domain, env["poi_m"], kinds)  # fmt: skip
            seen.add(small)
    assert seen == {True, False}


def test_resample_nan_domain_plants_infinity_under_maxpool_only():
    # bug caught: a NaN in a maxpool window. The HIP kernel never selects it, the CPU reference
    # keeps it only as the first element of the window, so the two disagree on most such windows
    sp = space_by_id("ResampleFwd/Default")
    template = json.loads(sp.template_path.read_text())
    x_uid = {t["name"]: str(t["uid"]) for t in template["tensors"]}["X"]
    seen = set()
    for seed in range(200):
        axes = sampler.sample(sp, seed).axis_values
        for domain in POISONING_DOMAINS:
            leaf = sp.render(sp.derive_env(dict(axes, domain=domain)))["metadata"][
                "inputs"
            ][x_uid]
            planted = "nan" if domain == "nan" and axes["mode"] != "maxpool" else "inf"
            assert (leaf["kind"], leaf["special"]) == ("special", planted), (
                seed,
                domain,
                leaf,
            )
            seen.add((domain, axes["mode"] == "maxpool"))
    assert seen == {("nan", True), ("nan", False), ("inf", True), ("inf", False)}


@pytest.mark.parametrize(
    "space_id", [s for s in DOMAIN_SPACE_IDS if s.startswith("SdpaFwd/")]
)
def test_sdpa_poisons_v_only_where_the_reference_and_the_kernels_agree(space_id):
    # bug caught: a non-finite V under a causal or window mask (probed on gfx950 with a causal
    # mask: NaN in the reference where the kernel is finite) and an infinity where P underflows
    # (half, or a softmax scale above 1 over bfloat16; probed: inf in one side and NaN in the
    # other, as when 0 * inf meets a P that
    # rounds to zero at different logit gaps); both fail a case that no engine defect explains
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    v_uid = {t["name"]: str(t["uid"]) for t in template["tensors"]}["V"]
    seen = set()
    for seed in range(300):
        axes = sampler.sample(sp, seed).axis_values
        for domain in POISONING_DOMAINS:
            env = sp.derive_env(dict(axes, domain=domain))
            meta = sp.render(env)["metadata"]
            underflows = axes["dtype"] == "half" or axes["scale"] > 1
            masked = (
                axes.get("causal", False)
                or axes.get("window", "none") != "none"
                or axes.get("mask", "none") != "none"
            )
            poisoned = not masked and not (domain == "inf" and underflows)
            leaf = meta["inputs"][v_uid]
            assert (leaf["kind"], meta["validator"]) == (
                ("special", "match_nonfinite") if poisoned else ("free", "default")
            ), (seed, domain, leaf)  # fmt: skip
            assert (leaf["lo"], leaf["hi"]) == (-1, 1)
            seen.add((domain, poisoned))
    # a half-only space never poisons with an infinity: P underflows there
    both = {("nan", True), ("nan", False), ("inf", True), ("inf", False)}
    assert seen == (
        both if "bfloat16" in sp.axes["dtype"].values else both - {("inf", True)}
    )


@pytest.mark.parametrize("space_id", DOMAIN_SPACE_IDS)
def test_rendered_metadata_has_only_what_the_harness_accepts(space_id):
    # bug caught: a metadata key or fill leaf the strict C++ parsers reject
    # (BundleMetadata.hpp validator, InputFillRecipes.hpp parseFill) fails every case of
    # the space at load, and a misspelled leaf value does so for one domain only
    sp = space_by_id(space_id)
    kinds = {"free", "fixed", "nonzero", "zeros", "large", "subnormal", "special"}
    entry_keys = set(LEAF_KEYS) | {"seed"}

    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    for domain, meta in _rendered_by_domain(sp).items():
        assert set(meta) <= {"purpose", "validator", "inputs", "placement", "placement_uid"}, (
            domain, set(meta),
        )  # fmt: skip
        assert meta["validator"] in ("default", "match_nonfinite")
        assert meta["placement"] in PLACEMENTS and isinstance(
            meta["placement_uid"], int
        )
        for uid, leaf in meta["inputs"].items():
            assert uid.isdigit() and set(leaf) <= entry_keys, (domain, uid, leaf)
            assert leaf["kind"] in kinds, (domain, uid, leaf)
            for key in ("lo", "hi", "value", "scale", "fraction"):
                assert key not in leaf or number(leaf[key]), (domain, uid, key, leaf)
            if leaf["kind"] in ("free", "nonzero", "special"):
                assert leaf["lo"] <= leaf["hi"], (domain, uid, leaf)
            if leaf["kind"] == "large":
                assert leaf["scale"] > 0, (domain, uid, leaf)
            if leaf["kind"] == "special":
                assert leaf["special"] in ("nan", "inf", "-inf"), (domain, uid, leaf)
                assert 0 <= leaf["fraction"] <= 1, (domain, uid, leaf)
    allowed = {
        "format_version", "generator", "notes", "purpose", "seed", "validator", "inputs",
        "placement", "placement_uid",
    }  # fmt: skip
    for seed in range(40):
        assert set(sampler.sample(sp, seed).row["metadata"]) <= allowed


@pytest.mark.parametrize("space_id", sorted(OLD_SHAPE_DIGESTS))
def test_the_domain_axis_keeps_the_shapes_that_recorded_seeds_fix(space_id):
    # bug caught: an appended axis drawn from the shared generator shifts every later draw
    # after a retry, so seeds recorded before the domain axis would fix other shapes
    sp = space_by_id(space_id)
    old = [name for name, axis in sp.axes.items() if not axis.late]
    digest = hashlib.sha256()
    for seed in range(300):
        drawn = sampler.sample(sp, seed).axis_values
        digest.update(
            json.dumps(
                {name: drawn[name] for name in old},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    assert digest.hexdigest() == OLD_SHAPE_DIGESTS[space_id]


def test_a_late_axis_does_not_move_the_draws_of_other_axes_across_retries(tmp_path):
    # bug caught: a late axis drawn from the shared generator shifts the following draws
    # once a constraint rejects an attempt
    base = {
        "a": {"values": [1, 2, 3]},
        "n": {"range": [1, 100], "boundaries": [1, 100]},
    }

    def drawn(axes, seed):
        sp = space_lib.load_space(write_space(tmp_path, axes=axes))
        s = sampler.sample(sp, seed)
        return (s.axis_values["a"], s.axis_values["n"]), s.attempts

    seeds = range(200)
    plain = {s: drawn(base, s) for s in seeds}
    assert any(attempts > 1 for _, attempts in plain.values()), "no seed retried"
    late = {**base, "z": {"values": [1, 2, 3], "late": True}}
    eager = {**base, "z": {"values": [1, 2, 3]}}
    assert {s: drawn(late, s) for s in seeds} == plain
    # the control: the same axis without `late` does move later draws of retried seeds
    assert {s: drawn(eager, s) for s in seeds} != plain


def test_a_late_axis_is_deterministic_per_seed_and_takes_every_value(tmp_path):
    # bug caught: a late stream that ignores the seed or the attempt draws one value always
    axes = {
        "a": {"values": [1, 2, 3]},
        "n": {"range": [1, 100], "boundaries": [1, 100]},
        "z": {"values": [1, 2, 3, 4], "late": True},
    }
    sp = space_lib.load_space(write_space(tmp_path, axes=axes, constraints=[]))
    first = [sampler.sample(sp, seed).axis_values["z"] for seed in range(100)]
    assert first == [sampler.sample(sp, seed).axis_values["z"] for seed in range(100)]
    assert set(first) == {1, 2, 3, 4}


def test_a_non_late_axis_after_a_late_axis_is_a_load_error(tmp_path):
    # bug caught: an eager axis behind a late one would still draw from the shared generator
    axes = {"a": {"values": [1], "late": True}, "n": {"values": [2]}}
    with pytest.raises(
        space_lib.SpaceError, match=r"Toy/Default.*'n'.*late axes come last"
    ):
        space_lib.load_space(
            write_space(tmp_path, axes=axes, derive={}, constraints=[])
        )


@pytest.mark.parametrize("late", ["yes", 1, None, [True]])
def test_late_must_be_a_bool(tmp_path, late):
    # bug caught: a truthy string or number silently marking an axis late
    axes = {"a": {"values": [1, 2]}, "z": {"values": [1], "late": late}}
    with pytest.raises(
        space_lib.SpaceError, match=r"Toy/Default.*'z'.*'late' must be true or false"
    ):
        space_lib.load_space(
            write_space(tmp_path, axes=axes, derive={}, constraints=[])
        )


# --------------------------------------------------------------------------
# Alignment and placement: the axes, tensor_patches and what the driver does with them
# --------------------------------------------------------------------------


def _device_tensor_uids(template):
    """The real device tensors of a template graph: not virtual, not pass-by-value, not a
    host scalar (a tensor that carries a ``value``)."""
    return sorted(
        t["uid"]
        for t in template["tensors"]
        if not t.get("virtual")
        and not t.get("is_runtime_pass_by_value")
        and "value" not in t
    )


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_every_alignment_and_placement_renders_the_placement_and_the_patches(space_id):
    # bug caught: a patch on a scalar or virtual tensor (the harness would reject or ignore
    # it), a missing operand, a placement_uid on another tensor, or an `under` at alignment 1
    # (nothing can be misaligned below 1, so the case could never pass)
    sp = space_by_id(space_id)
    template = json.loads(sp.template_path.read_text())
    uid_of = {t["name"]: t["uid"] for t in template["tensors"]}
    devices = _device_tensor_uids(template)
    sample = sampler.sample(sp, 11)
    for alignment in ALIGNMENTS:
        for placement in PLACEMENTS:
            axes = dict(sample.axis_values, alignment=alignment, placement=placement)
            case = sp.render(sp.derive_env(axes))
            meta = case["metadata"]
            effective = "exact" if placement == "under" and alignment < 2 else placement
            assert meta["placement"] == effective, (alignment, placement)
            assert not (meta["placement"] == "under" and alignment < 2)
            assert meta["placement_uid"] == uid_of[PLACEMENT_OPERAND[space_id]]
            patches = case["tensor_patches"]
            assert sorted(p["uid"] for p in patches) == devices
            assert all(p["set"] == {"alignment": alignment} for p in patches)
            assert all(type(p["uid"]) is int for p in patches)
            if meta["placement"] == "under":
                # the misaligned tensor declares an alignment it can be placed below
                assert meta["placement_uid"] in devices and alignment >= 2


def test_under_is_drawn_at_every_alignment_but_one_and_only_exact_replaces_it():
    # bug caught: an `under` row at alignment 1, or an exact that is lost at alignment >= 2
    for space_id in SPACE_IDS:
        sp = space_by_id(space_id)
        seen = set()
        for seed in range(400):
            s = sampler.sample(sp, seed)
            a, p = s.axis_values["alignment"], s.axis_values["placement"]
            seen.add((a, p, s.row["metadata"]["placement"]))
        assert (1, "under", "exact") in seen and (1, "under", "under") not in seen
        assert (2, "under", "under") in seen and (256, "under", "under") in seen
        assert (16, "exact", "exact") in seen and (16, "default", "default") in seen


def _toy_patch_space(tmp_path, patches):
    return space_lib.load_space(
        write_space(
            tmp_path,
            case={
                "values": {"a": "${a}", "n": "${n}"},
                "metadata": {"purpose": "probe"},
                "tensor_patches": patches,
            },
        )
    )


def test_a_toy_space_renders_tensor_patches_into_the_row_and_one_without_has_none(
    tmp_path,
):
    # bug caught: patches dropped between render and the row, or a key added to every row
    sp = _toy_patch_space(tmp_path, [{"uid": 3, "set": {"alignment": "${a * 2}"}}])
    s = sampler.sample(sp, 5)
    assert s.row["tensor_patches"] == [
        {"uid": 3, "set": {"alignment": s.axis_values["a"] * 2}}
    ]
    assert list(s.row)[:3] == ["id", "values", "metadata"]
    plain = space_lib.load_space(write_space(tmp_path))
    assert "tensor_patches" not in sampler.sample(plain, 5).row


@pytest.mark.parametrize(
    "patches, message",
    [
        ({"uid": 1}, "must be a list"),
        ([[1]], "must be an object with 'uid' and 'set'"),
        ([{"uid": 1}], "must be an object with 'uid' and 'set'"),
        (
            [{"uid": 1, "set": {"alignment": 4}, "x": 1}],
            "must be an object with 'uid' and 'set'",
        ),
        ([{"uid": "3", "set": {"alignment": 4}}], r"uid' must be an integer"),
        ([{"uid": True, "set": {"alignment": 4}}], r"uid' must be an integer"),
        ([{"uid": 1, "set": {"alignment": 4, "dims": 2}}], "only key is 'alignment'"),
        ([{"uid": 1, "set": {"data_type": "half"}}], "only key is 'alignment'"),
        ([{"uid": 1, "set": {"alignment": "x"}}], r"alignment' must be an integer"),
        ([{"uid": 1, "set": {"alignment": 1.5}}], r"alignment' must be an integer"),
    ],
)
def test_tensor_patches_declarations_are_validated(tmp_path, patches, message):
    # bug caught: a patch that sets anything but the alignment, or with a uid the harness
    # would reject, loads and fails every case at run time instead of at load
    with pytest.raises(space_lib.SpaceError, match=f"Toy/Default.*{message}"):
        _toy_patch_space(tmp_path, patches)


@pytest.mark.parametrize("alignment", ["0", "-4", "a - a", "a / 2 + 0.5", "True"])
def test_a_rendered_alignment_below_one_or_not_an_integer_is_an_error(
    tmp_path, alignment
):
    # bug caught: alignment 0 or a float reaches the harness, which has no pointer to place
    sp = _toy_patch_space(
        tmp_path, [{"uid": 3, "set": {"alignment": "${%s}" % alignment}}]
    )
    with pytest.raises(
        space_lib.SpaceError, match="Toy/Default.*rendered case.tensor_patches"
    ):
        sampler.sample(sp, 5)


def test_a_rendered_uid_that_is_not_an_integer_is_an_error(tmp_path):
    sp = _toy_patch_space(tmp_path, [{"uid": "${'u'}", "set": {"alignment": 4}}])
    with pytest.raises(space_lib.SpaceError, match="rendered case.tensor_patches"):
        sampler.sample(sp, 5)


def _toy_entries_space(tmp_path, **fields):
    return space_lib.load_space(
        write_space(
            tmp_path,
            case={
                "values": {"a": "${a}", "n": "${n}"},
                "metadata": {"purpose": "probe"},
                **fields,
            },
        )
    )


TOY_STATS = {"uid": 4, "name": "Stats", "dims": [1, "${n}"], "alignment": "${a * 2}"}
TOY_WIRE = {"node": "N", "section": "outputs", "set": {"stats_tensor_uid": 4}}


def test_a_toy_space_renders_add_tensors_and_node_patches_into_the_row(tmp_path):
    # bug caught: the new case keys dropped between render and the row, or added to rows of a
    # space that does not declare them, or a ${...} left in a sweep value the harness reads
    sp = _toy_entries_space(
        tmp_path,
        add_tensors=[TOY_STATS],
        node_patches=[
            TOY_WIRE,
            {
                "node": 0,
                "section": "attributes",
                "set": {"scale": "${1 / (a * 2)}", "on": "${a >= 2}"},
                "remove": ["old"],
            },
        ],
    )
    s = sampler.sample(sp, 5)
    a, n = s.axis_values["a"], s.axis_values["n"]
    assert s.row["add_tensors"] == [
        {"uid": 4, "name": "Stats", "dims": [1, n], "alignment": a * 2}
    ]
    assert s.row["node_patches"] == [
        TOY_WIRE,
        {
            "node": 0,
            "section": "attributes",
            "set": {"scale": 1 / (a * 2), "on": a >= 2},
            "remove": ["old"],
        },
    ]
    assert list(s.row) == ["id", "values", "metadata", "add_tensors", "node_patches"]
    assert "${" not in json.dumps(s.row)
    plain = space_lib.load_space(write_space(tmp_path))
    row = sampler.sample(plain, 5).row
    assert "add_tensors" not in row and "node_patches" not in row
    # a space's own tensor_patches follow them
    both = _toy_entries_space(
        tmp_path,
        add_tensors=[TOY_STATS],
        tensor_patches=[{"uid": 4, "set": {"alignment": 4}}],
    )
    assert list(sampler.sample(both, 5).row)[3:] == ["add_tensors", "tensor_patches"]


def test_when_keeps_or_drops_an_entry_and_never_reaches_the_row(tmp_path):
    # bug caught: no way to switch a feature off per sample, or a "when" key in the sweep row
    sp = _toy_entries_space(
        tmp_path,
        add_tensors=[{**TOY_STATS, "when": "${a >= 2}"}],
        node_patches=[{**TOY_WIRE, "when": "${a >= 2}"}, {**TOY_WIRE, "node": "M"}],
    )
    kept = dropped = 0
    for seed in range(40):
        s = sampler.sample(sp, seed)
        on = s.axis_values["a"] >= 2
        assert [t["uid"] for t in s.row["add_tensors"]] == ([4] if on else [])
        assert [p["node"] for p in s.row["node_patches"]] == (
            ["N", "M"] if on else ["M"]
        )
        assert "when" not in json.dumps(s.row)
        kept, dropped = kept + on, dropped + (not on)
    assert kept and dropped


@pytest.mark.parametrize(
    "key, entries, message",
    [
        ("add_tensors", {"uid": 4}, "'case.add_tensors' must be a list"),
        ("add_tensors", [7], r"add_tensors\[0\]' must be an object"),
        ("add_tensors", [{"name": "x"}], r"add_tensors\[0\].uid' must be an integer"),
        ("add_tensors", [{"uid": "4"}], r"add_tensors\[0\].uid' must be an integer"),
        ("add_tensors", [{"uid": True}], r"add_tensors\[0\].uid' must be an integer"),
        (
            "add_tensors",
            [{"uid": 4, "when": True}],
            r"when' must be a \$\{expression\}",
        ),
        ("add_tensors", [{"uid": "${a}", "name": "${nope}"}], "unknown name 'nope'"),
        ("node_patches", "x", "'case.node_patches' must be a list"),
        ("node_patches", [[1]], r"node_patches\[0\]' must be an object"),
        (
            "node_patches",
            [{**TOY_WIRE, "extra": 1}],
            r"node_patches\[0\]' has unknown keys \['extra'\]",
        ),
        (
            "node_patches",
            [{"section": "outputs", "set": {}}],
            r"node_patches\[0\]' needs 'node'",
        ),
        (
            "node_patches",
            [{"node": "N", "set": {}}],
            r"node_patches\[0\]' needs 'section'",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "node": ""}],
            r"node_patches\[0\].node' must be",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "node": 1.5}],
            r"node_patches\[0\].node' must be",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "node": True}],
            r"node_patches\[0\].node' must be",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "section": "attribute"}],
            r"node_patches\[0\].section' must be one of attributes, parameters, inputs, outputs",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "section": "${a}"}],
            r"node_patches\[0\].section' must be one of",
        ),
        (
            "node_patches",
            [{"node": "N", "section": "outputs"}],
            r"node_patches\[0\]' needs 'set' or 'remove'",
        ),
        (
            "node_patches",
            [{**TOY_WIRE, "set": [1]}],
            r"node_patches\[0\].set' must be an object",
        ),
        (
            "node_patches",
            [{"node": "N", "section": "outputs", "remove": "k"}],
            r"node_patches\[0\].remove' must be a list of key names",
        ),
        (
            "node_patches",
            [{"node": "N", "section": "outputs", "remove": [1]}],
            r"node_patches\[0\].remove' must be a list of key names",
        ),
    ],
)
def test_add_tensors_and_node_patches_declarations_are_validated(
    tmp_path, key, entries, message
):
    # bug caught: a malformed entry loads, and every sample then fails at run time as a bare
    # load failure with no detail, or reaches the harness as the typo it is
    with pytest.raises(space_lib.SpaceError, match=f"Toy/Default.*{message}"):
        _toy_entries_space(tmp_path, **{key: entries})


@pytest.mark.parametrize(
    "key, entry, message",
    [
        ("add_tensors", {"uid": "${'u'}"}, "the uid must be an integer"),
        ("add_tensors", {"uid": "${a / 2}"}, "the uid must be an integer"),
        ("add_tensors", {"uid": 4, "name": "x${a}"}, "still holds a '\\$\\{'"),
        (
            "add_tensors",
            {"uid": 4, "dims": ["${a}", "$${a}"]},
            "dims\\[1\\] still holds",
        ),
        ("node_patches", {**TOY_WIRE, "node": "${None}"}, "must be a node name"),
        ("node_patches", {**TOY_WIRE, "node": "${a / 2}"}, "must be a node name"),
        (
            "node_patches",
            {**TOY_WIRE, "set": {"k": "v${a}"}},
            "set.k still holds a '\\$\\{'",
        ),
    ],
)
def test_a_rendered_entry_that_is_not_literal_sweep_json_is_an_error(
    tmp_path, key, entry, message
):
    # bug caught: a half-rendered ${...}, a float uid or a missing node name reaches sweep.json
    sp = _toy_entries_space(tmp_path, **{key: [entry]})
    with pytest.raises(space_lib.SpaceError, match=f"Toy/Default.*rendered.*{message}"):
        sampler.sample(sp, 5)


def test_a_space_with_a_stats_axis_builds_the_graph_with_stats_only_when_on(tmp_path):
    # bug caught: the sample's stats value and the graph the harness builds disagree: an
    # added tensor nothing names, or a node that names a tensor that is not there
    shared = json.loads(
        (TESTS_ROOT / "tests/data/sweep_patch_parity.json").read_text(encoding="utf-8")
    )
    (tmp_path / "toy.template.json").write_text(json.dumps(shared["template"]))
    plain = next(e for e in shared["expands"] if e["case"]["id"] == "plain")["case"]
    stats_tensor = next(e for e in shared["expands"] if e["case"]["id"] == "stats_on")[
        "case"
    ]["add_tensors"][0]
    sp = space_lib.load_space(
        write_space(
            tmp_path,
            file_name="SdpaToy.space.json",
            operation="SdpaToy",
            axes={"stats": {"values": ["off", "on"]}},
            derive={},
            constraints=[],
            case={
                "values": plain["values"],
                "metadata": {"purpose": "probe"},
                "add_tensors": [{**stats_tensor, "when": "${stats == 'on'}"}],
                "node_patches": [
                    {
                        "node": "SdpaFwd_0",
                        "section": "attributes",
                        "set": {"generate_stats": "${stats == 'on'}"},
                    },
                    {
                        "when": "${stats == 'on'}",
                        "node": "SdpaFwd_0",
                        "section": "outputs",
                        "set": {"stats_tensor_uid": 4},
                    },
                ],
            },
        )
    )
    support = fuzz.load_support_matrix_units()
    seen = {}
    for seed in range(20):
        s = sampler.sample(sp, seed)
        graph = bu.expand(shared["template"], s.row["values"])
        support.apply_case_patches(graph, s.row)
        node = graph["nodes"][0]
        on = s.axis_values["stats"] == "on"
        assert node["attributes"]["generate_stats"] is on
        assert node["outputs"]["stats_tensor_uid"] == (4 if on else None)
        assert [t["uid"] for t in graph["tensors"]] == (
            [0, 1, 2, 3, 4] if on else [0, 1, 2, 3]
        )
        seen[on] = True
    assert seen == {True: True, False: True}


def test_the_driver_writes_the_new_case_keys_into_sweep_json(run, tmp_path):
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(
        spaces,
        case={
            "values": {"a": "${a}", "n": "${n}"},
            "metadata": {"purpose": "probe"},
            "add_tensors": [{**TOY_STATS, "when": "${a >= 2}"}],
            "node_patches": [TOY_WIRE],
        },
    )
    assert run.run("--samples", "6", "--spaces-dir", str(spaces), space=None) == 0
    sweep = json.loads(
        (run.work / "bundles/gauntlet/Toy/Default/sweep.json").read_text()
    )
    assert len(sweep["cases"]) == 6
    for row in sweep["cases"]:
        assert row["node_patches"] == [TOY_WIRE]
        assert "when" not in json.dumps(row) and "${" not in json.dumps(row)
    assert {len(row["add_tensors"]) for row in sweep["cases"]} == {0, 1}
    assert run.summary["outcomes"]["PASS"] == 6


def test_a_failure_folder_holds_the_graph_with_added_tensors_and_node_patches(
    run, tmp_path
):
    # bug caught: the fault folder's graph is the template expansion without the case's
    # patches, so freezing it records a different test than the one that failed
    shared = json.loads(
        (TESTS_ROOT / "tests/data/sweep_patch_parity.json").read_text(encoding="utf-8")
    )
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    stats_on = next(e for e in shared["expands"] if e["case"]["id"] == "stats_on")[
        "case"
    ]
    write_space(
        spaces,
        file_name="SdpaToy.space.json",
        axes={"a": {"values": [1]}},
        derive={},
        constraints=[],
        case={
            "values": stats_on["values"],
            "metadata": {"purpose": "probe"},
            "add_tensors": stats_on["add_tensors"],
            "node_patches": stats_on["node_patches"],
        },
    )
    (spaces / "toy.template.json").write_text(json.dumps(shared["template"]))
    run.control(default="wrong")
    assert run.run("--samples", "1", "--spaces-dir", str(spaces), space=None) == 1
    folder = run.failure_dir("s1000", space="SdpaToy_Default")
    graph = json.loads((folder / "SdpaToy_Default_s1000.json").read_text())
    expected = next(e for e in shared["expands"] if e["case"]["id"] == "stats_on")[
        "graph"
    ]
    # the folder holds the canonical-uid form of the graph the harness runs
    assert [t["name"] for t in graph["tensors"]] == sorted(
        t["name"] for t in expected["tensors"]
    )
    node = graph["nodes"][0]
    names = {t["uid"]: t["name"] for t in graph["tensors"]}
    assert names[node["outputs"]["stats_tensor_uid"]] == "SdpaFwd_0::Stats"
    assert node["attributes"]["generate_stats"] is True
    assert json.loads((folder / "repro.json").read_text())["sweep_row"] == json.loads(
        json.dumps(
            sampler.sample(
                space_lib.load_space(spaces / "SdpaToy.space.json"), 1000
            ).row
        )
    )


def test_remap_meta_inputs_remaps_placement_uid_with_the_inputs():
    # bug caught: after UIDs are canonicalized the misaligned pointer would belong to
    # another tensor than the one the case named
    meta = {
        "placement": "under",
        "placement_uid": 7,
        "inputs": {"7": {"kind": "zeros"}},
    }
    out = bu.remap_meta_inputs(meta, {7: 2})
    assert out["placement_uid"] == 2 and out["inputs"] == {"2": {"kind": "zeros"}}
    assert meta["placement_uid"] == 7  # a copy
    assert bu.remap_meta_inputs(meta, {})["placement_uid"] == 7  # unmapped stays
    assert "placement_uid" not in bu.remap_meta_inputs({"purpose": "probe"}, {7: 2})


def test_case_identity_of_a_case_without_patches_or_placement_is_unchanged():
    # bug caught: new identity fields that rehash, and so rename, every existing case id
    plain = {
        "values": {"a": 1},
        "metadata": {"inputs": {"1": {"kind": "free"}}, "seed": 3},
    }
    assert bu._case_identity(plain) == {
        "values": {"a": 1}, "inputs": {"1": {"kind": "free"}}, "seed": 3,
    }  # fmt: skip
    patched = dict(plain, tensor_patches=[{"uid": 1, "set": {"alignment": 2}}])
    placed = {**plain, "metadata": {**plain["metadata"], "placement": "exact"}}
    assert len({bu._case_hash(e) for e in (plain, patched, placed)}) == 3
    # the tensor_patches shipped sweeps carry (a scalar made pass-by-value) are not identity
    scalar = dict(
        plain,
        tensor_patches=[
            {"uid": 1, "set": {"is_runtime_pass_by_value": True}, "remove": ["value"]}
        ],
    )
    assert bu._case_identity(scalar) == bu._case_identity(plain)
    assert bu._case_hash(scalar) == bu._case_hash(plain)


def test_every_shipped_case_id_is_unchanged(monkeypatch):
    # bug caught: an identity field that rehashes cases which already ship, so regenerating
    # a sweep renames them (and every golden pointer, claim and TOML filter keyed on the id)
    def legacy_identity(entry):
        meta = entry.get("metadata", {})
        return {
            "values": entry.get("values", {}),
            "inputs": meta.get("inputs", {}),
            "seed": meta.get("seed"),
        }

    checked = with_patches = reproduced = 0
    for path in sorted(BUNDLES.rglob("sweep.json")):
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
        if not cases:
            continue
        current = copy.deepcopy(cases)
        bu.assign_case_ids(current)
        with monkeypatch.context() as patch:
            patch.setattr(bu, "_case_identity", legacy_identity)
            before = copy.deepcopy(cases)
            bu.assign_case_ids(before)
        assert [c["id"] for c in current] == [c["id"] for c in before], path
        checked += len(cases)
        with_patches += sum(1 for c in cases if c.get("tensor_patches"))
        reproduced += sum(1 for c, d in zip(cases, current) if c["id"] == d["id"])
    assert checked > 10000 and with_patches > 0, (checked, with_patches)
    print(f"shipped cases checked {checked}, with tensor_patches {with_patches}, "
          f"ids reproduced by re-running assign_case_ids {reproduced}")  # fmt: skip


@pytest.mark.parametrize("alignment", [2, 256, 16])
def test_import_graph_keeps_a_declared_alignment_as_tensor_patches(tmp_path, alignment):
    # bug caught: a graph at alignment N != the template's 16 fails the import round trip,
    # or is imported without its patches and then runs as a different test
    sp = space_by_id("SdpaFwd/Default")
    sample = sampler.sample(sp, 21)
    row = sp.render(sp.derive_env(dict(sample.axis_values, alignment=alignment)))
    row["id"] = sample.case_id
    template = json.loads(sp.template_path.read_text())
    graph = expanded_graph(template, row)
    graph_path = tmp_path / "case.json"
    graph_path.write_text(json.dumps(graph))

    tree = tmp_path / "tree"
    topology = tree / "quick" / "SdpaFwd" / sp.template_path.parent.name
    topology.mkdir(parents=True)
    for name in ("graph.template.json", "sweep.json", "support.json"):
        shutil.copyfile(sp.template_path.parent / name, topology / name)
    before = len(json.loads((topology / "sweep.json").read_text())["cases"])

    def import_it(path=graph_path):
        return subprocess.run(
            [sys.executable, str(MIGRATION_DIR / "import_graph.py"), "--graph", str(path),
             "--bundle-dir", str(tree), "--tier", "quick"],
            capture_output=True, text=True, cwd=MIGRATION_DIR, timeout=120,
        )  # fmt: skip

    done = import_it()
    assert done.returncode == 0, done.stderr
    cases = json.loads((topology / "sweep.json").read_text())["cases"]
    assert len(cases) == before + 1
    new = cases[-1]
    expected = [] if alignment == 16 else [
        {"uid": uid, "set": {"alignment": alignment}} for uid in _device_tensor_uids(template)
    ]  # fmt: skip
    assert new.get("tensor_patches", []) == expected
    assert ("tensor_patches" in new) == (alignment != 16)
    # the case expands, patched, to the graph that was imported
    imported_template = json.loads((topology / "graph.template.json").read_text())
    back = bu.expand(imported_template, new["values"])
    fuzz.load_support_matrix_units().apply_case_patches(back, new)
    assert bu.canon(back) == bu.canon(graph)
    # importing it again is a duplicate, even though the expansion needs the patches
    assert "DUPLICATE" in import_it().stderr
    assert len(json.loads((topology / "sweep.json").read_text())["cases"]) == before + 1


def test_import_graph_gives_two_alignments_of_one_graph_two_case_ids(tmp_path):
    # bug caught: the case id hash ignores the patches, so the second import of the same
    # shape at another alignment collides with the first and is refused or overwrites it
    sp = space_by_id("SdpaFwd/Default")
    sample = sampler.sample(sp, 21)
    template = json.loads(sp.template_path.read_text())
    tree = tmp_path / "tree"
    topology = tree / "quick" / "SdpaFwd" / sp.template_path.parent.name
    topology.mkdir(parents=True)
    for name in ("graph.template.json", "sweep.json", "support.json"):
        shutil.copyfile(sp.template_path.parent / name, topology / name)
    for alignment in (2, 4):
        row = sp.render(sp.derive_env(dict(sample.axis_values, alignment=alignment)))
        path = tmp_path / f"case{alignment}.json"
        path.write_text(json.dumps(expanded_graph(template, row)))
        done = subprocess.run(
            [sys.executable, str(MIGRATION_DIR / "import_graph.py"), "--graph", str(path),
             "--bundle-dir", str(tree), "--tier", "quick"],
            capture_output=True, text=True, cwd=MIGRATION_DIR, timeout=120,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
    cases = json.loads((topology / "sweep.json").read_text())["cases"]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert [c["tensor_patches"][0]["set"]["alignment"] for c in cases[-2:]] == [2, 4]


def test_summary_breaks_the_outcomes_down_by_every_values_axis(run):
    # bug caught: a run that cannot say which feature (alignment, placement, domain, dtype,
    # layout, ...) fails or never passes, so per-feature PASS counts need the log
    run.control(seed_mod={"m": 3, "map": {"0": "wrong", "1": "declined"}})
    assert (
        run.run("--samples", "30", "--batch-size", "30", space="RMSNorm/Default") == 1
    )
    sp = space_by_id("RMSNorm/Default")
    values_axes = [name for name, axis in sp.axes.items() if not axis.is_range]
    assert {"alignment", "placement", "domain", "dtype", "layout"} <= set(values_axes)
    assert len(values_axes) < len(sp.axes)
    expected = {}
    for seed in range(1000, 1030):
        axes = sampler.sample(sp, seed).axis_values
        outcome = {0: "WRONG", 1: "DECLINED"}.get(seed % 3, "PASS")
        for name in values_axes:
            by_value = expected.setdefault(name, {})
            by_value.setdefault(str(axes[name]), dict.fromkeys(fuzz.OUTCOMES, 0))[
                outcome
            ] += 1
    assert run.summary["axis_outcomes"] == {"RMSNorm/Default": expected}
    for name, by_value in expected.items():
        assert sum(sum(c.values()) for c in by_value.values()) == 30, name


def test_summary_counts_a_values_axis_per_value_and_skips_range_axes(run, tmp_path):
    # bug caught: an axis the space does not have, or a range axis (one entry per drawn
    # integer), shows up in the breakdown
    assert run.run("--samples", "4", space="Matmul/Align") == 0
    assert sorted(run.summary["axis_outcomes"]["Matmul/Align"]) == [
        "alignment",
        "placement",
    ]
    spaces = tmp_path / "toy"
    spaces.mkdir()
    write_space(spaces)
    assert run.run("--samples", "2", "--spaces-dir", str(spaces), space=None) == 0
    toy = run.summary["axis_outcomes"]
    assert [sorted(by_axis) for by_axis in toy.values()] == [["a"]]
    counts = next(iter(toy.values()))["a"]
    assert sum(sum(c.values()) for c in counts.values()) == 2


KERNEL_RESIDUES = {
    "0": "pass",
    "1": "declined",
    "2": "wrong",
    "3": "unexercised",
    "4": "crash_engine",
    "5": "crash_execute",
    "6": "crash",
}
KERNEL_RESIDUE_COUNT = 7
KERNEL_SAMPLES = 14


def test_summary_counts_the_kernels_each_engine_ran(run):
    # bug caught: a record that names its kernels losing them on the way to the summary, a case
    # counted under another case's kernel, or a kernel counted more than once for one case
    run.control(
        kernels_seed_mod={
            "m": 2,
            "map": {"0": ["k_even"], "1": ["k_odd", "k_both"]},
        },
    )
    assert run.run("--samples", "10", "--batch-size", "10") == 0
    seeds = range(1000, 1010)
    even = sum(1 for s in seeds if s % 2 == 0)
    engine = run.summary["engines"]["STUB_ENGINE"]
    assert engine["kernels"] == {
        "k_even": even,
        "k_odd": 10 - even,
        "k_both": 10 - even,
    }
    assert engine["kernel_unreported"] == 0
    for record in run.outcome_list:
        seed = int(record["case"].removeprefix("s"))
        assert record["kernels"] == (
            ["k_even"] if seed % 2 == 0 else ["k_odd", "k_both"]
        )


def test_a_case_that_reports_no_kernel_is_unreported_only_when_it_reached_execute(run):
    # bug caught: a decline, a setup failure or a plan-build crash counted as "executed without
    # a kernel line" (they never had a kernel to report), or a driver-made hang/death counted
    # as an engine that is silent; and the other way, an engine that ran and reports nothing
    # (HIP_MLOPS) not showing up
    run.control(seed_mod={"m": KERNEL_RESIDUE_COUNT, "map": KERNEL_RESIDUES})
    assert run.run("--samples", str(KERNEL_SAMPLES), "--batch-size", "7") == 1
    seeds = range(1000, 1000 + KERNEL_SAMPLES)
    ran = sum(1 for s in seeds if s % KERNEL_RESIDUE_COUNT in (0, 2, 5))
    engine = run.summary["engines"]["STUB_ENGINE"]
    assert engine["kernel_unreported"] == ran == 6
    assert engine["kernels"] == {}
    assert engine["samples"] == KERNEL_SAMPLES
    by_outcome = {}
    driver_made = 0
    for record in run.outcome_list:
        assert "kernels" not in record
        if record["source"] == "driver":
            driver_made += 1
            continue
        key = (record["outcome"], record["stage"])
        by_outcome[key] = by_outcome.get(key, 0) + 1
    assert by_outcome[("CRASH", "execute")] == 2
    assert by_outcome[("CRASH", "build")] == 2
    # the two process deaths are CRASH results the driver wrote without a finished record: they
    # carry the execute stage and still do not count as engines that ran silently
    assert driver_made == 2


def test_only_cases_that_built_a_plan_name_kernels(run):
    # bug caught: kernels credited to a decline, a setup failure or a plan-build crash, or
    # kernels and unreported counting different sets of cases
    run.control(
        seed_mod={"m": KERNEL_RESIDUE_COUNT, "map": KERNEL_RESIDUES},
        kernels=["k_main"],
    )
    assert run.run("--samples", str(KERNEL_SAMPLES), "--batch-size", "7") == 1
    engine = run.summary["engines"]["STUB_ENGINE"]
    assert engine["kernels"] == {"k_main": 6}
    assert engine["kernel_unreported"] == 0
    with_kernels = [r for r in run.outcome_list if "kernels" in r]
    assert {r["outcome"] for r in with_kernels} == {"PASS", "WRONG", "CRASH"}
    assert all(r["stage"] in ("compare", "execute") for r in with_kernels)


def test_a_failure_folder_carries_the_alignment_placement_and_patches(run):
    # bug caught: the graph of a failure folder at the template's alignment 16 although the
    # case ran at another one, or a meta sidecar without the placement that made it fail
    run.control(default="wrong")
    assert run.run("--samples", "1", "--seed", "4242", space="Matmul/Align") == 1
    sp = space_by_id("Matmul/Align")
    sample = sampler.sample(sp, 4242)
    folder = run.failure_dir("s4242", "Matmul_Align")
    graph = json.loads((folder / "Matmul_Align_s4242.json").read_text())
    assert {t["uid"]: t["alignment"] for t in graph["tensors"]} == {
        p["uid"]: p["set"]["alignment"] for p in sample.row["tensor_patches"]
    }
    meta = json.loads((folder / "Matmul_Align_s4242.meta.json").read_text())
    assert (meta["placement"], meta["placement_uid"]) == (
        sample.row["metadata"]["placement"], sample.row["metadata"]["placement_uid"],
    )  # fmt: skip
    repro = json.loads((folder / "repro.json").read_text())
    assert repro["sweep_row"] == sample.row


# --------------------------------------------------------------------------
# Driver fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("HIPDNN_FUZZ_") or name in (
            "HIPDNN_STUB_CONTROL",
            fuzz.HOST_FILL_ENV,
        ):
            monkeypatch.delenv(name)


class Run:
    """One driver run against the stub, with helpers to read what it left behind."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.work = tmp_path / "work"
        self.log = tmp_path / "stub-invocations.jsonl"
        self.control_path = tmp_path / "control.json"
        self.control()

    def control(self, **control):
        control.setdefault("log", str(self.log))
        self.control_path.write_text(json.dumps(control))
        self.monkeypatch.setenv("HIPDNN_STUB_CONTROL", str(self.control_path))

    def args(self, *extra, space="SdpaFwd/Default", seed=1000):
        args = [
            "--binary", str(STUB),
            "--work-dir", str(self.work),
            "--case-timeout", "30s",
            "--seed", str(seed),
        ]  # fmt: skip
        if space:
            args += ["--filter", space if space.startswith("^") else f"^{space}$"]
        return args + list(extra)

    def run(self, *extra, **kwargs):
        return fuzz.main(self.args(*extra, **kwargs))

    @property
    def summary(self):
        return json.loads((self.work / "summary.json").read_text())

    @property
    def outcome_list(self):
        return [
            json.loads(line)
            for line in (self.work / "outcomes.jsonl").read_text().splitlines()
        ]

    @property
    def invocations(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def failure_dir(self, case, space="SdpaFwd_Default"):
        return self.work / "failures" / f"{space}_{case}"


@pytest.fixture
def run(tmp_path, monkeypatch):
    return Run(tmp_path, monkeypatch)


# --------------------------------------------------------------------------
# R6: control surface
# --------------------------------------------------------------------------


def test_environment_variables_set_the_defaults_and_flags_override_them(monkeypatch):
    env = {
        "HIPDNN_FUZZ_SAMPLES": "7",
        "HIPDNN_FUZZ_SEED": "55",
        "HIPDNN_FUZZ_DATA_SEED": "66",
        "HIPDNN_FUZZ_FILTER": "^RMSNorm/",
        "HIPDNN_FUZZ_TIME_BUDGET": "2h",
        "HIPDNN_FUZZ_CASE_TIMEOUT": "90s",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = fuzz.parse_args(["--binary", "bin"])
    assert (a.samples, a.seed, a.data_seed, a.filter) == (7, 55, 66, "^RMSNorm/")
    assert (a.time_budget, a.case_timeout) == (7200.0, 90.0)
    a = fuzz.parse_args(
        ["--binary", "bin", "--samples", "3", "--seed", "1", "--data-seed", "2", "--filter", "x",
         "--time-budget", "10m", "--case-timeout", "5s"]
    )  # fmt: skip
    assert (a.samples, a.seed, a.data_seed, a.filter) == (3, 1, 2, "x")
    assert (a.time_budget, a.case_timeout) == (600.0, 5.0)


def test_defaults(monkeypatch):
    a = fuzz.parse_args(["--binary", "bin"])
    assert (
        a.samples == 32
        and a.data_seed is None
        and a.filter is None
        and a.time_budget is None
    )
    assert a.case_timeout == 600.0
    assert 0 <= a.seed < 2**31
    # a time budget without --samples means unlimited samples
    assert fuzz.parse_args(["--binary", "bin", "--time-budget", "30m"]).samples == 0
    assert (
        fuzz.parse_args(
            ["--binary", "bin", "--time-budget", "30m", "--samples", "5"]
        ).samples
        == 5
    )
    assert fuzz.parse_args(["--binary", "bin", "--samples", "0"]).samples == 0


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("90", 90.0),
        ("30m", 1800.0),
        ("4h", 14400.0),
        ("1h30m", 5400.0),
        ("0.5s", 0.5),
        ("2d", 172800.0),
        ("0", None),
        ("none", None),
    ],
)
def test_durations(text, seconds):
    assert fuzz.parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["soon", "5x", "m"])
def test_bad_duration_is_an_error(bad):
    with pytest.raises(ValueError):
        fuzz.parse_duration(bad)


@pytest.mark.parametrize(
    "env, value",
    [
        ("HIPDNN_FUZZ_SAMPLES", "many"),
        ("HIPDNN_FUZZ_SEED", "-3"),
        ("HIPDNN_FUZZ_DATA_SEED", str(2**32)),
        ("HIPDNN_FUZZ_TIME_BUDGET", "soon"),
    ],
)
def test_bad_environment_values_are_usage_errors(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(SystemExit) as info:
        fuzz.parse_args(["--binary", "bin"])
    assert info.value.code == 2


def test_seed_is_printed_and_replayable(run, capsys):
    assert (
        fuzz.main(
            [
                "--binary",
                str(STUB),
                "--work-dir",
                str(run.work),
                "--samples",
                "1",
                "--filter",
                "^SdpaFwd/Default$",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    seed = run.summary["seed"]
    assert f"seed {seed}" in out
    assert run.outcome_list[0]["case"] == f"s{seed}"


def test_default_run_covers_every_space_and_filter_narrows(run):
    assert run.run("--samples", "2", space=None) == 0
    assert run.summary["spaces"] == SPACE_IDS
    assert run.summary["samples_by_space"] == {s: 2 for s in SPACE_IDS}
    assert run.run("--samples", "2", space="RMSNorm/Default") == 0
    assert run.summary["spaces"] == ["RMSNorm/Default"]
    assert run.run("--samples", "2", space="SdpaFwd/.*") == 0
    assert run.summary["spaces"] == [
        "SdpaFwd/Default",
        "SdpaFwd/Flash2",
        "SdpaFwd/Flash2Region",
        "SdpaFwd/Gfx1151",
        "SdpaFwd/Gfx1151Decode",
        "SdpaFwd/Gfx950Dense",
    ]


def test_filter_matching_nothing_and_broken_spaces_are_usage_errors(
    run, tmp_path, capsys
):
    assert run.run("--samples", "1", space="NoSuchOp/Default") == 2
    assert "no space" in capsys.readouterr().err
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(spaces, template="missing.template.json")
    assert run.run("--samples", "1", "--spaces-dir", str(spaces), space=None) == 2
    assert "Toy/Default" in capsys.readouterr().err


def test_infeasible_space_ends_the_run_with_an_error(run, tmp_path, capsys):
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(
        spaces, constraints=[{"rule": "n > 1000", "reason": "n is capped at 100"}]
    )
    assert run.run("--samples", "1", "--spaces-dir", str(spaces), space=None) == 2
    err = capsys.readouterr().err
    assert "infeasible" in err and "n is capped at 100" in err


# --------------------------------------------------------------------------
# R6: batches and what the binary is given
# --------------------------------------------------------------------------


def test_batch_is_written_as_a_concrete_sweep_under_the_bundle_root(run):
    assert (
        run.run(
            "--samples",
            "3",
            "--test-article",
            "ART",
            "--test-engine",
            "ENG",
            "--test-config",
            "CFG",
            "--binary-arg=--extra",
        )
        == 0
    )
    sp = space_by_id("SdpaFwd/Default")
    names = [gtest("SdpaFwd/Default", s) for s in (1000, 1001, 1002)]
    directory = sweep_dir(run.work, "SdpaFwd/Default", 1000)
    assert (
        directory / "graph.template.json"
    ).read_bytes() == sp.template_path.read_bytes()
    sweep = json.loads((directory / "sweep.json").read_text())
    assert sweep["version"] == 1
    rows = sweep["cases"]
    assert [r["id"] for r in rows] == ["s1000", "s1001", "s1002"]
    for row, seed in zip(rows, (1000, 1001, 1002)):
        assert json.dumps(row) == json.dumps(sampler.sample(sp, seed).row)
        assert set(row["metadata"]) >= {
            "format_version",
            "generator",
            "seed",
            "purpose",
            "notes",
        }
        assert row["metadata"]["purpose"] == "probe"
    # discoverable the way the harness walks a golden-data-dir
    assert bundle_discovery.find_sweep_roots(run.work / "bundles") == [directory]
    # the binary is called with the documented command line
    modes = {i["mode"]: i["argv"] for i in run.invocations}
    for mode in ("list", "run"):
        argv = modes[mode]
        assert argv[argv.index("--golden-data-dir") + 1] == str(run.work / "bundles")
        assert "--gtest_filter=" + ":".join(names) in argv
        assert argv[argv.index("--test-article") + 1] == "ART"
        assert argv[argv.index("--test-engine") + 1] == "ENG"
        assert argv[argv.index("--test-config") + 1] == "CFG"
        assert "--extra" in argv
    assert (
        "--gtest_list_tests" in modes["list"]
        and "--outcome-report" not in modes["list"]
    )
    assert (
        "--gtest_list_tests" not in modes["run"] and "--outcome-report" in modes["run"]
    )
    assert {o["test"] for o in run.outcome_list} == set(names)


def test_each_space_gets_its_own_batch_and_gtest_names(run):
    run.run("--samples", "1", space=None)
    names = {o["test"] for o in run.outcome_list}
    assert names == {gtest(space_id, 1000) for space_id in SPACE_IDS}


# --------------------------------------------------------------------------
# A11: the engine TOML matches dtype tags, so the suite carries the input dtype
# --------------------------------------------------------------------------

HIP_MLOPS_TOML = (
    TESTS_ROOT.parent / "hip-kernel-provider" / "config" / "HIP_MLOPS_ENGINE.toml"
)


@pytest.mark.parametrize(
    "axes, expected",
    [
        ({"dtype": {"values": ["float"]}}, "fp32"),
        ({"dtype": {"values": ["half"]}}, "fp16"),
        ({"dtype": {"values": ["bfloat16"]}}, "bfp16"),
        # mixed precision: the input's dtype, not the one of the other tensors
        (
            {"dtype": {"values": ["float"]}, "x_dtype": {"values": ["bfloat16"]}},
            "bfp16",
        ),
        ({"dtype": {"values": ["double"]}}, "double"),
        ({}, None),
    ],
)
def test_dtype_tag_is_the_input_dtype_spelled_like_the_bundle_case_ids(
    tmp_path, axes, expected
):
    # bug caught: a tag from the wrong axis, or spelled `float`/`bfloat16`, misses the TOML globs
    base = {
        "a": {"values": [1, 2, 3]},
        "n": {"range": [1, 100], "boundaries": [1, 100]},
    }
    sp = space_lib.load_space(write_space(tmp_path, axes={**base, **axes}))
    assert fuzz.dtype_tag(sp, sampler.sample(sp, 3)) == expected


def test_a_batch_is_split_into_one_sweep_directory_per_input_dtype(run):
    # bug caught: one sweep directory per space gives every case the same suite, so the
    # per-dtype TOML tolerance cannot tell a half case from a bfloat16 one
    space_id = "RMSNorm/Default"
    seeds = range(1000, 1024)
    assert run.run("--samples", "24", "--batch-size", "24", space=space_id) == 0
    expected = {}
    for seed in seeds:
        expected.setdefault(input_dtype_tag(space_id, seed), []).append(f"s{seed}")
    assert len(expected) >= 2, "the seeds must draw more than one input dtype"
    base = run.work / "bundles" / "gauntlet" / "RMSNorm" / "Default"
    directories = sorted(base / f"{tag}_cases" for tag in expected)
    assert sorted(p for p in base.iterdir()) == directories
    template = space_by_id(space_id).template_path.read_bytes()
    for tag, ids in expected.items():
        directory = base / f"{tag}_cases"
        rows = json.loads((directory / "sweep.json").read_text())["cases"]
        assert [r["id"] for r in rows] == ids
        assert (directory / "graph.template.json").read_bytes() == template
    assert (
        sorted(bundle_discovery.find_sweep_roots(run.work / "bundles")) == directories
    )
    # the harness registers suite = sanitized relative path joined by '_'; the stub does the
    # same, and a name the driver got wrong would be absent from its listing: UNEXERCISED
    registered = {
        "_".join(d.relative_to(run.work / "bundles").parts) + "." + case
        for d in directories
        for case in expected[d.name.removesuffix("_cases")]
    }
    assert {o["outcome"] for o in run.outcome_list} == {"PASS"}
    assert {o["test"] for o in run.outcome_list} == registered
    assert registered == {gtest(space_id, seed) for seed in seeds}


def test_a_crash_in_one_dtype_group_costs_one_case_and_the_other_groups_run(run):
    # bug caught: crash recovery that only knows one sweep directory loses the cases
    # that the other dtype directories of the batch still have to run
    space_id = "RMSNorm/Default"
    seeds = range(1000, 1012)
    tags = {s: input_dtype_tag(space_id, s) for s in seeds}
    assert len(set(tags.values())) >= 2
    # the binary runs the sweep directories in path order, so a crash in the first one
    # leaves the cases of the other directories unresolved
    first = min(tags.values())
    crash = next(s for s in seeds if tags[s] == first)
    run.control(cases={f"s{crash}": "crash"})
    assert run.run("--samples", "12", "--batch-size", "12", space=space_id) == 1
    assert {o["case"]: o["outcome"] for o in run.outcome_list} == {
        f"s{s}": "CRASH" if s == crash else "PASS" for s in seeds
    }
    runs = [i["tests"] for i in run.invocations if i["mode"] == "run"]
    assert len(runs) == 2 and len(runs[0]) == 12
    assert gtest(space_id, crash) not in runs[1]
    assert any(f"_{first}_cases." not in name for name in runs[1])
    folder = run.failure_dir(f"s{crash}", "RMSNorm_Default")
    assert json.loads((folder / "repro.json").read_text())["gtest_name"] == gtest(
        space_id, crash
    )


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_suite_tag_is_the_dtype_of_the_graphs_input_tensor(run, space_id):
    # bug caught: a space whose dtype axes are not the ones the driver reads gets a tag
    # that disagrees with its graph, and the TOML then grades a bfloat16 case as fp32
    sp = space_by_id(space_id)
    f = _fuzz(run, space=space_id)
    template = json.loads(sp.template_path.read_text())
    for seed in range(60):
        s = sampler.sample(sp, seed)
        if space_id in FIXED_SHAPE_SPACES:
            # no dtype axis, so no dtype directory: the suite is the plain space suite
            assert f._test_name(sp, s) == f"gauntlet_{sp.operation}_{sp.name}.s{seed}"
            continue
        graph = bu.expand(template, s.row["values"])
        x = next(t for t in graph["tensors"] if t["name"].lower() in ("x", "q"))
        assert (
            f._test_name(sp, s)
            == f"gauntlet_{sp.operation}_{sp.name}_{TAG_OF_DTYPE[x['data_type']]}_cases.s{seed}"
        ), (space_id, seed, s.axis_values)


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_registered_names_carry_no_token_the_name_validator_forbids(run, space_id):
    # bug caught: a suffix such as `Fp16_cases` would fail test_name_validation if the
    # runtime-registered gauntlet names were ever listed there (test_name_validator.py:19-24)
    banned = ("Test", "Integration", "Gpu", "Bfp16", "Fp16", "Fp32", "Fp64")
    sp = space_by_id(space_id)
    f = _fuzz(run, space=space_id)
    for seed in range(30):
        name = f._test_name(sp, sampler.sample(sp, seed))
        assert not [t for t in banned if t in name], name


def _toml_rms_threshold(config, test_name, tensor):
    """The rms threshold the harness resolves: the last [[validator_overrides]] entry whose
    filters match the test name and whose tensors match the label wins (TestSettings.hpp
    findValidatorOverride; both globs are fnmatch-style)."""
    result = None
    for entry in config.get("validator_overrides", []):
        if any(fnmatch.fnmatchcase(test_name, g) for g in entry["filters"]) and any(
            fnmatch.fnmatchcase(tensor, g) for g in entry["tensors"]
        ):
            result = entry.get("rms_threshold")
    return result


def test_the_hip_mlops_dtype_thresholds_reach_layernorm_backward_cases(run):
    # bug caught (A11): the cases are named s<seed>, so `*LayernormBackward*fp16_*` never
    # matched and every fp16 or bfloat16 case was graded at the fp32 rms threshold 1e-4
    tomllib = pytest.importorskip("tomllib")
    if not HIP_MLOPS_TOML.is_file():
        pytest.skip("the hip-kernel-provider config is not in this checkout")
    config = tomllib.loads(HIP_MLOPS_TOML.read_text(encoding="utf-8"))
    space_id = "LayernormBackward/Default"
    sp = space_by_id(space_id)
    f = _fuzz(run, space=space_id)
    template = json.loads(sp.template_path.read_text())
    expected = {"float": 1e-4, "half": 1e-3, "bfloat16": 1e-2}
    seen = {}
    for seed in range(200):
        s = sampler.sample(sp, seed)
        graph = bu.expand(template, s.row["values"])
        x = next(t for t in graph["tensors"] if t["name"] == "X")
        for tensor in ("LayernormBackward_0::DSCALE", "LayernormBackward_0::DBIAS"):
            got = _toml_rms_threshold(config, f._test_name(sp, s), tensor)
            assert got == expected[x["data_type"]], (seed, x["data_type"], got)
        seen[x["data_type"]] = seen.get(x["data_type"], 0) + 1
    assert {"half", "bfloat16"} <= set(seen)


ASM_SDPA_TOML = (
    TESTS_ROOT.parent / "hip-kernel-provider" / "config" / "ASM_SDPA_ENGINE.toml"
)


def _gtest_glob_matches(pattern, text):
    """GTest's glob, as ``globMatches`` in ``src/harness/bundle/GTestFilter.hpp`` has it: ``*``
    matches any run of characters (none included) and ``?`` exactly one; nothing else is special
    (``fnmatch`` would also read ``[...]``)."""
    p = t = 0
    star = star_text = None
    while t < len(text):
        if p < len(pattern) and pattern[p] == "*":
            star, star_text = p, t
            p += 1
        elif p < len(pattern) and pattern[p] in ("?", text[t]):
            p += 1
            t += 1
        elif star is not None:
            star_text += 1
            p, t = star + 1, star_text
        else:
            return False
    while p < len(pattern) and pattern[p] == "*":
        p += 1
    return p == len(pattern)


def _toml_tolerance(config, test_name):
    """The (atol, rtol) the harness resolves for a test name: the last ``[[tolerance_overrides]]``
    entry with a matching filter wins (``TestSettings::findToleranceOverride``); None when none
    matches, which leaves the common suite's tolerance."""
    result = None
    for entry in config.get("tolerance_overrides", []):
        if any(_gtest_glob_matches(g, test_name) for g in entry["filters"]):
            result = (entry["atol"], entry["rtol"])
    return result


def test_the_asm_sdpa_backward_tolerances_reach_fuzz_and_quick_cases(run):
    # bug caught (A4): the filters read `*_SdpaBackward_*.*_fp16_*`, which needs the dtype token
    # in the case id. The fuzz cases are named s<seed> and carry it in the suite, so every fuzz
    # backward sample was graded at the tight forward tolerance instead of the calibrated one
    tomllib = pytest.importorskip("tomllib")
    if not ASM_SDPA_TOML.is_file():
        pytest.skip("the hip-kernel-provider config is not in this checkout")
    config = tomllib.loads(ASM_SDPA_TOML.read_text(encoding="utf-8"))
    space_id = "SdpaBackward/Default"
    sp = space_by_id(space_id)
    f = _fuzz(run, space=space_id)
    expected = {"half": (2.5e-3, 2.5e-3), "bfloat16": (1e-2, 1e-2)}
    seen = {}
    for seed in range(200):
        s = sampler.sample(sp, seed)
        name = f._test_name(sp, s)
        dtype = s.axis_values["dtype"]
        assert _toml_tolerance(config, name) == expected[dtype], (name, dtype)
        seen[dtype] = seen.get(dtype, 0) + 1
    assert set(seen) == set(expected)  # fp16 and bfp16 each drawn, and never crossed

    # the quick bundle cases resolve as before: the dtype token is in the case id
    quick_dir = sp.template_path.parent
    suite = "_".join(quick_dir.relative_to(BUNDLES).parts)
    assert suite == "quick_SdpaBackward_Default"
    quick = json.loads((quick_dir / "sweep.json").read_text())["cases"]
    tokens = set()
    for case in quick:
        io = case["values"]["io_data_type"]
        name = f"{suite}.{re.sub(r'[^A-Za-z0-9_]', '_', case['id'])}"
        assert _toml_tolerance(config, name) == expected[io], name
        tokens.add(TAG_OF_DTYPE[io])
    assert tokens == {"fp16", "bfp16"}

    # no other operation, fuzz or quick, picks up the backward tolerances
    for other in (
        "gauntlet_SdpaFwd_Default_fp16_cases.s7",
        "gauntlet_SdpaFwd_Default_bfp16_cases.s7",
        "quick_SdpaFwd_Default.1_1_256_128_fp16_nchw",
    ):
        assert _toml_tolerance(config, other) is None, other


def test_samples_are_split_into_batches_with_consecutive_seeds(run):
    assert run.run("--samples", "7", "--batch-size", "3") == 0
    runs = [i["tests"] for i in run.invocations if i["mode"] == "run"]
    assert [len(r) for r in runs] == [3, 3, 1]
    assert [o["case"] for o in run.outcome_list] == [f"s{1000 + i}" for i in range(7)]


def test_fixed_data_seed_applies_to_every_sample(run):
    run.run("--samples", "3", "--data-seed", "4021")
    rows = json.loads(
        (sweep_dir(run.work, "SdpaFwd/Default", 1000) / "sweep.json").read_text()
    )["cases"]
    assert {r["metadata"]["seed"] for r in rows} == {4021}
    assert [r["id"] for r in rows] == ["s1000", "s1001", "s1002"]


# --------------------------------------------------------------------------
# R6: outcomes, exit code, summary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "behaviour, outcome, exit_code",
    [
        ("pass", "PASS", 0),
        ("unclaimed_pass", "UNCLAIMED_PASS", 0),
        ("declined", "DECLINED", 0),
        ("unexercised", "UNEXERCISED", 0),
        ("wrong", "WRONG", 1),
        ("crash_engine", "CRASH", 1),
        ("crash", "CRASH", 1),
    ],
)
def test_exit_status_is_nonzero_only_for_wrong_and_crash(
    run, behaviour, outcome, exit_code
):
    run.control(default=behaviour)
    assert run.run("--samples", "2") == exit_code
    assert [o["outcome"] for o in run.outcome_list] == [outcome, outcome]
    assert run.summary["exit_code"] == exit_code
    assert run.summary["outcomes"][outcome] == 2


def test_summary_counts_every_sample_per_engine_and_space_with_reasons(run):
    run.control(
        seed_mod={
            "m": 6,
            "map": {
                "1": "declined",
                "2": "wrong",
                "3": "unexercised",
                "4": "load_skip",
                "5": "unclaimed_pass",
            },
        },
        engine="STUB_ENGINE",
    )
    assert run.run("--samples", "12", "--batch-size", "5", space=None) == 1
    summary = run.summary
    spaces = len(SPACE_IDS)
    total = 12 * spaces  # twelve samples in each space; every residue of seed % 6 twice
    assert summary["samples"] == total and summary["reconciled"] is True
    assert sum(summary["outcomes"].values()) == total
    assert summary["outcomes"] == {
        "PASS": 2 * spaces,
        "DECLINED": 2 * spaces,
        "WRONG": 2 * spaces,
        "CRASH": 0,
        "UNCLAIMED_PASS": 2 * spaces,
        "UNEXERCISED": 4 * spaces,
    }
    engine = summary["engines"]["STUB_ENGINE"]
    assert engine["samples"] == total and set(engine["spaces"]) == set(SPACE_IDS)
    for space_id in SPACE_IDS:
        sp = engine["spaces"][space_id]
        assert sp["samples"] == 12 and sum(sp["outcomes"].values()) == 12
        assert sp["outcomes"]["PASS"] == 2 and sp["outcomes"]["WRONG"] == 2
        assert list(sp["declined_reasons"].values()) == [2]
        assert "does not support this graph" in next(iter(sp["declined_reasons"]))
        assert sorted(sp["unexercised_reasons"].values()) == [2, 2]
        assert any(r.startswith("load:") for r in sp["unexercised_reasons"])
        assert any(r == "no oracle available" for r in sp["unexercised_reasons"])
        assert len(sp["wrong"]) == 2
    assert len(summary["findings"]) == 2 * spaces
    assert len(run.outcome_list) == total
    assert (run.work / "samples.jsonl").read_text().count("\n") == total


def test_summary_path_flag_and_outcome_records(run):
    target = run.tmp / "elsewhere" / "s.json"
    target.parent.mkdir()
    run.run("--samples", "2", "--summary", str(target))
    assert json.loads(target.read_text())["samples"] == 2
    assert not (run.work / "summary.json").exists()
    record = run.outcome_list[0]
    assert record["test"] == gtest("SdpaFwd/Default", 1000)
    assert record["engine"] == "STUB_ENGINE" and record["arch"] == "gfx950"
    assert (
        record["purpose"] == "probe"
        and record["outcome"] == "PASS"
        and record["source"] == "binary"
    )


def test_unrecognized_outcome_is_unexercised_not_a_pass(run, monkeypatch):
    # a binary that reports an outcome this driver does not know must not count green
    class Odd(fuzz.Fuzz):
        def _finish_record(self, sp, sample, record, attempt):
            super()._finish_record(sp, sample, dict(record, outcome="MAYBE"), attempt)

    args = fuzz.parse_args(run.args("--samples", "1"))
    assert Odd(args, fuzz.select_spaces(args)).run() == 0
    assert run.outcome_list[0]["outcome"] == "UNEXERCISED"
    assert "unrecognized outcome 'MAYBE'" in run.outcome_list[0]["reason"]


# --------------------------------------------------------------------------
# R6: crash, hang, load and registration recovery
# --------------------------------------------------------------------------


def test_crash_is_one_case_and_the_remainder_runs_again(run):
    run.control(cases={"s1002": "crash"})
    assert run.run("--samples", "6", "--batch-size", "6") == 1
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert {c: o["outcome"] for c, o in outcomes.items()} == {
        "s1000": "PASS", "s1001": "PASS", "s1002": "CRASH", "s1003": "PASS", "s1004": "PASS", "s1005": "PASS",
    }  # fmt: skip
    assert (
        outcomes["s1002"]["source"] == "driver"
        and outcomes["s1002"]["crash_kind"] == "death"
    )
    assert (
        "died" in outcomes["s1002"]["reason"] and "139" in outcomes["s1002"]["reason"]
    )
    runs = [i["tests"] for i in run.invocations if i["mode"] == "run"]
    assert len(runs) == 2 and len(runs[0]) == 6
    assert runs[1] == [gtest("SdpaFwd/Default", s) for s in (1003, 1004, 1005)]


def test_several_crashes_in_one_batch_each_cost_one_case(run):
    run.control(cases={"s1000": "crash", "s1002": "crash", "s1004": "crash"})
    assert run.run("--samples", "5", "--batch-size", "5") == 1
    assert [o["outcome"] for o in run.outcome_list] == [
        "CRASH",
        "PASS",
        "CRASH",
        "PASS",
        "CRASH",
    ]


def test_hang_is_killed_after_the_case_timeout_and_counts_as_crash(run):
    run.control(cases={"s1001": "hang"}, hang_seconds=120)
    started = time.monotonic()
    assert run.run("--samples", "3", "--batch-size", "3", "--case-timeout", "1s") == 1
    assert time.monotonic() - started < 60
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert (
        outcomes["s1001"]["outcome"] == "CRASH"
        and outcomes["s1001"]["crash_kind"] == "hang"
    )
    assert "hang" in outcomes["s1001"]["reason"]
    assert (
        outcomes["s1000"]["outcome"] == "PASS"
        and outcomes["s1002"]["outcome"] == "PASS"
    )
    assert run.summary["engines"]["STUB_ENGINE"]["spaces"]["SdpaFwd/Default"][
        "crash"
    ] == ["s1001"]


def test_case_skipped_at_load_is_unexercised_with_the_load_message(run):
    run.control(cases={"s1001": "load_skip"})
    assert run.run("--samples", "3") == 0
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert outcomes["s1001"]["outcome"] == "UNEXERCISED"
    assert (
        outcomes["s1001"]["reason"].startswith("load:")
        and "stub load failure" in outcomes["s1001"]["reason"]
    )
    assert outcomes["s1001"]["source"] == "driver"
    runs = [i["tests"] for i in run.invocations if i["mode"] == "run"]
    assert all(gtest("SdpaFwd/Default", 1001) not in names for names in runs)
    assert outcomes["s1000"]["outcome"] == outcomes["s1002"]["outcome"] == "PASS"


def test_registration_crash_is_bisected_to_the_case_that_causes_it(run):
    run.control(cases={"s1005": "reg_crash"})
    assert run.run("--samples", "8", "--batch-size", "8") == 0
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert outcomes["s1005"]["outcome"] == "UNEXERCISED"
    assert outcomes["s1005"]["reason"].startswith("load-crash")
    assert "134" in outcomes["s1005"]["reason"]
    assert {c: o["outcome"] for c, o in outcomes.items() if c != "s1005"} == {
        f"s{1000 + i}": "PASS" for i in range(8) if i != 5
    }
    assert (
        len([i for i in run.invocations if i["mode"] == "list"]) <= 8
    )  # log2 splits, not one per case


def test_two_registration_crashes_are_both_isolated(run):
    run.control(cases={"s1001": "reg_crash", "s1006": "reg_crash"})
    assert run.run("--samples", "8", "--batch-size", "8") == 0
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert sorted(c for c, o in outcomes.items() if o["outcome"] == "UNEXERCISED") == [
        "s1001",
        "s1006",
    ]
    assert sum(o["outcome"] == "PASS" for o in outcomes.values()) == 6


def test_a_run_that_dies_before_any_case_starts_is_bisected_too(run):
    run.control(cases={"s1003": "run_start_crash"})
    assert run.run("--samples", "5", "--batch-size", "5") == 0
    outcomes = {o["case"]: o for o in run.outcome_list}
    assert outcomes["s1003"]["outcome"] == "UNEXERCISED" and outcomes["s1003"][
        "reason"
    ].startswith("load-crash")
    assert sorted(c for c, o in outcomes.items() if o["outcome"] == "PASS") == [
        "s1000",
        "s1001",
        "s1002",
        "s1004",
    ]


def test_every_sample_has_exactly_one_outcome_under_mixed_failures(run):
    run.control(
        seed_mod={
            "m": 9,
            "map": {
                "1": "crash",
                "2": "load_skip",
                "3": "wrong",
                "4": "declined",
                "5": "reg_crash",
                "6": "hang",
            },
        },
        hang_seconds=60,
    )
    assert run.run("--samples", "9", "--batch-size", "9", "--case-timeout", "1s") == 1
    cases = [o["case"] for o in run.outcome_list]
    assert (
        sorted(cases) == sorted({f"s{1000 + i}" for i in range(9)}) and len(cases) == 9
    )
    assert run.summary["reconciled"] is True and run.summary["samples"] == 9
    assert run.summary["outcomes"]["CRASH"] == 2  # one death, one hang


def test_binary_that_cannot_run_is_a_usage_error(run, capsys):
    assert (
        fuzz.main(
            [
                "--binary",
                "definitely-not-a-binary-xyz",
                "--work-dir",
                str(run.work),
                "--samples",
                "1",
            ]
        )
        == 2
    )
    assert "not an executable" in capsys.readouterr().err


# --------------------------------------------------------------------------
# R6: SIGINT
# --------------------------------------------------------------------------


def _fuzz(run, *extra, **kwargs):
    args = fuzz.parse_args(run.args(*extra, **kwargs))
    return fuzz.Fuzz(args, fuzz.select_spaces(args))


def test_stop_request_finishes_the_batch_and_writes_the_report(run):
    run.control(delay=0.3)
    f = _fuzz(run, "--samples", "0", "--time-budget", "5m", "--batch-size", "3")
    timer = threading.Timer(0.5, f.request_stop)
    timer.start()
    assert f.run() == 0
    timer.join()
    summary = run.summary
    assert summary["stopped_because"] == "interrupted"
    assert summary["samples"] >= 3 and summary["samples"] % 3 == 0  # whole batches only
    assert (
        summary["reconciled"] is True
        and summary["outcomes"]["PASS"] == summary["samples"]
    )
    assert summary["elapsed_seconds"] < 60


def test_second_interrupt_kills_the_binary_without_inventing_crashes(run):
    run.control(delay=30)
    f = _fuzz(run, "--samples", "4", "--batch-size", "4")
    timer = threading.Timer(1.0, lambda: (f.request_stop(), f.request_stop(abort=True)))
    timer.start()
    started = time.monotonic()
    assert f.run() == 0
    timer.join()
    assert time.monotonic() - started < 25
    assert {o["outcome"] for o in run.outcome_list} == {"UNEXERCISED"}
    assert all(
        o["reason"] == "interrupted before the case ran" for o in run.outcome_list
    )
    assert run.summary["reconciled"] is True and run.summary["outcomes"]["CRASH"] == 0


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX SIGINT delivery")
def test_real_sigint_stops_the_driver_gracefully(run):
    run.control(delay=0.2)
    cmd = [sys.executable, str(fuzz.DRIVER_PATH)] + run.args(
        "--samples", "0", "--time-budget", "5m", "--batch-size", "2"
    )
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    deadline = time.monotonic() + 30
    seen = ""
    while "batch " not in seen and time.monotonic() < deadline:
        seen += proc.stdout.readline()
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert "finishing the current batch" in out
    assert (
        run.summary["stopped_because"] == "interrupted"
        and run.summary["outcomes"]["CRASH"] == 0
    )


# --------------------------------------------------------------------------
# R6b: continuous mode
# --------------------------------------------------------------------------


def test_continuous_mode_keeps_drawing_fresh_samples_until_the_time_budget(run):
    run.control(delay=0.02)
    started = time.monotonic()
    two = "^(RMSNorm/Default|SdpaFwd/Default)$"
    assert run.run("--time-budget", "2s", "--batch-size", "2", space=two) == 0
    elapsed = time.monotonic() - started
    assert 2 <= elapsed < 40
    summary = run.summary
    assert summary["continuous"] is True and summary["stopped_because"] == "time-budget"
    assert summary["samples_per_space"] is None
    samples = [
        json.loads(line)
        for line in (run.work / "samples.jsonl").read_text().splitlines()
    ]
    assert len(samples) >= 6  # several batches
    per_space = {}
    for s in samples:
        per_space.setdefault(s["space"], []).append(s["shape_seed"])
    assert set(per_space) == {"RMSNorm/Default", "SdpaFwd/Default"}
    for seeds in per_space.values():
        assert len(seeds) >= 2 and len(set(seeds)) == len(
            seeds
        )  # distinct ids across batches
        assert seeds == list(range(1000, 1000 + len(seeds)))  # fresh seeds every batch
    # round robin: batches alternate between the spaces
    batch_spaces = [samples[i]["space"] for i in range(0, len(samples), 2)]
    assert all(a != b for a, b in zip(batch_spaces, batch_spaces[1:]))
    # every sample has one outcome
    assert summary["reconciled"] is True and summary["samples"] == len(samples) == len(
        run.outcome_list
    )
    assert len({(o["space"], o["case"]) for o in run.outcome_list}) == len(samples)


def test_continuous_mode_logs_crashes_and_goes_on(run):
    run.control(seed_mod={"m": 3, "map": {"1": "crash"}}, delay=0.01)
    assert run.run("--samples", "0", "--time-budget", "2s", "--batch-size", "3") == 1
    summary = run.summary
    assert summary["outcomes"]["CRASH"] >= 2 and summary["outcomes"]["PASS"] >= 4
    assert summary["reconciled"] is True and summary["samples"] > 3
    assert len(list((run.work / "failures").iterdir())) == summary["outcomes"]["CRASH"]


def test_sample_cap_stops_a_budgeted_run_too(run):
    assert run.run("--samples", "4", "--time-budget", "5m", "--batch-size", "2") == 0
    assert (
        run.summary["stopped_because"] == "sample-cap" and run.summary["samples"] == 4
    )
    assert run.summary["continuous"] is False


# --------------------------------------------------------------------------
# R6c: fault artifacts
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def findings(tmp_path_factory):
    """One run with a WRONG, a crash and a hang, then a probe that checks the folders exist."""
    tmp = tmp_path_factory.mktemp("findings")
    mp = pytest.MonkeyPatch()
    for name in list(os.environ):
        if name.startswith("HIPDNN_FUZZ_") or name in (
            "HIPDNN_STUB_CONTROL",
            fuzz.HOST_FILL_ENV,
        ):
            mp.delenv(name)
    mp.setenv(fuzz.HOST_FILL_ENV, "1")
    mp.setenv("HIPDNN_FINDINGS_MARKER", "present")
    r = Run(tmp, mp)
    r.control(
        cases={
            "s2000": "wrong",
            "s2001": "crash",
            "s2002": "hang",
            "s2003": "probe_file",
        },
        hang_seconds=120,
        probe_path=str(r.failure_dir("s2002") / "repro.json"),
    )
    code = r.run(
        "--samples", "5", "--batch-size", "5", "--case-timeout", "1s", "--seed", "2000",
        "--test-article", "ART", "--test-config", "CFG",
    )  # fmt: skip
    assert code == 1
    yield r
    mp.undo()


def test_wrong_crash_and_hang_each_leave_a_complete_folder(findings):
    r = findings
    assert sorted(p.name for p in (r.work / "failures").iterdir()) == [
        "SdpaFwd_Default_s2000", "SdpaFwd_Default_s2001", "SdpaFwd_Default_s2002",
    ]  # fmt: skip
    for case, outcome in (("s2000", "WRONG"), ("s2001", "CRASH"), ("s2002", "CRASH")):
        folder = r.failure_dir(case)
        name = f"SdpaFwd_Default_{case}"
        files = {p.name for p in folder.iterdir()}
        assert files >= {
            f"{name}.json", f"{name}.meta.json", "repro.json", "outcome.json", "log_tail.txt",
            "replay.sh", "replay.txt",
        }  # fmt: skip
        assert (
            json.loads((folder / "outcome.json").read_text())["result"]["outcome"]
            == outcome
        )
        assert json.loads((folder / "repro.json").read_text())["outcome"] == outcome


def test_folders_are_written_before_the_run_continues(findings):
    r = findings
    probe = next(o for o in r.outcome_list if o["case"] == "s2003")
    assert probe["outcome"] == "PASS" and probe["reason"] == "probe:True"
    assert r.summary["reconciled"] is True and r.summary["outcomes"] == {
        "PASS": 2, "DECLINED": 0, "WRONG": 1, "CRASH": 2, "UNCLAIMED_PASS": 0, "UNEXERCISED": 0,
    }  # fmt: skip


def test_fault_graph_is_the_expanded_concrete_graph_with_canonical_uids(findings):
    sp = space_by_id("SdpaFwd/Default")
    template = json.loads(sp.template_path.read_text())
    for case, seed in (("s2000", 2000), ("s2001", 2001)):
        graph = json.loads(
            (findings.failure_dir(case) / f"SdpaFwd_Default_{case}.json").read_text()
        )
        row = sampler.sample(sp, seed).row
        expected = expanded_graph(
            template, row
        )  # with the row's alignment patches applied
        assert bu.canon(graph) == bu.canon(
            bu.remap_graph(expected, bu.canonical_uid_map_by_name(expected))
        )
        assert {t["alignment"] for t in graph["tensors"]} == {
            row["tensor_patches"][0]["set"]["alignment"]
        }
        assert [t["name"] for t in graph["tensors"]] == ["K", "Q", "SdpaFwd_0::O", "V"]
        assert [t["uid"] for t in graph["tensors"]] == [0, 1, 2, 3]
        assert "${" not in json.dumps(graph)


def test_meta_sidecar_records_seed_purpose_and_source(findings):
    sp = space_by_id("SdpaFwd/Default")
    meta = json.loads(
        (findings.failure_dir("s2000") / "SdpaFwd_Default_s2000.meta.json").read_text()
    )
    s = sampler.sample(sp, 2000)
    assert meta["format_version"] == 1 and meta["purpose"] == "regression"
    assert meta["seed"] == s.data_seed
    assert (
        "SdpaFwd/Default" in meta["reference_source"]
        and "2000" in meta["reference_source"]
    )
    assert "operation" not in meta  # a failure folder is not a capture suite


def test_repro_json_carries_everything_needed_to_replay(findings):
    sp = space_by_id("SdpaFwd/Default")
    repro = json.loads((findings.failure_dir("s2001") / "repro.json").read_text())
    s = sampler.sample(sp, 2001)
    assert repro["space"] == "SdpaFwd/Default"
    assert repro["space_sha256"] == hashlib.sha256(sp.path.read_bytes()).hexdigest()
    assert (
        repro["template_sha256"]
        == hashlib.sha256(sp.template_path.read_bytes()).hexdigest()
    )
    assert repro["shape_seed"] == 2001 and repro["data_seed"] == s.data_seed
    assert repro["axis_values"] == s.axis_values and repro["sweep_row"] == s.row
    assert (
        repro["input_fill"]["path"] == "host"
        and repro["input_fill"][fuzz.HOST_FILL_ENV] == "1"
    )
    assert repro["input_fill"]["data_seed"] == s.data_seed
    assert repro["engine"] == "STUB_ENGINE" and repro["arch"] == "gfx950"
    assert repro["test_article"] == "ART" and repro["test_config"] == "CFG"
    assert (
        "rocm_version" in repro and "git_sha" in repro
    )  # best effort: present, maybe null
    assert (
        repro["driver_flags"]["case_timeout"] == 1.0
        and repro["driver_flags"]["seed"] == 2000
    )
    assert (
        repro["hipdnn_env"][fuzz.HOST_FILL_ENV] == "1"
        and repro["hipdnn_env"]["HIPDNN_FINDINGS_MARKER"] == "present"
    )
    assert "HIPDNN_STUB_CONTROL" in repro["hipdnn_env"]
    assert repro["uids_canonical"] is True


def test_log_tail_keeps_the_binarys_last_words(findings):
    assert (
        "segmentation fault"
        in (findings.failure_dir("s2001") / "log_tail.txt").read_text()
    )
    assert (
        "[ RUN      ] " + gtest("SdpaFwd/Default", 2002)
        in (findings.failure_dir("s2002") / "log_tail.txt").read_text()
    )


def test_replay_files_name_one_command_and_the_fill_path(findings):
    folder = findings.failure_dir("s2000")
    repro = json.loads((folder / "repro.json").read_text())
    script = (folder / "replay.sh").read_text()
    assert script.startswith("#!/bin/sh\n")
    command = [line for line in script.splitlines() if line.startswith("exec ")]
    assert len(command) == 1
    text = command[0]
    assert (
        "--filter '^SdpaFwd/Default$' --seed 2000 --samples 1 --data-seed "
        + str(repro["data_seed"])
        in text
    )
    assert f"{fuzz.HOST_FILL_ENV}=1" in text and "HIPDNN_STUB_CONTROL=" in text
    assert (
        '--work-dir "$D/replay-work"' in text
        and "--test-article ART" in text
        and "--test-config CFG" in text
    )
    assert "host" in script and fuzz.HOST_FILL_ENV in script
    replay_txt = (folder / "replay.txt").read_text()
    assert replay_txt.splitlines()[0] == repro["replay"]["command"]
    assert "input fill path: host" in replay_txt and "import_graph.py" in replay_txt


def _replay(folder, work_name="replayed"):
    repro = json.loads((folder / "repro.json").read_text())
    argv = list(repro["replay"]["argv"])
    work = folder.parent.parent / work_name
    argv[argv.index("--work-dir") + 1] = str(work)
    env = {k: v for k, v in os.environ.items() if k not in repro["replay"]["env_unset"]}
    env.update(repro["replay"]["env_set"])
    done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)
    return repro, work, done


@pytest.mark.parametrize(
    "case, outcome", [("s2000", "WRONG"), ("s2001", "CRASH"), ("s2002", "CRASH")]
)
def test_replay_regenerates_the_identical_sweep_row_graph_and_seed(
    findings, case, outcome
):
    folder = findings.failure_dir(case)
    repro, work, done = _replay(folder, f"replayed_{case}")
    assert done.returncode == 1, done.stdout + done.stderr
    sweep = json.loads(
        (sweep_dir(work, "SdpaFwd/Default", int(case[1:])) / "sweep.json").read_text()
    )
    assert len(sweep["cases"]) == 1
    assert json.dumps(sweep["cases"][0]) == json.dumps(
        repro["sweep_row"]
    )  # byte-identical row
    assert sweep["cases"][0]["metadata"]["seed"] == repro["data_seed"]
    replayed_folder = work / "failures" / f"SdpaFwd_Default_{case}"
    name = f"SdpaFwd_Default_{case}.json"
    assert (replayed_folder / name).read_bytes() == (folder / name).read_bytes()
    result = json.loads((replayed_folder / "outcome.json").read_text())["result"]
    assert result["outcome"] == outcome
    assert json.loads((work / "summary.json").read_text())["seed"] == int(case[1:])


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_replay_script_runs(findings):
    folder = findings.failure_dir("s2000")
    done = subprocess.run(
        ["sh", str(folder / "replay.sh")], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 1
    assert (folder / "replay-work" / "failures" / "SdpaFwd_Default_s2000").is_dir()


# --------------------------------------------------------------------------
# R6c: freeze
# --------------------------------------------------------------------------


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_fault_folder_imports_for_real_and_expands_back_identically(
    run, tmp_path, space_id
):
    sp = space_by_id(space_id)
    run.control(default="wrong")
    assert run.run("--samples", "1", "--seed", "4242", space=space_id) == 1
    folder = run.failure_dir("s4242", f"{sp.operation}_{sp.name}")
    case = folder.name
    failing_graph = json.loads((folder / f"{case}.json").read_text())
    repro = json.loads((folder / "repro.json").read_text())

    tree = tmp_path / "tree"
    topology = tree / "quick" / sp.operation / sp.template_path.parent.name
    topology.mkdir(parents=True)
    for name in ("graph.template.json", "sweep.json", "support.json"):
        if (
            sp.template_path.parent / name
        ).is_file():  # SdpaBackward ships no support.json
            shutil.copyfile(sp.template_path.parent / name, topology / name)
    before = json.loads((topology / "sweep.json").read_text())["cases"]

    def import_it():
        return subprocess.run(
            [sys.executable, str(MIGRATION_DIR / "import_graph.py"), "--graph", str(folder / f"{case}.json"),
             "--bundle-dir", str(tree), "--tier", "quick"],
            capture_output=True, text=True, cwd=MIGRATION_DIR, timeout=120,
        )  # fmt: skip

    done = import_it()
    assert done.returncode == 0, done.stderr
    assert "appended case" in done.stderr
    template = json.loads((topology / "graph.template.json").read_text())
    cases = json.loads((topology / "sweep.json").read_text())["cases"]
    assert len(cases) == len(before) + 1
    new = cases[-1]
    expanded = bu.expand(template, new["values"])
    fuzz.load_support_matrix_units().apply_case_patches(expanded, new)
    assert bu.canon(expanded) == bu.canon(failing_graph)  # the same graph back
    # UIDs unchanged: the import did not renumber a tensor (that would change its input seed)
    sampled = bu.expand(template, repro["sweep_row"]["values"])
    # a tensor the row adds (add_tensors) is part of the graph too
    fuzz.load_support_matrix_units().apply_case_patches(sampled, repro["sweep_row"])
    assert [(t["name"], t["uid"]) for t in failing_graph["tensors"]] == [
        (t["name"], t["uid"]) for t in sampled["tensors"]
    ]
    assert [t["uid"] for t in new["values"]["tensors"]] == [
        t["uid"] for t in repro["sweep_row"]["values"]["tensors"]
    ]
    assert new["metadata"]["seed"] == repro["data_seed"]  # the same seed
    # the fill recipes survive the failure folder and the freeze: a frozen regression that
    # lost metadata.inputs would run the harness default fill and no longer reproduce
    recipes = repro["sweep_row"]["metadata"].get("inputs")  # Matmul/Align sets none
    meta = json.loads((folder / f"{case}.meta.json").read_text())
    assert meta.get("inputs") == recipes
    assert new["metadata"].get("inputs") == recipes
    # the declared alignments and the placement survive too: a regression frozen without them
    # would run at the template's alignment 16 and no longer be the case that failed
    placement = repro["sweep_row"]["metadata"]
    assert meta["placement"] == new["metadata"]["placement"] == placement["placement"]
    assert (
        meta["placement_uid"]
        == new["metadata"]["placement_uid"]
        == placement["placement_uid"]
    )
    patched = {
        p["uid"]: p["set"]["alignment"] for p in repro["sweep_row"]["tensor_patches"]
    }
    assert {
        t["uid"]: t["alignment"]
        for t in failing_graph["tensors"]
        if t["uid"] in patched
    } == patched
    # a patch exactly where the graph differs from what the template says (some templates
    # carry no alignment at all)
    base = {t["uid"]: t.get("alignment") for t in template["tensors"]}
    assert new.get("tensor_patches", []) == [
        {"uid": uid, "set": {"alignment": a}}
        for uid, a in sorted(patched.items())
        if base[uid] != a
    ]
    assert new["metadata"]["purpose"] == "regression"
    assert (
        space_id in new["metadata"]["reference_source"]
        and "4242" in new["metadata"]["reference_source"]
    )
    # importing again is a no-op (duplicate detection)
    again = import_it()
    assert again.returncode == 0 and "DUPLICATE" in again.stderr
    assert (
        len(json.loads((topology / "sweep.json").read_text())["cases"])
        == len(before) + 1
    )


def test_fault_folder_survives_a_later_crash(run):
    run.control(cases={"s1000": "wrong", "s1003": "crash", "s1005": "crash"})
    assert run.run("--samples", "8", "--batch-size", "4") == 1
    names = sorted(p.name for p in (run.work / "failures").iterdir())
    assert names == [
        "SdpaFwd_Default_s1000",
        "SdpaFwd_Default_s1003",
        "SdpaFwd_Default_s1005",
    ]
    for name in names:
        assert (run.work / "failures" / name / "repro.json").exists()
        assert (run.work / "failures" / name / "replay.sh").exists()
    # the later batches did not touch the first finding
    first = json.loads(
        (run.work / "failures/SdpaFwd_Default_s1000/outcome.json").read_text()
    )
    assert first["result"]["outcome"] == "WRONG"


# --------------------------------------------------------------------------
# A binary that cannot run at all, and a batch in which nothing registers
# --------------------------------------------------------------------------


def test_a_binary_that_fails_at_startup_stops_the_run_with_status_2(run, capsys):
    run.control(startup_fail=True)
    assert run.run("--samples", "8", "--batch-size", "8") == 2
    # Stopped by the preflight: no batch was written, listed or bisected.
    assert run.invocations == []
    assert not (run.work / "bundles" / "gauntlet").exists()
    assert run.outcome_list == []
    err = capsys.readouterr().err
    assert "exited with status 1" in err and "is not loaded" in err
    assert "load-crash" not in json.dumps(run.summary)


def test_the_preflight_lists_the_binary_on_an_empty_bundle_root(run):
    assert run.run("--samples", "2", "--batch-size", "2") == 0
    first = run.invocations[0]
    assert first["mode"] == "list" and first["tests"] == []
    assert not any(
        arg.startswith("--gtest_filter=") for arg in first["argv"]
    ), "the preflight must not filter"
    root = first["argv"][first["argv"].index("--golden-data-dir") + 1]
    assert list(Path(root).iterdir()) == []


def test_a_batch_in_which_nothing_registers_is_load_not_load_crash(run):
    run.control(cases={"s1000": "load_skip"})
    assert run.run("--samples", "1") == 0
    [outcome] = run.outcome_list
    assert outcome["case"] == "s1000" and outcome["outcome"] == "UNEXERCISED"
    assert (
        outcome["reason"].startswith("load:")
        and "stub load failure" in outcome["reason"]
    )
    assert "load-crash" not in outcome["reason"]
    listings = [i for i in run.invocations if i["mode"] == "list"]
    # One for the preflight, one for the batch; no bisection and no run.
    assert len(listings) == 2
    assert not [i for i in run.invocations if i["mode"] == "run"]


def test_every_sample_of_a_batch_failing_to_load_is_load_not_load_crash(run):
    run.control(default="load_skip")
    assert run.run("--samples", "4", "--batch-size", "4") == 0
    assert [o["outcome"] for o in run.outcome_list] == ["UNEXERCISED"] * 4
    assert all(o["reason"].startswith("load:") for o in run.outcome_list)
    assert len([i for i in run.invocations if i["mode"] == "list"]) == 2


def test_the_freeze_command_is_quoted_and_independent_of_the_working_directory(run):
    run.control(cases={"s1000": "wrong"})
    assert run.run("--samples", "1") == 1
    text = (run.failure_dir("s1000") / "replay.txt").read_text()
    [line] = [ln for ln in text.splitlines() if ln.startswith("freeze: ")]
    words = shlex.split(line[len("freeze: ") :])
    assert Path(words[0]).is_absolute() and Path(words[1]).name == "import_graph.py"
    assert Path(words[1]).is_absolute()
    graph = words[words.index("--graph") + 1]
    bundle_dir = words[words.index("--bundle-dir") + 1]
    assert Path(graph).is_absolute() and Path(graph).exists()
    assert Path(bundle_dir).is_absolute() and Path(bundle_dir).is_dir()
    assert words[words.index("--tier") + 1] == "quick"


def test_load_reason_falls_back_when_the_binary_logged_nothing_about_the_case(run):
    # The real binary logs the skipped bundle only with HIPDNN_LOG_LEVEL set. What it always
    # prints is the zero-tests diagnostic, which echoes the whole --gtest_filter, case id
    # included; that line is not a load message.
    run.control(cases={"s1000": "load_skip"}, quiet_load=True)
    assert run.run("--samples", "1") == 0
    [outcome] = run.outcome_list
    assert outcome["outcome"] == "UNEXERCISED"
    assert outcome["reason"] == (
        "load: absent from --gtest_list_tests (the harness skipped the case at load)"
    )
    assert "gtest_filter" not in outcome["reason"]


def test_preflight_accepts_a_binary_that_has_no_compiled_in_tests(run):
    run.control(no_builtin=True)
    assert run.run("--samples", "2", "--batch-size", "2") == 0
    assert [o["outcome"] for o in run.outcome_list] == ["PASS", "PASS"]


# --------------------------------------------------------------------------
# --row: one explicit sample
# --------------------------------------------------------------------------


def _write_row(tmp_path, space="SdpaFwd/Default", name="row.json", **fields):
    """A --row file for ``space``; ``fields`` replace or add keys (a None value drops one)."""
    spec = {"space": space, "axes": {}, "data_seed": 7}
    spec.update(fields)
    path = tmp_path / name
    path.write_text(json.dumps({k: v for k, v in spec.items() if v is not None}))
    return path


def _row_args(run, row_path, *extra):
    return [
        "--binary", str(STUB),
        "--work-dir", str(run.work),
        "--case-timeout", "30s",
        "--row", str(row_path),
        *extra,
    ]  # fmt: skip


def _run_row(run, row_path, *extra):
    return fuzz.main(_row_args(run, row_path, *extra))


def _row_of(sample, **fields):
    return {
        "space": sample.space.space_id,
        "axes": sample.axis_values,
        "data_seed": sample.data_seed,
        **fields,
    }


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_a_row_run_has_the_gtest_name_and_sweep_row_of_the_seeded_run_it_replays(
    run, tmp_path, space_id
):
    # bug caught: a --row replay lands in another suite or under another case id, so the engine
    # TOML's tolerance and skip globs grade it differently from the run that found the defect
    seed = 4321
    sample = sampler.sample(space_by_id(space_id), seed)
    assert run.run("--samples", "1", seed=seed, space=space_id) == 0
    drawn = run.outcome_list[0]["test"]
    assert drawn == gtest(space_id, seed)
    drawn_row = json.loads(
        (sweep_dir(run.work, space_id, seed) / "sweep.json").read_text()
    )["cases"][0]

    (tmp_path / "row").mkdir()
    row_run = Run(tmp_path / "row", run.monkeypatch)
    row = _write_row(row_run.tmp, **_row_of(sample, case_id=f"s{seed}"))
    assert _run_row(row_run, row) == 0
    (outcome,) = row_run.outcome_list
    assert outcome["test"] == drawn
    written = json.loads(
        (sweep_dir(row_run.work, space_id, seed) / "sweep.json").read_text()
    )["cases"]
    assert len(written) == 1
    # the same case, but for the note that says how it was made
    assert written[0]["metadata"].pop("notes") != drawn_row["metadata"].pop("notes")
    assert json.dumps(written[0]) == json.dumps(drawn_row)
    # the same layout under bundles/gauntlet/<Op>/<Space>[/<tag>_cases]
    assert (sweep_dir(row_run.work, space_id, seed) / "graph.template.json").is_file()
    # the binary was asked for exactly that test
    modes = {i["mode"]: i["argv"] for i in row_run.invocations}
    assert f"--gtest_filter={drawn}" in modes["run"]


def test_a_row_without_a_case_id_is_named_row_and_a_given_case_id_is_used(
    run, tmp_path
):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 9)
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample))) == 0
    suite = gtest("SdpaFwd/Default", 9).split(".")[0]
    assert run.outcome_list[0]["test"] == f"{suite}.row"
    assert run.outcome_list[0]["case"] == "row"
    assert (
        _run_row(run, _write_row(tmp_path, **_row_of(sample, case_id="ladder_3"))) == 0
    )
    assert run.outcome_list[0]["test"] == f"{suite}.ladder_3"
    rows = json.loads(
        (sweep_dir(run.work, "SdpaFwd/Default", 9) / "sweep.json").read_text()
    )["cases"]
    assert [r["id"] for r in rows] == [
        "ladder_3"
    ]  # one case: the old bundle is replaced


def test_a_row_run_records_its_outcome_summary_and_failure_folder(run, tmp_path):
    # bug caught: a one-case run that records nothing, or whose finding cannot be replayed
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 31)
    run.control(default="wrong")
    row = _write_row(tmp_path, **_row_of(sample, case_id="s31"))
    assert _run_row(run, row) == 1
    (outcome,) = run.outcome_list
    assert outcome["outcome"] == "WRONG" and outcome["case"] == "s31"
    assert outcome["shape_seed"] is None and outcome["data_seed"] == sample.data_seed
    summary = run.summary
    assert summary["outcomes"]["WRONG"] == 1 and summary["samples"] == 1
    assert summary["reconciled"] is True and summary["exit_code"] == 1
    assert summary["stopped_because"] == "row" and summary["continuous"] is False
    assert summary["seed"] is None and summary["data_seed"] == sample.data_seed
    assert summary["row"] == json.loads(row.read_text()) | {"case_id": "s31"}
    assert summary["findings"][0]["failure_dir"] == outcome["failure_dir"]
    assert json.loads((run.work / "samples.jsonl").read_text()) == {
        "space": "SdpaFwd/Default",
        "case": "s31",
        "shape_seed": None,
        "data_seed": sample.data_seed,
        "axes": sample.axis_values,
    }

    folder = run.failure_dir("s31")
    repro = json.loads((folder / "repro.json").read_text())
    assert repro["axis_values"] == sample.axis_values and repro["shape_seed"] is None
    assert repro["row"] == {**_row_of(sample), "case_id": "s31"}
    assert repro["gtest_name"] == outcome["test"]
    assert json.loads((folder / "row.json").read_text()) == repro["row"]
    replay = (folder / "replay.txt").read_text()
    argv = repro["replay"]["argv"]
    assert argv[argv.index("--row") + 1] == str(folder / "row.json")
    assert "--seed" not in argv and "--filter" not in argv and "--samples" not in argv
    assert "--row" in replay.splitlines()[0] and "--seed" not in replay.splitlines()[0]
    assert f"axes: {json.dumps(sample.axis_values, sort_keys=True)}" in replay
    script = (folder / "replay.sh").read_text()
    assert '--row "$D/row.json"' in script and "--filter" not in script
    assert "shape seed" not in (folder / "SdpaFwd_Default_s31.meta.json").read_text()
    assert "explicit row" in (folder / "SdpaFwd_Default_s31.meta.json").read_text()

    # the recorded replay command reproduces it: same GTest name, same row, same outcome
    _, work, done = _replay(folder)
    assert done.returncode == 1, done.stdout + done.stderr
    again = json.loads((work / "outcomes.jsonl").read_text())
    assert (again["test"], again["outcome"]) == (outcome["test"], "WRONG")
    replayed = json.loads(
        (sweep_dir(work, "SdpaFwd/Default", 31) / "sweep.json").read_text()
    )["cases"]
    assert json.dumps(replayed[0]) == json.dumps(repro["sweep_row"])


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_a_row_failure_folder_replay_script_runs(run, tmp_path):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 31)
    run.control(default="wrong")
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample, case_id="s31"))) == 1
    folder = run.failure_dir("s31")
    done = subprocess.run(
        ["sh", str(folder / "replay.sh")], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert (folder / "replay-work" / "summary.json").is_file()


def test_the_stub_can_key_a_behaviour_on_a_rendered_value_not_the_case_id(
    run, tmp_path
):
    # the dims of the sample's tensors are in the written row; the case ids here are all
    # alike, so only the rendered value can tell the two runs apart
    sp = space_by_id("SdpaFwd/Default")
    sample = sampler.sample(sp, 12)
    sq = sample.row["values"]["tensors"][0]["dims"][2]
    run.control(
        match=[
            {
                "where": [{"path": "values.tensors.0.dims.2", "op": ">=", "value": sq}],
                "behaviour": "wrong",
            }
        ]
    )
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample, case_id="same"))) == 1
    assert run.outcome_list[0]["outcome"] == "WRONG"
    run.control(
        match=[
            {
                "where": [{"path": "values.tensors.0.dims.2", "op": ">", "value": sq}],
                "behaviour": "wrong",
            }
        ]
    )
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample, case_id="same"))) == 0
    assert run.outcome_list[0]["outcome"] == "PASS"


def test_a_stub_match_needs_every_condition_and_falls_back_to_case_then_default(
    run, tmp_path
):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 12)
    sq = sample.row["values"]["tensors"][0]["dims"][2]
    seed = sample.data_seed
    rule = {
        "where": [
            {"path": "values.tensors.0.dims.2", "op": "==", "value": sq},
            {"path": "metadata.seed", "op": "!=", "value": seed},
        ],
        "behaviour": "wrong",
    }
    row = _write_row(tmp_path, **_row_of(sample, case_id="c1"))
    run.control(match=[rule])
    assert _run_row(run, row) == 0  # the second condition fails
    rule["where"][1]["value"] = seed + 1
    run.control(match=[rule])
    assert _run_row(run, row) == 1
    run.control(match=[rule], cases={"c1": "declined"})
    assert _run_row(run, row) == 0  # an explicit case entry comes first
    assert run.outcome_list[0]["outcome"] == "DECLINED"
    # a path the row lacks and operands that cannot be compared do not match
    run.control(
        match=[
            {"where": [{"path": "node_patches.0.set", "op": "==", "value": 1}],
             "behaviour": "wrong"},
            {"where": [{"path": "metadata.notes", "op": "<", "value": 5}],
             "behaviour": "wrong"},
        ]
    )  # fmt: skip
    assert _run_row(run, row) == 0


def test_a_row_that_violates_a_constraint_is_a_usage_error_that_writes_nothing(
    run, tmp_path, capsys
):
    # bug caught: an illegal sample runs, and the engine's answer to it is read as a finding
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(spaces)  # constraint: n >= a
    row = _write_row(
        tmp_path, space="Toy/Default", axes={"a": 3, "n": 1}, case_id="bad"
    )
    assert _run_row(run, row, "--spaces-dir", str(spaces)) == 2
    err = capsys.readouterr().err
    assert "Toy/Default" in err and "n >= a" in err and "n must reach a" in err
    assert not run.work.exists() and not run.log.exists()
    # the same axes with the constraint met run
    ok = _write_row(tmp_path, space="Toy/Default", axes={"a": 3, "n": 3})
    assert _run_row(run, ok, "--spaces-dir", str(spaces)) == 0


@pytest.mark.parametrize(
    "axes, message",
    [
        ({"a": 3}, r"missing axes \['n'\]"),
        ({"a": 3, "n": 3, "z": 1}, r"unknown axes \['z'\]"),
        ({"a": 4, "n": 3}, r"axis 'a': 4 is not one of \[1, 2, 3\]"),
        ({"a": "1", "n": 3}, r"axis 'a': '1' is not one of"),
        ({"a": True, "n": 3}, r"axis 'a': True is not one of"),
        ({"a": 1, "n": 101}, r"axis 'n': 101 is outside the range \[1, 100\]"),
        ({"a": 1, "n": 2.5}, r"axis 'n': 2.5 is not an integer"),
    ],
)
def test_a_row_whose_axis_values_the_space_cannot_draw_is_a_usage_error(
    run, tmp_path, capsys, axes, message
):
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(spaces)
    row = _write_row(tmp_path, space="Toy/Default", axes=axes)
    assert _run_row(run, row, "--spaces-dir", str(spaces)) == 2
    assert re.search(f"Toy/Default.*{message}", capsys.readouterr().err)
    assert not run.work.exists()


def test_late_axes_are_plain_values_in_a_row(run, tmp_path):
    # bug caught: a row cannot name a late axis, or rebuilds it from a stream and ignores it
    spaces = tmp_path / "spaces"
    spaces.mkdir()
    write_space(
        spaces,
        axes={
            "a": {"values": [1, 2, 3]},
            "late_one": {"values": ["x", "y"], "late": True},
        },
        derive={},
        constraints=[],
        case={
            "values": {"a": "${a}", "late": "${late_one}"},
            "metadata": {"purpose": "probe"},
        },
    )
    for value in ("x", "y"):
        row = _write_row(
            tmp_path, space="Toy/Default", axes={"a": 2, "late_one": value}
        )
        assert _run_row(run, row, "--spaces-dir", str(spaces)) == 0
        written = json.loads(
            (run.work / "bundles/gauntlet/Toy/Default/sweep.json").read_text()
        )
        assert written["cases"][0]["values"] == {"a": 2, "late": value}


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"space": "NoSuch/Space"}, "no space 'NoSuch/Space'"),
        ({"space": None}, r"missing keys \['space'\]"),
        ({"axes": None}, r"missing keys \['axes'\]"),
        ({"data_seed": None}, r"missing keys \['data_seed'\]"),
        ({"axes": [1]}, "'axes' must be an object"),
        ({"space": 3}, "'space' must be an <Op>/<Space> string"),
        ({"data_seed": -1}, "data seed must be an integer in"),
        ({"data_seed": 2**32}, "data seed must be an integer in"),
        ({"data_seed": True}, "data seed must be an integer in"),
        ({"case_id": "a b"}, "case id must be a non-empty string"),
        ({"case_id": ""}, "case id must be a non-empty string"),
        ({"shape_seed": 5}, r"unknown keys \['shape_seed'\]"),
    ],
)
def test_a_malformed_row_file_is_a_usage_error(run, tmp_path, capsys, fields, message):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 3)
    spec = {**_row_of(sample), **fields}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({k: v for k, v in spec.items() if v is not None}))
    try:
        code = _run_row(run, path)
    except SystemExit as exit_:  # argparse's usage error
        code = exit_.code
    assert code == 2
    assert re.search(message, capsys.readouterr().err)
    assert not run.work.exists()


def test_a_missing_or_unparsable_row_file_is_a_usage_error(run, tmp_path, capsys):
    for path in (tmp_path / "absent.json", tmp_path / "text.json"):
        if path.name == "text.json":
            path.write_text("{not json")
        with pytest.raises(SystemExit) as exit_:
            _run_row(run, path)
        assert exit_.value.code == 2
        assert str(path) in capsys.readouterr().err
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(SystemExit) as exit_:
        _run_row(run, tmp_path / "list.json")
    assert exit_.value.code == 2 and "expected a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--samples", "4"),
        ("--seed", "5"),
        ("--data-seed", "6"),
        ("--filter", "^SdpaFwd/"),
        ("--time-budget", "1m"),
    ],
)
def test_row_cannot_be_combined_with_the_flags_that_draw_samples(
    run, tmp_path, capsys, flag, value
):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 3)
    row = _write_row(tmp_path, **_row_of(sample))
    with pytest.raises(SystemExit) as exit_:
        _run_row(run, row, flag, value)
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    assert "--row" in err and flag in err
    assert not run.work.exists() and not run.log.exists()


def test_the_sampling_environment_variables_do_not_conflict_with_row(
    run, tmp_path, monkeypatch
):
    # a variable left in the shell is a default, not a request to draw
    for name, value in (
        ("HIPDNN_FUZZ_SAMPLES", "9"),
        ("HIPDNN_FUZZ_SEED", "5"),
        ("HIPDNN_FUZZ_DATA_SEED", "6"),
        ("HIPDNN_FUZZ_FILTER", "^RMSNorm/"),
        ("HIPDNN_FUZZ_TIME_BUDGET", "1h"),
    ):
        monkeypatch.setenv(name, value)
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 3)
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample))) == 0
    assert run.summary["samples"] == 1 and run.summary["data_seed"] == sample.data_seed
    assert run.summary["time_budget_seconds"] is None
    assert [o["data_seed"] for o in run.outcome_list] == [sample.data_seed]


def test_a_row_that_does_not_load_is_unexercised_like_any_other_case(run, tmp_path):
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 3)
    run.control(cases={"gone": "load_skip"})
    assert _run_row(run, _write_row(tmp_path, **_row_of(sample, case_id="gone"))) == 0
    (outcome,) = run.outcome_list
    assert outcome["outcome"] == "UNEXERCISED" and outcome["reason"].startswith("load:")


# --------------------------------------------------------------------------
# sample_from_axes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("space_id", SPACE_IDS)
def test_sample_from_axes_renders_what_sample_renders_for_the_same_axis_values(
    space_id,
):
    # bug caught: the explicit entry point renders another row than the drawing one, so a
    # replayed or shrunk sample is a different test from the one that was drawn
    sp = space_by_id(space_id)
    drawn = sampler.sample(sp, 515)
    row = sampler.sample_from_axes(
        sp, drawn.axis_values, drawn.data_seed, drawn.case_id
    )
    assert row.shape_seed is None and row.case_id == "s515"
    assert row.axis_values == drawn.axis_values
    expected = copy.deepcopy(drawn.row)
    got = copy.deepcopy(row.row)
    assert got["metadata"].pop("notes") != expected["metadata"].pop("notes")
    assert got == expected


def test_sample_from_axes_does_not_share_the_callers_dict_or_lists(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path, axes={"a": {"values": [[1, 2]]}},
        derive={}, constraints=[], case={"values": {"a": "${a}"}, "metadata": {}}))  # fmt: skip
    axes = {"a": [1, 2]}
    row = sampler.sample_from_axes(sp, axes, 3)
    axes["a"].append(9)
    assert row.axis_values == {"a": [1, 2]} and row.row["values"] == {"a": [1, 2]}


@pytest.mark.parametrize("case_id", ["", "a b", "s-1", "x.y", 7, None])
def test_sample_from_axes_rejects_a_case_id_that_is_not_a_test_name(tmp_path, case_id):
    sp = space_lib.load_space(write_space(tmp_path))
    with pytest.raises(sampler.InvalidRowError, match="case id"):
        sampler.sample_from_axes(sp, {"a": 1, "n": 1}, 3, case_id)


def test_sample_from_axes_names_the_violated_rule_and_its_reason(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path))
    with pytest.raises(
        sampler.InvalidRowError,
        match=r"Toy/Default.*violates 'n >= a' \(n must reach a\)",
    ):
        sampler.sample_from_axes(sp, {"a": 3, "n": 2}, 3)
    with pytest.raises(ValueError, match="data seed"):
        sampler.sample_from_axes(sp, {"a": 1, "n": 2}, -3)
    # sample() is untouched: a seed still draws the same legal sample
    assert sampler.sample(sp, 5).axis_values == sampler.sample(sp, 5).axis_values


# --------------------------------------------------------------------------
# --shrink: smaller samples for WRONG and CRASH findings
# --------------------------------------------------------------------------

SHRINK_ENV_NAME = "HIPDNN_SHRINK_TEST_MARK"  # a HIPDNN_ variable replays carry


def _shrink_space(tmp_path, **overrides):
    """A toy space with a dtype, two ``values`` axes and a ``range`` axis. ``n`` must reach ``a``
    and is no multiple of 3, so some values of ``n`` are not legal."""
    spaces = tmp_path / "spaces"
    spaces.mkdir(exist_ok=True)
    decl = {
        "axes": {
            "dtype": {"values": ["half", "bfloat16"]},
            "a": {"values": [1, 2, 3]},
            "b": {"values": ["x", "y", "z"]},
            "n": {"range": [1, 100], "boundaries": [1, 100]},
        },
        "derive": {},
        "constraints": [
            {"rule": "n >= a", "reason": "n must reach a"},
            {"rule": "n % 3 != 0", "reason": "n is no multiple of 3"},
        ],
        "case": {
            "values": {"dtype": "${dtype}", "a": "${a}", "b": "${b}", "n": "${n}"},
            "metadata": {"purpose": "probe"},
        },
    }
    decl.update(overrides)
    write_space(spaces, **decl)
    return spaces


FOUND = {"dtype": "bfloat16", "a": 3, "b": "z", "n": 97}


def _when(behaviour, *conditions):
    """A stub ``match`` rule on rendered values: ``(path, op, value)`` conditions."""
    return {
        "where": [{"path": p, "op": op, "value": v} for p, op, v in conditions],
        "behaviour": behaviour,
    }


def _shrink_run(run, spaces, budget, axes=None, extra=()):
    """Run one toy row under ``--shrink budget``; returns the driver's exit status."""
    row = _write_row(
        run.tmp,
        space="Toy/Default",
        axes=FOUND if axes is None else axes,
        case_id="found",
        data_seed=11,
    )
    return _run_row(
        run, row, "--spaces-dir", str(spaces), "--shrink", str(budget), *extra
    )


def _shrunk_dir(run, case="found"):
    return run.failure_dir(case, space="Toy_Default") / "shrunk"


def _candidate_runs(run):
    """The stub's run invocations that came from a shrink candidate's sandbox."""

    def golden(invocation):
        argv = invocation["argv"]
        return Path(argv[argv.index("--golden-data-dir") + 1])

    return [
        i
        for i in run.invocations
        if i["mode"] == "run" and (run.work / "shrink") in golden(i).parents
    ]


def _shrunk_axes(run, case="found"):
    return json.loads((_shrunk_dir(run, case) / "axes.json").read_text())


def test_a_wrong_gated_on_a_dim_shrinks_the_range_axis_to_the_smallest_legal_value(
    run, tmp_path
):
    # bug caught: a shrink that stops at a plausible value, or that treats a value the space
    # forbids (39, a multiple of 3) as the answer, instead of the exact legal threshold
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.n", ">=", 39))])
    assert _shrink_run(run, spaces, 50) == 1
    shrunk = _shrunk_axes(run)
    assert shrunk["original"] == FOUND
    assert shrunk["shrunk"] == {"dtype": "half", "a": 1, "b": "x", "n": 40}
    assert shrunk["changed"] == ["dtype", "a", "b", "n"]
    candidates = _candidate_runs(run)
    # a binary search, not a scan of the 60 values above the threshold
    assert len(candidates) == shrunk["runs"] <= 12
    # what the shrunk sample is: the row file, and the rendered row of the sample it names
    row = json.loads((_shrunk_dir(run) / "sweep_row.json").read_text())
    assert row == {
        "space": "Toy/Default",
        "axes": shrunk["shrunk"],
        "data_seed": 11,
        "case_id": "found",
    }
    sp = space_lib.load_space(spaces / "Toy.space.json")
    shrunk_sample = sampler.sample_from_axes(sp, row["axes"], 11, "found")
    graph = json.loads((_shrunk_dir(run) / "Toy_Default_found.json").read_text())
    assert graph == {"nodes": [], "tensors": []}
    meta = json.loads((_shrunk_dir(run) / "Toy_Default_found.meta.json").read_text())
    assert meta["seed"] == 11 and meta["purpose"] == "regression"
    assert "shrunk" in meta["reference_source"]
    result = json.loads((_shrunk_dir(run) / "outcome.json").read_text())["result"]
    assert result["outcome"] == "WRONG"
    assert result["test"] == "gauntlet_Toy_Default_fp16_cases.found"
    assert shrunk_sample.axis_values == shrunk["shrunk"]


def test_a_wrong_gated_on_a_values_axis_keeps_that_value_and_simplifies_the_rest(
    run, tmp_path
):
    # bug caught: shrinking moves the axis that causes the failure to an earlier entry, or
    # leaves the axes that do not matter where they were
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.b", "==", "z"))])
    assert _shrink_run(run, spaces, 50) == 1
    shrunk = _shrunk_axes(run)
    assert shrunk["shrunk"] == {"dtype": "half", "a": 1, "b": "z", "n": 1}
    assert shrunk["changed"] == ["dtype", "a", "n"]
    summary_finding = run.summary["findings"][0]
    assert summary_finding["shrunk"] == {
        "runs": shrunk["runs"],
        "accepted_steps": shrunk["accepted_steps"],
        "axes_changed": ["dtype", "a", "n"],
    }
    assert shrunk["runs"] == len(_candidate_runs(run))


def test_a_candidate_that_changes_the_outcome_class_is_rejected(run, tmp_path):
    # bug caught: shrinking walks from a CRASH into a WRONG (any failure counts) and reports a
    # different defect than the one found
    spaces = _shrink_space(
        tmp_path, constraints=[{"rule": "n >= a", "reason": "n must reach a"}]
    )
    run.control(
        match=[
            _when("crash", ("values.n", ">=", 60)),
            _when("wrong", ("values.n", ">=", 20)),
        ]
    )
    assert _shrink_run(run, spaces, 50, axes={**FOUND, "b": "x"}) == 1
    outcomes = run.outcome_list
    assert [o["outcome"] for o in outcomes] == ["CRASH"]
    shrunk = _shrunk_axes(run)
    assert shrunk["shrunk"]["n"] == 60  # not 20, where the same rule set answers WRONG
    result = json.loads((_shrunk_dir(run) / "outcome.json").read_text())["result"]
    assert result["outcome"] == "CRASH" and result["crash_kind"] == "death"
    assert run.summary["findings"][0]["shrunk"]["axes_changed"] == ["dtype", "a", "n"]


def test_a_candidate_that_hangs_is_not_a_wrong(run, tmp_path):
    # bug caught: a hang is a CRASH, so a candidate that hangs must not stand in for a WRONG
    spaces = _shrink_space(
        tmp_path, constraints=[{"rule": "n >= a", "reason": "n must reach a"}]
    )
    run.control(
        match=[
            _when("wrong", ("values.n", ">=", 50)),
            _when("hang", ("values.n", ">=", 10)),
        ],
        hang_seconds=60,
    )
    assert _shrink_run(run, spaces, 50, extra=("--case-timeout", "1s")) == 1
    assert _shrunk_axes(run)["shrunk"]["n"] == 50
    result = json.loads((_shrunk_dir(run) / "outcome.json").read_text())["result"]
    assert result["outcome"] == "WRONG"


def test_the_shrink_budget_counts_candidate_runs_and_keeps_the_progress_so_far(
    run, tmp_path
):
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.n", ">=", 39))])
    assert _shrink_run(run, spaces, 3) == 1
    assert len(_candidate_runs(run)) == 3
    shrunk = _shrunk_axes(run)
    # dtype, a and b went to their first entries; n was not reached
    assert shrunk["shrunk"] == {"dtype": "half", "a": 1, "b": "x", "n": 97}
    assert shrunk["runs"] == 3 and shrunk["accepted_steps"] == 3
    assert run.summary["findings"][0]["shrunk"] == {
        "runs": 3,
        "accepted_steps": 3,
        "axes_changed": ["dtype", "a", "b"],
    }


def test_a_candidate_the_space_forbids_costs_no_budget(run, tmp_path):
    # bug caught: n=1 and n=2 break 'n >= a' while a is 3; counting them would spend the whole
    # budget before a could move, and the search would end on n=3
    spaces = _shrink_space(
        tmp_path,
        axes={"n": {"range": [1, 100], "boundaries": [1]}, "a": {"values": [1, 2, 3]}},
        constraints=[{"rule": "n >= a", "reason": "n must reach a"}],
        case={"values": {"a": "${a}", "n": "${n}"}, "metadata": {"purpose": "probe"}},
    )
    run.control(default="wrong")
    assert _shrink_run(run, spaces, 2, axes={"n": 3, "a": 3}) == 1
    assert _shrunk_axes(run)["shrunk"] == {"n": 1, "a": 1}
    assert len(_candidate_runs(run)) == 2


def test_shrinking_is_off_unless_asked_for(run, tmp_path):
    spaces = _shrink_space(tmp_path)
    run.control(default="wrong")
    assert _shrink_run(run, spaces, 0) == 1
    assert not (run.work / "shrink").exists()
    assert not _shrunk_dir(run).exists()
    assert "shrunk" not in run.summary["findings"][0]
    assert run.summary["shrink_budget"] == 0


def test_the_shrink_budget_comes_from_the_environment_and_a_flag_overrides_it(
    run, tmp_path, monkeypatch
):
    spaces = _shrink_space(tmp_path)
    run.control(default="wrong")
    monkeypatch.setenv("HIPDNN_FUZZ_SHRINK", "2")
    row = _write_row(
        run.tmp, space="Toy/Default", axes=FOUND, case_id="found", data_seed=11
    )
    assert _run_row(run, row, "--spaces-dir", str(spaces)) == 1
    assert run.summary["shrink_budget"] == 2 and len(_candidate_runs(run)) == 2
    shutil.rmtree(run.work)
    run.log.unlink()
    assert _run_row(run, row, "--spaces-dir", str(spaces), "--shrink", "0") == 1
    assert run.summary["shrink_budget"] == 0 and not (run.work / "shrink").exists()


def test_a_negative_shrink_budget_is_a_usage_error(run, tmp_path, capsys):
    row = _write_row(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        _run_row(run, row, "--shrink", "-1")
    assert exit_info.value.code == 2
    assert "--shrink must be >= 0" in capsys.readouterr().err


def test_candidates_stay_out_of_outcomes_samples_and_counts(run, tmp_path):
    # bug caught: a candidate run is counted as a sample, so the summary no longer reconciles
    # and a shrink changes the verdict of the run it is part of
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.n", ">=", 39))])
    seeds = range(5, 11)
    drawn = [
        sampler.sample(space_lib.load_space(spaces / "Toy.space.json"), s)
        for s in seeds
    ]
    wrong = [s.case_id for s in drawn if s.axis_values["n"] >= 39]
    assert wrong and len(wrong) < len(drawn)  # the seeds draw both kinds
    assert run.run("--samples", "6", "--spaces-dir", str(spaces), "--shrink", "40",
                   "--seed", "5", space=None) == 1  # fmt: skip
    candidates = _candidate_runs(run)
    assert len(candidates) > 0
    outcomes = run.outcome_list
    assert [o["case"] for o in outcomes] == [s.case_id for s in drawn]
    assert [o["case"] for o in outcomes if o["outcome"] == "WRONG"] == wrong
    assert len((run.work / "samples.jsonl").read_text().splitlines()) == 6
    summary = run.summary
    assert summary["samples"] == 6 and summary["reconciled"] is True
    assert sum(summary["outcomes"].values()) == 6
    assert summary["outcomes"]["WRONG"] == len(wrong)
    (engine,) = summary["engines"].values()
    assert engine["samples"] == 6
    # each finding was shrunk, in its own folder
    assert sorted(f["case"] for f in summary["findings"]) == sorted(wrong)
    assert all(f["shrunk"]["runs"] > 0 for f in summary["findings"])
    for case in wrong:
        assert (_shrunk_dir(run, case) / "replay.sh").is_file()
        assert (run.work / "shrink" / f"Toy_Default_{case}").is_dir()
    # no candidate left a failure folder of its own or a reports/logs file in the run's dirs
    assert sorted(p.name for p in (run.work / "failures").iterdir()) == sorted(
        f"Toy_Default_{c}" for c in wrong
    )
    assert len(list((run.work / "reports").glob("*-run.jsonl"))) == 1


def test_a_shrunk_row_replays_under_the_same_gtest_name_and_outcome(run, tmp_path):
    # bug caught: the replay of a shrunk sample lands in another suite than the shrink run, so
    # the engine TOML's tolerance and skip globs grade it differently (RFC 0023, A6)
    spaces = _shrink_space(tmp_path)
    run.monkeypatch.setenv(SHRINK_ENV_NAME, "marked")
    run.control(
        match=[
            _when("wrong", ("values.dtype", "==", "bfloat16"), ("values.n", ">=", 39))
        ]
    )
    assert _shrink_run(
        run,
        spaces,
        50,
        extra=("--test-engine", "ENG", "--test-article", "ART", "--test-config", "CFG"),
    ) == 1  # fmt: skip
    script = (_shrunk_dir(run) / "replay.sh").read_text()
    shrunk = _shrunk_axes(run)
    assert shrunk["shrunk"] == {"dtype": "bfloat16", "a": 1, "b": "x", "n": 40}
    final = json.loads((_shrunk_dir(run) / "outcome.json").read_text())["result"]
    assert final["test"] == "gauntlet_Toy_Default_bfp16_cases.found"
    assert final["test"] == run.outcome_list[0]["test"]  # the finding's own name

    # what replay.sh runs: the driver on the row file, with the run's binary, engine and config
    assert '--row "$D/sweep_row.json"' in script
    for token in ("--test-engine ENG", "--test-article ART", "--test-config CFG",
                  f"--binary {fuzz._quote(str(STUB))}", "--spaces-dir", f"{SHRINK_ENV_NAME}=marked"):  # fmt: skip
        assert token in script
    assert not any(flag in script for flag in ("--shrink", "--seed", "--filter"))

    # running that row reproduces the WRONG under the same GTest name
    replay_work = tmp_path / "replayed"
    done = fuzz.main(
        [
            "--binary", str(STUB), "--work-dir", str(replay_work), "--case-timeout", "30s",
            "--spaces-dir", str(spaces),
            "--row", str(_shrunk_dir(run) / "sweep_row.json"),
        ]
    )  # fmt: skip
    assert done == 1
    (again,) = [
        json.loads(line)
        for line in (replay_work / "outcomes.jsonl").read_text().splitlines()
    ]
    assert (again["test"], again["outcome"]) == (final["test"], "WRONG")
    sp = space_lib.load_space(spaces / "Toy.space.json")
    expected = sampler.sample_from_axes(sp, shrunk["shrunk"], 11, "found").row
    written = json.loads(
        (
            replay_work / "bundles/gauntlet/Toy/Default/bfp16_cases/sweep.json"
        ).read_text()
    )["cases"]
    assert json.dumps(written) == json.dumps([expected])


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_the_shrunk_replay_script_runs_and_reproduces_the_finding(run, tmp_path):
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.n", ">=", 39))])
    assert _shrink_run(run, spaces, 50) == 1
    folder = _shrunk_dir(run)
    done = subprocess.run(
        ["sh", str(folder / "replay.sh")], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 1, done.stdout + done.stderr
    (again,) = [
        json.loads(line)
        for line in (folder / "replay-work" / "outcomes.jsonl").read_text().splitlines()
    ]
    final = json.loads((folder / "outcome.json").read_text())["result"]
    assert (again["test"], again["outcome"]) == (final["test"], "WRONG")


def test_the_report_prints_one_line_per_shrunk_finding(run, tmp_path, capsys):
    spaces = _shrink_space(tmp_path)
    run.control(match=[_when("wrong", ("values.b", "==", "z"))])
    assert _shrink_run(run, spaces, 50) == 1
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("shrunk:")]
    assert len(lines) == 1
    assert 'dtype "bfloat16" -> "half"' in lines[0] and "n 97 -> 1" in lines[0]
    assert str(_shrunk_dir(run) / "replay.sh") in lines[0]


def test_a_finding_nothing_simpler_reproduces_still_gets_a_shrunk_folder(run, tmp_path):
    spaces = _shrink_space(tmp_path)
    axes = {"dtype": "half", "a": 1, "b": "x", "n": 1}
    run.control(default="wrong")
    assert _shrink_run(run, spaces, 5, axes=axes) == 1
    shrunk = _shrunk_axes(run)
    assert shrunk["shrunk"] == axes and shrunk["changed"] == []
    assert shrunk["runs"] == 0 and shrunk["accepted_steps"] == 0
    assert _candidate_runs(run) == []
    assert run.summary["findings"][0]["shrunk"]["axes_changed"] == []


def test_shrink_search_ends_on_a_value_the_oracle_accepted_even_when_it_is_not_monotonic(
    tmp_path,
):
    # the sampler-level search assumes values below a refused one are refused too; whatever the
    # oracle says, what it returns is a value it accepted and nothing is asked twice
    sp = space_lib.load_space(
        write_space(
            tmp_path,
            axes={"n": {"range": [1, 64], "boundaries": [1, 64]}},
            derive={},
            constraints=[],
            case={"values": {"n": "${n}"}, "metadata": {}},
        )
    )
    asked = []

    def oracle(candidate):
        n = candidate.axis_values["n"]
        asked.append(n)
        return n in (64, 48, 33)

    axes, steps = sampler.shrink(sp, {"n": 64}, 3, "c", oracle)
    assert (axes["n"], steps) == (48, 1)  # 33 lies below the refused 40
    assert len(asked) == len(set(asked))  # no candidate is asked twice
    assert 64 not in asked  # the original is never run again


# --------------------------------------------------------------------------
# Pairwise generation (pairwise.py)
# --------------------------------------------------------------------------

# The completeness tests cover SDPA forward: the spaces of its three engines (ASM_SDPA,
# HIP_FLASH2, rocKE dense) and the probe space. The generator runs on any space; for the others
# only that it returns (and what it does when its search is capped) is tested.
SDPA_FWD_SPACES = [
    "SdpaFwd/Default",
    "SdpaFwd/Flash2Region",
    "SdpaFwd/Gfx950Dense",
    "SdpaFwd/Flash2",
]
PROBE_SPACE = "SdpaFwd/Flash2"
PAIRWISE_SEED = 1


def _same_value(a, b):
    return type(a) is type(b) and a == b


def _rendered(sp, values):
    """The case an axis-value row renders to, as text, without the fields that only name the
    row (the id and the notes); None when the row is not legal."""
    try:
        row = sampler.sample_from_axes(sp, values, 0).row
    except (sampler.InvalidRowError, space_lib.SpaceError):
        return None
    shown = {k: v for k, v in row.items() if k != "id"}
    shown["metadata"] = {k: v for k, v in row["metadata"].items() if k != "notes"}
    return json.dumps(shown, sort_keys=True, default=repr)


def _live_axes(sp, levels, values):
    """The axes whose value matters in a legal row: some other level of the axis gives a legal
    row that renders differently."""
    base = _rendered(sp, values)
    assert base is not None, values
    live = set()
    for axis, options in levels.items():
        for level in options:
            if _same_value(level, values[axis]):
                continue
            alt = _rendered(sp, {**values, axis: level})
            if alt is not None and alt != base:
                live.add(axis)
                break
    return live


def _tuple_key(a, va, b, vb):
    return ((a, repr(va)), (b, repr(vb)))


def _tuple_universe(levels):
    return {
        _tuple_key(a, va, b, vb)
        for a, b in itertools.combinations(levels, 2)
        for va in levels[a]
        for vb in levels[b]
    }


def _held_live(levels, values, live):
    """The tuples of a row whose two values are levels and live in it."""
    held = [
        a
        for a in levels
        if a in live and any(_same_value(values[a], level) for level in levels[a])
    ]
    return {
        _tuple_key(a, values[a], b, values[b])
        for a, b in itertools.combinations(held, 2)
    }


def _covered(sp, levels, rows):
    covered = set()
    for values in rows:
        covered |= _held_live(levels, values, _live_axes(sp, levels, values))
    return covered


def _flagged(result, kind):
    return {
        _tuple_key(a, va, b, vb)
        for (a, va), (b, vb) in (finding.pair for finding in getattr(result, kind))
    }


def _brute_force(sp, levels):
    """Every legal row of the product of the levels, and from them the tuples some legal row
    holds with both values live (covered), the tuples no legal row holds (illegal) and the
    rest (dead). No pruning: only for a space of a few axes."""
    covered, held = set(), set()
    for combo in itertools.product(*levels.values()):
        values = dict(zip(levels, combo))
        if _rendered(sp, values) is None:
            continue
        live = _live_axes(sp, levels, values)
        for a, b in itertools.combinations(levels, 2):
            key = _tuple_key(a, values[a], b, values[b])
            held.add(key)
            if a in live and b in live:
                covered.add(key)
    universe = _tuple_universe(levels)
    return covered, universe - held, held - covered


# A toy space for each way a tuple can fail to be covered, small enough for the product.
TOY_PAIRWISE = {
    # lv is used only while use is "on": with use "off" it is inert, so (off, lv) is dead.
    "gated": {
        "decl": {
            "axes": {
                "use": {"values": ["off", "on"]},
                "lv": {"values": [1, 2, 3]},
                "other": {"values": [True, False]},
            },
            "derive": {"eff": "lv if use == 'on' else 0"},
            "constraints": [],
            "case": {
                "values": {"use": "${use}", "eff": "${eff}", "other": "${other}"},
                "metadata": {},
            },
        },
        "illegal": set(),
        "dead": {_tuple_key("use", "off", "lv", lv) for lv in (1, 2, 3)},
    },
    # a + b <= 4 forbids three of the nine (a, b) tuples and, at (1, 3) and (3, 1), every other
    # level of one value, which leaves it inert.
    "constrained": {
        "decl": {
            "axes": {
                "a": {"values": [1, 2, 3]},
                "b": {"values": [1, 2, 3]},
                "c": {"values": [0, 1]},
            },
            "derive": {},
            "constraints": [{"rule": "a + b <= 4", "reason": "the sum must fit"}],
            "case": {
                "values": {"a": "${a}", "b": "${b}", "c": "${c}"},
                "metadata": {},
            },
        },
        "illegal": {_tuple_key("a", a, "b", b) for a, b in ((2, 3), (3, 2), (3, 3))},
        # a stays at 1 (or 3) only with b at 3 (or 1): every other level of a is illegal there, so
        # a is inert, and (1, 3), (3, 1) are dead.
        "dead": {_tuple_key("a", a, "b", b) for a, b in ((1, 3), (3, 1))},
    },
    # n = 1 is legal only with m = 1 (so m is inert at n = 1: (1, 1) is dead); the interior
    # level of n meets every m.
    "ranged": {
        "decl": {
            "axes": {
                "n": {"range": [1, 40], "boundaries": [1, 40]},
                "m": {"values": [1, 2, 3]},
            },
            "derive": {},
            "constraints": [
                {"rule": "n != 1 or m == 1", "reason": "n = 1 needs m = 1"}
            ],
            "case": {"values": {"n": "${n}", "m": "${m}"}, "metadata": {}},
        },
        "illegal": {_tuple_key("n", 1, "m", m) for m in (2, 3)},
        "dead": {_tuple_key("n", 1, "m", 1)},
    },
    # All three at once: n >= 3 * eff, so n = 1 never goes with use "on" (illegal), and goes
    # with lv only where use is "off" and lv is inert (dead).
    "mixed": {
        "decl": {
            "axes": {
                "use": {"values": ["off", "on"]},
                "lv": {"values": [1, 2, 3]},
                "n": {"range": [1, 30], "boundaries": [1, 30]},
            },
            "derive": {"eff": "lv if use == 'on' else 0"},
            "constraints": [{"rule": "n >= eff * 3", "reason": "n must reach 3 * eff"}],
            "case": {
                "values": {"use": "${use}", "eff": "${eff}", "n": "${n}"},
                "metadata": {},
            },
        },
        "illegal": {_tuple_key("use", "on", "n", 1)},
        "dead": {_tuple_key("use", "off", "lv", lv) for lv in (1, 2, 3)}
        | {_tuple_key("lv", lv, "n", 1) for lv in (1, 2, 3)},
    },
}


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("toy", sorted(TOY_PAIRWISE))
def test_pairwise_matches_a_brute_force_over_the_whole_product_of_a_toy_space(
    tmp_path, toy, seed
):
    sp = space_lib.load_space(write_space(tmp_path, **TOY_PAIRWISE[toy]["decl"]))
    result = pairwise.build(sp, seed)
    covered, illegal, dead = _brute_force(sp, result.levels)
    # the rows cover exactly what can be covered, and nothing else is left unexplained
    assert _covered(sp, result.levels, result.rows) == covered
    assert _flagged(result, "illegal") == illegal
    assert _flagged(result, "dead") == dead
    assert result.unproven == []
    assert result.tuples_total == len(covered) + len(illegal) + len(dead)
    # and the brute force sees the cases each toy is there for
    assert TOY_PAIRWISE[toy]["illegal"] <= illegal
    assert TOY_PAIRWISE[toy]["dead"] <= dead
    if toy in ("gated", "constrained", "ranged"):
        assert (illegal, dead) == (
            TOY_PAIRWISE[toy]["illegal"],
            TOY_PAIRWISE[toy]["dead"],
        )
    for values in result.rows:
        assert _rendered(sp, values) is not None
        assert all(
            any(_same_value(v, level) for level in result.levels[a])
            for a, v in values.items()
        )


def test_a_gated_value_is_dead_not_covered_and_a_constrained_pair_is_illegal(tmp_path):
    (tmp_path / "gated").mkdir()
    gated = space_lib.load_space(
        write_space(tmp_path / "gated", **TOY_PAIRWISE["gated"]["decl"])
    )
    result = pairwise.build(gated, 1)
    dead = TOY_PAIRWISE["gated"]["dead"]
    assert _flagged(result, "dead") == dead
    assert result.illegal == [] and result.unproven == []
    assert all("never both live" in f.detail for f in result.dead)
    # legal rows hold each of these tuples, but lv is inert in all of them
    assert not dead & _covered(gated, result.levels, result.rows)
    for _a, (_b, value) in (f.pair for f in result.dead):
        legal = {"use": "off", "lv": value, "other": True}
        assert _rendered(gated, legal) is not None
        assert "lv" not in _live_axes(gated, result.levels, legal)

    (tmp_path / "constrained").mkdir()
    constrained = space_lib.load_space(
        write_space(tmp_path / "constrained", **TOY_PAIRWISE["constrained"]["decl"])
    )
    result = pairwise.build(constrained, 1)
    assert _flagged(result, "illegal") == TOY_PAIRWISE["constrained"]["illegal"]
    assert _flagged(result, "dead") == TOY_PAIRWISE["constrained"]["dead"]
    assert result.unproven == []
    assert all("a + b <= 4" in f.detail for f in result.illegal)


RANGED = TOY_PAIRWISE["ranged"]["decl"]


@pytest.mark.parametrize("seed", range(8))
def test_a_range_axis_has_its_boundaries_and_one_seeded_interior_level(tmp_path, seed):
    sp = space_lib.load_space(write_space(tmp_path, **RANGED))
    found = pairwise.levels(sp, "n", seed)
    assert found[:2] == [1, 40] and len(found) == 3 and 1 < found[2] < 40
    assert (
        pairwise.levels(sp, sp.axes["n"], seed)
        == found
        == pairwise.levels(sp, "n", seed)
    )
    assert pairwise.build(sp, seed).levels["n"] == found
    # the interior level meets every value of the other axis
    rows = pairwise.build(sp, seed).rows
    assert {r["m"] for r in rows if r["n"] == found[2]} == {1, 2, 3}


def test_the_interior_level_depends_on_the_seed(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path, **RANGED))
    assert len({pairwise.levels(sp, "n", seed)[2] for seed in range(8)}) > 1


def test_range_ends_are_levels_only_when_declared_and_values_are_deduplicated(tmp_path):
    sp = space_lib.load_space(
        write_space(
            tmp_path,
            axes={
                "n": {"range": [5, 9]},
                "k": {"range": [1, 100], "boundaries": [1, 7]},
                "w": {"values": ["none", "none", "sliding", "none"]},
                "t": {"values": [1, True, 1.0, 1]},
            },
            derive={},
            constraints=[],
            case={
                "values": {"n": "${n}", "k": "${k}", "w": "${w}", "t": "${t}"},
                "metadata": {},
            },
        )
    )
    n = pairwise.levels(sp, "n", 4)
    assert len(n) == 1 and 5 < n[0] < 9  # no declared boundary: the interior alone
    k = pairwise.levels(sp, "k", 4)
    assert k[:2] == [1, 7] and len(k) == 3 and 1 < k[2] < 7
    assert pairwise.levels(sp, "w", 4) == ["none", "sliding"]
    assert [repr(v) for v in pairwise.levels(sp, "t", 4)] == ["1", "True", "1.0"]


def test_only_pairwise_strength_is_built(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path))
    with pytest.raises(ValueError, match="strength"):
        pairwise.build(sp, 1, strength=3)


def test_sample_from_axes_names_its_generator_and_origin_and_defaults_to_the_fuzzer(
    tmp_path,
):
    sp = space_lib.load_space(write_space(tmp_path))
    plain = sampler.sample_from_axes(sp, {"a": 1, "n": 5}, 3).row["metadata"]
    assert plain["generator"] == "gauntlet-fuzz"
    assert plain["notes"].startswith(
        "gauntlet-fuzz sample: space Toy/Default, explicit row,"
    )
    named = sampler.sample_from_axes(
        sp,
        {"a": 1, "n": 5},
        3,
        "pw2",
        generator="gauntlet-pairwise",
        origin="pairwise seed 9, row 2",
    ).row
    assert (
        named["id"] == "pw2" and named["metadata"]["generator"] == "gauntlet-pairwise"
    )
    assert named["metadata"]["notes"].startswith(
        "gauntlet-pairwise sample: space Toy/Default, pairwise seed 9, row 2, axes"
    )


@pytest.fixture(scope="module")
def pairwise_builds():
    """``build(space_id, seed)`` -> (``pairwise.build`` result, seconds): each (space, seed) is
    built once for the whole module."""
    cache = {}

    def build(space_id, seed=PAIRWISE_SEED):
        if (space_id, seed) not in cache:
            started = time.perf_counter()
            result = pairwise.build(space_by_id(space_id), seed)
            cache[space_id, seed] = (result, time.perf_counter() - started)
        return cache[space_id, seed]

    return build


@pytest.mark.parametrize("space_id", SDPA_FWD_SPACES)
def test_sdpa_forward_rows_re_render_and_every_tuple_is_accounted_for(
    pairwise_builds, space_id
):
    result, seconds = pairwise_builds(space_id)
    sp = space_by_id(space_id)
    for k, values in enumerate(result.rows):
        sample = sampler.sample_from_axes(
            sp,
            values,
            7,
            f"pw{k}",
            generator="gauntlet-pairwise",
            origin=f"pairwise seed {PAIRWISE_SEED}, row {k}",
        )
        assert sample.axis_values == values
    assert len(result.rows) >= result.lower_bound
    assert result.unproven == []
    universe = _tuple_universe(result.levels)
    assert result.tuples_total == len(universe)
    illegal, dead = _flagged(result, "illegal"), _flagged(result, "dead")
    assert len(illegal) == len(result.illegal) and len(dead) == len(result.dead)
    assert not illegal & dead and (illegal | dead) <= universe
    print(
        f"\n{space_id}: {len(result.rows)} rows (lower bound {result.lower_bound}),"
        f" {result.tuples_total} tuples, {len(illegal)} illegal, {len(dead)} dead,"
        f" {len(result.unproven)} unproven, built in {seconds:.1f} s"
    )


@pytest.mark.parametrize("space_id", SDPA_FWD_SPACES)
def test_sdpa_forward_rows_cover_every_tuple_that_is_not_illegal_or_dead(
    pairwise_builds, space_id
):
    result, _ = pairwise_builds(space_id)
    sp = space_by_id(space_id)
    # liveness recomputed here from the rendered rows, swapping each value for the other levels
    covered = _covered(sp, result.levels, result.rows)
    flagged = _flagged(result, "illegal") | _flagged(result, "dead")
    missed = _tuple_universe(result.levels) - covered - flagged
    assert (
        not missed
    ), f"{len(missed)} tuples have no row with both values live: {sorted(missed)[:5]}"
    assert not covered & flagged, sorted(covered & flagged)[:5]


@pytest.mark.parametrize("space_id", SDPA_FWD_SPACES)
def test_no_random_legal_row_covers_a_tuple_the_build_calls_illegal_or_dead(
    pairwise_builds, space_id
):
    result, _ = pairwise_builds(space_id)
    sp = space_by_id(space_id)
    flagged = _flagged(result, "illegal") | _flagged(result, "dead")
    rng = random.Random(f"pairwise-refutation:{space_id}")
    names = list(result.levels)
    checked = 0
    while checked < 150:
        values = {a: copy.deepcopy(rng.choice(result.levels[a])) for a in names}
        if _rendered(sp, values) is None:
            continue
        checked += 1
        held = _held_live(result.levels, values, _live_axes(sp, result.levels, values))
        assert not held & flagged, (values, sorted(held & flagged)[:3])


def _expression_names(source):
    return {
        n.id
        for n in ast.walk(ast.parse(source.strip(), mode="eval"))
        if isinstance(n, ast.Name)
    }


def _substitutions(node):
    """The expression of every ``${...}`` string in a JSON tree."""
    if isinstance(node, str):
        if node.startswith("${") and node.endswith("}"):
            yield node[2:-1]
    elif isinstance(node, list):
        for v in node:
            yield from _substitutions(v)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _substitutions(v)


class _Prover:
    """Decides whether a legal row holds two values with both live, by exhaustive search. Coded
    apart from pairwise.py: it shares only the space's ``derive_env``, ``render`` and constraint
    expressions. A row is legal when no constraint is false; an axis is live in a row when
    another of its levels gives a legal row whose whole render differs. The search is plain
    recursion over the axes the question depends on (the axes read together with either value,
    and by the constraints that reach them); the others take the values of one legal row.
    """

    def __init__(self, sp, levels):
        self.sp = sp
        self.levels = levels
        self.axes = list(sp.axes)
        reads = {a: {a} for a in self.axes}
        for name, expr in sp.derive.items():
            reads[name] = set().union(
                *(reads[n] for n in _expression_names(expr.source) if n in reads)
            )

        def axes_of(source):
            return frozenset().union(
                *(reads[n] for n in _expression_names(source) if n in reads)
            )

        self.rules = [(c.expr, axes_of(c.rule)) for c in sp.constraints]
        case = json.loads(sp.path.read_text(encoding="utf-8"))["case"]
        self.groups = []
        for key in ("values", "metadata"):
            self.groups += [axes_of(s) for s in _substitutions(case.get(key, {}))]
        for key in ("add_tensors", "node_patches", "tensor_patches"):
            for entry in case.get(key, []):
                self.groups.append(
                    frozenset().union(*(axes_of(s) for s in _substitutions(entry)))
                )
        self.reads = reads
        direct = {n: _expression_names(e.source) for n, e in sp.derive.items()}
        self.readers = {
            a: [n for n, names in direct.items() if a in names] for a in self.axes
        }
        self.in_case = set()
        for source in _substitutions(case):
            self.in_case |= _expression_names(source)
        self.filler = sampler.sample(sp, 1).axis_values
        self.memo = {}

    def reach(self, axis):
        """The axes read together with ``axis`` by a constraint or a case group, and itself."""
        found = {axis}
        for scope in [s for _, s in self.rules] + self.groups:
            if axis in scope:
                found |= scope
        return frozenset(found)

    def relevant(self, fixed):
        found = set()
        for axis in fixed:
            found |= self.reach(axis)
        grew = True
        while grew:
            grew = False
            for _rule, scope in self.rules:
                if scope & found and not scope <= found:
                    found |= scope
                    grew = True
        return found

    def _is_live(self, axis, values):
        sp = self.sp
        try:
            base = json.dumps(sp.render(sp.derive_env(values)), sort_keys=True)
        except space_lib.SpaceError:
            return False
        rules = [rule for rule, scope in self.rules if axis in scope]
        for level in self.levels[axis]:
            if _same_value(level, values[axis]):
                continue
            try:
                env = sp.derive_env({**values, axis: level})
                if not all(rule.evaluate(env) for rule in rules):
                    continue
                if json.dumps(sp.render(env), sort_keys=True) != base:
                    return True
            except space_lib.SpaceError:
                continue
        return False

    def holds(self, rule, values):
        try:
            return bool(rule.evaluate(self.sp.derive_env(values)))
        except space_lib.SpaceError:
            return True  # not decidable on a partial row: the full row is checked at the end

    def _some_alt(self, axis, rules, watch, values):
        """Does some other level of ``axis`` keep every rule of ``rules`` true and, if ``watch``
        names derived values, change one of them? (Undecidable on a partial row: yes.)
        """
        sp = self.sp
        try:
            before = sp.derive_env(values)
        except space_lib.SpaceError:
            return True
        for level in self.levels[axis]:
            if _same_value(level, values[axis]):
                continue
            try:
                env = sp.derive_env({**values, axis: level})
                if not all(rule.evaluate(env) for rule in rules):
                    continue
            except space_lib.SpaceError:
                return True
            if watch is None or any(repr(env[n]) != repr(before[n]) for n in watch):
                return True
        return False

    def search(self, fixed, must_be_live):
        """A row of axis values with the ``fixed`` ones (axis -> value) that is legal and has
        every axis of ``must_be_live`` live; None if there is none."""
        relevant = self.relevant(fixed)
        pending = [
            ("legal", scope, rule) for rule, scope in self.rules if scope <= relevant
        ]
        for axis in must_be_live:
            # cheap necessary conditions first, each readable from a few axes: some other level
            # keeps one constraint true; some derived value that reads the axis changes (unless
            # the case names the axis itself); then liveness itself
            for rule, scope in self.rules:
                if axis in scope:
                    pending.append(("alt", scope | {axis}, (axis, [rule], None)))
            if axis not in self.in_case:
                readers = self.readers[axis]
                scope = frozenset({axis}).union(*(self.reads[n] for n in readers))
                pending.append(("alt", scope, (axis, [], readers)))
            pending.append(("live", self.reach(axis), axis))
        pending = [c for c in pending if c[1] <= relevant]
        assigned = set(fixed)
        stages = []

        def take():
            nonlocal pending
            due = [c for c in pending if c[1] <= assigned]
            pending = [c for c in pending if not c[1] <= assigned]
            stages.append(due)

        take()
        order = list(fixed)
        rest = sorted(relevant - assigned, key=self.axes.index)
        while rest:
            nxt = min(
                rest,
                key=lambda a: (
                    -sum(1 for c in pending if c[1] <= assigned | {a}),
                    -sum(1 for c in pending if a in c[1]),
                    len(self.levels[a]),
                ),
            )
            rest.remove(nxt)
            assigned.add(nxt)
            order.append(nxt)
            take()
        values = {**self.filler, **fixed}

        def passes(stage):
            for kind, scope, what in stage:
                if kind == "legal":
                    if not self.holds(what, values):
                        return False
                    continue
                key = (
                    kind,
                    (
                        what
                        if kind == "live"
                        else (
                            what[0],
                            tuple(id(r) for r in what[1]),
                            tuple(what[2] or ()),
                        )
                    ),
                    tuple(repr(values[a]) for a in sorted(scope)),
                )
                if key not in self.memo:
                    self.memo[key] = (
                        self._is_live(what, values)
                        if kind == "live"
                        else self._some_alt(*what, values)
                    )
                if not self.memo[key]:
                    return False
            return True

        def descend(depth):
            if depth == len(order):
                row = dict(values)
                try:
                    legal = self.sp.first_violation(self.sp.derive_env(row)) is None
                except space_lib.SpaceError:
                    legal = False
                return row if legal else None
            axis = order[depth]
            kept = values[axis]
            for level in self.levels[axis]:
                values[axis] = level
                if passes(stages[depth + 1 - len(fixed)]):
                    found = descend(depth + 1)
                    if found is not None:
                        return found
            values[axis] = kept
            return None

        if any(len(self.levels[a]) < 2 for a in must_be_live):
            return None  # an axis with one level has no other level to change to
        if not passes(stages[0]):
            return None
        return descend(len(fixed))

    def classify(self, a, va, b, vb):
        fixed = {a: va, b: vb}
        if self.search(fixed, [a, b]) is not None:
            return "feasible"
        return "illegal" if self.search(fixed, []) is None else "dead"


@pytest.mark.parametrize("space_id", SDPA_FWD_SPACES)
def test_every_illegal_or_dead_tuple_is_confirmed_by_an_independent_search(
    pairwise_builds, space_id
):
    result, _ = pairwise_builds(space_id)
    prover = _Prover(space_by_id(space_id), result.levels)
    started = time.perf_counter()
    for kind in ("illegal", "dead"):
        for finding in getattr(result, kind):
            (a, va), (b, vb) = finding.pair
            assert prover.classify(a, va, b, vb) == kind, (space_id, finding)
    print(
        f"\n{space_id}: {len(result.illegal)} illegal and {len(result.dead)} dead tuples"
        f" confirmed in {time.perf_counter() - started:.1f} s"
    )


def test_the_independent_search_finds_a_row_for_a_tuple_that_can_be_covered(
    pairwise_builds,
):
    result, _ = pairwise_builds("SdpaFwd/Default")
    prover = _Prover(space_by_id("SdpaFwd/Default"), result.levels)
    flagged = _flagged(result, "dead")
    checked = 0
    for a, b in itertools.combinations(result.levels, 2):
        va, vb = result.levels[a][0], result.levels[b][-1]
        if _tuple_key(a, va, b, vb) in flagged:
            continue
        assert prover.classify(a, va, b, vb) == "feasible", (a, va, b, vb)
        checked += 1
    assert checked > 50


@pytest.mark.parametrize("space_id", SDPA_FWD_SPACES)
def test_the_same_seed_gives_the_same_rows(pairwise_builds, space_id):
    first, _ = pairwise_builds(space_id)
    again = pairwise.build(space_by_id(space_id), PAIRWISE_SEED)
    assert again.rows == first.rows
    assert _flagged(again, "illegal") == _flagged(first, "illegal")
    assert _flagged(again, "dead") == _flagged(first, "dead")


def test_the_rows_do_not_depend_on_the_hash_seed(pairwise_builds):
    first, _ = pairwise_builds("SdpaFwd/Flash2Region")
    code = (
        "import hashlib, json, pairwise, space\n"
        "sp = next(s for s in space.load_spaces() if s.space_id == 'SdpaFwd/Flash2Region')\n"
        "rows = pairwise.build(sp, 1).rows\n"
        "print(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=HERE,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert (
        out
        == hashlib.sha256(json.dumps(first.rows, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize("space_id", ["SdpaFwd/Default", "SdpaFwd/Flash2Region"])
def test_a_different_seed_gives_a_different_row_set(pairwise_builds, space_id):
    one, _ = pairwise_builds(space_id)
    other, _ = pairwise_builds(space_id, PAIRWISE_SEED + 1)

    def as_set(result):
        return {json.dumps(r, sort_keys=True) for r in result.rows}

    assert as_set(one) != as_set(other)
    assert other.unproven == []
    # the classification does not depend on the seed (levels do not change what is feasible)
    assert len(other.rows) >= other.lower_bound


def test_the_probe_space_is_all_pairs_of_27_rows(pairwise_builds):
    result, _ = pairwise_builds(PROBE_SPACE)
    assert len(result.rows) == 27
    assert result.illegal == [] and result.dead == [] and result.unproven == []


def _coverage_space(tmp_path):
    return space_lib.load_space(
        write_space(
            tmp_path,
            axes={
                "use": {"values": ["off", "on"]},
                "lv": {"values": [1, 2]},
                "n": {"range": [1, 9], "boundaries": [1, 9]},
            },
            derive={"eff": "lv if use == 'on' else 0"},
            constraints=[],
            case={
                "values": {"use": "${use}", "eff": "${eff}", "n": "${n}"},
                "metadata": {},
            },
        )
    )


def test_pair_coverage_counts_drawn_passed_and_declined_tuples_of_hand_made_records(
    tmp_path,
):
    sp = _coverage_space(tmp_path)
    levels = pairwise.build(sp, 1).levels
    interior = levels["n"][2]
    off_level = next(v for v in (2, 3, 4, 5, 6) if v not in levels["n"])

    def row(use, lv, n):
        return {"use": use, "lv": lv, "n": n}

    records = [
        (row("on", 1, 1), "PASS"),
        (row("on", 2, 9), "DECLINED"),
        (row("off", 1, interior), "UNCLAIMED_PASS"),  # lv is inert here
        (row("on", 1, off_level), "PASS"),  # n holds no level
        (row("on", 2, 1), "WRONG"),
    ]
    coverage = pairwise.pair_coverage(sp, records, levels)

    def key(*pair):
        return tuple(pair)

    assert coverage["tuples"] == 4 + 6 + 6
    # drawn: (on,1) (on,2) | (on,n=1) (on,n=9) (off,n=interior) | (1,n=1) (2,n=9) (2,n=1)
    assert coverage["drawn"] == 8
    # passed: (on,1) (on,n=1) (off,n=interior) (1,n=1)
    assert coverage["passed"] == 4
    # drawn only by the DECLINED record: (on,n=9) and (2,n=9); (on,2) was also WRONG
    assert coverage["declined_only"] == 2
    assert coverage["never_passed"] == 4
    not_passed = coverage["not_passed"]
    assert len(not_passed) == 12
    assert (("use", "on"), ("lv", 1)) not in not_passed
    assert (("use", "on"), ("n", 1)) not in not_passed
    assert (("use", "off"), ("n", interior)) not in not_passed
    assert (("lv", 1), ("n", 1)) not in not_passed
    # an inert value is not counted: lv = 1 in the off record draws nothing with n = interior
    assert (("lv", 1), ("n", interior)) in not_passed
    assert (("use", "off"), ("lv", 1)) in not_passed
    # a value that is not a level is not counted either
    assert all(("n", off_level) not in pair for pair in not_passed)
    assert (("use", "on"), ("n", 9)) in not_passed and (
        ("lv", 2),
        ("n", 1),
    ) in not_passed


def test_pairwise_runs_on_a_space_outside_sdpa_forward(pairwise_builds):
    # RMSNorm is outside the completeness tests: the build only has to return rows that render
    result, seconds = pairwise_builds("RMSNorm/Default")
    sp = space_by_id("RMSNorm/Default")
    assert result.rows
    for k, values in enumerate(result.rows):
        sampler.sample_from_axes(sp, values, 7, f"pw{k}")
    print(
        f"\nRMSNorm/Default: {len(result.rows)} rows, {len(result.illegal)} illegal,"
        f" {len(result.dead)} dead, {len(result.unproven)} unproven, {seconds:.1f} s"
    )


def test_a_capped_search_reports_unproven_and_never_counts_the_tuple_covered(tmp_path):
    sp = space_lib.load_space(write_space(tmp_path, **TOY_PAIRWISE["mixed"]["decl"]))
    full = pairwise.build(sp, 1)
    assert full.unproven == []
    # a search that may visit no node settles only the tuples whose two axes decide them
    capped = pairwise.build(sp, 1, cap=0)
    assert capped.unproven
    flagged = _flagged(capped, "illegal") | _flagged(capped, "dead")
    unproven = _flagged(capped, "unproven")
    assert len(unproven) == len(capped.unproven) and not unproven & flagged
    covered = _covered(sp, capped.levels, capped.rows)
    assert not covered & (flagged | unproven)
    # every tuple is covered, proven impossible or reported unproven: none is dropped
    assert covered | flagged | unproven == _tuple_universe(capped.levels)


# --------------------------------------------------------------------------
# Pairwise runs (--pairwise, --generate-only, pair coverage)
# --------------------------------------------------------------------------

# A dtype axis (three sweep directories), two feature axes and a range axis; `n` must reach `a`,
# so (a=2, n=1) and (a=3, n=1) are illegal. Small enough to run in a second or two.
PW_TOY = {
    "axes": {
        "dtype": {"values": ["half", "bfloat16", "float"]},
        "a": {"values": [1, 2, 3]},
        "mode": {"values": ["p", "q", "r"]},
        "n": {"range": [1, 100], "boundaries": [1, 100]},
    },
    "derive": {},
    "constraints": [{"rule": "n >= a", "reason": "n must reach a"}],
    "case": {
        "values": {
            "dtype": "${dtype}",
            "a": "${a}",
            "mode": "${mode}",
            "n": "${n}",
        },
        "metadata": {"purpose": "probe"},
    },
}
PW_TOY_TAG = {"half": "fp16", "bfloat16": "bfp16", "float": "fp32"}
PW_ILLEGAL = [[["a", 2], ["n", 1]], [["a", 3], ["n", 1]]]


def _pw_spaces(tmp_path, *names):
    spaces = tmp_path / "spaces"
    spaces.mkdir(exist_ok=True)
    for name in names or ("Toy.Default.space.json",):
        write_space(spaces, file_name=name, **PW_TOY)
    return spaces


def _pw_space(spaces, name="Toy.Default.space.json"):
    return space_lib.load_space(spaces / name)


def _pw_run(run, spaces, *extra, seed=3):
    return run.run(
        "--pairwise", "--spaces-dir", str(spaces), *extra, space=None, seed=seed
    )


def _pw_samples(run):
    return [
        json.loads(line)
        for line in (run.work / "samples.jsonl").read_text().splitlines()
    ]


def _pw_report(work, name="Toy_Default"):
    return json.loads((Path(work) / "pairwise" / f"{name}.json").read_text())


def _pw_expected_coverage(sp, levels, samples, outcomes):
    """What a run drew and passed, from its samples and outcomes, worked out with the test's own
    liveness: ``(drawn, passed, declined_only)`` as sets of tuple keys."""
    drawn = {}
    for s in samples:
        for key in _held_live(levels, s["axes"], _live_axes(sp, levels, s["axes"])):
            drawn.setdefault(key, set()).add(outcomes[s["case"]])
    passed = {k for k, o in drawn.items() if o & {"PASS", "UNCLAIMED_PASS"}}
    declined = {k for k, o in drawn.items() if o == {"DECLINED"}}
    return drawn, passed, declined


def test_pairwise_runs_every_generator_row_once_as_pw_k_and_records_the_generator_and_seed(
    run, tmp_path, capsys
):
    # bug caught: a row dropped or run twice, a pairwise case named like a drawn one, a sweep row
    # that does not say which generator and seed made it, data seeds that are constant or change
    # from one process to the next, a summary that calls the run a sampling run
    spaces = _pw_spaces(tmp_path)
    sp = _pw_space(spaces)
    rows = pairwise.build(sp, 3).rows
    assert len(rows) > 8
    assert _pw_run(run, spaces, "--batch-size", "100") == 0
    out = capsys.readouterr().out
    assert re.search(r"pairwise Toy/Default: \d+ tuples, 2 illegal, \d+ dead", out)

    outcomes = run.outcome_list
    assert [o["case"] for o in outcomes] == [f"pw{k}" for k in range(len(rows))]
    assert len({o["test"] for o in outcomes}) == len(rows)
    assert [s["axes"] for s in _pw_samples(run)] == rows
    assert all(s["shape_seed"] is None for s in _pw_samples(run))
    sweeps = sorted(
        (run.work / "bundles/gauntlet/Toy/Default").glob("*_cases/sweep.json")
    )
    assert {f.parent.name for f in sweeps} == {
        f"{PW_TOY_TAG[r['dtype']]}_cases" for r in rows
    }
    cases = [c for f in sweeps for c in json.loads(f.read_text())["cases"]]
    assert sorted(c["id"] for c in cases) == sorted(f"pw{k}" for k in range(len(rows)))
    for case in cases:
        k = int(case["id"][2:])
        meta = case["metadata"]
        assert meta["generator"] == "gauntlet-pairwise"
        assert f"pairwise seed 3, row {k}," in meta["notes"]
        assert case["values"] == {**rows[k]}
    seeds = [o["data_seed"] for o in outcomes]
    assert all(0 <= d < 2**32 for d in seeds) and len(set(seeds)) > len(rows) // 2
    assert sorted(c["metadata"]["seed"] for c in cases) == sorted(seeds)

    summary = run.summary
    assert summary["mode"] == "pairwise" and summary["seed"] == 3
    assert summary["stopped_because"] == "pairwise-complete"
    assert summary["continuous"] is False and summary["samples_per_space"] is None
    assert summary["samples"] == len(rows) and summary["reconciled"] is True

    report = _pw_report(run.work)
    assert report["space"] == "Toy/Default" and report["seed"] == 3
    assert report["rows"] == len(rows) and report["lower_bound"] <= len(rows)
    assert report["levels"]["n"] == pairwise.levels(sp, "n", 3)
    assert report["tuples_total"] == 6 * 9
    assert [f["pair"] for f in report["illegal"]] == PW_ILLEGAL
    assert all("n >= a" in f["detail"] for f in report["illegal"])
    # n = 1 is legal only with a = 1, so a is inert there
    assert [f["pair"] for f in report["dead"]] == [[["a", 1], ["n", 1]]]
    assert report["unproven"] == []
    assert report["repair"]["rounds"] == [] and report["repair"]["max_rounds"] == 2
    assert report["timings"]["total_s"] >= 0

    # the same seed gives the same data seeds in another process; --data-seed fixes them all
    assert _pw_run(run, spaces, "--batch-size", "100") == 0
    assert [o["data_seed"] for o in run.outcome_list] == seeds
    assert _pw_run(run, spaces, "--data-seed", "77") == 0
    assert {o["data_seed"] for o in run.outcome_list} == {77}


def test_pairwise_rows_go_through_the_binary_in_batches(run, tmp_path):
    # bug caught: a batch that holds the whole set, or one that drops the rows past the batch size
    spaces = _pw_spaces(tmp_path)
    rows = pairwise.build(_pw_space(spaces), 3).rows
    assert _pw_run(run, spaces, "--batch-size", "4") == 0
    runs = [i["tests"] for i in run.invocations if i["mode"] == "run"]
    assert [len(r) for r in runs] == [4] * (len(rows) // 4) + [len(rows) % 4] * (
        len(rows) % 4 > 0
    )
    assert run.summary["samples"] == len(rows)


def test_a_pairwise_wrong_on_a_rendered_value_replays_by_row_under_the_same_gtest_name(
    run, tmp_path
):
    # bug caught: a failure folder that replays with the shape seed a pairwise row does not have,
    # or has no row.json for the --row it names, an repro.json without the row, a replay that
    # runs under another GTest name; the case ids all differ here, so only the rendered value can
    # make the stub answer WRONG
    spaces = _pw_spaces(tmp_path)
    sp = _pw_space(spaces)
    rows = pairwise.build(sp, 3).rows
    interior = pairwise.levels(sp, "n", 3)[2]
    run.control(match=[_when("wrong", ("values.n", "==", interior))])
    assert _pw_run(run, spaces) == 1
    wrong = [o["case"] for o in run.outcome_list if o["outcome"] == "WRONG"]
    assert wrong and wrong == [
        f"pw{k}" for k, r in enumerate(rows) if r["n"] == interior
    ]
    case = wrong[0]
    k = int(case[2:])
    folder = run.failure_dir(case, "Toy_Default")
    repro = json.loads((folder / "repro.json").read_text())
    assert repro["shape_seed"] is None and repro["axis_values"] == rows[k]
    assert repro["row"] == json.loads((folder / "row.json").read_text())
    assert repro["row"]["case_id"] == case and repro["row"]["axes"] == rows[k]
    assert repro["sweep_row"]["metadata"]["generator"] == "gauntlet-pairwise"
    argv = repro["replay"]["argv"]
    assert argv[argv.index("--row") + 1] == str(folder / "row.json")
    assert not {"--seed", "--filter", "--samples", "--pairwise"} & set(argv)
    script = (folder / "replay.sh").read_text()
    assert '--row "$D/row.json"' in script and "--filter" not in script
    replay = (folder / "replay.txt").read_text()
    assert "--row" in replay.splitlines()[0]
    assert f"axes: {json.dumps(rows[k], sort_keys=True)}" in replay
    meta = (folder / f"Toy_Default_{case}.meta.json").read_text()
    assert "explicit row" in meta and "shape seed" not in meta
    finding = next(f for f in run.summary["findings"] if f["case"] == case)
    assert "--row" in finding["replay"]

    _, work, done = _replay(folder)
    assert done.returncode == 1, done.stdout + done.stderr
    (again,) = [
        json.loads(line) for line in (work / "outcomes.jsonl").read_text().splitlines()
    ]
    original = next(o for o in run.outcome_list if o["case"] == case)
    assert (again["test"], again["outcome"]) == (original["test"], "WRONG")


def test_pairwise_load_reasons_of_different_rows_group_under_one_key(run, tmp_path):
    # bug caught: every pw<k> case id makes its own reason group, so a systematic load failure
    # reads as one-off findings
    spaces = _pw_spaces(tmp_path)
    run.control(cases={"pw1": "load_skip", "pw3": "load_skip"})
    assert _pw_run(run, spaces) == 0
    reasons = {}
    for engine in run.summary["engines"].values():
        reasons.update(engine["spaces"]["Toy/Default"]["unexercised_reasons"])
    assert len(reasons) == 1 and list(reasons.values()) == [2]
    assert "pw1" not in next(iter(reasons)) and "<case>" in next(iter(reasons))


def test_reason_key_masks_pairwise_ids_like_drawn_ones_and_nothing_longer(run):
    # bug caught: a pw<k> id left in a reason, or a longer word that holds one masked
    f = _fuzz(run, "--samples", "1")
    assert (
        f._reason_key("load: pw12 and s7 and pw3.")
        == "load: <case> and <case> and <case>."
    )
    assert f._reason_key("spw1 pw1a xpw2 pw s") == "spw1 pw1a xpw2 pw s"


def test_the_repair_round_retries_declined_only_tuples_from_levels_that_passed(
    run, tmp_path
):
    # bug caught: a repair row that uses a level that never passed (it would only decline again),
    # a coverable declined-only tuple left uncovered, a never-passing value retried or lost from
    # the report, a repair that does not stop at its round limit
    spaces = _pw_spaces(tmp_path)
    sp = _pw_space(spaces)
    # mode q is declined whatever it meets; (a=1, n=100) is declined whatever else the row holds
    run.control(
        match=[
            _when("declined", ("values.mode", "==", "q")),
            _when("declined", ("values.a", "==", 1), ("values.n", "==", 100)),
        ]
    )
    assert _pw_run(run, spaces, "--pairwise-repair", "3") == 0
    report = _pw_report(run.work)
    samples = _pw_samples(run)
    outcomes = {o["case"]: o["outcome"] for o in run.outcome_list}
    main = report["rows"]
    repair_samples = samples[main:]
    assert report["repair"]["rows"] == len(repair_samples) > 0
    assert [s["case"] for s in samples] == [f"pw{k}" for k in range(len(samples))]

    # round 1's targets, worked out from the main pass alone
    levels = report["levels"]
    drawn, passed, declined = _pw_expected_coverage(
        sp, levels, samples[:main], outcomes
    )
    q_tuples = {k for k in declined if ("mode", "'q'") in k}
    coverable = declined - q_tuples - {_tuple_key("a", 1, "n", 100)}
    assert q_tuples and coverable  # both kinds of target exist for this seed
    first = report["repair"]["rounds"][0]
    assert first["targets"] == len(declined) and first["first_row"] == main
    assert first["never_passed"] == len(q_tuples)
    assert first["retried"] == len(declined) - len(q_tuples)

    # the rows use only levels that passed in the main pass: never mode q, every other level
    passing = {axis: [] for axis in levels}
    for s in samples[:main]:
        if outcomes[s["case"]] == "PASS":
            for axis, value in s["axes"].items():
                passing[axis].append(value)
    assert "q" not in passing["mode"] and set(passing["mode"]) == {"p", "r"}
    for s in repair_samples:
        for axis, value in s["axes"].items():
            assert value in passing[axis], (s["case"], axis, value)

    # every coverable tuple passes now; what is left is mode q (never retried) and the one
    # tuple the stub always declines
    cov = run.summary["pair_coverage"]["Toy/Default"]
    assert cov["not_passed_total"] == len(cov["not_passed"]) < 50
    for pair in cov["not_passed"]:
        assert ["mode", "q"] in pair or pair == [["a", 1], ["n", 100]]
    assert [["a", 1], ["n", 100]] in cov["not_passed"]
    assert cov["repair_rows"] == len(repair_samples) and cov["complete"] is True
    assert report["repair"]["never_passed"] and all(
        ["mode", "q"] in pair for pair in report["repair"]["never_passed"]
    )
    assert {
        _tuple_key(a, va, b, vb)
        for (a, va), (b, vb) in report["repair"]["never_passed"]
    } == q_tuples

    # one round per limit step: the always-declined tuple keeps coming back
    rounds = report["repair"]["rounds"]
    assert [r["round"] for r in rounds] == [1, 2, 3] and all(r["rows"] for r in rounds)
    assert rounds[1]["first_row"] == rounds[0]["first_row"] + rounds[0]["rows"]
    assert run.summary["stopped_because"] == "pairwise-complete"


def test_no_repair_rows_are_built_with_a_zero_round_limit(run, tmp_path):
    spaces = _pw_spaces(tmp_path)
    run.control(match=[_when("declined", ("values.mode", "==", "q"))])
    assert _pw_run(run, spaces, "--pairwise-repair", "0") == 0
    report = _pw_report(run.work)
    assert report["repair"]["rounds"] == [] and report["repair"]["rows"] == 0
    assert len(run.outcome_list) == report["rows"]
    assert run.summary["pair_coverage"]["Toy/Default"]["repair_rows"] == 0


def test_pair_coverage_counts_range_axes_and_not_shrink_candidates(run, tmp_path):
    # bug caught: tuples of a range axis never drawn (the outcome axis map drops them), a shrink
    # candidate (same case id, other axis values) counted in place of the row it shrinks, drawn
    # not adding up to the universe less what no legal row covers
    spaces = _pw_spaces(tmp_path)
    sp = _pw_space(spaces)
    run.control(
        match=[
            _when("wrong", ("values.a", "==", 3)),
            _when("declined", ("values.mode", "==", "q")),
        ]
    )
    assert _pw_run(run, spaces, "--shrink", "3") == 1
    assert _candidate_runs(run)  # a candidate ran, with the finding's own case id
    report = _pw_report(run.work)
    cov = run.summary["pair_coverage"]["Toy/Default"]
    assert cov["tuples"] == report["tuples_total"]
    flagged = cov["illegal"] + cov["dead"] + cov["unproven"]
    assert cov["drawn"] == cov["tuples"] - flagged
    assert cov["rows"] == report["rows"] and cov["complete"] is True

    samples = _pw_samples(run)
    outcomes = {o["case"]: o["outcome"] for o in run.outcome_list}
    drawn, passed, declined = _pw_expected_coverage(
        sp, report["levels"], samples, outcomes
    )
    assert (cov["drawn"], cov["passed"], cov["declined_only"]) == (
        len(drawn),
        len(passed),
        len(declined),
    )
    assert cov["never_passed"] == len(drawn) - len(passed)
    assert cov["declined_only"] > 0 and cov["never_passed"] > cov["declined_only"]
    assert any("n" in (a, b) for (a, _), (b, _) in drawn)  # range-axis tuples count
    assert any("n" in (a, b) for (a, _), (b, _) in passed)
    assert cov["not_passed_total"] == cov["tuples"] - cov["passed"] - flagged
    assert len(cov["not_passed"]) == min(50, cov["not_passed_total"])


def test_a_pairwise_run_of_the_flash2_probe_space_passes_every_tuple(run):
    # a shipped SDPA forward space through the driver: all rows run, all 27 tuples pass
    assert run.run("--pairwise", "--seed", "1", space="SdpaFwd/Flash2") == 0
    cov = run.summary["pair_coverage"]["SdpaFwd/Flash2"]
    assert (cov["tuples"], cov["rows"], cov["unproven"]) == (27, 27, 0)
    assert cov["drawn"] == cov["passed"] == 27 - cov["illegal"] - cov["dead"]
    assert cov["declined_only"] == cov["never_passed"] == cov["not_passed_total"] == 0


def test_a_pairwise_run_stopped_by_the_time_budget_marks_its_coverage_incomplete(
    run, tmp_path
):
    # bug caught: a stopped run that reports its partial coverage as the whole set's
    spaces = _pw_spaces(tmp_path)
    run.control(delay=0.4)
    assert _pw_run(run, spaces, "--batch-size", "2", "--time-budget", "1s") == 0
    summary = run.summary
    assert summary["stopped_because"] == "time-budget" and summary["reconciled"] is True
    cov = summary["pair_coverage"]["Toy/Default"]
    assert cov["complete"] is False and 0 < summary["samples"] < cov["rows"]
    assert cov["drawn"] < cov["tuples"] - cov["illegal"] - cov["dead"] - cov["unproven"]


def test_generate_only_writes_every_row_of_every_space_in_every_dtype_tag_and_runs_nothing(
    run, tmp_path, capsys
):
    # bug caught: a writer that clears the bundle root per space or batch (only the last survives),
    # a dtype tag or a row past the batch size missing, a binary that runs after all
    spaces = _pw_spaces(tmp_path, "Toy.Default.space.json", "Toy.Other.space.json")
    work = tmp_path / "generated"
    code = fuzz.main(
        [
            "--pairwise", "--generate-only", "--seed", "5", "--batch-size", "4",
            "--work-dir", str(work), "--spaces-dir", str(spaces),
        ]
    )  # fmt: skip
    assert code == 0
    assert not run.log.exists() and not (work / "logs").exists()
    for name in ("Default", "Other"):
        sp = _pw_space(spaces, f"Toy.{name}.space.json")
        rows = pairwise.build(sp, 5).rows
        assert len(rows) > 4  # more rows than a batch holds
        sweeps = sorted(
            (work / "bundles/gauntlet/Toy" / name).glob("*_cases/sweep.json")
        )
        assert {f.parent.name for f in sweeps} == {
            f"{PW_TOY_TAG[r['dtype']]}_cases" for r in rows
        } and len(sweeps) == 3
        found = {}
        for sweep in sweeps:
            assert (sweep.parent / "graph.template.json").is_file()
            for case in json.loads(sweep.read_text())["cases"]:
                found[case["id"]] = (sweep.parent.name, case)
        assert sorted(found) == sorted(f"pw{k}" for k in range(len(rows)))
        for k, values in enumerate(rows):
            tag_dir, case = found[f"pw{k}"]
            assert tag_dir == f"{PW_TOY_TAG[values['dtype']]}_cases"
            assert case["values"] == values
            assert case["metadata"]["generator"] == "gauntlet-pairwise"
            assert f"pairwise seed 5, row {k}," in case["metadata"]["notes"]
        report = _pw_report(work, f"Toy_{name}")
        assert report["rows"] == len(rows) and report["seed"] == 5
        assert report["repair"] == {
            "max_rounds": 0, "rows": 0, "rounds": [], "never_passed": [],
        }  # fmt: skip
        assert [f["pair"] for f in report["illegal"]] == PW_ILLEGAL
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--pairwise", "--row", "{row}"], "cannot be combined with --row"),
        (["--pairwise", "--samples", "3"], "cannot be combined with --samples"),
        (["--pairwise", "--samples", "0"], "cannot be combined with --samples"),
        (["--generate-only"], "--generate-only needs --pairwise"),
        (["--pairwise-repair", "2"], "--pairwise-repair needs --pairwise"),
        (["--pairwise", "--pairwise-repair", "-1"], "--pairwise-repair must be >= 0"),
    ],
)
def test_pairwise_flag_clashes_are_usage_errors(run, tmp_path, capsys, flags, message):
    # bug caught: a pairwise run that quietly ignores a row file or a sample count it was given
    sample = sampler.sample(space_by_id("SdpaFwd/Default"), 3)
    row = _write_row(tmp_path, **_row_of(sample))
    argv = [f.replace("{row}", str(row)) for f in flags]
    with pytest.raises(SystemExit) as exit_:
        fuzz.main(run.args(*argv))
    assert exit_.value.code == 2
    assert message in capsys.readouterr().err
    assert not run.work.exists() and not run.log.exists()


def test_pairwise_ignores_the_samples_variable_and_needs_a_binary_unless_it_generates(
    monkeypatch, capsys
):
    monkeypatch.setenv("HIPDNN_FUZZ_SAMPLES", "9")  # a default of the random runs only
    a = fuzz.parse_args(["--binary", "bin", "--pairwise"])
    assert a.samples is None and a.pairwise_repair == 2 and a.seed is not None
    g = fuzz.parse_args(["--pairwise", "--generate-only"])
    assert g.binary is None and g.generate_only
    with pytest.raises(SystemExit) as exit_:
        fuzz.parse_args(["--pairwise"])
    assert exit_.value.code == 2 and "--binary" in capsys.readouterr().err


def test_the_offline_coverage_helper_counts_a_random_runs_first_samples_against_a_report(
    run, tmp_path, capsys
):
    # bug caught: a random run's coverage read from the wrong samples (other spaces, past N), a
    # range value that is no level counted, outcomes joined to the wrong case
    spaces = _pw_spaces(tmp_path)
    sp = _pw_space(spaces)
    generated = tmp_path / "generated"
    assert (
        fuzz.main(
            [
                "--pairwise", "--generate-only", "--seed", "3",
                "--work-dir", str(generated), "--spaces-dir", str(spaces),
            ]
        )  # fmt: skip
        == 0
    )
    capsys.readouterr()
    reports = generated / "pairwise"
    report = _pw_report(generated)
    run.control(seed_mod={"m": 3, "map": {"0": "declined", "1": "wrong"}})
    run.run("--samples", "24", "--spaces-dir", str(spaces), space=None)
    samples = _pw_samples(run)
    outcomes = {o["case"]: o["outcome"] for o in run.outcome_list}

    counts = {}
    for first in (None, 7):
        found = fuzz.coverage_of(run.work, reports, first, str(spaces))
        assert list(found) == ["Toy/Default"]
        cov = found["Toy/Default"]
        used = samples if first is None else samples[:first]
        drawn, passed, declined = _pw_expected_coverage(
            sp, report["levels"], used, outcomes
        )
        assert (cov["samples"], cov["with_outcome"]) == (len(used), len(used))
        assert (cov["drawn"], cov["passed"], cov["declined_only"]) == (
            len(drawn),
            len(passed),
            len(declined),
        )
        assert cov["tuples"] == report["tuples_total"]
        assert cov["illegal"] == len(report["illegal"]) == 2
        assert cov["not_passed_total"] == (
            cov["tuples"] - cov["passed"] - cov["illegal"] - cov["dead"]
        )
        counts[first] = cov
    assert 0 < counts[7]["drawn"] < counts[None]["drawn"]
    assert counts[None]["passed"] > 0
    capsys.readouterr()  # what the random run printed

    code = fuzz.main(
        [
            "--coverage-of", str(run.work),
            "--levels-from", str(reports / "Toy_Default.json"),
            "--first", "7", "--spaces-dir", str(spaces),
        ]
    )  # fmt: skip
    assert code == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps({"Toy/Default": counts[7]})
    )


def test_the_offline_coverage_helper_rejects_a_missing_report(run, tmp_path, capsys):
    assert (
        fuzz.main(
            [
                "--coverage-of",
                str(tmp_path),
                "--levels-from",
                str(tmp_path / "none.json"),
            ]
        )
        == 2
    )
    assert "--levels-from" in capsys.readouterr().err


def _hand_records(rows, declined):
    """A run's records for ``rows``: ``DECLINED`` where ``declined(row)``, else ``PASS``."""
    return [(row, "DECLINED" if declined(row) else "PASS") for row in rows]


def test_repair_rows_are_deterministic_use_only_passing_levels_and_cover_what_is_coverable(
    tmp_path,
):
    # bug caught: a repair row that holds a level that never passed, a coverable target left
    # without a live row, a value that never passed retried, rows that change between calls
    sp = space_lib.load_space(write_space(tmp_path, **PW_TOY))
    built = pairwise.build(sp, 3)
    records = _hand_records(built.rows, lambda r: r["mode"] == "q")
    found = pairwise.repair_targets(sp, records, built.levels, built.liveness)
    assert found.targets and found.witness["mode"] != "q"
    assert found.allowed["mode"] == ["p", "r"]
    assert found.allowed["n"] == built.levels["n"]

    def call(round_number):
        return pairwise.repair(
            sp,
            3,
            found.targets,
            found.allowed,
            round_number=round_number,
            witness=found.witness,
            liveness=built.liveness,
        )

    first = call(1)
    assert call(1).rows == first.rows
    assert first.rows and all(row["mode"] in ("p", "r") for row in first.rows)
    assert first.unreachable == []
    retried = {_tuple_key(a, va, b, vb) for (a, va), (b, vb) in first.retried}
    never = {_tuple_key(a, va, b, vb) for (a, va), (b, vb) in first.never_passed}
    assert never and all(("mode", "'q'") in key for key in never)
    assert retried and not retried & never
    assert retried | never == {
        _tuple_key(a, va, b, vb) for (a, va), (b, vb) in found.targets
    }
    assert retried <= _covered(sp, built.levels, first.rows)
    assert len(first.rows) < len(
        first.retried
    )  # rows are packed with several targets each


def test_a_target_only_the_declined_level_can_cover_is_unreachable_not_retried(
    tmp_path,
):
    # bug caught: a repair that builds a row from a level that never passed to reach a tuple (it
    # would decline again), or drops the tuple without saying so
    sp = space_lib.load_space(
        write_space(
            tmp_path,
            axes={
                "a": {"values": [1, 2]},
                "b": {"values": ["x", "y"]},
                "c": {"values": ["p", "q"]},
            },
            derive={},
            constraints=[
                {
                    "rule": "a != 2 or b == 'x' or c == 'q'",
                    "reason": "a = 2 with b = y needs c = q",
                }
            ],
            case={
                "values": {"a": "${a}", "b": "${b}", "c": "${c}"},
                "metadata": {},
            },
        )
    )
    built = pairwise.build(sp, 1)
    records = _hand_records(built.rows, lambda r: r["c"] == "q")
    found = pairwise.repair_targets(sp, records, built.levels, built.liveness)
    target = (("a", 2), ("b", "y"))
    assert target in found.targets and found.allowed["c"] == ["p"]
    result = pairwise.repair(
        sp, 1, found.targets, found.allowed, round_number=1,
        witness=found.witness, liveness=built.liveness,
    )  # fmt: skip
    # with c held at p, a and b are inert in (a=1, b=y) and (a=2, b=x) (the other value of a
    # needs c = q), and (a=2, b=y) is illegal: no row of the passing levels covers them
    unreachable = {f.pair: f.detail for f in result.unreachable}
    assert set(unreachable) == {(("a", 1), ("b", "y")), (("a", 2), ("b", "x")), target}
    assert "a != 2 or b == 'x' or c == 'q'" in unreachable[target]
    assert result.rows == [] and result.never_passed  # the c = q tuples are not retried
    assert all(("c", "q") in pair for pair in result.never_passed)


def test_a_repair_round_that_must_hold_a_tuple_it_avoids_still_builds_its_rows(
    run, tmp_path
):
    # bug caught: round 2 avoids the tuples round 1 retried without success; a row whose every
    # allowed level of some axis holds one of them scored below the -1 the level search started
    # from, chose no level and the whole run died with "Cannot choose from an empty sequence"
    spaces = _pw_spaces(tmp_path)
    run.control(
        match=[
            _when("declined", ("values.a", "==", 1), ("values.dtype", "==", "half")),
            _when(
                "declined",
                ("values.mode", "==", "p"),
                ("values.a", "==", 3),
                ("values.dtype", "==", "half"),
            ),
        ]
    )
    assert _pw_run(run, spaces, "--pairwise-repair", "2", seed=2) == 0
    assert run.summary["stopped_because"] == "pairwise-complete"
    cov = run.summary["pair_coverage"]["Toy/Default"]
    assert cov["complete"] is True
    report = _pw_report(run.work)
    samples = _pw_samples(run)
    rounds = report["repair"]["rounds"]
    assert [r["round"] for r in rounds] == [1, 2] and rounds[1]["retried"] > 0
    for r in rounds:
        # every target is retried or never passed; an unreachable one was retried
        assert r["targets"] == r["retried"] + r["never_passed"]
        assert r["unreachable"] <= r["retried"] and r["rows"] > 0
    assert rounds[0]["first_row"] == report["rows"]
    assert rounds[1]["first_row"] == report["rows"] + rounds[0]["rows"]
    assert report["repair"]["rows"] == sum(r["rows"] for r in rounds)
    assert len(samples) == report["rows"] + report["repair"]["rows"]
    assert cov["repair_rows"] == report["repair"]["rows"]
    assert len(run.outcome_list) == len(samples)


def test_repair_rows_of_a_level_every_other_level_avoids_are_built_not_lost(tmp_path):
    # bug caught: the same crash as above, found by calling repair directly: `a` may only be 3
    # (the one level that passed) and the stubborn tuple (dtype half, a 3) is held by every row
    # of it, so a row for (dtype half, mode p) cannot avoid it
    sp = space_lib.load_space(write_space(tmp_path, **PW_TOY))
    built = pairwise.build(sp, 3)
    allowed = {name: list(levels) for name, levels in built.levels.items()}
    allowed["a"] = [3]
    target = (("dtype", "half"), ("mode", "p"))
    stubborn = (("dtype", "half"), ("a", 3))
    result = pairwise.repair(
        sp, 3, [target, stubborn], allowed, round_number=2,
        stubborn=[stubborn], liveness=built.liveness,
    )  # fmt: skip
    assert result.unreachable == [] and result.never_passed == []
    assert result.rows and all(row["a"] == 3 for row in result.rows)
    covered = _covered(sp, built.levels, result.rows)
    assert _tuple_key("dtype", "half", "mode", "p") in covered
    assert _tuple_key("dtype", "half", "a", 3) in covered  # its own row


def test_repair_with_no_allowed_level_for_an_axis_retries_nothing_and_says_why(
    tmp_path,
):
    # bug caught: an axis whose allowed set is empty (no level of it ever passed) with targets
    # that do not mention it, which asked the level search for a level of that axis
    sp = space_lib.load_space(write_space(tmp_path, **PW_TOY))
    built = pairwise.build(sp, 3)
    allowed = {name: list(levels) for name, levels in built.levels.items()}
    allowed["mode"] = []
    with_mode = (("a", 1), ("mode", "p"))
    without_mode = (("dtype", "half"), ("a", 2))
    result = pairwise.repair(
        sp, 3, [with_mode, without_mode], allowed, round_number=1,
        liveness=built.liveness,
    )  # fmt: skip
    assert result.rows == []
    assert result.never_passed == [with_mode]  # needs a level of mode: never passed
    assert result.retried == [without_mode]
    assert [(f.pair, "'mode'" in f.detail) for f in result.unreachable] == [
        (without_mode, True)
    ]
    # an axis left out of the map has no allowed level either
    del allowed["mode"]
    again = pairwise.repair(
        sp, 3, [with_mode, without_mode], allowed, round_number=1,
        liveness=built.liveness,
    )  # fmt: skip
    assert again.rows == [] and again.never_passed == [with_mode]
    assert [f.pair for f in again.unreachable] == [without_mode]
