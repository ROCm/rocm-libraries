#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Self-test for the migration toolchain on a synthetic fixture.

Creates 3 synthetic graphs: 2 isomorphic (same topology, different knobs)
and 1 distinct. Verifies:
  1. skeleton_hash groups the 2 isomorphic graphs together
  2. place_bundles produces 1 sweep (2 cases) + 1 standalone
  3. verify_migration round-trip passes for all 3
  4. import_graph detects an exact duplicate (skip) and a new case (append)

No C++ binary needed — everything is pure Python on synthetic data.

Usage::

    python3 test_migration.py [-v]
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def _make_graph(node_type, tensor_uids, dims, strides, data_type, io_data_type=None):
    """Build a minimal synthetic graph dict."""
    nodes = [
        {
            "type": f"{node_type}Attributes",
            "name": "",
            "inputs": {f"x_tensor_uid": tensor_uids[0]},
            "outputs": {f"y_tensor_uid": tensor_uids[-1]},
        }
    ]
    tensors = []
    for uid in tensor_uids:
        tensors.append(
            {
                "uid": uid,
                "name": "",
                "dims": dims,
                "strides": strides,
                "data_type": data_type,
                "virtual": False,
            }
        )
    graph = {
        "nodes": nodes,
        "tensors": tensors,
        "io_data_type": io_data_type or data_type,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }
    return graph


def _write_captured(capture_dir, suite, case_name, graph, meta=None):
    """Write a captured case in the Hop A directory structure."""
    case_dir = capture_dir / suite / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / f"{case_name}.json", "w") as f:
        json.dump(graph, f, indent=2)
    if meta is None:
        meta = {"format_version": 1, "seed": 42}
    with open(case_dir / f"{case_name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run(args, check=True, cwd=None):
    """Run a subprocess and return its result."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=cwd or SCRIPT_DIR,
    )
    if check and result.returncode != 0:
        print(f"  FAIL: {' '.join(str(a) for a in args)}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
    return result


def test_skeleton_grouping():
    """2 isomorphic graphs get the same skeleton hash; 1 distinct differs."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_utils import skeleton_hash

    g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
    g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
    g3 = _make_graph("Conv", [0, 1, 2], [2, 3, 4, 5], [60, 20, 5, 1], "float")

    h1 = skeleton_hash(g1)
    h2 = skeleton_hash(g2)
    h3 = skeleton_hash(g3)

    assert h1 == h2, f"isomorphic graphs should match: {h1} vs {h2}"
    assert h1 != h3, f"distinct graphs should differ: {h1} vs {h3}"
    print("  PASS: skeleton_grouping")


def test_place_and_verify():
    """End-to-end: capture -> place -> verify."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        bundle_dir = tmp / "bundles"

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
        g3 = _make_graph("Conv", [0, 1, 2], [2, 3, 4, 5], [60, 20, 5, 1], "float")

        _write_captured(capture_dir, "Smoke/IntegrationGpuReluFp32", "small_fp32", g1)
        _write_captured(capture_dir, "Smoke/IntegrationGpuReluFp16", "small_fp16", g2)
        _write_captured(capture_dir, "Smoke/IntegrationGpuConvFp32", "small_conv", g3)

        # Hop B: place
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "place_bundles.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"

        # Check output structure: should have 1 sweep (2 cases) + 1 standalone
        sweeps = list(bundle_dir.rglob("sweep.json"))
        templates = list(bundle_dir.rglob("graph.template.json"))
        standalones = [
            p
            for p in bundle_dir.rglob("*.json")
            if p.name not in ("sweep.json", "graph.template.json")
            and not p.name.endswith(".meta.json")
        ]

        assert len(sweeps) == 1, f"expected 1 sweep, got {len(sweeps)}: {sweeps}"
        assert len(templates) == 1, f"expected 1 template, got {len(templates)}"
        assert len(standalones) == 1, f"expected 1 standalone, got {len(standalones)}"

        with open(sweeps[0]) as f:
            sweep = json.load(f)
        assert (
            len(sweep["cases"]) == 2
        ), f"expected 2 sweep cases, got {len(sweep['cases'])}"

        # Hop C: verify
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "verify_migration.py"),
                "--capture-dir",
                str(capture_dir),
                "--bundle-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"verify_migration failed: {r.stderr}"

        print("  PASS: place_and_verify")


def test_import_dedup():
    """import_graph detects exact dups and appends new cases."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
        g_dup = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")

        # First import
        graph_path = tmp / "g1.json"
        with open(graph_path, "w") as f:
            json.dump(g1, f)

        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(graph_path),
                "--bundle-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"first import failed: {r.stderr}"

        # Import exact dup -> should skip
        dup_path = tmp / "g_dup.json"
        with open(dup_path, "w") as f:
            json.dump(g_dup, f)

        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(dup_path),
                "--bundle-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"dup import should skip (rc=0): {r.stderr}"
        assert "DUPLICATE" in r.stderr, "should report DUPLICATE"

        # Import exact dup with --strict -> should fail
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(dup_path),
                "--bundle-dir",
                str(bundle_dir),
                "--strict",
            ],
            check=False,
        )
        assert r.returncode == 1, "strict dup should exit 1"

        # Import structural match with new knobs -> should append
        g2_path = tmp / "g2.json"
        with open(g2_path, "w") as f:
            json.dump(g2, f)

        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(g2_path),
                "--bundle-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"new-case import failed: {r.stderr}"
        assert "appended" in r.stderr, "should report appended"

        # Verify the sweep now has 2 cases
        sweeps = list(bundle_dir.rglob("sweep.json"))
        assert len(sweeps) == 1
        with open(sweeps[0]) as f:
            sweep = json.load(f)
        assert (
            len(sweep["cases"]) == 2
        ), f"expected 2 cases after append, got {len(sweep['cases'])}"

        print("  PASS: import_dedup")


def test_round_trip_expansion():
    """expand(template, values) reproduces the original graph exactly."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_utils import canon, expand

    g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")

    template = {
        "nodes": [
            {
                "type": "ReluAttributes",
                "name": "",
                "inputs": {"x_tensor_uid": 0},
                "outputs": {"y_tensor_uid": 1},
            }
        ],
        "tensors": [
            {
                "uid": 0,
                "name": "",
                "dims": "${case.dims}",
                "strides": "${case.strides}",
                "data_type": "${case.data_type}",
                "virtual": False,
            },
            {
                "uid": 1,
                "name": "",
                "dims": "${case.dims}",
                "strides": "${case.strides}",
                "data_type": "${case.data_type}",
                "virtual": False,
            },
        ],
        "io_data_type": "${case.io_data_type}",
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }

    values = {
        "io_data_type": "float",
        "tensors": [
            {
                "uid": 0,
                "dims": [2, 3, 4, 5],
                "strides": [60, 20, 5, 1],
                "data_type": "float",
            },
            {
                "uid": 1,
                "dims": [2, 3, 4, 5],
                "strides": [60, 20, 5, 1],
                "data_type": "float",
            },
        ],
    }

    expanded = expand(template, values)
    assert canon(expanded) == canon(g), "round-trip expansion mismatch"
    print("  PASS: round_trip_expansion")


def test_case_ids():
    """Case ids are stable, bounded, and unique even for input-range-only diffs."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_utils import assign_case_ids

    def case(dims, strides, dtype, inputs=None, attrs=None):
        v = {
            "io_data_type": dtype,
            "tensors": [
                {"uid": 0, "dims": dims, "strides": strides, "data_type": dtype}
            ],
        }
        if attrs:
            v["attributes"] = attrs
        return {"values": v, "metadata": {"seed": 1, "inputs": inputs or {}}}

    # 1. Stability: same input list, run twice -> identical ids, order-independent.
    a = [
        case([2, 3, 4, 5], [60, 20, 5, 1], "float"),
        case([8, 3, 4, 5], [60, 20, 5, 1], "half"),
    ]
    b = [dict(c) for c in reversed(a)]  # reversed order
    assign_case_ids(a)
    assign_case_ids(b)
    ids_a = {id(x): x["id"] for x in a}
    # reversed list must yield the same id per identical case content
    assert a[0]["id"] != a[1]["id"], "distinct cases must get distinct ids"
    # re-run on a fresh copy -> identical
    a2 = [
        case([2, 3, 4, 5], [60, 20, 5, 1], "float"),
        case([8, 3, 4, 5], [60, 20, 5, 1], "half"),
    ]
    assign_case_ids(a2)
    assert [c["id"] for c in a] == [c["id"] for c in a2], "ids must be deterministic"

    # 2. Input-range-only difference: identical graph, different bias range.
    #    Must NOT collide — this is the case the readable tokens cannot express.
    r1 = case([2, 3, 4, 5], [60, 20, 5, 1], "float", inputs={"1": {"lo": -1, "hi": 1}})
    r2 = case([2, 3, 4, 5], [60, 20, 5, 1], "float", inputs={"1": {"lo": -2, "hi": 2}})
    pair = [r1, r2]
    assign_case_ids(pair)
    assert pair[0]["id"] != pair[1]["id"], (
        "cases differing only in input range must get distinct ids "
        f"(got {pair[0]['id']!r} twice)"
    )

    # 3. Uniqueness across a larger mixed sweep.
    many = [
        case(
            [1, 16, 3, 3], [144, 9, 3, 1], "half", attrs={"parameters__stride": [s, s]}
        )
        for s in (1, 2)
    ] + [r1, r2]
    assign_case_ids(many)
    assert len({c["id"] for c in many}) == len(many), "all ids must be unique"

    # 4. Bounded length: no id may be an unusable gtest name.
    assert all(len(c["id"]) <= 120 for c in many), "ids must stay bounded"

    print("  PASS: case_ids")


def test_inputs_uid_canonicalized_by_name():
    """Cases with shuffled auto-assigned UIDs compress, fill specs follow names.

    The C++ builder assigns UIDs non-deterministically (GraphTensorIds.hpp
    iterates an unordered_set), so two captures of the SAME topology can label
    the same logical tensor with different UIDs. This fixture reproduces that:
    both cases are the same ternary op, but case1 has x/dy/scale UIDs shuffled
    relative to case0, with each case's fill-spec map keyed to match itself.

    place_bundles canonicalizes UIDs by tensor NAME, so:
      * both cases collapse into ONE sweep (compression works despite shuffle),
      * each fill spec stays bound to its tensor (x keeps [-1,1] etc.) even
        though it moved to a different UID.
    Verified per-tensor by re-keying the placed inputs by name.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        bundle_dir = tmp / "bundles"

        def ternary_graph(uids, dims, strides):
            """uids = {name: uid}; out is virtual."""
            return {
                "nodes": [
                    {
                        "type": "TernaryAttributes",
                        "name": "",
                        "inputs": {
                            "dy_tensor_uid": uids["dy"],
                            "scale_tensor_uid": uids["scale"],
                            "x_tensor_uid": uids["x"],
                        },
                        "outputs": {"out_tensor_uid": uids["out"]},
                    }
                ],
                "tensors": [
                    {
                        "uid": uids[nm],
                        "name": nm,
                        "dims": dims,
                        "strides": strides,
                        "data_type": "float",
                        "virtual": (nm == "out"),
                    }
                    for nm in ("x", "dy", "scale", "out")
                ],
                "io_data_type": "float",
                "compute_data_type": "float",
                "intermediate_data_type": "float",
                "name": "",
            }

        # Fill spec per tensor, keyed by that case's own UID for that tensor.
        specs = {
            "x": {"kind": "free", "lo": -1.0, "hi": 1.0},
            "dy": {"kind": "free", "lo": -0.1, "hi": 0.1},
            "scale": {"kind": "free", "lo": -0.2, "hi": 0.2},
        }

        def meta_for(uids):
            return {
                "format_version": 1,
                "seed": 7,
                "inputs": {str(uids[nm]): specs[nm] for nm in ("x", "dy", "scale")},
            }

        # case0 and case1 describe the same op with DIFFERENT UID assignments.
        uids0 = {"x": 1, "dy": 2, "scale": 3, "out": 4}
        uids1 = {"x": 3, "dy": 4, "scale": 1, "out": 2}  # shuffled
        _write_captured(
            capture_dir,
            "Smoke/IntegrationGpuTernary",
            "case0",
            ternary_graph(uids0, [2, 3, 4, 5], [60, 20, 5, 1]),
            meta=meta_for(uids0),
        )
        _write_captured(
            capture_dir,
            "Smoke/IntegrationGpuTernary",
            "case1",
            ternary_graph(uids1, [4, 6, 8, 10], [480, 80, 10, 1]),
            meta=meta_for(uids1),
        )

        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "place_bundles.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"

        # Compression: shuffled-UID siblings must land in ONE sweep, not two.
        sweeps = list(bundle_dir.rglob("sweep.json"))
        standalones = [
            p
            for p in bundle_dir.rglob("*.json")
            if p.name not in ("sweep.json", "graph.template.json")
            and not p.name.endswith(".meta.json")
        ]
        assert len(sweeps) == 1, f"expected 1 sweep (compression), got {sweeps}"
        assert not standalones, f"no standalone fallback expected: {standalones}"
        with open(sweeps[0]) as f:
            sweep = json.load(f)
        assert len(sweep["cases"]) == 2, f"expected 2 cases, got {len(sweep['cases'])}"

        # Fill spec must follow the tensor by name, not by original UID. Re-key
        # placed inputs through the template's uid->name map and check each case.
        templates = list(bundle_dir.rglob("graph.template.json"))
        with open(templates[0]) as f:
            tpl = json.load(f)
        uid_to_name = {t["uid"]: t["name"] for t in tpl["tensors"]}
        for case in sweep["cases"]:
            by_name = {
                uid_to_name[int(u)]: spec
                for u, spec in case["metadata"]["inputs"].items()
            }
            assert by_name["x"] == specs["x"], f"x spec moved: {by_name}"
            assert by_name["dy"] == specs["dy"], f"dy spec moved: {by_name}"
            assert by_name["scale"] == specs["scale"], f"scale spec moved: {by_name}"

        # Hop C must pass (it compares metadata inputs by name).
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "verify_migration.py"),
                "--capture-dir",
                str(capture_dir),
                "--bundle-dir",
                str(bundle_dir),
            ]
        )
        assert r.returncode == 0, f"verify_migration failed: {r.stderr}"

        print("  PASS: inputs_uid_canonicalized_by_name")


def test_import_inputs_uid_canonicalized_by_name():
    """import_graph canonicalizes UIDs by name so imports match placed sweeps.

    A graph captured with one UID assignment and re-imported with a shuffled
    assignment must dedup against the existing case (same topology, same fill
    specs per tensor) rather than appearing as a spurious new case. This mirrors
    test_inputs_uid_canonicalized_by_name for the incremental import path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        def ternary_graph(uids):
            return {
                "nodes": [
                    {
                        "type": "TernaryAttributes",
                        "name": "",
                        "inputs": {
                            "dy_tensor_uid": uids["dy"],
                            "scale_tensor_uid": uids["scale"],
                            "x_tensor_uid": uids["x"],
                        },
                        "outputs": {"out_tensor_uid": uids["out"]},
                    }
                ],
                "tensors": [
                    {
                        "uid": uids[nm],
                        "name": nm,
                        "dims": [2, 3, 4, 5],
                        "strides": [60, 20, 5, 1],
                        "data_type": "float",
                        "virtual": (nm == "out"),
                    }
                    for nm in ("x", "dy", "scale", "out")
                ],
                "io_data_type": "float",
                "compute_data_type": "float",
                "intermediate_data_type": "float",
                "name": "",
            }

        specs = {
            "x": {"kind": "free", "lo": -1.0, "hi": 1.0},
            "dy": {"kind": "free", "lo": -0.1, "hi": 0.1},
            "scale": {"kind": "free", "lo": -0.2, "hi": 0.2},
        }

        def inputs_for(uids):
            return {str(uids[nm]): specs[nm] for nm in ("x", "dy", "scale")}

        uids0 = {"x": 1, "dy": 2, "scale": 3, "out": 4}
        uids1 = {"x": 3, "dy": 4, "scale": 1, "out": 2}  # shuffled

        # First import (case0).
        g0_path = tmp / "g0.json"
        with open(g0_path, "w") as f:
            json.dump(ternary_graph(uids0), f)
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(g0_path),
                "--bundle-dir",
                str(bundle_dir),
                "--seed",
                "7",
                "--meta",
                f"inputs={json.dumps(inputs_for(uids0))}",
            ]
        )
        assert r.returncode == 0, f"first import failed: {r.stderr}"

        # Re-import the same op with SHUFFLED UIDs + matching specs -> must dedup,
        # because canonicalization by name makes it identical to case0.
        g1_path = tmp / "g1.json"
        with open(g1_path, "w") as f:
            json.dump(ternary_graph(uids1), f)
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "import_graph.py"),
                "--graph",
                str(g1_path),
                "--bundle-dir",
                str(bundle_dir),
                "--seed",
                "7",
                "--meta",
                f"inputs={json.dumps(inputs_for(uids1))}",
            ]
        )
        assert r.returncode == 0, f"shuffled re-import failed: {r.stderr}"
        assert "DUPLICATE" in r.stderr, (
            "shuffled re-import of the same op must dedup by name, " f"got:\n{r.stderr}"
        )

        # Fill spec must be attached to the correctly-named tensor in the bundle.
        sweeps = list(bundle_dir.rglob("sweep.json"))
        assert len(sweeps) == 1, f"expected 1 sweep, got {sweeps}"
        with open(sweeps[0]) as f:
            sweep = json.load(f)
        templates = list(bundle_dir.rglob("graph.template.json"))
        with open(templates[0]) as f:
            tpl = json.load(f)
        uid_to_name = {t["uid"]: t["name"] for t in tpl["tensors"]}
        by_name = {
            uid_to_name[int(u)]: spec
            for u, spec in sweep["cases"][0]["metadata"]["inputs"].items()
        }
        assert by_name["x"] == specs["x"], f"x spec misbound: {by_name}"

        print("  PASS: import_inputs_uid_canonicalized_by_name")


def _write_sweep(bundle_dir, tier, operation, variant, cases):
    """Write a minimal sweep.json under <tier>/<operation>/<variant>/."""
    d = bundle_dir / tier / operation / variant
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "sweep.json", "w") as f:
        json.dump({"version": 1, "cases": cases}, f, indent=2)


def _gtest_json(cases):
    """Build a GTest --gtest_output=json document from {suite: {case: status}}."""
    suites = []
    for suite, by_case in cases.items():
        tc = []
        for case, status in by_case.items():
            entry = {"name": case, "status": "RUN", "result": "COMPLETED"}
            if status == "SKIP":
                entry["status"] = "NOTRUN"
                entry["result"] = "SKIPPED"
            elif status == "FAIL":
                entry["failures"] = [{"failure": "x"}]
            tc.append(entry)
        suites.append({"name": suite, "testsuite": tc})
    return {"testsuites": suites}


def test_diff_coverage_suite_qualified_join():
    """diff_coverage joins on (suite, case_id), not the collision-prone case_id.

    The same bare case_id can appear in sibling suites (e.g. Default and
    Variant2). If the join used the bare id, a PASS in one suite would falsely
    satisfy the coverage requirement for a C++ test whose bundle lives in — and
    is SKIPPED in — the other. Here two suites share case id ``shape_a`` mapped
    to two different C++ sources; only the Default one passes as a bundle. The
    Variant2 C++ source must be reported as a regression, proving the bare id
    does not leak coverage across suites.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        # Same bare case id "shape_a" in two suites, distinct C++ sources.
        _write_sweep(
            bundle_dir,
            "full",
            "Batchnorm",
            "Default",
            [
                {
                    "id": "shape_a",
                    "values": {},
                    "metadata": {
                        "reference_source": (
                            "c++ integration suite: Full/BnDefault.Correctness_0"
                        )
                    },
                }
            ],
        )
        _write_sweep(
            bundle_dir,
            "full",
            "Batchnorm",
            "Variant2",
            [
                {
                    "id": "shape_a",
                    "values": {},
                    "metadata": {
                        "reference_source": (
                            "c++ integration suite: Full/BnVariant2.Correctness_0"
                        )
                    },
                }
            ],
        )

        # C++: both sources PASSED.
        cpp = _gtest_json(
            {
                "Full/BnDefault": {"Correctness/0": "PASS"},
                "Full/BnVariant2": {"Correctness/0": "PASS"},
            }
        )
        # Bundle: Default's shape_a PASSED, Variant2's shape_a SKIPPED.
        bundle = _gtest_json(
            {
                "full_Batchnorm_Default": {"shape_a": "PASS"},
                "full_Batchnorm_Variant2": {"shape_a": "SKIP"},
            }
        )
        cpp_path = tmp / "cpp.json"
        bundle_path = tmp / "bundle.json"
        with open(cpp_path, "w") as f:
            json.dump(cpp, f)
        with open(bundle_path, "w") as f:
            json.dump(bundle, f)

        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "diff_coverage.py"),
                "--cpp",
                str(cpp_path),
                "--bundle",
                str(bundle_path),
                "--bundle-dir",
                str(bundle_dir),
            ],
            check=False,
        )
        # Variant2 source must be flagged as a regression (bundle SKIPPED it),
        # even though a same-named case passed in Default.
        assert r.returncode == 1, (
            "expected regression exit(1) when a cross-suite case_id collides; "
            f"got rc={r.returncode}\n{r.stderr}"
        )
        assert (
            "BnVariant2" in r.stderr
        ), f"Variant2 source should be the reported regression:\n{r.stderr}"
        assert (
            "BnDefault" not in r.stderr.split("regressions:")[-1]
        ), f"Default source passed and must not be a regression:\n{r.stderr}"
        print("  PASS: diff_coverage_suite_qualified_join")


def main() -> int:
    verbose = "-v" in sys.argv
    failures = 0

    tests = [
        test_skeleton_grouping,
        test_round_trip_expansion,
        test_case_ids,
        test_place_and_verify,
        test_import_dedup,
        test_inputs_uid_canonicalized_by_name,
        test_import_inputs_uid_canonicalized_by_name,
        test_diff_coverage_suite_qualified_join,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc()
            failures += 1

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
