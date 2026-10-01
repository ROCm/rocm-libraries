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
  5. import_graph round-trip refusals name the mismatch and write nothing
  6. import_graph takes metadata from the capture sidecar, as place_bundles does

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


def _inventory(root):
    """{relative path: file bytes, or None for a directory} under root."""
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def _import(tmp, name, graph, bundle_dir, *extra, check=True):
    """Write graph to tmp/<name>.json and run import_graph.py on it."""
    graph_path = tmp / f"{name}.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f)
    return run(
        [
            sys.executable,
            str(SCRIPT_DIR / "import_graph.py"),
            "--graph",
            str(graph_path),
            "--bundle-dir",
            str(bundle_dir),
            *extra,
        ],
        check=check,
    )


def _reported(stderr, label):
    """Rendered value of a '    <label>:  <value>  (<side>)' diagnostic line."""
    prefix = f"    {label}:"
    for line in stderr.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().rsplit("  (", 1)[0]
    raise AssertionError(f"no '{label}' line in stderr:\n{stderr}")


def _assert_refused(r, bundle_dir, before, where):
    """Import exited 1 at the named round-trip site and wrote nothing.

    The ERROR line directly follows the preamble's `operation:` line.
    """
    assert r.returncode == 1, f"expected exit 1, got {r.returncode}:\n{r.stderr}"
    lines = r.stderr.splitlines()
    operation = next(
        (i for i, line in enumerate(lines) if line.startswith("  operation:")), None
    )
    assert operation is not None and lines[operation + 1 : operation + 2] == [
        f"  ERROR: round-trip verify failed {where}"
    ], f"expected failure {where} directly after the preamble:\n{r.stderr}"
    assert "Traceback" not in r.stderr, f"diagnostics crashed:\n{r.stderr}"
    assert _inventory(bundle_dir) == before, "refused import changed the bundle tree"


def test_roundtrip_mismatch_walk():
    """_first_mismatch walks sorted keys then indices; _brief bounds renderings.

    An absent side is the _MISSING object, so a key that is absent stays
    distinguishable from a key whose value is null or a string that reads like
    the absence marker.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from import_graph import _MISSING, _brief, _first_mismatch

    found = _first_mismatch({"b": 1, "a": [1, 2]}, {"b": 2, "a": [1]})
    assert found[0] == "a[1]" and found[1] == 2, f"sorted-key walk: {found}"
    assert found[2] is _MISSING, f"missing list element must be _MISSING: {found}"

    found = _first_mismatch([1], [1, 3])
    assert found[0] == "[1]" and found[1] is _MISSING and found[2] == 3, found

    found = _first_mismatch({"x": {"y": [{"z": 1}]}}, {"x": {"y": [{"z": 2}]}})
    assert found == ("x.y[0].z", 1, 2), f"nested path: {found}"
    assert _first_mismatch(1, 2) == ("<root>", 1, 2)
    assert _first_mismatch({"a": 1}, {"a": 1.0}) is None, "walk adds no equality"

    found = _first_mismatch({"k": None}, {})
    assert found[1] is None and found[2] is _MISSING, f"null vs absent: {found}"
    assert (_brief(found[1]), _brief(found[2])) == ("null", "<absent>")

    found = _first_mismatch({"k": "<absent>"}, {})
    assert found[1] == "<absent>" and found[1] is not _MISSING, found
    assert found[2] is _MISSING, found
    assert _brief(found[1]) == '"<absent>"' and _brief(found[2]) == "<absent>"

    assert _brief("x" * 198) == '"' + "x" * 198 + '"', "200 chars must not truncate"
    over = _brief("x" * 199)
    assert len(over) == 200 and over.endswith("..."), f"201 chars: {over!r}"
    assert len(_brief("x" * 500)) == 200
    print("  PASS: roundtrip_mismatch_walk")


def test_import_extraction_failure_diagnostics():
    """A failed extraction round-trip names sweep, match counts and field; no write.

    Two structural matches exist (full/ then quick/); the tier-preferred quick/
    sweep is the one reported. The input's long top-level name is not a
    template placeholder, so the expanded template cannot reproduce it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir, "--tier", "full")
        assert r.returncode == 0, f"seed import failed: {r.stderr}"
        full_sweep = next((bundle_dir / "full").rglob("sweep.json"))
        quick_dir = (
            bundle_dir / "quick" / full_sweep.parent.relative_to(bundle_dir / "full")
        )
        shutil.copytree(full_sweep.parent, quick_dir)
        quick_sweep = quick_dir / "sweep.json"

        bad = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        bad["name"] = "n" * 500
        before = _inventory(bundle_dir)
        r = _import(tmp, "bad", bad, bundle_dir, check=False)
        _assert_refused(r, bundle_dir, before, "after extraction")

        assert (
            f"selected sweep: {quick_sweep} (of 2 structural match(es),"
            f" 1 in tier 'quick')" in r.stderr
        ), f"sweep path/count not reported:\n{r.stderr}"
        assert _reported(r.stderr, "field") == "name", r.stderr
        expected = _reported(r.stderr, "expected")
        assert len(expected) == 200 and expected.endswith("..."), expected
        assert expected.startswith('"nnn'), expected
        assert _reported(r.stderr, "actual") == '""', r.stderr
        print("  PASS: import_extraction_failure_diagnostics")


def test_import_canonical_only_difference():
    """int 1 vs float 1.0 compares equal field-wise; report canonical offsets.

    The walk finds no unequal field, so the report gives the first differing
    offset of the two canonical renderings and at most 60 characters of each.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_utils import canon, canonical_uid_map_by_name, remap_graph

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g_int = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g_int["nodes"][0]["alpha"] = 1
        r = _import(tmp, "g_int", g_int, bundle_dir)
        assert r.returncode == 0, f"seed import failed: {r.stderr}"

        g_float = json.loads(json.dumps(g_int))
        g_float["nodes"][0]["alpha"] = 1.0
        before = _inventory(bundle_dir)
        r = _import(tmp, "g_float", g_float, bundle_dir, check=False)
        _assert_refused(r, bundle_dir, before, "after extraction")
        assert not any(
            line.startswith("    field:") for line in r.stderr.splitlines()
        ), r.stderr

        want = canon(remap_graph(g_float, canonical_uid_map_by_name(g_float)))
        got = canon(remap_graph(g_int, canonical_uid_map_by_name(g_int)))
        at = next(i for i, (a, b) in enumerate(zip(want, got)) if a != b)
        assert len(want) - at > 60 and len(got) - at > 60, "fixture too short"
        assert _reported(r.stderr, "at offset") == str(at), r.stderr
        assert _reported(r.stderr, "expected") == want[at : at + 60], r.stderr
        assert _reported(r.stderr, "actual") == got[at : at + 60], r.stderr
        print("  PASS: import_canonical_only_difference")


def test_import_absent_vs_literal_marker():
    """A key absent from the template is not confused with a look-alike string.

    The input carries the literal string "<absent>" where the template has no
    key at all: the input side renders as a quoted JSON string, the absent side
    as the bare marker.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"seed import failed: {r.stderr}"

        bad = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        bad["nodes"][0]["note"] = "<absent>"
        before = _inventory(bundle_dir)
        r = _import(tmp, "bad", bad, bundle_dir, check=False)
        _assert_refused(r, bundle_dir, before, "after extraction")
        assert _reported(r.stderr, "field") == "nodes[0].note", r.stderr
        assert _reported(r.stderr, "expected") == '"<absent>"', r.stderr
        assert _reported(r.stderr, "actual") == "<absent>", r.stderr
        print("  PASS: import_absent_vs_literal_marker")


def test_import_new_template_failure_diagnostics():
    """A failed new-template round-trip names the field and writes nothing.

    A tensor without a uid cannot bind its per-tensor placeholders, so the
    expanded template leaves them unresolved. An unrelated existing bundle
    keeps the before/after inventory non-empty.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        conv = _make_graph("Conv", [0, 1, 2], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "conv", conv, bundle_dir)
        assert r.returncode == 0, f"seed import failed: {r.stderr}"

        bad = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        bad["tensors"].append(
            {
                "name": "orphan",
                "dims": [7],
                "strides": [1],
                "data_type": "float",
                "virtual": False,
            }
        )
        before = _inventory(bundle_dir)
        assert before, "fixture must start from a non-empty bundle tree"
        r = _import(tmp, "bad", bad, bundle_dir, check=False)
        _assert_refused(r, bundle_dir, before, "for new template")
        assert "selected sweep" not in r.stderr, r.stderr
        assert _reported(r.stderr, "field") == "tensors[1].data_type", r.stderr
        assert _reported(r.stderr, "tensor") == 'name="orphan"', r.stderr
        assert _reported(r.stderr, "expected") == '"float"', r.stderr
        assert _reported(r.stderr, "actual") == '"${UNRESOLVED:data_type}"', r.stderr
        print("  PASS: import_new_template_failure_diagnostics")


# --------------------------------------------------------------------------
# Capture sidecar (.meta.json) handling in import_graph
# --------------------------------------------------------------------------

_SUITE = "Smoke/IntegrationGpuTernaryFp32"


def _ternary_graph(uids, dims=(2, 3, 4, 5)):
    """Named-tensor ternary op; uids = {name: uid}, out is virtual."""
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
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
                "dims": list(dims),
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


_UIDS0 = {"x": 1, "dy": 2, "scale": 3, "out": 4}
_UIDS1 = {"x": 3, "dy": 4, "scale": 1, "out": 2}  # shuffled


def _sidecar(case_name, uids=None, seed=None, specs=None, suite=_SUITE):
    """A sidecar shaped like the one --capture-bundles writes.

    specs = {tensor name: fill spec}; keyed by that capture's own UIDs.
    """
    meta = {
        "format_version": 1,
        "operation": suite,
        "generator": "capture-bundles",
        "generator_version": "1.0.0",
    }
    if seed is not None:
        meta["seed"] = seed
    if specs is not None:
        meta["inputs"] = {str(uids[nm]): spec for nm, spec in specs.items()}
    meta["notes"] = f"Captured from C++ graph test {suite}.{case_name}"
    return meta


def _capture(capture_dir, case_name, graph, sidecar, suite=_SUITE):
    """Write a Hop A capture; sidecar is a dict, raw text, or None (absent)."""
    if isinstance(sidecar, dict):
        _write_captured(capture_dir, suite, case_name, graph, meta=sidecar)
    else:
        _write_captured(capture_dir, suite, case_name, graph)
        meta_path = capture_dir / suite / case_name / f"{case_name}.meta.json"
        if sidecar is None:
            meta_path.unlink()
        else:
            meta_path.write_text(sidecar)
    return capture_dir / suite / case_name / f"{case_name}.json"


def _import_capture(graph_path, bundle_dir, *extra):
    return run(
        [
            sys.executable,
            str(SCRIPT_DIR / "import_graph.py"),
            "--graph",
            str(graph_path),
            "--bundle-dir",
            str(bundle_dir),
            *extra,
        ],
        check=False,
    )


def _cases(bundle_dir):
    """All sweep cases under bundle_dir."""
    out = []
    for p in sorted(bundle_dir.rglob("sweep.json")):
        with open(p) as f:
            out.extend(json.load(f)["cases"])
    return out


def _only_metadata(bundle_dir):
    cases = _cases(bundle_dir)
    assert len(cases) == 1, f"expected 1 case, got {cases}"
    return cases[0]["metadata"]


def _inputs_by_name(bundle_dir, metadata):
    """Re-key a case's inputs from canonical UIDs to tensor names."""
    tpl_path = next(bundle_dir.rglob("graph.template.json"))
    with open(tpl_path) as f:
        uid_to_name = {t["uid"]: t["name"] for t in json.load(f)["tensors"]}
    return {uid_to_name[int(u)]: s for u, s in metadata["inputs"].items()}


def test_import_reads_sidecar():
    """With the sidecar present, import records what place_bundles records.

    No --meta/--seed: the seed, the explicit inputs entry, and every other
    sidecar field come from the capture. x's range [0.5, 2.0] is deliberately
    not the FREE[-1, 1] declaration default, so a dropped entry would show. The
    two captures use shuffled UIDs, so inputs must be remapped with the graph.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        specs = {
            "x": {"kind": "free", "lo": 0.5, "hi": 2.0},
            "scale": {"kind": "fixed", "value": 1.0},
        }
        paths = [
            _capture(
                capture_dir,
                "case0",
                _ternary_graph(_UIDS0),
                _sidecar("case0", _UIDS0, seed=1234, specs=specs),
            ),
            _capture(
                capture_dir,
                "case1",
                _ternary_graph(_UIDS1, dims=(4, 6, 8, 10)),
                _sidecar("case1", _UIDS1, seed=99, specs=specs),
            ),
        ]

        placed_dir = tmp / "placed"
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "place_bundles.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(placed_dir),
            ]
        )
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"

        imported_dir = tmp / "imported"
        imported_dir.mkdir()
        for p in paths:
            r = _import_capture(p, imported_dir)
            assert r.returncode == 0, f"import of {p} failed:\n{r.stderr}"
            assert "WARN" not in r.stderr, f"clean sidecar must be quiet:\n{r.stderr}"

        def by_source(cases):
            return {c["metadata"]["reference_source"]: c["metadata"] for c in cases}

        placed = by_source(_cases(placed_dir))
        imported = by_source(_cases(imported_dir))
        assert len(placed) == 2, f"place_bundles should sweep both: {placed}"
        assert imported == placed, (
            f"import metadata differs from place_bundles:\n"
            f"  imported: {imported}\n  placed:   {placed}"
        )
        for ref, meta in imported.items():
            assert meta["seed"] in (1234, 99), f"captured seed lost: {meta}"
            assert (
                _inputs_by_name(imported_dir, meta) == specs
            ), f"inputs entry not preserved by name for {ref}: {meta['inputs']}"
        print("  PASS: import_reads_sidecar")


def test_import_cli_overrides_sidecar():
    """--seed and --meta win over the sidecar; untouched sidecar fields stay."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        captured = {"x": {"kind": "free", "lo": 0.5, "hi": 2.0}}
        supplied = {"x": {"kind": "free", "lo": -3.0, "hi": 3.0}}
        graph_path = _capture(
            tmp / "captured",
            "case0",
            _ternary_graph(_UIDS1),
            _sidecar("case0", _UIDS1, seed=42, specs=captured),
        )
        inputs = {str(_UIDS1[nm]): s for nm, s in supplied.items()}
        r = _import_capture(
            graph_path,
            bundle_dir,
            "--seed",
            "7",
            "--meta",
            "notes=hand-written",
            "--meta",
            f"inputs={json.dumps(inputs)}",
        )
        assert r.returncode == 0, f"import failed:\n{r.stderr}"
        meta = _only_metadata(bundle_dir)
        assert meta["seed"] == 7, meta
        assert meta["notes"] == "hand-written", meta
        assert _inputs_by_name(bundle_dir, meta) == supplied, meta
        assert meta["generator"] == "capture-bundles", f"sidecar base lost: {meta}"
        for flag in ("--seed", "--meta notes", "--meta inputs"):
            assert (
                f"WARN: {flag} " in r.stderr
            ), f"no override warning for {flag}:\n{r.stderr}"
        print("  PASS: import_cli_overrides_sidecar")


def test_import_seed_override_warns():
    """Overriding the captured seed warns with both values and the sidecar path.

    Agreeing values, a sidecar without a seed, and no sidecar stay quiet;
    --strict turns the override into a refusal that writes nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        graph_path = _capture(
            capture_dir, "case0", _ternary_graph(_UIDS0), _sidecar("case0", seed=42)
        )
        sidecar_path = graph_path.with_suffix(".meta.json")

        bundle_dir = tmp / "override"
        bundle_dir.mkdir()
        r = _import_capture(graph_path, bundle_dir, "--seed", "7")
        assert r.returncode == 0, f"override must still import:\n{r.stderr}"
        warn = [ln for ln in r.stderr.splitlines() if "WARN" in ln]
        assert len(warn) == 1, f"expected one warning:\n{r.stderr}"
        for part in ("--seed 7", "captured 42", str(sidecar_path)):
            assert part in warn[0], f"warning lacks {part!r}: {warn[0]}"
        assert _only_metadata(bundle_dir)["seed"] == 7, "override not applied"

        quiet = {
            "agree": (graph_path, ["--seed", "42"]),
            "sidecar without seed": (
                _capture(
                    capture_dir, "case1", _ternary_graph(_UIDS0), _sidecar("case1")
                ),
                ["--seed", "7"],
            ),
            "no sidecar": (
                _capture(capture_dir, "case2", _ternary_graph(_UIDS0), None),
                ["--seed", "7"],
            ),
        }
        for label, (path, extra) in quiet.items():
            d = tmp / label.replace(" ", "_")
            d.mkdir()
            r = _import_capture(path, d, *extra)
            assert r.returncode == 0, f"{label}: import failed:\n{r.stderr}"
            assert "WARN" not in r.stderr, f"{label}: must be quiet:\n{r.stderr}"

        strict_dir = tmp / "strict"
        strict_dir.mkdir()
        r = _import_capture(graph_path, strict_dir, "--seed", "7", "--strict")
        assert r.returncode == 1, f"--strict override must fail:\n{r.stderr}"
        assert "ERROR: --seed 7 overrides captured 42" in r.stderr, r.stderr
        assert _inventory(strict_dir) == {}, "--strict refusal wrote files"
        print("  PASS: import_seed_override_warns")


def test_import_no_sidecar_unchanged():
    """Without a sidecar, metadata is exactly what the CLI says, as before."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph = _ternary_graph(_UIDS0)
        runs = {
            "loose graph, no flags": (tmp / "loose", (), {"format_version": 1}),
            "loose graph, flags": (
                tmp / "loose_flags",
                ("--seed", "5", "--meta", "notes=n"),
                {"format_version": 1, "seed": 5, "notes": "n"},
            ),
        }
        for label, (bundle_dir, extra, expected) in runs.items():
            bundle_dir.mkdir()
            r = _import(tmp, bundle_dir.name, graph, bundle_dir, *extra, check=False)
            assert r.returncode == 0, f"{label}: import failed:\n{r.stderr}"
            assert "WARN" not in r.stderr, f"{label}: must be quiet:\n{r.stderr}"
            meta = _only_metadata(bundle_dir)
            assert meta == expected, f"{label}: {meta} != {expected}"

        # Capture layout but the sidecar is missing: no reference_source is
        # invented, the metadata is still just the default.
        bundle_dir = tmp / "capture_layout"
        bundle_dir.mkdir()
        path = _capture(tmp / "captured", "case0", graph, None)
        r = _import_capture(path, bundle_dir)
        assert r.returncode == 0 and "WARN" not in r.stderr, r.stderr
        assert _only_metadata(bundle_dir) == {"format_version": 1}
        print("  PASS: import_no_sidecar_unchanged")


def _tiers_by_source(bundle_dir):
    """{reference_source: tier folder} for every sweep case and standalone."""
    out = {}
    for p in sorted(bundle_dir.rglob("*.json")):
        with open(p) as f:
            doc = json.load(f)
        if p.name == "sweep.json":
            metas = [c["metadata"] for c in doc["cases"]]
        elif p.name.endswith(".meta.json"):
            metas = [doc]  # place_bundles' single-case standalone bundle
        else:
            continue
        for meta in metas:
            out[meta.get("reference_source")] = p.relative_to(bundle_dir).parts[0]
    return out


def test_import_tier_from_sidecar():
    """Without --tier, the suite prefix picks the tier as in place_bundles.

    A Full/ capture must land in full/, not the quick/ default, or it moves to
    the smoke CI lane. Each capture is imported into an empty tree, so it takes
    the new-topology path that a misfiled tier would silently write to.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        suites = [
            "Smoke/IntegrationGpuTernaryFp32",
            "Full/IntegrationGpuTernaryFp32",
            "Standard/IntegrationGpuTernaryFp32",
            "IntegrationGpuTernaryFp32",  # no tier prefix
        ]
        paths = {
            suite: _capture(
                capture_dir,
                "case0",
                _ternary_graph(_UIDS0),
                _sidecar("case0", seed=seed, suite=suite),
                suite=suite,
            )
            for seed, suite in enumerate(suites)
        }

        placed_dir = tmp / "placed"
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "place_bundles.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(placed_dir),
            ]
        )
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"
        placed = _tiers_by_source(placed_dir)

        imported = {}
        for i, path in enumerate(paths.values()):
            bundle_dir = tmp / f"imported{i}"
            bundle_dir.mkdir()
            r = _import_capture(path, bundle_dir)
            assert r.returncode == 0, f"import of {path} failed:\n{r.stderr}"
            assert "--tier" not in r.stderr, r.stderr
            imported.update(_tiers_by_source(bundle_dir))

        assert len(placed) == len(suites), f"place_bundles lost cases: {placed}"
        assert imported == placed, (
            f"import tiers differ from place_bundles:\n"
            f"  imported: {imported}\n  placed:   {placed}"
        )
        assert sorted(imported.values()) == ["full", "quick", "quick", "standard"]

        # --tier still wins over the sidecar, but not silently.
        full_case = paths["Full/IntegrationGpuTernaryFp32"]
        override_dir = tmp / "override"
        override_dir.mkdir()
        r = _import_capture(full_case, override_dir, "--tier", "standard")
        assert r.returncode == 0, r.stderr
        assert "WARN: --tier" in r.stderr and '"full"' in r.stderr, r.stderr
        assert list(_tiers_by_source(override_dir).values()) == ["standard"]

        # A --tier that agrees with the sidecar is quiet.
        agree_dir = tmp / "agree"
        agree_dir.mkdir()
        r = _import_capture(full_case, agree_dir, "--tier", "full", "--strict")
        assert r.returncode == 0, r.stderr
        assert "--tier" not in r.stderr, r.stderr
        assert list(_tiers_by_source(agree_dir).values()) == ["full"]

        # Under --strict a contradicting --tier refuses and writes nothing.
        strict_dir = tmp / "strict"
        strict_dir.mkdir()
        r = _import_capture(full_case, strict_dir, "--tier", "quick", "--strict")
        assert r.returncode != 0, "strict tier override should fail"
        assert "ERROR: --tier" in r.stderr, r.stderr
        assert not any(strict_dir.iterdir()), list(strict_dir.rglob("*"))

        # No sidecar: quick, as before.
        full_case.with_suffix(".meta.json").unlink()
        bare_dir = tmp / "bare"
        bare_dir.mkdir()
        r = _import_capture(full_case, bare_dir)
        assert r.returncode == 0, r.stderr
        assert list(_tiers_by_source(bare_dir).values()) == ["quick"]

        # No sidecar means nothing to contradict: --tier is quiet even strict.
        bare_tier_dir = tmp / "bare_tier"
        bare_tier_dir.mkdir()
        r = _import_capture(full_case, bare_tier_dir, "--tier", "full", "--strict")
        assert r.returncode == 0, r.stderr
        assert "--tier" not in r.stderr, r.stderr
        assert list(_tiers_by_source(bare_tier_dir).values()) == ["full"]
        print("  PASS: import_tier_from_sidecar")


def test_import_meta_override_checks():
    """--meta is checked against everything the capture recorded, as typed values.

    A numeric --meta restating the captured value is quiet and stays a number
    (so it dedups); overriding the derived reference_source warns, and fails
    under --strict; an inputs map keyed by non-UIDs is refused, not a crash.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph_path = _capture(
            tmp / "captured",
            "case0",
            _ternary_graph(_UIDS0),
            _sidecar("case0", seed=42),
        )
        ref = f"c++ integration suite: {_SUITE}.case0"

        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        r = _import_capture(graph_path, bundle_dir, "--meta", "seed=42", "--strict")
        assert r.returncode == 0, f"restating the seed must pass:\n{r.stderr}"
        assert "--meta seed" not in r.stderr, r.stderr
        meta = _only_metadata(bundle_dir)
        assert meta["seed"] == 42 and isinstance(meta["seed"], int), meta
        assert meta["reference_source"] == ref, meta

        before = _inventory(bundle_dir)
        r = _import_capture(graph_path, bundle_dir, "--meta", "seed=42")
        assert r.returncode == 0, r.stderr
        assert "DUPLICATE" in r.stderr, f"--meta seed=42 must dedup:\n{r.stderr}"
        assert _inventory(bundle_dir) == before, "duplicate was written"

        quiet_dir = tmp / "ref_agree"
        quiet_dir.mkdir()
        r = _import_capture(
            graph_path, quiet_dir, "--meta", f"reference_source={ref}", "--strict"
        )
        assert r.returncode == 0, r.stderr
        assert "--meta reference_source" not in r.stderr, r.stderr

        warn_dir = tmp / "ref_override"
        warn_dir.mkdir()
        r = _import_capture(graph_path, warn_dir, "--meta", "reference_source=manual")
        assert r.returncode == 0, r.stderr
        warn = [
            ln for ln in r.stderr.splitlines() if "WARN: --meta reference_source" in ln
        ]
        assert len(warn) == 1 and ref in warn[0], f"no warning:\n{r.stderr}"
        assert _only_metadata(warn_dir)["reference_source"] == "manual"

        strict_dir = tmp / "ref_strict"
        strict_dir.mkdir()
        r = _import_capture(
            graph_path, strict_dir, "--meta", "reference_source=manual", "--strict"
        )
        assert r.returncode == 1, f"--strict override must fail:\n{r.stderr}"
        assert "ERROR: --meta reference_source" in r.stderr, r.stderr
        assert _inventory(strict_dir) == {}, "--strict refusal wrote files"

        bad_dir = tmp / "bad_inputs"
        bad_dir.mkdir()
        r = _import_capture(
            graph_path, bad_dir, "--meta", 'inputs={"x": {"kind": "free"}}'
        )
        assert r.returncode == 1, f"non-UID inputs key must fail:\n{r.stderr}"
        assert "Traceback" not in r.stderr, r.stderr
        assert "not a tensor UID" in r.stderr, r.stderr
        assert _inventory(bad_dir) == {}, "refusal wrote files"
        print("  PASS: import_meta_override_checks")


def test_import_malformed_sidecar_warns():
    """An unreadable sidecar warns and the import proceeds without it.

    Under --strict it is a refusal that writes nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bad = {
            "truncated": '{"seed": 4',
            "not an object": "[1, 2]",
            # Parses, but remap_meta_inputs would crash on int("x").
            "inputs key not a UID": '{"seed": 4, "inputs": {"x": {"kind": "free"}}}',
            "inputs not an object": '{"seed": 4, "inputs": [1]}',
        }
        for i, (label, text) in enumerate(bad.items()):
            path = _capture(tmp / "captured", f"case{i}", _ternary_graph(_UIDS0), text)
            sidecar_path = path.with_suffix(".meta.json")

            bundle_dir = tmp / f"lenient{i}"
            bundle_dir.mkdir()
            r = _import_capture(path, bundle_dir, "--seed", "3")
            assert r.returncode == 0, f"{label}: must not abort:\n{r.stderr}"
            assert "Traceback" not in r.stderr, f"{label}: crashed:\n{r.stderr}"
            assert (
                f"WARN: bad sidecar {sidecar_path}" in r.stderr
            ), f"{label}: no warning:\n{r.stderr}"
            meta = _only_metadata(bundle_dir)
            assert meta == {"format_version": 1, "seed": 3}, f"{label}: {meta}"

            strict_dir = tmp / f"strict{i}"
            strict_dir.mkdir()
            r = _import_capture(path, strict_dir, "--strict")
            assert r.returncode == 1, f"{label}: --strict must fail:\n{r.stderr}"
            assert f"ERROR: bad sidecar {sidecar_path}" in r.stderr, r.stderr
            assert _inventory(strict_dir) == {}, f"{label}: --strict wrote files"
        print("  PASS: import_malformed_sidecar_warns")


def test_import_seed_only_difference_distinct_cases():
    """Identical graphs that differ only by seed are distinct cases.

    Covers both a second captured seed and an unknown (absent) seed: neither
    may be treated as a duplicate of a case with an explicit seed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        graph = _ternary_graph(_UIDS0)
        seed1 = _capture(capture_dir, "seed1", graph, _sidecar("seed1", seed=1))
        seed2 = _capture(capture_dir, "seed2", graph, _sidecar("seed2", seed=2))

        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        for p in (seed1, seed2):
            r = _import_capture(p, bundle_dir)
            assert r.returncode == 0, f"import of {p} failed:\n{r.stderr}"
            assert (
                "DUPLICATE" not in r.stderr
            ), f"seed-only difference deduped:\n{r.stderr}"
        cases = _cases(bundle_dir)
        assert [c["metadata"]["seed"] for c in cases] == [1, 2], cases
        assert len({c["id"] for c in cases}) == 2, f"ids collide: {cases}"

        # Re-importing a captured seed is still a duplicate.
        r = _import_capture(seed1, bundle_dir)
        assert "DUPLICATE" in r.stderr, f"same seed must dedup:\n{r.stderr}"

        # An unknown seed does not match an explicit one.
        bundle_dir = tmp / "unknown"
        bundle_dir.mkdir()
        r = _import_capture(seed1, bundle_dir)
        assert r.returncode == 0, r.stderr
        r = _import(tmp, "no_seed", graph, bundle_dir, check=False)
        assert r.returncode == 0, r.stderr
        assert "DUPLICATE" not in r.stderr, f"unknown seed deduped:\n{r.stderr}"
        cases = _cases(bundle_dir)
        assert len(cases) == 2 and len({c["id"] for c in cases}) == 2, cases
        print("  PASS: import_seed_only_difference_distinct_cases")


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
        test_roundtrip_mismatch_walk,
        test_import_extraction_failure_diagnostics,
        test_import_canonical_only_difference,
        test_import_absent_vs_literal_marker,
        test_import_new_template_failure_diagnostics,
        test_import_reads_sidecar,
        test_import_cli_overrides_sidecar,
        test_import_seed_override_warns,
        test_import_no_sidecar_unchanged,
        test_import_tier_from_sidecar,
        test_import_meta_override_checks,
        test_import_malformed_sidecar_warns,
        test_import_seed_only_difference_distinct_cases,
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
