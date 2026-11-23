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
  6. place_bundles keeps manual cases of an existing sweep on regeneration
  7. import_graph takes metadata from the capture sidecar, as place_bundles does

No C++ binary needed — everything is pure Python on synthetic data.

Usage::

    python3 test_migration.py [-v]
"""

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def test_place_keeps_manual_cases():
    """Regenerating a sweep keeps its manual cases, or refuses and writes nothing.

    A manual case (metadata.generator == "manual") has no C++ test behind it, so
    no capture reproduces it. A rerun of place_bundles over the same tree must
    keep it. One that no longer fits the template fails the run and leaves the
    sweep as it was.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        bundle_dir = tmp / "bundles"
        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
        _write_captured(capture_dir, "Smoke/IntegrationGpuReluFp32", "small_fp32", g1)
        _write_captured(capture_dir, "Smoke/IntegrationGpuReluFp16", "small_fp16", g2)
        place = [
            sys.executable,
            str(SCRIPT_DIR / "place_bundles.py"),
            "--capture-dir",
            str(capture_dir),
            "--output-dir",
            str(bundle_dir),
        ]
        r = run(place)
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"
        (sweep_path,) = bundle_dir.rglob("sweep.json")

        sweep = json.loads(sweep_path.read_text())
        manual = copy.deepcopy(sweep["cases"][0])
        manual["id"] = "runtime_scalar"
        manual["tensor_patches"] = [
            {"uid": 0, "set": {"is_runtime_pass_by_value": True}, "remove": []}
        ]
        manual["metadata"] = {"format_version": 1, "generator": "manual"}
        sweep["cases"].append(manual)
        sweep_path.write_text(json.dumps(sweep, indent=2) + "\n")

        r = run(place)
        assert r.returncode == 0, f"place_bundles failed: {r.stderr}"
        cases = json.loads(sweep_path.read_text())["cases"]
        assert len(cases) == 3, [c["id"] for c in cases]
        assert cases[-1] == manual, f"manual case changed or dropped: {cases}"

        # A manual case missing a template tensor cannot be expanded any more.
        sweep = json.loads(sweep_path.read_text())
        sweep["cases"][-1]["values"]["tensors"].pop()
        stale = json.dumps(sweep, indent=2) + "\n"
        sweep_path.write_text(stale)

        r = run(place, check=False)
        assert r.returncode == 1, f"expected failure:\n{r.stderr}"
        assert "'runtime_scalar' does not fit" in r.stderr, r.stderr
        assert sweep_path.read_text() == stale, "sweep must be left unchanged"
        print("  PASS: place_keeps_manual_cases")


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


def test_import_append_preserves_case_fields():
    """Appending a new case must not drop existing keys like golden/
    tensor_patches, nor an unrelated top-level sweep key (mirrors
    integration-test-bundles/quick/BatchnormFwdInference/Inference/sweep.json,
    whose cases carry a 'golden' key alongside values/metadata).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"seed import failed: {r.stderr}"

        sweep_path = next(bundle_dir.rglob("sweep.json"))
        with open(sweep_path) as f:
            sweep = json.load(f)
        sweep["note"] = "hand-authored top-level key"
        sweep["cases"][0]["golden"] = {"path": "golden/g1/tensors.dvc"}
        sweep["cases"][0]["tensor_patches"] = {"1": {"seed": 7}}
        with open(sweep_path, "w") as f:
            json.dump(sweep, f, indent=2)
            f.write("\n")

        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
        r = _import(tmp, "g2", g2, bundle_dir)
        assert r.returncode == 0, f"append import failed: {r.stderr}"
        assert "appended" in r.stderr

        with open(sweep_path) as f:
            after = json.load(f)
        assert (
            after["note"] == "hand-authored top-level key"
        ), "unknown top-level sweep key must survive an append"
        assert len(after["cases"]) == 2
        assert after["cases"][0]["golden"] == {
            "path": "golden/g1/tensors.dvc"
        }, "existing case's golden must survive an append"
        assert after["cases"][0]["tensor_patches"] == {
            "1": {"seed": 7}
        }, "existing case's tensor_patches must survive an append"
        print("  PASS: import_append_preserves_case_fields")


def test_import_new_topology_never_clobbers():
    """A second, structurally distinct graph whose derived directory already
    holds a sweep must land in a fresh directory, following place_bundles.py's
    Default/Variant2/... convention; the original template/sweep survive
    byte-identical.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Matmul", [0, 1], [2, 3], [3, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"first import failed: {r.stderr}"

        default_dir = bundle_dir / "quick" / "Matmul" / "Default"
        assert (default_dir / "sweep.json").exists(), r.stderr
        before_template = (default_dir / "graph.template.json").read_bytes()
        before_sweep = (default_dir / "sweep.json").read_bytes()

        # Same op and same rank (-> same preferred "Default" topo name), but
        # an extra unconnected tensor gives it a different skeleton hash, so
        # it is NOT a structural match and must take the new-topology path.
        g2 = _make_graph("Matmul", [0, 1, 2], [2, 3], [3, 1], "float")
        r = _import(tmp, "g2", g2, bundle_dir)
        assert r.returncode == 0, f"second import failed: {r.stderr}"
        assert "created new bundle" in r.stderr, r.stderr

        variant_dir = bundle_dir / "quick" / "Matmul" / "Variant2"
        assert (
            variant_dir / "sweep.json"
        ).exists(), f"expected a fresh Variant2 dir, not a clobber:\n{r.stderr}"
        assert (
            default_dir / "graph.template.json"
        ).read_bytes() == before_template, "Default template must be untouched"
        assert (
            default_dir / "sweep.json"
        ).read_bytes() == before_sweep, "Default sweep must be untouched"
        print("  PASS: import_new_topology_never_clobbers")


def test_import_new_topology_case_id_is_descriptive():
    """A brand-new topology's sole case gets a real id, not the literal
    'case' -- the id doubles as the gtest name."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"import failed: {r.stderr}"

        sweep_path = next(bundle_dir.rglob("sweep.json"))
        with open(sweep_path) as f:
            sweep = json.load(f)
        case_id = sweep["cases"][0]["id"]
        assert (
            case_id != "case"
        ), f"new-topology case id must be descriptive, got {case_id!r}"
        assert re.match(r"^[a-z0-9_]+$", case_id), case_id
        print("  PASS: import_new_topology_case_id_is_descriptive")


def test_import_support_json_sidecar():
    """Every sweep import_graph creates or touches gets an empty canonical
    support.json that verify_support_claims.verify_all() accepts; an existing
    support.json is never modified.
    """
    sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))
    import verify_support_claims

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        # New topology -> gets a fresh, empty, canonical support.json.
        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"seed import failed: {r.stderr}"
        assert "created empty support.json" in r.stderr, r.stderr

        sweep_path = next(bundle_dir.rglob("sweep.json"))
        support_path = sweep_path.parent / "support.json"
        assert support_path.exists()
        raw = support_path.read_bytes()
        assert raw == b'{\n  "claims": {},\n  "version": 1\n}\n', raw
        assert b"\r\n" not in raw, "support.json must use LF line endings"

        with open(sweep_path) as f:
            case_id = json.load(f)["cases"][0]["id"]

        # Overwrite with a hand-written, valid, non-empty claims sidecar; an
        # append into this same sweep must leave it byte-for-byte alone.
        hand_written = (
            json.dumps(
                {
                    "claims": {
                        "SOME_ENGINE": [
                            {"cases": [case_id], "support": {"gfx90a": ["linux"]}}
                        ]
                    },
                    "version": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        support_path.write_bytes(hand_written)

        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "half")
        r = _import(tmp, "g2", g2, bundle_dir)
        assert r.returncode == 0, f"append import failed: {r.stderr}"
        assert "created empty support.json" not in r.stderr, r.stderr
        assert (
            support_path.read_bytes() == hand_written
        ), "an existing support.json must never be modified"

        errors = verify_support_claims.verify_all(bundle_dir)
        assert not errors, f"verify_support_claims flagged our output:\n{errors}"
        print("  PASS: import_support_json_sidecar")


def test_import_generator_metadata_default():
    """metadata['generator'] defaults to 'import_graph.py'; an explicit
    --meta generator=... wins, and a missing reference_source warns."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()

        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g1", g1, bundle_dir)
        assert r.returncode == 0, f"default-generator import failed: {r.stderr}"
        sweep_path = next(bundle_dir.rglob("sweep.json"))
        with open(sweep_path) as f:
            sweep = json.load(f)
        assert sweep["cases"][0]["metadata"]["generator"] == "import_graph.py"
        assert (
            "reference_source in metadata" in r.stderr
        ), "a missing --meta reference_source must warn"

        bundle_dir2 = tmp / "bundles2"
        bundle_dir2.mkdir()
        g2 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(
            tmp,
            "g2",
            g2,
            bundle_dir2,
            "--meta",
            "generator=custom_gen",
            "--meta",
            "reference_source=unit-test",
        )
        assert r.returncode == 0, f"explicit-generator import failed: {r.stderr}"
        sweep_path2 = next(bundle_dir2.rglob("sweep.json"))
        with open(sweep_path2) as f:
            sweep2 = json.load(f)
        assert sweep2["cases"][0]["metadata"]["generator"] == "custom_gen"
        assert (
            "reference_source in metadata" not in r.stderr
        ), "an explicit reference_source must not warn"
        print("  PASS: import_generator_metadata_default")


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


def test_resolve_plain_tensor_scoped_field_from_its_own_entry():
    """_resolve() must let a non-whitelisted, non-dotted field (e.g. alignment, as
    used by develop's SdpaBackward bundles) resolve from a tensor's own values.tensors[]
    entry -- it has no field-name whitelist, mirroring the C++ loader's lookupJsonPath(),
    which walks any placeholder path through the tensor's own entry with no restriction.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_utils import canon, expand

    g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
    g["tensors"][0]["alignment"] = 16

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
                "alignment": "${case.alignment}",
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
                "alignment": 16,
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
    assert canon(expanded) == canon(g), "plain tensor-scoped field did not resolve"
    print("  PASS: resolve_plain_tensor_scoped_field_from_its_own_entry")


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
        p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None
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
            assert "no reference_source" not in r.stderr, r.stderr

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
            "loose graph, no flags": (
                tmp / "loose",
                (),
                {"format_version": 1, "generator": "import_graph.py"},
            ),
            "loose graph, flags": (
                tmp / "loose_flags",
                ("--seed", "5", "--meta", "notes=n"),
                {
                    "format_version": 1,
                    "generator": "import_graph.py",
                    "seed": 5,
                    "notes": "n",
                },
            ),
        }
        for label, (bundle_dir, extra, expected) in runs.items():
            bundle_dir.mkdir()
            r = _import(tmp, bundle_dir.name, graph, bundle_dir, *extra, check=False)
            assert r.returncode == 0, f"{label}: import failed:\n{r.stderr}"
            assert "WARN" not in r.stderr, f"{label}: must be quiet:\n{r.stderr}"
            assert "no reference_source" in r.stderr, f"{label}: {r.stderr}"
            meta = _only_metadata(bundle_dir)
            assert meta == expected, f"{label}: {meta} != {expected}"

        # Capture layout but the sidecar is missing: no reference_source is
        # invented, the metadata is still just the default.
        bundle_dir = tmp / "capture_layout"
        bundle_dir.mkdir()
        path = _capture(tmp / "captured", "case0", graph, None)
        r = _import_capture(path, bundle_dir)
        assert r.returncode == 0 and "WARN" not in r.stderr, r.stderr
        assert "no reference_source" in r.stderr, r.stderr
        assert _only_metadata(bundle_dir) == {
            "format_version": 1,
            "generator": "import_graph.py",
        }
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
            "Comprehensive/IntegrationGpuTernaryFp32",
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
        assert sorted(imported.values()) == [
            "comprehensive",
            "full",
            "quick",
            "quick",
            "standard",
        ]

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


def test_import_sidecar_outside_capture_layout():
    """A sidecar still sets the tier when the graph is not at its capture path.

    The suite is recorded in the sidecar, so a capture copied to a flat
    directory keeps its tier; the case name is not, so no reference_source is
    derived. A hand-written sidecar with no ``operation`` has no suite at all:
    quick, as with no sidecar.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        suite = "Full/IntegrationGpuTernaryFp32"
        captured = _capture(
            tmp / "captured",
            "case0",
            _ternary_graph(_UIDS0),
            _sidecar("case0", seed=7, suite=suite),
            suite=suite,
        )
        flat = tmp / "flat"
        flat.mkdir()
        for p in (captured, captured.with_suffix(".meta.json")):
            (flat / p.name).write_text(p.read_text())

        bundle_dir = tmp / "moved"
        bundle_dir.mkdir()
        r = _import_capture(flat / captured.name, bundle_dir)
        assert r.returncode == 0, r.stderr
        assert "WARN" not in r.stderr, r.stderr
        assert list(_tiers_by_source(bundle_dir)) == [None], "reference_source invented"
        assert list(_tiers_by_source(bundle_dir).values()) == ["full"]
        assert _only_metadata(bundle_dir)["seed"] == 7

        # Hand-written sidecar, no operation: quick, no reference_source.
        no_op = _sidecar("case1", seed=8, suite=suite)
        del no_op["operation"]
        path = _capture(
            tmp / "captured", "case1", _ternary_graph(_UIDS0), no_op, suite=suite
        )
        bundle_dir = tmp / "no_operation"
        bundle_dir.mkdir()
        r = _import_capture(path, bundle_dir)
        assert r.returncode == 0, r.stderr
        assert "WARN" not in r.stderr, r.stderr
        assert _tiers_by_source(bundle_dir) == {None: "quick"}
        assert _only_metadata(bundle_dir)["seed"] == 8
        print("  PASS: import_sidecar_outside_capture_layout")


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

        r = _import_capture(graph_path, bad_dir, "--meta", "inputs=[1]")
        assert r.returncode == 1, f"non-object inputs must fail:\n{r.stderr}"
        assert "Traceback" not in r.stderr, r.stderr
        assert "inputs: expected a JSON object" in r.stderr, r.stderr
        assert _inventory(bad_dir) == {}, "refusal wrote files"
        print("  PASS: import_meta_override_checks")


def test_import_meta_value_types():
    """Only the non-string metadata keys parse as JSON; NaN and inf are refused.

    A string key keeps its text as given, so ``notes=NaN`` cannot write a bare
    NaN that makes the whole sweep.json unreadable. A typed key that is not a
    finite JSON value is refused with a message, not a traceback.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph_path = _capture(
            tmp / "captured", "case0", _ternary_graph(_UIDS0), _sidecar("case0")
        )

        bundle_dir = tmp / "strings"
        bundle_dir.mkdir()
        r = _import_capture(
            graph_path,
            bundle_dir,
            "--meta",
            "notes=NaN",
            "--meta",
            "precision_profile=123",
            "--meta",
            "minimum_vram_mb=512",
        )
        assert r.returncode == 0, r.stderr
        meta = _only_metadata(bundle_dir)
        assert meta["notes"] == "NaN", meta
        assert meta["precision_profile"] == "123", meta
        assert meta["minimum_vram_mb"] == 512, meta
        sweep = next(bundle_dir.rglob("sweep.json")).read_text()
        assert "NaN" not in sweep.replace('"NaN"', ""), "bare NaN written"

        for bad in ("seed=NaN", "seed=Infinity", "seed=1e400", "seed=abc"):
            bad_dir = tmp / f"bad_{bad.split('=')[1]}"
            bad_dir.mkdir()
            r = _import_capture(graph_path, bad_dir, "--meta", bad)
            assert r.returncode == 1, f"{bad} must fail:\n{r.stderr}"
            assert "Traceback" not in r.stderr, f"{bad}: crashed:\n{r.stderr}"
            assert "import_graph: --meta seed:" in r.stderr, r.stderr
            assert _inventory(bad_dir) == {}, f"{bad}: refusal wrote files"
        print("  PASS: import_meta_value_types")


def test_import_malformed_sidecar_refused():
    """A sidecar that exists but is unusable is refused, with or without --strict.

    Proceeding without it would land the case in quick/ with no
    reference_source. place_bundles, which knows the suite from the path, warns
    and places the case without the sidecar instead of crashing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        capture_dir = tmp / "captured"
        bad = {
            "truncated": '{"seed": 4',
            "not an object": "[1, 2]",
            # Parses, but remap_meta_inputs would crash on int("x").
            "inputs key not a UID": '{"seed": 4, "inputs": {"x": {"kind": "free"}}}',
            "inputs not an object": '{"seed": 4, "inputs": [1]}',
            "operation empty": '{"seed": 4, "operation": ""}',
            "operation not a string": '{"seed": 4, "operation": 7}',
        }
        for i, (label, text) in enumerate(bad.items()):
            path = _capture(capture_dir, f"case{i}", _ternary_graph(_UIDS0), text)
            sidecar_path = path.with_suffix(".meta.json")
            for strict in ([], ["--strict"]):
                bundle_dir = tmp / f"refused{i}{len(strict)}"
                bundle_dir.mkdir()
                r = _import_capture(path, bundle_dir, "--seed", "3", *strict)
                assert r.returncode == 1, f"{label} {strict}: must fail:\n{r.stderr}"
                assert "Traceback" not in r.stderr, f"{label}: crashed:\n{r.stderr}"
                assert (
                    f"ERROR: bad sidecar {sidecar_path}" in r.stderr
                ), f"{label}: no error:\n{r.stderr}"
                assert _inventory(bundle_dir) == {}, f"{label}: refusal wrote files"

        placed_dir = tmp / "placed"
        r = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "place_bundles.py"),
                "--capture-dir",
                str(capture_dir),
                "--output-dir",
                str(placed_dir),
            ],
            check=False,
        )
        assert r.returncode == 0, f"place_bundles failed:\n{r.stderr}"
        assert "Traceback" not in r.stderr, r.stderr
        assert "WARN: bad meta" in r.stderr and "not a tensor UID" in r.stderr, r.stderr
        placed = _tiers_by_source(placed_dir)
        assert len(placed) == len(bad), f"place_bundles lost cases: {placed}"
        print("  PASS: import_malformed_sidecar_refused")


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


def _retier_graph_template(clip):
    return {
        "name": "g",
        "io_data_type": "${case.io_data_type}",
        "tensors": [
            {
                "uid": 0,
                "dims": "${case.dims}",
                "strides": "${case.strides}",
                "data_type": "${case.data_type}",
            }
        ],
        "nodes": [{"type": "PointwiseAttributes", "inputs": {"relu_lower_clip": clip}}],
    }


def _retier_case(case_id, clip_attr=None):
    values = {
        "io_data_type": "float",
        "tensors": [
            {"uid": 0, "dims": [1, 2], "strides": [2, 1], "data_type": "float"}
        ],
    }
    if clip_attr is not None:
        values["attributes"] = {"relu_lower_clip": clip_attr}
    return {
        "id": case_id,
        "values": values,
        "metadata": {"format_version": 1, "notes": case_id},
    }


def _retier_write(root, tier, template, cases, claims):
    d = root / tier / "Op" / "Default"
    d.mkdir(parents=True)
    for name, obj in (
        ("graph.template.json", template),
        ("sweep.json", {"version": 1, "cases": cases}),
        ("support.json", {"version": 1, "claims": claims}),
    ):
        with open(d / name, "w") as f:
            json.dump(obj, f, indent=2)


def _retier_load(root, tier):
    d = root / tier / "Op" / "Default"
    return {
        n: json.load(open(d / n))
        for n in ("graph.template.json", "sweep.json", "support.json")
    }


def test_retier_bundles():
    """Moving cases between tiers keeps every case's expanded graph and claims.

    The quick template hard-codes relu_lower_clip while the full template takes it
    from the case, so a case moving from quick to a tier that uses the full
    template has to gain the literal as an explicit attribute. Cases that stay put,
    single-graph bundles and per-engine claims must come through unchanged, and a
    plan that loses or duplicates a case must be refused before anything is written.
    """
    script = SCRIPT_DIR / "retier_bundles.py"
    win = {"gfx1151": ["windows"]}
    lin = {"gfx942": ["linux"]}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        _retier_write(
            root,
            "quick",
            _retier_graph_template(None),
            [_retier_case("a"), _retier_case("b")],
            {
                "MIOPEN_ENGINE": [
                    {"cases": ["a"], "support": win},
                    {"cases": ["b"], "support": lin},
                ]
            },
        )
        _retier_write(
            root,
            "full",
            _retier_graph_template("${case.attributes.relu_lower_clip}"),
            [_retier_case("0c", clip_attr=0.5)],
            {"MIOPEN_ENGINE": [{"cases": ["0c"], "support": win}]},
        )
        single = root / "quick" / "Op2" / "nchw" / "fp32" / "Small"
        single.mkdir(parents=True)
        with open(single / "Small.json", "w") as f:
            json.dump({"name": "s"}, f)
        with open(single / "Small.meta.json", "w") as f:
            json.dump({"format_version": 1}, f)

        asg = Path(tmp) / "asg.json"

        def write_assignments(rows):
            with open(asg, "w") as f:
                json.dump(
                    [dict(current_tier=t, key=k, proposed_tier=n) for t, k, n in rows],
                    f,
                )

        plan = [
            ("quick", "Op/Default/a", "standard"),
            ("quick", "Op/Default/b", "quick"),
            ("full", "Op/Default/0c", "quick"),
            ("quick", "Op2/nchw/fp32/Small", "full"),
        ]

        # A case without an assignment is refused and nothing is written.
        write_assignments(plan[:-1])
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode != 0 and "no assignment" in r.stderr, r.stderr
        assert (single / "Small.json").is_file() and not (root / "standard").exists()

        # Two copies of one id may not land in the same sweep.
        dup = root / "full" / "Op" / "Default"
        sweep = json.load(open(dup / "sweep.json"))
        sweep["cases"].append(_retier_case("a", clip_attr=1.0))
        json.dump(sweep, open(dup / "sweep.json", "w"))
        write_assignments(plan + [("full", "Op/Default/a", "standard")])
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode != 0 and "twice" in r.stderr, r.stderr
        sweep["cases"].pop()
        json.dump(sweep, open(dup / "sweep.json", "w"))

        write_assignments(plan)
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ]
        )
        assert r.returncode == 0, r.stderr
        assert "identical before and after" in r.stdout, r.stdout

        std, quick = _retier_load(root, "standard"), _retier_load(root, "quick")
        assert not (root / "full" / "Op").exists(), "emptied tier dir should be removed"
        assert (
            root / "full" / "Op2" / "nchw" / "fp32" / "Small" / "Small.meta.json"
        ).is_file()
        assert not single.exists()
        # Unified template: the placeholder form, in every tier.
        placeholder = "${case.attributes.relu_lower_clip}"
        for tier_files in (std, quick):
            assert (
                tier_files["graph.template.json"]["nodes"][0]["inputs"][
                    "relu_lower_clip"
                ]
                == placeholder
            )
        # "a" came from the literal-null template and carries it explicitly.
        (case_a,) = std["sweep.json"]["cases"]
        assert case_a["id"] == "a" and case_a["values"]["attributes"] == {
            "relu_lower_clip": None
        }
        # "0c" kept its own value; "b" (stayed) gained the explicit null.
        by_id = {c["id"]: c for c in quick["sweep.json"]["cases"]}
        assert by_id["0c"]["values"]["attributes"] == {"relu_lower_clip": 0.5}
        assert by_id["b"]["values"]["attributes"] == {"relu_lower_clip": None}
        # Claims follow their case: b and 0c have different arch maps, so two groups.
        groups = quick["support.json"]["claims"]["MIOPEN_ENGINE"]
        assert {tuple(g["cases"]): g["support"] for g in groups} == {
            ("b",): lin,
            ("0c",): win,
        }
        # Canonical text: "0c" arrived after "b" but sorts first, and the sidecar
        # tools refuse any file whose groups are not in sorted-first-case order.
        sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))
        from support_sidecar import parse

        for tier in ("quick", "standard"):
            path = root / tier / "Op" / "Default" / "support.json"
            parse(path, path.read_text("utf-8"))
        assert std["support.json"]["claims"]["MIOPEN_ENGINE"] == [
            {"cases": ["a"], "support": win}
        ]
        print("  PASS: retier_bundles")


def test_retier_keeps_import_mode():
    """Moved cases keep their sweep's import_mode; mixed import_modes are refused."""
    script = SCRIPT_DIR / "retier_bundles.py"
    marker = {"grouping": ["template-attributes"]}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        template = _retier_graph_template(None)
        _retier_write(
            root, "quick", template, [_retier_case("a"), _retier_case("b")], {}
        )
        _retier_write(root, "full", template, [_retier_case("c")], {})
        for tier in ("quick", "full"):
            path = root / tier / "Op" / "Default" / "sweep.json"
            sweep = json.loads(path.read_text())
            if tier == "quick":
                sweep = {"version": 1, "import_mode": marker, "cases": sweep["cases"]}
            path.write_text(json.dumps(sweep))
        asg = Path(tmp) / "asg.json"

        def retier(rows):
            with open(asg, "w") as f:
                json.dump(
                    [dict(current_tier=t, key=k, proposed_tier=n) for t, k, n in rows],
                    f,
                )
            return run(
                [
                    sys.executable,
                    script,
                    "--bundle-dir",
                    root,
                    "--assignments",
                    asg,
                    "--apply",
                ],
                check=False,
            )

        r = retier(
            [
                ("quick", "Op/Default/a", "quick"),
                ("quick", "Op/Default/b", "full"),
                ("full", "Op/Default/c", "full"),
            ]
        )
        assert r.returncode != 0 and "import_mode" in r.stderr, r.stderr
        assert not (root / "standard").exists()

        r = retier(
            [
                ("quick", "Op/Default/a", "quick"),
                ("quick", "Op/Default/b", "standard"),
                ("full", "Op/Default/c", "full"),
            ]
        )
        assert r.returncode == 0, r.stderr
        moved = json.loads((root / "standard/Op/Default/sweep.json").read_text())
        assert moved.get("import_mode") == marker, moved
        assert [c["id"] for c in moved["cases"]] == ["b"], moved
        kept = json.loads((root / "full/Op/Default/sweep.json").read_text())
        assert "import_mode" not in kept, kept
        print("  PASS: retier_keeps_import_mode")


# --------------------------------------------------------------------------
# Grouped import (opt-in flags of import_graph.py)
# --------------------------------------------------------------------------

_GROUPED_FLAGS = (
    "--op-dir",
    "GroupedOp",
    "--template-attributes",
    "--template-alignment",
    "--skeleton-value-form",
    "--only-op-dir",
    "--require-marked-sweep",
)


def _grouped_graph(causal=False, alignment=16, scale=None, runtime=False, dims=None):
    """A one-node graph with node attributes, tensor alignment and a scale tensor.

    ``scale`` bakes a by-value scale tensor; ``runtime`` makes it a runtime
    pass-by-value tensor instead. With neither, the graph has no scale tensor.
    """
    dims = dims or [2, 3, 4, 5]
    strides = [60, 20, 5, 1] if dims == [2, 3, 4, 5] else [2 * 3 * 4, 3 * 4, 4, 1]
    graph = _make_graph("Grouped", [0, 1], dims, strides, "float")
    node = graph["nodes"][0]
    node["attributes"] = {"causal": causal, "window": None, "mode": "AUTO"}
    node["compute_data_type"] = "float"
    for t in graph["tensors"]:
        t["alignment"] = alignment
        t["is_runtime_pass_by_value"] = False
    graph["tensors"][0]["name"] = "X"
    graph["tensors"][1]["name"] = "Y"
    if scale is not None or runtime:
        node["inputs"]["scale_tensor_uid"] = 2
        tensor = {
            "uid": 2,
            "name": "SCALE",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            "alignment": alignment,
            "is_runtime_pass_by_value": runtime,
        }
        if not runtime:
            tensor["value"] = scale
            tensor["value_type"] = "FloatValue"
        graph["tensors"].append(tensor)
    return graph


def _sweep_dirs(bundle_dir):
    """Sorted sweep directories, relative to bundle_dir, as posix strings."""
    return sorted(
        p.parent.relative_to(bundle_dir).as_posix()
        for p in bundle_dir.rglob("sweep.json")
    )


def _load_sweep(bundle_dir, rel):
    with open(bundle_dir / rel / "sweep.json") as f:
        return json.load(f)


def _write_marked_sweep(bundle_dir, rel, graph, marker):
    """Write a one-case sweep for graph at bundle_dir/rel with an import_mode."""
    from bundle_utils import TENSOR_ALWAYS

    out = bundle_dir / rel
    out.mkdir(parents=True)
    template = copy.deepcopy(graph)
    for t in template["tensors"]:
        for fld in TENSOR_ALWAYS:
            t[fld] = f"${{case.{fld}}}"
    values = {
        "tensors": [
            {"uid": t["uid"], **{fld: t[fld] for fld in TENSOR_ALWAYS}}
            for t in graph["tensors"]
        ]
    }
    sweep = {"version": 1, "cases": [{"id": "only", "values": values, "metadata": {}}]}
    if marker is not None:
        sweep = {"version": 1, "import_mode": marker, "cases": sweep["cases"]}
    (out / "graph.template.json").write_text(json.dumps(template, indent=2) + "\n")
    (out / "sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")


def test_inventory_keys_are_forward_slash_normalized():
    """_inventory() keys stay forward-slash even when relative_to() would use backslashes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        nested = bundle_dir / "quick" / "Relu" / "Marked"
        nested.mkdir(parents=True)
        (nested / "sweep.json").write_bytes(b"{}")

        class _BackslashRelPath:
            def __init__(self, posix_str):
                self._posix = posix_str

            def __str__(self):
                return self._posix.replace("/", "\\")

            def as_posix(self):
                return self._posix

        real_relative_to = Path.relative_to

        def fake_relative_to(self, *args, **kwargs):
            return _BackslashRelPath(real_relative_to(self, *args, **kwargs).as_posix())

        with patch.object(Path, "relative_to", fake_relative_to):
            keys = _inventory(bundle_dir)
        assert "quick/Relu/Marked/sweep.json" in keys, keys
        assert not any("\\" in k for k in keys), keys
        print("  PASS: inventory_keys_are_forward_slash_normalized")


def test_grouped_default_mode_ignores_marked_sweep():
    """Default mode never appends into a marked sweep, but reports a duplicate."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        _write_marked_sweep(
            bundle_dir, "quick/Relu/Marked", g, {"grouping": ["template-attributes"]}
        )
        before = _inventory(bundle_dir)

        r = _import(tmp, "dup", g, bundle_dir, "--strict", check=False)
        assert r.returncode == 1, f"strict duplicate should exit 1:\n{r.stderr}"
        assert "DUPLICATE" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before, "duplicate check wrote files"

        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        r = _import(tmp, "new", g2, bundle_dir)
        assert r.returncode == 0, r.stderr
        assert "created new bundle" in r.stderr, r.stderr
        assert (bundle_dir / "quick/Relu/Marked/sweep.json").read_bytes() == before[
            "quick/Relu/Marked/sweep.json"
        ], "default mode changed a marked sweep"
        assert "import_mode" not in _load_sweep(bundle_dir, "quick/Relu/Nchw")
        assert _sweep_dirs(bundle_dir) == ["quick/Relu/Marked", "quick/Relu/Nchw"]
        print("  PASS: grouped_default_mode_ignores_marked_sweep")


def test_grouped_creates_marked_sweep():
    """Opt-in import writes import_mode and templates attributes and alignment."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g = _grouped_graph(alignment=2)
        r = _import(tmp, "g", g, bundle_dir, *_GROUPED_FLAGS, "--topology", "Topo")
        assert r.returncode == 0, r.stderr
        rel = "quick/GroupedOp/Topo"
        assert _sweep_dirs(bundle_dir) == [rel], _sweep_dirs(bundle_dir)
        sweep = _load_sweep(bundle_dir, rel)
        assert list(sweep) == ["version", "import_mode", "cases"], list(sweep)
        assert sweep["import_mode"] == {
            "grouping": [
                "skeleton-value-form",
                "template-alignment",
                "template-attributes",
            ],
            "format": "compact",
        }, sweep["import_mode"]
        with open(bundle_dir / rel / "graph.template.json") as f:
            template = json.load(f)
        node = template["nodes"][0]
        assert node["attributes"] == {
            "causal": "${case.attributes.attributes__causal}",
            "window": "${case.attributes.attributes__window}",
            "mode": "${case.attributes.attributes__mode}",
        }, node["attributes"]
        assert node["compute_data_type"] == "${case.attributes.compute_data_type}"
        assert (
            node["type"] == "GroupedAttributes" and node["inputs"]["x_tensor_uid"] == 0
        )
        assert [t["alignment"] for t in template["tensors"]] == [
            "${case.alignment.0}",
            "${case.alignment.1}",
        ]
        values = sweep["cases"][0]["values"]
        assert values["alignment"] == {"0": 2, "1": 2}, values
        assert values["attributes"]["attributes__window"] is None, values

        from bundle_utils import canon, expand

        assert canon(expand(template, values)) == canon(
            g
        ), "template does not round-trip"
        sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))
        from support_matrix_units import expand_template_graph

        loaded = expand_template_graph(template, sweep["cases"][0])
        assert canon(loaded) == canon(g), "the loader mirror expands it differently"
        print("  PASS: grouped_creates_marked_sweep")


def test_grouped_append_attributes_round_trip():
    """A case that differs only in attributes, alignment or baked value appends."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        from bundle_utils import canon, canonical_uid_map_by_name, expand, remap_graph

        graphs = [
            _grouped_graph(scale=0.5),
            _grouped_graph(causal=True, alignment=4, scale=-1.0),
        ]
        graphs[1]["nodes"][0]["attributes"]["window"] = 7
        for i, g in enumerate(graphs):
            r = _import(tmp, f"g{i}", g, bundle_dir, *_GROUPED_FLAGS, "--strict")
            assert r.returncode == 0, r.stderr
        assert "appended" in r.stderr, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        with open(bundle_dir / rel / "graph.template.json") as f:
            template = json.load(f)
        (scale,) = [t for t in template["tensors"] if t["name"] == "SCALE"]
        assert scale["value"] == "${case.value}", scale
        cases = _load_sweep(bundle_dir, rel)["cases"]
        assert len(cases) == 2, cases
        for case, g in zip(cases, graphs):
            want = remap_graph(g, canonical_uid_map_by_name(g))
            assert canon(expand(template, case["values"])) == canon(want), case["id"]
        print("  PASS: grouped_append_attributes_round_trip")


def test_grouped_value_forms_never_share_sweep():
    """Baked, runtime and absent scale tensors give three sweeps."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        forms = [
            _grouped_graph(scale=0.5),
            _grouped_graph(runtime=True),
            _grouped_graph(scale=0.25),
            _grouped_graph(),
        ]
        for i, g in enumerate(forms):
            r = _import(tmp, f"g{i}", g, bundle_dir, *_GROUPED_FLAGS, "--strict")
            assert r.returncode == 0, r.stderr
        dirs = _sweep_dirs(bundle_dir)
        assert len(dirs) == 3, dirs
        counts = sorted(len(_load_sweep(bundle_dir, d)["cases"]) for d in dirs)
        assert counts == [1, 1, 2], counts

        from bundle_utils import skeleton_hash

        baked, runtime = forms[0], forms[1]
        assert skeleton_hash(baked) == skeleton_hash(runtime)
        assert skeleton_hash(baked, value_form=True) != skeleton_hash(
            runtime, value_form=True
        )
        print("  PASS: grouped_value_forms_never_share_sweep")


def test_grouped_only_op_dir_blocks_cross_appends():
    """--only-op-dir never appends across tiers or operation folders."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        flags = ("--template-attributes", "--only-op-dir")
        g1 = _grouped_graph()
        g2 = _grouped_graph(causal=True)
        g3 = _grouped_graph(alignment=8)
        r = _import(tmp, "g1", g1, bundle_dir, *flags, "--op-dir", "OpA")
        assert r.returncode == 0, r.stderr
        r = _import(
            tmp, "g2", g2, bundle_dir, *flags, "--op-dir", "OpA", "--tier", "standard"
        )
        assert r.returncode == 0 and "created new bundle" in r.stderr, r.stderr
        r = _import(tmp, "g3", g3, bundle_dir, *flags, "--op-dir", "OpB")
        assert r.returncode == 0 and "created new bundle" in r.stderr, r.stderr
        assert _sweep_dirs(bundle_dir) == [
            "quick/OpA/Nchw",
            "quick/OpB/Nchw",
            "standard/OpA/Nchw",
        ], _sweep_dirs(bundle_dir)
        print("  PASS: grouped_only_op_dir_blocks_cross_appends")


def test_grouped_require_marked_sweep():
    """--require-marked-sweep refuses unmarked and differently marked sweeps."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        _write_marked_sweep(bundle_dir, "quick/Relu/Unmarked", g, None)
        _write_marked_sweep(
            bundle_dir, "quick/Relu/Other", g, {"grouping": ["template-alignment"]}
        )
        before = _inventory(bundle_dir)
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        flags = ("--template-attributes", "--only-op-dir", "--require-marked-sweep")
        r = _import(tmp, "g2", g2, bundle_dir, *flags, "--topology", "Mine")
        assert r.returncode == 0 and "created new bundle" in r.stderr, r.stderr
        for rel in ("quick/Relu/Unmarked", "quick/Relu/Other"):
            for name in ("graph.template.json", "sweep.json"):
                key = f"{rel}/{name}"
                assert (bundle_dir / key).read_bytes() == before[key], key

        g3 = _make_graph("Relu", [0, 1], [3, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(tmp, "g3", g3, bundle_dir, *flags, "--topology", "Mine")
        assert r.returncode == 0 and "appended" in r.stderr, r.stderr
        assert len(_load_sweep(bundle_dir, "quick/Relu/Mine")["cases"]) == 2
        print("  PASS: grouped_require_marked_sweep")


def test_grouped_case_id():
    """--case-id sets the id and refuses an id the sweep already has."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        r = _import(
            tmp,
            "g1",
            _grouped_graph(),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--case-id",
            "fp32_a1",
        )
        assert r.returncode == 0, r.stderr
        r = _import(
            tmp,
            "g2",
            _grouped_graph(causal=True),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--case-id",
            "fp32_b2",
        )
        assert r.returncode == 0 and "appended case 'fp32_b2'" in r.stderr, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        ids = [c["id"] for c in _load_sweep(bundle_dir, rel)["cases"]]
        assert ids == ["fp32_a1", "fp32_b2"], ids

        before = _inventory(bundle_dir)
        r = _import(
            tmp,
            "g3",
            _grouped_graph(alignment=4),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--case-id",
            "fp32_a1",
            check=False,
        )
        assert r.returncode == 1 and "already exists" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before, "refused import changed the tree"

        r = _import(
            tmp, "g4", _grouped_graph(), bundle_dir, "--case-id", "Bad-Id", check=False
        )
        assert r.returncode == 1 and "lowercase_snake_case" in r.stderr, r.stderr
        print("  PASS: grouped_case_id")


def _grouped_named_graph():
    """A ternary graph whose creation-order UIDs differ from sorted-name order."""
    graph = _ternary_graph({"x": 0, "dy": 1, "scale": 2, "out": 3})
    for t in graph["tensors"]:
        t["alignment"] = 16
        t["is_runtime_pass_by_value"] = False
    return graph


def _import_grouped_capture(tmp, graph, sidecar):
    """Grouped import of one capture; returns (bundle_dir, metadata, template, case)."""
    bundle_dir = tmp / "bundles"
    bundle_dir.mkdir()
    graph_path = _capture(tmp / "capture", "named_case", graph, sidecar)
    r = _import_capture(graph_path, bundle_dir, *_GROUPED_FLAGS, "--topology", "Topo")
    assert r.returncode == 0, r.stderr
    (rel,) = _sweep_dirs(bundle_dir)
    (case,) = _load_sweep(bundle_dir, rel)["cases"]
    with open(bundle_dir / rel / "graph.template.json") as f:
        template = json.load(f)
    return bundle_dir, case["metadata"], template, case


def test_grouped_import_keeps_values_inputs():
    """A format_version 2 case with a values input survives grouped import and re-expansion."""
    from bundle_utils import (
        canon,
        canonical_uid_map_by_name,
        expand,
        inputs_by_name,
        remap_graph,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph = _grouped_named_graph()
        uid_map = canonical_uid_map_by_name(graph)
        assert any(old != new for old, new in uid_map.items()), uid_map
        values = {"kind": "values", "values": [64, 32], "repeat": True, "seed": 11}
        sidecar = {
            "format_version": 2,
            "seed": 7,
            "inputs": {"0": values, "1": {"seed": 22}, "2": {"seed": 33}},
        }
        _, meta, template, case = _import_grouped_capture(tmp, graph, sidecar)
        assert meta["format_version"] == 2 and meta["seed"] == 7, meta
        by_name = inputs_by_name(meta, template)
        assert json.dumps(by_name, sort_keys=True) == json.dumps(
            inputs_by_name(sidecar, graph), sort_keys=True
        ), by_name
        assert uid_map[0] != 0, uid_map
        assert json.dumps(by_name["x"]) == json.dumps(values), by_name["x"]
        assert canon(expand(template, case["values"])) == canon(
            remap_graph(graph, uid_map)
        ), "the case does not re-expand to the imported graph"
        print("  PASS: grouped_import_keeps_values_inputs")


def test_grouped_import_keeps_per_input_seeds():
    """A grouped import that renumbers UIDs keeps each tensor's seed and fill spec."""
    from bundle_utils import canonical_uid_map_by_name, inputs_by_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph = _grouped_named_graph()
        uid_map = canonical_uid_map_by_name(graph)
        sidecar = {
            "format_version": 1,
            "seed": 1234,
            "inputs": {
                "0": {"seed": 101},
                "1": {"seed": 202},
                "2": {"kind": "fixed", "seed": 303, "value": 0.125},
            },
        }
        _, meta, template, _ = _import_grouped_capture(tmp, graph, sidecar)
        assert sorted(meta["inputs"]) == sorted(
            str(uid_map[int(u)]) for u in sidecar["inputs"]
        ), meta["inputs"]
        assert meta["inputs"] != sidecar["inputs"], "the import did not renumber UIDs"
        assert inputs_by_name(meta, template) == inputs_by_name(sidecar, graph), meta
        print("  PASS: grouped_import_keeps_per_input_seeds")


def test_grouped_import_keeps_aliases_start_offsets_and_live_rows():
    """A grouped import that renumbers UIDs keeps each alias, start offset, live-row entry and
    view seed."""
    from bundle_utils import aliases_by_name, canonical_uid_map_by_name, inputs_by_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graph = _grouped_named_graph()
        uid_map = canonical_uid_map_by_name(graph)
        assert uid_map[0] != 0 and uid_map[1] != 1, uid_map
        sidecar = {
            "format_version": 2,
            "seed": 1234,
            "inputs": {"0": {"seed": 101}, "1": {"seed": 202}, "2": {"seed": 303}},
            "aliases": [{"uid": 1, "of": 0, "byte_offset": 0}],
            "start_offsets": [{"uid": 0, "start_offset": 4}],
            "live_rows": [{"uid": 2, "axis": 1, "lengths": 1}],
        }
        _, meta, template, _ = _import_grouped_capture(tmp, graph, sidecar)
        assert meta["aliases"] == [
            {"uid": uid_map[1], "of": uid_map[0], "byte_offset": 0}
        ], meta["aliases"]
        assert meta["start_offsets"] == [{"uid": uid_map[0], "start_offset": 4}], meta[
            "start_offsets"
        ]
        assert meta["live_rows"] == [
            {"uid": uid_map[2], "axis": 1, "lengths": uid_map[1]}
        ], meta["live_rows"]
        assert aliases_by_name(meta, template) == aliases_by_name(sidecar, graph), meta
        assert aliases_by_name(meta, template) == [
            {"view": "dy", "of": "x", "byte_offset": 0}
        ], meta
        assert inputs_by_name(meta, template)["dy"] == {"seed": 202}, meta["inputs"]
        print("  PASS: grouped_import_keeps_aliases_start_offsets_and_live_rows")


def test_grouped_topology():
    """--topology names the folder and refuses a mismatch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        r = _import(
            tmp,
            "g1",
            _grouped_graph(scale=0.5),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--topology",
            "Baked",
        )
        assert r.returncode == 0, r.stderr
        assert _sweep_dirs(bundle_dir) == ["quick/GroupedOp/Baked"]
        before = _inventory(bundle_dir)

        r = _import(
            tmp,
            "g2",
            _grouped_graph(scale=0.25),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--topology",
            "Other",
            check=False,
        )
        assert r.returncode == 1 and "--topology Other" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before, "refused append changed the tree"

        r = _import(
            tmp,
            "g3",
            _grouped_graph(runtime=True),
            bundle_dir,
            *_GROUPED_FLAGS,
            "--topology",
            "Baked",
            check=False,
        )
        assert r.returncode == 1 and "holds another sweep" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before, "refused create changed the tree"

        r = _import(
            tmp, "g4", _grouped_graph(), bundle_dir, "--topology", "../x", check=False
        )
        assert r.returncode == 1 and "not a folder name" in r.stderr, r.stderr
        print("  PASS: grouped_topology")


def test_grouped_attribute_keys_prefixed_per_node():
    """An attribute key that two nodes share gets an n<index>__ prefix."""
    from bundle_utils import canon, expand

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g = _make_graph("Relu", [0, 1, 2], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        for i, t in enumerate(g["tensors"]):
            t["name"] = f"T{i}"
        g["nodes"][0]["outputs"] = {"y_tensor_uid": 1}
        g["nodes"].append(
            {
                "type": "ReluAttributes",
                "name": "",
                "inputs": {"x_tensor_uid": 1},
                "outputs": {"y_tensor_uid": 2},
            }
        )
        g["nodes"][0]["attributes"] = {"mode": "A", "only_first": 1}
        g["nodes"][1]["attributes"] = {"mode": "B"}
        r = _import(tmp, "g", g, bundle_dir, "--template-attributes")
        assert r.returncode == 0, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        with open(bundle_dir / rel / "graph.template.json") as f:
            template = json.load(f)
        assert template["nodes"][0]["attributes"] == {
            "mode": "${case.attributes.n0__attributes__mode}",
            "only_first": "${case.attributes.attributes__only_first}",
        }, template["nodes"][0]["attributes"]
        assert template["nodes"][1]["attributes"] == {
            "mode": "${case.attributes.n1__attributes__mode}"
        }, template["nodes"][1]["attributes"]
        values = _load_sweep(bundle_dir, rel)["cases"][0]["values"]
        assert canon(expand(template, values)) == canon(g), "no round-trip"
        print("  PASS: grouped_attribute_keys_prefixed_per_node")


def test_grouped_runs_are_deterministic():
    """Two identical grouped runs write identical bytes."""
    trees = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graphs = [
            _grouped_graph(scale=0.5),
            _grouped_graph(causal=True, scale=0.5),
            _grouped_graph(runtime=True, dims=[1, 2, 3, 4]),
            _grouped_graph(alignment=2, scale=2.0),
        ]
        for run_index in range(2):
            bundle_dir = tmp / f"bundles{run_index}"
            bundle_dir.mkdir()
            for i, g in enumerate(graphs):
                r = _import(tmp, f"g{i}", g, bundle_dir, *_GROUPED_FLAGS, "--strict")
                assert r.returncode == 0, r.stderr
            trees.append(_inventory(bundle_dir))
            for rel in _sweep_dirs(bundle_dir):
                text = (bundle_dir / rel / "sweep.json").read_text()
                assert _is_compact(text), f"{rel}/sweep.json is not compact"
    assert trees[0] == trees[1], "grouped runs differ"
    print("  PASS: grouped_runs_are_deterministic")


def _grouped_top_level_round_trip(field, first, second):
    """Two graphs that differ only in top-level ``field`` share one grouped sweep.

    Both cases must expand back to their graphs with ``expand()`` and with the
    loader mirror. A default-mode import keeps the field literal.
    """
    from bundle_utils import canon, canonical_uid_map_by_name, expand, remap_graph

    sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))
    from support_matrix_units import expand_template_graph

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        graphs = []
        for value in (first, second):
            g = _grouped_graph(scale=0.5)
            g["is_override_shape_enabled"] = False
            g["min_required_engine_api_version"] = {"major": 1, "minor": 0, "patch": 0}
            g[field] = value
            graphs.append(g)
        for i, g in enumerate(graphs):
            r = _import(tmp, f"g{i}", g, bundle_dir, *_GROUPED_FLAGS, "--strict")
            assert r.returncode == 0, r.stderr
        assert "appended" in r.stderr, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        with open(bundle_dir / rel / "graph.template.json") as f:
            template = json.load(f)
        for fld in ("is_override_shape_enabled", "min_required_engine_api_version"):
            assert template[fld] == f"${{case.{fld}}}", template[fld]
        cases = _load_sweep(bundle_dir, rel)["cases"]
        assert [c["values"][field] for c in cases] == [first, second], cases
        for case, g in zip(cases, graphs):
            want = canon(remap_graph(g, canonical_uid_map_by_name(g)))
            assert canon(expand(template, case["values"])) == want, case["id"]
            assert canon(expand_template_graph(template, case)) == want, case["id"]

        default_dir = tmp / "default"
        default_dir.mkdir()
        r = _import(tmp, "d", graphs[1], default_dir)
        assert r.returncode == 0, r.stderr
        (rel,) = _sweep_dirs(default_dir)
        with open(default_dir / rel / "graph.template.json") as f:
            assert json.load(f)[field] == second, "default mode templated the field"


def test_grouped_engine_api_version_per_case():
    """Cases with different min_required_engine_api_version share a sweep."""
    _grouped_top_level_round_trip(
        "min_required_engine_api_version",
        {"major": 1, "minor": 0, "patch": 0},
        {"major": 1, "minor": 3, "patch": 0},
    )
    print("  PASS: grouped_engine_api_version_per_case")


def test_grouped_override_shape_flag_per_case():
    """Cases with different is_override_shape_enabled share a sweep."""
    _grouped_top_level_round_trip("is_override_shape_enabled", False, True)
    print("  PASS: grouped_override_shape_flag_per_case")


# --------------------------------------------------------------------------
# sweep.json layout (dump_sweep and the import_mode format key)
# --------------------------------------------------------------------------


def _compact_sample_sweep():
    """A sweep with every member kind the compact layout treats differently."""
    case = _retier_case("a", clip_attr=0.5)
    case["values"]["alignment"] = {"0": 16}
    case["values"]["tensors"].append(
        {"uid": 1, "dims": [3], "strides": [1], "data_type": "half", "value": 0.25}
    )
    case["golden"] = {"path": "golden/a/a.json"}
    case["values"][7] = {1: "int key"}
    case[3] = [1, 2]
    case["tensor_patches"] = [{"uid": 1, "value": None}]
    empty = {"id": "b", "values": {}, "metadata": {}}
    no_tensors = {"id": "c", "values": {"tensors": []}, "metadata": {"seed": 1}}
    return {
        "version": 1,
        "import_mode": {"grouping": ["template-attributes"], "format": "compact"},
        "cases": [case, empty, no_tensors],
    }


def _is_compact(text):
    """True if text is the compact layout of the sweep it loads to."""
    from bundle_utils import dump_sweep

    return text == dump_sweep(json.loads(text), "compact")


def test_sweep_format_compact_round_trip():
    """dump_sweep compact loads to the same object, keeps key order, and puts one tensor per line."""
    from bundle_utils import dump_sweep, sweep_format

    sweep = _compact_sample_sweep()
    text = dump_sweep(sweep, "compact")
    back = json.loads(text)
    assert json.dumps(back) == json.dumps(sweep), text
    assert back["cases"][0]["values"]["7"] == {"1": "int key"}, text
    assert '        "7": {"1": "int key"}' in text.splitlines(), text
    assert '      "3": [1, 2],' in text.splitlines(), text
    top = {"version": 1, 5: {"x": 1}, "cases": [{"id": "a", 2: True}]}
    assert json.loads(dump_sweep(top, "compact")) == json.loads(json.dumps(top))
    lines = text.splitlines()
    assert '      "values": {' in lines, text
    assert '        "tensors": [' in lines, text
    tensor_lines = [ln for ln in lines if ln.startswith(" " * 10 + '{"uid": ')]
    assert len(tensor_lines) == 2, text
    assert (
        '        "attributes": {"relu_lower_clip": 0.5},' in lines
    ), "values members are not one compact line each"
    assert any(ln.startswith('      "metadata": {"format_version": 1') for ln in lines)
    assert '      "golden": {"path": "golden/a/a.json"},' in lines, text
    assert '  "import_mode": {' in lines, "top level is not written as indent=2"
    assert text.endswith("}\n") and _is_compact(text)
    assert sweep_format(sweep) == "compact"

    assert dump_sweep(sweep, "pretty") == json.dumps(sweep, indent=2) + "\n"
    assert (
        dump_sweep({"version": 1, "cases": []}, "compact")
        == json.dumps({"version": 1, "cases": []}, indent=2) + "\n"
    )
    assert sweep_format({"version": 1, "cases": []}) == "pretty"
    assert sweep_format({"import_mode": {"grouping": []}, "cases": []}) == "pretty"
    try:
        dump_sweep(sweep, "h4")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown format accepted")
    print("  PASS: sweep_format_compact_round_trip")


def test_grouped_append_keeps_sweep_format():
    """An append writes the layout recorded in the sweep, whatever --sweep-format says."""
    from bundle_utils import dump_sweep

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fmt, other in (("compact", "pretty"), ("pretty", "compact")):
            bundle_dir = tmp / f"bundles-{fmt}"
            bundle_dir.mkdir()
            create = ("--sweep-format", fmt) if fmt == "pretty" else ()
            r = _import(
                tmp,
                "g1",
                _grouped_graph(scale=0.5),
                bundle_dir,
                *_GROUPED_FLAGS,
                *create,
                "--strict",
            )
            assert r.returncode == 0 and "created new bundle" in r.stderr, r.stderr
            r = _import(
                tmp,
                "g2",
                _grouped_graph(causal=True, scale=0.5),
                bundle_dir,
                *_GROUPED_FLAGS,
                "--sweep-format",
                other,
                "--strict",
            )
            assert r.returncode == 0 and "appended" in r.stderr, r.stderr
            (rel,) = _sweep_dirs(bundle_dir)
            text = (bundle_dir / rel / "sweep.json").read_text()
            sweep = json.loads(text)
            assert sweep["import_mode"]["format"] == fmt, sweep["import_mode"]
            assert len(sweep["cases"]) == 2, sweep
            assert text == dump_sweep(sweep, fmt), f"append rewrote {fmt} as {other}"

        legacy = tmp / "legacy"
        legacy.mkdir()
        g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        marker = {"grouping": ["template-attributes"]}
        _write_marked_sweep(legacy, "quick/Relu/Old", g, marker)
        g2 = _make_graph("Relu", [0, 1], [3, 3, 4, 5], [60, 20, 5, 1], "float")
        r = _import(
            tmp,
            "g3",
            g2,
            legacy,
            "--template-attributes",
            "--require-marked-sweep",
            "--strict",
        )
        assert r.returncode == 0 and "appended" in r.stderr, r.stderr
        text = (legacy / "quick/Relu/Old/sweep.json").read_text()
        sweep = json.loads(text)
        assert sweep["import_mode"] == marker, sweep["import_mode"]
        assert (
            text == json.dumps(sweep, indent=2) + "\n"
        ), "grouping-only sweep left pretty"
        print("  PASS: grouped_append_keeps_sweep_format")


def test_sweep_format_key_does_not_split_sweeps():
    """--require-marked-sweep and retier compare import_mode without its format key."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g2 = _make_graph("Relu", [0, 1], [3, 3, 4, 5], [60, 20, 5, 1], "float")
        flags = ("--template-attributes", "--only-op-dir", "--require-marked-sweep")
        grouping = ["template-attributes"]
        for i, marker in enumerate(
            (
                {"grouping": grouping},
                {"grouping": grouping, "format": "pretty"},
                {"grouping": grouping, "format": "compact"},
            )
        ):
            bundle_dir = tmp / f"b{i}"
            bundle_dir.mkdir()
            _write_marked_sweep(bundle_dir, "quick/Relu/Mine", g, marker)
            for fmt in (None, "compact", "pretty"):
                extra = ("--sweep-format", fmt) if fmt else ()
                r = _import(tmp, "g2", g2, bundle_dir, *flags, *extra, "--dry-run")
                assert r.returncode == 0 and "appended" in r.stderr, (
                    marker,
                    fmt,
                    r.stderr,
                )
            assert _sweep_dirs(bundle_dir) == ["quick/Relu/Mine"]

        script = SCRIPT_DIR / "retier_bundles.py"
        root = tmp / "retier"
        template = _retier_graph_template(None)
        _retier_write(root, "quick", template, [_retier_case("a")], {})
        _retier_write(root, "full", template, [_retier_case("c")], {})
        for tier, marker in (
            ("quick", {"grouping": grouping, "format": "pretty"}),
            ("full", {"grouping": grouping}),
        ):
            path = root / tier / "Op" / "Default" / "sweep.json"
            cases = json.loads(path.read_text())["cases"]
            path.write_text(
                json.dumps({"version": 1, "import_mode": marker, "cases": cases})
            )
        asg = tmp / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="full"
                    ),
                    dict(current_tier="full", key="Op/Default/c", proposed_tier="full"),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode == 0, r.stderr
        merged = json.loads((root / "full/Op/Default/sweep.json").read_text())
        assert sorted(c["id"] for c in merged["cases"]) == ["a", "c"], merged
        print("  PASS: sweep_format_key_does_not_split_sweeps")


def test_retier_keeps_sweep_format():
    """retier_bundles writes a moved and a kept compact sweep in compact."""
    from bundle_utils import dump_sweep

    script = SCRIPT_DIR / "retier_bundles.py"
    marker = {"grouping": ["template-attributes"], "format": "compact"}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        template = _retier_graph_template(None)
        _retier_write(
            root, "quick", template, [_retier_case("a"), _retier_case("b")], {}
        )
        path = root / "quick" / "Op" / "Default" / "sweep.json"
        cases = json.loads(path.read_text())["cases"]
        path.write_text(
            dump_sweep({"version": 1, "import_mode": marker, "cases": cases}, "compact")
        )
        asg = Path(tmp) / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="quick"
                    ),
                    dict(
                        current_tier="quick",
                        key="Op/Default/b",
                        proposed_tier="standard",
                    ),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode == 0, r.stderr
        for tier, ids in (("quick", ["a"]), ("standard", ["b"])):
            text = (root / tier / "Op" / "Default" / "sweep.json").read_text()
            sweep = json.loads(text)
            assert sweep["import_mode"] == marker, sweep
            assert [c["id"] for c in sweep["cases"]] == ids, sweep
            assert (
                _is_compact(text) and text != json.dumps(sweep, indent=2) + "\n"
            ), tier
        print("  PASS: retier_keeps_sweep_format")


def test_retier_refuses_mixed_sweep_formats():
    """retier_bundles refuses to merge sweeps whose sweep.json formats differ."""
    from bundle_utils import dump_sweep

    script = SCRIPT_DIR / "retier_bundles.py"
    grouping = ["template-attributes"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        template = _retier_graph_template(None)
        _retier_write(root, "quick", template, [_retier_case("a")], {})
        _retier_write(root, "full", template, [_retier_case("c")], {})
        for tier, fmt in (("quick", "compact"), ("full", "pretty")):
            path = root / tier / "Op" / "Default" / "sweep.json"
            cases = json.loads(path.read_text())["cases"]
            marker = {"grouping": grouping, "format": fmt}
            sweep = {"version": 1, "import_mode": marker, "cases": cases}
            path.write_text(dump_sweep(sweep, fmt))
        before = _inventory(root)
        asg = Path(tmp) / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="full"
                    ),
                    dict(current_tier="full", key="Op/Default/c", proposed_tier="full"),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode != 0 and "sweep.json formats" in r.stderr, r.stderr
        assert _inventory(root) == before, "refused retier changed the tree"
        print("  PASS: retier_refuses_mixed_sweep_formats")


def test_retier_version_consistency():
    """build_plan checks the sweep-level ``version`` like it checks import_mode/format.

    Merging sweeps that disagree on ``version`` is refused, unless a merged
    case has ``absent_tensors`` (which needs sweep version 2): then the output
    is reconciled to version 2 regardless of what the source sweeps declared,
    instead of silently keeping whichever sweep happened to be listed first.
    """
    script = SCRIPT_DIR / "retier_bundles.py"
    template = _retier_graph_template(None)

    # No merged case needs absent_tensors: a version mismatch must be refused,
    # exactly like the import_mode/sweep-format checks.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        _retier_write(root, "quick", template, [_retier_case("a")], {})
        _retier_write(root, "full", template, [_retier_case("c")], {})
        path = root / "full" / "Op" / "Default" / "sweep.json"
        sweep = json.loads(path.read_text())
        sweep["version"] = 2
        path.write_text(json.dumps(sweep, indent=2))
        before = _inventory(root)
        asg = Path(tmp) / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="full"
                    ),
                    dict(current_tier="full", key="Op/Default/c", proposed_tier="full"),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode != 0 and "sweep versions" in r.stderr, r.stderr
        assert _inventory(root) == before, "refused retier changed the tree"

    # A merged case with absent_tensors forces version 2, even though the
    # sweep that would otherwise come first (by tier order) declares version 1.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        _retier_write(root, "quick", template, [_retier_case("a")], {})
        absent_case = _retier_case("c")
        absent_case["absent_tensors"] = [0]
        absent_case["metadata"]["format_version"] = 2
        _retier_write(root, "full", template, [absent_case], {})
        for tier, version in (("quick", 1), ("full", 2)):
            path = root / tier / "Op" / "Default" / "sweep.json"
            sweep = json.loads(path.read_text())
            sweep["version"] = version
            path.write_text(json.dumps(sweep, indent=2))
        asg = Path(tmp) / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="full"
                    ),
                    dict(current_tier="full", key="Op/Default/c", proposed_tier="full"),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode == 0, r.stderr
        merged = json.loads(
            (root / "full" / "Op" / "Default" / "sweep.json").read_text()
        )
        assert merged["version"] == 2, merged
        assert sorted(c["id"] for c in merged["cases"]) == ["a", "c"], merged

    # The reverse merge order: the case that stays in the destination tier
    # (and so sorts first) is the one from the lower-version sweep, and the
    # case needing absent_tensors arrives from a different, higher-version
    # sweep. The first-sorted sweep's own version must not simply be kept --
    # the merge still has to force version 2.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "integration-test-bundles"
        _retier_write(root, "full", template, [_retier_case("c")], {})
        absent_case = _retier_case("a")
        absent_case["absent_tensors"] = [0]
        absent_case["metadata"]["format_version"] = 2
        _retier_write(root, "quick", template, [absent_case], {})
        for tier, version in (("full", 1), ("quick", 2)):
            path = root / tier / "Op" / "Default" / "sweep.json"
            sweep = json.loads(path.read_text())
            sweep["version"] = version
            path.write_text(json.dumps(sweep, indent=2))
        asg = Path(tmp) / "asg.json"
        asg.write_text(
            json.dumps(
                [
                    dict(
                        current_tier="quick", key="Op/Default/a", proposed_tier="full"
                    ),
                    dict(current_tier="full", key="Op/Default/c", proposed_tier="full"),
                ]
            )
        )
        r = run(
            [
                sys.executable,
                script,
                "--bundle-dir",
                root,
                "--assignments",
                asg,
                "--apply",
            ],
            check=False,
        )
        assert r.returncode == 0, r.stderr
        merged = json.loads(
            (root / "full" / "Op" / "Default" / "sweep.json").read_text()
        )
        assert merged["version"] == 2, merged
        assert sorted(c["id"] for c in merged["cases"]) == ["a", "c"], merged
    print("  PASS: retier_version_consistency")


def test_default_mode_sweep_bytes_unchanged():
    """Default mode writes json.dumps(indent=2) with no import_mode, and refuses --sweep-format."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        for name, graph, word in (
            ("g", g, "created new bundle"),
            ("g2", g2, "appended"),
        ):
            r = _import(tmp, name, graph, bundle_dir)
            assert r.returncode == 0 and word in r.stderr, r.stderr
            (rel,) = _sweep_dirs(bundle_dir)
            raw = (bundle_dir / rel / "sweep.json").read_bytes()
            sweep = json.loads(raw)
            assert "import_mode" not in sweep, sweep
            assert raw == (json.dumps(sweep, indent=2) + "\n").encode(), raw
        before = _inventory(bundle_dir)
        r = _import(tmp, "g3", g, bundle_dir, "--sweep-format", "compact", check=False)
        assert r.returncode == 1 and "--sweep-format" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before
        print("  PASS: default_mode_sweep_bytes_unchanged")


# --------------------------------------------------------------------------
# Remove patches and superset templates with absent tensors
# --------------------------------------------------------------------------

_SUPERSET_FLAGS = _GROUPED_FLAGS + ("--superset-templates",)


def _optional_graph(sink=True, stats=True, bias=False, seq=4, virtual_stats=False):
    """A one-node graph with optional ports.

    Inputs q, k, v and, when asked, a sink and a bias; outputs o and, when asked,
    stats. An unused optional port holds null. The uids follow creation order,
    which differs from sorted-name order.
    """
    names = ["q", "k", "v", "o"]
    names += ["sink"] if sink else []
    names += ["bias"] if bias else []
    names += ["stats"] if stats else []
    uid = {name: i for i, name in enumerate(names)}
    node = {
        "type": "OptAttributes",
        "name": "opt",
        "attributes": {"causal": False},
        "inputs": {
            "q_tensor_uid": uid["q"],
            "k_tensor_uid": uid["k"],
            "v_tensor_uid": uid["v"],
            "sink_tensor_uid": uid.get("sink"),
            "bias_tensor_uid": uid.get("bias"),
        },
        "outputs": {"o_tensor_uid": uid["o"], "stats_tensor_uid": uid.get("stats")},
    }
    tensors = []
    for name in names:
        dims = [1, 2, seq, 1] if name in ("sink", "stats") else [1, 2, seq, 8]
        strides = [2 * seq * dims[3], seq * dims[3], dims[3], 1]
        tensors.append(
            {
                "uid": uid[name],
                "name": name,
                "dims": dims,
                "strides": strides,
                "data_type": "float",
                "virtual": virtual_stats and name == "stats",
                "alignment": 16,
                "is_runtime_pass_by_value": False,
            }
        )
    return {
        "nodes": [node],
        "tensors": tensors,
        "io_data_type": "float",
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }


def _optional_sidecar(graph, base_seed):
    """A sidecar with a per-input seed for each input tensor of the graph."""
    inputs = {
        str(t["uid"]): {"seed": base_seed + i}
        for i, t in enumerate(graph["tensors"])
        if t["name"] not in ("o", "stats")
    }
    return {"format_version": 1, "seed": base_seed, "inputs": inputs}


def _import_optional(tmp, bundle_dir, name, graph, sidecar, *flags):
    graph_path = _capture(tmp / "capture", name, graph, sidecar)
    return _import_capture(graph_path, bundle_dir, *flags)


def _sweep_and_template(bundle_dir, rel):
    sweep = _load_sweep(bundle_dir, rel)
    with open(bundle_dir / rel / "graph.template.json") as f:
        return sweep, json.load(f)


def _assert_absent_case_loads(sweep, template, case):
    """The case-side rules the C++ loader checks for a case with absent tensors."""
    from bundle_utils import absent_case_problems, absent_tensor_problems

    assert absent_case_problems(sweep, template, case) == [], case

    absent = case["absent_tensors"]
    assert sweep["version"] == 2, sweep["version"]
    assert case["metadata"]["format_version"] == 2, case["metadata"]
    assert not absent_tensor_problems(template, absent), absent
    assert not {e["uid"] for e in case["values"].get("tensors", [])} & set(absent)
    for field, value in case["values"].items():
        if field != "tensors" and isinstance(value, dict):
            assert not {str(u) for u in absent} & set(value), (field, value)
    assert not {int(u) for u in case["metadata"].get("inputs", {})} & set(absent)
    assert "golden" not in case and "tensor_patches" not in case, case


def test_import_remove_patch_round_trip():
    """A field the case's tensor lacks becomes a remove patch; a placeholder field does not."""
    from bundle_utils import canon, canonical_uid_map_by_name, expand_case, remap_graph

    sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))
    from support_matrix_units import apply_tensor_patches, expand_template_graph

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        g1 = _make_graph("Relu", [0, 1], [2, 3, 4, 5], [60, 20, 5, 1], "float")
        for t in g1["tensors"]:
            t["is_runtime_pass_by_value"] = False
        g2 = _make_graph("Relu", [0, 1], [4, 6, 8, 10], [480, 80, 10, 1], "float")
        g2["tensors"][0]["is_runtime_pass_by_value"] = False
        assert _import(tmp, "g1", g1, bundle_dir).returncode == 0
        r = _import(tmp, "g2", g2, bundle_dir)
        assert r.returncode == 0 and "appended" in r.stderr, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        sweep, template = _sweep_and_template(bundle_dir, rel)
        first, second = sweep["cases"]
        assert "tensor_patches" not in first, first
        uid_map = canonical_uid_map_by_name(g2)
        assert second["tensor_patches"] == [
            {"uid": uid_map[1], "remove": ["is_runtime_pass_by_value"]}
        ], second
        want = canon(remap_graph(g2, uid_map))
        assert canon(expand_case(template, second)) == want
        mirrored = expand_template_graph(template, second)
        apply_tensor_patches(mirrored, second)
        assert canon(mirrored) == want, "the loader mirror disagrees"
        r = _import(tmp, "g2again", g2, bundle_dir)
        assert r.returncode == 0 and "DUPLICATE" in r.stderr, r.stderr

        grouped_dir = tmp / "grouped"
        grouped_dir.mkdir()
        g3 = _grouped_graph(scale=0.5)
        g4 = _grouped_graph(scale=0.5, dims=[1, 2, 3, 4])
        del g4["tensors"][1]["alignment"]
        r = _import(tmp, "g3", g3, grouped_dir, *_GROUPED_FLAGS)
        assert r.returncode == 0, r.stderr
        before = _inventory(grouped_dir)
        r = _import(tmp, "g4", g4, grouped_dir, *_GROUPED_FLAGS, check=False)
        _assert_refused(r, grouped_dir, before, "after extraction")
        print("  PASS: import_remove_patch_round_trip")


def test_superset_import_nested_cases():
    """Cases whose tensor sets nest share one version 2 sweep and re-expand to their graphs."""
    from bundle_utils import (
        absent_by_name,
        canon,
        expand_case,
        inputs_by_name,
        remap_graph,
        uid_map_onto,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        graphs = [
            ("full", _optional_graph(), []),
            ("no_sink", _optional_graph(sink=False, seq=8), ["sink"]),
            ("no_stats", _optional_graph(stats=False, seq=16), ["stats"]),
        ]
        sidecars = {}
        for i, (name, graph, _) in enumerate(graphs):
            sidecars[name] = _optional_sidecar(graph, 100 * (i + 1))
            r = _import_optional(
                tmp,
                bundle_dir,
                name,
                graph,
                sidecars[name],
                *_SUPERSET_FLAGS,
                "--topology",
                "Opt",
                "--case-id",
                name,
                "--strict",
            )
            assert r.returncode == 0, r.stderr
            if i == 0:
                assert _load_sweep(bundle_dir, "quick/GroupedOp/Opt")["version"] == 1
        assert _sweep_dirs(bundle_dir) == ["quick/GroupedOp/Opt"]
        sweep, template = _sweep_and_template(bundle_dir, "quick/GroupedOp/Opt")
        assert sweep["version"] == 2, sweep
        cases = {c["id"]: c for c in sweep["cases"]}
        assert "absent_tensors" not in cases["full"], cases["full"]
        assert cases["full"]["metadata"]["format_version"] == 1
        for name, graph, absent_names in graphs:
            case = cases[name]
            assert absent_by_name(case, template) == absent_names, case
            if absent_names:
                _assert_absent_case_loads(sweep, template, case)
            want = remap_graph(graph, uid_map_onto(graph, template))
            assert canon(expand_case(template, case)) == canon(want), name
            assert inputs_by_name(case["metadata"], template) == inputs_by_name(
                sidecars[name], graph
            ), name

        before = _inventory(bundle_dir)
        r = _import_optional(
            tmp,
            bundle_dir,
            "no_sink_again",
            graphs[1][1],
            sidecars["no_sink"],
            *_SUPERSET_FLAGS,
        )
        assert r.returncode == 0 and "DUPLICATE" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before
        print("  PASS: superset_import_nested_cases")


def test_import_cache_matches_separate_runs():
    """Imports that share an ImportCache write the bytes that separate runs write."""
    import contextlib
    import io
    from unittest import mock

    import import_graph

    def flags(topology, case_id=None, strict=True):
        out = [*_SUPERSET_FLAGS, "--topology", topology]
        out += ["--case-id", case_id] if case_id is not None else []
        return out + (["--strict"] if strict else [])

    steps = [
        ("full", _optional_graph(), 100, flags("Opt", "full")),
        ("no_sink", _optional_graph(sink=False, seq=8), 200, flags("Opt", "no_sink")),
        ("no_stats", _optional_graph(stats=False, seq=16), 300, flags("Opt", None)),
        ("full_seq8", _optional_graph(seq=8), 400, flags("Opt", "full_seq8")),
        ("dup", _optional_graph(sink=False, seq=8), 200, flags("Opt", "dup")),
        (
            "dup_skip",
            _optional_graph(sink=False, seq=8),
            200,
            flags("Opt", strict=False),
        ),
        ("id_taken", _optional_graph(seq=32), 500, flags("Opt", "full")),
        ("bias", _optional_graph(bias=True), 600, flags("Bias", "bias")),
        ("bias_seq8", _optional_graph(bias=True, seq=8), 700, flags("Bias", None)),
        ("no_sink_32", _optional_graph(sink=False, seq=32), 800, flags("Opt", "ns32")),
    ]
    flush_after = {"no_stats", "dup_skip"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        separate, shared = tmp / "separate", tmp / "shared"
        separate.mkdir()
        shared.mkdir()
        cache = import_graph.ImportCache()
        rcs_separate, rcs_shared = [], []
        sweep_loads = []
        json_load = json.load

        def counting_load(f, *args, **kwargs):
            if Path(getattr(f, "name", "")).name == "sweep.json":
                sweep_loads.append(f.name)
            return json_load(f, *args, **kwargs)

        for name, graph, seed, step_flags in steps:
            graph_path = _capture(
                tmp / "capture", name, graph, _optional_sidecar(graph, seed)
            )
            rcs_separate.append(
                _import_capture(graph_path, separate, *step_flags).returncode
            )
            argv = ["--graph", str(graph_path), "--bundle-dir", str(shared)]
            with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
                import_graph.json, "load", counting_load
            ):
                rcs_shared.append(import_graph.main(argv + step_flags, cache))
            if name in flush_after:
                cache.flush()
                assert _inventory(shared) == _inventory(separate), name
        cache.flush()
        assert rcs_separate == [0, 0, 0, 0, 1, 0, 1, 0, 0, 0], rcs_separate
        assert rcs_shared == rcs_separate, rcs_shared
        assert _inventory(shared) == _inventory(separate)
        assert _sweep_dirs(shared) == ["quick/GroupedOp/Bias", "quick/GroupedOp/Opt"]
        assert len(_cases(shared)) == 7
        assert len(sweep_loads) == 2, sweep_loads
        assert all(e.get("memo") for e in cache._entries.values())
        print("  PASS: import_cache_matches_separate_runs")


def test_superset_import_needs_a_containing_template():
    """A graph joins only a template that holds every tensor of it, by the loader's rules."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def tree(name, first, second, flags=_SUPERSET_FLAGS):
            bundle_dir = tmp / name
            bundle_dir.mkdir()
            for i, graph in enumerate((first, second)):
                r = _import(tmp, f"{name}{i}", graph, bundle_dir, *flags)
                assert r.returncode == 0, r.stderr
            return bundle_dir, r.stderr

        bundle_dir, err = tree(
            "other", _optional_graph(), _optional_graph(stats=False, bias=True)
        )
        assert len(_sweep_dirs(bundle_dir)) == 2 and "missed merge" not in err, err
        for rel in _sweep_dirs(bundle_dir):
            sweep = _load_sweep(bundle_dir, rel)
            assert sweep["version"] == 1, sweep
            assert not any("absent_tensors" in c for c in sweep["cases"]), sweep

        bundle_dir, _ = tree(
            "virtual", _optional_graph(virtual_stats=True), _optional_graph(stats=False)
        )
        assert len(_sweep_dirs(bundle_dir)) == 2, "a virtual tensor was made absent"

        topo_dir = tmp / "topology"
        topo_dir.mkdir()
        for name, graph, topo in (
            ("full", _optional_graph(), "Full"),
            ("own", _optional_graph(sink=False), "Own"),
            ("joins", _optional_graph(stats=False, seq=8), "Full"),
            ("own_twin", _optional_graph(sink=False, seq=16), "Full"),
        ):
            r = _import(
                tmp, f"t_{name}", graph, topo_dir, *_SUPERSET_FLAGS, "--topology", topo
            )
            assert r.returncode == 0, r.stderr
            if name == "own":
                assert "NOTE: the graph also fits" in r.stderr, r.stderr
        assert _sweep_dirs(topo_dir) == ["quick/GroupedOp/Full", "quick/GroupedOp/Own"]
        assert [
            len(_load_sweep(topo_dir, rel)["cases"]) for rel in _sweep_dirs(topo_dir)
        ] == [3, 1]

        bundle_dir, _ = tree(
            "unmarked",
            _optional_graph(),
            _optional_graph(sink=False),
            flags=_GROUPED_FLAGS,
        )
        assert len(_sweep_dirs(bundle_dir)) == 2, "joined without --superset-templates"
        print("  PASS: superset_import_needs_a_containing_template")


def test_superset_import_order_of_smaller_cases():
    """After the largest case, any order of the smaller cases gives the same template and cases."""
    import itertools

    smaller = [
        ("no_sink", _optional_graph(sink=False, seq=8)),
        ("no_stats", _optional_graph(stats=False, seq=16)),
        ("neither", _optional_graph(sink=False, stats=False, seq=32)),
    ]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n, order in enumerate(itertools.permutations(smaller)):
            bundle_dir = tmp / f"bundles{n}"
            bundle_dir.mkdir()
            for name, graph in (("full", _optional_graph()),) + order:
                r = _import(
                    tmp,
                    f"{n}_{name}",
                    graph,
                    bundle_dir,
                    *_SUPERSET_FLAGS,
                    "--topology",
                    "Opt",
                    "--case-id",
                    name,
                    "--strict",
                )
                assert r.returncode == 0, r.stderr
            rel = "quick/GroupedOp/Opt"
            sweep = _load_sweep(bundle_dir, rel)
            template = (bundle_dir / rel / "graph.template.json").read_bytes()
            cases = {c["id"]: json.dumps(c, sort_keys=True) for c in sweep["cases"]}
            results.append((template, cases, sweep["version"]))
    assert len(results) == 6 and all(r == results[0] for r in results), "orders differ"
    print("  PASS: superset_import_order_of_smaller_cases")


def test_superset_import_reports_missed_merge():
    """A larger case after a smaller one starts a superset and leaves the smaller sweep unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        r = _import(
            tmp, "small", _optional_graph(sink=False), bundle_dir, *_SUPERSET_FLAGS
        )
        assert r.returncode == 0, r.stderr
        (small_rel,) = _sweep_dirs(bundle_dir)
        before = {
            k: v
            for k, v in _inventory(bundle_dir).items()
            if Path(k).as_posix().startswith(small_rel)
        }
        r = _import(tmp, "large", _optional_graph(), bundle_dir, *_SUPERSET_FLAGS)
        assert r.returncode == 0 and "created new bundle" in r.stderr, r.stderr
        assert "missed merge" in r.stderr, r.stderr
        assert str(bundle_dir / small_rel / "sweep.json") in r.stderr, r.stderr
        after = _inventory(bundle_dir)
        assert all(after[k] == v for k, v in before.items()), "smaller sweep changed"
        assert len(_sweep_dirs(bundle_dir)) == 2
        print("  PASS: superset_import_reports_missed_merge")


def test_absent_case_problems_mirror_the_loader():
    """absent_case_problems names each case rule of the C++ loader that a case breaks."""
    from bundle_utils import absent_case_problems

    template = {
        "nodes": [
            {
                "type": "OptAttributes",
                "inputs": {"q_tensor_uid": 0, "sink_tensor_uid": 1},
                "outputs": {"o_tensor_uid": 2},
            }
        ],
        "tensors": [
            {"uid": 0, "name": "q", "virtual": False},
            {"uid": 1, "name": "sink", "virtual": False},
            {"uid": 2, "name": "o", "virtual": False},
        ],
    }
    sweep = {"version": 2}
    good = {
        "id": "c",
        "absent_tensors": [1],
        "values": {"tensors": [{"uid": 0}, {"uid": 2}], "alignment": {"0": 16}},
        "metadata": {"format_version": 2, "inputs": {"0": {"seed": 1}}},
    }
    assert absent_case_problems(sweep, template, good) == []
    assert absent_case_problems(sweep, template, {"id": "full", "values": {}}) == []

    def broken(**changes):
        case = json.loads(json.dumps(good))
        for key, value in changes.items():
            case[key] = value
        return absent_case_problems(sweep, template, case)

    assert broken(absent_tensors=[]), "an empty list loads"
    assert broken(absent_tensors=[1, 1]), "a repeated uid loads"
    assert broken(absent_tensors=[7]), "a uid the template lacks loads"
    assert broken(absent_tensors=[0]), "a uid held by two ports loads"
    assert broken(values={"tensors": [{"uid": 1}]}), "a values.tensors entry loads"
    assert broken(values={"alignment": {"1": 16}}), "a values.alignment key loads"
    assert broken(tensor_patches=[{"uid": 1, "remove": ["x"]}])
    assert broken(metadata={"format_version": 2, "inputs": {"1": {"seed": 3}}})
    assert broken(metadata={"format_version": 2, "aliases": [{"uid": 0, "of": 1}]})
    assert broken(metadata={"format_version": 1}), "format_version 1 loads"
    assert broken(metadata=None), "no metadata loads"
    assert broken(golden={"path": "g"}), "golden data loads"
    assert absent_case_problems({"version": 1}, template, good), "sweep version 1 loads"
    print("  PASS: absent_case_problems_mirror_the_loader")


def test_import_refuses_metadata_uid_moves():
    """A metadata uid that a uid remap would attach to another tensor fails the import."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        from bundle_utils import remap_graph

        graph = _optional_graph(sink=False, stats=False)
        graph = remap_graph(graph, {t["uid"]: t["uid"] + 10 for t in graph["tensors"]})
        sidecar = {"format_version": 1, "seed": 1, "inputs": {"2": {"seed": 5}}}
        before = _inventory(bundle_dir)
        r = _import_optional(tmp, bundle_dir, "moved", graph, sidecar, *_GROUPED_FLAGS)
        assert r.returncode == 1 and "moves a metadata uid" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before

        full = _optional_graph()
        part = _optional_graph(stats=False, seq=8)
        for g in (full, part):
            for t in g["tensors"]:
                if t["name"] == "stats":
                    t["name"] = "z_stats"
        assert _import(tmp, "full", full, bundle_dir, *_SUPERSET_FLAGS).returncode == 0
        absent_uid = len(part["tensors"])
        sidecar = {"format_version": 2, "inputs": {str(absent_uid): {"seed": 5}}}
        before = _inventory(bundle_dir)
        r = _import_optional(tmp, bundle_dir, "part", part, sidecar, *_SUPERSET_FLAGS)
        assert r.returncode == 1 and "names absent tensors" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == before
        print("  PASS: import_refuses_metadata_uid_moves")


def test_superset_index_of_a_sweep_with_only_absent_cases():
    """A sweep whose every case has absent tensors still stands for its whole template."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        for name, graph in (
            ("full", _optional_graph()),
            ("no_sink", _optional_graph(sink=False, seq=8)),
        ):
            r = _import(
                tmp, name, graph, bundle_dir, *_SUPERSET_FLAGS, "--case-id", name
            )
            assert r.returncode == 0, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        sweep = _load_sweep(bundle_dir, rel)
        sweep["cases"] = [c for c in sweep["cases"] if c["id"] != "full"]
        (bundle_dir / rel / "sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
        for name, graph in (
            ("no_sink_2", _optional_graph(sink=False, seq=16)),
            ("full_2", _optional_graph(seq=16)),
        ):
            r = _import(
                tmp, name, graph, bundle_dir, *_SUPERSET_FLAGS, "--case-id", name
            )
            assert r.returncode == 0 and "appended" in r.stderr, r.stderr
        assert _sweep_dirs(bundle_dir) == [rel]
        cases = {c["id"]: c for c in _load_sweep(bundle_dir, rel)["cases"]}
        assert "absent_tensors" in cases["no_sink_2"], cases["no_sink_2"]
        assert "absent_tensors" not in cases["full_2"], cases["full_2"]
        print("  PASS: superset_index_of_a_sweep_with_only_absent_cases")


def test_superset_import_keeps_aliases_by_name():
    """A superset join keeps a case's aliases, inputs and seeds attached to the same tensors."""
    from bundle_utils import aliases_by_name, inputs_by_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        full = _optional_graph()
        assert _import(tmp, "full", full, bundle_dir, *_SUPERSET_FLAGS).returncode == 0
        part = _optional_graph(sink=False, seq=8)
        uid = {t["name"]: t["uid"] for t in part["tensors"]}
        sidecar = _optional_sidecar(part, 500)
        sidecar["format_version"] = 2
        sidecar["aliases"] = [{"uid": uid["v"], "of": uid["k"], "byte_offset": 0}]
        r = _import_optional(tmp, bundle_dir, "part", part, sidecar, *_SUPERSET_FLAGS)
        assert r.returncode == 0 and "appended" in r.stderr, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        sweep, template = _sweep_and_template(bundle_dir, rel)
        case = next(c for c in sweep["cases"] if c.get("absent_tensors"))
        _assert_absent_case_loads(sweep, template, case)
        assert aliases_by_name(case["metadata"], template) == aliases_by_name(
            sidecar, part
        ), case["metadata"]
        assert inputs_by_name(case["metadata"], template) == inputs_by_name(
            sidecar, part
        ), case["metadata"]
        print("  PASS: superset_import_keeps_aliases_by_name")


_RAGGED_LINKS = {
    "q": "ragged_offset_q",
    "o": "ragged_offset_q",
    "k": "ragged_offset_kv",
    "v": "ragged_offset_kv",
}


def _ragged_graph(sink=True, stats=True, seq=4):
    """:func:`_optional_graph` with ragged q, k, v and o.

    q and o link to the offset tensor ``ragged_offset_q``; k and v link to
    ``ragged_offset_kv``. The offset tensors take the next uids in creation
    order, so their sorted-name uids differ.
    """
    graph = _optional_graph(sink=sink, stats=stats, seq=seq)
    tensors = graph["tensors"]
    next_uid = 1 + max(t["uid"] for t in tensors)
    offsets = {}
    for i, name in enumerate(("ragged_offset_q", "ragged_offset_kv")):
        offsets[name] = next_uid + i
        tensors.append(
            {
                "uid": next_uid + i,
                "name": name,
                "dims": [2, 1, 1, 1],
                "strides": [1, 1, 1, 1],
                "data_type": "int32",
                "virtual": False,
                "alignment": 16,
                "is_runtime_pass_by_value": False,
            }
        )
    for t in tensors:
        if t["name"] in _RAGGED_LINKS:
            t["ragged_offset_tensor_uid"] = offsets[_RAGGED_LINKS[t["name"]]]
            t["ragged_offset_multiplier"] = 8
    return graph


def _ragged_links(graph):
    """{tensor name: name of the tensor its ragged_offset_tensor_uid holds}."""
    names = {t["uid"]: t["name"] for t in graph["tensors"]}
    return {
        t["name"]: names.get(t["ragged_offset_tensor_uid"])
        for t in graph["tensors"]
        if t.get("ragged_offset_tensor_uid") is not None
    }


def _ragged_sidecar(graph, base_seed, q_end, kv_end):
    """:func:`_optional_sidecar` with exact ``values`` fills for the offset tensors."""
    sidecar = _optional_sidecar(graph, base_seed)
    sidecar["format_version"] = 2
    uid = {t["name"]: t["uid"] for t in graph["tensors"]}
    for name, end in (("ragged_offset_q", q_end), ("ragged_offset_kv", kv_end)):
        sidecar["inputs"][str(uid[name])] = {
            "kind": "values",
            "values": [0, end],
            "repeat": False,
        }
    return sidecar


def test_remap_graph_remaps_tensor_links():
    """remap_graph, canonical_uid_map and skeleton_hash follow tensor-level uid links."""
    from bundle_utils import (
        absent_tensor_problems,
        canonical_uid_map,
        canonical_uid_map_by_name,
        remap_graph,
        skeleton_hash,
    )

    graph = _ragged_graph()
    assert _ragged_links(graph) == _RAGGED_LINKS
    by_name = canonical_uid_map_by_name(graph)
    offset_uids = [
        t["uid"] for t in graph["tensors"] if t["name"].startswith("ragged_offset")
    ]
    assert all(by_name[u] != u for u in offset_uids), by_name
    for make_map in (canonical_uid_map, canonical_uid_map_by_name):
        remapped = remap_graph(graph, make_map(graph))
        assert _ragged_links(remapped) == _RAGGED_LINKS, make_map.__name__
        assert skeleton_hash(remapped) == skeleton_hash(graph), make_map.__name__

    unlinked = copy.deepcopy(graph)
    for t in unlinked["tensors"]:
        t.pop("ragged_offset_tensor_uid", None)
    assert canonical_uid_map(graph) == canonical_uid_map(unlinked)

    swapped = copy.deepcopy(graph)
    q = next(t for t in swapped["tensors"] if t["name"] == "q")
    q["ragged_offset_tensor_uid"] = offset_uids[1]
    assert skeleton_hash(swapped) != skeleton_hash(graph)

    plain = _optional_graph()
    null_links = copy.deepcopy(plain)
    for t in null_links["tensors"]:
        t["ragged_offset_tensor_uid"] = None
    assert skeleton_hash(null_links) == skeleton_hash(plain)
    assert _ragged_links(remap_graph(null_links, canonical_uid_map(null_links))) == {}

    stale = copy.deepcopy(graph)
    q = next(t for t in stale["tensors"] if t["name"] == "q")
    q["ragged_offset_tensor_uid"] = 99
    assert 99 in canonical_uid_map(stale)
    try:
        remap_graph(stale, canonical_uid_map_by_name(stale))
    except ValueError as e:
        assert "99" in str(e), e
    else:
        raise AssertionError("a link to a uid without a map entry remaps")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        r = _import(tmp, "stale", stale, bundle_dir, check=False)
        assert r.returncode == 1 and "does not declare" in r.stderr, r.stderr
        assert _inventory(bundle_dir) == {}, _inventory(bundle_dir)

    problems = absent_tensor_problems(graph, [offset_uids[0]])
    assert any("ragged_offset_tensor_uid" in p for p in problems), problems
    print("  PASS: remap_graph_remaps_tensor_links")


def test_import_ragged_graph_keeps_offset_links():
    """An imported ragged graph keeps its offset links and offset values on renumbered uids."""
    from bundle_utils import (
        canon,
        canonical_uid_map_by_name,
        expand_case,
        inputs_by_name,
        remap_graph,
    )

    for mode, flags in (("default", ()), ("grouped", _GROUPED_FLAGS)):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bundle_dir = tmp / "bundles"
            bundle_dir.mkdir()
            graph = _ragged_graph()
            sidecar = _ragged_sidecar(graph, 300, 32, 24)
            r = _import_optional(tmp, bundle_dir, "ragged", graph, sidecar, *flags)
            assert r.returncode == 0, (mode, r.stderr)
            (rel,) = _sweep_dirs(bundle_dir)
            sweep, template = _sweep_and_template(bundle_dir, rel)
            (case,) = sweep["cases"]
            built = expand_case(template, case)
            assert _ragged_links(built) == _RAGGED_LINKS, (mode, built["tensors"])
            want = remap_graph(graph, canonical_uid_map_by_name(graph))
            assert canon(built) == canon(want), mode
            assert inputs_by_name(case["metadata"], template) == inputs_by_name(
                sidecar, graph
            ), (mode, case["metadata"])
    print("  PASS: import_ragged_graph_keeps_offset_links")


def test_superset_import_ragged_offset_values_per_case():
    """A ragged superset sweep expands each case to its offset links and offset values."""
    from bundle_utils import (
        canon,
        expand_case,
        inputs_by_name,
        remap_graph,
        uid_map_onto,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        graphs = [
            ("full", _ragged_graph(), 32, 24),
            ("no_sink", _ragged_graph(sink=False, seq=8), 64, 40),
            ("no_stats", _ragged_graph(stats=False, seq=16), 128, 96),
        ]
        sidecars = {}
        for i, (name, graph, q_end, kv_end) in enumerate(graphs):
            sidecars[name] = _ragged_sidecar(graph, 100 * (i + 1), q_end, kv_end)
            r = _import_optional(
                tmp,
                bundle_dir,
                name,
                graph,
                sidecars[name],
                *_SUPERSET_FLAGS,
                "--topology",
                "Ragged",
                "--case-id",
                name,
                "--strict",
            )
            assert r.returncode == 0, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        sweep, template = _sweep_and_template(bundle_dir, rel)
        assert sweep["version"] == 2, sweep
        cases = {c["id"]: c for c in sweep["cases"]}
        assert sorted(cases) == ["full", "no_sink", "no_stats"], sorted(cases)
        for name, graph, q_end, kv_end in graphs:
            case = cases[name]
            if case.get("absent_tensors"):
                _assert_absent_case_loads(sweep, template, case)
            built = expand_case(template, case)
            assert _ragged_links(built) == _RAGGED_LINKS, (name, built["tensors"])
            want = remap_graph(graph, uid_map_onto(graph, template))
            assert canon(built) == canon(want), name
            by_name = inputs_by_name(case["metadata"], template)
            assert by_name["ragged_offset_q"]["values"] == [0, q_end], (name, by_name)
            assert by_name["ragged_offset_kv"]["values"] == [0, kv_end], (name, by_name)
            assert by_name == inputs_by_name(sidecars[name], graph), name
        print("  PASS: superset_import_ragged_offset_values_per_case")


def test_superset_import_ragged_offset_multiplier_per_case():
    """Ragged graphs whose offset multipliers differ share one sweep and keep their own.

    The serialiser writes a multiplier of 1 on every tensor; only linked tensors take it per case.
    """
    from bundle_utils import canon, expand_case, remap_graph, uid_map_onto

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle_dir = tmp / "bundles"
        bundle_dir.mkdir()
        graphs = []
        for i, multiplier in enumerate((8, 24)):
            graph = _ragged_graph(seq=4 * (i + 1))
            for t in graph["tensors"]:
                linked = t.get("ragged_offset_tensor_uid") is not None
                t["ragged_offset_multiplier"] = multiplier if linked else 1
            graphs.append((f"m{multiplier}", graph, multiplier))
        for i, (name, graph, _) in enumerate(graphs):
            r = _import_optional(
                tmp,
                bundle_dir,
                name,
                graph,
                _ragged_sidecar(graph, 100 * (i + 1), 32, 24),
                *_SUPERSET_FLAGS,
                "--topology",
                "Ragged",
                "--case-id",
                name,
                "--strict",
            )
            assert r.returncode == 0, r.stderr
        (rel,) = _sweep_dirs(bundle_dir)
        sweep, template = _sweep_and_template(bundle_dir, rel)
        assert all(
            t["ragged_offset_multiplier"] == "${case.ragged_offset_multiplier}"
            for t in template["tensors"]
            if t.get("ragged_offset_tensor_uid") is not None
        ), template["tensors"]
        cases = {c["id"]: c for c in sweep["cases"]}
        for name, graph, multiplier in graphs:
            built = expand_case(template, cases[name])
            assert canon(built) == canon(
                remap_graph(graph, uid_map_onto(graph, template))
            ), name
            assert {
                t["ragged_offset_multiplier"]
                for t in built["tensors"]
                if t.get("ragged_offset_tensor_uid") is not None
            } == {multiplier}, name
        plain = [
            t
            for t in template["tensors"]
            if t.get("ragged_offset_tensor_uid") is None
            and "ragged_offset_multiplier" in t
        ]
        assert plain and all(t["ragged_offset_multiplier"] == 1 for t in plain), plain
        assert all(
            "ragged_offset_multiplier" not in v
            for c in sweep["cases"]
            for v in c["values"]["tensors"]
            if v["uid"] in {t["uid"] for t in plain}
        ), sweep["cases"]
        print("  PASS: superset_import_ragged_offset_multiplier_per_case")


def main() -> int:
    verbose = "-v" in sys.argv
    failures = 0

    tests = [
        test_skeleton_grouping,
        test_inventory_keys_are_forward_slash_normalized,
        test_round_trip_expansion,
        test_resolve_plain_tensor_scoped_field_from_its_own_entry,
        test_case_ids,
        test_place_and_verify,
        test_place_keeps_manual_cases,
        test_import_dedup,
        test_import_append_preserves_case_fields,
        test_import_new_topology_never_clobbers,
        test_import_new_topology_case_id_is_descriptive,
        test_import_support_json_sidecar,
        test_import_generator_metadata_default,
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
        test_import_sidecar_outside_capture_layout,
        test_import_meta_override_checks,
        test_import_meta_value_types,
        test_import_malformed_sidecar_refused,
        test_import_seed_only_difference_distinct_cases,
        test_diff_coverage_suite_qualified_join,
        test_retier_bundles,
        test_grouped_default_mode_ignores_marked_sweep,
        test_grouped_creates_marked_sweep,
        test_grouped_append_attributes_round_trip,
        test_grouped_value_forms_never_share_sweep,
        test_grouped_only_op_dir_blocks_cross_appends,
        test_grouped_require_marked_sweep,
        test_grouped_case_id,
        test_grouped_topology,
        test_grouped_import_keeps_values_inputs,
        test_grouped_import_keeps_per_input_seeds,
        test_grouped_import_keeps_aliases_start_offsets_and_live_rows,
        test_grouped_runs_are_deterministic,
        test_grouped_attribute_keys_prefixed_per_node,
        test_grouped_engine_api_version_per_case,
        test_grouped_override_shape_flag_per_case,
        test_retier_keeps_import_mode,
        test_sweep_format_compact_round_trip,
        test_grouped_append_keeps_sweep_format,
        test_sweep_format_key_does_not_split_sweeps,
        test_retier_keeps_sweep_format,
        test_retier_refuses_mixed_sweep_formats,
        test_retier_version_consistency,
        test_default_mode_sweep_bytes_unchanged,
        test_import_remove_patch_round_trip,
        test_superset_import_nested_cases,
        test_superset_import_needs_a_containing_template,
        test_superset_import_order_of_smaller_cases,
        test_superset_import_reports_missed_merge,
        test_absent_case_problems_mirror_the_loader,
        test_import_refuses_metadata_uid_moves,
        test_superset_index_of_a_sweep_with_only_absent_cases,
        test_superset_import_keeps_aliases_by_name,
        test_import_cache_matches_separate_runs,
        test_remap_graph_remaps_tensor_links,
        test_import_ragged_graph_keeps_offset_links,
        test_superset_import_ragged_offset_values_per_case,
        test_superset_import_ragged_offset_multiplier_per_case,
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
