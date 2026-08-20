"""Tests for verify_support_claims.py."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_support_claims import verify_all


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Schema — single graph
# ---------------------------------------------------------------------------


class TestSingleGraphSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_valid_sidecar_passes(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.meta.json",
            {"enforcement_level": "full"},
        )
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_bad_version_fails(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {"version": 2, "claims": {}},
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("version must be 1" in e for e in errors))

    def test_invalid_platform_fails(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["macos"]}},
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("invalid platform" in e for e in errors))

    def test_empty_claims_passes(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {"version": 1, "claims": {}},
        )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_null_claims_fails(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {"version": 1, "claims": None},
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("got null" in e for e in errors))


# ---------------------------------------------------------------------------
# Schema — sweep
# ---------------------------------------------------------------------------


class TestSweepSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_valid_sweep_passes(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_duplicate_case_id_fails(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {
                "version": 1,
                "cases": [
                    {"id": "c1", "values": {}},
                    {"id": "c2", "values": {}},
                ],
            },
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1"],
                            "support": {"gfx942": ["linux"]},
                        },
                        {
                            "cases": ["c1"],
                            "support": {"gfx90a": ["linux"]},
                        },
                    ]
                },
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("duplicate case id" in e for e in errors))

    def test_missing_cases_array_fails(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {"ENGINE": [{"support": {"gfx942": ["linux"]}}]},
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("non-empty 'cases' array" in e for e in errors))

    def test_missing_support_object_fails(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {"ENGINE": [{"cases": ["c1"]}]},
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("missing 'support' object" in e for e in errors))

    def test_null_claims_fails(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {"version": 1, "claims": None},
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("got null" in e for e in errors))


# ---------------------------------------------------------------------------
# enforcement_level
# ---------------------------------------------------------------------------


class TestEnforcementLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_missing_meta_json_passes(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_missing_enforcement_level_in_meta_passes(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(self.bundle_root / "A" / "Small.meta.json", {"seed": 42})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_invalid_enforcement_level_fails(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.meta.json",
            {"enforcement_level": "ultra"},
        )
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {
                "version": 1,
                "claims": {"ENGINE": {"gfx942": ["linux"]}},
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("invalid enforcement_level" in e for e in errors))

    def test_valid_enforcement_level_passes(self) -> None:
        for level in ("applicability", "buildable", "full"):
            d = self.bundle_root / level
            _write_json(d / "B.json", {})
            _write_json(d / "B.meta.json", {"enforcement_level": level})
            _write_json(
                d / "B.support.json",
                {
                    "version": 1,
                    "claims": {"ENGINE": {"gfx942": ["linux"]}},
                },
            )
        self.assertEqual(verify_all(self.bundle_root), [])

    def test_empty_claims_skips_enforcement_check(self) -> None:
        _write_json(self.bundle_root / "A" / "Small.json", {})
        _write_json(
            self.bundle_root / "A" / "Small.support.json",
            {"version": 1, "claims": {}},
        )
        self.assertEqual(verify_all(self.bundle_root), [])


# ---------------------------------------------------------------------------
# Sweep case id cross-check
# ---------------------------------------------------------------------------


class TestSweepCaseIds(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_orphan_case_id_fails(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1", "c_nonexistent"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("c_nonexistent" in e and "not found" in e for e in errors))

    def test_valid_case_ids_pass(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {
                "version": 1,
                "cases": [
                    {"id": "c1", "values": {}},
                    {"id": "c2", "values": {}},
                ],
            },
        )
        _write_json(
            sweep_dir / "support.json",
            {
                "version": 1,
                "claims": {
                    "ENGINE": [
                        {
                            "cases": ["c1", "c2"],
                            "support": {"gfx942": ["linux"]},
                        }
                    ]
                },
            },
        )
        self.assertEqual(verify_all(self.bundle_root), [])


# ---------------------------------------------------------------------------
# Orphaned sidecars
# ---------------------------------------------------------------------------


class TestOrphanedSidecars(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_orphaned_single_graph_sidecar_fails(self) -> None:
        _write_json(
            self.bundle_root / "A" / "Missing.support.json",
            {"version": 1, "claims": {}},
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("orphaned sidecar" in e for e in errors))

    def test_orphaned_sweep_support_json_fails(self) -> None:
        _write_json(
            self.bundle_root / "B" / "NotASweep" / "support.json",
            {"version": 1, "claims": {}},
        )
        errors = verify_all(self.bundle_root)
        self.assertTrue(any("not a sweep root" in e for e in errors))

    def test_sweep_support_json_with_sweep_json_passes(self) -> None:
        sweep_dir = self.bundle_root / "B" / "Default"
        _write_json(
            sweep_dir / "sweep.json",
            {"version": 1, "cases": [{"id": "c1", "values": {}}]},
        )
        _write_json(
            sweep_dir / "support.json",
            {"version": 1, "claims": {}},
        )
        self.assertEqual(verify_all(self.bundle_root), [])


# ---------------------------------------------------------------------------
# No sidecars at all
# ---------------------------------------------------------------------------


class TestNoSidecars(unittest.TestCase):
    def test_empty_tree_passes(self) -> None:
        bundle_root = Path(tempfile.mkdtemp())
        try:
            self.assertEqual(verify_all(bundle_root), [])
        finally:
            shutil.rmtree(bundle_root, ignore_errors=True)

    def test_nonexistent_root_passes(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        shutil.rmtree(tmp)
        self.assertEqual(verify_all(tmp / "does_not_exist"), [])


if __name__ == "__main__":
    unittest.main()
