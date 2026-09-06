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
    path.write_bytes(
        (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )


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
        self.assertEqual(len(errors), 1)
        self.assertIn("version must be 1", errors[0])

    def test_non_utf8_sidecar_reports_an_error(self) -> None:
        # A hook that dies with a traceback tells the committer nothing about
        # which file is bad.
        _write_json(self.bundle_root / "A" / "Small.json", {})
        bad = self.bundle_root / "A" / "Small.support.json"
        bad.write_bytes(b'{"version": 1, "claims": {"\xff\xfe": {}}}')
        errors = verify_all(self.bundle_root)
        self.assertEqual(len(errors), 1)
        self.assertIn("not valid UTF-8", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid platform", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("got null", errors[0])


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
        self.assertEqual(len(errors), 1)
        self.assertIn("duplicate case id", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("non-empty 'cases' array", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("missing 'support' object", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("got null", errors[0])


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
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid enforcement_level", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("c_nonexistent", errors[0])
        self.assertIn("not found", errors[0])

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
        self.assertEqual(len(errors), 1)
        self.assertIn("orphaned sidecar", errors[0])

    def test_orphaned_sweep_support_json_fails(self) -> None:
        _write_json(
            self.bundle_root / "B" / "NotASweep" / "support.json",
            {"version": 1, "claims": {}},
        )
        errors = verify_all(self.bundle_root)
        self.assertEqual(len(errors), 1)
        self.assertIn("not a sweep root", errors[0])

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
# Canonical form
# ---------------------------------------------------------------------------


class TestCanonicalForm(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_root = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.bundle_root, ignore_errors=True)

    def test_non_canonical_bytes_detected(self) -> None:
        (self.bundle_root / "A").mkdir(parents=True)
        (self.bundle_root / "A" / "Small.json").write_bytes(b"{}")
        non_canonical = json.dumps({"version": 1, "claims": {}}, indent=2)
        (self.bundle_root / "A" / "Small.support.json").write_bytes(
            non_canonical.encode("utf-8")
        )
        errors = verify_all(self.bundle_root)
        self.assertEqual(len(errors), 1)
        self.assertIn("not in canonical form", errors[0])


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
