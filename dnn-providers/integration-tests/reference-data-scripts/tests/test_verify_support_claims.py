#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for verify_support_claims.py (RFC 0015 §9.3 support-claim
pre-commit validation).

Run directly:

    python3 test_verify_support_claims.py
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from verify_support_claims import (  # noqa: E402
    find_graph_files,
    find_sweep_roots,
    validate_directory,
    validate_single_graph_bundle,
    validate_sweep_bundle,
)


def _write(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class SingleGraphBundleTests(unittest.TestCase):
    def test_clean_bundle_with_valid_claim_and_enforcement_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(
                root / "Foo.support.json",
                {
                    "version": 1,
                    "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux", "windows"]}},
                },
            )
            _write(root / "Foo.meta.json", {"enforcement_level": "full"})

            self.assertEqual(validate_single_graph_bundle(graph), [])
            self.assertEqual(validate_directory(root), [])

    def test_claim_bearing_bundle_missing_enforcement_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(
                root / "Foo.support.json",
                {"version": 1, "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux"]}}},
            )
            _write(root / "Foo.meta.json", {"some_other_field": True})

            errors = validate_single_graph_bundle(graph)
            self.assertEqual(len(errors), 1)
            self.assertIn("Foo.meta.json", errors[0])
            self.assertIn("enforcement_level", errors[0])

    def test_claim_bearing_bundle_invalid_enforcement_level_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(
                root / "Foo.support.json",
                {"version": 1, "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux"]}}},
            )
            _write(root / "Foo.meta.json", {"enforcement_level": "verification"})

            errors = validate_single_graph_bundle(graph)
            self.assertEqual(len(errors), 1)
            self.assertIn("Foo.meta.json", errors[0])
            self.assertIn("verification", errors[0])

    def test_non_claim_bearing_bundle_with_no_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            # No Foo.support.json, no Foo.meta.json at all.

            self.assertEqual(validate_single_graph_bundle(graph), [])

    def test_support_json_bad_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(
                root / "Foo.support.json",
                {"version": 2, "claims": {}},
            )
            _write(root / "Foo.meta.json", {"enforcement_level": "full"})

            errors = validate_single_graph_bundle(graph)
            self.assertEqual(len(errors), 1)
            self.assertIn("Foo.support.json", errors[0])
            self.assertIn("version", errors[0])

    def test_support_json_bad_platform_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(
                root / "Foo.support.json",
                {
                    "version": 1,
                    "claims": {"MIOPEN_ENGINE": {"gfx942": ["linux", "macos"]}},
                },
            )
            _write(root / "Foo.meta.json", {"enforcement_level": "full"})

            errors = validate_single_graph_bundle(graph)
            self.assertEqual(len(errors), 1)
            self.assertIn("macos", errors[0])

    def test_find_graph_files_excludes_companions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            graph = root / "Foo.json"
            _write(graph, {"nodes": []})
            _write(root / "Foo.meta.json", {"enforcement_level": "full"})
            _write(root / "Foo.support.json", {"version": 1, "claims": {}})

            found = find_graph_files(root)
            self.assertEqual(found, [graph])


class SweepBundleTests(unittest.TestCase):
    def _make_sweep(self, root: pathlib.Path, cases: list) -> pathlib.Path:
        sweep_dir = root / "TopologyName"
        _write(sweep_dir / "graph.template.json", {"nodes": []})
        _write(sweep_dir / "sweep.json", {"version": 1, "cases": cases})
        return sweep_dir

    def test_clean_sweep_bundle_with_valid_grouped_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cases = [
                {"id": "small_fp16_nchw", "metadata": {"enforcement_level": "full"}},
                {
                    "id": "small_fp32_nchw",
                    "metadata": {"enforcement_level": "buildable"},
                },
                {"id": "big_fp8_nhwc"},
            ]
            sweep_dir = self._make_sweep(root, cases)
            _write(
                sweep_dir / "support.json",
                {
                    "version": 1,
                    "claims": {
                        "MIOPEN_ENGINE": [
                            {
                                "cases": ["small_fp16_nchw", "small_fp32_nchw"],
                                "support": {"gfx942": ["linux", "windows"]},
                            }
                        ]
                    },
                },
            )

            self.assertEqual(validate_sweep_bundle(sweep_dir), [])
            self.assertEqual(validate_directory(root), [])
            self.assertEqual(find_sweep_roots(root), [sweep_dir])

    def test_orphaned_claim_case_id_not_in_sweep_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cases = [
                {"id": "small_fp16_nchw", "metadata": {"enforcement_level": "full"}}
            ]
            sweep_dir = self._make_sweep(root, cases)
            _write(
                sweep_dir / "support.json",
                {
                    "version": 1,
                    "claims": {
                        "MIOPEN_ENGINE": [
                            {
                                "cases": ["small_fp16_nchw", "does_not_exist"],
                                "support": {"gfx942": ["linux"]},
                            }
                        ]
                    },
                },
            )

            errors = validate_sweep_bundle(sweep_dir)
            self.assertEqual(len(errors), 1)
            self.assertIn("orphaned", errors[0])
            self.assertIn("does_not_exist", errors[0])

    def test_ambiguous_claim_same_case_id_twice_for_one_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cases = [
                {"id": "small_fp16_nchw", "metadata": {"enforcement_level": "full"}},
                {"id": "small_fp32_nchw", "metadata": {"enforcement_level": "full"}},
            ]
            sweep_dir = self._make_sweep(root, cases)
            _write(
                sweep_dir / "support.json",
                {
                    "version": 1,
                    "claims": {
                        "MIOPEN_ENGINE": [
                            {
                                "cases": ["small_fp16_nchw"],
                                "support": {"gfx942": ["linux"]},
                            },
                            {
                                "cases": ["small_fp16_nchw", "small_fp32_nchw"],
                                "support": {"gfx90a": ["linux"]},
                            },
                        ]
                    },
                },
            )

            errors = validate_sweep_bundle(sweep_dir)
            self.assertEqual(len(errors), 1)
            self.assertIn("ambiguous", errors[0])
            self.assertIn("small_fp16_nchw", errors[0])

    def test_claimed_case_missing_enforcement_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cases = [{"id": "small_fp16_nchw"}]  # no metadata at all
            sweep_dir = self._make_sweep(root, cases)
            _write(
                sweep_dir / "support.json",
                {
                    "version": 1,
                    "claims": {
                        "MIOPEN_ENGINE": [
                            {
                                "cases": ["small_fp16_nchw"],
                                "support": {"gfx942": ["linux"]},
                            }
                        ]
                    },
                },
            )

            errors = validate_sweep_bundle(sweep_dir)
            self.assertEqual(len(errors), 1)
            self.assertIn("small_fp16_nchw", errors[0])
            self.assertIn("enforcement_level", errors[0])

    def test_unclaimed_case_needs_no_enforcement_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cases = [
                {"id": "claimed_case", "metadata": {"enforcement_level": "full"}},
                {"id": "unclaimed_case"},  # not covered by any group, no metadata
            ]
            sweep_dir = self._make_sweep(root, cases)
            _write(
                sweep_dir / "support.json",
                {
                    "version": 1,
                    "claims": {
                        "MIOPEN_ENGINE": [
                            {
                                "cases": ["claimed_case"],
                                "support": {"gfx942": ["linux"]},
                            }
                        ]
                    },
                },
            )

            self.assertEqual(validate_sweep_bundle(sweep_dir), [])


if __name__ == "__main__":
    unittest.main()
