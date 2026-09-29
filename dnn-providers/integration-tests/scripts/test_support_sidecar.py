"""Tests for support_sidecar.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support_sidecar import Claim, SidecarError, load, parse, render, sidecar_path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent / "integration-test-bundles"


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class TestSidecarPath(unittest.TestCase):
    def test_sweep(self):
        self.assertEqual(
            sidecar_path("integration-test-bundles/quick/Conv/sweep.json"),
            PurePosixPath("integration-test-bundles/quick/Conv/support.json"),
        )

    def test_single_graph(self):
        self.assertEqual(
            sidecar_path("integration-test-bundles/quick/Bn/Small.json"),
            PurePosixPath("integration-test-bundles/quick/Bn/Small.support.json"),
        )

    def test_refuses_paths_outside_the_bundle_tree(self):
        for bundle in (
            "quick/Bn/Small.json",
            "/integration-test-bundles/quick/Small.json",
            "other/integration-test-bundles/Small.json",
            "integration-test-bundles/Small.json",
            "integration-test-bundles/../scripts/x.json",
            "integration-test-bundles/quick/./Small.json",
            "integration-test-bundles/quick//Small.json",
            "integration-test-bundles/quick/Small.bin",
        ):
            with self.subTest(bundle=bundle), self.assertRaises(SidecarError):
                sidecar_path(bundle)


class TestRoundTrip(unittest.TestCase):
    def test_every_checked_in_sidecar_renders_back_byte_for_byte(self):
        paths = sorted(BUNDLE_ROOT.rglob("*support.json"))
        self.assertTrue(paths, f"no sidecars under {BUNDLE_ROOT}")
        for path in paths:
            with self.subTest(path=str(path.relative_to(BUNDLE_ROOT))):
                self.assertEqual(render(path, load(path)), path.read_text("utf-8"))


class TestFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _write(self, name: str, data: object) -> Path:
        path = self.dir / name
        path.write_bytes(_dump(data).encode("utf-8"))
        return path

    def test_missing_sidecar_has_no_claims(self):
        self.assertEqual(load(self.dir / "support.json"), set())

    def test_single_graph_load(self):
        path = self._write(
            "X.support.json",
            {"claims": {"E": {"gfx942": ["linux", "windows"]}}, "version": 1},
        )
        self.assertEqual(
            load(path),
            {Claim("", "E", "gfx942", "linux"), Claim("", "E", "gfx942", "windows")},
        )

    def test_sweep_groups_cases_with_equal_support(self):
        claims = {
            Claim("b", "E", "gfx942", "linux"),
            Claim("a", "E", "gfx942", "linux"),
            Claim("c", "E", "gfx90a", "linux"),
            Claim("d", "E", "gfx942", "linux"),
        }
        self.assertEqual(
            json.loads(render(self.dir / "support.json", claims)),
            {
                "claims": {
                    "E": [
                        {"cases": ["a", "b", "d"], "support": {"gfx942": ["linux"]}},
                        {"cases": ["c"], "support": {"gfx90a": ["linux"]}},
                    ]
                },
                "version": 1,
            },
        )

    def test_adding_claims_keeps_every_old_one(self):
        path = self._write(
            "support.json",
            {
                "claims": {
                    "E": [
                        {"cases": ["a"], "support": {"gfx942": ["linux"]}},
                        {"cases": ["b"], "support": {"gfx90a": ["windows"]}},
                    ]
                },
                "version": 1,
            },
        )
        old = load(path)
        new = {Claim("a", "E", "gfx90a", "windows"), Claim("c", "F", "gfx90a", "linux")}
        path.write_bytes(render(path, old | new).encode("utf-8"))
        self.assertEqual(load(path), old | new)
        self.assertLess(old, load(path))

    def test_claim_of_the_wrong_shape_is_refused(self):
        with self.assertRaises(SidecarError):
            render(self.dir / "support.json", {Claim("", "E", "gfx942", "linux")})
        with self.assertRaises(SidecarError):
            render(self.dir / "X.support.json", {Claim("a", "E", "gfx942", "linux")})

    def test_refuses_what_a_rewrite_would_lose(self):
        # Each file is valid JSON, but rendering its claims gives other text.
        cases = {
            "extra key": {"claims": {}, "note": "x", "version": 1},
            "empty platforms": {"claims": {"E": {"gfx942": []}}, "version": 1},
            "version 2": {"claims": {}, "version": 2},
            "claims not an object": {"claims": [], "version": 1},
        }
        for name, data in cases.items():
            with self.subTest(name):
                path = self._write("X.support.json", data)
                with self.assertRaises(SidecarError):
                    load(path)

    def test_refuses_a_value_that_is_not_a_string(self):
        cases = {
            "platform": ("X.support.json", {"E": {"gfx942": [1]}}),
            "platforms of mixed types": ("X.support.json", {"E": {"gfx942": [1, "a"]}}),
            "case": (
                "support.json",
                {"E": [{"cases": [1], "support": {"gfx942": ["linux"]}}]},
            ),
        }
        for name, (file, claims) in cases.items():
            with self.subTest(name):
                path = self._write(file, {"claims": claims, "version": 1})
                with self.assertRaisesRegex(SidecarError, "not a string"):
                    load(path)

    def test_refuses_a_non_canonical_file(self):
        path = self.dir / "X.support.json"
        path.write_bytes(b'{"version": 1, "claims": {}}\n')
        with self.assertRaises(SidecarError):
            load(path)

    def test_refuses_a_sweep_group_without_support(self):
        path = self._write("support.json", {"claims": {"E": [{"cases": ["a"]}]}})
        with self.assertRaises(SidecarError):
            load(path)

    def test_parse_reads_text_of_a_sidecar_path(self):
        text = _dump({"claims": {"E": {"gfx942": ["linux"]}}, "version": 1})
        self.assertEqual(
            parse(self.dir / "X.support.json", text),
            {Claim("", "E", "gfx942", "linux")},
        )
        self.assertFalse((self.dir / "X.support.json").exists())
        with self.assertRaises(SidecarError):
            parse(self.dir / "support.json", text)

    def test_refuses_invalid_utf8(self):
        path = self.dir / "X.support.json"
        path.write_bytes(b"\xff")
        with self.assertRaisesRegex(SidecarError, "unreadable"):
            load(path)

    def test_refuses_a_sidecar_it_cannot_read(self):
        path = self.dir / "X.support.json"
        path.mkdir()
        with self.assertRaisesRegex(SidecarError, "X.support.json: unreadable"):
            load(path)

    def test_refuses_invalid_json(self):
        path = self.dir / "X.support.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(SidecarError):
            load(path)


if __name__ == "__main__":
    unittest.main()
