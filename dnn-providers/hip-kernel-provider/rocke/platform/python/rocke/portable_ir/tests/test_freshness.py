# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Freshness checks for the local portable-IR shared-engine cache."""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocke.portable_ir.src import online


class _BuildIdFunction:
    def __init__(self, value: str):
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value.encode()


class _Library:
    def __init__(self, build_id: str):
        self.rocke_build_id = _BuildIdFunction(build_id)


class TestSourceBuildId(unittest.TestCase):
    def test_matches_the_cmake_path_and_content_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = {
                "cpp/a.cpp": b"a\n",
                "cpp/sub/b.cpp": b"b\n",
                "cpp/include/rocke/a.h": b"h\n",
                "cpp/include/rocke/b.hpp": b"hpp\n",
                "cpp/core/private.h": b"private\n",
                "tests/portable_ir/replay_cli.cpp": b"cli\n",
                "CMakeLists.txt": b"root recipe\n",
                "tests/CMakeLists.txt": b"CLI recipe\n",
                "cmake/rocke_build_id.cmake": b"hash recipe\n",
                "cmake/generate_rocke_build_id.cmake": b"header recipe\n",
            }
            for relative, content in inputs.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            ignored = root / "cpp" / "bindings" / "rocke_engine.cpp"
            ignored.parent.mkdir(parents=True)
            ignored.write_text("not linked into the bundle\n")
            lines = [
                f"{relative}:{hashlib.sha256(content).hexdigest()}"
                for relative, content in inputs.items()
            ]
            expected = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[
                :16
            ]
            self.assertEqual(online.source_build_id(root), expected)

    def test_replay_cli_source_changes_the_bundle_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "cpp" / "engine.cpp"
            engine.parent.mkdir(parents=True)
            engine.write_text("engine\n")
            replay_cli = root / "tests" / "portable_ir" / "replay_cli.cpp"
            replay_cli.parent.mkdir(parents=True)
            replay_cli.write_text("first\n")
            first = online.source_build_id(root)
            replay_cli.write_text("second\n")
            self.assertNotEqual(first, online.source_build_id(root))

    def test_build_recipe_changes_the_bundle_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = root / "cpp" / "engine.cpp"
            engine.parent.mkdir(parents=True)
            engine.write_text("engine\n")
            recipe = root / "CMakeLists.txt"
            recipe.write_text("first\n")
            first = online.source_build_id(root)
            recipe.write_text("second\n")
            self.assertNotEqual(first, online.source_build_id(root))

    def test_python_id_matches_the_cmake_generator(self):
        source_cmake = Path(online._platform_root()) / "cmake"
        script_names = ("rocke_build_id.cmake", "generate_rocke_build_id.cmake")
        if not shutil.which("cmake") or not all(
            (source_cmake / name).is_file() for name in script_names
        ):
            self.skipTest("CMake and rocKE build-ID generator scripts required")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            private = root / "cpp" / "core" / "private.h"
            private.parent.mkdir(parents=True)
            private.write_text("private header\n")
            replay_cli = root / "tests" / "portable_ir" / "replay_cli.cpp"
            replay_cli.parent.mkdir(parents=True)
            replay_cli.write_text("replay CLI\n")
            (root / "CMakeLists.txt").write_text("root recipe\n")
            (root / "tests" / "CMakeLists.txt").write_text("CLI recipe\n")
            binding = root / "cpp" / "bindings" / "ignored.cpp"
            binding.parent.mkdir(parents=True)
            binding.write_text("not a bundle input\n")
            cmake_dir = root / "cmake"
            cmake_dir.mkdir()
            for name in script_names:
                shutil.copy2(source_cmake / name, cmake_dir / name)
            header = root / "generated" / "values.h"
            manifest = root / "generated" / "provenance.txt"
            command = [
                "cmake",
                f"-DROCKE_SOURCE_ROOT={root}",
                f"-DROCKE_BUILD_ID_HEADER={header}",
                f"-DROCKE_BUILD_ID_MANIFEST={manifest}",
                "-DROCKE_ENGINE_VERSION=test",
                "-P",
                str(cmake_dir / "generate_rocke_build_id.cmake"),
            ]
            subprocess.run(command, check=True)
            expected = online.source_build_id(root)
            first_header = header.read_text()
            first_manifest = manifest.read_text()
            self.assertIn(f"#define ROCKE_BUILD_ID {expected}", first_header)
            self.assertEqual(first_manifest, f"test {expected}\n")

            # Re-run the build-time generator against the same output without a
            # configure step. This is the incremental path used by the custom
            # target; OBJECT_DEPENDS then recompiles only the provenance object.
            private.write_text("changed private header\n")
            subprocess.run(command, check=True)
            refreshed = header.read_text()
            self.assertNotEqual(first_header, refreshed)
            refreshed_id = online.source_build_id(root)
            self.assertIn(f"#define ROCKE_BUILD_ID {refreshed_id}", refreshed)
            self.assertNotEqual(first_manifest, manifest.read_text())
            self.assertEqual(manifest.read_text(), f"test {refreshed_id}\n")

    def test_installed_tree_without_cpp_sources_has_no_expected_id(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(online.source_build_id(td))


class TestInstalledProvenanceManifest(unittest.TestCase):
    def test_source_less_tree_uses_installed_expected_id(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "provenance.txt"
            manifest.write_text("1.0.0 installed-build\n")
            with (
                mock.patch.dict(
                    "os.environ",
                    {"ROCKE_PROVENANCE_MANIFEST": str(manifest)},
                    clear=True,
                ),
                mock.patch.object(
                    online, "_current_source_build_id", return_value=None
                ),
            ):
                self.assertEqual(
                    online.provenance_manifest(), ("1.0.0", "installed-build")
                )
                self.assertEqual(online._expected_build_id(), "installed-build")

    def test_manifest_must_match_source_when_both_are_present(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "provenance.txt"
            manifest.write_text("1.0.0 installed-build\n")
            with (
                mock.patch.dict(
                    "os.environ",
                    {"ROCKE_PROVENANCE_MANIFEST": str(manifest)},
                    clear=True,
                ),
                mock.patch.object(
                    online, "_current_source_build_id", return_value="source-build"
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    online._expected_build_id()

    def test_malformed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "provenance.txt"
            manifest.write_text("missing-build-id\n")
            with self.assertRaisesRegex(RuntimeError, "invalid.*manifest"):
                online.provenance_manifest(manifest)


class _CleanEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self._environment = mock.patch.dict(online.os.environ, {}, clear=True)
        self._environment.start()

    def tearDown(self):
        self._environment.stop()


class TestLoadedEngineFreshness(_CleanEnvironmentTest):
    def test_stale_engine_fails_once_with_both_build_ids(self):
        lib = _Library("old-build")
        with mock.patch.object(
            online, "_current_source_build_id", return_value="current-build"
        ):
            with self.assertRaisesRegex(RuntimeError, "stale/mixed rocKE engine") as cm:
                online._check_source_freshness(lib, "cached/librocke.so")
        message = str(cm.exception)
        self.assertIn("old-build", message)
        self.assertIn("current-build", message)
        self.assertIn("provenance manifest", message)
        self.assertIs(lib.rocke_build_id.restype, ctypes.c_char_p)

    def test_matching_engine_is_accepted(self):
        lib = _Library("same-build")
        with mock.patch.object(
            online, "_current_source_build_id", return_value="same-build"
        ):
            online._check_source_freshness(lib, "librocke.so")

    def test_public_load_rejects_stale_engine_before_binding(self):
        lib = _Library("old-build")
        online._lib = None
        online._lib_path = None
        with (
            mock.patch.object(online, "resolve_library_path", return_value="stale.so"),
            mock.patch.object(online.ctypes, "CDLL", return_value=lib),
            mock.patch.object(
                online, "_current_source_build_id", return_value="current-build"
            ),
            mock.patch.object(online, "_bind") as bind,
        ):
            with self.assertRaisesRegex(RuntimeError, "stale/mixed rocKE engine"):
                online.load()
        bind.assert_not_called()

    def test_already_loaded_engine_does_not_rehash_sources(self):
        lib = _Library("current-build")
        online._lib = lib
        online._lib_path = str(Path("loaded.so").resolve())
        try:
            with mock.patch.object(
                online, "_current_source_build_id"
            ) as source_build_id:
                self.assertIs(online.load(online._lib_path), lib)
                self.assertIs(online.load(online._lib_path), lib)
            source_build_id.assert_not_called()
        finally:
            online._lib = None
            online._lib_path = None

    def test_loaded_engine_cannot_ignore_a_later_explicit_selection(self):
        online._lib = object()
        online._lib_path = str(Path("first.so").resolve())
        with mock.patch.dict("os.environ", {"ROCKE_ONLINE_LIB": "second.so"}):
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                online.load()
        online._lib = None
        online._lib_path = None


class TestReplayCliFreshness(_CleanEnvironmentTest):
    def test_source_less_matching_manifest_resolves_explicit_cli(self):
        proc = subprocess.CompletedProcess(
            ["cli", "--provenance"], 0, stdout="1.0 current-build\n", stderr=""
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cli = root / "replay-cli"
            cli.touch()
            manifest = root / "provenance.txt"
            manifest.write_text("1.0 current-build\n")
            with (
                mock.patch.object(online.subprocess, "run", return_value=proc),
                mock.patch.object(
                    online, "_current_source_build_id", return_value=None
                ),
                mock.patch.object(online, "build_replay_cli") as build,
                mock.patch.dict(
                    "os.environ",
                    {
                        "ROCKE_REPLAY_CLI": str(cli),
                        "ROCKE_PROVENANCE_MANIFEST": str(manifest),
                    },
                    clear=True,
                ),
            ):
                self.assertEqual(online.resolve_replay_cli_path(build=False), str(cli))
        build.assert_not_called()

    def test_source_less_installed_cli_is_checked_against_manifest(self):
        proc = subprocess.CompletedProcess(
            ["cli", "--provenance"], 0, stdout="1.0 stale-build\n", stderr=""
        )
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "provenance.txt"
            manifest.write_text("1.0 current-build\n")
            with (
                mock.patch.object(online.subprocess, "run", return_value=proc),
                mock.patch.object(
                    online, "_current_source_build_id", return_value=None
                ),
                mock.patch.dict(
                    "os.environ",
                    {"ROCKE_PROVENANCE_MANIFEST": str(manifest)},
                    clear=True,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "stale/mixed rocKE replay CLI"
                ) as cm:
                    online.replay_cli_provenance("cli")
        self.assertIn("stale-build", str(cm.exception))
        self.assertIn("current-build", str(cm.exception))

    def test_stale_cli_reports_both_build_ids(self):
        proc = subprocess.CompletedProcess(
            ["cli", "--provenance"], 0, stdout="1.0 old-build\n", stderr=""
        )
        with (
            mock.patch.object(online.subprocess, "run", return_value=proc),
            mock.patch.object(
                online, "_current_source_build_id", return_value="current-build"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "stale/mixed rocKE replay CLI"
            ) as cm:
                online.replay_cli_provenance("cli")
        self.assertIn("old-build", str(cm.exception))
        self.assertIn("current-build", str(cm.exception))


class TestArtifactBundleProvenance(_CleanEnvironmentTest):
    def test_matching_ids_allow_different_configure_date_versions(self):
        with (
            mock.patch.object(
                online, "library_provenance", return_value=("1.0.0+20260101", "same")
            ),
            mock.patch.object(
                online,
                "replay_cli_provenance",
                return_value=("1.0.0+20260102", "same"),
            ),
        ):
            library, cli = online.validate_artifact_bundle("library", "cli")
        self.assertEqual(library[1], cli[1])
        self.assertNotEqual(library[0], cli[0])

    def test_mixed_ids_are_rejected_without_a_source_tree(self):
        with (
            mock.patch.object(
                online, "library_provenance", return_value=("1.0.0", "library-id")
            ),
            mock.patch.object(
                online, "replay_cli_provenance", return_value=("1.0.0", "cli-id")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "mixed bundles") as cm:
                online.validate_artifact_bundle("library", "cli")
        self.assertIn("library-id", str(cm.exception))
        self.assertIn("cli-id", str(cm.exception))


class TestCacheSelection(_CleanEnvironmentTest):
    def test_default_cache_is_keyed_by_source_build_id(self):
        with mock.patch.object(
            online, "_current_source_build_id", return_value="abc123"
        ):
            path = Path(online._default_cache_lib())
        self.assertEqual(path.parent.name, "abc123")

    def test_completion_marker_must_match_current_source(self):
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "librocke.so"
            artifact.touch()
            online._complete_marker(artifact).write_text("old-build")
            with mock.patch.object(
                online, "_current_source_build_id", return_value="current-build"
            ):
                self.assertFalse(online._published(artifact))

    def test_cache_lock_reuses_an_unlocked_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".build.lock").write_text("0")
            with online._cache_lock(root):
                self.assertTrue((root / ".build.lock").is_file())
            self.assertTrue((root / ".build.lock").is_file())

    def test_source_less_non_building_probes_return_none(self):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(online, "_current_source_build_id", return_value=None),
        ):
            self.assertIsNone(online.resolve_library_path(build=False))
            self.assertIsNone(online.resolve_replay_cli_path(build=False))

    def test_explicit_missing_library_never_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.so"
            with self.assertRaisesRegex(FileNotFoundError, "Refusing to fall back"):
                online.resolve_library_path(str(missing))


if __name__ == "__main__":
    unittest.main()
