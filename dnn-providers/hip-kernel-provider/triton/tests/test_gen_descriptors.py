# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gen_descriptors re-derives the committed tree into a scratch root, replaces what it
owns (stale descriptors included), and leaves the output untouched on failure; its
``--check`` passes on the committed tree, writes nothing, never imports triton, and
fails naming the file on every kind of drift. Drift is planted in a scratch copy of
the generator and descriptor trees, never in this checkout."""

import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

import gen_descriptors  # noqa: E402
from tritongen import _family as F  # noqa: E402
from tritongen.ops import OPS  # noqa: E402


def run(*argv: str) -> tuple[int, str]:
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = gen_descriptors.main(list(argv))
    return code, out.getvalue() + err.getvalue()


def same_tree(left: Path, right: Path) -> bool:
    cmp = filecmp.dircmp(left, right)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(left, right, cmp.common_files, shallow=False)
    return (
        not mismatch
        and not errors
        and all(same_tree(left / d, right / d) for d in cmp.common_dirs)
    )


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class GenDescriptorsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_rederivation_matches_committed_tree(self):
        code, log = run("--in", str(F.CONTENT_DIR), "--out", str(self.tmp / "o"))
        self.assertEqual(code, 0, log)
        self.assertTrue(same_tree(F.CONTENT_DIR, self.tmp / "o"))

    def test_in_place_run_removes_stale_descriptors(self):
        root = self.tmp / "root"
        shutil.copytree(F.CONTENT_DIR, root)
        for op in OPS.values():
            stale_op = root / op.OP / "stale.uhd.json"
            stale_arch = root / op.OP / op.ARCHES[0] / "gone.ukd.json"
            stale_op.write_text("{}")
            stale_arch.write_text("{}")
            code, log = run("--in", str(root), "--out", str(root), "--op", op.OP)
            self.assertEqual(code, 0, log)
            self.assertFalse(stale_op.exists())
            self.assertFalse(stale_arch.exists())
        self.assertTrue(same_tree(F.CONTENT_DIR, root))
        self.assertEqual(
            sorted(p.name for p in root.iterdir()),
            sorted(OPS),
            "no staging dir left behind",
        )

    def test_in_place_run_removes_non_contract_arches(self):
        root = self.tmp / "root"
        shutil.copytree(F.CONTENT_DIR, root)
        planted = []
        for op in OPS.values():
            stale_dir = root / op.OP / "gfx90a"
            stale_dir.mkdir()
            planted += [
                stale_dir / "bad.ukd.json",
                stale_dir / "notes.hsaco",
                root / op.OP / "gfx90a.SOURCE.md",
            ]
        for path in planted:
            path.write_text("{}")
        code, log = run("--in", str(root), "--out", str(root))
        self.assertEqual(code, 0, log)
        for path in planted:
            self.assertFalse(path.exists(), path)
        self.assertTrue(same_tree(F.CONTENT_DIR, root))

    def test_failure_leaves_output_untouched(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                broken = self.tmp / op.OP / "in"
                shutil.copytree(F.CONTENT_DIR, broken)
                victim = (
                    broken
                    / op.OP
                    / op.ARCHES[-1]
                    / F.hsaco_file(op.instances()[0].name)
                )
                victim.write_bytes(victim.read_bytes() + b"x")
                out = self.tmp / op.OP / "out"
                shutil.copytree(F.CONTENT_DIR, out)
                code, log = run("--in", str(broken), "--out", str(out))
                self.assertEqual(code, 1)
                self.assertIn(f"error: {F.ERR_INPUT}: ", log)
                self.assertIn("SHA256", log)
                self.assertTrue(same_tree(F.CONTENT_DIR, out))


class CheckTest(unittest.TestCase):
    """``gen_descriptors.py --check`` run as a separate process on a scratch copy of the
    hip-kernel-provider tree (generator, descriptors, packaging library)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.hkp = Path(cls._tmp.name) / "hkp"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(F.TRITON_DIR, cls.hkp / "triton", ignore=ignore)
        shutil.copytree(
            F.CONTENT_DIR,
            cls.hkp / F.CONTENT_DIR.relative_to(F.HKP_DIR),
            ignore=ignore,
        )
        shutil.copytree(
            F.HKP_DIR / "descriptor-packaging" / "python",
            cls.hkp / "descriptor-packaging" / "python",
            ignore=ignore,
        )
        cls.pristine = cls.hkp.with_name("pristine")
        shutil.copytree(cls.hkp, cls.pristine)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        shutil.rmtree(self.hkp)
        shutil.copytree(self.pristine, self.hkp)
        self.triton = self.hkp / "triton"
        self.content = self.hkp / F.CONTENT_DIR.relative_to(F.HKP_DIR)

    def check(self, blocked: str | None = None) -> tuple[int, list[str], str]:
        """Run ``--check``; ``blocked`` names a module made unimportable first."""
        script = str(self.triton / "gen_descriptors.py")
        if blocked:
            argv = [
                sys.executable,
                "-c",
                f"import runpy, sys; sys.modules[{blocked!r}] = None; "
                f"sys.argv = [{script!r}, '--check']; "
                f"runpy.run_path({script!r}, run_name='__main__')",
            ]
        else:
            argv = [sys.executable, script, "--check"]
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        lines = [line for line in result.stderr.splitlines() if line]
        return result.returncode, lines, result.stdout

    def assert_fails_naming(
        self, planted: Path, *fragments: str, named: Path | None = None
    ):
        code, lines, out = self.check()
        self.assertEqual(code, 1, (lines, out))
        self.assertTrue(lines, out)
        for line in lines:
            self.assertTrue(line.startswith(f"{F.ERR_CHECK}: "), line)
        prefix = f"{F.ERR_CHECK}: {(named or planted).resolve()}: "
        hits = [line for line in lines if line.startswith(prefix)]
        self.assertTrue(hits, (prefix, lines))
        for fragment in fragments:
            self.assertTrue(any(fragment in line for line in hits), (fragment, hits))

    def each_op(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                yield op

    def first_ukd(self, op) -> Path:
        return self.content / op.OP / op.ARCHES[0] / F.ukd_file(op.instances()[0].name)

    def first_object(self, op) -> Path:
        return (
            self.content / op.OP / op.ARCHES[0] / F.hsaco_file(op.instances()[0].name)
        )

    def manifest(self, op) -> Path:
        return self.content / op.OP / op.ARCHES[0] / F.MANIFEST_FILE

    def test_clean_tree_passes_and_writes_nothing(self):
        before = snapshot(self.hkp)
        code, lines, out = self.check()
        self.assertEqual(code, 0, lines)
        self.assertEqual(lines, [])
        self.assertEqual(len(out.splitlines()), 1, out)
        self.assertEqual(snapshot(self.hkp), before)

    def test_clean_tree_passes_without_triton(self):
        code, lines, _ = self.check(blocked="triton")
        self.assertEqual(code, 0, lines)

    def test_missing_msgpack_is_one_env_error(self):
        code, lines, _ = self.check(blocked="msgpack")
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(f"error: {F.ERR_ENV}: "), lines)

    def test_non_contract_arch_fails(self):
        for op in self.each_op():
            arch_dir = self.content / op.OP / "gfx90a"
            shutil.copytree(self.content / op.OP / op.ARCHES[0], arch_dir)
            source_md = self.content / op.OP / f"gfx90a{F.SOURCE_MD_SUFFIX}"
            source_md.write_text("x")
            self.assert_fails_naming(arch_dir, "not a contract arch")
            self.assert_fails_naming(source_md, "not a contract arch")

    def test_stale_descriptor_fails(self):
        for op in self.each_op():
            stale = self.content / op.OP / op.ARCHES[0] / "gone.ukd.json"
            shutil.copy2(self.first_ukd(op), stale)
            self.assert_fails_naming(stale, "not derived")

    def test_missing_descriptor_fails(self):
        for op in self.each_op():
            ukd = self.first_ukd(op)
            ukd.unlink()
            self.assert_fails_naming(ukd, "missing")

    def test_hand_edited_descriptor_fails(self):
        for op in self.each_op():
            for path in (
                self.first_ukd(op),
                self.content / op.OP / F.op_file(op.OP, "ued"),
            ):
                with self.subTest(path=path.name):
                    self.setUp()
                    text = path.read_text(encoding="utf-8")
                    path.write_text(
                        text.replace('"version": "1.0"', '"version": "1.1"')
                    )
                    self.assert_fails_naming(path, "differs")

    def test_changed_kernel_source_fails(self):
        for op in self.each_op():
            source = self.triton / op.KERNEL_SOURCE_FILE
            with source.open("a", encoding="utf-8") as handle:
                handle.write("# edited\n")
            self.assert_fails_naming(
                source, "inputs_sha256", op.KERNEL_SOURCE_FILE, named=self.manifest(op)
            )

    def test_changed_generator_module_fails(self):
        for op in self.each_op():
            for module in (
                F.PACKAGE_DIR / "_codeobject.py",
                F.PACKAGE_DIR / "ops" / f"{op.OP}.py",
            ):
                rel = module.relative_to(F.TRITON_DIR).as_posix()
                with self.subTest(module=rel):
                    self.setUp()
                    with (self.triton / rel).open("a", encoding="utf-8") as handle:
                        handle.write("# edited\n")
                    self.assert_fails_naming(
                        self.triton / rel, "inputs_sha256", rel, named=self.manifest(op)
                    )

    def test_changed_requirements_fail(self):
        for op in self.each_op():
            requirements = self.triton / F.REQUIREMENTS.name
            with requirements.open("a", encoding="utf-8") as handle:
                handle.write("# edited\n")
            self.assert_fails_naming(
                requirements,
                "inputs_sha256",
                F.REQUIREMENTS.name,
                named=self.manifest(op),
            )

    def test_line_endings_are_not_a_change(self):
        source = self.triton / next(iter(OPS.values())).KERNEL_SOURCE_FILE
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
        code, lines, _ = self.check()
        self.assertEqual(code, 0, lines)

    def test_stray_object_fails(self):
        for op in self.each_op():
            stray = self.content / op.OP / op.ARCHES[0] / "stray.hsaco"
            shutil.copy2(self.first_object(op), stray)
            self.assert_fails_naming(stray, "does not name")

    def test_object_sha_mismatch_fails(self):
        for op in self.each_op():
            victim = self.first_object(op)
            victim.write_bytes(victim.read_bytes() + b"\0")
            self.assert_fails_naming(victim, "SHA256")


if __name__ == "__main__":
    unittest.main()
