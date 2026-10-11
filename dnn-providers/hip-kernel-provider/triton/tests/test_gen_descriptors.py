# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gen_descriptors re-derives the committed tree into a scratch root, replaces what it
owns (stale descriptors included), and leaves the output untouched on failure."""

import filecmp
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

import gen_descriptors  # noqa: E402
from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402


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


class GenDescriptorsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_rederivation_matches_committed_tree(self):
        code, log = run("--in", str(env.CONTENT_DIR), "--out", str(self.tmp / "o"))
        self.assertEqual(code, 0, log)
        self.assertTrue(same_tree(env.CONTENT_DIR, self.tmp / "o"))

    def test_in_place_run_removes_stale_descriptors(self):
        root = self.tmp / "root"
        shutil.copytree(env.CONTENT_DIR, root)
        stale_op = root / C.OP / "stale.uhd.json"
        stale_arch = root / C.OP / "gfx942" / "gone.ukd.json"
        stale_op.write_text("{}")
        stale_arch.write_text("{}")
        code, log = run("--in", str(root), "--out", str(root))
        self.assertEqual(code, 0, log)
        self.assertFalse(stale_op.exists())
        self.assertFalse(stale_arch.exists())
        self.assertTrue(same_tree(env.CONTENT_DIR, root))
        self.assertEqual(
            [p.name for p in root.iterdir()], [C.OP], "no staging dir left behind"
        )

    def test_in_place_run_removes_non_contract_arches(self):
        root = self.tmp / "root"
        shutil.copytree(env.CONTENT_DIR, root)
        stale_dir = root / C.OP / "gfx90a"
        stale_dir.mkdir()
        planted = [
            stale_dir / "bad.ukd.json",
            stale_dir / "notes.hsaco",
            root / C.OP / "gfx90a.SOURCE.md",
        ]
        for path in planted:
            path.write_text("{}")
        code, log = run("--in", str(root), "--out", str(root))
        self.assertEqual(code, 0, log)
        for path in planted:
            self.assertFalse(path.exists(), path)
        self.assertFalse(stale_dir.exists())
        self.assertTrue(same_tree(env.CONTENT_DIR, root))

    def test_failure_leaves_output_untouched(self):
        broken = self.tmp / "in"
        shutil.copytree(env.CONTENT_DIR, broken)
        victim = broken / C.OP / "gfx950" / C.hsaco_file(C.instances()[0].name)
        victim.write_bytes(victim.read_bytes() + b"x")
        out = self.tmp / "out"
        shutil.copytree(env.CONTENT_DIR, out)
        code, log = run("--in", str(broken), "--out", str(out))
        self.assertEqual(code, 1)
        self.assertIn(f"error: {C.ERR_INPUT}: ", log)
        self.assertIn("SHA256", log)
        self.assertTrue(same_tree(env.CONTENT_DIR, out))


if __name__ == "__main__":
    unittest.main()
