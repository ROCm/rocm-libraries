# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's rk picks the code a process tree runs, once, at the tree's start: the release installed over the
image, only over the image it was installed on and while it is whole, else the image's code; RKMON_CODE=image picks
the image's. It exports its pick (RKMON_CODE), and every rk the tree starts runs that code (the rk script;
rkmon_lib/code_volume.py)."""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import sandbox  # noqa: E402
from rkmon_lib import code_volume, container_layout, releases  # noqa: E402

# rkmon.py of the tests' code: it prints the code it runs (the launcher's RKMON_CODE) and its working folder
RKMON_PY = "import os\nprint(os.environ['RKMON_CODE'], os.getcwd())\n"


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        base = pathlib.Path(self.d.name).resolve()
        self.image = base / "opt" / "rocke_monitor"
        self.code = base / "code"
        self.release = self.code / "releases" / "r1"
        self.other = self.code / "releases" / "r2"
        for tree in (self.image, self.release, self.other):
            (tree / "rkmon_lib").mkdir(parents=True)
            shutil.copy2(ROOT / "rk", tree / "rk")
            (tree / "rkmon.py").write_text(RKMON_PY)
            (tree / "rkmon_lib" / "store.py").write_text("# the code\n")
            sandbox.writable(tree)  # copy2 keeps a read-only package folder's modes
        for tree in (self.release, self.other):
            (tree / ".rkmon_release.json").write_text(
                json.dumps({"id": tree.name, "base_image": "img1"}) + "\n"
            )
        os.symlink("releases/r1", self.code / "current")
        self.idf = base / "image_code_id"
        self.idf.write_text("img1\n")
        self.env = dict(
            os.environ,
            RKMON_CONTAINER="1",
            RKMON_CODE_DIR=str(self.code),
            RKMON_IMAGE_CODE=str(self.image),
            RKMON_IMAGE_ID_FILE=str(self.idf),
        )
        for name in ("RKMON_CODE", "RKMON_WRAPPER"):
            self.env.pop(name, None)

    def rk(self, *args, rk=None, **env) -> subprocess.CompletedProcess:
        """rk ARGS in the container, the image's rk or the one at RK, with ENV."""
        return subprocess.run(
            ["sh", str(rk or self.image / "rk"), *args],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(self.env, **env),
        )

    def which(self, **env) -> str:
        p = self.rk("--which-code", **env)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.strip()

    def test_the_release_over_this_image_runs(self):
        self.assertEqual(self.which(), str(self.release))

    def test_a_release_over_another_image_does_not(self):
        self.idf.write_text("img2\n")
        self.assertEqual(self.which(), str(self.image))

    def test_a_damaged_release_runs_the_image_code_and_says_so(self):
        """A file of code_volume.WHOLE missing or empty: rk runs the image's code and says so, with the host's command
        that goes back to the code before it, and releases agrees (no release runs).
        """
        said = (
            f"rk: the code release in {self.release} is damaged (a file is missing or empty), so the image's code "
            "runs; to go back to the code before it: rkmon rollback"
        )
        for rel in code_volume.WHOLE:
            f = self.release / rel
            text = f.read_bytes()
            for how in ("empty", "missing"):
                with self.subTest(rel=rel, how=how):
                    if how == "empty":
                        f.write_bytes(b"")
                    else:
                        f.unlink()
                    p = self.rk("--which-code")
                    self.assertEqual(
                        (p.returncode, p.stdout.strip(), " ".join(p.stderr.split())),
                        (0, str(self.image), said),
                    )
                    with mock.patch.object(releases, "IMAGE_ID_FILE", self.idf):
                        self.assertIsNone(releases.active_release(self.code))
                    f.write_bytes(text)
        self.assertEqual(self.which(), str(self.release))
        # the fix as the operator types it, through the wrapper the command came in by
        (self.release / "rkmon.py").write_text("")
        p = self.rk("--which-code", RKMON_WRAPPER="./rkmon")
        self.assertTrue(
            " ".join(p.stderr.split()).endswith(
                "to go back to the code before it: ./rkmon rollback"
            ),
            p.stderr,
        )

    def test_rk_checks_the_files_code_volume_whole_names(self):
        """rk's whole() checks the files of code_volume.WHOLE, in its order, and no other."""
        text = (ROOT / "rk").read_text(encoding="utf-8")
        body = re.search(r"\n    whole\(\) \{\n(.*?)\n    \}\n", text, re.S).group(1)
        self.assertEqual(
            re.findall(r'\[ -s "\$1/([^"]+)" \]', body), list(code_volume.WHOLE)
        )

    def test_rk_and_releases_name_the_code_volume(self):
        """Without RKMON_CODE_DIR, rk and releases.code_dir name the code volume's folder: container_layout.CODE_DIR."""
        rk = re.search(
            r"\$\{RKMON_CODE_DIR:-([^}]+)\}", (ROOT / "rk").read_text(encoding="utf-8")
        ).group(1)
        self.assertEqual(
            (rk, str(releases.code_dir({}))),
            (container_layout.CODE_DIR, container_layout.CODE_DIR),
        )

    def test_rk_and_releases_name_the_folder_the_image_s_code_is_in(self):
        """Without RKMON_IMAGE_CODE, rk and releases.IMAGE_CODE name the folder docker/Dockerfile copies the code
        to."""
        rk = re.search(
            r"\$\{RKMON_IMAGE_CODE:-([^}]+)\}",
            (ROOT / "rk").read_text(encoding="utf-8"),
        ).group(1)
        env = {k: v for k, v in os.environ.items() if k != "RKMON_IMAGE_CODE"}
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import releases; "
            "print(releases.IMAGE_CODE)"
        )
        py = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        copied = re.search(r"^COPY \. (\S+)$", dockerfile, re.M).group(1)
        self.assertEqual((rk, py.stdout.strip()), (copied, copied), py.stderr)

    def test_the_image_code_on_request_and_outside_the_container(self):
        for env in ({"RKMON_CODE": "image"}, {"RKMON_CONTAINER": ""}):
            with self.subTest(env=env):
                p = self.rk("--which-code", **env)
                self.assertEqual(
                    (p.returncode, p.stdout.strip(), p.stderr), (0, str(self.image), "")
                )

    def test_a_tree_runs_the_code_its_first_rk_picked(self):
        """The rk that starts a tree exports the code it picked: every rk the tree starts, the image's or a
        release's, runs that code in its folder, even after an install put another release in place; a tree started
        after that runs the new release."""
        p = self.rk("status")
        self.assertEqual(
            (p.returncode, p.stdout.split()), (0, [str(self.release)] * 2), p.stderr
        )
        (self.code / "current").unlink()
        os.symlink("releases/r2", self.code / "current")  # an install, with no restart
        for rk in (self.image / "rk", self.release / "rk", self.other / "rk"):
            with self.subTest(rk=rk):
                p = self.rk("status", rk=rk, RKMON_CODE=str(self.release))
                self.assertEqual(p.stdout.split(), [str(self.release)] * 2, p.stderr)
        self.assertEqual(self.rk("status").stdout.split(), [str(self.other)] * 2)

    def test_a_picked_code_that_is_gone_or_damaged_never_runs(self):
        """The code a tree picked, gone or damaged since: rk runs the image's code, and says so."""
        gone = str(self.code / "releases" / "r9")
        (self.release / "rkmon_lib" / "store.py").write_text("")
        for pinned in (gone, str(self.release)):
            with self.subTest(pinned=pinned):
                p = self.rk("--which-code", RKMON_CODE=pinned)
                self.assertEqual(
                    (p.returncode, p.stdout.strip(), " ".join(p.stderr.split())),
                    (
                        0,
                        str(self.image),
                        f"rk: the code this command was started with ({pinned}) is gone or damaged, so the "
                        "image's code runs",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
