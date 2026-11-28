# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A review round's impact score (rkmon_lib/hybrid/impact.py) reads its paths from below the folder the configuration
names, and looks for twins, names and importers in that folder's tree; rk spend passes the folder in. A temp git repo,
no network."""

import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import spend  # noqa: E402
from rkmon_lib.hybrid import impact, size  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402

FOLDER = "src/proj"


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class TestFolder(unittest.TestCase):
    def test_a_path_is_read_from_below_the_folder(self):
        """The engines and the path signals read a path from below the folder given; a path outside it, or any path
        with a folder of none (the repository's root), in full."""
        cpp, header = f"{FOLDER}/platform/cpp/a.cpp", f"{FOLDER}/CMakeLists.txt"
        self.assertEqual(impact.rel(cpp, FOLDER), "platform/cpp/a.cpp")
        self.assertEqual(impact.rel(cpp, "lib/k"), cpp)
        self.assertEqual(impact.rel("platform/cpp/a.cpp", ""), "platform/cpp/a.cpp")
        self.assertEqual(impact.engine(cpp, FOLDER), "cpp")
        self.assertEqual(impact.engine(f"{FOLDER}/library/a.py", FOLDER), "py")
        self.assertIsNone(impact.engine(cpp, "lib/k"))
        self.assertEqual(impact.engine("platform/cpp/a.cpp", ""), "cpp")
        self.assertIn("registry", impact.path_signals(header, FOLDER))
        h = {"lines": ["+add_test(NAME x COMMAND y)"]}
        self.assertEqual(impact.content_signals(header, h, FOLDER), {"ctest_line"})

    def test_the_tree_is_the_folder_s(self):
        """A round's tree holds the folder's files only, and finds a file's twin and its importers in it; with a
        folder of none, the whole repository's."""
        with tempfile.TemporaryDirectory() as d:
            repo = pathlib.Path(d)
            g(repo, "init", "-q")
            files = {
                f"{FOLDER}/platform/cpp/include/widget.h": "int widget(int a);\n",
                f"{FOLDER}/platform/cpp/widget.cpp": '#include "widget.h"\n',
                f"{FOLDER}/library/widget.py": "def widget(a):\n    return a\n",
                "other/platform/cpp/user.cpp": '#include "widget.h"\n',
            }
            for name, text in files.items():
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                (repo / name).write_text(text)
            g(repo, "add", "-A")
            g(repo, "commit", "-q", "-m", "x")
            tree = impact.Tree(str(repo), "HEAD", FOLDER)
            self.assertEqual(
                sorted(tree.files), sorted(f for f in files if f.startswith(FOLDER))
            )
            header = f"{FOLDER}/platform/cpp/include/widget.h"
            self.assertEqual(
                tree.twin(f"{FOLDER}/library/widget.py"),
                [header, f"{FOLDER}/platform/cpp/widget.cpp"],
            )
            self.assertEqual(tree.importers(header), 1)
            self.assertEqual(
                tree.files_naming(["widget"])["widget"],
                {f for f in files if f.startswith(FOLDER)},
            )
            whole = impact.Tree(str(repo), "HEAD", "")
            self.assertEqual(sorted(whole.files), sorted(files))
            self.assertEqual(whole.importers(header), 2)

    def test_rk_spend_scores_a_round_in_the_configured_folder(self):
        """rk spend's impact of a post asks the round's score for the folder the configuration names."""
        seen = []

        def score_round(*a, folder, **kw):
            seen.append(folder)
            return {"ok": True, "tier": "low", "score": 1}

        post = {"head": "h" * 40, "old": ""}
        cfg = {"rocke_path": f"/{FOLDER}/", "paths": {"repo": "/r"}}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            spend, "_develop", return_value="d" * 40
        ), mock.patch.object(
            size, "measure", return_value=({"eff": 3, "tier": "S", "src": "pr"}, [], [])
        ), mock.patch.object(
            impact, "score_round", score_round
        ):
            spend.measure(cfg, Store(pathlib.Path(d)), [post], impact_for=[post])
        self.assertEqual(seen, [FOLDER])
        self.assertEqual(post["impact"], "low (1)")


if __name__ == "__main__":
    unittest.main()
