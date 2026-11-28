# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ABI homes on the merge (lessons L-579): abi.h's define and newest history line, abi.py's mirror and the layout
canaries must agree across all files, else HALF RENUMBER; and the clash scan's ABI lane: two open PRs claiming the same
counter (the second to land renumbers). Temp git repos, no network."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rkmon_lib import abi, clashes, review  # noqa: E402

HEADER = """/* Binary ABI.
 * 2: rocke_memory_caps_t gained mid-struct fields.
 * 3: rocke_mma_op_t gained fields at its end. */
#define ROCKE_ABI_VERSION 3

/* Wire ABI.
 * 2: list-tagged kernel attrs. */
#define ROCKE_RECIPE_ABI 2
"""
MIRROR = "RECIPE_ABI = 2\n\nBINARY_ABI = 3\n"
CANARY = (
    "static_assert(sizeof(rocke_memory_caps_t) == 20 && ROCKE_ABI_VERSION >= {n},\n"
    '              "rocke_memory_caps_t changed size: bump ROCKE_ABI_VERSION");\n'
)


def bump(text: str, n: int, what: str) -> str:
    return text.replace(
        " at its end. */\n#define ROCKE_ABI_VERSION 3",
        f" at its end.\n * {n}: {what}. */\n#define ROCKE_ABI_VERSION {n}",
    )


class Repo:
    def __init__(self, d: str):
        self.d = d
        self.g("init", "-q", "-b", "develop")
        self.write(
            {
                f"rk/{abi.ABI_H}": HEADER,
                f"rk/{abi.ABI_PY}": MIRROR,
                "rk/platform/tests/core/test_caps.cpp": CANARY.format(n=2),
                "rk/README.md": "x\n",
            }
        )
        self.g("add", "-A")
        self.g("commit", "-qm", "base")
        self.base = self.g("rev-parse", "HEAD")

    def g(self, *a) -> str:
        return subprocess.run(
            ["git", "-C", self.d, "-c", "user.name=t", "-c", "user.email=t@t", *a],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, files: dict) -> None:
        for path, text in files.items():
            p = pathlib.Path(self.d, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def commit(self, files: dict, msg: str) -> str:
        self.write(files)
        self.g("add", "-A")
        self.g("commit", "-qm", msg)
        return self.g("rev-parse", "HEAD")

    def branch(self, name: str, files: dict) -> str:
        self.g("checkout", "-q", "-b", name, self.base)
        sha = self.commit(files, name)
        self.g("checkout", "-q", "develop")
        return sha


class TestHomes(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.r = Repo(self.t.name)

    def test_homes_that_agree(self):
        h = abi.homes(self.t.name, "rk", self.r.base)
        self.assertEqual(
            abi.homes_lines(h),
            [
                "homes binary: define 3 · history 3 · abi.py 3 · canary 2 · OK",
                "homes recipe: define 2 · history 2 · abi.py 2 · OK",
            ],
        )

    def test_each_half_renumber(self):
        h = abi.homes(
            self.t.name,
            "rk",
            self.r.commit(
                {f"rk/{abi.ABI_H}": bump(HEADER, 4, "rocke_caps_t gained b")},
                "abi.h only",
            ),
        )
        self.assertEqual(
            abi.homes_lines(h)[0],
            "homes binary: define 4 · history 4 · abi.py 3 · canary 2 · HALF RENUMBER: abi.py BINARY_ABI 3 but "
            "abi.h ROCKE_ABI_VERSION 4",
        )
        # The define moved back but the history line and a canary stayed: three homes disagree.
        sha = self.r.commit(
            {
                f"rk/{abi.ABI_H}": bump(HEADER, 4, "x").replace(
                    "#define ROCKE_ABI_VERSION 4", "#define ROCKE_ABI_VERSION 3"
                ),
                "rk/library/cpp/tests/conv_host.cpp": CANARY.format(n=4),
            },
            "define only",
        )
        probs = abi.homes(self.t.name, "rk", sha)["binary"]["problems"]
        self.assertEqual(
            probs,
            [
                "abi.h ROCKE_ABI_VERSION 3 but its newest history line is 4",
                "a layout canary needs ROCKE_ABI_VERSION >= 4 (library/cpp/tests/conv_host.cpp:1) but abi.h has 3",
            ],
        )
        # A history number used twice (a keep-both resolution of two bumps).
        dup = HEADER.replace(
            " * 3: rocke_mma_op_t", " * 3: another.\n * 3: rocke_mma_op_t"
        )
        sha = self.r.commit(
            {
                f"rk/{abi.ABI_H}": dup,
                "rk/library/cpp/tests/conv_host.cpp": CANARY.format(n=2),
            },
            "dup",
        )
        self.assertEqual(
            abi.homes(self.t.name, "rk", sha)["binary"]["problems"],
            ["history number(s) [3] appear more than once"],
        )

    def test_no_abi_h(self):
        sha = self.r.commit({f"rk/{abi.ABI_H}": ""}, "gone")
        self.assertEqual(abi.homes(self.t.name, "rk", sha), {})
        self.assertEqual(abi.homes_lines({}), ["homes: no abi.h in that tree"])

    def scan(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "tools/abi_scan.py"),
                self.t.name,
                "rk",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

    def test_the_scan_reads_the_homes_on_the_merge(self):
        # The PR bumps abi.h and abi.py together to 4; develop meanwhile adds a canary that needs 5. Each side is
        # whole, but their merge is not: CI ran neither combination.
        head = self.r.branch(
            "pr",
            {
                f"rk/{abi.ABI_H}": bump(HEADER, 4, "rocke_caps_t gained b"),
                f"rk/{abi.ABI_PY}": MIRROR.replace("BINARY_ABI = 3", "BINARY_ABI = 4"),
            },
        )
        self.r.commit(
            {"rk/platform/tests/core/test_op.cpp": CANARY.format(n=5)}, "develop canary"
        )
        self.r.g("checkout", "-q", "--detach", head)
        self.r.g("merge", "-q", "--no-edit", "develop")
        merge = self.r.g("rev-parse", "HEAD")
        out = self.scan(head, "develop", merge)
        self.assertIn(f"on the merge {merge[:10]}:", out)
        self.assertTrue(
            any(
                ln.startswith(
                    "homes binary: define 4 · history 4 · abi.py 4 · canary 5 · HALF RENUMBER"
                )
                for ln in out
            ),
            out,
        )
        self.assertTrue(
            out[-1].startswith("ABI fail HALF RENUMBER on the merge: a layout canary"),
            out[-1],
        )
        # Without a merged tree the homes are not read, and the scan passes on the head's own bump.
        out = self.scan(head, "develop", "-")
        self.assertIn("homes: not read (no merged tree)", out)
        self.assertTrue(
            out[-1].startswith("ABI pass ROCKE_ABI_VERSION 3 -> 4"), out[-1]
        )

    def test_the_checks_pass_the_merge(self):
        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
            "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
        }
        plan = review.plan_checks(cfg, {"n": 7}, ["rk/docs/x.md"], {})
        plan["_dirs"] = {}
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", plan, None)
        self.assertIn(
            'AM=-; [ "$MERGED" = 1 ] && AM=$(git -C $W rev-parse HEAD', script
        )
        self.assertIn(f"$M/tools/abi_scan.py $R0 $K {'a' * 40} $BREF $AM", script)


class TestClashLane(unittest.TestCase):
    """Two open PRs bump ROCKE_ABI_VERSION to 4 with different history lines; a third carries the first's very hunk
    (one claim); a fourth bumps nothing."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        r = self.r = Repo(self.t.name)
        one = {
            f"rk/{abi.ABI_H}": bump(HEADER, 4, "rocke_caps_t gained b"),
            f"rk/{abi.ABI_PY}": MIRROR.replace("BINARY_ABI = 3", "BINARY_ABI = 4"),
        }
        self.heads = {
            1: r.branch("p1", one),
            2: r.branch(
                "p2",
                {
                    f"rk/{abi.ABI_H}": bump(HEADER, 4, "rocke_op_t gained c"),
                    f"rk/{abi.ABI_PY}": MIRROR.replace(
                        "BINARY_ABI = 3", "BINARY_ABI = 4"
                    ),
                },
            ),
            3: r.branch("p3", {**one, "rk/README.md": "y\n"}),
            4: r.branch("p4", {"rk/README.md": "z\n"}),
        }
        self.cfg = {"rocke_path": "rk", "paths": {"repo": self.t.name}}
        self.st = {
            "prs": {
                str(k): {
                    "status": "open",
                    "head": h,
                    "base": "develop",
                    "head_ref": f"p{k}",
                }
                for k, h in self.heads.items()
            }
        }

    def test_the_same_counter_claimed_twice_is_a_number_clash(self):
        prs = clashes.open_prs(self.st)
        a = clashes.abi_lane(self.cfg, self.st, 1, prs, self.r.base)
        self.assertEqual(a["develop"], {"binary": 3, "recipe": 2})
        self.assertEqual(a["clashes"], [{"lane": "binary", "n": 2, "to": 4, "mine": 4}])
        self.assertEqual(
            clashes.abi_lines(a),
            [
                "ABI NUMBER CLASH binary ABI (ROCKE_ABI_VERSION): develop 3; this PR claims 4; #2 claims 4: whichever "
                "lands second renumbers to 5 in every home (abi.h define and history line, abi.py BINARY_ABI, the "
                "layout canaries)"
            ],
        )
        res = {"n": 1, "pairs": [], "not_run": [], "abi": a}
        self.assertEqual(clashes.verdict(res), ("fail", "ABI number clash with #2"))
        # A PR that bumps nothing has no lane line and no ABI clash.
        none = clashes.abi_lane(self.cfg, self.st, 4, prs, self.r.base)
        self.assertEqual(
            (none["mine"], none["clashes"], clashes.abi_lines(none)), ({}, [], [])
        )

    def test_alone_in_the_lane_and_the_tick_s_claims(self):
        st = {"prs": {k: v for k, v in self.st["prs"].items() if k in ("1", "4")}}
        prs = clashes.open_prs(st)
        a = clashes.abi_lane(self.cfg, st, 1, prs, self.r.base)
        self.assertEqual(
            clashes.abi_lines(a),
            [
                "ABI lane binary ABI (ROCKE_ABI_VERSION): develop 3; this PR claims 4; no other open PR claims this counter"
            ],
        )
        # The tick's cached claim is used while its head is the one compared.
        st["prs"]["4"]["abi_cache"] = {
            "head": self.heads[4],
            "v": abi.CACHE_VERSION,
            "lanes": {"binary": {"from": 3, "to": 4, "hunk": "cached"}},
        }
        a = clashes.abi_lane(self.cfg, st, 1, prs, self.r.base)
        self.assertEqual([c["n"] for c in a["clashes"]], [4])

    def test_the_scan_s_section_shows_the_lane(self):
        res, lines = clashes.run(self.cfg, self.st, 1, develop=self.r.base, budget_s=60)
        self.assertTrue(
            any(ln.startswith("ABI NUMBER CLASH binary") for ln in lines), lines
        )
        self.assertTrue(
            lines[-1].startswith("CLASH fail ABI number clash with #2"), lines[-1]
        )


if __name__ == "__main__":
    unittest.main()
