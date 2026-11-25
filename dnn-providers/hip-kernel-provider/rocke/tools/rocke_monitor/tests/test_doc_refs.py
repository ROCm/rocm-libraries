# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/doc_refs.py: doc lines that name what the PR removes or renames, and new env vars, IR ops and dtypes the
reference docs do not name (agent_script_overlap.md: docs and reference text against code, by hand in every review).
Leads, not verdicts."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "doc_refs.py"
sys.path.insert(0, str(TOOL.parent))
import doc_refs as dr  # noqa: E402

K = "prov/rk"
REF = f"{K}/platform/dsl_docs/reference"


def g(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestDocRefs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name)
        self.write(
            {
                f"{K}/platform/python/rocke/core/ir.py": 'F16 = Type("f16")\nOPS = ["arith.add", "arith.mul"]\n',
                f"{K}/platform/python/rocke/tools/run.py": (
                    "import os\n\n\ndef build_kernel(x):\n    return x\n\n\n"
                    'def keep():\n    return os.environ.get("ROCKE_KEEP")\n\n\n'
                    'LEVEL = os.environ.get("ROCKE_OLD_LEVEL", "1")\n'
                    'parser.add_argument("--old-flag")\nARCHS = ["gfx942", "gfx908"]\n'
                ),
                f"{K}/README.md": "Call `build_kernel(x)`; set ROCKE_OLD_LEVEL=2; pass --old-flag; runs on gfx908.\n"
                "Unrelated: the build kernel step and ROCKE_KEEP.\n",
                f"{K}/CHANGELOG.md": "Removed build_kernel( in 1.0 and ROCKE_OLD_LEVEL.\n",
                f"{REF}/env_flags.md": "| `ROCKE_KEEP` | keep |\n| `ROCKE_OLD_LEVEL` | level |\n",
                f"{REF}/op_vocabulary.md": "| `arith.add` | add |\n| `arith.mul` | mul |\n",
                f"{REF}/dtype_support.md": "| `f16` | half |\n",
            }
        )
        g(r, "init", "-q", "-b", "develop")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        self.write(
            {
                f"{K}/platform/python/rocke/core/ir.py": 'F16 = Type("f16")\nFP9 = Type("fp9")\n'
                'OPS = ["arith.add", "arith.mul", "arith.newop", "nodialect.x"]\n',
                f"{K}/platform/python/rocke/tools/run.py": (
                    "import os\n\n\ndef make_kernel(x):\n    return x\n\n\n"
                    'def keep():\n    return os.environ.get("ROCKE_KEEP")\n\n\n'
                    'LEVEL = os.environ.get("ROCKE_NEW_LEVEL", "1")\nARCHS = ["gfx942"]\n'
                ),
                f"{K}/library/tests/test_run.py": 'import os\nX = os.environ.get("ROCKE_TEST_ONLY")\n',
            }
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")

    def write(self, files: dict) -> None:
        for path, text in files.items():
            p = self.repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def test_stale_doc_lines_and_unlisted_new_names_are_leads(self):
        res = dr.check(str(self.repo), self.head, K=K, base="develop")
        self.assertIsNone(res["error"])
        self.assertEqual(
            sorted(res["gone"]),
            ["--old-flag", "ROCKE_OLD_LEVEL", "build_kernel", "gfx908"],
        )
        self.assertEqual(
            sorted((x["doc"], x["line"], x["name"]) for x in res["stale"]),
            [
                ("README.md", 1, "--old-flag"),
                ("README.md", 1, "ROCKE_OLD_LEVEL"),
                ("README.md", 1, "build_kernel"),
                ("README.md", 1, "gfx908"),
                ("platform/dsl_docs/reference/env_flags.md", 2, "ROCKE_OLD_LEVEL"),
            ],
        )
        # A test's env var, an op of a dialect the vocabulary does not use, a listed name: none is a lead.
        self.assertEqual(
            [(x["kind"], x["name"], x["at"]) for x in res["new"]],
            [
                ("dtype", "fp9", "platform/python/rocke/core/ir.py:2"),
                ("op", "arith.newop", "platform/python/rocke/core/ir.py:3"),
                ("env", "ROCKE_NEW_LEVEL", "platform/python/rocke/tools/run.py:12"),
            ],
        )
        out = dr.render(res)
        self.assertIn(
            "LEAD README.md:1 names build_kernel, a function or class the PR removes or renames (it was in "
            "platform/python/rocke/tools/run.py)",
            out,
        )
        self.assertIn(
            "LEAD new environment variable ROCKE_NEW_LEVEL (platform/python/rocke/tools/run.py:12) is not in "
            "platform/dsl_docs/reference/env_flags.md",
            out,
        )
        self.assertTrue(
            out[-1].startswith(
                "DOCREFS fail 8 leads: 5 doc lines name what the PR removes"
            ),
            out[-1],
        )

    def test_a_moved_old_name_is_not_new_and_a_clean_pr_passes(self):
        g(self.repo, "checkout", "-q", "develop")
        g(self.repo, "checkout", "-qb", "clean")
        self.write(
            {
                f"{K}/platform/python/rocke/core/ir.py": 'OPS = ["arith.mul", "arith.add"]\nF16 = Type("f16")\n'
            }
        )
        g(self.repo, "commit", "-qam", "reorder")
        res = dr.check(
            str(self.repo), g(self.repo, "rev-parse", "HEAD"), K=K, base="develop"
        )
        self.assertEqual((res["gone"], res["stale"], res["new"]), ({}, [], []))
        self.assertTrue(dr.render(res)[-1].startswith("DOCREFS pass "))

    def test_the_checks_script_runs_it_as_a_lead_section(self):
        sys.path.insert(0, str(TOOL.parents[1]))
        from rkmon_lib import review

        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(
            {
                "rocke_path": "rk",
                "repo": "o/r",
                "paths": {"working_dir": "/w"},
            },
            7,
            "a" * 40,
            "follow-up",
            plan,
            None,
        )
        sec = script.split(f"sec '{review.DOCREFS_SECTION}'", 1)[1].split("\nsec ", 1)[
            0
        ]
        self.assertIn(
            f"tools/doc_refs.py --repo $R0 --head {'a' * 40} --rocke-path $K --base $BREF --budget 120",
            sec,
        )
        self.assertIn("st docrefs ", sec)
        self.assertIn("docrefs", review.LEAD_SECTIONS)


if __name__ == "__main__":
    unittest.main()
