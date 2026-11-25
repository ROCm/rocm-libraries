# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/import_probe.py and its child: the stand-in arguments and engine, which
entry points are called and how, which modules are probed, the leads, the report, the checks wiring, and the probe end
to end on a small repo whose PR moves a lowering function into a library bridge that imports rocke_library_engine
before it chooses a backend (as #11111's first round did, engine_bridge.py:596), adds a module that imports it at its
top level, a script and a test. The repo's tree holds a real rocke_library_engine.py on the import path: the probe
still reads it as not built. Temp git repos, no network."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import import_probe as ip  # noqa: E402
import import_probe_child as child  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestChild(unittest.TestCase):
    def test_a_stand_in_answers_everything_and_equals_nothing(self):
        s = child.StandIn()
        self.assertIsInstance(s.tile.m(3)[0], child.StandIn)
        self.assertFalse(s == -1)
        self.assertFalse(s < 2)
        self.assertEqual((int(s), list(s), len(s), str(s)), (1, [], 0, "stand-in"))
        with self.assertRaises(AttributeError):
            s.__dataclass_fields__  # noqa: B018 (dunders stay missing: a dataclass check says no)

    def test_the_cheapest_entry_point_first_and_its_arguments(self):
        mod = type(sys)("m")
        exec(
            textwrap.dedent(
                """\
            def lower(spec, *, arch, kind="16c", backend=None): pass
            def resolve(backend=None): pass
            def _private(backend=None): pass
            def other(x): pass
            """
            ),
            mod.__dict__,
        )
        self.assertEqual(
            [n for _, _, n in child.entry_points(mod)], ["resolve", "lower"]
        )
        args, kwargs = child.arguments(mod.lower)
        self.assertEqual((len(args), sorted(kwargs)), (1, ["arch", "backend"]))
        self.assertIsInstance(args[0], child.StandIn)
        self.assertEqual((kwargs["arch"], kwargs["backend"]), ("gfx950", "python"))

    def test_the_engines_built_and_not_built(self):
        eng = child.Engines(["rocke_engine"], "/nowhere")
        spec = eng.find_spec("rocke_engine")
        self.assertEqual(spec.origin, "rkmon stand-in")
        with self.assertRaises(ModuleNotFoundError) as cm:
            eng.find_spec("rocke_library_engine")
        self.assertEqual(cm.exception.name, "rocke_library_engine")
        self.assertIsNone(eng.find_spec("rocke.core"))
        self.assertEqual(
            [(x["module"], x["built"]) for x in eng.log],
            [("rocke_engine", True), ("rocke_library_engine", False)],
        )
        try:
            try:
                raise cm.exception
            except ImportError as e:
                raise RuntimeError(
                    "the C++ engine extension 'rocke_library_engine' is not importable"
                ) from e
        except RuntimeError as wrapped:
            got = child.describe(wrapped, eng)
        self.assertEqual(
            (got["type"], got["missing"], got["stand_in"]),
            ("RuntimeError", "rocke_library_engine", False),
        )


class TestParent(unittest.TestCase):
    def test_the_modules_probed_and_scripts(self):
        cs = [
            {
                "status": "R",
                "old": "rk/platform/python/rocke/a.py",
                "new": "rk/library/kernels/a.py",
            },
            {"status": "A", "old": None, "new": "rk/library/tests/test_a.py"},
            {"status": "D", "old": "rk/platform/python/rocke/gone.py", "new": None},
            {
                "status": "M",
                "old": "rk/platform/tools/x.py",
                "new": "rk/platform/tools/x.py",
            },
        ]
        heads, bases = ip.side_modules(cs, "rk")
        self.assertEqual([h["module"] for h in heads], ["kernels.a"])
        self.assertEqual([b["module"] for b in bases], ["rocke.a", "rocke.gone"])
        self.assertTrue(ip.is_script("import sys\n\ndef main():\n    pass\n\nmain()\n"))
        self.assertFalse(
            ip.is_script(
                "X = compute()\nregister('x')\nif __name__ == '__main__':\n    main()\n"
            )
        )

    def test_a_lead_names_both_sides(self):
        head_e = {
            "name": "lower",
            "line": 5,
            "ok": False,
            "type": "BackendError",
            "msg": "not importable",
            "missing": "rocke_library_engine",
            "stand_in": False,
            "frames": [
                ["library/k/bridge.py", 9, "lower"],
                ["platform/python/rocke/b.py", 20, "import_engine"],
            ],
            "engines": [
                {
                    "module": "rocke_library_engine",
                    "built": False,
                    "site": [
                        ["library/k/bridge.py", 9, "lower"],
                        ["platform/python/rocke/b.py", 20, "import_engine"],
                    ],
                }
            ],
        }
        base_e = {
            "name": "lower",
            "line": 30,
            "ok": True,
            "missing": None,
            "stand_in": False,
            "frames": [],
            "engines": [
                {
                    "module": "rocke_engine",
                    "built": True,
                    "site": [["platform/python/rocke/b.py", 33, "lower"]],
                }
            ],
        }
        heads = [
            {
                "module": "kernels.k.bridge",
                "path": "rk/library/k/bridge.py",
                "change": {"status": "A", "old": None, "new": "rk/library/k/bridge.py"},
            }
        ]
        b_py = "rk/platform/python/rocke/b.py"
        bases = [
            {
                "module": "rocke.b",
                "path": b_py,
                "change": {"status": "M", "old": b_py, "new": b_py},
            }
        ]
        H = {
            "kernels.k.bridge": {
                "module": "kernels.k.bridge",
                "import": {"ok": True},
                "engines": [],
                "entries": [head_e],
                "names": [["lower", 5, 1], ["other", 50, 1]],
            }
        }
        B = {
            "rocke.b": {
                "module": "rocke.b",
                "import": {"ok": True},
                "engines": [],
                "entries": [base_e],
                "names": [["lower", 30, 1]],
            }
        }
        leads = ip.compare(heads, H, bases, B)
        self.assertEqual(len(leads), 1)
        self.assertEqual(
            ip.describe(leads[0], "rk"),
            "kernels.k.bridge.lower (library/k/bridge.py:5, the cheapest of 2 with a backend parameter): "
            'backend="python" imports rocke_library_engine (not built) at library/k/bridge.py:9, then '
            "raises BackendError: rocke_library_engine is not built; at the merge base rocke.b.lower "
            "imports rocke_engine at platform/python/rocke/b.py:33, then returns",
        )
        res = {
            "error": None,
            "heads": [
                {
                    "module": "kernels.k.bridge",
                    "path": "rk/library/k/bridge.py",
                    "status": "A",
                }
            ],
            "bases": [{}],
            "leads": leads,
        }
        self.assertEqual(
            ip.verdict(res),
            (
                "fail",
                "1 lead: bridge.lower needs rocke_library_engine with "
                'backend="python" (bridge.py:9); at the base it imports '
                "rocke_engine first (b.py:33)",
            ),
        )
        # The base failing the same way is no lead.
        B["rocke.b"]["entries"] = [
            dict(base_e, ok=False, missing="rocke_library_engine")
        ]
        self.assertEqual(ip.compare(heads, H, bases, B), [])
        self.assertEqual(
            ip.verdict({"error": None, "heads": [], "bases": [], "leads": []})[1],
            "nothing to probe: the PR changes no Python module under library/ or platform/python/",
        )
        # A probe cut short reads NOT RUN, never clean.
        cut = {
            "error": None,
            "heads": [{}],
            "bases": [],
            "leads": [],
            "head_ok": 1,
            "head_ran": 1,
            "head_stop": "the time budget ran out; not reached: kernels.q",
        }
        self.assertEqual(
            ip.verdict(cut),
            (
                "skip",
                "NOT RUN in part (head: the time budget ran out; not reached: "
                "kernels.q)",
            ),
        )
        self.assertEqual(
            ip.render({"error": "x"}, "rk"), ["NOT RUN: x", "IMPORTS skip NOT RUN: x"]
        )


class TestChecksWiring(unittest.TestCase):
    PLAN = {
        "families": [],
        "tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_every_review_runs_it_at_the_head_with_the_checks_python_and_a_gate_does_not(
        self,
    ):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f"sec '{review.IMPORTS_SECTION}'", 1)[1].split("\nsec ", 1)[
            0
        ]
        self.assertIn(
            f"nice -n 10 timeout {review.IMPORTS_BUDGET_S + 120} python -I -B $M/tools/import_probe.py --repo "
            f"$R0 --head {'a' * 40} --out $SO/imports --rocke-path $K --base $BREF --python \"$(command -v python)\" "
            f"--budget {review.IMPORTS_BUDGET_S}",
            sec,
        )
        self.assertIn(
            '\nprr imports "$OUT/imports"\n', sec
        )  # the results copied back, as data
        self.assertIn("st imports ", sec)
        self.assertIn("IMPORTS skip no result", sec)
        self.assertIn("imports", review.LEAD_SECTIONS)
        self.assertGreater(
            script.index(review.IMPORTS_SECTION),
            script.rindex('\nif [ "$MERGED" = 1 ]; then\n'),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            cfg, 7, "a" * 40, "first-round", self.PLAN, None, mode="gate"
        )
        self.assertNotIn("import_probe.py", gate)


BACKEND_BASE = """\
import importlib


def import_engine(module="rocke_engine"):
    return importlib.import_module(module)


def resolve_backend(backend=None):
    return backend or "python"


def lower_x(spec, *, arch, backend=None):
    eng = import_engine()
    if spec.split == -1:
        raise ValueError("auto split")
    return "python" if backend == "python" else eng.lower(spec)
"""

BRIDGE = """\
from rocke.backend import import_engine


def lower_x(spec, *, arch, backend=None):
    eng = import_engine("rocke_library_engine")
    if spec.split == -1:
        raise ValueError("auto split")
    return "python" if backend == "python" else eng.lower(spec)
"""


@unittest.skipUnless(sys.version_info >= (3, 11), "the child runs with python -P")
class TestProbe(unittest.TestCase):
    """rocKE at prov/rk: the base defines lower_x in rocke.backend (rocke_engine first, then the backend); the PR moves
    it to kernels.x.bridge with rocke_library_engine first."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        self.write(
            {
                "prov/rk/platform/python/rocke/__init__.py": "",
                "prov/rk/platform/python/rocke/backend.py": BACKEND_BASE,
                # A real module of the name, on the import path: the probe never imports it.
                "prov/rk/library/rocke_library_engine.py": "def lower(spec):\n    return 'cpp'\n",
            }
        )
        g(r, "init", "-q", "-b", "develop")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        self.write(
            {
                "prov/rk/platform/python/rocke/backend.py": BACKEND_BASE.split(
                    "\n\ndef lower_x"
                )[0]
                + "\n",
                "prov/rk/library/kernels/x/bridge.py": BRIDGE,
                "prov/rk/library/kernels/y.py": "import rocke_library_engine\n\n\n"
                "def g(*, backend=None):\n    return 1\n",
                "prov/rk/library/kernels/z.py": "def ok(*, backend=None):\n    return 1\n",
                "prov/rk/library/kernels/tool.py": "def main():\n    print('ran')\n\n\nmain()\n",
                "prov/rk/library/tests/test_x.py": "def test_x():\n    pass\n",
            }
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")
        self.tmp = pathlib.Path(self.d.name) / "tmp"
        self.tmp.mkdir()

    def write(self, files: dict) -> None:
        for path, text in files.items():
            p = self.repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def test_the_moved_function_needs_the_library_engine_at_the_head_only(self):
        with mock.patch.dict(os.environ, {"TMPDIR": str(self.tmp)}):
            res = ip.run(
                str(self.repo),
                self.head,
                pathlib.Path(self.d.name) / "out",
                K="prov/rk",
                base="develop",
                python=sys.executable,
            )
        self.assertIsNone(res["error"], res)
        self.assertEqual(list(self.tmp.iterdir()), [])  # the scratch trees are gone
        self.assertEqual(g(self.repo, "status", "--porcelain"), "")
        self.assertEqual(res["scripts"], ["library/kernels/tool.py"])
        self.assertEqual(
            sorted(res["head_records"]),
            ["kernels.x.bridge", "kernels.y", "kernels.z", "rocke.backend"],
        )
        self.assertEqual(
            [(x["kind"], x["module"]) for x in res["leads"]],
            [("entry", "kernels.x.bridge"), ("import", "kernels.y")],
        )
        e, be = res["leads"][0]["entry"], res["leads"][0]["base_entry"]
        self.assertEqual((e["name"], e["missing"]), ("lower_x", "rocke_library_engine"))
        self.assertEqual(
            ip.frame(e["engines"][0]["site"], "lower_x"),
            ["library/kernels/x/bridge.py", 5, "lower_x"],
        )
        self.assertEqual(
            (res["leads"][0]["base_module"], be["ok"]), ("rocke.backend", True)
        )
        self.assertEqual(
            [(x["module"], x["built"]) for x in be["engines"]], [("rocke_engine", True)]
        )
        self.assertEqual(
            ip.frame(be["engines"][0]["site"], "lower_x")[:2],
            ["platform/python/rocke/backend.py", 13],
        )
        self.assertEqual(res["leads"][1]["head"]["missing"], "rocke_library_engine")
        lines = ip.render(res, "prov/rk")
        self.assertEqual(
            lines[-1],
            "IMPORTS fail 2 leads: bridge.lower_x needs rocke_library_engine with "
            'backend="python" (bridge.py:5); at the base it imports rocke_engine first '
            "(backend.py:13); +1 more",
        )
        self.assertIn('   kernels.z.ok: backend="python" returns', lines)
        self.assertTrue((pathlib.Path(self.d.name) / "out" / "head_probe.log").exists())

    def test_the_cli_prints_the_section_and_writes_the_json(self):
        out = pathlib.Path(self.d.name) / "cli"
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOLS / "import_probe.py"),
                "--repo",
                str(self.repo),
                "--head",
                self.head,
                "--out",
                str(out),
                "--rocke-path",
                "prov/rk",
                "--base",
                "develop",
                "--python",
                sys.executable,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "TMPDIR": str(self.tmp)},
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(p.stdout.splitlines()[-1].startswith("IMPORTS fail 2 leads"))
        self.assertEqual(
            len(json.loads((out / "import_probe.json").read_text())["leads"]), 2
        )


if __name__ == "__main__":
    unittest.main()
