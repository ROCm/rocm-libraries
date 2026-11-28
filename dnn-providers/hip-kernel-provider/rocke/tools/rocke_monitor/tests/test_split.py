# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk review-split N`: a review too large for one verifier cut into parts by area, files that change together kept
together, data summarized, a file over the limit cut into slices, the checklist quoted from the team file, and the
dispatch plan in waves under the subagent cap."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import review, split  # noqa: E402
from rkmon_lib.hybrid import size  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

R = "prov/rocke"

TEAM = """### What to Check

**8. What peer reviewers catch.**
- **New test assertions:** one code change that makes each fail. Flag tautologies,
  NaN-blind tolerances.
- **New spec or request fields:**
  - the validator rejects wrong types;
  - the field is in the kernel name.
- **New guards, defaults and `None` handling:** quote the C++ and the Python line.
- **Launch paths:** which stream each allocation uses.
- Three more check sections are leads:
  - **"CTest labels and tiers":** a CTest entry that loses a tier;
"""

QUALITY = """### The non-negotiables

1. **Single responsibility.** A unit does one job.

### What to Check

1. **Single responsibility.** What does each new or changed unit do? A bool flag
   that picks a behaviour is a second job.
2. **DRY.** For each new helper, search the tree first:
   - `git grep` by name and synonyms;
   - the shared homes.

   Name the existing home and say which copy should stay.
3. **Primitives and layers.** Which layer does each change sit in?
8. **Overlapping work in other open PRs.** Each lead read at both heads.

### Output Format
"""


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(repo, files, msg):
    for f, text in files.items():
        p = pathlib.Path(repo, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    g(repo, "add", "-A")
    g(repo, "commit", "-qm", msg)
    return g(repo, "rev-parse", "HEAD")


def code(n, word="x"):
    """n lines of code with names (never a table)."""
    return "".join(f"{word}_{i} = compute({word}, value_{i})\n" for i in range(n))


def fake(path, a, cls=None, text="v = f(a, b)"):
    """A parsed file of `a` added lines, as hybrid.size.parse_patch gives it."""
    lines = [f"+{text}"] * a
    f = {
        "p": path,
        "op": path,
        "st": "A",
        "sim": None,
        "ob": "",
        "nb": "1",
        "bin": False,
        "a": a,
        "d": 0,
        "hunks": [
            {"h": "", "os": 0, "ol": 0, "ns": 1, "nl": a, "lines": lines, "n": a}
        ],
    }
    f["cls"] = cls or size.classify(path)
    return f


class PartSizeTest(unittest.TestCase):
    def test_a_part_holds_up_to_8000_changed_lines_and_the_docs_say_so(self):
        """`rk review-split N` cuts at features.review_automation.large_prs.split_part_lines (8000 by default) unless
        --max-lines says otherwise, and the docs give that number."""
        import rkmon

        root = pathlib.Path(__file__).resolve().parent.parent
        n = split.part_lines({})
        self.assertEqual(n, 8000)
        asked = []

        def run(cfg, st, n, out, rnd, max_lines, *rest, **kw):
            asked.append(max_lines)
            return [], 0

        cfgs = iter([{}, {}, {"split_part_lines": 5000}])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            rkmon, "load_config", lambda: next(cfgs)
        ), mock.patch.object(
            rkmon, "Store", lambda: Store(pathlib.Path(d))
        ), mock.patch.object(
            split, "default_out", lambda cfg, st, n, rnd: (pathlib.Path(d), "fr")
        ), mock.patch.object(
            split, "run", run
        ), mock.patch(
            "builtins.print"
        ):
            self.assertEqual(rkmon.main(["review-split", "7"]), 0)
            self.assertEqual(rkmon.main(["review-split", "7", "--max-lines", "500"]), 0)
            self.assertEqual(
                rkmon.main(["review-split", "7"]), 0
            )  # the deploy settings' part size
        self.assertEqual(asked, [n, 500, 5000])
        from rkmon_lib import file_types, settings_file

        self.assertEqual(
            settings_file.schema_at(
                file_types.DEPLOY.load(),
                "features.review_automation.large_prs.split_part_lines",
            )["default"],
            n,
        )
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"each filled up to\n  {n} changed lines", readme)
        self.assertIn(f"(default {n})", rkmon.__doc__)
        self.assertIn(
            f"({n} by default)",
            (root / "docs" / "review_pipeline.md").read_text(encoding="utf-8"),
        )


class AreasTest(unittest.TestCase):
    def test_areas_follow_the_tree(self):
        cases = {
            f"{R}/library/kernels/gfx950/attn.py": ("library/gfx950", "kernels"),
            f"{R}/library/builders/gfx950/attn_build.py": ("library/gfx950", "kernels"),
            f"{R}/library/tests/test_attn.py": ("library/tests", "tests"),
            f"{R}/library/dispatch/route.py": ("library/dispatch", "library"),
            f"{R}/platform/cpp/bindings/py.cpp": ("platform/cpp/bindings", "bindings"),
            f"{R}/platform/cpp/core/ir.cpp": ("platform/cpp/core", "platform_cpp"),
            f"{R}/platform/python/rocke/runtime/launch.py": (
                "platform/python/runtime",
                "runtime",
            ),
            f"{R}/platform/python/rocke/core/ir.py": (
                "platform/python/core",
                "platform_python",
            ),
            f"{R}/platform/tests/core/test_ir.py": ("platform/tests/core", "tests"),
            f"{R}/platform/rust/codeobj/src/lib.rs": ("platform/rust/codeobj", "rust"),
            f"{R}/platform/dsl_docs/reference/env.md": ("docs", "docs"),
            f"{R}/platform/CMakeLists.txt": ("build", "build"),
            "prov/x_test_categories_integration.yaml": ("build", "build"),
            ".github/workflows/ci.yml": ("build", "build"),
            "projects/lib/library/src/backends/b.cpp": (
                "projects/lib/library/src",
                "code",
            ),
            "projects/lib/test/00_unit/b_test.cpp": (
                "projects/lib/test/00_unit",
                "tests",
            ),
            "projects/lib/docs/guide.rst": ("projects/lib/docs", "docs"),
        }
        for path, want in cases.items():
            self.assertEqual(split.area_of(path, R), want, path)

    def test_data_is_summarized_not_read(self):
        self.assertEqual(
            split.reclass(
                fake(
                    f"{R}/platform/tests/core/expected_ir.json", 2000, text='  "op": 1,'
                )
            ),
            "data",
        )  # a large JSON outside a golden folder: generated
        self.assertEqual(
            split.reclass(
                fake(f"{R}/platform/tests/core/runs.jsonl", 5, text='{"a": 1}')
            ),
            "data",
        )
        self.assertEqual(
            split.reclass(
                fake("p/select/model.txt", 3, cls="code", text="v3;" + "x," * 400)
            ),
            "data",
        )
        self.assertEqual(
            split.reclass(fake("p/tables.cpp", 60, cls="code", text="{1, 2, 3, 4},")),
            "data",
        )
        self.assertEqual(split.reclass(fake("p/code.cpp", 60, cls="code")), "code")
        self.assertEqual(
            split.reclass(fake(f"{R}/library/tests/golden/a_ir.json", 9)), "golden"
        )  # size's class

    def test_small_hand_edited_configs_are_read(self):
        """A config JSON or YAML of a few changed lines is read, even when its lines are all literals; one marked
        generated near its top, or one past CONFIG_MAX_LINES, is summarized."""
        cfg = fake(f"{R}/library/configs/tuning.json", 40, text='  "block_m": 128,')
        self.assertEqual(split.reclass(cfg), "code")
        self.assertEqual(
            split.reclass(fake(f"{R}/platform/tests/core/fixture.json", 10)), "test"
        )
        self.assertEqual(
            split.reclass(
                fake(f"{R}/library/configs/arches.yaml", 80, text="  - gfx950: 128")
            ),
            "code",
        )
        marked = fake(f"{R}/library/configs/table.json", 20, text='  "x": 1,')
        marked["hunks"][0]["lines"][
            0
        ] = '+{"_note": "generated by tools/make_table.py",'
        self.assertEqual(split.reclass(marked), "data")
        big = fake(
            f"{R}/library/configs/tuning.json",
            split.CONFIG_MAX_LINES + 1,
            text='  "block_m": 128,',
        )
        self.assertEqual(split.reclass(big), "data")
        # a large YAML is no JSON: read unless its lines are a table
        self.assertEqual(
            split.reclass(
                fake(f"{R}/library/configs/big.yaml", 300, text="  - gfx950: 128")
            ),
            "data",
        )
        self.assertEqual(
            split.reclass(
                fake(f"{R}/library/configs/big.yaml", 300, text="key_a: value_b(other)")
            ),
            "code",
        )


class LinksTest(unittest.TestCase):
    def test_files_that_change_together(self):
        files = [
            fake(f"{R}/platform/cpp/core/lower_mma.cpp", 10),
            fake(f"{R}/platform/python/rocke/core/lower_mma.py", 10),
            fake(f"{R}/platform/cpp/include/rocke/conv_spec.h", 5),
            fake(f"{R}/platform/cpp/core/conv_spec.cpp", 5),
            fake(f"{R}/library/kernels/gfx950/attention_bwd.py", 10),
            fake(f"{R}/library/kernels/gfx942/attention_bwd.py", 10),
            fake(f"{R}/library/tests/test_attention_bwd_gfx950_numeric.py", 10),
            fake(
                f"{R}/library/tests/test_misc.py",
                1,
                text="from rocke.core.lower_mma import emit",
            ),
            fake(
                f"{R}/platform/tests/CMakeLists.txt", 1, text="add_test(test_misc.py)"
            ),
            fake("projects/lib/src/lower_mma.cpp", 10),
        ]
        got = {(a, b): why for a, b, why in split.links(files, R)}

        def linked(a, b):
            return (a, b) in got or (b, a) in got

        self.assertTrue(
            linked(
                f"{R}/platform/cpp/core/lower_mma.cpp",
                f"{R}/platform/python/rocke/core/lower_mma.py",
            )
        )
        self.assertTrue(
            linked(
                f"{R}/platform/cpp/include/rocke/conv_spec.h",
                f"{R}/platform/cpp/core/conv_spec.cpp",
            )
        )
        self.assertTrue(
            linked(
                f"{R}/library/tests/test_attention_bwd_gfx950_numeric.py",
                f"{R}/library/kernels/gfx950/attention_bwd.py",
            )
        )
        self.assertFalse(
            linked(
                f"{R}/library/tests/test_attention_bwd_gfx950_numeric.py",
                f"{R}/library/kernels/gfx942/attention_bwd.py",
            )
        )  # the other family
        self.assertTrue(
            linked(
                f"{R}/library/tests/test_misc.py",
                f"{R}/platform/python/rocke/core/lower_mma.py",
            )
        )
        self.assertTrue(
            linked(
                f"{R}/platform/tests/CMakeLists.txt", f"{R}/library/tests/test_misc.py"
            )
        )
        self.assertFalse(
            any("projects/lib/src/lower_mma.cpp" in k for k in got)
        )  # another project: no twin


class PartitionTest(unittest.TestCase):
    def test_parts_hold_at_most_the_limit_and_keep_twins_together(self):
        files = [fake(f"{R}/library/kernels/gfx950/k{i}.py", 400) for i in range(10)]
        files += [
            fake(f"{R}/platform/cpp/core/lower_mma.cpp", 300),
            fake(f"{R}/platform/python/rocke/core/lower_mma.py", 300),
        ]
        files += [fake(f"{R}/platform/python/rocke/dispatch/huge.py", 2500)]
        files += [
            fake(f"{R}/docs/a.md", 20),
            fake(f"{R}/library/tests/golden/g.json", 5000),
        ]
        plan = split.partition(files, R, 1000)
        parts = plan["parts"]
        self.assertTrue(
            all(p["lines"] <= 1000 for p in parts), [p["lines"] for p in parts]
        )
        where = split.holders(parts)
        self.assertEqual(
            where[f"{R}/platform/cpp/core/lower_mma.cpp"],
            where[f"{R}/platform/python/rocke/core/lower_mma.py"],
        )
        self.assertEqual(
            len(where[f"{R}/platform/python/rocke/dispatch/huge.py"]), 3
        )  # cut into three slices
        self.assertNotIn(
            f"{R}/library/tests/golden/g.json", where
        )  # summarized, never in a part
        self.assertEqual(sum(p["lines"] for p in parts), 4000 + 600 + 2500 + 20)
        # As few parts as these files allow: each slice of huge.py fills most of a part, two 400-line kernels fill
        # another; the docs file shares the first part with two kernels, and its areas are both.
        self.assertEqual(len(parts), 9)
        self.assertEqual(parts[0]["areas"], ["library/gfx950", "docs"])
        self.assertEqual(parts[0]["kinds"], ["kernels", "docs"])

    def test_parts_fill_to_the_limit(self):
        """A PR of 80,000 changed lines over many areas takes about ten parts of 8,000 lines, as many as its lines need
        or one more: a part takes several areas, a slice of a file over the limit shares its part with other files,
        and files that change together stay together."""
        files = [
            fake(f"{R}/library/kernels/gfx942/k{i:02d}.py", 480) for i in range(25)
        ]
        files += [
            fake(f"{R}/library/kernels/gfx950/k{i:02d}.py", 500) for i in range(30)
        ]
        files += [fake(f"{R}/library/dispatch/routes.py", 20000)]
        files += [
            fake(f"{R}/platform/python/rocke/core/emit{i:02d}.py", 250)
            for i in range(26)
        ]
        trios = [
            (
                f"{R}/platform/cpp/core/lower_op{i:02d}.cpp",
                f"{R}/platform/python/rocke/core/lower_op{i:02d}.py",
                f"{R}/platform/tests/core/test_lower_op{i:02d}.py",
            )
            for i in range(14)
        ]  # a C++ mirror, its Python twin and the test named after them
        files += [fake(p, n) for trio in trios for p, n in zip(trio, (500, 250, 200))]
        files += [
            fake(f"{R}/platform/tests/core/test_misc{i:02d}.py", 200) for i in range(46)
        ]
        files += [fake(f"{R}/docs/guide{i}.md", 300) for i in range(10)]
        files += [fake(f"{R}/platform/cmake/t{i}.cmake", 100) for i in range(10)]
        total = sum(split.file_lines(f) for f in files)
        self.assertEqual(total, 80000)
        parts = split.partition(files, R, 8000)["parts"]
        self.assertEqual(split.parts_for(total, 8000), 10)
        self.assertLessEqual(len(parts), split.parts_for(total, 8000) + 1)
        self.assertTrue(all(p["lines"] <= 8000 for p in parts))
        self.assertGreaterEqual(
            sum(p["lines"] >= 7800 for p in parts), 10
        )  # full, but for the last
        where = split.holders(parts)
        self.assertEqual(len(where[f"{R}/library/dispatch/routes.py"]), 3)
        for k in where[f"{R}/library/dispatch/routes.py"]:
            self.assertGreater(len(parts[k - 1]["files"]), 1)  # a slice shares its part
        for trio in trios:
            self.assertEqual(len({tuple(where[p]) for p in trio}), 1, trio)
        self.assertTrue(any(len(p["areas"]) > 2 for p in parts))
        # The estimate counts by the same rule: about 11 agents, the parts and the cross pass.
        self.assertEqual(split.estimate({}, None, total), "about 11 agents")

    def test_a_file_over_the_limit_is_cut_at_hunks(self):
        def hunks(*sizes):
            f = fake("p/big.py", 0)
            f["hunks"] = [
                {
                    "h": "",
                    "os": 1,
                    "ol": 0,
                    "ns": i * 2000 + 1,
                    "nl": n,
                    "lines": ["+x"] * n,
                    "n": n,
                }
                for i, n in enumerate(sizes)
            ]
            f["a"] = sum(sizes)
            return f

        cs = split.chunks(
            hunks(900, 900), 1000
        )  # the even cut falls on the hunk boundary
        self.assertEqual([c["lines"] for c in cs], [900, 900])
        self.assertEqual(
            [cs[0]["new"], cs[1]["new"], cs[0]["old"]], [(1, 900), (2001, 2900), None]
        )
        cs = split.chunks(
            hunks(950, 100, 950), 1000
        )  # the boundary at 950 would need a third slice: cut evenly
        self.assertEqual([c["lines"] for c in cs], [1000, 1000])
        cs = split.chunks(hunks(2500), 1000)  # one hunk: even slices by line
        self.assertEqual([c["lines"] for c in cs], [834, 833, 833])
        self.assertEqual(cs[1]["new"], (835, 1667))

    def test_waves_stay_under_the_cap(self):
        parts = [{"k": k} for k in range(1, 15)]
        ws = split.waves(parts, 6)
        self.assertEqual([len(w) for w in ws], [6, 6, 2])  # a verifier per part
        self.assertEqual([p["k"] for w in ws for p in w], list(range(1, 15)))


class ChecklistTest(unittest.TestCase):
    def test_bullets_are_quoted_whole(self):
        lines = split.checklist(TEAM, ["tests", "kernels"])
        text = "\n".join(lines)
        self.assertIn(
            "- **New test assertions:** one code change that makes each fail. Flag tautologies,\n  NaN-blind",
            text,
        )
        self.assertIn(
            "- **New spec or request fields:**\n  - the validator rejects wrong types;\n  - the field is in",
            text,
        )
        self.assertIn('- **"CTest labels and tiers":** a CTest entry', text)
        self.assertIn(
            "- !! not in the verifier team file: **Pointers the kernel declares**", text
        )

    def test_without_the_file_the_leads_are_named(self):
        self.assertEqual(
            split.checklist(None, ["docs"])[0],
            "- **Changed docs, comments and PR-body claims** (the verifier team file, What to Check)",
        )

    def test_the_code_quality_checks_are_quoted_whole_from_what_to_check(self):
        text = "\n".join(split.quality_checklist(QUALITY))
        self.assertTrue(
            text.startswith(
                "1. **Single responsibility.** What does each new or changed unit do? A bool flag\n"
                "   that picks a behaviour is a second job.\n2. **DRY.**"
            ),
            text,
        )  # What to Check's item, not the non-negotiables'
        self.assertIn(
            "2. **DRY.** For each new helper, search the tree first:\n   - `git grep` by name and synonyms;\n"
            "   - the shared homes.\n\n   Name the existing home and say which copy should stay.\n"
            "3. **Primitives and layers.**",
            text,
        )
        self.assertIn(
            "- !! not in the code quality reviewer's team file: **Interfaces from**",
            text,
        )
        self.assertNotIn("Overlapping work", text)  # the cross pass's
        self.assertNotIn("### Output Format", text)
        self.assertEqual(
            split.quality_checklist(None)[1],
            "- **DRY.** (the code quality reviewer's team file, What to Check)",
        )

    def test_the_leads_are_the_code_quality_reviewers(self):
        """Each lead a part quotes is an item of What to Check in the package's code quality reviewer's team file."""
        from rkmon_lib import paths

        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / "agentic" / paths.QUALITY_FILE).read_text(encoding="utf-8")
        self.assertFalse([x for x in split.quality_checklist(text) if "!! not in" in x])


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        g(self.repo, "init", "-q")
        base = commit(
            self.repo, {f"{R}/platform/CMakeLists.txt": "project(x)\n"}, "base"
        )
        g(self.repo, "update-ref", "refs/remotes/origin/develop", base)
        files = {
            f"{R}/library/kernels/gfx950/k{i}.py": code(300, f"k{i}") for i in range(6)
        }
        files[f"{R}/library/tests/test_k1.py"] = code(50, "t")
        files[f"{R}/platform/cpp/core/lower_mma.cpp"] = code(200, "c")
        files[f"{R}/platform/python/rocke/core/lower_mma.py"] = code(200, "p")
        files[f"{R}/platform/cpp/include/rocke/abi.h"] = code(5, "abi")
        files[f"{R}/platform/tests/golden/g.json"] = '{"a": 1}\n' * 400
        files[f"{R}/library/tests/sdpa/expected_out.json"] = (
            '  "o": 1,\n' * 1500
        )  # a large golden, no golden folder
        files[f"{R}/library/configs/tuning.json"] = (
            '  "block_m": 128,\n' * 20
        )  # a small hand-edited config
        files[f"{R}/docs/guide.md"] = "# Guide\n" * 30
        self.head = commit(self.repo, files, "head")
        self.cfg = {
            "rocke_path": R,
            "hot_spots": {"ABI": f"{R}/platform/cpp/include/rocke/abi.h"},
            "paths": {
                "repo": str(self.repo),
                "root": str(root),
                "workers": str(root / "workers"),
            },
        }
        self.out = root / "tmp" / "v7fr"

    def tearDown(self):
        self.tmp.cleanup()

    def test_parts_cross_and_plan(self):
        st = empty_state()
        lines, code_ = split.run(
            self.cfg, st, 7, self.out, "fr", 800, 3, team_file=None, head=self.head
        )
        self.assertEqual(code_, 0)
        self.assertTrue(lines[0].startswith("SPLIT #7 (fr): "), lines[0])
        plan = json.loads((self.out / "parts" / "plan.json").read_text())
        self.assertTrue(all(p["lines"] <= 800 for p in plan["parts"]))
        self.assertGreaterEqual(len(plan["parts"]), 3)
        self.assertEqual(
            sum(p["lines"] for p in plan["parts"]), 1800 + 50 + 400 + 5 + 30 + 20
        )
        # The small config is read in a part; the large golden is summarized, never in a part.
        self.assertTrue(
            any(f"{R}/library/configs/tuning.json" in p["files"] for p in plan["parts"])
        )
        self.assertFalse(
            any(
                f"{R}/library/tests/sdpa/expected_out.json" in p["files"]
                for p in plan["parts"]
            )
        )
        twins = [
            p["k"]
            for p in plan["parts"]
            if f"{R}/platform/cpp/core/lower_mma.cpp" in p["files"]
        ]
        self.assertEqual(
            twins,
            [
                p["k"]
                for p in plan["parts"]
                if f"{R}/platform/python/rocke/core/lower_mma.py" in p["files"]
            ],
        )
        # Waves: a verifier per part, which runs its part's code quality checks too (no lane beside it), at most 3.
        self.assertEqual([len(w) for w in plan["waves"]], [3, 1])
        self.assertFalse([p for p in plan["parts"] if "lane" in p])
        self.assertTrue(any(x.startswith("  wave 2 (1 agent)") for x in lines), lines)
        self.assertTrue(
            any(x.startswith("  then the cross pass, once every part") for x in lines)
        )
        self.assertIn(
            "cost: about 5 agents: 4 part verifiers in 2 waves, then the cross pass",
            lines,
        )  # the parts and the cross pass; no spend card here, so no dollars
        self.assertNotIn("lane", "\n".join(lines))
        self.assertTrue(lines[-1].startswith("!! the verifier team file was not read"))
        part = (self.out / "parts" / "part1.md").read_text()
        self.assertIn("## What to check", part)
        self.assertIn("(the verifier team file, What to Check)", part)
        self.assertIn(
            "## Code quality\n\nFrom the code quality reviewer's team file (not read: it lies beside "
            "paths.verifier_team_file), What to Check: run these on this part's files and write their items into "
            "`part1_findings.md`.",
            part,
        )
        self.assertIn(
            "- **Single responsibility.** (the code quality reviewer's team file, What to Check)",
            part,
        )
        files = (self.out / "parts" / "part1.files").read_text().split()
        self.assertTrue(files and all(f.startswith(R) for f in files))
        cross = (self.out / "parts" / "cross.md").read_text()
        self.assertIn(f"`{R}/platform/cpp/include/rocke/abi.h`: 5 lines, part", cross)
        self.assertIn(
            f"- ABI `{R}/platform/cpp/include/rocke/abi.h`: 1 files, 5 lines; part",
            cross,
        )
        self.assertIn(f"`{R}/platform/tests/golden/`: 1 golden, 400 lines", cross)
        self.assertIn(f"`{R}/library/tests/sdpa/`: 1 data, 1500 lines", cross)
        self.assertIn("- not in the checks yet: run `rk clash 7`", cross)

    def test_each_agent_gets_a_description_and_a_brief_that_name_its_review(self):
        """The plan gives each part verifier and the cross pass the Agent description and the brief to dispatch it
        with. Each names the PR and the round where the orchestrator reads them (orchestrator.brief_pr), so a ready
        review whose parts are running is not handed back to the session."""
        from rkmon_lib import orchestrator

        lines, _ = split.run(
            self.cfg,
            empty_state(),
            7,
            self.out,
            "fr",
            800,
            3,
            team_file=None,
            head=self.head,
        )
        calls = [x for x in lines if x.startswith("    part")]
        self.assertEqual(len(calls), 4, lines)
        self.assertIn(
            f'    part1: description "PR 7 fr part 1 of 4", brief {self.out / "parts" / "part1.md"}',
            calls,
        )
        (cross,) = [x for x in lines if x.startswith("  then the cross pass")]
        self.assertIn(
            f'description "PR 7 fr cross pass", brief {self.out / "parts" / "cross.md"}',
            cross,
        )
        for text in calls + [cross]:
            self.assertEqual(orchestrator.brief_pr(text), (7, "fr"), text)
        # The description alone, or the brief's path alone, names the review too, in a follow-up as well.
        for rnd in ("fr", "fu2"):
            for k in ((2, 4), ()):
                desc, brief = split.agent_call(
                    7, rnd, self.out.with_name(f"v7{rnd}"), *k
                )
                self.assertEqual(orchestrator.brief_pr(desc), (7, rnd), desc)
                self.assertEqual(orchestrator.brief_pr(str(brief)), (7, rnd), brief)

    def test_the_verifiers_run_at_the_verifier_s_effort(self):
        """The plan dispatches the part verifiers and the cross pass at the effort the verifier's is set to (its own,
        else claude.effort, else the model's default), as the procedure's verifier is.
        """
        from rkmon_lib import config

        for over, effort in (
            ({}, "high"),
            ({"effort": {"verify": "max"}}, "max"),
            (
                {"effort": {"verify": ""}, "claude": {"effort": ""}},
                config.MODEL_DEFAULT,
            ),
        ):
            lines, _ = split.run(
                {**self.cfg, **over},
                empty_state(),
                7,
                self.out,
                "fr",
                800,
                3,
                team_file=None,
                head=self.head,
            )
            (parts,) = [x for x in lines if x.startswith("  part verifiers: ")]
            (cross,) = [x for x in lines if x.startswith("  then the cross pass")]
            self.assertTrue(
                parts.startswith(
                    f"  part verifiers: rocke-review-verifier, effort {effort}; "
                ),
                parts,
            )
            self.assertIn(
                f": rocke-review-verifier, effort {effort}, description", cross
            )

    def test_a_follow_up_skips_parts_with_no_change(self):
        self.out.mkdir(parents=True)
        (self.out / "classify.txt").write_text(
            json.dumps(
                {
                    "kind": "substantive",
                    "old": "a" * 40,
                    "new": self.head,
                    "changed_files": ["platform/cpp/core/lower_mma.cpp"],
                }
            )
        )
        team = pathlib.Path(self.tmp.name) / "team.md"
        team.write_text(TEAM)
        team.with_name("rocke_code_quality_reviewer.md").write_text(QUALITY)
        lines, _ = split.run(
            self.cfg,
            empty_state(),
            7,
            self.out,
            "fu1",
            800,
            20,
            team_file=str(team),
            head=self.head,
        )
        plan = json.loads((self.out / "parts" / "plan.json").read_text())
        changed = [p["k"] for p in plan["parts"] if p["changed"]]
        self.assertEqual([k for w in plan["waves"] for k in w], changed)
        self.assertTrue(any(x.startswith("  no verifier for ") for x in lines), lines)
        self.assertFalse(any("team file was not read" in x for x in lines))

    def test_each_part_runs_its_code_quality_checks_and_the_cost_counts_its_agents(
        self,
    ):
        """No code quality agent beside a part: its verifier runs the checks quoted from the code quality reviewer's
        team file beside the verifier team file. The cost: the parts and the cross pass, each about the spend card's
        mean review."""
        from rkmon_lib import spend

        folder = pathlib.Path(self.tmp.name) / "agentic" / "team_members"
        folder.mkdir(parents=True)
        (folder / "rocke_review_verifier.md").write_text(TEAM)
        (folder / "rocke_code_quality_reviewer.md").write_text(QUALITY)
        state = pathlib.Path(self.tmp.name) / "state"
        state.mkdir()
        days = [(4, 2.5), (1, 5.0), (0, None)]  # (reviews, mean): $3.00 a review
        (state / spend.SUMMARY).write_text(
            json.dumps({"days": [{"reviews": n, "mean": m} for n, m in days]})
        )
        cfg = dict(self.cfg, prices={"claude-x": {"input": 1.0}})
        lines, _ = split.run(
            cfg,
            empty_state(),
            7,
            self.out,
            "fr",
            800,
            3,
            team_file=str(folder / "rocke_review_verifier.md"),
            head=self.head,
            store=Store(state),
        )
        self.assertIn(
            "cost: about 5 agents, about $15: 4 part verifiers in 2 waves, then the cross pass; each agent about "
            "one review's cost (the spend card's mean review)",
            lines,
        )
        self.assertFalse([x for x in lines if x.startswith("!!")], lines)
        part = (self.out / "parts" / "part2.md").read_text()
        self.assertIn(
            f"From the code quality reviewer's team file ({folder / 'rocke_code_quality_reviewer.md'}), What to "
            "Check: run these on this part's files and write their items into `part2_findings.md`.",
            part,
        )
        self.assertIn(
            "\n2. **DRY.** For each new helper, search the tree first:\n   - `git grep` by name and synonyms;\n",
            part,
        )
        self.assertNotIn("Each lead read at both heads", part)  # the cross pass's item
        self.assertNotIn("_quality.md", part)
        # Without the code quality reviewer's file beside the verifier's, the parts name its leads and the output says
        # so.
        (folder / "rocke_code_quality_reviewer.md").unlink()
        lines, _ = split.run(
            cfg,
            empty_state(),
            7,
            self.out,
            "fr",
            800,
            3,
            team_file=str(folder / "rocke_review_verifier.md"),
            head=self.head,
        )
        self.assertEqual(
            lines[-1],
            "!! the code quality reviewer's team file was not read "
            f"({folder / 'rocke_code_quality_reviewer.md'}, beside the verifier team file): the parts name its "
            "checks' leads only",
        )
        self.assertIn("cost: about 5 agents: 4 part verifiers", lines[-2])  # no store

    def test_one_part_is_no_split(self):
        lines, _ = split.run(
            self.cfg, empty_state(), 7, self.out, "fr", 3000, 20, head=self.head
        )
        self.assertTrue(
            any(x.startswith("dispatch: one part, so no split") for x in lines), lines
        )

    def test_the_cross_pass_finds_the_links_past_its_first_300_in_the_plan(self):
        real = split.partition

        def partition(*a):  # the real parts, and 301 more links between the first two
            plan = real(*a)
            one, two = (p["files"][0] for p in plan["parts"][:2])
            plan["cut"] += [(one, two, f"link {i}") for i in range(301)]
            return plan

        with mock.patch.object(split, "partition", partition), mock.patch.object(
            split, "PLAN", "links.json"
        ):
            split.run(
                self.cfg, empty_state(), 7, self.out, "fr", 800, 4, head=self.head
            )
        plan = json.loads((self.out / "parts" / "links.json").read_text())
        cut = len(plan["links_between_parts"])
        self.assertIn(
            f"- … {cut - 300} more (parts/links.json)",
            (self.out / "parts" / "cross.md").read_text(),
        )


class ManualLineTest(unittest.TestCase):
    def test_the_manual_line_names_the_split(self):
        st = empty_state()
        st["prs"] = {"9": {"n": 9, "head": "9" * 40, "size": 50000, "status": "open"}}
        st[review.APPROVED] = [review.approval(9)]  # its in-depth review, approved
        plan = [{"kind": "FIRST_ROUND", "pr": 9, "text": "first-round review of #9"}]
        with tempfile.TemporaryDirectory() as d:
            line = review.manual_lines(
                {"autopilot_max_lines": 30000, "settle_min": 30},
                st,
                Store(pathlib.Path(d)),
                plan,
            )[0]
        self.assertIn("`rk review-start 9`", line)
        self.assertIn("`rk review-split 9`", line)


if __name__ == "__main__":
    unittest.main()
