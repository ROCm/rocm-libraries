# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/tidy_check.py: clang-tidy with the provider's .clang-tidy and clang -Werror over the TUs a PR
touches, feature macros off and on; the checks' "Superbuild tidy" section, which runs it as PR code after its --fetch;
in a slot, where git fetches nothing, its reads of a blobless clone; `rk job-log --errors`.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
TOOL = ROOT / "tools" / "tidy_check.py"

# Fake LLVM tools: clang-tidy reports one error in a.cpp only with the feature macro on; clang++ refuses a GCC-only flag
# (the check drops it and runs again), then reports an unused variable in a.cpp. Each logs its arguments.
FAKE_TIDY = """#!/bin/bash
echo "$*" >> "$(dirname "$0")/tidy.log"
for a in "$@"; do case "$a" in *.cpp) f=$a;; esac; done
case "$f" in */tree/*) case "$*" in *-DROCKE_FEATURE_X=1*) [ "${f##*/}" = a.cpp ] && echo "$f:3:5: error: narrowing conversion [bugprone-narrowing-conversions,-warnings-as-errors]";; esac;; esac
echo "1 warning generated."
exit 0
"""
FAKE_CLANGXX = """#!/bin/bash
echo "$*" >> "$(dirname "$0")/clang.log"
case "$*" in *-fgcc-only*) echo "clang++: error: unknown argument: '-fgcc-only'"; exit 1;; esac
f="${@: -1}"
[ "${f##*/}" = a.cpp ] && { echo "$f:7:9: error: unused variable 'x' [-Werror,-Wunused-variable]"; exit 1; }
exit 0
"""


class Fixture:
    def __init__(self, base: pathlib.Path):
        self.tree = base / "tree"
        self.build = base / "build"
        self.bin = base / "bin"
        src = self.tree / "rk" / "src"
        src.mkdir(parents=True)
        self.build.mkdir()
        self.bin.mkdir()
        (src / "a.cpp").write_text(
            "#ifdef ROCKE_FEATURE_X\nint f();\n#endif\nint g() { return 1; }\n"
        )
        (src / "b.cpp").write_text('#include "h.hpp"\nint b() { return 2; }\n')
        (src / "c.cpp").write_text("int c() { return 3; }\n")
        (src / "h.hpp").write_text("#if defined(ROCKE_DEFINED)\n#endif\n")
        (self.tree / "rk" / ".clang-tidy").write_text(
            "Checks: 'bugprone-*'\nWarningsAsErrors: '*'\n"
        )
        cmds = []
        for name in ("a.cpp", "b.cpp", "c.cpp"):
            cmds.append(
                {
                    "directory": str(self.build),
                    "file": str(src / name),
                    "arguments": [
                        "/usr/bin/c++",
                        "-DROCKE_DEFINED=1",
                        f"-I{src}",
                        "-std=c++17",
                        "-fgcc-only",
                        "-Wno-foo",
                        "-o",
                        f"{name}.o",
                        "-c",
                        str(src / name),
                    ],
                }
            )
        (self.build / "compile_commands.json").write_text(json.dumps(cmds))
        for name, body in (("clang-tidy", FAKE_TIDY), ("clang++", FAKE_CLANGXX)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)

    def run(self, *files, **kw):
        argv = [
            sys.executable,
            "-I",
            "-B",
            str(TOOL),
            "--tree",
            str(self.tree),
            "--build",
            str(kw.get("build", self.build)),
            "--rocke-path",
            "rk",
            "--llvm-bin",
            str(kw.get("bin", self.bin)),
            *(["--dev-tree", str(kw["dev"])] if kw.get("dev") else []),
            "--files",
            *files,
        ]
        env = {k: v for k, v in os.environ.items() if k != "RKMON_LLVM_BIN"}
        if kw.get("no_path"):
            env["PATH"] = "/nonexistent"
        return subprocess.run(
            argv, capture_output=True, text=True, env=env, timeout=120
        ).stdout


class TestTidyCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.f = Fixture(pathlib.Path(self.d.name))

    def tearDown(self):
        self.d.cleanup()

    def test_errors_of_both_runs_feature_macros_off_and_on(self):
        out = self.f.run(
            "rk/src/a.cpp", "rk/src/h.hpp", "rk/src/missing.cpp", "rk/docs/x.md"
        ).splitlines()
        self.assertEqual(
            out[0],
            "Superbuild tidy (ran): 2 errors in 2 TUs (clang-tidy + clang -Werror; feature macros off and on: "
            "ROCKE_FEATURE_X; 6 runs)",
        )
        self.assertIn(
            "   rk/src/a.cpp:3:5: error: narrowing conversion [bugprone-narrowing-conversions,-warnings-as-errors]"
            "  [tidy, macros on]",
            out,
        )
        self.assertIn(
            "   rk/src/a.cpp:7:9: error: unused variable 'x' [-Werror,-Wunused-variable]"
            "  [clang -Werror, macros off; clang -Werror, macros on]",
            out,
        )
        self.assertIn("   flags clang does not know, dropped: -fgcc-only", out)
        self.assertIn(
            "   rk/src/missing.cpp: no TU of the native build (not compiled here)", out
        )
        self.assertEqual(out[-1], "TIDY fail 2 errors in 2 TUs")
        clang = (self.f.bin / "clang.log").read_text()
        self.assertIn(
            "-fsyntax-only -ferror-limit=0 -Wall -Wextra -Werror -DROCKE_DEFINED=1",
            clang,
        )
        self.assertNotIn(
            "-Wno-foo", clang
        )  # the GCC command's own warning flags are left out
        self.assertNotIn(" -o ", clang)
        self.assertNotIn(
            "c.cpp", clang
        )  # an untouched TU that includes nothing changed
        tidy = (self.f.bin / "tidy.log").read_text()
        self.assertIn(
            f"--quiet --config-file={self.f.tree}/rk/.clang-tidy {self.f.tree}/rk/src/a.cpp -- ",
            tidy,
        )
        self.assertIn("b.cpp", tidy)  # the TU including the changed header

    def test_an_error_develop_has_too_is_not_counted(self):
        import shutil

        dev = pathlib.Path(self.d.name) / "dev"
        shutil.copytree(self.f.tree, dev)
        out = self.f.run("rk/src/a.cpp", dev=dev).splitlines()
        self.assertTrue(
            out[0].startswith("Superbuild tidy (ran): 1 error in 1 TU ("), out
        )
        self.assertIn("develop compared", out[0])
        self.assertIn(
            "   1 more error(s) develop has too, not counted; e.g. rk/src/a.cpp:7:9: error: unused variable 'x' "
            "[-Werror,-Wunused-variable]",
            out,
        )
        self.assertEqual(out[-1], "TIDY fail 1 error in 1 TU")

    def test_a_provider_config_the_sparse_tree_links_to_is_read_from_its_commit(self):
        cfg_dir = self.f.tree / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / ".clang-tidy").write_text("Checks: 'misc-*'\n")
        (self.f.tree / "rk" / ".clang-tidy").unlink()
        (self.f.tree / "rk" / ".clang-tidy").symlink_to("../cfg/.clang-tidy")
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "commit.gpgsign=false",
                    *args,
                ],
                cwd=self.f.tree,
                check=True,
                capture_output=True,
            )
        import shutil

        shutil.rmtree(
            cfg_dir
        )  # a sparse checkout without the folder the link points into
        out = self.f.run("rk/src/c.cpp").splitlines()
        self.assertIn(
            "   rk/.clang-tidy: read from the tree's commit (cfg/.clang-tidy)", out
        )
        tidy = (self.f.bin / "tidy.log").read_text()
        self.assertRegex(tidy, r"--config-file=\S+/clang-tidy-\d+\.yaml ")
        self.assertEqual(out[-1], "TIDY pass 0 errors in 1 TU (no feature macro)")

    def test_no_config_above_the_tu_runs_clang_alone(self):
        (self.f.tree / "rk" / ".clang-tidy").unlink()
        out = self.f.run("rk/src/c.cpp").splitlines()
        self.assertIn(
            "   clang-tidy not run (no .clang-tidy above the TU): rk/src/c.cpp", out
        )
        self.assertFalse((self.f.bin / "tidy.log").exists())

    def test_a_clean_change(self):
        out = self.f.run("rk/src/c.cpp").splitlines()
        self.assertEqual(out[-1], "TIDY pass 0 errors in 1 TU (no feature macro)")

    def test_nothing_to_check_or_nothing_to_check_with(self):
        self.assertEqual(
            self.f.run("rk/docs/x.md").splitlines()[-1],
            "TIDY pass no C/C++ file changed",
        )
        self.assertEqual(
            self.f.run("rk/src/a.cpp", build=self.f.tree).splitlines()[-1],
            "TIDY skip no compile_commands.json",
        )
        self.assertEqual(
            self.f.run("rk/src/a.cpp", bin=self.f.tree, no_path=True).splitlines()[-1],
            "TIDY skip no clang-tidy or clang++",
        )


def git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestInASlot(unittest.TestCase):
    """A sparse worktree of a blobless clone, as the checks read one: the provider's .clang-tidy is a link into a folder
    the tree does not check out, and the head moves a source with a line changed. In a PR runner's slot git fetches
    nothing (GIT_NO_LAZY_FETCH): the changed files come from the trees alone, and the link's file from the clone, once
    the check's --fetch, as the monitor runs it, has fetched its blob."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        origin = d / "origin"
        (origin / "cfg").mkdir(parents=True)
        (origin / "rk" / "src").mkdir(parents=True)
        git(origin, "init", "-q", "-b", "develop")
        (origin / "cfg" / ".clang-tidy").write_text("Checks: 'misc-*'\n")
        (origin / "rk" / ".clang-tidy").symlink_to("../cfg/.clang-tidy")
        body = "".join(f"int v{i}() {{ return {i}; }}\n" for i in range(30))
        (origin / "rk" / "src" / "a.cpp").write_text(body)
        (origin / "rk" / "src" / "c.cpp").write_text("int c() { return 3; }\n")
        git(origin, "add", "-A")
        git(origin, "commit", "-qm", "base")
        self.base = git(origin, "rev-parse", "HEAD")
        git(origin, "mv", "rk/src/a.cpp", "rk/src/m.cpp")
        (origin / "rk" / "src" / "m.cpp").write_text(body + "int w() { return 0; }\n")
        (origin / "rk" / "src" / "c.cpp").write_text("int c() { return 4; }\n")
        git(origin, "commit", "-qam", "head: a move with a line changed")
        head = git(origin, "rev-parse", "HEAD")
        git(origin, "config", "uploadpack.allowFilter", "true")
        git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
        self.clone, self.wt = d / "clone", d / "wt"
        git(
            d,
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            origin.as_uri(),
            "clone",
        )
        git(
            self.clone,
            "worktree",
            "add",
            "-q",
            "--no-checkout",
            "--detach",
            str(self.wt),
            head,
        )
        git(self.wt, "sparse-checkout", "set", "--no-cone", "rk")
        git(self.wt, "reset", "-q", "--hard")
        self.build, self.bin = d / "build", d / "bin"
        self.build.mkdir()
        self.bin.mkdir()
        src = self.wt / "rk" / "src"
        cmds = [
            {
                "directory": str(self.build),
                "file": str(src / name),
                "arguments": ["/usr/bin/c++", "-std=c++17", "-c", str(src / name)],
            }
            for name in ("m.cpp", "c.cpp")
        ]
        (self.build / "compile_commands.json").write_text(json.dumps(cmds))
        for name, text in (("clang-tidy", FAKE_TIDY), ("clang++", FAKE_CLANGXX)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)

    def tool(self, *extra, fetch=False) -> subprocess.CompletedProcess:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("RKMON_LLVM_BIN", "GIT_NO_LAZY_FETCH")
        }
        if not fetch:
            env["GIT_NO_LAZY_FETCH"] = "1"  # as the launcher sets it in a slot
        argv = [sys.executable, "-I", "-B", str(TOOL), "--tree", str(self.wt)]
        argv += ["--build", str(self.build), "--rocke-path", "rk", "--base", self.base]
        argv += ["--llvm-bin", str(self.bin), *extra]
        return subprocess.run(
            argv + (["--fetch"] if fetch else []),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    def test_the_changed_files_are_read_from_the_trees_alone(self):
        out = self.tool().stdout.splitlines()
        self.assertEqual(out[-1], "TIDY pass 0 errors in 2 TUs (no feature macro)")
        self.assertIn(
            "   clang-tidy not run (no .clang-tidy above the TU): rk/src/c.cpp rk/src/m.cpp",
            out,
        )
        self.assertIn(
            "   rk/.clang-tidy links to cfg/.clang-tidy, which the tree and its commit lack",
            out,
        )

    def test_a_base_git_cannot_read_is_no_pass(self):
        r = self.tool("--base", "0" * 40)
        self.assertEqual(
            r.stdout.splitlines()[-1], "TIDY skip no list of the changed files"
        )

    def test_its_fetch_brings_the_file_the_providers_link_names(self):
        r = self.tool("--repo", str(self.clone), fetch=True)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))
        out = self.tool().stdout.splitlines()
        self.assertIn(
            "   rk/.clang-tidy: read from the tree's commit (cfg/.clang-tidy)", out
        )
        self.assertRegex(
            (self.bin / "tidy.log").read_text(),
            r"--config-file=\S+/clang-tidy-\d+\.yaml ",
        )
        self.assertEqual(out[-1], "TIDY pass 0 errors in 2 TUs (no feature macro)")

    def test_its_fetch_passes_over_a_link_out_of_the_tree(self):
        (self.wt / "rk" / "x").mkdir()
        (self.wt / "rk" / "x" / ".clang-tidy").symlink_to("/nonexistent/.clang-tidy")
        r = self.tool("--repo", str(self.clone), fetch=True)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))

    def test_its_fetch_needs_the_clone(self):
        r = self.tool(fetch=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--fetch needs --repo", r.stderr)


class TestChecksSection(unittest.TestCase):
    PLAN = {
        "families": [],
        "tests": [],
        "build": True,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_a_build_runs_it_after_the_native_build_as_a_lead(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f"sec '{review.TIDY_SECTION}'", 1)[1].split("\nsec ", 1)[0]
        self.assertIn('if [ -f "$OUT/native_ok" ]; then', sec)
        self.assertIn(
            "$M/tools/tidy_check.py --tree $W --build $SB/${WTN}/core --rocke-path $K --repo $R0",
            sec,
        )
        # as PR code, after its fetch as the monitor
        self.assertIn("net prf python3 -I -B $M/tools/tidy_check.py --tree $W", sec)
        self.assertIn("net prc nice -n 10 timeout", sec)
        self.assertIn("TIDY skip no result", sec)
        self.assertIn("st tidy skip 'no native build of this tree'", sec)
        self.assertIn("tidy", review.LEAD_SECTIONS)
        self.assertGreater(
            script.index(review.TIDY_SECTION),
            script.index("sec 'Native build + ctest'"),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        plan = dict(self.PLAN, build=False)
        self.assertNotIn(
            review.TIDY_SECTION,
            review.checks_script(cfg, 7, "a" * 40, "first-round", plan, None),
        )

    def test_the_native_build_writes_its_compile_commands(self):
        self.assertIn(
            '"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"',
            (ROOT / "worker" / "build_native.py").read_text(encoding="utf-8"),
        )


class TestLogErrors(unittest.TestCase):
    def test_every_distinct_error_and_the_sources_never_compiled(self):
        from rkmon_lib import ci_runs

        log = (
            "2026-11-05T10:00:00.1Z [1/3] Building CXX object src/CMakeFiles/x.dir/a.cpp.o\n"
            "2026-11-05T10:00:01.1Z /src/a.cpp:3:5: error: narrowing conversion\n"
            "2026-11-05T10:00:02.1Z /src/a.cpp:3:5: error: narrowing conversion\n"
            "summary: error: 0\n"
            "/src/b.cpp:9:1: error: unknown type name 'foo'\n"
        )
        lines = ci_runs.log_errors(
            log, ["rk/src/a.cpp", "rk/src/b.cpp", "rk/src/z.cpp", "rk/README.md"]
        )
        self.assertEqual(
            lines,
            [
                "2 distinct error lines",
                "2:/src/a.cpp:3:5: error: narrowing conversion",
                "5:/src/b.cpp:9:1: error: unknown type name 'foo'",
                "never compiled: rk/src/b.cpp",
                "never compiled: rk/src/z.cpp",
            ],
        )


if __name__ == "__main__":
    unittest.main()
