# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's files against rocKE's rules for its files and the checks that rocKE and the repository it sits in run
on them: rocKE's two-line header (every package file that can hold a comment carries it, after the front matter in
skill and agent files and right after the #! line in scripts; tools/make_package.py checks the built folder the same
way); an executable file's #! line; black, check-yaml and check-added-large-files (no file over MAX_KB), as the
repository's pre-commit runs them; and rocKE's relative-path guard, which reads every file of rocKE's tree, the
package's too.
"""

import ast
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

import make_package  # noqa: E402

try:
    import yaml  # PyYAML, for TestYaml
except ImportError:  # TestYaml is skipped
    yaml = None

# rocKE's relative-path guard (relative_path_guard in its platform/tests/run_all.py, the guard stage of its
# tools/run_checks.py too), as its file has it: the files it reads (by their suffix or name, outside the folders it
# skips) and the patterns no line of them may match. rocKE's tree is copied verbatim between repositories, so no code
# or test names a folder of one of them. Each "/" of a pattern is written "\x2f": the guard reads this file too, and
# finds most of them written as they are.
GUARD = {
    "_FORBIDDEN": (
        "\x2fworkspace\\b",
        "rocm-libraries(?:-[a-z-]+)?\x2f",
        "projects\x2fcomposablekernel",
        "dnn-providers\x2f",
    ),
    "_GUARD_SUFFIXES": {".py", ".cmake", ".toml", ".ini", ".sh", ".cfg"},
    "_GUARD_NAMES": {"CMakeLists.txt"},
    "_GUARD_SKIP_DIRS": {".git", "__pycache__", "build", "dsl_docs", "examples"},
    "_GUARD_SKIP_PREFIXES": (".venv",),
}
# The guard's own file, while the package sits in rocKE's tree: two folders up, at rocKE's top.
GUARD_FILE = ROOT.parent.parent / "platform" / "tests" / "run_all.py"
# The largest file the repository takes, in KB of 1024 bytes: its pre-commit's check-added-large-files fails a pull
# request that adds a file over it, the file's size rounded up to a whole KB, as that hook counts it.
MAX_KB = 500


def package_files():
    p = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    names = (
        p.stdout.splitlines()
        if p.returncode == 0
        else [str(f.relative_to(ROOT)) for f in ROOT.rglob("*")]
    )
    for n in names:
        f = ROOT / n
        if (
            f.is_file()
            and "__pycache__" not in f.parts
            and ".git" not in f.parts
            and f.name != "config.json"
        ):
            yield f


def guard_findings(root: pathlib.Path, files) -> list:
    """`path:line: text` for each line that one of GUARD's patterns matches, in each of FILES (in the folder ROOT) that
    the guard reads."""
    forbidden = [re.compile(p) for p in GUARD["_FORBIDDEN"]]
    out = []
    for f in files:
        rel = f.relative_to(root)
        skipped = any(
            p in GUARD["_GUARD_SKIP_DIRS"]
            or p.startswith(GUARD["_GUARD_SKIP_PREFIXES"])
            for p in rel.parts
        )
        read = f.suffix in GUARD["_GUARD_SUFFIXES"] or f.name in GUARD["_GUARD_NAMES"]
        if skipped or not read:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        out += [
            f"{rel.as_posix()}:{n}: {line.strip()}"
            for n, line in enumerate(text.splitlines(), 1)
            if any(p.search(line) for p in forbidden)
        ]
    return out


def guard_of(path: pathlib.Path) -> dict:
    """GUARD's names as the guard's file PATH has them, read without running it: each set and tuple as it is, and each
    of _FORBIDDEN's patterns by its text (one compiled with more than its text, as written, which no copy matches).
    """

    def pattern(node):
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            return ast.literal_eval(node.args[0])
        return ast.unparse(node)

    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in GUARD:
                out[t.id] = (
                    tuple(pattern(e) for e in value.elts)
                    if t.id == "_FORBIDDEN"
                    else ast.literal_eval(value)
                )
    return out


class TestHeaders(unittest.TestCase):
    def test_every_file_that_can_hold_a_comment_has_it(self):
        bad = [
            f"{f.relative_to(ROOT)}: {why}"
            for f in package_files()
            for why in [make_package.header_problem(f)]
            if why
        ]
        self.assertEqual(bad, [])

    def test_placement(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            ok = pathlib.Path(d) / "a.md"
            ok.write_text(
                "---\nname: x\n---\n\n<!--\n"
                + "\n".join(make_package.HEADER_LINES)
                + "\n-->\n# T\n"
            )
            self.assertEqual(make_package.header_problem(ok), "")
            late = pathlib.Path(d) / "b.md"
            late.write_text(
                "<!--\n"
                + "\n".join(make_package.HEADER_LINES)
                + "\n-->\n---\nname: x\n---\n"
            )
            self.assertEqual(
                make_package.header_problem(late),
                "the header must follow the front matter",
            )
            sh = pathlib.Path(d) / "c.sh"
            sh.write_text(
                "#!/bin/sh\necho\n# " + "\n# ".join(make_package.HEADER_LINES) + "\n"
            )
            self.assertEqual(
                make_package.header_problem(sh),
                "the header must be the lines right after #!",
            )
            js = pathlib.Path(d) / "d.json"
            js.write_text("{}\n")
            self.assertEqual(make_package.header_problem(js), "")


class TestModes(unittest.TestCase):
    def test_an_executable_file_starts_with_a_hash_bang_line(self):
        """An executable file of the package starts with a #! line, which names the program that runs it: run by its
        path, a file with none fails, or runs in a shell it was not written for."""

        def starts_with_hash_bang(f):
            with open(f, "rb") as h:
                return h.read(2) == b"#!"

        bad = [
            f.relative_to(ROOT).as_posix()
            for f in package_files()
            if f.stat().st_mode & 0o111 and not starts_with_hash_bang(f)
        ]
        self.assertEqual(bad, [])


class TestBlack(unittest.TestCase):
    """The Python files black formats (tools/make_package.py --black, the repository's pre-commit hook): the .py files
    and the executable ones with a python #! line and no suffix; the POSIX sh entry points are not among them.
    """

    def test_the_files_black_reads(self):
        names = {
            f.relative_to(ROOT).as_posix() for f in make_package.python_files(ROOT)
        }
        for name in (
            "rkmon.py",
            "tools/tick_loop.py",
            "docker/rkmon",
            "tests/fakes/fake_atlassian",
            "tests/fakes/gpu/bin/srun",
        ):
            self.assertIn(name, names)
        for name in (
            "rk",
            "docker/entrypoint.sh",
        ):
            self.assertNotIn(name, names)

    @unittest.skipUnless(
        os.environ.get("RKMON_BLACK"),
        "RKMON_BLACK names black (the release the repository's pre-commit pins)",
    )
    def test_the_source_is_black_formatted(self):
        files = [str(f) for f in make_package.python_files(ROOT)]
        # Black keeps its cache in a folder of the test's own (BLACK_CACHE_DIR), never in a home folder.
        with tempfile.TemporaryDirectory() as cache:
            p = subprocess.run(
                [os.environ["RKMON_BLACK"], "--check", "-q", *files],
                capture_output=True,
                text=True,
                env=dict(os.environ, BLACK_CACHE_DIR=cache),
            )
        self.assertEqual(p.returncode, 0, (p.stdout + p.stderr)[-2000:])


class TestRelativePathGuard(unittest.TestCase):
    """rocKE's relative-path guard (GUARD, its copy here) finds nothing in the package."""

    def test_no_file_the_guard_reads_names_a_folder_of_one_repository(self):
        self.assertEqual(guard_findings(ROOT, package_files()), [])

    def test_what_the_guard_reads(self):
        """A line that names a folder of one repository, in a file the guard reads, is found; a repository's name
        alone, a file of another kind and a folder the guard skips are not."""
        texts = {
            "a.py": "x = 'dnn-providers\x2fx'",
            "CMakeLists.txt": "set(P \x2fworkspace)",
            "b.sh": "cd projects\x2fcomposablekernel",
            "c.toml": "x = 'rocm-libraries-internal\x2f'",
            "d.cfg": "x = rocm-libraries",
            "e.md": "dnn-providers\x2fx",
            "build/f.py": "x = 'dnn-providers\x2fx'",
            ".venv-x/g.py": "x = 'dnn-providers\x2fx'",
        }
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for rel, text in texts.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(f"ok\n{text}\n", encoding="utf-8")
            found = guard_findings(root, sorted(root / rel for rel in texts))
        self.assertEqual(
            found,
            [
                f"{rel}:2: {texts[rel]}"
                for rel in ("CMakeLists.txt", "a.py", "b.sh", "c.toml")
            ],
        )

    def test_the_guard_s_names_are_read_from_its_file(self):
        """guard_of reads GUARD's names from a file of the guard's form, without running it."""
        lines = [
            "import re",
            "raise SystemExit('run')",
            "_GUARD_NAMES: set[str] = {'CMakeLists.txt'}",
            *(
                f"{name} = {value!r}"
                for name, value in GUARD.items()
                if name not in ("_FORBIDDEN", "_GUARD_NAMES")
            ),
            "_FORBIDDEN = [",
            *(f"    re.compile({p!r})," for p in GUARD["_FORBIDDEN"]),
            "]",
        ]
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "run_all.py"
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(guard_of(f), GUARD)
            # a pattern compiled with a flag differs from the copy's
            f.write_text(f.read_text("utf-8").replace("),\n]", ", re.I),\n]"), "utf-8")
            self.assertNotEqual(guard_of(f), GUARD)

    def test_the_copy_is_the_guard_s_own(self):
        """While the package sits in rocKE's tree, GUARD is what the guard's own file has."""
        if not GUARD_FILE.is_file():
            self.skipTest("the package sits in no rocKE tree")
        self.assertEqual(guard_of(GUARD_FILE), GUARD)


class TestSize(unittest.TestCase):
    def test_no_file_is_over_the_size_the_repository_takes(self):
        """No file of the package is over MAX_KB, its size counted as check-added-large-files counts it: rocm-libraries'
        pre-commit runs that hook, which fails a pull request that adds a larger file.
        """
        sizes = {f: math.ceil(f.stat().st_size / 1024) for f in package_files()}
        bad = [
            f"{f.relative_to(ROOT).as_posix()}: {kb} KB, over the limit of {MAX_KB} KB"
            for f, kb in sizes.items()
            if kb > MAX_KB
        ]
        self.assertEqual(
            bad,
            [],
            "rocm-libraries' pre-commit (check-added-large-files) fails a pull request that adds a file over "
            f"{MAX_KB} KB: split each file named into smaller ones",
        )


class TestYaml(unittest.TestCase):
    @unittest.skipIf(yaml is None, "PyYAML is not installed")
    def test_each_yaml_file_loads(self):
        """Each .yaml and .yml file of the package loads with PyYAML's safe_load, as the repository's pre-commit
        (check-yaml) loads each YAML file a pull request changes."""
        bad = []
        for f in package_files():
            if f.suffix in (".yaml", ".yml"):
                try:
                    yaml.safe_load(f.read_text(encoding="utf-8"))
                except yaml.YAMLError as exc:
                    bad.append(f"{f.relative_to(ROOT)}: {' '.join(str(exc).split())}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
