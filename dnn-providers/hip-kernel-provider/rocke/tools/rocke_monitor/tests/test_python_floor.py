# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every Python file of the package parses on the monitor's oldest Python, doctor.PY_MIN (the image's base runs that
release). ast.parse refuses the grammar a later release added (feature_version), but not the f-strings a later
release reads: from 3.12 a replacement field may hold a string or an f-string in its own f-string's quote, a
backslash or a comment, and before 3.12 each is a syntax error that stops the module's import. The tokens 3.12 gives
an f-string find those.

docker/rkmon runs on the host's Python, deployment.HOST_PYTHON, an older one, and so does tools/make_package.py, which
builds the package there: the two, each module the wrapper loads by name, each program it runs on its own Python, and
each of the package's files those import parse there, and define what they define there. A file's imports are each
import in it, a function's too: rkmon may call any function of a module it loads, and what that function imports
loads on the host's Python then. A relative import names a file of the importing file's folder; an import of
rkmon_lib, one of its modules; a plain import, a file of the importing program's own folder or of worker/, which the
library's modules that import one put on sys.path. Python evaluates an annotation as it defines a function, or a
module's or a class's name, and before 3.10 an X | Y annotation is then a TypeError that stops the module's import,
unless the module imports annotations from __future__, which keeps each annotation a string. And re reads an atomic
group or a possessive quantifier only from 3.11: an older one refuses the pattern as it compiles it, as the module
loads or as the function that holds it runs.
"""

import ast
import io
import pathlib
import re
import subprocess
import sys
import tokenize
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import deployment, doctor  # noqa: E402

PREFIXES = "rRbBuUfFtT"
WRAPPER = ROOT / "docker" / "rkmon"
MAKE_PACKAGE = ROOT / "tools" / "make_package.py"
LIB = ROOT / "rkmon_lib"
WORKER = ROOT / "worker"


def python_files() -> list[pathlib.Path]:
    """Each file of the package (tracked, or new and not ignored) named *.py or started by a python #! line."""
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
    out = []
    for n in names:
        f = ROOT / n
        if not f.is_file() or {"__pycache__", ".git"} & set(f.parts):
            continue
        if f.suffix == ".py":
            out.append(f)
        elif not f.suffix:
            with f.open("rb") as fh:
                first = fh.readline()
            if first.startswith(b"#!") and b"python" in first:
                out.append(f)
    return out


def _quote(token: str) -> str:
    body = token.lstrip(PREFIXES)
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def _clashes(inner: str, outer: str) -> bool:
    """Before 3.12, a string inside a replacement field that opens with its f-string's quote ends that f-string."""
    return inner[0] == outer if len(outer) == 1 else inner == outer


def later_fstrings(source: str) -> list[tuple[int, str]]:
    """(line, what) of each part of an f-string that only 3.12 or newer reads. [] before 3.12, whose tokens hold no
    f-string's parts: there, ast.parse refuses such an f-string itself."""
    start = getattr(tokenize, "FSTRING_START", None)
    if start is None:
        return []
    out, quotes = [], []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == start:
            q = _quote(t.string)
            if any(_clashes(q, o) for o in quotes):
                out.append((t.start[0], f"an f-string inside another in its quote {q}"))
            quotes.append(q)
        elif t.type == tokenize.FSTRING_END:
            quotes.pop()
        elif quotes and t.type == tokenize.STRING:
            q = _quote(t.string)
            if any(_clashes(q, o) for o in quotes):
                out.append((t.start[0], f"a string in its f-string's quote {q}"))
            if "\\" in t.string:
                out.append((t.start[0], "a backslash in a replacement field"))
        elif quotes and t.type == tokenize.COMMENT:
            out.append((t.start[0], "a comment in a replacement field"))
    return out


def too_new(source: str, floor: tuple[int, int]) -> list[tuple[int, str]]:
    """(line, why) of what Python FLOOR cannot parse in SOURCE."""
    try:
        ast.parse(source, feature_version=floor)
    except SyntaxError as exc:
        return [(exc.lineno or 0, exc.msg)]
    return later_fstrings(source)


def loaded_by_name(text: str) -> set[str]:
    """The package's modules docker/rkmon (TEXT) loads by name: lib("NAME") in either quote, and "rkmon_lib.NAME"."""
    return set(re.findall(r"""\blib\(["']([a-z_0-9]+)["']\)""", text)) | set(
        re.findall(r"""["']rkmon_lib\.([a-z_0-9]+)["']""", text)
    )


def run_on_its_python(text: str) -> set[pathlib.Path]:
    """The package's programs docker/rkmon (TEXT) runs on its own Python: each file a command that starts with
    sys.executable names as os.path.join(self.pkg, PART, ...)."""
    out = set()
    for node in ast.walk(ast.parse(text)):
        if not (
            isinstance(node, ast.List)
            and node.elts
            and ast.unparse(node.elts[0]) == "sys.executable"
        ):
            continue
        for e in node.elts[1:]:
            if (
                isinstance(e, ast.Call)
                and ast.unparse(e.func) == "os.path.join"
                and e.args
                and ast.unparse(e.args[0]) == "self.pkg"
                and all(
                    isinstance(a, ast.Constant) and isinstance(a.value, str)
                    for a in e.args[1:]
                )
            ):
                out.add(ROOT.joinpath(*(a.value for a in e.args[1:])))
    return out


def imported(tree: ast.Module, folder: pathlib.Path) -> set[pathlib.Path]:
    """The package's files a file of FOLDER (TREE) imports: each import in it, in a function, a class's body, an if or
    a try too. A relative import names a file of FOLDER; an import of rkmon_lib, one of its modules; a plain import, a
    file of FOLDER (a program's own) or of worker/."""
    out = set()

    def plain(name: str) -> None:
        out.update(
            f for f in (folder / f"{name}.py", WORKER / f"{name}.py") if f.parent != LIB
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 1:
                out |= {
                    folder / f"{n}.py"
                    for n in (
                        [module.split(".")[0]]
                        if module
                        else [a.name for a in node.names]
                    )
                }
            elif module == "rkmon_lib":
                out |= {LIB / f"{a.name}.py" for a in node.names}
            elif module.startswith("rkmon_lib."):
                out.add(LIB / f"{module.split('.')[1]}.py")
            elif module and not node.level:
                plain(module.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("rkmon_lib."):
                    out.add(LIB / f"{a.name.split('.')[1]}.py")
                else:
                    plain(a.name.split(".")[0])
    return {f for f in out if f.is_file()}


def host_modules() -> list[pathlib.Path]:
    """The files that run on the host's Python: docker/rkmon, each module it loads by name and each program it runs on
    its own Python; tools/make_package.py; and each of the package's files those import (imported).
    """
    text = WRAPPER.read_text(encoding="utf-8")
    todo = {
        WRAPPER,
        MAKE_PACKAGE,
        *run_on_its_python(text),
        *(LIB / f"{name}.py" for name in loaded_by_name(text)),
    }
    seen = set()
    while todo:
        f = todo.pop()
        if f not in seen and f.is_file():
            seen.add(f)
            todo |= imported(ast.parse(f.read_text(encoding="utf-8")), f.parent)
    return sorted(seen)


def evaluated_unions(tree: ast.Module) -> list[int]:
    """The line of each X | Y annotation that Python evaluates as it defines a function (its parameters and return)
    or a module's or a class's name. None in a module that imports annotations from __future__, and none on a
    function's own names, whose annotations are never evaluated."""
    if any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    ):
        return []
    out, todo = [], [(node, False) for node in tree.body]
    while todo:
        node, local = todo.pop()
        found = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            found = [
                p.annotation
                for p in (*a.posonlyargs, *a.args, a.vararg, *a.kwonlyargs, a.kwarg)
                if p
            ]
            found.append(node.returns)
        elif isinstance(node, ast.AnnAssign) and not local:
            found = [node.annotation]
        out += [
            x.lineno
            for x in found
            if x is not None
            and any(
                isinstance(y, ast.BinOp) and isinstance(y.op, ast.BitOr)
                for y in ast.walk(x)
            )
        ]
        # A function's body is its own; a class's body runs as the class is defined, wherever that is.
        inner = isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ) or (local and not isinstance(node, ast.ClassDef))
        todo += [(child, inner) for child in ast.iter_child_nodes(node)]
    return sorted(out)


def later_pattern(pattern: str) -> bool:
    """Whether the regular expression PATTERN holds what only Python 3.11 or newer reads: an atomic group or a
    possessive quantifier. An older re refuses them as it compiles the pattern."""
    parser = getattr(re, "_parser", None)
    if parser is None:  # before 3.11, whose re refuses them itself
        try:
            re.compile(pattern)
        except re.error:
            return True
        return False
    todo = [parser.parse(pattern)]
    while todo:
        x = todo.pop()
        if isinstance(x, parser.SubPattern):
            for op, av in x:
                if getattr(op, "name", "") in ("ATOMIC_GROUP", "POSSESSIVE_REPEAT"):
                    return True
                todo.append(av)
        elif isinstance(x, (list, tuple)):
            todo += x
    return False


def later_patterns(tree: ast.Module) -> list[int]:
    """The line of each pattern given as text to a function of re that only Python 3.11 or newer reads (later_pattern):
    the compile is an error before then, as the module loads or as the function that compiles it runs.
    """
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and later_pattern(node.args[0].value)
    )


def host_problems(source: str) -> list[tuple[int, str]]:
    """(line, why) of what the host's Python, deployment.HOST_PYTHON, cannot read in SOURCE: the grammar and the
    f-strings a later Python reads (too_new), each X | Y annotation it evaluates (evaluated_unions), and each pattern
    only a later re reads (later_patterns)."""
    tree = ast.parse(source)
    return [
        *too_new(source, deployment.HOST_PYTHON),
        *(
            (line, "an X | Y annotation, evaluated as it is defined")
            for line in evaluated_unions(tree)
        ),
        *(
            (line, "a pattern that only Python 3.11 or newer compiles")
            for line in later_patterns(tree)
        ),
    ]


class TestPythonFloor(unittest.TestCase):
    def test_every_python_file_parses_on_the_oldest_python_it_runs_on(self):
        files = python_files()
        for rel in ("rkmon.py", "rkmon_lib/doctor.py", "tools/make_package.py"):
            self.assertIn(ROOT / rel, files, "the files read")
        bad = [
            f"{f.relative_to(ROOT)}:{line}: {why}"
            for f in files
            for line, why in too_new(f.read_text(encoding="utf-8"), doctor.PY_MIN)
        ]
        self.assertEqual(bad, [], f"not read by Python {doctor.PY_MIN}")

    def test_what_a_later_python_alone_reads_is_found(self):
        floor = (3, 10)
        for later in (
            'x = f"{a}; {b(c, "d")}"\n',
            'x = f"{f"{a}"}"\n',
            r'''x = f"{'\n'.join(a)}"''' + "\n",
            'x = f"""{a  # why\n}"""\n',
            "try:\n    pass\nexcept* ValueError:\n    pass\n",
            "type X = int\n",
        ):
            with self.subTest(later=later):
                self.assertTrue(too_new(later, floor))
        for fine in (
            "x = f\"{a}; {b(c, 'd')}\"\n",
            'x = f"""{d["k"]}"""\n',
            'x = f"{a}" "b"\n',
            "match x:\n    case 1:\n        pass\n",
        ):
            with self.subTest(fine=fine):
                self.assertEqual(too_new(fine, floor), [])

    def test_what_runs_on_the_host_reads_on_the_host_s_python(self):
        files = host_modules()
        for rel in (
            "docker/rkmon",
            "tools/make_package.py",
            *(
                f"rkmon_lib/{n}.py"
                for n in ("deployment", "settings_file", "privacy", "checks")
            ),
            # loaded only as a function runs: the secrets registry's entries read gh's and atlassian's facts
            "rkmon_lib/gh.py",
            "rkmon_lib/atlassian.py",
            # a plain import of a library module's, from worker/
            "worker/pr_user.py",
            # a program rkmon runs on its own Python, and a file it imports from its own folder
            "tests/container/image.py",
            "tests/container/hostrun.py",
            # loaded as make_package's functions run: a program of its own folder
            "tools/generated.py",
        ):
            self.assertIn(ROOT / rel, files, "the files read")
        bad = [
            f"{f.relative_to(ROOT)}:{line}: {why}"
            for f in files
            for line, why in host_problems(f.read_text(encoding="utf-8"))
        ]
        self.assertEqual(bad, [], f"not read by Python {deployment.HOST_PYTHON}")

    def test_what_a_module_loads_and_evaluates_as_it_loads_is_found(self):
        self.assertEqual(
            loaded_by_name(
                'self.lib("a").b(self.lib(\'c\')) sys.modules.get("rkmon_lib.d") lib(name)'
            ),
            {"a", "c", "d"},
        )
        self.assertEqual(
            run_on_its_python(
                'argv = [sys.executable, "-B", os.path.join(self.pkg, "tests", "container", "image.py"), "--x"]\n'
                'other = ["docker", os.path.join(self.pkg, "tools", "make_package.py")]\n'
            ),
            {ROOT / "tests" / "container" / "image.py"},
        )
        self.assertEqual(
            imported(
                ast.parse(
                    "import os\nimport rkmon_lib.paths\nfrom rkmon_lib import checks\nfrom rkmon_lib.steps import x\n"
                    "from . import outcome, states\nfrom .image import y\n"
                    "try:\n    from . import identity\nexcept ImportError:\n    pass\n"
                    "class C:\n    from . import wrapping\n"
                    "def f():\n    from . import gh\n    import pr_user\n"
                    "from . import no_such_module\nimport no_such_module\n"
                ),
                LIB,
            ),
            {
                *(
                    LIB / f"{n}.py"
                    for n in (
                        "paths",
                        "checks",
                        "steps",
                        "outcome",
                        "states",
                        "image",
                        "identity",
                        "wrapping",
                        "gh",
                    )
                ),
                WORKER / "pr_user.py",
            },
        )
        here = ROOT / "tests" / "container"
        self.assertEqual(
            imported(ast.parse("import json\nimport hostrun\n"), here),
            {here / "hostrun.py"},
        )
        for later in (
            "def f(a: int | None): pass\n",
            "def f(*a: int | str, **k: int | str): pass\n",
            "def f() -> list[int | None]: pass\n",
            "x: int | None = None\n",
            "def f():\n    def g(a: int | None): pass\n",
            "def f():\n    class C:\n        x: int | None = None\n",
        ):
            with self.subTest(later=later):
                self.assertTrue(evaluated_unions(ast.parse(later)))
        for fine in (
            "from __future__ import annotations\ndef f(a: int | None): pass\n",
            "def f():\n    x: int | None = None\n",
            'def f(a: "int | None"): pass\n',
            "x = a | b\n",
        ):
            with self.subTest(fine=fine):
                self.assertEqual(evaluated_unions(ast.parse(fine)), [])
        for later in (
            'P = re.compile(r"(?>a|b)c")\n',
            'P = re.compile("x(?:a++)")\n',
            "def f(s):\n    return re.fullmatch('a*+b', s)\n",
        ):
            with self.subTest(later=later):
                self.assertTrue(later_patterns(ast.parse(later)))
        for fine in (
            'P = re.compile(r"(?=(a|b))\\1c")\n',
            'P = re.compile("[*+]+a{2}")\n',
            'P = parser.compile("(?>a)")\n',
        ):
            with self.subTest(fine=fine):
                self.assertEqual(later_patterns(ast.parse(fine)), [])


if __name__ == "__main__":
    unittest.main()
