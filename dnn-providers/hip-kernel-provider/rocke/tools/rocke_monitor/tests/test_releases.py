# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The files a new image needs (rkmon_lib/code_volume.py IMAGE_BOUND): what the image is built from, and what it runs
from its own folder (its scripts as root and through sudo, the GitHub request cap's counter), which loads the standard
library and a few files of the package by their paths only; and the container's code as the package reads it
(rkmon_lib/releases.py): `rk code`'s refusal, since root alone changes the code, and the doctor's row that says which
code runs.
"""

import ast
import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import checks, code_volume, doctor, releases  # noqa: E402

# The package's files that each file the image runs from its own folder loads: by their paths (runpy, from the image's
# folder), from its own folder (an import there), or by a name no import statement gives (paths.worker_module, whose
# loads are followed where it is called). The files the image runs that this does not list load the standard library
# only.
LOADED_BY_PATH = {
    "docker/rk_gh.py": ("rkmon_lib/container_layout.py", "rkmon_lib/gh_cap.py"),
    "docker/rk_ids.py": (
        "rkmon_lib/container_layout.py",
        "rkmon_lib/whole_file.py",
        "worker/perms.py",
    ),
    "docker/rk_layout.py": (
        "rkmon_lib/container_layout.py",
        "rkmon_lib/paths.py",
        "worker/perms.py",
    ),
    "docker/rk_prcode.py": (
        "rkmon_lib/container_layout.py",
        "worker/perms.py",
        "worker/pr_env.py",
        "worker/pr_results.py",
    ),
    "docker/rk_root.py": (
        "docker/rk_write.py",
        "rkmon_lib/claude_env_shape.py",
        "rkmon_lib/code_volume.py",
        "rkmon_lib/container_layout.py",
        "rkmon_lib/whole_file.py",
    ),
    "docker/rk_write.py": (
        "rkmon_lib/code_volume.py",
        "rkmon_lib/container_layout.py",
    ),
    "rkmon_lib/claude_env_shape.py": ("rkmon_lib/custom_headers.py",),
    "rkmon_lib/gh_cap.py": (
        "rkmon_lib/paths.py",
        "rkmon_lib/whole_file.py",
        "worker/perms.py",
    ),
    "rkmon_lib/paths.py": (),
    "worker/pr_results.py": ("worker/perms.py",),
}
LOADERS = {"spec_from_file_location", "import_module", "__import__", "run_path"}


def _module(base: pathlib.Path, dotted: str):
    """The file of the module DOTTED in the folder BASE: its .py, or its package's __init__.py; None for neither."""
    p = base.joinpath(*dotted.split("."))
    for f in (p.with_suffix(".py"), p / "__init__.py"):
        if f.is_file():
            return f
    return None


def _in_image(node):
    """The file of the image's folder that NODE names, as the image's scripts name it from CODE (that folder):
    os.path.join(CODE, "a", "b") or f"{CODE}/a/b"; None for any other node."""
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "join":
        first, *rest = node.args or [None]
        if getattr(first, "id", "") == "CODE" and all(
            isinstance(a, ast.Constant) for a in rest
        ):
            return "/".join(a.value for a in rest)
    if isinstance(node, ast.JoinedStr) and len(node.values) == 2:
        code, rest = node.values
        if (
            isinstance(code, ast.FormattedValue)
            and getattr(code.value, "id", "") == "CODE"
        ):
            if isinstance(rest, ast.Constant) and rest.value.startswith("/"):
                return rest.value[1:]
    return None


def loads(rel: str) -> tuple:
    """(the package's files that the Python file REL imports, at its top or in a function, loads or runs from the
    image's folder; whether it also loads one by a path or a name that no import statement gives).
    """
    path = ROOT / rel
    here = path.parent
    # Where an absolute import finds the package's modules: its top folder (rkmon.py and the image's scripts put it on
    # the path), worker/ (paths.worker_module, and the modules that put it there), and a script's own folder.
    bases = [ROOT, ROOT / "worker"] + (
        [] if (here / "__init__.py").is_file() else [here]
    )
    found, by_path = set(), False
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), rel)):
        if isinstance(node, ast.Import) or (
            isinstance(node, ast.ImportFrom) and not node.level
        ):
            if isinstance(node, ast.Import):
                names = [(a.name, ()) for a in node.names]
            else:
                names = [(node.module, [a.name for a in node.names])]
            for name, members in names:
                parts = name.split(".")
                for base in bases:
                    found.update(
                        _module(base, ".".join(parts[: n + 1]))
                        for n in range(len(parts))
                    )
                    found.update(_module(base, f"{name}.{m}") for m in members)
        elif isinstance(node, ast.ImportFrom):
            base = here if node.level == 1 else here.parents[node.level - 2]
            mod = node.module
            found.add(_module(base, mod) if mod else None)
            found.update(
                _module(base, f"{mod}.{a.name}" if mod else a.name) for a in node.names
            )
        elif isinstance(node, (ast.Call, ast.JoinedStr)):
            image = _in_image(node)
            if image:
                found.add(ROOT / image)
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                first = node.args[0] if node.args else None
                if name == "worker_module" and isinstance(first, ast.Constant):
                    found.add(ROOT / "worker" / f"{first.value}.py")
                elif name in LOADERS and not (first is not None and _in_image(first)):
                    by_path = True
    files = {
        f.relative_to(ROOT).as_posix() for f in found if f is not None and f.is_file()
    }
    return files - {rel}, by_path


def image_runs(named: set) -> tuple:
    """(the files NAMED, and in turn each file of the package one of them imports, loads or runs from the image's
    folder (`loads`), with the files LOADED_BY_PATH names for those that load by a path; the files that do).
    """
    seen, todo, by_path = set(named), sorted(named), set()
    while todo:
        rel = todo.pop()
        if not rel.endswith(".py"):
            continue  # the launcher and the entry point: what they run from the image's folder is NAMED
        found, dynamic = loads(rel)
        if dynamic:
            by_path.add(rel)
            found |= set(LOADED_BY_PATH.get(rel, ()))
        for f in sorted(found - seen):
            seen.add(f)
            todo.append(f)
    return seen, by_path


class ImageBound(unittest.TestCase):
    def test_every_build_step_the_dockerfile_copies_needs_a_new_image(self):
        """Each file of docker/ the Dockerfile copies into the image's build, and the requirements its venvs are made
        from, are image-bound: a change to one changes what a new image is built from.
        """
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        copied = {
            f
            for line in text.splitlines()
            if line.startswith("COPY ") and "--from=" not in line
            for f in line.split()[1:-1]
            if f.startswith(("docker/", "requirements/"))
        }
        self.assertIn("requirements/", copied)
        self.assertIn("docker/fetch.sh", copied)
        self.assertEqual(sorted(copied - set(code_volume.IMAGE_BUILT_FROM)), [])

    def named(self) -> set:
        """What the image runs from its own folder by name: the files the entry point names there ($PKG), and those
        the Dockerfile names there (the sudoers lines, the ENV lines, the rk link, the entry point).
        """
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        entry = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        image = re.search(r"^COPY \. (\S+)$", dockerfile, re.M).group(1)
        self.assertEqual(re.search(r"^PKG=(\S+)", entry, re.M).group(1), image)
        named = set(re.findall(r'"\$PKG/([^"]+)"', entry))
        return named | set(re.findall(re.escape(image) + r"/([\w./-]+)", dockerfile))

    def test_what_the_image_runs_from_its_own_folder(self):
        """IMAGE_RUNS is each file the image runs or loads from its own folder, and nothing else: those it names
        (`named`), and in turn every file of the package one of them imports, at its top or in a function, or loads
        (LOADED_BY_PATH, which lists what each loads). The writer's code is not among them: the broker runs the code
        tree it checked, a release or the image's own, and a release carries its own.
        """
        runs, by_path = image_runs(self.named())
        self.assertLessEqual(
            {
                "rk",
                "docker/entrypoint.sh",
                "docker/rk_ids.py",
                "docker/rk_layout.py",
                "docker/rk_root.py",
                "docker/rk_write.py",
                "docker/rk_prcode.py",
                "worker/pr_results.py",
            },
            runs,
        )
        self.assertEqual(
            sorted(
                runs
                & {
                    "rkmon.py",
                    "rkmon_lib/writer.py",
                    "rkmon_lib/writer_run.py",
                    "rkmon_lib/config.py",
                }
            ),
            [],
            "the writer's code is the code tree's the broker checked, not the image's own",
        )
        self.assertLessEqual(
            by_path,
            set(LOADED_BY_PATH),
            "LOADED_BY_PATH lists each file the image runs that loads another by a name no import statement gives",
        )
        for rel in sorted(f for f in runs if f.endswith(".py")):
            with self.subTest(rel=rel):
                self.assertEqual(
                    sorted(loads(rel)[0]),
                    sorted(LOADED_BY_PATH.get(rel, ())),
                    "LOADED_BY_PATH lists the package's files each file the image runs loads",
                )
        self.assertEqual(sorted(set(LOADED_BY_PATH) - runs), [])
        self.assertEqual(
            sorted(runs - set(code_volume.IMAGE_RUNS)),
            [],
            "the image runs these from its own folder: add them to code_volume.IMAGE_RUNS",
        )
        self.assertEqual(
            sorted(set(code_volume.IMAGE_RUNS) - runs),
            [],
            "the image does not run these: take them out of code_volume.IMAGE_RUNS",
        )

    def test_what_the_image_runs_loads_the_standard_library_only(self):
        """Each Python file of IMAGE_RUNS imports Python's standard library only, but the files of the package
        LOADED_BY_PATH says it loads from its own folder."""
        for rel in (f for f in code_volume.IMAGE_RUNS if f.endswith(".py")):
            own = {pathlib.PurePosixPath(f).stem for f in LOADED_BY_PATH.get(rel, ())}
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [("." * node.level) + (node.module or "")]
                else:
                    continue
                for name in names:
                    with self.subTest(rel=rel, name=name):
                        top = name.split(".")[0]
                        self.assertTrue(
                            top in sys.stdlib_module_names
                            or top == "__future__"
                            or name in own,
                            f"{rel} imports {name}",
                        )

    def test_a_change_to_what_an_image_is_built_from(self):
        """image_changes names each image-bound file that differs by its bytes or its execute bit, or that one tree
        lacks; compose.yaml is read on the host, and no image is built from it."""
        with tempfile.TemporaryDirectory() as t:
            a, b = pathlib.Path(t, "a"), pathlib.Path(t, "b")
            for top in (a, b):
                write(top, "docker/Dockerfile", "FROM scratch\n")
                write(top, "docker/entrypoint.sh", "#!/bin/sh\n")
                write(top, "docker/compose.yaml", "services: {}\n")
                write(top, "requirements/checks.txt", "pytest\n")
                write(top, "rkmon_lib/writer.py", "# the writer\n")
            self.assertEqual(code_volume.image_changes(a, b), [])
            write(b, "docker/compose.yaml", "services: {x: {}}\n")
            write(
                b, "rkmon_lib/writer.py", "# the writer, changed\n"
            )  # a release carries it
            self.assertEqual(code_volume.image_changes(a, b), [])
            write(b, "docker/Dockerfile", "FROM other\n")
            (b / "docker" / "entrypoint.sh").chmod(0o755)
            write(b, "requirements/more.txt", "black\n")
            self.assertEqual(
                code_volume.image_changes(a, b),
                ["docker/Dockerfile", "docker/entrypoint.sh", "requirements/more.txt"],
            )


class RootChangesTheCode(unittest.TestCase):
    """`rk code install` and `rk code back` refuse, exit 4, nothing changed: root alone changes the container's code,
    at the host's request; the next line names the host's commands (rkmon package update, then rkmon confirm and rkmon
    apply; rkmon rollback), as the wrapper was run (RKMON_WRAPPER). rk's usage error for `rk update` names the same
    commands (tests/test_control.py).
    """

    def rk_code(self, argv, environ) -> tuple:
        """(rk ARGV's exit code, what it said, joined by one space) with ENVIRON as the environment."""
        import rkmon

        out = io.StringIO()
        kept = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
        with mock.patch.dict(
            os.environ, {**kept, **environ}, clear=True
        ), contextlib.redirect_stdout(out):
            try:
                rc = rkmon.main(argv)
            except SystemExit as exc:
                rc = exc.code
        return rc, " ".join(out.getvalue().split())

    def test_in_the_container_root_alone_changes_the_code(self):
        container = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        install = "next: ./rkmon package update, then ./rkmon confirm and ./rkmon apply"
        for argv, then in (
            (["code", "install"], install),
            (["code", "install", "--from", "-"], install),
            (["code", "back"], "next: ./rkmon rollback"),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(
                    self.rk_code(argv, container),
                    (
                        4,
                        "REFUSED: in the container root alone changes the code, at the host's request; nothing "
                        f"changed {then}",
                    ),
                )

    def test_on_a_host_there_is_no_code_volume(self):
        self.assertEqual(
            self.rk_code(["code", "install"], {}),
            (
                4,
                "REFUSED: rk code works only in the monitor's container; nothing changed",
            ),
        )

    def test_the_help_names_the_host_s_commands(self):
        import rkmon

        text = " ".join(rkmon.__doc__.split())
        self.assertIn(
            "code install | back refused, exit 4: in the container root alone changes the code "
            "(rkmon_lib/releases.py): rkmon apply and every rkmon start on the host put this package's code in, and "
            "rkmon rollback goes back to the code kept",
            text,
        )


class TheDoctorSaysWhichCodeRuns(unittest.TestCase):
    def test_the_image_s_own_or_the_release_over_it(self):
        """The doctor's code row: the image's own code, with its digest, or the release root installed over it, with
        its version and when, and the host's command that goes back to the code before it.
        """
        with tempfile.TemporaryDirectory() as t:
            base = pathlib.Path(t)
            idf, code = base / "image_code_id", base / "code"
            idf.write_text("img1\n")
            rel = code / code_volume.RELEASES / "2026.11.19-0123456789ab"
            for f in code_volume.WHOLE:
                write(rel, f, "x\n")
            write(
                rel,
                code_volume.RELEASE_FILE,
                json.dumps(
                    {
                        "id": rel.name,
                        "version": "2026.11.19",
                        "at": "2026-11-19T10:00:00Z",
                        "base_image": "img1",
                    }
                ),
            )
            env = {"RKMON_CODE_DIR": str(code), "RKMON_WRAPPER": "./rkmon"}
            with mock.patch.object(releases, "IMAGE_ID_FILE", idf), mock.patch.dict(
                os.environ, env
            ):
                self.assertEqual(
                    doctor.code_check({}),
                    checks.Check("code", "PASS", "the image's own (img1)"),
                )
                os.symlink(
                    f"{code_volume.RELEASES}/{rel.name}", code / code_volume.CURRENT
                )
                self.assertEqual(
                    doctor.code_check({}),
                    checks.Check(
                        "code",
                        "PASS",
                        f"release {rel.name} over the image (version 2026.11.19, installed 2026-11-19T10:00:00Z); "
                        "to go back to the code before it: ./rkmon rollback",
                    ),
                )
                idf.write_text("img2\n")  # a new image: its own code runs
                self.assertEqual(
                    doctor.code_check({}),
                    checks.Check("code", "PASS", "the image's own (img2)"),
                )


def write(top: pathlib.Path, rel: str, text: str) -> None:
    """The file REL in the folder TOP, holding TEXT."""
    p = top / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
