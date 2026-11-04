# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the added-lines-only absolute import checker.

Pure Python plus a throwaway git repository; no rocisa / HIP needed.
"""

import importlib.util
import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "check_no_absolute_imports.py"
)
_spec = importlib.util.spec_from_file_location("check_no_absolute_imports", _SCRIPT)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

TL = "projects/hipblaslt/tensilelite"
OLD_PKG = f"{TL}/Tensile"
NEW_PKG = f"{TL}/tensilelite"


@pytest.mark.parametrize(
    "stmt, module",
    [
        ("import Tensile", "Tensile"),
        ("import Tensile.Common.Utilities", "Tensile.Common.Utilities"),
        ("import Tensile.Common as C", "Tensile.Common"),
        ("from Tensile import Common", "Tensile"),
        ("from Tensile.Common import Utilities", "Tensile.Common"),
        ("import tensilelite", "tensilelite"),
        ("import tensilelite.Common.Utilities", "tensilelite.Common.Utilities"),
        ("from tensilelite import Common", "tensilelite"),
        ("from tensilelite.Common import Utilities", "tensilelite.Common"),
        ("import os, Tensile.Common", "Tensile.Common"),
    ],
)
def test_banned_forms_are_reported(stmt, module):
    assert hook.banned_imports(stmt + "\n", "x.py") == [(1, 0, module)]


@pytest.mark.parametrize(
    "stmt",
    [
        "from . import Common",
        "from .Common import Utilities",
        "from ..Common import Utilities",
        "import os",
        "import tensilelite_tensile_compat",
        "from TensileCreateLibrary import x",
        "from mytensilelite import x",
        "import numpy.Tensile",
    ],
)
def test_allowed_forms_are_ignored(stmt):
    assert hook.banned_imports(stmt + "\n", "x.py") == []


def test_text_that_only_looks_like_an_import_is_ignored():
    source = (
        '"""\n'
        "from Tensile import Common\n"
        '"""\n'
        "# import tensilelite\n"
        'code = "from tensilelite import x"\n'
    )
    assert hook.banned_imports(source, "x.py") == []


def test_nested_imports_are_reported():
    source = "def f():\n    if True:\n        from Tensile.Common import x\n"
    assert hook.banned_imports(source, "x.py") == [(3, 8, "Tensile.Common")]


def test_bridge_import_is_exempt_only_in_the_alias_package():
    stmt = "from tensilelite._namespace_bridge import install_alias\n"
    assert hook.banned_imports(stmt, f"{OLD_PKG}/__init__.py") == []
    assert hook.banned_imports(stmt, f"{NEW_PKG}/__init__.py") != []
    other = "from tensilelite.Common import x\n"
    assert hook.banned_imports(other, f"{OLD_PKG}/__init__.py") != []


def test_parse_added_lines_reads_hunk_headers():
    diff = (
        "diff --git a/p/a.py b/p/a.py\n"
        "--- a/p/a.py\n"
        "+++ b/p/a.py\n"
        "@@ -3,0 +4,2 @@ ctx\n"
        "+x\n"
        "+y\n"
        "@@ -10 +12 @@\n"
        "-old\n"
        "+new\n"
        "@@ -20,2 +21,0 @@\n"
        "-gone\n"
        "-gone\n"
        "diff --git a/p/b.py b/p/b.py\n"
        "deleted file mode 100644\n"
        "--- a/p/b.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-a\n"
        "-b\n"
    )
    assert hook.parse_added_lines(diff) == {"p/a.py": {4, 5, 12}}


def _git(repo, *args):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "base")
    monkeypatch.chdir(tmp_path)
    for var in ("PRE_COMMIT_FROM_REF", "PRE_COMMIT_TO_REF"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


LEGACY = (
    "import os\n"
    "from Tensile.Common import a\n"
    "from Tensile.Common import b\n"
    "import tensilelite.Common.c\n"
    "\n"
    "\n"
    "def f():\n"
    "    return a, b, os\n"
)


def _commit_all(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def test_only_the_added_line_is_reported_in_a_file_with_existing_violations(
    repo, capsys
):
    mod = f"{OLD_PKG}/mod.py"
    _write(repo, mod, LEGACY)
    _commit_all(repo, "base")

    _write(
        repo,
        mod,
        LEGACY.replace("\n\ndef f", "from tensilelite.Common import new\n\n\ndef f", 1),
    )
    _git(repo, "add", "-A")  # staged, as in a commit hook

    assert hook.main([mod]) == 1
    out = capsys.readouterr().out
    assert f"{mod}:5:1: absolute import of 'tensilelite.Common'" in out
    assert out.count("absolute import of") == 1


def test_touching_a_file_without_adding_imports_passes(repo):
    mod = f"{NEW_PKG}/mod.py"
    _write(repo, mod, LEGACY)
    _commit_all(repo, "base")

    _write(repo, mod, LEGACY.replace("return a, b, os", "return os, b, a"))
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 0


def test_relative_import_added_next_to_existing_absolute_ones_passes(repo):
    mod = f"{NEW_PKG}/mod.py"
    _write(repo, mod, LEGACY)
    _commit_all(repo, "base")

    _write(repo, mod, LEGACY.replace("import os\n", "import os\nfrom . import d\n"))
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 0


def test_appending_a_name_to_an_existing_multiline_import_is_not_reported(repo):
    mod = f"{NEW_PKG}/mod.py"
    _write(repo, mod, "from Tensile.Common import (\n    a,\n)\n")
    _commit_all(repo, "base")

    _write(repo, mod, "from Tensile.Common import (\n    a,\n    b,\n)\n")
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 0


def test_new_file_reports_every_absolute_import(repo, capsys):
    _write(repo, f"{NEW_PKG}/seed.py", "x = 1\n")
    _commit_all(repo, "base")

    mod = f"{NEW_PKG}/new.py"
    _write(repo, mod, LEGACY)
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 1
    assert capsys.readouterr().out.count("absolute import of") == 3


def test_renamed_file_with_unchanged_imports_passes(repo):
    old = f"{OLD_PKG}/mod.py"
    new = f"{NEW_PKG}/mod.py"
    _write(repo, old, LEGACY * 3)
    _commit_all(repo, "base")

    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    (repo / old).rename(repo / new)
    _git(repo, "add", "-A")

    assert hook.main([new]) == 0


def test_files_outside_the_package_roots_are_ignored(repo):
    other = f"{TL}/scripts/tool.py"
    _write(repo, other, "from Tensile import x\n")
    _write(repo, f"{NEW_PKG}/seed.py", "x = 1\n")
    _git(repo, "add", "-A")

    assert hook.main([other]) == 0


@pytest.mark.parametrize("root", [OLD_PKG, NEW_PKG])
def test_tests_directories_are_never_checked(repo, root):
    test_file = f"{root}/Tests/unit/test_x.py"
    _write(repo, test_file, "from Tensile.Common import a\nimport tensilelite\n")
    _write(repo, f"{NEW_PKG}/seed.py", "x = 1\n")
    _git(repo, "add", "-A")

    assert hook.main([test_file]) == 0
    assert not hook._in_scope(test_file)
    assert hook._in_scope(f"{root}/Common/x.py")


def test_missing_package_root_is_skipped_silently(repo, capsys):
    mod = f"{NEW_PKG}/mod.py"  # the Tensile root does not exist at all
    _write(repo, mod, "import os\n")
    _commit_all(repo, "base")
    _write(repo, mod, "import os\nimport sys\n")
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 0
    assert capsys.readouterr().out == ""


def test_ref_mode_checks_the_range_and_reads_the_to_ref(repo, monkeypatch, capsys):
    mod = f"{OLD_PKG}/mod.py"
    _write(repo, mod, LEGACY)
    _commit_all(repo, "base")
    _git(repo, "branch", "develop")

    _write(repo, mod, LEGACY + "from Tensile.Common import late\n")
    _commit_all(repo, "adds one")
    # A working-tree edit after the commit must not influence a ref-range check.
    _write(repo, mod, LEGACY)

    monkeypatch.setenv("PRE_COMMIT_FROM_REF", "develop")
    monkeypatch.setenv("PRE_COMMIT_TO_REF", "HEAD")

    assert hook.main([mod]) == 1
    out = capsys.readouterr().out
    assert f"{mod}:9:1:" in out
    assert out.count("absolute import of") == 1


def test_unparseable_touched_file_fails_loudly(repo, capsys):
    mod = f"{NEW_PKG}/mod.py"
    _write(repo, mod, "x = 1\n")
    _commit_all(repo, "base")
    _write(repo, mod, "x = 1\ndef (:\n")
    _git(repo, "add", "-A")

    assert hook.main([mod]) == 1
    assert "cannot parse" in capsys.readouterr().out
