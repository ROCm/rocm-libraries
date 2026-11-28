# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A script's import closure holds the files its imports reach and names the rest.

``rocke.tests.import_closure.script_import_closure`` reads a script's imports
without running it, resolving them in the script's directory and then in the
search roots. Each case writes a small tree of real Python files.
"""

from __future__ import annotations

from pathlib import Path

from rocke.tests.import_closure import script_import_closure


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_sibling_package_and_function_level_imports_are_all_reached(tmp_path):
    _write(
        tmp_path,
        {
            "scripts/main.py": "import sibling\n\n\ndef run():\n    from pkg import tool\n",
            "scripts/sibling.py": "import json\n",
            "lib/pkg/__init__.py": "from .tool import VALUE\n",
            "lib/pkg/tool.py": "VALUE = 1\n",
            "lib/pkg/unused.py": "VALUE = 2\n",
        },
    )
    closure = script_import_closure(tmp_path / "scripts/main.py", [tmp_path / "lib"])
    assert closure.files == {
        (tmp_path / relative).resolve()
        for relative in (
            "scripts/main.py",
            "scripts/sibling.py",
            "lib/pkg/__init__.py",
            "lib/pkg/tool.py",
        )
    }
    # json is the standard library's; "from .tool import VALUE" names no module.
    assert closure.unresolved == frozenset()


def test_an_import_found_nowhere_is_named_by_its_top_level_package(tmp_path):
    _write(tmp_path, {"main.py": "import absent_helper.part\nimport json\n"})
    closure = script_import_closure(tmp_path / "main.py", [])
    assert closure.unresolved == {"absent_helper"}


def test_a_directory_the_script_adds_to_sys_path_is_not_searched(tmp_path):
    _write(
        tmp_path,
        {
            "main.py": "import sys\n\nsys.path.insert(0, 'extra')\nimport helper\n",
            "extra/helper.py": "VALUE = 1\n",
        },
    )
    closure = script_import_closure(tmp_path / "main.py", [])
    assert (tmp_path / "extra/helper.py").resolve() not in closure.files
    assert closure.unresolved == {"helper"}


def test_a_namespace_package_is_searched_and_holds_no_file(tmp_path):
    """A directory without ``__init__.py`` is a namespace package: its modules
    are reached through it, in every search root that holds a part of it."""
    _write(
        tmp_path,
        {
            "main.py": "import space.first\nfrom space import second\n",
            "lib_a/space/first.py": "VALUE = 1\n",
            "lib_b/space/second.py": "VALUE = 2\n",
        },
    )
    closure = script_import_closure(
        tmp_path / "main.py", [tmp_path / "lib_a", tmp_path / "lib_b"]
    )
    assert closure.files == {
        (tmp_path / relative).resolve()
        for relative in ("main.py", "lib_a/space/first.py", "lib_b/space/second.py")
    }
    assert closure.unresolved == frozenset()
