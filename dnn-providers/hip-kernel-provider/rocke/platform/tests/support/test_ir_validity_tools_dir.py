# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``ROCKE_PLATFORM_TOOLS_DIR`` tells the compile-and-link test helper where the
IR-validity tool is. A directory without the tool is refused, so a wrong path in
a CI entry fails loudly instead of turning every compile check into a skip, and
each directory's own copy of the tool is used."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest import mock

import pytest
from rocke.tests.ir_validity import _validity_tool, arbiter_verdict

_MODULE = """\
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k() {
  ret void
}
"""


def test_a_tools_dir_without_the_tool_is_refused(tmp_path):
    with (
        mock.patch.dict(os.environ, {"ROCKE_PLATFORM_TOOLS_DIR": str(tmp_path)}),
        pytest.raises(FileNotFoundError) as refused,
    ):
        arbiter_verdict(_MODULE, "gfx942")
    assert str(refused.value) == (
        f"ROCKE_PLATFORM_TOOLS_DIR={str(tmp_path)!r} holds no check_ir_validity.py "
        f"(resolved to {tmp_path.resolve()}); point it at a directory that does, "
        "or unset it"
    )


def test_each_tools_dir_gets_its_own_copy_of_the_tool(tmp_path):
    """Two directories each hold a copy of the tool and the modules it imports.
    The helper loads the tool from whichever directory the variable names, also
    after loading the other, and the tool's imports come from its own
    directory."""
    tool = _validity_tool()
    if tool is None:
        pytest.skip("rocke platform tools not found")
    source = Path(tool.__file__).parent
    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir()
        for module_file in source.glob("*.py"):
            shutil.copy(module_file, directory)
        with mock.patch.dict(os.environ, {"ROCKE_PLATFORM_TOOLS_DIR": str(directory)}):
            loaded = _validity_tool()
        assert Path(loaded.__file__).parent == directory.resolve()
        for imported in (
            loaded._requires_newer_flavor,
            loaded.available_cpus,
            loaded.tool_identity,
        ):
            assert Path(imported.__code__.co_filename).parent == directory.resolve()
