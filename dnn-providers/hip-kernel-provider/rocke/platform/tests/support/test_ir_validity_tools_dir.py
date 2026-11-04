# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``ROCKE_PLATFORM_TOOLS_DIR`` tells the compile-and-link test helper where the
IR-validity tool is. A directory without the tool is refused, so a wrong path in
a CI entry fails loudly instead of turning every compile check into a skip."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from rocke.tests.ir_validity import arbiter_verdict

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
