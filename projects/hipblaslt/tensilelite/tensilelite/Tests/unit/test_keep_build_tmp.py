################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest

pytestmark = pytest.mark.unit

from pathlib import Path

from Tensile import Tensile

_CONFIG = Path(__file__).parent / "test_data" / "keep_build_tmp.yaml"


@pytest.mark.parametrize("keep_build_tmp", [False, True])
def test_keep_build_tmp(tensile_args: list[str], tmp_path: Path, keep_build_tmp: bool) -> None:
    output_dir = tmp_path / "output"
    args = [
        str(_CONFIG), str(output_dir), "--build-only",
        "--global-parameters", f"KeepBuildTmp={keep_build_tmp}",
        *tensile_args,
    ]

    Tensile.Tensile(args)

    leftover = [p for p in output_dir.rglob("build_tmp") if p.is_dir()]
    if keep_build_tmp:
        assert leftover
    else:
        assert not leftover
