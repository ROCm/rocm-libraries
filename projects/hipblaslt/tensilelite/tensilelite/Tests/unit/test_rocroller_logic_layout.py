# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every rocRoller logic file must describe the layout of the kernels it registers.

A rocRoller kernel is built for one layout, written in its name as
``RR_GEMM_<A><B>_`` (``T`` for a transposed operand, ``N`` otherwise). The
library offers a solution only for problems whose transposes equal its logic
file's ``ProblemType``, so a kernel registered under another layout's file is
launched with the wrong strides, and the scale operands are read along the
wrong dimension. Each layout has its own files, named by its index string.
"""

import re

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import corpus, rocroller_logic_files

pytestmark = [pytest.mark.unit, needs_logic_corpus]

_NAME = re.compile(r"^RR_GEMM_([NT])([NT])_")
_CIJK = {
    ("T", "N"): "_Cijk_Alik_Bljk_",
    ("N", "N"): "_Cijk_Ailk_Bljk_",
    ("N", "T"): "_Cijk_Ailk_Bjlk_",
    ("T", "T"): "_Cijk_Alik_Bjlk_",
}


def test_every_rocroller_kernel_sits_in_a_file_of_its_layout():
    wrong = []
    layouts = set()
    for path in rocroller_logic_files():
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        problem = doc["ProblemType"]
        layout = ("T" if problem["TransposeA"] else "N", "T" if problem["TransposeB"] else "N")
        if _CIJK[layout] not in path.name:
            wrong.append(f"{path.name}: ProblemType is {''.join(layout)}")
        for solution in doc["Solutions"]:
            name = (solution.get("CustomKernel") or {}).get("name", "")
            if not name.startswith("RR_GEMM_"):
                continue
            match = _NAME.match(name)
            if not match or match.groups() != layout:
                wrong.append(f"{path.name}: {name}")
            layouts.add(layout)
    assert layouts, f"no RR_GEMM_ solutions under {corpus()}"
    assert not wrong, f"rocRoller kernels registered under another layout: {wrong}"
