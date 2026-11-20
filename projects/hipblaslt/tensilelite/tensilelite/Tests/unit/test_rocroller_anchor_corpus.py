# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every shipped rocRoller kernel must own a GridBased ExactLogic point.

A GridBased library returns only the solutions that own an ExactLogic entry
from findAllSolutions(GEMM_TYPE_ONLY), so a kernel without one is invisible to
every caller that searches by GEMM type. Such a kernel still builds and ships,
and nothing else reports it.
"""

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import corpus, rocroller_logic_files

RR_KERNEL_PREFIX = "RR_GEMM_"

pytestmark = [pytest.mark.unit, needs_logic_corpus]


def test_every_rocroller_kernel_owns_an_exact_logic_point():
    unanchored = []
    seen = 0
    for path in rocroller_logic_files():
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        owned = {entry[1][0] for entry in doc["ExactLogic"] or []}
        for solution in doc["Solutions"]:
            name = (solution.get("CustomKernel") or {}).get("name", "")
            if not name.startswith(RR_KERNEL_PREFIX):
                continue
            seen += 1
            if solution["SolutionIndex"] not in owned:
                unanchored.append(f"{path.name}: [{solution['SolutionIndex']}] {name}")
    assert seen, f"no {RR_KERNEL_PREFIX} solutions under {corpus()}"
    assert not unanchored, f"rocRoller kernels with no ExactLogic point: {unanchored}"
