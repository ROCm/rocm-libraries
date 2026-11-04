# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every shipped rocRoller kernel must own a GridBased ExactLogic point.

A GridBased library returns only the solutions that own an ExactLogic entry,
so hipBLASLt's checked-in rocRoller selector, which starts from
findAllSolutions(GEMM_TYPE_ONLY), never sees a kernel without one. Such a
kernel still builds and ships, and nothing else reports it.
"""

import pytest
import yaml

from rocroller_logic_corpus import CORPUS, LOGIC, rocroller_logic_files

RR_KERNEL_PREFIX = "RR_GEMM_"

# Keyed on the logic tree, not the rocRoller files, so moved files still fail here.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not LOGIC.is_dir(),
    reason=f"source-tree check: no hipBLASLt logic tree at {LOGIC}; the installed "
    "test payload does not ship it",
    ),
]


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
    assert seen, f"no {RR_KERNEL_PREFIX} solutions under {CORPUS}"
    assert not unanchored, f"rocRoller kernels with no ExactLogic point: {unanchored}"
