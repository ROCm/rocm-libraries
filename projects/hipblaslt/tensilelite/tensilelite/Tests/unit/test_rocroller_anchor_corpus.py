# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every shipped rocRoller kernel must own a GridBased ExactLogic point.

A GridBased library returns only the solutions that own an ExactLogic entry,
so hipBLASLt's checked-in rocRoller selector, which starts from
findAllSolutions(GEMM_TYPE_ONLY), never sees a kernel without one. Such a
kernel still builds and ships, and nothing else reports it.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

RR_KERNEL_PREFIX = "RR_GEMM_"

_HERE = Path(__file__).resolve().parent
_CORPUS = (
    _HERE.parents[3]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
    / "gfx950/gfx950_id75a3/GridBased"
)


def test_every_rocroller_kernel_owns_an_exact_logic_point():
    unanchored = []
    seen = 0
    for path in sorted(_CORPUS.glob("*.yaml")):
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        owned = {entry[1][0] for entry in doc["ExactLogic"] or []}
        for solution in doc["Solutions"]:
            name = (solution.get("CustomKernel") or {}).get("name", "")
            if not name.startswith(RR_KERNEL_PREFIX):
                continue
            seen += 1
            if solution["SolutionIndex"] not in owned:
                unanchored.append(f"{path.name}: [{solution['SolutionIndex']}] {name}")
    assert seen, f"no {RR_KERNEL_PREFIX} solutions under {_CORPUS}"
    assert not unanchored, f"rocRoller kernels with no ExactLogic point: {unanchored}"
