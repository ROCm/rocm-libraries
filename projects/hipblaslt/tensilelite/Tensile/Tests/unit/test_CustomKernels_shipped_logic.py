# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compatibility sweep over the custom kernels the shipped logic actually uses.

Runs the real comparison over every (logic file, custom kernel) pair under the
hipBLASLt logic root -- the set parseLibraryLogicData walks during a device
library build. The other tests build synthetic ProblemTypes, so a rule can be
self-consistent there and still reject pairings that ship.

The logic root lives outside tensilelite, so this skips when tensilelite is
checked out or installed on its own.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from Tensile.CustomKernelCompatibility import compareCustomKernelProblemTypes
from Tensile.CustomKernels import getCustomKernelConfig
from Tensile.LibraryIO import readYAML


pytestmark = pytest.mark.unit

_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
)


def _mentionsCustomKernel(path: Path) -> bool:
    # Few logic files reference a custom kernel, and parsing them all costs far
    # more than a line scan.
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("CustomKernelName:") and not stripped.endswith("''"):
                return True
    return False


def _logicFilesWithCustomKernels():
    if not _LOGIC_ROOT.is_dir():
        return []
    return sorted(f for f in _LOGIC_ROOT.rglob("*.yaml") if _mentionsCustomKernel(f))


def _customSolutions(data):
    if isinstance(data, dict):
        return data.get("ProblemType"), data.get("Solutions") or []
    return data[4], data[5]


@pytest.fixture
def isolated_valid_parameters():
    """Keep ``getCustomKernelConfig``'s global side effect inside this test.

    It folds ``newMIValidParameters`` into the process-wide ``validParameters``
    registry. Harmless in a build, but in a test session it leaks into whatever
    runs next: the registry and parameter-type tests assert on that dict's exact
    contents.
    """
    from Tensile.Common.ValidParameters import validParameters

    saved = deepcopy(validParameters)
    yield
    validParameters.clear()
    validParameters.update(saved)


def test_shipped_logic_only_references_compatible_custom_kernels(
    isolated_valid_parameters,
):
    files = _logicFilesWithCustomKernels()
    if not files:
        pytest.skip(f"hipBLASLt logic root not available at {_LOGIC_ROOT}")

    incompatible = defaultdict(list)
    checked = 0
    # Each lookup re-reads and re-parses a whole .s file, and far fewer distinct
    # kernels are referenced than there are solutions. Caching per name is safe:
    # this reads only the embedded ProblemType, which InternalSupportParams --
    # the one part of the config that varies per solution -- cannot affect.
    configs = {}
    for path in files:
        problemType, solutions = _customSolutions(readYAML(str(path)))
        for index, solution in enumerate(solutions):
            name = solution.get("CustomKernelName")
            if not name:
                continue
            if name not in configs:
                configs[name] = getCustomKernelConfig(
                    name, solution.get("InternalSupportParams", {})
                )
            config = configs[name]
            checked += 1
            mismatches = compareCustomKernelProblemTypes(
                problemType, config.get("ProblemType")
            )
            if mismatches:
                rel = path.relative_to(_LOGIC_ROOT).as_posix()
                fields = sorted(mismatch.field for mismatch in mismatches)
                incompatible[(rel, name)].append((index, fields))

    assert checked, "no custom-kernel solutions found; the sweep is not doing anything"

    if incompatible:
        report = "\n".join(
            f"  {rel}\n    solution #{entries[0][0]} {name}\n      fields: "
            + ", ".join(entries[0][1])
            for (rel, name), entries in sorted(incompatible.items())
        )
        pytest.fail(
            f"{sum(len(v) for v in incompatible.values())} shipped solution(s) "
            f"reference a custom kernel that cannot service the logic's "
            f"ProblemType:\n{report}\n"
            "Remove the solution from the logic file, or use a kernel whose "
            "custom.config covers it."
        )
