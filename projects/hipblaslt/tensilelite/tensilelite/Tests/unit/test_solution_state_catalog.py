# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ chip ids have to stay the Python chip ids.

solutionFromState keeps a DeviceNames chip id that ChipIdRegistry (in
AMDGPUPredicates.hpp) knows, and emits PciChipId predicates only for the
architectures ChipIdRegistry::supportsChipIdPredicate accepts. The installed
tensilelite test tree does not contain that header, so this test compares
Python to the checked-in list, and SolutionRecord.CatalogMatchesSolutionState
compares the registry to it. The list holds each chip id with its architecture,
then a "PciChipId <architecture>" line for each architecture that uses chip-id
predicates.
"""

from pathlib import Path

import pytest

from tensilelite.Common.Architectures import GFX_CHIP_IDS, supportsChipIdPredicate

pytestmark = pytest.mark.unit


def _lines(name):
    path = Path(__file__).resolve().parent / "test_data" / name
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_build_chip_ids_match_python():
    python = [f"{chip} {gfx}" for gfx, chips in GFX_CHIP_IDS.items() for chip in chips]
    python += [f"PciChipId {gfx}" for gfx in GFX_CHIP_IDS if supportsChipIdPredicate(gfx)]
    assert _lines("catalog_chip_ids.txt") == python
