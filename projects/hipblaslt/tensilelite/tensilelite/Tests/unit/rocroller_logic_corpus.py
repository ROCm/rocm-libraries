# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where the checked-in rocRoller logic files live, for the tests that read them."""

from pathlib import Path

LOGIC = (
    Path(__file__).resolve().parents[4]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)

# The base gfx950 directory, not a chip-id overlay: the files declare
# 75a0, which every gfx950 chip id falls back to, so they load on MI350X
# as well as MI355X.
CORPUS = LOGIC / "gfx950/gfx950/GridBased"

# CORPUS also holds the hipBLASLt-tuned gfx950 logic. Only the rocRoller files
# carry this suffix; test_rocroller_logic_placement.py checks that it still
# selects every logic file that names a rocRoller kernel.
RR_LOGIC_GLOB = "*_MXAE8B32_MXBE8B32_BH_UserArgs.yaml"


def rocroller_logic_files():
    return sorted(CORPUS.glob(RR_LOGIC_GLOB))
