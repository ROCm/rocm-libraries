# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where the checked-in rocRoller logic files live, for the tests that read them.

The files are part of the production logic corpus, so they are reached only
through ``logic_corpus``: a test that calls these helpers skips where the corpus
is absent (the installed test payload) and fails there when
``HIPBLASLT_REQUIRE_LOGIC_CORPUS`` is set.
"""

from pathlib import Path

from logic_corpus import require_asm_full_root

# The base gfx950 directory, not a chip-id overlay: next to 75a3 and 75a2 the
# files name 75a0, which every gfx950 chip id falls back to, so they load on
# MI350X as well as MI355X.
CORPUS_REL = "gfx950/gfx950/GridBased"

# The corpus directory also holds the hipBLASLt-tuned gfx950 logic. Only the
# rocRoller files carry this suffix; test_rocroller_logic_placement.py checks
# that it still selects every logic file that names a rocRoller kernel.
RR_LOGIC_GLOB = "*_MXAE8B32_MXBE8B32_BH_UserArgs.yaml"


def asm_full() -> Path:
    return require_asm_full_root()


def corpus() -> Path:
    return asm_full() / CORPUS_REL


def rocroller_logic_files():
    return sorted(corpus().glob(RR_LOGIC_GLOB))
