# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import yaml

from toolchain_test_helpers import toolchain_for
from tensilelite.retune_library import libraryArchitectureName, parseCurrentLibrary


def test_retune_parses_mapping_architecture_and_solutions(tmp_path):
    source = Path(__file__).parent / "data/gfx1100/SB.yaml"
    logic = yaml.safe_load(source.read_text())
    logic[2] = {"Architecture": "gfx1100", "CUCount": 96}
    path = tmp_path / "logic.yaml"
    path.write_text(yaml.safe_dump(logic))
    assembler, isa_info = toolchain_for("gfx1100")
    raw, solutions, sizes = parseCurrentLibrary(path, None, assembler, isa_info, False)
    assert libraryArchitectureName(raw) == "gfx1100"
    assert solutions
    assert sizes.problems
