# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The amax, layer norm and softmax generators write their .s files directly, so each one applies
the gfx1200/gfx1201 SGPR read-hazard pass itself. Runs the generators the way
device-library/extops/CMakeLists.txt does. The ``cxx_compiler`` fixture fails rather than skips
when amdclang++ is missing, so these tests cannot pass by not running."""

import os
import subprocess
import sys

import pytest

from tensilelite.SgprHazards import analyze

pytestmark = pytest.mark.unit

TENSILE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# (module, arguments, whether its gfx12 kernel has hazard sites to cover)
GENERATORS = [
    ("amax_generator", ["-t", "S", "-d", "S", "-w", "256", "-c", "4"], True),
    ("layer_norm_generator", ["-w", "256", "-c", "4", "--sweep-once", "0"], True),
    ("softmax_generator", ["-m", "8", "-n", "32"], False),
]


def _generate(tmp_path, compiler, module, args, arch):
    out = tmp_path / f"{module}_{arch}.s"
    result = subprocess.run([sys.executable, "-m", f"tensilelite._extops.{module}", "-o", str(out),
                             *args, "--arch", arch, "--toolchain", str(compiler)],
                            cwd=TENSILE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return out.read_text()


@pytest.mark.parametrize("module, args, hasSites", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_extop_generator_covers_hazards_on_gfx12(tmp_path, cxx_compiler, module, args, hasSites):
    text = _generate(tmp_path, cxx_compiler, module, args, "gfx1201")
    assert ("SGPR read hazard" in text) == hasSites
    assert analyze(text.split("\n"))[1] == {}


@pytest.mark.parametrize("module, args, hasSites", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_extop_generator_leaves_other_arches_alone(tmp_path, cxx_compiler, module, args, hasSites):
    assert "SGPR read hazard" not in _generate(tmp_path, cxx_compiler, module, args, "gfx1100")
