# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The amax, layer norm and softmax generators write their .s files directly, so each one applies
the gfx1200/gfx1201 SGPR read-hazard pass itself. Runs the generators the way
device-library/extops/CMakeLists.txt does."""

import os
import shutil
import subprocess
import sys

import pytest

from tensilelite.SgprHazards import analyze

pytestmark = pytest.mark.unit

TENSILE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
TOOLCHAIN = shutil.which("amdclang++", path=os.pathsep.join(
    [os.path.join(_ROCM, "bin"), os.path.join(_ROCM, "lib", "llvm", "bin")]))

# (module, arguments, whether its gfx12 kernel has hazard sites to cover)
GENERATORS = [
    ("amax_generator", ["-t", "S", "-d", "S", "-w", "256", "-c", "4"], True),
    ("layer_norm_generator", ["-w", "256", "-c", "4", "--sweep-once", "0"], True),
    ("softmax_generator", ["-m", "8", "-n", "32"], False),
]


def _generate(tmp_path, module, args, arch):
    out = tmp_path / f"{module}_{arch}.s"
    subprocess.run([sys.executable, "-m", f"tensilelite._extops.{module}", "-o", str(out), *args,
                    "--arch", arch, "--toolchain", TOOLCHAIN],
                   cwd=TENSILE_ROOT, check=True, capture_output=True)
    return out.read_text()


@pytest.mark.skipif(TOOLCHAIN is None, reason="amdclang++ not found")
@pytest.mark.parametrize("module, args, hasSites", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_extop_generator_covers_hazards_on_gfx12(tmp_path, module, args, hasSites):
    text = _generate(tmp_path, module, args, "gfx1201")
    assert ("SGPR read hazard" in text) == hasSites
    assert analyze(text.split("\n"))[1] == {}


@pytest.mark.skipif(TOOLCHAIN is None, reason="amdclang++ not found")
@pytest.mark.parametrize("module, args, hasSites", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_extop_generator_leaves_other_arches_alone(tmp_path, module, args, hasSites):
    assert "SGPR read hazard" not in _generate(tmp_path, module, args, "gfx1100")
