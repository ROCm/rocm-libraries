################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""macroAndSetImplClassic MX-scale VGPR-macro cold-branch characterization.

Target code:
  - KernelWriterAssembly.py macroAndSetImplClassic MX-scale VALU/G2L/pack RegSet
    macros. The gfx1250 MX-F4 NN config with VectorWidth{A,B}=1 routes through
    the not-lrvwTile pack arms, recording lines 1146 and 1150 (MXSA) and 1186
    (MXSB) before the emit fails fast.
  - The raise is the intended M-major MX-scale local-read guard in
    Tensile/Components/LocalRead.py:628 (commit 4ab8940).

Background:
  macroAndSetImplClassic runs only when UseSubtileImpl is False. gfx950 forces
  UseSubtileImpl=True for MX (reject pre-emit), so these classic MX macro arms
  are unreachable there. gfx1250 admits the solution and emission reaches the
  MX-scale VGPR-macro cold branch, emitting the RegSet macros, then localReadMX.
  In this M-major (UnrollMajorLDS==0) layout a single local read spans fewer
  bytes than one MX scale unit, so tilePerRead floors to 0; localReadMX guards
  the resulting divide-by-zero with a descriptive Exception. The M-major
  MX-scale local-read layout is unimplemented (MX scales are per-32-K-block, so
  K-major LDS is the only implemented layout).

  Asserting the guarded Exception proves the macroAndSetImplClassic cold branch
  (lines 1146, 1150, 1186) executed and was recorded before the fail-fast.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s00_macroandsetimplclassic_mx_scale.yaml",
)


def test_s00_macroandsetimplclassic_mx_scale_guard_raises():
    """macroAndSetImplClassic MX-scale cold branch runs, then the guard raises.

    Emitting the gfx1250 MX-F4 NN config reaches macroAndSetImplClassic and
    emits the MX-scale VGPR RegSet macros (KernelWriterAssembly.py lines 1146,
    1150, 1186), then localReadMX fails fast with the intended M-major MX-scale
    local-read guard (LocalRead.py:628, commit 4ab8940). The pre-raise lines are
    recorded before the descriptive Exception is raised.
    """
    from config_harness import emit_kernels_from_config

    with pytest.raises(Exception, match=r"unsupported M-major MX-scale local read"):
        emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
