# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MX 128x128 (MXBlockFree) LRA scale-row characterization.

Target code: LraTileAssignmentMFMA, the ``if mxShift:`` block that maps a lane
to its 2D scale row (offset / MXBlockFree). The config emits one kernel per
VectorWidthA/B pair; each arm leaves its own asm comment.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "mx_block128x128_lra.yaml",
)

_ARMS = {
    "nonsplit": "MXS 2D TileSpan non-split",
    "wavesplit": "MXS 2D TileSpan wave-split",
    "onewave": "MXS 2D: one wave along free dim",
    "multiwave": "MXS 2D: wtid = tid",
}


@pytest.fixture(scope="module")
def kernels():
    return emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)


def test_mx_block128x128_lra_emits(kernels):
    assert len(kernels) == 4, f"expected 4 kernels, got {len(kernels)}"
    for base, src, err in kernels:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert "MXAE8B128x128" in base and "MXBE8B128x128" in base


def test_mx_block128x128_lra_covers_every_arm(kernels):
    hit = {arm for _b, src, _e in kernels for arm, tag in _ARMS.items() if tag in src}
    assert hit == set(_ARMS), f"missing arms: {sorted(set(_ARMS) - hit)}"


def test_mx_block128x128_lra_shifts_by_mxblockfree(kernels):
    """Every 2D arm except one-wave ends with offset >> log2(128)."""
    for base, src, _e in kernels:
        for line in src.splitlines():
            if "scale row = " in line and "MXBlockFree(128)" in line:
                assert line.split()[0] == "v_lshrrev_b32", f"{base}: {line}"
                assert "0x7" in line, f"{base}: {line}"
