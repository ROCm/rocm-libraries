# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 companion carrier-ring VgprGroup characterization.

Target code: LoopModel/adapter.py ``_set_register_depths``, the companion branch.

``VgprGroup`` cuts the buffer one prefetch unit fills. A scale companion whose buffer is already
divided by a carrier ring must not take its parent's cut on top: two carriers then land in one
slot, the later ``ds_load`` overwrites the earlier, and ``check_register_dataflow`` refuses the
plan with WRONG SOURCE before any assembly is emitted. The companion ring is derived from the
parent's tile coverage, and the cut has to be derived with it.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")

# The UseLoopModel arm of mxf8_tn_maf.yaml: MXSA carries a 2-carrier ring beside VgprGroupA=2.
_CONFIG = "loopmodel_mxf8_companion_carrier_vgroup.yaml"


@pytest.fixture(scope="module")
def kernels():
    return emit_kernels_from_config(os.path.join(_DESIGNED, _CONFIG), limit=8, arch=_ARCH)


def test_companion_carrier_ring_keeps_its_own_vgpr_group(kernels):
    """The kernel emits. Inheriting VgprGroupA=2 onto MXSA fails this with WRONG SOURCE."""
    assert kernels, "no kernel was emitted for %s" % _CONFIG
    for basename, _source, err in kernels:
        assert err == 0, "%s did not emit (err=%s)" % (basename, err)
