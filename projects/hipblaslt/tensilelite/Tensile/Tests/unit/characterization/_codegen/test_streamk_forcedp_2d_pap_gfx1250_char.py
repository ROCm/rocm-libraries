# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""ForceDPOnly 2-D dual-operand multicast + PrefetchAcrossPersistent -- gfx1250
characterization (CPU-only).

PAP companion to ``test_streamk_forcedp_2d_gfx1250_char.py``. A genuine 2-D
cluster ClusterDim=[Cs,Ck]=[2,2] where BOTH operands are multicast (Cs X-peers
reuse B on M-adjacent tiles; Ck Y-peers reuse A on N-adjacent tiles -- an
A-multicast, NOT a K-split reduction), with PrefetchAcrossPersistent on.

Because A IS a real multicast (aPeers = Ck > 1), the
``ClusterLoad.papDropsSelfOnlyMaskA`` refinement does NOT free maskA here: BOTH
multicast masks must stay live across the PAP refresh. This is the counterpart
to the factored (FDPO=0, self-only maskA FREED) PAP config and guards against
the refinement accidentally dropping a real A-multicast. It already fit the SGPR
budget before the refinement, so it must remain a real kernel (err==0).

Asserts:
  * every kernel emits real gfx1250 assembly with ``err == 0``;
  * BOTH multicast masks stay live and are applied (MulticastMaskA to the A
    descriptor, MulticastMaskB to the B descriptor);
  * the cluster-scope barrier handshake is present; and
  * the kernel fits the 106-SGPR budget.

CPU-only: no GPU required. Real-HW / FFM validation of the dual-2D + PAP point
is part of the residual re-validation scope.
"""

import os
import re

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
    "streamk_forcedp_2d_pap.yaml",
)

_SGPR_BUDGET = 106
_NEXT_FREE_SGPR = re.compile(r"\.amdhsa_next_free_sgpr\s+(\d+)")


def _max_sgpr(src):
    return max((int(m.group(1)) for m in _NEXT_FREE_SGPR.finditer(src)), default=0)


def test_streamk_forcedp_2d_pap_gfx1250_keeps_both_masks_live():
    """Dual-2D (FDPO=1) + PAP emits real assembly (err==0) and keeps BOTH multicast
    masks live -- the real A-multicast is NOT freed by the guard refinement."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src and "gfx1250" in src, (
            f"Kernel {base!r} missing gfx1250 target"
        )
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"

        # Both masks stay live and are applied (dual-operand multicast under PAP).
        assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in src, (
            f"Kernel {base!r} dropped the real A-multicast mask (MulticastMaskA) -- "
            "the guard refinement must NOT free A on a dual-2D cluster"
        )
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r} dropped the B-multicast mask (MulticastMaskB)"
        )

        # Cluster-scope barrier handshake present.
        assert src.count("s_barrier_signal -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert src.count("s_barrier_wait -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )

        sgprs = _max_sgpr(src)
        assert 0 < sgprs <= _SGPR_BUDGET, (
            f"Kernel {base!r} uses {sgprs} SGPRs, exceeds the {_SGPR_BUDGET} budget"
        )


def test_streamk_forcedp_2d_pap_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the dual-2D PAP emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
