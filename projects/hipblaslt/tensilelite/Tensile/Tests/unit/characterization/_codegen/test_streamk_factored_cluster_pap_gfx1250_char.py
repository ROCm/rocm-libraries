# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Factored 2-D StreamK cluster + PrefetchAcrossPersistent -- gfx1250
characterization (CPU-only).

PAP companion to ``test_streamk_factored_cluster_gfx1250_char.py``. Pins the
config that PREVIOUSLY OVERFLOWED the SGPR budget: a genuine 2-D FACTORED
cluster ClusterDim=[Cs,Ck] (Cs = B-multicast, Ck = K-split reduction) with
PrefetchAcrossPersistent + StreamKForceDPOnly=0.

On the factored path the A operand is still per-workgroup (Ck is a REDUCTION
axis, not an A-multicast axis), so maskA is self-only. Before the
``ClusterLoad.papDropsSelfOnlyMaskA`` refinement the self-only maskA was
conservatively kept live under PAP whenever ClusterDim[1] > 1, pushing these
kernels to sgprs=107 -> replaced by an ``s_endpgm`` stub (output unwritten).
The refined guard frees the self-only maskA on the factored path (aPeers==1),
so the kernel now fits <=106 SGPRs and emits REAL assembly.

Asserts:
  * every kernel emits real gfx1250 assembly with ``err == 0``;
  * the B-multicast mask stays live (MulticastMaskB OR'd into a B descriptor);
  * the self-only A mask is FREED -- it is NOT applied to the A descriptor under
    PAP (the specific behavior the guard refinement enables);
  * the intra-cluster split-barrier reduction is present (Ck axis); and
  * the kernel fits the 106-SGPR budget (``.amdhsa_next_free_sgpr <= 106``),
    proving it is a real kernel rather than the overflow ``s_endpgm`` stub.

CPU-only: no GPU required. Real-HW / FFM validation of the factored 2-D + PAP
point is part of the residual re-validation scope.
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
    "streamk_factored_cluster_pap.yaml",
)

_SGPR_BUDGET = 106
_NEXT_FREE_SGPR = re.compile(r"\.amdhsa_next_free_sgpr\s+(\d+)")


def _max_sgpr(src):
    return max((int(m.group(1)) for m in _NEXT_FREE_SGPR.finditer(src)), default=0)


def test_streamk_factored_cluster_pap_gfx1250_frees_selfonly_maskA():
    """Factored 2-D cluster + PAP + FDPO=0 emits real assembly (err==0), keeps the
    B-multicast mask live, FREES the self-only A mask, and fits <=106 SGPRs (was
    107 -> s_endpgm stub before the guard refinement)."""
    results = emit_kernels_from_config(_CONFIG, limit=12, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source (overflow stub?)"
        )
        assert ".amdgcn_target" in src and "gfx1250" in src, (
            f"Kernel {base!r} missing gfx1250 target"
        )
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"

        # B-multicast mask stays live and is applied to the B descriptor.
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r} dropped the live B-multicast mask (MulticastMaskB)"
        )
        # The self-only A mask is FREED under PAP on the factored path: it must
        # NOT be OR'd into the A descriptor (would reference an undeclared SGPR).
        assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" not in src, (
            f"Kernel {base!r} still applies the self-only MulticastMaskA under PAP "
            "(guard refinement did not free it)"
        )

        # Ck axis: intra-cluster split-barrier reduction present.
        assert src.count("s_barrier_signal -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert src.count("s_barrier_wait -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )

        # Fits the SGPR budget (real kernel, not the overflow stub).
        sgprs = _max_sgpr(src)
        assert 0 < sgprs <= _SGPR_BUDGET, (
            f"Kernel {base!r} uses {sgprs} SGPRs, exceeds the {_SGPR_BUDGET} budget"
        )


def test_streamk_factored_cluster_pap_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the factored PAP emit."""
    results = emit_kernels_from_config(_CONFIG, limit=12, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
