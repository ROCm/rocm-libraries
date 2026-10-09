################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 padded-WG edge-size cluster codegen characterization.

A non-StreamK ClusterDim=[2,2] GEMM whose tile count (23x23 with MT32x32) is not
a multiple of ClusterDim. The launch grid is padded up to a multiple, so the
kernel must:
  1. early-exit the padded work-groups before any load/barrier
     (label 'ClusterPad_EarlyStop'), and
  2. reduce the TDM multicast mask to the WGs actually present in the boundary
     cluster ('reduce multicast mask to real WGs in cluster') so the broadcast
     does not wait for the multicast timeout on the missing padded WGs.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "cluster_padding.yaml",
)

_CONFIG_NPOT_MT1 = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "cluster_padding_npot_mt1.yaml",
)

# Instructions whose first operand is a source, not a destination.
_NO_DST_PREFIXES = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_nop")


def _sgpr_written(line, reg):
    """True if the instruction on ``line`` writes SGPR ``reg`` (sN or s[a:b])."""
    code = line.split("//", 1)[0].strip()
    if not code.startswith("s_") or code.startswith(_NO_DST_PREFIXES):
        return False
    m = re.match(r"\S+\s+s(?:\[(\d+):(\d+)\]|(\d+))\b", code)
    if not m:
        return False
    if m.group(3) is not None:
        return int(m.group(3)) == reg
    return int(m.group(1)) <= reg <= int(m.group(2))


def test_cluster_padding_gfx1250_emits_early_exit_and_reduced_mask():
    """Non-SK ClusterDim=[2,2] padding size: padded early-exit + reduced mask."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        # Cluster WG-id decode arm must be present.
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode ('RemapWorkGroupDone')"
        )
        # Padded work-groups must early-exit before any load/barrier.
        assert "ClusterPad_EarlyStop" in src, (
            f"Kernel {base!r}: missing padded-WG early-exit ('ClusterPad_EarlyStop')"
        )
        assert "s_endpgm" in src, f"Kernel {base!r}: early-exit missing s_endpgm"
        # The multicast mask must be reduced to the real WGs of the cluster.
        assert "reduce multicast mask to real WGs in cluster" in src, (
            f"Kernel {base!r}: multicast mask not reduced for boundary cluster"
        )


def test_cluster_padding_npot_mt1_keeps_wg_x_for_mask_shift():
    """Non-power-of-2 MT1: the mask-reduction scratch must not clobber wg_x.

    The reduction ceil-divides SizeJ by MT1=176 with a magic multiply into SGPR
    scratch. If that scratch aliases the decoded wg_x SGPR, maskA is shifted by
    floor(SizeJ/MT1)*MT1 instead of wg_x and the A/MXSA multicast targets WGs
    outside the cluster (AIHPBLAS-5043: GPU memory fault past the MXSA buffer).
    """
    results = emit_kernels_from_config(_CONFIG_NPOT_MT1, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "STATIC_DIV: divisor=176" in src, (
            f"Kernel {base!r}: expected the magic-number divide by MT1=176"
        )
        lines = src.splitlines()
        wgx_idx = next(i for i, l in enumerate(lines) if "Etract wg_x." in l)
        wgx = int(re.match(r"\s*s_and_b32\s+s(\d+),", lines[wgx_idx]).group(1))
        shift_idx = next(i for i, l in enumerate(lines)
                         if l.lstrip().startswith("s_lshl_b32") and "Setting maskA" in l)
        shift = lines[shift_idx].split("//", 1)[0].rstrip().rsplit(",", 1)[1].strip()
        assert shift == f"s{wgx}", (
            f"Kernel {base!r}: maskA shifted by {shift}, expected wg_x s{wgx}"
        )
        clobbers = [l.strip() for l in lines[wgx_idx + 1:shift_idx] if _sgpr_written(l, wgx)]
        assert not clobbers, (
            f"Kernel {base!r}: wg_x s{wgx} overwritten before the maskA shift: {clobbers}"
        )
