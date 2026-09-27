#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit test: ProblemType OutputAmaxD is rejected with Stream-K.
#
# AmaxD's cross-workgroup reduction counts NumWorkGroups0 * NumWorkGroups1 (a
# tile count) while Stream-K launches sk.grid workgroups, so the last arriver is
# never identified; Stream-K's no-work-workgroup exit additionally returns
# before insertAmaxD. Solution.assignDerivedParameters therefore rejects the
# combination instead of emitting a kernel with a silently stale AmaxD output.
#
# The tests drive the real config -> Solution derivation path over the 2x2
# matrix of (OutputAmaxD, StreamK) so that only the combined cell is rejected,
# proving the differentiator is the combination rather than either feature.
#
# Usage:
#   pytest test_streamk_amaxd_reject.py -v
################################################################################

import copy
import os
import sys

import pytest

pytestmark = pytest.mark.unit

_DESIGNED = os.path.join(
    os.path.dirname(__file__), "characterization",
    "_codegen", "data", "test_data", "_designed", "gfx1250")
# A plain (non-Stream-K) gfx1250 GEMM config; all four cells of the matrix below
# derive solutions on it before the reject is added.
_BASE = os.path.join(_DESIGNED, "rich_gemm_sb.yaml")

_ARCH = "gfx1250"


def _write_variant(tmp_path, name, amaxd, streamK):
    """Copy _BASE with the given OutputAmaxD problem type and StreamK fork."""
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(_BASE))
    cfg["BenchmarkProblems"][0][0]["OutputAmaxD"] = amaxd
    fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in fork:
        if "StreamK" in entry:
            entry["StreamK"] = [streamK]
            break
    else:
        fork.append({"StreamK": [streamK]})
    out = tmp_path / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)
    return str(out)


def _derive_states(cfg_path):
    from config_harness import derive_states
    return derive_states(cfg_path, arch=_ARCH, limit_solutions=8)


def test_amaxd_with_streamk_rejected(tmp_path):
    """OutputAmaxD + Stream-K derives no solutions."""
    cfg = _write_variant(tmp_path, "amaxd_sk3.yaml", amaxd=True, streamK=3)
    assert _derive_states(cfg) == [], (
        "OutputAmaxD with Stream-K must be rejected (AmaxD counts tiles, "
        "Stream-K launches sk.grid workgroups)")


@pytest.mark.parametrize("amaxd,streamK", [(True, 0), (False, 3), (False, 0)])
def test_amaxd_or_streamk_alone_still_valid(tmp_path, amaxd, streamK):
    """Control: every other cell of the matrix still derives solutions, so the
    reject above is caused by the combination, not by either feature alone."""
    cfg = _write_variant(tmp_path, f"amaxd{int(amaxd)}_sk{streamK}.yaml",
                         amaxd=amaxd, streamK=streamK)
    assert _derive_states(cfg), (
        f"OutputAmaxD={amaxd} with StreamK={streamK} should still derive "
        "solutions")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
