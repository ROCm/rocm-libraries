# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared HalfPLR preparation preserves ordinary scheduling and issue ownership."""

import re

import pytest
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from test_pap_pipeline_overlap_gfx1250 import _config
from test_dataparallel_pap_address_pipeline_gfx1250 import _scalar_destinations
from Tensile.Components.PersistentLoop import PersistentKernelState

pytestmark = pytest.mark.unit


def assert_shared_dispatch_assembly(src):
    """Also usable against the exact runtime binaries without regenerating ISA."""
    lines = src.splitlines()
    def label(name):
        return next(i for i, line in enumerate(lines) if line.startswith("label_" + name + ":"))
    start, issue, done = map(label, ("HalfPlrPrefetchAcrossPersistent", "HalfPlrPAPIssue", "HalfPlrPAPPreparedReturn"))
    preparation = lines[start:issue]
    assert "PapHalfPlrOrdinaryCompute" not in src
    assert "PAP preserve incoming compute SCC" not in src
    assert "HalfPLR PAP prepare at two or issue at one" in src
    assert "HalfPLR PAP shared dispatch prepares only at counter two" in "\n".join(preparation)
    assert not any(re.match(r"(?:ds_|tensor_|s_barrier|s_wait_dscnt|s_wait_tensorcnt|v_wmma)", line.strip()) for line in preparation)
    assert not any(re.match(r"s_\w+\s+s\[sgprtdm", line) for line in preparation)
    assert not any("Reserve next static assignment" in line for line in preparation)
    assert preparation[-1].startswith("s_branch label_HalfPlrPAPPreparedReturn")
    assert "PAP record branchless address preparation validity" in "\n".join(preparation)
    for special in ("EXEC", "VCC"):
        assert sum("PAP mapper restore exact current " + special in line for line in preparation) == 1
    assert "Reserve next static assignment once" in "\n".join(lines[issue:done])
    assert "PAP next addresses prepared by NGLL?" in src
    assert "PAP use ordinary mapping on preparation bypass" in src
    assert "PAP second PGR group already issued?" not in src
    assert "PAP skip redundant first local reads" not in src
    assert "PAP activate retained" not in src

    payload_start = int(re.search(r"s_mov_b32 s(\d+), exec_lo\s+// PAP mapper save current EXEC", src)[1])
    scratch = int(re.search(r"s_sub_u32 s(\d+), s\[sgprLoopCounterL\], 1\s+// HalfPLR PAP dispatch range", src)[1])
    payload = set(range(payload_start, scratch))
    selector = int(re.search(r"s_mov_b32 s(\d+), 0\s+// select branch-back label for HalfPLR loop copy 0", src)[1])
    assert selector not in payload | {scratch}
    assert all(selector not in _scalar_destinations(line) for line in preparation)
    loop = lines[label("ReturnFromHalfPlrPAP_0"):label("LoopEndL")]
    assert all(not payload & _scalar_destinations(line) for line in loop)
    resets = [i for i, line in enumerate(lines) if "invalidate prepared address" in line]
    assert resets and max(resets) < label("ReturnFromHalfPlrPAP_0")

    phases = []
    for copy in (1, 2):
        begin, end = label(f"ReturnFromHalfPlrPAP_{copy}"), label(f"ReturnFromHalfPlrPAP_{copy + 1}")
        phase = lines[begin:end]
        assert phase[1].startswith("s_cmp_eq_i32 s[sgprLoopCounterL]")
        issues = [i for i, line in enumerate(phase) if line.startswith("tensor_load_to_lds")]
        assert len(issues) >= 2
        remaining = phase[issues[-1] + 1:]
        assert any(line.startswith("v_wmma") for line in remaining)
        waits = [int(line.split()[1]) for line in remaining if line.startswith("s_wait_tensorcnt")]
        assert all(count >= 2 for count in waits), remaining
        phases.append({"copy": copy, "wmma_after_pair": sum(line.startswith("v_wmma") for line in remaining), "post_pair_tensor_waits": waits})
    for copy in range(4):
        assert f"select branch-back label for HalfPLR loop copy {copy}" in src
        assert f"label_ReturnFromHalfPlrPAP_{copy}" in "\n".join(lines[done:])
    return {"ordinary_phases": phases, "carrier": sorted(payload), "dispatch_scratch": scratch, "return_selector": selector}


@pytest.mark.parametrize("halfplr", [1, 2, 3])
@pytest.mark.parametrize("cluster", [[1, 1], [2, 4]])
def test_halfplr_prepares_addresses_in_shared_dispatch(tmp_path, halfplr, cluster):
    config = _config(tmp_path, HalfPLR=halfplr, SuppressNoLoadLoop=True,
                     AssertSummationElementMultiple=256, ClusterDim=cluster)
    results = emit_kernels_from_config(config, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        (tmp_path / "halfplr_shared_dispatch.s").write_text(src)
        assert_shared_dispatch_assembly(src)
        assert max(map(int, re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src))) <= 106


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
@pytest.mark.parametrize("fallback", ["resources", "tail"])
def test_halfplr_preparation_falls_back_to_shared_handoff(tmp_path, monkeypatch, fallback, strategy):
    overrides = dict(HalfPLR=1, SuppressNoLoadLoop=True, TileProcessingStrategy=strategy,
                     AssertSummationElementMultiple=256)
    if fallback == "resources":
        monkeypatch.setattr(PersistentKernelState, "papAddressCarrierPlan", lambda *args, **kwargs: None)
    elif fallback == "tail":
        overrides["AssertSummationElementMultiple"] = 32
    results = emit_kernels_from_config(_config(tmp_path, **overrides), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "HalfPLR PAP before final unrolled-loop trip" in src
        assert "HalfPLR PAP prepare at two or issue at one" not in src
        assert "HalfPLR PAP shared dispatch prepares only at counter two" not in src
        assert "PAP mapper save current VCC" not in src
        assert "PAP borrow prepared next coordinate" not in src
        assert "checkpoint WorkGroup0 for PAP restore" in src
        if strategy == "StreamK":
            assert "restore HalfPLR loop-copy selector after PAP" in src
            assert "restore current StreamKLocalEnd after PAP" in src
        for copy in range(4):
            assert f"select branch-back label for HalfPLR loop copy {copy}" in src
