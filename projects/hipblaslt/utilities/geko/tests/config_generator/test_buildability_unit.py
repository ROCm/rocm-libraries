# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the buildable-MI search: presets from the geko profiles, the resource measure, the repair stage,
stage-3 starts and the count of failures no modelled resource explains (fake TensileLite)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from geko.config_generator import buildability as B
from geko.config_generator import fork_params
from geko.config_generator.constants import _ARCH_SPECS
from geko.schemas import GemmConfig, GemmType

MI = (16, 16, 4, 1, 1, 1, 1, 4, 1)  # 4 waves: a single MI entry (LSU 1)
ENTRY = {"MatrixInstruction": list(MI)}
REASON = "FP4 TN requires SourceSwap"


def _search(arch: str, fork: dict, wavefront: int = 64) -> B.SettingSearch:
    gt = GemmType.from_tensile("N", "N", "S", "S", "S")
    space = B.TuningSpace(Path("probe.yaml"), fork, {}, "group_MatrixInstruction", wavefront)
    return B.SettingSearch(space, {"ARCH": arch, "GemmProblems": [GemmConfig(gt, [[64, 64, 1, 64]])]})


# ---------------------------------------------------------------------------- presets and factorial
def test_gfx1250_presets_vary_halfplr_and_clusterlocalread() -> None:
    s = _search(
        "gfx1250",
        {"DepthU": [64, 128], "HalfPLR": [0, 1, 2, 3], "ClusterLocalRead": [0, 1], "PrefetchGL2": [2, 1, 0]},
    )
    presets = s.presets(MI, ENTRY)
    assert [(p["HalfPLR"], p["ClusterLocalRead"]) for p in presets] == [
        (h, c) for c in (0, 1) for h in (0, 2, 1, 3)
    ]
    assert {p["DepthU"] for p in presets} == {64}  # the smallest legal DepthU
    assert {p["PrefetchGL2"] for p in presets} == {0}  # LOW_RESOURCE, not the space's first value


def test_gfx1250_presets_without_clusterlocalread_deduplicate() -> None:
    s = _search("gfx1250-strict", {"DepthU": [64, 128], "HalfPLR": [0, 1, 2, 3]})
    assert [p["HalfPLR"] for p in s.presets(MI, ENTRY)] == [0, 2, 1, 3]


def test_factorial_includes_halfplr_where_the_space_tunes_it() -> None:
    s = _search("gfx1250", {"DepthU": [64, 128, 256], "HalfPLR": [0, 1, 2, 3], "ClusterLocalRead": [0, 1]})
    core = list(s.core(MI, ENTRY))
    assert len(core) == 2 * 4 * 2  # two smallest DepthUs x HalfPLR x ClusterLocalRead
    assert {p["HalfPLR"] for p in core} == {0, 1, 2, 3}


@pytest.mark.parametrize("arch", ["gfx1250", "gfx950"])
def test_factorial_without_halfplr_in_the_space(arch: str) -> None:
    core = list(_search(arch, {"DepthU": [64, 128, 256], "ClusterLocalRead": [0, 1]}).core(MI, ENTRY))
    assert len(core) == 2 * 2
    assert all("HalfPLR" not in p for p in core)


@pytest.mark.parametrize("arch", sorted(_ARCH_SPECS))
def test_presets_follow_the_architectures_generic_profile(arch: str) -> None:
    s = _search(arch, {"DepthU": [64, 128]})
    assert s.presets_table is B.LIST_PRESETS[fork_params.get_profile_class(arch, B.LIST_SEARCH_SPACE)]
    if arch.startswith("gfx1250"):
        assert [(p["HalfPLR"], p["ClusterLocalRead"]) for p in s.presets_table][:2] == [(0, 0), (2, 0)]
    else:
        assert s.presets_table is B.LIST_PRESETS[fork_params.get_profile_class("gfx950", B.LIST_SEARCH_SPACE)]


def test_a_profile_without_presets_starts_from_the_factorial(monkeypatch) -> None:
    class NewArch(fork_params.optimization_param.BaseOptimizationParams):
        pass

    monkeypatch.setattr(B, "get_profile_class", lambda arch, space: NewArch)
    s = _search("gfx9999", {"DepthU": [64, 128], "SourceSwap": [False, True]})
    assert s.presets(MI, ENTRY) == []
    v = B.classify(_FakeTensile(lambda s: True, lambda s: s["DepthU"] == 128), s, MI)
    assert v.buildable and v.stage == "core" and v.checks == 2


# ---------------------------------------------------------------------------- resource measure
def _measured(vgpr, agpr, vgpr_cap, occ_regs, occ_cap) -> B.Measurement:
    return B.Measurement(
        err=1, vgpr=vgpr, agpr=agpr, sgpr=50, lds=1000, vgpr_cap=vgpr_cap, sgpr_cap=106, lds_cap=327680,
        occ_regs=occ_regs, occ_cap=occ_cap,
    )


def test_gfx1250_vgprs_count_against_the_unified_1024_file() -> None:
    # No AGPRs; 128 threads on a WGP of two CUs: occupancy allows 2 * 131072 / 128 = 2048 per lane.
    assert _measured(900, 0, 1024, 900, 2048).overflow() < 0
    m = _measured(1033, 0, 1024, 1033, 2048)
    assert m.resource() == "vgpr"  # not split by whether the count is a multiple of 8
    assert math.isclose(m.overflow(), 1033 / 1024 - 1)


def test_gfx9_unified_vgpr_agpr_file_is_the_occupancy_limit() -> None:
    # 256 threads: 131072 / 256 = 512 per lane for VGPRs (to a multiple of 8) plus AGPRs.
    m = _measured(250, 300, 256, 256 + 300, 512)
    assert m.resource() == "occupancy"
    assert math.isclose(m.overflow(), 556 / 512 - 1)


def test_refusals_have_no_distance_except_lds() -> None:
    assert B.Measurement(err=-1, reason=REASON).overflow() == math.inf
    lds = B.Measurement(err=-1, lds=400000, lds_cap=327680)
    assert lds.resource() == "lds" and math.isclose(lds.overflow(), 400000 / 327680 - 1)


def test_failures_no_modelled_resource_explains_are_counted_per_code() -> None:
    t = _FakeTensile(lambda s: True, lambda s: False)
    t.measure = lambda s: B.Measurement(  # every modelled resource fits, yet code generation failed (code 13)
        err=13, vgpr=100, agpr=0, sgpr=50, lds=1000, vgpr_cap=256, sgpr_cap=106, lds_cap=65536, occ_regs=100, occ_cap=512
    )
    v = B.classify(t, _search("gfx950", {"DepthU": [64, 128]}), MI)
    assert not v.buildable
    assert v.unmodelled == {13: v.checks}


# ---------------------------------------------------------------------------- stage-3 starts
def test_starts_skip_refused_settings() -> None:
    assert B._starts([(math.inf, "", {"x": 0}), (math.inf, "", {"x": 1})]) == []
    refused, measured = {"x": 0}, {"x": 1}
    assert B._starts([(math.inf, "", refused), (0.1, "vgpr", measured)]) == [measured]


def test_starts_are_the_closest_measured_failures() -> None:
    fails = [(ov, "vgpr", {"x": i}) for i, ov in enumerate([0.6, 0.2, 0.5, 0.1, 0.4, 0.3])]
    assert B._starts(fails) == [{"x": 3}, {"x": 1}, {"x": 5}, {"x": 4}]


# ---------------------------------------------------------------------------- repair
class _FakeTensile:
    """``valid(s)`` decides validation; valid settings build when ``builds(s)``, else overflow VGPRs."""

    def __init__(self, valid, builds, reason=REASON):
        self.valid, self.builds, self.reason = valid, builds, reason
        self.validated = []

    def accepts(self, s):
        self.validated.append(s)
        return self.valid(s)

    def measure(self, s):
        if not self.valid(s):
            return B.Measurement(err=-1, reason=self.reason)
        if self.builds(s):
            return B.Measurement(err=0)
        return _measured(300, 0, 256, 300, 512)


FORK = {"DepthU": [64, 128], "SourceSwap": [False, True], "X": [0, 1, 2]}


def test_repair_builds_when_every_preset_and_factorial_setting_is_refused() -> None:
    t = _FakeTensile(lambda s: s["SourceSwap"], lambda s: True)
    v = B.classify(t, _search("gfx1250", FORK), MI)
    assert v.buildable and v.stage == "repair"
    assert v.witness["SourceSwap"] is True and v.witness["DepthU"] == 64
    assert v.checks == 3  # preset (DepthU 64), factorial DepthU 128, the repaired setting
    assert v.validations == 4  # every single change of the refused preset
    assert t.validated[0]["SourceSwap"] is True  # the parameter the refusal names is tried first


def test_repair_reaches_refused_settings_next_to_measured_ones() -> None:
    # DepthU 128 generates code but overflows; DepthU 64 builds only with SourceSwap. Stage 3 from the
    # measured DepthU 128 kernel cannot reach it in one change, the repair of the refused DepthU 64 can.
    t = _FakeTensile(lambda s: s["DepthU"] == 128 or s["SourceSwap"], lambda s: s["DepthU"] == 64)
    v = B.classify(t, _search("gfx1250", FORK), MI)
    assert v.buildable and v.stage == "repair"
    assert (v.witness["DepthU"], v.witness["SourceSwap"]) == (64, True)


def test_repair_tries_pairs_when_no_single_change_is_accepted() -> None:
    t = _FakeTensile(lambda s: s["SourceSwap"] and s["X"] == 2, lambda s: True, reason="unnamed rule")
    v = B.classify(t, _search("gfx1250", FORK), MI)
    assert v.buildable and v.stage == "repair"
    assert (v.witness["SourceSwap"], v.witness["X"]) == (True, 2)


def test_repair_pairs_stay_within_budget(monkeypatch) -> None:
    monkeypatch.setattr(B, "REPAIR_PAIRS", 3)
    s = _search("gfx1250", {"DepthU": [64, 128], "X": list(range(10)), "Y": list(range(10))})
    refused = {"DepthU": 64, "X": 0, "Y": 0, s.space.mi_group: ENTRY}
    accepted: dict = {}
    assert B.repair(_FakeTensile(lambda s: False, None), s, refused, REASON, [ENTRY], accepted) == []
    singles = len(list(s.changes(refused, [ENTRY])))
    assert len(accepted) == singles + 3


def test_nothing_accepted_ends_unbuildable_with_the_refusal() -> None:
    t = _FakeTensile(lambda s: False, None)
    v = B.classify(t, _search("gfx1250", FORK), MI)
    assert not v.buildable and v.stage == ""
    assert v.overflow == math.inf and v.reason == REASON
    assert v.checks == 2  # refused kernels are never stage-3 starts
    assert 0 < v.validations <= 2 * (4 + B.REPAIR_PAIRS)  # two refused settings repaired


def test_one_rule_broken_with_different_values_is_repaired_once() -> None:
    t = _FakeTensile(lambda s: False, None)
    t.measure = lambda s: B.Measurement(err=-1, reason=f"PrefetchGlobalRead({s['PrefetchGlobalRead']}) refused")
    v = B.classify(t, _search("gfx950", {"DepthU": [64], "PrefetchGlobalRead": [1, 2]}), MI)
    assert v.checks == 2  # the factorial's PrefetchGlobalRead 2 and 1, both refused
    assert v.validations == 1  # one repair: the single change of the first; none for the second


def _lsu(s) -> int:
    return s["group_MatrixInstruction"].get("WorkGroup", [0, 0, 1])[2]


def test_repair_kernels_do_not_crowd_out_the_factorials_stage3_starts() -> None:
    # LSU 1 kernels are 50% over and build once Y=1 (one stage-3 change). LSU 2/4 kernels are refused without
    # SourceSwap or X > 0; repairing them yields kernels only 10% over that never build. Those must not take
    # the stage-3 start that leads to the build.
    t = _FakeTensile(lambda s: _lsu(s) == 1 or s["SourceSwap"] or s["X"] > 0, lambda s: _lsu(s) == 1 and s["Y"] == 1)
    t.measure = lambda s: (
        B.Measurement(err=-1, reason=REASON) if not t.valid(s)
        else B.Measurement(err=0) if t.builds(s)
        else _measured(384, 0, 256, 384, 512) if _lsu(s) == 1 else _measured(282, 0, 256, 282, 512)
    )
    fork = {"DepthU": [64], "X": [0, 1, 2, 3, 4, 5], "Y": [0, 1], "SourceSwap": [False, True]}
    v = B.classify(t, _search("gfx950", fork), (16, 16, 4, 1, 1, 1, 1, 1, 1))  # one wave: LSU 1, 2 and 4
    assert v.buildable and v.stage == "one-at-a-time" and _lsu(v.witness) == 1 and v.witness["Y"] == 1


@pytest.mark.parametrize("wavefront, wg1", [(64, 4), (32, 2)])
def test_mi_entries_use_the_spaces_wavefront(wavefront: int, wg1: int) -> None:
    s = _search("gfx950", {"DepthU": [64, 128]}, wavefront)
    entries = s.mi_entries((16, 16, 4, 1, 1, 1, 1, 1, 1))  # one 16-lane-wide wave per LSU slice
    assert [e.get("WorkGroup") for e in entries] == [None, [16, wg1, 2], [16, wg1, 4]]


def test_direct_to_vgpr_read_width_follows_the_wavefront() -> None:
    fork = {"DepthU": [64], "DirectToVgprA": [False, True], "GlobalReadVectorWidthA": [-1, 1, 2, 4]}
    entry = {"MatrixInstruction": [16, 16, 4, 1, 1, 1, 1, 4, 1]}  # 16 x 4 A inputs per MFMA
    for lanes, width in ((64, 1), (32, 2)):
        core = list(_search("gfx950", fork, lanes).core(MI, entry))
        assert any(p["DirectToVgprA"] and p["GlobalReadVectorWidthA"] == width for p in core), lanes


def test_a_kernel_built_over_a_modelled_cap_is_reported() -> None:
    t = _FakeTensile(lambda s: True, lambda s: True)
    t.measure = lambda s: B.Measurement(  # TensileLite built it with 300 of 256 VGPRs: the model is out of date
        err=0, vgpr=300, agpr=0, sgpr=50, lds=1000, vgpr_cap=256, sgpr_cap=106, lds_cap=65536, occ_regs=300, occ_cap=512
    )
    v = B.classify(t, _search("gfx950", {"DepthU": [64]}), MI)
    assert v.buildable and v.over_cap == ["vgpr"]


# ---------------------------------------------------------------------------- list file and probe config
def test_written_lists_read_back_even_when_no_mi_builds(tmp_path: Path) -> None:
    from geko.config_generator.mi_candidates import load_buildable_list

    built = B.Verdict(MI, True, "preset", 1, depthu=[64, 128])
    rejected = B.Verdict((16, 16, 4, 1, 1, 1, 2, 4, 1), False, "", 9, overflow=0.1, reason="too many vgprs")
    for verdicts, listed in (([built, rejected], [MI]), ([rejected], [])):
        path = tmp_path / "SSS_NN.yaml"
        B.write_buildable_list(path, verdicts, arch="gfx950", gemm_name="SSS_NN", source="test", depthu=[32, 64, 128])
        bl = load_buildable_list(path)
        assert bl.mis == listed and bl.searched_depthu == [32, 64, 128]
        assert bl.depthu == ({MI: [64, 128]} if listed else {})


def test_probe_config_ignores_exported_overrides(monkeypatch, tmp_path: Path) -> None:
    from geko.config_generator import config_generator

    seen = {}

    def fake_run(cfg, hipblaslt_path, workdir, write_shell_scripts):
        seen.update(StreamK=cfg["StreamK"], BUILDABLE_MIS=cfg["BUILDABLE_MIS"])
        (workdir / f"{cfg['GemmProblems'][0].name}_0.yaml").write_text("")

    monkeypatch.setattr(config_generator, "run", fake_run)
    monkeypatch.setenv("StreamK", "1")
    monkeypatch.setenv("BUILDABLE_MIS", "1")
    cfg = dict(TRANSA="N", TRANSB="N", DataType="S", DestDataType="S", ComputeDataType="S", ARCH="gfx950")
    B.write_probe_config(cfg, tmp_path, tmp_path / "probe")
    assert seen == {"StreamK": B.LIST_STREAMK, "BUILDABLE_MIS": False}
    assert os.environ["StreamK"] == "1" and os.environ["BUILDABLE_MIS"] == "1"  # restored afterwards
