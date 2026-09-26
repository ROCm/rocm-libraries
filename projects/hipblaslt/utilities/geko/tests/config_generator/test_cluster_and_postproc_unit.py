# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import os

from geko.config_generator import cluster_sizes as cs
from geko.config_generator.fork_params import post_processor as base_pp
from geko.config_generator.fork_params.hw_profiles.gfx950 import post_processor as gfx950_pp
from geko.config_generator.fork_params import optimization_param as opt_param
from geko.config_generator.mi_designer import MFMAParameters
from geko.config_generator.shared_utils import ConfigEntry, ForkParameter
from geko.schemas import GemmType


def _entry_with_groups(size, groups):
    return ConfigEntry(
        sizes=[list(size)],
        fork_params={"Groups": ForkParameter(name="Groups", values=[groups])},
        nkernels=1,
        mis_per_size={tuple(size): len(groups)},
    )


def test_cluster_sizes_algorithms(monkeypatch) -> None:
    e0 = _entry_with_groups((32, 16, 1, 16), [])
    e1 = _entry_with_groups((16, 32, 1, 16), [])

    one = cs.do_cluster([e0, e1], CUs=256, is_one_size_per_config=True)
    assert one == {0: [0], 1: [1]}

    all_in = cs.do_cluster([e0, e1], CUs=256, cluster_algo=0)
    assert all_in == {0: [0, 1]}

    monkeypatch.setattr(cs, "_cluster_sizes_mi", lambda entries, CUs: {7: [1, 0]})
    out = cs.do_cluster([e0, e1], CUs=256, cluster_algo=99)
    assert out[7] == [1, 0]


def test_cluster_sizes_mi_and_reorder(monkeypatch) -> None:
    groups = [{"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[16, 16, 16, 1, 1, 1, 1, 2, 2])}]
    e0 = _entry_with_groups((64, 16, 1, 16), groups)
    e1 = _entry_with_groups((16, 64, 1, 16), groups)

    monkeypatch.setattr(cs.MIDesign, "calculate_mfma_parameters", lambda _mi: MFMAParameters(MT0=64, MT1=32, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1))
    out = cs._cluster_sizes_mi([e0, e1], CUs=256)
    assert list(out.keys()) == [(64, 32, 1, 1)]
    assert out[(64, 32, 1, 1)] == [0, 1]

    # Empty groups fall back to (0,0,0,0)
    out2 = cs._cluster_sizes_mi([_entry_with_groups((16, 16, 1, 16), [])], CUs=256)
    assert out2[(0, 0, 0, 0)] == [0]


def _post_cfg(mt_du=None):
    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    return {
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
        "ARCH": "gfx950",
        "StreamK": False,
        "CMS": False,
        **({"MT_DU": mt_du} if mt_du is not None else {}),
    }


def test_base_postprocessor_mt_du_and_matcher(monkeypatch) -> None:
    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})
    pp = base_pp.BasePostProcessor(_post_cfg(mt_du=[64, 32, 16]))
    fork = {
        "DepthU": ForkParameter(name="DepthU", values=[8, 16, 32]),
        "WorkGroupMapping": ForkParameter(name="WorkGroupMapping", values=[16]),
    }
    groups = [
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[16, 16, 4, 1, 1, 2, 2, 2, 2])},
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[32, 32, 4, 2, 1, 2, 2, 2, 2])},
    ]

    monkeypatch.setattr(base_pp.MIDesign, "calculate_mfma_parameters", lambda mi: MFMAParameters(MT0=64, MT1=32, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1) if mi.M == 16 else MFMAParameters(MT0=32, MT1=32, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1))
    f2, g2 = pp.apply(fork, groups, (16, 16, 1, 16))
    assert f2["DepthU"].values == [16]
    assert f2["WorkGroupMapping"].values == [0]
    assert len(g2) == 1

    assert base_pp._mi_matches_mt(groups[0], 64, 32) is True


def test_gfx950_postprocessor_adjustments(monkeypatch) -> None:
    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})
    pp = gfx950_pp.GFX950PostProcessor(_post_cfg())
    fork = {
        "PrefetchGlobalRead": ForkParameter(name="PrefetchGlobalRead", values=[2]),
        "DepthU": ForkParameter(name="DepthU", values=[16, 32, 64]),
        "WorkGroupMapping": ForkParameter(name="WorkGroupMapping", values=[16, 32]),
    }
    groups = [
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[16, 16, 4, 1, 1, 2, 2, 1, 1])},
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[32, 32, 4, 2, 1, 4, 4, 2, 2])},
    ]

    def _calc(mi):
        if mi.M == 16:
            return MFMAParameters(MT0=16, MT1=16, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1)
        return MFMAParameters(MT0=256, MT1=256, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1)

    monkeypatch.setattr(gfx950_pp.MIDesign, "calculate_mfma_parameters", _calc)
    f2, g2 = pp.apply(fork, groups, (128, 128, 1, 2048))
    assert 1 in f2["PrefetchGlobalRead"].values
    assert f2["DepthU"].values[-1] == 128
    assert 1 in f2["WorkGroupMapping"].values
    assert "MIArchVgpr" in g2[1]
    assert f2["UseCustomMainLoopSchedule"].values == [0]


def test_load_cms_groups_import_error(monkeypatch) -> None:
    monkeypatch.setattr(os.path, "isdir", lambda _p: False)
    raised = False
    try:
        gfx950_pp.load_CMS_groups("H", "N", "N", lambda *a, **k: ForkParameter(name=a[0], values=a[1]))
    except ImportError:
        raised = True
    assert raised is True
