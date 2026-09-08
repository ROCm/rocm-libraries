# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from geko.config_generator import config_generator as cg
from geko.config_generator import fork_param_generator as fpg
from geko.config_generator.fork_params import optimization_param as opt_param
from geko.config_generator.fork_params import post_processor as base_pp
from geko.config_generator.fork_params import get_optimization_params, get_post_processor
from geko.config_generator.fork_params import param_meta as pm
from geko.config_generator.fork_params.hw_profiles.gfx942 import optimization_param as g942_opt
from geko.config_generator.fork_params.hw_profiles.gfx942 import post_processor as g942_pp
from geko.config_generator.fork_params.hw_profiles.gfx950 import optimization_param as g950_opt
from geko.config_generator.fork_params.hw_profiles.gfx950 import post_processor as g950_pp
from geko.config_generator.mi_designer import MFMAParameters
from geko.config_generator.shared_utils import ConfigEntry, ForkParameter
from geko.schemas import GemmType


def _cfg(
    arch: str = "gfx950",
    dt: str = "H",
    dd: str | None = None,
    cd: str | None = None,
    trans_a: str = "N",
    trans_b: str = "N",
    search_space: str = "heuristic",
    streamk: bool = False,
    cms: bool = False,
) -> dict:
    dd = dd or dt
    cd = cd or ("S" if dt not in ("D", "Z") else dt)
    gt = GemmType.from_tensile(trans_a, trans_b, dt, dd, cd)
    return {
        "ARCH": arch,
        "CUs": 256 if arch.startswith("gfx950") else 304,
        "XCC": 8,
        "WGMUnit": 8,
        "StreamK": streamk,
        "CMS": cms,
        "search_space": search_space,
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
    }


def test_param_meta_helpers_and_loader_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pm._format_range(7) == "7"
    assert pm._format_range([1, 2, 3]) == "[1, 2, 3]"
    assert pm._format_range(list(range(10)), start_elements=2, end_elements=2) == "[0, 1, ..., 8, 9]"

    gp_mod = types.ModuleType("Tensile.Common.GlobalParameters")
    gp_mod.defaultBenchmarkCommonParameters = [{"A": 11}, {"B": 22}]
    vp_mod = types.ModuleType("Tensile.Common.ValidParameters")
    vp_mod.validParameters = {
        "A": [0, 1, 2, 3, 4, 5],
        "B": [7],
        "C": [1, 2, 3],
    }

    monkeypatch.setitem(sys.modules, "Tensile.Common.GlobalParameters", gp_mod)
    monkeypatch.setitem(sys.modules, "Tensile.Common.ValidParameters", vp_mod)

    pm.load_tensile_metadata.cache_clear()
    m1 = pm.load_tensile_metadata()
    m2 = pm.load_tensile_metadata()
    assert m1 is m2
    assert set(m1.keys()) == {"A", "B"}
    assert m1["A"].default_value == 11
    assert "..." in m1["A"].valid_range


def test_registry_and_fork_generator_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})

    c0 = _cfg(arch="gfx950", search_space="heuristic")
    assert isinstance(get_optimization_params(c0), g950_opt.GFX950Params)
    assert isinstance(get_post_processor(c0), g950_pp.GFX950PostProcessor)

    c1 = _cfg(arch="gfx950", search_space="generic")
    assert isinstance(get_optimization_params(c1), g950_opt.GFX950GAParams)
    assert isinstance(get_post_processor(c1), g950_pp.GFX950GAPostProcessor)

    c2 = _cfg(arch="gfx942", search_space="heuristic")
    assert isinstance(get_optimization_params(c2), g942_opt.GFX942Params)
    assert isinstance(get_post_processor(c2), g942_pp.GFX942PostProcessor)

    # Unknown search space falls back to heuristic registries.
    c3 = _cfg(arch="gfx942", search_space="unknown")
    assert isinstance(get_optimization_params(c3), g942_opt.GFX942Params)
    assert isinstance(get_post_processor(c3), g942_pp.GFX942PostProcessor)

    # Unknown arch -> no post processor.
    c4 = dict(c3)
    c4["ARCH"] = "gfx000"
    assert get_post_processor(c4) is None

    class _MI:
        def generate_for_size(self, _size):
            return [{"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[1, 2, 3])}]

    class _OPT:
        def generate_for_size(self, _size):
            return ({"DepthU": ForkParameter(name="DepthU", values=[16])}, [[{"X": ForkParameter(name="X", values=[1])}]])

    class _PP:
        def apply(self, fp, mi, _size):
            fp["Applied"] = ForkParameter(name="Applied", values=[1])
            return fp, mi

    monkeypatch.setattr(fpg, "count_kernels", lambda _fp: 7)
    fp0, nmis0, nk0 = fpg.generate_fork_params(_MI(), _OPT(), {}, (16, 16, 1, 16), post_processor=None)
    assert nmis0 == 1 and nk0 == 7 and "Groups" in fp0

    fp1, nmis1, nk1 = fpg.generate_fork_params(_MI(), _OPT(), {}, (16, 16, 1, 16), post_processor=_PP())
    assert nmis1 == 1 and nk1 == 7 and "Applied" in fp1


def test_post_processors_and_optimization_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})

    def _calc(mi):
        if mi.M == 16:
            return MFMAParameters(MT0=16, MT1=16, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1)
        return MFMAParameters(MT0=512, MT1=512, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1)

    monkeypatch.setattr(g950_pp.MIDesign, "calculate_mfma_parameters", _calc)
    monkeypatch.setattr(g942_pp.MIDesign, "calculate_mfma_parameters", _calc)

    groups = [
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[16, 16, 4, 1, 1, 2, 2, 1, 1])},
        {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[32, 32, 4, 2, 1, 2, 2, 2, 2])},
    ]

    p942 = g942_pp.GFX942PostProcessor(_cfg(arch="gfx942", dt="H", search_space="heuristic"))
    f942, g942 = p942.apply({}, list(groups), (64, 64, 1, 64))
    assert isinstance(f942, dict)
    assert "MIArchVgpr" in g942[1]

    p942ga = g942_pp.GFX942GAPostProcessor(_cfg(arch="gfx942", dt="H", search_space="generic"))
    _, g942ga = p942ga.apply({}, list(groups), (64, 64, 1, 64))
    assert "MIArchVgpr" in g942ga[1]

    monkeypatch.setattr(g950_pp, "load_CMS_groups", lambda *a, **k: [{"CMS": ForkParameter(name="CMS", values=[1])}])
    p950 = g950_pp.GFX950PostProcessor(_cfg(arch="gfx950", dt="H", cms=True, streamk=True, search_space="heuristic"))
    f950, g950 = p950.apply({"PrefetchGlobalRead": ForkParameter(name="PrefetchGlobalRead", values=[2]), "DepthU": ForkParameter(name="DepthU", values=[16, 32, 64]), "WorkGroupMapping": ForkParameter(name="WorkGroupMapping", values=[16])}, list(groups), (256, 64, 1, 4096))
    assert "UseCustomMainLoopSchedule" not in f950
    assert len(g950) >= 1

    p950ga = g950_pp.GFX950GAPostProcessor(_cfg(arch="gfx950", dt="H", cms=True, streamk=False, search_space="generic"))
    _, g950ga = p950ga.apply({}, list(groups), (64, 64, 1, 64))
    assert len(g950ga) >= 1

    # Optimization params: execute both heuristic and generic families with a few sizes
    # and dtype/layout combos to cover many branches compactly.
    profs = [
        g950_opt.GFX950Params(_cfg(arch="gfx950", dt="H", trans_a="N", trans_b="N", streamk=True, cms=True)),
        g950_opt.GFX950Params(_cfg(arch="gfx950", dt="D", trans_a="T", trans_b="N")),
        g950_opt.GFX950Params(_cfg(arch="gfx950", dt="C", cd="C", trans_a="N", trans_b="T")),
        g950_opt.GFX950GAParams(_cfg(arch="gfx950", dt="X", trans_a="N", trans_b="N", search_space="generic", streamk=True)),
        g950_opt.GFX950GAParams(_cfg(arch="gfx950", dt="H", trans_a="T", trans_b="N", search_space="generic")),
        g942_opt.GFX942Params(_cfg(arch="gfx942", dt="B", trans_a="N", trans_b="T", streamk=True)),
        g942_opt.GFX942Params(_cfg(arch="gfx942", dt="C", cd="C", trans_a="N", trans_b="T")),
        g942_opt.GFX942GAParams(_cfg(arch="gfx942", dt="X", trans_a="N", trans_b="N", search_space="generic", streamk=True)),
    ]
    sizes = [(4, 4, 1, 128), (128, 8192, 1, 16384), (4096, 256, 1, 8192)]
    for p in profs:
        for sz in sizes:
            try:
                params, groups_out = p.generate_for_size(sz)
            except ValueError:
                continue
            assert isinstance(params, dict)
            assert isinstance(groups_out, list)

    # Call inactive group helpers directly to mark them covered.
    p = g950_opt.GFX950GAParams(_cfg(arch="gfx950", dt="H", search_space="generic"))
    ctx = type("Ctx", (), {"M": 128, "N": 128, "B": 1, "K": 128})()
    assert len(p.tailloop_stagger_group(ctx)) >= 1
    assert len(p.extra_latency_dtv_group(ctx)) >= 1

    p2 = g942_opt.GFX942GAParams(_cfg(arch="gfx942", dt="H", search_space="generic"))
    assert len(p2.tailloop_stagger_group(ctx)) >= 1
    assert len(p2.extra_latency_dtv_group(ctx)) >= 1


def test_config_generator_orchestrators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_run_per_gemm_type = cg._run_per_gemm_type
    monkeypatch.setattr(cg, "build_tensilelite_client", lambda *_a, **_k: tmp_path / "client")

    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    gp0 = type("GP", (), {"gemm_type": gt, "sizes": [[16, 16, 1, 16]]})()
    gp1 = type("GP", (), {"gemm_type": gt, "sizes": [[32, 32, 1, 32]]})()

    calls = []
    monkeypatch.setattr(cg, "_run_per_gemm_type", lambda conf, *_a, **_k: calls.append(conf["GemmProblem"]))

    base_cfg = {"GemmProblems": [gp0, gp1], "BUILD_DIR": None}
    cg.run(base_cfg, tmp_path, tmp_path / "out", write_shell_scripts=True)
    cg.run(base_cfg, tmp_path, tmp_path / "out2", write_shell_scripts=False)
    assert len(calls) == 4
    monkeypatch.setattr(cg, "_run_per_gemm_type", real_run_per_gemm_type)

    # _fork_params_entry_for_size success + failure
    monkeypatch.setattr(cg, "generate_fork_params", lambda *_a, **_k: ({"A": ForkParameter(name="A", values=[1])}, 2, 3))
    c = {"GemmProblem": type("GP", (), {"gemm_type": gt})(), "LOG_LEVEL": 20}
    e = cg._fork_params_entry_for_size((16, 16, 1, 16), object(), object(), c, None)
    assert e.nkernels == 3 and e.mis_per_size[(16, 16, 1, 16)] == 2

    monkeypatch.setattr(cg, "generate_fork_params", lambda *_a, **_k: ({}, 0, 0))
    with pytest.raises(ValueError, match="No kernels found"):
        cg._fork_params_entry_for_size((16, 16, 1, 16), object(), object(), c, None)

    # _emit_entity_files path
    written = {}

    class _CSG:
        def build_config(self, *_a, **_k):
            return {"ok": 1}

        def generate_comment(self, _nk):
            return "# h\n"

    class _W:
        def write_entity_files_only(self, entry, built, header, entity_name):
            written[entity_name] = (entry, built, header)

    c2 = {"backend": "ductile", "CMS_PRIORITY": False, "MACROTILE_OPT": False, "LOG_LEVEL": 20}
    cg._emit_entity_files((3, e), _CSG(), _W(), "HHS_NN", c2)
    assert "HHS_NN_3" in written

    # _run_per_gemm_type with heavy dependencies patched out.
    entries = [
        ConfigEntry(
            sizes=[[16, 16, 1, 16]],
            fork_params={"Groups": ForkParameter(name="Groups", values=[[{"X": ForkParameter(name="X", values=[1])}]])},
            nkernels=4,
            mis_per_size={(16, 16, 1, 16): 2},
        ),
        ConfigEntry(
            sizes=[[32, 32, 1, 32]],
            fork_params={"Groups": ForkParameter(name="Groups", values=[[{"X": ForkParameter(name="X", values=[2])}]])},
            nkernels=5,
            mis_per_size={(32, 32, 1, 32): 3},
        ),
    ]

    monkeypatch.setattr(cg, "MIDesign", lambda *_a, **_k: object())
    monkeypatch.setattr(cg, "get_optimization_params", lambda _cfg: object())
    monkeypatch.setattr(cg, "get_post_processor", lambda _cfg: object())
    monkeypatch.setattr(cg, "do_cluster", lambda *_a, **_k: {0: [0], 1: [1]})
    monkeypatch.setattr(cg, "do_merge", lambda *_a, **_k: entries)

    class _CSG2:
        def __init__(self, _cfg):
            pass

        def build_config(self, entry, **_k):
            return {"entry": entry.nkernels}

        def generate_comment(self, nk):
            return f"# {nk}\n"

    app = []

    class _W2:
        def __init__(self, *_a, **_k):
            pass

        def write_entity_files_only(self, *_a, **_k):
            return None

        def append_aggregate_metadata(self, name, *_a, **_k):
            app.append(name)

    monkeypatch.setattr(cg, "ConfigSectionGenerator", _CSG2)
    monkeypatch.setattr(cg, "EntityOutputWriter", _W2)

    def _pf(fn, iterable):
        lst = list(iterable)
        if lst and isinstance(lst[0], tuple) and len(lst[0]) == 2 and isinstance(lst[0][0], int):
            for item in lst:
                fn(item)
            return None
        return entries

    monkeypatch.setattr(cg, "parallel_for", _pf)

    cfg_run = {
        "GemmProblem": gp0,
        "CUs": 256,
        "CLUSTER": 0,
        "ONE_SIZE_PER_CONFIG": False,
        "MAX_NUM_KERNELS_PER_CONFIG": 123,
        "CMS_PRIORITY": False,
        "MACROTILE_OPT": False,
        "backend": "ductile",
    }

    cg._run_per_gemm_type(cfg_run, tmp_path, tmp_path / "o3", write_shell_scripts=False, client_path=None)
    assert len(app) == 2


def test_gfx950_post_processor_cms_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _calc(_mi):
        return MFMAParameters(MT0=64, MT1=64, TT0=1, TT1=1, WG0=2, WG1=2, MIBlockM=1)

    monkeypatch.setattr(g950_pp.MIDesign, "calculate_mfma_parameters", _calc)

    # Cover GA CMS-disabled branch.
    pp = g950_pp.GFX950GAPostProcessor(_cfg(arch="gfx950", dt="H", search_space="generic", cms=False))
    fp, _ = pp.apply({}, [{"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[16, 16, 4, 1, 1, 2, 2, 1, 1])}], (64, 64, 1, 64))
    assert fp["UseCustomMainLoopSchedule"].values == [0]

    # Cover direct MI reconstruction helper.
    mi = g950_pp._reconstruct_matrix_instruction(
        {
            "MatrixInstruction": [16, 16, 4, 1],
            "MIWaveGroup": [2, 2],
            "MacroTile0": 64,
            "MacroTile1": 64,
        }
    )
    assert len(mi) == 9

    # Cover CMS loading path with injected Tensile modules.
    cs_mod = types.ModuleType("Tensile.Components.CustomSchedule")
    cs_mod.query_cms_kernels = lambda **_k: [
        {
            "MatrixInstruction": [16, 16, 4, 1],
            "MIWaveGroup": [2, 2],
            "MacroTile0": 64,
            "MacroTile1": 64,
            "DepthU": "32",
            "GlobalSplitU": "bad",  # force failed cast warning branch
            "NoSuchKey": 7,
        }
    ]
    vp_mod = types.ModuleType("Tensile.Common.ValidParameters")
    vp_mod.validParameters = {"DepthU": [1], "GlobalSplitU": [1]}
    monkeypatch.setitem(sys.modules, "Tensile.Components.CustomSchedule", cs_mod)
    monkeypatch.setitem(sys.modules, "Tensile.Common.ValidParameters", vp_mod)

    def _mk(name, values, **kwargs):
        return ForkParameter(
            name=name,
            values=values,
            comment=kwargs.get("comment", ""),
            metadata=kwargs.get("metadata"),
        )

    groups = g950_pp.load_CMS_groups("h", "N", "T", _mk, MT_DU=[64, 64, "32"])
    assert len(groups) == 1
    assert groups[0]["UseCustomMainLoopSchedule"].values == [1]
    assert "MatrixInstruction" in groups[0]

    # Unsupported dtype short-circuit.
    assert g950_pp.load_CMS_groups("q", "N", "N", _mk) == []
