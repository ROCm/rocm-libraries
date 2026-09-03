# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from geko.config_generator import config_merger as cm
from geko.config_generator import output_writer as ow
from geko.config_generator import config_sections_generator as csg
from geko.config_generator.fork_params.hw_profiles.gfx942 import optimization_param as g942
from geko.config_generator.shared_utils import ConfigEntry, ForkParameter
from geko.schemas import GemmType


def _fp(name, values, active=True, comment="", metadata=None):
    return ForkParameter(name=name, values=values, active=active, comment=comment, metadata=metadata or {})


def _mk_entry(size, params, mis=None):
    return ConfigEntry(sizes=[list(size)], fork_params=params, nkernels=1, mis_per_size=mis or {tuple(size): 1})


def test_config_merger_primitives_and_trim() -> None:
    groups = _fp(
        "Groups",
        [[
            {"MatrixInstruction": _fp("MatrixInstruction", [1, 2, 3])},
            {"MatrixInstruction": _fp("MatrixInstruction", [4, 5, 6])},
        ]],
    )
    params = {"DepthU": _fp("DepthU", [16, 32]), "Groups": groups}
    assert cm._first_group_mi_count(groups) == 2
    assert cm._count_kernels_without_mi(params) == 2

    trimmed = cm._trim_mi_to_fit(params, max_kernels=2)
    assert trimmed <= 2
    assert len(params["Groups"].values[0]) == 1


def test_config_merger_group_key_and_merge_paths() -> None:
    e0 = {"A": _fp("A", [1]), "B": _fp("B", [2])}
    e1 = {"B": _fp("B", [2]), "A": _fp("A", [1])}
    assert cm._group_entry_key(e0) == cm._group_entry_key(e1)

    base = {"DepthU": _fp("DepthU", [16]), "Groups": _fp("Groups", [[e0]])}
    other = {"DepthU": _fp("DepthU", [32]), "Groups": _fp("Groups", [[e1, {"A": _fp("A", [3]), "B": _fp("B", [4])}]])}
    merged = cm._merge_two_param_dicts(base, other)
    assert merged["DepthU"].values == [16, 32]
    assert len(merged["Groups"].values[0]) == 2


def test_config_merger_cluster_split_and_do_merge(monkeypatch) -> None:
    p0 = {
        "DepthU": _fp("DepthU", [16]),
        "Groups": _fp("Groups", [[{"MatrixInstruction": _fp("MatrixInstruction", [1])}]]),
    }
    p1 = {
        "DepthU": _fp("DepthU", [32]),
        "Groups": _fp("Groups", [[{"MatrixInstruction": _fp("MatrixInstruction", [2])}]]),
    }
    entries = [_mk_entry((16, 16, 1, 16), p0), _mk_entry((32, 32, 1, 32), p1)]

    monkeypatch.setattr(cm, "count_kernels", lambda params: len(params.get("DepthU", _fp("x", [])).values) * len(params["Groups"].values[0]))
    out = cm.merge_sizes_in_cluster([0, 1], entries, max_kernels=1)
    assert len(out) == 2

    merged_all = cm.do_merge({0: [0], 1: [1]}, entries, max_kernels=1)
    assert len(merged_all) == 2


def test_output_writer_formatters_and_sections(tmp_path: Path) -> None:
    tw = ow.TuningConfigWriter()
    assert tw._format_scalar(True) == "true"
    assert tw._quote_if_string("x") == '"x"'
    assert tw._format_fork_values([1, "a"]) == '[1, "a"]'

    cfg = {
        "GlobalParameters": {"A": 1},
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM"},
                {
                    "InitialSolutionParameters": "",
                    "BenchmarkCommonParameters": [{"KernelLanguage": "[Assembly]"}],
                    "ForkParameters": {"DepthU": _fp("DepthU", [16]), "Groups": _fp("Groups", [[{"MatrixInstruction": _fp("MatrixInstruction", [1], comment="c")}]])},
                    "BenchmarkJoinParameters": "",
                    "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": "[ 16, 16, 1, 16 ]"}]}, {"BiasTypeArgs": "[S]"}],
                },
            ]
        ],
        "LibraryLogic": {"ScheduleName": '"gfx950"'},
        "#LibraryClient": "",
        "Backend": {"Name": "Ductile", "Config": {"soo": True, "weights": [{"group_0": "[1.0]"}]}},
    }
    out = tmp_path / "a.yaml"
    tw.write(cfg, "# hdr\n", out)
    txt = out.read_text(encoding="utf-8")
    assert "BenchmarkProblems:" in txt
    assert "ForkParameters:" in txt
    assert "weights:" in txt


def test_output_writer_scripts_and_orchestrator(tmp_path: Path) -> None:
    run_all = tmp_path / "run_all.sh"
    ow.init_run_all_script(run_all)
    ow.append_run_all(run_all, "x.sh", "1/2")
    assert "./x.sh" in run_all.read_text(encoding="utf-8")

    cfg_log = tmp_path / "Config_X.log"
    ow.write_config_log(cfg_log, "e1", 3, {(16, 16, 1, 16): 2})
    assert "#kernels 3" in cfg_log.read_text(encoding="utf-8")

    hip = tmp_path / "hip"
    (hip / "tensilelite/Tensile/bin").mkdir(parents=True)
    script = tmp_path / "e1.sh"
    ow.write_run_script(script, "e1", hip, client_path=hip / "client")
    assert script.is_file()

    writer = ow.EntityOutputWriter(tmp_path / "out", "HHS_NN", hip, write_shell_scripts=False)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    entry = _mk_entry((16, 16, 1, 16), {"DepthU": _fp("DepthU", [16]), "Groups": _fp("Groups", [[{"MatrixInstruction": _fp("MatrixInstruction", [1])}]])})
    config = {
        "GlobalParameters": {"A": 1},
        "BenchmarkProblems": [[{"OperationType": "GEMM"}, {"InitialSolutionParameters": "", "BenchmarkCommonParameters": [], "ForkParameters": entry.fork_params, "BenchmarkJoinParameters": "", "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": "[ 16, 16, 1, 16 ]"}]}]}]],
        "LibraryLogic": {"ScheduleName": '"gfx950"'},
        "Backend": {"Name": "Tensile"},
    }
    writer.write_entity_files_only(entry, config, "# h\n", "e1")
    writer.append_aggregate_metadata("e1", entry, progress="1/1")
    assert (tmp_path / "out/e1.yaml").is_file()
    assert (tmp_path / "Config_HHS_NN.log").is_file()


def _section_cfg(dtype="H", epilogues=True, backend="tensile", search_space="heuristic"):
    gt = GemmType.from_tensile("N", "N", dtype, dtype, "S" if dtype != "D" else "D")
    return {
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
        "ARCH": "gfx950",
        "CUs": 256,
        "XCC": 8,
        "EPILOGUES": epilogues,
        "backend": backend,
        "search_space": search_space,
        "SIZE_OPTION": 0,
    }


def test_config_sections_generator_paths(monkeypatch) -> None:
    gen = csg.ConfigSectionGenerator(_section_cfg(dtype="H", epilogues=True, backend="tensile", search_space="heuristic"))
    assert gen._is_tf32("X") is True
    assert gen._convert_type("X1") == "B"
    assert gen._calc_iters(16, 16, 1, 16) >= 5

    e = _mk_entry((16, 16, 1, 16), {"Groups": _fp("Groups", [[{"MatrixInstruction": _fp("MatrixInstruction", [1], metadata={"MT": (64, 64), "wave": (2, 2), "GSU": 1, "LSU": 1})}]])})
    cfg = gen.build_config(e, backend="tensile")
    assert "GlobalParameters" in cfg
    assert "BenchmarkProblems" in cfg

    # Ductile path with deterministic cost function inputs.
    gen_ga = csg.ConfigSectionGenerator(_section_cfg(dtype="H", epilogues=True, backend="ductile", search_space="generic"))
    cfg2 = gen_ga.build_config(e, backend="ductile", config_name="x", cms_priority=False, soo=True)
    assert cfg2["Backend"]["Name"] == "Ductile"
    assert "Config" in cfg2["Backend"]
    assert "weights" not in cfg2["Backend"]["Config"]

    # Multiple non-CMS groups should emit weights.
    e2 = _mk_entry(
        (32, 32, 1, 32),
        {
            "Groups": _fp(
                "Groups",
                [[
                    {"MatrixInstruction": _fp("MatrixInstruction", [1], metadata={"MT": (64, 64), "wave": (2, 2), "GSU": 1, "LSU": 1})},
                    {"MatrixInstruction": _fp("MatrixInstruction", [2], metadata={"MT": (128, 64), "wave": (2, 2), "GSU": 2, "LSU": 1})},
                ]],
            )
        },
    )
    cfg3 = gen_ga.build_config(e2, backend="ductile", config_name="x2", cms_priority=False, soo=True)
    assert "weights" in cfg3["Backend"]["Config"]
    assert "# Total #kernels" in gen_ga.generate_comment(10)

    # Invalid validation profile branch.
    bad = _section_cfg(dtype="H", epilogues=True, backend="ductile", search_space="generic")
    bad["DUCTILE_VALIDATION_PROFILE"] = 999
    gen_bad = csg.ConfigSectionGenerator(bad)
    with pytest.raises(ValueError, match="Invalid DUCTILE_VALIDATION_PROFILE"):
        gen_bad._build_ductile(e.fork_params, e.sizes, "x", False, False)


def test_gfx942_params_branches(monkeypatch) -> None:
    from geko.config_generator.fork_params import optimization_param as opt_param

    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})
    gt = GemmType.from_tensile("N", "T", "H", "H", "S")
    cfg = {
        "ARCH": "gfx942",
        "CUs": 304,
        "XCC": 8,
        "WGMUnit": 8,
        "StreamK": True,
        "CMS": True,
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
    }
    p = g942.GFX942Params(cfg)
    params, groups = p.generate_for_size((4096, 256, 1, 8192))
    assert "DepthU" in params
    assert "WorkGroupMapping" in params
    assert "StreamK" in params
    assert len(groups) >= 1

    ga = g942.GFX942GAParams(cfg)
    params_ga, groups_ga = ga.generate_for_size((256, 256, 1, 1024))
    assert "DepthU" in params_ga
    assert "GlobalReadVectorWidthA" in params_ga
    assert len(groups_ga) >= 1
