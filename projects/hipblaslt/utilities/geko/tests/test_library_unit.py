# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from geko.bench.log import ADAPTIVE_ROW_KEYS
from geko.library import Library, LibraryCollection
from geko.library.library import SUPPORTED_INITIALIZATIONS


def _mk_lib(problem_overrides=None, sizes=None, solutions=None, name="lib.yaml") -> Library:
    problem = {
        "TransposeA": 0,
        "TransposeB": 1,
        "DataType": 4,
        "DestDataType": 4,
        "ComputeDataType": 0,
        "UseScaleAB": "Scalar",
        "HighPrecisionAccumulate": 1,
    }
    if problem_overrides:
        problem.update(problem_overrides)

    if solutions is None:
        solutions = [
            {
                "SolutionIndex": 0,
                "StaggerU": 0,
                "BaseName": "foo_UserArgs",
                "KernelNameMin": "bar_UserArgs",
                "SolutionNameMin": "baz_UserArgs",
            }
        ]
    if sizes is None:
        sizes = [[[16, 32, 1, 64], [0, 200.0]]]

    data = [
        None,
        None,
        "gfx950",
        None,
        problem,
        solutions,
        [2, 3, 0, 1],
        sizes,
        None,
        None,
        "DeviceEfficiency",
        "Equality",
    ]
    return Library(data, name)


def test_create_bench_input_rejects_invalid_initialization(tmp_path: Path) -> None:
    lib = _mk_lib()
    with pytest.raises(ValueError, match="Must be on of"):
        lib.create_bench_input(tmp_path, initialization="not_valid")


def test_create_bench_input_generates_bench_and_verify(tmp_path: Path) -> None:
    lib = _mk_lib()
    bench_file, verify_file = lib.create_bench_input(
        tmp_path,
        verify=True,
        initialization=SUPPORTED_INITIALIZATIONS[0],
        duration=0.01,
    )
    assert Path(bench_file).is_file()
    assert verify_file is not None
    assert Path(verify_file).is_file()


def test_create_bench_input_adaptive_by_default_and_verify_has_no_adaptive_keys(tmp_path: Path) -> None:
    lib = _mk_lib()
    bench_file, verify_file = lib.create_bench_input(tmp_path, verify=True, duration=0.5)
    bench_rows = yaml.safe_load(Path(bench_file).read_text())
    assert bench_rows[0]["adaptive"] is True
    assert bench_rows[0]["measure_time"] == pytest.approx(500.0)
    assert "iters" in bench_rows[0]

    verify_rows = yaml.safe_load(Path(verify_file).read_text())
    assert verify_rows[0]["iters"] == 1
    assert verify_rows[0]["cold_iters"] == 0
    for key in ADAPTIVE_ROW_KEYS:
        assert key not in verify_rows[0]


def test_create_bench_input_non_adaptive_has_no_adaptive_keys(tmp_path: Path) -> None:
    lib = _mk_lib()
    bench_file, _ = lib.create_bench_input(tmp_path, verify=False, duration=0.5, adaptive=False)
    rows = yaml.safe_load(Path(bench_file).read_text())
    for key in ADAPTIVE_ROW_KEYS:
        assert key not in rows[0]
    assert "iters" in rows[0]


def test_create_bench_input_no_compute_dtype_path(tmp_path: Path) -> None:
    lib = _mk_lib(problem_overrides={"DataType": 5, "DestDataType": 5})
    # Remove ComputeDataType to exercise fallback path.
    del lib.problem["ComputeDataType"]
    bench_file, verify_file = lib.create_bench_input(tmp_path, verify=False, initialization="rand_int")
    assert Path(bench_file).is_file()
    assert verify_file is None


def test_create_bench_input_mx_block_uses_arch_mx_scale(tmp_path: Path) -> None:
    lib = _mk_lib(problem_overrides={"MXBlockA": 32, "MXBlockB": 32})
    bench_file, _ = lib.create_bench_input(tmp_path, verify=False)
    gemms = yaml.safe_load(Path(bench_file).read_text())
    assert gemms[0]["scaleA"] == 1001  # gfx950 mx_scale
    assert gemms[0]["scaleB"] == 1001


def test_create_bench_input_mx_block_unknown_arch_uses_canonical_scale(tmp_path: Path) -> None:
    lib = _mk_lib(problem_overrides={"MXBlockA": 32, "MXBlockB": 32})
    lib.data[2] = "gfx_unknown"
    bench_file, _ = lib.create_bench_input(tmp_path, verify=False)
    gemms = yaml.safe_load(Path(bench_file).read_text())
    assert gemms[0]["scaleA"] == 3
    assert gemms[0]["scaleB"] == 3


def test_add_epilogues_updates_solution_names() -> None:
    lib = _mk_lib(problem_overrides={"DataType": 11, "DestDataType": 11, "UseScaleAB": "Vector"})
    lib.add_epilogues()
    assert lib.problem["Activation"] is True
    assert "BiasDataTypeList" in lib.problem
    assert "Bias" in lib.solutions[0]["BaseName"]


def _mk_i8_lib(compute=0, base="Cijk_Ailk_Bljk_I8II_BH_UserArgs_MT64x64") -> Library:
    # Shipped int8 libraries carry an empty (off) UseScaleAB.
    overrides = {"DataType": 8, "DestDataType": 8, "ComputeDataType": compute, "UseScaleAB": ""}
    solutions = [{"SolutionIndex": 0, "StaggerU": 0, "BaseName": base}]
    lib = _mk_lib(problem_overrides=overrides, solutions=solutions)
    if compute is None:
        del lib.problem["ComputeDataType"]
    return lib


def test_add_epilogues_i8_float_compute_has_bias() -> None:
    lib = _mk_i8_lib(compute=0)
    lib.add_epilogues()
    assert lib.problem["UseBias"] == 1
    assert lib.problem["BiasDataTypeList"] == [0]
    assert lib.solutions[0]["BaseName"] == "Cijk_Ailk_Bljk_I8II_BH_BiasS_HAS_SAV_UserArgs_MT64x64"
    assert lib.problem["UseScaleAB"] == ""


@pytest.mark.parametrize("compute", [6, None])
def test_add_epilogues_i8_int_or_missing_compute_has_no_bias(compute) -> None:
    lib = _mk_i8_lib(compute=compute)
    lib.add_epilogues()
    assert lib.problem["UseBias"] == 0
    assert lib.problem["BiasDataTypeList"] == []
    # TensileLite tags the activation with the compute type: int32 -> HAI.
    assert lib.solutions[0]["BaseName"] == "Cijk_Ailk_Bljk_I8II_BH_HAI_SAV_UserArgs_MT64x64"


def test_add_epilogues_replaces_stale_has_tag_with_hai() -> None:
    lib = _mk_i8_lib(compute=6, base="Cijk_I8II_BH_HAS_SAV_UserArgs_MT64x64")
    lib.add_epilogues()
    assert lib.solutions[0]["BaseName"] == "Cijk_I8II_BH_HAI_SAV_UserArgs_MT64x64"


def test_add_epilogues_strips_stale_int_bias_tag() -> None:
    lib = _mk_i8_lib(compute=0, base="Cijk_I8II_BH_BiasI_UserArgs_MT64x64")
    lib.add_epilogues()
    assert lib.solutions[0]["BaseName"] == "Cijk_I8II_BH_BiasS_HAS_SAV_UserArgs_MT64x64"


def test_add_epilogues_i8_name_update_is_idempotent() -> None:
    lib = _mk_i8_lib(compute=0)
    lib.add_epilogues()
    once = lib.solutions[0]["BaseName"]
    lib.add_epilogues()
    assert lib.solutions[0]["BaseName"] == once


def test_create_bench_input_i8_forces_rand_int(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lib = _mk_i8_lib(compute=0)
    with caplog.at_level("WARNING"):
        bench_file, _ = lib.create_bench_input(tmp_path, verify=False, initialization="trig_float")
    assert any("rand_int" in rec.getMessage() for rec in caplog.records)
    gemms = yaml.safe_load(Path(bench_file).read_text())
    assert gemms[0]["initialization"] == "rand_int"
    assert gemms[0]["a_type"] == "i8_r"
    assert gemms[0]["compute_type"] == "c_f32_r"


def test_trim_removes_duplicate_size_entries() -> None:
    sols = [
        {"SolutionIndex": 0, "StaggerU": 0},
        {"SolutionIndex": 1, "StaggerU": 0},
    ]
    sizes = [
        [[16, 16, 1, 16], [0, 50.0]],
        [[16, 16, 1, 16], [1, 100.0]],
    ]
    lib = _mk_lib(solutions=sols, sizes=sizes)
    lib.trim()
    assert len(lib.sizes) == 1
    assert len(lib.solutions) == 1
    assert lib.sizes[0][1][0] == 0


def test_library_setters_validate_types() -> None:
    lib = _mk_lib()
    with pytest.raises(TypeError, match="Must be a list"):
        lib.solutions = "bad"
    with pytest.raises(TypeError, match="Must be a list"):
        lib.order = "bad"
    with pytest.raises(TypeError, match="Must be a list"):
        lib.sizes = "bad"
    with pytest.raises(TypeError, match="Must be a string"):
        lib.metric = 1
    with pytest.raises(TypeError, match="Must be a string"):
        lib.type = 1


def test_library_collection_validation_and_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _mk_lib()

    with pytest.raises(TypeError, match="type 'list'"):
        LibraryCollection(libs="bad")
    with pytest.raises(TypeError, match="type 'Library'"):
        LibraryCollection(libs=["bad"])

    c = LibraryCollection([lib])
    assert len(c) == 1
    assert c[0] is lib
    assert list(iter(c))[0] is lib

    with pytest.raises(ValueError, match="type 'Library'"):
        c.append("bad")

    called = {"add": 0, "bench": 0, "trim": 0, "dump": 0}

    monkeypatch.setattr(lib, "add_epilogues", lambda: called.__setitem__("add", called["add"] + 1))
    monkeypatch.setattr(
        lib,
        "create_bench_input",
        lambda *_a, **_k: called.__setitem__("bench", called["bench"] + 1),
    )
    monkeypatch.setattr(lib, "trim", lambda: called.__setitem__("trim", called["trim"] + 1))

    def _fake_parallel(fn, seq):
        for item in seq:
            fn(item)
        return []

    monkeypatch.setattr("geko.library.library.parallel_for", _fake_parallel)
    monkeypatch.setattr(lib, "dump", lambda _out: called.__setitem__("dump", called["dump"] + 1))

    c.add_epilogues()
    c.create_bench_input(tmp_path)
    c.trim()
    c.dump(tmp_path)

    assert called == {"add": 1, "bench": 1, "trim": 1, "dump": 1}


def test_dict_library_supports_property_access_and_dump(tmp_path: Path) -> None:
    data = {
        "ArchitectureName": "gfx950",
        "ProblemType": {"TransposeA": 0, "TransposeB": 1, "DataType": 4, "DestDataType": 4},
        "DefaultSolution": {"StaggerU": 0},
        "Solutions": [{"SolutionIndex": 0, "StaggerU": 0}],
        "IndexOrder": [2, 3, 0, 1],
        "ExactLogic": [[[16, 32, 1, 64], [0, 200.0]]],
        "PerfMetric": "DeviceEfficiency",
        "LibraryType": "Equality",
    }

    lib = Library(data, "dict_lib.yaml")
    assert lib.format == "dict"
    assert lib.arch == "gfx950"
    assert lib.problem["DataType"] == 4
    assert lib.solutions[0]["SolutionIndex"] == 0
    assert lib.order == [2, 3, 0, 1]
    assert lib.sizes == [[[16, 32, 1, 64], [0, 200.0]]]
    assert lib.metric == "DeviceEfficiency"
    assert lib.type == "Equality"

    lib.solutions = [{"SolutionIndex": 1, "StaggerU": 1}]
    lib.order = [1, 0, 2, 3]
    lib.sizes = [[[8, 8, 1, 8], [1, 100.0]]]
    lib.metric = "OtherMetric"
    lib.type = "GridBased"

    out_dir = tmp_path / "out"
    lib.dump(out_dir)
    dumped = yaml.safe_load((out_dir / "dict_lib.yaml").read_text())
    assert dumped["Solutions"][0]["SolutionIndex"] == 1
    assert dumped["IndexOrder"] == [1, 0, 2, 3]
    assert dumped["ExactLogic"] == [[[8, 8, 1, 8], [1, 100.0]]]
    assert dumped["PerfMetric"] == "OtherMetric"
    assert dumped["LibraryType"] == "GridBased"


@pytest.mark.parametrize(
    "schedule_name,arch,expected",
    [
        ("gfx1250-strict", "gfx1250-strict", "gfx1250-strict"),
        ("gfx1250", "gfx1250", "gfx1250"),
        ("aquavanjaram", "gfx942", "gfx942"),
        (None, "gfx950", "gfx950"),
    ],
)
def test_schedule_is_an_architecture_create_library_accepts(schedule_name, arch, expected) -> None:
    lib = _mk_lib()
    lib.data[1] = schedule_name
    lib.data[2] = arch
    assert lib.schedule == expected


def test_schedule_rejects_retired_gfx1250v0_logic() -> None:
    """Old A0 logic would otherwise be built into the gfx1250 (B0) library."""
    lib = _mk_lib()
    lib.data[1] = "gfx1250v0"
    lib.data[2] = "gfx1250"
    with pytest.raises(ValueError, match="gfx1250-strict"):
        lib.schedule
