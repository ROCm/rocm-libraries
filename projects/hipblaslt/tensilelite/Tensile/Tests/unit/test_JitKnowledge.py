# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest
import yaml

from Tensile import JitKnowledge


pytestmark = pytest.mark.unit

PROBLEM = {"OperationType": "GEMM", "DataType": 7, "DestDataType": 7, "ComputeDataType": 0,
           "HighPrecisionAccumulate": True, "TransposeA": False, "TransposeB": True,
           "UseBeta": True, "Batched": True, "UseBias": 1, "BiasDataTypeList": [0, 7],
           "Activation": True, "ActivationType": "hipblaslt_all"}


def solution(index, **fields):
    s = {"SolutionIndex": index, "MatrixInstruction": [16, 16, 32, 1], "MIWaveTile": [2, 2],
         "MIWaveGroup": [2, 2], "MacroTile0": 64, "MacroTile1": 64, "DepthU": 64,
         "WorkGroupMapping": 6}
    s.update(fields)
    return s


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def dictLogic(solutions, rows, library="Equality", cuCount=80, defaults=None, problem=PROBLEM):
    return {"MinimumRequiredVersion": "5.0.0", "ScheduleName": "s", "ArchitectureName": "gfx942",
            "CUCount": cuCount, "DeviceNames": ["Device 74a1"], "ProblemType": problem,
            "DefaultSolution": defaults or {}, "Solutions": solutions,
            "IndexOrder": [0, 1, 2, 3], "ExactLogic": rows, "RangeLogic": None,
            "LibraryType": library}


def listLogic(solutions, rows):
    return [{"MinimumRequiredVersion": "4.33.0"}, "s", "gfx942", ["Device 74a1"], PROBLEM,
            solutions, [0, 1, 2, 3], rows, None, None, "DeviceEfficiency", "GridBased"]


@pytest.fixture
def logic(tmp_path):
    root = tmp_path / "aquavanjaram"
    name = "Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_UserArgs"
    # Legacy list layout with the retired Stream-K spelling.
    write(root, f"gfx942/GridBased/gfx942_{name}.yaml", listLogic(
        [solution(0, StreamK=5, StreamKXCCMapping=8, GlobalSplitU=0), solution(1, DepthU=32)],
        [[[512, 512, 1, 512, 512, 512, 512, 512], [0, 1.0]], [[64, 64, 1, 64], [1, 2.0]]]))
    # Dict layout: file defaults, a custom kernel, a duplicate and an assert.
    write(root, f"gfx942_80cu/Equality/gfx942_{name}.yaml", dictLogic(
        [solution(0), solution(1, CustomKernelName="hand_written"), solution(2),
         solution(3, AssertFree0ElementMultiple=8, WorkGroup=[16, 4, 4])],
        [[[128, 256, 2, 1024], [0, 1.0]], [[1, 1, 1, 1], [1, 1.0]], [[128, 128, 1, 128], [2, 1.0]],
         [[256, 256, 1, 256], [3, 1.0]]],
        defaults={"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}))
    write(root, f"gfx942/Range/gfx942_{name}.yaml", dictLogic(
        [solution(0)], [[[100, 200, 300, 500, 1, 1, 64, 128], [0, 1.0]]], library="Range",
        cuCount=None))
    write(root, f"gfx942/FreeSize/gfx942_{name}.yaml", dictLogic(
        [solution(0)], [], library="FreeSize", cuCount=None))
    return root


def extract(root, output):
    summary = JitKnowledge.extract(root, output, "gfx942", jobs=2)
    header, start = JitKnowledge.readHeader(output)
    return summary, header, start


def params(group, s):
    return dict(group["param_dictionary"][i] for i in s["params"])


def test_extracts_both_layouts_into_branch_groups(logic, tmp_path):
    output = tmp_path / "k.dat.zlib"
    summary, header, start = extract(logic, output)
    assert summary["logic"] == 3 and summary["skipped_solutions"] == 1
    assert header["generator"] == "tensilelite-logic-knowledge" and header["arch"] == "gfx942"
    assert header["library_arch"] == "gfx942" and len(header["content_hash"]) == 16
    assert header["branches"] == [{"kind": "cu", "cu_count": 80, "pci_ids": []},
                                  {"kind": "generic", "cu_count": None, "pci_ids": []}]
    cu, generic = header["index"]
    assert (cu["branch"], generic["branch"]) == (0, 1)
    assert cu["core_key"] == generic["core_key"] and cu["core_key"].startswith(
        "DataTypeA=7,DataTypeB=7,MacDataTypeA=7,MacDataTypeB=7,DestDataType=7,ComputeDataType=0,")
    assert cu["problem_type"] == {"name": "Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_UserArgs",
                                  "features": {"Bias": [0, 7], "Activation": True}}

    group = JitKnowledge.readGroup(output, start, cu)
    # The custom kernel's row is dropped and the duplicate shares its set.
    assert group["rows"] == [[128, 256, 2, 1024, 0, 0], [128, 128, 1, 128, 0, 0],
                             [256, 256, 1, 256, 1, 0]]
    first, asserted = group["sets"]
    assert first["gsu"] == -1 and first["gsu_algorithm"] == "MultipleBufferSingleKernel"
    assert first["policy"] == {"strategy": "None", "assignment": "StaticGrid"}
    assert first["macro_tile"] == [64, 64] and first["waves"] == [2, 2]
    assert first["instruction"] == [16, 16, 32, 1] and first["depth_u"] == 64
    assert first["source"] == {"file": "gfx942_80cu/Equality/gfx942_"
                               "Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_UserArgs.yaml", "index": 0}
    assert first["asserts"] == {} and asserted["asserts"] == {"AssertFree0ElementMultiple": 8}
    p = params(group, first)
    assert p["MatrixInstruction"] == [16, 16, 32, 1, 1, 2, 2, 2, 2] and p["WorkGroupMapping"] == 6
    assert p["GlobalSplitU"] == -1 and p["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel"
    assert "AssertFree0ElementMultiple" not in params(group, asserted)
    # WorkGroup's third value is the tuned LocalSplitU.
    assert params(group, asserted)["WorkGroup"] == [16, 4, 4]

    group = JitKnowledge.readGroup(output, start, generic)
    # GridBased takes the first four sizes; Range the middle of each range.
    assert [r[:4] + r[5:] for r in group["rows"]] == [
        [512, 512, 1, 512, 1], [64, 64, 1, 64, 1], [150, 400, 1, 96, 2]]
    streamK = group["sets"][group["rows"][0][4]]
    assert streamK["policy"] == {"strategy": "StreamK", "assignment": "Hybrid"}
    assert streamK["gsu"] == 0
    p = params(group, streamK)
    assert p["TileProcessingStrategy"] == "StreamK" and p["WorkAssignment"] == "Hybrid"
    assert p["PersistentXCCMapping"] == 8
    assert not {"StreamK", "StreamKXCCMapping", "_PersistentLoop"} & set(p)


def test_index_offsets_round_trip(logic, tmp_path):
    output = tmp_path / "k.dat.zlib"
    _, header, start = extract(logic, output)
    offset = 0
    for entry in header["index"]:
        assert entry["offset"] == offset
        group = JitKnowledge.readGroup(output, start, entry)
        assert len(group["rows"]) == entry["rows"] and len(group["sets"]) == entry["sets"]
        assert all(0 <= row[4] < entry["sets"] for row in group["rows"])
        offset += entry["length"]
    assert os.path.getsize(output) == start + offset


def test_output_is_deterministic_and_the_hash_follows_content(logic, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    extract(logic, first)
    extract(logic, second)
    assert first.read_bytes() == second.read_bytes()
    write(logic, "gfx942_80cu/Equality/gfx942_Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_UserArgs.yaml",
          dictLogic([solution(0)], [[[128, 256, 2, 2048], [0, 1.0]]]))
    _, changed, _ = extract(logic, tmp_path / "c")
    assert changed["content_hash"] != JitKnowledge.readHeader(first)[0]["content_hash"]


def test_problem_types_with_other_epilogues_are_other_groups(tmp_path):
    root = tmp_path / "logic"
    plain = dict(PROBLEM, UseBias=0, Activation=False, ActivationType="none",
                 UseScaleAB="Vector")
    write(root, "gfx942/Equality/gfx942_a.yaml", dictLogic([solution(0)], [[[8, 8, 1, 8], [0, 1.0]]],
                                                          cuCount=None))
    write(root, "gfx942/Equality/gfx942_b.yaml", dictLogic([solution(0)], [[[8, 8, 1, 8], [0, 1.0]]],
                                                          cuCount=None, problem=plain))
    _, header, _ = extract(root, tmp_path / "k")
    assert sorted(e["problem_type"]["features"] == {"UseScaleAB": "Vector"}
                  for e in header["index"]) == [False, True]


def test_files_with_only_skipped_solutions_add_no_group(tmp_path):
    root = tmp_path / "logic"
    plain = dict(PROBLEM, UseBias=0, Activation=False, ActivationType="none")
    write(root, "gfx942/Equality/gfx942_a.yaml", dictLogic([solution(0)], [[[8, 8, 1, 8], [0, 1.0]]],
                                                          cuCount=None))
    write(root, "gfx942_80cu/Equality/gfx942_b.yaml", dictLogic(
        [solution(0, CustomKernelName="hand_written")], [[[8, 8, 1, 8], [0, 1.0]]], problem=plain))
    summary, header, _ = extract(root, tmp_path / "k")
    assert summary["skipped_solutions"] == 1 and len(header["branches"]) == 1
    assert [e["problem_type"]["features"] for e in header["index"]] == [
        {"Bias": [0, 7], "Activation": True}]


def test_rejects_other_files(tmp_path):
    bad = tmp_path / "bad"
    bad.write_bytes(b"HJKN" + (2).to_bytes(4, "little") + (0).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="schema 2"):
        JitKnowledge.readHeader(bad)
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="not a JIT knowledge file"):
        JitKnowledge.readHeader(bad)
