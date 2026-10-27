# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent payload compatibility at generation and prebuilt ingestion boundaries."""

from copy import deepcopy
import json

import pytest
import yaml

from tensilelite.Contractions import SizeMapping
from tensilelite.CustomKernels import getCustomKernelConfig, validateCustomPersistentArgs
from tensilelite.ExecutionPolicy import normalize_execution_policy
from tensilelite.SolutionStructs.Naming import getKernelFileBase

pytestmark = pytest.mark.unit


def _persistent_args_config(layout=1):
    config = {
        "TileProcessingStrategy": "DataParallel",
        "WorkAssignment": "StaticGrid",
        "InternalSupportParams": {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 1},
        "CustomKernel": {
            "name": "data_parallel_args_v1",
            "args": [
                {"type": "uint32", "semantic": "ItersPerTile"},
                {"type": "uint32", "semantic": "PersistentGrid"},
            ],
        },
    }

    if layout == 2:
        config.update(TileProcessingStrategy="StreamK", ClusterDim=[4, 2])
        config["InternalSupportParams"]["PersistentLoopArgsVersion"] = 2
        config["CustomKernel"]["args"] = [{"type": "address", "semantic": "AddressFlags"}] + [
            {"type": "uint32", "semantic": semantic} for semantic in (
                "ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                "SKItersPerWG", "SKGrid", "SKTilesAndSplit")
        ] + [{"type": "address", "semantic": "AddressWorkspace"}]
    return config


def _write_custom(tmp_path, config, name="prebuilt_dp"):
    payload = yaml.safe_dump({"custom.config": config}, sort_keys=False)
    (tmp_path / (name + ".s")).write_text(".amdgpu_metadata\n---\n" + payload + "...\n.end_amdgpu_metadata\n")
    return name


@pytest.mark.parametrize("outer", [0, 1, 2, 3])
def test_missing_prebuilt_layout_is_legacy_even_when_consumer_uses_v1(tmp_path, outer):
    config = _persistent_args_config()
    config["InternalSupportParams"] = {"KernArgsVersion": outer}
    config["CustomKernel"]["args"] = [
        {"type": "uint32", "semantic": semantic}
        for semantic in ["ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                         "SKItersPerWG", "SKGrid", "SKTilesAndSplit"]
    ]
    name = _write_custom(tmp_path, config)
    result = getCustomKernelConfig(name, {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 1}, str(tmp_path))
    assert result["InternalSupportParams"] == {"KernArgsVersion": outer, "PersistentLoopArgsVersion": 0}
    assert result["CustomKernel"]["args"] == config["CustomKernel"]["args"]
    assert getKernelFileBase(False, result) == name
    assert normalize_execution_policy(result)["InternalSupportParams"] == result["InternalSupportParams"]


@pytest.mark.parametrize("layout,workspace", [(1, None), (2, "None"), (2, "StreamKWithReduction")])
def test_custom_descriptor_round_trips_without_reordering(tmp_path, layout, workspace):
    config = _persistent_args_config(layout)
    if workspace is not None:
        config["CustomKernel"]["workspaceType"] = workspace
    before = deepcopy(config)
    validateCustomPersistentArgs(config)
    assert config == before
    config["CustomKernel"]["args"].insert(0, {"type": "address", "semantic": "AddressA"})
    config["CustomKernel"]["args"].append({"type": "float", "semantic": "Alpha"})
    name = _write_custom(tmp_path, config)
    result = getCustomKernelConfig(name, {"PersistentLoopArgsVersion": 0}, str(tmp_path))
    assert result["InternalSupportParams"]["PersistentLoopArgsVersion"] == layout
    assert result["CustomKernel"]["args"] == config["CustomKernel"]["args"]
    assert getKernelFileBase(False, result) == name


@pytest.mark.parametrize("outer", [0, 1, 2, 3])
def test_prebuilt_streamk_cluster_keeps_legacy_metadata_and_assembly(tmp_path, outer):
    from types import SimpleNamespace
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    config = _persistent_args_config()
    config.update(TileProcessingStrategy="StreamK", ClusterDim=[2, 1], ISA=[12, 5, 0])
    config["InternalSupportParams"] = {"KernArgsVersion": outer}
    config["CustomKernel"]["args"] = [
        {"type": "uint32", "semantic": semantic}
        for semantic in ["ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                         "SKItersPerWG", "SKGrid", "SKTilesAndSplit"]
    ]
    name = _write_custom(tmp_path, config, "prebuilt_streamk_cluster")
    loaded = getCustomKernelConfig(name, {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 2}, str(tmp_path))
    state = normalize_execution_policy(loaded)
    assert state["InternalSupportParams"] == {"KernArgsVersion": outer, "PersistentLoopArgsVersion": 0}
    assert state["CustomKernel"]["args"] == config["CustomKernel"]["args"]
    assert getKernelFileBase(False, state) == name

    class Kernel(dict):
        duplicate = False

    writer = SimpleNamespace(debugConfig=SimpleNamespace(splitGSU=False),
                             assembler=SimpleNamespace(rocm_version=SimpleNamespace(major=7, patch=0)),
                             states=SimpleNamespace())
    writer._getCustomKernelSource = lambda kernel: KernelWriterAssembly._getCustomKernelSource(
        writer, kernel, str(tmp_path))
    # A geometry-only test must never send this prebuilt payload into the
    # source generator, which would switch it to the spatial-cluster schedule.
    result, source = KernelWriterAssembly.getSourceFileString(writer, Kernel(state))
    assert result == 0
    assert source == (tmp_path / (name + ".s")).read_text()


@pytest.mark.parametrize("layout", [1, 2])
@pytest.mark.parametrize("mutation", ["wide", "padding", "separated", "reverse", "duplicate", "legacy", "workspace"])
def test_custom_descriptor_rejects_incompatible_payload(layout, mutation):
    config = _persistent_args_config(layout)
    args = config["CustomKernel"]["args"]
    first = int(layout == 2)  # ABI2 puts AddressFlags before the scheduling block.
    if mutation == "wide":
        args[first + 1]["type"] = "uint64"
    elif mutation == "padding":
        args[first]["padding"] = 4
    elif mutation == "separated":
        args.insert(first + 1, {"type": "float", "semantic": "Alpha"})
    elif mutation == "reverse":
        args[first], args[first + 1] = args[first + 1], args[first]
    elif mutation == "duplicate":
        args.append(deepcopy(args[first + 1]))
    elif mutation == "legacy":
        args.append({"type": "uint32", "semantic": "TotalIters" if layout == 2 else "SKItersPerWG"})
    elif mutation == "workspace":
        if layout == 2:
            config["CustomKernel"]["workspaceType"] = "GSU"
        else:
            config["CustomKernel"]["workspaceSizePerElemC"] = 4
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(config)


@pytest.mark.parametrize("mutation", ["missing_grid", "missing_flags", "duplicate_flags", "wide_flags", "dp_grid", "old_outer", "missing_descriptor"])
def test_cluster_custom_descriptor_rejects_incompatible_scheduling(mutation):
    config = _persistent_args_config(2)
    args = config["CustomKernel"]["args"]
    if mutation == "missing_grid":
        del args[5]
    elif mutation == "missing_flags":
        del args[0]
    elif mutation == "duplicate_flags":
        args.append(deepcopy(args[0]))
    elif mutation == "wide_flags":
        args[0]["type"] = "uint64"
    elif mutation == "dp_grid":
        args[5]["semantic"] = "PersistentGrid"
    elif mutation == "old_outer":
        config["InternalSupportParams"]["KernArgsVersion"] = 2
    elif mutation == "missing_descriptor":
        del config["CustomKernel"]
        config["CustomKernelName"] = "external_cluster"
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(config)


def test_data_parallel_custom_claim_requires_descriptor_and_version():
    config = _persistent_args_config()
    config["InternalSupportParams"]["PersistentLoopArgsVersion"] = 0
    with pytest.raises(ValueError, match="PersistentGrid requires"):
        validateCustomPersistentArgs(config)
    config = _persistent_args_config()
    del config["CustomKernel"]
    config["CustomKernelName"] = "data_parallel_args_v1"
    with pytest.raises(ValueError, match="argument descriptor"):
        validateCustomPersistentArgs(config)


@pytest.mark.parametrize("semantic", ["AddressSynchronizer", "Synchronizer", "GSUSync"])
def test_data_parallel_custom_descriptor_rejects_partial_flag_aliases(semantic):
    config = _persistent_args_config()
    config["CustomKernel"]["args"].append({"type": "address", "semantic": semantic})
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(config)


@pytest.mark.parametrize("version", [-1, 3, 99, True, "1"])
@pytest.mark.parametrize("regenerate", [False, True])
def test_unknown_layout_rejected_before_regeneration(version, regenerate):
    config = _persistent_args_config()
    config.pop("CustomKernel")
    config["InternalSupportParams"]["PersistentLoopArgsVersion"] = version
    with pytest.raises(ValueError, match="Unsupported PersistentLoopArgsVersion"):
        normalize_execution_policy(config, regenerate=regenerate)


@pytest.mark.parametrize("outer", [0, 1, 2, 3])
def test_generated_legacy_dp_upgrades_both_layout_versions(outer):
    config = {"StreamK": 3, "StreamKForceDPOnly": 1,
              "InternalSupportParams": {"KernArgsVersion": outer},
              "AssignedDerivedParameters": True,
              "AssignedProblemIndependentDerivedParameters": True}
    prebuilt = normalize_execution_policy(config, regenerate=False)
    regenerated = normalize_execution_policy(config)
    assert prebuilt["InternalSupportParams"] == {"KernArgsVersion": outer, "PersistentLoopArgsVersion": 0}
    assert regenerated["InternalSupportParams"] == {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 1}
    assert regenerated["AssignedDerivedParameters"] is False
    assert regenerated["AssignedProblemIndependentDerivedParameters"] is False


@pytest.mark.parametrize("outer", [-1, 4, 99, True, "3"])
def test_unknown_outer_version_rejected_before_data_parallel_regeneration(outer):
    config = _persistent_args_config()
    config.pop("CustomKernel")
    config["InternalSupportParams"] = {"KernArgsVersion": outer, "PersistentLoopArgsVersion": 0}
    with pytest.raises(ValueError, match="KernArgsVersion"):
        normalize_execution_policy(config)


@pytest.mark.parametrize("outer", [0, 1, 2])
def test_data_parallel_payload_rejects_legacy_outer_version(outer):
    config = _persistent_args_config()
    config["InternalSupportParams"]["KernArgsVersion"] = outer
    with pytest.raises(ValueError, match="KernArgsVersion"):
        normalize_execution_policy(config, regenerate=False)


def test_legacy_disabled_atomic_option_does_not_survive_size_mapping():
    from test_streamk_force_dp_only import minimal_size_mapping_state

    state = minimal_size_mapping_state()
    state.update(StreamK=0, StreamKAtomic=1)
    mapping = SizeMapping.FromOriginalState(state)
    assert mapping.tileProcessingStrategy == "None"
    assert mapping.streamKAtomic == 0
    assert state["StreamKAtomic"] == 1


@pytest.mark.parametrize("declared,regenerate,expected", [(0, False, 0), (2, False, 2), (0, True, 2)])
def test_streamk_cluster_serializes_its_scheduling_abi(declared, regenerate, expected):
    from tensilelite.Common import state as serialized_state
    from tensilelite.Common.GlobalParameters import defaultInternalSupportParams
    from tensilelite.Contractions import InternalArgsSupport
    from test_streamk_force_dp_only import minimal_size_mapping_state

    source = minimal_size_mapping_state()
    source.update(ClusterDim=[2, 1], InternalSupportParams=dict(defaultInternalSupportParams))
    source["InternalSupportParams"].update(KernArgsVersion=3, PersistentLoopArgsVersion=declared)
    normalized = normalize_execution_policy(source, regenerate=regenerate)
    mapping = serialized_state(SizeMapping.FromOriginalState(normalized))
    support = serialized_state(InternalArgsSupport.FromOriginalState(normalized))
    assert mapping["tileProcessingStrategy"] == "StreamK"
    assert mapping["workAssignment"] == "StaticGrid"
    assert support["persistentLoopArgsVersion"] == expected


@pytest.mark.parametrize("strategy,use_beta,initial_strides", [
    ("DataParallel", False, False), ("DataParallel", False, True),
    ("DataParallel", True, False), ("DataParallel", True, True),
    ("StreamK", True, False),
])
def test_generated_signature_descriptor_and_reader_agree(tmp_path, record_property, strategy, use_beta, initial_strides):
    from config_harness import _emit_one, _isolated_globals_with_isa, _toolchain_for, _solutions_from_config_unguarded
    from test_persistent_config_generation import _config
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    config = _config({"TileProcessingStrategy": [strategy], "WorkAssignment": ["StaticGrid"]})
    if strategy == "StreamK":
        from pathlib import Path
        fixture = Path(__file__).parents[1] / "common/streamk/gfx1250/static_streamk_cluster_multicast_boundaries.yaml"
        config = yaml.safe_load(fixture.read_text())
        config["BenchmarkProblems"][0] = config["BenchmarkProblems"][0][:2]
        fork = config["BenchmarkProblems"][0][1]["ForkParameters"]
        fork[:] = [entry for entry in fork if "ClusterDim" not in entry] + [{"ClusterDim": [[2, 2]]}]
    config["BenchmarkProblems"][0][0].update(UseBeta=use_beta, UseInitialStridesAB=initial_strides,
                                           UseInitialStridesCD=initial_strides)
    config_path = tmp_path / "persistent.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    assembler, isa = _toolchain_for("gfx1250" if strategy == "StreamK" else "gfx942")
    with _isolated_globals_with_isa(isa):
        solutions = _solutions_from_config_unguarded(config_path, assembler, isa, limit_solutions=1)
        assert len(solutions) == 1
        kernel = generateKernelObjectsFromSolutions(solutions)[0]
        writer = KernelWriterAssembly(assembler, DebugConfig())
        name, source, error = _emit_one(writer, kernel, False, False)
        assert error == 0
        generated = writer.kernelArgDefs
    (tmp_path / (name + ".s")).write_text(source)
    loaded = getCustomKernelConfig(name, {}, str(tmp_path))
    assert loaded["InternalSupportParams"]["KernArgsVersion"] == 3
    assert loaded["InternalSupportParams"]["PersistentLoopArgsVersion"] == (2 if strategy == "StreamK" else 1)
    assert loaded["TileProcessingStrategy"] == strategy
    if strategy == "DataParallel":
        assert loaded["CustomKernel"]["args"] == generated
    else:
        # StreamK ingestion derives the descriptor from the finalized signature;
        # its advisory writer list predates the TDM address ordering.
        generated = loaded["CustomKernel"]["args"]
    metadata = yaml.safe_load(source.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])
    arguments = metadata["amdhsa.kernels"][0][".args"]
    assert len(arguments) == len(generated)
    sizes = {"address": 8, "uint32": 4, "uint64": 8, "float32": 4}
    offset = 0
    for argument, descriptor in zip(arguments, generated):
        assert argument[".offset"] == offset
        assert argument[".size"] == sizes[descriptor["type"]]
        offset += argument[".size"]
    names = [arg[".name"] for arg in arguments]
    first = names.index("ItersPerTile")
    scheduling = (["ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                   "SKItersPerWG", "skGrid", "skTiles"] if strategy == "StreamK"
                  else ["ItersPerTile", "PersistentGrid"])
    assert names[first:first + len(scheduling)] == scheduling
    if strategy == "StreamK":
        assert loaded["CustomKernel"]["grid"] == ["PersistentGrid", "One", "One"]
        assert [arg["semantic"] for arg in generated[first:first + 6]] == [
            "ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
            "SKItersPerWG", "SKGrid", "SKTilesAndSplit"]
        assert any(arg["semantic"] == "AddressFlags" for arg in generated)
    scalar = first + len(scheduling)
    assert names[scalar:scalar + 2] == ["alpha", "beta" if use_beta else "betapad"]
    assert names[-4:] == ["batchOffsetD", "batchOffsetC", "batchOffsetA", "batchOffsetB"]
    assert offset == metadata["amdhsa.kernels"][0][".kernarg_segment_size"]
    record_property("arguments", json.dumps(arguments))
    record_property("descriptor", json.dumps(generated))

    if strategy == "DataParallel" and use_beta and not initial_strides:
        # Exercise a packed C1 group at the signature boundary without asking
        # solution derivation to accept a fabricated contraction geometry.
        from tensilelite.Components.Signature import SignatureDefault
        from tensilelite.CustomKernels import _metadataArgToCustomArg

        packed = writer.states.kernel
        packed["PackedC1IdxChars"] = ["J", "K"]
        packed["PackedC1IndicesX"] = [1, 2]
        writer.kernelArgDefs = []
        writer._registerKernelArgs(packed)
        packed_source = str(SignatureDefault()(writer))
        packed_metadata = yaml.safe_load(packed_source.split(".amdgpu_metadata", 1)[1]
                                        .split(".end_amdgpu_metadata", 1)[0])
        packed_arguments = packed_metadata["amdhsa.kernels"][0][".args"]
        inferred = [_metadataArgToCustomArg(argument) for argument in packed_arguments]
        assert inferred == writer.kernelArgDefs
        magic = [argument for argument in packed_arguments if argument[".name"].endswith("SizeJ")]
        assert [argument[".name"] for argument in magic] == ["MagicNumberSizeJ", "MagicShiftSizeJ"]
        assert magic[1][".offset"] == magic[0][".offset"] + 4
        assert packed_arguments[-4][".offset"] == arguments[-4][".offset"] + 8


def test_codegen_rejects_invalid_data_parallel_descriptor(monkeypatch, tmp_path):
    from config_harness import _isolated_globals_with_isa, _toolchain_for, _solutions_from_config_unguarded
    from test_persistent_config_generation import _config
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    config = _config({"TileProcessingStrategy": ["DataParallel"], "WorkAssignment": ["StaticGrid"]})
    path = tmp_path / "data_parallel.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    register = KernelWriterAssembly._registerKernelArgs

    def omit_grid(writer, kernel):
        register(writer, kernel)
        writer.kernelArgDefs = [arg for arg in writer.kernelArgDefs
                                if arg["semantic"] != "PersistentGrid"]

    monkeypatch.setattr(KernelWriterAssembly, "_registerKernelArgs", omit_grid)
    assembler, isa = _toolchain_for("gfx942")
    with _isolated_globals_with_isa(isa):
        solutions = _solutions_from_config_unguarded(path, assembler, isa, limit_solutions=1)
        assert len(solutions) == 1
        kernel = generateKernelObjectsFromSolutions(solutions)[0]
        writer = KernelWriterAssembly(assembler, DebugConfig())
        with pytest.raises(ValueError, match="scheduling arguments"):
            writer._getKernelSource(kernel)
