# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent payload compatibility at generation and prebuilt ingestion boundaries."""

from copy import deepcopy
import json

import pytest
import yaml

from Tensile.Contractions import SizeMapping
from Tensile.CustomKernels import getCustomKernelConfig, validateCustomPersistentArgs
from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.SolutionStructs.Naming import getKernelFileBase

pytestmark = pytest.mark.unit


def _native_config():
    return {
        "TileProcessingStrategy": "DataParallel",
        "WorkAssignment": "StaticGrid",
        "InternalSupportParams": {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 1},
        "CustomKernel": {
            "name": "native_dp",
            "args": [
                {"type": "uint32", "semantic": "ItersPerTile"},
                {"type": "uint32", "semantic": "PersistentGrid"},
            ],
        },
    }


def _write_custom(tmp_path, config, name="prebuilt_dp"):
    payload = yaml.safe_dump({"custom.config": config}, sort_keys=False)
    (tmp_path / (name + ".s")).write_text(".amdgpu_metadata\n---\n" + payload + "...\n.end_amdgpu_metadata\n")
    return name


@pytest.mark.parametrize("outer", [0, 1, 2, 3])
def test_missing_prebuilt_layout_is_legacy_even_when_consumer_is_native(tmp_path, outer):
    config = _native_config()
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


def test_native_custom_descriptor_round_trips_without_reordering(tmp_path):
    config = _native_config()
    config["CustomKernel"]["args"].insert(0, {"type": "address", "semantic": "AddressA"})
    config["CustomKernel"]["args"].append({"type": "float", "semantic": "Alpha"})
    name = _write_custom(tmp_path, config)
    result = getCustomKernelConfig(name, {"PersistentLoopArgsVersion": 0}, str(tmp_path))
    assert result["InternalSupportParams"]["PersistentLoopArgsVersion"] == 1
    assert result["CustomKernel"]["args"] == config["CustomKernel"]["args"]
    assert getKernelFileBase(False, result) == name


@pytest.mark.parametrize("mutation", ["wide", "padding", "separated", "reverse", "duplicate", "legacy", "workspace"])
def test_native_custom_descriptor_rejects_incompatible_payload(mutation):
    config = _native_config()
    args = config["CustomKernel"]["args"]
    if mutation == "wide":
        args[1]["type"] = "uint64"
    elif mutation == "padding":
        args[0]["padding"] = 4
    elif mutation == "separated":
        args.insert(1, {"type": "float", "semantic": "Alpha"})
    elif mutation == "reverse":
        args.reverse()
    elif mutation == "duplicate":
        args.append(deepcopy(args[1]))
    elif mutation == "legacy":
        args.append({"type": "uint32", "semantic": "SKItersPerWG"})
    elif mutation == "workspace":
        config["CustomKernel"]["workspaceSizePerElemC"] = 4
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(config)


def test_native_custom_claim_requires_descriptor_and_version():
    config = _native_config()
    config["InternalSupportParams"]["PersistentLoopArgsVersion"] = 0
    with pytest.raises(ValueError, match="PersistentGrid requires"):
        validateCustomPersistentArgs(config)
    config = _native_config()
    del config["CustomKernel"]
    config["CustomKernelName"] = "native_dp"
    with pytest.raises(ValueError, match="argument descriptor"):
        validateCustomPersistentArgs(config)


@pytest.mark.parametrize("semantic", ["AddressSynchronizer", "Synchronizer", "GSUSync"])
def test_native_custom_descriptor_rejects_partial_flag_aliases(semantic):
    config = _native_config()
    config["CustomKernel"]["args"].append({"type": "address", "semantic": semantic})
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(config)


@pytest.mark.parametrize("version", [-1, 2, 99, True, "1"])
@pytest.mark.parametrize("regenerate", [False, True])
def test_unknown_layout_rejected_before_regeneration(version, regenerate):
    config = _native_config()
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
def test_unknown_outer_version_rejected_before_native_regeneration(outer):
    config = _native_config()
    config.pop("CustomKernel")
    config["InternalSupportParams"] = {"KernArgsVersion": outer, "PersistentLoopArgsVersion": 0}
    with pytest.raises(ValueError, match="KernArgsVersion"):
        normalize_execution_policy(config)


@pytest.mark.parametrize("outer", [0, 1, 2])
def test_native_payload_rejects_legacy_outer_version(outer):
    config = _native_config()
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


@pytest.mark.parametrize("use_beta", [False, True])
@pytest.mark.parametrize("initial_strides", [False, True])
def test_native_generated_signature_descriptor_and_reader_agree(tmp_path, record_property, use_beta, initial_strides):
    from config_harness import _emit_one, _isolated_globals_with_isa, _toolchain_for, _solutions_from_config_unguarded
    from test_persistent_config_generation import _config
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    config = _config({"TileProcessingStrategy": ["DataParallel"], "WorkAssignment": ["StaticGrid"]})
    config["BenchmarkProblems"][0][0].update(UseBeta=use_beta, UseInitialStridesAB=initial_strides,
                                           UseInitialStridesCD=initial_strides)
    config_path = tmp_path / "native.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    assembler, isa = _toolchain_for("gfx942")
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
    assert loaded["InternalSupportParams"]["PersistentLoopArgsVersion"] == 1
    assert loaded["TileProcessingStrategy"] == "DataParallel"
    assert loaded["CustomKernel"]["args"] == generated
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
    assert names[first:first + 2] == ["ItersPerTile", "PersistentGrid"]
    assert names[first + 2:first + 4] == ["alpha", "beta" if use_beta else "betapad"]
    assert names[-4:] == ["batchOffsetD", "batchOffsetC", "batchOffsetA", "batchOffsetB"]
    assert offset == metadata["amdhsa.kernels"][0][".kernarg_segment_size"]
    record_property("arguments", json.dumps(arguments))
    record_property("descriptor", json.dumps(generated))

    if use_beta and not initial_strides:
        # Exercise a packed C1 group at the signature boundary without asking
        # solution derivation to accept a fabricated contraction geometry.
        from Tensile.Components.Signature import SignatureDefault
        from Tensile.CustomKernels import _metadataArgToCustomArg

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
