# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regular GEMM descriptors and explicit-index client input, without a GPU."""

import os
import subprocess
import sys
import zlib
from itertools import product

import msgpack
import pytest
from tensilelite.Utilities.gemm_problem_for_solution import (
    UnsupportedSolution,
    load_library_metadata,
)

pytestmark = pytest.mark.unit


def solution(index=17, *, trans_a=False, trans_b=False, split=1):
    a, b = ("lik" if trans_a else "ilk"), ("jlk" if trans_b else "ljk")
    return {
        "index": index,
        "name": "solution-name",
        "kernelName": "kernel-name",
        "problemType": {
            "operationIdentifier": f"Contraction_l_A{a}_B{b}_Cijk_Dijk",
            "transA": trans_a,
            "transB": trans_b,
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
            "computeType": "Float",
        },
        "sizeMapping": {"macroTile": [128, 96, 1], "depthU": 32, "globalSplitU": split},
    }


def write_shard(directory, records, name="TensileLibrary_test_gfx942.dat"):
    path = directory / name
    path.write_bytes(msgpack.packb({"solutions": records, "library": {"type": "Matching"}}))
    return path


@pytest.mark.parametrize(
    "trans_a,trans_b,split", list(product((False, True), (False, True), (1, 4, 8)))
)
def test_full_tile_and_eight_blocks_per_split(tmp_path, trans_a, trans_b, split):
    write_shard(tmp_path, [solution(trans_a=trans_a, trans_b=trans_b, split=split)])
    library = load_library_metadata(tmp_path, target="gfx942")
    problem = library.choose_single_problem(17)
    assert (problem.m, problem.n, problem.k) == (128, 96, 256 * split)
    assert problem.leading_dimensions == (
        problem.k if trans_a else 128,
        96 if trans_b else problem.k,
        128,
        128,
    )
    assert problem.batch_strides == (128 * problem.k, 96 * problem.k, 12288, 12288)
    assert (problem.alpha, problem.beta) == (1, 0)
    # Generation reuses the loaded records, independent of the files afterward.
    for path in tmp_path.iterdir():
        path.unlink()
    assert library.choose_single_problem(17) == problem


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
@pytest.mark.parametrize("compressed", [False, True])
def test_formats_selection_and_identity(tmp_path, library_format, compressed):
    from tensilelite.Utilities.BinaryLibraryIO import writeMessagePack

    writeMessagePack(
        tmp_path / "TensileLibrary_test_gfx942.dat",
        {"solutions": [solution(9), solution(17)], "library": {"type": "Matching"}},
        indexed=library_format == "msgpack-indexed",
    )
    if not compressed:
        path = tmp_path / "TensileLibrary_test_gfx942.dat.zlib"
        path.with_suffix("").write_bytes(zlib.decompress(path.read_bytes()))
        path.unlink()
    library = load_library_metadata(tmp_path, target="gfx942", solution_indices=[17])
    assert library.indices == (17,)
    p = library.choose_single_problem(17)
    assert (p.solution_name, p.kernel_name, p.library_directory, p.target) == (
        "solution-name",
        "kernel-name",
        tmp_path,
        "gfx942",
    )
    with pytest.raises(KeyError):
        library.choose_single_problem(9)


@pytest.mark.parametrize(
    "key,value",
    [
        ("adaptiveGemmGSUA", 1),
        ("globalSplitU", 0),
        ("depthU", 0),
        ("depthU", True),
    ],
)
def test_unsupported_scheduling_is_distinct_from_invalid_metadata(tmp_path, key, value):
    record = solution()
    record["sizeMapping"][key] = value
    write_shard(tmp_path, [record])
    if key in {"tileProcessingStrategy", "adaptiveGemmGSUA"} or value == -1:
        with pytest.raises(UnsupportedSolution) as caught:
            load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
        assert caught.value.solution_index == 17
        assert caught.value.reasons
    else:
        with pytest.raises(ValueError):
            load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize("blocks,runtime_split", [(1, 1), (3, 2), (8, 5)])
def test_automatic_split_k_spans_two_groups_of_blocks(tmp_path, blocks, runtime_split):
    write_shard(tmp_path, [solution(split=-1)])
    p = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=blocks
    )
    assert (p.m, p.n, p.k) == (128, 96, 2 * blocks * 32)
    # ContractionSolution::calculateAutoGSU for one output tile and enough CUs.
    assert max(1, p.k // 32 // 3) == runtime_split


def test_automatic_split_k_honors_minimum_k(tmp_path):
    record = solution(split=-1)
    record["problemPredicate"] = {"type": "SizeGreaterThan", "index": 3, "value": 300}
    write_shard(tmp_path, [record])
    p = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=3
    )
    assert p.k == 320


@pytest.mark.parametrize(
    "per_tile,split,supported",
    [
        ([4, 2, 4], 4, True),
        # Generated metadata can round the stored elements per store to zero.
        ([1, 0, 4], 4, True),
        ([4, 2, 4], -1, True),
        ([409600, 16, 2], 4, False),
        ([409600, 16, 2], 1, True),
    ],
)
def test_synchronizer_size_check_matches_runtime(tmp_path, per_tile, split, supported):
    record = solution(split=split)
    record["problemPredicate"] = {
        "type": "SynchronizerSizeCheck",
        "index": 0,
        "value": [128, 96, *per_tile, split],
    }
    write_shard(tmp_path, [record])
    library = load_library_metadata(tmp_path, target="gfx942")
    if supported:
        p = library.choose_single_problem(17)
        assert (p.m, p.n) == (128, 96)
    else:
        with pytest.raises(UnsupportedSolution, match="synchronizer"):
            library.choose_single_problem(17)


@pytest.mark.parametrize(
    "value", [[128, 96, 4, 2, 4], [128, 0, 4, 2, 4, 2], [128, 96, -1, 2, 4, 2], "x"]
)
def test_invalid_synchronizer_size_check_is_an_error(tmp_path, value):
    record = solution(split=2)
    record["problemPredicate"] = {"type": "SynchronizerSizeCheck", "index": 0, "value": value}
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError, match="SynchronizerSizeCheck"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize("split", [0, 1, 4])
@pytest.mark.parametrize("blocks", [1, 8, 17])
@pytest.mark.parametrize(
    "policy",
    [
        {"tileProcessingStrategy": "StreamK", "workAssignment": assignment}
        for assignment in ("StaticGrid", "DynamicWorkQueue", "Hybrid")
    ]
    + [{"tileProcessingStrategy": "DataParallel"}],
)
def test_persistent_scheduling_uses_no_fixed_split(tmp_path, split, blocks, policy):
    record = solution(split=split)
    record["sizeMapping"].update(policy)
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=blocks
    )
    assert (problem.m, problem.n, problem.k) == (128, 96, blocks * 32)


@pytest.mark.parametrize(
    "mapping",
    [
        {"tileProcessingStrategy": "Persistent"},
        {"tileProcessingStrategy": "Streamk"},
        {"workAssignment": "typo"},
        {"tileProcessingStrategy": "DataParallel", "workAssignment": "Hybrid"},
        {"streamK": 2},
        {"streamK": 3, "tileProcessingStrategy": "None"},
    ],
)
def test_invalid_execution_policy_is_an_error(tmp_path, mapping):
    record = solution()
    record["sizeMapping"].update(mapping)
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError, match="Solution 17"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize(
    "mode,force,strategy",
    [(3, 0, "StreamK"), (4, 0, "StreamK"), (5, 0, "StreamK"), (3, 1, "DataParallel")],
)
def test_legacy_persistent_scheduling_matches_canonical(tmp_path, mode, force, strategy):
    record = solution(split=0)
    record["sizeMapping"].update(streamK=mode, streamKForceDPOnly=force)
    write_shard(tmp_path, [record])
    legacy = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    record["sizeMapping"].pop("streamK")
    record["sizeMapping"].pop("streamKForceDPOnly")
    record["sizeMapping"].update(
        tileProcessingStrategy=strategy,
        workAssignment={3: "StaticGrid", 4: "DynamicWorkQueue", 5: "Hybrid"}[mode],
    )
    write_shard(tmp_path, [record])
    assert legacy == load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert legacy.k == 256


@pytest.mark.parametrize(
    "key,value",
    [
        ("sparse", 1),
        ("useE", True),
        ("activationType", "Gelu"),
        ("operationIdentifier", "another contraction"),
    ],
)
def test_rejects_unsupported_problem_features(tmp_path, key, value):
    record = solution()
    record["problemType"][key] = value
    write_shard(tmp_path, [record])
    with pytest.raises(UnsupportedSolution) as caught:
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert caught.value.solution_index == 17
    assert caught.value.reasons


def epilogue_solution(*, gradient=False, aux=False, amax=False, sources=(1,), activations=None):
    record = solution()
    record["problemType"].update(
        dType="Half",
        cType="Half",
        eType="Half",
        useBias=1,
        biasSrcWhiteList=list(sources),
        biasDataTypeWhiteList=["Float", "Half"],
        useGradient=gradient,
        useE=aux,
        outputAmaxD=amax,
        activationType="Hipblaslt_all" if aux else "None",
    )
    if activations is not None:
        record["problemPredicate"] = {
            "type": "And",
            "value": [{"type": "ActivationEnumWhiteList", "value": list(activations)}],
        }
    return record


@pytest.mark.parametrize(
    "gradient,aux,source,activations,expected",
    [
        # Bias gradient reduced from B, without an activation.
        (True, False, 1, None, ("B", "None")),
        (True, False, 0, None, ("A", "None")),
        # Forward ReLU that also writes E.
        (False, True, 3, ["None", "Relu"], ("D", "Relu")),
        # Gradient that reads E and applies the ReLU derivative.
        (True, True, 3, ["Dgelu", "Drelu"], ("D", "Relu")),
    ],
)
def test_gradient_and_aux_epilogues(tmp_path, gradient, aux, source, activations, expected):
    write_shard(
        tmp_path,
        [epilogue_solution(gradient=gradient, aux=aux, sources=(source,), activations=activations)],
    )
    p = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert p.gradient == gradient
    assert p.aux_type == ("Half" if aux else None)
    assert p.bias is not None and p.bias.dtype == "Float"
    assert (p.bias.source, p.activation) == expected


def test_amax_d_needs_no_bias_or_activation(tmp_path):
    write_shard(tmp_path, [epilogue_solution(amax=True)])
    p = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert (p.amax_d, p.gradient, p.aux_type, p.bias) == (True, False, None, None)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"useBias": 0}, "without bias support"),
        ({"biasSrcWhiteList": [2]}, "bias sources"),
        ({"biasDataTypeWhiteList": []}, "bias sources"),
    ],
)
def test_gradient_without_usable_bias_is_unsupported(tmp_path, changes, reason):
    record = epilogue_solution(gradient=True)
    record["problemType"].update(changes)
    write_shard(tmp_path, [record])
    with pytest.raises(UnsupportedSolution, match=reason):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


def test_aux_requires_an_allowed_relu(tmp_path):
    write_shard(tmp_path, [epilogue_solution(aux=True, sources=(3,), activations=["Gelu"])])
    with pytest.raises(UnsupportedSolution, match="Relu not allowed"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize("bias,scaling", [(False, ""), (True, ""), (True, "Scalar")])
def test_grouped_problem_uses_two_packed_members_and_required_options(tmp_path, bias, scaling):
    record = solution(trans_a=True, split=4)
    record["problemType"].update(
        groupedGemm=True,
        useBias=int(bias),
        biasDataTypeWhiteList=["Half", "Float"],
        biasSrcWhiteList=[3],
        useScaleAB=scaling,
    )
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=3, alpha=2, beta=1
    )
    assert problem.grouped_members == ((128, 96, 384), (128, 96, 384))
    assert (problem.alpha, problem.beta) == (2, 1)
    assert (problem.bias is not None) == bias
    if bias:
        assert (problem.bias.dtype, problem.bias.source) == ("Float", "D")
    assert problem.scale_ab == (scaling or None)


@pytest.mark.parametrize("swizzle_a,swizzle_b", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", ["Half", "BFloat16", "Float8_fnuz", "Float4"])
def test_swizzled_operands_keep_logical_shape_and_request_device_layout(
    tmp_path, swizzle_a, swizzle_b, dtype
):
    record = solution(trans_a=True, trans_b=False, split=4)
    record["problemType"].update(
        aType=dtype,
        bType=dtype,
        cType="Half",
        dType="Half",
        swizzleTensorA=swizzle_a,
        swizzleTensorB=swizzle_b,
    )
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=3, alpha=2, beta=1
    )
    assert (problem.swizzle_a, problem.swizzle_b) == (swizzle_a, swizzle_b)
    assert (problem.m, problem.n, problem.k) == (128, 96, 384)


@pytest.mark.parametrize("dtype", ["Float4", "Float8"])
@pytest.mark.parametrize("output", ["Half", "BFloat16", "Float"])
@pytest.mark.parametrize("depth", [256, 512])
def test_gfx950_mx_streamk_preserves_scales_and_original_index(tmp_path, dtype, output, depth):
    record = solution(trans_a=True, split=0)
    record["sizeMapping"].update(depthU=depth, tileProcessingStrategy="StreamK")
    record["problemType"].update(
        aType=dtype,
        bType=dtype,
        cType=output,
        dType=output,
        mxBlockA=32,
        mxBlockB=32,
        mxTypeA="E8",
        mxTypeB="E8",
        mxScaleFormat=1,
    )
    write_shard(tmp_path, [record], "TensileLibrary_mx_gfx950.dat")
    problem = load_library_metadata(tmp_path, target="gfx950").choose_single_problem(17)
    assert (
        problem.mx_block_a,
        problem.mx_block_b,
        problem.mx_type_a,
        problem.mx_type_b,
        problem.mx_scale_format,
    ) == (32, 32, "E8", "E8", 1)
    assert (problem.m, problem.n, problem.k) == (128, 96, 8 * depth)
    assert problem.solution_index == 17


@pytest.mark.parametrize(
    "field,value",
    [
        ("mxBlockA", -1),
        ("mxBlockB", True),
        ("mxScaleFormat", 3),
        ("mxScaleFormat", True),
        ("mxTypeA", "unknown"),
    ],
)
def test_invalid_mx_metadata_is_an_error(tmp_path, field, value):
    record = solution()
    record["problemType"].update(mxBlockA=32)
    record["problemType"][field] = value
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError, match="Solution 17"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


def test_generation_does_not_import_generator_or_rocisa(tmp_path):
    mx = solution(18, trans_a=True, split=0)
    mx["sizeMapping"].update(streamK=3, depthU=256)
    mx["problemType"].update(
        aType="Float4",
        bType="Float4",
        mxBlockA=32,
        mxBlockB=32,
        mxTypeA="E8",
        mxTypeB="E8",
        mxScaleFormat=1,
    )
    write_shard(tmp_path, [solution(), mx], "TensileLibrary_test_gfx950.dat")
    script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('rocisa', 'tensilelite.Common', 'tensilelite.SolutionStructs',
                                'Tensile.Common', 'Tensile.SolutionStructs')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from tensilelite.Utilities.gemm_problem_for_solution import load_library_metadata
library = load_library_metadata(sys.argv[1], target='gfx950')
for index, k in [(17, 256), (18, 2048)]:
    problem = library.choose_single_problem(index)
    assert (problem.m, problem.n, problem.k) == (128, 96, k)
    assert problem.solution_index == index
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


def test_compressed_precedence_missing_and_duplicate_indices(tmp_path):
    directory = tmp_path / "gfx942"
    directory.mkdir()
    path = write_shard(directory, [solution(17)])
    path.with_suffix(".dat.zlib").write_bytes(zlib.compress(path.read_bytes()))
    write_shard(directory, [solution(9)])
    assert load_library_metadata(tmp_path, target="gfx942").indices == (17,)
    with pytest.raises(KeyError, match="404"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=[404])
    write_shard(directory, [solution(17)], "TensileLibrary_other_gfx942.dat")
    with pytest.raises(ValueError, match="duplicate solution index 17"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=[404])


def test_catalog_retains_only_problem_construction_metadata(tmp_path):
    record = solution()
    record.update(
        hardwarePredicate={"type": "TruePred"},
        problemPredicate={"type": "TruePred"},
        customKernel={"config": "unused kernel configuration"},
        ideals={"64": 100.0},
    )
    write_shard(tmp_path, [record])
    library = load_library_metadata(tmp_path, target="gfx942")
    assert set(library._records[17]) == {
        "name",
        "kernelName",
        "problemType",
        "sizeMapping",
        "sizePredicates",
        "activationWhiteList",
    }
    assert library._records[17]["sizePredicates"] == ()
    assert library._records[17]["activationWhiteList"] is None
    assert library.choose_single_problem(17).k == 256


def test_rocroller_2205_size_bounds_override_preferred_dimensions(tmp_path):
    record = solution(2205, trans_a=True)
    record["sizeMapping"].update(macroTile=[32, 32, 1], depthU=64)
    record["problemType"].update(
        aType="BFloat8",
        bType="BFloat8",
        cType="BFloat16",
        dType="BFloat16",
        mxBlockA=32,
        mxBlockB=32,
        mxScaleFormat=0,
    )
    record["problemPredicate"] = {
        "type": "And",
        "value": [
            {"type": "SizeGreaterThan", "index": 0, "value": 63},
            {"type": "SizeLessThan", "index": 0, "value": 129},
            {"type": "SizeGreaterThan", "index": 1, "value": 63},
            {"type": "SizeLessThan", "index": 1, "value": 129},
            {"type": "SizeGreaterThan", "index": 3, "value": 127},
            {"type": "SizeLessThan", "index": 3, "value": 129},
        ],
    }
    write_shard(tmp_path, [record], "TensileLibrary_mx_gfx950.dat")
    library = load_library_metadata(tmp_path, target="gfx950")
    for blocks in (1, 8, 17):
        problem = library.choose_single_problem(2205, reduction_blocks=blocks)
        assert (problem.m, problem.n, problem.k) == (64, 64, 128)


def test_nested_size_bounds_and_multiples(tmp_path):
    record = solution()
    record["problemPredicate"] = {
        "type": "And",
        "value": [
            {"type": "LeadingFree0SizesGreaterOrEqual", "value": 129},
            {"type": "Free0SizeMultiple", "index": 0, "value": 8},
            {"type": "SizeEqual", "index": 1, "value": 1},
            {
                "type": "And",
                "value": [
                    {"type": "SizeInRange", "index": 3, "value": {"min": 65, "max": 200}},
                    {"type": "BoundSizeMultiple", "index": -1, "value": 48},
                ],
            },
        ],
    }
    write_shard(tmp_path, [record])
    p = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert (p.m, p.n, p.k) == (136, 1, 192)


@pytest.mark.parametrize(
    "predicates,error,match",
    [
        (
            [
                {"type": "SizeGreaterThan", "index": 0, "value": 128},
                {"type": "SizeLessThan", "index": 0, "value": 129},
            ],
            ValueError,
            "contradictory",
        ),
        ([{"type": "SizeMultiple", "index": 3, "value": 0}], ValueError, "value"),
        ([{"type": "SizeEqual", "index": True, "value": 32}], ValueError, "index"),
        ([{"type": "SizeEqual", "index": 4, "value": 32}], ValueError, "index"),
        ([{"type": "SizeEqual", "index": 3, "value": 17}], UnsupportedSolution, "reduction blocks"),
        ([{"type": "BatchSizeEqual", "index": 0, "value": 2}], UnsupportedSolution, "batch count"),
    ],
)
def test_invalid_size_metadata_is_distinct_from_unsupported_workload(
    tmp_path, predicates, error, match
):
    record = solution()
    record["problemPredicate"] = {"type": "And", "value": predicates}
    write_shard(tmp_path, [record])
    with pytest.raises(error, match=match):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize("dtype", ["Half", "BFloat16", "Float8_fnuz", "BFloat8_fnuz", "Int8"])
def test_storage_and_compute_types(tmp_path, dtype):
    record = solution()
    record["problemType"].update(aType=dtype, bType=dtype)
    if dtype == "Int8":
        record["problemType"].update(cType="Int32", dType="Int32", computeType="Int32")
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert problem.a_type == problem.b_type == dtype
    assert problem.compute_type == ("Int32" if dtype == "Int8" else "Float")
    assert problem.c_type == problem.d_type == problem.compute_type


@pytest.mark.parametrize("output_type", ["Int8", "Int32"])
def test_float_compute_int8_inputs_use_native_derived_input_type(tmp_path, output_type):
    record = solution()
    record["problemType"].update(
        aType="Int8",
        bType="Int8",
        cType=output_type,
        dType=output_type,
        computeType="Float",
        computeInputTypeA="Int8",
        computeInputTypeB="Int8",
    )
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert (problem.compute_input_a, problem.compute_input_b) == ("Int8", "Int8")


@pytest.mark.parametrize("blocks", [1, 3, 8, 17])
def test_reduction_work_is_configurable(tmp_path, blocks):
    write_shard(tmp_path, [solution(split=4)])
    library = load_library_metadata(tmp_path, target="gfx942")
    assert library.choose_single_problem(17, reduction_blocks=blocks).k == blocks * 32 * 4


def test_nondefault_scalars_are_retained(tmp_path):
    write_shard(tmp_path, [solution(split=4)])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
        17, reduction_blocks=3, alpha=2, beta=1
    )
    assert (problem.k, problem.alpha, problem.beta) == (384, 2, 1)


def test_scalar_values_respect_problem_capability(tmp_path):
    record = solution()
    record["problemType"]["useBeta"] = False
    write_shard(tmp_path, [record])
    library = load_library_metadata(tmp_path, target="gfx942")
    with pytest.raises(UnsupportedSolution, match="nonzero beta"):
        library.choose_single_problem(17, beta=1)
    for alpha, beta in ((float("nan"), 0), (1, float("inf")), (True, 0), (10**1000, 0)):
        with pytest.raises(ValueError, match="finite real numbers"):
            library.choose_single_problem(17, alpha=alpha, beta=beta)
    record["problemType"].update(useBeta=True, computeType="Int32")
    write_shard(tmp_path, [record])
    with pytest.raises(UnsupportedSolution, match="integer-compute scalar must fit Int32"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17, alpha=0.5)
    with pytest.raises(UnsupportedSolution, match="integer-compute scalar must fit Int32"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17, alpha=2**31)


@pytest.mark.parametrize("blocks", [0, -1, True, 8.0])
def test_invalid_reduction_blocks_are_errors(tmp_path, blocks):
    write_shard(tmp_path, [solution()])
    with pytest.raises(ValueError, match="reduction_blocks"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(
            17, reduction_blocks=blocks
        )


@pytest.mark.parametrize("target", ["gfx90a", "gfx942", "gfx950", "gfx1250-strict"])
@pytest.mark.parametrize("dtype", ["Double", "ComplexFloat", "ComplexDouble"])
def test_ordinary_types_and_architectures(tmp_path, target, dtype):
    record = solution()
    record["problemType"].update({f"{key}Type": dtype for key in "abcd"})
    record["problemType"]["computeType"] = dtype
    write_shard(tmp_path, [record], f"TensileLibrary_test_{target}.dat")
    problem = load_library_metadata(tmp_path, target=target).choose_single_problem(17)
    assert problem.a_type == problem.compute_type == dtype


@pytest.mark.parametrize("dtype", ["Int64", "Int8x4", "Half"])
def test_descriptor_accepts_metadata_types_without_client_restrictions(tmp_path, dtype):
    record = solution()
    record["problemType"].update(aType=dtype, bType=dtype, computeType=dtype)
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert problem.a_type == dtype


def test_complex_conjugate_transpose(tmp_path):
    record = solution(trans_a=True)
    record["problemType"].update({f"{key}Type": "ComplexFloat" for key in "abcd"})
    record["problemType"].update(
        computeType="ComplexFloat", operationIdentifier="Contraction_l_AlikC_Bljk_Cijk_Dijk"
    )
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert problem.conjugate_a and problem.trans_a


def test_sweep_can_continue_only_for_explicit_unsupported_cases(tmp_path):
    unsupported = solution(9)
    unsupported["problemType"].update(groupedGemm=True, useGateResidual=True)
    write_shard(tmp_path, [unsupported, solution(17)])
    library = load_library_metadata(tmp_path, target="gfx942")
    executed, gaps = [], {}
    for index in library.indices:
        try:
            executed.append(library.choose_single_problem(index))
        except UnsupportedSolution as error:
            gaps[error.solution_index] = error.reasons
    assert [p.solution_index for p in executed] == [17]
    assert set(gaps[9]) == {"useGateResidual"}
    # A broken record must escape the sweep's UnsupportedSolution handler.
    broken = solution(0)
    del broken["sizeMapping"]["depthU"]
    write_shard(tmp_path, [broken])
    with pytest.raises(KeyError, match="depthU"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(0)


def test_unknown_metadata_type_is_an_error(tmp_path):
    record = solution()
    record["problemType"]["aType"] = "typo"
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError, match="unknown metadata datatype"):
        load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)


@pytest.mark.parametrize("per_operand", [False, True])
@pytest.mark.parametrize(
    "pair,a_type,b_type",
    [
        ("Float8BFloat8", "Float8", "BFloat8"),
        ("BFloat8Float8", "BFloat8", "Float8"),
        ("Float8BFloat8_fnuz", "Float8_fnuz", "BFloat8_fnuz"),
        ("BFloat8Float8_fnuz", "BFloat8_fnuz", "Float8_fnuz"),
    ],
)
def test_mixed_fp8_compute_inputs(tmp_path, per_operand, pair, a_type, b_type):
    record = solution()
    problem_type = record["problemType"]
    problem_type.update(aType=a_type, bType=b_type)
    if per_operand:
        problem_type.update(computeInputTypeA=pair, computeInputTypeB=pair)
    else:
        problem_type["computeInputType"] = pair
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").choose_single_problem(17)
    assert (problem.compute_input_a, problem.compute_input_b) == (a_type, b_type)
