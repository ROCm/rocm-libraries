# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assemble input-bias gradients and check completion ordering on gfx1250."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from Tensile import __file__ as tensile_file

pytestmark = pytest.mark.unit


def _input_bias_config(strategy, source, feature=None):
    root = Path(tensile_file).parent
    config = yaml.safe_load((root / "Tests/common/gradient/gfx1250/hhs_bgrada_gfx1250.yaml").read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, group = config["BenchmarkProblems"][0]
    problem["BiasSrc"] = source
    values = {key: value[:1] for entry in group["ForkParameters"] for key, value in entry.items()}
    values.update(TileProcessingStrategy=[strategy], WorkAssignment=["StaticGrid"],
                  StreamKFixupTreeReduction=[int(strategy == "StreamK")], GlobalSplitU=[0],
                  PrefetchAcrossPersistent=[0], ClusterDim=[[2, 2]], StaggerU=[0],
                  ForceDisableShadowInit=[True], ScheduleIterAlg=[0], DepthU=[64])
    if feature == "sav":
        problem["UseScaleAlphaVec"] = 1
    elif feature == "scale_ab":
        problem["UseScaleAB"] = "Vector"
    elif feature == "remap":
        values["StoreRemapVectorWidth"] = [4]
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    group["BenchmarkFinalParameters"] = [
        {"ProblemSizes": [{"Exact": [16, 16, 1, 1024]}, {"Exact": [17, 17, 1, 1025]}]},
        {"BiasTypeArgs": ["s"]}]
    return config


def _assert_alpha_zero_product_discarded(assembly):
    # The GEMM accumulators must be overwritten, not merely multiplied by zero:
    # even a zero-filled unused tensor can yield Inf*0 from the bias source.
    blocks = re.findall(
        r"v_cmp_eq_f32[^\n]*sgprAlpha[^\n]*\n"
        r"s_cbranch_vccz (label_InputBiasProductReady\w*)[^\n]*\n(.*?)^\1:",
        assembly, re.M | re.S)
    assert blocks, "alpha zero must discard the unused GEMM product"
    wmma_dests = re.findall(r"v_wmma\S*\s+v\[vgprValuC\+(\d+):vgprValuC\+\d+\+(\d+)\]", assembly)
    assert wmma_dests, "test must exercise real GEMM accumulator writes"
    expected = {i for base, width in wmma_dests
                for i in range(int(base), int(base) + int(width) + 1)}
    for _, block in blocks:
        cleared = {int(i) for i in re.findall(r"v_mov_b32 v\[vgprValuC\+(\d+)\], 0\s", block)}
        assert cleared == expected, "every WMMA result must be discarded"
        assert "ValuSum" not in block, "the independent bias sums must survive"
    discard = assembly.index("alpha-zero input bias: discard GEMM product")
    assert discard < assembly.index("// local store bias")
    alpha_scale = re.search(r"v_(?:pk_)?mul_f32[^\n]*sgprAlpha", assembly)
    assert alpha_scale and discard < alpha_scale.start()
    if "publish input bias partial before ready flag" in assembly:
        assert discard < assembly.index("publish input bias partial before ready flag")


def _assert_alpha_zero_tdm_inputs(assembly, source):
    unused = "B" if source == "A" else "A"
    prologue = assembly[:assembly.index("tensor_load_to_lds")]
    for tc in (source, unused):
        loads = list(re.finditer(f"load {tc} matrix address from pointer array", prologue))
        assert loads, "the runtime general-batch path must be generated"
        for load in loads:
            begin = prologue.rfind("ArgType == 3 for General Batched GEMM", 0, load.start())
            assert begin != -1
            guard = prologue[begin:load.start()]
            assert "skip pointer dereference when summation size is zero" in guard
            if tc == source:
                assert "sgprAlpha" not in guard, "alpha zero still needs the bias-source pointer"
            else:
                assert re.search(r"v_cmp_eq_f32.*sgprAlpha.*\n"
                                 rf"s_cbranch_vccnz label_TDMGeneralBatchedAddress{tc}Ready", guard)
    # tensor_dim0 is split over words 1 and 2. Clear just those bits, leaving
    # tile dimensions and completion enabled, and reapply after tail resets.
    assert f"alpha-zero unused {source}:" not in assembly
    guards = re.findall(
        r"v_cmp_eq_f32[^\n]*sgprAlpha[^\n]*\n"
        rf"s_cbranch_vccz (label_InputBias{unused}TensorReady\w*)[^\n]*\n(.*?)^\1:",
        assembly, re.M | re.S)
    assert len(guards) >= 2, "both initial and tail descriptors need zero bounds"
    for _, block in guards:
        assert re.search(r"s_and_b32 s\[([^]]+\+1)\], s\[\1\], 65535", block)
        assert re.search(r"s_and_b32 s\[([^]]+\+2)\], s\[\1\], 4294901760", block)
    tail = assembly[assembly.index("TDM reset tensor dim for tail"):]
    assert f"alpha-zero unused {unused}: tensor_dim0 low = 0" in tail


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
@pytest.mark.parametrize("source", ["A", "B"])
def test_input_bias_gradient_completion_assembles(tmp_path, strategy, source):
    path = tmp_path / "input_bias.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config(strategy, source)))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        _assert_alpha_zero_tdm_inputs(assembly, source)
        _assert_alpha_zero_product_discarded(assembly)
        bias_store = assembly.index("global store bias")
        # Wave 0/1 may read partials written by wave 2 for BiasSrc=B. A
        # barrier alone does not complete those stores on gfx1250.
        partial_store = assembly.rindex("local store bias", 0, bias_store)
        publication_barrier = assembly.index("s_barrier_signal -1", partial_store)
        assert re.search(r"s_wait_dscnt 0[^\n]*finish bias LDS stores before publication",
                         assembly[partial_store:publication_barrier])
        # A/B bias sums use raw inputs even when GEMM alpha is zero. Neither
        # the prologue nor the StreamK activation/main-loop setup may skip K.
        compute_setup = assembly[:assembly.index("tensor_load_to_lds")]
        assert "SKAlphaCheck" not in compute_setup
        assert "PersistentAlphaCheck" not in compute_setup
        assert "Set summation dim=0 if Alpha == 0" not in compute_setup
        assert "Skip iterations" not in compute_setup
        if strategy == "DataParallel":
            assert "publish input bias partial" not in assembly
            continue
        reduction = assembly.index("accumulate peer input bias partial")
        assert assembly.index("phantom tiles skip the store") < reduction < bias_store
        owner_gate = assembly.rfind("does wg start tile?", reduction, bias_store)
        assert owner_gate != -1
        assert "label_SK_Partials" in assembly[owner_gate:bias_store]
        publication = assembly.index("publish input bias partial before ready flag")
        flag = assembly.index("// set flag", publication)
        assert "global_wb" in assembly[publication:flag]
        assert "wait for data store" in assembly[publication:flag]
        assert "input bias partial slot offset (high word)" in assembly
        assert "s[sgprGSUSumIdx]" not in assembly


@pytest.mark.parametrize("source", ["A", "B"])
@pytest.mark.parametrize("feature", ["sav", "scale_ab", "remap"])
def test_input_bias_partials_survive_other_epilogue_lds_users(tmp_path, source, feature):
    path = tmp_path / "input_bias_combined.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config("StreamK", source, feature)))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        bias_store = assembly.index("global store bias")
        assert assembly.index("accumulate peer input bias partial") < bias_store
        if feature == "remap":
            # The raw SumUnroll sums retain their LDS0 layout; remap uses its
            # separate final D-store path only after the bias is consumed.
            assert "add bias lds offset" not in assembly
            continue
        marker = "store scaleAlpha" if feature == "sav" else "store scale"
        stores = [line for line in assembly.splitlines() if "ds_store" in line and marker in line]
        assert stores
        # These kernels have MT16, maxKId2, float accumulation: LDS[0:128]
        # contains the raw input-bias reduction and must survive vector staging.
        offsets = [int(re.search(r"offset:(\d+)", line).group(1)) for line in stores]
        assert min(offsets) >= 128


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
def test_output_bias_gradient_intermediate_and_reduction_helper(tmp_path, strategy):
    from config_harness import solutions_from_config
    from Tensile.KernelHelperNaming import initReductionKernelObjects

    path = tmp_path / "output_bias.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config(strategy, "D")))
    solutions = solutions_from_config(str(path), arch="gfx1250", limit_solutions=1)
    assert len(solutions) == 1
    helpers = initReductionKernelObjects(solutions[0])
    assert len(helpers) == len(solutions[0]["ProblemType"]["BiasDataTypeList"])
    assert all(h.state["ProblemType"]["ComputeDataType"].isSingle() for h in helpers)
    helper = next(h for h in helpers if h.state["ProblemType"]["BiasDataType"].isSingle())
    assert helper.state["ProblemType"]["BiasSrc"] == "D"
    assert helper.state["ProblemType"]["ComputeDataType"].isSingle()
    assert helper.state["ProblemType"]["BiasDataType"].isSingle()
    assert helper.getKernelName().endswith("_SS_Reduction")
    assert "float *in, float* out" in helper.getHeaderFileString()

    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        # D-gradient reduces the completed GEMM result through an intermediate
        # FP32 bias matrix, including when the final output is half precision.
        stores = [line for line in assembly.splitlines()
                  if "buffer_store_b32" in line and "// store Bias" in line]
        assert stores and all("sgprSrdBias" in line for line in stores)
        first_store = assembly.index(stores[0])
        assert assembly.index("phantom tiles skip the store") < first_store
        if strategy == "StreamK":
            assert assembly.index("flag offset based on physical partial slot") < first_store
            assert "publish input bias partial" not in assembly


@pytest.mark.parametrize("source", ["A", "B"])
@pytest.mark.parametrize("nan_unused", [False, True])
def test_alpha_zero_hardware_config_assembles_all_variants(tmp_path, source, nan_unused):
    root = Path(tensile_file).parent
    suffix = ("_nan_b" if source == "A" else "_nan_a") if nan_unused else ""
    config = yaml.safe_load((root / f"Tests/common/streamk/gfx1250/persistent_input_bias_alpha0{suffix}.yaml").read_text())
    config["BenchmarkProblems"] = [entry for entry in config["BenchmarkProblems"]
                                   if entry[0]["BiasSrc"] == source]
    path = tmp_path / "alpha_zero.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=4, arch="gfx1250", canonical=False)
    assert len(results) == 4, "both persistent strategies and prefetch depths must generate"
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        assert "global store bias" in assembly
        _assert_alpha_zero_tdm_inputs(assembly, source)
        _assert_alpha_zero_product_discarded(assembly)
        compute_setup = assembly[:assembly.index("tensor_load_to_lds")]
        assert "SKAlphaCheck" not in compute_setup
        assert "PersistentAlphaCheck" not in compute_setup
        assert "Set summation dim=0 if Alpha == 0" not in compute_setup
        assert "Skip iterations" not in compute_setup


@pytest.mark.parametrize("source", ["A", "B"])
@pytest.mark.parametrize("variant", ["multiwave", "prefetch", "buffer", "buffer_stagger", "buffer_32bit"])
def test_alpha_zero_input_bias_load_variants(tmp_path, source, variant):
    strategy = "DataParallel" if variant.startswith("buffer") else "StreamK"
    config = _input_bias_config(strategy, source)
    group = config["BenchmarkProblems"][0][1]
    values = {key: value for entry in group["ForkParameters"] for key, value in entry.items()}
    if variant == "multiwave":
        values["MatrixInstruction"] = [[16, 16, 32, 1, 1, 1, 1, 2, 2]]
    elif variant == "prefetch":
        values["PrefetchGL2"] = [1]
    else:
        values.update(TDMInst=[0], ClusterDim=[[1, 1]], ClusterLocalRead=[0])
        if variant == "buffer_stagger":
            values["StaggerU"] = [32]
        elif variant == "buffer_32bit":
            values["Use64bShadowLimit"] = [False]
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    path = tmp_path / "input_bias_load.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    unused = "B" if source == "A" else "A"
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        _assert_alpha_zero_product_discarded(assembly)
        if variant in ("multiwave", "prefetch"):
            _assert_alpha_zero_tdm_inputs(assembly, source)
        if variant == "prefetch":
            # Guard the address calculation (which may dereference a pointer
            # array) as well as every global_prefetch instruction.
            assert f"s_cbranch_vccnz label_InputBias{unused}PrefetchAddressDone" in assembly
            guards = re.findall(
                rf"s_cbranch_vccnz (label_InputBias{unused}PrefetchDone\w*)[^\n]*\n(.*?)^\1:",
                assembly, re.M | re.S)
            assert guards and all("global_prefetch_b8" in block for _, block in guards)
            assert f"InputBias{source}PrefetchDone" not in assembly
        elif variant in ("buffer", "buffer_stagger"):
            loop = re.search(r"^label_LoopBeginL:$(.*?)^label_LoopEndL:", assembly, re.M | re.S)
            assert loop and "Alpha" not in loop[1]
            assert f"buffer_load" in loop[1]
            # Both setup and tail must clear every packed bound bit and the
            # shadow; otherwise an increment or tail rewind can reopen reads.
            guards = re.findall(
                rf"s_cbranch_vccz (label_\w*InputBias{unused}BufferReady\w*)[^\n]*\n(.*?)^\1:",
                assembly, re.M | re.S)
            assert len(guards) >= 2
            for _, block in guards:
                assert f"s_mov_b32 s[sgprSrd{unused}+2], 0" in block
                assert f"s_and_b32 s[sgprSrd{unused}+1], s[sgprSrd{unused}+1], 33554431" in block
                assert f"s_mov_b32 s[sgprShadowLimit{unused}], 0" in block
                assert f"s_mov_b32 s[sgprShadowLimit{unused}+1], 2147483648" in block
                assert "s_mov_b32 s[sgprInputBiasBufferLimit], 0" in block
            assert "s[sgprInputBiasBufferLimit]" in loop[1]
            assert assembly.index(f"InputBias{unused}BufferReady") < assembly.index("buffer_load")
            assert f"alpha-zero unused {source}: suppress buffer reads" not in assembly
        elif variant == "buffer_32bit":
            # Check each actual unused-input load, including tail loads, so
            # an SRD increment cannot accidentally restore a nonzero bound.
            pos = 0
            for load in re.finditer(rf"^buffer_load[^\n]*sgprSrd{unused}:", assembly, re.M):
                block = assembly[pos:load.start()]
                assert f"s_mov_b32 s[sgprSrd{unused}+2], 0" in block
                assert re.search(rf"s_cbranch_vccz label_\w*InputBias{unused}BufferReady", block)
                # gfx125x stores the low seven limit bits in Srd+1; clearing
                # only Srd+2 still permits a short read through a null operand.
                assert f"s_and_b32 s[sgprSrd{unused}+1], s[sgprSrd{unused}+1], 33554431" in block
                pos = load.end()
            assert pos
            assert f"alpha-zero unused {source}: suppress buffer reads" not in assembly


@pytest.mark.parametrize("source", [None, "A", "B"])
def test_two_peer_streamk_has_no_multicast_or_unpaired_cluster_wait(tmp_path, source):
    config = _input_bias_config("StreamK", source or "A")
    problem, group = config["BenchmarkProblems"][0]
    if source is None:
        problem.update(UseBias=0, Gradient=False)
    for entry in group["ForkParameters"]:
        if "ClusterDim" in entry:
            entry["ClusterDim"] = [[2, 1]]
    path = tmp_path / "independent_cluster.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(kernels) == 1
    for name, assembly, error in kernels:
        assert error == 0
        assert_assembles(assembly, name)
        assert "sgprStreamKClusterPeer" in assembly
        assert "sgprPersistentPhantomTile" in assembly
        assert "tensor_load_to_lds" in assembly
        assert "MulticastMask" not in assembly
        assert not re.search(r"s_barrier_(signal|wait) -3", assembly)
        if source:
            _assert_alpha_zero_product_discarded(assembly)
            _assert_alpha_zero_tdm_inputs(assembly, source)
