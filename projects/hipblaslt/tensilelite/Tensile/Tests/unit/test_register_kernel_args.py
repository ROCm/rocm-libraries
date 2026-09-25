# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Coverage for ``KernelWriter._registerKernelArgs`` — the pure argument-metadata
builder that produces the ``CustomKernel.args`` list for Tensile-generated
kernels (the kernarg-buffer layout the C++ runtime packs against).

Scope / what this does and does NOT guarantee:

* It drives the *real* method body over controlled ``self.states`` / ``kernel``
  inputs and pins the arg **ordering** and per-feature **inclusion** (Sparse ->
  AddressMetadata, StreamK -> SK scalar args, MBSK -> trailing sync args, ...).
  That layout is a real contract: the code must match ``Signature.py`` and the
  assembly prologue, so a reordered/dropped arg is a genuine regression this
  catches.
* It is a change-detector for that logic, **not** a cross-check against the live
  ``Signature.py`` kernarg emission — feeding fabricated ``states`` cannot prove
  the two stay consistent. That end-to-end consistency is exercised by the
  ``Tests/custom/*.yaml`` kernels that build + dispatch on real hardware.

``_registerKernelArgs`` reads only attributes/inputs (no rocisa/asm emission), so
it is called unbound with a stub ``self`` — no GPU, assembler, or kernel build.
"""

from types import SimpleNamespace

import pytest

from Tensile.Activation import ActivationType
from Tensile.Common import DataDirection
from Tensile.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


def _writer(useBias=DataDirection.NONE, needBiasType=False, bpeCinternal=4,
            numActivationArgSize=1, debugKernel=False,
            numSgprSizesFree=2, numSgprSizesSum=1, strides=2):
    ns = SimpleNamespace
    states = ns(
        numSgprSizesFree=numSgprSizesFree, numSgprSizesSum=numSgprSizesSum,
        d=ns(numSgprStrides=strides), c=ns(numSgprStrides=strides),
        a=ns(numSgprStrides=strides), b=ns(numSgprStrides=strides),
        mxsa=ns(numSgprStrides=1), mxsb=ns(numSgprStrides=1),
        m=ns(numSgprStrides=1), e=ns(numSgprStrides=strides),
        bpeCinternal=bpeCinternal, useBias=useBias, needBiasType=needBiasType,
        numActivationArgSize=numActivationArgSize, kernelName="tensile_kernel",
    )
    return ns(states=states, debugConfig=ns(debugKernel=debugKernel), kernelArgDefs=[])


def _kernel(**over):
    problem_type = {
        "MXBlockA": False, "MXBlockB": False, "Sparse": False, "UseBeta": True,
        "UseScaleAB": False, "UseScaleCD": False, "UseScaleAlphaVec": 0, "UseE": False,
        "ActivationType": ActivationType("none"), "OutputAmaxD": False, "UseBias": 0,
    }
    problem_type.update(over.pop("ProblemType", {}))
    kernel = {
        "InternalSupportParams": {"KernArgsVersion": 1},
        "ProblemType": problem_type,
        "StreamK": 0, "StreamKAtomic": 0,
        "PackedC0IdxChars": ["I"], "PackedC0IndicesX": [0],
        "PackedC1IdxChars": ["J"], "PackedC1IndicesX": [1],
        "GlobalSplitUAlgorithm": "SingleBuffer", "AdaptiveGemmGSUA": 0,
        "ActivationFused": False,
    }
    kernel.update(over)
    return kernel


def _register(writer, kernel):
    KernelWriter._registerKernelArgs(writer, kernel)
    return writer.kernelArgDefs


def _sems(writer, kernel):
    return [a["semantic"] for a in _register(writer, kernel)]


# --------------------------------------------------------------------------- #
# Base layout + header
# --------------------------------------------------------------------------- #


def test_base_layout_exact_ordering():
    assert _sems(_writer(), _kernel()) == [
        "GemmInfo", "InternalArgs", "InternalArgs1", "NumWorkGroups",
        "SizeFree0", "SizeFree1", "SizeSum",
        "AddressD", "AddressC", "AddressA", "AddressB",
        "StrideD0", "StrideD1", "StrideC0", "StrideC1",
        "StrideA0", "StrideA1", "StrideB0", "StrideB1",
        "Alpha", "Beta",
    ]


def test_kernargs_version0_omits_v1_header():
    sems = _sems(_writer(), _kernel(InternalSupportParams={"KernArgsVersion": 0}))
    assert sems[:2] == ["GemmInfo", "InternalArgs"]
    assert "InternalArgs1" not in sems
    assert "NumWorkGroups" not in sems


def test_debug_buffer_between_sizes_and_addresses():
    sems = _sems(_writer(debugKernel=True), _kernel())
    assert sems.index("SizeSum") < sems.index("DebugBuffer") < sems.index("AddressD")


# --------------------------------------------------------------------------- #
# Tensor-address / stride variants
# --------------------------------------------------------------------------- #


def test_mxblock_adds_scale_addresses_and_strides():
    sems = _sems(_writer(), _kernel(ProblemType={"MXBlockA": True, "MXBlockB": True}))
    assert sems[sems.index("AddressA") + 1] == "AddressMXScaleA"
    assert sems[sems.index("AddressB") + 1] == "AddressMXScaleB"
    assert "StrideScaleA0" in sems and "StrideScaleB0" in sems


def test_sparse_adds_metadata_address_and_strides():
    sems = _sems(_writer(), _kernel(ProblemType={"Sparse": True}))
    assert "AddressMetadata" in sems
    assert "StrideMetadata0" in sems


def test_packed_magic_divisors_carry_index():
    kernel = _kernel(PackedC0IdxChars=["I", "J"], PackedC0IndicesX=[2, 3])
    defs = _register(_writer(), kernel)
    magics = [a for a in defs if a["semantic"] in ("MagicNumberSize", "MagicShiftSize")]
    # one (number, shift) pair for the (n-1) leading packed chars
    assert [a["semantic"] for a in magics] == ["MagicNumberSize", "MagicShiftSize"]
    assert all(a["index"] == 2 for a in magics)


def test_packed_c1_magic_divisors():
    # The PackedC1 loop is independent of PackedC0; exercise it directly.
    kernel = _kernel(PackedC1IdxChars=["J", "K"], PackedC1IndicesX=[4, 5])
    defs = _register(_writer(), kernel)
    magics = [a for a in defs if a["semantic"] in ("MagicNumberSize", "MagicShiftSize")]
    assert [a["semantic"] for a in magics] == ["MagicNumberSize", "MagicShiftSize"]
    assert all(a["index"] == 4 for a in magics)


# --------------------------------------------------------------------------- #
# Alpha / Beta
# --------------------------------------------------------------------------- #


def test_no_beta_when_use_beta_false():
    sems = _sems(_writer(), _kernel(ProblemType={"UseBeta": False}))
    assert "Alpha" in sems and "Beta" not in sems


def test_alpha_beta_wide_compute_are_float64():
    defs = _register(_writer(bpeCinternal=8), _kernel())
    types = {a["semantic"]: a["type"] for a in defs}
    assert types["Alpha"] == "float64" and types["Beta"] == "float64"


def test_alpha_narrow_compute_is_uint32():
    defs = _register(_writer(bpeCinternal=4), _kernel())
    types = {a["semantic"]: a["type"] for a in defs}
    assert types["Alpha"] == "uint32"


# --------------------------------------------------------------------------- #
# StreamK scalar args
# --------------------------------------------------------------------------- #


def test_streamk_two_tile_adds_workspace_and_all_scalar_args():
    sems = _sems(_writer(), _kernel(StreamK=2, StreamKAtomic=0))
    for expected in ["AddressWorkspace", "AddressFlags", "ItersPerTile",
                     "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                     "SKItersPerWG", "SKGrid", "SKTilesAndSplit"]:
        assert expected in sems


def test_streamk_basic_omits_grid_and_split():
    sems = _sems(_writer(), _kernel(StreamK=1, StreamKAtomic=0))
    assert "SKItersPerWG" in sems
    assert "SKGrid" not in sems and "SKTilesAndSplit" not in sems


def test_streamk_atomic_omits_workspace_addresses():
    sems = _sems(_writer(), _kernel(StreamK=2, StreamKAtomic=1))
    assert "AddressWorkspace" not in sems and "AddressFlags" not in sems


# --------------------------------------------------------------------------- #
# Scale / bias / factordim / E / activation / amax / MBSK
# --------------------------------------------------------------------------- #


def test_scale_addresses():
    sems = _sems(_writer(), _kernel(
        ProblemType={"UseScaleAB": True, "UseScaleCD": True, "UseScaleAlphaVec": 1}))
    for expected in ["AddressScaleA", "AddressScaleB", "AddressScaleC",
                     "AddressScaleD", "AddressScaleAlphaVec"]:
        assert expected in sems


def test_bias_address_with_type():
    sems = _sems(_writer(useBias=DataDirection.READ, needBiasType=True), _kernel())
    assert "AddressBias" in sems and "BiasType" in sems and "StrideBias" in sems


def test_bias_address_without_type():
    sems = _sems(_writer(useBias=DataDirection.READ, needBiasType=False), _kernel())
    assert "AddressBias" in sems
    assert "BiasType" not in sems and "StrideBias" not in sems


def test_factordim_from_scale_alpha_vec():
    assert "FactorDim" in _sems(_writer(), _kernel(ProblemType={"UseScaleAlphaVec": 3}))


def test_factordim_from_bias():
    sems = _sems(_writer(needBiasType=True), _kernel(ProblemType={"UseBias": 3}))
    assert "FactorDim" in sems


def test_use_e_adds_address_and_strides():
    sems = _sems(_writer(), _kernel(ProblemType={"UseE": True}))
    assert "AddressE" in sems and "StrideE0" in sems


def test_activation_args_and_type_arg():
    kernel = _kernel(ProblemType={"ActivationType": ActivationType("all")}, ActivationFused=True)
    sems = _sems(_writer(), kernel)
    assert sems.count("ActivationArg") == 2   # activationAlpha, activationBeta
    assert "ActivationTypeArg" in sems         # 'all' -> runtime type selector


def test_activation_skipped_when_not_fused():
    kernel = _kernel(ProblemType={"ActivationType": ActivationType("all")}, ActivationFused=False)
    sems = _sems(_writer(), kernel)
    assert "ActivationArg" not in sems


def test_activation_non_all_omits_type_arg():
    # A specific fused activation with its own arg(s) emits ActivationArg but no
    # runtime ActivationTypeArg (that is only for the 'all'/'hipblaslt_all' enums).
    kernel = _kernel(ProblemType={"ActivationType": ActivationType("leakyrelu")},
                     ActivationFused=True)
    sems = _sems(_writer(), kernel)
    assert sems.count("ActivationArg") == 1
    assert "ActivationTypeArg" not in sems


def test_amax_output_addresses():
    sems = _sems(_writer(), _kernel(ProblemType={"OutputAmaxD": True}))
    for expected in ["AddressAmaxOut", "AmaxWS", "AmaxSync"]:
        assert expected in sems


def test_mbsk_trailing_args_from_algorithm():
    sems = _sems(_writer(), _kernel(GlobalSplitUAlgorithm="MultipleBufferSingleKernel"))
    assert sems[-3:] == ["AddressTD", "Synchronizer", "GSUSync"]


def test_mbsk_trailing_args_from_adaptive_gsua():
    sems = _sems(_writer(), _kernel(AdaptiveGemmGSUA=1))
    for expected in ["AddressTD", "Synchronizer", "GSUSync"]:
        assert expected in sems
