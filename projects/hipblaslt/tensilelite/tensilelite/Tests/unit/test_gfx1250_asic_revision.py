# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx1250 ASIC revision split via a distinct architecture name.

gfx1250 ships in two ASIC revisions that report the same ISA and use the same
compiler target. v0 is modelled as the architecture name ``gfx1250v0``; v1 keeps
the plain ``gfx1250`` name. Both canonicalize to ``IsaVersion(12,5,0)`` and both
assemble at ``-mcpu=gfx1250``, so the ASIC revision is invisible below the build's
capability map.

Because the two ASIC revisions are indistinguishable by ISA, the assembler-probed
capability table cannot tell them apart. The v0 deltas are therefore *declared*
in ``ARCH_CAP_OVERRIDES`` and applied on top of the probed caps, which turns the
two silicon differences v0 has to express -- no TDM-multicast, no fp4 32x16 WMMA
-- into ordinary capability reads in Solution derivation.

There is deliberately no solution parameter and no kernel-name difference: one
build targets exactly one ASIC revision, so same-named kernels never coexist.

The tests that derive real solutions need gfx1250 capabilities (``amdclang++``
targeting gfx1250) and are skipped when the toolchain is unavailable.
"""

import copy
import inspect
import os
import sys
from pathlib import Path

import pytest

from Tensile.Common.Architectures import (
    ARCH_CAP_OVERRIDES,
    ARCH_COMPILER_TARGET,
    SUPPORTED_GFX,
    architectureMap,
    expandAllArchitectures,
    gfxToCompilerTarget,
    gfxToIsa,
    isaToGfx,
)
from Tensile.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.Types import IsaInfo, IsaVersion
from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

# The `codegen_harness` helper (shared by the characterization codegen suites)
# lives in a sibling directory that is not on sys.path for this test root, so the
# codegen test adds it lazily inside `_emit` below.
_CODEGEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "characterization", "_codegen"
)

GFX1250 = "gfx1250"
GFX1250V0 = "gfx1250v0"
ISA_GFX1250 = IsaVersion(12, 5, 0)

# The two capabilities v0 lacks. Absent from the probed table, so every consumer
# reads them with a True default and no other architecture changes behavior.
# HasTDMMulticast is architectural (archCaps); HasWMMA_f4_32x16 is an opcode (asmCaps).
CAP_MULTICAST = "HasTDMMulticast"
CAP_FP4_32X16 = "HasWMMA_f4_32x16"
# A probed archCap v0 shares with v1 (present in the table, NOT overridden: v0
# has the same XNACK-replay hazard and keeps the drain).
CAP_XCNT = "RequiresXCntForVolatileVMEM"

FP4_32X16_REASON = "does not support the fp4 32x16 matrix-instruction shape"

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# =========================================================================== #
# Architecture registration. No toolchain needed.
# =========================================================================== #
def test_gfx1250v0_is_registered_as_an_architecture():
    """``--architecture gfx1250v0`` must survive the build driver's allow-list."""
    assert GFX1250V0 in architectureMap


def test_both_asic_revisions_canonicalize_to_the_gfx1250_compiler_target():
    """The whole design rests on this: the arch name carries the ASIC revision, the
    ISA tuple does not, so every target string derived from the tuple
    (``-mcpu``, ``--offload-arch``, ``.amdgcn_target``) stays gfx1250 for both."""
    assert gfxToIsa(GFX1250V0) == ISA_GFX1250
    assert gfxToIsa(GFX1250) == ISA_GFX1250
    assert isaToGfx(gfxToIsa(GFX1250V0)) == GFX1250


def test_gfx1250v0_is_absent_from_the_isa_derived_names():
    """``SUPPORTED_GFX`` -- what ``all`` expands to -- is derived from ISA tuples,
    so it can name only one architecture per ISA and that one is v1. v0 is a
    bring-up target and must be requested explicitly."""
    assert GFX1250V0 not in SUPPORTED_GFX
    assert GFX1250 in SUPPORTED_GFX


def test_all_keeps_architectures_its_expansion_does_not_cover():
    """``all`` cannot name v0, so an explicit ``gfx1250v0`` alongside it has to
    survive rather than be dropped -- being dropped would turn a request for an
    ASIC revision that cannot be built together with v1 into a silent v1-only build.
    Kept, it reaches the mixed-build guard and reports the conflict."""
    expanded = expandAllArchitectures(["all", GFX1250V0])
    assert GFX1250V0 in expanded
    assert set(SUPPORTED_GFX).issubset(expanded)


def test_all_expansion_does_not_duplicate_covered_architectures():
    assert expandAllArchitectures(["all", "gfx942"]) == SUPPORTED_GFX


def test_expansion_is_a_passthrough_without_the_all_keyword():
    assert expandAllArchitectures([GFX1250V0, "gfx942"]) == [GFX1250V0, "gfx942"]


@pytest.mark.parametrize("spec", ["gfx950[cu=64]", "gfx942:xnack+", "gfx942[id=74a0]"])
def test_all_absorbs_qualified_specs_of_architectures_it_covers(spec):
    """Only names ``all`` genuinely cannot express may survive it. A predicate or
    xnack spec names an architecture the expansion already covers, so keeping it
    would both change behavior for architectures unrelated to the ASIC revision split
    and hand the predicate splitter a duplicate of that architecture."""
    assert expandAllArchitectures(["all", spec]) == SUPPORTED_GFX


@pytest.mark.parametrize("padding", ["", " ", "\t"])
def test_all_tolerates_empty_entries(padding):
    """cmake joins GPU_TARGETS with ``;``, so an empty element arrives here as an
    empty spec (``--architecture=all;``). Everything beside ``all`` used to be
    discarded, empty entries included; keeping only genuinely uncoverable names
    must not turn that into a hard build failure, since the predicate splitter
    rejects any spec it cannot recognize."""
    assert expandAllArchitectures(["all", padding]) == SUPPORTED_GFX


@pytest.mark.parametrize("padding", ["", " ", "\t"])
def test_empty_entries_are_dropped_without_the_all_keyword(padding):
    """``GPU_TARGETS=gfx1250v0;`` arrives as a trailing empty spec too, and there is
    no ``all`` to absorb it: the list is returned verbatim, so the empty entry
    reaches the predicate splitter and fails the build on a request that is
    otherwise valid."""
    assert expandAllArchitectures([GFX1250V0, padding]) == [GFX1250V0]


@pytest.mark.parametrize("keyword", [" all", "all ", "\tall"])
def test_all_is_recognized_despite_surrounding_whitespace(keyword):
    """Membership was tested on the raw entry while the filter compared the
    stripped one, so a padded keyword skipped expansion and was handed to the
    predicate splitter as an architecture named ``all``."""
    assert expandAllArchitectures([keyword, GFX1250V0]) == SUPPORTED_GFX + [GFX1250V0]


# =========================================================================== #
# Compiler target for each architecture name. The ASIC revision lives in the name
# only; every compiler invocation has to fall back to a target clang knows.
# =========================================================================== #
def test_v0_compiles_at_the_plain_gfx1250_target():
    """``--offload-arch=gfx1250v0`` fails with `unsupported HIP gpu
    architecture`, so the HIP helper-kernel compile must be given gfx1250."""
    assert gfxToCompilerTarget(GFX1250V0) == GFX1250


def test_compiler_target_preserves_xnack_qualifiers():
    """The mapping must not detour through the ISA tuple: that drops ``:xnack±``,
    which is part of the target for every gfx9 architecture."""
    assert gfxToCompilerTarget("gfx942:xnack+") == "gfx942:xnack+"
    assert gfxToCompilerTarget("gfx950:xnack-") == "gfx950:xnack-"
    assert gfxToCompilerTarget(GFX1250) == GFX1250


def test_every_architecture_resolves_to_a_supported_compiler_target():
    """A future ASIC revision name added to ``architectureMap`` without an alias here
    would reach ``--offload-arch`` verbatim and break the build."""
    placeholders = {"all", "gfx000"}
    for name in architectureMap:
        if name in placeholders:
            continue
        base = gfxToCompilerTarget(name).split(":")[0]
        assert base in SUPPORTED_GFX, f"{name} has no compiler target clang accepts"


def test_only_v0_needs_a_compiler_target_alias():
    assert set(ARCH_COMPILER_TARGET) == {GFX1250V0}


# =========================================================================== #
# The capability-override mechanism itself. No toolchain needed: the override
# step only rewrites dict entries, so a synthetic capability map exercises it.
# =========================================================================== #
def _synthetic_iim():
    """An ISA map with the two v0-sensitive caps absent, as the probe leaves them."""
    return {ISA_GFX1250: IsaInfo({"SupportedISA": True}, {}, {}, {})}


def test_overrides_turn_off_both_v0_capabilities():
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [GFX1250V0])
    assert iim[ISA_GFX1250].archCaps[CAP_MULTICAST] is False
    assert iim[ISA_GFX1250].asmCaps[CAP_FP4_32X16] is False


def test_v1_leaves_both_capabilities_at_their_default():
    """v1 declares no overrides, so the caps stay absent and consumers'
    ``get(..., True)`` default keeps today's behavior."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [GFX1250])
    assert CAP_MULTICAST not in iim[ISA_GFX1250].archCaps
    assert CAP_FP4_32X16 not in iim[ISA_GFX1250].asmCaps


def test_overrides_declared_only_for_v0():
    assert set(ARCH_CAP_OVERRIDES) == {GFX1250V0}


def test_unknown_arch_name_is_ignored_by_the_override_step():
    """Names without declared deltas must pass through untouched rather than
    raising, since every build passes its full requested-arch list."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, ["gfx942", "gfx950"])
    assert iim[ISA_GFX1250].asmCaps == {"SupportedISA": True}


def test_mixed_asic_revision_build_is_rejected():
    """Both names key the same IsaVersion, so a single capability map cannot
    describe both; combined with identical kernel names a mixed build would
    silently emit one ASIC revision's kernels under the other's caps."""
    # Matched on the conflict wording, not just the name: the same function also
    # raises for a requested ASIC revision absent from the map, and either message
    # would otherwise satisfy this test.
    with pytest.raises(ValueError, match="share ISA"):
        applyArchCapOverrides(_synthetic_iim(), [GFX1250, GFX1250V0])


@pytest.mark.parametrize("spec", ["gfx1250v0[cu=64]", "gfx1250v0[id=1250]"])
def test_predicated_asic_revision_still_gets_its_overrides(spec):
    """``--gpu-targets`` accepts a predicate on any architecture and forwards the
    spec verbatim, so the lookup has to see past it. Missing here is the worst
    case the split can produce: the build is accepted, reports v0, and derives
    every solution under the shipping ASIC revision's capabilities."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [spec])
    assert iim[ISA_GFX1250].archCaps[CAP_MULTICAST] is False
    assert iim[ISA_GFX1250].asmCaps[CAP_FP4_32X16] is False


def test_predicated_asic_revision_still_conflicts_with_the_other_asic_revision():
    """The mixed-build guard compares declared deltas, so a predicate that hides
    the deltas also hides the conflict."""
    with pytest.raises(ValueError, match="share ISA"):
        applyArchCapOverrides(_synthetic_iim(), [GFX1250, "gfx1250v0[cu=64]"])


def test_repeating_one_asic_revision_is_not_a_conflict():
    """Only *differing* capabilities conflict. A name repeated by the caller, and
    two qualified variants of one architecture (which share an ISA and declare no
    deltas), must both still build."""
    applyArchCapOverrides(_synthetic_iim(), [GFX1250V0, GFX1250V0])
    applyArchCapOverrides(_synthetic_iim(), ["gfx942:xnack+", "gfx942:xnack-"])


def test_requested_asic_revision_missing_from_the_capability_map_is_an_error():
    """Silently skipping would hand v0 the shipping ASIC revision's capabilities --
    the one outcome the override step exists to prevent -- so a map that cannot
    carry the declared deltas has to fail loudly."""
    with pytest.raises(ValueError, match=GFX1250V0):
        applyArchCapOverrides({}, [GFX1250V0])


def test_makeisainfomap_keeps_its_two_argument_signature():
    """Suites across the repo stub this seam with ``lambda isa_list, _compiler``.
    Applying the overrides is therefore a separate step at the entry points
    rather than an extra argument here, which would break every one of them."""
    assert list(inspect.signature(makeIsaInfoMap).parameters) == [
        "targetIsas",
        "cxxCompiler",
    ]


# =========================================================================== #
# Shared toolchain harness (real gfx1250 caps + assembler). The base solution is
# the MXFP4 (F4/F4/S) TN config from ``mxf4_gfx1250.yaml``; each test flips only
# MatrixInstruction / ClusterDim, and selects an ASIC revision by capability map.
# =========================================================================== #
@pytest.fixture(scope="module")
def gfx1250_cxx():
    """The compiler the capability probe assembles with, or a clean skip."""
    from Tensile.Toolchain.Validators import validateToolchain

    try:
        return validateToolchain("amdclang++")
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"amdclang++ is unavailable: {e}")


@pytest.fixture(scope="module")
def gfx1250_iim(gfx1250_cxx):
    """v1: exactly what an entry point produces for ``--architecture gfx1250``."""
    iim = makeIsaInfoMap([ISA_GFX1250], gfx1250_cxx)
    if not iim[ISA_GFX1250].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    applyArchCapOverrides(iim, [GFX1250])
    return iim


@pytest.fixture(scope="module")
def gfx1250v0_iim(gfx1250_iim):
    """v0: the same two steps an entry point runs for ``--architecture gfx1250v0``.

    Routed through the production override function rather than hand-patching
    keys, so the fixture cannot drift from what a real v0 build sees.
    """
    iim = copy.deepcopy(gfx1250_iim)
    applyArchCapOverrides(iim, [GFX1250V0])
    return iim


def test_probe_leaves_both_capabilities_absent_for_the_override_to_set(gfx1250_iim):
    """The premise the ``True`` defaults in Solution rest on: the real assembler
    probe never reports these keys, so v1 is byte-identical to before the split
    and v0 is the only architecture that changes behavior."""
    assert CAP_MULTICAST not in gfx1250_iim[ISA_GFX1250].archCaps
    assert CAP_FP4_32X16 not in gfx1250_iim[ISA_GFX1250].asmCaps


def test_overrides_apply_on_top_of_really_probed_capabilities(gfx1250v0_iim):
    """Probe-then-override composed on a real capability map, not a synthetic one."""
    info = gfx1250v0_iim[ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False
    # Probed entries survive the override, which only adds the declared keys.
    assert info.asmCaps["SupportedISA"]
    assert "HasWMMA" in info.asmCaps


def test_xcnt_is_a_really_probed_archcap_v0_inherits(gfx1250_iim):
    """RequiresXCntForVolatileVMEM is a key rocisa really probes (True by
    default). v0 shares gfx1250's XNACK-replay hazard, so it deliberately does
    NOT override this cap and inherits the probed default -- keeping the drain.
    (HasTDMMulticast also lives in archCaps but is a fill-missing key, guarded by
    the absent-key test above.)"""
    assert CAP_XCNT in gfx1250_iim[ISA_GFX1250].archCaps


@pytest.fixture(scope="module")
def assembler(gfx1250_cxx):
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(gfx1250_cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _problem_type():
    return {
        "OperationType": "GEMM",
        "DataType": "F4",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,  # TN
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "f8",
        "DataTypeMXSB": "f8",
    }


def _make_params(iim, mi, **overrides):
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    problem_type = _problem_type()
    problem_type.update(overrides.pop("ProblemType", {}))

    params = {
        "ProblemType": problem_type,
        "ISA": ISA_GFX1250,
        "MatrixInstruction": mi,
        "WavefrontSize": 32,
        "DepthU": 128,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "TransposeLDS": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": 1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": 32,
        "GlobalReadVectorWidthB": 32,
        "LocalReadVectorWidth": 32,
        "SourceSwap": True,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "TDMInst": 3,
        "LDSTrInst": False,
        "StreamK": 0,
        "StreamKForceDPOnly": 0,
        "PrefetchAcrossPersistent": 0,
        "PrefetchGL2": 0,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
        "UseSgprForGRO": 0,
        "ForceDisableShadowInit": True,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "GlobalReadPerMfma": 1.0,
        "LocalWritePerMfma": -1,
        "AssertSummationElementMultiple": 32,
    }
    params.update(overrides)
    params.update(
        matrixInstructionToMIParameters(
            mi, ISA_GFX1250, params["WavefrontSize"], problem_type, None, iim
        )
    )
    return params


def _derive(iim, assembler, capsys, mi, **overrides):
    sol = Solution(_make_params(iim, mi, **overrides), False, True, False, assembler, iim)
    return sol, capsys.readouterr().out


# =========================================================================== #
# fp4 32x16 gating.
# =========================================================================== #
FP4_32X16_MI = [32, 16, 128, 1, 1, 2, 1, 2, 1]


def test_fp4_32x16_accepted_on_gfx1250(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, FP4_32X16_MI)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert FP4_32X16_REASON not in out
    # Anchor the physical-vs-effective distinction: SourceSwap=True transposes the
    # effective MatrixInstM to 16 while the physical MIBlock[0] stays 32. This is
    # what makes the reject test below a genuine test of MIBlock gating.
    assert sol["MIBlock"][0] == 32 and sol["MIBlock"][1] == 16
    assert sol["MatrixInstM"] == 16 and sol["MatrixInstN"] == 32


def test_fp4_32x16_rejected_on_gfx1250v0(_gp_gfx1250, gfx1250v0_iim, assembler, capsys):
    """v0 lacks the fp4 32x16 WMMA opcode.

    With SourceSwap=True the effective MatrixInstM is 16, so a guard on
    MatrixInstM==32 would NOT fire here -- this only passes because the guard
    uses the physical MIBlock[0]==32.
    """
    sol, out = _derive(gfx1250v0_iim, assembler, capsys, FP4_32X16_MI)
    assert sol.get("Valid") is False
    assert FP4_32X16_REASON in out


# v0's 16x16 fp4 acceptance (the restriction is specific to 32x16) is covered by
# test_multicast_forced_off_on_gfx1250v0 below, which derives a valid 16x16 v0
# solution -- so no separate 16x16-accept test is needed.


def test_fp4_32x16_rejected_on_gfx1250v0_without_source_swap(
    _gp_gfx1250, gfx1250_iim, gfx1250v0_iim, assembler, capsys
):
    """The other half of the SourceSwap space, and a real shipped shape
    (streamk/gfx1250/sk_mxf4gemm_tdm_ext.yaml is 32x16 fp4 with SourceSwap false).

    Here the effective dims equal the physical ones, so this passes under either
    gate -- together with the SourceSwap=True case above it pins that the gate
    covers both, which only the physical MIBlock does. The v1 leg keeps the
    rejection attributable to the ASIC revision rather than to the config.
    """
    v1, _ = _derive(gfx1250_iim, assembler, capsys, FP4_32X16_MI, SourceSwap=False)
    assert v1.get("Valid") is True, "SourceSwap=False 32x16 fp4 must derive on v1"

    sol, out = _derive(
        gfx1250v0_iim, assembler, capsys, FP4_32X16_MI, SourceSwap=False
    )
    assert sol["MIBlock"][0] == 32 and sol["MIBlock"][1] == 16
    assert sol.get("Valid") is False
    assert FP4_32X16_REASON in out


MIXED_MAC = {"DataType": "F8", "MacDataTypeA": "F8", "MacDataTypeB": "F4",
             "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"}

# The mixed-operand shape needs the vector widths the shipped config uses
# (sk_mxf8f4gemm_tdm.yaml: all three auto, DepthU 256, no SourceSwap). With the
# fp4 base's fixed widths it is rejected on *both* ASIC revisions -- F8 wants
# lrvwA == 16 while F4 wants lrvwB == 32, and one LocalReadVectorWidth cannot be
# both -- which would make the reject below prove nothing about the ASIC revision.
MIXED_MAC_PARAMS = dict(
    GlobalReadVectorWidthA=-1,
    GlobalReadVectorWidthB=-1,
    LocalReadVectorWidth=-1,
    DepthU=256,
    SourceSwap=False,
    TransposeLDS=-1,
)


def test_mixed_mac_type_fp4_is_covered_by_the_same_gate(
    _gp_gfx1250, gfx1250_iim, gfx1250v0_iim, assembler, capsys
):
    """The opcode is selected by the WMMA operand format (MacDataType), so a gate
    keyed on the memory format (DataTypeA/B) would look like a hole for configs
    such as sk_mxf8f4gemm_tdm.yaml (``MacDataTypeA: F8, MacDataTypeB: F4``).

    It is not one: ``DataTypeB`` is *defaulted from* ``MacDataTypeB`` whenever the
    config does not set it explicitly (Problem.py), and ``getRealDataTypeB`` only
    remaps the F8/BF8 pair -- never fp4. So MacDataType fp4 always implies
    DataType fp4 and the two keys cannot diverge here. This pins that equivalence
    so the gate is not "fixed" into keying on something it need not.

    The v1 leg is what makes the v0 reject meaningful: it shows the config is
    derivable, so the rejection comes from the ASIC revision and not from the config.
    """
    v1, _ = _derive(
        gfx1250_iim, assembler, capsys, FP4_32X16_MI,
        ProblemType=dict(MIXED_MAC), **MIXED_MAC_PARAMS
    )
    assert v1.get("Valid") is True, "mixed-operand config must be derivable on v1"
    assert v1["ProblemType"]["MacDataTypeB"].isFloat4()
    assert v1["ProblemType"]["DataTypeB"].isFloat4()
    assert v1["MIBlock"][0] == 32 and v1["MIBlock"][1] == 16

    v0, out = _derive(
        gfx1250v0_iim, assembler, capsys, FP4_32X16_MI,
        ProblemType=dict(MIXED_MAC), **MIXED_MAC_PARAMS
    )
    assert v0.get("Valid") is False
    assert FP4_32X16_REASON in out


# =========================================================================== #
# TDM-multicast gating (ClusterDim != [1,1], StreamK == 0).
# =========================================================================== #
MULTICAST_MI = [16, 16, 128, 1, 1, 2, 2, 2, 2]


def test_multicast_on_for_gfx1250(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["Multicast"] is True
    assert sol["ClusterBarrier"] is True


def test_multicast_forced_off_on_gfx1250v0(_gp_gfx1250, gfx1250v0_iim, assembler, capsys):
    """v0 has no TDM-multicast, but clustering itself stays valid."""
    sol, out = _derive(gfx1250v0_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["Multicast"] is False
    # ClusterBarrier is a separate feature v0 supports; it must stay enabled.
    assert sol["ClusterBarrier"] is True


# =========================================================================== #
# Naming invariant. The ASIC revisions are separate builds, so their kernels must NOT
# be named apart -- an ASIC revision token in the name would desynchronize the shipped
# library logic (which stores KernelNameMin at tuning time) from the emitted
# symbol. This fails loudly if anyone reintroduces one.
# =========================================================================== #
def test_kernel_names_identical_across_asic_revisions(
    _gp_gfx1250, gfx1250_iim, gfx1250v0_iim, assembler, capsys
):
    v1, _ = _derive(gfx1250_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    v0, _ = _derive(gfx1250v0_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    assert v1.get("Valid") is True and v0.get("Valid") is True
    # Derived state genuinely differs, so this is not a vacuous comparison.
    assert v1["Multicast"] != v0["Multicast"]
    assert getKernelNameMin(v1, False) == getKernelNameMin(v0, False)
    assert getSolutionNameMin(v1, False) == getSolutionNameMin(v0, False)


# =========================================================================== #
# End-to-end codegen split. Every TDM-multicast emission site gates on the
# derived ``kernel["Multicast"]``, so the capability override drives the codegen
# difference with no new codegen branch.
# =========================================================================== #
# Multicast is purely descriptor-driven: the ONLY writer of the descriptor's
# multicast bit is the `setMulticastMask` component (Components/TensorDataMover.py),
# which reads the `MulticastMask*` sgprs. `MulticastMask` appears as a non-comment
# `.set sgprMulticastMask*` directive, so it survives DisableAsmComments and
# canonicalize_asm.
MULTICAST_MARKERS = ("MulticastMask", "multicast mask")


def _emit(archName):
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    if _CODEGEN_DIR not in sys.path:
        sys.path.insert(0, _CODEGEN_DIR)
    from codegen_harness import (
        _init_rocisa_for,
        _isolated_globals,
        _prepare_kernel,
        canonicalize_asm,
        get_assembler,
        get_isa_info_map,
    )

    asm = get_assembler()
    iim = copy.deepcopy(get_isa_info_map())
    if not iim[ISA_GFX1250].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    applyArchCapOverrides(iim, [archName])

    # Solution derivation + assignGlobalParameters mutate process-global state
    # (globalParameters / validParameters); isolate so this never leaks.
    with _isolated_globals():
        sol = Solution(
            _make_params(iim, MULTICAST_MI, ClusterDim=[2, 1]),
            False,
            True,
            False,
            asm,
            iim,
        )
        assert sol.get("Valid") is True, f"{archName} base solution unexpectedly rejected"

        kernels = generateKernelObjectsFromSolutions([sol])
        assert len(kernels) == 1
        kernel = kernels[0]
        assert kernel["Multicast"] is (archName != GFX1250V0)

        kwa = KernelWriterAssembly(asm, DebugConfig())
        ri = _init_rocisa_for(kernel)
        _prepare_kernel(kernel, False)
        res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        return canonicalize_asm(res.src), res.err


def test_gfx1250_emits_multicast_gfx1250v0_does_not(gfx1250_cxx):
    v1_src, v1_err = _emit(GFX1250)
    v0_src, v0_err = _emit(GFX1250V0)
    assert v1_err == 0 and v0_err == 0

    assert any(m in v1_src for m in MULTICAST_MARKERS), (
        "gfx1250 with ClusterDim!=[1,1] should emit multicast mask setup"
    )
    assert not any(m in v0_src for m in MULTICAST_MARKERS), (
        "gfx1250v0 must not emit any multicast instructions"
    )
    assert v1_src != v0_src


# =========================================================================== #
# XNACK-replay drain (``s_wait_xcnt``). Unlike multicast, this is a codegen-time
# arch capability, not a solution parameter, so it does not travel through
# ``applyArchCapOverrides``/``isaInfoMap``. The kernel writer reads
# ``RequiresXCntForVolatileVMEM`` straight from the rocisa singleton
# (``ti.getArchCaps()``), which is keyed by ISA (12,5,0) and cannot tell
# gfx1250's ASIC revisions apart. v0 shares the same XNACK-replay hazard as v1, so it
# keeps the drain: ``RequiresXCntForVolatileVMEM`` is intentionally NOT in v0's
# ARCH_CAP_OVERRIDES. The test derives on the gfx1250 (v1) capability map and
# emits the same kernel under both ASIC revision signals to pin that v0 no longer
# drops the drain -- both must emit it, in equal number.
# =========================================================================== #
_STREAMK_CONFIG = os.path.join(
    _CODEGEN_DIR, "data", "test_data", "_designed", "gfx1250", "streamk.yaml"
)

# StreamK tags every XNACK-replay drain with this fixed comment (its
# ``preVolatileVmem`` call sites in Components/StreamK.py). Counting the marker
# isolates the drain exactly, independent of any other ``s_wait_xcnt`` a pass
# might emit, so the v1-vs-v0 comparison cannot be confounded.
_XCNT_DRAIN_MARKER = "drain xnacks before volatile VMEM"


def _emit_streamk_srcs(asic_revision):
    """Emit the gfx1250 StreamK config's kernels, optionally under a v0 ASIC revision.

    Mirrors ``config_harness.emit_kernels_from_config`` but sets
    ``globalParameters["StinkyTofuArchName"]`` inside the isolated globals so the
    codegen seam sees the ASIC revision (the harness never sets it). Returns the list
    of canonicalized assembly strings.
    """
    if _CODEGEN_DIR not in sys.path:
        sys.path.insert(0, _CODEGEN_DIR)
    import codegen_harness as ch
    import config_harness as cfgh
    from Tensile.Common.GlobalParameters import globalParameters
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    assembler, iim = cfgh._toolchain_for(GFX1250)
    if not iim[ISA_GFX1250].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")

    srcs = []
    with cfgh._isolated_globals_with_isa(iim):
        if asic_revision:
            globalParameters["StinkyTofuArchName"] = asic_revision
        sols = cfgh._solutions_from_config_unguarded(
            _STREAMK_CONFIG, assembler, iim, limit_solutions=8
        )
        kernels = generateKernelObjectsFromSolutions(sols)
        kernels = sorted(kernels, key=lambda k: getKernelFileBase(False, k))[:8]
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        for kernel in kernels:
            ri = ch._init_rocisa_for(kernel)
            ch._prepare_kernel(kernel, False)
            res = processKernelSource(
                kwa, ri.getData(), ri.getOutputOptions(), False, kernel
            )
            srcs.append(ch.canonicalize_asm(res.src))
    return srcs


def test_gfx1250v0_streamk_keeps_the_xcnt_drain(gfx1250_cxx):
    """The XNACK-replay drain is gated on ``RequiresXCntForVolatileVMEM``. v0
    shares gfx1250's hazard and does NOT override that cap, so it must keep the
    drain. Emitting the *same* StreamK kernel under v1 vs v0 and counting the
    drain marker isolates exactly the drain: v1 must keep it (the ``> 0`` check
    guards against a vacuous pass) and v0 must keep every one too."""
    v1_drains = sum(s.count(_XCNT_DRAIN_MARKER) for s in _emit_streamk_srcs(asic_revision=None))
    v0_drains = sum(s.count(_XCNT_DRAIN_MARKER) for s in _emit_streamk_srcs(asic_revision=GFX1250V0))
    assert v1_drains > 0, (
        "gfx1250 (v1) StreamK should emit the XNACK-replay drain; without it "
        "the v0 assertion below would pass vacuously"
    )
    assert v0_drains == v1_drains, (
        f"gfx1250 v0 must keep every RequiresXCntForVolatileVMEM drain "
        f"(v1 drains={v1_drains}, v0 drains={v0_drains})"
    )


# =========================================================================== #
# Round-trip. The ASIC revision lives in the build's capability map, not in the
# solution, so a re-parsed solution re-derives Multicast from whichever map the
# reading build uses. That is inherent to having no solution parameter; this
# pins it as a known property rather than leaving it a latent surprise.
# =========================================================================== #
def test_multicast_rederived_from_build_caps_after_yaml_roundtrip(
    _gp_gfx1250, gfx1250_iim, gfx1250v0_iim, assembler, capsys, tmp_path
):
    from Tensile import LibraryIO
    from Tensile.SolutionStructs import ProblemSizes

    sol, out = _derive(gfx1250v0_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    assert sol.get("Valid") is True, f"base solution rejected: {out!r}"
    assert sol["Multicast"] is False

    problemSizes = ProblemSizes(sol["ProblemType"], [{"Exact": [128, 128, 1, 128]}])
    path = str(tmp_path / "solutions.yaml")
    LibraryIO.writeSolutions(path, problemSizes, None, None, [sol])
    data = LibraryIO.read(path)

    # ISA round-trips as a plain list; Solution construction normalizes it back to
    # the tuple the capability map is keyed by, which is what the re-derivation
    # below relies on.
    assert data[-1]["ISA"] == list(ISA_GFX1250)

    _sizes, parsed = LibraryIO.parseSolutionsData(
        data,
        path,
        assembler,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        isaInfoMap=gfx1250v0_iim,
    )
    capsys.readouterr()
    assert len(parsed) == 1
    assert parsed[0].get("Valid") is True
    assert parsed[0]["Multicast"] is False

    # The discriminating half: re-read the *same* v0-written file under the v1
    # capability map. Multicast comes back True, which shows the value is derived
    # from the reading build's caps rather than restored from the file.
    _sizes, reparsed = LibraryIO.parseSolutionsData(
        LibraryIO.read(path),
        path,
        assembler,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        isaInfoMap=gfx1250_iim,
    )
    capsys.readouterr()
    assert reparsed[0]["Multicast"] is True


# =========================================================================== #
# Entry-point wiring.
#
# Everything above proves the override mechanism works *when called*. These prove
# the production entry points actually call it, which nothing else in the repo
# does: deleting the `applyArchCapOverrides` calls and the `gfxToCompilerTarget`
# normalization leaves the rest of this file (and the TensileCreateLibrary
# characterization suites) green, because a missing override simply leaves the
# capability map at v1's values -- the exact silent failure the feature exists to
# prevent.
#
# Both tests stub the whole pipeline, so they need no toolchain and no GPU.
# =========================================================================== #
def _stub_iim():
    """What `makeIsaInfoMap` returns for a gfx1250 build, minus the real probe."""
    return {ISA_GFX1250: IsaInfo({"SupportedISA": True}, {}, {}, {})}


@pytest.fixture
def restore_global_parameters():
    from Tensile.Common.GlobalParameters import globalParameters

    saved = copy.deepcopy(dict(globalParameters))
    yield globalParameters
    globalParameters.clear()
    globalParameters.update(saved)


def _write_min_config(tmp_path, **globalParams):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "GlobalParameters": {
                    "MinimumRequiredVersion": "5.0.0",
                    **globalParams,
                },
                "BenchmarkProblems": [],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _stub_tensile_pipeline(monkeypatch, captured):
    """Stubs every expensive step of `Tensile()` and captures what the benchmark
    pipeline is handed. Mirrors the stub set in test_tensile_backend_config.py."""
    import types

    from Tensile import Tensile as TensileModule

    monkeypatch.setattr(
        TensileModule, "validateToolchain", lambda *a: ("cxx", "cc", "bundler")
    )
    monkeypatch.setattr(
        TensileModule,
        "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        TensileModule,
        "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeIsaInfoMap", lambda _isas, _compiler: _stub_iim()
    )
    monkeypatch.setattr(TensileModule, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(TensileModule, "argUpdatedGlobalParameters", lambda _args: {})
    monkeypatch.setattr(
        TensileModule,
        "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )

    def _capture(config, outputPath, asmToolchain, srcToolchain, isaInfoMap, *a, **kw):
        captured["isaInfoMap"] = isaInfoMap
        captured["archNames"] = kw.get("archNames")

    monkeypatch.setattr(TensileModule, "executeStepsInConfig", _capture)
    return TensileModule


def test_tensile_entry_point_applies_the_v0_overrides(
    monkeypatch, tmp_path, restore_global_parameters
):
    """``Tensile --gpu-targets gfx1250v0`` must reach the benchmark pipeline with
    v0 capabilities. This is the only channel the ASIC revision travels through."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile(
        [config, str(tmp_path / "out"), "--gpu-targets", GFX1250V0]
    )

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False


def test_tensile_entry_point_leaves_v1_capabilities_untouched(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The same path for v1 must not invent either key, so a plain gfx1250 build
    is byte-identical to one from before the split."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile([config, str(tmp_path / "out"), "--gpu-targets", GFX1250])

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert CAP_FP4_32X16 not in info.asmCaps


# --------------------------------------------------------------------------- #
# `GlobalParameters: Architecture:` in the config. The key predates the split
# and appears in 120 configs, where it was silently ignored. It is the only
# place a config can state an ASIC revision, since the ISA does not distinguish them,
# so it selects the ASIC revision -- and nothing else, to keep those 120 unchanged.
# --------------------------------------------------------------------------- #
def test_config_architecture_selects_the_asic_revision(
    monkeypatch, tmp_path, restore_global_parameters
):
    """A config alone must be able to ask for v0, without the caller having to
    remember ``--gpu-targets``."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path, Architecture=GFX1250V0, ISA=[[12, 5, 0]])

    TensileModule.Tensile([config, str(tmp_path / "out")])

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False
    assert captured["archNames"] == [GFX1250V0]


def test_config_architecture_of_the_shipping_asic_revision_adds_nothing(
    monkeypatch, tmp_path, restore_global_parameters
):
    """What all 120 pre-existing configs say. Honouring the key must leave them
    deriving exactly what they derived when it was ignored."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path, Architecture=GFX1250, ISA=[[12, 5, 0]])

    TensileModule.Tensile([config, str(tmp_path / "out")])

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert CAP_FP4_32X16 not in info.asmCaps


def test_gpu_targets_overrides_the_config_architecture(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The flag is the more specific statement of intent, so a config tuned for v0
    must still be buildable for v1 without editing it."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path, Architecture=GFX1250V0, ISA=[[12, 5, 0]])

    TensileModule.Tensile([config, str(tmp_path / "out"), "--gpu-targets", GFX1250])

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert captured["archNames"] == [GFX1250]


def test_config_architecture_for_an_isa_not_being_built_is_ignored(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The key does not select the ISA, so a name for an ISA this build does not
    cover cannot be adopted -- that would apply an unrelated architecture's
    capabilities. Ignoring it is what happened before the key was honoured."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path, Architecture="gfx942", ISA=[[12, 5, 0]])

    TensileModule.Tensile([config, str(tmp_path / "out")])

    assert captured["archNames"] == []
    assert CAP_MULTICAST not in captured["isaInfoMap"][ISA_GFX1250].archCaps


def test_config_architecture_naming_an_asic_revision_of_another_isa_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The one case where silently ignoring the name is not acceptable: it asks
    for an ASIC revision, and ignoring it builds the shipping one instead."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(tmp_path, Architecture=GFX1250V0, ISA=[[9, 4, 2]])

    with pytest.raises(ValueError) as excinfo:
        TensileModule.Tensile([config, str(tmp_path / "out")])

    assert GFX1250V0 in str(excinfo.value)


@pytest.mark.parametrize("arch", ["gfx1250v1", "gfx1250V0", "gfx1250v"])
def test_unrecognized_config_architecture_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters, arch
):
    """The same near-miss names ``--gpu-targets`` rejects: each resolves to
    (12,5,0) by the ISA regex alone, so without a name check the config would
    quietly build the shipping ASIC revision."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(tmp_path, Architecture=arch, ISA=[[12, 5, 0]])

    with pytest.raises(ValueError) as excinfo:
        TensileModule.Tensile([config, str(tmp_path / "out")])

    assert arch in str(excinfo.value)


def test_mixed_asic_revisions_in_the_config_architecture_are_rejected(
    monkeypatch, tmp_path, restore_global_parameters
):
    """One build is one ASIC revision, whichever layer asked for both."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(
        tmp_path, Architecture=f"{GFX1250};{GFX1250V0}", ISA=[[12, 5, 0]]
    )

    with pytest.raises(ValueError):
        TensileModule.Tensile([config, str(tmp_path / "out")])


def test_tensile_entry_point_records_the_requested_names(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The tuning flow re-spawns TensileCreateLibrary for the client library, and
    the ISA cannot say which ASIC revision this build is for. The requested names have
    to reach the steps so that the re-spawn can ask for the right one."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile(
        [config, str(tmp_path / "out"), "--gpu-targets", GFX1250V0]
    )

    assert captured["archNames"] == [GFX1250V0]


# Deliberately malformed inputs, not spellings this project uses: a suffix that is
# not a registered name, the wrong case, and a truncated suffix. The only correct
# spellings are GFX1250 and GFX1250V0.
@pytest.mark.parametrize("target", ["gfx1250v1", "gfx1250V0", "gfx1250v"])
def test_unknown_gpu_target_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters, target
):
    """Every one of these resolves to ISA (12,5,0) through ``gfxToIsa`` (the regex
    stops at the first non-hex character), so an ISA check alone accepts them and
    silently builds v1. A typo in an ASIC revision name must not produce the other
    ASIC revision."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(tmp_path)

    with pytest.raises(ValueError, match=target):
        TensileModule.Tensile([config, str(tmp_path / "out"), "--gpu-targets", target])


@pytest.mark.parametrize(
    "target",
    [
        "gfx942:sramecc+:xnack-",
        "gfx90a:sramecc+:xnack-",
        "gfx950[cu=64]",
        "gfx942[id=74a0]",
    ],
)
def test_qualified_gpu_targets_stay_accepted(
    monkeypatch, tmp_path, restore_global_parameters, target
):
    """Guarding against ASIC revision typos must not narrow what a GPU target may be.
    The first two are the target-ID form ``rocm_agent_enumerator -v`` prints and a
    user copy-pastes from ``offload-arch``; the last two are the predicate form
    ``--architecture`` accepts. All four resolved to an ISA before the ASIC revision
    split, so rejecting them would be a regression -- and would make the two
    flags disagree about what a GPU target is."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile([config, str(tmp_path / "out"), "--gpu-targets", target])

    assert captured["archNames"] == [target]


def _run_createlibrary(monkeypatch, tmp_path, arch, logicArchName=None):
    """Drives ``TensileCreateLibrary.run()`` for one requested architecture with
    every expensive step stubbed, and returns what the production wiring produced.

    Several separate behaviors are observed here -- the capability overrides, the
    compiler-target normalization, and which logic files survive the filter --
    and they get one test each, so that none is left without coverage if another
    breaks first.

    ``logicArchName`` writes one logic file declaring that architecture into the
    otherwise empty logic directory. The real glob and filter always run; without
    a file they simply have nothing to select.
    """
    from unittest.mock import MagicMock

    import Tensile.TensileCreateLibrary.Run as RunModule

    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    if logicArchName is not None:
        # The filter reads only the third sequence item (CustomYamlLoader's
        # load_logic_gfx_arch), so a full logic file is not needed to exercise it.
        (logic_dir / f"{logicArchName}_test.yaml").write_text(
            "- {MinimumRequiredVersion: 4.33.0}\n"
            f"- {SCHEDULE_NAME}\n"
            f"- {logicArchName}\n"
        )
    captured = {}
    writeSignature = inspect.signature(RunModule.writeSolutionsAndKernelsTCL)

    class _Stop(Exception):
        """Ends run() once both observations are made, so the test does not have
        to stub the whole tail of the function."""

    def _capture_gp(_arguments, isaInfoMap):
        captured["isaInfoMap"] = isaInfoMap

    def _capture_archs(*args, **kwargs):
        # Bound by name against the real signature, captured above before the
        # patch: the argument's position is an implementation detail of
        # writeSolutionsAndKernelsTCL and reading it positionally makes this
        # break confusingly when that signature grows.
        captured["cmdlineArchs"] = writeSignature.bind(*args, **kwargs).arguments[
            "cmdlineArchs"
        ]
        raise _Stop

    monkeypatch.setattr(
        RunModule,
        "parseArguments",
        lambda: {
            "PrintLevel": 1,
            "OutputPath": str(tmp_path / "out"),
            "CxxCompiler": "/fake/hipcc",
            "CCompiler": "/fake/hipcc",
            "OffloadBundler": "/fake/clang-offload-bundler",
            "Assembler": "/fake/assembler",
            "CodeObjectVersion": "4",
            "BuildIdKind": "sha1",
            "AsmDebug": False,
            "AsanBuild": False,
            "Architecture": arch,
            "LogicPath": str(logic_dir),
            "LogicFormat": "yaml",
            "LibraryFormat": "msgpack",
            "CpuThreads": 1,
            "LazyLibraryLoading": True,
            "GenSolTable": False,
            "Experimental": False,
            "LogicFilter": "*",
            "DisableAsmComments": False,
            "UseCompression": False,
            "KeepBuildTmp": False,
        },
    )
    monkeypatch.setattr(RunModule, "setVerbosity", lambda *a, **kw: None)
    monkeypatch.setattr(
        RunModule,
        "validateToolchain",
        lambda *a: ("/fake/hipcc", None, "/fake/bundler", None, None),
    )
    monkeypatch.setattr(RunModule, "makeIsaInfoMap", lambda _isas, _cxx: _stub_iim())
    monkeypatch.setattr(RunModule, "assignGlobalParameters", _capture_gp)
    monkeypatch.setattr(RunModule, "makeAssemblyToolchain", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(RunModule, "makeSourceToolchain", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(RunModule, "KernelWriterAssembly", lambda *a, **kw: MagicMock())
    def _capture_logic_files(logicFiles, *a, **kw):
        captured["logicFiles"] = [Path(f).name for f in logicFiles]
        return ([], {GFX1250: MagicMock(solutions={}, lazyLibraries={})}, {})

    monkeypatch.setattr(
        RunModule, "generateLogicDataAndSolutions", _capture_logic_files
    )
    monkeypatch.setattr(
        RunModule, "generateKernelObjectsFromSolutions", lambda *a, **kw: []
    )
    monkeypatch.setattr(RunModule, "generateKernelHelperObjects", lambda *a, **kw: [])
    monkeypatch.setattr(RunModule, "copyStaticFiles", lambda *a, **kw: [])
    monkeypatch.setattr(RunModule, "writeSolutionsAndKernelsTCL", _capture_archs)

    with pytest.raises(_Stop):
        RunModule.run()

    return captured


def test_createlibrary_entry_point_applies_the_v0_overrides(
    monkeypatch, tmp_path, restore_global_parameters
):
    """``TensileCreateLibrary --architecture=gfx1250v0`` must derive its solutions
    under v0 capabilities; this is the second of the two entry points, and it
    reaches the capability map by a different route than ``Tensile()``."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250V0)

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False


def test_createlibrary_entry_point_normalizes_the_compiler_target(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The ASIC revision name must not reach the kernel writers: they pass it to
    ``--offload-arch``, where clang rejects it as an unsupported architecture."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250V0)

    assert captured["cmdlineArchs"] == [GFX1250]


def test_createlibrary_entry_point_leaves_v1_capabilities_untouched(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The same route for a plain gfx1250 build must invent neither key, so this
    entry point is byte-identical to before the split as well."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250)

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert CAP_FP4_32X16 not in info.asmCaps
    assert captured["cmdlineArchs"] == [GFX1250]


def test_v0_build_consumes_the_architectures_logic_files(
    monkeypatch, tmp_path, restore_global_parameters
):
    """v0 has no tuned logic of its own and is not planned to get any: it reuses
    gfx1250's and re-derives under v0 capabilities. The only thing that lets it is
    ``archMatch``'s ``a.startswith(arch)`` clause, written for xnack variants, so
    the whole feature rests on the ASIC revision name extending its architecture's.

    Pinned here because it fails silently: a name that is not a prefix-extension
    (``gfx1250-v0``) filters every logic file out and the build then reports
    success having written an empty library.
    """
    captured = _run_createlibrary(
        monkeypatch, tmp_path, GFX1250V0, logicArchName=GFX1250
    )

    assert captured["logicFiles"] == [f"{GFX1250}_test.yaml"]


def test_v0_build_ignores_an_unrelated_architectures_logic_files(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The control for the prefix rule above: it must stay a prefix match on the
    requested name, not degrade into accepting whatever logic is on disk."""
    captured = _run_createlibrary(
        monkeypatch, tmp_path, GFX1250V0, logicArchName="gfx942"
    )

    assert captured["logicFiles"] == []


# =========================================================================== #
# Tuning flow. `Tensile` re-spawns `TensileCreateLibrary` to build the client
# library from the logic it just tuned. That re-spawn is a fresh process whose
# only statement of what to build is `--architecture=`, so it is the one place
# the ASIC revision can be lost after being correctly applied everywhere else.
# =========================================================================== #
def _buildTargetGfx(archNames):
    from Tensile.ClientWriter import buildTargetGfx

    return buildTargetGfx(_stub_iim(), archNames)


def test_client_library_is_rebuilt_for_the_requested_asic_revision():
    assert _buildTargetGfx([GFX1250V0]) == GFX1250V0


def test_client_library_target_falls_back_to_the_isa_derived_name():
    """The ISA and auto-detect entry paths never learn a name, so the ISA-derived
    target stays the default rather than the lookup being mandatory. Both an empty
    list and an omitted argument have to behave that way."""
    assert _buildTargetGfx([]) == GFX1250
    assert _buildTargetGfx(None) == GFX1250

    from Tensile.ClientWriter import buildTargetGfx

    assert buildTargetGfx(_stub_iim()) == GFX1250


def test_client_library_target_ignores_names_from_other_architectures():
    """A multi-architecture build must not label this ISA with an unrelated
    requested name; the lookup is keyed by ISA, not by position."""
    assert _buildTargetGfx(["gfx942"]) == GFX1250


def test_client_library_target_ignores_qualifiers_of_other_architectures():
    """Only an ASIC revision needs a name the ISA cannot express. Forwarding a requested
    qualifier verbatim would rebuild the client library for one xnack setting where
    an xnack-agnostic code object is wanted -- and since the library directory and
    the .co filter both still resolve, that surfaces when the device loads it, not
    at build time."""
    iim = {IsaVersion(9, 4, 2): IsaInfo({"SupportedISA": True}, {}, {}, {})}

    from Tensile.ClientWriter import buildTargetGfx

    assert buildTargetGfx(iim, ["gfx942:xnack+"]) == "gfx942"


def test_client_library_target_keeps_the_asic_revision_under_a_predicate():
    """The predicate is dropped -- the re-spawn resolves logic files itself, and an
    unqualified name is what every other architecture already gets here -- but the
    ASIC revision must survive, or the tuning flow rebuilds the client library for the
    shipping ASIC revision while reporting v0."""
    assert _buildTargetGfx(["gfx1250v0[cu=64]"]) == GFX1250V0


def test_client_library_target_picks_the_name_matching_the_rebuilt_isa():
    """A multi-architecture build (``--gpu-targets 'gfx942;gfx1250v0'`` is allowed,
    the ISAs differ) must resolve the name for the ISA actually being rebuilt.
    Taking the sole requested name, or the first one, would rebuild v0's client
    library against the shipping ASIC revision the moment a second architecture is asked
    for -- silently, since the name it lands on is still a valid target."""
    from Tensile.ClientWriter import buildTargetGfx

    caps = ({"SupportedISA": True}, {}, {}, {})
    both = ["gfx942", GFX1250V0]

    assert buildTargetGfx(
        {ISA_GFX1250: IsaInfo(*caps), IsaVersion(9, 4, 2): IsaInfo(*caps)}, both
    ) == GFX1250V0
    # The other order rebuilds gfx942, which needs no alias, so the v0 name in the
    # list must not follow it there.
    assert buildTargetGfx(
        {IsaVersion(9, 4, 2): IsaInfo(*caps), ISA_GFX1250: IsaInfo(*caps)}, both
    ) == "gfx942"


def test_client_writer_receives_the_requested_names(monkeypatch, tmp_path):
    """The names travel as an argument rather than through globalParameters, so
    nothing else can clobber them; this pins the single hop between the entry point
    and the re-spawn."""
    import types

    from Tensile import Tensile as TensileModule

    captured = {}
    monkeypatch.setattr(
        TensileModule.ClientWriter,
        "main",
        lambda *a, **kw: captured.update(kw),
    )

    TensileModule.executeStepsInConfig(
        {"LibraryClient": None},
        tmp_path,
        types.SimpleNamespace(assembler="assembler"),
        types.SimpleNamespace(compiler="compiler"),
        _stub_iim(),
        "cc",
        types.SimpleNamespace(),
        0,
        {},
        archNames=[GFX1250V0],
    )

    assert captured["archNames"] == [GFX1250V0]


# =========================================================================== #
# Logic-file architecture names. v0 has no tuned logic of its own: it reuses
# gfx1250's and re-derives every solution under v0 capabilities, which is what
# makes the ASIC revision free of solution parameters. A logic file that names the
# ASIC revision instead is therefore a mistake, and one that would otherwise cost a
# whole build to notice -- masterLibraries is keyed by the declared name while
# the per-architecture writes are keyed by the ISA-derived one.
# =========================================================================== #
SCHEDULE_NAME = "Aldebaran_Cijk_Ailk_Bljk_SB"


@pytest.fixture
def _restore_type_mismatch_collector():
    """``generateLogicDataAndSolutions`` resets Solution.py's module-level type
    mismatch collector and replaces it with its own aggregate. Sibling suites reset
    it in setup rather than teardown, so today nothing breaks -- restore it anyway
    rather than depend on that."""
    from Tensile.SolutionStructs.Solution import (
        getTypeMismatchCollector,
        mergeTypeMismatchCollector,
        resetTypeMismatchCollector,
    )

    saved = copy.deepcopy(getTypeMismatchCollector())
    yield
    resetTypeMismatchCollector()
    mergeTypeMismatchCollector(saved)


def _generateLogicData(monkeypatch, *architectureNames):
    """Runs the real merge loop over one synthetic parsed logic file per name.

    Built as the real ``LibraryIO.LibraryLogic`` rather than a bare tuple, so the
    stub cannot drift from the parser's contract and a refactor from positional to
    field access does not look like a production failure.
    """
    from unittest.mock import MagicMock

    import Tensile.LibraryIO as LibraryIO
    import Tensile.TensileCreateLibrary.Run as RunModule

    libraries = {}
    parsed = []
    for name in architectureNames:
        libraries[name] = MagicMock(solutions={}, lazyLibraries={})
        parsed.append(
            LibraryIO.LibraryLogic(
                schedule=SCHEDULE_NAME,
                architecture=name,
                problemType=MagicMock(),
                solutions=[],
                exactLogic=None,
                library=libraries[name],
                typeMismatches={},
            )
        )
    monkeypatch.setattr(RunModule, "ParallelMap2", lambda _fn, _iter, *a, **kw: parsed)
    args = {
        "Architecture": GFX1250,
        "CodeObjectVersion": "4",
        "LazyLibraryLoading": True,
        "GenSolTable": False,
    }
    result = RunModule.generateLogicDataAndSolutions(
        ["fake.yaml"] * len(parsed), args, MagicMock(), _stub_iim()
    )
    return result, libraries


@pytest.mark.parametrize("asic_revision, architecture", sorted(ARCH_COMPILER_TARGET.items()))
def test_logic_file_naming_the_asic_revision_is_rejected(
    monkeypatch, _restore_type_mismatch_collector, asic_revision, architecture
):
    """Silently dropping it is the failure mode to avoid: the key would not match
    the ISA-derived name the writes are gated on, so the build would report
    success and ship an empty library.

    Driven from the alias table so a second ASIC revision is covered the day it is
    added, not the day someone remembers this test.
    """
    with pytest.raises(ValueError) as excinfo:
        _generateLogicData(monkeypatch, asic_revision)

    message = str(excinfo.value)
    assert asic_revision in message
    # Quoted, because the architecture name is a *substring* of the ASIC revision name:
    # a bare `architecture in message` is satisfied by the ASIC revision name alone and
    # pins nothing, which is exactly the requirement this test exists for.
    assert f"'{architecture}'" in message
    # Without this a user is told a name is wrong but not which of hundreds of
    # logic files says it.
    assert SCHEDULE_NAME in message


def test_logic_file_naming_the_architecture_is_accepted(
    monkeypatch, _restore_type_mismatch_collector
):
    """The control: gfx1250 logic is what a v0 build is *meant* to consume, so the
    guard must not narrow the fallback the whole ASIC revision design depends on."""
    (_, masterLibraries, _), _ = _generateLogicData(monkeypatch, GFX1250)

    assert list(masterLibraries) == [GFX1250]


def test_fallback_logic_is_still_merged_and_popped(
    monkeypatch, _restore_type_mismatch_collector
):
    """``fallback`` is the one architecture name deliberately designed not to name
    an architecture, so it is what an over-broad guard breaks first.

    Paired with a real architecture on purpose: alone, the fallback handling is a
    merge over an empty loop followed by a pop, so an empty result cannot tell
    "handled" apart from "silently discarded".
    """
    (_, masterLibraries, _), libraries = _generateLogicData(
        monkeypatch, GFX1250, "fallback"
    )

    assert list(masterLibraries) == [GFX1250]
    libraries[GFX1250].merge.assert_called_once_with(libraries["fallback"])
