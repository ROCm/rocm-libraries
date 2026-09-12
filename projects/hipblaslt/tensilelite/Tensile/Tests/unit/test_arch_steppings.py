# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx1250 stepping split via a distinct architecture name.

gfx1250 ships in two steppings that report the same ISA but are separate compiler
targets. v0 is modelled as the architecture name ``gfx1250-strict``; v1 keeps the
plain ``gfx1250`` name. Both canonicalize to ``IsaVersion(12,5,0)``, so the
stepping is invisible below the build's capability map and has to be carried by
name to reach the assembler as ``-mcpu=gfx1250-strict``. Getting that wrong is not
cosmetic: the two targets emit different ELF machine codes (0xEB and 0x49) and
their code objects will not load on each other's silicon.

Because the two steppings are indistinguishable by ISA, the assembler-probed
capability table cannot tell them apart. The v0 deltas are therefore *declared*
in ``ARCH_CAP_OVERRIDES`` and applied on top of the probed caps, which turns the
two silicon differences v0 has to express -- no TDM-multicast, no fp4 32x16 WMMA
-- into ordinary capability reads in Solution derivation.

There is deliberately no solution parameter and no kernel-name difference: one
build targets exactly one stepping, so same-named kernels never coexist.

The tests that derive real solutions need gfx1250 capabilities (``amdclang++``
targeting gfx1250) and are skipped when the toolchain is unavailable.
"""

import copy
import inspect
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from Tensile.Common.Architectures import (
    _REPORTED_ARCH_RE,
    SUPPORTED_GFX,
    SUPPORTED_ISA,
    archMacroNames,
    archNameForIsa,
    architectureMap,
    baseArchName,
    expandAllArchitectures,
    gfxToIsa,
    isaToGfx,
    steppingArchOf,
)
from Tensile import GpuArch
from Tensile.CustomYamlLoader import archMatch
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
GFX1250_STRICT = "gfx1250-strict"
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
def test_gfx1250_strict_is_registered_as_an_architecture():
    """``--architecture gfx1250-strict`` must survive the build driver's allow-list."""
    assert GFX1250_STRICT in architectureMap


def test_both_steppings_canonicalize_to_the_gfx1250_compiler_target():
    """The whole design rests on this asymmetry: the arch name carries the
    stepping, the ISA tuple does not, so anything derived from the tuple can only
    ever say gfx1250. That is why the requested name, not the ISA, is what reaches
    the compiler target and the output subtree."""
    assert gfxToIsa(GFX1250_STRICT) == ISA_GFX1250
    assert gfxToIsa(GFX1250) == ISA_GFX1250
    assert isaToGfx(gfxToIsa(GFX1250_STRICT)) == GFX1250


def test_gfx1250_strict_is_absent_from_the_isa_derived_names():
    """``SUPPORTED_GFX`` -- what ``all`` expands to -- is derived from ISA tuples,
    so it can name only one architecture per ISA, and for (12,5,0) that one is
    gfx1250. The stepping is a bring-up target and must be requested by name."""
    assert GFX1250_STRICT not in SUPPORTED_GFX
    assert GFX1250 in SUPPORTED_GFX


def test_generated_source_is_guarded_on_both_steppings_macros():
    """Generated helper-kernel source is guarded on the macro clang predefines,
    which is named after the compiler target, not the ISA. Guarding on the
    ISA-derived ``__gfx1250__`` alone drops the code from every strict build."""
    assert archMacroNames(ISA_GFX1250) == ["__gfx1250__", "__gfx1250_strict__"]


@pytest.mark.parametrize("isa", [isa for isa in SUPPORTED_ISA if isa != ISA_GFX1250])
def test_every_other_isa_still_names_exactly_one_macro(isa):
    """gfx1250 is the only ISA two architectures share, so the guard other
    architectures get is unchanged -- including gfx942 and gfx950, whose xnack
    spellings must collapse to one macro rather than repeat it."""
    assert archMacroNames(isa) == ["__" + isaToGfx(isa) + "__"]


def test_an_isa_naming_no_architecture_yields_no_macro():
    """``all`` has no ISA, and matching it would emit a macro no compilation
    defines -- a guard that silently never fires."""
    assert archMacroNames(None) == []


def test_all_keeps_architectures_its_expansion_does_not_cover():
    """``all`` cannot name v0, so an explicit ``gfx1250-strict`` alongside it has to
    survive rather than be dropped -- being dropped would turn a request for an
    stepping that cannot be built together with v1 into a silent v1-only build.
    Kept, it reaches the mixed-build guard and reports the conflict."""
    expanded = expandAllArchitectures(["all", GFX1250_STRICT])
    assert GFX1250_STRICT in expanded
    assert set(SUPPORTED_GFX).issubset(expanded)


def test_all_expansion_does_not_duplicate_covered_architectures():
    assert expandAllArchitectures(["all", "gfx942"]) == SUPPORTED_GFX


def test_expansion_is_a_passthrough_without_the_all_keyword():
    assert expandAllArchitectures([GFX1250_STRICT, "gfx942"]) == [GFX1250_STRICT, "gfx942"]


@pytest.mark.parametrize("spec", ["gfx950[cu=64]", "gfx942:xnack+", "gfx942[id=74a0]"])
def test_all_absorbs_qualified_specs_of_architectures_it_covers(spec):
    """Only names ``all`` genuinely cannot express may survive it. A predicate or
    xnack spec names an architecture the expansion already covers, so keeping it
    would both change behavior for architectures unrelated to the stepping split
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
    """``GPU_TARGETS=gfx1250-strict;`` arrives as a trailing empty spec too, and there is
    no ``all`` to absorb it: the list is returned verbatim, so the empty entry
    reaches the predicate splitter and fails the build on a request that is
    otherwise valid."""
    assert expandAllArchitectures([GFX1250_STRICT, padding]) == [GFX1250_STRICT]


@pytest.mark.parametrize("keyword", [" all", "all ", "\tall"])
def test_all_is_recognized_despite_surrounding_whitespace(keyword):
    """Membership was tested on the raw entry while the filter compared the
    stripped one, so a padded keyword skipped expansion and was handed to the
    predicate splitter as an architecture named ``all``."""
    assert expandAllArchitectures([keyword, GFX1250_STRICT]) == SUPPORTED_GFX + [GFX1250_STRICT]


# =========================================================================== #
# The capability-override mechanism itself. No toolchain needed: the override
# step only rewrites dict entries, so a synthetic capability map exercises it.
# =========================================================================== #
def _synthetic_iim():
    """An ISA map with the two v0-sensitive caps absent, as the probe leaves them."""
    return {ISA_GFX1250: IsaInfo({"SupportedISA": True}, {}, {}, {})}


def test_overrides_turn_off_both_strict_capabilities():
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [GFX1250_STRICT])
    assert iim[ISA_GFX1250].archCaps[CAP_MULTICAST] is False
    assert iim[ISA_GFX1250].asmCaps[CAP_FP4_32X16] is False


def test_gfx1250_leaves_both_capabilities_at_their_default():
    """v1 declares no overrides, so the caps stay absent and consumers'
    ``get(..., True)`` default keeps today's behavior."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [GFX1250])
    assert CAP_MULTICAST not in iim[ISA_GFX1250].archCaps
    assert CAP_FP4_32X16 not in iim[ISA_GFX1250].asmCaps


def test_unknown_arch_name_is_ignored_by_the_override_step():
    """Names without declared deltas must pass through untouched rather than
    raising, since every build passes its full requested-arch list."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, ["gfx942", "gfx950"])
    assert iim[ISA_GFX1250].asmCaps == {"SupportedISA": True}


def test_mixed_stepping_build_is_rejected():
    """Both names key the same IsaVersion, so a single capability map cannot
    describe both; combined with identical kernel names a mixed build would
    silently emit one stepping's kernels under the other's caps."""
    # Matched on the conflict wording, not just the name: the same function also
    # raises for a requested stepping absent from the map, and either message
    # would otherwise satisfy this test.
    with pytest.raises(ValueError, match="share ISA"):
        applyArchCapOverrides(_synthetic_iim(), [GFX1250, GFX1250_STRICT])


@pytest.mark.parametrize("spec", ["gfx1250-strict[cu=64]", "gfx1250-strict[id=1250]"])
def test_predicated_stepping_still_gets_its_overrides(spec):
    """``--gpu-targets`` accepts a predicate on any architecture and forwards the
    spec verbatim, so the lookup has to see past it. Missing here is the worst
    case the split can produce: the build is accepted, reports v0, and derives
    every solution under the shipping stepping's capabilities."""
    iim = _synthetic_iim()
    applyArchCapOverrides(iim, [spec])
    assert iim[ISA_GFX1250].archCaps[CAP_MULTICAST] is False
    assert iim[ISA_GFX1250].asmCaps[CAP_FP4_32X16] is False


def test_predicated_stepping_still_conflicts_with_the_other_stepping():
    """The mixed-build guard compares declared deltas, so a predicate that hides
    the deltas also hides the conflict."""
    with pytest.raises(ValueError, match="share ISA"):
        applyArchCapOverrides(_synthetic_iim(), [GFX1250, "gfx1250-strict[cu=64]"])


def test_repeating_one_stepping_is_not_a_conflict():
    """Only *differing* capabilities conflict. A name repeated by the caller, and
    two qualified variants of one architecture (which share an ISA and declare no
    deltas), must both still build."""
    applyArchCapOverrides(_synthetic_iim(), [GFX1250_STRICT, GFX1250_STRICT])
    applyArchCapOverrides(_synthetic_iim(), ["gfx942:xnack+", "gfx942:xnack-"])


def test_requested_stepping_missing_from_the_capability_map_is_an_error():
    """Silently skipping would hand v0 the shipping stepping's capabilities --
    the one outcome the override step exists to prevent -- so a map that cannot
    carry the declared deltas has to fail loudly."""
    with pytest.raises(ValueError, match=GFX1250_STRICT):
        applyArchCapOverrides({}, [GFX1250_STRICT])


# =========================================================================== #
# Shared toolchain harness (real gfx1250 caps + assembler). The base solution is
# the MXFP4 (F4/F4/S) TN config from ``mxf4_gfx1250.yaml``; each test flips only
# MatrixInstruction / ClusterDim, and selects an stepping by capability map.
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
def gfx1250_strict_iim(gfx1250_iim):
    """v0: the same two steps an entry point runs for ``--architecture gfx1250-strict``.

    Routed through the production override function rather than hand-patching
    keys, so the fixture cannot drift from what a real v0 build sees.
    """
    iim = copy.deepcopy(gfx1250_iim)
    applyArchCapOverrides(iim, [GFX1250_STRICT])
    return iim


def test_probe_leaves_both_capabilities_absent_for_the_override_to_set(gfx1250_iim):
    """The premise the ``True`` defaults in Solution rest on: the real assembler
    probe never reports these keys, so v1 is byte-identical to before the split
    and v0 is the only architecture that changes behavior."""
    assert CAP_MULTICAST not in gfx1250_iim[ISA_GFX1250].archCaps
    assert CAP_FP4_32X16 not in gfx1250_iim[ISA_GFX1250].asmCaps


def test_overrides_apply_on_top_of_really_probed_capabilities(gfx1250_strict_iim):
    """Probe-then-override composed on a real capability map, not a synthetic one."""
    info = gfx1250_strict_iim[ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False
    # Probed entries survive the override, which only adds the declared keys.
    assert info.asmCaps["SupportedISA"]
    assert "HasWMMA" in info.asmCaps


def test_xcnt_is_a_really_probed_archcap_strict_inherits(gfx1250_iim):
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


def test_fp4_32x16_rejected_on_gfx1250_strict(_gp_gfx1250, gfx1250_strict_iim, assembler, capsys):
    """v0 lacks the fp4 32x16 WMMA opcode.

    With SourceSwap=True the effective MatrixInstM is 16, so a guard on
    MatrixInstM==32 would NOT fire here -- this only passes because the guard
    uses the physical MIBlock[0]==32.
    """
    sol, out = _derive(gfx1250_strict_iim, assembler, capsys, FP4_32X16_MI)
    assert sol.get("Valid") is False
    assert FP4_32X16_REASON in out


# gfx1250-strict's 16x16 fp4 acceptance (the restriction is specific to 32x16) is covered by
# test_multicast_forced_off_on_gfx1250_strict below, which derives a valid 16x16 v0
# solution -- so no separate 16x16-accept test is needed.


def test_fp4_32x16_rejected_on_gfx1250_strict_without_source_swap(
    _gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys
):
    """The other half of the SourceSwap space, and a real shipped shape
    (streamk/gfx1250/sk_mxf4gemm_tdm_ext.yaml is 32x16 fp4 with SourceSwap false).

    Here the effective dims equal the physical ones, so this passes under either
    gate -- together with the SourceSwap=True case above it pins that the gate
    covers both, which only the physical MIBlock does. The v1 leg keeps the
    rejection attributable to the stepping rather than to the config.
    """
    v1, _ = _derive(gfx1250_iim, assembler, capsys, FP4_32X16_MI, SourceSwap=False)
    assert v1.get("Valid") is True, "SourceSwap=False 32x16 fp4 must derive on v1"

    sol, out = _derive(
        gfx1250_strict_iim, assembler, capsys, FP4_32X16_MI, SourceSwap=False
    )
    assert sol["MIBlock"][0] == 32 and sol["MIBlock"][1] == 16
    assert sol.get("Valid") is False
    assert FP4_32X16_REASON in out


MIXED_MAC = {"DataType": "F8", "MacDataTypeA": "F8", "MacDataTypeB": "F4",
             "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"}

# The mixed-operand shape needs the vector widths the shipped config uses
# (sk_mxf8f4gemm_tdm.yaml: all three auto, DepthU 256, no SourceSwap). With the
# fp4 base's fixed widths it is rejected on *both* steppings -- F8 wants
# lrvwA == 16 while F4 wants lrvwB == 32, and one LocalReadVectorWidth cannot be
# both -- which would make the reject below prove nothing about the stepping.
MIXED_MAC_PARAMS = dict(
    GlobalReadVectorWidthA=-1,
    GlobalReadVectorWidthB=-1,
    LocalReadVectorWidth=-1,
    DepthU=256,
    SourceSwap=False,
    TransposeLDS=-1,
)


def test_mixed_mac_type_fp4_is_covered_by_the_same_gate(
    _gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys
):
    """The opcode is selected by the WMMA operand format (MacDataType), so a gate
    keyed on the memory format (DataTypeA/B) would look like a hole for configs
    such as sk_mxf8f4gemm_tdm.yaml (``MacDataTypeA: F8, MacDataTypeB: F4``).

    It is not one: ``DataTypeB`` is *defaulted from* ``MacDataTypeB`` whenever the
    config does not set it explicitly (Problem.py), and ``getRealDataTypeB`` only
    remaps the F8/BF8 pair -- never fp4. So MacDataType fp4 always implies
    DataType fp4 and the two keys cannot diverge here. This pins that equivalence
    so the gate is not "fixed" into keying on something it need not.

    The v1 leg is what makes the gfx1250-strict reject meaningful: it shows the config is
    derivable, so the rejection comes from the stepping and not from the config.
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
        gfx1250_strict_iim, assembler, capsys, FP4_32X16_MI,
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


def test_multicast_forced_off_on_gfx1250_strict(_gp_gfx1250, gfx1250_strict_iim, assembler, capsys):
    """v0 has no TDM-multicast, but clustering itself stays valid."""
    sol, out = _derive(gfx1250_strict_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["Multicast"] is False
    # ClusterBarrier is a separate feature v0 supports; it must stay enabled.
    assert sol["ClusterBarrier"] is True


# =========================================================================== #
# Naming invariant. The steppings are separate builds, so their kernels must NOT
# be named apart -- an stepping token in the name would desynchronize the shipped
# library logic (which stores KernelNameMin at tuning time) from the emitted
# symbol. This fails loudly if anyone reintroduces one.
# =========================================================================== #
def test_kernel_names_identical_across_steppings(
    _gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys
):
    v1, _ = _derive(gfx1250_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
    v0, _ = _derive(gfx1250_strict_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
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
        assert kernel["Multicast"] is (archName != GFX1250_STRICT)

        kwa = KernelWriterAssembly(asm, DebugConfig())
        ri = _init_rocisa_for(kernel)
        _prepare_kernel(kernel, False)
        res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        return canonicalize_asm(res.src), res.err


def test_gfx1250_emits_multicast_gfx1250_strict_does_not(gfx1250_cxx):
    v1_src, v1_err = _emit(GFX1250)
    v0_src, v0_err = _emit(GFX1250_STRICT)
    assert v1_err == 0 and v0_err == 0

    assert any(m in v1_src for m in MULTICAST_MARKERS), (
        "gfx1250 with ClusterDim!=[1,1] should emit multicast mask setup"
    )
    assert not any(m in v0_src for m in MULTICAST_MARKERS), (
        "gfx1250-strict must not emit any multicast instructions"
    )
    assert v1_src != v0_src


# =========================================================================== #
# XNACK-replay drain (``s_wait_xcnt``). Unlike multicast, this is a codegen-time
# arch capability, not a solution parameter, so it does not travel through
# ``applyArchCapOverrides``/``isaInfoMap``. The kernel writer reads
# ``RequiresXCntForVolatileVMEM`` straight from the rocisa singleton
# (``ti.getArchCaps()``), which is keyed by ISA (12,5,0) and cannot tell
# gfx1250's steppings apart. v0 shares the same XNACK-replay hazard as v1, so it
# keeps the drain: ``RequiresXCntForVolatileVMEM`` is intentionally NOT in gfx1250-strict's
# ARCH_CAP_OVERRIDES. The test derives on the gfx1250 (gfx1250) capability map and
# emits the same kernel under both stepping signals to pin that v0 no longer
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


def _emit_streamk_srcs(stepping):
    """Emit the gfx1250 StreamK config's kernels, optionally under a v0 stepping.

    Mirrors ``config_harness.emit_kernels_from_config`` but sets
    ``globalParameters["StinkyTofuArchName"]`` inside the isolated globals so the
    codegen seam sees the stepping (the harness never sets it). Returns the list
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
        if stepping:
            globalParameters["StinkyTofuArchName"] = stepping
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


def test_gfx1250_strict_streamk_keeps_the_xcnt_drain(gfx1250_cxx):
    """The XNACK-replay drain is gated on ``RequiresXCntForVolatileVMEM``. v0
    shares gfx1250's hazard and does NOT override that cap, so it must keep the
    drain. Emitting the *same* StreamK kernel under v1 vs v0 and counting the
    drain marker isolates exactly the drain: v1 must keep it (the ``> 0`` check
    guards against a vacuous pass) and v0 must keep every one too."""
    v1_drains = sum(s.count(_XCNT_DRAIN_MARKER) for s in _emit_streamk_srcs(stepping=None))
    v0_drains = sum(s.count(_XCNT_DRAIN_MARKER) for s in _emit_streamk_srcs(stepping=GFX1250_STRICT))
    assert v1_drains > 0, (
        "gfx1250 (gfx1250) StreamK should emit the XNACK-replay drain; without it "
        "the gfx1250-strict assertion below would pass vacuously"
    )
    assert v0_drains == v1_drains, (
        f"gfx1250 v0 must keep every RequiresXCntForVolatileVMEM drain "
        f"(v1 drains={v1_drains}, v0 drains={v0_drains})"
    )


# =========================================================================== #
# Round-trip. The stepping lives in the build's capability map, not in the
# solution, so a re-parsed solution re-derives Multicast from whichever map the
# reading build uses. That is inherent to having no solution parameter; this
# pins it as a known property rather than leaving it a latent surprise.
# =========================================================================== #
def test_multicast_rederived_from_build_caps_after_yaml_roundtrip(
    _gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys, tmp_path
):
    from Tensile import LibraryIO
    from Tensile.SolutionStructs import ProblemSizes

    sol, out = _derive(gfx1250_strict_iim, assembler, capsys, MULTICAST_MI, ClusterDim=[2, 1])
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
        isaInfoMap=gfx1250_strict_iim,
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
# does: deleting the `applyArchCapOverrides` calls leaves the rest of this file
# (and the TensileCreateLibrary characterization suites) green, because a missing
# override simply leaves the capability map at gfx1250's values -- the exact
# silent failure the feature exists to prevent.
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


def test_tensile_entry_point_applies_the_strict_overrides(
    monkeypatch, tmp_path, restore_global_parameters
):
    """``Tensile --gpu-targets gfx1250-strict`` must reach the benchmark pipeline with
    v0 capabilities. This is the only channel the stepping travels through."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile(
        [config, str(tmp_path / "out"), "--gpu-targets", GFX1250_STRICT]
    )

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False


def test_tensile_entry_point_leaves_gfx1250_capabilities_untouched(
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
# place a config can state an stepping, since the ISA does not distinguish them,
# so it selects the stepping -- and nothing else, to keep those 120 unchanged.
# --------------------------------------------------------------------------- #
def test_config_architecture_selects_the_stepping(
    monkeypatch, tmp_path, restore_global_parameters
):
    """A config alone must be able to ask for v0, without the caller having to
    remember ``--gpu-targets``."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path, Architecture=GFX1250_STRICT, ISA=[[12, 5, 0]])

    TensileModule.Tensile([config, str(tmp_path / "out")])

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False
    assert captured["archNames"] == [GFX1250_STRICT]


def test_config_architecture_of_the_shipping_stepping_adds_nothing(
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
    config = _write_min_config(tmp_path, Architecture=GFX1250_STRICT, ISA=[[12, 5, 0]])

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


def test_config_architecture_naming_an_stepping_of_another_isa_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The one case where silently ignoring the name is not acceptable: it asks
    for an stepping, and ignoring it builds the shipping one instead."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(tmp_path, Architecture=GFX1250_STRICT, ISA=[[9, 4, 2]])

    with pytest.raises(ValueError) as excinfo:
        TensileModule.Tensile([config, str(tmp_path / "out")])

    assert GFX1250_STRICT in str(excinfo.value)


@pytest.mark.parametrize("arch", ["gfx1250v1", "gfx1250V0", "gfx1250v"])
def test_unrecognized_config_architecture_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters, arch
):
    """The same near-miss names ``--gpu-targets`` rejects: each resolves to
    (12,5,0) by the ISA regex alone, so without a name check the config would
    quietly build the shipping stepping."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(tmp_path, Architecture=arch, ISA=[[12, 5, 0]])

    with pytest.raises(ValueError) as excinfo:
        TensileModule.Tensile([config, str(tmp_path / "out")])

    assert arch in str(excinfo.value)


def test_mixed_steppings_in_the_config_architecture_are_rejected(
    monkeypatch, tmp_path, restore_global_parameters
):
    """One build is one stepping, whichever layer asked for both."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    config = _write_min_config(
        tmp_path, Architecture=f"{GFX1250};{GFX1250_STRICT}", ISA=[[12, 5, 0]]
    )

    with pytest.raises(ValueError):
        TensileModule.Tensile([config, str(tmp_path / "out")])


@pytest.mark.parametrize(
    "detected,expected",
    [
        (GFX1250_STRICT, {"gfx1250-strict_Cijk_A.yaml"}),
        (GFX1250, {"gfx1250_Cijk_A.yaml"}),
        ("gfx942", {"aquavanjaram_Cijk_A.yaml"}),
    ],
)
def test_the_summation_step_selects_exactly_one_steppings_logic_files(
    tmp_path, detected, expected
):
    """``gfx1250-strict`` is a literal prefix extension of ``gfx1250``, so the
    unanchored glob this step used to run matched both. On plain gfx1250 silicon
    that benchmarked the other stepping's solutions -- against code objects the
    agent cannot load -- and wrote the resulting model back into its logic files.
    Every logic filename is ``<codename>_...``, so the separator separates them."""
    import glob

    from Tensile.Common.Architectures import gfxToSwCodename

    for name in (
        "gfx1250_Cijk_A.yaml",
        "gfx1250-strict_Cijk_A.yaml",
        "aquavanjaram_Cijk_A.yaml",
    ):
        (tmp_path / name).touch()

    pattern = os.path.join(str(tmp_path), "{}_*".format(gfxToSwCodename(detected)))
    assert {os.path.basename(p) for p in glob.glob(pattern)} == expected


# =========================================================================== #
# Which tool is asked, and what is kept of its answer. `GpuArch` is the source
# the rest of Tensile names architectures from, so what it decides here is what
# ends up in file names, directory names and `-mcpu`.
# =========================================================================== #
def _fakeTool(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + "".join(f"echo {line}\n" for line in lines))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def rocmRoot(tmp_path, monkeypatch):
    """An empty ROCM_PATH, with nothing of interest left on PATH."""
    root = tmp_path / "rocm"
    root.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    return root


@pytest.mark.parametrize("layout", ["llvm/bin", "lib/llvm/bin"])
def test_amdgpu_arch_is_found_in_either_install_layout(rocmRoot, layout):
    """ROCm puts it under llvm/bin; TheRock and the Windows SDK under lib/llvm/bin."""
    _fakeTool(rocmRoot / layout / "amdgpu-arch", "gfx950")

    assert GpuArch.detect_gpu_archs() == ["gfx950"]


def test_the_install_is_searched_ahead_of_PATH(rocmRoot, tmp_path, monkeypatch):
    """Both amdgpu-arch layouts have to be tried before PATH is consulted at all.

    Resolving one layout the whole way to PATH before trying the next would let
    a stray amdgpu-arch beat the one inside the install ROCM_PATH names -- and
    the two layouts differ only by a "lib" prefix, so the miss is routine. A
    test pointing ROCM_PATH at a fixture would then be answered by the real
    machine, which is how this was found.
    """
    strayDir = tmp_path / "stray"
    _fakeTool(strayDir / "amdgpu-arch", "gfx942")
    monkeypatch.setenv("PATH", str(strayDir))

    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch", GFX1250_STRICT)

    assert GpuArch.detect_gpu_archs() == [GFX1250_STRICT]


def test_PATH_still_answers_when_the_install_has_neither_layout(rocmRoot, tmp_path, monkeypatch):
    strayDir = tmp_path / "stray"
    _fakeTool(strayDir / "amdgpu-arch", "gfx942")
    monkeypatch.setenv("PATH", str(strayDir))

    assert GpuArch.detect_gpu_archs() == ["gfx942"]


def test_the_stepping_suffix_survives_the_probe(rocmRoot):
    """gfx1250 and gfx1250-strict reject each other's code objects, so a name
    truncated to its hex part is not a smaller answer, it is a wrong one."""
    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch", GFX1250_STRICT)

    assert GpuArch.detect_gpu_archs() == [GFX1250_STRICT]
    assert GpuArch.detect_gpu_arch() == GFX1250_STRICT


def test_every_device_is_reported_in_enumeration_order(rocmRoot):
    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch", "gfx950", "gfx942")

    assert GpuArch.detect_gpu_archs() == ["gfx950", "gfx942"]
    assert GpuArch.detect_gpu_arch() == "gfx950"


def test_repeated_devices_are_each_reported(rocmRoot):
    """One entry per agent, not per distinct architecture.

    Callers index this positionally to answer "what is device N", so collapsing
    a homogeneous four-GPU box to one entry makes every device but 0 undetectable.
    """
    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch", "gfx950", "gfx950", "gfx950")

    assert GpuArch.detect_gpu_archs() == ["gfx950", "gfx950", "gfx950"]


def test_a_list_valued_ROCM_PATH_is_searched_entry_by_entry(rocmRoot, tmp_path, monkeypatch):
    """ROCM_PATH holds an os.pathsep-separated list often enough that
    validateToolchain splits it -- and a box with two ROCm SDKs installed is
    both when it does and when picking the wrong one picks the wrong stepping.
    Reading it as a single directory name matches nothing and falls to PATH.
    """
    strayDir = tmp_path / "stray"
    _fakeTool(strayDir / "amdgpu-arch", "gfx942")
    monkeypatch.setenv("PATH", str(strayDir))

    second = tmp_path / "rocm-strict"
    _fakeTool(second / "lib/llvm/bin/amdgpu-arch", GFX1250_STRICT)
    monkeypatch.setenv("ROCM_PATH", os.pathsep.join([str(rocmRoot), str(second)]))

    assert GpuArch.detect_gpu_archs() == [GFX1250_STRICT]


def test_a_GPU_less_machine_reports_nothing(rocmRoot):
    """gfx000 is the placeholder such a host enumerates, not an architecture."""
    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch", GpuArch._PLACEHOLDER_ARCH)

    assert GpuArch.detect_gpu_archs() == []
    assert GpuArch.detect_gpu_arch() is None


def test_rocminfo_answers_when_amdgpu_arch_cannot(rocmRoot):
    """amdgpu-arch needs a device it can open, so it is not always the one that
    answers; rocminfo names the agent in a line this parses."""
    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch")  # runs, prints nothing
    _fakeTool(
        rocmRoot / "bin/rocminfo",
        "'  Name:                    {}'".format(GFX1250_STRICT),
    )

    assert GpuArch.detect_gpu_archs() == [GFX1250_STRICT]


def test_no_ROCm_at_all_is_reported_differently_from_no_GPU(rocmRoot, capsys):
    """Both answer None, and the two need different things done about them."""
    assert GpuArch.detect_gpu_arch() is None
    assert "Please install ROCm" in capsys.readouterr().err

    _fakeTool(rocmRoot / "lib/llvm/bin/amdgpu-arch")

    assert GpuArch.detect_gpu_arch() is None
    assert "Failed to detect" in capsys.readouterr().err


def _detectionReturning(monkeypatch, stdout, returncode=0):
    """Detection with the enumerator fallback wired to ``stdout``.

    ``detect_gpu_archs`` is silenced because it is consulted first: left alone it
    answers from the real device on any machine with a GPU, and the fixture would
    never be reached.

    The pin variables are cleared for the same reason: they reorder the sources,
    so a machine that happens to set one would otherwise change what these tests
    are measuring. Tests about the pins set them back.
    """
    import Tensile.Common.Architectures as Arch

    class _Proc:
        pass

    _Proc.returncode = returncode
    _Proc.stdout = stdout
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.delenv("ROCM_TARGET_LST", raising=False)
    monkeypatch.setattr(Arch, "detect_gpu_archs", lambda: [])
    monkeypatch.setattr(Arch, "run", lambda *a, **k: _Proc())
    return Arch


def test_the_stepping_survives_detection(monkeypatch):
    """The name used to be rebuilt from its ISA, and the two steppings share
    (12,5,0), so the rebuild reported gfx1250 for a gfx1250-strict agent. They
    reject each other's code objects, so that answer was silently wrong. What
    the tool said is now what detection reports.
    """
    Arch = _detectionReturning(monkeypatch, GFX1250_STRICT.encode() + b"\n")

    assert Arch.detectGlobalCurrentArch(0, "/my/enum") == GFX1250_STRICT
    # The ISA is still derivable from it, so the older callers are unaffected.
    assert tuple(Arch.detectGlobalCurrentISA(0, "/my/enum")) == ISA_GFX1250


def test_the_enumerator_is_asked_before_the_hardware(monkeypatch):
    """It is the only source that reads a target.lst or HSA_OVERRIDE_GFX_VERSION
    pin -- amdgpu-arch and rocminfo answer from the hardware. That is how a
    sandboxed image or a near-miss card is built for, so asking the hardware
    first would discard the pin without a word. It is also the tool
    --rocm-agent-enumerator names, which would otherwise do nothing.
    """
    Arch = _detectionReturning(monkeypatch, b"gfx90a\n")
    monkeypatch.setattr(Arch, "detect_gpu_archs", lambda: ["gfx942"])

    assert Arch.detectGlobalCurrentArch(0, "/my/enum") == "gfx90a"
    assert Arch.detectHostGfxArchs() == ["gfx90a"]


def test_the_hardware_answers_when_the_enumerator_cannot(monkeypatch):
    """The enumerator needs the render group and a device it can open, so it is
    not always the one that answers. Such a host used to fail detection outright.
    """
    Arch = _detectionReturning(monkeypatch, b"")
    monkeypatch.setattr(Arch, "detect_gpu_archs", lambda: ["gfx942"])

    assert Arch.detectGlobalCurrentArch(0, "/my/enum") == "gfx942"


def test_every_device_stays_addressable_on_a_homogeneous_box(monkeypatch):
    """Detection answers "what is device N" positionally, so the list it indexes
    has to keep one entry per agent. De-duplicating it first collapses a
    four-GPU box to a single entry, and `-d 1` upwards then fails to detect at
    all -- which is the ordinary way to tune on one GPU of a multi-GPU node.
    """
    Arch = _detectionReturning(monkeypatch, (GFX1250_STRICT + "\n").encode() * 4)

    for deviceId in range(4):
        assert Arch.detectGlobalCurrentArch(deviceId, "/my/enum") == GFX1250_STRICT

    # The set-shaped question still gets a set-shaped answer.
    assert Arch.detectHostGfxArchs() == [GFX1250_STRICT]


def test_a_mixed_box_names_each_device_separately(monkeypatch):
    """The two steppings cannot be told apart by ISA, so an index that slipped
    would hand one device the other's compiler target with nothing to catch it."""
    Arch = _detectionReturning(
        monkeypatch, "gfx950\n{}\ngfx1250\n".format(GFX1250_STRICT).encode()
    )

    assert Arch.detectGlobalCurrentArch(0, "/my/enum") == "gfx950"
    assert Arch.detectGlobalCurrentArch(1, "/my/enum") == GFX1250_STRICT
    assert Arch.detectGlobalCurrentArch(2, "/my/enum") == "gfx1250"


def test_a_labelled_enumerator_line_still_yields_its_architecture(monkeypatch):
    """hipinfo, the enumerator on Windows, prints properties as `key: value`.
    Splitting such a line at its colon to strip target features would keep the
    label and drop the name, leaving Windows with no detection at all."""
    Arch = _detectionReturning(monkeypatch, b"gcnArchName:                gfx1100\n")

    assert Arch.detectGlobalCurrentArch(0, "hipinfo") == "gfx1100"


def test_target_features_are_still_stripped_from_a_reported_name(monkeypatch):
    """The reason the colon was being split on in the first place."""
    Arch = _detectionReturning(monkeypatch, b"gfx942:sramecc+:xnack-\ngfx950[cu=64]\n")

    assert Arch.detectGlobalCurrentArch(0, "/my/enum") == "gfx942"
    assert Arch.detectGlobalCurrentArch(1, "/my/enum") == "gfx950"


def test_detection_falls_back_to_the_enumerator_when_nothing_else_answers(monkeypatch):
    """amdgpu-arch needs a device it can open and rocminfo needs the render
    group, so the enumerator is kept as the last source rather than dropped."""
    Arch = _detectionReturning(monkeypatch, b"gfx942\n")

    assert Arch.detectGlobalCurrentArch(0, "rocm_agent_enumerator") == "gfx942"


def test_detection_reports_the_architecture_the_enumerator_named(monkeypatch):
    """amdgpu-arch and rocminfo both print the full name; it is Tensile that used
    to throw it away by round-tripping through the ISA."""
    Arch = _detectionReturning(monkeypatch, b"gfx1250-strict\n")

    assert Arch.detectGlobalCurrentArch(0, "amdgpu-arch") == GFX1250_STRICT
    # The ISA is still derivable from it, so the older callers are unaffected.
    assert tuple(Arch.detectGlobalCurrentISA(0, "amdgpu-arch")) == ISA_GFX1250


def test_detection_refuses_a_name_that_only_looks_like_a_stepping(monkeypatch):
    """gfxToIsa runs a regex that stops at the first non-hex character, so
    gfx1250v1 resolves to (12,5,0) too. Accepting it on the strength of its ISA
    would build it as gfx1250 without a word; there is no such architecture, so
    failing loudly is the only honest answer."""
    Arch = _detectionReturning(monkeypatch, b"gfx1250v1\n")

    # Matched on the message: a bare `Exception` would also be satisfied by an
    # OSError out of `run`, which is a different failure entirely.
    with pytest.raises(Exception, match="Failed to detect current architecture"):
        Arch.detectGlobalCurrentArch(0, "amdgpu-arch")


def test_detection_failure_still_raises(monkeypatch):
    Arch = _detectionReturning(monkeypatch, b"", returncode=5)

    with pytest.raises(Exception, match="Failed to detect current architecture"):
        Arch.detectGlobalCurrentArch(0, "amdgpu-arch")


def test_auto_detect_tunes_for_the_architecture_it_found(
    monkeypatch, tmp_path, restore_global_parameters
):
    """With neither --gpu-targets nor ISA:, the enumerator is the only statement of
    what to build, and it names the architecture. Keeping only its ISA would tune
    gfx1250-strict silicon under gfx1250's capabilities and then build code
    objects that silicon rejects -- on the one path that runs on the very hardware
    it is tuning for."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    monkeypatch.setattr(
        TensileModule, "detectGlobalCurrentArch", lambda *a, **kw: GFX1250_STRICT
    )
    config = _write_min_config(tmp_path)

    TensileModule.Tensile([config, str(tmp_path / "out")])

    assert captured["archNames"] == [GFX1250_STRICT]
    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False


def test_auto_detect_on_the_base_architecture_is_unchanged(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The regression guard: detection reporting a name rather than an ISA must
    leave every architecture that does not share one deriving exactly what it
    derived before."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    monkeypatch.setattr(
        TensileModule, "detectGlobalCurrentArch", lambda *a, **kw: GFX1250
    )
    config = _write_min_config(tmp_path)

    TensileModule.Tensile([config, str(tmp_path / "out")])

    assert captured["archNames"] == [GFX1250]
    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert CAP_FP4_32X16 not in info.asmCaps


def test_auto_detect_still_rejects_a_misspelt_config_architecture(
    monkeypatch, tmp_path, restore_global_parameters
):
    """Auto-detect supplying the name must not cost the config its spellcheck.
    The key no longer decides which architecture is built here, but a typo in it
    is still a mistake worth reporting rather than silently dropping."""
    TensileModule = _stub_tensile_pipeline(monkeypatch, {})
    monkeypatch.setattr(
        TensileModule, "detectGlobalCurrentArch", lambda *a, **kw: GFX1250
    )
    config = _write_min_config(tmp_path, Architecture="gfx1250-stict")

    with pytest.raises(ValueError) as excinfo:
        TensileModule.Tensile([config, str(tmp_path / "out")])

    assert "gfx1250-stict" in str(excinfo.value)


def test_tensile_entry_point_records_the_requested_names(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The tuning flow re-spawns TensileCreateLibrary for the client library, and
    the ISA cannot say which stepping this build is for. The requested names have
    to reach the steps so that the re-spawn can ask for the right one."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile(
        [config, str(tmp_path / "out"), "--gpu-targets", GFX1250_STRICT]
    )

    assert captured["archNames"] == [GFX1250_STRICT]


# Deliberately malformed inputs, not spellings this project uses: a suffix that is
# not a registered name, the wrong case, and a truncated suffix. The only correct
# spellings are GFX1250 and GFX1250_STRICT.
@pytest.mark.parametrize("target", ["gfx1250v1", "gfx1250V0", "gfx1250v"])
def test_unknown_gpu_target_is_rejected(
    monkeypatch, tmp_path, restore_global_parameters, target
):
    """Every one of these resolves to ISA (12,5,0) through ``gfxToIsa`` (the regex
    stops at the first non-hex character), so an ISA check alone accepts them and
    silently builds v1. A typo in an stepping name must not produce the other
    stepping."""
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
    """Guarding against stepping typos must not narrow what a GPU target may be.
    The first two are the target-ID form ``rocm_agent_enumerator -v`` prints and a
    user copy-pastes from ``offload-arch``; the last two are the predicate form
    ``--architecture`` accepts. All four resolved to an ISA before the stepping
    split, so rejecting them would be a regression -- and would make the two
    flags disagree about what a GPU target is."""
    captured = {}
    TensileModule = _stub_tensile_pipeline(monkeypatch, captured)
    config = _write_min_config(tmp_path)

    TensileModule.Tensile([config, str(tmp_path / "out"), "--gpu-targets", target])

    assert captured["archNames"] == [target]


def _logicFileName(architectureName, scheduleName, tag=""):
    return f"{scheduleName}_{architectureName}{tag}.yaml"


def _run_createlibrary(monkeypatch, tmp_path, arch, logicFiles=()):
    """Drives ``TensileCreateLibrary.run()`` for one requested architecture with
    every expensive step stubbed. ``logicFiles`` writes minimal logic files
    (arch name, schedule name) into the logic dir; the real glob and filter run,
    so the selection observed is the production wiring's.
    """
    from unittest.mock import MagicMock

    import Tensile.TensileCreateLibrary.Run as RunModule

    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    for architectureName, scheduleName, *tag in logicFiles:
        # The filter reads only the third sequence item (CustomYamlLoader's
        # load_logic_gfx_arch), so a full logic file is not needed to exercise
        # it; the ScheduleName is written anyway to keep the file well formed.
        (logic_dir / _logicFileName(architectureName, scheduleName, *tag)).write_text(
            "- {MinimumRequiredVersion: 4.33.0}\n"
            f"- {scheduleName}\n"
            f"- {architectureName}\n"
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


def test_createlibrary_entry_point_applies_the_strict_overrides(
    monkeypatch, tmp_path, restore_global_parameters
):
    """``TensileCreateLibrary --architecture=gfx1250-strict`` must derive its solutions
    under v0 capabilities; this is the second of the two entry points, and it
    reaches the capability map by a different route than ``Tensile()``."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250_STRICT)

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert info.archCaps[CAP_MULTICAST] is False
    assert info.asmCaps[CAP_FP4_32X16] is False


def test_createlibrary_entry_point_forwards_the_requested_name(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The stepping name has to reach the kernel writers intact: it is what they
    hand ``--offload-arch``, and it is the only thing that decides the ELF machine
    code the code objects carry. Normalizing it to gfx1250 here would assemble the
    whole build for silicon that cannot load it."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250_STRICT)

    assert captured["cmdlineArchs"] == [GFX1250_STRICT]


def test_createlibrary_entry_point_leaves_gfx1250_capabilities_untouched(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The same route for a plain gfx1250 build must invent neither key, so this
    entry point is byte-identical to before the split as well."""
    captured = _run_createlibrary(monkeypatch, tmp_path, GFX1250)

    info = captured["isaInfoMap"][ISA_GFX1250]
    assert CAP_MULTICAST not in info.archCaps
    assert CAP_FP4_32X16 not in info.asmCaps
    assert captured["cmdlineArchs"] == [GFX1250]


# One logic tree holds both steppings' tuning, so every selection test below
# runs against a directory holding both: what has to be pinned is the partition,
# not that a lone file survives. Each entry is (ArchitectureName, ScheduleName),
# and each stepping declares its own name -- the two are separate architectures
# here, and sharing an ISA is not enough to make them share tuning.
_ARCH_LOGIC = (GFX1250, GFX1250)
_STRICT_LOGIC = (GFX1250_STRICT, GFX1250_STRICT)
_OTHER_ARCH_LOGIC = ("gfx942", "aquavanjaram")


def test_strict_build_selects_only_the_steppings_logic(
    monkeypatch, tmp_path, restore_global_parameters
):
    """Excluding the base architecture's own logic is the half that fails
    silently: those solutions were selected under capabilities the stepping does
    not have, and the build would report success having shipped them.

    It rests on ``archMatch`` comparing whole names -- a prefix match would let
    a request for ``gfx1250-strict`` claim ``gfx1250``'s logic as well.
    """
    captured = _run_createlibrary(
        monkeypatch, tmp_path, GFX1250_STRICT, logicFiles=[_ARCH_LOGIC, _STRICT_LOGIC]
    )

    assert captured["logicFiles"] == [_logicFileName(*_STRICT_LOGIC)]


def test_gfx1250_build_selects_only_the_architectures_logic(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The mirror, and the isolation gfx1250 needs: gfx1250-strict's logic sits
    in the same tree, so a plain gfx1250 build globs it up unless the declared
    architecture excludes it. It would otherwise ship strict-derived solutions
    in every gfx1250 library built after the stepping lands.
    """
    captured = _run_createlibrary(
        monkeypatch, tmp_path, GFX1250, logicFiles=[_ARCH_LOGIC, _STRICT_LOGIC]
    )

    assert captured["logicFiles"] == [_logicFileName(*_ARCH_LOGIC)]


def test_all_build_excludes_the_steppings_logic(
    monkeypatch, tmp_path, restore_global_parameters
):
    """``all`` is the default distribution build and reaches the selection as the
    architectures it expands to -- a list built from ISAs, which cannot name a
    stepping. The stepping's logic must therefore be excluded from it, and every
    other architecture's must survive.
    """
    captured = _run_createlibrary(
        monkeypatch,
        tmp_path,
        "all",
        logicFiles=[_ARCH_LOGIC, _STRICT_LOGIC, _OTHER_ARCH_LOGIC],
    )

    assert sorted(captured["logicFiles"]) == sorted(
        [_logicFileName(*_ARCH_LOGIC), _logicFileName(*_OTHER_ARCH_LOGIC)]
    )


def test_stepping_selection_spares_other_architectures(
    monkeypatch, tmp_path, restore_global_parameters
):
    """A multi-architecture build that includes the stepping must still consume
    every other architecture's logic. Tightening the name comparison to exclude
    the stepping's base is what could take these with it.
    """
    captured = _run_createlibrary(
        monkeypatch,
        tmp_path,
        f"gfx942_{GFX1250_STRICT}",
        logicFiles=[_ARCH_LOGIC, _STRICT_LOGIC, _OTHER_ARCH_LOGIC],
    )

    assert sorted(captured["logicFiles"]) == sorted(
        [_logicFileName(*_STRICT_LOGIC), _logicFileName(*_OTHER_ARCH_LOGIC)]
    )


def test_strict_build_ignores_an_unrelated_architectures_logic_files(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The control for the rule above: it must stay a match on the requested
    name, not degrade into accepting whatever logic is on disk."""
    captured = _run_createlibrary(
        monkeypatch, tmp_path, GFX1250_STRICT, logicFiles=[_OTHER_ARCH_LOGIC]
    )

    assert captured["logicFiles"] == []


# =========================================================================== #
# Tuning flow. `Tensile` re-spawns `TensileCreateLibrary` to build the client
# library from the logic it just tuned. That re-spawn is a fresh process whose
# only statement of what to build is `--architecture=`, so it is the one place
# the stepping can be lost after being correctly applied everywhere else.
# =========================================================================== #
def test_client_library_is_rebuilt_for_the_requested_stepping():
    assert archNameForIsa(ISA_GFX1250, [GFX1250_STRICT]) == GFX1250_STRICT


def test_client_library_target_falls_back_to_the_isa_derived_name():
    """The ISA and auto-detect entry paths never learn a name, so the ISA-derived
    target stays the default rather than the lookup being mandatory. An empty
    list, an explicit None, and an omitted argument all have to behave that way."""
    assert archNameForIsa(ISA_GFX1250, []) == GFX1250
    assert archNameForIsa(ISA_GFX1250, None) == GFX1250
    assert archNameForIsa(ISA_GFX1250) == GFX1250


def test_client_library_target_ignores_names_from_other_architectures():
    """A multi-architecture build must not label this ISA with an unrelated
    requested name; the lookup is keyed by ISA, not by position."""
    assert archNameForIsa(ISA_GFX1250, ["gfx942"]) == GFX1250


def test_client_library_target_ignores_qualifiers_of_other_architectures():
    """Only a stepping needs a name the ISA cannot express. Forwarding a requested
    qualifier verbatim would rebuild the client library for one xnack setting where
    an xnack-agnostic code object is wanted -- and since the library directory and
    the .co filter both still resolve, that surfaces when the device loads it, not
    at build time."""
    assert archNameForIsa(IsaVersion(9, 4, 2), ["gfx942:xnack+"]) == "gfx942"


def test_client_library_target_keeps_the_stepping_under_a_predicate():
    """The predicate is dropped -- the re-spawn resolves logic files itself, and an
    unqualified name is what every other architecture already gets here -- but the
    stepping must survive, or the tuning flow rebuilds the client library for the
    shipping stepping while reporting the stepping."""
    assert archNameForIsa(ISA_GFX1250, ["gfx1250-strict[cu=64]"]) == GFX1250_STRICT


def test_client_library_target_picks_the_name_matching_the_rebuilt_isa():
    """A multi-architecture build (``--gpu-targets 'gfx942;gfx1250-strict'`` is allowed,
    the ISAs differ) must resolve the name for the ISA actually being rebuilt.
    Taking the sole requested name, or the first one, would rebuild gfx1250-strict's
    client library against the shipping stepping the moment a second architecture is
    asked for -- silently, since the name it lands on is still a valid target."""
    both = ["gfx942", GFX1250_STRICT]

    assert archNameForIsa(ISA_GFX1250, both) == GFX1250_STRICT
    # The other ISA rebuilds gfx942, which needs no alias, so the gfx1250-strict
    # name in the same list must not follow it there.
    assert archNameForIsa(IsaVersion(9, 4, 2), both) == "gfx942"


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
        archNames=[GFX1250_STRICT],
    )

    assert captured["archNames"] == [GFX1250_STRICT]


# =========================================================================== #
# Logic-file architecture names. A logic file declares the architecture it was
# tuned for, and gfx1250-strict declares itself -- it is not gfx1250 tuning with
# a tag on it. That is what makes the two agree: masterLibraries is keyed by the
# declared name and the per-architecture writes are keyed by the requested name,
# so a file that declared the ISA-derived name instead would key a library no
# write ever addresses, and the build would ship an empty subtree in silence.
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


@pytest.mark.parametrize(
    "stepping, architecture",
    sorted((n, steppingArchOf(n)) for n in architectureMap if steppingArchOf(n)),
)
def test_logic_file_naming_the_stepping_keys_its_own_library(
    monkeypatch, _restore_type_mismatch_collector, stepping, architecture
):
    """A stepping is an architecture here, so its logic keys a library of its own
    rather than merging into its base's. Sharing a key is the failure to avoid:
    the two are tuned under different capabilities, and one write would land on
    whichever name the writes happen to be gated on.

    Driven from the alias table so a second stepping is covered the day it is
    added, not the day someone remembers this test.
    """
    (_, masterLibraries, _), _ = _generateLogicData(monkeypatch, stepping)

    assert list(masterLibraries) == [stepping]
    assert architecture not in masterLibraries


def test_logic_file_naming_the_architecture_is_accepted(
    monkeypatch, _restore_type_mismatch_collector
):
    """The control: the base architecture's own logic still keys its own library,
    unaffected by the stepping sharing its ISA."""
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


# =========================================================================== #
# The one place the ISA-derived name is still the right answer: the Processor
# predicate inside the master library. The runtime derives its own Processor by
# substring match on the agent name, so a gfx1250-strict agent reports
# Processor::gfx1250 -- there is no gfx1250-strict enumerator in AMDGPU.hpp, and
# the msgpack loader hard-fails on an enum name its table does not list. The two
# steppings are kept apart by their directories, not by this predicate.
# =========================================================================== #
def _hardwareRowFor(arch):
    from unittest.mock import MagicMock

    from Tensile.SolutionLibrary import MasterSolutionLibrary

    return MasterSolutionLibrary.hardware(
        {"ArchitectureName": arch, "CUCount": None},
        MagicMock(),
        "TensileLibrary_lazy",
        lazyLibrary=True,
    )


def test_a_steppings_predicate_names_the_architecture_it_shares_an_isa_with():
    """Writing "gfx1250-strict" here would make the .dat unloadable: the enum
    table in Serialization/Predicates.hpp has no such case, and the loader turns
    an unlisted name into a null library and a bare "Could not initialize Tensile
    library" with no mention of the cause."""
    lib, _ = _hardwareRowFor(GFX1250_STRICT)

    processor = lib.rows[0]["predicate"].value
    assert processor.tag == "Processor"
    assert processor.value == GFX1250

    # Identical to the base architecture's, which is what lets one runtime
    # Processor value serve both steppings.
    baseLib, _ = _hardwareRowFor(GFX1250)
    assert baseLib.rows[0]["predicate"].value.value == GFX1250


def test_a_steppings_placeholder_name_carries_the_stepping():
    """The sibling invariant, and the reason the two must not be unified: the
    shard filename does take the requested name, so TensileCreateLibrary's Mapping
    filter (``endswith("_" + archName)``) keeps the stepping's entries. Predicate
    and filename deliberately disagree."""
    _, placeholderName = _hardwareRowFor(GFX1250_STRICT)

    assert placeholderName == "TensileLibrary_lazy_" + GFX1250_STRICT


# =========================================================================== #
# Architecture identity. gfx1250 and gfx1250-strict share ISA 12.5.0, so the ISA
# names neither the compiler target, nor the code object, nor the output subtree.
# archNamesByIsa() carries the requested name to all three. It is the identity
# for every architecture that does not share its ISA, which is what keeps their
# output byte-identical.
# =========================================================================== #
def test_arch_names_by_isa_is_identity_for_ordinary_archs():
    """An ordinary build maps every ISA back to the name it was asked for, so
    threading the map through the writers cannot move or rename a single
    non-stepping artifact."""
    from Tensile.Common.Architectures import SUPPORTED_ISA, archNamesByIsa

    for isa in SUPPORTED_ISA:
        if isa == ISA_GFX1250:
            continue
        name = isaToGfx(isa)
        assert archNamesByIsa([name]) == {isa: name}


def test_arch_names_by_isa_names_the_stepping_not_its_base():
    """The whole point: 12.5.0 resolves to whichever of the two architectures was
    requested, so a strict build never falls back onto gfx1250's name."""
    from Tensile.Common.Architectures import archNamesByIsa

    assert archNamesByIsa([GFX1250_STRICT]) == {ISA_GFX1250: GFX1250_STRICT}
    assert archNamesByIsa([GFX1250]) == {ISA_GFX1250: GFX1250}


def test_arch_names_by_isa_drops_qualifiers_and_predicates():
    """Qualifiers name an architecture the ISA already describes; forwarding
    gfx942:xnack+ to --offload-arch would pin the code object to one xnack setting
    instead of leaving it xnack-agnostic. Predicates are resolved elsewhere."""
    from Tensile.Common.Architectures import archNamesByIsa

    assert archNamesByIsa(["gfx942:xnack+", "gfx942:xnack-"]) == {
        gfxToIsa("gfx942"): "gfx942"
    }
    assert archNamesByIsa([GFX1250_STRICT + "[cu=64]"]) == {
        ISA_GFX1250: GFX1250_STRICT
    }


def test_a_build_matching_no_logic_files_says_so(
    monkeypatch, tmp_path, capsys, restore_global_parameters
):
    """It otherwise exits 0 having written a subtree with no master and no
    Mapping, and the only trace is a zero in the log that looks like every other
    zero-match build. A strict build is the likely victim: it is the one whose
    logic lives in a tree of its own, so a path or filter mistake matches nothing
    at all rather than merely less than expected."""
    _run_createlibrary_to_writes(
        monkeypatch, tmp_path, GFX1250_STRICT, GFX1250_STRICT, "prefix_" + GFX1250_STRICT
    )

    warning = capsys.readouterr().out
    assert "No logic files matched" in warning, warning
    assert GFX1250_STRICT in warning, warning


def test_arch_names_by_isa_rejects_two_architectures_sharing_one_isa():
    """One key cannot name both. Keeping either silently would give the master and
    the mapping the requested names -- those loops iterate the names -- while the
    assembler target, the code object filename and its subtree all resolved
    through this one key, so one architecture would ship a library whose kernels
    were assembled for, and written under, the other."""
    from Tensile.Common.Architectures import archNamesByIsa

    with pytest.raises(ValueError, match="share ISA"):
        archNamesByIsa([GFX1250, GFX1250_STRICT])
    # A predicate must not be able to hide the conflict.
    with pytest.raises(ValueError, match="share ISA"):
        archNamesByIsa([GFX1250_STRICT + "[cu=64]", GFX1250])


# =========================================================================== #
# Scratch space. The runs covering two architectures that share an ISA write one
# output directory, so each has to keep its intermediates to itself: kernel
# basenames are derived from the ISA, making them identical across the two while
# the machine code inside differs.
# =========================================================================== #
def test_two_steppings_get_different_scratch_directories():
    """The collision this exists to prevent: same output directory, same kernel
    basenames, different machine code."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    assert buildTmpDir("/out/Tensile", [GFX1250]) != buildTmpDir(
        "/out/Tensile", [GFX1250_STRICT]
    )


def test_scratch_directory_ignores_order_and_qualifiers():
    """One run's scratch has to resolve to one directory however its
    architectures were spelled, or the cleanup would not find what the writer
    made."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    assert buildTmpDir("/out/Tensile", ["gfx942", GFX1250]) == buildTmpDir(
        "/out/Tensile", [GFX1250, "gfx942"]
    )
    assert buildTmpDir("/out/Tensile", ["gfx942:xnack+", "gfx942:xnack-"]) == (
        buildTmpDir("/out/Tensile", ["gfx942"])
    )


def test_scratch_stays_under_the_output_directory():
    """It is removed with rmtree, so the tag must not be able to steer that
    anywhere but inside build_tmp."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    path = buildTmpDir("/out/Tensile", [GFX1250_STRICT])

    assert path.parent == Path("/out/Tensile/build_tmp")
    assert path.name.isascii() and "/" not in path.name and ".." not in path.name


def test_scratch_keeps_its_old_name_when_no_stepping_was_asked_for():
    """Every architecture set that could be built before steppings existed has to
    land where it always did, or upgrading orphans the scratch of every build in
    flight and every tool that reaches into it by path."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    for archs in ([], ["gfx942"], ["gfx942", "gfx950", "gfx1200"], ["gfx942:xnack+"]):
        assert buildTmpDir("/out/Tensile", archs) == Path("/out/Tensile/build_tmp/TENSILE")


def test_only_the_stepping_side_of_a_shared_isa_is_renamed():
    """The two runs covering a shared ISA never occupy one group, so naming only
    the stepping's group apart is enough to separate them -- and it leaves the
    other run on the path it had."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    assert buildTmpDir("/out/Tensile", [GFX1250]) == Path("/out/Tensile/build_tmp/TENSILE")
    assert buildTmpDir("/out/Tensile", [GFX1250_STRICT]) == (
        Path(f"/out/Tensile/build_tmp/TENSILE-{GFX1250_STRICT}")
    )


def test_scratch_name_survives_an_output_path_with_no_stem():
    """Path("/").stem is empty, and "<root>" / "" is "<root>" -- the run would
    take the shared parent as its own scratch and rmtree a sibling's work."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    for outputPath in ("/", ""):
        path = buildTmpDir(outputPath, [GFX1250_STRICT])
        assert path.name and path.parent.name == "build_tmp"


def test_scratch_name_ignores_architectures_the_table_does_not_know():
    """An architecture spec is free text and gfxToIsa reads only the leading gfx
    digits, so "gfx1250/.." parses as a stepping. The name reaches rmtree, so
    only names the table vouches for may reach the name."""
    from Tensile.TensileCreateLibrary.Run import buildTmpDir

    for hostile in ("gfx1250/..", "gfx1250-../..", "gfx1250 ", "../gfx1250-x"):
        path = buildTmpDir("/out/Tensile", [hostile])
        assert path.parent == Path("/out/Tensile/build_tmp")
        assert "/" not in path.name and ".." not in path.name


def test_gfx_names_that_spell_no_version_are_not_an_error():
    """buildTmpDir asks whether a spec is a stepping, so gfxToIsa has to answer
    for anything a user can type. The hex digits its pattern accepts are not all
    parseable as a version, and the promise it makes for those is None."""
    from Tensile.Common.Architectures import gfxToIsa

    assert gfxToIsa("gfx1250") == (12, 5, 0)
    assert gfxToIsa("gfx90a") == (9, 0, 10)
    for unparseable in ("gfxabc", "gfxa00", "gfxbeef"):
        assert gfxToIsa(unparseable) is None


# =========================================================================== #
# Partitioning. One run cannot name two architectures sharing an ISA, so the
# build asks isaCollisionFreeGroups how many runs it takes to cover what was
# requested. The groups feed one run each, all writing one output directory.
# =========================================================================== #
def test_ordinary_architectures_all_fit_in_one_group():
    """Nothing collides among them, so the partitioning has to stay invisible:
    one group means one run, which is what every build did before steppings
    existed."""
    from Tensile.Common.Architectures import isaCollisionFreeGroups

    assert isaCollisionFreeGroups(["gfx942", "gfx950", "gfx1200"]) == [
        ["gfx942", "gfx950", "gfx1200"]
    ]
    assert isaCollisionFreeGroups([]) == []


def test_a_stepping_and_its_base_land_in_different_groups():
    """The pairing archNamesByIsa rejects is exactly the one that must split."""
    from Tensile.Common.Architectures import isaCollisionFreeGroups

    assert isaCollisionFreeGroups([GFX1250, GFX1250_STRICT]) == [
        [GFX1250],
        [GFX1250_STRICT],
    ]
    # Requested order is kept, so neither is privileged.
    assert isaCollisionFreeGroups([GFX1250_STRICT, GFX1250]) == [
        [GFX1250_STRICT],
        [GFX1250],
    ]


def test_qualified_specs_of_one_architecture_are_not_split():
    """They share an ISA *and* a name, so one run covers both. Splitting them
    would double a build that has no collision in it at all."""
    from Tensile.Common.Architectures import isaCollisionFreeGroups

    assert isaCollisionFreeGroups(["gfx942:xnack+", "gfx942:xnack-"]) == [
        ["gfx942:xnack+", "gfx942:xnack-"]
    ]


def test_specs_come_back_out_of_a_group_unchanged():
    """Groups are handed to a build as-is, so a spec has to survive whole. A
    predicate carries commas of its own, which is what makes the separator
    between specs a matter of correctness rather than taste."""
    from Tensile.Common.Architectures import isaCollisionFreeGroups

    spec = "gfx942[id=74a0,cu=80]"
    assert isaCollisionFreeGroups([spec, "gfx950"]) == [[spec, "gfx950"]]
    # A predicate must not hide a collision either.
    assert isaCollisionFreeGroups([GFX1250_STRICT + "[cu=64]", GFX1250]) == [
        [GFX1250_STRICT + "[cu=64]"],
        [GFX1250],
    ]


def test_every_group_is_one_archNamesByIsa_accepts():
    """The partitioning is only useful if each group can actually be built, which
    is what archNamesByIsa raising on a group would deny."""
    from Tensile.Common.Architectures import archNamesByIsa, isaCollisionFreeGroups

    for group in isaCollisionFreeGroups(["all", GFX1250_STRICT]):
        archNamesByIsa(group)


def test_all_is_expanded_before_partitioning():
    """``all`` cannot name a stepping, so a request for both arrives as the
    keyword beside a name it does not cover; the stepping has to survive into its
    own group rather than be absorbed or dropped."""
    from Tensile.Common.Architectures import isaCollisionFreeGroups

    groups = isaCollisionFreeGroups(["all", GFX1250_STRICT])

    assert len(groups) == 2
    assert GFX1250 in groups[0] and "gfx942" in groups[0]
    assert groups[1] == [GFX1250_STRICT]


# =========================================================================== #
# Fan-out. Each group is handed to its own process, because a run settles
# arch-dependent state process-wide. The children are this same entry point
# re-invoked, so what they are handed has to be a faithful narrowing of what
# this process was handed.
# =========================================================================== #
PARENT_ARGV = [
    "--architecture=gfx1250;gfx1250-strict",
    "--cxx-compiler=/opt/rocm/bin/amdclang++",
    "--jobs=64",
    "--disable-asm-comments",
    "/src/library",
    "/build/Tensile",
    "HIP",
]


def _spawnedCommands(monkeypatch, groups, argv=PARENT_ARGV, requestedJobs=-1):
    """The argv and environment of every child _buildGroupsSeparately starts.

    ``requestedJobs`` is the parsed ``CpuThreads``; -1 is its default, meaning
    "every CPU".
    """
    from Tensile.TensileCreateLibrary import Run

    spawned = []

    class _Proc:
        def wait(self):
            return 0

        def poll(self):
            return 0

    def _popen(cmd, env=None):
        spawned.append((cmd, env))
        return _Proc()

    monkeypatch.setattr(Run.sys, "argv", ["TensileCreateLibrary"] + argv)
    monkeypatch.setattr(Run.subprocess, "Popen", _popen)
    # Pinned so the job split is the machine-independent part of the arithmetic.
    monkeypatch.setattr(Run, "_cpuCount", lambda: 128)
    Run._buildGroupsSeparately(groups, requestedJobs)
    return spawned


def test_each_group_is_built_for_exactly_the_architectures_it_holds(monkeypatch):
    spawned = _spawnedCommands(monkeypatch, [[GFX1250, "gfx942"], [GFX1250_STRICT]])

    archArgs = [[t for t in cmd if t.startswith("--architecture=")] for cmd, _ in spawned]

    assert archArgs == [
        ["--architecture=gfx1250;gfx942"],
        ["--architecture=gfx1250-strict"],
    ]


def test_everything_the_caller_asked_for_survives_into_the_children(monkeypatch):
    """A group is the same build, only narrower. Anything dropped here is a
    setting that silently stops applying the moment a stepping is requested."""
    spawned = _spawnedCommands(monkeypatch, [[GFX1250], [GFX1250_STRICT]])

    for cmd, _ in spawned:
        # The three leading tokens are the interpreter and -m entry point.
        passedThrough = [
            t for t in cmd[3:] if not t.startswith(("--architecture=", "--jobs="))
        ]
        assert passedThrough == [
            "--cxx-compiler=/opt/rocm/bin/amdclang++",
            "--disable-asm-comments",
            "/src/library",
            "/build/Tensile",
            "HIP",
        ]


def test_the_requested_parallelism_is_split_across_the_groups(monkeypatch):
    """Each group is a whole build, so passing the count through unchanged would
    spend it per group and oversubscribe the machine by the number of groups.

    The count comes from the parsed argument, not from re-reading argv: argparse
    accepts ``-j=8`` and any unambiguous abbreviation of ``--jobs``, and a
    scanner that misses one hands the group an invented number instead.
    """
    spawned = _spawnedCommands(
        monkeypatch, [[GFX1250], [GFX1250_STRICT]], requestedJobs=64
    )

    for cmd, _ in spawned:
        assert "--jobs=32" in cmd
        assert "--jobs=64" not in cmd


def test_the_larger_group_gets_the_larger_share(monkeypatch):
    """The split that actually happens is lopsided: a stepping collides with the
    one architecture it steps from, so it builds alone while every other
    architecture goes in the other group. Splitting evenly would run that group at
    half speed and leave half the machine idle as soon as the single-architecture
    group finishes."""
    spawned = _spawnedCommands(
        monkeypatch,
        [["gfx942", "gfx950", "gfx1200"], [GFX1250_STRICT]],
        requestedJobs=64,
    )

    jobs = [t for cmd, _ in spawned for t in cmd if t.startswith("--jobs=")]
    assert jobs == ["--jobs=48", "--jobs=16"]
    # Still bounded by what the caller asked for, so the machine is not
    # oversubscribed the way one count per group would do.
    assert sum(int(j.split("=")[1]) for j in jobs) <= 64


def test_disabled_threading_survives_the_split(monkeypatch):
    """0 means "no threading", not "a number to divide" -- 0 // n is 0 either way,
    but rounding it up to 1 would start a worker the caller switched off."""
    spawned = _spawnedCommands(
        monkeypatch, [[GFX1250], [GFX1250_STRICT]], requestedJobs=0
    )

    for cmd, _ in spawned:
        assert "--jobs=0" in cmd


def test_children_can_import_tensile(monkeypatch):
    """The children are started with -m, which does not inherit the sys.path
    edit that bin/TensileCreateLibrary makes when Tensile is not installed."""
    import Tensile

    spawned = _spawnedCommands(monkeypatch, [[GFX1250], [GFX1250_STRICT]])
    packageRoot = str(Path(Tensile.__file__).resolve().parent.parent)

    for _, env in spawned:
        assert packageRoot in env["PYTHONPATH"].split(os.pathsep)


def test_children_are_told_they_share_a_scratch_parent(monkeypatch):
    """Scratch cleanup is the one place a group build must behave differently: a
    child may take only its own subdirectory, because a sibling is writing into
    the same parent. Every other run reclaims the whole tree, so the children
    have to be marked or a fan-out would delete a live sibling's scratch."""
    from Tensile.TensileCreateLibrary import Run

    spawned = _spawnedCommands(monkeypatch, [[GFX1250], [GFX1250_STRICT]])

    for _, env in spawned:
        assert env[Run._GROUP_BUILD_ENV] == "1"


def test_a_failed_group_is_not_swallowed(monkeypatch):
    """A group that fails while the other succeeds has to fail the run: a zero
    exit here lets the build be stamped with one stepping's library missing."""
    from Tensile.TensileCreateLibrary import Run

    class _Proc:
        def __init__(self, rc):
            self._rc = rc

        def wait(self):
            return self._rc

        def poll(self):
            return self._rc

    returnCodes = iter([0, 1])
    monkeypatch.setattr(Run.sys, "argv", ["TensileCreateLibrary"] + PARENT_ARGV)
    monkeypatch.setattr(
        Run.subprocess, "Popen", lambda cmd, env=None: _Proc(next(returnCodes))
    )

    with pytest.raises(SystemExit):
        Run._buildGroupsSeparately([[GFX1250], [GFX1250_STRICT]], -1)


def test_the_architecture_flag_is_replaced_whatever_its_spelling(monkeypatch):
    """Replaced, not merely overridden. argparse's last-wins would pick the
    appended one either way, so what this pins is that the child's target does
    not rest on that: its command line says one architecture, the one it builds.
    """
    argv = ["--architecture", "gfx1250;gfx1250-strict", "/src", "/out", "HIP"]
    spawned = _spawnedCommands(monkeypatch, [[GFX1250_STRICT]], argv=argv)

    cmd = spawned[0][0]

    assert "--architecture" not in cmd
    assert "gfx1250;gfx1250-strict" not in cmd
    assert "--architecture=gfx1250-strict" in cmd


def _run_createlibrary_to_writes(
    monkeypatch, tmp_path, arch, masterKey, mappingValue, shardNames=()
):
    """Drives ``run()`` all the way through the per-arch master/mapping write loops
    with the heavy steps stubbed, capturing every ``LibraryIO.write`` path and the
    arguments handed to ``writeSolutionsAndKernelsTCL``. Unlike ``_run_createlibrary``
    this does not stop early, so the output-naming of the writes is what is pinned.

    ``shardNames`` populates the master library's ``lazyLibraries`` so the shard
    write loop (``writeMsl``) actually runs; the ``ParallelMap2`` stub invokes the
    callable rather than swallowing it, so the shard routing is exercised too.
    """
    from unittest.mock import MagicMock

    import Tensile.TensileCreateLibrary.Run as RunModule

    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    captured = {"writes": [], "wsk": {}}
    writeSignature = inspect.signature(RunModule.writeSolutionsAndKernelsTCL)

    lazyLibraries = {name: MagicMock() for name in shardNames}

    def _iim(isas, _cxx):
        return {isa: IsaInfo({"SupportedISA": True}, {}, {}, {}) for isa in isas}

    def _wsk(*args, **kwargs):
        bound = writeSignature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured["wsk"]["cmdlineArchs"] = bound.arguments["cmdlineArchs"]
        return (0, [], [])

    def _glds(logicFiles, *a, **kw):
        return ([], {masterKey: MagicMock(lazyLibraries=lazyLibraries)}, {0: mappingValue})

    def _capture_write(filename, *a, **kw):
        captured["writes"].append(str(filename))

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
            "KeepBuildTmp": True,
        },
    )
    monkeypatch.setattr(RunModule, "setVerbosity", lambda *a, **kw: None)
    monkeypatch.setattr(
        RunModule,
        "validateToolchain",
        lambda *a: ("/fake/hipcc", None, "/fake/bundler", None, None),
    )
    monkeypatch.setattr(RunModule, "makeIsaInfoMap", _iim)
    monkeypatch.setattr(RunModule, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(RunModule, "makeAssemblyToolchain", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(RunModule, "makeSourceToolchain", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(RunModule, "KernelWriterAssembly", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(RunModule, "generateLogicDataAndSolutions", _glds)
    monkeypatch.setattr(
        RunModule, "generateKernelObjectsFromSolutions", lambda *a, **kw: []
    )
    monkeypatch.setattr(RunModule, "generateKernelHelperObjects", lambda *a, **kw: [])
    monkeypatch.setattr(RunModule, "copyStaticFiles", lambda *a, **kw: [])
    monkeypatch.setattr(RunModule, "writeSolutionsAndKernelsTCL", _wsk)
    monkeypatch.setattr(RunModule, "passPostKernelInfoToLibrary", lambda *a, **kw: None)
    monkeypatch.setattr(
        RunModule, "ParallelMap2", lambda fn, it, *a, **kw: [fn(*x) for x in it]
    )
    monkeypatch.setattr(RunModule, "state", lambda x: x)
    monkeypatch.setattr(RunModule.LibraryIO, "write", _capture_write)

    RunModule.run()
    return captured


def test_strict_build_writes_master_and_mapping_into_its_own_subtree(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The runtime forms both the subtree and the filename from the architecture
    the driver reports, which for v0 silicon is gfx1250-strict -- so both must
    carry the stepping. A master written as TensileLibrary_lazy_gfx1250 into
    library/gfx1250-strict/ is a file no lookup ever asks for."""
    captured = _run_createlibrary_to_writes(
        monkeypatch, tmp_path, GFX1250_STRICT, GFX1250_STRICT, "prefix_" + GFX1250_STRICT
    )

    writes = captured["writes"]
    assert any(
        w.endswith(f"library/{GFX1250_STRICT}/TensileLibrary_lazy_{GFX1250_STRICT}")
        for w in writes
    ), writes
    assert any(
        w.endswith(
            f"library/{GFX1250_STRICT}/TensileLiteLibrary_lazy_{GFX1250_STRICT}_Mapping"
        )
        for w in writes
    ), writes
    # Nothing for this build may fall back into the base architecture's subtree.
    assert not any(f"library/{GFX1250}/" in w for w in writes), writes


def test_a_strict_builds_shards_route_to_its_subtree_under_its_own_name(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The shards are the actual kernel payload: a master in library/gfx1250-strict/
    whose shards landed in library/gfx1250/ is the silent-empty-library failure
    this whole design exists to prevent. The shard name comes from the master's
    own name, so it carries the stepping too."""
    shard = "TensileLibrary_lazy_" + GFX1250_STRICT + "_0"
    captured = _run_createlibrary_to_writes(
        monkeypatch,
        tmp_path,
        GFX1250_STRICT,
        GFX1250_STRICT,
        "prefix_" + GFX1250_STRICT,
        shardNames=(shard,),
    )

    writes = captured["writes"]
    assert any(
        w.endswith(f"library/{GFX1250_STRICT}/{shard}") for w in writes
    ), writes
    assert not any(f"library/{GFX1250}/" in w for w in writes), writes


def test_strict_build_hands_its_own_name_to_the_code_object_builders(
    monkeypatch, tmp_path
):
    """run() -> writer -> the two builders is the chain that routes every .co and
    .hsaco. The writer must resolve the requested name itself: the assembly builder
    gets it keyed by ISA (its kernels carry only ISAs), the source builder gets the
    command line verbatim and reads it back off the bundler's targets."""
    from unittest.mock import MagicMock

    import Tensile.TensileCreateLibrary.Run as RunModule

    seen = {}
    srcSignature = inspect.signature(RunModule.buildSourceCodeObjectFiles)

    def _asm(*a, **kw):
        seen["asm"] = kw.get("archNames")
        return []

    def _src(*a, **kw):
        bound = srcSignature.bind(*a, **kw)
        bound.apply_defaults()
        seen["src"] = bound.arguments["cmdlineArchs"]
        return []

    monkeypatch.setattr(RunModule, "buildAssemblyCodeObjectFiles", _asm)
    monkeypatch.setattr(RunModule, "buildSourceCodeObjectFiles", _src)
    monkeypatch.setattr(RunModule, "ParallelMap2", lambda *a, **kw: [])
    monkeypatch.setattr(RunModule, "writeHelpers", lambda *a, **kw: None)
    monkeypatch.setattr(RunModule, "rocisa", MagicMock())

    RunModule.writeSolutionsAndKernelsTCL(
        str(tmp_path),          # outputPath
        MagicMock(),            # asmToolchain
        MagicMock(),            # srcToolchain
        [],                     # solutions
        [],                     # kernels
        [],                     # kernelHelperObjs
        MagicMock(),            # kernelWriterAssembly
        [GFX1250_STRICT],       # cmdlineArchs
    )

    assert seen["asm"] == {ISA_GFX1250: GFX1250_STRICT}
    assert seen["src"] == [GFX1250_STRICT]


def test_strict_build_threads_its_own_name_to_the_kernel_writer(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The .co and helper kernels are routed by the writer, not this loop, so the
    requested name has to reach it intact rather than collapsing to gfx1250."""
    captured = _run_createlibrary_to_writes(
        monkeypatch, tmp_path, GFX1250_STRICT, GFX1250_STRICT, "prefix_" + GFX1250_STRICT
    )

    assert captured["wsk"]["cmdlineArchs"] == [GFX1250_STRICT]


def test_ordinary_build_output_paths_are_unchanged(
    monkeypatch, tmp_path, restore_global_parameters
):
    """The regression guard: a gfx942 build must write exactly where it does today,
    so the naming change is provably inert for every architecture but the
    stepping."""
    captured = _run_createlibrary_to_writes(
        monkeypatch, tmp_path, "gfx942", "gfx942", "prefix_gfx942"
    )

    writes = captured["writes"]
    assert any(
        w.endswith("library/gfx942/TensileLibrary_lazy_gfx942") for w in writes
    ), writes
    assert any(
        w.endswith("library/gfx942/TensileLiteLibrary_lazy_gfx942_Mapping")
        for w in writes
    ), writes
    assert captured["wsk"]["cmdlineArchs"] == ["gfx942"]


# =========================================================================== #
# Toolchain destination routing. Code objects and helper kernels are fanned out
# into library/<arch>/ by the assembly and source builders. The assembly builder
# groups kernels by the ISA they canonicalize to, so it is the one place that
# cannot recover the architecture on its own and has to be told.
# =========================================================================== #
def test_assembly_co_carries_the_stepping_in_both_subtree_and_filename(tmp_path):
    """The default (non-lazy) code object is named after the architecture, and the
    runtime looks it up by the name the driver reports. Deriving it from the ISA
    would write TensileLibrary_gfx1250.co for a build no gfx1250 agent can use."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles

    coFiles = buildAssemblyCodeObjectFiles(
        MagicMock(),
        MagicMock(),
        [{"ISA": ISA_GFX1250, "BaseName": "k0"}],
        tmp_path,
        tmp_path,
        compress=True,
        archNames={ISA_GFX1250: GFX1250_STRICT},
    )

    assert len(coFiles) == 1
    assert str(coFiles[0]).endswith(
        f"{GFX1250_STRICT}/TensileLibrary_{GFX1250_STRICT}.co"
    ), coFiles
    assert (tmp_path / GFX1250_STRICT).is_dir()
    assert not (tmp_path / GFX1250).exists()


def test_assembly_shard_keeps_its_recorded_name_in_the_stepping_subtree(tmp_path):
    """A lazy shard is named by the master library that references it, so the
    builder must not rename it -- only the subtree it lands in is the builder's
    to choose."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles

    shard = "TensileLibrary_lazy_" + GFX1250_STRICT + "_0"
    coFiles = buildAssemblyCodeObjectFiles(
        MagicMock(),
        MagicMock(),
        [{"ISA": ISA_GFX1250, "BaseName": "k0", "codeObjectFile": shard}],
        tmp_path,
        tmp_path,
        compress=True,
        archNames={ISA_GFX1250: GFX1250_STRICT},
    )

    assert len(coFiles) == 1
    assert str(coFiles[0]).endswith(f"{GFX1250_STRICT}/{shard}.co"), coFiles


def test_assembly_co_is_unchanged_for_ordinary_archs(tmp_path):
    from unittest.mock import MagicMock

    from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles

    kernel = {
        "ISA": gfxToIsa("gfx942"),
        "BaseName": "k0",
        "codeObjectFile": "TensileLibrary_lazy_gfx942",
    }
    coFiles = buildAssemblyCodeObjectFiles(
        MagicMock(), MagicMock(), [kernel], tmp_path, tmp_path, compress=True
    )

    assert len(coFiles) == 1
    assert str(coFiles[0]).endswith("gfx942/TensileLibrary_lazy_gfx942.co"), coFiles


# =========================================================================== #
# Compiler target. A stepping shares its ISA with the architecture it steps, so
# the target cannot be derived from the ISA a kernel canonicalizes to; the
# requested name is carried down to -mcpu and to the bundle entry instead.
# =========================================================================== #
def test_strict_is_assembled_for_its_own_target(monkeypatch):
    from Tensile.Toolchain import Component as ComponentMod

    captured = []
    monkeypatch.setattr(ComponentMod, "_getVersion", lambda *a, **k: None)
    monkeypatch.setattr(ComponentMod, "_invoke", lambda args, desc: captured.append(args))
    assembler = ComponentMod.Assembler(Path("amdclang++"), 5)
    assembler(GFX1250_STRICT, 32, "k.s", "k.o")
    assembler(GFX1250, 32, "k.s", "k.o")

    strictArgs, baseArgs = captured
    assert f"-mcpu={GFX1250_STRICT}" in strictArgs, strictArgs
    # -mcpu decides the ELF machine code, and it is the only thing that differs;
    # the stepping keeps gfx1250's +real-true16.
    assert [a.replace(GFX1250_STRICT, GFX1250) for a in strictArgs] == baseArgs


def _compressTargetOf(tmp_path, isa, archNames=None):
    from unittest.mock import MagicMock

    from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles

    bundler = MagicMock()
    kernel = {"ISA": isa, "BaseName": "k0", "codeObjectFile": "TensileLibrary_lazy_x"}
    buildAssemblyCodeObjectFiles(
        MagicMock(), bundler, [kernel], tmp_path, tmp_path, compress=True,
        archNames=archNames,
    )
    return bundler.compress.call_args[0][2]


def test_assembly_bundle_is_tagged_with_the_target_it_was_built_for(tmp_path):
    """The runtime unbundles only the entry matching the agent it is loading onto,
    so a strict bundle tagged gfx1250 yields no code object at all."""
    from Tensile.Common.Architectures import archNamesByIsa

    assert _compressTargetOf(
        tmp_path, ISA_GFX1250, archNames=archNamesByIsa([GFX1250_STRICT])
    ) == GFX1250_STRICT
    assert _compressTargetOf(tmp_path, gfxToIsa("gfx942")) == "gfx942"


def _run_build_source(tmp_path, monkeypatch, bundlerTarget, cmdlineArchs):
    from unittest.mock import MagicMock

    from Tensile.Toolchain import Source as SourceMod

    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
    monkeypatch.setattr(SourceMod.shutil, "move", lambda s, d: None)

    bundler = MagicMock()
    bundler.targets = lambda objPath: [bundlerTarget]
    kernelPath = tmp_path / "Kernels.cpp"
    kernelPath.write_text("")

    return SourceMod.buildSourceCodeObjectFiles(
        MagicMock(),
        bundler,
        tmp_path / "lib",
        tmp_path / "tmpobj",
        tmp_path / "inc",
        kernelPath,
        cmdlineArchs,
    )


def test_source_helper_co_is_named_by_the_target_it_was_compiled_for(
    tmp_path, monkeypatch
):
    """The helper kernel is compiled with --offload-arch=gfx1250-strict, so the
    bundler reports that target back and both the subtree and the filename follow
    it -- no separate map is needed, and none may override it."""
    coPaths = _run_build_source(
        tmp_path, monkeypatch, GFX1250_STRICT, [GFX1250_STRICT]
    )

    assert len(coPaths) == 1
    assert str(coPaths[0]).endswith(
        f"{GFX1250_STRICT}/Kernels.so-000-{GFX1250_STRICT}.hsaco"
    ), coPaths


def test_source_helper_co_is_unchanged_for_ordinary_archs(tmp_path, monkeypatch):
    coPaths = _run_build_source(tmp_path, monkeypatch, "gfx942", ["gfx942"])

    assert len(coPaths) == 1
    assert str(coPaths[0]).endswith("gfx942/Kernels.so-000-gfx942.hsaco"), coPaths


# =========================================================================== #
# Helper kernel cache. The entry is keyed on the command line, which now spells
# the two steppings differently, so they no longer share one entry and the
# subtree names inside it round trip unchanged.
# =========================================================================== #
HELPER_HSACO = f"Kernels.so-000-{GFX1250_STRICT}.hsaco"


def test_the_two_steppings_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    """Sharing an entry would let a gfx1250 build restore code objects assembled
    for gfx1250-strict, whose ELF machine code no gfx1250 agent will load."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain import HelperKernelCache as HKC

    monkeypatch.setattr(HKC, "_STATIC_HEADER_FILES", ())
    (tmp_path / "Kernels.cpp").write_text("")
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "Kernels.h").write_text("")

    compiler = MagicMock()
    compiler.default_args = []

    keys = {
        HKC._computeCacheKey(tmp_path / "Kernels.cpp", inc, [arch], compiler)
        for arch in (GFX1250, GFX1250_STRICT)
    }
    assert len(keys) == 2


def _cacheAfterAStrictStore(tmp_path, monkeypatch):
    """Runs a strict build's cache miss and subsequent store, and returns the cache
    root."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain import HelperKernelCache as HKC

    monkeypatch.setattr(HKC, "_computeCacheKey", lambda *a, **kw: "KEY")
    cacheRoot = tmp_path / "cache"
    monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cacheRoot))
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

    libDir = tmp_path / "libstrict" / GFX1250_STRICT
    libDir.mkdir(parents=True)
    (libDir / HELPER_HSACO).write_text("data")

    cache = HKC.HelperKernelCache()
    hit, _ = cache.restore(
        tmp_path / "Kernels.cpp",
        tmp_path / "inc",
        [GFX1250_STRICT],
        MagicMock(),
        tmp_path / "libstrict",
    )
    assert not hit
    cache.store([str(libDir / HELPER_HSACO)])
    return cacheRoot


def test_helper_cache_stores_the_stepping_subtree_under_its_own_name(
    tmp_path, monkeypatch
):
    """The stored layout mirrors the install layout, and the install layout for a
    strict build is library/gfx1250-strict/. Rewriting it to the base name would
    make the entry restore into the wrong subtree."""
    cacheRoot = _cacheAfterAStrictStore(tmp_path, monkeypatch)

    assert (cacheRoot / "KEY" / GFX1250_STRICT / HELPER_HSACO).is_file()
    assert not (cacheRoot / "KEY" / GFX1250).exists()


def test_a_strict_builds_cache_entry_restores_into_the_stepping_subtree(
    tmp_path, monkeypatch
):
    """The end-to-end consequence: a second strict build in the same workspace
    hits the cache and must get its helper kernel back in library/gfx1250-strict/."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain import HelperKernelCache as HKC

    _cacheAfterAStrictStore(tmp_path, monkeypatch)

    cache = HKC.HelperKernelCache()
    hit, coPaths = cache.restore(
        tmp_path / "Kernels.cpp",
        tmp_path / "inc",
        [GFX1250_STRICT],
        MagicMock(),
        tmp_path / "libnext",
    )

    assert hit
    assert [str(p) for p in coPaths] == [
        str(tmp_path / "libnext" / GFX1250_STRICT / HELPER_HSACO)
    ]


def test_helper_cache_store_is_unchanged_for_ordinary_archs(tmp_path, monkeypatch):
    """The regression guard: the stored layout is the directory layout, exactly as
    before."""
    from unittest.mock import MagicMock

    from Tensile.Toolchain import HelperKernelCache as HKC

    monkeypatch.setattr(HKC, "_computeCacheKey", lambda *a, **kw: "KEY")
    cacheRoot = tmp_path / "cache"
    monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cacheRoot))
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

    libDir = tmp_path / "lib" / "gfx942"
    libDir.mkdir(parents=True)
    (libDir / "Kernels.so-000-gfx942.hsaco").write_text("data")

    cache = HKC.HelperKernelCache()
    cache.restore(
        tmp_path / "Kernels.cpp",
        tmp_path / "inc",
        ["gfx942"],
        MagicMock(),
        tmp_path / "lib",
    )
    cache.store([str(libDir / "Kernels.so-000-gfx942.hsaco")])

    assert (cacheRoot / "KEY" / "gfx942" / "Kernels.so-000-gfx942.hsaco").is_file()


# =========================================================================== #
# The helper-kernel generators' auto-detect path.
#
# AMaxGenerator/SoftmaxGenerator/LayerNormGenerator take the architecture from
# --arch; when it is missing they fall back to asking the device. That branch
# lives in `if __name__ == '__main__'`, so it is only reachable by running the
# script, which is why these spawn one rather than importing it.
# =========================================================================== #
_TENSILELITE = Path(__file__).resolve().parents[3]


def _rocmShimReporting(tmp_path, arch):
    """A ROCM_PATH whose detection tools all name ``arch``.

    amdgpu-arch has to be here too, not just the enumerator: detection asks it
    first, and a shim that left it out would fall through to the real one on
    PATH and answer from the machine running the test. The tools are shimmed
    through ROCM_PATH rather than PATH because both resolvers look under the
    ROCm install first; everything else the script needs is still found on PATH.
    """
    root = tmp_path / "rocm"
    for relative in ("bin/rocm_agent_enumerator", "lib/llvm/bin/amdgpu-arch"):
        tool = root / relative
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_text(f"#!/bin/sh\necho {arch}\n")
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return root


@pytest.mark.parametrize(
    "generator", ["AMaxGenerator.py", "SoftmaxGenerator.py", "LayerNormGenerator.py"]
)
@pytest.mark.parametrize("reported", [GFX1250, GFX1250_STRICT])
def test_the_generators_build_for_the_architecture_the_device_reported(
    tmp_path, generator, reported
):
    """Both steppings report ISA (12,5,0), so a generator that kept only the ISA
    would build gfx1250 on either and hand the strict device code it rejects.
    The gfx1250 case is the regression fence: naming the architecture must not
    change what an ordinary architecture builds."""
    script = _TENSILELITE / generator
    out = tmp_path / "kernel.s"
    env = dict(
        os.environ,
        ROCM_PATH=str(_rocmShimReporting(tmp_path, reported)),
        PYTHONPATH=str(_TENSILELITE),
    )

    result = subprocess.run(
        [sys.executable, str(script), "--arch", "", "-o", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f'.amdgcn_target "amdgcn-amd-amdhsa--{reported}"' in out.read_text()


# --- every architecture still parses the way it always did --------------------
#
# Giving one ISA two names changed the parsing every architecture goes through,
# not just the stepping's, so these sweep all of them. The regression they exist
# to catch is a legacy architecture quietly resolving differently, which no
# gfx1250 test would notice.


def _spellings(name):
    """Every spelling CMake may put in GPU_TARGETS for one architecture."""
    if ":" in name:
        return [name]
    return [name, f"{name}:xnack+", f"{name}:xnack-", f"{name}[cu=64]", f"{name}[id=74a0]"]


_ARCH_NAMES = sorted(n for n in architectureMap if n != "all")
_BARE_NAMES = sorted({baseArchName(n) for n in _ARCH_NAMES})
_ALL_SPECS = [(baseArchName(n), s) for n in _ARCH_NAMES for s in _spellings(n)]
_SPEC_IDS = [s for _, s in _ALL_SPECS]


@pytest.mark.parametrize("bare,spec", _ALL_SPECS, ids=_SPEC_IDS)
def test_an_architecture_matches_its_own_logic_header(bare, spec):
    """A header naming the architecture claims every spelling of a request for it.

    Logic file headers carry a bare name, so the qualifier and the predicate have
    to come off the request before the comparison, or tuning is dropped for a
    target that asked for it with either. Paired with the next test: this one
    catches a filter tightened until it drops real files, that one catches a
    filter loosened until "gfx1250" claims "gfx1250-strict". Neither alone would
    have caught the regression they were written for.
    """
    assert archMatch(bare, [spec]) is True


@pytest.mark.parametrize("bare,spec", _ALL_SPECS, ids=_SPEC_IDS)
def test_an_architecture_claims_no_other_architectures_request(bare, spec):
    """No header claims a request for a different architecture.

    Whole-name, not prefix: "gfx1250" is a prefix of "gfx1250-strict", and the
    two are separate compiler targets whose code objects will not load on each
    other's silicon.
    """
    for other in _BARE_NAMES:
        if other == bare:
            continue
        assert archMatch(other, [spec]) is False, f"{other!r} wrongly claimed {spec!r}"


@pytest.mark.parametrize("bare,spec", _ALL_SPECS, ids=_SPEC_IDS)
def test_qualifiers_and_predicates_do_not_change_the_isa(bare, spec):
    """Target features and predicates describe a configuration, not a target."""
    assert gfxToIsa(spec) == gfxToIsa(bare)


@pytest.mark.parametrize("bare,spec", _ALL_SPECS, ids=_SPEC_IDS)
def test_base_arch_name_strips_both_qualifier_and_predicate(bare, spec):
    assert baseArchName(spec) == bare


@pytest.mark.parametrize("name", [n for n in _BARE_NAMES if n != "gfx1250-strict"])
def test_only_the_stepping_reports_a_base_architecture(name):
    """A name that merely looks like a stepping must not answer.

    If it did, the architecture it names would be built under another's
    capability overrides.
    """
    assert steppingArchOf(name) is None


@pytest.mark.parametrize(
    "reported",
    ["gfx942", "gfx90a", "gfx1250", "gfx1250-strict",
     "gfx9-4-generic", "gfx12-5-generic", "gfx10-3-generic"],
)
def test_a_reported_name_is_captured_whole(reported):
    """The stepping suffix survives detection.

    rocm_agent_enumerator's own regex stops at the hyphen group it was written
    for, so it answers "gfx1250" for a gfx1250-strict agent -- a name Tensile
    accepts, which is why that truncation is silent rather than an error.
    """
    m = _REPORTED_ARCH_RE.search(f"  Name:   {reported}  ")
    assert m is not None and m.group(0) == reported


@pytest.mark.parametrize("reported", ["gfx942:sramecc+:xnack-", "gfx90a:xnack+"])
def test_target_features_are_not_part_of_the_captured_name(reported):
    assert _REPORTED_ARCH_RE.search(reported).group(0) == reported.split(":")[0]


def test_the_isa_regex_reads_only_the_leading_hex_digits():
    """gfxToIsa stops at the first non-hex character, by design and by hazard.

    By design, because a stepping has to resolve to the ISA it shares. By hazard,
    because a name that merely looks like one resolves too: "gfx1250v1" answers
    (12,5,0), which is why detection filters against architectureMap rather than
    trusting the parse.
    """
    assert gfxToIsa("gfx1250v1") == gfxToIsa("gfx1250")
    assert "gfx1250v1" not in architectureMap


@pytest.mark.parametrize("name", _BARE_NAMES)
def test_every_known_architecture_resolves_to_an_isa(name):
    assert gfxToIsa(name) is not None


def test_a_name_cannot_smuggle_a_path_separator():
    """Names become directory components, so a parse must not yield a path.

    gfxToIsa reads the leading digits and ignores the rest, so "gfx1250/.."
    parses as an ISA; what stops it is that architectureMap does not vouch for it.
    """
    assert gfxToIsa("gfx1250/..") == gfxToIsa("gfx1250")
    assert "gfx1250/.." not in architectureMap
    assert not re.fullmatch(r"[A-Za-z0-9_.+-]+", "gfx1250/..")
