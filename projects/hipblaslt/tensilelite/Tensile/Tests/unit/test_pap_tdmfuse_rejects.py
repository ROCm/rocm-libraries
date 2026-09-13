# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PrefetchAcrossPersistent against the non-default TDM descriptor groupings.

PAP hands a persistent tile over by rebuilding the TDM descriptors and by
restoring the LDS bank the next-tile prefetch landed in. Every helper on that
path walks the tensors as the two pairs (A, B) and (MXSA, MXSB) and applies
exactly one offset per pair, which is only the ownership the default grouping
has: {A,B} on one descriptor set and {MXSA,MXSB} on another. A grouping that
seats a scale tensor on a data tensor's set makes tdmMXSAGroup0/tdmMXSBGroup0
RegSet aliases of tdmAGroup0/tdmBGroup0, and then:

  * `papTdmRestoreLdsBank` and `papTdmUpdateDescriptor` add the bank offset once
    under tdmAGroup0 and once under tdmMXSAGroup0. Aliased, that is one physical
    register twice while its sibling range is never shifted. Both are plain
    `s_add_u32` with no normalize-first step, so unlike `papTdmSetTailLdsBank`
    they are NOT idempotent. Measured on gfx1250 at TDMFuse=1: the emitted
    kernel carries `.set sgprtdmMXSAGroup0, sgprtdmAGroup0+0`, both adds land on
    s25, and B's s29 is never touched. It assembles cleanly, so it is silent.
  * `tdmApplyStreamKTailOffsetWaveSeparated` names tdmMXSAMXSBIncs, whose
    defineSgpr is guarded by `not tdmFuseAMx`. Measured at TDMFuse=2: the symbol
    is referenced with no `.set`, and the assembler refuses the kernel with
    "expected absolute expression".

The rejection is derived from group membership rather than from a TDMFuse value,
so a grouping row nobody has wired to an integer yet inherits it. What follows
pins the firing, the accept controls that stop it over-rejecting, the precedence
against TDMFuse's own guards, and the assembly-level invariant behind it.
"""

import copy
import re

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Components.TDMFuse import (
    TDM_GROUPS,
    tdmGrouping,
    tdmPapRejectReason,
    tdmScaleSharesDataSet,
)

pytestmark = pytest.mark.unit

# Sibling unit tests mutate the process-global defaultSolution in place; snapshot
# it at import (collection precedes execution) so Solution construction here is
# not order-dependent.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

# MIWaveGroup [2,2] -> NumWaves 4, which is what TDMFuse=2 names explicitly and
# what TDMFuse=1 needs at least two waves for.
_MI_W4 = [16, 16, 128, 1, 1, 2, 4, 2, 2]

# The reject's own words, enough of them to tell it from its neighbours.
_PAP_GROUPING_MSG = "requires every TDM scale tensor to own its own descriptor set"
# TDMFuse's own drift guards, which must keep precedence over the one above.
_FUSE_DECLINED_MSG = "passed its solution-level guards but"


# ---------------------------------------------------------------------------
# Pure predicate: derived from the grouping, not from the TDMFuse integer.
# ---------------------------------------------------------------------------
def _ks(fuse=1, **ov):
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "NumWaves": 4,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": -1,
        "PrefetchGlobalReadB": -1,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    ks.update(ov)
    return ks


@pytest.mark.parametrize("fuse, expected", [(0, ()), (1, (("A", "MXSA"), ("MXSB", "B"))),
                                            (2, (("A", "MXSA", "MXSB"),))])
def test_shared_sets_read_off_the_resolved_grouping(fuse, expected):
    assert tdmScaleSharesDataSet(_ks(fuse=fuse)) == expected


@pytest.mark.parametrize("fuse", [1, 2])
def test_reason_fires_for_every_sharing_grouping(fuse):
    reason = tdmPapRejectReason(_ks(fuse=fuse))
    assert reason is not None
    assert _PAP_GROUPING_MSG in reason
    # It must explain the aliasing, not merely name the knob: a reader who has
    # only this string has to be able to find the two register ranges.
    assert "tdmMXSAGroup0/tdmMXSBGroup0" in reason
    assert "tdmAGroup0/tdmBGroup0" in reason
    assert tdmGrouping(_ks(fuse=fuse)).name in reason


def test_default_grouping_is_not_rejected():
    assert tdmPapRejectReason(_ks(fuse=0)) is None


@pytest.mark.parametrize("decline", [{"NumWaves": 1}, {"TDMSplit": True},
                                     {"UseSubtileImpl": True}, {"TDMInst": 1}])
def test_declined_grouping_answers_with_the_fallback(decline):
    """A grouping TDMFuse asked for but the predicates declined shares nothing.

    The writer falls back to {A,B} + {MXSA,MXSB} in that case, so rejecting on
    the requested value rather than the resolved grouping would refuse a
    solution whose descriptors are in fact separate. TDMFuse's own guards refuse
    these for their own reasons; this only pins that the reason here is silent.
    """
    assert tdmPapRejectReason(_ks(fuse=1, **decline)) is None
    assert tdmPapRejectReason(_ks(fuse=2, **decline)) is None


@pytest.mark.parametrize("missing", [{"MXBlockA": 0, "MXBlockB": 32},
                                     {"MXBlockA": 32, "MXBlockB": 0},
                                     {"MXBlockA": 0, "MXBlockB": 0}])
def test_scale_less_types_share_nothing(missing):
    """No live MXSA/MXSB means no scale to seat on a data tensor's set."""
    assert tdmPapRejectReason(_ks(fuse=1, ProblemType=missing)) is None


@pytest.mark.parametrize("row, shares", [("MX_AB", False), ("AB", False), ("None", False),
                                         ("paired", True), ("A_MX", True), ("B_MX", True)])
def test_every_grouping_row_answers_without_a_new_branch(row, shares):
    """Rows no TDMFuse integer selects inherit the right answer as data.

    B_MX in particular is A_MX mirrored and gets refused for free, which is the
    property that keeps this reject from needing an edit per new row.
    """
    assert bool(tdmScaleSharesDataSet(_ks(), TDM_GROUPS[row])) is shares


# ---------------------------------------------------------------------------
# Solution level: real gfx1250 caps + assembler, assignDerivedParameters
# end-to-end. Mirrors test_halfplr_streamk_rejects.py's harness.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
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


def _make_params(gfx1250_iim, mi=None, **overrides):
    """A TN MXF8F4 StreamK=3 solution: the smallest shape PAP+TDM accepts.

    StreamKForceDPOnly=1 because PAP+TDM only emits there -- at
    StreamKForceDPOnly=0 the writer fails for every grouping including the
    default, which is a separate pre-existing limitation and not what these
    tests are about.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = mi or _MI_W4
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM", "MacDataTypeA": "F8", "MacDataTypeB": "F4",
        "DataType": "F8", "DestDataType": "s", "ComputeDataType": "s",
        "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
        "UseBeta": True, "Batched": True, "MXBlockA": 32, "MXBlockB": 32,
        "DataTypeMXSA": "E8", "DataTypeMXSB": "E8",
    }
    problem_type.update(pt)
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 32, "DepthU": 256,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0,
        "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1, "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3, "LDSTrInst": False,
        "StreamK": 3, "StreamKForceDPOnly": 1, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": False, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 1,
        "TDMFuse": 0, "TDMSplit": False, "TDMCross": 0, "InitCIterWmma": 0,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim))
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from Tensile.SolutionStructs.Solution import Solution
    sol = Solution(_make_params(gfx1250_iim, **overrides), False, True, False,
                   assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


def test_pap_on_the_default_grouping_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The accept control. Without it every reject below is vacuous."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert sol.get("PrefetchAcrossPersistent") == 1, \
        "PAP was silently reset, so this control proves nothing"
    assert sol.get("LdsOffsetA_Blk") != 0, \
        "LdsOffsetA_Blk == 0 folds the LDS-bank helpers out entirely"


@pytest.mark.parametrize("fuse", [1, 2])
def test_pap_with_a_shared_scale_set_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=1, TDMFuse=fuse)
    assert sol.get("Valid") is False, "TDMFuse=%d + PAP was accepted" % fuse
    assert _PAP_GROUPING_MSG in out, "rejected for another reason: %r" % out
    # Diagnosis, not just validity: if an earlier guard starts shadowing this
    # one the message goes and only this assertion notices.
    assert _FUSE_DECLINED_MSG not in out


@pytest.mark.parametrize("fuse", [1, 2])
def test_the_same_grouping_without_pap_is_still_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The rejection is about PAP, not about the grouping."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=0, TDMFuse=fuse)
    assert sol.get("Valid") is True, "expected accept, rejected with: %r" % out
    assert _PAP_GROUPING_MSG not in out


def test_tdmfuse_own_guards_keep_precedence(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """A grouping that cannot be built reports its own reason, not this one.

    NumWaves=8 satisfies PAP's wave-separated gate but not TDMFuse=2's explicit
    four waves, so tdmFuseAMx declines, the writer falls back to the default
    grouping, and nothing is aliased for PAP to complain about.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=1,
                       TDMFuse=2, mi=[16, 16, 128, 1, 1, 2, 4, 2, 4])
    assert sol.get("Valid") is False
    assert "TDMFuse=2 dispatches its shared descriptor three ways" in out
    assert _PAP_GROUPING_MSG not in out


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_no_accepted_pap_solution_aliases_a_scale_onto_a_data_set(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The invariant every PAP TDM helper is written against.

    Stated over whatever survives validation rather than over a list of values,
    so it keeps its teeth if the reject is ever re-expressed. Against the
    unfixed tree TDMFuse=1 and 2 are accepted and this fails on both.
    """
    sol, _ = _derive(gfx1250_iim, assembler, capsys,
                     PrefetchAcrossPersistent=1, TDMFuse=fuse)
    if not sol.get("Valid"):
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to check" % fuse)
    assert tdmScaleSharesDataSet(sol) == (), (
        "accepted a PAP solution whose %s grouping shares a descriptor set with "
        "a scale tensor" % tdmGrouping(sol).name)


# ---------------------------------------------------------------------------
# Assembly level: the behaviour the reject exists to prevent.
#
# The tests above assert a message and a predicate. This one emits the real
# kernel and looks at the registers, which is the only level at which the two
# defects are visible: RegSet aliasing means one physical SGPR reached by two
# different spellings, and a missing defineSgpr means a symbol with no .set.
# ---------------------------------------------------------------------------

# Both of these are plain `s_add_u32 dst, dst, x`. papTdmSetTailLdsBank is
# deliberately excluded: it normalizes to bank 0 before adding, so it IS
# idempotent and is applied more than once per register by design.
_NON_IDEMPOTENT_BANK_SITES = {
    "papTdmRestoreLdsBank": ("shift A/B descriptor to PAP bank",
                             "shift MX descriptor to PAP bank"),
    "papTdmUpdateDescriptor": ("restore PAP LDS bank after descriptor refresh",),
}
_LDS_ADDR_SYMBOLS = ("sgprtdmAGroup0+1", "sgprtdmBGroup0+1",
                     "sgprtdmMXSAGroup0+1", "sgprtdmMXSBGroup0+1")


def _resolve_sets(asm):
    """symbol -> physical register, honouring the first .set and skipping UNDEF."""
    raw = {}
    for m in re.finditer(r"^\s*\.set\s+(\S+?),\s*(.+?)\s*$", asm, re.M):
        name, value = m.group(1), m.group(2).strip()
        if name in raw or value == "UNDEF":
            continue
        raw[name] = value

    def resolve(name, depth=0):
        if depth > 40:
            return None
        value = raw.get(name)
        if value is None:
            return None
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*\+\s*(\d+)", value)
        if m:
            base = resolve(m.group(1), depth + 1)
            return None if base is None else base + int(m.group(2))
        return resolve(value, depth + 1)

    out = {name: resolve(name) for name in raw}

    def phys(symbol):
        m = re.fullmatch(r"(\w+?)\+(\d+)", symbol)
        if m:
            base = out.get(m.group(1))
            return None if base is None else base + int(m.group(2))
        return out.get(symbol)

    return out, phys


def _emit_asm(gfx1250_iim, assembler, **overrides):
    """(solution, assembly text or None). CPU-only; no GPU is touched."""
    import shutil
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.TensileCreateLibrary.Run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    sol = Solution(_make_params(gfx1250_iim, **overrides), False, True, False,
                   assembler, gfx1250_iim)
    if not sol.get("Valid"):
        return sol, None
    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        pieces = []
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]),
                    shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
    return sol, "\n".join(pieces)


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_pap_shifts_each_descriptor_lds_bank_exactly_once(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every non-idempotent bank shift lands on its own physical register.

    Against the unfixed tree TDMFuse=1 and 2 are accepted, tdmMXSAGroup0 is a
    RegSet alias of tdmAGroup0, and both sites shift one register twice while
    the sibling range is never shifted -- which is what this fails on.
    """
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, TDMFuse=fuse)
    capsys.readouterr()
    if asm is None:
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to emit" % fuse)
    _, phys = _resolve_sets(asm)

    shifted = {}
    for line in asm.splitlines():
        m = re.search(r"s_add_u32 s\[(sgprtdm\w+Group0\+1)\], s\[\1\], \S+\s*//\s*(.*)",
                      line)
        if not m:
            continue
        comment = m.group(2).strip()
        for site, comments in _NON_IDEMPOTENT_BANK_SITES.items():
            if comment in comments:
                shifted.setdefault(site, []).append((m.group(1), phys(m.group(1))))

    for site, hits in shifted.items():
        regs = [reg for _, reg in hits]
        duplicated = sorted({r for r in regs if regs.count(r) > 1})
        assert not duplicated, (
            "%s shifts s%s more than once via aliased spellings %s" %
            (site, duplicated, [sym for sym, _ in hits]))

    # And the complement: no allocated descriptor set is left behind. Aliasing
    # does not only double one shift, it drops the sibling's shift entirely.
    if "papTdmRestoreLdsBank" in shifted:
        allocated = {phys(s) for s in _LDS_ADDR_SYMBOLS if phys(s) is not None}
        assert {reg for _, reg in shifted["papTdmRestoreLdsBank"]} == allocated, (
            "papTdmRestoreLdsBank shifted %s but the kernel allocates %s" %
            (sorted({reg for _, reg in shifted["papTdmRestoreLdsBank"]}), sorted(allocated)))


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_pap_never_names_an_unallocated_tdm_increment(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every tdm*Incs the kernel reads has a defineSgpr behind it.

    tdmMXSAMXSBIncs is allocated only when `not tdmFuseAMx`, while the PAP tail
    reset builds the name unconditionally. Against the unfixed tree TDMFuse=2 +
    PAP emits `s[sgprtdmMXSAMXSBIncs]` with no .set, and the assembler refuses
    it with "expected absolute expression".
    """
    sol, asm = _emit_asm(gfx1250_iim, assembler, PrefetchAcrossPersistent=1, TDMFuse=fuse)
    capsys.readouterr()
    if asm is None:
        pytest.skip("TDMFuse=%d + PAP is refused, nothing to emit" % fuse)
    resolved, _ = _resolve_sets(asm)
    referenced = set(re.findall(r"\bsgprtdm\w*Incs\b", asm))
    undefined = sorted(s for s in referenced if resolved.get(s) is None)
    assert not undefined, "referenced with no .set: %s" % undefined
