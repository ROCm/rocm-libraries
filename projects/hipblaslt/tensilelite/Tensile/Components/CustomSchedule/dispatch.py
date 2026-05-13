################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Dispatch, decorator, and registries for the CustomSchedule package."""

from copy import deepcopy
from dataclasses import asdict
from enum import Enum, auto
from itertools import product
from typing import Callable, Optional, Tuple

from rocisa.code import Macro, Module, TextBlock, ValueElseIf, ValueEndif, ValueIf
from rocisa.instruction import (
    MFMAInstruction, SCBranchSCC1, SMFMAInstruction, SNop, SWaitCnt,
)

import Tensile.Components.CMSValidator as cmsv
import Tensile.Components.ScheduleCapture as scap
from Tensile.Common import IsaVersion
from Tensile.Common.Utilities import printWarning

from .shared import (
    CMSKernelInfo,
    ScheduleInfo,
    TileConfig,
    _DTYPE_PREDICATE_NAMES,
    is16bit,
    is8bit,
    isMixed,
    isNN,
    isNT,
    isTF32,
    isTN,
    isTT,
)


# Enum to distinguish between different schedule matching outcomes
class ScheduleMatchStatus(Enum):
    FOUND = auto()                  # Schedule found and supported
    NO_MATCH = auto()               # Criteria don't match, continue searching
    UNSUPPORTED_VARIANT = auto()    # Criteria match but variant unsupported, stop searching


# Global registry for schedule functions.
# Mutated at module-import time by `@RegisterSchedule`. The
# layout-autodetection test save/restores this list, so it must remain a
# live module-level binding (re-exported, never deepcopied).
_SCHEDULE_REGISTRY = []

_SCHEDULE_METADATA: list[CMSKernelInfo] = []


def removeComments(module):
    retModule = Module()
    for i in module.flatitems():
        if type(i) != TextBlock and not isinstance(i, (SCBranchSCC1, SNop)):
            retModule.add(i)
    return retModule.flatitems()


def customMainLoopSchedule(writer, kernel, tensorParametersA, tensorParametersB, \
                      globalReadIncACode, globalReadIncBCode, \
                      LRCodeA, PackCodeA, LRCodeB, PackCodeB,\
                      LRSwapA, LRSwapB, \
                      globalReadA, globalReadB, \
                      LWSwapA, LWSwapB, \
                      mfmaCode, loopCounterCode, \
                      nta=0, ntb=0, \
                      ):
    strNta = "" if kernel["AdaptiveGemmNTAB"] == 0 else "_NTA%s"%nta
    strNtb = "" if kernel["AdaptiveGemmNTAB"] == 0 else "_NTB%s"%ntb
    module = Module()

    globalReadIncACode = removeComments(globalReadIncACode)
    globalReadIncBCode = removeComments(globalReadIncBCode)
    numLoopIter = kernel["LoopIters"]
    ph = -2 # placeholder index

    if numLoopIter > 1:
        for uIdx in range(0, numLoopIter):
            LRCodeA[uIdx] = removeComments(LRCodeA[uIdx])
            LRCodeB[uIdx] = removeComments(LRCodeB[uIdx])
            PackCodeA[uIdx] = removeComments(PackCodeA[uIdx])
            PackCodeB[uIdx] = removeComments(PackCodeB[uIdx])
    else: # numIterPLR = 0 case
        numLoopIter = 2
        # Split instruction stream for numiterPLR=0.
        # split_for_plr lives in ScheduleCapture so the default-side capture
        # path can apply the same split and its idMap matches CMS's.
        LRCodeA = scap.split_for_plr(LRCodeA[0])
        LRCodeB = scap.split_for_plr(LRCodeB[0])
        PackCodeA = scap.split_for_plr(PackCodeA[0])
        PackCodeB = scap.split_for_plr(PackCodeB[0])


    LRSwapA = removeComments(LRSwapA)
    LRSwapB = removeComments(LRSwapB)
    globalReadA = removeComments(globalReadA)
    globalReadB = removeComments(globalReadB)
    localWriteA = removeComments(writer.codes.localWriteA)
    localWriteB = removeComments(writer.codes.localWriteB)
    LWSwapA = removeComments(LWSwapA)
    LWSwapB = removeComments(LWSwapB)
    loopCounterCode = removeComments(loopCounterCode)
    mfmaCode = removeComments(mfmaCode)

    _, opt1 = hasCustomSchedule(kernel)
    numCodePath = opt1.numCodePaths
    assert opt1.numMfma == len(mfmaCode)


    for _, indexList in opt1.optSchedule.items():
        assert len(indexList) <= opt1.numCodePaths

    if len(opt1.mfmaReorder) > 0:
        mfmaCode = [mfmaCode[x] for x in opt1.mfmaReorder]

    idMap = scap.build_idmap(
        num_loop_iter=numLoopIter,
        LRCodeA=LRCodeA, PackCodeA=PackCodeA,
        LRCodeB=LRCodeB, PackCodeB=PackCodeB,
        globalReadA=globalReadA, globalReadB=globalReadB,
        globalReadIncACode=globalReadIncACode,
        globalReadIncBCode=globalReadIncBCode,
        localWriteA=localWriteA, localWriteB=localWriteB,
        LRSwapA=LRSwapA, LRSwapB=LRSwapB,
        LWSwapA=LWSwapA, LWSwapB=LWSwapB,
        loopCounterCode=loopCounterCode,
        syncCode=opt1.syncCode,
        snopCode=opt1.snopCode,
    )

    # Expose the real idMap and mfmaCode for test infrastructure.
    # Tests call _getKernelSource() then read writer._last_id_map to get
    # real rocisa instruction objects with correct register assignments.
    writer._last_id_map = idMap
    writer._last_mfma_code = mfmaCode

    # create the case str (TN, NT, TT, or NN)
    if isTN(kernel):
        case_str = "TN"
    elif isNT(kernel):
        case_str = "NT"
    elif isTT(kernel):
        case_str = "TT"
    elif isNN(kernel):
        case_str = "NN"
    else:
        case_str = "Unknown"
    # isValid raises ValidationError on failure; trap it here only to
    # re-raise with kernel-shape context attached. A pre-existing silent
    # bool-drop here would have been a defect, and now the contract makes
    # that drop impossible.
    try:
        cmsv.isValid(opt1, cmsv.ValidationContext(kernel=kernel, id_map=idMap, mfma_code=mfmaCode))
    except cmsv.ValidationError as exc:
        raise AssertionError(
            f"CMS validation failed for kernel "
            f"{kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']} "
            f"{case_str}: {exc}"
        ) from exc

    # Snapshot opt1 BEFORE scheduleInst() mutates opt1.optSchedule[key][cp][i] = ph
    # at line ~422 below. The capture machinery and any future rules that want
    # the as-built schedule recipe inspect this copy.
    opt1_for_capture = deepcopy(opt1)

    # Side-channel populated as instructions are added to the macro: maps the
    # Python id() of each emitted Instruction to its CMS id_map category. The
    # macro walker (Tensile.Components.ScheduleCapture.expand_cms_macro) reads
    # this map to recover tags after macro expansion. Deepcopied SWaitCnts from
    # nllvmcntHandling are not in the map but the walker falls back to SYNC via
    # isinstance checks against sync_class.
    tag_by_origin_id: dict = {}
    # Approach 2 (rocm-libraries-dfd8): parallel side-channel mapping each
    # emitted Instruction's id() to its rocisa-derived `source_module_id`
    # (the closest-named-ancestor `Module.name` of the source Module — e.g.
    # "globalReadIncrementA" for SCC cmps emitted from the GRIncA bucket).
    # Built from the ORIGINAL named source Modules (`writer.codes.*`)
    # because `idMap` here holds removed-comments lists that have lost the
    # enclosing Module wrapper and therefore the `.name` attribute. Leaf
    # `id()` is preserved by `removeComments` (it filters TextBlocks and
    # re-adds the surviving leaves to a fresh Module without cloning).
    # The default-side path (KernelWriter.py:_loopBody) calls the same
    # inversion against its own NAMED idmap, so both sides end up with
    # matching `source_module_id` strings on the same leaves — which is
    # what cross-build identity stability requires.
    _src_idmap_for_inversion = {
        'GRIncA': writer.codes.globalReadIncrements.findNamedItem("globalReadIncrementA")
                    if hasattr(writer.codes, 'globalReadIncrements') and writer.codes.globalReadIncrements is not None
                    else None,
        'GRIncB': writer.codes.globalReadIncrements.findNamedItem("globalReadIncrementB")
                    if hasattr(writer.codes, 'globalReadIncrements') and writer.codes.globalReadIncrements is not None
                    else None,
        'GRA':    writer.codes.globalReadA,
        'GRB':    writer.codes.globalReadB,
        'LWA':    writer.codes.localWriteA,
        'LWB':    writer.codes.localWriteB,
    }
    source_module_by_origin_id: dict = scap.invert_idmap_to_id_to_source_module(
        _src_idmap_for_inversion)

    InstStreams = {key: [stream, idMap[key]] for key, stream in opt1.optSchedule.items()}

    macro = Macro("MAINLOOP%s%s"%(strNta, strNtb), ["ID", "useGR=1", "usePLR=1", "useGRInc=1", "useLoop=1"])

    lastIter = numLoopIter - 1

    for miIndex in range(-1, len(mfmaCode)):
        if miIndex >= 0:
            macro.addComment0("mfmaIndex:%u"%(miIndex))
            mfmaItem = mfmaCode[miIndex]
            tag_by_origin_id[id(mfmaItem)] = "MFMA"
            macro.add(mfmaItem)

        def scheduleInst(keyName, indexList, instructionList):
            ret = [None]*len(indexList)
            totalNumInst = len(instructionList)
            for i in range(len(indexList)):
                if indexList[i]: # For specific codepath
                    cc = 0
                    # Add slower, but allow reordering of instructions
                    while miIndex in indexList[i]:
                        ind = indexList[i].index(miIndex)
                        if ind >= totalNumInst:
                            raise IndexError(
                                f"CMS schedule stream {keyName}[{i}] references instruction index {ind} "
                                f"but only {totalNumInst} instructions exist "
                                f"(indices={indexList[i]}, "
                                f"MT={kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']})")
                        if cc == 0:
                            ret[i] = Module()
                        ret[i].add(instructionList[ind])
                        cc += 1
                        indexList[i][ind] = ph
            if ret.count(None) == len(ret):
                return [None]
            else:
                return ret

        ToSched = {k: scheduleInst(k, stream[0], stream[1]) for k, stream in InstStreams.items()}

        def nllvmcntHandling(inst, shift0, shift1):
            if isinstance(inst, SWaitCnt) and (inst.vlcnt != -1 or (inst.dscnt != -1 and opt1.nllZeroDscnt)):
                macro.add(ValueIf("\\useGR == 1 && \\usePLR == 1")) # in main loop
                macro.addComment0("vmcnt used in main loop")
                macro.add(inst)
                macro.add(ValueElseIf("\\useGR == 0 && \\usePLR == 1")) # in NGL
                instModified = deepcopy(inst)
                if inst.vlcnt != -1:
                    macro.addComment0("vmcnt used in ngl, applying %u shift"%shift0)
                    instModified.vlcnt = max(0, instModified.vlcnt - shift0)
                macro.add(instModified)
                macro.add(ValueElseIf("\\useGR == 0 && \\usePLR == 0")) # in NLL
                instModified = deepcopy(inst)
                if inst.vlcnt != -1:
                    macro.addComment0("vmcnt used in nll, applying %u shift"%shift1)
                    instModified.vlcnt = max(0, instModified.vlcnt - shift1)
                if (inst.dscnt != -1 and opt1.nllZeroDscnt):
                    macro.addComment0("setting dscnt = 0 for NLL")
                    instModified.dscnt = 0
                macro.add(instModified)
                macro.add(ValueEndif())
            else:
                macro.add(inst)

        def get_macro_guard(key):
            """Determine the macro guard for a given instruction key."""
            if key in ['GRIncA', 'GRIncB']:
                return "\\useGRInc == 1"
            elif key in ['GRA', 'GRB', 'LWA', 'LWB', 'LWSA', 'LWSB']:
                return "\\useGR == 1"
            elif key in ['LRA%u' % lastIter, 'LRB%u' % lastIter, 'LRSA', 'LRSB']:
                return "\\usePLR == 1"
            elif key in ['LCC']:
                return "\\useLoop == 1"
            return ""

        def emit_instructions(instModule, macroGuard: str, category: str = ""):
            """Emit instructions from a module with optional macro guard.

            category: id_map key under which these instructions are being
            emitted (e.g. 'GRA', 'LRA0', 'PackB1', 'SYNC'). Used to populate
            tag_by_origin_id for the macro walker. Empty string means do not
            tag (caller did its own tagging).
            """
            if instModule is not None:
                for inst in instModule.flatitems():
                    if category:
                        tag_by_origin_id[id(inst)] = category
                    if isinstance(inst, SWaitCnt):
                        nllvmcntHandling(inst, opt1.nglshift, opt1.nllshift)
                    else:
                        if macroGuard:
                            macro.add(ValueIf(macroGuard))
                        macro.add(inst)
                        if macroGuard:
                            macro.add(ValueEndif(comment="EndIf %s" % macroGuard))

        for k, ts in ToSched.items():
            macroGuard = get_macro_guard(k)

            if len(ts) == 1:
                emit_instructions(ts[0], macroGuard, category=k)
            elif len(ts) == numCodePath:
                # Multi codepath - emit inside ID conditionals
                for codepath in range(numCodePath):
                    if codepath == 0:
                        macro.add(ValueIf("\\ID == %u" % codepath))
                    else:
                        macro.add(ValueElseIf("\\ID == %u" % codepath))
                    emit_instructions(ts[codepath], macroGuard, category=k)
                macro.add(ValueEndif(comment="EndIf \\ID checks"))
            else:
                raise ValueError(f"Invalid number of instructions for {k}: {len(ts)}")

    module.add(macro)

    # Stash the inputs needed to expand the CMS-side FourPartCapture; the
    # actual expansion is deferred until the default-side capture is
    # available (see KernelWriter.kernelBody, where ctx.default is built
    # from main + ctx.default_n_gl/_n_ll, then the deferred CMS expansion
    # runs and mirrors that body shape by construction).
    #
    # Why deferred: customMainLoopSchedule runs before noLoadLoop populates
    # ctx.default_n_gl / ctx.default_n_ll, so the default-side capture
    # doesn't exist yet. Building the CMS capture here would require
    # re-deriving body presence from kernel config (PGR/SuppressNoLoadLoop)
    # — exactly the predicate-shaped duplication that rocm-libraries-dj1g
    # eliminated. By deferring, the CMS expander observes the default-side
    # capture as the single source of truth.
    #
    # The macro walker reads tag_by_origin_id (populated above) and falls
    # back to isinstance checks against sync_class / snop_class /
    # mfma_classes for instructions added directly to the macro (MFMAs at
    # mfmaCode[miIndex], deepcopied SWaitCnts from nllvmcntHandling).
    mfma_classes = (MFMAInstruction, SMFMAInstruction)
    try:
        from rocisa.instruction import MXMFMAInstruction
        mfma_classes = mfma_classes + (MXMFMAInstruction,)
    except ImportError:
        pass
    writer._pending_cms_capture_inputs = scap.CmsCaptureInputs(
        macro=macro,
        num_codepaths=numCodePath,
        tag_by_origin_id=tag_by_origin_id,
        source_module_by_origin_id=source_module_by_origin_id,
        sync_class=SWaitCnt,
        snop_class=SNop,
        mfma_classes=mfma_classes,
        # Same numMfmaPerIter the writer used during emission. Both
        # default-side and CMS-side captures need the same value so
        # build_dataflow_graph derives matching MFMA-subiter classifications.
        # (Upstream Tensile naming uses "Iter" but this is the inner unroll
        # subiteration count.)
        num_mfma_per_subiter=getattr(writer.states, 'numMfmaPerIter', 0),
        # `regset_stream` is populated at the deferred-expansion site in
        # KernelWriter.kernelBody (after all RegSet directives have been
        # emitted by `_loopBody` / `_noLoadLoopBody`). Leaving it empty
        # here would mean only the macro-vgpr RegSets emitted before
        # customMainLoopSchedule are visible, missing body-level ones
        # like LocalReadAddrA / GlobalReadOffsetA. See rocm-libraries-bb34
        # and INTRA_GRAPH_7A_REGSET_INVESTIGATION.md for the timing audit.
    )
    writer._last_opt1_for_capture = opt1_for_capture

    return module, numCodePath


def hasCustomSchedule(kernel):
    """
    Trampoline function that checks if a custom schedule is available.
    Iterates through registered schedule functions and returns the first match.

    Validator-coverage note: a False return here causes Solution.py to set
    UseCustomMainLoopSchedule=0, which causes the kernel to be built via the
    non-CMS path. The CMS validator (Tensile/Components/CMSValidator.py) is
    only invoked on CMS=1 kernels — solutions that fall through this check
    are NOT validated by the dataflow-graph machinery. A missing registration
    is therefore a silent loss of validator coverage, not just a missing
    optimization.
    """
    if not kernel["UseCustomMainLoopSchedule"]:
        return False, None
    if not kernel["EnableMatrixInstruction"]:
        return False, None
    kernel_isa = tuple(kernel["ISA"]) if not isinstance(kernel["ISA"], tuple) else kernel["ISA"]
    if kernel_isa not in (IsaVersion(9,5,0), IsaVersion(11,5,1)):
        return False, None
    if isMixed(kernel):
        return False, None
    useLDSTr = kernel["LDSTrInst"]
    TLDS = kernel["TransposeLDS"]
    for schedule_func in _SCHEDULE_REGISTRY:
        status, schedule = schedule_func(kernel, useLDSTr, TLDS)
        if status == ScheduleMatchStatus.FOUND:
            return True, schedule
        elif status == ScheduleMatchStatus.UNSUPPORTED_VARIANT:
            # Criteria matched but variant unsupported - stop searching
            return False, None
        # status == NO_MATCH: continue to next schedule

    return False, None


def query_cms_kernels(dtype: Optional[str] = None, layout: Optional[str] = None) -> list[dict]:
    """Query for available CMS kernels matching the given data type and/or layout.

    This function searches the CMS kernel registry and returns the minimum
    combination of parameters needed for each matching CMS kernel.

    Args:
        dtype:  Data type filter (case-insensitive).
                Accepted values: "16bit", "8bit", "TF32", or None for all.
        layout: Layout / transpose e.g. ("TN", "NT", "NN", "TT", or None for all)


    Returns:
        A list of dicts, each containing the minimum parameter combination
        needed for a matching CMS kernel. Each dict includes the minimal parameters/values combinations needed for using each CMS kernel.

    """
    results = []
    for info in _SCHEDULE_METADATA:
        if info.matches(dtype=dtype, layout=layout):
            results.append(asdict(info))
    return results


def get_cms_kernel_info_objects(dtype: Optional[str] = None, layout: Optional[str] = None) -> list[CMSKernelInfo]:
    """Query for available CMS kernels and return CMSKernelInfo objects.

    Same filtering as :func:`query_cms_kernels` but returns the raw
    ``CMSKernelInfo`` dataclass instances instead of dicts.

    Args:
        dtype:  Data type filter (case-insensitive), or None for all.
        layout: Layout filter (case-insensitive), or None for all.

    Returns:
        A list of CMSKernelInfo objects matching the filters.
    """
    return [info for info in _SCHEDULE_METADATA if info.matches(dtype=dtype, layout=layout)]


def get_available_dtypes() -> set[str]:
    """Return set of all data type strings that have at least one CMS kernel."""
    return {info.dtype for info in _SCHEDULE_METADATA}


def get_available_layouts(dtype: Optional[str] = None) -> set[str]:
    """Return a set of all layout strings available for the given data type.

    Args:
        dtype: Optional data type filter, or None for all data types.

    Returns:
        Sorted list of unique layout strings (e.g. ["NN", "NT", "TN", "TT"]).
    """
    def as_str(transpose: bool) -> str:
        return "T" if transpose else "N"

    layouts: set[str] = set()
    for info in _SCHEDULE_METADATA:
        if dtype is None or info.dtype.lower() == dtype.lower():
            layouts.add(as_str(info.TransposeA) + as_str(info.TransposeB))
    return layouts


class _ProbeDataType:
    """Minimal DataType stub for layout probing at registration time."""
    def isHalf(self): return False
    def isBFloat16(self): return False
    def isInt8(self): return False
    def is8bitFloat(self): return False
    def numBytes(self): return 2


class RegisterSchedule:
    """
    Decorator that registers a schedule function with its matching criteria.
    The function is wrapped with logic that checks if the kernel matches the criteria.
    Supported layouts are auto-detected by probing the inner function at registration time.

    Usage:
        @RegisterSchedule(
            tile_config=TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2]
        )
        def _get_schedule_256x96x64_16bit(kernel, useLDSTr, TLDS):
            ...
    """

    def __init__(self, tile_config: TileConfig, dtype_predicate: Callable, vector_widths: list[int], matrix_inst: list[int], mfma_wave_group: list[int]):
        """
        Initialize the registration decorator with matching criteria.

        Args:
            tile_config:        TileConfig object
            dtype_predicate:    Callable that takes kernel and returns True if dtype matches
            vector_widths:      List of [GRVWA, GRVWB, LRVW]
            matrix_inst:        List [M, N, K, B] for MI
            mfma_wave_group:    List [rows, cols] for MIWG
        """
        self.tile_config = tile_config
        self.dtype_predicate = dtype_predicate
        self.vector_widths = vector_widths
        self.matrix_inst = matrix_inst
        self.mfma_wave_group = mfma_wave_group

    def _make_probe_kernel(self, transA: bool, transB: bool, useLDSTr: bool, TLDS: int, vectorWidthA: int, vectorWidthB: int) -> dict:
        """Build a synthetic kernel dict for probing layout support."""
        tc = self.tile_config
        mi = self.matrix_inst
        miwg = self.mfma_wave_group
        probe_dtype = _ProbeDataType()
        return {
            "ProblemType": {
                "DataType": probe_dtype,
                "DataTypeA": probe_dtype,
                "DataTypeB": probe_dtype,
                "TransposeA": transA,
                "TransposeB": transB,
            },
            "MacroTile0": tc.macro_tile_size_0,
            "MacroTile1": tc.macro_tile_size_1,
            "DepthU": tc.depth_u,
            "PrefetchGlobalRead": tc.prefetch_global_read,
            "PrefetchLocalRead": tc.prefetch_local_read,
            "DirectToLds": tc.direct_to_lds,
            "WaveSeparateGlobalReadA": tc.wave_separate_global_read_a,
            "WaveSeparateGlobalReadB": tc.wave_separate_global_read_b,
            "GlobalReadVectorWidthA": self.vector_widths[0],
            "GlobalReadVectorWidthB": self.vector_widths[1],
            "VectorWidthA": vectorWidthA,
            "VectorWidthB": vectorWidthB,
            "LocalReadVectorWidth": self.vector_widths[2],
            "MatrixInstruction": list(self.matrix_inst),
            "MIWaveGroup": list(self.mfma_wave_group),
            "LDSTrInst": useLDSTr,
            "TransposeLDS": TLDS,
            "MIWaveTileA": tc.macro_tile_size_0 // (mi[0] * miwg[0]),
            "MIWaveTileB": tc.macro_tile_size_1 // (mi[1] * miwg[1]),
            # Standard flags that inner functions may read/write
            "UseCustomMainLoopSchedule": True,
            "EnableMatrixInstruction": True,
            "UnrollLoopSwapGlobalReadOrder": False,
            "ISA": IsaVersion(*tc.isa),
            "WavefrontSize": tc.wavefront_size,
            "Use64bShadowLimit": 1,
            "ForceUnrollSubIter": False,
            "SwapGlobalReadOrder": False,
            "UsePLRPack": False,
            "UseF32XEmulation": False,
            "UseDirect32XEmulation": False,
            "MfmaInitCVgprs": False,
        }

    def _detect_supported_layouts(self, func: Callable) -> list[Tuple[bool, bool, bool, int]]:
        """Probe the inner function to discover which layouts it actually handles."""
        def as_str(transpose: bool) -> str:
            return "T" if transpose else "N"

        valid_vector_widths = [1, 2, 3, 4, 6, 8]
        detected = set()
        for transA, transB in product([True, False], repeat=2):
            for useLDSTr, TLDS in product([True, False], [1, 0]):
                for vwA, vwB in product(valid_vector_widths, repeat=2):
                    probe = self._make_probe_kernel(transA, transB, useLDSTr, TLDS, vwA, vwB)
                    try:
                        found, _ = func(probe, useLDSTr, TLDS)
                        if found:
                            detected_info_tuple = (transA, transB, useLDSTr, TLDS)
                            detected.add(detected_info_tuple)
                    except (ValueError, KeyError) as e:
                        layout = as_str(transA) + as_str(transB)
                        printWarning(
                            f"Layout probe failed for func '{func.__name__}' "
                            f"with layout={layout}, useLDSTr={useLDSTr}, TLDS={TLDS}, "
                            f"VectorWidthA={vwA}, VectorWidthB={vwB}\n"
                            f"  Kernel: {probe['MacroTile0']}x{probe['MacroTile1']}x{probe['DepthU']} {layout}\n"
                            f"  Error: {e}"
                        )
                        continue

        return list(detected)

    def __call__(self, func: Callable) -> Callable:
        """Wrap the function with matching logic and register it."""
        def wrapped_func(kernel: dict, useLDSTr: bool, TLDS: int) -> tuple[ScheduleMatchStatus, Optional[ScheduleInfo]]:
            # TODO: Currently ULSGRO not checked for in CMS, disabled for now
            if kernel["UnrollLoopSwapGlobalReadOrder"]:
                return ScheduleMatchStatus.NO_MATCH, None

            if not self.dtype_predicate(kernel):
                return ScheduleMatchStatus.NO_MATCH, None

            MT0, MT1, DU = kernel["MacroTile0"], kernel["MacroTile1"], kernel["DepthU"]
            PGR, PLR, DTL, DPLB = kernel["PrefetchGlobalRead"], kernel["PrefetchLocalRead"], kernel["DirectToLds"], kernel["DtlPlusLdsBuf"]
            WSGRA, WSGRB = kernel["WaveSeparateGlobalReadA"], kernel["WaveSeparateGlobalReadB"]
            kernel_isa = tuple(kernel["ISA"]) if hasattr(kernel["ISA"], '__iter__') else (kernel["ISA"],)
            kernel_tile_config = TileConfig(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB,
                                            isa=kernel_isa, wavefront_size=kernel["WavefrontSize"])
            if self.tile_config != kernel_tile_config:
                return ScheduleMatchStatus.NO_MATCH, None

            GRVWA, GRVWB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
            LRVWA, LRVWB = kernel["LocalReadVectorWidthA"], kernel["LocalReadVectorWidthB"]
            kernel_vector_widths = [GRVWA, GRVWB, LRVWA, LRVWB]
            # WA: if need to support different LRVW for A and B, add a new parameter to vector_widths
            extended_vector_widths = self.vector_widths + [self.vector_widths[2]]
            if extended_vector_widths != kernel_vector_widths:
                return ScheduleMatchStatus.NO_MATCH, None

            if self.matrix_inst != kernel["MatrixInstruction"]:
                return ScheduleMatchStatus.NO_MATCH, None

            if self.mfma_wave_group != kernel["MIWaveGroup"]:
                return ScheduleMatchStatus.NO_MATCH, None

            # All wrapper criteria matched - call inner function
            match, schedule = func(kernel, useLDSTr, TLDS)

            if match:
                return ScheduleMatchStatus.FOUND, schedule
            # Inner function returned False - variant unsupported, stop searching
            return ScheduleMatchStatus.UNSUPPORTED_VARIANT, None

        _SCHEDULE_REGISTRY.append(wrapped_func)

        # Auto-detect supported layouts by probing the inner function
        detected_infos = self._detect_supported_layouts(func)

        # Store metadata for query API
        dtype_name = _DTYPE_PREDICATE_NAMES.get(self.dtype_predicate, str(self.dtype_predicate))
        tc = self.tile_config
        for detected_info in detected_infos:
            _transA, _transB, _useLDSTr, _TLDS = detected_info
            _SCHEDULE_METADATA.append(CMSKernelInfo(
                name=func.__name__,
                dtype=dtype_name,
                TransposeA=_transA,
                TransposeB=_transB,
                MacroTile0=tc.macro_tile_size_0,
                MacroTile1=tc.macro_tile_size_1,
                DepthU=tc.depth_u,
                PrefetchGlobalRead=tc.prefetch_global_read,
                PrefetchLocalRead=tc.prefetch_local_read,
                DirectToLds=tc.direct_to_lds,
                DtlPlusLdsBuf=tc.dtl_plus_lds_buf,
                WaveSeparateGlobalReadA=tc.wave_separate_global_read_a,
                WaveSeparateGlobalReadB=tc.wave_separate_global_read_b,
                GlobalReadVectorWidthA=self.vector_widths[0],
                GlobalReadVectorWidthB=self.vector_widths[1],
                LocalReadVectorWidth=self.vector_widths[2],
                MatrixInstruction=list(self.matrix_inst),
                MIWaveGroup=list(self.mfma_wave_group),
                LDSTrInst=_useLDSTr,
                TransposeLDS=_TLDS,
            ))

        # Return original function unchanged (so it can still be called directly)
        return func
