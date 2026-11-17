# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover reject harvest -- general derivation rejection branches.

Companion to ``test_setcover_reject_char.py`` (Stream-K guards). Targets the
reachable-invalid rejection branches in ``Solution.assignDerivedParameters``
that the tuned/emit configs never select: BufferLoad=0 constraints, the
MultipleBufferSingleKernel epilogue guards, StoreRemap/SourceSwap conflicts,
UseCustomMainLoopSchedule limits, and a handful of standalone feature guards
(InterleaveAlpha, ConvertAfterDS, ActivationAlt, wavefront/source, AF0EM).

Each case takes a fully-derived base solution, applies parameter overrides that
a caller *can legitimately supply* but that the code is designed to reject,
resets the derivation flags, and re-runs derivation with
``printRejectionReason=False`` so the guard sets ``Valid=False`` instead of
raising. Every override is a user-settable tuning knob or ProblemType flag --
these are reachable-invalid states (category A), not pipeline-impossible states.

Trip-cases were verified line-by-line under ``sys.settrace`` before authoring
(work/mutcov-evidence/sol_reject_probe.py): every case flips at least one
currently-missing Solution.py line. The BufferLoad=0 guards are a chain of
``reject(); return`` statements, so each is reached by one config that clears
the earlier guards; the MBSK epilogue guards do not early-return, so a single
maximal-violation config trips several. The golden pins the ordered rejection
reasons together with deterministic reject-relevant state; every case asserts
``Valid is False`` since these configs are genuinely invalid.

Runs under global-state isolation (derivation mutates globalParameters /
validParameters) so it does not leak into other suites.
"""

import copy
import pytest

from codegen_harness import _isolated_globals  # shared isolation context
from reject_harness import apply_overrides, derive_with_rejections

pytestmark = pytest.mark.unit

# (case_id, base_label, {overrides}) -- overrides support dotted keys for
# nested ProblemType.* fields.
_TRIPS = {
    # standalone feature guards (one reject each)
    "interleave_alpha": ("gfx942_HSS", {"InterleaveAlpha": 1}),
    "activationalt": ("gfx942_HSS", {"ActivationAlt": True}),
    "convertafterds": ("gfx942_HSS", {"ConvertAfterDS": True}),
    "wf32_source": ("gfx942_HSS", {"WavefrontSize": 32, "KernelLanguage": "Source"}),
    "ucms_tailloop": ("gfx942_HSS", {"UseCustomMainLoopSchedule": 1, "TailloopInNll": True}),
    "af0em_nohpa": ("gfx942_HSS", {"ProblemType.HighPrecisionAccumulate": False, "AssertFree0ElementMultiple": 1}),
    "srmap_sourceswap": ("gfx942_HSS", {"StoreRemapVectorWidth": 4, "SourceSwap": True}),
    # BufferLoad=0 returning-reject chain (one config clears the earlier guards)
    "nobuf_pgr2": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 2}),
    "nobuf_dtva": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": True}),
    "nobuf_dtvb": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": False, "DirectToVgprB": True}),
    "nobuf_usebias": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": False, "DirectToVgprB": False}),
    "nobuf_sparse": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": False, "DirectToVgprB": False, "ProblemType.UseBias": 0, "ProblemType.Sparse": 1}),
    "nobuf_gg": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": False, "DirectToVgprB": False, "ProblemType.UseBias": 0, "ProblemType.Sparse": 0, "ProblemType.GroupedGemm": True}),
    "nobuf_subdword": ("gfx942_HSS", {"BufferLoad": False, "PrefetchGlobalRead": 1, "DirectToVgprA": False, "DirectToVgprB": False, "ProblemType.UseBias": 0, "ProblemType.Sparse": 0, "ProblemType.GroupedGemm": False}),
    # MBSK epilogue non-returning guards (combined maximal violation)
    "mbsk_maxviol": ("gfx942_BBS", {"ProblemType.UseScaleCD": True, "ProblemType.UseE": True, "ProblemType.BiasSrc": "A", "NumElementsPerBatchStore": 1}),
}

_KEYS = [
    "Valid", "BufferLoad", "PrefetchGlobalRead", "DirectToVgprA",
    "DirectToVgprB", "StoreRemapVectorWidth", "_GlobalAccumulation",
]


@pytest.mark.parametrize("case", sorted(_TRIPS.keys()))
def test_setcover_reject_other(case, reject_bases, isa_info_map, assembler, monkeypatch, snapshot):
    label, overrides = _TRIPS[case]
    base = reject_bases[label]
    rocm = assembler.rocm_version
    with _isolated_globals():
        s = copy.deepcopy(base)
        apply_overrides(s, overrides)
        out = derive_with_rejections(s, isa_info_map, rocm, monkeypatch, _KEYS)
    assert out.get("Valid") is False
    assert out["rejections"]
    assert out == snapshot
