"""Mutation-killing tests for
``Solution.assignProblemIndependentDerivedParameters``.

Every test seeds a real, fully-derived solution ``_state`` (the ``real_state``
fixture), resets the ``AssignedProblemIndependentDerivedParameters`` flag, drives
one branch of the function, and asserts the exact resulting key set and the exact
values of every key the branch writes. Because the input state already contains
all output keys with their correct values, a mutant that (a) writes to a wrong
key adds a bogus key -> the key-set assertion fails, or (b) writes a wrong value
overwrites the correct value -> the value assertion fails.

Rejection-reason string mutants are killed via ``capsys``: with
``printRejectionReason=True`` and a real ``SolutionIndex`` present, ``reject``
prints the reason and then raises, so the exact message text is observable.
"""

import copy
import importlib

import pytest

import rocisa

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

APIDP = S.Solution.assignProblemIndependentDerivedParameters

def _reset(state):
    state["AssignedProblemIndependentDerivedParameters"] = False
    return state

def _gfx(isa_info_map, mm):
    for k in isa_info_map:
        if tuple(k)[:2] == mm:
            return k
    raise AssertionError("no isa for %r" % (mm,))

def _expect_reject(state, isa_info_map, capsys, needles):
    """Run with printRejectionReason=True; reject must print `needles` then raise."""
    with pytest.raises(Exception):
        APIDP(state, True, isa_info_map)
    out = capsys.readouterr().out
    for n in needles:
        assert n in out, "missing %r in reject output %r" % (n, out)
    return out

_BASE_VALS = {
    "AssignedProblemIndependentDerivedParameters": True,
    "Valid": True,
    "_ScheduleIterAlg": 3,
    "_StinkyTofuOptLevel": 0,
    "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": 16,
    "MatrixInstB": 1, "MatrixInstBM": 1, "MatrixInstBN": 1,
    "LocalSplitU": 1, "NumWaveSplitK": 1,
    "MIOutputVectorWidth": 4, "MIRegPerOut": 1,
    "ThreadTile0": 4, "ThreadTile1": 1,
    "SubGroup0": 8, "SubGroup1": 32,
    "NumThreads": 256, "NumWaves": 4,
    "MacroTile0": 32, "MacroTile1": 32,
    "MIInputPerThreadA": 4, "MIInputPerThreadB": 4,
    "AssertSummationElementMultiple": 1,
}
_BASE_FALSE = [
    "UseDotInstruction", "NonDTLTailLoopA", "NonDTLTailLoopB",
    "NonDTLTailLoopMetadata", "DirectToLdsA", "DirectToLdsB",
    "reorderGRInstForDTVA", "reorderGRInstForDTVB",
    "UseDot2F32XEmulation", "UseMFMAF32XEmulation", "UseDirect32XEmulation",
    "MfmaInitCVgprs", "UseSubtileImpl", "Multicast", "ClusterBarrier",
    "UseDualFMAC",
]
_BASE_TRUE = ["tailLoopOptA", "tailLoopOptB"]

def test_base_happy_key_set_is_invariant(real_state, isa_info_map):
    st = _reset(real_state)
    before = set(st)
    APIDP(st, False, isa_info_map)

    assert set(st) == before

def test_base_happy_scalar_values(real_state, isa_info_map):
    st = _reset(real_state)
    APIDP(st, False, isa_info_map)
    for k, v in _BASE_VALS.items():
        assert st[k] == v, k

def test_base_happy_boolean_values(real_state, isa_info_map):
    st = _reset(real_state)
    APIDP(st, False, isa_info_map)
    for k in _BASE_FALSE:
        assert st[k] is False, k
    for k in _BASE_TRUE:
        assert st[k] is True, k

def test_early_return_leaves_state_untouched(real_state, isa_info_map):
    st = real_state
    st["AssignedProblemIndependentDerivedParameters"] = True
    st["MacroTile0"] = 999999
    APIDP(st, False, isa_info_map)

    assert st["MacroTile0"] == 999999
    assert st["AssignedProblemIndependentDerivedParameters"] is True

def test_flag_reset_to_false_then_true(real_state, isa_info_map):

    st = _reset(real_state)
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["AssignedProblemIndependentDerivedParameters"] is True
    assert set(st) == before

def test_valid_absent_is_set_true(real_state, isa_info_map):
    st = _reset(real_state)
    st.pop("Valid", None)
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["Valid"] is True
    assert set(st) == before | {"Valid"}

def test_valid_present_sentinel_not_overwritten(real_state, isa_info_map):

    st = _reset(real_state)
    st["Valid"] = "SENTINEL"
    APIDP(st, False, isa_info_map)
    assert st["Valid"] == "SENTINEL"

def test_sia4_backend_missing_rejects(real_state, isa_info_map, monkeypatch, capsys):
    monkeypatch.setattr(rocisa, "hasStinkyTofuBackend", lambda: False)
    st = _reset(real_state)
    st["ScheduleIterAlg"] = 4
    _expect_reject(st, isa_info_map, capsys,
                   ["ScheduleIterAlg=4 requires the StinkyTofu backend",
                    "rocisa was built without it"])

def test_sia4_unsupported_isa_rejects(real_state, isa_info_map, capsys):

    st = _reset(real_state)
    st["ScheduleIterAlg"] = 4
    _expect_reject(st, isa_info_map, capsys,
                   ["ScheduleIterAlg=4 is not supported for",
                    "no StinkyTofu backend for this architecture.",
                    "Supported: ["])

def test_sia4_supported_sets_opt_level(real_state, isa_info_map, monkeypatch):
    monkeypatch.setattr(rocisa, "isSupportedByStinkyTofu", lambda isa: True)
    st = _reset(real_state)
    st["ScheduleIterAlg"] = 4
    APIDP(st, False, isa_info_map)
    assert st["_ScheduleIterAlg"] == 0
    assert st["_StinkyTofuOptLevel"] == 3
    assert st["Valid"] is True

def test_sia_else_records_optlevel_zero(real_state, isa_info_map):

    st = _reset(real_state)
    st["ScheduleIterAlg"] = 3
    APIDP(st, False, isa_info_map)
    assert st["_ScheduleIterAlg"] == 3
    assert st["_StinkyTofuOptLevel"] == 0

def test_reject_general_batched_requires_batched(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["ProblemType"]["StridedBatched"] = False
    st["ProblemType"]["Batched"] = False
    _expect_reject(st, isa_info_map, capsys,
                   ["General Batched GEMM only support Batched Problem"])

def test_reject_general_batched_requires_gemm(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["ProblemType"]["StridedBatched"] = False
    st["ProblemType"]["Batched"] = True
    st["ProblemType"]["OperationType"] = "NOTGEMM"
    _expect_reject(st, isa_info_map, capsys,
                   ["General Batched GEMM only support GEMM OperationType"])

def test_emi_autodetect_true_from_mi_keys(real_state, isa_info_map):
    st = _reset(real_state)
    st["EnableMatrixInstruction"] = None
    APIDP(st, False, isa_info_map)
    assert st["MatrixInstM"] == 16
    assert st["ThreadTile0"] == 4 and st["ThreadTile1"] == 1
    assert st["SubGroup0"] == 8 and st["SubGroup1"] == 32
    assert st["NumThreads"] == 256

def test_emi_autodetect_false_from_workgroup_keys(real_state, isa_info_map):
    st = _reset(real_state)
    st["EnableMatrixInstruction"] = None
    st["MIBlock"] = [1, 2, 3]
    APIDP(st, False, isa_info_map)
    assert st["ThreadTile0"] == 1 and st["ThreadTile1"] == 1
    assert st["SubGroup0"] == 32 and st["SubGroup1"] == 8
    assert st["NumThreads"] == 256

def test_emi_undetermined_rejects(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["EnableMatrixInstruction"] = None
    st["MIBlock"] = [1, 2]
    st["WorkGroup"] = [1, 1]
    _expect_reject(st, isa_info_map, capsys,
                   ["EnableMatrixInstruction undetermined"])

def test_matrix_inst_m4_branch(real_state, isa_info_map):
    st = _reset(real_state)
    st["MIBlock"] = [4, 16, 16, 1, 1, 1]
    APIDP(st, False, isa_info_map)
    assert st["MatrixInstM"] == 4
    assert st["ThreadTile0"] == 4 and st["ThreadTile1"] == 1
    assert st["SubGroup0"] == 2 and st["SubGroup1"] == 32
    assert st["NumThreads"] == 64
    assert st["MacroTile0"] == 8 and st["MacroTile1"] == 32
    assert st["Valid"] is True

def test_mi_input_per_thread_defaulting(real_state, isa_info_map):

    st = _reset(real_state)
    st.pop("MIInputPerThreadA", None)
    st.pop("MIInputPerThreadB", None)
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["MIInputPerThreadA"] == 4
    assert st["MIInputPerThreadB"] == 4
    assert set(st) == before | {"MIInputPerThreadA", "MIInputPerThreadB"}

def test_emi_false_branch_values(real_state, isa_info_map):
    st = _reset(real_state)
    st["EnableMatrixInstruction"] = False
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["ThreadTile0"] == 1 and st["ThreadTile1"] == 1
    assert st["SubGroup0"] == 32 and st["SubGroup1"] == 8
    assert st["LocalSplitU"] == 1 and st["NumWaveSplitK"] == 1
    assert st["MIWaveGroup"] == [0, 8]
    assert st["NumThreads"] == 256
    assert st["MacroTile0"] == 32 and st["MacroTile1"] == 8
    assert st["UseDotInstruction"] is True
    assert st["NumDotElements"] == 2
    assert set(st) == before | {"NumDotElements"}

def test_emi_false_wavesplitk_split_mode(real_state, isa_info_map):

    st = _reset(real_state)
    st["EnableMatrixInstruction"] = False
    st["WaveSplitK"] = True
    APIDP(st, False, isa_info_map)
    assert st["LocalSplitU"] == 1
    assert st["NumWaveSplitK"] == 1
    assert st["Valid"] is True

def test_reject_numthreads_not_multiple_of_wavefront(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["EnableMatrixInstruction"] = False
    st["WorkGroup"] = [7, 8, 1]
    _expect_reject(st, isa_info_map, capsys,
                   ["size of WorkGroup", "should be multiple of WavefrontSize"])

def test_reject_macrotile_mismatch(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["MacroTile"] = [999, 999]
    _expect_reject(st, isa_info_map, capsys, ["MacroTile mismatch"])

def test_wavesplitk_without_dot_raises(real_state, isa_info_map):

    st = _reset(real_state)
    st["WaveSplitK"] = True
    with pytest.raises(Exception):
        APIDP(st, False, isa_info_map)

@pytest.mark.parametrize("dtl,ea,eb", [(1, True, True), (2, True, False), (3, False, True)])
def test_direct_to_lds_ab(real_state, isa_info_map, dtl, ea, eb):
    st = _reset(real_state)
    st["DirectToLds"] = dtl
    APIDP(st, False, isa_info_map)
    assert st["DirectToLdsA"] is ea
    assert st["DirectToLdsB"] is eb

def test_nondtl_tailloop_a(real_state, isa_info_map):

    st = _reset(real_state)
    st["DirectToLds"] = 2
    APIDP(st, False, isa_info_map)
    assert st["DirectToLdsA"] is True and st["DirectToLdsB"] is False
    assert st["NonDTLTailLoopA"] is True
    assert st["tailLoopOptA"] is False
    assert st["NonDTLTailLoopB"] is False
    assert st["tailLoopOptB"] is True

def test_nondtl_tailloop_b(real_state, isa_info_map):
    st = _reset(real_state)
    st["ProblemType"]["TLUB"] = False
    st["DirectToLds"] = 1
    APIDP(st, False, isa_info_map)
    assert st["DirectToLdsB"] is True
    assert st["NonDTLTailLoopB"] is True
    assert st["tailLoopOptB"] is False

def test_no_bufferload_disables_tailloopopt(real_state, isa_info_map):
    st = _reset(real_state)
    st["BufferLoad"] = False
    APIDP(st, False, isa_info_map)
    assert st["tailLoopOptA"] is False
    assert st["tailLoopOptB"] is False

def test_directtovgpra_disables_tailloopopt_a(real_state, isa_info_map):
    st = _reset(real_state)
    st["DirectToVgprA"] = True
    APIDP(st, False, isa_info_map)
    assert st["tailLoopOptA"] is False
    assert st["tailLoopOptB"] is True

def test_dot_instruction_disables_tailloopopt(real_state, isa_info_map):

    st = _reset(real_state)
    st["EnableMatrixInstruction"] = False
    APIDP(st, False, isa_info_map)
    assert st["tailLoopOptA"] is False
    assert st["tailLoopOptB"] is False

def test_mx_blocks_derivation(real_state, isa_info_map):
    st = _reset(real_state)
    st["ProblemType"]["MXBlockA"] = 32
    st["ProblemType"]["MXBlockB"] = 32
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["AssertSummationElementMultiple"] == 32
    assert st["tailLoopOptMXSA"] is False
    assert st["tailLoopOptMXSB"] is False
    assert st["NonDTLTailLoopMXSA"] is False
    assert st["NonDTLTailLoopMXSB"] is False
    assert st["tailLoopOptA"] is False
    assert st["tailLoopOptB"] is False
    assert set(st) == before | {
        "tailLoopOptMXSA", "tailLoopOptMXSB",
        "NonDTLTailLoopMXSA", "NonDTLTailLoopMXSB",
    }

def test_reorder_gr_dtv_sia3(real_state, isa_info_map):
    st = _reset(real_state)
    st["ScheduleIterAlg"] = 3
    APIDP(st, False, isa_info_map)
    assert st["reorderGRInstForDTVA"] is False
    assert st["reorderGRInstForDTVB"] is False

def test_reorder_gr_dtv_non_sia3(real_state, isa_info_map):
    st = _reset(real_state)
    st["ScheduleIterAlg"] = 1
    APIDP(st, False, isa_info_map)
    assert st["reorderGRInstForDTVA"] is False
    assert st["reorderGRInstForDTVB"] is False

def test_f32x_emulation(real_state, isa_info_map):
    st = _reset(real_state)
    st["UseF32XEmulation"] = True
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["UseF32XEmulation"] is True
    assert st["UseDirect32XEmulation"] is True
    assert st["UseDirect32XEmulationInterleaveTreg"] is False
    assert st["UseMFMAF32XEmulation"] is True
    assert st["UseDot2F32XEmulation"] is False
    assert set(st) == before | {"UseDirect32XEmulationInterleaveTreg"}

def test_f32x_emulation_off_defaults(real_state, isa_info_map):
    st = _reset(real_state)
    st["UseF32XEmulation"] = False
    APIDP(st, False, isa_info_map)
    assert st["UseF32XEmulation"] is False
    assert st["UseDirect32XEmulation"] is False
    assert st["UseMFMAF32XEmulation"] is False
    assert st["UseDot2F32XEmulation"] is False

def test_dual_fmac_disabled(real_state, isa_info_map):
    st = _reset(real_state)
    st["UseDualFMAC"] = True
    APIDP(st, False, isa_info_map)

    assert st["UseDualFMAC"] is False

def test_multicast_off_when_cluster_default(real_state, isa_info_map):
    st = _reset(real_state)
    APIDP(st, False, isa_info_map)
    assert st["Multicast"] is False
    assert st["ClusterBarrier"] is False

def test_multicast_on_when_clusterdim_nondefault(real_state, isa_info_map):
    st = _reset(real_state)
    st["ClusterDim"] = [2, 1]
    APIDP(st, False, isa_info_map)
    assert st["Multicast"] is True
    assert st["ClusterBarrier"] is False

def test_cluster_barrier_on_with_cap_and_tdm(real_state, isa_info_map, monkeypatch):
    st = _reset(real_state)
    st["ClusterDim"] = [2, 1]
    st["TDMInst"] = 1
    caps = isa_info_map[st["ISA"]].asmCaps
    monkeypatch.setitem(caps, "HasClusterBarrier", 1)
    APIDP(st, False, isa_info_map)
    assert st["Multicast"] is True
    assert st["ClusterBarrier"] is True

def _subtile_state(real_state, isa_info_map, **mods):
    st = _reset(real_state)
    st["ISA"] = _gfx(isa_info_map, (9, 5))
    st["UseSubtileImpl"] = True
    st.update(mods)
    return st

def test_subtile_happy_path(real_state, isa_info_map):
    st = _subtile_state(real_state, isa_info_map)
    before = set(st)
    APIDP(st, False, isa_info_map)
    assert st["UseSubtileImpl"] is True
    assert st["VectorWidthA"] == 1 and st["VectorWidthB"] == 1
    assert st["SourceSwap"] is False
    assert st["BufferStore"] is True
    assert st["Use64bShadowLimit"] is False
    assert st["Use64bShadowLimitMX"] is False
    assert st["_ABTilePairA"] == "AB_B16"
    assert st["_ABTilePairB"] == "AB_B16_TLU1"
    assert st["DepthU"] == 32
    assert st["InternalSupportParams"]["SupportUserGSU"] is False
    assert st["Valid"] is True
    assert set(st) == before | {"_ABTilePairA", "_ABTilePairB"}

def test_subtile_reject_depthu(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, DepthU=48)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 support only DepthU multiple of"])

def test_subtile_reject_pgr(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, PrefetchGlobalRead=3)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 requires PrefetchGlobalRead 0, 1 or 2, got 3"])

def test_subtile_reject_mi_shape(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, MIBlock=[32, 16, 16, 1, 1, 1])
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 requires MatrixInst 16x16"])

def test_subtile_reject_sia(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, ScheduleIterAlg=1)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 does not support ScheduleIterAlg"])

def test_subtile_reject_gsu(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, GlobalSplitU=2)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 with StreamK=0 requires GlobalSplitU=1",
                    "no GSU reduction support"])

def test_subtile_reject_debug_streamk(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map, DebugStreamK=1)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 does not support DebugStreamK (must be 0)"])

def test_subtile_reject_pap_pgr(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map,
                        PrefetchAcrossPersistent=1, PrefetchGlobalRead=1)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 PrefetchAcrossPersistent requires PrefetchGlobalRead=2"])

def test_subtile_reject_pap_dtv_mx(real_state, isa_info_map, capsys):
    st = _subtile_state(real_state, isa_info_map,
                        PrefetchAcrossPersistent=1, PrefetchGlobalRead=2,
                        DirectToVgprMXSA=True)
    _expect_reject(st, isa_info_map, capsys,
                   ["UseSubtileImpl=1 PrefetchAcrossPersistent not supported with "
                    "DirectToVgpr MX scale tensors"])

def test_gfx950_mx_requires_subtile(real_state, isa_info_map, capsys):
    st = _reset(real_state)
    st["ISA"] = _gfx(isa_info_map, (9, 5))
    st["UseSubtileImpl"] = False
    st["ProblemType"]["MXBlockA"] = 32
    _expect_reject(st, isa_info_map, capsys, ["gfx950 MX requires UseSubtileImpl"])
