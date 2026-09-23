import pytest
pytestmark = pytest.mark.unit

def test_state_keys_present_after_eval():
    from Tensile.SolutionStructs.segment_interleave import evaluate
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    s = _vw8_state()
    res = evaluate(s)
    # Production stores only the offsets in state; the applied bool resolves into
    # LDSSegmentInterleave (0/1), so emit sites gate on that (== 1) rather than a separate key.
    s["LDSSegInterleaveOffsets"] = res["offsets"]
    assert res["applicable"] is True
    assert s["LDSSegInterleaveOffsets"]["ldsBaseB"] == 33024


def _resolve_lsi(state):
    """Mirror Solution.py's LDSSegmentInterleave resolution (~4851-4907), including the deferred
    re-eval when the ONLY blocker is an unresolved 1LDSBuffer(-1 -> 0). The oracle's 1LDSBuffer gate
    short-circuits before other gates, so its skip can mask the real decision until 1LDSBuffer is
    resolved; production defers the resolve/reject and re-runs the oracle then. Returns
    (resolved_LDSSegmentInterleave, rejected)."""
    from Tensile.SolutionStructs.segment_interleave import evaluate
    _lsiRequested = state.get("LDSSegmentInterleave", -1)
    _lsiOneLdsBufAtEval = state.get("1LDSBuffer", 0)
    res = evaluate(state)
    _lsiBufWillResolve = _lsiOneLdsBufAtEval == -1 and res["reason"] == "needs 1LDSBuffer==0"
    if not _lsiBufWillResolve:
        # 1 = applied (split or bcontig alike), 0 = not.
        return (1 if res["applicable"] else 0, _lsiRequested == 1 and not res["applicable"])
    # Deferred: 1LDSBuffer resolves to 0 (the case that suppressed the reject); LDSSegmentInterleave
    # is kept at _lsiRequested (not forced to 0) so the re-eval oracle does not short-circuit on mode==0.
    s2 = dict(state); s2["1LDSBuffer"] = 0
    res2 = evaluate(s2)
    if res2["applicable"] and not res2["aligned"]:
        return 1, False                       # tight/bcontig are pure reorders -> size-safe to apply post-hoc
    return 0, (_lsiRequested == 1)            # aligned (LDS unreserved) or genuinely inapplicable -> reject if forced


def test_deferred_1ldsbuffer_auto_applies_tight():
    # auto(-1) + 1LDSBuffer=-1, tight-eligible: was silently baseline before the deferred re-eval;
    # now the free tight branch is applied once 1LDSBuffer resolves to 0.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(**{"1LDSBuffer": -1, "LDSSegmentInterleave": -1})) == (1, False)


def test_deferred_1ldsbuffer_forceon_applies_tight():
    # force-on + 1LDSBuffer=-1, tight-eligible: applies (not silently downgraded to baseline).
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(**{"1LDSBuffer": -1, "LDSSegmentInterleave": 1})) == (1, False)


def test_deferred_1ldsbuffer_forceon_aligned_rejects():
    # aligned grows LDS never reserved before 1LDSBuffer resolved; a forced aligned request rejects
    # rather than silently running as baseline.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2,
                                   **{"1LDSBuffer": -1, "LDSSegmentInterleave": 1})) == (0, True)


def test_deferred_1ldsbuffer_forceon_genuine_conflict_rejects():
    # A real disqualifier (fine VW) masked by the 1LDSBuffer short-circuit at first eval is caught
    # by the re-eval and rejected with the real reason -- not accepted as baseline.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(VectorWidthA=4,
                                   **{"1LDSBuffer": -1, "LDSSegmentInterleave": 1})) == (0, True)


def test_resolved_1ldsbuffer_forceon_conflict_still_rejects():
    # 1LDSBuffer already resolved (==1) is the non-deferred path: force-on that cannot apply rejects.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(**{"1LDSBuffer": 1, "LDSSegmentInterleave": 1})) == (0, True)
    assert _resolve_lsi(_vw8_state(VectorWidthA=4, LDSSegmentInterleave=1)) == (0, True)


def test_unrelated_disqualifier_rejects_even_with_unresolved_buffer():
    # A disqualifier that fires before the 1LDSBuffer short-circuit rejects (does not defer).
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(**{"UseSubtileImpl": 1, "1LDSBuffer": -1,
                                      "LDSSegmentInterleave": 1})) == (0, True)

def test_bcontig_resolves_to_1():
    # bcontig (unsplittable B) applies and resolves to 1, like split.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(MacroTile1=224, VectorWidthB=1)) == (1, False)


def test_deferred_1ldsbuffer_bcontig_resolves_to_1():
    # Same bcontig case, blocked only by 1LDSBuffer(-1): still resolves to 1 after the deferred re-eval.
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    assert _resolve_lsi(_vw8_state(MacroTile1=224, VectorWidthB=1, **{"1LDSBuffer": -1})) == (1, False)
