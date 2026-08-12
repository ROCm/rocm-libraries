import importlib
import sys
import types

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

EXPECTED_MSG = (
    "UseSubtileImpl=1 hits the subtile GR K-partition bug on tensor A: "
    "loadRatioGR=2 with localSubtileGrid=(3, 2) (M-subtile count 3 is not a "
    "multiple of loadRatioGR and there is more than one K-partition)"
)

def _install_fake_kernel(monkeypatch, loadRatioGR, localSubtileGrid):
    """Inject a fake Tensile.Components.Subtile.Kernel so the function's lazy
    ``from ... import selectABGeometry, TileInfo`` resolves to controllable
    geometry instead of the real (deadlock-prone) Kernel module. This lets the
    test drive loadRatioGR / localSubtileGrid to exactly the values that do or
    do not trip _subtileGRKPartitionIsBuggy."""
    fake = types.ModuleType("Tensile.Components.Subtile.Kernel")

    def selectABGeometry(state, tc):
        return ("geom", tc)

    class TileInfo:
        def __init__(self, geometry, tc, _unused, state):
            self.loadRatioGR = loadRatioGR
            self.localSubtileGrid = localSubtileGrid

    fake.selectABGeometry = selectABGeometry
    fake.TileInfo = TileInfo
    monkeypatch.setitem(sys.modules, "Tensile.Components.Subtile.Kernel", fake)

def _install_reject_recorder(monkeypatch):
    """Replace the module-global reject with a recorder that captures the exact
    (state, printRejectionReason, *args) passed. Capturing the args is the only
    way to observe the rejection-reason string (real reject only prints it and
    would raise on a fully-derived state carrying a SolutionIndex)."""
    calls = []

    def rec(state, printReason=True, *args):
        calls.append((state, printReason, args))
        return True

    monkeypatch.setattr(S, "reject", rec)
    return calls

def test_returns_true_and_no_reject_when_subtile_disabled(monkeypatch):
    calls = _install_reject_recorder(monkeypatch)
    state = {"UseSubtileImpl": False, "ISA": [9, 5, 0]}
    assert S._validateSubtileGRKPartition(state, True) is True
    assert calls == []

def test_returns_true_and_no_reject_when_isa_not_gfx950(monkeypatch):
    calls = _install_reject_recorder(monkeypatch)
    _install_fake_kernel(monkeypatch, loadRatioGR=2, localSubtileGrid=(3, 2))
    state = {"UseSubtileImpl": True, "ISA": [9, 4, 2]}
    assert S._validateSubtileGRKPartition(state, True) is True
    assert calls == []

def test_returns_true_and_no_reject_when_not_buggy_on_gfx950(monkeypatch):
    calls = _install_reject_recorder(monkeypatch)
    _install_fake_kernel(monkeypatch, loadRatioGR=2, localSubtileGrid=(4, 2))
    state = {"UseSubtileImpl": True, "ISA": [9, 5, 0]}
    assert S._validateSubtileGRKPartition(state, True) is True
    assert calls == []

def test_rejects_and_returns_false_on_buggy_gfx950(monkeypatch):
    calls = _install_reject_recorder(monkeypatch)
    _install_fake_kernel(monkeypatch, loadRatioGR=2, localSubtileGrid=(3, 2))
    state = {"UseSubtileImpl": True, "ISA": [9, 5, 0]}

    rv = S._validateSubtileGRKPartition(state, True)

    assert rv is False
    assert len(calls) == 1
    captured_state, captured_print, captured_args = calls[0]
    assert captured_state is state
    assert captured_print is True
    assert len(captured_args) == 1
    assert captured_args[0] == EXPECTED_MSG

def test_isa_membership_requires_exact_tuple(monkeypatch):
    """ISA (9,5,0) must take the buggy path (rejects, returns False); any other
    ISA short circuits to True. Locks both the exact comparison tuple and the
    ``!=`` operator."""
    _install_fake_kernel(monkeypatch, loadRatioGR=2, localSubtileGrid=(3, 2))

    calls = _install_reject_recorder(monkeypatch)
    buggy = {"UseSubtileImpl": True, "ISA": [9, 5, 0]}
    assert S._validateSubtileGRKPartition(buggy, True) is False
    assert len(calls) == 1

    for isa in ([10, 5, 0], [9, 6, 0], [9, 5, 1]):
        calls2 = _install_reject_recorder(monkeypatch)
        state = {"UseSubtileImpl": True, "ISA": list(isa)}
        assert S._validateSubtileGRKPartition(state, True) is True
        assert calls2 == []
