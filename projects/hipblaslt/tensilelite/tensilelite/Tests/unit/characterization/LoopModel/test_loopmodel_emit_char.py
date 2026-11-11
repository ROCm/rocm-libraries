# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for ``Components.LoopModel.Emit``.

The module is the fork between GIR and the legacy scaffold: it decides which
GIR block each scaffold site emits, with which op kinds, and what the scaffold
accumulates alongside. It owns no codegen of its own -- every emit goes through
``girEmitStage`` -- so the tests stub that seam and the cached program, and
assert the sequence of (phase, kinds) the module asks for. That sequence is the
behavior; the instructions behind it belong to ``GirToRocisa``.
"""

import pytest

from tensilelite.Components.LoopModel import Emit


class _States:
    def __init__(self, numItersPLR=2, subTileIdx=0):
        self.numItersPLR = numItersPLR
        self.SubTileIdx = subTileIdx


class _Codes:
    def __init__(self, perIterLocalWrite=None):
        self.perIterLocalWrite = perIterLocalWrite or []


class _Writer:
    """Records the scaffold calls Emit makes around each GIR stage."""

    def __init__(self, numItersPLR=2, perIterLocalWrite=None):
        self.states = _States(numItersPLR)
        self.codes = _Codes(perIterLocalWrite)
        self.calls = []

    def openPrefetchGlobalRead2orMore(self, kernel, gen):
        self.calls.append(("open", gen))
        return f"open{gen}"

    def closePrefetchGlobalRead2orMore(self, kernel, tPA, tPB, idxPgr):
        self.calls.append(("close", idxPgr))
        return f"close{idxPgr}"


class _Module:
    """Stand-in for rocisa Module: records what was added, in order."""

    def __init__(self):
        self.added = []

    def add(self, item):
        self.added.append(item)
        return item

    def count(self):
        return len(self.added)


def _kernel(**over):
    kernel = {
        "UseLoopModel": True,
        "HalfPLR": 0,
        "PrefetchGlobalRead": 2,
        "LoopIters": 2,
        "ExpandPointerSwap": False,
        "numSubTiles": 2,
    }
    kernel.update(over)
    return kernel


@pytest.fixture
def seam(monkeypatch):
    """Replace the two seams and hand back the recorded emit list."""
    emitted = []

    def fakeEmit(writer, kernel, tPA, tPB, phase, *, internalPointerSwap=False, kinds=None):
        emitted.append((phase, kinds, internalPointerSwap))
        return f"emit:{phase}"

    monkeypatch.setattr(Emit, "girEmitStage", fakeEmit)
    return emitted


def _program(monkeypatch, meta=None, chain=("steady",)):
    prog = type("P", (), {"meta": meta or {}})()
    monkeypatch.setattr(Emit, "loopModelGirProgram", lambda writer, kernel: prog)
    monkeypatch.setattr(Emit, "steady_chain", lambda p: list(chain))
    return prog


# --- loopModelSteadyBlocks / loopModelSteadyPhase -------------------------


def test_steady_blocks_is_the_programs_chain(monkeypatch):
    _program(monkeypatch, chain=("a", "b"))
    assert Emit.loopModelSteadyBlocks(_Writer(), _kernel()) == ["a", "b"]


def test_steady_phase_falls_back_to_steady_on_an_empty_chain(monkeypatch):
    _program(monkeypatch, chain=())
    assert Emit.loopModelSteadyPhase(_Writer(), _kernel(), 0) == "steady"


@pytest.mark.parametrize("loopCopy", [0, 1, 7])
def test_steady_phase_serves_one_block_to_every_scaffold_copy(monkeypatch, loopCopy):
    """A single block means the copies differ only in scaffold-owned state."""
    _program(monkeypatch, chain=("only",))
    assert Emit.loopModelSteadyPhase(_Writer(), _kernel(), loopCopy) == "only"


@pytest.mark.parametrize("loopCopy, expected", [(0, "s0"), (1, "s1"), (2, "s2")])
def test_steady_phase_indexes_a_longer_chain_by_copy(monkeypatch, loopCopy, expected):
    _program(monkeypatch, chain=("s0", "s1", "s2"))
    assert Emit.loopModelSteadyPhase(_Writer(), _kernel(), loopCopy) == expected


@pytest.mark.parametrize("loopCopy", [-1, 3])
def test_steady_phase_rejects_a_copy_the_chain_has_no_block_for(monkeypatch, loopCopy):
    _program(monkeypatch, chain=("s0", "s1", "s2"))
    with pytest.raises(RuntimeError, match="has no GIR steady block"):
        Emit.loopModelSteadyPhase(_Writer(), _kernel(), loopCopy)


# --- _prologueGuardLadder -------------------------------------------------


def test_guard_ladder_emits_nothing_without_a_prefetch_guard(monkeypatch, seam):
    _program(monkeypatch, meta={})
    module = _Module()
    Emit._prologueGuardLadder(_Writer(), _kernel(), None, None, module)
    assert seam == [] and module.added == []


def test_guard_ladder_brackets_the_block_and_joins_after_every_close(monkeypatch, seam):
    _program(monkeypatch, meta={"prefetch_guard": {"gen": 1, "block": "g", "join": "j"}})
    writer = _Writer()
    module = _Module()

    Emit._prologueGuardLadder(writer, _kernel(PrefetchGlobalRead=2), None, None, module)

    assert [p for p, _k, _s in seam] == ["g", "j"]
    # One open, then one close per prefetch index 0..PGR inclusive.
    assert writer.calls == [("open", 1), ("close", 0), ("close", 1), ("close", 2)]
    assert module.added == ["open1", "emit:g", "close0", "close1", "close2", "emit:j"]


def test_guard_ladder_emits_the_skip_block_after_the_single_trip_close(monkeypatch, seam):
    _program(monkeypatch,
             meta={"prefetch_guard": {"gen": 1, "block": "g", "join": "j", "skip": "sk"}})
    module = _Module()

    Emit._prologueGuardLadder(_Writer(), _kernel(PrefetchGlobalRead=2), None, None, module)

    assert [p for p, _k, _s in seam] == ["g", "sk", "j"]
    assert module.added.index("emit:sk") == module.added.index("close1") + 1


# --- loopModelPrologueCopies ----------------------------------------------


def test_prologue_copies_moves_only_the_global_fill_under_half_plr(monkeypatch, seam):
    _program(monkeypatch, meta={})
    module = _Module()

    Emit.loopModelPrologueCopies(_Writer(), _kernel(HalfPLR=1), None, None, module)

    assert seam == [("prologue", Emit.PROLOGUE_MOVEMENT, False)]


@pytest.mark.parametrize(
    "kernel",
    [
        _kernel(HalfPLR=0),                       # every other shape keeps one site
        _kernel(HalfPLR=1, UseLoopModel=False),   # legacy path is untouched
    ],
)
def test_prologue_copies_is_inert_off_the_half_plr_path(monkeypatch, seam, kernel):
    _program(monkeypatch, meta={})
    module = _Module()
    Emit.loopModelPrologueCopies(_Writer(), kernel, None, None, module)
    assert seam == [] and module.added == []


# --- loopModelPrologue ----------------------------------------------------


def test_prologue_emits_the_whole_block_and_the_ladder_without_half_plr(monkeypatch, seam):
    _program(monkeypatch, meta={"prefetch_guard": {"gen": 1, "block": "g", "join": "j"}})
    writer = _Writer(numItersPLR=3)
    module, pack, packPre = _Module(), {}, {}

    Emit.loopModelPrologue(writer, _kernel(), None, None, module, pack, packPre)

    assert [p for p, _k, _s in seam] == ["prologue", "g", "j"]
    assert seam[0][1] is None, "no kind filter: the one site emits everything"
    assert sorted(pack) == [0, 1, 2] and sorted(packPre) == [0, 1, 2]
    assert all(m.count() == 0 for m in pack.values())


def test_prologue_emits_only_the_fill_when_half_plr_took_the_copies_early(monkeypatch, seam):
    _program(monkeypatch, meta={"prefetch_guard": {"gen": 1, "block": "g", "join": "j"}})
    module, pack, packPre = _Module(), {}, {}

    Emit.loopModelPrologue(_Writer(), _kernel(HalfPLR=1), None, None, module, pack, packPre)

    assert seam == [("prologue", Emit.PROLOGUE_FILL, False)], "the ladder went with the copies"


def test_prologue_advances_the_subtile_index_modulo_the_count(monkeypatch, seam):
    _program(monkeypatch, meta={})
    writer = _Writer()
    writer.states.SubTileIdx = 1
    Emit.loopModelPrologue(writer, _kernel(numSubTiles=2), None, None, _Module(), {}, {})
    assert writer.states.SubTileIdx == 0


# --- loopModelSteadyIter / loopModelDrainIter -----------------------------


def test_steady_iter_accumulates_the_scaffold_and_emits_at_the_last_substep(monkeypatch, seam):
    _program(monkeypatch, chain=("steady",))
    perLW = _Module()
    perLW.add("lw")
    writer = _Writer(perIterLocalWrite=[(None, _Module()), (None, perLW)])
    module, scaffold = _Module(), _Module()

    # u = 0: scaffold only, nothing emitted yet.
    Emit.loopModelSteadyIter(writer, _kernel(LoopIters=2), None, None, module, scaffold,
                             0, "waitLW0", "sync0")
    assert seam == [] and module.added == []
    assert scaffold.added == ["waitLW0", "sync0"]

    # u = 1: the last substep emits the block, then the accumulated scaffold.
    Emit.loopModelSteadyIter(writer, _kernel(LoopIters=2), None, None, module, scaffold,
                             1, "waitLW1", "sync1")
    assert seam == [("steady", None, False)]
    assert scaffold.added == ["waitLW0", "sync0", "waitLW1", "sync1", perLW]
    assert module.added == ["emit:steady", scaffold]


def test_steady_iter_forwards_expand_pointer_swap(monkeypatch, seam):
    _program(monkeypatch, chain=("steady",))
    Emit.loopModelSteadyIter(_Writer(), _kernel(LoopIters=1, ExpandPointerSwap=True),
                             None, None, _Module(), _Module(), 0, "w", "s")
    assert seam == [("steady", None, True)]


def test_steady_iter_skips_an_empty_per_iteration_local_write(monkeypatch, seam):
    _program(monkeypatch, chain=("steady",))
    writer = _Writer(perIterLocalWrite=[(None, _Module())])
    scaffold = _Module()
    Emit.loopModelSteadyIter(writer, _kernel(LoopIters=1), None, None, _Module(), scaffold,
                             0, "w", "s")
    assert scaffold.added == ["w", "s"]


@pytest.mark.parametrize("remainPgr, expected", [(0, "drain1"), (1, "drain0")])
def test_drain_iter_names_the_block_by_steps_remaining(monkeypatch, seam, remainPgr, expected):
    _program(monkeypatch)
    Emit.loopModelDrainIter(_Writer(), _kernel(LoopIters=1, PrefetchGlobalRead=2),
                            None, None, _Module(), _Module(), 0, "w", "s", remainPgr)
    assert seam == [(expected, None, False)]


def test_drain_iter_holds_the_scaffold_until_the_last_substep(monkeypatch, seam):
    _program(monkeypatch)
    module, scaffold = _Module(), _Module()
    Emit.loopModelDrainIter(_Writer(), _kernel(LoopIters=2), None, None, module, scaffold,
                            0, "w", "s", 0)
    assert seam == [] and module.added == []
    assert scaffold.added == ["w", "s"]


def test_drain_iter_tolerates_an_absent_per_iteration_local_write(monkeypatch, seam):
    _program(monkeypatch)
    writer = _Writer(perIterLocalWrite=[])
    scaffold = _Module()
    Emit.loopModelDrainIter(writer, _kernel(LoopIters=1), None, None, _Module(), scaffold,
                            0, "w", "s", 0)
    assert scaffold.added == ["w", "s"]
