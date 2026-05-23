# Unit tests for InstructionEmitter.emit_lr and emit_gr in the
# *mixed-datatype* regime (A and B have different element widths,
# e.g. one is F8 and the other is F4).
#
# These tests are intentionally hermetic:
#   * no GPU is required
#   * rocisa does NOT need to be initialised
#   * NO real LRPlacement / GRPlacement / SchedulerConfig / Module
#     classes are used — everything other than the function under
#     test is a duck-typed stub.
#
# This isolation means the test cannot be derailed by any unrelated
# change to internal data classes or by missing AMD toolchain on the
# test host. If the test fails, the *only* thing that can have changed
# is the per-tensor subtileShape[1] formula inside emit_lr / emit_gr.
import pytest
from types import SimpleNamespace
from unittest.mock import patch

# Only the public surface we are testing is imported from the project:
#   - TileInfo: builds the per-tensor geometry object
#   - AB_B4 / AB_B8: pre-baked geometry constants (F4 and F8 tile pairs)
#   - InstructionEmitter module (imported as IE so we can patch names in it)
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4, AB_B8
from Tensile.Components.Subtile import InstructionEmitter as IE


# ------------------------------------------------------------------------
# 1. FakeModule -- replaces rocisa.code.Module inside InstructionEmitter
#                  for the duration of the test (via patch.object below).
# ------------------------------------------------------------------------
# emit_lr / emit_gr call exactly three Module methods:
#     m = Module()
#     m.add(<something>)
#     return list(m.flatitems())
# so a 4-line stand-in is enough. We deliberately ignore None and []
# arguments to .add() so the recorders below can return whatever they want.
class FakeModule:
    def __init__(self):
        self._items = []

    def add(self, x):
        if x is not None and x != []:
            self._items.append(x)
        return x

    def flatitems(self):
        return list(self._items)


# ------------------------------------------------------------------------
# 2. Kernel-dict factory -- only the AB-relevant keys are read by TileInfo.
# ------------------------------------------------------------------------
# Values match the two mixed f4f8 yaml configuration:
#   MT 128x128, DepthU=256, MFMA 16x16x128, 2x2 wave group.
# TileInfo for AB tensors reads ONLY these kernel keys; it does *not*
# read DataTypeA/B (the bpe and subtileShape come from the geometry
# constant we pass in -- AB_B8 or AB_B4 -- not from the kernel dict).
def _kernel():
    return {
        "DepthU": 256,
        "_DepthUA": 256,
        "_DepthUB": 256,
        "MacroTileA": 128,
        "MacroTileB": 128,
        "MacroTile0": 128,
        "MacroTile1": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": {
            # numBytes() is the only method TileInfo ever calls on these.
            "DataTypeA": SimpleNamespace(numBytes=lambda: 1),
            "DataTypeB": SimpleNamespace(numBytes=lambda: 1),
            "ComputeDataType": SimpleNamespace(numBytes=lambda: 4),
        },
    }


# ------------------------------------------------------------------------
# 3. Duck-typed ReadGranularity (only .mn and .k are accessed)
#    and a stub SchedulerConfig (only .lrA, .lrB, .grA, .grB are accessed
#    by emit_lr / emit_gr for the A/B branch we exercise here).
# ------------------------------------------------------------------------
def _RG(mn, k):
    return SimpleNamespace(mn=mn, k=k)


def _make_emitter(geo_a, geo_b):
    """Build (emitter, tiA, tiB, numK) for the requested (A, B) geometry pair.
    geo_a / geo_b are project geometry constants (AB_B8 or AB_B4).
    The returned emitter has only the state needed by emit_lr / emit_gr.
    """
    kernel = _kernel()
    tiA = TileInfo(geo_a, "A", None, kernel)  # writer=None: no register alloc
    tiB = TileInfo(geo_b, "B", None, kernel)
    # numK = #K-tiles per DU iteration. With DU=256 and instK=128 it is 2
    # for both F8 and F4 (so A and B agree even when their subtileShape[1]
    # differs). We assert this so a future geometry change fails loudly.
    assert (
        tiA.localMMATileGrid[1] == tiB.localMMATileGrid[1]
    ), "Test setup requires A and B to share localMMATileGrid[1] (numK)"
    numK = tiA.localMMATileGrid[1]
    # Minimal config stub. emit_lr reads cfg.lrA.mn / cfg.lrA.k
    # (or lrB), emit_gr reads cfg.grA / cfg.grB.
    cfg = SimpleNamespace(
        lrA=_RG(mn=1, k=1),  # one ds_read per (tile,k) -- finest granularity
        lrB=_RG(mn=1, k=1),
        grA=_RG(mn=1, k=tiA.subtileShape[1]),  # one buffer_load per K-subtile
        grB=_RG(mn=1, k=tiB.subtileShape[1]),
    )
    # All these are unused by emit_lr / emit_gr's A/B branch -- placeholders.
    writer = SimpleNamespace()
    dtileInfo = SimpleNamespace()
    # vgprTiles* must be subscriptable by the integers in tile_map[tileId].
    # We use sentinel strings so any accidental cross-tensor reference is
    # visible in the recorded calls.
    vgprTilesA = [f"A_vt_{i}" for i in range(64)]
    vgprTilesB = [f"B_vt_{i}" for i in range(64)]
    emitter = IE.InstructionEmitter(
        writer,
        kernel,
        cfg,
        tiA,
        tiB,
        dtileInfo,
        vgprTilesA,
        vgprTilesB,
    )
    return emitter, tiA, tiB, numK


# ------------------------------------------------------------------------
# 4. Placement stubs. emit_lr / emit_gr read exactly these attributes:
#    placement.tensor                          -> 'A' or 'B'
#    placement.tiles.tileId_start / .tileId_end
#    placement.tiles.subIterK_start / .subIterK_end
#    placement.vgpr_tile_map[unroll_iter]      -> dict {tileId -> vgprTileId}
# That's it. SimpleNamespace covers all of them with zero dependency on
# the real LRPlacement / GRPlacement / MFMATileRange classes.
# ------------------------------------------------------------------------
def _lr_placement(tensor, numK):
    return SimpleNamespace(
        tensor=tensor,
        tiles=SimpleNamespace(
            tileId_start=0,
            tileId_end=1,
            subIterK_start=0,
            subIterK_end=numK,
        ),
        vgpr_tile_map=[{0: 0}],  # 1 unroll iter, 1 tileId -> vgprTileId 0
    )


def _gr_placement(tensor, numK):
    return SimpleNamespace(
        tensor=tensor,
        tiles=SimpleNamespace(
            tileId_start=0,
            tileId_end=1,
            subIterK_start=0,
            subIterK_end=numK,
        ),
    )


# ------------------------------------------------------------------------
# 5. Recorders -- callable objects that capture (tc, sId0, sId1, subIterK)
#    or (tc, sId0, sId1) and return None (FakeModule.add ignores None).
# ------------------------------------------------------------------------
class _LRRecorder:
    """Replaces emitSingleDsRead(tileInfo, sId0, sId1, subIterK, dstTile)."""

    def __init__(self):
        self.calls = []

    def __call__(self, tileInfo, sId0, sId1, subIterK, dstTile):
        self.calls.append((tileInfo.tc, sId0, sId1, subIterK))
        return None


class _GRRecorder:
    """Replaces emitSingleBufferLoad(tileInfo, kernel, sId0, sId1)."""

    def __init__(self):
        self.calls = []

    def __call__(self, tileInfo, kernel, sId0, sId1):
        self.calls.append((tileInfo.tc, sId0, sId1))
        return None


# ------------------------------------------------------------------------
# 6. Expected outputs computed straight from each tensor's OWN
#    subtileShape[1].
# ------------------------------------------------------------------------
def _expected_lr(ti, numK):
    sk = ti.subtileShape[1]
    return [(ti.tc, 0, k // sk, k % sk) for k in range(numK)]


def _expected_gr(ti, numK):
    sk = ti.subtileShape[1]
    # GR loop steps by grGran.k = ti.subtileShape[1], so it only fires for
    # K values that start a new K-subtile: k = 0, sk, 2*sk, ...
    return [(ti.tc, 0, k // sk) for k in range(0, numK, sk)]


# ------------------------------------------------------------------------
# 7. Drivers: patch IE.Module + IE.emit* primitives, then run the function.
# ------------------------------------------------------------------------
def _run_emit_lr(emitter, tensor, numK):
    rec = _LRRecorder()
    with patch.object(IE, "Module", FakeModule), patch.object(
        IE, "emitSingleDsRead", rec
    ):
        emitter.emit_lr(_lr_placement(tensor, numK))
    return rec.calls


def _run_emit_gr(emitter, tensor, numK):
    rec = _GRRecorder()
    with patch.object(IE, "Module", FakeModule), patch.object(
        IE, "emitSingleBufferLoad", rec
    ):
        emitter.emit_gr(_gr_placement(tensor, numK))
    return rec.calls


# ========================================================================
#                  TESTS -- A=F8, B=F4
# ========================================================================
class TestMixed_F8A_F4B:
    """A=F8 (subtileShape[1]=1), B=F4 (subtileShape[1]=2)."""

    @pytest.fixture
    def env(self):
        return _make_emitter(AB_B8, AB_B4)

    def test_geometry_sanity(self, env):
        _, tiA, tiB, _ = env
        assert tiA.subtileShape[1] == 1, "A=F8 must have subtileShape[1]=1"
        assert tiB.subtileShape[1] == 2, "B=F4 must have subtileShape[1]=2"

    def test_emit_lr_A_uses_A_subtileShape(self, env):
        emitter, tiA, _, numK = env
        assert _run_emit_lr(emitter, "A", numK) == _expected_lr(tiA, numK)

    def test_emit_lr_B_uses_B_subtileShape(self, env):
        # ** This is the test that catches the legacy A-only bug. **
        emitter, _, tiB, numK = env
        got = _run_emit_lr(emitter, "B", numK)
        assert got == _expected_lr(tiB, numK), (
            f"B's LR must use B's subtileShape[1]={tiB.subtileShape[1]}, "
            f"not A's. Got: {got}"
        )

    def test_emit_gr_A_uses_A_subtileShape(self, env):
        emitter, tiA, _, numK = env
        assert _run_emit_gr(emitter, "A", numK) == _expected_gr(tiA, numK)

    def test_emit_gr_B_uses_B_subtileShape(self, env):
        emitter, _, tiB, numK = env
        assert _run_emit_gr(emitter, "B", numK) == _expected_gr(tiB, numK)


# ========================================================================
#                  TESTS -- A=F4, B=F8
# ========================================================================
class TestMixed_F4A_F8B:
    """Mirror image of the previous case."""

    @pytest.fixture
    def env(self):
        return _make_emitter(AB_B4, AB_B8)

    def test_geometry_sanity(self, env):
        _, tiA, tiB, _ = env
        assert tiA.subtileShape[1] == 2
        assert tiB.subtileShape[1] == 1

    def test_emit_lr_A_uses_A_subtileShape(self, env):
        emitter, tiA, _, numK = env
        assert _run_emit_lr(emitter, "A", numK) == _expected_lr(tiA, numK)

    def test_emit_lr_B_uses_B_subtileShape(self, env):
        emitter, _, tiB, numK = env
        got = _run_emit_lr(emitter, "B", numK)
        assert got == _expected_lr(tiB, numK), (
            f"B's LR must use B's subtileShape[1]={tiB.subtileShape[1]}, "
            f"not A's. Got: {got}"
        )

    def test_emit_gr_A_uses_A_subtileShape(self, env):
        emitter, tiA, _, numK = env
        assert _run_emit_gr(emitter, "A", numK) == _expected_gr(tiA, numK)

    def test_emit_gr_B_uses_B_subtileShape(self, env):
        emitter, _, tiB, numK = env
        assert _run_emit_gr(emitter, "B", numK) == _expected_gr(tiB, numK)


# ========================================================================
#         REGRESSION GUARD -- pure same-type A=B must keep working
# ========================================================================
@pytest.mark.parametrize("geo,sk", [(AB_B8, 1), (AB_B4, 2)])
def test_symmetric_AB_no_regression(geo, sk):
    """Same type for A and B (pure F8 or pure F4) must keep producing the
    same (sId1, subIterK) sequence."""
    emitter, tiA, tiB, numK = _make_emitter(geo, geo)
    assert tiA.subtileShape[1] == sk and tiB.subtileShape[1] == sk
    for tc, ti in (("A", tiA), ("B", tiB)):
        assert _run_emit_lr(emitter, tc, numK) == _expected_lr(ti, numK)
        assert _run_emit_gr(emitter, tc, numK) == _expected_gr(ti, numK)
