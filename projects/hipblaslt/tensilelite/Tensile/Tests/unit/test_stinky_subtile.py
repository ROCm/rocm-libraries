# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the StinkySubtile post-pass hook in ``Tensile.KernelWriter``.

No GPU required -- the StinkyTofu pipeline and rocIsaPass are stubbed out, so
these exercise only the Python glue: which module options the hook builds, which
passes each StinkySubtile level turns on, and the fallback that keeps the
un-optimized subtile assembly when rocIsaPass rejects a kernel.
"""

from types import SimpleNamespace

import pytest

import Tensile.KernelWriter as KW
from Tensile.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


GFX1250 = (12, 5, 0)


def _make_kernel(**overrides):
    """A kernel dict carrying just the keys the hook reads."""
    kernel = {
        "StinkySubtile": 1,
        "ThreadTile0": 4,
        "ThreadTile1": 8,
        "MacroTile0": 128,
        "WavefrontSize": 32,
        "SubGroup0": 16,
        "SubGroup1": 8,
        "MIWaveGroup": [2, 4],
        "VectorWidthA": 1,
        "VectorWidthB": 2,
        "GlobalReadVectorWidthA": 4,
        "GlobalReadVectorWidthB": 8,
        "DirectToLdsA": 0,
        "DirectToLdsB": 1,
        "_UseSgprForGRO": 0,
        "ActivationFuncCall": False,
        "StreamK": 0,
        "ProblemType": {
            "ActivationType": "none",
            "DataType": SimpleNamespace(isDouble=lambda: False),
        },
    }
    kernel.update(overrides)
    return kernel


def _make_writer(swPrefetchAbsBaseSgpr=-1, kernelName="test_kernel"):
    """A stand-in for KernelWriter carrying only the state the hook reads."""
    return SimpleNamespace(
        states=SimpleNamespace(
            version=GFX1250,
            kernelName=kernelName,
            swPrefetchAbsBaseSgpr=swPrefetchAbsBaseSgpr,
            archCaps={
                "RequiresXCntForVolatileVMEM": True,
                "EnableXnackReplay": False,
            },
        )
    )


# -- the shared module-options builder --

class TestModuleOptions:
    """_stinkyTofuModuleOptions carries the caller-independent half of the dict."""

    def test_tile_shape_is_forwarded(self):
        opts = KernelWriter._stinkyTofuModuleOptions(_make_writer(), _make_kernel())
        assert opts["TileA0"] == 4
        assert opts["TileB0"] == 8
        assert opts["TileM0"] == 128
        assert opts["wavefrontSize"] == 32
        assert opts["SubGroup0"] == 16
        assert opts["SubGroup1"] == 8

    def test_wave_group_is_split_into_two_keys(self):
        opts = KernelWriter._stinkyTofuModuleOptions(_make_writer(), _make_kernel())
        assert opts["WaveGroup0"] == 2
        assert opts["WaveGroup1"] == 4

    def test_direct_to_lds_is_coerced_to_bool(self):
        opts = KernelWriter._stinkyTofuModuleOptions(_make_writer(), _make_kernel())
        assert opts["DirectToLdsA"] is False
        assert opts["DirectToLdsB"] is True

    def test_arch_caps_are_read_from_writer_state(self):
        opts = KernelWriter._stinkyTofuModuleOptions(_make_writer(), _make_kernel())
        assert opts["RequiresXCntForVolatileVMEM"] is True
        assert opts["EnableXnackReplay"] is False

    def test_prefetch_scratch_sgpr_is_an_int(self):
        opts = KernelWriter._stinkyTofuModuleOptions(
            _make_writer(swPrefetchAbsBaseSgpr=72), _make_kernel())
        assert opts["SwInstructionPrefetchAbsBaseSgpr"] == 72
        assert isinstance(opts["SwInstructionPrefetchAbsBaseSgpr"], int)

    def test_pipeline_configuration_is_left_to_the_caller(self):
        """OptLevel and the per-pass gates belong to whichever path calls this."""
        opts = KernelWriter._stinkyTofuModuleOptions(_make_writer(), _make_kernel())
        for key in ("OptLevel", "EnableHazardCoverage", "EnableMovePropagation",
                    "EnableWaitCntInsertion", "EnableESM2"):
            assert key not in opts


# -- the level -> pass mapping --

@pytest.fixture
def captured_options(monkeypatch):
    """Run _stinkyTofuSubtileOptimize with the pipeline stubbed, return its options."""
    seen = {}

    class _StubPassOption:
        def __init__(self):
            self.getCycles = True
            self.insertDelayAlu = True
            self.removeDupFunc = False
            self.removeDupAssign = True

    def _stub_to_module(body, version, name, signature=None, options=None):
        seen.update(options or {})
        return SimpleNamespace(runOptimizationPipeline=lambda: None,
                               emitAssembly=lambda: "optimized-asm")

    monkeypatch.setattr(KW, "rocIsaPassOption", _StubPassOption)
    monkeypatch.setattr(KW, "rocIsaPass", lambda *a, **k: None)
    monkeypatch.setattr(KW, "print2", lambda *a, **k: None)
    monkeypatch.setattr(KW, "rocisa", SimpleNamespace(toStinkyTofuModule=_stub_to_module))
    monkeypatch.setattr(KW, "resolveSwInstructionPrefetch", lambda *a, **k: (True, False))

    def _run(level, **kernel_overrides):
        writer = _make_writer()
        # Isolate the level gating from the shared option builder.
        writer._stinkyTofuModuleOptions = lambda kernel: {}
        body = SimpleNamespace(body=SimpleNamespace(setParent=lambda: None))
        seen.clear()
        asm = KernelWriter._stinkyTofuSubtileOptimize(
            writer, _make_kernel(**kernel_overrides), body, None, level)
        return asm, dict(seen)

    return _run


class TestLevelGating:
    """Each level adds exactly one pass, and no level enables scheduling."""

    def test_returns_the_optimized_assembly(self, captured_options):
        asm, _ = captured_options(1)
        assert asm == "optimized-asm"

    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_opt_level_stays_zero(self, captured_options, level):
        """OptLevel 0 is what keeps the DAG scheduler out of the pipeline."""
        _, opts = captured_options(level)
        assert opts["OptLevel"] == 0

    @pytest.mark.parametrize("level,expected", [(1, False), (2, True), (3, True), (4, True)])
    def test_hazard_coverage_starts_at_level_2(self, captured_options, level, expected):
        _, opts = captured_options(level)
        assert opts["EnableHazardCoverage"] is expected

    @pytest.mark.parametrize("level,expected", [(1, False), (2, False), (3, True), (4, True)])
    def test_move_propagation_starts_at_level_3(self, captured_options, level, expected):
        _, opts = captured_options(level)
        assert opts["EnableMovePropagation"] is expected

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_prefetch_is_off_below_level_4(self, captured_options, level):
        _, opts = captured_options(level)
        assert opts["EnableSwInstructionPrefetchRelStatic"] is False
        assert opts["EnableSwInstructionPrefetchAbs"] is False

    def test_prefetch_follows_the_existing_knob_at_level_4(self, captured_options):
        """Level 4 defers the flavour to SwInstructionPrefetch, not a new control."""
        _, opts = captured_options(4)
        assert opts["EnableSwInstructionPrefetchRelStatic"] is True
        assert opts["EnableSwInstructionPrefetchAbs"] is False


# -- the fallback that keeps a kernel rather than dropping it --

class TestPostPassFallback:
    """rocIsaPass rejects some large subtile kernels; the hook must not drop them."""

    def test_returns_optimized_assembly_on_success(self):
        writer = _make_writer()
        writer._stinkyTofuSubtileOptimize = lambda *a, **k: "optimized-asm"
        out = KernelWriter._stinkyTofuSubtilePostPass(
            writer, _make_kernel(StinkySubtile=2), None, None)
        assert out == "optimized-asm"

    def test_passes_the_level_through(self):
        seen = {}

        def _optimize(kernel, body, fs, level):
            seen["level"] = level
            return "asm"

        writer = _make_writer()
        writer._stinkyTofuSubtileOptimize = _optimize
        KernelWriter._stinkyTofuSubtilePostPass(
            writer, _make_kernel(StinkySubtile=3), None, None)
        assert seen["level"] == 3

    def test_runtime_error_keeps_the_unoptimized_kernel(self, monkeypatch):
        """Returning None tells the caller to keep the subtile assembly as-is."""
        warnings = []
        monkeypatch.setattr(KW, "printWarning", lambda msg: warnings.append(msg))

        def _boom(*a, **k):
            raise RuntimeError("GPR index out of range")

        writer = _make_writer(kernelName="big_subtile_kernel")
        writer._stinkyTofuSubtileOptimize = _boom
        out = KernelWriter._stinkyTofuSubtilePostPass(
            writer, _make_kernel(StinkySubtile=4), None, None)

        assert out is None
        assert len(warnings) == 1
        # The kernel has to be named, or the drop is invisible in the log.
        assert "big_subtile_kernel" in warnings[0]
        assert "GPR index out of range" in warnings[0]
        assert "StinkySubtile=4" in warnings[0]

    def test_non_runtime_errors_still_propagate(self, monkeypatch):
        """Only the known rocIsaPass limitation is swallowed."""
        monkeypatch.setattr(KW, "printWarning", lambda msg: None)

        def _boom(*a, **k):
            raise ValueError("unrelated")

        writer = _make_writer()
        writer._stinkyTofuSubtileOptimize = _boom
        with pytest.raises(ValueError):
            KernelWriter._stinkyTofuSubtilePostPass(
                writer, _make_kernel(StinkySubtile=1), None, None)
