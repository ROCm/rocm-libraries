################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################
"""Unit tests for the ``TDMStoreInst`` store parameter (gfx1250).

``TDMStoreInst`` is the single control for the store-to-final-D epilogue: when
enabled, every alpha/beta store-to-D branch collapses to one whole-MacroTile
``tensor_store_from_lds`` (no per-element ``buffer_store`` fallback). These
tests are pure-Python (parameter validation + source-level wiring guards) and
do not require a GPU.
"""

import io
import os

import pytest

pytestmark = pytest.mark.unit

import Tensile
from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.TypeValidationErrors import ConfigTypeError

_TENSILE_DIR = os.path.dirname(Tensile.__file__)


def _read(*parts):
    with io.open(os.path.join(_TENSILE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


def _default_entry(name):
    hits = [d for d in defaultBenchmarkCommonParameters if name in d]
    return hits


class TestTDMStoreInstParameter:
    def test_in_valid_parameters(self):
        assert "TDMStoreInst" in validParameters
        assert validParameters["TDMStoreInst"] == [False, True]

    def test_has_single_default_false(self):
        hits = _default_entry("TDMStoreInst")
        assert len(hits) == 1
        assert hits[0]["TDMStoreInst"] == [False]

    def test_accepts_bool_values(self):
        # Must not raise for the valid bool domain.
        checkParametersAreValid(("TDMStoreInst", [False]), validParameters)
        checkParametersAreValid(("TDMStoreInst", [True]), validParameters)
        checkParametersAreValid(("TDMStoreInst", [False, True]), validParameters)

    def test_rejects_int(self):
        # TDMStoreInst is a bool toggle; an int must be rejected (bool/int trap).
        with pytest.raises(ConfigTypeError):
            checkParametersAreValid(("TDMStoreInst", [1]), validParameters)

    @pytest.mark.parametrize("removed", ["TDMSubtileHybrid", "TDMStoreEdge"])
    def test_superseded_store_params_removed(self, removed):
        # The old multi-toggle store knobs are gone; TDMStoreInst is the only one.
        assert removed not in validParameters
        assert _default_entry(removed) == []


class TestTDMStoreInstCodegenWiring:
    def test_store_functions_renamed(self):
        # The TDM store helpers must carry TDMStore* names with no
        # subtile / hybrid / ping-pong residue.
        src = _read("KernelWriterAssembly.py")
        for new in ("def _emitTDMStore(",
                    "def _emitTDMStoreScratch",
                    "def _emitTDMStoreBaseSetup"):
            assert new in src, new
        for old in ("_emitTdmSubtileHybridFlush",
                    "_emitSubtileHybridScratchStore",
                    "_emitTdmHybBaseSetup",
                    "TDMSubtileHybrid",
                    "TDMStoreEdge"):
            assert old not in src, old

    def test_store_gate_uses_tdmstoreinst(self):
        # The store-to-D dispatch gate is driven by TDMStoreInst and keeps the
        # MultipleBuffer(SingleKernel) workspace path out of scope (untouched).
        src = _read("Components", "GlobalWriteBatch.py")
        assert 'self.kernel.get("TDMStoreInst")' in src
        assert "TDMSubtileHybrid" not in src
        assert "isSubtileTDMStore" not in src
        assert '"MultipleBufferSingleKernel", "MultipleBuffer"' in src

    def test_store_base_setup_single_vgpr(self):
        # _emitTDMStoreBaseSetup must hold exactly ONE persistent VGPR for the
        # M-contiguous LDS scratch base.  A second live VGPR overflows StreamK's
        # store phase (already at the 1024-VGPR ceiling) to 1025 ("too many vgprs").
        src = _read("KernelWriterAssembly.py")
        setup = src.split("def _emitTDMStoreBaseSetup", 1)[1].split("\n  def ", 1)[0]
        assert 'checkOut(1, "tdmStoreBase")' in setup
        assert 'checkOut(2, "tdmStoreBase")' not in setup


class TestTDMStoreInstStreamKGate:
    """Solution-time gate for the StreamK narrow-scope TDM support.

    StreamK non-atomic (``_GlobalAccumulation == 'PartialsBuffer'``) keeps its
    fp32 partial-tile store on ``buffer_store`` (a separate StreamK path) and
    routes only the fixup-owner's final bf16->D store through the regular
    ``globalWriteBatch`` -- exactly where the TDM store fires.  These guards lock
    the accept/reject envelope in ``SolutionStructs/Solution.py``.
    """

    def test_streamk_partialsbuffer_is_the_allowed_mode(self):
        src = _read("SolutionStructs", "Solution.py")
        # Only the non-atomic PartialsBuffer StreamK mode is allowed; the atomic
        # path (not PartialsBuffer) is rejected.
        assert "!= 'PartialsBuffer'" in src
        assert "atomic StreamK reduction path does not route the final D store" in src

    def test_streamk_requires_usebeta_true(self):
        # StreamK aliases sgprSkPartialIdx onto sgprBeta, which is unallocated when
        # UseBeta=False -> cleanly rejected instead of an assembler error.
        src = _read("SolutionStructs", "Solution.py")
        assert 'state["StreamK"] != 0 and not pt.get("UseBeta", False)' in src
        assert "TDMStoreInst + StreamK requires UseBeta=True" in src

    def test_gsu_gt1_still_rejected(self):
        src = _read("SolutionStructs", "Solution.py")
        assert "TDMStoreInst does not yet support GlobalSplitU>1" in src

    def test_gate_comment_documents_streamk_partialsbuffer(self):
        # The GlobalWriteBatch gate comment must explain that PartialsBuffer is
        # intentionally NOT excluded (only the fixup-owner's D store reaches here).
        src = _read("Components", "GlobalWriteBatch.py")
        assert "PartialsBuffer" in src
        assert "fixup-owner" in src

