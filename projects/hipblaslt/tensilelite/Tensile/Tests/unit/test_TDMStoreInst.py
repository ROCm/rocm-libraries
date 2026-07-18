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
from Tensile.Common.DataType import DataType
from rocisa.enum import DataTypeEnum
from Tensile.SolutionStructs.Validators.TDMStoreInst import validateTDMStoreInst

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
        # Gate is a whitelist of D-writing accumulation modes (robust vs. a blacklist).
        assert 'tdmStoreDestModes = (None, "SingleBuffer", "PartialsBuffer")' in src
        assert 'self.kernel["_GlobalAccumulation"] in tdmStoreDestModes' in src

    def test_store_base_setup_single_vgpr(self):
        # _emitTDMStoreBaseSetup must hold exactly ONE persistent VGPR for the
        # M-contiguous LDS scratch base.  A second live VGPR overflows StreamK's
        # store phase (already at the 1024-VGPR ceiling) to 1025 ("too many vgprs").
        src = _read("KernelWriterAssembly.py")
        setup = src.split("def _emitTDMStoreBaseSetup", 1)[1].split("\n  def ", 1)[0]
        assert 'checkOut(1, "tdmStoreBase")' in setup
        assert 'checkOut(2, "tdmStoreBase")' not in setup


def _tdm_state(**overrides):
    """Smallest Solution state read by validateTDMStoreInst (dict-in / bool-out).

    Defaults to an accepted GSU=1 bf16/HPA/single config on gfx1250.  Pass
    ``pt={...}`` to override ProblemType fields and any other key as a top-level
    override (ISA, StreamK, _GlobalAccumulation, GlobalSplitU, ...).
    """
    pt = dict(
        DestDataType=DataType(DataTypeEnum.BFloat16),
        ComputeDataType=DataType(DataTypeEnum.Float),
        HighPrecisionAccumulate=True,
        UseBeta=False,
        ActivationType="none",
    )
    pt.update(overrides.pop("pt", {}))
    state = dict(
        TDMStoreInst=True,
        ISA=[12, 5, 0],
        StreamK=0,
        _GlobalAccumulation="SingleBuffer",
        GlobalSplitU=1,
        StoreRemapVectorWidth=0,
        ProblemType=pt,
    )
    state.update(overrides)
    return state


class TestTDMStoreInstValidator:
    """Behavioral accept/reject tests for ``validateTDMStoreInst``.

    Builds a real Solution state and asserts the return value and
    ``state["Valid"]`` (which ``reject`` sets to False).  Pure dict-in / bool-out:
    no client build, no GPU.
    """

    def test_valid_gsu1_accepted(self):
        st = _tdm_state()
        assert validateTDMStoreInst(st, printRejectionReason=False) is True
        assert st.get("Valid") is not False

    def test_tdmstoreinst_false_is_noop(self):
        # TDMStoreInst=False short-circuits: even an otherwise-illegal ISA is untouched.
        st = _tdm_state(TDMStoreInst=False, ISA=[9, 4, 2])
        assert validateTDMStoreInst(st, printRejectionReason=False) is True
        assert "Valid" not in st

    def test_gsu_minus1_accepted(self):
        # auto-GSU resolving to SingleBuffer direct-to-D is allowed (it is not > 1).
        st = _tdm_state(GlobalSplitU=-1)
        assert validateTDMStoreInst(st, printRejectionReason=False) is True

    @pytest.mark.parametrize("isa", [[9, 4, 2], [9, 5, 0], [12, 0, 0]],
                             ids=["gfx942", "gfx950", "gfx1200"])
    def test_non_gfx1250_rejected(self, isa):
        st = _tdm_state(ISA=isa)
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_non_bf16_dest_rejected(self):
        st = _tdm_state(pt={"DestDataType": DataType(DataTypeEnum.Half)})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_non_hpa_rejected(self):
        st = _tdm_state(pt={"HighPrecisionAccumulate": False})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False

    def test_non_single_compute_rejected(self):
        st = _tdm_state(pt={"ComputeDataType": DataType(DataTypeEnum.Double)})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False

    def test_gsu_gt1_rejected(self):
        st = _tdm_state(GlobalSplitU=2)
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_storeremap_rejected(self):
        st = _tdm_state(StoreRemapVectorWidth=4)
        assert validateTDMStoreInst(st, printRejectionReason=False) is False

    # --- StreamK envelope ---
    def test_streamk_partialsbuffer_beta_accepted(self):
        st = _tdm_state(StreamK=3, _GlobalAccumulation="PartialsBuffer", pt={"UseBeta": True})
        assert validateTDMStoreInst(st, printRejectionReason=False) is True
        assert st.get("Valid") is not False

    def test_streamk_usebeta_false_rejected(self):
        st = _tdm_state(StreamK=3, _GlobalAccumulation="PartialsBuffer")
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_streamk_atomic_rejected(self):
        # StreamK != 0 but not the PartialsBuffer workspace mode (atomic reduction path).
        st = _tdm_state(StreamK=2, _GlobalAccumulation="SingleBuffer", pt={"UseBeta": True})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    # --- unvalidated epilogue features ---
    def test_activation_rejected(self):
        # Regression guard for the reject key: it must test ActivationType (not a
        # non-existent "Activation" key that silently never fires).
        st = _tdm_state(pt={"ActivationType": "relu"})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_activation_none_accepted(self):
        st = _tdm_state(pt={"ActivationType": "none"})
        assert validateTDMStoreInst(st, printRejectionReason=False) is True

    @pytest.mark.parametrize("feat,val", [
        ("UseE", True), ("UseBias", 1), ("UseScaleAlphaVec", 1),
        ("UseScaleAB", "Scalar"), ("UseScaleCD", True),
    ])
    def test_unsupported_epilogue_features_rejected(self, feat, val):
        st = _tdm_state(pt={feat: val})
        assert validateTDMStoreInst(st, printRejectionReason=False) is False
        assert st["Valid"] is False

    def test_gate_comment_documents_streamk_partialsbuffer(self):
        # The GlobalWriteBatch gate comment must explain that PartialsBuffer is
        # intentionally NOT excluded (only the fixup-owner's D store reaches here).
        src = _read("Components", "GlobalWriteBatch.py")
        assert "PartialsBuffer" in src
        assert "fixup-owner" in src

