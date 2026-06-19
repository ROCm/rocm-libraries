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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Characterization tests for ClientWriter.py -- config-writing path.

Targets remaining missing coverage in:
  - checkConstStride (lines 406-412)
  - problemSizeParams (lines 415-515) -- useE, useBias, sparse, setConstStride* branches
  - dataInitParams (lines 517-554)
  - boundsCheckName, pruneModeName (lines 556-570)
  - writeClientConfigIni (lines 572-731)
  - writeClientConfig (lines 733-774)
  - CreateBenchmarkClientParametersForSizes (lines 776-793)

Strategy: build real ContractionsProblemType objects via FromOriginalState (the same dict
structure used by parseLibraryLogicFile) and drive writeClientConfigIni with
CPU-only overrides for all globalParameters keys.  No GPU, no subprocesses.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import Tensile.ClientWriter as CW
from Tensile.Contractions import ProblemType as ContractionsProblemType, FreeIndex, BatchIndex
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.SolutionStructs.Problem import ProblemSizesMockDummy
from Tensile.SolutionStructs import FactorDimArgs

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canonical minimal problem-type dict for a plain HSS GEMM (no bias, no E)
# Note: UseScaleAlphaVec and UseBias keys are included so FactorDimArgs works.
# ---------------------------------------------------------------------------
_PLAIN_GEMM_PT_DICT = {
    "TotalIndices": 4,
    "NumIndicesC": 3,
    "IndicesBatch": [2],
    "IndicesFree": [0, 1],
    "IndicesSummation": [3],
    "IndexAssignmentsA": [3, 0, 2],
    "IndexAssignmentsB": [1, 3, 2],
    "ComplexConjugateA": False,
    "ComplexConjugateB": False,
    "TransposeA": 1,
    "TransposeB": 1,
    "DataType": 4,      # BFloat16
    "DestDataType": 0,  # Float
    "ComputeDataType": 0,
    "ActivationComputeDataType": 0,
    "Batched": True,
    "StridedBatched": True,
    "UseBeta": True,
    "F32XdlMathOp": 0,
    # Required by FactorDimArgs constructor
    "UseScaleAlphaVec": 0,
    "UseBias": 0,
    "BiasDataTypeList": [],
    # BiasSrc must be set so biasSrcWhiteList is non-empty — writeClientConfigIni
    # calls biasSrcWhiteList[0] unconditionally on line 608.
    "BiasSrc": "D",
}

# Problem type with Bias enabled (useBias=1, biasSrcWhiteList non-empty)
_BIAS_GEMM_PT_DICT = dict(_PLAIN_GEMM_PT_DICT)
_BIAS_GEMM_PT_DICT.update({
    "UseBias": 1,
    "BiasDataTypeList": [0],   # Float bias
    "BiasSrc": "D",
})

# Problem type with UseE enabled
_USE_E_PT_DICT = dict(_PLAIN_GEMM_PT_DICT)
_USE_E_PT_DICT.update({
    "UseE": True,
    "DataTypeE": 0,  # Float
})

# Problem type with OutputAmaxD
_AMAX_PT_DICT = dict(_PLAIN_GEMM_PT_DICT)
_AMAX_PT_DICT.update({
    "OutputAmaxD": True,
    "DataTypeAmaxD": 0,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem_type(d):
    """Build a Contractions.ProblemType from a state dict."""
    return ContractionsProblemType.FromOriginalState(d)


def _make_factor_dim_args(pt_dict):
    """Build a FactorDimArgs using the original dict-format problemType."""
    return FactorDimArgs(pt_dict, [0])


def _set_all_gp(monkeypatch):
    """Set every globalParameters key read by writeClientConfigIni to a valid default."""
    keys = {
        "LibraryFormat": "yaml",
        "PerformanceMetric": "DeviceEfficiency",
        "DataInitValueActivationArgs": None,
        "DataInitSeed": 0,
        "CEqualD": False,
        "PrintTensorA": 0,
        "PrintTensorB": 0,
        "PrintTensorC": 0,
        "PrintTensorD": 0,
        "PrintTensorRef": 0,
        "PrintTensorBias": 0,
        "PrintTensorScaleAlphaVec": 0,
        "PrintTensorAmaxD": 0,
        "DumpTensors": 0,
        "ExitOnFails": 1,
        "PruneSparseMode": 0,
        "BoundsCheck": 0,
        "ValidationPrintValids": False,
        "ValidationMaxToPrint": 4,
        "NumBenchmarks": 1,
        "NumElementsToValidate": 0,
        "NumElementsToValidateWinner": -1,
        "EnqueuesPerSync": 1,
        "MaxEnqueuesPerSync": -1,
        "SyncsPerBenchmark": 1,
        "SkipSlowSolutionRatio": 0,
        "KernelTime": False,
        "HardwareMonitor": True,
        "NumWarmups": 0,
        "MinFlopsPerSync": 0,
        "SleepPercent": 0,
        "PerfModelL2ReadHits": 0.0,
        "PerfModelL2WriteHits": 0.15,
        "PerfModelL2ReadBwMul": 2,
        "PerfModelReadEfficiency": 1.0,
        "CSVExportWinner": False,
        "CSVMergeSameProblemID": False,
        "ClientLogLevel": 1,
        "MaxWorkspaceSize": 32 * 1024 * 1024,
        "PrintWinnersOnly": False,
        "GranularityThreshold": 0.0,
        "PredictionThreshold": 100.0,
        "PristineOnGPU": True,
        "LibraryUpdateFile": "",
        "LibraryUpdateComment": "",
        "UseUserArgs": False,
        "RotatingBufferSize": 0,
        "RotatingMode": 0,
        "RocProfCounter": [],
        # DataInit keys for dataInitParams
        "DataInitTypeA": -1,
        "DataInitTypeB": -1,
        "DataInitTypeC": 3,
        "DataInitTypeD": 0,
        "DataInitTypeE": 0,
        "DataInitTypeAlpha": 1,
        "DataInitTypeBeta": 1,
        "DataInitTypeBias": 0,
        "DataInitTypeScaleA": 0,
        "DataInitTypeScaleB": 0,
        "DataInitTypeScaleC": 0,
        "DataInitTypeScaleD": 0,
        "DataInitTypeScaleAlphaVec": 0,
        "DataInitTypeMXSA": 0,
        "DataInitTypeMXSB": 0,
        "DataInitTypeAB": 3,
    }
    for k, v in keys.items():
        monkeypatch.setitem(globalParameters, k, v)


def _write_ini(tmp_path, monkeypatch, pt, pt_dict, problem_sizes=None,
               forBenchmark=False, libraryFile="/lib/TensileLibrary.yaml",
               codeObjectFiles=None, biasTypeArgs="", factorDimArgs=None,
               activationArgs="", icacheFlushArgs="", probSolMap=None,
               extra_gp=None):
    """Call writeClientConfigIni with a real ContractionsProblemType and return content."""
    _set_all_gp(monkeypatch)
    if extra_gp:
        for k, v in extra_gp.items():
            monkeypatch.setitem(globalParameters, k, v)

    if problem_sizes is None:
        problem_sizes = ProblemSizesMockDummy()
    if codeObjectFiles is None:
        codeObjectFiles = []
    if factorDimArgs is None:
        factorDimArgs = _make_factor_dim_args(pt_dict)
    if probSolMap is None:
        probSolMap = {}

    source_dir = str(tmp_path / "source")
    os.makedirs(source_dir, exist_ok=True)
    out_file = str(tmp_path / "Params.ini")

    CW.writeClientConfigIni(
        forBenchmark=forBenchmark,
        problemSizes=problem_sizes,
        biasTypeArgs=biasTypeArgs,
        factorDimArgs=factorDimArgs,
        activationArgs=activationArgs,
        icacheFlushArgs=icacheFlushArgs,
        problemType=pt,
        sourceDir=source_dir,
        codeObjectFiles=codeObjectFiles,
        resultsFileName="/data/Results.csv",
        parametersFilePath=out_file,
        deviceId=0,
        gfxName="gfx942",
        libraryFile=libraryFile,
        probSolMap=probSolMap,
    )
    return Path(out_file).read_text()


# ===========================================================================
# 1. checkConstStride  (lines 406-412)
# ===========================================================================

class TestCheckConstStride:
    """checkConstStride -- lines 406-412."""

    def test_key_found_returns_val(self):
        """Lines 408-410: matching mapIdx returns val."""
        result = CW.checkConstStride([(0, 42), (1, 99)], 1)
        assert result == 99

    def test_key_not_found_returns_none(self):
        """Line 407: no match -- finalVal stays None."""
        result = CW.checkConstStride([(0, 10), (2, 20)], 5)
        assert result is None

    def test_last_match_wins(self):
        """Lines 408-410: later entry overwrites earlier when both match keyIdx."""
        result = CW.checkConstStride([(3, 7), (3, 99)], 3)
        assert result == 99

    def test_empty_map(self):
        """Empty map -- returns None."""
        result = CW.checkConstStride([], 0)
        assert result is None


# ===========================================================================
# 2. boundsCheckName / pruneModeName  (lines 556-570)
# ===========================================================================

class TestBoundsCheckName:
    """boundsCheckName -- lines 556-561."""

    def test_mode_0(self):  assert CW.boundsCheckName(0) == 'Disable'
    def test_mode_1(self):  assert CW.boundsCheckName(1) == 'NaN'
    def test_mode_2(self):  assert CW.boundsCheckName(2) == 'GuardPageFront'
    def test_mode_3(self):  assert CW.boundsCheckName(3) == 'GuardPageBack'
    def test_mode_4(self):  assert CW.boundsCheckName(4) == 'GuardPageAll'


class TestPruneModeName:
    """pruneModeName -- lines 563-570."""

    def test_mode_0(self):  assert CW.pruneModeName(0) == 'PruneRandom'
    def test_mode_1(self):  assert CW.pruneModeName(1) == 'PruneXX00'
    def test_mode_2(self):  assert CW.pruneModeName(2) == 'PruneX0X0'
    def test_mode_3(self):  assert CW.pruneModeName(3) == 'Prune0XX0'
    def test_mode_4(self):  assert CW.pruneModeName(4) == 'PruneX00X'
    def test_mode_5(self):  assert CW.pruneModeName(5) == 'Prune0X0X'
    def test_mode_6(self):  assert CW.pruneModeName(6) == 'Prune00XX'


# ===========================================================================
# 3. dataInitParams  (lines 517-554)
# ===========================================================================

class TestDataInitParams:
    """dataInitParams -- lines 517-554."""

    def _set_keys(self, monkeypatch, **overrides):
        defaults = {
            "DataInitTypeA": -1,
            "DataInitTypeB": -1,
            "DataInitTypeC": 3,
            "DataInitTypeD": 0,
            "DataInitTypeE": 0,
            "DataInitTypeAlpha": 1,
            "DataInitTypeBeta": 1,
            "DataInitTypeBias": 0,
            "DataInitTypeScaleA": 0,
            "DataInitTypeScaleB": 0,
            "DataInitTypeScaleC": 0,
            "DataInitTypeScaleD": 0,
            "DataInitTypeScaleAlphaVec": 0,
            "DataInitTypeMXSA": 0,
            "DataInitTypeMXSB": 0,
            "DataInitTypeAB": 3,
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            monkeypatch.setitem(globalParameters, k, v)

    def _make_pt(self, use_beta=True):
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        pt.useBeta = use_beta
        return pt

    def test_ab_resolved_from_ab_default(self, monkeypatch):
        """Lines 537-538: DataInitTypeA/B=-1 -- resolved from DataInitTypeAB."""
        self._set_keys(monkeypatch, DataInitTypeA=-1, DataInitTypeB=-1, DataInitTypeAB=3)
        result = dict(CW.dataInitParams(self._make_pt()))
        assert result["init-a"] == "Random"
        assert result["init-b"] == "Random"

    def test_ab_explicit(self, monkeypatch):
        """Lines 518-519: explicit DataInitTypeA/B used directly."""
        self._set_keys(monkeypatch, DataInitTypeA=1, DataInitTypeB=2)
        result = dict(CW.dataInitParams(self._make_pt()))
        assert result["init-a"] == "One"
        assert result["init-b"] == "Two"

    def test_beta_zeroed_when_no_beta(self, monkeypatch):
        """Lines 534-535: initBeta set to 0 when problemType.useBeta=False."""
        self._set_keys(monkeypatch, DataInitTypeBeta=1)
        result = dict(CW.dataInitParams(self._make_pt(use_beta=False)))
        assert result["init-beta"] == "Zero"

    def test_all_keys_present(self, monkeypatch):
        """Lines 540-554: all 15 init parameters are returned."""
        self._set_keys(monkeypatch)
        result = dict(CW.dataInitParams(self._make_pt()))
        expected_keys = [
            "init-a", "init-b", "init-c", "init-d", "init-e",
            "init-alpha", "init-beta", "init-bias",
            "init-scaleA", "init-scaleB", "init-scaleC", "init-scaleD",
            "init-scaleAlphaVec", "init-mx-a", "init-mx-b",
        ]
        for key in expected_keys:
            assert key in result

    def test_mx_keys_present(self, monkeypatch):
        """Lines 531-532: DataInitTypeMXSA/B used for init-mx-a/b."""
        self._set_keys(monkeypatch, DataInitTypeMXSA=0, DataInitTypeMXSB=0)
        result = dict(CW.dataInitParams(self._make_pt()))
        assert result["init-mx-a"] == "Zero"
        assert result["init-mx-b"] == "Zero"


# ===========================================================================
# 4. writeClientConfigIni -- plain GEMM (lines 572-731)
# ===========================================================================

class TestWriteClientConfigIniPlain:
    """writeClientConfigIni basic plain GEMM -- exercises lines 572-731."""

    def test_library_file_written(self, tmp_path, monkeypatch):
        """Lines 580-583: library-file key uses provided libraryFile."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             libraryFile="/explicit/TensileLibrary.yaml")
        assert "library-file=/explicit/TensileLibrary.yaml" in content

    def test_library_file_none_raises(self, tmp_path, monkeypatch):
        """Line 582: libraryFile=None is rejected -- AssertionError."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        _set_all_gp(monkeypatch)
        monkeypatch.setitem(globalParameters, "LibraryFormat", "yaml")

        source_dir = str(tmp_path / "source")
        os.makedirs(source_dir, exist_ok=True)
        out_file = str(tmp_path / "Params.ini")
        factor_dim = _make_factor_dim_args(_PLAIN_GEMM_PT_DICT)

        with pytest.raises(AssertionError, match="libraryFile is required"):
            CW.writeClientConfigIni(
                forBenchmark=False,
                problemSizes=ProblemSizesMockDummy(),
                biasTypeArgs="",
                factorDimArgs=factor_dim,
                activationArgs="",
                icacheFlushArgs="",
                problemType=pt,
                sourceDir=source_dir,
                codeObjectFiles=[],
                resultsFileName="/data/Results.csv",
                parametersFilePath=out_file,
                deviceId=0,
                gfxName="gfx942",
                libraryFile=None,
            )

    def test_library_file_dat_written_verbatim(self, tmp_path, monkeypatch):
        """Line 588: a caller-supplied .dat path is written to library-file verbatim."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        _set_all_gp(monkeypatch)
        monkeypatch.setitem(globalParameters, "LibraryFormat", "msgpack")

        source_dir = str(tmp_path / "source")
        os.makedirs(source_dir, exist_ok=True)
        out_file = str(tmp_path / "Params.ini")
        factor_dim = _make_factor_dim_args(_PLAIN_GEMM_PT_DICT)

        CW.writeClientConfigIni(
            forBenchmark=False,
            problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="",
            factorDimArgs=factor_dim,
            activationArgs="",
            icacheFlushArgs="",
            problemType=pt,
            sourceDir=source_dir,
            codeObjectFiles=[],
            resultsFileName="/data/Results.csv",
            parametersFilePath=out_file,
            deviceId=0,
            gfxName="gfx942",
            libraryFile="/lib/TensileLibrary.dat",
        )
        content = Path(out_file).read_text()
        assert "library-file=/lib/TensileLibrary.dat" in content

    def test_code_object_filtering_by_gfxname(self, tmp_path, monkeypatch):
        """Lines 585-587: code-object written only when gfxName in coFile or no 'gfx' in coFile."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        co_files = ["kernel_gfx942.co", "kernel_gfx90a.co", "kernel_noarch.co"]
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             codeObjectFiles=co_files,
                             libraryFile="/lib/TensileLibrary.yaml")
        # gfx942 and noarch should be written; gfx90a should not
        assert "kernel_gfx942.co" in content
        assert "kernel_noarch.co" in content
        assert "kernel_gfx90a.co" not in content

    def test_essential_problem_params_written(self, tmp_path, monkeypatch):
        """Lines 589-615: core problem-type parameters written to INI."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT)
        assert "results-file=" in content
        assert "performance-metric=" in content
        assert "problem-identifier=" in content
        assert "a-type=" in content
        assert "b-type=" in content
        assert "c-type=" in content
        assert "d-type=" in content
        assert "alpha-type=" in content
        assert "beta-type=" in content

    def test_device_and_seed_written(self, tmp_path, monkeypatch):
        """Lines 658-660: device-idx and init-seed written."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT)
        assert "device-idx=0" in content
        assert "init-seed=" in content

    def test_benchmark_flag_affects_numElementsToValidate(self, tmp_path, monkeypatch):
        """Lines 695-700: forBenchmark=True skips the winner logic."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             forBenchmark=True,
                             extra_gp={
                                 "NumElementsToValidate": 100,
                                 "NumElementsToValidateWinner": -1,
                             })
        assert "num-elements-to-validate=100" in content

    def test_non_benchmark_winner_max_used(self, tmp_path, monkeypatch):
        """Lines 696-699: forBenchmark=False with positive winner -- max taken."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             forBenchmark=False,
                             extra_gp={
                                 "NumElementsToValidate": 50,
                                 "NumElementsToValidateWinner": 200,
                             })
        assert "num-elements-to-validate=200" in content

    def test_non_benchmark_winner_negative_sets_neg1(self, tmp_path, monkeypatch):
        """Lines 696-697: winner=-1 -- numElementsToValidate=-1."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             forBenchmark=False,
                             extra_gp={
                                 "NumElementsToValidate": 50,
                                 "NumElementsToValidateWinner": -1,
                             })
        assert "num-elements-to-validate=-1" in content

    def test_print_tensor_flags_written_when_set(self, tmp_path, monkeypatch):
        """Lines 667-686: PrintTensor* flags written to INI when enabled."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={
                                 "PrintTensorA": 1,
                                 "PrintTensorB": 1,
                                 "PrintTensorD": 1,
                                 "ExitOnFails": 2,
                             })
        assert "print-tensor-a=1" in content
        assert "print-tensor-b=1" in content
        assert "print-tensor-d=1" in content
        assert "exit-on-error=1" in content

    def test_print_tensor_bias_and_scale_alpha_vec(self, tmp_path, monkeypatch):
        """Lines 677-681: PrintTensorBias/ScaleAlphaVec/AmaxD written when enabled."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={
                                 "PrintTensorBias": 1,
                                 "PrintTensorScaleAlphaVec": 1,
                                 "PrintTensorAmaxD": 1,
                                 "DumpTensors": 1,
                             })
        assert "print-tensor-bias=1" in content
        assert "print-tensor-scale-alpha-vec=1" in content
        assert "print-tensor-amaxd=1" in content
        assert "dump-tensors=1" in content

    def test_rocprof_counter_written_when_set(self, tmp_path, monkeypatch):
        """Lines 729-731: rocprof-counter written for each counter."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={"RocProfCounter": ["SQ_INSTS_VALU", "TA_BUSY"]})
        assert "rocprof-counter=SQ_INSTS_VALU" in content
        assert "rocprof-counter=TA_BUSY" in content

    def test_bounds_and_prune_mode_in_ini(self, tmp_path, monkeypatch):
        """Lines 688-689: prune-mode and bounds-check use correct name lookups."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={
                                 "BoundsCheck": 1,      # NaN
                                 "PruneSparseMode": 2,   # PruneX0X0
                             })
        assert "bounds-check=NaN" in content
        assert "prune-mode=PruneX0X0" in content

    def test_log_level_written(self, tmp_path, monkeypatch):
        """Line 716: log-level written from ClientLogLevel enum."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={"ClientLogLevel": 2})
        assert "log-level=Verbose" in content


# ===========================================================================
# 5. writeClientConfigIni -- UseE path (lines 598-599)
# ===========================================================================

class TestWriteClientConfigIniUseE:
    """writeClientConfigIni with UseE=True -- lines 598-599."""

    def test_e_type_written_when_use_e(self, tmp_path, monkeypatch):
        """Lines 598-599: e-type param emitted when problemType.useE is True."""
        pt = _make_problem_type(_USE_E_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _USE_E_PT_DICT)
        assert "e-type=" in content
        assert "use-e=True" in content


# ===========================================================================
# 6. writeClientConfigIni -- OutputAmaxD path (lines 600-601)
# ===========================================================================

class TestWriteClientConfigIniAmaxD:
    """writeClientConfigIni with OutputAmaxD=True -- lines 600-601."""

    def test_amaxD_type_written_when_set(self, tmp_path, monkeypatch):
        """Lines 600-601: amaxD-type param emitted when problemType.outputAmaxD is True."""
        pt = _make_problem_type(_AMAX_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _AMAX_PT_DICT)
        assert "amaxD-type=" in content
        assert "output-amaxD=True" in content


# ===========================================================================
# 7. writeClientConfigIni -- activation additional args (line 655-656)
# ===========================================================================

class TestWriteClientConfigIniActivation:
    """writeClientConfigIni with DataInitValueActivationArgs -- line 655-656."""

    def test_activation_additional_args_written(self, tmp_path, monkeypatch):
        """Lines 655-656: activation-additional-args written when DataInitValueActivationArgs is set."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             extra_gp={"DataInitValueActivationArgs": [1.0, 2.5]})
        assert "activation-additional-args=1.0,2.5" in content


# ===========================================================================
# 8. writeClientConfigIni -- probSolMap (lines 645-647)
# ===========================================================================

class TestWriteClientConfigIniProbSolMap:
    """writeClientConfigIni with probSolMap -- lines 645-647."""

    def test_prob_sol_map_written_when_present(self, tmp_path, monkeypatch):
        """Lines 645-647: prob-sol-map entry written for matching probIdx."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             probSolMap={0: 3})
        assert "prob-sol-map=0,3" in content


# ===========================================================================
# 9. writeClientConfigIni -- icacheFlushArgs (lines 631-633)
# ===========================================================================

class TestWriteClientConfigIniIcacheFlush:
    """writeClientConfigIni with icacheFlushArgs -- lines 631-633."""

    def test_icache_flush_args_written(self, tmp_path, monkeypatch):
        """Lines 631-633: icache-flush-args written for each opt."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        icache_args = [True, False]   # raw values that will be written
        content = _write_ini(tmp_path, monkeypatch, pt, _PLAIN_GEMM_PT_DICT,
                             icacheFlushArgs=icache_args)
        assert "icache-flush-args=True" in content
        assert "icache-flush-args=False" in content


# ===========================================================================
# 10. writeClientConfig (lines 733-774)
# ===========================================================================

class TestWriteClientConfig:
    """writeClientConfig -- lines 733-774."""

    def _build_new_library(self, problem_type):
        """Build a minimal SolutionLibrary with one solution for writeClientConfig."""
        fake_solution = SimpleNamespace(problemType=problem_type)
        new_library = SimpleNamespace(solutions={"sol0": fake_solution})
        return new_library

    def test_raises_when_no_solutions(self, tmp_path, monkeypatch):
        """Lines 762-763: empty solutions -- RuntimeError."""
        _set_all_gp(monkeypatch)
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        empty_lib = SimpleNamespace(solutions={})

        with pytest.raises(RuntimeError, match="No valid solutions found"):
            CW.writeClientConfig(
                forBenchmark=False,
                solutions=None,
                problemSizes=ProblemSizesMockDummy(),
                biasTypeArgs="",
                factorDimArgs=_make_factor_dim_args(_PLAIN_GEMM_PT_DICT),
                activationArgs="",
                icacheFlushArgs="",
                stepName="Test",
                stepBaseDir=str(tmp_path),
                newLibrary=empty_lib,
                codeObjectFiles=[],
                tileAwareSelection=False,
                deviceId=0,
                gfxName="gfx942",
                libraryFile="/lib/TensileLibrary.yaml",
            )

    def test_creates_ini_file(self, tmp_path, monkeypatch):
        """Lines 754-772: creates an .ini file with expected content."""
        _set_all_gp(monkeypatch)
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        new_library = self._build_new_library(pt)

        filename = CW.writeClientConfig(
            forBenchmark=False,
            solutions=None,
            problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="",
            factorDimArgs=_make_factor_dim_args(_PLAIN_GEMM_PT_DICT),
            activationArgs="",
            icacheFlushArgs="",
            stepName="MyTest",
            stepBaseDir=str(tmp_path),
            newLibrary=new_library,
            codeObjectFiles=[],
            tileAwareSelection=False,
            deviceId=0,
            gfxName="gfx942",
            libraryFile="/lib/TensileLibrary.yaml",
            sourceDir=str(source_dir),
        )
        assert os.path.exists(filename)
        content = Path(filename).read_text()
        assert "library-file=/lib/TensileLibrary.yaml" in content

    def test_tileAwareSelection_uses_granularity_filename(self, tmp_path, monkeypatch):
        """Lines 757-758: tileAwareSelection=True -- _Granularity.ini filename."""
        _set_all_gp(monkeypatch)
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        new_library = self._build_new_library(pt)

        filename = CW.writeClientConfig(
            forBenchmark=False,
            solutions=None,
            problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="",
            factorDimArgs=_make_factor_dim_args(_PLAIN_GEMM_PT_DICT),
            activationArgs="",
            icacheFlushArgs="",
            stepName="MyTest",
            stepBaseDir=str(tmp_path),
            newLibrary=new_library,
            codeObjectFiles=[],
            tileAwareSelection=True,
            deviceId=0,
            gfxName="gfx942",
            libraryFile="/lib/TensileLibrary.yaml",
            sourceDir=str(source_dir),
        )
        assert "_Granularity.ini" in filename

    def test_non_tile_uses_plain_filename(self, tmp_path, monkeypatch):
        """Line 760: tileAwareSelection=False -- plain .ini filename."""
        _set_all_gp(monkeypatch)
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        new_library = self._build_new_library(pt)

        filename = CW.writeClientConfig(
            forBenchmark=False,
            solutions=None,
            problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="",
            factorDimArgs=_make_factor_dim_args(_PLAIN_GEMM_PT_DICT),
            activationArgs="",
            icacheFlushArgs="",
            stepName="MyTest",
            stepBaseDir=str(tmp_path),
            newLibrary=new_library,
            codeObjectFiles=[],
            tileAwareSelection=False,
            deviceId=0,
            gfxName="gfx942",
            libraryFile="/lib/TensileLibrary.yaml",
            sourceDir=str(source_dir),
        )
        assert "_Granularity.ini" not in filename
        assert filename.endswith(".ini")

    def test_default_sourceDir_from_stepBaseDir(self, tmp_path, monkeypatch):
        """Line 754-755: sourceDir=None -- defaults to stepBaseDir/source."""
        _set_all_gp(monkeypatch)
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        # Create the default source subdir
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)
        new_library = self._build_new_library(pt)

        filename = CW.writeClientConfig(
            forBenchmark=False,
            solutions=None,
            problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="",
            factorDimArgs=_make_factor_dim_args(_PLAIN_GEMM_PT_DICT),
            activationArgs="",
            icacheFlushArgs="",
            stepName="MyTest",
            stepBaseDir=str(tmp_path),
            newLibrary=new_library,
            codeObjectFiles=[],
            tileAwareSelection=False,
            deviceId=0,
            gfxName="gfx942",
            libraryFile="/lib/TensileLibrary.yaml",
            sourceDir=None,  # triggers default computation
        )
        assert os.path.exists(filename)


# ===========================================================================
# 11. CreateBenchmarkClientParametersForSizes (lines 776-793)
# ===========================================================================

class TestCreateBenchmarkClientParametersForSizes:
    """CreateBenchmarkClientParametersForSizes -- lines 776-793."""

    def test_creates_config_with_problemtype_dict(self, tmp_path, monkeypatch):
        """Lines 782-793: problemTypeDict provided -- ContractionsProblemType.FromOriginalState used."""
        _set_all_gp(monkeypatch)

        lib_root = tmp_path / "libroot"
        lib_root.mkdir(parents=True)
        lib_inner = lib_root / "library"
        lib_inner.mkdir(parents=True)
        (lib_inner / "kernel.co").write_text("fake")

        out_file = str(tmp_path / "Config.ini")

        with patch.object(CW, "libraryDir", return_value=lib_inner):
            with patch.object(CW, "writeClientConfigIni") as mock_ini:
                CW.CreateBenchmarkClientParametersForSizes(
                    libraryRootPath=str(lib_root),
                    problemSizes=ProblemSizesMockDummy(),
                    dataFilePath="/data/Results.csv",
                    configFile=out_file,
                    deviceId=0,
                    gfxName="gfx942",
                    problemTypeDict=_PLAIN_GEMM_PT_DICT,
                )
        mock_ini.assert_called_once()
        call_args = mock_ini.call_args
        # First positional arg is forBenchmark=True
        assert call_args[0][0] == True

    def test_creates_config_from_metadata_file(self, tmp_path, monkeypatch):
        """Lines 785-791: no problemTypeDict -- reads metadata.yaml from library dir."""
        _set_all_gp(monkeypatch)

        lib_root = tmp_path / "libroot"
        lib_root.mkdir(parents=True)
        lib_inner = lib_root / "library"
        lib_inner.mkdir(parents=True)
        (lib_inner / "kernel.co").write_text("fake")

        import yaml
        metadata = {"ProblemType": _PLAIN_GEMM_PT_DICT}
        (lib_inner / "metadata.yaml").write_text(yaml.dump(metadata))

        out_file = str(tmp_path / "Config.ini")

        with patch.object(CW, "libraryDir", return_value=lib_inner):
            with patch.object(CW, "writeClientConfigIni") as mock_ini:
                CW.CreateBenchmarkClientParametersForSizes(
                    libraryRootPath=str(lib_root),
                    problemSizes=ProblemSizesMockDummy(),
                    dataFilePath="/data/Results.csv",
                    configFile=out_file,
                    deviceId=0,
                    gfxName="gfx942",
                    problemTypeDict=None,  # use metadata
                )
        mock_ini.assert_called_once()

    def test_exits_when_no_metadata_file(self, tmp_path, monkeypatch):
        """Lines 787-788: missing metadata.yaml triggers printExit -- SystemExit."""
        _set_all_gp(monkeypatch)

        lib_root = tmp_path / "libroot"
        lib_root.mkdir(parents=True)
        lib_inner = lib_root / "library"
        lib_inner.mkdir(parents=True)
        # No metadata.yaml created

        out_file = str(tmp_path / "Config.ini")

        with patch.object(CW, "libraryDir", return_value=lib_inner):
            with pytest.raises(SystemExit):
                CW.CreateBenchmarkClientParametersForSizes(
                    libraryRootPath=str(lib_root),
                    problemSizes=ProblemSizesMockDummy(),
                    dataFilePath="/data/Results.csv",
                    configFile=out_file,
                    deviceId=0,
                    gfxName="gfx942",
                    problemTypeDict=None,
                )


# ===========================================================================
# 12. problemSizeParams -- useE stride (line 491-492)
# ===========================================================================

class TestProblemSizeParamsUseE:
    """problemSizeParams useE branch -- lines 491-492."""

    def test_e_strides_added_when_use_e(self):
        """Lines 491-492: useE=True adds 'e-strides' equal to dstrides."""
        pt = _make_problem_type(_USE_E_PT_DICT)
        from Tensile.SolutionStructs.Problem import Problem

        # sizes=[M, N, batch, K] = 4 indices matching TotalIndices
        problem = Problem([128, 64, 1, 512])
        result = dict(CW.problemSizeParams(pt, problem, [0]))
        assert "e-strides" in result

    def test_no_e_strides_without_use_e(self):
        """No e-strides when useE is False."""
        pt = _make_problem_type(_PLAIN_GEMM_PT_DICT)
        from Tensile.SolutionStructs.Problem import Problem

        problem = Problem([128, 64, 1, 512])
        result = dict(CW.problemSizeParams(pt, problem, [0]))
        assert "e-strides" not in result


# ===========================================================================
# 13. DataInitName enum coverage (lines 53-80)
# ===========================================================================

class TestDataInitNameEnum:
    """DataInitName enum -- lines 53-80, exercised through DataInitName lookups."""

    def test_all_enum_values_have_names(self):
        """All defined enum values resolve to a string name."""
        for val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27]:
            name = CW.DataInitName(val).name
            assert isinstance(name, str) and len(name) > 0

    def test_specific_enum_names(self):
        """Check several specific enum names."""
        assert CW.DataInitName(0).name == "Zero"
        assert CW.DataInitName(3).name == "Random"
        assert CW.DataInitName(27).name == "UniformLowPrecision"


# ===========================================================================
# 14. ClientLogLevel enum (lines 82-86)
# ===========================================================================

class TestClientLogLevelEnum:
    """ClientLogLevel -- lines 82-86."""

    def test_all_log_level_values(self):
        """All log levels have expected names."""
        assert CW.ClientLogLevel(0).name == "Error"
        assert CW.ClientLogLevel(1).name == "Terse"
        assert CW.ClientLogLevel(2).name == "Verbose"
        assert CW.ClientLogLevel(3).name == "Debug"
