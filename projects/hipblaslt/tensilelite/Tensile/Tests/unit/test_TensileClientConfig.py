################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import sys
import tempfile
import os


# Lazy import to avoid module-level import errors
@pytest.fixture(scope="module")
def tcc():
    """Lazy import TensileClientConfig to handle import dependencies"""
    import Tensile.TensileClientConfig
    return Tensile.TensileClientConfig


@pytest.mark.unit
class TestGetGlobalParams:
    """Tests for getGlobalParams function"""

    def test_with_valid_global_parameters(self, tcc):
        """Test getGlobalParams with valid GlobalParameters"""
        config = {
            "GlobalParameters": {
                "MinimumRequiredVersion": "4.0.0",
                "PrintLevel": 1
            }
        }

        result = tcc.getGlobalParams(config)

        assert result is not None
        assert result == config["GlobalParameters"]
        assert "MinimumRequiredVersion" in result

    def test_with_missing_global_parameters(self, tcc):
        """Test getGlobalParams when GlobalParameters key is missing"""
        config = {
            "SomeOtherKey": "value"
        }

        result = tcc.getGlobalParams(config)

        assert result is None

    def test_with_none_config(self, tcc):
        """Test getGlobalParams with None config"""
        result = tcc.getGlobalParams(None)
        assert result is None

    def test_with_empty_dict(self, tcc):
        """Test getGlobalParams with empty dict"""
        result = tcc.getGlobalParams({})
        assert result is None

    def test_with_non_dict_config(self, tcc):
        """Test getGlobalParams with non-dict config"""
        result = tcc.getGlobalParams("not a dict")
        assert result is None

        result = tcc.getGlobalParams([1, 2, 3])
        assert result is None


@pytest.mark.unit
class TestGetProblemDict:
    """Tests for getProblemDict function"""

    def test_from_benchmark_problems(self, tcc):
        """Test getProblemDict from BenchmarkProblems structure"""
        config = {
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM", "DataType": "d"},
                    {"BenchmarkFinalParameters": []}
                ]
            ]
        }

        result = tcc.getProblemDict(config)

        assert result is not None
        assert result == {"OperationType": "GEMM", "DataType": "d"}

    def test_from_benchmark_problems_multiple_warns(self, tcc):
        """Test getProblemDict warns when multiple BenchmarkProblems exist"""
        config = {
            "BenchmarkProblems": [
                [{"OperationType": "GEMM"}, {}],
                [{"OperationType": "GEMM2"}, {}]
            ]
        }

        with patch("Tensile.TensileClientConfig.printWarning") as mock_warn:
            result = tcc.getProblemDict(config)

            assert result == {"OperationType": "GEMM"}
            mock_warn.assert_called_once()
            assert "More than one" in mock_warn.call_args[0][0]

    def test_from_problem_type(self, tcc):
        """Test getProblemDict from ProblemType key"""
        config = {
            "ProblemType": {
                "OperationType": "GEMM",
                "DataType": "s"
            }
        }

        result = tcc.getProblemDict(config)

        assert result is not None
        assert result == config["ProblemType"]

    def test_benchmark_problems_takes_precedence(self, tcc):
        """Test that BenchmarkProblems takes precedence over ProblemType"""
        config = {
            "BenchmarkProblems": [
                [{"OperationType": "GEMM_BP"}, {}]
            ],
            "ProblemType": {"OperationType": "GEMM_PT"}
        }

        result = tcc.getProblemDict(config)

        # BenchmarkProblems is checked first and should be returned
        assert result == {"OperationType": "GEMM_BP"}

    def test_with_no_problem_info(self, tcc):
        """Test getProblemDict when no problem info exists"""
        config = {"SomeOtherKey": "value"}

        result = tcc.getProblemDict(config)

        assert result is None

    def test_with_none_config(self, tcc):
        """Test getProblemDict with None config"""
        result = tcc.getProblemDict(None)
        assert result is None

    def test_with_invalid_benchmark_problems_structure(self, tcc):
        """Test getProblemDict with malformed BenchmarkProblems"""
        config = {
            "BenchmarkProblems": [[]]
        }

        result = tcc.getProblemDict(config)

        # Should fall through to checking ProblemType, which doesn't exist
        assert result is None


@pytest.mark.unit
class TestGetSizeList:
    """Tests for getSizeList function"""

    def test_from_benchmark_problems(self, tcc):
        """Test getSizeList from BenchmarkProblems structure"""
        size_list = [
            {"Exact": [256, 256, 1, 256]},
            {"Exact": [512, 512, 1, 512]}
        ]
        config = {
            "BenchmarkProblems": [
                [
                    {},
                    {
                        "BenchmarkFinalParameters": [
                            {"ProblemSizes": size_list}
                        ]
                    }
                ]
            ]
        }

        result = tcc.getSizeList(config)

        assert result is not None
        assert result == size_list
        assert len(result) == 2

    def test_config_is_size_list(self, tcc):
        """Test getSizeList when config itself is a size list"""
        size_list = [
            {"Exact": [128, 128, 1, 128]},
            {"Range": [[64, 256], [64, 256], [1], [64, 256]]}
        ]

        result = tcc.getSizeList(size_list)

        assert result is not None
        assert result == size_list

    def test_invalid_size_list_missing_exact_or_range(self, tcc):
        """Test getSizeList rejects list without Exact or Range"""
        config = [
            {"SomeKey": [128, 128, 1, 128]},
            {"OtherKey": [256, 256, 1, 256]}
        ]

        result = tcc.getSizeList(config)

        assert result is None

    def test_size_list_with_exact_only(self, tcc):
        """Test getSizeList with only Exact entries"""
        config = [
            {"Exact": [64, 64, 1, 64]},
            {"Exact": [128, 128, 1, 128]}
        ]

        result = tcc.getSizeList(config)

        assert result == config

    def test_size_list_with_range_only(self, tcc):
        """Test getSizeList with only Range entries"""
        config = [
            {"Range": [[64, 256], [64, 256], [1], [64]]}
        ]

        result = tcc.getSizeList(config)

        assert result == config

    def test_size_list_mixed_exact_and_range(self, tcc):
        """Test getSizeList with mixed Exact and Range"""
        config = [
            {"Exact": [64, 64, 1, 64]},
            {"Range": [[64, 256], [64, 256], [1], [64]]}
        ]

        result = tcc.getSizeList(config)

        assert result == config

    def test_non_list_config(self, tcc):
        """Test getSizeList with non-list config"""
        result = tcc.getSizeList({"NotAList": "value"})
        assert result is None

    def test_none_config(self, tcc):
        """Test getSizeList with None config"""
        result = tcc.getSizeList(None)
        assert result is None

    def test_empty_list(self, tcc):
        """Test getSizeList with empty list"""
        result = tcc.getSizeList([])
        # Empty list is technically a valid size list (though useless)
        assert result == []

    def test_list_with_non_dict_elements(self, tcc):
        """Test getSizeList rejects list with non-dict elements"""
        config = [
            {"Exact": [64, 64, 1, 64]},
            "not a dict"
        ]

        result = tcc.getSizeList(config)

        assert result is None


@pytest.mark.unit
class TestParseConfig:
    """Tests for parseConfig function"""

    def test_parse_complete_config(self, tcc):
        """Test parseConfig with all components present"""
        config = {
            "GlobalParameters": {"MinimumRequiredVersion": "4.0.0"},
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM", "DataType": "d"},
                    {
                        "BenchmarkFinalParameters": [
                            {
                                "ProblemSizes": [
                                    {"Exact": [256, 256, 1, 256]}
                                ]
                            }
                        ]
                    }
                ]
            ]
        }

        globalParams, problemDict, sizeList = tcc.parseConfig(config)

        assert globalParams == {"MinimumRequiredVersion": "4.0.0"}
        assert problemDict == {"OperationType": "GEMM", "DataType": "d"}
        assert sizeList == [{"Exact": [256, 256, 1, 256]}]

    def test_parse_config_missing_global_params(self, tcc):
        """Test parseConfig when GlobalParameters is missing"""
        config = {
            "ProblemType": {"OperationType": "GEMM"},
        }

        globalParams, problemDict, sizeList = tcc.parseConfig(config)

        assert globalParams is None
        assert problemDict == {"OperationType": "GEMM"}
        assert sizeList is None

    def test_parse_config_only_size_list(self, tcc):
        """Test parseConfig when config is just a size list"""
        config = [
            {"Exact": [128, 128, 1, 128]},
            {"Exact": [256, 256, 1, 256]}
        ]

        globalParams, problemDict, sizeList = tcc.parseConfig(config)

        assert globalParams is None
        assert problemDict is None
        assert sizeList == config

    def test_parse_empty_config(self, tcc):
        """Test parseConfig with empty config"""
        globalParams, problemDict, sizeList = tcc.parseConfig({})

        assert globalParams is None
        assert problemDict is None
        assert sizeList is None

    def test_parse_none_config(self, tcc):
        """Test parseConfig with None"""
        globalParams, problemDict, sizeList = tcc.parseConfig(None)

        assert globalParams is None
        assert problemDict is None
        assert sizeList is None
