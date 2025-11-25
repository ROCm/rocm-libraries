################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
from unittest.mock import mock_open, patch
from pathlib import Path
from Tensile.Common.Utilities import isRhel8
from Tensile.Common.Architectures import (
    SUPPORTED_ARCH_DEVICE_IDS,
    SUPPORTED_ARCH_CU_COUNTS,
    ARCH_DEVICE_ID_FALLBACKS,
    ArchInfo,
    splitArchsFromPredicates,
    LogicFileError,
    filterLogicFilesByPredicates,
    _extractArchInfo,
    _verifyPredicate,
)

# Test data
VALID_LOGIC_FILE_CONTENT = """- {MinimumRequiredVersion: 4.33.0}
- gfx950 
- gfx950
- [Device 75a2]
"""

VALID_LOGIC_FILE_WITH_CU = """- {MinimumRequiredVersion: 4.33.0}
- aquavanjaram
- {Architecture: gfx942, CUCount: 228}
- [Device 74a0]
"""

INVALID_VERSION_FILE = """- Invalid Version Line
- aquavanjaram
- gfx950
- [Device 75a0]
"""

INVALID_ARCH_FILE = """- {MinimumRequiredVersion: 4.33.0}
- aquavanjaram
- invalid_arch_line
- [Device 75a0]
"""

INVALID_DEVICE_FILE = """- {MinimumRequiredVersion: 4.33.0}
- aquavanjaram
- gfx950
- invalid_device_line
"""

@pytest.fixture
def mock_logic_file():
    with patch("builtins.open", mock_open(read_data=VALID_LOGIC_FILE_CONTENT)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_logic_file_with_cu():
    with patch("builtins.open", mock_open(read_data=VALID_LOGIC_FILE_WITH_CU)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_logic_file_invalid_version():
    with patch("builtins.open", mock_open(read_data=INVALID_VERSION_FILE)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_logic_file_invalid_arch():
    with patch("builtins.open", mock_open(read_data=INVALID_ARCH_FILE)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_logic_file_invalid_device():
    with patch("builtins.open", mock_open(read_data=INVALID_DEVICE_FILE)) as mock_file:
        yield mock_file

def test_extractArchInfo_success(mock_logic_file):
    result = _extractArchInfo("dummy.yaml")
    assert isinstance(result, ArchInfo)
    assert result.Name == "gfx950"
    assert result.Gfx == "gfx950"
    assert result.DeviceIds == {"id=75a2"}
    assert result.CUCount is None

def test_extractArchInfo_with_cu_count(mock_logic_file_with_cu):
    result = _extractArchInfo("dummy.yaml")
    assert isinstance(result, ArchInfo)
    assert result.Name == "aquavanjaram"
    assert result.Gfx == "gfx942"
    assert result.DeviceIds == {"id=74a0"}
    assert result.CUCount == "cu=228"

def test_extractArchInfo_with_invalid_version(mock_logic_file_invalid_version):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_extractArchInfo_with_invalid_arch(mock_logic_file_invalid_arch):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_extractArchInfo_with_invalid_device(mock_logic_file_invalid_device):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_filterLogicFilesByPredicates_fallback_id_match(mock_logic_file):
    logicFiles = ["file1.yaml", "file2.yaml"]
    predicateMap = {
        "gfx950": {
            "id=75a2": set(),  # Should fallback to 75a0
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=75a0"}, None)
        result = filterLogicFilesByPredicates(logicFiles, predicateMap)
        assert len(result) == 2
        assert "file1.yaml" in result
        assert "file2.yaml" in result
        assert all(isinstance(r, str) for r in result)

def test_filterLogicFilesByPredicates_cu_match(mock_logic_file):
    logicFiles = ["file1.yaml"]
    predicateMap = {
        "gfx942": {
            "cu=80": set(),
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx942", {"id=74a0"}, "cu=80")
        result = filterLogicFilesByPredicates(logicFiles, predicateMap)
        assert len(result) == 1
        assert "file1.yaml" in result

def test_filterLogicFilesByPredicates_cu_fallback(mock_logic_file):
    logicFiles = ["file1.yaml"]
    predicateMap = {
        "gfx942": {
            "cu=96": set(),  # Should fallback to None
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx942", {"id=74a0"}, None)
        result = filterLogicFilesByPredicates(logicFiles, predicateMap)
        assert len(result) == 1
        assert "file1.yaml" in result

def test_filterLogicFilesByPredicates_mixed_match_fallback(mock_logic_file):
    logicFiles = ["file1.yaml", "file2.yaml"]
    predicateMap = {
        "gfx942": {
            "cu=96": set(),  # Should fallback to None
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract1:
        mock_extract1.return_value = ArchInfo("test", "gfx942", {"id=74a0"}, None)
        with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract2:
            mock_extract2.return_value = ArchInfo("test", "gfx942", {"id=74a0"}, "cu=96")
            result = filterLogicFilesByPredicates(logicFiles, predicateMap)
            assert len(result) == 2
            assert "file1.yaml" in result
            assert "file2.yaml" in result

def test_filterLogicFilesByPredicates_no_match(mock_logic_file):
    logicFiles = ["file1.yaml"]
    predicateMap = {
        "gfx950": {
            "id=75a0": set(),
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=75a3"}, None)
        result = filterLogicFilesByPredicates(logicFiles, predicateMap)
        assert len(result) == 0

def test_filterLogicFilesByPredicates_match_emulation_ids(mock_logic_file):
    logicFiles = ["file1.yaml"]
    predicateMap = {
        "gfx950": {
            "id=75a0": set(),
            "fallback": set()
        }
    }
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=0049"}, None)
        result = filterLogicFilesByPredicates(logicFiles, predicateMap)
        assert len(result) == 1
        assert "file1.yaml" in result

def test_splitArchsFromPredicates_with_variants():
    archSpecs = ["gfx950[id=75a0]", "gfx906"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert "gfx906" in archs and "gfx950" in archs
    assert variants["gfx950"] == ["id=75a0"]

def test_splitArchsFromPredicates_with_multiple_variants():
    archSpecs = ["gfx950[id=75a0,id=75a2]", "gfx942"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert "gfx942" in archs and "gfx950" in archs
    assert set(variants["gfx950"]) == {"id=75a0", "id=75a2"}

def test_splitArchsFromPredicates_with_mixed_predicates():
    archSpecs = ["gfx942[id=74a0,cu=80]", "gfx950[id=75a0]"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert "gfx942" in archs and "gfx950" in archs
    assert set(variants["gfx942"]) == {"id=74a0", "cu=80"}
    assert variants["gfx950"] == ["id=75a0"]

def test_splitArchsFromPredicates_no_variants():
    archSpecs = ["gfx950", "gfx906"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert "gfx906" in archs and "gfx950" in archs
    assert variants is None

def test_splitArchsFromPredicates_empty():
    archSpecs = []
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert archs == []
    assert variants is None

def test_verifyPredicate_valid_device_id():
    for device_id in SUPPORTED_ARCH_DEVICE_IDS:
        arch = SUPPORTED_ARCH_DEVICE_IDS[device_id]
        assert _verifyPredicate(device_id, arch) == device_id

def test_verifyPredicate_valid_cu_count():
    for cu_count in SUPPORTED_ARCH_CU_COUNTS:
        arch = SUPPORTED_ARCH_CU_COUNTS[cu_count]
        assert _verifyPredicate(cu_count, arch) == cu_count

def test_verifyPredicate_invalid_device_id():
    with pytest.raises(ValueError) as exc_info:
        _verifyPredicate("id=invalid", "gfx950")
    assert "device ID not supported" in str(exc_info.value)

def test_verifyPredicate_mismatched_device_id():
    with pytest.raises(ValueError) as exc_info:
        _verifyPredicate("id=75a0", "gfx942")  # 75a0 is for gfx950
    assert "not associated with" in str(exc_info.value)

def test_verifyPredicate_invalid_cu_count():
    with pytest.raises(ValueError) as exc_info:
        _verifyPredicate("cu=999", "gfx942")
    assert "CU count not supported" in str(exc_info.value)

def test_verifyPredicate_mismatched_cu_count():
    with pytest.raises(ValueError) as exc_info:
        _verifyPredicate("cu=80", "gfx950")  # cu=80 is for gfx942
    assert "not associated with" in str(exc_info.value)

def test_verifyPredicate_invalid_predicate():
    with pytest.raises(ValueError) as exc_info:
        _verifyPredicate("invalid=value", "gfx950")
    assert "only device ID and CU count-based predicates" in str(exc_info.value)

