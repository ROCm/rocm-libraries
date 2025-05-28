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
    _extractArchInfo,
    filterLogicFilesByArchPredicates,
    splitArchsFromPredicates,
    verifyPredicate,
    ArchInfo,
    LogicFileError,
    ARCH_DEVICE_ID_FALLBACKS,
    SUPPORTED_ARCH_DEVICE_IDS,
)

# Test data
VALID_LOGIC_FILE_CONTENT = """- {MinimumRequiredVersion: 4.33.0}
- gfx950 
- gfx950
- [Device 75a2]
"""

VALID_LOGIC_FILE_WITH_CU = """- {MinimumRequiredVersion: 4.33.0}
- aquavanjaram
- {Architecture: gfx942, CUCount: 256}
- [Device 74a3]
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
    assert result.CUCount == "cu=256"
    assert result.DeviceIds == {"id=74a3"}

def test_extractArchInfo_with_invalid_version(mock_logic_file_invalid_version):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_extractArchInfo_with_invalid_arch(mock_logic_file_invalid_arch):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_extractArchInfo_with_invalid_device(mock_logic_file_invalid_device):
    with pytest.raises(LogicFileError):
        _extractArchInfo("dummy.yaml")

def test_filterLogicFiles_exact_match(mock_logic_file):
    logicFiles = ["file1.yaml", "file2.yaml"]
    archs = {"gfx950"}
    deviceIds = {"id=75a0"}
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=75a0"})
        result = filterLogicFilesByArchPredicates(logicFiles, archs, deviceIds)
        assert len(result) == 2
        assert "file1.yaml" in result
        assert "file2.yaml" in result
        assert all(isinstance(r, str) for r in result)

def test_filterLogicFiles_fallback_match(mock_logic_file):
    logicFiles = ["file1.yaml", "file2.yaml"]
    archs = {"gfx950"}
    deviceIds = {"id=75a2"}  # Should fallback to 75a0
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=75a0"})
        result = filterLogicFilesByArchPredicates(logicFiles, archs, deviceIds)
        assert len(result) == 2
        assert "file1.yaml" in result
        assert "file2.yaml" in result
        assert all(isinstance(r, str) for r in result)

def test_filterLogicFiles_no_match(mock_logic_file):
    logicFiles = ["file1.yaml"]
    archs = {"gfx950"}
    deviceIds = {"id=75a0"}
    
    with patch("Tensile.Common.Architectures._extractArchInfo") as mock_extract:
        mock_extract.return_value = ArchInfo("test", "gfx950", {"id=75a3"})
        result = filterLogicFilesByArchPredicates(logicFiles, archs, deviceIds)
        assert len(result) == 0

def test_splitArchsFromPredicates_with_variants():
    archSpecs = ["gfx950[id=75a0]", "gfx906"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert archs == ["gfx950", "gfx906"]
    assert variants == {"id=75a0"}

def test_splitArchsFromPredicates_with_multiple_variants():
    archSpecs = ["gfx950[id=75a0,id=75a2]", "gfx942[id=74a2,id=74a3]"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert archs == ["gfx950", "gfx942"]
    assert variants == {"id=75a0", "id=75a2", "id=74a2", "id=74a3"}

def test_splitArchsFromPredicates_no_variants():
    archSpecs = ["gfx950", "gfx906"]
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert archs == ["gfx950", "gfx906"]
    assert variants is None

def test_splitArchsFromPredicates_empty():
    archSpecs = []
    archs, variants = splitArchsFromPredicates(archSpecs)
    assert archs == []
    assert variants is None

def test_verifyPredicate_valid():
    for device_id in SUPPORTED_ARCH_DEVICE_IDS:
        assert verifyPredicate(device_id) == device_id

def test_verifyPredicate_invalid():
    with pytest.raises(ValueError) as exc_info:
        verifyPredicate("id=invalid")
    assert "device ID not supported" in str(exc_info.value)

def test_verifyPredicate_non_id():
    with pytest.raises(ValueError, match=r"Invalid predicate: only device ID-based predicates are currently supported: (.*)"):
        verifyPredicate("cu=304")

def test_verifyPredicate_invalid_predicate():
    with pytest.raises(ValueError, match=r"Invalid predicate: only device ID-based predicates are currently supported: (.*)"):
        verifyPredicate("invalid=value")

