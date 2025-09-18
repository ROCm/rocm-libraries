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

@pytest.fixture
def mock_openFile():
    with patch("builtins.open", mock_open()) as mock:
        yield mock

@pytest.fixture
def mock_exists():
    with patch.object(Path, "exists", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_notExists():
    with patch.object(Path, "exists", return_value=False) as mock:
        yield mock

# Test cases for isRhel8
def test_isRhel8_true(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.4"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_non_rhel(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Ubuntu" VERSION_ID="20.04"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_new_version(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="9.0"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_true_with_warning(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.5"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_file_not_found(mock_notExists, mock_openFile):
    assert isRhel8() is False
    mock_notExists.assert_called_once_with()
    mock_openFile.assert_not_called()  # No file open attempt if the file doesn't exist
