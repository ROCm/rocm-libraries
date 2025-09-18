################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from Tensile.TensileCreateLib.KernelFileContext import (
    KernelFileContextManager,
    _closeKernelFiles,
    _openKernelFiles,
)


@pytest.fixture
def mock_openFile():
    mockFile = MagicMock()
    mockOpen = MagicMock(return_value=mockFile)
    with patch("builtins.open", mockOpen):
        yield mockOpen, mockFile


@pytest.fixture
def mock_pathResolve():
    with patch("pathlib.Path.resolve", return_value=Path("/fake/path")) as mockResolve:
        yield mockResolve


def test_openKernelFiles_withOnlyMergeFilesEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(0, True, False, Path("/some/path"), [])
    mock_openFile[0].assert_called()
    assert mock_openFile[0].call_args_list == [
        call(Path("/fake/path/Kernels.cpp"), "a", encoding="utf-8"),
        call(Path("/fake/path/Kernels.h"), "a", encoding="utf-8"),
    ]


def test_openKernelFiles_withOnlyLazyLoadingEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(0, False, True, Path("/some/path"), [])
    mock_openFile[0].assert_called()
    assert mock_openFile[0].call_args_list == [
        call(Path("/fake/path/Kernels.cpp"), "a", encoding="utf-8"),
        call(Path("/fake/path/Kernels.h"), "a", encoding="utf-8"),
    ]


def test_openKernelFiles_withMergeAndLazyEnabled(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(2, True, True, Path("/fake/path"), [])
    mock_openFile[0].assert_called()
    assert mock_openFile[0].call_args_list == [
        call(Path("/fake/path/Kernels.cpp"), "a", encoding="utf-8"),
        call(Path("/fake/path/Kernels.h"), "a", encoding="utf-8"),
    ]


def test_openKernelFiles_withAll(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(2, True, True, Path("/fake/path"), ["a_kernel"])
    mock_openFile[0].assert_called()
    assert mock_openFile[0].call_args_list == [
        call("a_kernel.cpp", "a", encoding="utf-8"),
        call("a_kernel.h", "a", encoding="utf-8"),
    ]


def test_openKernelFiles_withNumMergedFiles(mock_openFile, mock_pathResolve):
    x, y = _openKernelFiles(2, False, False, Path("/some/path"), ["kernel1.cpp", "kernel2.cpp"])
    assert mock_openFile[0].call_args_list == [
        call("kernel1.cpp", "a", encoding="utf-8"),
        call("kernel1.h", "a", encoding="utf-8"),
    ]


def test_openKernelFiles_noAction(mock_openFile, mock_pathResolve):
    _openKernelFiles(2, False, False, Path("/some/path"), [])
    mock_openFile[0].assert_not_called()


def test_closeKerneFiles(mock_openFile, mock_pathResolve):
    _, mock_file = mock_openFile
    _closeKernelFiles(mock_file, mock_file)
    assert mock_file.close.call_count == 2


def test_KernelFileContextManager_withOnlyLazyLoadingEnabled(mock_openFile, mock_pathResolve):
    with pytest.raises(ValueError, match="If lazy loading is enabled, merge files must be as well"):
        with KernelFileContextManager(True, False, 1, Path("/some/path"), []):
            pass


def test_KernelFileContextManager_withOnlyMergeFilesEnabled(mock_openFile, mock_pathResolve):
    with KernelFileContextManager(False, True, 1, Path("/some/path"), []):
        mock_openFile[0].assert_called()


def test_KernelFileContextManager_withNoMergeFilesAndNum(mock_openFile, mock_pathResolve):
    with KernelFileContextManager(False, False, 3, Path("/some/path"), ["a_kernel.cpp"]) as (x, y):
        assert x is None and y is None, "We expect nothing to be opened in this situation"


def test_KernelFileContextManager_withNoOutputPath(mock_openFile, mock_pathResolve):
    with pytest.raises(ValueError, match="Output path cannot be empty."):
        with KernelFileContextManager(False, False, 3, "", ["a_kernel.cpp"]):
            pass


def test_KernelFileContextManager_withNoKernelFiles(mock_openFile, mock_pathResolve):
    with pytest.raises(ValueError, match=r"(.*) Provide at least one kernel file."):
        with KernelFileContextManager(False, True, 3, Path("/fake/path"), []):
            pass
