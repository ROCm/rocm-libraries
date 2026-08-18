################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, modify, merge, publish, distribute, sublicense, and/or sell
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

"""Characterization tests for TensileRetuneLibrary argument parsing and main flow."""

import pytest
import sys
import os
from pathlib import Path
from unittest import mock

import Tensile.TensileRetuneLibrary as TRL


pytestmark = pytest.mark.unit


def test_working_path_functions():
    """Test pushWorkingPath, popWorkingPath, ensurePath, setWorkingPath (lines 46-67)."""
    from Tensile.TensileRetuneLibrary import pushWorkingPath, popWorkingPath, setWorkingPath, ensurePath, globalParameters, workingDirectoryStack

    original_path = globalParameters.get("WorkingPath", "/tmp")
    globalParameters["WorkingPath"] = original_path

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test", "nested")
        result = ensurePath(test_path)
        assert os.path.exists(result)
        assert result == test_path

        # Test pushWorkingPath appends to path
        globalParameters["WorkingPath"] = tmpdir
        pushWorkingPath("subdir")
        assert globalParameters["WorkingPath"] == os.path.join(tmpdir, "subdir")

        # Test popWorkingPath removes last component
        popWorkingPath()
        assert globalParameters["WorkingPath"] == tmpdir

        # Test setWorkingPath saves old and sets new
        workingDirectoryStack.clear()
        setWorkingPath(os.path.join(tmpdir, "another"))
        assert globalParameters["WorkingPath"] == os.path.join(tmpdir, "another")
        assert len(workingDirectoryStack) == 1

    globalParameters["WorkingPath"] = original_path


def test_TensileRetuneLibrary_invalid_args():
    """Test TensileRetuneLibrary with invalid arguments (line 165)."""
    # Missing required arguments should fail during argparse
    with pytest.raises(SystemExit):
        TRL.TensileRetuneLibrary([])


def test_TensileRetuneLibrary_help_arg():
    """Test TensileRetuneLibrary --help (line 165 argparse)."""
    with pytest.raises(SystemExit) as exc_info:
        TRL.TensileRetuneLibrary(["--help"])
    assert exc_info.value.code == 0


def test_main_function_delegates():
    """Test that main() calls TensileRetuneLibrary (line 236)."""
    with mock.patch('Tensile.TensileRetuneLibrary.TensileRetuneLibrary') as mock_trl:
        orig_argv = sys.argv
        try:
            sys.argv = ["prog", "dummy_lib", "dummy_out"]
            TRL.main()
            # Verify TensileRetuneLibrary was called with sys.argv[1:]
            mock_trl.assert_called_once()
            args = mock_trl.call_args[0][0]
            assert args == ["dummy_lib", "dummy_out"]
        finally:
            sys.argv = orig_argv
