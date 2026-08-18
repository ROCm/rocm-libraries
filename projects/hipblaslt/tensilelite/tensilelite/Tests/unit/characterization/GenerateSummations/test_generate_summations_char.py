################################################################################
# Characterization tests for tensilelite.GenerateSummations — summation model fitting.
#
# Characterization tests for the GenerateSummations benchmark-library wrapper.
################################################################################
import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


M = importlib.import_module("tensilelite.GenerateSummations")


# ---------------------------------------------------------------------------
# Test: createLibraryForBenchmark package-handler invocation
# ---------------------------------------------------------------------------
def test_create_library_for_benchmark_success():
    """
    Pin that createLibraryForBenchmark forwards the correct argument list to the
    package-local create-library handler.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")
        current_path = str(tmpdir / "work")

        with patch.object(M, "createLibrary") as mock_create:
            M.createLibraryForBenchmark(logic_path, lib_path, current_path)

            mock_create.assert_called_once()
            cmd = mock_create.call_args.args[0]

            # Verify command structure
            assert len(cmd) == 8
            assert "--new-client-only" in cmd
            assert "--no-short-file-names" in cmd
            assert "--architecture=all" in cmd
            assert "--code-object-version=default" in cmd
            assert "--library-format=yaml" in cmd
            assert os.path.abspath(logic_path) in cmd
            assert os.path.abspath(lib_path) in cmd
            assert "HIP" in cmd


# ---------------------------------------------------------------------------
# Test: createLibraryForBenchmark handler error handling
# ---------------------------------------------------------------------------
def test_create_library_for_benchmark_error_handling():
    """
    Pin that package-handler errors are caught and handled.
    This exercises lines 60–63 (the try/except block).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")
        current_path = str(tmpdir)

        for error in (RuntimeError("handler failed"), OSError("File not found"), SystemExit(1)):
            with patch.object(M, "createLibrary", side_effect=error), patch.object(M, "printExit") as mock_exit:
                M.createLibraryForBenchmark(logic_path, lib_path, current_path)
                mock_exit.assert_called_once()




