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

        with patch.object(M, "createLibrary") as mock_create:
            M.createLibraryForBenchmark(logic_path, lib_path, "gfx1250-strict")

            mock_create.assert_called_once()
            cmd = mock_create.call_args.args[0]

            # Verify command structure
            assert len(cmd) == 8
            assert "--new-client-only" in cmd
            assert "--no-short-file-names" in cmd
            # Not "all": that is expanded from the supported ISAs, so it cannot
            # name a stepping that shares another architecture's ISA, and the
            # library would be built somewhere this does not read it back from.
            assert "--architecture=gfx1250-strict" in cmd
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
        for error in (RuntimeError("handler failed"), OSError("File not found"), SystemExit(1)):
            with (
                patch.object(M, "createLibrary", side_effect=error),
                patch.object(M, "printExit") as mock_exit,
            ):
                M.createLibraryForBenchmark(logic_path, lib_path, "gfx942")
                mock_exit.assert_called_once()

def test_main_parses_paths_and_returns_zero():
    with patch.object(M, "GenerateSummations") as generate_summations:
        assert M.main(["logic", "output"]) == 0

    generate_summations.assert_called_once_with(["logic", "output"])
