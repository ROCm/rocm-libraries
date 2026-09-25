################################################################################
# Characterization tests for tensilelite.GenerateSummations — summation model fitting.
#
# Characterization tests for the GenerateSummations create-library dispatch.
# CSV parsing behavior is covered separately by the focused csv/NumPy unit test.
################################################################################
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


from tensilelite import GenerateSummations as M


# ---------------------------------------------------------------------------
# Test: createLibraryForBenchmark in-process dispatch
# ---------------------------------------------------------------------------
def test_create_library_for_benchmark_success():
    """
    Pin that createLibraryForBenchmark constructs the canonical argument list
    and invokes the in-process create-library API.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")

        with patch.object(M, "createLibrary") as create_library:
            M.createLibraryForBenchmark(logic_path, lib_path)
            create_library.assert_called_once()
            cmd = create_library.call_args.args[0]

            # Verify command structure
            assert len(cmd) == 6
            assert "--architecture=all" in cmd
            assert "--code-object-version=default" in cmd
            assert "--library-format=yaml" in cmd
            assert logic_path in cmd
            assert lib_path in cmd
            assert "HIP" in cmd


# ---------------------------------------------------------------------------
# Test: createLibraryForBenchmark API error handling
# ---------------------------------------------------------------------------
def test_create_library_for_benchmark_error_handling():
    """
    Pin that create-library errors are caught and handled.
    This exercises lines 60–63 (the try/except block).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")

        for error in (RuntimeError("failed"), OSError("File not found"), SystemExit(1)):
            with patch.object(M, "createLibrary", side_effect=error), pytest.raises(SystemExit):
                M.createLibraryForBenchmark(logic_path, lib_path)


def test_main_parses_paths_and_returns_zero():
    with patch.object(M, "GenerateSummations") as generate_summations:
        assert M.main(["logic", "output"]) == 0

    generate_summations.assert_called_once_with(["logic", "output"])
