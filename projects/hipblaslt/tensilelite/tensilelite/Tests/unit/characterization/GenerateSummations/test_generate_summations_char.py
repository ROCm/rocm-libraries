################################################################################
# Characterization tests for tensilelite.GenerateSummations — summation model fitting.
#
# Characterization tests for the GenerateSummations create-library dispatch.
# CSV parsing behavior is covered separately by the focused csv/NumPy unit test.
################################################################################
import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# Import the real production module; it has no optional dataframe dependency.
try:
    M = importlib.import_module("tensilelite.GenerateSummations")
    _PANDAS_AVAILABLE = True
except ImportError as e:
    if "numpy" in str(e):
        M = None
        _PANDAS_AVAILABLE = False
    else:
        raise


# ---------------------------------------------------------------------------
# Test: createLibraryForBenchmark in-process dispatch
# ---------------------------------------------------------------------------
@pytest.mark.skipif(M is None, reason="Module import failed")
def test_create_library_for_benchmark_success():
    """
    Pin that createLibraryForBenchmark constructs the canonical argument list
    and invokes the in-process create-library API.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")
        current_path = str(tmpdir / "work")

        with patch.object(M, "createLibrary") as create_library:
            M.createLibraryForBenchmark(logic_path, lib_path, current_path)
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
@pytest.mark.skipif(M is None, reason="Module import failed")
def test_create_library_for_benchmark_error_handling():
    """
    Pin that create-library errors are caught and handled.
    This exercises lines 60–63 (the try/except block).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logic_path = str(tmpdir / "logic")
        lib_path = str(tmpdir / "lib")
        current_path = str(tmpdir)

        with patch.object(M, "createLibrary", side_effect=RuntimeError("failed")):
            with pytest.raises(SystemExit):
                M.createLibraryForBenchmark(logic_path, lib_path, current_path)

        # Test OSError
        with patch.object(M, "createLibrary", side_effect=OSError("File not found")):
            with pytest.raises(SystemExit):
                M.createLibraryForBenchmark(logic_path, lib_path, current_path)



