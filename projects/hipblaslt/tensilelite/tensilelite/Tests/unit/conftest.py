"""pytest configuration and CLI options for store-D unit tests."""

import os
import sys

import pytest

# characterization/_codegen has no __init__.py, so tests that reuse its
# config_harness (the shared config -> Solution/assembly emit helpers) need the
# directory itself on sys.path to import it bare.
_CODEGEN_DIR = os.path.join(os.path.dirname(__file__), "characterization", "_codegen")
if _CODEGEN_DIR not in sys.path:
    sys.path.insert(0, _CODEGEN_DIR)

from streamk5_test_helpers import mock_streamk_writer  # noqa: F401
from tensilelite.Tests.gpu_detection import arch_matches, detect_gpu_target, hip_available
from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Skip ``gpu``-marked tests the GPU can't run.

    ``gpu`` = any GPU; ``gpu("gfx950", ...)`` = one of those (``gfx1250`` covers
    ``gfx1250-strict``). Nearest mark wins: function/param > class > module.
    Don't mark both a function and its params; the function's mark wins.
    Runs after ``-m`` deselection, so ``-m "not gpu"`` never probes the GPU.
    """
    for item in items:
        if not str(item.path).startswith(_UNIT_DIR + os.sep):
            continue
        mark = item.get_closest_marker("gpu")
        if mark is None:
            continue
        if not hip_available():
            item.add_marker(pytest.mark.skip(reason="needs hip-python"))
            continue
        found = detect_gpu_target()
        if found is None:
            item.add_marker(pytest.mark.skip(reason="needs a GPU (none visible to HIP)"))
        elif mark.args and not any(arch_matches(found, a) for a in mark.args):
            item.add_marker(pytest.mark.skip(reason=f"needs {'/'.join(mark.args)}, found {found}"))


@pytest.fixture(autouse=True)
def _isolate_rocisa_state():
    with preserve_rocisa_kernel_state():
        yield


@pytest.fixture(autouse=True)
def skip_parametrized_if_cli(request):
    """Skip all tests except test_storeD_cli when CLI options (--mn) are provided."""
    if (request.config.getoption("--mn", default=None) is not None
            and request.function.__name__ != "test_storeD_cli"):
        pytest.skip("--mn specified: only test_storeD_cli runs")


def pytest_addoption(parser):
    parser.addoption(
        "--mn", nargs="+", metavar="M,N",
        help="List of M,N pairs to test, e.g. --mn 23,17 32,32 16,16",
    )
    parser.addoption(
        "--mt", nargs="+", metavar="MT0,MT1",
        help="List of MacroTile pairs to test, e.g. --mt 16,16 32,32",
    )
    parser.addoption(
        "--wave-config", nargs="+", metavar="WG0,WG1",
        help="List of MIWaveGroup pairs to test, e.g. --wave-config 1,1 2,2",
    )
    parser.addoption(
        "--dump-asm", action="store_true", default=False,
        help="Dump generated assembly and store module text for each test case",
    )
    parser.addoption(
        "--dump-store-insts", action="store_true", default=False,
        help="Print only the buffer_store_* instructions emitted by the store-D path, "
             "one per line with the preceding comment for context",
    )
    parser.addoption(
        "--asm-output-dir", default=None, metavar="DIR",
        help="Write the full assembled kernel source (.s file) for each CLI test case "
             "to DIR/test_<label>.s for offline inspection",
    )
    parser.addoption(
        "--print-ref", action="store_true", default=False,
        help="Print the reference C matrix alongside the D output (only applies to "
             "random init mode where a reference matrix is available)",
    )
    parser.addoption(
        "--subtile-map", action="store_true", default=False,
        help="Print one value per MMA subtile (0,0) instead of the full matrix — "
             "useful for large tiles where the full print is unwieldy",
    )
    parser.addoption(
        "--init-mode", default="matrix", choices=["matrix", "wave_id", "random"],
        help="Accvgpr initialisation mode for the CLI test: "
             "'matrix' (default) loads a full M×N host matrix; "
             "'wave_id' writes float(wave_id) to every accvgpr (no verification); "
             "'random' loads uniform random values in [-9, 9] per thread per accvgpr (no verification).",
    )
    parser.addoption(
        "--dtype", default="fp32", choices=["fp32", "bf16"],
        help="Output destination data type for the CLI test: "
             "'fp32' (default) or 'bf16' (uses HighPrecisionAccumulate with bf16 conversion).",
    )
