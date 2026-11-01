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
from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state


@pytest.fixture(autouse=True)
def _isolate_rocisa_state():
    with preserve_rocisa_kernel_state():
        yield


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_parallel_map():
    """Run the unit tests' ``ParallelMap``/``ParallelMap2`` calls in-process.

    Why here: ``CPUThreadCount()`` sizes its pool from ``globalParameters["CpuThreads"]``
    (default -1, meaning every CPU the process can see). That is the machine's CPU count,
    not the CI job's CPU allocation, so each pytest/xdist process launches a pool of
    roughly that many workers (96 on a 96-CPU host) for tests that map over one or a few
    items. The pool never speeds those tests up, and it multiplies CPU use by the number
    of xdist workers. A session-scoped autouse fixture in this conftest runs once in every
    process that runs unit tests, each xdist worker included, so no test or CLI flag has to
    opt in.

    Why a fixture and not a ``pytest_configure`` hook: pytest runs a nested conftest's
    ``pytest_configure`` as soon as it imports that conftest during collection, so a run
    that collects all of ``Tests`` (``tox -e py3 -- tensilelite/Tests -m common``) would
    also have set ``CpuThreads`` for the non-unit tests that call ``Tensile`` in-process.
    Autouse fixtures only reach tests under ``unit/``, and the patches are undone when the
    session ends.

    Why both values: ``restoreDefaultGlobalParameters()``, called by the ``Tensile``,
    ``TensileUpdateLibrary`` and ``TensileRetuneLibrary`` entry points the tests exercise,
    clears ``globalParameters`` and refills it from ``defaultGlobalParameters``. Setting
    only the live value is undone by the first such call and the pools come back.
    A test that needs a different value sets it for itself, as several already do. The
    real worker-pool path stays covered by the device-library builds, by the
    subprocess-based ``Tests/common/test_config.py`` runs, and by one small test in
    ``characterization/Parallel``.
    """
    from tensilelite.Common import GlobalParameters

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(GlobalParameters.globalParameters, "CpuThreads", 1)
        patch.setitem(GlobalParameters.defaultGlobalParameters, "CpuThreads", 1)
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
