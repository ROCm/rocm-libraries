"""pytest configuration and CLI options for store-D unit tests."""

import pytest


@pytest.fixture
def _force_strict_gate_on(monkeypatch):
    """Force _STRICT_GATE_ENABLED True for tests that exercise the strict gate.

    The kill switch lives at module scope in
    Tensile/Common/TypeValidationErrors.py and is False so the
    validation infrastructure can land inert in the first commit.
    Tests that exercise gate-dependent validators opt in via
    ``pytestmark`` or by requesting this fixture directly.

    This fixture is removed in the same commit that removes the
    kill switch.
    """
    monkeypatch.setattr(
        "Tensile.Common.TypeValidationErrors._STRICT_GATE_ENABLED",
        True,
    )


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
