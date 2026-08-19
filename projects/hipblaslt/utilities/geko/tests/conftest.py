# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest configuration and CLI options for geko tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from geko.constants import SUPPORTED_ARCH

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config) -> None:
    """Register custom pytest markers for this package."""
    config.addinivalue_line(
        "markers",
        "slow: long-running tests (subprocess e2e, GPU / hipBLASLt); skip with --skip-slow.",
    )
    config.addinivalue_line(
        "markers",
        "cg_integration: config_generator end-to-end (CLI or API; needs hipBLASLt + YAML).",
    )
    config.addinivalue_line(
        "markers",
        "cg_cli_guard: config_generator main() path validation only (no hipBLASLt).",
    )
    config.addinivalue_line(
        "markers",
        "cg_components: MIDesign / optimization params / fork_param_generator (needs TensileLite).",
    )
    config.addinivalue_line(
        "markers",
        "geko_bin: subprocess smoke tests for bin/geko (skip with pytest --skip-geko-bin).",
    )


def pytest_addoption(parser) -> None:
    """Add CLI options for hipBLASLt path, config paths, and optional skips."""
    parser.addoption(
        "--hipblaslt-path",
        action="store",
        default=None,
        help="Path to local hipBLASLt repository for integration tests",
    )
    parser.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to input YAML configuration file (for config_generator tests)",
    )
    parser.addoption(
        "--workload",
        action="store",
        default=None,
        help="Path to hipBLASLt YAML workload/log file (for configure tests)",
    )
    parser.addoption(
        "--hw",
        action="store",
        default="gfx950",
        choices=SUPPORTED_ARCH,
        help="Target GPU architecture for configure integration (scripts/configure.py --architecture).",
    )
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip tests marked @pytest.mark.slow (long e2e / GPU runs).",
    )
    parser.addoption(
        "--skip-geko-bin",
        action="store_true",
        default=False,
        help="Skip tests marked @pytest.mark.geko_bin (subprocess bin/geko smoke).",
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Apply optional skips for slow e2e tests and bin/geko subprocess smoke tests."""
    if config.getoption("--skip-slow"):
        skip = pytest.mark.skip(reason="skipped: --skip-slow")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip)

    if config.getoption("--skip-geko-bin"):
        skip = pytest.mark.skip(reason="skipped: --skip-geko-bin")
        for item in items:
            if item.get_closest_marker("geko_bin"):
                item.add_marker(skip)


@pytest.fixture
def hipblaslt_path(request):
    p = request.config.getoption("--hipblaslt-path")
    return str(Path(p).resolve()) if p is not None else None


@pytest.fixture
def config_path(request):
    return request.config.getoption("--config")


@pytest.fixture
def workload_path(request):
    return request.config.getoption("--workload")


@pytest.fixture
def hw_arch(request) -> str:
    """Architecture string passed to ``configure.py --architecture`` in integration tests."""
    return request.config.getoption("--hw")


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tensilelite_sys_path(hipblaslt_path):
    if not hipblaslt_path:
        pytest.skip("Requires --hipblaslt-path")
    if not (Path(hipblaslt_path) / "tensilelite").is_dir():
        pytest.skip(f"tensilelite not found under {hipblaslt_path}")
    pytest.importorskip(
        "tensilelite",
        reason="TensileLite must be installed in the active GEKO interpreter.",
    )
    pytest.importorskip(
        "rocisa",
        reason="rocisa must be importable in the active GEKO interpreter.",
    )
    yield
