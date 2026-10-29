# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest configuration and CLI options for geko tests."""

from __future__ import annotations

import sys
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
        "cg_components: MIDesign / optimization params / fork_param_generator (needs Tensile).",
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
    tl = Path(hipblaslt_path) / "tensilelite"
    if not tl.is_dir():
        pytest.skip(f"tensilelite not found under {hipblaslt_path}")
    s = str(tl)
    sys.path.insert(0, s)
    # The directory existing does not guarantee rocisa was built/installed in this
    # checkout. Tensile imports rocisa (a compiled nanobind extension) lazily, so
    # without this guard a missing rocisa surfaces as an error deep inside a test
    # rather than a clean skip. See tox -e integration to provision the env.
    pytest.importorskip(
        "rocisa",
        reason=f"rocisa not importable from {tl}; build it (e.g. tox -e integration).",
    )
    try:
        yield
    finally:
        try:
            sys.path.remove(s)
        except ValueError:
            pass
