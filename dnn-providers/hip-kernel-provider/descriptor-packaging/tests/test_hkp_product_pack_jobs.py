"""_hkp_resolve_product_pack_jobs: what HIPKERNELPROVIDER_PRODUCT_PACK_JOBS wires.

Runs the real function from HkpPackaging.cmake under ``cmake -P``. Empty, the default,
must wire nothing, so the production root keeps the packer's own pool size; a positive
integer becomes the root's ``PACK_JOBS``; anything else fails configure rather than the
pack step.
"""

import subprocess
from pathlib import Path

import pytest

_CMAKE_DIR = Path(__file__).resolve().parents[1] / "cmake"
_PROVIDER_CMAKE_DIR = _CMAKE_DIR.parents[1] / "cmake"

pytestmark = pytest.mark.quick


def _resolve(cmake, tmp_path, value):
    """Return the completed ``cmake -P`` run resolving ``value``."""
    script = tmp_path / "resolve.cmake"
    script.write_text(
        f'list(APPEND CMAKE_MODULE_PATH "{_CMAKE_DIR.as_posix()}" '
        f'"{_PROVIDER_CMAKE_DIR.as_posix()}")\n'
        "include(HkpPackaging)\n"
        "_hkp_resolve_product_pack_jobs(_args)\n"
        'message(STATUS "ARGS=[${_args}]")\n'
    )
    return subprocess.run(
        [cmake, f"-DHIPKERNELPROVIDER_PRODUCT_PACK_JOBS={value}", "-P", str(script)],
        capture_output=True,
        text=True,
    )


def _args(result):
    out = result.stdout + result.stderr
    line = next(line for line in out.splitlines() if "ARGS=[" in line)
    return line.split("ARGS=[")[1].rstrip("]")


def test_empty_wires_no_pack_jobs(cmake, tmp_path):
    result = _resolve(cmake, tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert _args(result) == ""


@pytest.mark.parametrize("value", ["1", "48", "128"])
def test_a_positive_integer_becomes_pack_jobs(cmake, tmp_path, value):
    result = _resolve(cmake, tmp_path, value)
    assert result.returncode == 0, result.stderr
    assert _args(result) == f"PACK_JOBS;{value}"


@pytest.mark.parametrize("value", ["0", "-2", "auto", "4x"])
def test_anything_else_is_a_configure_error(cmake, tmp_path, value):
    result = _resolve(cmake, tmp_path, value)
    assert result.returncode != 0
    assert "HIPKERNELPROVIDER_PRODUCT_PACK_JOBS" in result.stderr
