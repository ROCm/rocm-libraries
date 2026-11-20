# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Handle lifetime, and the device check that keeps hipblasLtCreate from ending the process."""

import os
import subprocess
import sys
import textwrap

import pytest

import hipblaslt

c = hipblaslt._core


def _child(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    env.pop("HIPBLASLT_PY_LIBRARY", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_handle_without_a_device_raises():
    # In a subprocess with the device hidden, so this runs on GPU lanes too.
    r = _child(
        """
        import warnings; warnings.simplefilter("ignore")
        import hipblaslt
        try:
            hipblaslt._core.Handle()
        except hipblaslt.HipblasLtError as e:
            print("raised:", e)
        """
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "raised:" in r.stdout and "no HIP device is visible" in r.stdout


# ROCM-33629: hipblasLtCreate calls exit(EXIT_FAILURE) when there's no device
# (library/src/amd_detail/hipblaslt.cpp, through CHECK_HIP_ERROR), so a caller can't recover.
# This asserts the fixed behavior, an error status, and is quarantined until the fix lands.
# Time-box: revisit by the next ROCm release. Delete this marker in the PR that fixes
# ROCM-33629.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="ROCM-33629: hipblasLtCreate exits the process instead of returning an error",
)
def test_rocm_33629_create_without_a_device_returns_an_error():
    r = _child(
        """
        import warnings; warnings.simplefilter("ignore")
        import hipblaslt
        print("status:", hipblaslt._core._unchecked_create_status())
        """
    )
    assert (
        r.returncode == 0 and "status:" in r.stdout
    ), f"exit {r.returncode}, stdout {r.stdout!r}: hipblasLtCreate ended the process"
    assert "status: 0" not in r.stdout


@pytest.mark.gpu
def test_first_handle_runs_the_version_check():
    # In a fresh process: import leaves the check alone, and the first Handle() runs it once.
    code = """
        import warnings; warnings.simplefilter("ignore")
        import hipblaslt
        before = hipblaslt._loader._version_checked
        hipblaslt._core.Handle().close()
        print("checked:", before, hipblaslt._loader._version_checked)
        """
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "checked: False True" in r.stdout, r.stdout


@pytest.mark.gpu
def test_handle_closes_once_and_raises_after():
    h = c.Handle()
    assert not h.closed and h.ptr
    h.close()
    h.close()
    assert h.closed
    with pytest.raises(hipblaslt.HipblasLtError, match="closed"):
        h.ptr


@pytest.mark.gpu
def test_handle_as_a_context_manager():
    with c.Handle() as h:
        assert h.ptr
    assert h.closed


@pytest.mark.gpu
def test_dropping_a_handle_is_safe():
    h = c.Handle()
    del h
    import gc

    gc.collect()
    with c.Handle() as again:
        assert again.ptr
