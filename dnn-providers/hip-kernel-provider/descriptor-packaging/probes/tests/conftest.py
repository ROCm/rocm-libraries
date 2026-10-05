import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBES_TESTS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PROBES_TESTS_DIR.parent.parent
_HKP_DIR = _PKG_DIR.parent

ARCH = "gfx950"
STAMP_NAME = ".hkp-packed.stamp"
HKP_PACK = _PKG_DIR / "tools" / "hkp_pack.py"
PROBE_ASSERT = _PKG_DIR / "tools" / "hkp_probe_assert.py"
ROCKE_FIXTURE = _PKG_DIR / "tests" / "fixtures" / "rocke"
_ROCKE_SOURCE_DIRS = (
    _HKP_DIR / "rocke" / "platform" / "python",
    _HKP_DIR / "rocke" / "library",
)


@pytest.fixture(scope="session")
def rocm_kpack_dir():
    """The rocm_kpack python dir; a missing one is a hard failure, never a skip."""
    value = os.environ.get("HIPKERNELPROVIDER_ROCM_KPACK_DIR")
    if not value or not Path(value).is_dir():
        pytest.fail(
            "HIPKERNELPROVIDER_ROCM_KPACK_DIR must name the rocm_kpack python dir"
        )
    return value


@pytest.fixture(scope="session")
def hipcc():
    value = os.environ.get("HKP_HIPCC")
    if not value:
        pytest.fail("HKP_HIPCC must name the hipcc driver")
    return value


@pytest.fixture(scope="session")
def comgr_lib():
    """The comgr library the pack is steered to, or None when the environment
    does not pin one (rocke then resolves its own)."""
    return os.environ.get("ROCKE_COMGR_LIB") or None


@pytest.fixture(scope="module")
def packed_root(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    """Pack the rocke fixture once for gfx950 and return the output root.

    The stamp file is written by CMake in production; it is created here so the
    stamp assertion has something to find.
    """
    work = tmp_path_factory.mktemp("probe_pack")
    out = work / "out"
    wheel_stamp = work / "rocke-wheel.stamp"
    wheel_stamp.write_text(hashlib.sha256(b"probe-test-wheel").hexdigest() + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in _ROCKE_SOURCE_DIRS] + [env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if comgr_lib:
        env["ROCKE_COMGR_LIB"] = comgr_lib
    result = subprocess.run(
        [
            sys.executable,
            str(HKP_PACK),
            "--source-root",
            str(ROCKE_FIXTURE),
            "--out-root",
            str(out),
            "--arches",
            ARCH,
            "--hipcc",
            hipcc,
            "--inter-root",
            str(work / "inter"),
            "--kpack-python-dir",
            rocm_kpack_dir,
            "--source-label",
            "probe_assert_test",
            "--rocke-wheel-stamp",
            str(wheel_stamp),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"hkp_pack failed ({result.returncode}):\n{result.stderr}")
    (out / STAMP_NAME).write_text("")
    return out
