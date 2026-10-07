"""Paths, helpers and fixtures shared by the probe tooling tests.

There is no conftest.py here: tests/ imports its own conftest by module name, and a
second conftest.py would replace it when both directories are collected in one run.
Test modules import the fixtures they use from this module.
"""

import hashlib
import json
import os
import shutil
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
PROBE_DERIVE = _PKG_DIR / "tools" / "hkp_probe_derive_root.py"
FIXTURES = _PKG_DIR / "tests" / "fixtures"
ROCKE_FIXTURE = FIXTURES / "rocke"
MAIN_FIXTURE = FIXTURES / "main"
EMPTY_ARCH_FIXTURE = FIXTURES / "empty_arch"
HSACO_FIXTURES = FIXTURES / "hsaco"
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


def pack_root(src, work, rocm_kpack_dir, hipcc, comgr_lib):
    """Pack `src` for ARCH with the real packer; return the output root.

    The stamp file is written by CMake in production; it is created here so the
    stamp assertion has something to find.
    """
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
            str(src),
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


@pytest.fixture(scope="module")
def packed_root(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    """The rocke fixture packed once for ARCH."""
    work = tmp_path_factory.mktemp("probe_pack")
    return pack_root(ROCKE_FIXTURE, work, rocm_kpack_dir, hipcc, comgr_lib)


def run_assert(out_root, expect_path, rocm_kpack_dir, expect_comgr=None):
    cmd = [
        sys.executable,
        str(PROBE_ASSERT),
        "--out-root",
        str(out_root),
        "--arch",
        ARCH,
        "--expect",
        str(expect_path),
        "--kpack-python-dir",
        rocm_kpack_dir,
        "--stamp-name",
        STAMP_NAME,
    ]
    if expect_comgr is not None:
        cmd += ["--expect-comgr", str(expect_comgr)]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_derive(src, arch, out, ukds=()):
    """Run the derive tool; one --ukd per entry of `ukds`."""
    cmd = [
        sys.executable,
        str(PROBE_DERIVE),
        "--from",
        str(src),
        "--arch",
        arch,
        "--out",
        str(out),
    ]
    for name in ukds:
        cmd += ["--ukd", name]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def hip_source(dst):
    """The main fixture without its standalone UKD, which derive refuses."""
    shutil.copytree(MAIN_FIXTURE, dst)
    (dst / "pointwise_add_b128.ukd.json").unlink()
    kdp = json.loads((dst / "pointwise.kdp.json").read_text())
    kdp["kernelDescriptors"] = [
        u for u in kdp["kernelDescriptors"] if isinstance(u, dict)
    ]
    (dst / "pointwise.kdp.json").write_text(json.dumps(kdp, indent=2))
    return dst


def derive_and_pack(src, work, rocm_kpack_dir, hipcc, comgr_lib, ukds=()):
    """Derive `src` for ARCH, pack the derived root; return (out root, expect path)."""
    r = run_derive(src, ARCH, work / "probe" / "root", ukds)
    assert r.returncode == 0, r.stderr
    out = pack_root(work / "probe" / "root", work, rocm_kpack_dir, hipcc, comgr_lib)
    return out, work / "probe" / "expect.json"
