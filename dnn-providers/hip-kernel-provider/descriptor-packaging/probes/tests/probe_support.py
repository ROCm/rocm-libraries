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

sys.path.insert(0, str(_PKG_DIR / "python"))
from hkp_pack import provenance_sidecar  # noqa: E402

ARCH = "gfx950"
STAMP_NAME = ".hkp-packed.stamp"
HKP_PACK = _PKG_DIR / "tools" / "hkp_pack.py"
PROBE_ASSERT = _PKG_DIR / "tools" / "hkp_probe_assert.py"
PROBE_DERIVE = _PKG_DIR / "tools" / "hkp_probe_derive_root.py"
GENERIC_TARGETS_JSON = (
    _HKP_DIR.parent.parent
    / "projects"
    / "hipdnn"
    / "plugin_sdk"
    / "data"
    / "gpu_generic_targets.json"
)
FIXTURES = _PKG_DIR / "tests" / "fixtures"
ROCKE_FIXTURE = FIXTURES / "rocke"
MULTI_LAUNCH_FIXTURES = (
    FIXTURES / "multi_launch_decode_merge",
    FIXTURES / "multi_launch_same_kernel_twice",
)
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


def pack_root(
    src,
    work,
    rocm_kpack_dir,
    hipcc,
    comgr_lib,
    *,
    rocke=False,
    filters=((), ()),
    arch=ARCH,
):
    """Pack `src` for `arch` with the real packer; return the output root.

    `rocke` packs with the rocKE toolchain (wheel stamp, in-tree rocke on PYTHONPATH,
    comgr), as a build with HIPKERNELPROVIDER_ENABLE_ROCKE=ON does; the default packs
    as an OFF build does, with none of it, so a test that does not pack rocKE content
    needs no rocKE toolchain. `filters` is (exclude_folders, disabled_kinds), the
    packer's own flags.

    The stamp file is written by CMake in production; it is created here so the
    stamp assertion has something to find.
    """
    out = work / "out"
    env = dict(os.environ)
    cmd = [
        sys.executable,
        str(HKP_PACK),
        "--source-root",
        str(src),
        "--out-root",
        str(out),
        "--arches",
        arch,
        "--hipcc",
        hipcc,
        "--generic-targets-json",
        str(GENERIC_TARGETS_JSON),
        "--inter-root",
        str(work / "inter"),
        "--kpack-python-dir",
        rocm_kpack_dir,
        "--source-label",
        "probe_assert_test",
    ]
    if rocke:
        wheel_stamp = work / "rocke-wheel.stamp"
        wheel_stamp.write_text(hashlib.sha256(b"probe-test-wheel").hexdigest() + "\n")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(p) for p in _ROCKE_SOURCE_DIRS] + [env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        if comgr_lib:
            env["ROCKE_COMGR_LIB"] = comgr_lib
        cmd += ["--rocke-wheel-stamp", str(wheel_stamp)]
    exclude_folders, disabled_kinds = filters
    for folder in exclude_folders:
        cmd += ["--exclude-folder", folder]
    for kind in disabled_kinds:
        cmd += ["--disable-kind", kind]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"hkp_pack failed ({result.returncode}):\n{result.stderr}")
    (out / STAMP_NAME).write_text("")
    return out


def read_packed(path):
    """A packed descriptor with each UKD's provenance attached from its sidecar."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    provenance_sidecar.attach(Path(path), doc)
    return doc


def write_packed(path, doc):
    """Write `doc` (UKD provenance inline, as read_packed returns it) back as a
    packed descriptor: provenance in a fresh sidecar, digests matching the UKDs."""
    path = Path(path)
    sidecar_name, data = provenance_sidecar.detach(path.name, doc)
    (path.parent / sidecar_name).write_bytes(data)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def packed_root(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    """The rocke fixture packed once for ARCH, with the rocKE toolchain."""
    work = tmp_path_factory.mktemp("probe_pack")
    return pack_root(ROCKE_FIXTURE, work, rocm_kpack_dir, hipcc, comgr_lib, rocke=True)


def run_assert(out_root, expect_path, rocm_kpack_dir, expect_comgr=None, arch=ARCH):
    cmd = [
        sys.executable,
        str(PROBE_ASSERT),
        "--out-root",
        str(out_root),
        "--arch",
        arch,
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


def run_derive(src, arch, out, ukds=(), filters=((), ())):
    """Run the derive tool; one --ukd per entry of `ukds`, and the same
    (exclude_folders, disabled_kinds) `filters` pack_root takes."""
    cmd = [
        sys.executable,
        str(PROBE_DERIVE),
        "--from",
        str(src),
        "--generic-targets-json",
        str(GENERIC_TARGETS_JSON),
        "--arch",
        arch,
        "--out",
        str(out),
    ]
    for name in ukds:
        cmd += ["--ukd", name]
    exclude_folders, disabled_kinds = filters
    for folder in exclude_folders:
        cmd += ["--exclude-folder", folder]
    for kind in disabled_kinds:
        cmd += ["--disable-kind", kind]
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


def derive_and_pack(
    src,
    work,
    rocm_kpack_dir,
    hipcc,
    comgr_lib,
    ukds=(),
    *,
    rocke=False,
    filters=((), ()),
    arch=ARCH,
):
    """Derive `src` for `arch`, pack the derived root under the same `filters`; return
    (out root, expect path). `rocke` as in pack_root."""
    r = run_derive(src, arch, work / "probe" / "root", ukds, filters)
    assert r.returncode == 0, r.stderr
    out = pack_root(
        work / "probe" / "root",
        work,
        rocm_kpack_dir,
        hipcc,
        comgr_lib,
        rocke=rocke,
        filters=filters,
        arch=arch,
    )
    return out, work / "probe" / "expect.json"
