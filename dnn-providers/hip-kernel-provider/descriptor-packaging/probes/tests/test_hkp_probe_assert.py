import json
import shutil
import subprocess
import sys

import pytest

from conftest import ARCH, PROBE_ASSERT, STAMP_NAME

KDP = f"{ARCH}/attention.kdp.json"


class _Tree:
    """A mutable copy of the packed output plus the assert-tool invocation."""

    def __init__(self, root, rocm_kpack_dir, comgr_path):
        self.root = root
        self.rocm_kpack_dir = rocm_kpack_dir
        self.comgr_path = comgr_path

    def kdp(self):
        return json.loads((self.root / KDP).read_text(encoding="utf-8"))

    def mutate_ukd(self, edit):
        kdp = self.kdp()
        edit(kdp["kernelDescriptors"][0])
        (self.root / KDP).write_text(json.dumps(kdp, indent=2), encoding="utf-8")

    def mutate_kdp(self, edit):
        kdp = self.kdp()
        edit(kdp)
        (self.root / KDP).write_text(json.dumps(kdp, indent=2), encoding="utf-8")

    def run(self, expect_ukds=1, expect_comgr=None, root=None):
        cmd = [
            sys.executable,
            str(PROBE_ASSERT),
            "--out-root",
            str(self.root if root is None else root),
            "--arch",
            ARCH,
            "--kind",
            "rocke",
            "--kpack-python-dir",
            self.rocm_kpack_dir,
            "--stamp-name",
            STAMP_NAME,
            "--expect-ukds",
            str(expect_ukds),
        ]
        if expect_comgr is not None:
            cmd += ["--expect-comgr", str(expect_comgr)]
        return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture
def tree(packed_root, tmp_path, rocm_kpack_dir, comgr_lib):
    copy = tmp_path / "out"
    shutil.copytree(packed_root, copy)
    comgr = (
        comgr_lib
        or json.loads((copy / KDP).read_text(encoding="utf-8"))["kernelDescriptors"][0][
            "provenance"
        ]["comgr_path"]
    )
    return _Tree(copy, rocm_kpack_dir, comgr)


def _assert_fails(result, assertion_id):
    assert result.returncode == 1, result.stdout + result.stderr
    lines = [
        ln
        for ln in result.stderr.splitlines()
        if ln.startswith("hkp_probe_assert: FAIL")
    ]
    assert lines, result.stderr
    assert any(f"FAIL {assertion_id}:" in ln for ln in lines), result.stderr


def test_pristine_pack_passes(tree):
    result = tree.run()
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_pristine_pack_passes_with_expected_comgr(tree):
    result = tree.run(expect_comgr=tree.comgr_path)
    assert result.returncode == 0, result.stderr


def test_out_root_missing(tree):
    result = tree.run(root=tree.root / "nope")
    _assert_fails(result, "out-root-missing")
    assert "build target hkp_packaging_probes first" in result.stderr


def test_stamp_missing(tree):
    (tree.root / STAMP_NAME).unlink()
    _assert_fails(tree.run(), "stamp-missing")


def test_arch_dir_missing(tree):
    shutil.rmtree(tree.root / ARCH)
    _assert_fails(tree.run(), "arch-dir-missing")


def test_extra_arch_dir(tree):
    (tree.root / "gfx942").mkdir()
    _assert_fails(tree.run(), "extra-arch-dir")


def test_kpack_missing(tree):
    (tree.root / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").unlink()
    _assert_fails(tree.run(), "kpack-missing")


def test_kpack_empty(tree):
    (tree.root / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").write_bytes(b"")
    _assert_fails(tree.run(), "kpack-empty")


def test_no_kdp(tree):
    (tree.root / KDP).unlink()
    _assert_fails(tree.run(), "no-kdp")


def test_ukd_count_mismatch(tree):
    _assert_fails(tree.run(expect_ukds=2), "ukd-count")


def test_ukd_count_zero_is_failure(tree):
    tree.mutate_kdp(lambda kdp: kdp.update(kernelDescriptors=[]))
    _assert_fails(tree.run(expect_ukds=0), "ukd-count")


def test_ukd_kind(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(kind="rocke"))
    _assert_fails(tree.run(), "ukd-kind")


def test_arch_field_ukd(tree):
    tree.mutate_ukd(lambda u: u.update(arch=["gfx942"]))
    _assert_fails(tree.run(), "arch-field")


def test_arch_field_kdp(tree):
    tree.mutate_kdp(lambda kdp: kdp.update(arch=[ARCH, "gfx942"]))
    _assert_fails(tree.run(), "arch-field")


def test_kpack_toc(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(toc_key="no_such_key"))
    _assert_fails(tree.run(), "kpack-toc")


def test_sha256(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(sha256="0" * 64))
    _assert_fails(tree.run(), "sha256")


def test_signature(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(signature=[]))
    _assert_fails(tree.run(), "signature")


def test_symbol(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(symbol="no_such_symbol_xyz"))
    _assert_fails(tree.run(), "symbol")


def test_provenance_origin(tree):
    tree.mutate_ukd(lambda u: u["provenance"].update(origin_kind="hip"))
    _assert_fails(tree.run(), "provenance-origin")


def test_provenance_wheel_absent(tree):
    tree.mutate_ukd(lambda u: u["provenance"].pop("rocke_wheel_sha256"))
    _assert_fails(tree.run(), "provenance-wheel")


def test_provenance_wheel_empty(tree):
    tree.mutate_ukd(lambda u: u["provenance"].update(rocke_wheel_sha256=""))
    _assert_fails(tree.run(), "provenance-wheel")


def test_provenance_comgr_wrong_expected(tree):
    result = tree.run(expect_comgr=tree.root / "not" / "libamd_comgr.so")
    _assert_fails(result, "provenance-comgr")


def test_provenance_comgr_empty_without_expectation(tree):
    tree.mutate_ukd(lambda u: u["provenance"].update(comgr_path=""))
    _assert_fails(tree.run(), "provenance-comgr")


def test_corrupt_kpack_reports_toc_failure(tree):
    kpack_path = tree.root / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    kpack_path.write_bytes(b"not a kpack archive" * 8)
    result = tree.run()
    _assert_fails(result, "kpack-toc")
    assert "Traceback" not in result.stderr


def test_malformed_kdp_json_reports_no_kdp(tree):
    (tree.root / KDP).write_text("{ not json", encoding="utf-8")
    result = tree.run()
    _assert_fails(result, "no-kdp")
    assert "Traceback" not in result.stderr
