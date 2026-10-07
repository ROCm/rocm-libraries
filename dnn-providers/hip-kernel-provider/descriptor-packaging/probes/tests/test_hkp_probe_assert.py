import json
import shutil

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    ARCH,
    STAMP_NAME,
    comgr_lib,
    hipcc,
    packed_root,
    rocm_kpack_dir,
    run_assert,
)

KDP = f"{ARCH}/attention.kdp.json"
KDP_IN_ARCH_DIR = "attention.kdp.json"


class _Tree:
    """A mutable copy of the packed rocke output plus the assert-tool invocation."""

    def __init__(self, root, expect_path, rocm_kpack_dir, comgr_path):
        self.root = root
        self.expect_path = expect_path
        self.rocm_kpack_dir = rocm_kpack_dir
        self.comgr_path = comgr_path
        self.ukd_name = self.kdp()["kernelDescriptors"][0]["name"]
        self.write_expect([self.entry()])

    def entry(self, name=None, kind="rocke"):
        return {
            "kdp": KDP_IN_ARCH_DIR,
            "name": self.ukd_name if name is None else name,
            "kind": kind,
        }

    def write_expect(self, entries):
        self.expect_path.write_text(json.dumps(entries), encoding="utf-8")

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

    def run(self, expect_comgr=None, root=None):
        return run_assert(
            self.root if root is None else root,
            self.expect_path,
            self.rocm_kpack_dir,
            expect_comgr,
        )


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
    return _Tree(copy, tmp_path / "expect.json", rocm_kpack_dir, comgr)


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
    tree.write_expect([tree.entry(), tree.entry(name="a_second_ukd")])
    _assert_fails(tree.run(), "ukd-count")


def test_ukd_shipped_twice_fails_count(tree):
    # Every shipped UKD is in the expect list, but one ships twice: matched > expected.
    tree.mutate_kdp(
        lambda kdp: kdp["kernelDescriptors"].append(kdp["kernelDescriptors"][0])
    )
    _assert_fails(tree.run(), "ukd-count")


def test_ukd_count_zero_is_failure(tree):
    tree.mutate_kdp(lambda kdp: kdp.update(kernelDescriptors=[]))
    tree.write_expect([])
    _assert_fails(tree.run(), "ukd-count")


def test_shipped_ukd_missing_from_expect(tree):
    # Same count, different identity: the shipped UKD is not the one expected.
    tree.write_expect([tree.entry(name="not_the_shipped_ukd")])
    result = tree.run()
    _assert_fails(result, "ukd-count")
    assert tree.ukd_name in result.stderr


def test_ukd_kind(tree):
    tree.mutate_ukd(lambda u: u["kernel_source"].update(kind="rocke"))
    _assert_fails(tree.run(), "ukd-kind")


def test_arch_field_ukd(tree):
    tree.mutate_ukd(lambda u: u.update(arch=["gfx942"]))
    _assert_fails(tree.run(), "arch-field")


def test_arch_field_ukd_absent(tree):
    tree.mutate_ukd(lambda u: u.pop("arch"))
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


def test_expected_kind_differs_from_shipped_origin(tree):
    # The pack records rocke; the expectation says hip.
    tree.write_expect([tree.entry(kind="hip")])
    _assert_fails(tree.run(), "provenance-origin")


def test_expected_kind_without_rules(tree):
    tree.write_expect([tree.entry(kind="mystery")])
    result = tree.run()
    _assert_fails(result, "no-rules-for-kind")
    assert "mystery" in result.stderr


def test_hip_kind_needs_no_rocke_provenance(tree):
    # A hip UKD records neither the wheel digest nor comgr; stripping them from a
    # UKD expected as hip must not fail.
    def as_hip(u):
        u["provenance"] = {"origin_kind": "hip", "hipcc_version": "test"}

    tree.mutate_ukd(as_hip)
    tree.write_expect([tree.entry(kind="hip")])
    result = tree.run()
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("content", ["{ not json", "{}", '[{"kdp": "x"}]'])
def test_unusable_expect_exits_2(tree, content):
    tree.expect_path.write_text(content, encoding="utf-8")
    result = tree.run()
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr


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


def test_non_object_kdp_reports_no_kdp(tree):
    (tree.root / KDP).write_text("[]", encoding="utf-8")
    result = tree.run()
    _assert_fails(result, "no-kdp")
    assert "Traceback" not in result.stderr


def test_duplicate_expect_entry_exits_2(tree):
    tree.write_expect([tree.entry(), tree.entry()])
    result = tree.run()
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr


def test_rocm_kpack_not_importable_exits_2(tree, tmp_path):
    # A rocm_kpack package without the kpack module: the import fails whatever the
    # interpreter has installed, because --kpack-python-dir is searched first.
    (tmp_path / "fake" / "rocm_kpack").mkdir(parents=True)
    (tmp_path / "fake" / "rocm_kpack" / "__init__.py").write_text("")
    result = run_assert(tree.root, tree.expect_path, str(tmp_path / "fake"))
    assert result.returncode == 2, result.stderr
    assert "cannot import rocm_kpack" in result.stderr
    assert "Traceback" not in result.stderr


def test_malformed_kdp_json_reports_no_kdp(tree):
    (tree.root / KDP).write_text("{ not json", encoding="utf-8")
    result = tree.run()
    _assert_fails(result, "no-kdp")
    assert "Traceback" not in result.stderr
