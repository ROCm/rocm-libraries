"""Pass-through output (embedded_source): derive -> real pack -> assert.

The roots are packed by the real packer. A root holding only pass-through UKDs
ships descriptors and no kpack archive, which is what these tests rely on.
"""

import json
import shutil

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    ARCH,
    EMPTY_ARCH_FIXTURE,
    comgr_lib,
    derive_and_pack,
    hip_source,
    hipcc,
    rocm_kpack_dir,
    run_assert,
)

_EMBEDDED = {
    "kind": "embedded_source",
    "source_file": "kernels/PointwiseAdd.cpp",
    "entry_point": "PointwiseAdd",
}
KDP = "emb/solo.kdp.json"


def _embedded_source(dst):
    shutil.copytree(EMPTY_ARCH_FIXTURE, dst)
    kdp = json.loads((dst / "solo.kdp.json").read_text())
    kdp["arch"] = [ARCH]
    kdp["kernelDescriptors"][0]["kernel_source"] = dict(_EMBEDDED)
    (dst / "solo.kdp.json").write_text(json.dumps(kdp, indent=2))
    return dst


class _Tree:
    def __init__(self, root, expect, rocm_kpack_dir, kdp):
        self.root, self.expect_path, self.rocm_kpack_dir, self.kdp_rel = (
            root,
            expect,
            rocm_kpack_dir,
            kdp,
        )

    def _path(self):
        return self.root / ARCH / self.kdp_rel

    def mutate(self, name, edit):
        kdp = json.loads(self._path().read_text())
        (ukd,) = [u for u in kdp["kernelDescriptors"] if u["name"] == name]
        edit(ukd)
        self._path().write_text(json.dumps(kdp, indent=2))

    def mutate_expect(self, edit):
        entries = json.loads(self.expect_path.read_text())
        edit(entries)
        self.expect_path.write_text(json.dumps(entries))

    def run(self):
        return run_assert(self.root, self.expect_path, self.rocm_kpack_dir)


def _fails(result, assertion_id):
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAIL {assertion_id}:" in result.stderr, result.stderr


@pytest.fixture(scope="module")
def emb_packed(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("emb_probe")
    _embedded_source(work / "src" / "emb")
    return derive_and_pack(work / "src", work, rocm_kpack_dir, hipcc, comgr_lib)


@pytest.fixture(scope="module")
def mixed_packed(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("emb_hip_probe")
    src = work / "src"
    _embedded_source(src / "emb")
    hip_source(src / "hip")
    return derive_and_pack(src, work, rocm_kpack_dir, hipcc, comgr_lib)


@pytest.fixture
def emb(emb_packed, tmp_path, rocm_kpack_dir):
    out, expect = emb_packed
    shutil.copytree(out, tmp_path / "out")
    shutil.copyfile(expect, tmp_path / "expect.json")
    tree = _Tree(
        tmp_path / "out", tmp_path / "expect.json", rocm_kpack_dir, "emb/solo.kdp.json"
    )
    tree.name = json.loads(expect.read_text())[0]["name"]
    return tree


@pytest.fixture
def mixed(mixed_packed, tmp_path, rocm_kpack_dir):
    out, expect = mixed_packed
    shutil.copytree(out, tmp_path / "out")
    shutil.copyfile(expect, tmp_path / "expect.json")
    return tmp_path / "out", tmp_path / "expect.json"


def test_passthrough_only_root_ships_no_archive_and_passes(emb_packed, rocm_kpack_dir):
    out, expect = emb_packed
    entries = json.loads(expect.read_text())
    assert [(e["kdp"], e["kind"]) for e in entries] == [(KDP, "embedded_source")]
    assert entries[0]["kernel_source"] == _EMBEDDED
    # The packer's real output: descriptors, no archive.
    assert not (out / ARCH / "kpack").exists()
    assert (out / ARCH / KDP).is_file()
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_stray_archive_is_ignored_without_expected_kpack_ukds(emb, rocm_kpack_dir):
    (emb.root / ARCH / "kpack").mkdir()
    (emb.root / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").write_bytes(b"")
    result = emb.run()
    assert result.returncode == 0, result.stderr


def test_mixed_hip_and_embedded_source_root_passes(mixed, rocm_kpack_dir):
    out, expect = mixed
    kinds = {e["kdp"]: e["kind"] for e in json.loads(expect.read_text())}
    assert kinds == {
        "emb/solo.kdp.json": "embedded_source",
        "hip/pointwise.kdp.json": "hip",
        "hip/pointwise_wild.kdp.json": "hip",
    }
    assert (out / ARCH / "kpack").is_dir()
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_mixed_root_still_runs_archive_checks_for_the_kpack_ukds(mixed, rocm_kpack_dir):
    out, expect = mixed
    kdp = out / ARCH / "hip" / "pointwise.kdp.json"
    doc = json.loads(kdp.read_text())
    doc["kernelDescriptors"][0]["kernel_source"]["sha256"] = "0" * 64
    kdp.write_text(json.dumps(doc))
    _fails(run_assert(out, expect, rocm_kpack_dir), "sha256")


def test_mixed_root_missing_archive_fails(mixed, rocm_kpack_dir):
    out, expect = mixed
    shutil.rmtree(out / ARCH / "kpack")
    _fails(run_assert(out, expect, rocm_kpack_dir), "kpack-missing")


# --- one mutation per pass-through assertion and sub-condition ---------------
def test_passthrough_source_edited(emb):
    emb.mutate(emb.name, lambda u: u["kernel_source"].update(source_file="other.cpp"))
    _fails(emb.run(), "passthrough-source")


def test_passthrough_source_entry_point_edited(emb):
    emb.mutate(emb.name, lambda u: u["kernel_source"].update(entry_point="Other"))
    _fails(emb.run(), "passthrough-source")


def test_passthrough_arch_absent(emb):
    emb.mutate(emb.name, lambda u: u.pop("arch"))
    _fails(emb.run(), "arch-field")


def test_passthrough_arch_wrong(emb):
    emb.mutate(emb.name, lambda u: u.update(arch=["gfx942"]))
    _fails(emb.run(), "arch-field")


def test_passthrough_origin(emb):
    emb.mutate(emb.name, lambda u: u["provenance"].update(origin_kind="hip"))
    _fails(emb.run(), "provenance-origin")


def test_passthrough_source_label_empty(emb):
    emb.mutate(emb.name, lambda u: u["provenance"].update(source_label=""))
    _fails(emb.run(), "passthrough-provenance")


def test_passthrough_source_label_absent(emb):
    emb.mutate(emb.name, lambda u: u["provenance"].pop("source_label"))
    _fails(emb.run(), "passthrough-provenance")


def test_passthrough_provenance_source_file_differs(emb):
    emb.mutate(emb.name, lambda u: u["provenance"].update(source_file="x.cpp"))
    _fails(emb.run(), "passthrough-provenance")


def test_passthrough_shipped_as_kpack_fails_ukd_kind(emb):
    emb.mutate(emb.name, lambda u: u["kernel_source"].update(kind="kpack"))
    _fails(emb.run(), "ukd-kind")


def test_kpack_kind_shipped_as_passthrough_fails_ukd_kind(mixed, rocm_kpack_dir):
    out, expect = mixed
    kdp = out / ARCH / "hip" / "pointwise.kdp.json"
    doc = json.loads(kdp.read_text())
    doc["kernelDescriptors"][0]["kernel_source"] = dict(_EMBEDDED)
    kdp.write_text(json.dumps(doc))
    _fails(run_assert(out, expect, rocm_kpack_dir), "ukd-kind")


def test_expect_entry_of_passthrough_kind_without_kernel_source_exits_2(emb):
    emb.mutate_expect(lambda entries: entries[0].pop("kernel_source"))
    result = emb.run()
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr


def test_expected_passthrough_but_shipped_ukd_missing_fails_count(emb):
    emb.mutate_expect(lambda entries: entries.append(dict(entries[0], name="another")))
    _fails(emb.run(), "ukd-count")


def test_two_listed_passthrough_ukds_in_one_kdp_pack_and_pass(
    tmp_path, rocm_kpack_dir, hipcc, comgr_lib
):
    # UKDS may keep several pass-through UKDs of one KDP; default mode keeps one.
    src = _embedded_source(tmp_path / "src" / "emb")
    kdp = json.loads((src / "solo.kdp.json").read_text())
    first = kdp["kernelDescriptors"][0]
    second = json.loads(json.dumps(first))
    second.update(id="ukd-solo-second", name="second")
    second["kernel_source"]["entry_point"] = "PointwiseMul"
    kdp["kernelDescriptors"].append(second)
    (src / "solo.kdp.json").write_text(json.dumps(kdp, indent=2))
    listed = [first["name"], "second"]
    out, expect = derive_and_pack(
        tmp_path / "src", tmp_path, rocm_kpack_dir, hipcc, comgr_lib, ukds=listed
    )
    entries = json.loads(expect.read_text())
    assert sorted(e["name"] for e in entries) == sorted(listed)
    assert {e["kernel_source"]["entry_point"] for e in entries} == {
        "PointwiseAdd",
        "PointwiseMul",
    }
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr
