"""Multi-launch UKDs (`sources`) authored as hip or hsaco entries.

An entry is the single-launch hip or hsaco `kernel_source`, so launches that share
one compile share one code object, and each keeps its own symbol and its own
signature, read from the object. The hip cases compile for real; the hsaco cases
are compiler-free and use the committed code objects under `fixtures/hsaco/`.
The validation of a malformed entry is in `test_hkp_pack_sources.py`.
"""

import hashlib
import json
import shutil

import pytest

from hkp_pack import desk_check
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.hsaco_source import hsaco_variant_key
from hkp_pack.kernel_signature import exported_kernels, kernel_signature
from hkp_pack.pipeline import compile_intermediate
from pack_helpers import GENERIC_TARGETS, read_shipped, run_pipeline

ARCH = "gfx942"
CO_NAME = "HsacoFixture.co"
NO_HIPCC = "hipcc-not-invoked"
KDP = "solo.kdp.json"

PAIR_CPP = """\
#include <hip/hip_runtime.h>

extern "C" __global__ void KernelAdd(const float* a, const float* b, float* c)
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

extern "C" __global__ void KernelScale(float* x, float s)
{
    x[threadIdx.x] *= s;
}

// No launch below names this one.
extern "C" __global__ void KernelUnclaimed(float* x)
{
    x[threadIdx.x] = 0.0f;
}
"""

OTHER_CPP = """\
#include <hip/hip_runtime.h>

extern "C" __global__ void KernelOther(float* x)
{
    x[threadIdx.x] += 1.0f;
}
"""

BUILD = {"defines": {"HKP_TEST": 1}}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, doc):
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _hip(source="Pair.cpp", entry="KernelAdd", build=BUILD):
    return {"kind": "hip", "source": source, "entry": entry, "build": build}


def _hsaco(symbol, file=CO_NAME):
    return {"kind": "hsaco", "file": file, "symbol": symbol}


def _root(tmp_path, empty_arch_fixture, entries, arch=None):
    """`empty_arch` with its one UKD's `kernel_source` replaced by `sources`."""
    root = tmp_path / "root"
    shutil.copytree(empty_arch_fixture, root)
    (root / "Pair.cpp").write_text(PAIR_CPP, encoding="utf-8")
    (root / "Other.cpp").write_text(OTHER_CPP, encoding="utf-8")
    kdp_path = root / KDP
    doc = _read(kdp_path)
    ukd = doc["kernelDescriptors"][0]
    del ukd["kernel_source"]
    ukd["sources"] = entries
    if arch is not None:
        ukd["arch"] = arch
    _write(kdp_path, doc)
    return root


def _add_fixture_file(root, hsaco_fixture_dir, name=CO_NAME):
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hsaco_fixture_dir / ARCH / CO_NAME, dest)
    return dest


def _pack(root, tmp_path, rocm_kpack_dir, hipcc=NO_HIPCC):
    return run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


def _shipped(tmp_path):
    return read_shipped(tmp_path / "out" / ARCH / KDP)["kernelDescriptors"][0]


def _blob(results, rocm_kpack_dir, toc_key):
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    return bytes(archive.get_kernel(toc_key, ARCH))


def _assert_passes_the_desk_check(tmp_path, rocm_kpack_dir):
    """A kernel with no compiled bindings is unclaimed, never failed for lacking
    the producing compiler's evidence."""
    failures, unclaimed, verified = desk_check.compiled_agreement(
        tmp_path / "out" / ARCH / KDP, GENERIC_TARGETS, rocm_kpack_dir
    )
    assert (failures, len(unclaimed), verified) == ([], 1, 0)


# --- hip: compiled for real -------------------------------------------------------
def test_two_launches_on_one_unit_share_a_blob_and_keep_their_own_symbols(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """One translation unit and build is one compile and one toc_key. Each launch
    keeps its authored `entry` and the argument list of that kernel, so a signature
    cached per blob, or a symbol taken from the first launch, would show here. The
    unit also exports a kernel no launch names, which is no error."""
    entries = {
        "add": _hip(entry="KernelAdd"),
        "scale": _hip(entry="KernelScale"),
    }
    root = _root(tmp_path, empty_arch_fixture, entries)

    results = _pack(root, tmp_path, rocm_kpack_dir, hipcc)

    ukd = _shipped(tmp_path)
    assert "kernel_source" not in ukd and ukd["version"] == "1.1"
    add, scale = ukd["sources"]["add"], ukd["sources"]["scale"]
    assert add["kind"] == scale["kind"] == "kpack"
    assert add["toc_key"] == scale["toc_key"]
    assert (add["symbol"], scale["symbol"]) == ("KernelAdd", "KernelScale")
    blob = _blob(results, rocm_kpack_dir, add["toc_key"])
    assert add["sha256"] == scale["sha256"] == hashlib.sha256(blob).hexdigest()
    assert add["signature"] == kernel_signature(blob, "KernelAdd", "test")
    assert scale["signature"] == kernel_signature(blob, "KernelScale", "test")
    assert [a["kind"] for a in add["signature"]] == ["global_buffer"] * 3
    assert [a["kind"] for a in scale["signature"]] == ["global_buffer", "by_value"]
    assert "KernelUnclaimed" in exported_kernels(blob, "test")


@pytest.mark.parametrize(
    "other",
    [
        _hip(source="Other.cpp", entry="KernelOther"),
        _hip(entry="KernelScale", build={"defines": {"HKP_TEST": 2}}),
    ],
    ids=["other-unit", "other-build"],
)
def test_launches_on_distinct_compiles_pack_distinct_blobs(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir, other
):
    root = _root(
        tmp_path, empty_arch_fixture, {"first": _hip(entry="KernelAdd"), "next": other}
    )

    results = _pack(root, tmp_path, rocm_kpack_dir, hipcc)

    sources = _shipped(tmp_path)["sources"]
    assert sources["first"]["toc_key"] != sources["next"]["toc_key"]
    for launch in sources.values():
        blob = _blob(results, rocm_kpack_dir, launch["toc_key"])
        assert launch["sha256"] == hashlib.sha256(blob).hexdigest()
        assert launch["symbol"] in exported_kernels(blob, "test")


def test_a_hip_sources_ukd_records_each_launchs_authored_identity(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """Provenance names the shared kind and what each launch asked for. No
    compiler observed anything about the kernels, so there is no
    `effective_spec`, and the desk check leaves the kernel unclaimed."""
    entries = {
        "add": _hip(entry="KernelAdd"),
        "other": _hip(source="Other.cpp", entry="KernelOther"),
    }
    root = _root(tmp_path, empty_arch_fixture, entries)
    authored = _read(root / KDP)["kernelDescriptors"][0]["provenance"]

    _pack(root, tmp_path, rocm_kpack_dir, hipcc)

    provenance = _shipped(tmp_path)["provenance"]
    assert provenance["origin_kind"] == "hip"
    assert provenance["sources"] == {
        name: {"source": ks["source"], "entry": ks["entry"], "build": ks["build"]}
        for name, ks in entries.items()
    }
    assert provenance["hipcc_version"]
    assert "effective_spec" not in provenance
    assert provenance["specialization_contract"] == authored["specialization_contract"]
    _assert_passes_the_desk_check(tmp_path, rocm_kpack_dir)


# --- hsaco: compiler-free ---------------------------------------------------------
@pytest.mark.quick
def test_one_file_serves_two_launches_as_one_entry(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """Two symbols of one file share a toc_key and one archive entry; each keeps
    its own argument list. The packed UKD names the file and its digest per launch
    and carries no toolchain claim, and nothing is staged for an authored object."""
    entries = {
        "add": _hsaco("HsacoFixtureAdd"),
        "scale": _hsaco("HsacoFixtureScale"),
    }
    root = _root(tmp_path, empty_arch_fixture, entries, arch=[ARCH])
    fixture = _add_fixture_file(root, hsaco_fixture_dir).read_bytes()
    digest = hashlib.sha256(fixture).hexdigest()

    results = _pack(root, tmp_path, rocm_kpack_dir)

    ukd = _shipped(tmp_path)
    add, scale = ukd["sources"]["add"], ukd["sources"]["scale"]
    assert add["toc_key"] == scale["toc_key"] == hsaco_variant_key(CO_NAME)
    assert (add["symbol"], scale["symbol"]) == ("HsacoFixtureAdd", "HsacoFixtureScale")
    assert (len(add["signature"]), len(scale["signature"])) == (4, 3)
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    assert list(archive.toc) == [add["toc_key"]]
    assert bytes(archive.get_kernel(add["toc_key"], ARCH)) == fixture

    provenance = ukd["provenance"]
    assert provenance["origin_kind"] == "hsaco"
    assert provenance["sources"] == {
        "add": {"file": CO_NAME, "sha256": digest, "symbol": "HsacoFixtureAdd"},
        "scale": {"file": CO_NAME, "sha256": digest, "symbol": "HsacoFixtureScale"},
    }
    assert "hipcc_version" not in provenance and "effective_spec" not in provenance

    assert not list((tmp_path / "out").rglob("*.co"))
    assert not list((tmp_path / "inter" / ARCH).rglob("*.co"))
    staged = _read(tmp_path / "inter" / ARCH / KDP)["kernelDescriptors"][0]
    assert staged["sources"] == entries
    _assert_passes_the_desk_check(tmp_path, rocm_kpack_dir)


@pytest.mark.quick
def test_launches_on_two_files_pack_two_entries(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    entries = {
        "add": _hsaco("HsacoFixtureAdd"),
        "scale": _hsaco("HsacoFixtureScale", file=f"other/{CO_NAME}"),
    }
    root = _root(tmp_path, empty_arch_fixture, entries, arch=[ARCH])
    _add_fixture_file(root, hsaco_fixture_dir)
    _add_fixture_file(root, hsaco_fixture_dir, f"other/{CO_NAME}")

    results = _pack(root, tmp_path, rocm_kpack_dir)

    sources = _shipped(tmp_path)["sources"]
    assert sources["add"]["toc_key"] == hsaco_variant_key(CO_NAME)
    assert sources["scale"]["toc_key"] == hsaco_variant_key(f"other/{CO_NAME}")
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    assert sorted(archive.toc) == sorted(v["toc_key"] for v in sources.values())


@pytest.mark.quick
def test_a_symbol_the_file_lacks_names_its_launch(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    entries = {"add": _hsaco("HsacoFixtureAdd"), "gone": _hsaco("NoSuchKernel")}
    root = _root(tmp_path, empty_arch_fixture, entries, arch=[ARCH])
    _add_fixture_file(root, hsaco_fixture_dir)

    with pytest.raises(HkpPackError, match="launch 'gone' declares symbol"):
        _pack(root, tmp_path, rocm_kpack_dir)


@pytest.mark.quick
def test_hsaco_launches_need_a_ukd_arch(tmp_path, empty_arch_fixture):
    entries = {"add": _hsaco("HsacoFixtureAdd"), "scale": _hsaco("HsacoFixtureScale")}
    root = _root(tmp_path, empty_arch_fixture, entries)

    with pytest.raises(HkpPackError, match="hsaco requires a non-empty 'arch'"):
        load_flat_input(root, GENERIC_TARGETS)


# --- compiled bindings --------------------------------------------------------------
@pytest.mark.quick
@pytest.mark.parametrize(
    "entries, arch",
    [
        ({"a": _hip(entry="KernelAdd"), "b": _hip(entry="KernelScale")}, None),
        ({"a": _hsaco("HsacoFixtureAdd"), "b": _hsaco("HsacoFixtureScale")}, [ARCH]),
    ],
    ids=["hip", "hsaco"],
)
def test_a_declaration_binding_compiled_fields_is_refused(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, entries, arch
):
    """No compiler observed these kernels, so a contract naming compiled fields has
    nothing to be fulfilled by."""
    root = _root(tmp_path, empty_arch_fixture, entries, arch=arch)
    _add_fixture_file(root, hsaco_fixture_dir)
    kdp_path = root / KDP
    doc = _read(kdp_path)
    (consumer,) = doc["kernelDescriptors"][0]["provenance"]["specialization_contract"][
        "consumers"
    ]
    consumer["metadata_fields"] = ["block_size"]
    consumer["matcher_only_fields"] = ["dtype"]
    consumer["bindings"] = {"block_size": {"field": "block_size"}}
    _write(kdp_path, doc)
    flat = load_flat_input(root, GENERIC_TARGETS)

    with pytest.raises(HkpPackError, match="source cannot fulfil"):
        compile_intermediate(flat, root, ARCH, NO_HIPCC, tmp_path / "inter")
