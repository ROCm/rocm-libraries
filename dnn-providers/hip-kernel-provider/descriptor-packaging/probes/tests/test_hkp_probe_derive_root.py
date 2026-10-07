"""Tests for hkp_probe_derive_root.py (pure stdlib; no compile, no kpack)."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from probe_support import MAIN_FIXTURE, PROBE_DERIVE, run_derive


def _derive(src: Path, arch: str, out: Path, ukds=()):
    r = run_derive(src, arch, out, ukds)
    assert r.returncode == 0, r.stderr
    return json.loads((out.parent / "expect.json").read_text())


def _ukd(name, kernel_source, arch=None, ukd_id=None):
    ukd = {
        "version": "1.0",
        "id": ukd_id or f"id-{name}",
        "name": name,
        "kernel_source": kernel_source,
        "metadata": {},
        "priority": 0,
    }
    if arch is not None:
        ukd["arch"] = arch
    return ukd


def _rocke(name, builder="b1", arch=None, source="s.py", spec=None, ukd_id=None):
    ks = {
        "kind": "rocke",
        "source": source,
        "builder": builder,
        "spec": spec if spec is not None else {"block": 64},
    }
    return _ukd(name, ks, arch, ukd_id)


def _hip(name, source="a.cpp", build=None, arch=None):
    ks = {
        "kind": "hip",
        "source": source,
        "entry": name,
        "build": build if build is not None else {},
    }
    return _ukd(name, ks, arch)


def _hsaco(name, arch=("gfx950",)):
    ks = {"kind": "hsaco", "file": f"{name}.hsaco", "symbol": name}
    return _ukd(name, ks, list(arch))


def _kpack(name, arch=None):
    ks = {
        "kind": "kpack",
        "library": "kpack/x.kpack",
        "toc_key": name,
        "symbol": name,
        "sha256": "0" * 64,
        "signature": "s",
    }
    return _ukd(name, ks, arch)


def _embedded(name, ukd_id=None):
    ks = {"kind": "embedded_source", "source_file": "k/a.cpp", "entry_point": "Entry"}
    return _ukd(name, ks, ukd_id=ukd_id)


# The generic descriptors every KDP of _write_root references.
_GENERICS = {
    "x.umd.json": {"name": "m", "scope": "kernel", "match_symbol": "m"},
    "x.ued.json": {"name": "t:e", "heuristic": "uhd", "metadata": "kmd"},
    "x.udd.json": {"name": "d", "dispatch_symbol": "d"},
    "x.uhd.json": {"name": "h", "kind": "native", "payload": "h"},
    "x.kmd.json": {"name": "k", "fields": []},
}


def _write_root(root: Path, ukds, kdp_arch=("gfx950",), kdp_file="x.kdp.json"):
    root.mkdir(parents=True, exist_ok=True)
    kdp = {
        "version": "1.0",
        "id": f"kdp-{kdp_file}",
        "name": "x",
        "arch": list(kdp_arch) if kdp_arch is not None else [],
        "matchers": ["umd"],
        "engine": "ued",
        "dispatch": "udd",
        "kernelDescriptors": list(ukds),
    }
    (root / kdp_file).parent.mkdir(parents=True, exist_ok=True)
    (root / kdp_file).write_text(json.dumps(kdp, indent=2) + "\n")
    for file, doc in _GENERICS.items():
        doc = {"version": "1.0", "id": file.split(".")[1], **doc}
        (root / file).write_text(json.dumps(doc) + "\n")


def _write_standalone(root: Path, ukd_id="ukd-standalone-id", arch=None):
    ukd = _rocke("standalone", builder="b9", arch=arch, ukd_id=ukd_id)
    (root / "s.ukd.json").write_text(json.dumps(ukd) + "\n")


def _kept(out: Path, kdp_file="x.kdp.json"):
    kdp = json.loads((out / kdp_file).read_text())
    return [d["name"] for d in kdp["kernelDescriptors"]]


# --- compile-group selection ------------------------------------------------
def test_one_ukd_per_compile_group(tmp_path):
    ukds = [
        _rocke("r_b", "b1"),
        _rocke("r_a", "b1"),
        _rocke("r_c", "b2"),
        _hip("h_b", "a.cpp", {"defines": {"X": 1}}),
        _hip("h_a", "a.cpp", {"defines": {"X": 1}}),
        _hip("h_other_build", "a.cpp", {"defines": {"X": 2}}),
        _hip("h_other_source", "b.cpp", {"defines": {"X": 1}}),
    ]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    # rocke: 2 builders; hip: 3 (source, build) pairs.
    assert {(e["kind"], e["name"]) for e in expect} == {
        ("rocke", "r_a"),
        ("rocke", "r_c"),
        ("hip", "h_a"),
        ("hip", "h_other_build"),
        ("hip", "h_other_source"),
    }
    assert sorted(_kept(tmp_path / "w" / "root")) == sorted(e["name"] for e in expect)


def test_rocke_group_key_includes_source(tmp_path):
    # The same builder name in two modules is two compile paths.
    ukds = [
        _rocke("r_a", "b1", source="common.py"),
        _rocke("r_b", "b1", source="arch.py"),
    ]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert sorted(e["name"] for e in expect) == ["r_a", "r_b"]


def test_hsaco_compiles_nothing_so_one_per_kdp(tmp_path):
    ukds = [_hsaco("z_hsaco"), _hsaco("y_hsaco"), _rocke("r")]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert {(e["kind"], e["name"]) for e in expect} == {
        ("hsaco", "y_hsaco"),
        ("rocke", "r"),
    }


def test_hsaco_ukd_for_other_arch_is_ignored(tmp_path):
    ukds = [
        _hsaco("a_942", arch=["gfx942"]),
        _hsaco("b_950", arch=["gfx950"]),
    ]
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert [e["name"] for e in expect] == ["b_950"]


def test_passthrough_kind_expect_carries_authored_kernel_source(tmp_path):
    emb = _embedded("e")
    other = _embedded("e2")
    _write_root(tmp_path / "src", [other, emb, _rocke("r")])
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    by_name = {e["name"]: e for e in expect}
    # One passthrough UKD per KDP (nothing compiles); it records its authored block.
    assert set(by_name) == {"e", "r"}
    assert by_name["e"]["kernel_source"] == emb["kernel_source"]
    # kpack-output entries keep the three-field schema.
    assert set(by_name["r"]) == {"kdp", "name", "kind"}


def test_non_varying_group_keeps_first_by_name(tmp_path):
    # Every spec is equal, so the group keeps one UKD whatever the authored order.
    ukds = [_rocke(n) for n in ("m", "c", "x", "a2", "a1")]
    # Ids sort opposite to names, so a pick by id would choose differently.
    for i, u in enumerate(sorted(ukds, key=lambda u: u["name"])):
        u["id"] = f"id-{len(ukds) - i}"
    for i, order in enumerate(itertools.permutations(ukds, 3)):
        src, out = tmp_path / f"src{i}", tmp_path / f"w{i}" / "root"
        _write_root(src, order)
        expect = _derive(src, "gfx950", out)
        want = min(u["name"] for u in order)
        assert [e["name"] for e in expect] == [want]
        assert _kept(out) == [want]


def _sweep_group():
    """A rocke group: dtype x causal x block_n vary, head does not."""
    ukds = []
    for dtype, causal, block_n in itertools.product(
        ("bf16", "fp16"), (False, True), (64, 128, 256)
    ):
        name = f"k_{dtype}_c{int(causal)}_n{block_n}"
        spec = {"dtype": dtype, "causal": causal, "block_n": block_n, "head": 128}
        ukds.append(_rocke(name, spec=spec))
    # An authored order that is neither sorted nor reverse-sorted.
    return ukds[3:] + ukds[:3]


def _pairs(ukd):
    return {(k, json.dumps(v)) for k, v in ukd["kernel_source"]["spec"].items()}


def test_sweep_covers_every_varying_value(tmp_path):
    ukds = _sweep_group()
    _write_root(tmp_path / "src", ukds)
    out = tmp_path / "w" / "root"
    kept = [e["name"] for e in _derive(tmp_path / "src", "gfx950", out)]
    by_name = {u["name"]: u for u in ukds}
    covered = set().union(*(_pairs(by_name[n]) for n in kept))
    varying = {p for u in ukds for p in _pairs(u) if p[0] != "head"}
    assert varying <= covered
    # block_n has three values, so three UKDs cover the group; the greedy pick finds
    # that cover instead of packing all twelve.
    assert len(kept) == 3
    # Written in authored order.
    assert _kept(out) == [u["name"] for u in ukds if u["name"] in kept]


def test_sweep_is_deterministic(tmp_path):
    ukds = _sweep_group()
    picks = []
    for i, order in enumerate([ukds, ukds, ukds[::-1]]):
        src = tmp_path / f"src{i}"
        _write_root(src, order)
        picks.append(
            sorted(e["name"] for e in _derive(src, "gfx950", tmp_path / f"w{i}/root"))
        )
    assert picks[0] == picks[1] == picks[2]


def test_sweep_is_per_compile_group(tmp_path):
    # A value varying across groups but not within one is not swept.
    ukds = [
        _rocke("a1", "b1", spec={"n": 1}),
        _rocke("a2", "b1", spec={"n": 1}),
        _rocke("b1", "b2", spec={"n": 2}),
    ]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert [e["name"] for e in expect] == ["a1", "b1"]


def test_hip_build_key_ignores_dict_key_order(tmp_path):
    ukds = [
        _hip("h1", build={"defines": {"A": 1, "B": 2}}),
        _hip("h2", build={"defines": {"B": 2, "A": 1}}),
    ]
    _write_root(tmp_path / "src", ukds)
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["h1"]


def test_pick_only_among_ukds_shipping_for_arch(tmp_path):
    # "a_*" sorts first but ships for gfx942 only; a wildcard UKD and an explicit
    # match remain candidates of the same group.
    ukds = [
        _rocke("a_942_only", arch=["gfx942"]),
        _rocke("b_wild"),
        _rocke("c_explicit", arch=["gfx950"]),
        _rocke("d_empty", arch=[]),
    ]
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert [e["name"] for e in expect] == ["b_wild"]
    expect = _derive(tmp_path / "src", "gfx942", tmp_path / "w2" / "root")
    assert [e["name"] for e in expect] == ["a_942_only"]


def test_kdp_not_shipping_for_arch_is_copied_unchanged(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a"), _rocke("b", "b2")], kdp_arch=("gfx942",))
    assert _derive(src, "gfx950", tmp_path / "w" / "root") == []
    assert (tmp_path / "w/root/x.kdp.json").read_bytes() == (
        src / "x.kdp.json"
    ).read_bytes()


def test_wildcard_kdp_ships_everywhere(tmp_path):
    _write_root(tmp_path / "src", [_rocke("a")], kdp_arch=None)
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["a"]


def test_expect_records_kdp_relative_path(tmp_path):
    _write_root(tmp_path / "src", [_rocke("a")], kdp_file="sub/dir/x.kdp.json")
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert expect == [{"kdp": "sub/dir/x.kdp.json", "name": "a", "kind": "rocke"}]


def test_dot_prefixed_kdp_is_copied_but_not_expected(tmp_path):
    # The packer skips dot-prefixed paths, so derive must not expect their UKDs.
    src = tmp_path / "src"
    _write_root(src, [_rocke("a")])
    _write_root(src, [_rocke("hidden")], kdp_file=".hidden/x.kdp.json")
    expect = _derive(src, "gfx950", tmp_path / "w" / "root")
    assert [e["name"] for e in expect] == ["a"]
    assert (tmp_path / "w/root/.hidden/x.kdp.json").read_bytes() == (
        src / ".hidden/x.kdp.json"
    ).read_bytes()


# --- configure-time refusals (exit 2, nothing written) ----------------------
def _refused(tmp_path, ukds, message, kdp_arch=("gfx950",)):
    _write_root(tmp_path / "src", ukds, kdp_arch=kdp_arch)
    _write_standalone(tmp_path / "src")
    r = run_derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2, r.stderr
    assert r.stderr.startswith("hkp_probe_derive:")
    assert message in r.stderr
    assert "Traceback" not in r.stderr
    assert not (tmp_path / "w").exists()


def test_standalone_ukd_reference_exits_2(tmp_path):
    _refused(
        tmp_path,
        [_rocke("a"), "ukd-standalone-id"],
        "standalone UKD references unsupported by probe derive",
    )


def test_standalone_reference_in_real_fixture_exits_2(tmp_path):
    # tests/fixtures/main: pointwise.kdp.json ships for gfx950 and references
    # ukd-pointwise-add-f32-b128 by id.
    r = run_derive(MAIN_FIXTURE, "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2, r.stderr
    assert "pointwise.kdp.json" in r.stderr
    assert "ukd-pointwise-add-f32-b128" in r.stderr
    assert not (tmp_path / "w").exists()


def test_standalone_reference_in_other_arch_kdp_is_ignored(tmp_path):
    _write_root(
        tmp_path / "src", [_rocke("a"), "ukd-standalone-id"], kdp_arch=("gfx942",)
    )
    _write_standalone(tmp_path / "src")
    assert _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root") == []


def test_kind_without_probe_support_exits_2(tmp_path):
    # kpack is a kind the packer accepts but the probe has no entry for.
    _refused(tmp_path, [_rocke("a"), _kpack("m")], "kind 'kpack' has no probe support")


def test_unsupported_kind_for_other_arch_is_ignored(tmp_path):
    ukds = [_rocke("a"), _kpack("m", arch=["gfx942"])]
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["a"]


@pytest.mark.parametrize(
    "ukd, message",
    [
        (_ukd("m", {"kind": "mystery"}), "unsupported kind 'mystery'"),
        (_ukd("h", {"kind": "hip", "source": "a.cpp", "entry": "h"}), "invalid build"),
        (
            _ukd("r", {"kind": "rocke", "builder": "b", "spec": {}}),
            "missing required field 'source'",
        ),
    ],
    ids=["unknown-kind", "hip-build", "rocke-source"],
)
def test_packer_rejection_exits_2(tmp_path, ukd, message):
    # Derive loads the root with the packer's loader, so what the packer rejects
    # fails configure, with the packer's message.
    _refused(tmp_path, [_rocke("a"), ukd], message)


def test_ukd_arch_outside_kdp_arch_exits_2(tmp_path):
    # The packer rejects this root; derive must not hide it by dropping the UKD.
    _refused(tmp_path, [_rocke("a", arch=["gfx942"])], "is not a subset of KDP")


def test_kept_ukds_with_equal_names_exit_2(tmp_path):
    ukds = [_rocke("same", "b1"), _rocke("same", "b2", ukd_id="id-same-2")]
    _refused(tmp_path, ukds, "share a name")


def test_malformed_kdp_exits_2(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.kdp.json").write_text("{ not json")
    r = run_derive(src, "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2
    assert "malformed descriptor JSON in bad.kdp.json" in r.stderr
    assert "Traceback" not in r.stderr


# --- --ukd: keep exactly the listed UKDs -------------------------------------
def test_ukd_list_keeps_exactly_the_listed_ukds(tmp_path):
    # Several of one compile group, and a pick the default mode would not make.
    ukds = [_rocke("r_a"), _rocke("r_b"), _rocke("r_c"), _hip("h_a"), _hip("h_b")]
    _write_root(tmp_path / "src", ukds)
    out = tmp_path / "w" / "root"
    expect = _derive(tmp_path / "src", "gfx950", out, ukds=["r_c", "r_b", "h_b"])
    assert [(e["kind"], e["name"]) for e in expect] == [
        ("rocke", "r_b"),
        ("rocke", "r_c"),
        ("hip", "h_b"),
    ]
    assert _kept(out) == ["r_b", "r_c", "h_b"]


def test_ukd_list_omits_kdps_keeping_nothing(tmp_path):
    src, out = tmp_path / "src", tmp_path / "w" / "root"
    _write_root(src, [_rocke("a")], kdp_file="keep.kdp.json")
    _write_root(src, [_rocke("b")], kdp_file="drop.kdp.json")
    # Derived once without --ukd, so the omission must also remove the stale copy.
    _derive(src, "gfx950", out)
    expect = _derive(src, "gfx950", out, ukds=["a"])
    assert [(e["kdp"], e["name"]) for e in expect] == [("keep.kdp.json", "a")]
    assert not (out / "drop.kdp.json").exists()
    assert (out / "x.kmd.json").is_file()


def test_ukd_list_copies_kdps_not_shipping_for_arch(tmp_path):
    src, out = tmp_path / "src", tmp_path / "w" / "root"
    _write_root(src, [_rocke("a")], kdp_file="keep.kdp.json")
    _write_root(src, [_rocke("b")], kdp_arch=("gfx942",), kdp_file="other.kdp.json")
    _derive(src, "gfx950", out, ukds=["a"])
    copied = (out / "other.kdp.json").read_bytes()
    assert copied == (src / "other.kdp.json").read_bytes()


def _ukd_refused(tmp_path, ukds, listed, message):
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    _write_standalone(tmp_path / "src")
    r = run_derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root", ukds=listed)
    assert r.returncode == 2, r.stderr
    assert r.stderr.startswith("hkp_probe_derive:")
    assert message in r.stderr
    assert not (tmp_path / "w").exists()


def test_ukd_list_unknown_name_exits_2(tmp_path):
    _ukd_refused(tmp_path, [_rocke("a")], ["a", "nope"], "['nope'] matched no")


def test_ukd_list_name_only_for_other_arch_exits_2(tmp_path):
    ukds = [_rocke("a"), _rocke("b", arch=["gfx942"])]
    _ukd_refused(tmp_path, ukds, ["a", "b"], "['b'] matched no")


def test_ukd_list_name_only_in_kdp_for_other_arch_exits_2(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a")], kdp_file="keep.kdp.json")
    _write_root(src, [_rocke("b")], kdp_arch=("gfx942",), kdp_file="other.kdp.json")
    r = run_derive(src, "gfx950", tmp_path / "w" / "root", ukds=["a", "b"])
    assert r.returncode == 2, r.stderr
    assert "['b'] matched no" in r.stderr


def test_ukd_list_duplicate_entry_exits_2(tmp_path):
    _ukd_refused(tmp_path, [_rocke("a")], ["a", "a"], "['a'] more than once")


def test_ukd_list_ignores_standalone_ref_in_kdp_keeping_nothing(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a")], kdp_file="keep.kdp.json")
    _write_root(src, [_rocke("b"), "ukd-standalone-id"], kdp_file="drop.kdp.json")
    _write_standalone(src)
    expect = _derive(src, "gfx950", tmp_path / "w" / "root", ukds=["a"])
    assert [e["name"] for e in expect] == ["a"]


def test_ukd_list_standalone_ref_in_kept_kdp_exits_2(tmp_path):
    _ukd_refused(
        tmp_path,
        [_rocke("a"), "ukd-standalone-id"],
        ["a"],
        "standalone UKD references unsupported by probe derive",
    )


def test_ukd_list_arch_outside_kdp_arch_exits_2(tmp_path):
    ukds = [_rocke("a", arch=["gfx90a", "gfx950"])]
    _ukd_refused(tmp_path, ukds, ["a"], "is not a subset of KDP")


def test_ukd_list_kept_ukds_with_equal_names_exit_2(tmp_path):
    ukds = [_rocke("same", "b1"), _rocke("same", "b2", ukd_id="id-same-2")]
    _ukd_refused(tmp_path, ukds, ["same"], "share a name")


def test_ukd_list_unregistered_kind_of_kept_ukd_exits_2(tmp_path):
    _ukd_refused(tmp_path, [_rocke("a"), _kpack("m")], ["m"], "has no probe support")


def test_ukd_list_ignores_unlisted_ukds_of_kept_kdp(tmp_path):
    # The kind of an unlisted UKD does not matter.
    ukds = [_rocke("a"), _kpack("m")]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root", ukds=["a"])
    assert [e["name"] for e in expect] == ["a"]


# --- idempotence ------------------------------------------------------------
def _stamp_old(out: Path):
    for p in [*out.rglob("*"), out.parent / "expect.json"]:
        if p.is_file():
            os.utime(p, ns=(10**18, 10**18))


def _mtimes(out: Path):
    return {
        p.relative_to(out.parent).as_posix(): p.stat().st_mtime_ns
        for p in [*out.rglob("*"), out.parent / "expect.json"]
        if p.is_file()
    }


def test_second_derive_touches_nothing(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("b"), _rocke("a")])
    out = tmp_path / "w" / "root"
    first = _derive(src, "gfx950", out)
    _stamp_old(out)
    before = _mtimes(out)
    assert before and set(before) >= {
        "root/x.kdp.json",
        "root/x.kmd.json",
        "expect.json",
    }
    assert _derive(src, "gfx950", out) == first
    assert _mtimes(out) == before


def test_rederive_follows_source_changes(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("b"), _rocke("a")])
    out = tmp_path / "w" / "root"
    _derive(src, "gfx950", out)
    (out / "stale.txt").write_text("stale")
    (out / "stale_dir").mkdir()
    (out / "stale_dir" / "f.json").write_text("{}")
    _write_root(src, [_rocke("b"), _rocke("0first")])
    kmd = '{"version": "1.0", "id": "kmd", "name": "k2", "fields": []}\n'
    (src / "x.kmd.json").write_text(kmd)
    expect = _derive(src, "gfx950", out)
    assert [e["name"] for e in expect] == ["0first"]
    assert _kept(out) == ["0first"]
    assert (out / "x.kmd.json").read_text() == kmd
    assert sorted(p.name for p in out.rglob("*")) == sorted(["x.kdp.json", *_GENERICS])


# --- --list-arches ----------------------------------------------------------
def _list_arches(src: Path):
    r = subprocess.run(
        [sys.executable, str(PROBE_DERIVE), "--list-arches", "--from", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_list_arches_unions_explicit_kdp_and_ukd_arches(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a", arch=["gfx942"])], kdp_arch=("gfx950", "gfx942"))
    # A wildcard KDP contributes nothing itself; its explicit UKD arches do, inline
    # and standalone.
    wild = [_rocke("b", arch=["gfx1201"]), _rocke("c"), "ukd-standalone-id"]
    _write_root(src, wild, kdp_arch=None, kdp_file="wild.kdp.json")
    _write_standalone(src, arch=["gfx90a"])
    assert _list_arches(src) == ["gfx1201", "gfx90a", "gfx942", "gfx950"]


def test_list_arches_excludes_wildcards(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a"), _rocke("b", arch=[])], kdp_arch=None)
    assert _list_arches(src) == []


def test_list_arches_packer_rejection_exits_2(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a", arch=["gfx942"])])
    r = subprocess.run(
        [sys.executable, str(PROBE_DERIVE), "--list-arches", "--from", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
    assert r.stderr.startswith("hkp_probe_derive:")
    assert "is not a subset of KDP" in r.stderr


# --- the kinds registry -----------------------------------------------------
def _tool_module(name):
    sys.path.insert(0, str(PROBE_DERIVE.parent))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_kinds_registry_is_complete():
    kinds = _tool_module("hkp_probe_kinds").KINDS
    checks = _tool_module("hkp_probe_assert")._PROVENANCE_CHECKS
    assert kinds
    used = set()
    for kind, entry in kinds.items():
        assert all(isinstance(f, str) and f for f in entry.group_by), kind
        # Every named provenance check has an implementation in the assert tool.
        assert set(entry.provenance) <= set(checks), kind
        used |= set(entry.provenance)
    # ... and no implementation is dead.
    assert used == set(checks)


def test_every_registered_output_type_has_checks():
    kinds = _tool_module("hkp_probe_kinds").KINDS
    outputs = _tool_module("hkp_probe_assert")._OUTPUT_CHECKS
    assert {k.output for k in kinds.values()} <= set(outputs)
    # ... and no check set is dead.
    assert set(outputs) <= {k.output for k in kinds.values()}
