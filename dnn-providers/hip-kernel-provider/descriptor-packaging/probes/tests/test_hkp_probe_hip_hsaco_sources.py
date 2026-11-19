"""hkp_probe_assert.py on a packed hip or hsaco multi-launch (`sources`) UKD.

These drive the tool's check functions directly, as test_hkp_probe_launches.py does for
rocke: the archive is a stand-in holding one blob that every launch shares, and the
argument list each symbol "exports" is a table. A hip or hsaco launch has no compiler
evidence (`provenance.effective_spec`) and no role, so what is pinned here is that the
structural per-launch checks apply to it and the evidence checks do not, while rocke
keeps demanding the evidence.
"""

import hashlib
from types import SimpleNamespace

import pytest

from probe_support import StandInArchive, tool_module

PASS_ARCH = "gfx950"
TOC_KEY = "module"
DECODE = "k_decode"
MERGE = "k_merge"
BLOB = f"ELF {DECODE} {MERGE}".encode()
DECODE_SIG = [{"kind": "global_buffer", "size": 8, "offset": 0}]
MERGE_SIG = [{"kind": "by_value", "size": 4, "offset": 0}]
SIGNATURES = {DECODE: DECODE_SIG, MERGE: MERGE_SIG}


tool = tool_module("hkp_probe_assert")


def _source(symbol, signature=None, **extra):
    return {
        "kind": "kpack",
        "library": "../kpack/x.kpack",
        "toc_key": TOC_KEY,
        "symbol": symbol,
        "sha256": hashlib.sha256(BLOB).hexdigest(),
        "signature": SIGNATURES[symbol] if signature is None else signature,
        **extra,
    }


def _provenance(kind):
    """What the packer records for a packed `sources` UKD of `kind`: no effective_spec."""
    if kind == "hip":
        return {"origin_kind": "hip", "hipcc_version": "v"}
    return {"origin_kind": "hsaco"}


def _ukd(kind="hip", **overrides):
    ukd = {
        "name": "m",
        "arch": [PASS_ARCH],
        "sources": {"decode": _source(DECODE), "merge": _source(MERGE)},
        "provenance": _provenance(kind),
    }
    ukd.update(overrides)
    return ukd


def _expected(kind="hip"):
    return {
        "kdp": "x.kdp.json",
        "name": "m",
        "kind": kind,
        "arch": PASS_ARCH,
        "launches": {"decode": None, "merge": None},
    }


@pytest.fixture(autouse=True)
def _signatures(monkeypatch):
    """What the blob exports: each symbol of the module takes its own SIGNATURES row."""

    def fake(blob, symbol, label):
        if symbol not in SIGNATURES:
            raise tool.HkpPackError(f"{label}: no kernel {symbol!r}")
        return SIGNATURES[symbol]

    monkeypatch.setattr(tool, "kernel_signature", fake)


def _run_lines(ukd, kind="hip", expected=None):
    failures = tool._Failures()
    archives = {PASS_ARCH: StandInArchive({TOC_KEY: BLOB})}
    args = SimpleNamespace(expect_comgr=None)
    tool._check_kpack_ukd(
        args,
        "x.kdp.json:m",
        ukd,
        _expected(kind) if expected is None else expected,
        archives,
        failures,
    )
    return [line.split(": ", 2)[1:] for line in failures.lines]


def _run(ukd, kind="hip", expected=None):
    return [found.removeprefix("FAIL ") for found, _ in _run_lines(ukd, kind, expected)]


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_well_formed_packed_sources_ukd_passes_without_compiler_evidence(kind):
    assert "effective_spec" not in _ukd(kind)["provenance"]
    assert _run(_ukd(kind), kind) == []


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_launches_sharing_one_blob_each_keep_their_own_signature(kind):
    # The two launches read one toc_key; a check that compared against the first
    # launch's signature alone would fail one of them.
    ukd = _ukd(kind)
    assert ukd["sources"]["decode"]["toc_key"] == ukd["sources"]["merge"]["toc_key"]
    assert ukd["sources"]["decode"]["signature"] != ukd["sources"]["merge"]["signature"]
    assert _run(ukd, kind) == []


def test_a_rocke_sources_ukd_without_compiler_evidence_still_fails():
    # The same shape that passes for hip: only the kind decides that evidence is owed.
    ukd = _ukd("hip")
    ukd["provenance"] = {
        "origin_kind": "rocke",
        "rocke_wheel_sha256": "w",
        "comgr_path": "c",
    }
    assert _run(ukd, "rocke") == ["launches", "launches"]
    assert _run(_ukd("hip"), "hip") == []


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_signature_that_disagrees_with_the_object_names_the_launch(kind):
    ukd = _ukd(kind)
    ukd["sources"]["merge"]["signature"] = DECODE_SIG
    found = _run_lines(ukd, kind)
    assert [f for f, _ in found] == ["FAIL launch-signature"]
    assert "launch 'merge'" in found[0][1]


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_symbol_the_object_does_not_hold_names_the_launch(kind):
    ukd = _ukd(kind)
    ukd["sources"]["merge"] = _source("k_absent", MERGE_SIG)
    found = _run_lines(ukd, kind)
    assert sorted(f for f, _ in found) == ["FAIL launch-signature", "FAIL symbol"]
    assert all("launch 'merge'" in detail for _, detail in found)


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_launch_sha256_that_is_not_the_blobs_names_the_launch(kind):
    ukd = _ukd(kind)
    ukd["sources"]["decode"]["sha256"] = "0" * 64
    found = _run_lines(ukd, kind)
    assert [f for f, _ in found] == ["FAIL sha256"]
    assert "launch 'decode'" in found[0][1]


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_launch_missing_from_the_ukd_is_reported(kind):
    ukd = _ukd(kind)
    del ukd["sources"]["merge"]
    assert "launches" in _run(ukd, kind)


@pytest.mark.parametrize("kind", ["hip", "hsaco"])
def test_a_launch_whose_toc_key_is_not_in_the_archive_is_reported(kind):
    ukd = _ukd(kind)
    ukd["sources"]["merge"]["toc_key"] = "other"
    found = _run_lines(ukd, kind)
    assert [f for f, _ in found] == ["FAIL kpack-toc"]
    assert "launch 'merge'" in found[0][1]
