"""The `launches` checks of hkp_probe_assert.py, on a packed multi-launch UKD.

These drive the tool's check functions directly: the archive is a stand-in holding one
module blob, and the argument list the blob "exports" is a table. What they pin is the
assertion logic, not the packer's output, so no compile and no kpack archive is needed.
"""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from probe_support import PROBE_ASSERT

PASS_ARCH = "gfx950"
TOC_KEY = "module"
DECODE = "k_decode"
MERGE = "k_merge"
BLOB = f"ELF {DECODE} {MERGE}".encode()
SIGNATURE = [{"kind": "global_buffer", "size": 8, "offset": 0}]


def _assert_tool():
    sys.path.insert(0, str(PROBE_ASSERT.parent))
    try:
        return __import__("hkp_probe_assert")
    finally:
        sys.path.pop(0)


tool = _assert_tool()


class _Archive:
    def __init__(self, blobs):
        self._blobs = blobs

    def get_kernel(self, toc_key, arch):
        return self._blobs.get(toc_key)


def _source(symbol, signature=None, **extra):
    return {
        "kind": "kpack",
        "library": "../kpack/x.kpack",
        "toc_key": TOC_KEY,
        "symbol": symbol,
        "sha256": hashlib.sha256(BLOB).hexdigest(),
        "signature": SIGNATURE if signature is None else signature,
        **extra,
    }


def _evidence(decode=DECODE, merge=MERGE):
    """The compiler's record for each launch of one module compiled once."""
    observed = {"symbol": decode, "companions": {"merge": merge}}
    return {"decode": {"observations": observed}, "merge": {"observations": observed}}


def _ukd(**overrides):
    ukd = {
        "name": "m",
        "arch": [PASS_ARCH],
        "sources": {"decode": _source(DECODE), "merge": _source(MERGE)},
        "provenance": {
            "origin_kind": "rocke",
            "rocke_wheel_sha256": "w",
            "comgr_path": "c",
            "effective_spec": _evidence(),
        },
    }
    ukd.update(overrides)
    return ukd


EXPECTED = {
    "kdp": "x.kdp.json",
    "name": "m",
    "kind": "rocke",
    "arch": PASS_ARCH,
    "launches": {"decode": None, "merge": "merge"},
}


@pytest.fixture(autouse=True)
def _signatures(monkeypatch):
    """What the blob exports: every symbol of the module takes SIGNATURE."""

    def fake(blob, symbol, label):
        if symbol not in (DECODE, MERGE):
            raise tool.HkpPackError(f"{label}: no kernel {symbol!r}")
        return SIGNATURE

    monkeypatch.setattr(tool, "kernel_signature", fake)


def _run(ukd, expected=EXPECTED):
    failures = tool._Failures()
    archives = {PASS_ARCH: _Archive({TOC_KEY: BLOB})}
    args = SimpleNamespace(expect_comgr=None)
    tool._check_kpack_ukd(args, "x.kdp.json:m", ukd, expected, archives, failures)
    return [line.split(": ", 2)[1].removeprefix("FAIL ") for line in failures.lines]


def test_a_well_formed_multi_launch_ukd_passes():
    assert _run(_ukd()) == []


def test_a_single_launch_ukd_is_unaffected():
    ukd = _ukd()
    del ukd["sources"]
    ukd["kernel_source"] = _source(DECODE)
    del ukd["provenance"]["effective_spec"]
    expected = {k: v for k, v in EXPECTED.items() if k != "launches"}
    assert _run(ukd, expected) == []


def test_kernel_source_shipped_where_launches_are_expected():
    ukd = _ukd()
    del ukd["sources"]
    ukd["kernel_source"] = _source(DECODE)
    assert _run(ukd) == ["launches"]


def test_sources_shipped_where_none_are_expected():
    expected = {k: v for k, v in EXPECTED.items() if k != "launches"}
    assert _run(_ukd(), expected) == ["launches"]


def test_both_keys_shipped():
    assert _run(_ukd(kernel_source=_source(DECODE))) == ["launches"]


def test_sources_that_is_not_an_object():
    assert _run(_ukd(sources=[_source(DECODE), _source(MERGE)])) == ["launches"]


def test_a_dropped_launch():
    ukd = _ukd()
    del ukd["sources"]["merge"]
    assert "launches" in _run(ukd)


def test_a_renamed_launch():
    ukd = _ukd()
    ukd["sources"]["reduce"] = ukd["sources"].pop("merge")
    assert "launches" in _run(ukd)


def test_symbols_swapped_between_launches():
    # Both symbols are in the one module and take the same arguments, so only the
    # compiler's record of which launch is which can see the swap.
    ukd = _ukd(sources={"decode": _source(MERGE), "merge": _source(DECODE)})
    assert _run(ukd) == ["launches", "launches"]


def test_a_launch_with_a_role_is_compared_with_the_recorded_companion():
    # `decode` takes the primary and `merge` the companion `merge`: pointing the
    # companion at the primary's symbol must not pass because the primary matches.
    ukd = _ukd(sources={"decode": _source(DECODE), "merge": _source(DECODE)})
    assert _run(ukd) == ["launches"]


def test_a_launch_without_recorded_observations():
    ukd = _ukd()
    ukd["provenance"]["effective_spec"]["merge"] = {}
    assert _run(ukd) == ["launches"]


def test_a_ukd_with_no_compiler_evidence_is_reported():
    ukd = _ukd()
    del ukd["provenance"]["effective_spec"]
    # Each launch is unbound, so each reports it; the swap check cannot run without it.
    assert _run(ukd) == ["launches", "launches"]


def test_a_signature_that_disagrees_with_the_object():
    wrong = [{"kind": "by_value", "size": 4, "offset": 0}]
    ukd = _ukd(sources={"decode": _source(DECODE), "merge": _source(MERGE, wrong)})
    assert _run(ukd) == ["launch-signature"]


def test_a_symbol_the_object_does_not_hold():
    ukd = _ukd(sources={"decode": _source(DECODE), "merge": _source("k_absent")})
    failures = _run(ukd)
    assert "symbol" in failures
    assert "launch-signature" in failures


def test_a_launch_that_is_not_kpack():
    ukd = _ukd()
    ukd["sources"]["merge"] = {"kind": "rocke"}
    assert _run(ukd) == ["ukd-kind"]


def test_an_empty_signature_is_reported_for_the_launch():
    ukd = _ukd()
    ukd["sources"]["merge"]["signature"] = []
    assert "signature" in _run(ukd)


@pytest.mark.parametrize(
    "launches",
    [{}, [], {"decode": 1}, "decode"],
    ids=["empty", "list", "role-not-a-string", "string"],
)
def test_a_malformed_launches_entry_in_expect_is_a_usage_error(tmp_path, launches):
    path = tmp_path / "expect.json"
    path.write_text(json.dumps([{**EXPECTED, "launches": launches}]), encoding="utf-8")
    with pytest.raises(tool._UsageError):
        tool._load_expect(path)


def test_a_well_formed_launches_entry_in_expect_loads(tmp_path):
    path = tmp_path / "expect.json"
    path.write_text(json.dumps([EXPECTED]), encoding="utf-8")
    assert (
        tool._load_expect(path)[("x.kdp.json", "m")]["launches"] == EXPECTED["launches"]
    )
