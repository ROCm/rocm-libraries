# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""check_native_golden.check_family against a scripted emitter.

Pins the exit-code contract: 0 emits (blessed, matching), 1 refuses (not
blessed, no stdout), 2 "unknown config" ends the family; any other code is a
failure, including a Windows crash code, which is positive. The family must
also end where the Python reference ends, which the golden alone cannot show
for trailing refusals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_native_golden as cng  # noqa: E402 -- after the sys.path line above

LL = b"define void @k() {\n  ret void\n}\n"
GOOD = cng.sh(LL)
END = (2, b"", "unknown config index", False)
WINDOWS_ACCESS_VIOLATION = 0xC0000005


def _python_ends_at(monkeypatch, end):
    """The Python reference answers "unknown config" from index `end` on."""
    monkeypatch.setattr(
        cng, "reference_is_end", lambda family, idxs: tuple(i >= end for i in idxs)
    )


def _problems(monkeypatch, runs, blessed, python_end=None):
    """check_family over scripted (rc, stdout, stderr, had_cr) per config.

    The Python reference ends where C++ does unless `python_end` says not."""
    runs = list(runs) + [END]
    _python_ends_at(monkeypatch, len(runs) - 1 if python_end is None else python_end)
    monkeypatch.setattr(cng, "config_limit", lambda family: len(runs))
    monkeypatch.setattr(cng, "run_config", lambda exe, idx: runs[idx])
    return cng.check_family("fam", Path("emit"), blessed)


def test_contract_holds(monkeypatch):
    runs = [(0, LL, "", False), (1, b"", "build failed: refused", False)]
    assert _problems(monkeypatch, runs, {"0": GOOD}) == []


@pytest.mark.parametrize(
    "run, blessed, expect",
    [
        # right .ll, then a failing exit: not a pass (the reviewed hole)
        ((3, LL, "late error", False), {"0": GOOD}, "emitter failed (exit 3)"),
        ((1, LL, "", False), {"0": GOOD}, "C++ rejects (exit 1)"),
        # a "refusal" that exits 0 with nothing printed
        ((0, b"", "", False), {}, "exits 0 with no output"),
        # a crash on a config Python refuses is not a refusal
        (
            (WINDOWS_ACCESS_VIOLATION, b"", "", False),
            {},
            "emitter failed (exit 3221225477)",
        ),
        ((-11, b"", "segfault", False), {}, "emitter failed (exit -11)"),
        # emits where the golden blesses nothing
        ((0, LL, "", False), {}, "C++ emits, the golden blesses no .ll for it"),
        # refuses but prints
        ((1, LL, "", False), {}, "refuses (exit 1) but prints output"),
        # emits the wrong .ll
        ((0, b"other\n", "", False), {"0": GOOD}, ".ll differs from the golden"),
    ],
    ids=[
        "match-then-exit3",
        "blessed-refused",
        "exit0-no-output",
        "windows-crash",
        "posix-crash",
        "emits-unblessed",
        "refuses-with-output",
        "wrong-ll",
    ],
)
def test_contract_breaks_are_reported(monkeypatch, run, blessed, expect):
    problems = _problems(monkeypatch, [run], blessed)
    assert any(expect in p for p in problems), problems


def test_ending_before_a_blessed_config(monkeypatch):
    problems = _problems(
        monkeypatch, [(0, LL, "", False)], {"0": GOOD, "1": GOOD}, python_end=2
    )
    assert problems == [
        "C++ ends at config 1; the Python reference has more configs "
        "(refusals included)",
        "C++ ends before blessed configs ['1']",
    ]


REFUSED = (1, b"", "build failed: refused", False)


def test_a_truncated_refusal_tail_fails(monkeypatch):
    """C++ drops the last refused configs: every blessed one still matches."""
    runs = [(0, LL, "", False), REFUSED]
    problems = _problems(monkeypatch, runs, {"0": GOOD}, python_end=4)
    assert problems == [
        "C++ ends at config 2; the Python reference has more configs "
        "(refusals included)"
    ]


def test_an_extra_refusal_tail_fails(monkeypatch):
    """C++ refuses configs past the Python reference's last one."""
    runs = [(0, LL, "", False), REFUSED, REFUSED]
    problems = _problems(monkeypatch, runs, {"0": GOOD}, python_end=1)
    assert problems == ["C++ answers config 2, past the Python reference's last config"]


def test_an_unreachable_python_reference_fails(monkeypatch):
    def broken(family, idxs):
        raise RuntimeError("no Python reference fam_emit.py")

    runs = [(0, LL, "", False)] + [END]
    monkeypatch.setattr(cng, "reference_is_end", broken)
    monkeypatch.setattr(cng, "config_limit", lambda family: len(runs))
    monkeypatch.setattr(cng, "run_config", lambda exe, idx: runs[idx])
    assert cng.check_family("fam", Path("emit"), {"0": GOOD}) == [
        "cannot ask the Python reference where the family ends: "
        "no Python reference fam_emit.py"
    ]


def _capped(monkeypatch, runs, cap, blessed):
    """check_family with config_limit = cap; index cap answers runs[cap]."""
    _python_ends_at(monkeypatch, cap)
    monkeypatch.setattr(cng, "config_limit", lambda family: cap)
    monkeypatch.setattr(cng, "run_config", lambda exe, idx: runs[idx])
    return cng.check_family("fam", Path("emit"), blessed)


def test_a_family_ending_exactly_at_the_cap_passes(monkeypatch):
    runs = [(0, LL, "", False), (0, LL, "", False), END]
    assert _capped(monkeypatch, runs, 2, {"0": GOOD, "1": GOOD}) == []


def test_a_config_past_the_cap_fails(monkeypatch):
    runs = [(0, LL, "", False), (0, LL, "", False), (0, LL, "", False)]
    problems = _capped(monkeypatch, runs, 2, {"0": GOOD, "1": GOOD})
    assert problems == [
        "config 2 exists past the cap of 2 configs (exit 0); "
        "raise FAMILY_MAX_CFG['fam'] in run_diff.py"
    ]
