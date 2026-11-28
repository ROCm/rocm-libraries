# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The golden fingerprint (rocke.core.golden_file) and what it is for.

Two changes that re-record the same golden family on different trees must
conflict in git, so the second is re-recorded on the merged tree instead of
landing a golden that matches neither. The merge tests below build real git
histories from files the real record writers produce.
"""

from __future__ import annotations

import json

import pytest
from rocke.core.golden_file import (
    FINGERPRINT_KEY,
    fingerprint,
    golden_json_text,
    rerecord_command,
    rerecord_hint,
    stamp_fingerprint,
    verify_fingerprint,
    write_golden_json,
)
from rocke.core.ir_golden import (
    GOLDEN_ALIASES,
    GOLDEN_PROFILE_FLAVORS,
    GOLDEN_SCHEMA,
    check_golden,
    compare,
    drift_families,
    drift_hint,
    write_golden_shards,
)
from rocke.tests.git_merge import merges_cleanly


def _cases(**shas: str) -> dict:
    return {
        "cases": {name: {"sha256": sha} for name, sha in sorted(shas.items())},
        "schema": "sample/v1",
    }


def _profile_document(shas: dict[str, str]) -> dict:
    sub = {
        "cases": {case: {"sha256": sha} for case, sha in shas.items()},
        "expected_failures": {},
    }
    return {
        "schema": GOLDEN_SCHEMA,
        "aliases": GOLDEN_ALIASES,
        "flavors": {flavor: sub for flavor in GOLDEN_PROFILE_FLAVORS},
    }


def test_fingerprint_is_the_sha256_of_canonical_json():
    # Key order and whitespace do not move it; content does.
    assert fingerprint({"b": 1, "a": [1, 2]}) == fingerprint({"a": [1, 2], "b": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert fingerprint({}) == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_stamped_document_verifies_and_covers_everything_but_itself():
    stamped = stamp_fingerprint(_cases(x="aa", y="bb"))
    assert stamped[FINGERPRINT_KEY] == fingerprint(_cases(x="aa", y="bb"))
    assert verify_fingerprint(stamped) == (True, "")
    # Re-stamping replaces the old fingerprint rather than covering it.
    assert stamp_fingerprint(stamped) == stamped


def test_written_golden_reads_back_verified(tmp_path):
    path = tmp_path / "golden" / "sample.json"
    write_golden_json(path, _cases(x="aa"))
    assert path.read_text(encoding="utf-8") == golden_json_text(_cases(x="aa"))
    assert verify_fingerprint(json.loads(path.read_text())) == (True, "")


def test_missing_fingerprint_is_refused():
    ok, reason = verify_fingerprint(_cases(x="aa"))
    assert not ok
    assert reason == "no 'fingerprint' recorded"


def test_edited_entries_make_the_fingerprint_stale():
    edited = stamp_fingerprint(_cases(x="aa"))
    edited["cases"]["x"]["sha256"] = "a2"
    ok, reason = verify_fingerprint(edited)
    assert not ok
    assert reason.startswith("stale 'fingerprint' - recorded=")
    assert "the entries changed without a re-record" in reason


def test_stamp_refuses_a_non_object():
    with pytest.raises(TypeError, match="golden document must be a JSON object"):
        stamp_fingerprint(["not", "an", "object"])


def test_rerecord_command_names_the_tool_golden_and_families():
    assert rerecord_command("kda_gfx942") == (
        "python tools/golden.py rerecord kda_gfx942"
    )
    assert rerecord_command("representative", ["gemm", "attention", "gemm"]) == (
        "python tools/golden.py rerecord representative --only attention,gemm"
    )
    assert rerecord_hint("conv").endswith(
        "re-record from the rocke directory:\n"
        "    python tools/golden.py rerecord conv"
    )


@pytest.mark.parametrize("name", ["", "  "])
def test_rerecord_command_refuses_an_empty_name(name):
    with pytest.raises(ValueError, match="expected a name"):
        rerecord_command(name)


# --- the drift a shard golden reports names its families -------------------------


def test_stale_shard_golden_prints_the_command_for_its_drifting_families(tmp_path):
    recorded = {"gemm/gfx942/a": "aa", "gemm/gfx942/b": "bb", "moe/gfx950/c": "cc"}
    write_golden_shards(tmp_path, _profile_document(recorded))
    current = {**recorded, "gemm/gfx942/b": "b2", "conv/gfx950/new": "dd"}
    del current["moe/gfx950/c"]

    def run(flavor: str) -> dict:
        return _profile_document(current)["flavors"][flavor]

    drift = check_golden(tmp_path, run)
    assert drift_families(drift) == ["conv", "gemm", "moe"]
    assert drift_hint("representative", drift).endswith(
        "python tools/golden.py rerecord representative --only conv,gemm,moe"
    )


def test_every_kind_of_compare_drift_names_its_family():
    # drift_families parses compare's message text: each kind of line compare
    # writes, bare (one flavor checked) and flavor-prefixed, maps to its family.
    base = {
        "cases": {"gemm/gfx942/a": {"sha256": "aa"}, "moe/gfx950/b": {"sha256": "bb"}},
        "expected_failures": {"conv/gfx942/c": {"type": "ValueError", "message": "x"}},
    }
    current = {
        "cases": {"gemm/gfx942/a": {"sha256": "a2"}, "lse/gfx942/d": {"sha256": "dd"}},
        "expected_failures": {
            "conv/gfx942/c": {"type": "ValueError", "message": "y"},
            "attn/gfx950/e": {"type": "KeyError", "message": "z"},
        },
    }
    drift = compare(base, current)
    assert len(drift) == 5  # missing, two new, a changed hash, a changed failure
    expected = ["attn", "conv", "gemm", "lse", "moe"]
    assert drift_families(drift) == expected
    assert drift_families(f"[llvm22] {line}" for line in drift) == expected


def test_drift_naming_no_case_rerecords_the_whole_golden():
    drift = ["golden profile documents ['llvm20'] do not match [...]; re-bless it"]
    assert drift_families(drift) is None
    assert drift_hint("conv", drift).endswith("python tools/golden.py rerecord conv")


# --- two re-records of one family conflict in git; of two families they merge ---


def _shard_text(tmp_path, name: str, shas: dict[str, str]) -> str:
    directory = tmp_path / name
    write_golden_shards(directory, _profile_document(shas))
    (shard,) = directory.glob("*.json")
    return shard.read_text()


def test_two_rerecords_of_one_document_family_conflict(tmp_path):
    base = {"g.json": golden_json_text(_cases(x="aa", y="bb"))}
    # Each change re-records a different case of the same family, so their hash
    # edits alone would merge into a file that matches neither tree.
    left = {"g.json": golden_json_text(_cases(x="a2", y="bb"))}
    right = {"g.json": golden_json_text(_cases(x="aa", y="b2"))}
    assert not merges_cleanly(tmp_path, base, left, right)


def test_without_the_fingerprint_the_same_rerecords_merge_silently(tmp_path):
    def plain(document: dict) -> str:
        return json.dumps(document, indent=2, sort_keys=True) + "\n"

    base = {"g.json": plain(_cases(x="aa", y="bb"))}
    left = {"g.json": plain(_cases(x="a2", y="bb"))}
    right = {"g.json": plain(_cases(x="aa", y="b2"))}
    assert merges_cleanly(tmp_path, base, left, right)


def test_two_rerecords_of_one_shard_conflict_and_of_two_shards_merge(tmp_path):
    def shard(tag: str, family: str, first: str, second: str) -> dict[str, str]:
        text = _shard_text(
            tmp_path / tag,
            family,
            {f"{family}/gfx942/first": first, f"{family}/gfx942/second": second},
        )
        return {f"{family}.json": text}

    base = {**shard("b1", "gemm", "aa", "bb"), **shard("b2", "moe", "cc", "dd")}
    same_left = shard("l1", "gemm", "a2", "bb")
    same_right = shard("r1", "gemm", "aa", "b2")
    assert not merges_cleanly(tmp_path / "same", base, same_left, same_right)
    other_right = shard("r2", "moe", "c2", "dd")
    assert merges_cleanly(tmp_path / "other", base, same_left, other_right)


def test_the_merge_helper_raises_on_a_failing_git_step(tmp_path):
    # A broken step must not read as "no conflict": an empty change set leaves
    # nothing to commit, so the commit fails and the helper raises.
    with pytest.raises(RuntimeError, match=r"git commit --quiet -m \w+ failed"):
        merges_cleanly(tmp_path, {"g.json": "{}\n"}, {"g.json": "{}\n"}, {})
