# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What `catalog_diff.py` calls the same catalog, and what it refuses to.

The gate exists because byte comparison cannot work here: `generator.py` mints
every descriptor id with `uuid.uuid4()`, so two runs of one config differ
everywhere. Masking ids is therefore the premise, and the risk that comes with
it is a mask that hides a real change. Each case below is one thing the mask
must NOT swallow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import catalog_diff  # noqa: E402


def _entry(name: str, block_n: int = 64, priority: int = 0) -> dict:
    return {
        "version": "1.0",
        "id": f"id-of-{name}",
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "source": "kernels/gfx950/attention_dense.py",
            "builder": "build_attention_dense",
            "spec": {"dtype": "bf16", "block_m": 256, "block_n": block_n},
        },
        "metadata": {"dtype": "BF16", "block_m": 256, "block_n": block_n},
        "priority": priority,
    }


def _kdp(entries: list[dict], **overrides) -> dict:
    kdp = {
        "version": "1.0",
        "id": "kdp-id",
        "name": "hipkernel:stub",
        "arch": ["gfx950"],
        "matchers": ["umd-id"],
        "engine": "ued-id",
        "dispatch": "udd-id",
        "provenance": {"specialization_contract": {"schema_version": 1}},
        "kernelDescriptors": entries,
    }
    kdp.update(overrides)
    return kdp


@pytest.fixture
def run(tmp_path):
    def go(shipped: dict, regenerated: dict, *flags: str) -> int:
        left, right = tmp_path / "a.kdp.json", tmp_path / "b.kdp.json"
        left.write_text(json.dumps(shipped))
        right.write_text(json.dumps(regenerated))
        return catalog_diff.main(
            ["--shipped", str(left), "--regenerated", str(right), *flags]
        )

    return go


class TestTheMaskHidesIdsAndNothingElse:
    def test_regenerated_ids_alone_are_not_a_difference(self, run):
        """The premise. Every id differs between two runs of one config, so if
        this failed the gate could never pass and would be abandoned."""
        shipped = _kdp([_entry("k0")])
        regenerated = json.loads(json.dumps(shipped))
        regenerated["id"] = "fresh-kdp-id"
        regenerated["kernelDescriptors"][0]["id"] = "fresh-entry-id"
        regenerated["engine"] = "fresh-ued-id"
        assert run(shipped, regenerated, "--ignore-ids") == 0

    def test_a_dropped_kernel_is_caught(self, run, capsys):
        """The failure the gate is FOR: a profile change that silently ships
        less than it did. Counting alone would catch this one."""
        assert run(_kdp([_entry("k0"), _entry("k1", 32)]), _kdp([_entry("k0")]),
                   "--ignore-ids") == 1
        out = capsys.readouterr().out
        assert "shipped 2, regenerated 1 (-1)" in out, out
        assert "only in shipped" in out and "k1" in out, out

    def test_a_changed_spec_under_an_unchanged_name_is_caught(self, run, capsys):
        """The failure counting MISSES: same count, same names, different
        compiled kernel. This is why the key carries the whole spec."""
        shipped = _kdp([_entry("k0", block_n=64)])
        regenerated = _kdp([_entry("k0", block_n=128)])
        # The name is identical on both sides, so only the spec can distinguish.
        regenerated["kernelDescriptors"][0]["name"] = "k0"
        assert run(shipped, regenerated, "--ignore-ids") == 1
        out = capsys.readouterr().out
        assert "only in shipped" in out and "only in regenerated" in out, out

    def test_a_changed_priority_is_caught(self, run):
        """Priority orders selection, so two catalogs that differ only there
        dispatch differently for the same graph."""
        assert run(
            _kdp([_entry("k0", priority=0)]),
            _kdp([_entry("k0", priority=5)]),
            "--ignore-ids",
        ) == 1

    def test_a_changed_specialization_contract_is_caught(self, run, capsys):
        """The contract is what forces kernel_source.spec and metadata to agree.
        Masked ids must not mask the fields around them."""
        regenerated = _kdp([_entry("k0")])
        regenerated["provenance"]["specialization_contract"]["metadata_fields"] = [
            "dtype"
        ]
        assert run(_kdp([_entry("k0")]), regenerated, "--ignore-ids") == 1
        assert "specialization_contract" in capsys.readouterr().out

    def test_a_dropped_matcher_is_caught_even_with_ids_masked(self, run, capsys):
        """Masking a reference's VALUE must not mask its absence: a KDP that
        lost its matcher list is a different engine however equal its kernels."""
        assert run(_kdp([_entry("k0")]), _kdp([_entry("k0")], matchers=[]),
                   "--ignore-ids") == 1
        assert "reference" in capsys.readouterr().out

    def test_a_changed_arch_is_caught(self, run):
        """Arch drives pruning; equal metadata on disjoint arches is legal, so a
        silently widened arch list changes which catalog a device loads."""
        assert run(
            _kdp([_entry("k0")]), _kdp([_entry("k0")], arch=["gfx942"]), "--ignore-ids"
        ) == 1


class TestTheDuplicateSpecCheck:
    def test_two_entries_naming_one_binary_are_reported(self, run, capsys):
        """The packer dedups compilation on the spec, so two entries with one
        spec are one kernel under two names -- and if their completed metadata
        also collides, the loader keeps one and ships less than it claims."""
        twins = _kdp([_entry("k0"), _entry("k1")])  # same spec, different names
        assert run(twins, twins, "--ignore-ids") == 1
        assert "share one spec" in capsys.readouterr().out


class TestExpectIds:
    def test_preserved_ids_pass_and_fresh_ones_fail(self, run, capsys):
        """`--expect-ids` is for checking a SPLICE, where reusing the shipped
        UUIDs is the obligation -- `extend.md`'s 'resolve by UUID, not
        filename' made checkable rather than left to review."""
        shipped = _kdp([_entry("k0")])
        assert run(shipped, json.loads(json.dumps(shipped)), "--expect-ids") == 0

        renumbered = json.loads(json.dumps(shipped))
        renumbered["engine"] = "fresh-ued-id"
        assert run(shipped, renumbered, "--ignore-ids", "--expect-ids") == 1
        assert "was not preserved" in capsys.readouterr().out


class TestUsageErrorsAreNotDifferences:
    def test_a_missing_file_exits_2(self, tmp_path, capsys):
        """Exit 1 means 'the catalogs differ'. An unreadable input has compared
        nothing, and returning 1 for it would read as a real difference."""
        existing = tmp_path / "a.kdp.json"
        existing.write_text(json.dumps(_kdp([_entry("k0")])))
        assert (
            catalog_diff.main(
                ["--shipped", str(existing), "--regenerated", str(tmp_path / "nope")]
            )
            == 2
        )
        assert "cannot read" in capsys.readouterr().err

    def test_a_file_that_is_not_a_kdp_exits_2(self, tmp_path, capsys):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps(_kdp([_entry("k0")])))
        b.write_text(json.dumps({"version": "1.0"}))
        assert catalog_diff.main(["--shipped", str(a), "--regenerated", str(b)]) == 2
        assert "no 'kernelDescriptors'" in capsys.readouterr().err
