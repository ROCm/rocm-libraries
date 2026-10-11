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
        assert (
            run(
                _kdp([_entry("k0"), _entry("k1", 32)]),
                _kdp([_entry("k0")]),
                "--ignore-ids",
            )
            == 1
        )
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
        assert (
            run(
                _kdp([_entry("k0", priority=0)]),
                _kdp([_entry("k0", priority=5)]),
                "--ignore-ids",
            )
            == 1
        )

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
        assert (
            run(_kdp([_entry("k0")]), _kdp([_entry("k0")], matchers=[]), "--ignore-ids")
            == 1
        )
        assert "reference" in capsys.readouterr().out

    def test_a_changed_arch_is_caught(self, run):
        """Arch drives pruning; equal metadata on disjoint arches is legal, so a
        silently widened arch list changes which catalog a device loads."""
        assert (
            run(
                _kdp([_entry("k0")]),
                _kdp([_entry("k0")], arch=["gfx942"]),
                "--ignore-ids",
            )
            == 1
        )


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


def _kmd(fields: list[dict]) -> dict:
    return {"version": "1.0", "id": "kmd-id", "name": "stub fields", "fields": fields}


def _field(name: str, type_: str = "int", default=None) -> dict:
    f: dict = {"name": name, "type": type_}
    if default is not None:
        f["default_value"] = default
    return f


class TestKmdFieldComparison:
    @pytest.fixture
    def run_kmd(self, tmp_path):
        kdp = _kdp([_entry("k0")])

        def go(shipped_kmd: dict, regen_kmd: dict, *extra_flags: str) -> int:
            a_kdp = tmp_path / "a.kdp.json"
            b_kdp = tmp_path / "b.kdp.json"
            a_kmd = tmp_path / "a.kmd.json"
            b_kmd = tmp_path / "b.kmd.json"
            a_kdp.write_text(json.dumps(kdp))
            b_kdp.write_text(json.dumps(kdp))
            a_kmd.write_text(json.dumps(shipped_kmd))
            b_kmd.write_text(json.dumps(regen_kmd))
            return catalog_diff.main(
                [
                    "--shipped",
                    str(a_kdp),
                    "--regenerated",
                    str(b_kdp),
                    "--shipped-kmd",
                    str(a_kmd),
                    "--regenerated-kmd",
                    str(b_kmd),
                    "--ignore-ids",
                    *extra_flags,
                ]
            )

        return go

    def test_identical_kmd_passes(self, run_kmd):
        kmd = _kmd([_field("dtype", "string"), _field("block_n", "int", 64)])
        assert run_kmd(kmd, kmd) == 0

    def test_changed_default_value_is_caught(self, run_kmd, capsys):
        """A default_value change silently changes how the matcher completes partial
        records -- e.g., block_n: 64 vs 128 dispatches to a different binary even
        though both catalogs look identical."""
        shipped = _kmd([_field("block_n", "int", 64)])
        regen = _kmd([_field("block_n", "int", 128)])
        assert run_kmd(shipped, regen) == 1
        out = capsys.readouterr().out
        assert "default_value" in out and "block_n" in out, out

    def test_changed_type_is_caught(self, run_kmd, capsys):
        shipped = _kmd([_field("causal", "bool")])
        regen = _kmd([_field("causal", "int")])
        assert run_kmd(shipped, regen) == 1
        out = capsys.readouterr().out
        assert "type" in out and "causal" in out, out

    def test_dropped_field_is_caught(self, run_kmd, capsys):
        shipped = _kmd([_field("dtype", "string"), _field("block_n", "int")])
        regen = _kmd([_field("dtype", "string")])
        assert run_kmd(shipped, regen) == 1
        assert "block_n" in capsys.readouterr().out

    def test_added_field_is_caught(self, run_kmd, capsys):
        shipped = _kmd([_field("dtype", "string")])
        regen = _kmd([_field("dtype", "string"), _field("block_n", "int")])
        assert run_kmd(shipped, regen) == 1
        assert "block_n" in capsys.readouterr().out

    def test_only_shipped_kmd_exits_2(self, tmp_path, capsys):
        """Both KMD flags must be given together; one alone is a usage error."""
        kdp = tmp_path / "a.kdp.json"
        kmd = tmp_path / "a.kmd.json"
        kdp.write_text(json.dumps(_kdp([_entry("k0")])))
        kmd.write_text(json.dumps(_kmd([_field("x", "int")])))
        assert (
            catalog_diff.main(
                [
                    "--shipped",
                    str(kdp),
                    "--regenerated",
                    str(kdp),
                    "--shipped-kmd",
                    str(kmd),
                ]
            )
            == 2
        )
        assert "--regenerated-kmd" in capsys.readouterr().err


class TestShippedFieldKeying:
    """The comparison key uses the shipped entry's OWN spec fields, not the
    metadata_fields filter. This catches cases the narrower filter would miss."""

    def test_non_metadata_spec_field_change_is_caught(self, run, capsys):
        """A pin flip that changes a spec field outside metadata_fields
        (e.g. `persistent`) must be detected, not masked by the filter."""
        shipped_entry = {
            "version": "1.0",
            "id": "id-k0",
            "name": "k0",
            "kernel_source": {
                "kind": "rocke",
                "source": "kernels/gfx950/attention_dense.py",
                "builder": "build_attention_dense",
                "spec": {
                    "dtype": "bf16",
                    "block_m": 256,
                    "block_n": 64,
                    "persistent": False,
                },
            },
            "metadata": {"dtype": "BF16", "block_m": 256, "block_n": 64},
            "priority": 0,
        }
        regen_entry = json.loads(json.dumps(shipped_entry))
        regen_entry["kernel_source"]["spec"]["persistent"] = True
        shipped = _kdp([shipped_entry])
        regenerated = _kdp([regen_entry])
        assert run(shipped, regenerated, "--ignore-ids") == 1
        out = capsys.readouterr().out
        assert "only in shipped" in out or "only in regenerated" in out, out

    def test_regenerated_only_spec_fields_are_not_a_difference(self, run):
        """Regenerated catalogs may carry newly defaulted spec fields (e.g.
        `lds_k_row_pad`) that the shipped catalog never had. These must NOT
        produce false differences -- only shipped fields are compared."""
        shipped_entry = {
            "version": "1.0",
            "id": "id-k0",
            "name": "k0",
            "kernel_source": {
                "kind": "rocke",
                "source": "kernels/gfx950/attention_dense.py",
                "builder": "build_attention_dense",
                "spec": {"dtype": "bf16", "block_m": 256, "block_n": 64},
            },
            "metadata": {"dtype": "BF16", "block_m": 256, "block_n": 64},
            "priority": 0,
        }
        regen_entry = json.loads(json.dumps(shipped_entry))
        # Regenerated side carries an extra field the shipped spec never had.
        regen_entry["kernel_source"]["spec"]["lds_k_row_pad"] = 8
        regen_entry["id"] = "fresh-id"
        shipped = _kdp([shipped_entry])
        regenerated = _kdp([regen_entry])
        assert run(shipped, regenerated, "--ignore-ids") == 0

    def test_changed_kernel_source_kind_is_caught(self, run, capsys):
        """kernel_source.kind is now part of the key. Changing the source kind
        while leaving the spec identical is a different binary."""
        shipped_entry = {
            "version": "1.0",
            "id": "id-k0",
            "name": "k0",
            "kernel_source": {
                "kind": "rocke",
                "source": "kernels/gfx950/attention_dense.py",
                "builder": "build_attention_dense",
                "spec": {"dtype": "bf16"},
            },
            "metadata": {"dtype": "BF16"},
            "priority": 0,
        }
        regen_entry = json.loads(json.dumps(shipped_entry))
        regen_entry["kernel_source"]["kind"] = "hiprtc"
        regen_entry["id"] = "fresh-id"
        shipped = _kdp([shipped_entry])
        regenerated = _kdp([regen_entry])
        assert run(shipped, regenerated, "--ignore-ids") == 1


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
