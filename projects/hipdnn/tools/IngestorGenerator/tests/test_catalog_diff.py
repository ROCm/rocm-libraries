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
        "provenance": {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": "ued-id",
                        "kmd_id": "kmd-id",
                        "metadata_fields": ["dtype", "block_m", "block_n"],
                    }
                ],
            }
        },
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

    @pytest.mark.parametrize("key", ["engine_id", "kmd_id"])
    def test_the_contract_consumer_ids_are_masked(self, run, key):
        """The contract references the freshly minted UED and KMD, so a straight
        generate.py output differs from the shipped KDP in these two keys."""
        shipped = _kdp([_entry("k0")])
        regenerated = json.loads(json.dumps(shipped))
        consumer = regenerated["provenance"]["specialization_contract"]["consumers"][0]
        consumer[key] = "fresh-minted-id"
        assert run(shipped, regenerated, "--ignore-ids") == 0
        assert run(shipped, regenerated) == 1

    def test_a_dropped_kernel_is_caught(self, run, capsys):
        """The failure the gate is FOR: a manifest or registry change that silently
        ships less than it did. Counting alone would catch this one."""
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

    @pytest.mark.parametrize(
        "edit",
        [
            lambda e: e["kernel_source"].update(builder="build_attention_other"),
            lambda e: e["kernel_source"].update(source="kernels/gfx950/other.py"),
            lambda e: e.update(version="2.0"),
            lambda e: e["metadata"].update(dtype="FP16"),
            lambda e: e["kernel_source"].update(target="gfx942"),
            lambda e: e["kernel_source"].update(entry_point="other"),
            lambda e: e.update(arch=["gfx942"]),
        ],
        ids=["builder", "source", "version", "metadata", "target", "entry", "arch"],
    )
    @pytest.mark.parametrize("flags", [(), ("--subset",), ("--strict-spec",)])
    def test_each_key_component_alone_is_caught(self, run, edit, flags):
        """One flip at a time, so removing any one component from `entry_key`
        leaves its own case green-to-red. Metadata moves without the spec: the
        spec and metadata agreeing is the contract's job, not the key's."""
        shipped = _kdp([_entry("k0")])
        regenerated = json.loads(json.dumps(shipped))
        edit(regenerated["kernelDescriptors"][0])
        assert run(shipped, regenerated, "--ignore-ids", *flags) == 1

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
        consumer = regenerated["provenance"]["specialization_contract"]["consumers"][0]
        consumer["metadata_fields"] = ["dtype"]
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

    def test_distinct_prebuilt_objects_with_no_spec_are_not_duplicates(self, run):
        """A prebuilt object carries no spec: what names its binary is the file."""

        def hsaco(name, file):
            entry = _entry(name)
            entry["kernel_source"] = {"kind": "hsaco", "file": file, "symbol": "k"}
            return entry

        kdp = _kdp([hsaco("k0", "a.hsaco"), hsaco("k1", "b.hsaco")])
        assert run(kdp, kdp, "--ignore-ids") == 0

    def test_one_binary_on_two_arches_is_not_a_duplicate(self, run):
        twin = _entry("k1")
        twin["arch"] = ["gfx942"]
        kdp = _kdp([_entry("k0"), twin])
        assert run(kdp, kdp, "--ignore-ids") == 0


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


class TestExpectIdsPerKernel:
    """`--expect-ids` used to check only the KDP's own id and references, so a
    splice that renumbered every kernel descriptor still passed."""

    def test_a_renumbered_kernel_is_not_preserved(self, run, capsys):
        shipped = _kdp([_entry("k0", block_n=32), _entry("k1", block_n=64)])
        regenerated = json.loads(json.dumps(shipped))
        regenerated["kernelDescriptors"][1]["id"] = "fresh-k1"
        assert run(shipped, regenerated, "--ignore-ids", "--expect-ids") == 1
        out = capsys.readouterr().out
        assert "'k1'" in out and "was not preserved" in out, out
        assert "'k0'" not in out

    def test_preserved_kernel_ids_pass_in_any_order(self, run):
        shipped = _kdp([_entry("k0", block_n=32), _entry("k1", block_n=64)])
        regenerated = json.loads(json.dumps(shipped))
        regenerated["kernelDescriptors"].reverse()
        assert run(shipped, regenerated, "--ignore-ids", "--expect-ids") == 0

    @pytest.mark.parametrize("key", ["engine_id", "kmd_id"])
    def test_contract_engine_and_kmd_ids_must_be_preserved(self, run, capsys, key):
        shipped = _kdp([_entry("k0")])
        regenerated = json.loads(json.dumps(shipped))
        consumer = regenerated["provenance"]["specialization_contract"]["consumers"][0]
        consumer[key] = "fresh-minted-id"
        assert run(shipped, regenerated, "--ignore-ids", "--expect-ids") == 1
        assert "engine_id/kmd_id" in capsys.readouterr().out

    def test_repeated_names_are_keyed_by_spec(self, run):
        a, b = _entry("dup", block_n=32), _entry("dup", block_n=64)
        a["id"], b["id"] = "id-a", "id-b"
        shipped = _kdp([a, b])
        regenerated = json.loads(json.dumps(shipped))
        regenerated["kernelDescriptors"].reverse()
        assert run(shipped, regenerated, "--ignore-ids", "--expect-ids") == 0
        regenerated["kernelDescriptors"][0]["id"] = "other"
        assert run(shipped, regenerated, "--ignore-ids", "--expect-ids") == 1

    def test_per_arch_twins_sharing_a_name_pass_when_identical(self, run):
        """Two entries with one name and one spec differ only by arch; each is keyed
        by its binary identity, so the pair is not ambiguous."""
        a, b = _entry("twin"), _entry("twin")
        a["arch"], b["arch"] = ["gfx950"], ["gfx942"]
        a["id"], b["id"] = "id-a", "id-b"
        shipped = _kdp([a, b])
        assert run(shipped, json.loads(json.dumps(shipped)), "--expect-ids") == 0

    def test_a_name_repeated_on_one_side_only_cannot_be_checked(self, run, capsys):
        """The shipped `dup` is keyed by name, the regenerated pair by name and spec,
        so its id would go unchecked; `--subset` accepts the extra entry."""
        shipped = _kdp([_entry("dup", block_n=32)])
        regenerated = _kdp([_entry("dup", block_n=32), _entry("dup", block_n=64)])
        flags = ("--ignore-ids", "--subset", "--expect-ids")
        assert run(shipped, regenerated, *flags) == 2
        assert "dup" in capsys.readouterr().err


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

    @staticmethod
    def _pair():
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
        regen_entry["kernel_source"]["spec"]["lds_k_row_pad"] = 8
        regen_entry["id"] = "fresh-id"
        return _kdp([shipped_entry]), _kdp([regen_entry])

    def test_regenerated_only_spec_field_is_not_compared_by_default(self, run, capsys):
        """The registry spec dataclass can be wider than the shipped spec; the
        extra fields are default-valued (hkp_pack hydrates Spec(**fields)), so by
        default the gate is keyed on the shipped fields."""
        shipped, regenerated = self._pair()
        assert run(shipped, regenerated, "--ignore-ids") == 0
        out = capsys.readouterr().out
        assert "1 regenerated-only spec fields NOT compared: lds_k_row_pad" in out, out

    def test_subset_names_the_fields_it_did_not_compare(self, run, capsys):
        shipped, regenerated = self._pair()
        assert run(shipped, regenerated, "--ignore-ids", "--subset") == 0
        out = capsys.readouterr().out
        assert "NOT compared: lds_k_row_pad" in out, out

    def test_no_regenerated_only_field_is_said_so(self, run, capsys):
        kdp = _kdp([_entry("k0")])
        assert run(kdp, kdp, "--ignore-ids") == 0
        assert "adds no spec field" in capsys.readouterr().out

    def test_strict_spec_makes_a_regenerated_only_field_a_difference(self, run, capsys):
        shipped, regenerated = self._pair()
        assert run(shipped, regenerated, "--ignore-ids", "--strict-spec") == 1
        out = capsys.readouterr().out
        assert "only in shipped" in out and "only in regenerated" in out, out
        assert "strict (full spec)" in out

    def test_a_dropped_field_is_caught_by_default(self, run):
        """Keying on shipped fields tolerates ADDED fields only. A field the
        shipped spec has and the regenerated one lost is a difference."""
        shipped, regenerated = self._pair()
        # swap roles: the shipped side now carries the extra field
        assert run(regenerated, shipped, "--ignore-ids") == 1

    def test_changed_value_of_a_shipped_field_is_caught_by_default(self, run):
        shipped, regenerated = self._pair()
        regenerated["kernelDescriptors"][0]["kernel_source"]["spec"]["block_n"] = 128
        assert run(shipped, regenerated, "--ignore-ids") == 1

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


def _write(path: Path, document) -> str:
    path.write_text(json.dumps(document))
    return str(path)


@pytest.mark.parametrize(
    "entry",
    [
        "k0",
        {"name": "k0", "kernel_source": None},
        {"name": "k0", "kernel_source": {"spec": ["block_n", 64]}},
        {"name": 5, "kernel_source": {"spec": {}}},
        {"name": None, "kernel_source": {"spec": {}}},
        {"name": "k0", "priority": [1]},
        {"name": "k0", "version": {"major": 1}},
    ],
    ids=[
        "string-entry",
        "null-kernel-source",
        "list-spec",
        "numeric-name",
        "null-name",
        "list-priority",
        "object-version",
    ],
)
def test_a_malformed_catalog_entry_exits_2(tmp_path, capsys, entry):
    """Exit 1 means 'the catalogs differ'; an entry the comparison cannot read
    compared nothing."""
    good = _write(tmp_path / "a.kdp.json", _kdp([_entry("k0")]))
    bad = _write(tmp_path / "b.kdp.json", _kdp([entry]))
    assert catalog_diff.main(["--shipped", good, "--regenerated", bad]) == 2
    assert "kernelDescriptors[0]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "provenance",
    [
        "x",
        {"specialization_contract": "x"},
        {"specialization_contract": {"consumers": "x"}},
        {"specialization_contract": {"consumers": ["x"]}},
    ],
    ids=["string", "string-contract", "string-consumers", "string-consumer"],
)
@pytest.mark.parametrize("flags", [(), ("--expect-ids",)])
def test_a_malformed_provenance_block_exits_2(tmp_path, capsys, provenance, flags):
    good = _write(tmp_path / "a.kdp.json", _kdp([_entry("k0")]))
    bad = _write(tmp_path / "b.kdp.json", _kdp([_entry("k0")], provenance=provenance))
    assert catalog_diff.main(["--shipped", good, "--regenerated", bad, *flags]) == 2
    assert "provenance" in capsys.readouterr().err


@pytest.mark.parametrize(
    "fields",
    [None, ["dtype"], {"name": "dtype"}],
    ids=["null", "string-item", "object"],
)
def test_a_malformed_kmd_exits_2(tmp_path, capsys, fields):
    kdp = _write(tmp_path / "a.kdp.json", _kdp([_entry("k0")]))
    good = _write(tmp_path / "a.kmd.json", _kmd([_field("dtype")]))
    bad = _write(tmp_path / "b.kmd.json", {"id": "kmd-id", "fields": fields})
    argv = ["--shipped", kdp, "--regenerated", kdp]
    assert (
        catalog_diff.main(argv + ["--shipped-kmd", good, "--regenerated-kmd", bad]) == 2
    )
    assert "KMD 'fields' must be an array of objects" in capsys.readouterr().err


@pytest.mark.parametrize("name", [5, ["dtype"], None], ids=["numeric", "list", "null"])
def test_a_kmd_field_with_a_non_string_name_exits_2(tmp_path, capsys, name):
    kdp = _write(tmp_path / "a.kdp.json", _kdp([_entry("k0")]))
    good = _write(tmp_path / "a.kmd.json", _kmd([_field("dtype")]))
    bad = _write(tmp_path / "b.kmd.json", _kmd([_field("dtype"), {"name": name}]))
    argv = ["--shipped", kdp, "--regenerated", kdp]
    assert (
        catalog_diff.main(argv + ["--shipped-kmd", good, "--regenerated-kmd", bad]) == 2
    )
    err = capsys.readouterr().err
    assert "fields[1]" in err
    assert "Traceback" not in err


def test_a_json_array_is_not_a_kdp(tmp_path, capsys):
    good = _write(tmp_path / "a.kdp.json", _kdp([_entry("k0")]))
    bad = _write(tmp_path / "b.json", [])
    assert catalog_diff.main(["--shipped", good, "--regenerated", bad]) == 2
    assert "no 'kernelDescriptors'" in capsys.readouterr().err


def _subset_case(change: str) -> tuple[dict, dict]:
    shipped = _kdp([_entry("k0"), _entry("k1", 32)])
    if change == "dropped":
        return shipped, _kdp([_entry("k0")])
    if change == "spec":
        return shipped, _kdp([_entry("k0"), _entry("k1", 128)])
    return shipped, _kdp([_entry("k0"), _entry("k1", 32), _entry("k_extra", 16)])


@pytest.mark.parametrize("change,rc", [("dropped", 1), ("spec", 1), ("extra", 0)])
def test_subset(run, capsys, change, rc):
    shipped, regenerated = _subset_case(change)
    assert run(shipped, regenerated, "--ignore-ids", "--subset") == rc
    out = capsys.readouterr().out
    if rc:
        assert "only in shipped" in out and "k1" in out, out
    else:
        assert "registry extras (not shipped): 1" in out and "k_extra" in out, out


def test_subset_excludes_strict_spec(run):
    kdp = _kdp([_entry("k0")])
    assert run(kdp, kdp, "--subset", "--strict-spec") == 2


def _multi_entry(name: str = "k0") -> dict:
    """A multi-launch entry: `sources` in place of `kernel_source`, at UKD 1.1."""
    entry = _entry(name)
    source = entry.pop("kernel_source")
    entry["version"] = "1.1"
    entry["sources"] = {
        "segment": source,
        "reduce": {
            "kind": "rocke",
            "source": "kernels/gfx950/attention_dense.py",
            "builder": "build_reduce",
            "spec": {"num_splits": 4},
            "role": "reduce",
        },
    }
    return entry


class TestMultiLaunchEntries:
    """An entry naming its kernels in `sources` is compared launch by launch: an
    entry key that read only `kernel_source` would call two different
    multi-launch catalogs the same."""

    def test_equal_multi_launch_catalogs_are_the_same(self, run):
        assert run(_kdp([_multi_entry()]), _kdp([_multi_entry()]), "--ignore-ids") == 0

    @pytest.mark.parametrize(
        "edit",
        [
            lambda e: e["sources"]["reduce"].update(role="merge"),
            lambda e: e["sources"]["reduce"].update(builder="build_other"),
            lambda e: e["sources"]["reduce"]["spec"].update(num_splits=8),
            lambda e: e["sources"]["segment"]["spec"].update(block_n=32),
            lambda e: e["sources"].pop("reduce"),
            lambda e: e["sources"].update(merge=e["sources"].pop("reduce")),
            lambda e: e.update(
                kernel_source=e.pop("sources")["segment"], version="1.0"
            ),
        ],
        ids=[
            "role",
            "builder",
            "second_launch_spec",
            "first_launch_spec",
            "dropped_launch",
            "renamed_launch",
            "collapsed_to_kernel_source",
        ],
    )
    @pytest.mark.parametrize("flags", [(), ("--subset",), ("--strict-spec",)])
    def test_a_change_to_any_launch_is_caught(self, run, edit, flags):
        shipped = _kdp([_multi_entry()])
        regenerated = json.loads(json.dumps(shipped))
        edit(regenerated["kernelDescriptors"][0])
        assert run(shipped, regenerated, "--ignore-ids", *flags) == 1

    def test_a_spec_field_only_a_launch_carries_is_keyed_by_default(self, run):
        """The shipped catalog's spec fields decide what is compared; a field only
        its second launch carries must be among them."""
        shipped = _kdp([_multi_entry()])
        regenerated = json.loads(json.dumps(shipped))
        regenerated["kernelDescriptors"][0]["sources"]["reduce"]["spec"] = {
            "num_splits": 8
        }
        assert run(shipped, regenerated, "--ignore-ids") == 1

    def test_two_entries_with_one_set_of_launches_are_duplicates(self, capsys):
        kdp = _kdp([_multi_entry("a"), _multi_entry("b")])
        assert len(catalog_diff.find_duplicate_specs(kdp)) == 1

    def test_an_entry_with_both_spellings_is_refused(self, run):
        entry = _multi_entry()
        entry["kernel_source"] = entry["sources"]["segment"]
        assert run(_kdp([entry]), _kdp([entry]), "--ignore-ids") == 2

    @pytest.mark.parametrize(
        "sources",
        [[], "x", None, {"a": "x"}, {"a": {"spec": []}}],
        ids=["list", "string", "null", "launch_not_object", "spec_not_object"],
    )
    def test_malformed_sources_exit_2(self, run, sources):
        entry = _entry("k0")
        entry.pop("kernel_source")
        entry["sources"] = sources
        assert run(_kdp([entry]), _kdp([entry]), "--ignore-ids") == 2
