# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/generator.py: rendered content (not golden-file diffing),
UUID cross-reference threading, allow-listed JSON keys, the UMD policy and the shape
of the gtest cases the emitted C++ stubs carry."""

import copy
import dataclasses
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from codegen.generator import (
    PLACEHOLDER_MARKER,
    _dedup_key,
    build_kdp,
    build_kdp_documents,
    build_kmd,
    build_ued,
    emitted_inventory,
    mint_ids,
    resolve_ids,
)
from codegen.models import (
    DEFAULT_FIXTURE_ARCH,
    DIALECT_PACKAGED,
    KernelSource,
    KmdField,
)
from codegen.prior_ids import discover_siblings, find_engine, read_prior_bundle
from tests.helpers import (
    all_ids,
    home_and_satellite,
    json_bytes,
    kdps,
    make_engine,
    make_kernel,
    make_minimal_config,
    make_pack,
    make_packaged_config,
    make_rocke_kernel,
    read_docs,
    render_resolved,
)


def _distinct_from(value):
    """A value of the SAME declared type that is not ``value``: an untyped stand-in
    would be refused for the type rather than the value."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 448
    return f"{value}_other"


def emitted_cases(rendered: str, macro: str, suite: str) -> dict:
    """The gtest cases one suite of an emitted C++ stub carries, name -> body.

    Keyed by each case's OWN name: a substring check for text a case contains still
    passes once the case is gone. A body ends at the first closing brace in column 0.
    """
    pattern = rf"{macro}\({re.escape(suite)}, (\w+)\)\n\{{\n(.*?)\n\}}"
    return {m.group(1): m.group(2) for m in re.finditer(pattern, rendered, re.DOTALL)}


class TestUuidThreading:
    """Every cross-reference must be the SAME id minted for the referenced
    descriptor."""

    def test_ued_metadata_references_kmd_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        ued = build_ued(scale_add_config, ids)
        assert ued["metadata"] == ids["kmd"]

    def test_ued_sort_kernel_catalog_references_uhd_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        ued = build_ued(scale_add_config, ids)
        assert ued["sort_kernel_catalog"] == {"default": ids["uhd"]}
        assert "heuristic" not in ued

    def test_kdp_engine_references_ued_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        pack = scale_add_config.packs[0]
        kdp = build_kdp(scale_add_config, pack, ids)
        assert kdp["engine"] == ids["ued"]

    def test_kdp_dispatch_references_udd_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        pack = scale_add_config.packs[0]
        kdp = build_kdp(scale_add_config, pack, ids)
        assert kdp["dispatch"] == ids["udd"]

    def test_kdp_matchers_reference_umd_ids(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        pack = scale_add_config.packs[0]
        kdp = build_kdp(scale_add_config, pack, ids)
        assert ids["kernel_match"] in kdp["matchers"]

    def test_multi_pack_kdp_references_its_own_operation_umd(self, binary_ops_config):
        ids = mint_ids(binary_ops_config)
        add_pack = binary_ops_config.packs[0]
        max_pack = binary_ops_config.packs[1]
        add_kdp = build_kdp(binary_ops_config, add_pack, ids)
        max_kdp = build_kdp(binary_ops_config, max_pack, ids)
        assert ids[("operation_umd", 0)] in add_kdp["matchers"]
        assert ids[("operation_umd", 1)] in max_kdp["matchers"]
        # Each pack's own operation matcher must NOT appear on the other pack.
        assert ids[("operation_umd", 0)] not in max_kdp["matchers"]
        assert ids[("operation_umd", 1)] not in add_kdp["matchers"]

    def test_kernel_ids_are_unique_and_distinct_from_pack_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        pack = scale_add_config.packs[0]
        kdp = build_kdp(scale_add_config, pack, ids)
        kernel_ids = [k["id"] for k in kdp["kernelDescriptors"]]
        assert len(kernel_ids) == len(set(kernel_ids))
        assert kdp["id"] not in kernel_ids

    def test_every_minted_id_is_distinct(self, binary_ops_config):
        """Pack names, and kernel names off the `load_config` path, are not unique, so
        no id may be derived from one."""
        ids = mint_ids(binary_ops_config)
        values = list(ids.values())
        assert len(values) == len(set(values))


class TestDuplicateKernelNamesAreSurvivable:
    """Kernel names are unique on the ``load_config`` path but nothing else enforces
    it, so a programmatic config can repeat one. Neither ids nor their reuse may key
    on a name: two variants sharing an id lose one with no error."""

    def test_same_name_different_metadata_gets_distinct_ids(self, scale_add_config):
        import copy

        config = copy.deepcopy(scale_add_config)
        pack = config.packs[0]
        original = pack.kernels[0]
        twin = copy.deepcopy(original)
        twin.name = original.name  # the same name, deliberately
        key = next(iter(twin.metadata))
        value = twin.metadata[key]
        twin.metadata[key] = (value + 1) if isinstance(value, int) else "other"
        pack.kernels.append(twin)

        kdp = build_kdp(config, pack, mint_ids(config))
        emitted = kdp["kernelDescriptors"]
        assert len(emitted) == len(pack.kernels), "a distinct variant was dropped"
        ids = [k["id"] for k in emitted]
        assert len(ids) == len(set(ids)), "same-named variants collided on id"


class TestVariantDeduplication:
    """Overlapping generation expressions are expected; the emitted KDP is unique. Two
    entries with identical matcher-visible metadata are ONE candidate to the runtime."""

    def test_duplicate_metadata_is_emitted_once(self, scale_add_config):
        import copy

        config = copy.deepcopy(scale_add_config)
        pack = config.packs[0]
        original = pack.kernels[0]
        clone = copy.deepcopy(original)
        # Same metadata, different name: what two overlapping expressions produce.
        clone.name = original.name + "_from_second_expression"
        pack.kernels.append(clone)

        ids = mint_ids(config)
        kdp = build_kdp(config, pack, ids)

        emitted = [k["name"] for k in kdp["kernelDescriptors"]]
        assert clone.name not in emitted, "duplicate metadata must not be emitted twice"
        assert original.name in emitted, "the first entry wins"

        seen = [
            json.dumps(k["metadata"], sort_keys=True) for k in kdp["kernelDescriptors"]
        ]
        assert len(seen) == len(set(seen)), "every emitted entry has unique metadata"

    def test_distinct_metadata_is_kept(self, scale_add_config):
        """De-duplication keys on metadata, so a real variant is never dropped."""
        import copy

        config = copy.deepcopy(scale_add_config)
        pack = config.packs[0]
        original = pack.kernels[0]
        variant = copy.deepcopy(original)
        variant.name = original.name + "_real_variant"
        key = next(iter(variant.metadata))
        value = variant.metadata[key]
        variant.metadata[key] = (value + 1) if isinstance(value, int) else "other"
        pack.kernels.append(variant)

        kdp = build_kdp(config, pack, mint_ids(config))
        assert variant.name in [k["name"] for k in kdp["kernelDescriptors"]]

    def test_unset_tristate_never_collides_with_the_schema_default(
        self, scale_add_config
    ):
        """The loader substitutes a field's KMD ``default_value`` for anything absent
        and then requires the tuple to be unique per device, so "omit the key" and
        "write the default" are one catalog entry and a collision drops the whole
        engine."""
        import copy

        config = copy.deepcopy(scale_add_config)
        pack = config.packs[0]
        optional = next(
            (
                f
                for f in config.kmd_fields
                if not f.is_mandatory and f.default_value is not None
            ),
            None,
        )
        if optional is None:
            pytest.skip("fixture engine declares no optional KMD field")

        # One kernel pins the field to the schema default; its twin states it nowhere
        # and lets the spec decide. That is the only spelling of "unset" a bundle can
        # ship: a literal null is a type the loader reads nothing onto, and a knob
        # stated in NEITHER layer is refused
        # (`test_a_knob_stated_in_neither_layer_is_refused`).
        pinned = copy.deepcopy(pack.kernels[0])
        pinned.name = pack.kernels[0].name + "_pinned_to_default"
        pinned.metadata[optional.name] = optional.default_value
        unset = copy.deepcopy(pack.kernels[0])
        unset.name = pack.kernels[0].name + "_left_to_the_spec"
        unset.metadata.pop(optional.name, None)
        unset.kernel_source.spec = dict(unset.kernel_source.spec or {})
        unset.kernel_source.spec[optional.name] = _distinct_from(optional.default_value)
        pack.kernels.extend([pinned, unset])

        kdp = build_kdp(config, pack, mint_ids(config))
        defaults = {f.name: f.default_value for f in config.kmd_fields}
        names = [f.name for f in config.kmd_fields]
        tuples = [
            tuple(k["metadata"].get(n, defaults.get(n)) for n in names)
            for k in kdp["kernelDescriptors"]
        ]
        assert len(tuples) == len(set(tuples)), (
            "two descriptors resolve to one catalog tuple once KMD defaults are "
            "applied; the loader rejects the engine outright"
        )

    def test_a_knob_stated_in_neither_layer_is_refused(self, scale_add_config):
        """Never mentioning an optional knob produces a descriptor that looks clean and
        is not: at load the KMD's ``default_value`` becomes the catalog key while the
        binary was compiled from the BUILDER's own default."""
        import copy

        config = copy.deepcopy(scale_add_config)
        pack = config.packs[0]
        optional = next((f for f in config.kmd_fields if not f.is_mandatory), None)
        if optional is None:
            pytest.skip("fixture engine declares no optional KMD field")

        silent = copy.deepcopy(pack.kernels[0])
        silent.name = pack.kernels[0].name + "_states_it_nowhere"
        silent.metadata.pop(optional.name, None)
        if silent.kernel_source.spec:
            silent.kernel_source.spec.pop(optional.name, None)
        pack.kernels.append(silent)

        with pytest.raises(ValueError, match="neither its metadata nor"):
            build_kdp(config, pack, mint_ids(config))

    def test_a_knob_the_spec_pins_needs_no_metadata_entry(
        self, gfx950_attention_dense_config
    ):
        """A knob absent from metadata but PINNED in ``kernel_source.spec`` is fully
        decided, so it must emit with the spec's value derived into metadata. Runs on
        the PACKAGED fixture, the only dialect carrying a ``kernel_source.spec``."""
        import copy

        config = copy.deepcopy(gfx950_attention_dense_config)
        pack = config.packs[0]
        optional = next((f for f in config.kmd_fields if not f.is_mandatory), None)
        assert optional is not None, "packaged fixture must declare an optional field"
        assert pack.kernels[0].kernel_source.spec, "packaged fixture must carry a spec"

        pinned = copy.deepcopy(pack.kernels[0])
        pinned.name = pack.kernels[0].name + "_pinned_in_spec_only"
        pinned.metadata.pop(optional.name, None)
        pinned.kernel_source.spec[optional.name] = 1
        pack.kernels.append(pinned)

        kdp = build_kdp(config, pack, mint_ids(config))
        emitted = next(k for k in kdp["kernelDescriptors"] if k["name"] == pinned.name)
        assert emitted["metadata"][optional.name] == 1, (
            "a spec-pinned knob must reach metadata; the matcher compares metadata "
            "and would otherwise never see the value the binary was built with"
        )

    def test_duplicate_metadata_is_dropped_ACROSS_packs_not_just_within_one(
        self, generator, binary_ops_config, tmp_path
    ):
        """The loader groups packs by ENGINE ID, so identical metadata across two packs
        is a duplicate CATALOG TUPLE, which drops the WHOLE ENGINE."""
        import copy

        config = copy.deepcopy(binary_ops_config)
        assert len(config.packs) >= 2, "fixture must be multi-pack"
        source = config.packs[0].kernels[0]
        clone = copy.deepcopy(source)
        clone.name = source.name + ".same_metadata_other_pack"
        config.packs[1].kernels.append(clone)

        written = generator.render(config, tmp_path)
        emitted = []
        for path in written:
            if path.endswith(".kdp.json"):
                doc = json.loads((tmp_path / path).read_text())
                emitted += [
                    json.dumps(k["metadata"], sort_keys=True)
                    for k in doc["kernelDescriptors"]
                ]
        assert len(emitted) == len(set(emitted)), (
            "two packs of one engine emitted the same matcher-visible metadata; the "
            "loader would see a duplicate catalog tuple and drop the engine"
        )
        assert clone.name not in json.dumps(
            [
                json.loads((tmp_path / p).read_text())
                for p in written
                if p.endswith(".kdp.json")
            ]
        ), "the cross-pack duplicate should have been dropped, not renamed"

    def test_a_distinct_variant_in_a_second_pack_is_kept(
        self, generator, binary_ops_config, tmp_path
    ):
        """The converse: without it the test above passes on a generator that drops
        every kernel after the first."""
        import copy

        config = copy.deepcopy(binary_ops_config)
        source = config.packs[0].kernels[0]
        variant = copy.deepcopy(source)
        variant.name = source.name + ".genuinely_different"
        key = next(k for k, v in variant.metadata.items() if isinstance(v, int))
        variant.metadata[key] = variant.metadata[key] + 1000
        config.packs[1].kernels.append(variant)

        written = generator.render(config, tmp_path)
        names = []
        for path in written:
            if path.endswith(".kdp.json"):
                doc = json.loads((tmp_path / path).read_text())
                names += [k["name"] for k in doc["kernelDescriptors"]]
        assert variant.name in names, (
            "a variant differing in matcher-visible metadata is real coverage and "
            "must survive de-duplication"
        )


class TestUmdPolicy:
    """Emit a UMD only for genuine per-pack narrowing; a single-pack engine gets zero
    graph-scoped UMDs (mirrors TestConvFwdPack.cpp)."""

    def test_single_pack_engine_emits_no_operation_umd_file(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        umd_files = [f for f in written if f.endswith(".umd.json")]
        # Only the shared kernel-scoped matcher -- no operation-scoped UMD.
        assert len(umd_files) == 1
        assert "kernel_dtype_matches_graph.umd.json" in umd_files[0]

    def test_multi_pack_engine_emits_one_operation_umd_per_pack(
        self, generator, binary_ops_config, tmp_path
    ):
        written = generator.render(binary_ops_config, tmp_path)
        umd_files = [f for f in written if f.endswith(".umd.json")]
        # One shared kernel-scoped matcher + one operation matcher per pack.
        assert len(umd_files) == 1 + len(binary_ops_config.packs)

    def test_single_pack_kdp_carries_no_operation_umd_reference(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        kdp_path = [tmp_path / f for f in written if f.endswith(".kdp.json")][0]
        kdp = json.loads(kdp_path.read_text())
        # Only one matcher on a single-pack engine: the shared kernel-scoped one.
        assert len(kdp["matchers"]) == 1


class TestAllowListedKeys:
    """Emitted descriptor JSON uses only allow-listed keys per type."""

    _KMD_KEYS = {"version", "id", "name", "fields"}
    _UED_KEYS = {
        "version",
        "id",
        "name",
        "graph_match",
        "sort_kernel_catalog",
        "metadata",
        "knobs",
        "behavior_notes",
        "numerical_notes",
        "sdk_version",
    }
    _UMD_KEYS = {"version", "id", "name", "scope", "match_symbol"}
    _UDD_KEYS = {"version", "id", "name", "dispatch_symbol"}
    _UHD_KEYS = {"version", "id", "name", "adapter", "objective", "native"}
    _KDP_KEYS = {
        "version",
        "id",
        "name",
        "arch",
        "matchers",
        "engine",
        "dispatch",
        "kernelDescriptors",
        # The packager's extension key the loader ignores without a warning
        # (DescriptorLoader.hpp isExtensionKey), carrying the pack's one
        # specialization_contract.
        "provenance",
    }
    _UKD_KEYS = {
        "version",
        "id",
        "name",
        "kernel_source",
        "metadata",
        "priority",
        "arch",
        "provenance",
    }
    _KERNEL_SOURCE_KEYS = {
        "kind",
        "source_file",
        "entry_point",
        "library",
        "toc_key",
        "symbol",
        "sha256",
    }

    def _rendered_json(self, generator, config, tmp_path, suffix):
        written = generator.render(config, tmp_path)
        paths = [tmp_path / f for f in written if f.endswith(suffix)]
        return [json.loads(p.read_text()) for p in paths]

    def test_kmd_keys_allow_listed(self, generator, scale_add_config, tmp_path):
        for obj in self._rendered_json(
            generator, scale_add_config, tmp_path, ".kmd.json"
        ):
            assert set(obj.keys()) <= self._KMD_KEYS

    def test_ued_keys_allow_listed(self, generator, scale_add_config, tmp_path):
        for obj in self._rendered_json(
            generator, scale_add_config, tmp_path, ".ued.json"
        ):
            assert set(obj.keys()) <= self._UED_KEYS

    def test_umd_keys_allow_listed(self, generator, binary_ops_config, tmp_path):
        for obj in self._rendered_json(
            generator, binary_ops_config, tmp_path, ".umd.json"
        ):
            assert set(obj.keys()) <= self._UMD_KEYS

    def test_udd_keys_allow_listed(self, generator, scale_add_config, tmp_path):
        for obj in self._rendered_json(
            generator, scale_add_config, tmp_path, ".udd.json"
        ):
            assert set(obj.keys()) <= self._UDD_KEYS

    def test_uhd_keys_allow_listed(self, generator, scale_add_config, tmp_path):
        for obj in self._rendered_json(
            generator, scale_add_config, tmp_path, ".uhd.json"
        ):
            assert set(obj.keys()) <= self._UHD_KEYS

    def test_kdp_keys_allow_listed(self, generator, scale_add_config, tmp_path):
        for obj in self._rendered_json(
            generator, scale_add_config, tmp_path, ".kdp.json"
        ):
            assert set(obj.keys()) <= self._KDP_KEYS
            for kernel in obj["kernelDescriptors"]:
                assert set(kernel.keys()) <= self._UKD_KEYS
                assert set(kernel["kernel_source"].keys()) <= self._KERNEL_SOURCE_KEYS

    def test_every_string_field_non_empty(self, generator, scale_add_config, tmp_path):
        """The loader rejects any empty string field."""
        written = generator.render(scale_add_config, tmp_path)
        json_files = [tmp_path / f for f in written if f.endswith(".json")]

        def check(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        assert value != "", f"empty string field '{key}'"
                    else:
                        check(value)
            elif isinstance(obj, list):
                for item in obj:
                    check(item)

        for path in json_files:
            check(json.loads(path.read_text()))


class TestRenderWritesEveryFile:
    def test_scale_add_render_matches_preview(
        self, generator, scale_add_config, tmp_path
    ):
        preview = generator.preview_files(scale_add_config)
        written = generator.render(scale_add_config, tmp_path)
        assert sorted(preview) == sorted(written)

    def test_binary_ops_render_matches_preview(
        self, generator, binary_ops_config, tmp_path
    ):
        preview = generator.preview_files(binary_ops_config)
        written = generator.render(binary_ops_config, tmp_path)
        assert sorted(preview) == sorted(written)

    def test_every_written_file_exists_on_disk(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        for rel in written:
            assert (tmp_path / rel).exists(), rel

    def test_every_emitted_cpp_file_has_copyright_header(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        for rel in written:
            if rel.endswith(".cpp"):
                text = (tmp_path / rel).read_text()
                assert text.startswith("// Copyright")
                assert "SPDX-License-Identifier:  MIT" in text


class TestFragmentsNameRealFiles:
    """Every descriptor file the CMake fragment names must exist on disk.

    Installation is by DIRECTORY and the fragment tells the author which files belong at
    that path, so a name with no file behind it loses the descriptor with no build
    error. The single-pack case is discriminating: ``kdp_stem()`` is the BARE slug
    there.
    """

    @staticmethod
    def _fragment_descriptor_names(tmp_path):
        text = (tmp_path / "fragments" / "cmake_descriptor_files.txt").read_text()
        names = [line.strip().lstrip("#").strip() for line in text.splitlines()]
        return [name for name in names if name.endswith(".json")]

    def test_single_pack_fragment_names_all_exist(
        self, generator, scale_add_config, tmp_path
    ):
        generator.render(scale_add_config, tmp_path)
        listed = self._fragment_descriptor_names(tmp_path)
        assert listed, "fragment named no descriptor files at all"
        ddir = tmp_path / scale_add_config.descriptor_dir
        for name in listed:
            assert (
                ddir / name
            ).exists(), f"fragment names {name}, which the generator never wrote"

    def test_multi_pack_fragment_names_all_exist(
        self, generator, binary_ops_config, tmp_path
    ):
        generator.render(binary_ops_config, tmp_path)
        listed = self._fragment_descriptor_names(tmp_path)
        assert listed, "fragment named no descriptor files at all"
        ddir = tmp_path / binary_ops_config.descriptor_dir
        for name in listed:
            assert (
                ddir / name
            ).exists(), f"fragment names {name}, which the generator never wrote"

    def test_fragment_names_every_descriptor_written(
        self, generator, binary_ops_config, tmp_path
    ):
        """A descriptor written but not named is one the author never learns to
        place."""
        written = generator.render(binary_ops_config, tmp_path)
        prefix = binary_ops_config.descriptor_dir + "/"
        on_disk = {
            rel[len(prefix) :]
            for rel in written
            if rel.startswith(prefix) and rel.endswith(".json")
        }
        assert on_disk == set(self._fragment_descriptor_names(tmp_path))


class TestSpecializationContractEmission:
    """The bundle is checked on a machine without the rocKE that compiled it, so these
    assert what each kernel RESOLVES to, not where the declaration is written."""

    @staticmethod
    def _consumers(kdp):
        """What each kernel of this KDP resolves to, through the real reader."""
        agreement = _import_hkp_pack("agreement")
        if agreement is None:
            pytest.skip("hkp_pack is not importable from this checkout")
        return [
            agreement.resolved_contract(kernel, kdp)
            for kernel in kdp["kernelDescriptors"]
        ]

    def test_the_entry_carries_the_minted_engine_and_kmd_ids(self, scale_add_config):
        """Ids are threaded from the one mint: a re-derived one points at whatever
        engine shares a name."""
        ids = mint_ids(scale_add_config)
        kdp = build_kdp(scale_add_config, scale_add_config.packs[0], ids)
        for contract in self._consumers(kdp):
            assert contract["schema_version"] == 1
            assert len(contract["consumers"]) == 1
            consumer = contract["consumers"][0]
            assert consumer["engine_id"] == ids["ued"]
            assert consumer["kmd_id"] == ids["kmd"]

    def test_the_entry_carries_exactly_the_six_contract_keys(self, scale_add_config):
        """The consumer rejects a missing OR an unknown key, so an extra one is as fatal
        as an absent one."""
        kdp = build_kdp(
            scale_add_config, scale_add_config.packs[0], mint_ids(scale_add_config)
        )
        for contract in self._consumers(kdp):
            assert set(contract["consumers"][0]) == {
                "engine_id",
                "kmd_id",
                "metadata_fields",
                "matcher_only_fields",
                "bindings",
                "vocabulary",
            }

    def test_a_direct_load_declaration_emits_its_matcher_only_partition(self):
        config = make_minimal_config(
            specialization={
                "metadata_fields": [],
                "matcher_only_fields": ["block_size", "dtype"],
                "bindings": {},
                "vocabulary": {},
            }
        )
        kdp = build_kdp(config, config.packs[0], mint_ids(config))
        consumer = self._consumers(kdp)[0]["consumers"][0]
        assert consumer["metadata_fields"] == []
        assert sorted(consumer["matcher_only_fields"]) == ["block_size", "dtype"]

    def test_the_declaration_is_written_once_for_the_whole_pack(self, scale_add_config):
        """Every inline kernel of a bundle is one engine's, one KMD's, one field
        partition's, so repeating the declaration per kernel multiplies the file size of
        a large pack."""
        kdp = build_kdp(
            scale_add_config, scale_add_config.packs[0], mint_ids(scale_add_config)
        )
        assert "specialization_contract" in kdp["provenance"]
        assert kdp["kernelDescriptors"]
        assert not any("provenance" in k for k in kdp["kernelDescriptors"])

    @pytest.mark.parametrize(
        "dialect,kind",
        [("packaged", "rocke"), ("direct_load", "embedded_source")],
    )
    def test_a_config_with_no_declaration_refuses_to_emit(self, dialect, kind):
        """Descriptors with no declaration cannot be checked against the builder they
        were compiled from. The direct-load row is parametrized alongside the rocKE one
        because a gate firing only on the dialect an author is already careful about
        never fires."""
        config = make_minimal_config(
            dialect=dialect,
            kernel_source_kind=kind,
            packs=[make_pack(arch=["gfx942"])],
            specialization={},
        )
        with pytest.raises(ValueError, match="specialization"):
            build_kdp(config, config.packs[0], mint_ids(config))

    def test_the_emitted_entry_satisfies_the_packagers_own_validator(
        self, scale_add_config
    ):
        """Asserting the shape here and hoping it matches ``hkp_pack`` is how the two
        drift."""
        agreement = _import_hkp_pack("agreement")
        if agreement is None:
            pytest.skip("hkp_pack is not importable from this checkout")
        ids = mint_ids(scale_add_config)
        kmd = build_kmd(scale_add_config, ids)
        kdp = build_kdp(scale_add_config, scale_add_config.packs[0], ids)
        for kernel in kdp["kernelDescriptors"]:
            consumers = agreement.contracts(kernel, {ids["kmd"]: kmd}, kdp)
            assert len(consumers) == 1
            agreement.validate_consumer(consumers[0], kmd)

    def test_the_packaged_bundle_passes_the_packers_load(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        """hkp_pack loads the rendered tree as it would at pack time: every id and
        cross-reference a packed bundle carries (the UED's `id` and `metadata` among
        them) must be a UUID the packer resolves."""
        descriptors = _import_hkp_pack("descriptors")
        generic_targets = _import_hkp_pack("generic_targets")
        if descriptors is None or generic_targets is None:
            pytest.skip("hkp_pack is not importable from this checkout")
        config = gfx950_attention_dense_config
        generator.render(config, tmp_path)
        # The shipped generic target table, as the packer loads it at pack time.
        flat = descriptors.load_flat_input(
            tmp_path / config.descriptor_dir,
            generic_targets.GenericTargets.load(generic_targets.DEFAULT_TABLE_PATH),
            log=lambda _message: None,
        )
        # A wrong root would load nothing and pass vacuously.
        assert flat.by_type("ued") and flat.by_type("kmd")


def _import_hkp_pack(module: str):
    """``hkp_pack.<module>``, or ``None`` where the provider tree is absent: descriptor
    generation must not require the kernel toolchain, so this is a test-only bridge."""
    import importlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    package = root / "dnn-providers/hip-kernel-provider/descriptor-packaging/python"
    if not (package / "hkp_pack" / f"{module}.py").exists():
        return None
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    try:
        return importlib.import_module(f"hkp_pack.{module}")
    except ImportError:
        return None


class TestCatalogIdentity:
    """Identity is the COMPLETED tuple plus the devices it covers: the loader
    substitutes each absent field's KMD ``default_value`` and compares per device, so
    the emitted JSON is not the catalog key and a duplicate tuple drops the whole
    engine."""

    @staticmethod
    def _twin_config(**config_overrides):
        """Two kernels, same source, whose metadata the caller sets."""
        left = make_kernel(name="twin.left")
        right = make_kernel(name="twin.right")
        pack = make_pack(kernels=[left, right], arch=["gfx942"])
        config = make_minimal_config(packs=[pack], **config_overrides)
        return config, pack, left, right

    def test_equal_tuples_on_disjoint_arches_both_survive(self):
        """Two devices, two binaries: dropping either leaves the engine matching nothing
        on that device."""
        config, pack, left, right = self._twin_config()
        pack.arch = ["gfx942", "gfx950"]
        left.arch = ["gfx942"]
        right.arch = ["gfx950"]

        kdp = build_kdp(config, pack, mint_ids(config))
        assert [k["name"] for k in kdp["kernelDescriptors"]] == [
            "twin.left",
            "twin.right",
        ]

    @pytest.mark.parametrize(
        ("pack_arch", "left_arch", "right_arch", "other_binary"),
        [
            (["gfx942"], [], [], True),
            (["gfx942", "gfx950"], ["gfx942"], ["gfx942", "gfx950"], False),
            (["gfx11-generic"], [], [], True),
        ],
    )
    def test_equal_tuples_tied_on_a_device_are_refused_naming_both(
        self, pack_arch, left_arch, right_arch, other_binary
    ):
        config, pack, left, right = self._twin_config()
        pack.arch = pack_arch
        left.arch = left_arch
        right.arch = right_arch
        if other_binary:
            right.kernel_source.entry_point = "ScaleAddOther"

        with pytest.raises(ValueError) as excinfo:
            build_kdp(config, pack, mint_ids(config))
        message = str(excinfo.value)
        assert "twin.left" in message and "twin.right" in message

    def test_a_wildcard_arch_and_a_concrete_one_with_one_tuple_both_survive(self):
        """A concrete arch outranks an empty one on its own device, so they never tie."""
        config, pack, left, right = self._twin_config()
        pack.arch = []
        left.arch = []  # every device
        right.arch = ["gfx942"]
        right.kernel_source.entry_point = "ScaleAddOther"

        kdp = build_kdp(config, pack, mint_ids(config))
        assert [k["name"] for k in kdp["kernelDescriptors"]] == [
            "twin.left",
            "twin.right",
        ]

    def test_an_omitted_field_and_the_kmd_default_are_one_key(self):
        """One tuple omits an optional field; the other states it at exactly the KMD
        default. Keyed on the identity function rather than an emitted pair, because
        `_check_metadata_resolved` refuses to EMIT a descriptor omitting a defaulted
        field."""
        config, _pack, _left, _right = self._twin_config()
        omitted = _dedup_key({"dtype": "FLOAT"}, config)
        stated = _dedup_key({"dtype": "FLOAT", "block_size": 64}, config)
        assert omitted == stated
        # The control: a value that is NOT the default stays a different key.
        assert _dedup_key({"dtype": "FLOAT", "block_size": 128}, config) != stated

    def test_an_int_and_a_float_spelling_of_one_float_value_are_one_entry(self):
        """``1`` and ``1.0`` in a FLOAT field compare equal in Python and serialize to
        different bytes, so an identity keyed on the document sees two entries."""
        config, pack, left, right = self._twin_config(
            kmd_fields=[
                KmdField(name="scale", type="float", default_value=1.0),
                KmdField(name="dtype", type="string"),
            ]
        )
        for kernel, spelling in ((left, 1), (right, 1.0)):
            kernel.metadata = {"scale": spelling, "dtype": "FLOAT"}

        kdp = build_kdp(config, pack, mint_ids(config))
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["twin.left"]
        assert kdp["kernelDescriptors"][0]["metadata"]["scale"] == 1.0

    def test_a_bool_typed_field_keeps_its_boolean_spelling(self):
        """A BOOL field's matcher holds ``true``, not ``1``: an integer there declines
        every graph while the engine still loads."""
        config, pack, left, right = self._twin_config(
            kmd_fields=[
                KmdField(name="causal", type="bool", default_value=False),
                KmdField(name="dtype", type="string"),
            ]
        )
        pack.kernels = [left]
        left.metadata = {"dtype": "FLOAT"}
        left.kernel_source.spec = {"causal": True}

        kdp = build_kdp(config, pack, mint_ids(config))
        emitted = kdp["kernelDescriptors"][0]["metadata"]["causal"]
        assert emitted is True, f"a bool field shipped {emitted!r}"

    def test_a_builder_boolean_targeting_an_int_field_ships_as_an_integer(self):
        """The converse: a builder spec spells a flag ``true`` where an ``int``-typed
        field carries ``1``. Without both halves, keeping booleans is indistinguishable
        from never converting."""
        config, pack, left, right = self._twin_config(
            kmd_fields=[
                KmdField(name="causal", type="int", default_value=0),
                KmdField(name="dtype", type="string"),
            ]
        )
        pack.kernels = [left]
        left.metadata = {"dtype": "FLOAT"}
        left.kernel_source.spec = {"causal": True}

        kdp = build_kdp(config, pack, mint_ids(config))
        emitted = kdp["kernelDescriptors"][0]["metadata"]["causal"]
        assert emitted == 1 and emitted is not True, f"an int field shipped {emitted!r}"


class TestAPackEmptiedByDeduplicationIsRefused:
    """``DescriptorLoader`` drops a pack that "declares no kernels" at load, while the
    census counts every KDP the generator wrote, so a pack whose kernels were all
    absorbed asserts one more pack than the runtime holds."""

    @staticmethod
    def _two_packs(left_kernels, right_kernels):
        return make_minimal_config(
            packs=[
                make_pack(
                    name="alpha",
                    discriminator="alpha",
                    arch=["gfx942"],
                    kernels=left_kernels,
                ),
                make_pack(
                    name="beta",
                    discriminator="beta",
                    arch=["gfx942"],
                    kernels=right_kernels,
                ),
            ]
        )

    def test_a_pack_whose_every_kernel_was_absorbed_is_refused(self):
        """Both kernels complete to the same catalog tuple on the same architecture from
        the same source at the same priority, so engine-wide de-duplication empties
        ``beta``."""
        config = self._two_packs(
            [make_kernel(name="absorbing")], [make_kernel(name="absorbed")]
        )

        with pytest.raises(ValueError) as excinfo:
            build_kdp_documents(config, mint_ids(config))
        message = str(excinfo.value)
        assert "beta" in message, message
        assert "alpha" in message, message
        assert "absorbed" in message and "absorbing" in message, message

    def test_two_packs_of_distinct_kernels_still_both_ship(self):
        """The control: nothing is absorbed, so nothing is refused."""
        config = self._two_packs(
            [make_kernel(name="on_alpha")],
            [
                make_kernel(
                    name="on_beta", metadata={"block_size": 128, "dtype": "FLOAT"}
                )
            ],
        )

        inventory = emitted_inventory(
            config, build_kdp_documents(config, mint_ids(config))
        )
        entry = inventory["arches"]["gfx942"]
        assert entry["pack_names"] == ["test_alpha", "test_beta"]
        assert entry["pack_count"] == 2
        assert inventory["total_descriptor_count"] == 2


class TestEmittedInventory:
    """The context's view of what SHIPS, not what was authored: a census rendered from
    the authored count reports a shortfall whenever generation expressions overlap."""

    def test_the_count_is_the_deduplicated_one_not_the_authored_one(self):
        config = make_minimal_config(
            packs=[
                make_pack(
                    kernels=[
                        make_kernel(name="expr_a"),
                        make_kernel(name="expr_b"),  # same metadata, second expression
                    ],
                    arch=["gfx942"],
                )
            ]
        )
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        assert inventory["total_descriptor_count"] == 1
        assert inventory["arches"]["gfx942"]["descriptor_count"] == 1

    def test_multi_arch_entries_are_disjoint_and_union_to_the_emitted_set(self):
        config = make_minimal_config(
            dialect="packaged",
            kernel_source_kind="rocke",
            packs=[
                make_pack(
                    name="a",
                    arch=["gfx942"],
                    discriminator="a",
                    kernels=[make_kernel(name="on_942")],
                ),
                make_pack(
                    name="b",
                    arch=["gfx950"],
                    discriminator="b",
                    kernels=[
                        make_kernel(
                            name="on_950",
                            metadata={"block_size": 128, "dtype": "FLOAT"},
                        )
                    ],
                ),
            ],
        )
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        arches = inventory["arches"]
        assert set(arches) == {"gfx942", "gfx950"}
        left = set(arches["gfx942"]["descriptor_names"])
        right = set(arches["gfx950"]["descriptor_names"])
        assert not left & right
        assert left | right == {"on_942", "on_950"}
        assert inventory["total_descriptor_count"] == 2

    def _two_packs_on_one_arch(self, left_names, right_names):
        """Two packs on one architecture, so every descriptor lands in one bucket. Every
        kernel gets its own ``block_size``, so de-duplication keeps them all."""
        block_size = iter(range(64, 512))

        def pack(name, kernel_names):
            return make_pack(
                name=name,
                discriminator=name,
                arch=["gfx942"],
                kernels=[
                    make_kernel(
                        name=kernel_name,
                        metadata={"block_size": next(block_size), "dtype": "FLOAT"},
                    )
                    for kernel_name in kernel_names
                ],
            )

        return make_minimal_config(
            packs=[pack("a", left_names), pack("b", right_names)]
        )

    def test_three_distinct_descriptors_over_two_packs_count_as_three(self):
        """The control: a count derived from the name set or the pack count reads 2
        here."""
        config = self._two_packs_on_one_arch(["on_a"], ["on_b", "on_c"])
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        entry = inventory["arches"]["gfx942"]
        assert entry["descriptor_count"] == 3
        assert inventory["total_descriptor_count"] == 3
        assert entry["descriptor_names"] == ["on_a", "on_b", "on_c"]

    def _inventory_for_pack_arch(self, tmp_path, pack_arch):
        """The inventory of a one-kernel engine, read through ``load_config`` because
        the arch normalisation is the LOADER's."""
        import yaml

        from codegen.config_loader import load_config

        raw = {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "specialization": {
                "metadata_fields": [],
                "matcher_only_fields": ["block_size"],
                "bindings": {},
                "vocabulary": {},
            },
            "packs": [
                {
                    "name": "p",
                    "arch": pack_arch,
                    "kernels": [
                        {
                            "name": "only",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "k.hip",
                                "entry_point": "k",
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        config = load_config(path)
        ids = mint_ids(config)
        return emitted_inventory(config, build_kdp_documents(config, ids))

    def test_a_repeated_arch_counts_its_one_descriptor_once(self, tmp_path):
        """The inventory appends per arch ENTRY, so a repeated arch value reports a
        per-arch count above ``total_descriptor_count``."""
        inventory = self._inventory_for_pack_arch(tmp_path, ["gfx942", "gfx942"])
        assert inventory["arches"]["gfx942"]["descriptor_count"] == 1
        assert inventory["total_descriptor_count"] == 1

    def test_two_distinct_arches_each_count_the_descriptor_once(self, tmp_path):
        """The control: collapsing the arch list to its first entry reads 1 for gfx942
        here exactly as the repeat above does."""
        inventory = self._inventory_for_pack_arch(tmp_path, ["gfx942", "gfx950"])
        assert set(inventory["arches"]) == {"gfx942", "gfx950"}
        assert inventory["arches"]["gfx942"]["descriptor_count"] == 1
        assert inventory["arches"]["gfx950"]["descriptor_count"] == 1
        assert inventory["total_descriptor_count"] == 1

    def test_inventory_gives_every_member_the_generic_rows(self):
        """Each member's row holds the generic's descriptors, its own, and the wildcard's."""
        generic = make_pack(
            name="generic",
            arch=["gfx11-generic"],
            kernels=[make_kernel(name="generic.k", metadata={"block_size": 64})],
        )
        explicit = make_pack(
            name="explicit",
            arch=["gfx1151"],
            kernels=[make_kernel(name="explicit.k", metadata={"block_size": 128})],
        )
        other = make_pack(
            name="other",
            arch=["gfx942"],
            kernels=[make_kernel(name="other.k", metadata={"block_size": 256})],
        )
        everywhere = make_pack(
            name="everywhere",
            arch=[],
            kernels=[make_kernel(name="any.k", metadata={"block_size": 512})],
        )
        config = make_minimal_config(packs=[generic, explicit, other, everywhere])
        documents = build_kdp_documents(config, mint_ids(config))
        arches = emitted_inventory(config, documents)["arches"]

        names = {a: set(row["descriptor_names"]) for a, row in arches.items()}
        assert names["gfx1100"] == {"generic.k", "any.k"}
        assert names["gfx1153"] == {"generic.k", "any.k"}
        assert names["gfx1151"] == {"generic.k", "explicit.k", "any.k"}
        assert names["gfx942"] == {"other.k", "any.k"}
        assert names["gfx11-generic"] == {"generic.k", "any.k"}
        assert "gfx1154" not in names
        stems = {config.kdp_stem(p) for p in (generic, everywhere)}
        assert set(arches["gfx1100"]["pack_names"]) == stems

    def test_the_packaged_dialect_reports_the_kind_that_actually_ships(self):
        """hkp_pack lowers ``rocke``/``hip`` to ``kpack`` before the loader reads it, so
        a census expecting the AUTHORED kind expects one that never arrives."""
        config = make_minimal_config(
            dialect="packaged",
            kernel_source_kind="rocke",
            packs=[make_pack(arch=["gfx950"])],
        )
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        assert inventory["source_kind"] == "kpack"

    def test_the_direct_load_dialect_reports_its_authored_kind(self):
        """Nothing lowers a direct-load bundle, so what it authored is what the loader
        reads."""
        config = make_minimal_config()
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        assert inventory["source_kind"] == "embedded_source"

    @staticmethod
    def _census_row(rendered: str, arch: str) -> str:
        """The body of ``arch``'s rendered ``ExpectedInventory``, scoped to the ONE row
        the census selects and compares by set equality."""
        match = re.search(
            r'\{"' + re.escape(arch) + r'",\s*\n\s*ExpectedInventory\{(.*?)\n\s*\}\},',
            rendered,
            re.DOTALL,
        )
        assert match, f"no census row rendered for {arch}"
        return match.group(1)

    @staticmethod
    def _assert_every_inventory_row_is_rendered(generator, config):
        """Every arch, pack and descriptor name the inventory holds reaches the rendered
        census. Returns the rendered text."""
        rendered = generator._render_template("test_packs.cpp.j2", config)
        inventory = emitted_inventory(
            config, build_kdp_documents(config, mint_ids(config))
        )
        assert f'std::string("{inventory["sdk_version"]}")' in rendered
        assert inventory["arches"], "fixture engine emits no inventory rows"
        for arch, entry in inventory["arches"].items():
            row = TestEmittedInventory._census_row(rendered, arch)
            for stem in entry["pack_names"]:
                assert f'"{config.engine.namespace}:{stem}",' in row
            for name in entry["descriptor_names"]:
                assert f'"{name}",' in row
        return rendered

    def test_the_inventory_reaches_the_rendered_census(
        self, generator, scale_add_config
    ):
        """Asserted on the RENDERED TEXT rather than the context dict: a key read by no
        template is threading with nothing behind it. Run over the MIXED bundle too,
        because the wildcard/concrete union is what a single-arch config cannot
        exercise."""
        self._assert_every_inventory_row_is_rendered(generator, scale_add_config)

        mixed = TestWildcardInventoryUnion._mixed_bundle()
        rendered = self._assert_every_inventory_row_is_rendered(generator, mixed)
        concrete = self._census_row(rendered, "gfx942")
        assert '"on_942",' in concrete
        assert '"anywhere",' in concrete, (
            "the wildcard kernel loads on gfx942 too, so a gfx942 row without it "
            "asserts a short set against a loader holding both"
        )
        wildcard_pack = f'"{mixed.engine.namespace}:{mixed.kdp_stem(mixed.packs[0])}",'
        assert wildcard_pack in concrete, (
            "the pack names no arch, so it loads on gfx942 as well and the row that "
            "device reads has to name it"
        )
        assert '"on_942",' not in self._census_row(rendered, "*"), (
            "the control: the union runs one way only, so the wildcard row must not "
            "advertise a concrete-arch kernel on every device"
        )


#: The two cases ``test_matchers.cpp.j2`` must hand every generated engine, named once
#: because the shipped-config and heuristic-free checks assert the same pair.
_BOTH_DIRECTIONS = {
    "AcceptsAGraphThisEngineServes",
    "DeclinesAGraphOutsideItsApplicability",
}


class TestMatcherStubDirections:
    """An accept-only test passes for a matcher stuck on ``true`` and a decline-only
    test for one stuck on ``false``, so both directions must ship."""

    @staticmethod
    def _cases(config, rendered: str) -> dict:
        return emitted_cases(rendered, "TEST", f"{config.engine.pascal_name}Matchers")

    def test_the_stub_emits_an_accept_case_and_a_decline_case(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template("test_matchers.cpp.j2", scale_add_config)
        cases = self._cases(scale_add_config, rendered)
        missing = _BOTH_DIRECTIONS - set(cases)
        assert not missing, (
            f"the matcher stub is missing {sorted(missing)} and emitted "
            f"{sorted(cases)}. An engine handed one direction is an engine whose "
            "matcher can be stuck on a constant and pass everything before a device"
        )

    def test_each_direction_instructs_the_expectation_that_proves_it(
        self, generator, scale_add_config
    ):
        """A pair that both say ``EXPECT_TRUE`` restores the hole two cases exist to
        close."""
        rendered = generator._render_template("test_matchers.cpp.j2", scale_add_config)
        cases = self._cases(scale_add_config, rendered)
        symbol = scale_add_config.graph_match_symbol
        accept = cases["AcceptsAGraphThisEngineServes"]
        decline = cases["DeclinesAGraphOutsideItsApplicability"]
        assert f"EXPECT_TRUE({symbol})" in accept
        assert f"EXPECT_FALSE({symbol})" not in accept
        assert f"EXPECT_FALSE({symbol})" in decline
        assert f"EXPECT_TRUE({symbol})" not in decline

    def test_every_emitted_case_announces_itself_as_unfilled(
        self, generator, scale_add_config
    ):
        """Both directions ship as skips, so the suite reports success for an engine
        nothing has matched -- tolerable only because the skip carries the marker
        ``unfilled_placeholders`` counts."""
        rendered = generator._render_template("test_matchers.cpp.j2", scale_add_config)
        cases = self._cases(scale_add_config, rendered)
        assert cases
        for name, body in cases.items():
            assert "GTEST_SKIP()" in body, name
            assert PLACEHOLDER_MARKER in body, name


class TestHeuristicFreeArms:
    """``heuristic: none``: the arm every shipped config declines to take.

    All three configs under ``configs/`` declare ``heuristic: native``, so nothing else
    renders the ``{% else %}`` arms of ``test_packs.cpp.j2`` and
    ``test_matchers.cpp.j2`` or the omitted ``SCORE_SYMBOL`` paths of ``native.cpp.j2``.
    Such an engine ranks on priority, then descriptor id.
    """

    def test_every_template_renders_for_an_engine_with_no_ranking_model(
        self, generator, heuristic_free_config
    ):
        """``StrictUndefined`` makes an arm naming anything unavailable a render-time
        error."""
        assert not heuristic_free_config.engine.has_heuristic
        for name in ("native.cpp.j2", "test_matchers.cpp.j2", "test_packs.cpp.j2"):
            assert generator._render_template(name, heuristic_free_config)

    def test_the_native_stub_omits_the_score_symbol_and_still_registers_the_rest(
        self, generator, heuristic_free_config
    ):
        config = heuristic_free_config
        rendered = generator._render_template("native.cpp.j2", config)
        assert "SCORE_SYMBOL" not in rendered
        assert "scoreKernel" not in rendered
        assert config.score_symbol not in rendered
        # The omission is of the score symbol alone: an arm that also dropped the
        # kernel matcher or the dispatch handler would register an engine that
        # loads and then serves nothing.
        assert (
            "scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);" in rendered
        )
        assert "scope.add(std::string(DISPATCH_SYMBOL)" in rendered
        assert config.graph_match_symbol in rendered
        assert config.kernel_match_symbol in rendered

    def test_the_native_stub_declares_a_score_symbol_when_the_engine_ranks(
        self, generator, scale_add_config
    ):
        """The control: a template emitting nothing satisfies every 'not in' check on
        the heuristic-free arm."""
        rendered = generator._render_template("native.cpp.j2", scale_add_config)
        assert f'SCORE_SYMBOL = "{scale_add_config.score_symbol}"' in rendered
        assert "double scoreKernel(" in rendered
        assert "scope.add(std::string(SCORE_SYMBOL), &scoreKernel);" in rendered

    def test_the_matcher_stub_drops_the_score_case_and_keeps_both_directions(
        self, generator, heuristic_free_config
    ):
        """The two directions sit outside the heuristic branch, so an arm that swallowed
        them would leave a matcher suite testing no direction."""
        config = heuristic_free_config
        rendered = generator._render_template("test_matchers.cpp.j2", config)
        cases = emitted_cases(rendered, "TEST", f"{config.engine.pascal_name}Matchers")
        assert "ScoreReadsAtLeastOneDescriptorField" not in cases
        assert not _BOTH_DIRECTIONS - set(cases), sorted(cases)
        assert config.kernel_match_symbol in rendered
        assert config.score_symbol not in rendered

    def test_the_pack_census_asserts_the_absence_of_a_ranking_model(
        self, generator, heuristic_free_config
    ):
        """Loading successfully cannot show a registration did NOT happen, so the arm
        asks the registry about the symbol this engine WOULD have used."""
        config = heuristic_free_config
        rendered = generator._render_template("test_packs.cpp.j2", config)
        suite = f"Test{config.engine.pascal_name}Packs"
        cases = emitted_cases(rendered, "TEST_F", suite)
        assert "RanksThroughItsRegisteredScoreSymbol" not in cases
        body = cases["ShipsNoHeuristicAndRegistersNoScoreSymbol"]
        assert "EXPECT_FALSE(_set->heuristic.has_value())" in body
        registration = f'isRegistered("{config.score_symbol}")'
        assert f"EXPECT_FALSE(ingestor::ScoreRegistry::{registration})" in body
        assert 'isRegistered("")' not in rendered

    def test_the_pack_census_checks_the_registration_when_the_engine_ranks(
        self, generator, scale_add_config
    ):
        """The control: the two arms must be opposites, not one arm twice."""
        config = scale_add_config
        rendered = generator._render_template("test_packs.cpp.j2", config)
        suite = f"Test{config.engine.pascal_name}Packs"
        cases = emitted_cases(rendered, "TEST_F", suite)
        assert "ShipsNoHeuristicAndRegistersNoScoreSymbol" not in cases
        body = cases["RanksThroughItsRegisteredScoreSymbol"]
        assert "ASSERT_TRUE(_set->heuristic.has_value())" in body
        registration = f'isRegistered("{config.score_symbol}")'
        assert f"EXPECT_TRUE(ingestor::ScoreRegistry::{registration})" in body

    def test_the_uhd_is_not_written_for_an_engine_with_no_ranking_model(
        self, generator, heuristic_free_config, tmp_path
    ):
        """Such a bundle must also SHIP no UHD, or the census asserts the absence of a
        descriptor sitting beside it."""
        config = heuristic_free_config
        written = generator.render(config, tmp_path)
        assert not [rel for rel in written if rel.endswith(".uhd.json")]
        assert sorted(written) == sorted(generator.preview_files(config))


class TestMatcherStubDeviceFixture:
    """``warpSize`` takes part in ``DeviceKey``'s equality AND its hash
    (``plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DeviceKey.hpp``), so the arch and
    the wave size are one fact about one device: templating the arch while writing 64
    beside it emits a device that exists on no wave32 target."""

    @staticmethod
    def _fixture_body(rendered: str) -> str:
        """The body of the emitted ``fixedDeviceProperties()``, scoped to that function
        because the surrounding comment names both fields and the wave sizes in
        prose."""
        match = re.search(
            r"DeviceProperties fixedDeviceProperties\(\)\n\{\n(.*?)\n\}",
            rendered,
            re.DOTALL,
        )
        assert match, "the matcher stub no longer emits a by-value device fixture"
        return match.group(1)

    def _rendered_fixture(self, generator, config) -> str:
        return self._fixture_body(
            generator._render_template("test_matchers.cpp.j2", config)
        )

    def test_a_wave64_arch_renders_a_wave64_device(self, generator, scale_add_config):
        assert scale_add_config.packs[0].arch == ["gfx942"], "fixture arch changed"
        body = self._rendered_fixture(generator, scale_add_config)
        assert 'properties.gcnArchName = "gfx942";' in body
        assert "properties.warpSize = 64;" in body

    def test_a_wave32_arch_renders_a_wave32_device(self, generator):
        """Every config under ``configs/`` targets CDNA, so only an RDNA arch catches a
        constant 64."""
        config = make_minimal_config(packs=[make_pack(arch=["gfx1250"])])
        body = self._rendered_fixture(generator, config)
        assert 'properties.gcnArchName = "gfx1250";' in body
        assert "properties.warpSize = 32;" in body
        assert "properties.warpSize = 64;" not in body

    def test_a_config_naming_no_arch_still_gets_the_documented_default(
        self, generator, binary_ops_config
    ):
        """A bundle restricting no architecture still needs a device to be, and a
        fallback arch paired with the other family's wave is the same impossible
        device."""
        assert not any(pack.arch for pack in binary_ops_config.packs), (
            "fixture now names an architecture, so it no longer exercises the "
            "no-arch fallback"
        )
        body = self._rendered_fixture(generator, binary_ops_config)
        # Spelled out rather than interpolated from DEFAULT_FIXTURE_ARCH: a test
        # written against the constant agrees with whatever the constant becomes,
        # including a value whose wave size the fallback then states wrongly.
        assert 'properties.gcnArchName = "gfx942";' in body
        assert "properties.warpSize = 64;" in body
        assert DEFAULT_FIXTURE_ARCH == "gfx942"


#: A kernel name carrying the three things a C++ string literal cannot hold raw: a
#: double quote, a backslash and a control character. Nothing validates a kernel name's
#: charset -- the loader's patterns cover every other string, never ``kernel.name``.
_HOSTILE_KERNEL_NAME = 'scale_add."f32"\\path\x01'

#: The same name as C++ SOURCE TEXT. Octal, not ``\x01``, because a C++ hex escape is
#: maximal-munch -- see `cpp_escape`.
_ESCAPED_KERNEL_NAME = 'scale_add.\\"f32\\"\\\\path\\001'


#: A Jinja construct of any kind, masked out before the C++ scan so that quotes,
#: slashes and apostrophes belonging to the TEMPLATE language are never mistaken for
#: C++ punctuation. DOTALL because an interpolation may wrap across lines.
_JINJA_CONSTRUCT = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)

#: The subset of the above that EMITS a value, and so is what this check is about.
_JINJA_INTERPOLATION = re.compile(r"\{\{.*?\}\}", re.DOTALL)

#: ``cpp_escape`` applied as the LAST filter of the chain. Last, not merely present:
#: a filter running after it could reintroduce exactly the characters it removed.
_ESCAPE_APPLIED = re.compile(r"\|\s*cpp_escape\s*$")

#: Templates rendering C++ translation-unit text, by name. The ``.cpp.j2``/``.hpp.j2``
#: suites plus the ``_cpp.j2``/``_hpp.j2`` splice fragments, whose emitted ``.txt``
#: is prose wrapped around C++ lines a human pastes into a real translation unit.
_CXX_TEMPLATE_NAME = re.compile(r"[._](cpp|hpp)\.j2$")

#: Templates rendering CMake, which carry no C++ string literal and are out of scope.
_CMAKE_TEMPLATE_NAME = re.compile(r"^cmake_")


def _mask_jinja(text: str) -> tuple[str, list[tuple[int, str]]]:
    """``text`` with every Jinja construct blanked to NULs of the SAME LENGTH, so every
    offset still addresses the same character of the original, plus the
    interpolations."""
    masked = _JINJA_CONSTRUCT.sub(lambda m: "\0" * (m.end() - m.start()), text)
    sites = [(m.start(), m.group(0)) for m in _JINJA_INTERPOLATION.finditer(text)]
    return masked, sites


def _offsets_inside_a_string_literal(masked: str, name: str) -> bytearray:
    """One flag per character of ``masked``: is it inside a C++ ``"..."``?

    A real (small) lexer rather than a quote count, because the templates carry every
    construct that defeats counting: apostrophes in prose, a ``'`` inside a literal,
    ``//`` and ``/** */`` comments holding both quote characters, and ``##`` prose lines
    that are not C++ at all. A site wrongly read as outside a literal is silently
    exempted, so the scanner asserts its assumptions: no literal stays open across a
    newline, and none is a raw literal.
    """
    inside = bytearray(len(masked))
    state = "code"
    index = 0
    line = 1
    at_line_start = True
    while index < len(masked):
        character = masked[index]
        if character == "\n":
            assert state in ("code", "block", "line"), (
                f"{name}:{line}: a {state} literal is still open at end of line. "
                "The scanner does not model line-spanning literals, and would "
                "mis-classify every interpolation after this point."
            )
            state = "code" if state == "line" else state
            line, at_line_start, index = line + 1, True, index + 1
            continue
        if state == "code" and at_line_start:
            if character in " \t":
                index += 1
                continue
            at_line_start = False
            if masked.startswith("##", index):
                # Splice-fragment prose: rendered into a .txt for a human to read,
                # never compiled. Its apostrophes are English, not char literals.
                state = "line"
                continue
        if state == "code":
            if masked.startswith("//", index):
                state, index = "line", index + 2
            elif masked.startswith("/*", index):
                state, index = "block", index + 2
            elif character == '"':
                assert index == 0 or masked[index - 1] not in "Ru8L", (
                    f"{name}:{line}: raw or encoded string literal prefix, whose "
                    "delimiters this scanner does not model"
                )
                state, index = "string", index + 1
            elif character == "'":
                state, index = "char", index + 1
            else:
                index += 1
            continue
        if state == "block":
            state, index = (
                ("code", index + 2)
                if masked.startswith("*/", index)
                else (state, index + 1)
            )
            continue
        if state == "line":
            index += 1
            continue
        if character == "\\":  # string or char: the escape hides the next character
            index += 2
            continue
        if character == ('"' if state == "string" else "'"):
            state, index = "code", index + 1
            continue
        inside[index] = 1 if state == "string" else 0
        index += 1
    assert state in ("code", "line"), f"{name}: template ends inside a {state}"
    return inside


class TestCppStringEscaping:
    """Values interpolated into a generated C++ string literal must be escaped."""

    def test_every_interpolation_inside_a_literal_applies_the_escape(
        self, template_dir
    ):
        """The two value-specific cases below pin one interpolation each; every OTHER
        interpolation inside a ``"..."`` could lose its ``| cpp_escape`` unnoticed. The
        site list is READ OUT OF the template text at run time, with no exemption
        list."""
        unclassified = [
            path.name
            for path in sorted(template_dir.rglob("*.j2"))
            if not _CXX_TEMPLATE_NAME.search(path.name)
            and not _CMAKE_TEMPLATE_NAME.search(path.name)
        ]
        assert not unclassified, (
            f"{unclassified} match neither the C++ nor the CMake naming rule, so "
            "this check silently skipped them. Name them so one rule claims them."
        )

        unescaped, scanned = [], {}
        for path in sorted(template_dir.rglob("*.j2")):
            if not _CXX_TEMPLATE_NAME.search(path.name):
                continue
            text = path.read_text(encoding="utf-8")
            masked, sites = _mask_jinja(text)
            inside = _offsets_inside_a_string_literal(masked, path.name)
            in_literal = [(o, e) for o, e in sites if inside[o]]
            scanned[path] = (len(in_literal), text)
            for offset, expression in in_literal:
                body = expression[2:-2].strip().strip("-").strip()
                if not _ESCAPE_APPLIED.search(body):
                    line = text.count("\n", 0, offset) + 1
                    unescaped.append(f"{path.name}:{line}: {expression.strip()}")

        assert not unescaped, (
            "these interpolations sit inside a C++ string literal but do not end "
            "in `| cpp_escape`, so the value they emit reaches the literal raw:\n  "
            + "\n  ".join(unescaped)
        )
        blind = [
            path.name
            for path, (count, text) in scanned.items()
            if count == 0 and "cpp_escape" in text
        ]
        assert not blind, (
            f"{blind} apply `cpp_escape` somewhere, yet the scan found no site "
            "inside a literal -- the masking or the lexer above has broken and "
            "this check is passing vacuously."
        )

    def test_a_kernel_name_is_escaped_into_the_census_expectation(self, generator):
        config = make_minimal_config(
            packs=[make_pack(kernels=[make_kernel(name=_HOSTILE_KERNEL_NAME)])]
        )
        rendered = generator._render_template("test_packs.cpp.j2", config)
        assert f'"{_ESCAPED_KERNEL_NAME}",' in rendered
        assert _HOSTILE_KERNEL_NAME not in rendered, (
            "the raw name reached the file, so the literal it sits in is closed by "
            "the name's own quote"
        )
        assert "\x01" not in rendered

    def test_the_sdk_version_is_escaped_into_the_census_expectation(self, generator):
        """The second genuinely unvalidated value: ``sdk_version`` is free-form."""
        config = make_minimal_config(engine=make_engine(sdk_version='1.0.0"); //'))
        rendered = generator._render_template("test_packs.cpp.j2", config)
        assert 'std::string("1.0.0\\"); //")' in rendered
        assert 'std::string("1.0.0");' not in rendered


class TestWildcardInventoryUnion:
    """A wildcard descriptor ships on every device, so every concrete row holds it: the
    census selects ONE row and compares it with what loaded by SET EQUALITY."""

    @staticmethod
    def _mixed_bundle():
        return make_minimal_config(
            packs=[
                make_pack(
                    arch=[],
                    kernels=[
                        make_kernel(
                            name="anywhere",
                            metadata={"block_size": 64, "dtype": "FLOAT"},
                        ),
                        make_kernel(
                            name="on_942",
                            arch=["gfx942"],
                            metadata={"block_size": 128, "dtype": "FLOAT"},
                        ),
                    ],
                )
            ]
        )

    def _inventory(self):
        config = self._mixed_bundle()
        ids = mint_ids(config)
        return config, emitted_inventory(config, build_kdp_documents(config, ids))

    def test_the_concrete_row_carries_the_wildcard_pack_and_kernel(self):
        config, inventory = self._inventory()
        row = inventory["arches"]["gfx942"]
        assert set(row["descriptor_names"]) == {"anywhere", "on_942"}
        assert row["descriptor_count"] == 2
        assert row["pack_names"] == [config.kdp_stem(config.packs[0])], (
            "the pack is arch-independent, so it loads on gfx942 -- a row that "
            "names no pack asserts an empty pack set against a loader that has one"
        )
        assert row["pack_count"] == 1

    def test_the_wildcard_row_keeps_only_what_ships_everywhere(self):
        """The control: the union runs one way only."""
        _config, inventory = self._inventory()
        assert set(inventory["arches"]["*"]["descriptor_names"]) == {"anywhere"}
        assert inventory["total_descriptor_count"] == 2


class TestPlaceholderGateOnUnreadableFiles:
    """A file the scan could not read is not a file the scan found clean."""

    def test_an_undecodable_located_file_fails_the_scan_naming_it(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        target = tmp_path / "packs" / "ScaleAddNative.cpp"
        target.write_bytes(
            b"// \xff\xfe not utf-8\n// TODO - " + PLACEHOLDER_MARKER.encode() + b"\n"
        )
        with pytest.raises(ValueError) as excinfo:
            generator.unfilled_placeholders([tmp_path], written)
        assert "packs/ScaleAddNative.cpp" in str(excinfo.value)

    def test_a_readable_tree_still_reports_its_counts(
        self, generator, scale_add_config, tmp_path
    ):
        """The control: the failure above must be about the undecodable file."""
        written = generator.render(scale_add_config, tmp_path)
        unfilled = generator.unfilled_placeholders([tmp_path], written)
        assert unfilled, "a freshly generated engine must carry unfilled stubs"

    def test_the_cli_does_not_claim_coverage_it_did_not_have(self, tmp_path):
        import subprocess
        import sys
        from pathlib import Path

        tool_root = Path(__file__).parent.parent
        out = tmp_path / "out"

        def run(*args):
            return subprocess.run(
                [sys.executable, str(tool_root / "generate.py"), *args],
                cwd=tool_root,
                capture_output=True,
                text=True,
            )

        generated = run(
            "--config",
            str(tool_root / "configs" / "scale_add.yaml"),
            "--output-dir",
            str(out),
        )
        assert generated.returncode == 0, generated.stderr
        # Fill every placeholder, so the only thing between this tree and a clean
        # gate is the one file the scan cannot read.
        for path in out.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                if PLACEHOLDER_MARKER in text:
                    path.write_text(
                        text.replace(PLACEHOLDER_MARKER, "done"), encoding="utf-8"
                    )
        target = out / "packs" / "ScaleAddNative.cpp"
        target.write_bytes(
            b"// \xff\xfe not utf-8\n// TODO - " + PLACEHOLDER_MARKER.encode() + b"\n"
        )

        checked = run(
            "--config",
            str(tool_root / "configs" / "scale_add.yaml"),
            "--output-dir",
            str(out),
            "--check-placeholders",
        )
        assert checked.returncode == 1, checked.stdout
        assert "No unfilled placeholders" not in checked.stdout
        assert "packs/ScaleAddNative.cpp" in checked.stdout + checked.stderr

    def test_a_fresh_generation_fails_on_a_file_it_cannot_read_back(
        self, tmp_path, monkeypatch, capsys
    ):
        """The other CLI arm: ``--check-placeholders`` returns before generation and
        raises from its own handler, leaving the fresh-generation scan with no case at
        all.

        Driven in-process because the fault has to land BETWEEN the write and the
        read-back: a corrupt copy seeded beforehand resolves the relative path TWICE and
        is discarded as ambiguous. Only ``render`` is substituted.
        """
        import sys
        from pathlib import Path

        import generate

        target = "packs/ScaleAddNative.cpp"

        class GeneratorThatLosesAFileAfterWriting(generate.IngestorGenerator):
            def render(self, config, output_dir, **kwargs):
                written = super().render(config, output_dir, **kwargs)
                (output_dir / target).write_bytes(b"// \xff\xfe not utf-8\n")
                return written

        tool_root = Path(__file__).parent.parent
        out = tmp_path / "out"
        monkeypatch.setattr(
            generate, "IngestorGenerator", GeneratorThatLosesAFileAfterWriting
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate.py",
                "--config",
                str(tool_root / "configs" / "scale_add.yaml"),
                "--output-dir",
                str(out),
            ],
        )

        with pytest.raises(SystemExit) as exit_info:
            generate.main()
        captured = capsys.readouterr()
        assert exit_info.value.code == 1, captured.out
        assert "Error:" in captured.err
        assert target in captured.err
        assert (
            "unfilled placeholder(s)" not in captured.out
        ), "a count printed beside an unread file reads as the whole bundle's"


class TestProjectedMetadataIsTypeChecked:
    """A spec value projected into metadata is checked against its declared type."""

    def test_a_spec_pinned_string_for_an_int_field_is_refused(
        self, gfx950_attention_dense_config
    ):
        import copy

        config = copy.deepcopy(gfx950_attention_dense_config)
        pack = config.packs[0]
        kernel = pack.kernels[0]
        field = next(f for f in config.kmd_fields if f.type == "int")
        kernel.metadata.pop(field.name, None)
        kernel.kernel_source.spec[field.name] = "128"

        with pytest.raises(ValueError, match=f"declares as type '{field.type}'"):
            build_kdp(config, pack, mint_ids(config))

    @pytest.mark.parametrize(
        "config_name", ["variants_example.yaml", "axes_example.yaml"]
    )
    def test_an_expansion_path_is_not_double_rejected(
        self, load_test_config, config_name
    ):
        """The control: ``variants`` projects at expansion time and ``axes``
        materialises metadata before validation, so both reach the generator
        type-correct."""
        config = load_test_config(config_name)
        documents = build_kdp_documents(config, mint_ids(config))
        assert any(document["kernelDescriptors"] for _pack, document in documents)


class TestPackagedHsacoKdp:
    def test_hsaco_kernel_inheriting_a_generic_pack_arch_conforms_to_the_pack_rule(
        self,
    ):
        from codegen.config_loader import _check_generic_arch

        kernel = make_kernel(
            name="test.hsaco",
            kernel_source=KernelSource(
                kind="hsaco", file="HsacoFixture.co", symbol="HsacoFixtureAdd"
            ),
        )
        config = make_minimal_config(
            dialect="packaged",
            kernel_source_kind="hsaco",
            packs=[make_pack(kernels=[kernel], arch=["gfx942", "gfx11-generic"])],
        )
        _check_generic_arch(config)
        kdp = build_kdp(config, config.packs[0], mint_ids(config))
        assert kdp["kernelDescriptors"][0]["arch"] == ["gfx942", "gfx11-generic"]

    def test_every_hsaco_ukd_carries_exactly_kind_file_symbol(self):
        """hkp_pack validates a closed field set per kind, so a UKD carrying any other
        kind's key is refused at pack time."""
        kernels = [
            make_kernel(
                name=f"test.{symbol}",
                kernel_source=KernelSource(
                    kind="hsaco", file="HsacoFixture.co", symbol=symbol
                ),
                metadata={"block_size": block_size, "dtype": "FLOAT"},
            )
            for symbol, block_size in (
                ("HsacoFixtureAdd", 64),
                ("HsacoFixtureScale", 128),
            )
        ]
        config = make_minimal_config(
            dialect="packaged",
            kernel_source_kind="hsaco",
            packs=[make_pack(kernels=kernels, arch=["gfx942"])],
        )
        kdp = build_kdp(config, config.packs[0], mint_ids(config))
        assert [k["kernel_source"] for k in kdp["kernelDescriptors"]] == [
            {"kind": "hsaco", "file": "HsacoFixture.co", "symbol": "HsacoFixtureAdd"},
            {"kind": "hsaco", "file": "HsacoFixture.co", "symbol": "HsacoFixtureScale"},
        ]


# ---------------------------------------------------------------------------
# Stable descriptor ids
# ---------------------------------------------------------------------------


def _kernel_ids(docs: dict) -> dict:
    """Kernel name -> id over every KDP; only meaningful where names are unique."""
    return {
        entry["name"]: entry["id"]
        for kdp in kdps(docs)
        for entry in kdp["kernelDescriptors"]
    }


def _copy_bundle(source: Path, root: Path, config) -> Path:
    """A copy of a descriptor directory at ``config.descriptor_dir`` under ``root``,
    so an edited copy still passes for that bundle's own directory."""
    target = root / config.descriptor_dir
    shutil.copytree(source, target)
    return target


def _regenerated(generator, before, after, tmp_path):
    """Render ``before``, regenerate ``after`` over that tree and return both trees'
    documents, ``(before_docs, after_docs)``. Both configs share one descriptor
    directory."""
    directory = before.descriptor_dir
    generator.render(before, tmp_path / "before")
    render_resolved(
        generator, after, tmp_path / "after", ids_from=tmp_path / "before" / directory
    )
    return (
        read_docs(tmp_path / "before" / directory),
        read_docs(tmp_path / "after" / directory),
    )


def _edit_json(path: Path, **changes) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(changes)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


class TestStableIds:
    """Regenerating a bundle keeps the id of every descriptor that is still there.

    Ids are reused, never derived: a descriptor that is new gets a fresh ``uuid4``,
    and an id that is no longer used is never handed to a different descriptor.
    """

    @pytest.mark.parametrize("fixture", ["scale_add_config", "binary_ops_config"])
    def test_regenerating_an_unchanged_config_rewrites_nothing(
        self, generator, request, tmp_path, fixture
    ):
        """No flag is passed: the output tree itself is the previous version."""
        config = request.getfixturevalue(fixture)
        written = generator.render(config, tmp_path)
        directory = tmp_path / config.descriptor_dir
        first = json_bytes(directory)
        assert {f"{config.descriptor_dir}/{name}" for name in first} == {
            rel for rel in written if rel.endswith(".json")
        }
        assert config.is_multi_pack == any(
            name.startswith("operation_is_") for name in first
        ), "the multi-pack case must cover the operation UMDs' ids"

        generator.render(config, tmp_path)

        assert json_bytes(directory) == first
        assert all(b"\r" not in data for data in first.values()), (
            "descriptors must be LF on every platform, or an unchanged bundle "
            "differs from its git blob and its DVC object"
        )

    def test_adding_a_kernel_changes_only_the_added_kernels_id(
        self, generator, scale_add_config, tmp_path
    ):
        grown = copy.deepcopy(scale_add_config)
        added = copy.deepcopy(grown.packs[0].kernels[0])
        added.name = "scale_add.f32_block128"
        added.metadata["block_size"] = 128
        # In front of the others: ids keyed on position would all shift.
        grown.packs[0].kernels.insert(0, added)
        before, after = _regenerated(generator, scale_add_config, grown, tmp_path)
        for name, doc in before.items():
            if not name.endswith(".kdp.json"):
                assert after[name] == doc, f"{name} changed for an added kernel"
        (kdp_before,) = kdps(before)
        (kdp_after,) = kdps(after)
        assert kdp_after["id"] == kdp_before["id"]
        assert kdp_after["kernelDescriptors"][0]["name"] == added.name
        ids_before, ids_after = _kernel_ids(before), _kernel_ids(after)
        for name, kernel_id in ids_before.items():
            assert ids_after[name] == kernel_id, f"{name} was renumbered"
        new_id = ids_after[added.name]
        assert new_id not in all_ids(before)
        assert uuid.UUID(new_id).version == 4, "a new id is minted, not derived"

    def test_a_renamed_kernel_keeps_its_id(self, generator, scale_add_config, tmp_path):
        """A name is not part of the catalog identity, so renaming changes no id."""
        renamed = copy.deepcopy(scale_add_config)
        renamed.packs[0].kernels[0].name = "scale_add.renamed"
        before, after = map(
            _kernel_ids, _regenerated(generator, scale_add_config, renamed, tmp_path)
        )
        assert list(after.values()) == list(before.values())
        assert "scale_add.renamed" in after

    def test_a_retuned_kernel_gets_a_new_id_and_the_rest_keep_theirs(
        self, generator, scale_add_config, tmp_path
    ):
        """The same tuple built from different compiled bytes is not the kernel it
        replaces: an id kept across it would resolve to other code for anything that
        recorded the id."""
        config = copy.deepcopy(scale_add_config)
        extra = copy.deepcopy(config.packs[0].kernels[0])
        extra.name = "scale_add.f32_block128"
        extra.metadata["block_size"] = 128
        config.packs[0].kernels.append(extra)
        retuned = copy.deepcopy(config)
        retuned.packs[0].kernels[0].kernel_source.entry_point = "ScaleAddRetuned"
        before, after = map(
            _kernel_ids, _regenerated(generator, config, retuned, tmp_path)
        )
        first, second = config.packs[0].kernels[0].name, "scale_add.f32_block128"
        assert after[first] != before[first]
        assert after[second] == before[second]

    def test_a_retune_that_also_moves_the_arch_still_gets_a_new_id(
        self, generator, scale_add_config, tmp_path
    ):
        moved = copy.deepcopy(scale_add_config)
        moved.packs[0].arch = [*moved.packs[0].arch, "gfx950"]
        moved.packs[0].kernels[0].kernel_source.entry_point = "ScaleAddRetuned"
        before, after = map(
            _kernel_ids, _regenerated(generator, scale_add_config, moved, tmp_path)
        )
        name = scale_add_config.packs[0].kernels[0].name
        assert after[name] != before[name]

    @pytest.mark.parametrize(
        "knob, keeps",
        [
            ({"block": 64}, True),
            ({"block": 64, "stage": 2}, True),
            ({"block": 128}, False),
            ({"stage": 2}, False),
        ],
        ids=["unchanged", "gained-a-field", "changed-a-field", "dropped-a-field"],
    )
    def test_only_a_changed_or_dropped_spec_field_is_a_retune(
        self, generator, tmp_path, knob, keeps
    ):
        """A registry that starts stating a knob it used to omit must not remint
        every kernel, but a value the prior stated and the new one changes or drops
        is different code."""
        before = make_packaged_config("rocKE/a", [64])
        after = copy.deepcopy(before)
        after.packs[0].kernels[0].kernel_source.spec = dict(knob)
        (kept,), (now,) = (
            _kernel_ids(docs).values()
            for docs in _regenerated(generator, before, after, tmp_path)
        )
        assert (now == kept) is keeps

    def test_a_source_field_outside_spec_is_a_retune(self, generator, tmp_path):
        """A newly stated ``target`` changes what is compiled; only ``spec`` may grow
        without changing the kernel."""
        before = make_packaged_config("rocKE/a", [64])
        after = copy.deepcopy(before)
        after.packs[0].kernels[0].kernel_source.target = "generic"
        (kept,), (now,) = (
            _kernel_ids(docs).values()
            for docs in _regenerated(generator, before, after, tmp_path)
        )
        assert now != kept

    def test_a_retuned_kernel_keeps_the_id_minted_for_it_across_reruns(
        self, generator, tmp_path
    ):
        """The installed bundle holds the kernel at its old source; each rerun into
        the scratch directory must hand the retune the id the first run minted."""
        before = make_packaged_config("rocKE/a", [64, 128])
        generator.render(before, tmp_path / "installed")
        installed = tmp_path / "installed" / before.descriptor_dir
        retuned = copy.deepcopy(before)
        retuned.packs[0].kernels[0].kernel_source.spec = {"block": 99}
        scratch = tmp_path / "scratch"
        runs = []
        for _ in range(3):
            render_resolved(
                generator,
                retuned,
                scratch,
                ids_from=installed,
                also_from=scratch / retuned.descriptor_dir,
            )
            runs.append(_kernel_ids(read_docs(scratch / retuned.descriptor_dir)))

        first, other = (k.name for k in retuned.packs[0].kernels)
        installed_ids = _kernel_ids(read_docs(installed))
        assert runs[0] == runs[1] == runs[2]
        assert runs[0][first] not in installed_ids.values()
        assert runs[0][other] == installed_ids[other]

    def test_the_installed_id_wins_over_a_scratch_pass_of_the_same_source(
        self, generator, tmp_path
    ):
        before = make_packaged_config("rocKE/a", [64])
        generator.render(before, tmp_path / "installed")
        installed = tmp_path / "installed" / before.descriptor_dir
        scratch_dir = tmp_path / "scratch" / before.descriptor_dir
        shutil.copytree(installed, scratch_dir)
        scratch_kdp = next(scratch_dir.glob("*.kdp.json"))
        scratch_id = str(uuid.uuid4())
        _edit_json(
            scratch_kdp,
            kernelDescriptors=[
                {**entry, "id": scratch_id}
                for entry in read_docs(scratch_dir)[scratch_kdp.name][
                    "kernelDescriptors"
                ]
            ],
        )
        (installed_id,) = _kernel_ids(read_docs(installed)).values()

        resolution = resolve_ids(
            before, tmp_path / "again", ids_from=installed, also_from=scratch_dir
        )

        assert resolution.ids[("kernel", 0, 0)] == installed_id

    def test_a_duplicate_the_deduplication_drops_does_not_make_an_arch_move_ambiguous(
        self, generator, tmp_path
    ):
        before = make_packaged_config("rocKE/a", [64])
        after = copy.deepcopy(before)
        after.packs[0].arch = ["gfx950", "gfx942"]
        twin = copy.deepcopy(after.packs[0].kernels[0])
        twin.name = "a.twin"
        after.packs[0].kernels.append(twin)
        docs_before, docs_after = _regenerated(generator, before, after, tmp_path)
        assert [
            e["name"] for k in kdps(docs_after) for e in k["kernelDescriptors"]
        ] == [before.packs[0].kernels[0].name]
        assert _kernel_ids(docs_after) == _kernel_ids(docs_before)

    def test_a_kernel_moved_to_another_pack_keeps_its_id(
        self, generator, binary_ops_config, tmp_path
    ):
        """The key is the catalog entry, not the pack it sits in."""
        before_config = copy.deepcopy(binary_ops_config)
        extra = copy.deepcopy(before_config.packs[0].kernels[0])
        extra.name = "binary_ops_add.f32_block128"
        extra.metadata["block_size"] = 128
        before_config.packs[0].kernels.append(extra)
        moved = copy.deepcopy(before_config)
        moved.packs[1].kernels.append(moved.packs[0].kernels.pop())
        before, after = _regenerated(generator, before_config, moved, tmp_path)
        assert _kernel_ids(after) == _kernel_ids(before)
        assert {n: d["id"] for n, d in after.items()} == {
            n: d["id"] for n, d in before.items()
        }

    def test_a_removed_kernels_id_is_not_handed_to_a_new_kernel(
        self, generator, scale_add_config, tmp_path
    ):
        changed = copy.deepcopy(scale_add_config)
        pack = changed.packs[0]
        removed = pack.kernels.pop(1)
        replacement = copy.deepcopy(removed)
        replacement.name = "scale_add.f32_block512"
        replacement.metadata["block_size"] = 512
        pack.kernels.append(replacement)
        before, after = _regenerated(generator, scale_add_config, changed, tmp_path)
        ids_before, ids_after = _kernel_ids(before), _kernel_ids(after)
        assert ids_before[removed.name] not in all_ids(after)
        survivor = pack.kernels[0].name
        assert ids_after[survivor] == ids_before[survivor]
        assert ids_after[replacement.name] not in all_ids(before)

    def test_same_named_kernels_keep_their_own_ids_when_reordered(
        self, generator, scale_add_config, tmp_path
    ):
        """Neither position nor name identifies a kernel: here two share a name and
        only the metadata tells them apart."""
        before_config = copy.deepcopy(scale_add_config)
        pack = before_config.packs[0]
        twin = copy.deepcopy(pack.kernels[0])
        twin.metadata["block_size"] = 512
        pack.kernels.append(twin)
        assert twin.name == pack.kernels[0].name
        reordered = copy.deepcopy(before_config)
        reordered.packs[0].kernels.reverse()

        def by_block(docs):
            (kdp,) = kdps(docs)
            return {
                e["metadata"]["block_size"]: e["id"] for e in kdp["kernelDescriptors"]
            }

        before, after = map(
            by_block, _regenerated(generator, before_config, reordered, tmp_path)
        )
        assert len(set(before.values())) == 3
        assert after == before

    def test_two_packs_with_one_name_still_get_distinct_ids(self, generator, tmp_path):
        """Pack names are not validated unique either, so a prior pack id is claimed
        once: the second pack of that name cannot share it."""

        def build(pack_names):
            return make_minimal_config(
                packs=[
                    make_pack(
                        name=name,
                        discriminator=f"d{index}",
                        kernels=[
                            make_kernel(
                                name=f"k{index}",
                                metadata={
                                    "block_size": 64 * (index + 1),
                                    "dtype": "FLOAT",
                                },
                            )
                        ],
                    )
                    for index, name in enumerate(pack_names)
                ]
            )

        first = build(["x", "dup"])
        generator.render(first, tmp_path / "first")
        prior = read_docs(tmp_path / "first" / first.descriptor_dir)
        prior_dup_id = prior[f"{first.kdp_stem(first.packs[1])}.kdp.json"]["id"]

        second = build(["dup", "dup"])
        resolution = resolve_ids(
            second,
            tmp_path / "second",
            ids_from=tmp_path / "first" / first.descriptor_dir,
        )

        ids = resolution.ids
        assert [ids[("pack", 0)], ids[("pack", 1)]].count(prior_dup_id) == 1
        values = list(ids.values())
        assert len(values) == len(set(values)), "two descriptors share an id"

    def test_a_new_defaulted_kmd_field_keeps_every_kernel_id(
        self, generator, scale_add_config, tmp_path
    ):
        """A kernel's key completes its metadata with the CURRENT KMD, on both sides:
        the prior entry, which never stated the field, is the same catalog entry as one
        that states the new field's default."""
        before_root = tmp_path / "before"
        directory = scale_add_config.descriptor_dir
        generator.render(scale_add_config, before_root)
        before = read_docs(before_root / directory)
        odd_name = scale_add_config.packs[0].kernels[1].name

        def regenerate(label, overrides):
            config = copy.deepcopy(scale_add_config)
            config.kmd_fields.append(KmdField(name="gain", type="int", default_value=7))
            for kernel in config.packs[0].kernels:
                kernel.metadata["gain"] = overrides.get(kernel.name, 7)
            root = tmp_path / label
            render_resolved(generator, config, root, ids_from=before_root / directory)
            return read_docs(root / directory)

        same = regenerate("same", {})
        assert _kernel_ids(same) == _kernel_ids(before)
        assert same["scale_add.kmd.json"]["id"] == before["scale_add.kmd.json"]["id"]

        odd = regenerate("odd", {odd_name: 8})
        ids_before, ids_odd = _kernel_ids(before), _kernel_ids(odd)
        assert {n for n in ids_before if ids_odd[n] != ids_before[n]} == {odd_name}

    def test_dropping_a_kmd_field_does_not_reuse_an_ambiguous_id(
        self, generator, scale_add_config, tmp_path
    ):
        """Two kernels that differed only in a field the KMD no longer has now share a
        key. Generation still succeeds, and neither id is reused for the survivor."""
        before_config = copy.deepcopy(scale_add_config)
        before_config.kmd_fields.append(
            KmdField(name="gain", type="int", default_value=0)
        )
        pack = before_config.packs[0]
        first = pack.kernels[0]
        first.metadata["gain"] = 1
        second = copy.deepcopy(first)
        second.name = "scale_add.f32_block64_gain2"
        second.metadata["gain"] = 2
        pack.kernels = [first, second]
        after_config = copy.deepcopy(before_config)
        after_config.kmd_fields = [
            f for f in after_config.kmd_fields if f.name != "gain"
        ]
        survivor = after_config.packs[0].kernels[0]
        survivor.metadata.pop("gain")
        after_config.packs[0].kernels = [survivor]
        before, after = _regenerated(generator, before_config, after_config, tmp_path)

        prior_kernel_ids = set(_kernel_ids(before).values())
        assert prior_kernel_ids.isdisjoint(all_ids(after))
        (survivor_id,) = _kernel_ids(after).values()
        assert uuid.UUID(survivor_id).version == 4

    def test_an_explicit_ids_from_wins_over_the_output_directory(
        self, generator, scale_add_config, tmp_path
    ):
        first, second = tmp_path / "first", tmp_path / "second"
        directory = scale_add_config.descriptor_dir
        generator.render(scale_add_config, first)
        generator.render(scale_add_config, second)
        wanted = read_docs(first / directory)
        assert (
            read_docs(second / directory)["scale_add.ued.json"]["id"]
            != wanted["scale_add.ued.json"]["id"]
        ), "the two trees must differ for the test to mean anything"

        render_resolved(generator, scale_add_config, second, ids_from=first / directory)

        assert read_docs(second / directory) == wanted

    def test_a_satellite_shares_the_engine_and_ships_only_its_kdp(
        self, generator, tmp_path
    ):
        """A second bundle of one engine is a satellite: it reuses the home's engine
        level ids so the loader sees one engine, but its own KDP is a new id (two KDPs
        sharing one would conflict) and it writes no engine-level file (hkp_pack
        rejects a repeated id, identical content or not)."""
        bundles = home_and_satellite(generator, tmp_path)
        resolution = bundles.satellite_resolution
        home = read_docs(bundles.home_dir)
        satellite = read_docs(bundles.satellite_dir)

        assert resolution.satellite
        assert resolution.ids["ued"] == home["test.ued.json"]["id"]
        assert resolution.ids["kmd"] == home["test.kmd.json"]["id"]
        assert resolution.ids["udd"] == home["test.udd.json"]["id"]
        assert resolution.ids["uhd"] == home["test.uhd.json"]["id"]
        assert (
            resolution.ids["kernel_match"]
            == home["kernel_dtype_matches_graph.umd.json"]["id"]
        )

        assert sorted(satellite) == ["test.kdp.json"]
        (kdp,) = kdps(satellite)
        (home_kdp,) = kdps(home)
        assert kdp["engine"] == home["test.ued.json"]["id"]
        assert kdp["dispatch"] == home["test.udd.json"]["id"]
        assert home["kernel_dtype_matches_graph.umd.json"]["id"] in kdp["matchers"]
        (consumer,) = kdp["provenance"]["specialization_contract"]["consumers"]
        assert consumer["kmd_id"] == home["test.kmd.json"]["id"]
        assert kdp["id"] != home_kdp["id"]
        assert set(_kernel_ids(satellite).values()).isdisjoint(
            _kernel_ids(home).values()
        )

        descriptors = [w for w in bundles.satellite_written if w.endswith(".json")]
        assert descriptors == [f"{bundles.satellite.descriptor_dir}/test.kdp.json"]
        preview = generator.preview_files(bundles.satellite, resolution=resolution)
        assert preview == [f"{bundles.satellite.descriptor_dir}/test.kdp.json"]

    def test_a_satellite_drops_a_kernel_its_sibling_already_ships(
        self, generator, tmp_path
    ):
        """The overlap (block 128) is one catalog entry to the loader, which would
        refuse the engine on a duplicate tuple."""
        bundles = home_and_satellite(generator, tmp_path)
        (kdp,) = kdps(read_docs(bundles.satellite_dir))
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["b.b256"]

    def test_a_home_regeneration_sees_the_satellites_kernels(self, generator, tmp_path):
        """The seed runs both ways: the home may not add what a satellite ships."""
        bundles = home_and_satellite(generator, tmp_path)
        grown = copy.deepcopy(bundles.home)
        grown.packs[0].kernels.append(make_rocke_kernel("a.dup_of_b", 256))

        render_resolved(
            generator,
            grown,
            bundles.home_root,
            engine_ids_from=[bundles.satellite_dir],
        )

        (kdp,) = kdps(read_docs(bundles.home_dir))
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["a.b64", "a.b128"]

    def test_a_home_regeneration_with_satellites_still_writes_the_engine_files(
        self, generator, tmp_path
    ):
        """Only a satellite drops its engine-level files: a home that has satellites
        is the one bundle that holds them. Rendered into a tree that holds none of
        its files, so a skipped write cannot hide behind an earlier run."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        fresh = tmp_path / "fresh"

        resolution, _written = render_resolved(
            generator,
            bundles.home,
            fresh,
            ids_from=bundles.home_dir,
            engine_ids_from=[bundles.satellite_dir],
        )

        assert not resolution.satellite
        home = read_docs(bundles.home_dir)
        emitted = read_docs(fresh / bundles.home.descriptor_dir)
        assert sorted(emitted) == sorted(home)
        assert {
            "test.kmd.json",
            "test.ued.json",
            "test.udd.json",
            "test.uhd.json",
            "kernel_dtype_matches_graph.umd.json",
        } <= set(emitted)
        assert emitted == home

    def test_a_satellite_refuses_a_competing_different_candidate(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        clash = make_packaged_config("rocKE/b", [128, 256])
        clash.packs[0].kernels[0].kernel_source.spec["block"] = 999
        with pytest.raises(ValueError, match="catalog tuple"):
            resolve_ids(clash, tmp_path / "clash", engine_ids_from=[bundles.home_dir])

    def test_a_refused_candidate_names_the_pack_of_the_sibling_it_clashes_with(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        clash = make_packaged_config("rocKE/b", [128, 256])
        clash.packs[0].kernels[0].kernel_source.spec["block"] = 999
        (home_kdp,) = kdps(read_docs(bundles.home_dir))
        with pytest.raises(ValueError) as refused:
            resolve_ids(clash, tmp_path / "clash", engine_ids_from=[bundles.home_dir])
        assert f"(pack {home_kdp['name']!r})" in str(refused.value)

    def test_a_kernel_differing_from_the_siblings_only_in_priority_is_refused(
        self, generator, tmp_path
    ):
        """Priority is part of what a candidate is, so the same binary at another
        priority is not the sibling's kernel and may not share its tuple."""
        home = make_packaged_config("rocKE/a", [64, 128])
        home.packs[0].kernels[1].priority = 5
        home_root = tmp_path / "a"
        generator.render(home, home_root)
        satellite = make_packaged_config("rocKE/b", [128, 256])
        with pytest.raises(ValueError, match="not the same candidate"):
            resolve_ids(
                satellite,
                tmp_path / "b",
                engine_ids_from=[home_root / home.descriptor_dir],
            )

    def test_a_satellite_whose_kmd_differs_from_the_homes_is_refused(
        self, generator, tmp_path
    ):
        """Its KDPs are completed against, and point at, the home's KMD."""
        bundles = home_and_satellite(generator, tmp_path)
        drifted = make_packaged_config("rocKE/b", [128, 256])
        drifted.kmd_fields[0] = KmdField(
            name="block_size", type="int", default_value=65
        )
        with pytest.raises(ValueError, match="(?i)kmd"):
            resolve_ids(drifted, tmp_path / "drift", engine_ids_from=[bundles.home_dir])

    @pytest.mark.parametrize(
        ("document", "field", "name"),
        [
            ("test.ued.json", "sdk_version", "ued"),
            ("test.udd.json", "dispatch_symbol", "udd"),
            ("kernel_dtype_matches_graph.umd.json", "match_symbol", "kernel match UMD"),
            ("test.uhd.json", "payload", "uhd"),
        ],
        ids=["ued", "udd", "kernel-match-umd", "uhd"],
    )
    def test_a_satellite_whose_engine_document_differs_from_the_homes_is_refused(
        self, generator, tmp_path, document, field, name
    ):
        """Each engine-level document of a satellite is the home's, or the satellite
        was generated from a different engine block."""
        bundles = home_and_satellite(generator, tmp_path)
        _edit_json(bundles.home_dir / document, **{field: "changed-in-the-home"})
        with pytest.raises(
            ValueError, match=f"this config's {name} differs from the home"
        ):
            resolve_ids(
                bundles.satellite,
                tmp_path / "again",
                engine_ids_from=[bundles.home_dir],
            )

    def test_a_home_and_a_satellite_pack_into_one_engine(self, generator, tmp_path):
        """The packer walks one source root and refuses a repeated id anywhere in it."""
        import importlib
        import sys

        package = (
            Path(__file__).resolve().parents[5]
            / "dnn-providers/hip-kernel-provider/descriptor-packaging/python"
        )
        if not (package / "hkp_pack" / "descriptors.py").exists():
            pytest.skip("hkp_pack is not importable from this checkout")
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        descriptors = importlib.import_module("hkp_pack.descriptors")
        targets = importlib.import_module("hkp_pack.generic_targets")

        bundles = home_and_satellite(generator, tmp_path)
        root = tmp_path / "source"
        shutil.copytree(bundles.home_root / "descriptors", root)
        shutil.copytree(
            bundles.satellite_root / "descriptors", root, dirs_exist_ok=True
        )

        flat = descriptors.load_flat_input(
            root,
            targets.GenericTargets.load(targets.DEFAULT_TABLE_PATH),
            log=lambda *_: None,
        )

        assert len(flat.kdps()) == 2
        assert [d.type for d in flat.descriptors].count("ued") == 1

    # -- reader and resolution errors -------------------------------------------

    def test_an_explicit_ids_from_without_the_engine_is_refused(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        other = _copy_bundle(bundles.home_dir, tmp_path / "x", bundles.home)
        _edit_json(other / "test.ued.json", name="hipkernel:Other")
        with pytest.raises(ValueError, match="no UED"):
            resolve_ids(bundles.home, tmp_path / "y", ids_from=other)

    def test_two_ueds_of_one_name_in_a_root_are_ambiguous(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        doubled = _copy_bundle(bundles.home_dir, tmp_path / "x", bundles.home)
        shutil.copy(doubled / "test.ued.json", doubled / "other.ued.json")
        _edit_json(doubled / "other.ued.json", id=str(uuid.uuid4()))
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_ids(bundles.home, tmp_path / "y", ids_from=doubled)

    def test_the_engines_ued_in_two_roots_is_refused(self, generator, tmp_path):
        """Two homes: packing would reject the second set of engine-level files, and
        the loader drops both UEDs of one name."""
        bundles = home_and_satellite(generator, tmp_path)
        other = make_packaged_config("rocKE/c", [512])
        other_root = tmp_path / "c"
        generator.render(other, other_root)
        with pytest.raises(ValueError, match="UED"):
            resolve_ids(
                bundles.home,
                bundles.home_root,
                engine_ids_from=[other_root / other.descriptor_dir],
            )

    def test_siblings_none_of_which_holds_the_ued_are_refused(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        third = make_packaged_config("rocKE/c", [512])
        with pytest.raises(ValueError, match="no sibling"):
            resolve_ids(third, tmp_path / "c", engine_ids_from=[bundles.satellite_dir])

    def test_an_explicit_satellite_prior_must_belong_to_the_homes_engine(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        stray = _copy_bundle(bundles.satellite_dir, tmp_path / "x", bundles.satellite)
        _edit_json(stray / "test.kdp.json", engine=str(uuid.uuid4()))
        with pytest.raises(ValueError, match="(?i)kdp"):
            resolve_ids(
                bundles.satellite,
                tmp_path / "y",
                ids_from=stray,
                engine_ids_from=[bundles.home_dir],
            )

    @pytest.mark.parametrize("explicit", [True, False], ids=["ids-from", "fallback"])
    def test_a_ued_without_its_kdp_names_the_dvc_pointer(
        self, generator, tmp_path, explicit
    ):
        """gfx950's KDP is DVC-only. Without it every kernel id would be reminted
        silently, so both ways of finding the prior refuse and say where it is."""
        bundles = home_and_satellite(generator, tmp_path)
        root = tmp_path / "stub"
        stub = _copy_bundle(bundles.home_dir, root, bundles.home)
        (stub / "test.kdp.json").unlink()
        (stub / "artifacts.dvc").write_text("outs: []\n", encoding="utf-8")
        kwargs = {"ids_from": stub} if explicit else {}
        with pytest.raises(ValueError, match=r"\.dvc"):
            resolve_ids(bundles.home, root, **kwargs)

    def test_kdps_of_one_engine_must_agree_on_the_dispatch(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        doubled = _copy_bundle(bundles.home_dir, tmp_path / "x", bundles.home)
        extra = json.loads((doubled / "test.kdp.json").read_text(encoding="utf-8"))
        extra.update(id=str(uuid.uuid4()), name="hipkernel:extra")
        extra["dispatch"] = str(uuid.uuid4())
        for entry in extra["kernelDescriptors"]:
            entry["id"] = str(uuid.uuid4())
        (doubled / "extra.kdp.json").write_text(json.dumps(extra), encoding="utf-8")
        with pytest.raises(ValueError, match="dispatch"):
            resolve_ids(bundles.home, tmp_path / "y", ids_from=doubled)

    def test_a_satellite_kdp_must_use_the_homes_dispatch(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        stray = _copy_bundle(bundles.satellite_dir, tmp_path / "x", bundles.satellite)
        _edit_json(stray / "test.kdp.json", dispatch=str(uuid.uuid4()))
        with pytest.raises(ValueError, match="dispatch"):
            resolve_ids(bundles.home, bundles.home_root, engine_ids_from=[stray])

    def test_a_sibling_that_is_this_bundle_is_refused(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        with pytest.raises(ValueError, match="(?i)sibling"):
            resolve_ids(
                bundles.home, bundles.home_root, engine_ids_from=[bundles.home_dir]
            )

    def test_a_kernel_id_shared_by_the_own_prior_and_a_sibling_is_refused(
        self, generator, tmp_path
    ):
        """Two kernels of one id are rejected at pack time, so the run says which two
        files hold it rather than carrying the id into the new descriptors."""
        bundles = home_and_satellite(generator, tmp_path)
        (satellite_kernel,) = (
            e["id"]
            for kdp in kdps(read_docs(bundles.satellite_dir))
            for e in kdp["kernelDescriptors"]
        )
        home_kdp = bundles.home_dir / "test.kdp.json"
        document = json.loads(home_kdp.read_text(encoding="utf-8"))
        document["kernelDescriptors"][0]["id"] = satellite_kernel
        home_kdp.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="also in") as refused:
            resolve_ids(
                bundles.home,
                bundles.home_root,
                ids_from=bundles.home_dir,
                engine_ids_from=[bundles.satellite_dir],
            )
        assert str(bundles.home_dir / "test.kdp.json") in str(refused.value)
        assert str(bundles.satellite_dir / "test.kdp.json") in str(refused.value)

    @pytest.mark.parametrize("field", ["name", "kernel_source"])
    def test_a_sibling_kernel_missing_a_required_field_is_refused_naming_the_file(
        self, generator, tmp_path, field
    ):
        bundles = home_and_satellite(generator, tmp_path)
        satellite_kdp = bundles.satellite_dir / "test.kdp.json"
        document = json.loads(satellite_kdp.read_text(encoding="utf-8"))
        del document["kernelDescriptors"][0][field]
        satellite_kdp.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=field) as refused:
            resolve_ids(
                bundles.home,
                bundles.home_root,
                ids_from=bundles.home_dir,
                engine_ids_from=[bundles.satellite_dir],
            )
        assert str(satellite_kdp) in str(refused.value)

    def test_a_sibling_under_a_linked_descriptors_directory_is_found(
        self, generator, tmp_path
    ):
        """The family folder is read from the path as given: a ``descriptors``
        directory that is a link to a tree of another name is still the family
        folder's parent."""
        home = make_packaged_config("rocKE/a", [64, 128])
        generator.render(home, tmp_path / "real")
        target = tmp_path / "linked_target"
        shutil.copytree(tmp_path / "real" / "descriptors", target)
        link = tmp_path / "installed" / "descriptors"
        link.parent.mkdir()
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            made = (
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    capture_output=True,
                )
                if os.name == "nt"
                else None
            )
            if made is None or made.returncode != 0:
                pytest.skip("neither a symlink nor a junction can be created here")
        satellite = make_packaged_config("rocKE/b", [128, 256])

        resolution = resolve_ids(
            satellite,
            tmp_path / "out",
            engine_ids_from=[link / "rocKE" / "a"],
        )

        assert resolution.satellite

    def test_a_sibling_that_is_a_copy_of_another_sibling_is_refused(
        self, generator, tmp_path
    ):
        """Two KDPs of one id are a conflict at load, whatever folder they sit in."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        copied = _copy_bundle(bundles.satellite_dir, tmp_path / "x", bundles.satellite)
        with pytest.raises(ValueError, match="KDP id .* is in both"):
            resolve_ids(
                bundles.home,
                bundles.home_root,
                ids_from=bundles.home_dir,
                engine_ids_from=[bundles.satellite_dir, copied],
            )

    def test_a_sibling_with_a_graph_scope_operation_matcher_is_refused(
        self, generator, tmp_path
    ):
        """A graph-scope matcher names a symbol only its own bundle registers, so a
        multi-pack bundle cannot be a sibling."""
        bundles = home_and_satellite(generator, tmp_path)
        stray = _copy_bundle(bundles.satellite_dir, tmp_path / "x", bundles.satellite)
        matcher_id = str(uuid.uuid4())
        (stray / "operation_is_b.umd.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "id": matcher_id,
                    "name": "graph operation is b",
                    "scope": "graph",
                    "match_symbol": "OperationIsB",
                }
            ),
            encoding="utf-8",
        )
        kdp = json.loads((stray / "test.kdp.json").read_text(encoding="utf-8"))
        _edit_json(stray / "test.kdp.json", matchers=[matcher_id, *kdp["matchers"]])
        with pytest.raises(ValueError, match="graph-scope operation matcher"):
            resolve_ids(
                bundles.home,
                bundles.home_root,
                ids_from=bundles.home_dir,
                engine_ids_from=[stray],
            )

    def test_a_sibling_in_this_bundles_folder_is_refused(self, generator, tmp_path):
        """A copied config writes its KDP into the home's folder."""
        bundles = home_and_satellite(generator, tmp_path)
        inside = tmp_path / "s" / bundles.home.descriptor_dir
        inside.mkdir(parents=True)
        shutil.copy(bundles.satellite_dir / "test.kdp.json", inside)
        with pytest.raises(ValueError, match="authored_subpath"):
            resolve_ids(bundles.home, bundles.home_root, engine_ids_from=[inside])

    def test_ids_from_must_be_this_bundles_directory(self, generator, tmp_path):
        """A sibling passed with the wrong flag would lend this bundle its KDP id."""
        bundles = home_and_satellite(generator, tmp_path)
        with pytest.raises(ValueError, match="(?i)this bundle"):
            resolve_ids(bundles.home, bundles.home_root, ids_from=bundles.satellite_dir)

    def test_siblings_need_a_packaged_config(self, scale_add_config, tmp_path):
        with pytest.raises(ValueError, match="packaged"):
            resolve_ids(scale_add_config, tmp_path / "o", engine_ids_from=[tmp_path])

    def test_siblings_need_a_single_pack_config(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        multi = make_minimal_config(
            dialect=DIALECT_PACKAGED,
            kernel_source_kind="rocke",
            authored_subpath="rocKE/m",
            packs=[
                make_pack(
                    name=name,
                    discriminator=name,
                    kernels=[make_rocke_kernel(f"m.{name}", block)],
                    arch=["gfx950"],
                )
                for name, block in (("p", 512), ("q", 1024))
            ],
        )
        with pytest.raises(ValueError, match="(?i)multi-pack|single-pack"):
            resolve_ids(multi, tmp_path / "m", engine_ids_from=[bundles.home_dir])

    def test_a_satellite_must_sit_in_the_homes_family_folder(self, generator, tmp_path):
        """A disabled family folder is dropped whole by the packer, which would leave
        the satellite's references dangling."""
        bundles = home_and_satellite(generator, tmp_path)
        outside = make_packaged_config("hip/b", [256])
        with pytest.raises(ValueError, match="(?i)family"):
            resolve_ids(outside, tmp_path / "z", engine_ids_from=[bundles.home_dir])

    def test_a_home_outside_any_descriptors_tree_is_refused(self, generator, tmp_path):
        """The family folder is read from the path, so a home copied out of the
        ``descriptors/`` tree gives the satellite nothing to compare against."""
        bundles = home_and_satellite(generator, tmp_path)
        plain = tmp_path / "plain"
        shutil.copytree(bundles.home_dir, plain)
        with pytest.raises(ValueError, match="no 'descriptors/<family>' component"):
            resolve_ids(bundles.satellite, tmp_path / "z", engine_ids_from=[plain])

    def test_a_satellites_output_tree_without_its_siblings_is_refused(
        self, generator, tmp_path
    ):
        """Regenerating a satellite while forgetting its sibling would mint a second
        UED of the same name, and the loader drops both."""
        bundles = home_and_satellite(generator, tmp_path)
        with pytest.raises(ValueError, match="satellite"):
            resolve_ids(bundles.satellite, bundles.satellite_root)

    def test_an_empty_descriptor_dir_is_not_a_prior(
        self, generator, scale_add_config, tmp_path
    ):
        """A failed first run leaves one behind; it must not brick the rerun."""
        (tmp_path / scale_add_config.descriptor_dir).mkdir(parents=True)

        resolution = resolve_ids(scale_add_config, tmp_path)

        assert sum(resolution.report.reused.values()) == 0
        generator.render(scale_add_config, tmp_path, resolution=resolution)
        assert (
            tmp_path / scale_add_config.descriptor_dir / "scale_add.ued.json"
        ).is_file()

    # -- kernel matching across an arch change ----------------------------------

    @pytest.mark.parametrize(
        ("before_arch", "after_arch"),
        [
            (["gfx942"], ["gfx942", "gfx90a"]),
            (["gfx942", "gfx90a"], ["gfx942"]),
            (["gfx942"], ["gfx90a"]),
        ],
        ids=["widened", "narrowed", "moved"],
    )
    def test_changing_a_packs_arch_keeps_every_kernel_id(
        self, generator, scale_add_config, tmp_path, before_arch, after_arch
    ):
        """Tuples are unique here, so the tuple alone tells the kernels apart."""
        before = copy.deepcopy(scale_add_config)
        before.packs[0].arch = before_arch
        after = copy.deepcopy(scale_add_config)
        after.packs[0].arch = after_arch
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, before, after, tmp_path)
        )
        assert len(set(ids_before.values())) == 2
        assert ids_after == ids_before

    def test_changing_one_kernels_arch_keeps_every_kernel_id(
        self, generator, scale_add_config, tmp_path
    ):
        after = copy.deepcopy(scale_add_config)
        after.packs[0].kernels[0].arch = ["gfx90a"]
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, scale_add_config, after, tmp_path)
        )
        assert ids_after == ids_before

    @staticmethod
    def _one_tuple_at_two_arches(config, arch_x, arch_y):
        """Two kernels with ONE completed tuple, told apart by arch alone."""
        config = copy.deepcopy(config)
        pack = config.packs[0]
        pack.arch = []
        x = pack.kernels[0]
        x.arch = arch_x
        y = copy.deepcopy(x)
        y.name = "scale_add.f32_block64_other_arch"
        y.arch = arch_y
        pack.kernels = [x, y]
        return config

    @pytest.mark.parametrize("moved", ["neither", "x", "y"])
    def test_one_tuple_at_disjoint_arches_keeps_each_kernels_own_id(
        self, generator, scale_add_config, tmp_path, moved
    ):
        """The whole key pairs the kernel whose arch stayed, which leaves the other
        the only candidate for its tuple; without that, a swap of the two ids would
        go unnoticed."""
        before = self._one_tuple_at_two_arches(scale_add_config, ["gfx942"], ["gfx90a"])
        after = self._one_tuple_at_two_arches(
            scale_add_config,
            ["gfx908"] if moved == "x" else ["gfx942"],
            ["gfx908"] if moved == "y" else ["gfx90a"],
        )
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, before, after, tmp_path)
        )
        assert len(set(ids_before.values())) == 2
        assert ids_after == ids_before

    def test_two_new_kernels_for_one_prior_kernel_of_the_tuple_both_get_new_ids(
        self, generator, scale_add_config, tmp_path
    ):
        """Which of the two the prior id belonged to is a guess, so neither takes it."""
        before = self._one_tuple_at_two_arches(scale_add_config, ["gfx942"], ["gfx90a"])
        before.packs[0].kernels.pop()
        after = self._one_tuple_at_two_arches(scale_add_config, ["gfx90a"], ["gfx908"])
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, before, after, tmp_path)
        )
        prior = set(ids_before.values())
        assert len(ids_after) == 2
        assert prior.isdisjoint(ids_after.values())
        assert all(uuid.UUID(i).version == 4 for i in ids_after.values())

    def test_the_order_an_arch_list_is_written_in_does_not_identify_a_kernel(
        self, generator, scale_add_config, tmp_path
    ):
        """Two kernels of one tuple, each at two arches: only the arch set tells them
        apart, so a reordered list must still pair with its own prior id."""
        before = self._one_tuple_at_two_arches(
            scale_add_config, ["gfx942", "gfx90a"], ["gfx908", "gfx950"]
        )
        after = self._one_tuple_at_two_arches(
            scale_add_config, ["gfx90a", "gfx942"], ["gfx950", "gfx908"]
        )
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, before, after, tmp_path)
        )
        assert len(set(ids_before.values())) == 2
        assert ids_after == ids_before

    def test_a_kernel_stating_no_arch_is_keyed_by_its_packs(
        self, generator, scale_add_config, tmp_path
    ):
        """One tuple at the pack's arch (stated by none of its kernels) and at a
        second arch that moves: the inheriting kernel keeps its id by its whole key,
        which leaves the moved kernel the one candidate for its tuple."""
        before = self._one_tuple_at_two_arches(scale_add_config, None, ["gfx90a"])
        before.packs[0].arch = ["gfx942"]
        after = self._one_tuple_at_two_arches(scale_add_config, None, ["gfx908"])
        after.packs[0].arch = ["gfx942"]
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, before, after, tmp_path)
        )
        assert len(set(ids_before.values())) == 2
        assert ids_after == ids_before

    def test_a_metadata_fix_remints_only_that_kernels_id(
        self, generator, scale_add_config, tmp_path
    ):
        """A different tuple is a different catalog entry, whatever its arch."""
        after = copy.deepcopy(scale_add_config)
        fixed, untouched = after.packs[0].kernels
        fixed.metadata["block_size"] = 65
        ids_before, ids_after = map(
            _kernel_ids, _regenerated(generator, scale_add_config, after, tmp_path)
        )
        assert ids_after[untouched.name] == ids_before[untouched.name]
        assert ids_after[fixed.name] not in ids_before.values()

    # -- fresh ids --------------------------------------------------------------

    def test_fresh_ids_ignore_the_bundle_already_in_the_output_directory(
        self, generator, scale_add_config, tmp_path
    ):
        generator.render(scale_add_config, tmp_path)
        directory = tmp_path / scale_add_config.descriptor_dir
        existing = read_docs(directory)

        reused = resolve_ids(scale_add_config, tmp_path)
        assert set(reused.ids.values()) <= all_ids(existing)

        resolution = resolve_ids(scale_add_config, tmp_path, fresh=True)

        assert set(resolution.ids.values()).isdisjoint(all_ids(existing))
        assert sum(resolution.report.reused.values()) == 0
        generator.render(scale_add_config, tmp_path, resolution=resolution)
        assert all_ids(read_docs(directory)).isdisjoint(all_ids(existing))

    @pytest.mark.parametrize("flag", ["ids_from", "engine_ids_from"])
    def test_fresh_ids_contradict_the_reuse_inputs(
        self, generator, scale_add_config, tmp_path, flag
    ):
        generator.render(scale_add_config, tmp_path)
        given = tmp_path / scale_add_config.descriptor_dir
        kwargs = {flag: given if flag == "ids_from" else [given]}
        with pytest.raises(ValueError, match="fresh"):
            resolve_ids(scale_add_config, tmp_path / "other", fresh=True, **kwargs)

    # -- warnings ---------------------------------------------------------------

    def test_a_sibling_left_unnamed_in_the_family_folder_is_warned_about(
        self, generator, tmp_path
    ):
        """Left out, the sibling's kernels fall out of the engine-wide duplicate check
        and a tuple competing with one of them drops the whole engine at load."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)

        forgot = resolve_ids(bundles.home, bundles.home_root, ids_from=bundles.home_dir)
        named = resolve_ids(
            bundles.home,
            bundles.home_root,
            ids_from=bundles.home_dir,
            engine_ids_from=[bundles.satellite_dir],
        )

        assert any(str(bundles.satellite_dir) in w for w in forgot.report.warnings)
        assert named.report.warnings == ()

    @staticmethod
    def _dvc_pointer(directory: Path, tracked: Path) -> None:
        import hashlib

        (directory / f"{tracked.name}.dvc").write_text(
            "outs:\n"
            f"- md5: {hashlib.md5(tracked.read_bytes()).hexdigest()}\n"
            f"  size: {tracked.stat().st_size}\n"
            f"  path: {tracked.name}\n",
            encoding="utf-8",
        )

    def test_an_edited_dvc_tracked_file_is_reported_and_an_untouched_one_is_not(
        self, tmp_path
    ):
        from codegen.prior_ids import dvc_mismatches

        tracked = tmp_path / "kernels.kdp.json"
        tracked.write_bytes(b'{"id": "x"}\n')
        self._dvc_pointer(tmp_path, tracked)
        assert dvc_mismatches(tmp_path) == []

        tracked.write_bytes(b'{"id": "y"}\n')

        (message,) = dvc_mismatches(tmp_path)
        assert str(tracked) in message

    def test_a_stale_prior_artifact_is_a_warning_not_a_refusal(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        kdp = bundles.home_dir / "test.kdp.json"
        self._dvc_pointer(bundles.home_dir, kdp)
        assert resolve_ids(bundles.home, bundles.home_root).report.warnings == ()

        kdp.write_bytes(kdp.read_bytes() + b"\n")

        resolution = resolve_ids(bundles.home, bundles.home_root)
        assert [str(kdp) in w for w in resolution.report.warnings] == [True]
        assert sum(resolution.report.reused.values()) > 0

    def test_the_shipped_gfx950_artifact_matches_its_dvc_pointer(self, tmp_path):
        """The directory recipe: the pointer records a manifest hash of ``artifacts/``,
        which only holds once the KDP has been fetched."""
        from codegen.prior_ids import dvc_mismatches

        shipped = (
            Path(__file__).resolve().parents[5]
            / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
            / "descriptors/rocKE/gfx950_attention_dense"
        )
        kdp = shipped / "artifacts" / "gfx950_attention_dense.kdp.json"
        if not kdp.is_file():
            pytest.skip(f"shipped KDP not in this checkout: {kdp}")
        assert dvc_mismatches(shipped) == []

        copy_dir = tmp_path / "gfx950"
        shutil.copytree(shipped, copy_dir)
        edited = copy_dir / "artifacts" / kdp.name
        edited.write_bytes(edited.read_bytes() + b"\n")

        (message,) = dvc_mismatches(copy_dir)
        assert str(edited.parent) in message


class TestDiscoverSiblings:
    """The other bundles of an engine, found among the directories beside one."""

    ENGINE = "hipkernel:Test"

    def test_a_satellite_finds_the_home(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        found = discover_siblings(
            bundles.home_dir.parent, self.ENGINE, bundles.satellite_dir
        )
        assert found == [bundles.home_dir]

    def test_a_bundle_not_installed_yet_finds_the_home_and_the_satellites(
        self, generator, tmp_path
    ):
        """The home holds the UED; the satellite is found through the KDP naming it."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        found = discover_siblings(
            bundles.home_dir.parent, self.ENGINE, bundles.home_dir.parent / "c"
        )
        assert found == [bundles.home_dir, bundles.satellite_dir]

    def test_a_home_finds_its_satellites_through_its_own_ued(self, generator, tmp_path):
        """No other directory holds a UED, so the satellite is only found because the
        home's own directory is searched for the engine's id."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        found = discover_siblings(
            bundles.home_dir.parent, self.ENGINE, bundles.home_dir
        )
        assert found == [bundles.satellite_dir]

    def test_another_engines_bundle_is_not_a_sibling(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        other = make_packaged_config(
            "rocKE/o", [7], engine=make_engine(name="hipkernel:Other")
        )
        render_resolved(generator, other, bundles.home_root)

        found = discover_siblings(
            bundles.home_dir.parent, self.ENGINE, bundles.home_dir
        )

        assert found == [bundles.satellite_dir]

    @pytest.mark.parametrize(
        ("file", "content"),
        [
            ("x.kdp.json", '[{{"engine": "{id}"}}]'),
            ("x.kdp.json", '{{"engine": "{id}"'),
            ("x.ued.json", '[{{"name": "hipkernel:Test", "id": "{id}"}}]'),
            ("x.ued.json", '{{"name": "hipkernel:Test", "id": "{id}"'),
        ],
        ids=[
            "kdp-not-an-object",
            "kdp-unparseable",
            "ued-not-an-object",
            "ued-unparseable",
        ],
    )
    def test_a_malformed_document_names_no_engine(
        self, generator, tmp_path, file, content
    ):
        """A malformed descriptor beside the bundle, even one holding the engine's
        name and id, is skipped rather than raised on."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        ued = json.loads((bundles.home_dir / "test.ued.json").read_text("utf-8"))
        stray = bundles.home_dir.parent / "stray"
        stray.mkdir()
        (stray / file).write_text(content.format(id=ued["id"]), encoding="utf-8")

        found = discover_siblings(
            bundles.home_dir.parent, self.ENGINE, bundles.home_dir
        )

        assert found == [bundles.satellite_dir]

    def test_nothing_is_found_without_the_engines_ued(self, tmp_path):
        (tmp_path / "x").mkdir()
        (tmp_path / "own").mkdir()
        assert (
            discover_siblings(tmp_path / "missing", self.ENGINE, tmp_path / "own") == []
        )
        assert discover_siblings(tmp_path, self.ENGINE, tmp_path / "own") == []


def _prior_bundle(config, root: Path):
    """The previous version of ``config``'s bundle under ``root``, as the reader sees
    it."""
    directory = root / config.descriptor_dir
    engine = find_engine(directory, config.engine.name)
    return read_prior_bundle(directory, engine.ued["id"], engine.kernel_match_umd["id"])


class TestSupplementAndReport:
    """A second previous version that only fills gaps, and what the report says."""

    def test_a_supplement_fills_gaps_and_never_overrides_the_primary(
        self, generator, scale_add_config, tmp_path
    ):
        primary_config = copy.deepcopy(scale_add_config)
        primary_config.packs[0].kernels = primary_config.packs[0].kernels[:1]
        generator.render(primary_config, tmp_path / "primary")
        generator.render(scale_add_config, tmp_path / "supplement")
        primary = _prior_bundle(scale_add_config, tmp_path / "primary")
        supplement = _prior_bundle(scale_add_config, tmp_path / "supplement")
        shared, gap = scale_add_config.packs[0].kernels
        extra = copy.deepcopy(gap)
        extra.name, extra.metadata["block_size"] = "scale_add.f32_block512", 512
        config = copy.deepcopy(scale_add_config)
        config.packs[0].kernels.append(extra)

        ids = mint_ids(config, primary, supplement=supplement)

        primary_ids = {e["name"]: e["id"] for e in primary.kdps[0].entries}
        supplement_ids = {e["name"]: e["id"] for e in supplement.kdps[0].entries}
        assert ids[("kernel", 0, 0)] == primary_ids[shared.name]
        assert primary_ids[shared.name] != supplement_ids[shared.name]
        assert ids[("kernel", 0, 1)] == supplement_ids[gap.name]
        assert ids[("kernel", 0, 2)] not in {
            *primary.all_ids,
            *supplement.all_ids,
        }
        assert ids["ued"] == primary.engine.ued["id"]
        assert len(set(ids.values())) == len(ids)

    def test_a_pack_the_primary_lacks_takes_its_ids_from_the_supplement(
        self, generator, binary_ops_config, tmp_path
    ):
        primary_config = copy.deepcopy(binary_ops_config)
        primary_config.packs[1].name = "elsewhere"
        generator.render(primary_config, tmp_path / "primary")
        generator.render(binary_ops_config, tmp_path / "supplement")
        primary = _prior_bundle(binary_ops_config, tmp_path / "primary")
        supplement = _prior_bundle(binary_ops_config, tmp_path / "supplement")
        wanted = f"{binary_ops_config.engine.namespace}:" + binary_ops_config.kdp_stem(
            binary_ops_config.packs[1]
        )
        (held,) = (k for k in supplement.kdps if k.name == wanted)
        assert wanted not in {k.name for k in primary.kdps}

        ids = mint_ids(binary_ops_config, primary, supplement=supplement)

        assert ids[("pack", 1)] == held.id
        assert ids[("operation_umd", 1)] == held.operation_umd_id

    def test_a_supplement_id_the_primary_holds_for_another_kernel_stays_with_it(
        self, generator, scale_add_config, tmp_path
    ):
        """The supplement fills gaps only: an id the primary already shipped for one
        kernel is not handed to a different kernel, whichever the config lists
        first."""
        shared, gap = scale_add_config.packs[0].kernels
        primary_config = copy.deepcopy(scale_add_config)
        primary_config.packs[0].kernels = [shared]
        generator.render(primary_config, tmp_path / "primary")
        generator.render(scale_add_config, tmp_path / "supplement")
        primary = _prior_bundle(scale_add_config, tmp_path / "primary")
        (shared_id,) = (e["id"] for e in primary.kdps[0].entries)
        supplement_kdp = tmp_path / "supplement" / scale_add_config.descriptor_dir
        supplement_kdp = next(supplement_kdp.glob("*.kdp.json"))
        document = json.loads(supplement_kdp.read_text(encoding="utf-8"))
        for entry in document["kernelDescriptors"]:
            if entry["name"] == gap.name:
                entry["id"] = shared_id
        supplement_kdp.write_text(json.dumps(document), encoding="utf-8")
        supplement = _prior_bundle(scale_add_config, tmp_path / "supplement")
        config = copy.deepcopy(scale_add_config)
        config.packs[0].kernels.reverse()

        ids = mint_ids(config, primary, supplement=supplement)

        assert ids[("kernel", 0, 1)] == shared_id
        assert ids[("kernel", 0, 0)] != shared_id

    def test_a_pack_arch_widening_is_reported_and_an_unchanged_arch_is_not(
        self, generator, scale_add_config, tmp_path
    ):
        import generate

        directory = scale_add_config.descriptor_dir
        generator.render(scale_add_config, tmp_path / "before")
        same = resolve_ids(
            scale_add_config, tmp_path / "x", ids_from=tmp_path / "before" / directory
        )
        assert same.report.arch_moved == ()
        assert "arch change" not in generate.describe_ids(same.report)

        wider = copy.deepcopy(scale_add_config)
        wider.packs[0].arch = ["gfx942", "gfx90a"]
        resolution = resolve_ids(
            wider, tmp_path / "x", ids_from=tmp_path / "before" / directory
        )

        names = tuple(k.name for k in scale_add_config.packs[0].kernels)
        assert resolution.report.arch_moved == names
        summary = generate.describe_ids(resolution.report)
        assert "2 kept an id across an arch change" in summary
        assert names[0] in summary

    def test_an_unfetched_dvc_artifact_beside_the_bundle_is_warned_about(
        self, generator, tmp_path
    ):
        """A satellite whose KDP is a DVC object that was never pulled has no
        descriptor discovery could find, so its kernels would silently fall out of
        the duplicate check."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        unpulled = bundles.home_dir.parent / "c"
        unpulled.mkdir()
        pointer = unpulled / "artifacts.dvc"
        pointer.write_text(
            "outs:\n- md5: abc.dir\n  path: artifacts\n", encoding="utf-8"
        )

        def warnings():
            return resolve_ids(
                bundles.home,
                bundles.home_root,
                ids_from=bundles.home_dir,
                engine_ids_from=[bundles.satellite_dir],
            ).report.warnings

        (warning,) = warnings()
        assert str(pointer) in warning

        (unpulled / "artifacts").mkdir()
        assert warnings() == ()

    def test_an_unfetched_artifact_in_an_identified_directory_is_not_repeated(
        self, generator, tmp_path
    ):
        """A directory holding a UED or a KDP is some engine's bundle already, and a
        missing artifact there is reported when that bundle is read."""
        bundles = home_and_satellite(generator, tmp_path, same_tree=True)
        other = make_packaged_config(
            "rocKE/o", [7], engine=make_engine(name="hipkernel:Other")
        )
        render_resolved(generator, other, bundles.home_root)
        (bundles.home_root / other.descriptor_dir / "artifacts.dvc").write_text(
            "outs:\n- md5: abc.dir\n  path: artifacts\n", encoding="utf-8"
        )

        resolution = resolve_ids(
            bundles.home,
            bundles.home_root,
            ids_from=bundles.home_dir,
            engine_ids_from=[bundles.satellite_dir],
        )

        assert resolution.report.warnings == ()

    def test_an_empty_prior_kdp_is_warned_about(
        self, generator, scale_add_config, tmp_path
    ):
        generator.render(scale_add_config, tmp_path)
        directory = tmp_path / scale_add_config.descriptor_dir
        assert not any(
            "holds no kernels" in w
            for w in resolve_ids(scale_add_config, tmp_path).report.warnings
        )
        kdp = directory / "scale_add.kdp.json"
        _edit_json(kdp, kernelDescriptors=[])

        resolution = resolve_ids(scale_add_config, tmp_path)

        (warning,) = (w for w in resolution.report.warnings if "holds no kernels" in w)
        assert str(kdp) in warning

    def test_the_report_names_a_retired_kernel(
        self, generator, scale_add_config, tmp_path
    ):
        generator.render(scale_add_config, tmp_path / "before")
        after = copy.deepcopy(scale_add_config)
        removed = after.packs[0].kernels.pop()

        resolution = resolve_ids(
            after,
            tmp_path / "after",
            ids_from=tmp_path / "before" / after.descriptor_dir,
        )

        assert resolution.report.retired == (removed.name,)

    def test_the_report_names_a_retired_pack(
        self, generator, binary_ops_config, tmp_path
    ):
        generator.render(binary_ops_config, tmp_path / "before")
        after = copy.deepcopy(binary_ops_config)
        gone = f"{after.engine.namespace}:{after.kdp_stem(after.packs[1])}"
        after.packs[1].name = "elsewhere"

        resolution = resolve_ids(
            after,
            tmp_path / "after",
            ids_from=tmp_path / "before" / after.descriptor_dir,
        )

        assert resolution.report.retired == (gone,)

    def test_the_report_counts_the_operation_matchers_it_reused(
        self, generator, binary_ops_config, tmp_path
    ):
        generator.render(binary_ops_config, tmp_path / "before")

        resolution = resolve_ids(
            binary_ops_config,
            tmp_path / "after",
            ids_from=tmp_path / "before" / binary_ops_config.descriptor_dir,
        )

        assert resolution.report.reused["operation_umds"] == len(
            binary_ops_config.packs
        )

    def test_a_retired_kernel_is_counted_once_across_both_previous_versions(
        self, generator, scale_add_config, tmp_path
    ):
        """The installed bundle and a scratch pass of it hold the same kernel; it is
        one retired kernel, not two."""
        generator.render(scale_add_config, tmp_path / "before")
        directory = scale_add_config.descriptor_dir
        shutil.copytree(
            tmp_path / "before" / directory, tmp_path / "scratch" / directory
        )
        after = copy.deepcopy(scale_add_config)
        removed = after.packs[0].kernels.pop()

        resolution = resolve_ids(
            after,
            tmp_path / "after",
            ids_from=tmp_path / "before" / directory,
            also_from=tmp_path / "scratch" / directory,
        )

        assert resolution.report.retired == (removed.name,)


def _two_launches(config) -> None:
    """Give every kernel of ``config`` a ``segment`` and a ``reduce`` launch."""
    for kernel in config.packs[0].kernels:
        source = kernel.kernel_source
        kernel.kernel_source = dataclasses.replace(
            source,
            launches={
                "segment": source,
                "reduce": dataclasses.replace(source, role="reduce"),
            },
        )


class TestMultiLaunchIds:
    """A kernel with ``launches`` is written as ``sources``; its ids follow the same
    rules as a single-source kernel, launch by launch."""

    @staticmethod
    def _config(blocks=(64, 128), subpath="rocKE/a"):
        config = make_packaged_config(subpath, list(blocks))
        _two_launches(config)
        return config

    def test_regenerating_an_unchanged_multi_launch_bundle_keeps_every_id(
        self, generator, tmp_path
    ):
        config = self._config()
        generator.render(config, tmp_path)
        first = json_bytes(tmp_path / config.descriptor_dir)

        resolution = resolve_ids(config, tmp_path)
        generator.render(config, tmp_path, resolution=resolution)

        assert (
            "sources"
            in kdps(read_docs(tmp_path / config.descriptor_dir))[0][
                "kernelDescriptors"
            ][0]
        )
        assert resolution.report.minted == 0
        assert json_bytes(tmp_path / config.descriptor_dir) == first

    @pytest.mark.parametrize(
        "change, keeps",
        [
            (lambda launches: None, True),
            (lambda launches: launches["reduce"].spec.update(stage=2), True),
            (lambda launches: setattr(launches["reduce"], "role", "other"), False),
            (lambda launches: setattr(launches["reduce"], "target", "generic"), False),
            (lambda launches: launches["reduce"].spec.update(block=7), False),
            (lambda launches: launches.pop("reduce"), False),
            (lambda launches: launches.update(extra=launches["segment"]), False),
        ],
        ids=[
            "unchanged",
            "launch-spec-gains-a-field",
            "launch-role-changed",
            "launch-gains-a-target",
            "launch-spec-value-changed",
            "launch-dropped",
            "launch-added",
        ],
    )
    def test_each_launch_is_judged_alone(self, generator, tmp_path, change, keeps):
        before = self._config(blocks=(64,))
        after = self._config(blocks=(64,))
        source = after.packs[0].kernels[0].kernel_source
        source.launches = {
            name: dataclasses.replace(launch, spec=dict(launch.spec))
            for name, launch in source.launches.items()
        }
        change(source.launches)
        if len(source.launches) < 2:
            source.launches["other"] = dataclasses.replace(
                source.launches["segment"], role="other"
            )
        (kept,), (now,) = (
            _kernel_ids(docs).values()
            for docs in _regenerated(generator, before, after, tmp_path)
        )
        assert (now == kept) is keeps

    def test_a_single_source_kernel_that_gains_launches_is_a_retune(
        self, generator, tmp_path
    ):
        before = make_packaged_config("rocKE/a", [64])
        after = self._config(blocks=(64,))
        (kept,), (now,) = (
            _kernel_ids(docs).values()
            for docs in _regenerated(generator, before, after, tmp_path)
        )
        assert now != kept

    def test_a_satellite_deduplicates_against_a_multi_launch_home(
        self, generator, tmp_path
    ):
        home = self._config(blocks=(64, 128), subpath="rocKE/a")
        home_root = tmp_path / "a"
        render_resolved(generator, home, home_root)
        home_dir = home_root / home.descriptor_dir
        satellite = self._config(blocks=(128, 256), subpath="rocKE/b")

        resolution, _written = render_resolved(
            generator, satellite, tmp_path / "b", engine_ids_from=[home_dir]
        )

        (document,) = (kdp for _pack, kdp in resolution.kdp_documents)
        names = [e["name"] for e in document["kernelDescriptors"]]
        assert names == [satellite.packs[0].kernels[1].name]

    def test_a_sibling_kernel_naming_neither_source_key_is_refused(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        kdp_path = next(bundles.satellite_dir.glob("*.kdp.json"))
        document = json.loads(kdp_path.read_text(encoding="utf-8"))
        del document["kernelDescriptors"][0]["kernel_source"]
        kdp_path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ValueError, match="kernel_source"):
            resolve_ids(
                bundles.home,
                tmp_path / "a",
                ids_from=bundles.home_dir,
                engine_ids_from=[bundles.satellite_dir],
            )
