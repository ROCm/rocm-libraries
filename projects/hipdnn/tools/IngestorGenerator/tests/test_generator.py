# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/generator.py: rendered content (not golden-file diffing),
UUID cross-reference threading, allow-listed JSON keys, the UMD policy and the shape
of the gtest cases the emitted C++ stubs carry."""

import json
import re

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
)
from codegen.models import DEFAULT_FIXTURE_ARCH, KmdField
from tests.helpers import make_engine, make_kernel, make_minimal_config, make_pack


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

    def test_ued_heuristic_references_uhd_id(self, scale_add_config):
        ids = mint_ids(scale_add_config)
        ued = build_ued(scale_add_config, ids)
        assert ued["heuristic"] == ids["uhd"]

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

    def test_ids_are_minted_fresh_per_call(self, scale_add_config):
        """One mint per RUN, not one globally; within a run every id must still be
        unique."""
        ids_a = mint_ids(scale_add_config)
        ids_b = mint_ids(scale_add_config)
        assert ids_a["ued"] != ids_b["ued"]

    def test_every_minted_id_is_distinct(self, binary_ops_config):
        """Neither kernel names nor pack names are unique by the config, so no id may be
        derived from one."""
        ids = mint_ids(binary_ops_config)
        values = list(ids.values())
        assert len(values) == len(set(values))


class TestDuplicateKernelNamesAreSurvivable:
    """Kernel names are NOT validated unique, so nothing may be keyed on them: two
    variants sharing an id lose one with no error, and ids are random."""

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
        "heuristic",
        "metadata",
        "knobs",
        "behavior_notes",
        "numerical_notes",
        "sdk_version",
    }
    _UMD_KEYS = {"version", "id", "name", "scope", "match_symbol"}
    _UDD_KEYS = {"version", "id", "name", "dispatch_symbol"}
    _UHD_KEYS = {"version", "id", "name", "kind", "payload"}
    _KDP_KEYS = {
        "version",
        "id",
        "name",
        "arch",
        "matchers",
        "engine",
        "dispatch",
        "kernelDescriptors",
        # An extension key the loader warns about and ignores
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
        agreement = _import_agreement()
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
        agreement = _import_agreement()
        if agreement is None:
            pytest.skip("hkp_pack is not importable from this checkout")
        ids = mint_ids(scale_add_config)
        kmd = build_kmd(scale_add_config, ids)
        kdp = build_kdp(scale_add_config, scale_add_config.packs[0], ids)
        for kernel in kdp["kernelDescriptors"]:
            consumers = agreement.contracts(kernel, {ids["kmd"]: kmd}, kdp)
            assert len(consumers) == 1
            agreement.validate_consumer(consumers[0], kmd)


def _import_agreement():
    """``hkp_pack.agreement``, or ``None`` where the provider tree is absent: descriptor
    generation must not require the kernel toolchain, so this is a test-only bridge."""
    import importlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    package = root / "dnn-providers/hip-kernel-provider/descriptor-packaging/python"
    if not (package / "hkp_pack" / "agreement.py").exists():
        return None
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    try:
        return importlib.import_module("hkp_pack.agreement")
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

    def test_equal_tuples_on_overlapping_arches_are_refused_naming_both(self):
        """Same tuple, same device, DIFFERENT binary: dropping the second discards a
        binary somebody built, keeping both drops the engine at load, so the generator
        names both."""
        config, pack, left, right = self._twin_config()
        # Same matcher-visible metadata, different compiled source.
        right.kernel_source.entry_point = "ScaleAddOther"

        with pytest.raises(ValueError) as excinfo:
            build_kdp(config, pack, mint_ids(config))
        message = str(excinfo.value)
        assert "twin.left" in message and "twin.right" in message

    def test_one_candidate_on_unequal_overlapping_arches_names_the_coverage(self):
        """Same tuple, same device, SAME binary: there is no kernel_source or priority
        difference to find, so the diagnostic must name the coverage."""
        config, pack, left, right = self._twin_config()
        pack.arch = ["gfx942", "gfx950"]
        left.arch = ["gfx942"]
        right.arch = ["gfx942", "gfx950"]

        with pytest.raises(ValueError) as excinfo:
            build_kdp(config, pack, mint_ids(config))
        message = str(excinfo.value)
        assert "SAME candidate" in message
        assert "kernel_source/priority differ" not in message

    def test_a_wildcard_arch_overlapping_a_concrete_one_is_refused(self):
        """An absent ``arch`` is the loader's wildcard: treated as no coverage it sits
        beside a concrete candidate holding the same tuple and collides on that
        device."""
        config, pack, left, right = self._twin_config()
        pack.arch = []
        left.arch = []  # every device
        right.arch = ["gfx942"]
        right.kernel_source.entry_point = "ScaleAddOther"

        with pytest.raises(ValueError, match="overlapping architectures"):
            build_kdp(config, pack, mint_ids(config))

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
            def render(self, config, output_dir):
                written = super().render(config, output_dir)
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
