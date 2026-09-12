# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/generator.py.

Covers: rendered content assertions (not golden-file diffing), UUID
cross-reference threading, allow-listed JSON keys, and the UMD policy
(single-pack -> zero graph-scoped UMDs, multi-pack -> one per pack).
"""

import json

import pytest

from codegen.generator import (
    _dedup_key,
    build_kdp,
    build_kdp_documents,
    build_kmd,
    build_ued,
    emitted_inventory,
    mint_ids,
)
from codegen.models import KmdField
from tests.helpers import make_kernel, make_minimal_config, make_pack


class TestUuidThreading:
    """Every cross-reference must be the SAME id minted for the referenced
    descriptor -- ids come from one dict, never retyped."""

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
        """mint_ids() must never return the same UUID twice across two calls --
        AC #4 is one mint per RUN, not one mint globally, but within a run every
        id must still be unique."""
        ids_a = mint_ids(scale_add_config)
        ids_b = mint_ids(scale_add_config)
        assert ids_a["ued"] != ids_b["ued"]

    def test_every_minted_id_is_distinct(self, binary_ops_config):
        """Randomness carries uniqueness, so assert it rather than assume it.

        Deriving ids from names was tried and reverted: it made an id only as unique
        as the field it keyed on, and neither kernel names nor pack names are
        guaranteed unique by the config. This is the property that replaced it.
        """
        ids = mint_ids(binary_ops_config)
        values = list(ids.values())
        assert len(values) == len(set(values))


class TestDuplicateKernelNamesAreSurvivable:
    """Kernel names are NOT validated unique, so nothing may be keyed on them.

    Nothing in the config loader rejects two kernels sharing a name within a pack.
    An earlier revision keyed both the id and the id-lookup on the name, which gave
    two genuinely distinct variants one id -- and the loader de-duplicates catalog
    entries by id, so a real variant vanished with no error. Ids are random and the
    lookup is keyed on position; this test is what stops either regressing.
    """

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
    """Overlapping generation expressions are expected; the emitted KDP is unique.

    An author should be able to write "the model-trace shapes" and "the published
    sweep shapes" without hand-partitioning them. Two entries with identical
    matcher-visible metadata are ONE candidate to the runtime, so emitting both buys
    nothing and costs a compile, catalog space and a benchmark iteration.
    """

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
        """An omitted optional field must not collapse onto an explicit one.

        The loader substitutes a field's KMD ``default_value`` for anything absent,
        then requires the resulting tuple to be unique per device. So "omit the key"
        and "write the default" are the SAME catalog entry at load time even though
        the JSON differs -- and a collision is not a dropped entry, it drops the whole
        engine, which reaches production as an arm that silently serves nothing.

        The assertion above (unique raw metadata) does not catch this: the two entries
        differ on disk and collide only after defaults are applied. That gap shipped a
        variant set whose engine failed to load while every generator test passed.
        """
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

        # One kernel pins the field to the schema default; its twin leaves it unset.
        pinned = copy.deepcopy(pack.kernels[0])
        pinned.name = pack.kernels[0].name + "_pinned_to_default"
        pinned.metadata[optional.name] = optional.default_value
        unset = copy.deepcopy(pack.kernels[0])
        unset.name = pack.kernels[0].name + "_left_unset"
        unset.metadata[optional.name] = None
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
        """The sentinel's silent twin: say nothing at all, anywhere.

        ``-1`` is a value somebody chose to write, so it can be grepped for and it
        gets caught. Simply never mentioning an optional knob produces a descriptor
        that looks clean and is not: at load the KMD's ``default_value`` becomes the
        catalog key, while the binary was compiled from the BUILDER's own default.
        Nothing requires those two defaults to agree, and when they disagree the
        descriptor advertises a kernel other than the one it names.

        This is the direction that actually shipped. A check that only rejects the
        stated sentinel catches the careful author and waves the hurried one through.
        """
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
        """The converse, so the check above cannot pass by refusing everything.

        A knob absent from metadata but PINNED in ``kernel_source.spec`` is fully
        decided -- the spec is what the binary is built from -- so it must emit, with
        the spec's value derived into the metadata the matcher reads.

        Runs on the PACKAGED fixture deliberately. Only that dialect carries a
        ``kernel_source.spec``, so on the embedded-source config this assertion would
        skip -- and a guard whose positive half never executes is indistinguishable
        from one that refuses everything.
        """
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
        """The loader groups packs by ENGINE ID, so the scope has to be the engine.

        Per-pack de-duplication cannot see a variant the sibling pack already emitted.
        Both then ship: the runtime benchmarks two candidates that can never resolve
        to different code, and -- worse -- identical metadata is a duplicate CATALOG
        TUPLE, which does not drop the entry but drops the WHOLE ENGINE at load.

        This is the mechanism that makes "ship a second bundle for the coverage gap"
        the wrong instinct: the right shape is one de-duplicated union per engine.
        """
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
        """The converse: engine-wide de-duplication must not eat real coverage.

        Without this the test above passes on a generator that drops every kernel
        after the first, which is a far worse defect than the one it fixes.
        """
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
    """Emit a UMD only for genuine per-pack narrowing; a single-pack engine
    gets zero graph-scoped UMDs (mirrors TestConvFwdPack.cpp)."""

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
    """Every descriptor path a CMake fragment lists must exist on disk.

    ``HIPDNN_DESCRIPTOR_FILES`` is the single list driving staging, install, and
    the dependency edge. A fragment naming a file the generator did not write
    installs nothing for that entry, and the engine loses the descriptor with no
    build error -- the same silent-drop class the generator exists to prevent.

    Regression: the fragment template hardcoded ``<slug>_<pack>.kdp.json`` while
    the writer uses ``kdp_stem()``, which is the BARE slug for a single-pack
    engine. Every single-pack bundle therefore shipped a fragment pointing at a
    nonexistent ``<slug>_<slug>.kdp.json``. Multi-pack happened to agree, which
    is why the existing suite stayed green -- so the single-pack case below is
    the one that actually defends the fix.
    """

    @staticmethod
    def _fragment_descriptor_paths(tmp_path):
        text = (tmp_path / "fragments" / "cmake_descriptor_files.txt").read_text()
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_single_pack_fragment_paths_all_exist(
        self, generator, scale_add_config, tmp_path
    ):
        generator.render(scale_add_config, tmp_path)
        listed = self._fragment_descriptor_paths(tmp_path)
        assert listed, "fragment listed no descriptor files at all"
        for rel in listed:
            assert (
                tmp_path / "descriptors" / rel
            ).exists(), f"fragment names {rel}, which the generator never wrote"

    def test_multi_pack_fragment_paths_all_exist(
        self, generator, binary_ops_config, tmp_path
    ):
        generator.render(binary_ops_config, tmp_path)
        listed = self._fragment_descriptor_paths(tmp_path)
        assert listed, "fragment listed no descriptor files at all"
        for rel in listed:
            assert (
                tmp_path / "descriptors" / rel
            ).exists(), f"fragment names {rel}, which the generator never wrote"

    def test_fragment_lists_every_descriptor_written(
        self, generator, binary_ops_config, tmp_path
    ):
        """The converse: a descriptor written but not listed never ships."""
        written = generator.render(binary_ops_config, tmp_path)
        on_disk = {
            rel[len("descriptors/") :]
            for rel in written
            if rel.startswith("descriptors/") and rel.endswith(".json")
        }
        listed = set(self._fragment_descriptor_paths(tmp_path))
        assert on_disk == listed


class TestSpecializationContractEmission:
    """The declaration in force for every kernel of a bundle.

    The bundle is checked on a machine that does not have the rocKE that compiled
    it. The contract riding on the descriptors is the whole input to that check:
    which metadata fields the compiler consumed, how each is read back off the
    builder object, and which are the matcher's alone. Where it is WRITTEN is the
    packager's resolution rule and not a property these assert; what each kernel
    resolves to is.
    """

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
        """Ids are threaded from the one mint, never re-derived.

        A re-derived id points at whatever engine shares a name, and the ids are
        random precisely because names guarantee nothing.
        """
        ids = mint_ids(scale_add_config)
        kdp = build_kdp(scale_add_config, scale_add_config.packs[0], ids)
        for contract in self._consumers(kdp):
            assert contract["schema_version"] == 1
            assert len(contract["consumers"]) == 1
            consumer = contract["consumers"][0]
            assert consumer["engine_id"] == ids["ued"]
            assert consumer["kmd_id"] == ids["kmd"]

    def test_the_entry_carries_exactly_the_six_contract_keys(self, scale_add_config):
        """The consumer of this data rejects a missing OR an unknown key, so an
        extra one is as fatal as an absent one."""
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
        """The non-compiled path declares its fields rather than staying silent."""
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
        partition's, so repeating the declaration per kernel says nothing extra --
        and on a 2733-kernel pack it was 2.8x the file."""
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
        """Silence is not a waiver, on either path.

        An engine emitting descriptors with no declaration ships UKDs that cannot
        be checked against the builder they were compiled from -- and on the
        receiving machine there is no builder left to ask. The rocKE row is the one
        with a real obligation to waive; the direct-load row is there because a
        gate that only fires on the dialect an author is already careful about is
        the gate that never fires.
        """
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
        """The cross-check: the consumer of this data validates it exactly.

        Asserting the shape here and hoping it matches ``hkp_pack`` is how the two
        drift. Import the real validator when the provider tree is available and
        hand it what actually ships.
        """
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
    """``hkp_pack.agreement``, or ``None`` where the provider tree is absent.

    The generator deliberately does not depend on the packager -- descriptor
    generation must not require the kernel toolchain -- so this is a test-only
    bridge to the module that consumes what the generator emits.
    """
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
    """Identity is the COMPLETED tuple plus the devices it covers.

    The loader substitutes each absent field's KMD ``default_value`` before
    comparing, and compares per device. So the emitted JSON is not the catalog key:
    two documents that differ can be one entry, and two that agree can be two.
    Getting this wrong does not drop a descriptor -- a duplicate tuple drops the
    whole engine at load.
    """

    @staticmethod
    def _twin_config(**config_overrides):
        """Two kernels, same source, whose metadata the caller sets."""
        left = make_kernel(name="twin.left")
        right = make_kernel(name="twin.right")
        pack = make_pack(kernels=[left, right], arch=["gfx942"])
        config = make_minimal_config(packs=[pack], **config_overrides)
        return config, pack, left, right

    def test_equal_tuples_on_disjoint_arches_both_survive(self):
        """Two devices, two binaries. Dropping either leaves the engine matching
        nothing on that device -- a coverage hole no count reconciles."""
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
        """Same tuple, same device, DIFFERENT binary.

        Dropping the second discards a binary somebody deliberately built; keeping
        both ships a duplicate catalog tuple that drops the engine at load. Neither
        is a defensible silent outcome, so the generator refuses and names both.
        """
        config, pack, left, right = self._twin_config()
        # Same matcher-visible metadata, different compiled source.
        right.kernel_source.entry_point = "ScaleAddOther"

        with pytest.raises(ValueError) as excinfo:
            build_kdp(config, pack, mint_ids(config))
        message = str(excinfo.value)
        assert "twin.left" in message and "twin.right" in message

    def test_one_candidate_on_unequal_overlapping_arches_names_the_coverage(self):
        """Same tuple, same device, SAME binary -- and unequal coverage.

        Refusing is right: the tuple appears twice on the shared device. But there
        is no kernel_source or priority difference for the author to find, so a
        diagnostic asserting one sends them hunting for a difference that does not
        exist. The coverage is what differs and the coverage is what it must name.
        """
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
        """An absent ``arch`` is the loader's wildcard, not "no coverage".

        Treated as no coverage, a wildcard candidate sits happily beside a concrete
        one holding the same tuple -- and collides on precisely the device the
        concrete one names.
        """
        config, pack, left, right = self._twin_config()
        pack.arch = []
        left.arch = []  # every device
        right.arch = ["gfx942"]
        right.kernel_source.entry_point = "ScaleAddOther"

        with pytest.raises(ValueError, match="overlapping architectures"):
            build_kdp(config, pack, mint_ids(config))

    def test_an_omitted_field_and_the_kmd_default_are_one_key(self):
        """The collision that only appears after the loader completes the tuple.

        One tuple omits an optional field; the other states it at exactly the KMD
        default. The documents differ, the catalog keys must not -- the loader
        substitutes the default before comparing, so those are one entry to it.

        Keyed on the identity function rather than on an emitted pair, because
        `_check_metadata_resolved` refuses to EMIT a descriptor that omits a
        defaulted field at all: an omission that reached a descriptor would already
        have been rejected one layer up. What has to hold here is that the key
        cannot be fooled if it ever did.
        """
        config, _pack, _left, _right = self._twin_config()
        omitted = _dedup_key({"dtype": "FLOAT"}, config)
        stated = _dedup_key({"dtype": "FLOAT", "block_size": 64}, config)
        assert omitted == stated
        # The control: a value that is NOT the default stays a different key.
        assert _dedup_key({"dtype": "FLOAT", "block_size": 128}, config) != stated

    def test_an_int_and_a_float_spelling_of_one_float_value_are_one_entry(self):
        """``1`` and ``1.0`` in a FLOAT field are one catalog entry.

        They compare equal in Python and serialize to different bytes, so an
        identity keyed on the emitted document sees two entries the runtime cannot
        tell apart -- which is a duplicate tuple, which drops the engine.
        """
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
        """A BOOL field's matcher holds ``true``, not ``1``.

        Projecting every boolean to 0/1 destroys that: the descriptor then states
        an integer where the matcher compares a boolean, and declines every graph
        while the engine still loads and every count still reconciles.
        """
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
        """The converse, one field type over.

        A builder spec spells a flag ``true`` while an ``int``-typed KMD field
        carries ``1``; the loader compares an int alternative, so the boolean
        matches nothing. Without both halves, a guard that keeps booleans is
        indistinguishable from one that never converts.
        """
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


class TestEmittedInventory:
    """The template context's view of what SHIPS, not of what was authored.

    A census rendered from the authored count reports a shortfall on every run of
    any config whose generation expressions overlap, and teaches its reader to
    ignore it.
    """

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

    def test_the_packaged_dialect_reports_the_kind_that_actually_ships(self):
        """hkp_pack lowers ``rocke``/``hip`` to ``kpack`` before the loader reads
        it, so a census expecting the AUTHORED kind expects one that never
        arrives."""
        config = make_minimal_config(
            dialect="packaged",
            kernel_source_kind="rocke",
            packs=[make_pack(arch=["gfx950"])],
        )
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        assert inventory["source_kind"] == "kpack"

    def test_the_direct_load_dialect_reports_its_authored_kind(self):
        """The converse: nothing lowers a direct-load bundle, so what it authored
        is what the loader reads."""
        config = make_minimal_config()
        ids = mint_ids(config)
        inventory = emitted_inventory(config, build_kdp_documents(config, ids))
        assert inventory["source_kind"] == "embedded_source"

    def test_every_template_receives_the_inventory(self, generator, scale_add_config):
        """It is exported through the EXISTING context, beside ``config`` and
        ``ids`` -- not through a manifest file or a second CMake input."""
        captured = {}
        original = generator.env.get_template

        class _Recorder:
            def __init__(self, template):
                self._template = template

            def render(self, **context):
                captured.update(context)
                return self._template.render(**context)

        generator.env.get_template = lambda name: _Recorder(original(name))
        try:
            generator._render_template("native.cpp.j2", scale_add_config)
        finally:
            generator.env.get_template = original
        assert "emitted" in captured
        assert captured["emitted"]["sdk_version"] == scale_add_config.engine.sdk_version
