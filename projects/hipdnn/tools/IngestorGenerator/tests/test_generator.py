# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/generator.py.

Covers: rendered content assertions (not golden-file diffing), UUID
cross-reference threading, allow-listed JSON keys, and the UMD policy
(single-pack -> zero graph-scoped UMDs, multi-pack -> one per pack).

The two REQUIRED content assertions (Task 2A.4, non-negotiable) live in
``TestRequiredTrapAssertions`` below: the graph_match stub's whole-catalog
blast-radius warning, and TestMatchers.cpp's by-value DeviceProperties
construction.
"""

import json

import pytest

from codegen.generator import build_kdp, build_ued, mint_ids


class TestRequiredTrapAssertions:
    """Task 2A.4's two non-negotiable mechanical checks."""

    def test_graph_match_stub_carries_whole_catalog_warning(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        assert "empties" in rendered and "WHOLE catalog" in rendered
        assert "KernelIngestorStateManager.hpp:450-455" in rendered
        assert (
            "every remaining pack" in rendered.lower()
            or "EVERY remaining pack" in rendered
        )

    def test_graph_match_stub_carries_whole_catalog_warning_multi_pack(
        self, generator, binary_ops_config
    ):
        """The warning must survive the multi-pack shape too, not just single-pack."""
        rendered = generator._render_template(
            "native.cpp.j2", binary_ops_config, ids=mint_ids(binary_ops_config)
        )
        assert "WHOLE catalog" in rendered
        assert "KernelIngestorStateManager.hpp:450-455" in rendered

    def test_matchers_stub_constructs_device_properties_by_value(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template(
            "test_matchers.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        # By-value construction: a local DeviceProperties built and returned, never
        # a query through a real device call. Check for an actual CALL (with
        # parens directly preceded by no explanatory prose marker), not merely the
        # function's name -- the doc comment legitimately mentions
        # hipGetDeviceProperties() in prose to explain what NOT to do.
        assert "DeviceProperties properties;" in rendered
        assert "hipGetDeviceProperties(&properties" not in rendered
        assert "= hipGetDeviceProperties(" not in rendered
        assert ".getDeviceProperties()" not in rendered
        assert "BY VALUE" in rendered or "by value" in rendered.lower()


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
    }
    _UKD_KEYS = {
        "version",
        "id",
        "name",
        "kernel_source",
        "metadata",
        "priority",
        "arch",
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
