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
        assert ids[("operation_umd", "add")] in add_kdp["matchers"]
        assert ids[("operation_umd", "max")] in max_kdp["matchers"]
        # Each pack's own operation matcher must NOT appear on the other pack.
        assert ids[("operation_umd", "add")] not in max_kdp["matchers"]
        assert ids[("operation_umd", "max")] not in add_kdp["matchers"]

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
