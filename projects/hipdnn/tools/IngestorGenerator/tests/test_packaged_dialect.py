# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The packaged dialect: rocKE/hip descriptors that hkp_pack lowers.

``arch`` is required by hkp_pack and not by the runtime loader, so a pack
missing it passes every runtime-shaped check and fails only at pack time.
"""

import json
import re

import pytest
from codegen.config_loader import ConfigError, load_config
from codegen.models import (
    DIALECT_DIRECT_LOAD,
    DIALECT_PACKAGED,
    KernelSource,
)
from tests.helpers import make_kernel, make_minimal_config, make_pack


class TestKernelSourceEmission:
    """``as_document()`` emits exactly its kind's keys, never the union."""

    def test_rocke_emits_only_rocke_keys(self):
        ks = KernelSource(
            kind="rocke",
            source="kernels/gfx950/attention_dense.py",
            builder="build_attention_dense",
            spec={"batch": 1},
        )
        doc = ks.as_document()
        assert set(doc) == {"kind", "source", "builder", "spec"}
        assert "source_file" not in doc
        assert "entry_point" not in doc

    def test_embedded_emits_only_embedded_keys(self):
        ks = KernelSource(kind="embedded_source", source_file="X.cpp", entry_point="X")
        doc = ks.as_document()
        assert set(doc) == {"kind", "source_file", "entry_point"}
        assert "builder" not in doc
        assert "spec" not in doc

    def test_hip_emits_only_hip_keys(self):
        ks = KernelSource(
            kind="hip", source="X.cpp", entry="X", build={"defines": {"A": 1}}
        )
        doc = ks.as_document()
        assert set(doc) == {"kind", "source", "entry", "build"}

    def test_unemittable_kind_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="no emitter"):
            KernelSource(kind="kpack").as_document()


class TestPackagedLayout:
    def test_packaged_descriptors_land_at_the_authored_subpath(
        self, gfx950_attention_dense_config
    ):
        config = gfx950_attention_dense_config
        assert config.is_packaged
        assert config.descriptor_dir == "descriptors/rocKE/gfx950_attention_dense"

    def test_direct_load_writes_into_its_authored_set(self, scale_add_config):
        assert not scale_add_config.is_packaged
        assert scale_add_config.descriptor_dir == "test_descriptors/unit/scale_add"

    def test_authored_subpath_defaults_to_kind_over_slug(self):
        config = make_minimal_config(
            dialect=DIALECT_PACKAGED, kernel_source_kind="rocke"
        )
        assert config.descriptor_dir == "descriptors/rocke/test"

    def test_render_writes_every_file_under_the_authored_subpath(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        written = generator.render(gfx950_attention_dense_config, tmp_path)
        descriptors = [w for w in written if w.endswith(".json")]
        assert descriptors, "no descriptors written"
        for rel in descriptors:
            assert rel.startswith("descriptors/rocKE/gfx950_attention_dense/")
            assert (tmp_path / rel).exists()


class TestPackagedKdp:
    def test_kdp_carries_the_rocke_source_verbatim(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        written = generator.render(gfx950_attention_dense_config, tmp_path)
        kdp_path = [tmp_path / w for w in written if w.endswith(".kdp.json")][0]
        kdp = json.loads(kdp_path.read_text())
        ks = kdp["kernelDescriptors"][0]["kernel_source"]
        assert ks["kind"] == "rocke"
        assert ks["source"] == "kernels/gfx950/attention_dense.py"
        assert ks["builder"] == "build_attention_dense"
        # Every non-defaulted AttentionDenseSpec field must be present: hkp_pack
        # hydrates with Spec(**fields), so a missing one is a TypeError at pack time.
        for required in (
            "batch",
            "seqlen_q",
            "seqlen_kv",
            "num_query_heads",
            "num_kv_heads",
            "head_size",
        ):
            assert required in ks["spec"], required

    def test_packaged_kdp_always_carries_arch(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        written = generator.render(gfx950_attention_dense_config, tmp_path)
        kdp_path = [tmp_path / w for w in written if w.endswith(".kdp.json")][0]
        assert json.loads(kdp_path.read_text())["arch"] == ["gfx950"]


class TestPackagedFragments:
    """Neither dialect emits a CMake edit for descriptor installation."""

    @staticmethod
    def _payload(tmp_path, name):
        text = (tmp_path / "fragments" / name).read_text()
        return text, [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_descriptor_files_fragment_splices_nothing(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        generator.render(gfx950_attention_dense_config, tmp_path)
        text, payload = self._payload(tmp_path, "cmake_descriptor_files.txt")
        assert payload == [], (
            "a packaged bundle emitted CMake payload lines; a list naming an "
            "unlowered rocke descriptor installs a copy the runtime loader rejects"
        )
        assert "hkp_pack" in text and "authored subpath" in text.lower()

    def test_census_fragment_registers_the_suite_this_run_writes(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        """The call is spliced verbatim, so its suite must be one this run actually
        wrote: a census whose gtest filter matches nothing runs zero cases and reports
        success."""
        written = generator.render(gfx950_attention_dense_config, tmp_path)
        text, _payload = self._payload(tmp_path, "cmake_test_sources.txt")
        assert "hkp_register_census_tests(TARGET hip_kernel_provider_tests" in text
        assert "PACK_NAME product" in text
        suite = re.search(r"^\s*SUITES (\w+)$", text, re.MULTILINE)
        assert suite, f"the census call names no single suite:\n{text}"
        assert re.search(
            r"^\s*\)$", text, re.MULTILINE
        ), f"the census call is never closed, so the splice would not parse:\n{text}"
        assert f"tests/{suite.group(1)}.cpp" in written

    def test_direct_load_fragment_splices_nothing_either(
        self, generator, scale_add_config, tmp_path
    ):
        """Installation is by directory in BOTH roots: a hand-maintained list would put
        the shard's contents under two authorities that disagree when a file is
        added."""
        generator.render(scale_add_config, tmp_path)
        text, payload = self._payload(tmp_path, "cmake_descriptor_files.txt")
        assert payload == []
        assert scale_add_config.descriptor_dir in text


class TestPackagedValidation:
    def test_packaged_pack_without_arch_rejected(self):
        from codegen.config_loader import _check_dialect

        kernel = make_kernel(
            kernel_source=KernelSource(
                kind="rocke", source="m.py", builder="build_x", spec={"a": 1}
            )
        )
        config = make_minimal_config(
            dialect=DIALECT_PACKAGED,
            kernel_source_kind="rocke",
            packs=[make_pack(kernels=[kernel], arch=[])],
        )
        with pytest.raises(ConfigError, match="packaged dialect requires"):
            _check_dialect(config)

    def test_unknown_dialect_rejected(self):
        from codegen.config_loader import _check_dialect

        config = make_minimal_config(dialect="sideways")
        with pytest.raises(ConfigError, match="must be one of"):
            _check_dialect(config)

    def test_rocke_kernel_missing_builder_rejected(self):
        from codegen.config_loader import _check_kernel_source_fields

        kernel = make_kernel(
            kernel_source=KernelSource(kind="rocke", source="m.py", spec={"a": 1})
        )
        config = make_minimal_config(
            dialect=DIALECT_PACKAGED,
            kernel_source_kind="rocke",
            packs=[make_pack(kernels=[kernel], arch=["gfx950"])],
        )
        with pytest.raises(ConfigError, match="supplies no 'builder'"):
            _check_kernel_source_fields(config)

    def test_rocke_kernel_missing_spec_rejected(self):
        from codegen.config_loader import _check_kernel_source_fields

        kernel = make_kernel(
            kernel_source=KernelSource(
                kind="rocke", source="m.py", builder="build_x", spec={}
            )
        )
        config = make_minimal_config(
            dialect=DIALECT_PACKAGED,
            kernel_source_kind="rocke",
            packs=[make_pack(kernels=[kernel], arch=["gfx950"])],
        )
        with pytest.raises(ConfigError, match="supplies no 'spec'"):
            _check_kernel_source_fields(config)

    def test_the_shipped_gfx950_config_loads_clean(self, gfx950_attention_dense_config):
        config = gfx950_attention_dense_config
        assert config.dialect == DIALECT_PACKAGED
        assert config.kernel_source_kind == "rocke"
        assert config.packs[0].arch == ["gfx950"]


class TestDialectDefaultIsBackwardCompatible:
    def test_config_without_dialect_key_is_direct_load(self, scale_add_config):
        assert scale_add_config.dialect == DIALECT_DIRECT_LOAD

    def test_direct_load_output_is_unchanged_by_the_dialect_work(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        kdp_path = [tmp_path / w for w in written if w.endswith(".kdp.json")][0]
        ks = json.loads(kdp_path.read_text())["kernelDescriptors"][0]["kernel_source"]
        assert set(ks) == {"kind", "source_file", "entry_point"}
        assert ks["kind"] == "embedded_source"
