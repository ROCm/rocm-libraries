# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The packaged dialect: rocKE/hip descriptors that hkp_pack lowers.

``arch`` is required by hkp_pack and not by the runtime loader, so a pack
missing it passes every runtime-shaped check and fails only at pack time.
"""

import json
import re

import pytest
import yaml
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

    def test_rocke_target_is_emitted_only_when_set(self):
        ks = KernelSource(
            kind="rocke",
            source="kernels/gfx1151/wmma_fmha_fwd_v2.py",
            builder="build_wmma_fmha_fwd_v2",
            spec={"head_dim": 64},
            target="generic",
        )
        # Key order is part of the descriptor bytes: target sits before the spec.
        assert list(ks.as_document()) == ["kind", "source", "builder", "target", "spec"]
        assert ks.as_document()["target"] == "generic"

    def test_rocke_companion_kernels_are_emitted_only_when_set(self):
        ks = KernelSource(
            kind="rocke",
            source="kernels/gfx1151/wmma_fmha_decode.py",
            builder="build_wmma_fmha_decode",
            spec={"head_size": 64},
            target="generic",
            companion_kernels=["merge"],
        )
        assert list(ks.as_document()) == [
            "kind",
            "source",
            "builder",
            "target",
            "companion_kernels",
            "spec",
        ]
        assert ks.as_document()["companion_kernels"] == ["merge"]
        plain = KernelSource(kind="rocke", source="m.py", builder="b", spec={"a": 1})
        assert "companion_kernels" not in plain.as_document()

    def test_hip_never_emits_a_target(self):
        ks = KernelSource(kind="hip", source="X.cpp", entry="X", target="generic")
        assert "target" not in ks.as_document()

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

    def test_hsaco_emits_exactly_kind_file_symbol(self):
        ks = KernelSource(
            kind="hsaco",
            file="HsacoFixture.co",
            symbol="HsacoFixtureAdd",
            source="X.cpp",
            spec={"batch": 1},
        )
        assert ks.as_document() == {
            "kind": "hsaco",
            "file": "HsacoFixture.co",
            "symbol": "HsacoFixtureAdd",
        }

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
        success.

        Every assertion reads the comment-stripped payload rather than the raw
        fragment, because the prose around the call names the same keywords and would
        otherwise satisfy a check the emitted call had stopped meeting."""
        written = generator.render(gfx950_attention_dense_config, tmp_path)
        _text, payload = self._payload(tmp_path, "cmake_test_sources.txt")
        spliced = "\n".join(payload)
        # The census binary and the ordinary unit binary are not interchangeable:
        # naming the unit target puts package-census state on every ordinary run.
        assert (
            "hkp_register_census_tests(TARGET hip_kernel_provider_census_tests"
            in spliced
        ), f"the census call does not name the census target:\n{spliced}"
        assert "PACK_NAME product" in spliced
        suite = re.search(r"^\s*SUITES (\w+)$", spliced, re.MULTILINE)
        assert suite, f"the census call names no single suite:\n{spliced}"
        assert re.search(
            r"^\s*\)$", spliced, re.MULTILINE
        ), f"the census call is never closed, so the splice would not parse:\n{spliced}"
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


class TestCompanionKernels:
    """A rocke builder emitting a merge kernel into its primary's code object.

    The UKD declares the role; hkp_pack writes ``merge_symbol`` and
    ``merge_signature`` from the compiled object, so the config declares those
    KMD fields matcher-only and never authors them.
    """

    def _raw(self, config_path):
        raw = yaml.safe_load(config_path("gfx950_attention_dense.yaml").read_text())
        raw["packs"][0]["kernel_defaults"]["companion_kernels"] = ["merge"]
        raw["kmd_fields"] += [
            {"name": "merge_symbol", "type": "string"},
            {"name": "merge_signature", "type": "string"},
        ]
        raw["specialization"]["matcher_only_fields"] = [
            "merge_symbol",
            "merge_signature",
        ]
        return raw

    def _load(self, tmp_path, raw):
        path = tmp_path / "companions.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return load_config(path)

    def test_companion_fields_are_left_for_the_packer(
        self, generator, config_path, tmp_path
    ):
        config = self._load(tmp_path, self._raw(config_path))
        out = tmp_path / "out"
        written = generator.render(config, out)
        kdp = json.loads(
            next(out / w for w in written if w.endswith(".kdp.json")).read_text()
        )
        kmd = json.loads(
            next(out / w for w in written if w.endswith(".kmd.json")).read_text()
        )
        # Mandatory in the KMD, so a shipped kernel the packer did not complete
        # fails at load rather than matching without its merge kernel.
        assert {"name": "merge_symbol", "type": "string"} in kmd["fields"]
        assert {"name": "merge_signature", "type": "string"} in kmd["fields"]
        for ukd in kdp["kernelDescriptors"]:
            assert ukd["kernel_source"]["companion_kernels"] == ["merge"]
            assert "merge_symbol" not in ukd["metadata"]
            assert "merge_signature" not in ukd["metadata"]

    def test_optional_companion_fields_are_left_for_the_packer(
        self, generator, config_path, tmp_path
    ):
        """An engine mixing kernels with and without the companion declares its
        fields optional (empty default). The packer still writes them for the
        kernels declaring the role, so generation must not demand them there."""
        raw = self._raw(config_path)
        for field in raw["kmd_fields"]:
            if field["name"] in ("merge_symbol", "merge_signature"):
                field["default_value"] = ""
        config = self._load(tmp_path, raw)
        out = tmp_path / "out"
        written = generator.render(config, out)
        kdp = json.loads(
            next(out / w for w in written if w.endswith(".kdp.json")).read_text()
        )
        for ukd in kdp["kernelDescriptors"]:
            assert "merge_symbol" not in ukd["metadata"]
            assert "merge_signature" not in ukd["metadata"]

    def test_a_kernel_without_the_companion_states_its_empty_fields(
        self, generator, config_path, tmp_path
    ):
        """Optional companion fields of a kernel that does not declare the role are
        ordinary metadata: nothing writes them, so the config states them."""
        raw = self._raw(config_path)
        del raw["packs"][0]["kernel_defaults"]["companion_kernels"]
        for field in raw["kmd_fields"]:
            if field["name"] in ("merge_symbol", "merge_signature"):
                field["default_value"] = ""
        with pytest.raises(ValueError, match="merge_s"):
            generator.render(self._load(tmp_path, raw), tmp_path / "out")

    @pytest.mark.parametrize("roles", [[], ["Merge"], ["merge", "merge"], "merge"])
    def test_an_unusable_role_list_is_rejected(self, config_path, tmp_path, roles):
        raw = self._raw(config_path)
        raw["packs"][0]["kernel_defaults"]["companion_kernels"] = roles
        with pytest.raises(ConfigError, match="companion_kernels"):
            self._load(tmp_path, raw)

    def test_the_kmd_must_declare_each_companion_field(self, config_path, tmp_path):
        raw = self._raw(config_path)
        raw["kmd_fields"] = [
            f for f in raw["kmd_fields"] if f["name"] != "merge_signature"
        ]
        raw["specialization"]["matcher_only_fields"] = ["merge_symbol"]
        with pytest.raises(
            ConfigError, match=re.escape("kmd_fields entry 'merge_signature'")
        ):
            self._load(tmp_path, raw)

    def test_a_companion_field_must_be_a_string(self, config_path, tmp_path):
        raw = self._raw(config_path)
        for field in raw["kmd_fields"]:
            if field["name"] == "merge_symbol":
                field["type"] = "int"
        with pytest.raises(ConfigError, match="of type 'string'"):
            self._load(tmp_path, raw)

    def test_a_companion_field_is_never_authored(self, config_path, tmp_path):
        raw = self._raw(config_path)
        raw["packs"][0]["variants"][0]["metadata"].append("merge_symbol")
        raw["packs"][0]["variants"][0]["spec_defaults"]["merge_symbol"] = "forged"
        with pytest.raises(ConfigError, match="hkp_pack writes from the compiled"):
            self._load(tmp_path, raw)

    def test_a_companion_field_is_not_spec_bound(self, config_path, tmp_path):
        raw = self._raw(config_path)
        raw["specialization"]["matcher_only_fields"] = ["merge_signature"]
        raw["specialization"]["metadata_fields"].append("merge_symbol")
        raw["specialization"]["bindings"]["merge_symbol"] = {"field": "dtype"}
        with pytest.raises(ConfigError, match="matcher_only_fields"):
            self._load(tmp_path, raw)
