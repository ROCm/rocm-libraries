# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The `hiprtc_file` drop-in kind: config loading, emission, bundle staging.

What separates this kind from `embedded_source` is that nothing downstream
gets a second chance at it. An `embedded_source` mistake fails a build. A
`hiprtc_file` mistake generates clean, installs clean, and shows up as one
`LOG_ERROR` and a missing engine on a machine that has neither the config nor
the bundle -- so every rejection below is a failure moved from there to here.

The substituter's own rules are tested in `test_kernel_defines.py` against the
C++ table; this file tests that the loader actually asks it, and that the
emitter writes and stages what the loader will look for.
"""

from pathlib import Path

import pytest
import yaml

from codegen.config_loader import ConfigError, load_config
from codegen.generator import BundleError
from codegen.models import (
    EMITTABLE_KINDS_BY_DIALECT,
    KERNEL_SOURCE_KIND_HIPRTC_FILE,
    DIALECT_DIRECT_LOAD,
    DIALECT_PACKAGED,
    KernelSource,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hiprtc_dropin"


@pytest.fixture
def dropin_config(load_test_config):
    """The shipped worked example (`configs/hiprtc_dropin.yaml`)."""
    return load_test_config("hiprtc_dropin.yaml")


def load_fixture(name: str):
    return load_config(FIXTURES / name)


class TestKernelSourceDocument:
    """`as_document` writes the kind's own keys and nothing else -- the loader
    hard-fails an unknown key in `kernel_source`."""

    def test_hiprtc_file_emits_its_own_keys(self):
        document = KernelSource(
            kind=KERNEL_SOURCE_KIND_HIPRTC_FILE,
            bundle="sources",
            source_file="attention.hip",
            entry_point="attention_fwd",
            defines={"DTYPE": "$kernel.dtype"},
        ).as_document()
        assert document == {
            "kind": "hiprtc_file",
            "bundle": "sources",
            "source_file": "attention.hip",
            "entry_point": "attention_fwd",
            "defines": {"DTYPE": "$kernel.dtype"},
        }

    def test_hiprtc_file_does_not_borrow_the_packaged_build_key(self):
        """`build` belongs to `packaged`/`hip` and is validated by `hkp_pack`
        against a different schema. A kind carrying both keys would make each
        one's rules the other's silent trap."""
        document = KernelSource(
            kind=KERNEL_SOURCE_KIND_HIPRTC_FILE,
            bundle="sources",
            source_file="attention.hip",
            entry_point="attention_fwd",
            defines={"DTYPE": "float"},
            build={"defines": {"DTYPE": "double"}},
        ).as_document()
        assert "build" not in document
        assert document["defines"] == {"DTYPE": "float"}

    def test_hiprtc_file_is_a_direct_load_kind(self):
        assert (
            KERNEL_SOURCE_KIND_HIPRTC_FILE
            in EMITTABLE_KINDS_BY_DIALECT[DIALECT_DIRECT_LOAD]
        )
        assert (
            KERNEL_SOURCE_KIND_HIPRTC_FILE
            not in EMITTABLE_KINDS_BY_DIALECT[DIALECT_PACKAGED]
        )


class TestConfigLoading:
    def test_worked_example_loads(self, dropin_config):
        kernels = dropin_config.packs[0].kernels
        assert [k.kernel_source.kind for k in kernels] == ["hiprtc_file"] * 2
        # The point of the kind: one source file, two kernels, two binaries.
        assert {k.kernel_source.source_file for k in kernels} == {"ScaleAddHiprtc.hip"}
        assert [k.metadata["dtype"] for k in kernels] == ["float", "_Float16"]

    def test_defines_survive_loading(self, dropin_config):
        assert dropin_config.packs[0].kernels[0].kernel_source.defines == {
            "SCALE_ADD_DTYPE": "$kernel.dtype",
            "SCALE_ADD_BLOCK_SIZE": "$kernel.block_size",
        }

    def test_bundle_names_are_deduplicated(self, dropin_config):
        """One bundle serves many kernels, so the emitter stages it once."""
        assert dropin_config.bundle_names == ["scale_add_sources"]

    def test_config_dir_is_recorded(self, dropin_config):
        assert Path(dropin_config.config_dir).name == "configs"

    def test_valid_fixture_loads(self):
        assert load_fixture("valid.yaml").bundle_names == ["sources"]

    def test_bundle_is_required(self, tmp_path):
        raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
        del raw["packs"][0]["kernels"][0]["kernel_source"]["bundle"]
        path = tmp_path / "no_bundle.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="supplies no 'bundle'"):
            load_config(path)


class TestDefinesRejections:
    """Each of these is a pack dropped at load time on the target machine if it
    is not caught here."""

    def test_undeclared_field(self):
        with pytest.raises(ConfigError, match="does not declare"):
            load_fixture("undeclared_field.yaml")

    def test_float_field(self):
        with pytest.raises(ConfigError, match="is a float"):
            load_fixture("float_field.yaml")

    def test_expression_value(self):
        with pytest.raises(ConfigError, match="literal token replacement"):
            load_fixture("expression_value.yaml")

    def test_non_kernel_token(self):
        with pytest.raises(ConfigError, match=r"\$kernel\."):
            load_fixture("non_kernel_token.yaml")

    def test_non_string_define(self):
        with pytest.raises(ConfigError, match="string->string"):
            load_fixture("non_string_define.yaml")

    def test_diagnostic_names_the_kernel_and_the_key(self):
        """A `defines` message that named only the offending value would leave
        an author grepping a generated variant set for it."""
        with pytest.raises(ConfigError) as error:
            load_fixture("undeclared_field.yaml")
        message = str(error.value)
        assert "fixture.f32" in message
        assert "FIXTURE_DTYPE" in message


class TestBundleRejections:
    def test_escaping_bundle_is_refused(self):
        with pytest.raises(ConfigError, match=r"no '\.\.' component"):
            load_fixture("escaping_bundle.yaml")

    def test_absolute_bundle_is_refused(self, tmp_path):
        raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
        raw["packs"][0]["kernels"][0]["kernel_source"]["bundle"] = "/opt/sources"
        path = tmp_path / "absolute_bundle.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="relative path"):
            load_config(path)

    def test_missing_source_file_is_refused_at_generation(self, generator, tmp_path):
        """Loads clean -- the loader does not read the bundle -- and fails at
        staging, where the file list is known."""
        config = load_fixture("missing_source_file.yaml")
        with pytest.raises(BundleError, match="NotThere.hip"):
            generator.render(config, tmp_path)

    def test_in_memory_config_says_why_it_cannot_stage(self, generator, tmp_path):
        config = load_fixture("valid.yaml")
        config.config_dir = ""
        with pytest.raises(BundleError, match="built in memory"):
            generator.render(config, tmp_path)


class TestEmission:
    @pytest.fixture
    def rendered(self, generator, dropin_config, tmp_path):
        written = generator.render(dropin_config, tmp_path)
        return written, tmp_path

    def test_kdp_carries_the_hiprtc_kernel_source(self, rendered, dropin_config):
        import json

        _written, out = rendered
        kdp = json.loads(
            (out / "descriptors/scale_add_hiprtc/scale_add_hiprtc.kdp.json").read_text()
        )
        sources = [k["kernel_source"] for k in kdp["kernelDescriptors"]]
        assert {s["kind"] for s in sources} == {"hiprtc_file"}
        assert {s["bundle"] for s in sources} == {"scale_add_sources"}
        # Templates ship VERBATIM: the descriptor is resolved on the target
        # against that kernel's completed metadata, not here.
        assert sources[0]["defines"]["SCALE_ADD_DTYPE"] == "$kernel.dtype"

    def test_bundle_is_staged_beside_the_descriptors(self, rendered):
        _written, out = rendered
        staged = out / "descriptors/scale_add_hiprtc/scale_add_sources"
        assert sorted(p.name for p in staged.iterdir()) == [
            "ScaleAddCommon.h",
            "ScaleAddHiprtc.hip",
        ]

    def test_staged_sources_are_byte_identical(self, rendered, dropin_config):
        _written, out = rendered
        authored = Path(dropin_config.config_dir) / "scale_add_sources"
        staged = out / "descriptors/scale_add_hiprtc/scale_add_sources"
        for name in ("ScaleAddHiprtc.hip", "ScaleAddCommon.h"):
            assert (staged / name).read_bytes() == (authored / name).read_bytes()

    def test_headers_ship_too(self, rendered):
        """No descriptor names `ScaleAddCommon.h`. A bundle that staged only
        the named sources would compile for the author and fail on the target
        at the first #include."""
        written, _out = rendered
        assert (
            "descriptors/scale_add_hiprtc/scale_add_sources/ScaleAddCommon.h" in written
        )

    def test_preview_matches_what_render_writes(self, generator, dropin_config):
        """`--dry-run` is the only view a reviewer gets of a drop-in unit, so a
        preview that omitted the bundle would describe a different artifact
        than the one that ships."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            written = generator.render(dropin_config, Path(tmp))
        assert sorted(generator.preview_files(dropin_config)) == sorted(written)
