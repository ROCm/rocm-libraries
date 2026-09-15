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

import json
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
        # Flatbuffers enum spellings, because a kernel-scoped dtype matcher
        # compares metadata.dtype to EnumNameDataType(<the graph's dtype>). The
        # C++ spellings match nothing, silently, on every graph.
        assert [k.metadata["dtype"] for k in kernels] == ["FLOAT", "HALF"]

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
        _written, out = rendered
        kdp = json.loads(
            (
                out
                / "descriptors/scale_add_hiprtc/scale_add_hiprtc_f32_block64.kdp.json"
            ).read_text()
        )
        source = kdp["kernelDescriptors"][0]["kernel_source"]
        assert source["kind"] == "hiprtc_file"
        assert source["bundle"] == "scale_add_sources"
        # Templates ship VERBATIM: the descriptor is resolved on the target
        # against that kernel's completed metadata, not here.
        assert source["defines"]["SCALE_ADD_DTYPE"] == "$kernel.dtype"

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


class TestOneKdpPerKernel:
    """A drop-in is staged by copying descriptors into an installed tree, so
    the KDP is the unit of staging. With every variant inline in one file
    there is no way to add the second without re-shipping the first."""

    @pytest.fixture
    def rendered(self, generator, dropin_config, tmp_path):
        generator.render(dropin_config, tmp_path)
        return tmp_path / "descriptors/scale_add_hiprtc"

    def test_each_kernel_gets_its_own_kdp_named_after_it(self, rendered):
        assert sorted(p.name for p in rendered.glob("*.kdp.json")) == [
            "scale_add_hiprtc_f16_block256.kdp.json",
            "scale_add_hiprtc_f32_block64.kdp.json",
        ]

    def test_each_kdp_holds_exactly_its_own_kernel(self, rendered):
        by_file = {
            p.name: json.loads(p.read_text()) for p in rendered.glob("*.kdp.json")
        }
        assert [
            k["name"]
            for k in by_file["scale_add_hiprtc_f32_block64.kdp.json"][
                "kernelDescriptors"
            ]
        ] == ["scale_add_hiprtc.f32_block64"]
        assert [
            k["name"]
            for k in by_file["scale_add_hiprtc_f16_block256.kdp.json"][
                "kernelDescriptors"
            ]
        ] == ["scale_add_hiprtc.f16_block256"]

    def test_the_two_kdps_have_distinct_ids(self, rendered):
        """The loader keys the catalog by descriptor id. Two KDPs sharing one
        id are one entry, so staging the second would replace the first rather
        than add to it -- which is the whole point of the split."""
        ids = {json.loads(p.read_text())["id"] for p in rendered.glob("*.kdp.json")}
        assert len(ids) == 2

    def test_everything_else_is_shared_verbatim(self, rendered):
        """They are one pack: same engine, same dispatch, same matcher list,
        same arch. Only id, name and kernelDescriptors may differ."""
        documents = [json.loads(p.read_text()) for p in rendered.glob("*.kdp.json")]
        shared = [
            {k: v for k, v in d.items() if k not in ("id", "name", "kernelDescriptors")}
            for d in documents
        ]
        assert shared[0] == shared[1]

    def test_an_embedded_source_engine_still_gets_one_kdp_per_pack(
        self, generator, scale_add_config, tmp_path
    ):
        """The split is drop-in-only. An `embedded_source` engine ships in the
        provider binary; there is nothing to stage a file at a time."""
        generator.render(scale_add_config, tmp_path)
        emitted = tmp_path / f"descriptors/{scale_add_config.engine.slug}"
        assert [p.name for p in emitted.glob("*.kdp.json")] == [
            f"{scale_add_config.engine.slug}.kdp.json"
        ]

    def test_colliding_kernel_stems_are_refused(self, tmp_path):
        """Kernel names are not unique by construction and the stem collapses
        punctuation, so `fixture.f32` and `fixture_f32` reduce to one file. The
        second would overwrite the first and a variant would vanish."""
        raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
        twin = yaml.safe_load(yaml.safe_dump(raw["packs"][0]["kernels"][0]))
        twin["name"] = "fixture_f32"
        twin["metadata"]["block_size"] = 128
        raw["packs"][0]["kernels"].append(twin)
        path = tmp_path / "colliding_stems.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="same output file"):
            load_config(path)


class TestNoProvenance:
    """`provenance` is an extension the runtime loader does not parse: it logs
    `extension key 'provenance' ... ignoring it` once per KDP, in exactly the
    log an author reads to find the one LOG_ERROR that means their set was
    dropped."""

    def test_a_dropin_kdp_carries_none(self, generator, dropin_config, tmp_path):
        generator.render(dropin_config, tmp_path)
        emitted = tmp_path / "descriptors/scale_add_hiprtc"
        for path in emitted.glob("*.kdp.json"):
            assert "provenance" not in json.loads(path.read_text())

    def test_a_declared_specialization_block_is_still_not_emitted(
        self, generator, tmp_path
    ):
        """The removal is keyed on the dialect, not on the declaration's
        absence: `valid.yaml` declares a full contract and still ships none."""
        config = load_fixture("valid.yaml")
        assert config.specialization
        generator.render(config, tmp_path)
        path = tmp_path / "descriptors/fixture/fixture_f32.kdp.json"
        assert "provenance" not in json.loads(path.read_text())

    def test_a_dropin_config_need_not_declare_one_at_all(
        self, generator, dropin_config, tmp_path
    ):
        """Presence is enforced at emission, so dropping the block from the
        emitted document is what makes the config key optional."""
        assert not dropin_config.specialization
        generator.render(dropin_config, tmp_path)

    def test_a_packaged_kdp_still_carries_it(
        self, generator, gfx950_attention_dense_config, tmp_path
    ):
        """The packaged and rocKE paths check a received binary against the
        shipped contract, which is the only thing they have. B.3 must not
        reach them."""
        generator.render(gfx950_attention_dense_config, tmp_path)
        config = gfx950_attention_dense_config
        emitted = tmp_path / config.descriptor_dir
        documents = [json.loads(p.read_text()) for p in emitted.glob("*.kdp.json")]
        assert documents
        for document in documents:
            assert document["provenance"]["specialization_contract"]["consumers"]


class TestNativeSymbolNamespace:
    """G1: the namespace was derived from the engine name, so a set reusing an
    installed pack's symbols had to author the installed engine's name -- and
    therefore hash to its id."""

    def test_the_override_decouples_symbols_from_the_engine_name(self):
        config = load_fixture("installed_symbols_new_name.yaml")
        assert config.engine.name == "hipkernel:FixtureHiprtcDropin"
        assert config.native_symbol_namespace == "hipkernel.fixture"
        assert config.graph_match_symbol == "hipkernel.fixture.graph_match"
        assert config.dispatch_symbol == "hipkernel.fixture.dispatch"
        assert config.score_symbol == "hipkernel.fixture.score"
        assert config.kernel_match_symbol == "hipkernel.fixture.kernel_match"

    def test_the_emitted_descriptors_carry_both(self, generator, tmp_path):
        """The negative B.4 names: the installed symbols AND the new engine
        name, in one run, with no hand edit between them."""
        config = load_fixture("installed_symbols_new_name.yaml")
        generator.render(config, tmp_path)
        emitted = tmp_path / "descriptors/fixture_hiprtc_dropin"
        ued = json.loads((emitted / "fixture_hiprtc_dropin.ued.json").read_text())
        udd = json.loads((emitted / "fixture_hiprtc_dropin.udd.json").read_text())
        assert ued["name"] == "hipkernel:FixtureHiprtcDropin"
        assert ued["graph_match"] == {"native": "hipkernel.fixture.graph_match"}
        assert udd["dispatch_symbol"] == "hipkernel.fixture.dispatch"

    def test_omitting_it_derives_from_the_name_as_before(self):
        config = load_fixture("valid.yaml")
        assert config.native_symbol_namespace == "hipkernel.fixture"

    def test_a_malformed_namespace_is_refused(self, tmp_path):
        """The loader pre-flights every match/dispatch/score symbol and drops
        the whole engine on a miss, with one log line."""
        raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
        raw["engine"]["native_symbol_namespace"] = "hipkernel:fixture"
        path = tmp_path / "bad_namespace.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="dotted symbol prefix"):
            load_config(path)


class TestSinglePackDiscriminatorWarning:
    """Review M3. `build_operation_umd` returned None for a single-pack engine
    with no warning, so a drop-in silently claimed every operation its reused
    `graph_match` admits -- the pointwise pack's MUL and SUB graphs included.

    The condition, verbatim and shared with `hiprtc-mining.md`:
    `generator.SINGLE_PACK_DISCRIMINATOR_RULE`."""

    def test_a_single_pack_engine_warns_with_the_remedy(
        self, generator, dropin_config, tmp_path, capsys
    ):
        generator.render(dropin_config, tmp_path)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "hipkernel:ScaleAddHiprtc" in out
        # The consequence...
        assert "claim every operation its `graph_match` admits" in out
        # ...and the remedy, which is the actionable half.
        assert "operation_is_<op>" in out
        assert "matchers" in out

    def test_pack_discriminates_silences_it(self, generator, tmp_path, capsys):
        """The same single-pack shape, opted out. The fact is not derivable --
        the shipped conv pack self-discriminates inside native code this
        generator cannot read -- so the author is the only source for it."""
        config = load_fixture("pack_discriminates.yaml")
        assert config.engine.pack_discriminates
        generator.render(config, tmp_path)
        assert "WARNING" not in capsys.readouterr().out

    def test_it_still_emits_no_graph_scoped_umd_either_way(self, generator, tmp_path):
        """The warning changes what is SAID, never what is emitted:
        TestConvFwdPack.cpp asserts zero graph-scoped matchers for this shape,
        and §5 rejected both refusing and deriving a discriminator."""
        config = load_fixture("pack_discriminates.yaml")
        written = generator.render(config, tmp_path)
        assert not [f for f in written if "operation_is_" in f]
        kdp = json.loads(
            (tmp_path / "descriptors/fixture/fixture_f32.kdp.json").read_text()
        )
        assert len(kdp["matchers"]) == 1

    def test_a_multi_pack_engine_never_warns(
        self, generator, binary_ops_config, tmp_path, capsys
    ):
        generator.render(binary_ops_config, tmp_path)
        assert "WARNING" not in capsys.readouterr().out

    def test_pack_discriminates_is_refused_on_a_multi_pack_engine(self):
        with pytest.raises(ConfigError, match="pack_discriminates"):
            load_fixture("multi_pack_discriminates.yaml")
