# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/config_loader.py: the happy path over both worked-example
configs, the five pre-mint loader-mirroring checks, deprecated- and duplicate-key
rejection, and the kernel_source_kind rejections."""

import pytest
import yaml

from codegen.config_loader import ConfigError, load_config
from codegen.models import KernelSource
from tests.helpers import (
    make_engine,
    make_kernel,
    make_kmd_field,
    make_minimal_config,
    make_pack,
)


class TestLoadHappyPath:
    def test_scale_add_loads(self, scale_add_config):
        assert scale_add_config.engine.name == "hipkernel:ScaleAdd"
        assert len(scale_add_config.packs) == 1
        assert scale_add_config.packs[0].kernels[0].metadata["block_size"] == 64

    def test_binary_ops_loads(self, binary_ops_config):
        assert binary_ops_config.engine.name == "hipkernel:BinaryOps"
        assert len(binary_ops_config.packs) == 2
        assert binary_ops_config.is_multi_pack

    def test_scale_add_is_single_pack(self, scale_add_config):
        assert not scale_add_config.is_multi_pack


class TestEngineNameCheck:
    """Pre-mint check #1: engine.name must be scoped namespace:local."""

    def test_unscoped_name_rejected(self):
        from codegen.config_loader import _check_engine_name_scoped

        config = make_minimal_config(engine=make_engine(name="pointwise"))
        with pytest.raises(ConfigError, match="scoped"):
            _check_engine_name_scoped(config)

    def test_scoped_name_accepted(self):
        from codegen.config_loader import _check_engine_name_scoped

        config = make_minimal_config(engine=make_engine(name="hipkernel:Pointwise"))
        _check_engine_name_scoped(config)  # does not raise

    def test_invalid_heuristic_value_rejected(self):
        from codegen.config_loader import _check_engine_name_scoped

        config = make_minimal_config(engine=make_engine(heuristic="bogus"))
        with pytest.raises(ConfigError, match="heuristic"):
            _check_engine_name_scoped(config)

    def test_load_config_rejects_unscoped_name(self, tmp_path):
        raw = {
            "engine": {"name": "unscoped"},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "K.cpp",
                                "entry_point": "K",
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="scoped"):
            load_config(path)


class TestKnobsIntTypedCheck:
    """Pre-mint check #2: every knob names a declared, int-typed KMD field."""

    def test_undeclared_knob_rejected(self):
        from codegen.config_loader import _check_knobs_int_typed

        config = make_minimal_config(engine=make_engine(knobs=["nonexistent"]))
        with pytest.raises(ConfigError, match="no kmd_fields entry declares"):
            _check_knobs_int_typed(config)

    def test_non_int_knob_rejected(self):
        from codegen.config_loader import _check_knobs_int_typed

        config = make_minimal_config(
            kmd_fields=[
                make_kmd_field(name="dtype", type="string", default_value=None)
            ],
            engine=make_engine(knobs=["dtype"]),
        )
        with pytest.raises(ConfigError, match="int-typed"):
            _check_knobs_int_typed(config)

    def test_int_typed_knob_accepted(self):
        from codegen.config_loader import _check_knobs_int_typed

        config = make_minimal_config(engine=make_engine(knobs=["block_size"]))
        _check_knobs_int_typed(config)  # does not raise


class TestKernelMetadataAgainstKmdCheck:
    """Pre-mint check #3: kernel metadata type-checks against the KMD, no
    mandatory field omitted."""

    def test_undeclared_metadata_key_rejected(self):
        from codegen.config_loader import _check_kernel_metadata_against_kmd

        kernel = make_kernel(metadata={"block_size": 64, "bogus_field": 1})
        config = make_minimal_config(packs=[make_pack(kernels=[kernel])])
        with pytest.raises(ConfigError, match="no kmd_fields entry declares"):
            _check_kernel_metadata_against_kmd(config)

    def test_omitted_mandatory_field_rejected(self):
        from codegen.config_loader import _check_kernel_metadata_against_kmd

        # 'dtype' has no default_value in make_minimal_config's kmd_fields, so it
        # is mandatory; omit it from metadata.
        kernel = make_kernel(metadata={"block_size": 64})
        config = make_minimal_config(packs=[make_pack(kernels=[kernel])])
        with pytest.raises(ConfigError, match="omits mandatory metadata field"):
            _check_kernel_metadata_against_kmd(config)

    def test_wrong_type_metadata_rejected(self):
        from codegen.config_loader import _check_kernel_metadata_against_kmd

        kernel = make_kernel(metadata={"block_size": "sixty-four", "dtype": "FLOAT"})
        config = make_minimal_config(packs=[make_pack(kernels=[kernel])])
        with pytest.raises(
            ConfigError, match="does not match its declared kmd_fields type"
        ):
            _check_kernel_metadata_against_kmd(config)

    def test_valid_metadata_accepted(self):
        from codegen.config_loader import _check_kernel_metadata_against_kmd

        config = make_minimal_config()
        _check_kernel_metadata_against_kmd(config)  # does not raise


class TestKernelArchSubsetOfPackCheck:
    """Pre-mint check #4: a kernel's arch must be a subset of its pack's."""

    def test_kernel_arch_reaching_past_pack_rejected(self):
        from codegen.config_loader import _check_kernel_arch_subset_of_pack

        kernel = make_kernel(arch=["gfx950"])
        pack = make_pack(arch=["gfx942"], kernels=[kernel])
        config = make_minimal_config(packs=[pack])
        with pytest.raises(ConfigError, match="reaches past the pack's arch"):
            _check_kernel_arch_subset_of_pack(config)

    def test_kernel_arch_subset_accepted(self):
        from codegen.config_loader import _check_kernel_arch_subset_of_pack

        kernel = make_kernel(arch=["gfx942"])
        pack = make_pack(arch=["gfx942", "gfx950"], kernels=[kernel])
        config = make_minimal_config(packs=[pack])
        _check_kernel_arch_subset_of_pack(config)  # does not raise

    def test_empty_kernel_arch_inherits_pack(self):
        from codegen.config_loader import _check_kernel_arch_subset_of_pack

        kernel = make_kernel(arch=[])
        pack = make_pack(arch=["gfx942"], kernels=[kernel])
        config = make_minimal_config(packs=[pack])
        _check_kernel_arch_subset_of_pack(config)  # does not raise

    def test_empty_pack_arch_covers_everything(self):
        from codegen.config_loader import _check_kernel_arch_subset_of_pack

        kernel = make_kernel(arch=["gfx942"])
        pack = make_pack(arch=[], kernels=[kernel])
        config = make_minimal_config(packs=[pack])
        _check_kernel_arch_subset_of_pack(config)  # does not raise


class TestArchShapeCheck:
    """Pre-mint check #5: arch entries are plausible gfx-prefixed base ids
    (error), and unrecognized-but-well-formed ids warn rather than error."""

    @pytest.mark.parametrize(
        "bad_arch", ["GFX942", " gfx942", "gfx942:sramecc+", "notgfx"]
    )
    def test_malformed_arch_rejected(self, bad_arch):
        from codegen.config_loader import _check_arch_shape

        pack = make_pack(arch=[bad_arch])
        config = make_minimal_config(packs=[pack])
        with pytest.raises(ConfigError, match="not a plausible"):
            _check_arch_shape(config)

    def test_well_formed_unrecognized_arch_warns_not_errors(self):
        from codegen.config_loader import _check_arch_shape

        # gfx94 is well-formed (matches the gfx+lowercase-alnum shape) but not a
        # real device id -- exactly the documented gfx94/gfx942 typo trap.
        pack = make_pack(arch=["gfx94"])
        config = make_minimal_config(packs=[pack])
        with pytest.warns(UserWarning, match="well-formed but not a recognized"):
            warnings_out = _check_arch_shape(config)
        assert len(warnings_out) == 1

    def test_recognized_arch_produces_no_warning(self):
        from codegen.config_loader import _check_arch_shape

        pack = make_pack(arch=["gfx942"])
        config = make_minimal_config(packs=[pack])
        warnings_out = _check_arch_shape(config)
        assert warnings_out == []


class TestKernelSourceKindRejection:
    """Each rejection must name the DIALECT, not merely 'unsupported': the common
    mistake is a real kind under the wrong dialect, whose fix is a one-line ``dialect:``
    change."""

    def test_hsaco_file_rejected_naming_prerequisite(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="hsaco_file")
        with pytest.raises(ConfigError, match="supportsSourceKind"):
            _check_kernel_source_kind_implemented(config)

    def test_kpack_rejected_as_produced_not_authored(self):
        """kpack is what hkp_pack WRITES; authoring it is a second source of truth for
        library/toc_key/symbol/sha256 that can disagree with the archive it
        describes."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="kpack")
        with pytest.raises(ConfigError, match="PRODUCED kind, never an authored one"):
            _check_kernel_source_kind_implemented(config)

    def test_rocke_builder_rejected_pointing_at_the_packaged_spelling(self):
        """The runtime enum spelling parses and nothing dispatches it: a rocKE kernel
        reaches the loader already lowered to kpack, so the authored spelling is
        'rocke'."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="rocke_builder")
        with pytest.raises(ConfigError, match="never reaches the runtime as rocKE"):
            _check_kernel_source_kind_implemented(config)

    def test_rocke_under_direct_load_names_the_right_dialect(self):
        """'rocke' is real, just not in direct_load."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="rocke")
        with pytest.raises(ConfigError, match="belongs to dialect 'packaged'"):
            _check_kernel_source_kind_implemented(config)

    def test_embedded_source_under_packaged_names_the_right_dialect(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented
        from codegen.models import DIALECT_PACKAGED

        config = make_minimal_config(
            kernel_source_kind="embedded_source", dialect=DIALECT_PACKAGED
        )
        with pytest.raises(ConfigError, match="belongs to dialect 'direct_load'"):
            _check_kernel_source_kind_implemented(config)

    def test_embedded_source_accepted(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config()
        _check_kernel_source_kind_implemented(config)  # does not raise

    def test_rocke_accepted_under_packaged(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented
        from codegen.models import DIALECT_PACKAGED

        kernel = make_kernel(
            kernel_source=KernelSource(
                kind="rocke",
                source="kernels/gfx950/attention_dense.py",
                builder="build_attention_dense",
                spec={"batch": 1},
            )
        )
        config = make_minimal_config(
            kernel_source_kind="rocke",
            dialect=DIALECT_PACKAGED,
            packs=[make_pack(kernels=[kernel], arch=["gfx950"])],
        )
        _check_kernel_source_kind_implemented(config)  # does not raise

    def test_per_kernel_kind_also_checked(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented

        kernel = make_kernel(
            kernel_source=KernelSource(
                kind="hsaco_file", source_file="X.cpp", entry_point="X"
            )
        )
        config = make_minimal_config(packs=[make_pack(kernels=[kernel])])
        with pytest.raises(ConfigError, match="supportsSourceKind"):
            _check_kernel_source_kind_implemented(config)


class TestPackDiscriminatorsCheck:
    def test_multi_pack_missing_discriminator_rejected(self):
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(
            packs=[
                make_pack(name="a", discriminator=""),
                make_pack(name="b", discriminator="b"),
            ]
        )
        with pytest.raises(ConfigError, match="discriminator"):
            _check_pack_discriminators(config)

    def test_single_pack_with_discriminator_rejected(self):
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(packs=[make_pack(discriminator="add")])
        with pytest.raises(ConfigError, match="only one pack"):
            _check_pack_discriminators(config)

    def test_multi_pack_duplicate_discriminators_rejected(self):
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(
            packs=[
                make_pack(name="a", discriminator="x"),
                make_pack(name="b", discriminator="x"),
            ]
        )
        with pytest.raises(ConfigError, match="duplicate discriminators"):
            _check_pack_discriminators(config)

    def test_duplicate_pack_names_rejected(self):
        """A pack name keys its descriptor id AND its output filename, so two packs
        sharing a name collide twice: same pack id, and the second
        `<slug>_<pack>.kdp.json` overwrites."""
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(
            packs=[
                make_pack(name="same", discriminator="x"),
                make_pack(name="same", discriminator="y"),
            ]
        )
        with pytest.raises(ConfigError, match="duplicate names"):
            _check_pack_discriminators(config)

    def test_no_packs_rejected(self):
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(packs=[])
        with pytest.raises(ConfigError, match="at least one pack"):
            _check_pack_discriminators(config)

    def test_pack_with_no_kernels_rejected(self):
        from codegen.config_loader import _check_pack_discriminators

        config = make_minimal_config(packs=[make_pack(kernels=[])])
        with pytest.raises(ConfigError, match="no kernels"):
            _check_pack_discriminators(config)


class TestEmittedIdentifierShape:
    """Names this config splices into generated C++ IDENTIFIERS.

    `native.cpp.j2` builds `<NAME>_FIELD` from every kmd field name and
    `<NAME>_MATCHER_SYMBOL` plus `<name>OperationMatches` from every pack discriminator,
    so a name outside the identifier shape is a syntax error in a file nobody edited.
    """

    def test_a_valid_config_is_accepted(self):
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            kmd_fields=[
                make_kmd_field(name="head_dim"),
                make_kmd_field(name="block_size"),
            ],
            packs=[
                make_pack(name="a", discriminator="add_fast"),
                make_pack(name="b", discriminator="_scale2"),
            ],
        )
        _check_emitted_identifiers(config)  # does not raise

    def test_an_ordinary_engine_local_name_is_accepted(self):
        """Both shipped spellings are ordinary: `ConvFwd` (direct-load) and
        `gfx950_attention_dense` (packaged) derive different identifiers from the same
        rule."""
        from codegen.config_loader import _check_emitted_identifiers

        for name in ("hipkernel:ConvFwd", "hkp_example:gfx950_attention_dense"):
            config = make_minimal_config(engine=make_engine(name=name))
            _check_emitted_identifiers(config)  # does not raise

    def test_a_kebab_case_engine_local_name_is_accepted(self):
        """Holding the SLUG to the C++ identifier rule would refuse kebab-case, which
        the tool converts on purpose: `_to_pascal_case` splits on `-` and folds it away,
        so every identifier `attn-v2` derives is valid and only the slug keeps the
        hyphen."""
        from codegen.config_loader import _check_emitted_identifiers

        engine = make_engine(name="hipkernel:attn-v2")
        assert (engine.slug, engine.pascal_name, engine.camel_name) == (
            "attn-v2",
            "AttnV2",
            "attnV2",
        )
        _check_emitted_identifiers(make_minimal_config(engine=engine))  # no raise

    def test_a_local_name_leading_with_a_hyphen_is_rejected(self):
        """`_to_pascal_case` drops the empty leading part, so `-foo` derives a good
        `Foo` while the slug keeps the hyphen and names a directory that reads as a
        command-line option."""
        from codegen.config_loader import _check_emitted_identifiers

        engine = make_engine(name="hipkernel:-foo")
        assert (engine.pascal_name, engine.slug) == ("Foo", "-foo")
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(make_minimal_config(engine=engine))
        message = str(excinfo.value)
        assert "path stem" in message, message
        assert "-foo" in message, message

    def test_a_dotted_engine_local_name_is_rejected(self):
        """`hipkernel:attn.v2` emits `class Attn.v2DispatchHandler`: `_to_pascal_case`
        splits on `_` and `-` only, so the dot survives into every identifier and file
        stem."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(engine=make_engine(name="hipkernel:attn.v2"))
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        message = str(excinfo.value)
        assert "attn.v2" in message, message
        assert "Attn.v2" in message, message

    def test_an_engine_local_name_starting_with_a_digit_is_rejected(self):
        """`hipkernel:2dConv` emits `class 2dConvDispatchHandler`; nothing uppercases a
        leading digit away."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(engine=make_engine(name="hipkernel:2dConv"))
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        assert "2dConv" in str(excinfo.value)

    def test_an_engine_local_name_of_dots_is_rejected(self):
        """The path-shaped half of the same defect: the slug names this bundle's
        descriptor directory and `..` names its parent. The identifier rule is reached
        first, so the stem rule is asserted directly -- it is what still refuses `..` if
        the spellings change."""
        from codegen.config_loader import _check_emitted_identifiers
        from codegen.models import PATH_STEM_PATTERN

        engine = make_engine(name="hipkernel:..")
        assert engine.slug == ".."
        assert not PATH_STEM_PATTERN.match(engine.slug)
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(make_minimal_config(engine=engine))
        assert "engine.name" in str(excinfo.value)

    def test_load_config_runs_the_engine_name_check(self, tmp_path):
        """`_check_engine_name_scoped` accepts the name, so nothing else stops it."""
        raw = {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:attn.v2"},
            "kmd_fields": [{"name": "head_dim", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "k.hip",
                                "entry_point": "k",
                            },
                            "metadata": {"head_dim": 64},
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        assert "attn.v2" in str(excinfo.value)

    def test_a_hyphenated_field_name_is_rejected(self):
        """`head-dim` emits `constexpr std::string_view HEAD-DIM_FIELD`."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(kmd_fields=[make_kmd_field(name="head-dim")])
        with pytest.raises(ConfigError, match="head-dim"):
            _check_emitted_identifiers(config)

    def test_two_field_names_differing_only_in_case_are_rejected_naming_both(self):
        """`dtype` and `dType` both uppercase to `DTYPE_FIELD`. Each is a good
        identifier alone, so the diagnostic must carry both spellings: `DTYPE` names
        neither field."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            kmd_fields=[
                make_kmd_field(name="dtype", type="string", default_value="FLOAT"),
                make_kmd_field(name="dType", type="string", default_value="FLOAT"),
            ]
        )
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        message = str(excinfo.value)
        assert "'dtype'" in message and "'dType'" in message, message

    def test_two_fields_spelled_identically_are_rejected(self):
        """The same name twice emits `BLOCK_SIZE_FIELD` twice: a check comparing how the
        two entries describe themselves sees one description and lets the second claim
        it."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            kmd_fields=[
                make_kmd_field(name="block_size", type="int", default_value=64),
                make_kmd_field(name="block_size", type="int", default_value=128),
            ]
        )
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        assert "BLOCK_SIZE_FIELD" in str(excinfo.value), str(excinfo.value)

    def test_two_discriminators_spelled_identically_are_rejected(self):
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="left", discriminator="add", kernels=[make_kernel()]),
                make_pack(name="right", discriminator="add", kernels=[make_kernel()]),
            ]
        )
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        assert "ADD_MATCHER_SYMBOL" in str(excinfo.value), str(excinfo.value)

    def test_a_hyphenated_discriminator_is_rejected(self):
        """`add-fast` emits `ADD-FAST_MATCHER_SYMBOL` and `add-fastOperationMatches`."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="a", discriminator="add-fast"),
                make_pack(name="b", discriminator="mul"),
            ]
        )
        with pytest.raises(ConfigError, match="add-fast"):
            _check_emitted_identifiers(config)

    def test_two_discriminators_differing_only_in_case_are_rejected_naming_both(self):
        """`add` and `Add` both uppercase to `ADD_MATCHER_SYMBOL`, and
        `_check_pack_discriminators` dedups by EXACT match, so both survive it."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="a", discriminator="add"),
                make_pack(name="b", discriminator="Add"),
            ]
        )
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        message = str(excinfo.value)
        assert "'add'" in message and "'Add'" in message, message

    def test_a_discriminator_on_a_reserved_stem_is_rejected(self):
        """`graph` and `kernel` are taken by the template's fixed constants, emitted
        outside the per-pack loop, so the collision is invisible in the YAML."""
        from codegen.config_loader import _check_emitted_identifiers

        for reserved in ("graph", "Kernel"):
            config = make_minimal_config(
                packs=[
                    make_pack(name="a", discriminator=reserved),
                    make_pack(name="b", discriminator="mul"),
                ]
            )
            with pytest.raises(ConfigError) as excinfo:
                _check_emitted_identifiers(config)
            assert f"'{reserved}'" in str(excinfo.value), reserved

    def test_a_field_and_a_discriminator_may_share_a_name(self):
        """`<NAME>_FIELD` and `<NAME>_MATCHER_SYMBOL` are different identifiers; one
        shared map would reject `add` as a metadata field beside an `add` pack."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            kmd_fields=[make_kmd_field(name="add"), make_kmd_field(name="mul")],
            packs=[
                make_pack(name="a", discriminator="add"),
                make_pack(name="b", discriminator="mul"),
            ],
        )
        _check_emitted_identifiers(config)  # does not raise

    def test_the_reserved_stems_are_the_ones_the_template_actually_emits(
        self, template_dir
    ):
        """The reserved list is checked against its source: a fixed
        `<STEM>_MATCHER_SYMBOL` added to the template and not to the tuple reopens the
        hole silently."""
        import re

        from codegen.config_loader import RESERVED_MATCHER_SYMBOL_STEMS

        source = (template_dir / "native.cpp.j2").read_text()
        # Only the LITERAL stems: the per-pack constant interpolates
        # `{{ pack.discriminator.upper() }}` and so cannot match.
        emitted = re.findall(
            r"^constexpr std::string_view ([A-Z0-9_]+)_MATCHER_SYMBOL",
            source,
            re.MULTILINE,
        )
        assert sorted(emitted) == sorted(RESERVED_MATCHER_SYMBOL_STEMS), emitted

    def test_load_config_runs_the_check(self, tmp_path):
        raw = {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "head-dim", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "k.hip",
                                "entry_point": "k",
                            },
                            "metadata": {"head-dim": 64},
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="head-dim"):
            load_config(path)


class TestPackNamesAreSingleFileStems:
    """A pack name is spliced into the KDP's FILENAME, so it is held to the stem rule.

    ``kdp_stem`` builds ``<engine-slug>_<pack-name>``, ``render`` writes it as
    ``<descriptor_dir>/<stem>.kdp.json``, and the same stem is the descriptor's runtime
    ``name``. Kebab-case passes: only the pack's ``discriminator`` becomes a C++ name,
    and it carries its own check.
    """

    def test_a_normal_multi_pack_engine_still_loads(self):
        """``configs/binary_ops.yaml`` names its packs ``add`` and ``max``, so refusing
        this shape would refuse what ships."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="add", discriminator="add"),
                make_pack(name="sub", discriminator="sub"),
            ]
        )
        _check_emitted_identifiers(config)  # does not raise

    def test_a_kebab_case_pack_name_is_accepted(self):
        """``add-fast`` emits ``binary_ops_add-fast.kdp.json``, an ordinary filename,
        and nothing derives a C++ name from a pack name."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="add-fast", discriminator="addFast"),
                make_pack(name="mul", discriminator="mul"),
            ]
        )
        _check_emitted_identifiers(config)  # does not raise

    def test_a_pack_name_that_walks_up_is_rejected(self):
        """``../evil`` puts ``..`` inside the stem of a file this tool writes."""
        from codegen.config_loader import _check_emitted_identifiers
        from codegen.models import PATH_STEM_PATTERN

        config = make_minimal_config(
            packs=[
                make_pack(name="../evil", discriminator="evil"),
                make_pack(name="mul", discriminator="mul"),
            ]
        )
        assert config.kdp_stem(config.packs[0]) == "test_../evil"
        assert not PATH_STEM_PATTERN.match("../evil")
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        message = str(excinfo.value)
        assert "../evil" in message, message
        assert "path stem" in message, message

    def test_a_pack_name_carrying_a_separator_is_rejected(self):
        """A separator makes the stem name a place, not a file. Both spellings, because
        the backslash is a separator on the platform this bundle is usually generated
        for."""
        from codegen.config_loader import _check_emitted_identifiers

        for name in ("add/fast", "add\\fast"):
            config = make_minimal_config(
                packs=[
                    make_pack(name=name, discriminator="addFast"),
                    make_pack(name="mul", discriminator="mul"),
                ]
            )
            with pytest.raises(ConfigError) as excinfo:
                _check_emitted_identifiers(config)
            assert name in str(excinfo.value), name

    def test_an_empty_pack_name_is_rejected(self):
        """``name: ""`` passes the required-key check and emits
        ``binary_ops_.kdp.json``, the stem every empty-named pack of one engine lands on
        together."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="", discriminator="add"),
                make_pack(name="mul", discriminator="mul"),
            ]
        )
        with pytest.raises(ConfigError) as excinfo:
            _check_emitted_identifiers(config)
        assert "path stem" in str(excinfo.value), str(excinfo.value)

    def test_a_pack_name_leading_with_a_hyphen_is_rejected(self):
        """A leading ``-`` reads as an option wherever the filename is passed on a
        command line."""
        from codegen.config_loader import _check_emitted_identifiers

        config = make_minimal_config(
            packs=[
                make_pack(name="-fast", discriminator="fast"),
                make_pack(name="mul", discriminator="mul"),
            ]
        )
        with pytest.raises(ConfigError, match="-fast"):
            _check_emitted_identifiers(config)

    def test_load_config_runs_the_pack_name_check(self, tmp_path):
        raw = {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": name,
                    "discriminator": discriminator,
                    "kernels": [
                        {
                            "name": f"k_{discriminator}",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "k.hip",
                                "entry_point": "k",
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
                for name, discriminator in (("../evil", "evil"), ("mul", "mul"))
            ],
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="path stem"):
            load_config(path)


class TestArchListsAreNormalisedAtLoad:
    """A repeated arch entry is collapsed at load, which settles it for every consumer
    at once; see ``config_loader._unique_arch`` for where a repeat bites."""

    def _load(self, tmp_path, pack_arch, kernel_arch):
        raw = {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": pack_arch,
                    "kernels": [
                        {
                            "name": "k",
                            "arch": kernel_arch,
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
        return load_config(path)

    def test_a_repeated_arch_entry_is_collapsed(self, tmp_path):
        config = self._load(
            tmp_path, ["gfx942", "gfx950", "gfx942"], ["gfx950", "gfx950"]
        )
        assert config.packs[0].arch == ["gfx942", "gfx950"]
        assert config.packs[0].kernels[0].arch == ["gfx950"]

    def test_distinct_entries_are_all_kept_in_authored_order(self, tmp_path):
        """Not a sort and not a truncation: arch order reaches the descriptor bytes, so
        a reordering normalisation would rewrite what ships while membership tests still
        pass."""
        config = self._load(tmp_path, ["gfx950", "gfx942"], ["gfx942"])
        assert config.packs[0].arch == ["gfx950", "gfx942"]
        assert config.packs[0].kernels[0].arch == ["gfx942"]


class TestKernelNameUniquenessIsEngineScoped:
    """The loader collects an engine's packs into ONE `DescriptorSet` by engine id, and
    the de-duplication pass keys on resolved metadata rather than on the name."""

    def _raw(self, left_names, right_names):
        def kernel(name, block_size):
            return {
                "name": name,
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "k.hip",
                    "entry_point": "k",
                },
                "metadata": {"block_size": block_size},
            }

        return {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "left",
                    "discriminator": "left",
                    # Distinct metadata per kernel, so nothing here depends on the
                    # de-duplication pass.
                    "kernels": [
                        kernel(name, 64 + i) for i, name in enumerate(left_names)
                    ],
                },
                {
                    "name": "right",
                    "discriminator": "right",
                    "kernels": [
                        kernel(name, 128 + i) for i, name in enumerate(right_names)
                    ],
                },
            ],
        }

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def test_distinct_names_across_two_packs_still_load(self, tmp_path):
        """The widened check must not reject an ordinary multi-pack engine."""
        config = self._load(tmp_path, self._raw(["a"], ["b"]))
        assert [k.name for pack in config.packs for k in pack.kernels] == ["a", "b"]

    def test_one_name_in_two_packs_is_rejected_naming_both_packs(self, tmp_path):
        with pytest.raises(ConfigError, match="duplicated kernel name") as excinfo:
            self._load(tmp_path, self._raw(["same"], ["same"]))
        message = str(excinfo.value)
        assert "'same'" in message, message
        assert "'left'" in message and "'right'" in message, message

    def test_one_name_twice_in_one_pack_is_still_rejected(self, tmp_path):
        """An engine-wide check that lost the within-pack case would trade one silent
        collision for another."""
        with pytest.raises(ConfigError, match="duplicated kernel name") as excinfo:
            self._load(tmp_path, self._raw(["dup", "dup"], ["b"]))
        assert "'left'" in str(excinfo.value)


class TestDeprecatedKeys:
    def _base_raw(self):
        return {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "K.cpp",
                                "entry_point": "K",
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }

    # Each pattern is text only `_reject_deprecated_keys` produces. Every one of these
    # keys is an unknown key too, so a pattern like "optional" would pass whichever
    # check ran first and say nothing about which one did.
    def test_kmd_field_optional_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["kmd_fields"][0]["optional"] = True
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="MetadataField has no such member"):
            load_config(path)

    def test_kmd_field_optional_key_rejected_even_when_false(self, tmp_path):
        raw = self._base_raw()
        raw["kmd_fields"][0]["optional"] = False
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="MetadataField has no such member"):
            load_config(path)

    def test_kmd_field_default_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["kmd_fields"][0]["default"] = 1
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="the loader spells it default_value"):
            load_config(path)

    def test_top_level_schema_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["schema"] = "hipdnn.ued/v1"
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match=r"tag 'hipdnn\.ued/v1'"):
            load_config(path)

    def test_top_level_descriptor_files_var_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["descriptor_files_var"] = "HKP_DESCRIPTOR_FILES"
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(
            ConfigError, match=r"cmake_descriptor_files\.txt fragment already states"
        ):
            load_config(path)

    def test_top_level_pack_kernels_var_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["pack_kernels_var"] = "HKP_PACK_KERNELS"
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(
            ConfigError, match=r"cmake_target_sources\.txt fragment already states"
        ):
            load_config(path)

    def test_top_level_delegates_to_existing_plan_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["delegates_to_existing_plan"] = False
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(
            ConfigError, match="deleting the line changes nothing about the bundle"
        ):
            load_config(path)

    def test_top_level_delegates_to_existing_plan_key_rejected_when_true(
        self, tmp_path
    ):
        raw = self._base_raw()
        raw["delegates_to_existing_plan"] = True
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(
            ConfigError, match="deleting the line changes nothing about the bundle"
        ):
            load_config(path)


_AUTHORED_HEAD = (
    "authored_subpath: unit\n"
    "engine:\n"
    '  name: "hipkernel:Test"\n'
    "  knobs: [block_size]\n"
    "kmd_fields:\n"
    "  - {name: block_size, type: int, default_value: 64}\n"
    "packs:\n"
    "  - name: p\n"
    "    kernels:\n"
)

_AUTHORED_KERNEL = (
    "      - name: k\n"
    "        kernel_source:\n"
    "          kind: embedded_source\n"
    "          source_file: K.cpp\n"
    "          entry_point: K\n"
    "        metadata:\n"
    "          block_size: 64\n"
)


def _mark_of(text, needle, occurrence=0):
    """0-based ``(line, column)`` of ``needle``'s ``occurrence``-th match, as a
    PyYAML ``Mark`` reports it."""
    index = -1
    for _ in range(occurrence + 1):
        index = text.index(needle, index + 1)
    return text.count("\n", 0, index), index - (text.rfind("\n", 0, index) + 1)


def _parse(text):
    from codegen.config_loader import _DuplicateKeySafeLoader

    return yaml.load(text, Loader=_DuplicateKeySafeLoader)  # nosec B506


class TestRepeatedYamlKeys:
    """Stock PyYAML keeps the last of two equal mapping keys and drops the other
    silently, so the value an author reads first is not the one generated. Merge keys,
    aliases and YAML's scalar resolution stay what stock SafeLoader makes of them."""

    def _load(self, tmp_path, text):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        return load_config(path)

    def _assert_rejected(self, tmp_path, text, key, first, second):
        with pytest.raises(yaml.constructor.ConstructorError) as excinfo:
            self._load(tmp_path, text)
        error = excinfo.value
        assert repr(key) in str(error), str(error)
        assert (error.context_mark.line, error.context_mark.column) == first
        assert (error.problem_mark.line, error.problem_mark.column) == second

    def test_a_top_level_key_declared_twice_is_rejected_at_both_marks(self, tmp_path):
        text = _AUTHORED_HEAD + _AUTHORED_KERNEL + "authored_subpath: other\n"
        self._assert_rejected(
            tmp_path,
            text,
            "authored_subpath",
            _mark_of(text, "authored_subpath: unit"),
            _mark_of(text, "authored_subpath: other"),
        )

    def test_a_nested_key_declared_twice_is_rejected_at_both_marks(self, tmp_path):
        text = _AUTHORED_HEAD + _AUTHORED_KERNEL + "          block_size: 32\n"
        self._assert_rejected(
            tmp_path,
            text,
            "block_size",
            _mark_of(text, "block_size: 64"),
            _mark_of(text, "block_size: 32"),
        )

    def test_a_key_repeated_inside_a_merge_source_is_rejected(self, tmp_path):
        """The inline source is never constructed as a mapping of its own: only
        flattening into its parent ever reads it."""
        text = _AUTHORED_HEAD + (
            "      - name: k\n"
            "        kernel_source:\n"
            "          <<: {kind: embedded_source, source_file: K.cpp, "
            "source_file: J.cpp}\n"
            "          entry_point: K\n"
            "        metadata:\n"
            "          block_size: 64\n"
        )
        self._assert_rejected(
            tmp_path,
            text,
            "source_file",
            _mark_of(text, "source_file: K.cpp"),
            _mark_of(text, "source_file: J.cpp"),
        )

    def test_a_repeated_merge_key_is_rejected(self, tmp_path):
        text = _AUTHORED_HEAD + (
            "      - name: k1\n"
            "        kernel_source: &src\n"
            "          kind: embedded_source\n"
            "          source_file: K.cpp\n"
            "          entry_point: K\n"
            "        metadata:\n"
            "          block_size: 64\n"
            "      - name: k2\n"
            "        kernel_source:\n"
            "          <<: *src\n"
            "          <<: *src\n"
            "        metadata:\n"
            "          block_size: 32\n"
        )
        self._assert_rejected(
            tmp_path,
            text,
            "<<",
            _mark_of(text, "<<: *src"),
            _mark_of(text, "<<: *src", occurrence=1),
        )

    def test_merges_and_reused_aliases_load_as_stock_yaml(self, tmp_path):
        """k2 overrides a merged key explicitly; k3 merges ``[*k2src, *src]``, where
        the first source wins, and reuses k2src after it was flattened with both its
        inherited and its own ``entry_point``."""
        text = _AUTHORED_HEAD + (
            "      - name: k1\n"
            "        kernel_source: &src\n"
            "          kind: embedded_source\n"
            "          source_file: K.cpp\n"
            "          entry_point: K\n"
            "        metadata: &md\n"
            "          block_size: 64\n"
            "      - name: k2\n"
            "        kernel_source: &k2src\n"
            "          <<: *src\n"
            "          entry_point: K2\n"
            "        metadata:\n"
            "          block_size: 128\n"
            "      - name: k3\n"
            "        kernel_source:\n"
            "          <<: [*k2src, *src]\n"
            "        metadata:\n"
            "          <<: *md\n"
            "          block_size: 32\n"
        )
        kernels = self._load(tmp_path, text).packs[0].kernels
        assert [
            (k.name, k.kernel_source.source_file, k.kernel_source.entry_point)
            for k in kernels
        ] == [("k1", "K.cpp", "K"), ("k2", "K.cpp", "K2"), ("k3", "K.cpp", "K2")]
        assert [k.metadata["block_size"] for k in kernels] == [64, 128, 32]
        assert _parse(text) == yaml.safe_load(text)

    def test_quoted_and_unquoted_off_are_different_keys_and_values(self):
        text = 'off: 1\n"off": 2\nflag: off\nname: "off"\n'
        parsed = _parse(text)
        assert parsed == {False: 1, "off": 2, "flag": False, "name": "off"}
        assert parsed == yaml.safe_load(text)

    def test_two_spellings_yaml_resolves_to_one_key_are_rejected(self):
        """``off`` and ``no`` are both the boolean false, so stock YAML keeps one."""
        text = "off: 1\nno: 2\n"
        with pytest.raises(yaml.constructor.ConstructorError) as excinfo:
            _parse(text)
        error = excinfo.value
        assert (error.context_mark.line, error.context_mark.column) == (0, 0)
        assert (error.problem_mark.line, error.problem_mark.column) == (1, 0)

    @pytest.mark.parametrize(
        "entry",
        [
            "extra: !!python/object/apply:os.getcwd []\n",
            "? !!python/object/apply:os.getcwd []\n: 1\n",
        ],
        ids=["value", "key"],
    )
    def test_an_unsafe_tag_is_still_rejected(self, tmp_path, entry):
        text = _AUTHORED_HEAD + _AUTHORED_KERNEL + entry
        with pytest.raises(yaml.constructor.ConstructorError) as excinfo:
            self._load(tmp_path, text)
        mark = excinfo.value.problem_mark
        assert (mark.line, mark.column) == _mark_of(text, "!!python")


class TestShippedExampleConfigsLoad:
    """Every config under `configs/` is a worked example a reader copies, so a retired
    key they still set would make each copy a config the loader refuses."""

    def test_every_shipped_config_loads(self, configs_dir):
        paths = sorted(configs_dir.glob("*.yaml"))
        assert len(paths) == 5
        for path in paths:
            assert load_config(path).engine.name


class TestBehaviorNotesVocabulary:
    def test_unknown_behavior_note_rejected(self, tmp_path):
        raw = TestDeprecatedKeys()._base_raw()
        raw["engine"]["behavior_notes"] = ["not_a_real_note"]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="closed vocabulary"):
            load_config(path)

    def test_runtime_compilation_accepted(self, tmp_path):
        raw = TestDeprecatedKeys()._base_raw()
        raw["engine"]["behavior_notes"] = ["runtime_compilation"]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        config = load_config(path)
        assert config.engine.behavior_notes == ["runtime_compilation"]


class TestDirectLoadAuthoredSubpath:
    """Each set under ``test_descriptors/`` is its own pack target, walked by directory
    and read by a different binary, so a bundle naming none has no shard to land in and
    a default would file it in one no suite reads."""

    def test_a_direct_load_config_without_authored_subpath_is_rejected(self, tmp_path):
        raw = TestDeprecatedKeys()._base_raw()
        del raw["authored_subpath"]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError) as excinfo:
            load_config(path)
        message = str(excinfo.value)
        assert "authored_subpath" in message
        for value in ("shared", "unit", "integration", "archive_fixture"):
            assert value in message, message


class TestPackagedAuthoredSubpathIsContained:
    """A packaged bundle's subpath stays under the ``descriptors/`` root it names.

    ``descriptor_dir`` joins the two as a STRING, so a ``..`` component survives the
    join and a subpath that walks up writes the whole bundle above ``--output-dir``. An
    absolute subpath does not join either: ``C:/x`` turns the result drive-relative.
    """

    def _raw(self, subpath):
        raw = {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx950"],
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "rocke",
                                "source": "kernels/gfx950/attention_dense.py",
                                "builder": "build_attention_dense",
                                "spec": {"seqlen_q": 256},
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }
        if subpath is not None:
            raw["authored_subpath"] = subpath
        return raw

    def _load(self, tmp_path, subpath):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(self._raw(subpath)))
        return load_config(path)

    def test_the_shipped_subpath_still_loads(self, tmp_path):
        """The control, taken from configs/gfx950_attention_dense.yaml."""
        config = self._load(tmp_path, "rocKE/gfx950_attention_dense")
        assert config.descriptor_dir == "descriptors/rocKE/gfx950_attention_dense"

    def test_an_omitted_subpath_still_falls_back_to_kind_over_slug(self, tmp_path):
        """The packaged dialect has a default, unlike direct-load, and the containment
        rule must not turn it into a requirement."""
        config = self._load(tmp_path, None)
        assert config.descriptor_dir == "descriptors/rocke/test"

    def test_a_subpath_that_walks_out_of_the_descriptor_root_is_rejected(
        self, tmp_path
    ):
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "../../../../g2demo_pwned")
        message = str(excinfo.value)
        assert "authored_subpath" in message, message
        assert "g2demo_pwned" in message, message

    def test_a_subpath_that_descends_before_walking_out_is_rejected(self, tmp_path):
        """``rocKE/../../x`` spends one component going down and two coming back up, so
        a check looking only at the first component would pass it."""
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "rocKE/../../g2demo_pwned")
        assert "g2demo_pwned" in str(excinfo.value)

    def test_a_drive_qualified_subpath_is_rejected(self, tmp_path):
        """``C:/x`` never reaches ``..``: it makes the join drive-relative."""
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "C:/g2demo_abs")
        assert "g2demo_abs" in str(excinfo.value)

    def test_a_drive_relative_subpath_is_rejected(self, tmp_path):
        """``C:x`` has no root at all: a path resolved against a drive's own current
        directory, which no caller named."""
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "C:g2demo_abs")
        assert "g2demo_abs" in str(excinfo.value)

    def test_a_rooted_subpath_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "/g2demo_abs")
        assert "g2demo_abs" in str(excinfo.value)

    def test_a_backslash_separated_escape_is_rejected(self, tmp_path):
        """The config is read on one platform and generated on another, so a Windows
        separator has to escape on the reader too."""
        with pytest.raises(ConfigError) as excinfo:
            self._load(tmp_path, "..\\..\\g2demo_pwned")
        assert "g2demo_pwned" in str(excinfo.value)


class TestPackKernelDefaults:
    """A pack may hoist what every kernel repeats; a kernel overrides by restating.
    Without it a generated variant set restates `kind`, `source`, `builder` once per
    kernel."""

    def _raw(self, **pack_extra):
        return {
            # `rocke` is a packaged-dialect kind; the loader cross-checks the two.
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/test",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernels": [
                        {
                            "name": "k1",
                            "kernel_source": {"spec": {"seqlen_q": 256}},
                            "metadata": {"block_size": 64},
                        },
                        {
                            "name": "k2",
                            "kernel_source": {"spec": {"seqlen_q": 512}},
                            "metadata": {"block_size": 64},
                        },
                    ],
                    **pack_extra,
                }
            ],
        }

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def test_defaults_supply_kernel_source_and_spec(self, tmp_path):
        raw = self._raw(
            kernel_defaults={
                "kind": "rocke",
                "source": "kernels/gfx942/attention_dense.py",
                "builder": "build_attention_dense",
                "spec": {"head_size": 128, "dtype": "bf16"},
            }
        )
        config = self._load(tmp_path, raw)
        ks = config.packs[0].kernels
        assert [k.kernel_source.kind for k in ks] == ["rocke", "rocke"]
        assert [k.kernel_source.builder for k in ks] == ["build_attention_dense"] * 2
        assert [k.kernel_source.spec["head_size"] for k in ks] == [128, 128]
        assert [k.kernel_source.spec["seqlen_q"] for k in ks] == [256, 512]

    def test_kernel_overrides_a_default_by_restating_it(self, tmp_path):
        raw = self._raw(
            kernel_defaults={
                "kind": "rocke",
                "source": "s.py",
                "builder": "b",
                "spec": {"head_size": 128},
            }
        )
        raw["packs"][0]["kernels"][1]["kernel_source"]["spec"]["head_size"] = 64
        config = self._load(tmp_path, raw)
        assert [k.kernel_source.spec["head_size"] for k in config.packs[0].kernels] == [
            128,
            64,
        ]

    def test_missing_kind_still_rejected_when_no_defaults(self, tmp_path):
        raw = self._raw()
        with pytest.raises(ConfigError, match="kind"):
            self._load(tmp_path, raw)

    def test_a_kernel_of_another_kind_is_named_by_the_defaults_rejection(
        self, tmp_path
    ):
        """One `kernel_defaults` cannot serve two kinds, and the message says whose."""
        raw = self._raw(
            kernel_defaults={"kind": "rocke", "source": "s.py", "builder": "b"}
        )
        raw["packs"][0]["kernels"][1]["kernel_source"] = {
            "kind": "hip",
            "source": "k.cpp",
            "entry": "k",
        }
        with pytest.raises(
            ConfigError, match=r"kernel_defaults \(as merged for kernel 'k2'\)"
        ):
            self._load(tmp_path, raw)


class TestGzippedConfig:
    """A generated variant set belongs in the repo as plain text, so `.gz` is a
    supported input rather than the expected shipping form."""

    def _raw(self):
        return {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/test",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernels": [
                        {
                            "name": "k1",
                            "kernel_source": {
                                "kind": "rocke",
                                "source": "kernels/gfx942/attention_dense.py",
                                "builder": "build_attention_dense",
                                "spec": {"seqlen_q": 256},
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }

    def test_gzipped_config_matches_plaintext(self, tmp_path):
        import gzip as _gzip

        text = yaml.dump(self._raw())
        plain = tmp_path / "c.yaml"
        plain.write_text(text)
        packed = tmp_path / "c.yaml.gz"
        with _gzip.open(packed, "wt") as f:
            f.write(text)

        a = load_config(plain)
        b = load_config(packed)
        assert a.engine.name == b.engine.name
        assert len(a.packs[0].kernels) == len(b.packs[0].kernels)
        assert (
            a.packs[0].kernels[0].kernel_source.spec
            == b.packs[0].kernels[0].kernel_source.spec
        )

    def test_gzipped_config_still_validated(self, tmp_path):
        import gzip as _gzip

        raw = self._raw()
        raw["packs"][0]["kernels"][0]["kernel_source"].pop("kind")
        packed = tmp_path / "c.yaml.gz"
        with _gzip.open(packed, "wt") as f:
            f.write(yaml.dump(raw))
        with pytest.raises(ConfigError, match="kind"):
            load_config(packed)


class TestAxisExpansion:
    """Pack-level `axes` cross-products a `kernel_template` into enumerated kernels at
    load, so a pack author declares the axes rather than the enumeration."""

    def _raw(self, axes, spec_extra=None, clear_template_spec=False, pack_extra=None):
        template = {
            "name": "dense",
            "kernel_source": {
                "kind": "rocke",
                "source": "kernels/gfx942/dense.py",
                "builder": "build_dense",
                "spec": {} if clear_template_spec else {"seqlen_q": 256},
            },
            "metadata": {},
        }
        if spec_extra:
            template["kernel_source"]["spec"].update(spec_extra)
        pack = {
            "name": "p",
            "arch": ["gfx942"],
            "axes": axes,
            "kernel_template": template,
        }
        if pack_extra:
            pack.update(pack_extra)
        return {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/test",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [
                {"name": "block_size", "type": "int", "default_value": 64},
                {"name": "block_n", "type": "int", "default_value": 64},
                {"name": "waves_per_eu", "type": "int", "default_value": 2},
            ],
            "packs": [pack],
        }

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def test_two_axes_over_one_template_yields_the_cross_product(self, tmp_path):
        raw = self._raw({"block_n": [64, 32], "waves_per_eu": [2, 4]})
        config = self._load(tmp_path, raw)
        kernels = config.packs[0].kernels
        assert len(kernels) == 4

        names = [k.name for k in kernels]
        assert len(set(names)) == 4, f"expanded kernel names collide: {names}"

        by_combo = {
            (k.kernel_source.spec["block_n"], k.kernel_source.spec["waves_per_eu"]): k
            for k in kernels
        }
        assert set(by_combo) == {(64, 2), (64, 4), (32, 2), (32, 4)}
        for (block_n, waves), kernel in by_combo.items():
            # Every axis value must land in BOTH the spec and the metadata --
            # the metadata is what the runtime and the dedup pass actually see.
            assert kernel.metadata["block_n"] == block_n
            assert kernel.metadata["waves_per_eu"] == waves
            assert kernel.kernel_source.spec["seqlen_q"] == 256

    def test_expanded_names_are_distinct_by_construction_not_luck(self, tmp_path):
        """Every axis value is encoded into the name in a fixed order, asserted directly
        rather than by trusting that the chosen axes vary it."""
        raw = self._raw({"block_n": [1, 2, 3], "waves_per_eu": [10, 20, 30]})
        config = self._load(tmp_path, raw)
        names = [k.name for k in config.packs[0].kernels]
        assert len(names) == 9
        assert len(set(names)) == 9

    def test_axis_not_in_kmd_fields_is_rejected_naming_the_field(self, tmp_path):
        raw = self._raw({"totally_undeclared_field": [1, 2]})
        with pytest.raises(ConfigError, match="totally_undeclared_field"):
            self._load(tmp_path, raw)

    def test_empty_axis_list_is_rejected(self, tmp_path):
        raw = self._raw({"block_n": []})
        with pytest.raises(ConfigError, match="non-empty"):
            self._load(tmp_path, raw)

    def test_single_valued_axis_warns(self, tmp_path):
        """A lone value contributes nothing to the cross-product and usually means a
        typo."""
        raw = self._raw({"block_n": [64], "waves_per_eu": [2, 4]})
        with pytest.warns(UserWarning, match="single value"):
            config = self._load(tmp_path, raw)
        assert len(config.packs[0].kernels) == 2

    def test_axes_compose_with_kernel_defaults(self, tmp_path):
        """An axis-expanded kernel is another entry in the per-kernel loop that merges
        kernel_defaults underneath it."""
        raw = self._raw(
            {"block_n": [64, 32]},
            clear_template_spec=True,
            pack_extra={
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "kernels/gfx942/dense.py",
                    "builder": "build_dense",
                    "spec": {"seqlen_q": 999},
                }
            },
        )
        config = self._load(tmp_path, raw)
        kernels = config.packs[0].kernels
        assert len(kernels) == 2
        for k in kernels:
            assert k.kernel_source.kind == "rocke"
            assert k.kernel_source.builder == "build_dense"
            assert k.kernel_source.spec["seqlen_q"] == 999
        assert {k.kernel_source.spec["block_n"] for k in kernels} == {64, 32}

    def test_template_field_already_stated_is_not_overwritten_by_the_axis(
        self, tmp_path
    ):
        raw = self._raw(
            {"block_n": [64, 32]},
            spec_extra={"block_n": -1},
        )
        config = self._load(tmp_path, raw)
        for k in config.packs[0].kernels:
            assert k.kernel_source.spec["block_n"] == -1

    def test_kernel_template_without_axes_is_rejected(self, tmp_path):
        raw = self._raw({"block_n": [64, 32]})
        del raw["packs"][0]["axes"]
        with pytest.raises(ConfigError, match="kernel_template"):
            self._load(tmp_path, raw)

    def test_axes_without_kernel_template_is_rejected(self, tmp_path):
        raw = self._raw({"block_n": [64, 32]})
        del raw["packs"][0]["kernel_template"]
        with pytest.raises(ConfigError, match="kernel_template"):
            self._load(tmp_path, raw)

    def test_axes_must_be_a_mapping(self, tmp_path):
        raw = self._raw({"block_n": [64, 32]})
        raw["packs"][0]["axes"] = ["block_n", "waves_per_eu"]
        with pytest.raises(ConfigError, match="mapping"):
            self._load(tmp_path, raw)


class TestExpansionCarriesEveryAuthoredKind:
    """`axes` and `variants` expand under every authored kind, not just `rocke`.

    Both expanders write the kernel's `kernel_source.spec` themselves, whatever the
    kind, and only `rocke` READS a spec. Judging a generated spec as though the author
    typed it would make every `hip` and `embedded_source` config using expansion
    unloadable.
    """

    #: One authored `kernel_source` per kind whose vocabulary excludes `spec`.
    SOURCES = {
        "hip": {"kind": "hip", "source": "k.cpp", "entry": "k"},
        "embedded_source": {
            "kind": "embedded_source",
            "source_file": "K.cpp",
            "entry_point": "K",
        },
    }
    #: Each kind's dialect and the subpath that dialect demands -- `hip` is emitted
    #: by a packaged bundle, `embedded_source` by a direct-load one.
    TOP_LEVEL = {
        "hip": {
            "dialect": "packaged",
            "kernel_source_kind": "hip",
            "authored_subpath": "hip/test",
        },
        "embedded_source": {"authored_subpath": "unit"},
    }

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def _axes_raw(self, kind):
        return {
            **self.TOP_LEVEL[kind],
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_n", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "axes": {"block_n": [64, 32]},
                    "kernel_template": {
                        "name": "t",
                        "kernel_source": dict(self.SOURCES[kind]),
                        "metadata": {},
                    },
                }
            ],
        }

    def _variants_raw(self, kind, spec_defaults=None):
        defaults = dict(self.SOURCES[kind])
        if spec_defaults is not None:
            defaults["spec"] = dict(spec_defaults)
        return {
            **self.TOP_LEVEL[kind],
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_m", "type": "int", "default_value": 256}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernel_defaults": defaults,
                    "variants": [
                        {
                            "name": "dense_bm{block_m}_{tag}",
                            "metadata": ["block_m"],
                            "spec_order": ["block_m"],
                            "knob_sets": {"pinned": [{"block_m": 256, "tag": "a"}]},
                            "shapes": [{"knobs": "pinned"}],
                        }
                    ],
                }
            ],
        }

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_axes_expand_under_a_kind_that_reads_no_spec(self, tmp_path, kind):
        config = self._load(tmp_path, self._axes_raw(kind))
        kernels = config.packs[0].kernels
        assert len(kernels) == 2
        assert {k.kernel_source.kind for k in kernels} == {kind}
        # The metadata is what the runtime matches on, and it carries the axis
        # whether or not the kind has a spec to also record it in.
        assert {k.metadata["block_n"] for k in kernels} == {64, 32}

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_variants_expand_under_a_kind_that_reads_no_spec(self, tmp_path, kind):
        config = self._load(tmp_path, self._variants_raw(kind))
        kernels = config.packs[0].kernels
        assert len(kernels) == 1
        assert kernels[0].kernel_source.kind == kind
        assert kernels[0].metadata["block_m"] == 256

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_variants_read_a_pack_level_spec_default_under_any_kind(
        self, tmp_path, kind
    ):
        """`kernel_defaults.spec` is the group's spec floor for every kind, reaching
        `shape_spec` and from there the kernel name and resolved metadata."""
        raw = self._variants_raw(kind, spec_defaults={"block_m": 256})
        raw["packs"][0]["variants"][0]["knob_sets"]["pinned"] = [{"tag": "a"}]
        config = self._load(tmp_path, raw)
        kernels = config.packs[0].kernels
        assert len(kernels) == 1
        assert kernels[0].name == "dense_bm256_a"
        assert kernels[0].metadata["block_m"] == 256

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_a_typo_in_variant_kernel_defaults_is_still_refused(self, tmp_path, kind):
        """Only `spec` is the expander's to read; the rest keeps its closed
        vocabulary."""
        raw = self._variants_raw(kind, spec_defaults={"block_m": 256})
        raw["packs"][0]["kernel_defaults"]["buid"] = {"defines": {}}
        with pytest.raises(ConfigError, match=r"declares \['buid'\]"):
            self._load(tmp_path, raw)

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_a_pack_level_spec_without_variants_is_still_refused(self, tmp_path, kind):
        """The exemption belongs to the expander that reads the key: with no `variants`
        to consume it, a pack-level spec under such a kind really is dropped."""
        raw = self._variants_raw(kind, spec_defaults={"block_m": 256})
        pack = raw["packs"][0]
        pack.pop("variants")
        pack["kernels"] = [{"name": "k", "kernel_source": {}, "metadata": {}}]
        with pytest.raises(ConfigError, match=r"declares \['spec'\]"):
            self._load(tmp_path, raw)

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_a_typo_in_the_template_kernel_source_is_still_refused(
        self, tmp_path, kind
    ):
        """The template's own keys ARE authored, so `buid` for `build` is caught."""
        raw = self._axes_raw(kind)
        raw["packs"][0]["kernel_template"]["kernel_source"]["buid"] = {"defines": {}}
        with pytest.raises(
            ConfigError, match=r"kernel_template kernel_source declares \['buid'\]"
        ):
            self._load(tmp_path, raw)

    @pytest.mark.parametrize("kind", ["hip", "embedded_source"])
    def test_a_hand_written_spec_under_a_kind_that_reads_none_is_still_refused(
        self, tmp_path, kind
    ):
        """The exemption is for the key the EXPANSION writes, not the spelling: a
        hand-typed spec under a kind that has none is still dropped and still
        reported."""
        raw = self._axes_raw(kind)
        raw["packs"][0]["kernel_template"]["kernel_source"]["spec"] = {"block_n": 64}
        with pytest.raises(
            ConfigError, match=r"kernel_template kernel_source declares \['spec'\]"
        ):
            self._load(tmp_path, raw)


class TestUnknownKeysAreRefused:
    """A silently dropped key means exit 0, a success banner and a bundle missing what
    the author configured -- `engine.knobbs` for `engine.knobs` emits a UED with no
    knobs."""

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def _valid(self):
        return {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
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

    def test_the_valid_config_still_loads(self, tmp_path):
        """The control: every rejection below is worthless without it."""
        assert self._load(tmp_path, self._valid()) is not None

    def test_a_typo_in_an_engine_key_is_refused(self, tmp_path):
        raw = self._valid()
        raw["engine"]["knobbs"] = ["block_size"]
        with pytest.raises(ConfigError, match="knobbs"):
            self._load(tmp_path, raw)

    def test_a_typo_in_a_pack_key_is_refused(self, tmp_path):
        raw = self._valid()
        raw["packs"][0]["discriminador"] = "x"
        with pytest.raises(ConfigError, match="discriminador"):
            self._load(tmp_path, raw)

    def test_a_typo_in_a_kernel_key_is_refused(self, tmp_path):
        raw = self._valid()
        raw["packs"][0]["kernels"][0]["metadatas"] = {}
        with pytest.raises(ConfigError, match="metadatas"):
            self._load(tmp_path, raw)

    def test_a_typo_at_the_top_level_is_refused(self, tmp_path):
        raw = self._valid()
        raw["dialects"] = "packaged"
        with pytest.raises(ConfigError, match="dialects"):
            self._load(tmp_path, raw)

    def test_the_diagnostic_lists_the_keys_that_ARE_read(self, tmp_path):
        raw = self._valid()
        raw["engine"]["knobbs"] = []
        with pytest.raises(ConfigError, match="Known keys"):
            self._load(tmp_path, raw)


class TestMappingShapedKeysAreGuarded:
    """Unguarded, `dict("oops")` raises a `ValueError` from inside the merge naming
    neither the kernel nor the key, and generate.py catches only ConfigError."""

    def _load(self, tmp_path, mutate):
        raw = {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/t",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [{"name": "block_n", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "rocke",
                                "source": "kernels/x.py",
                                "builder": "build_x",
                                "spec": {"block_n": 64},
                            },
                            "metadata": {"block_n": 64},
                        }
                    ],
                }
            ],
        }
        mutate(raw)
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    @pytest.mark.parametrize("bad", ["oops", [1, 2, 3], 42])
    def test_a_non_mapping_spec_is_a_named_error(self, tmp_path, bad):
        with pytest.raises(ConfigError, match="must be a mapping"):
            self._load(
                tmp_path,
                lambda r: r["packs"][0]["kernels"][0]["kernel_source"].__setitem__(
                    "spec", bad
                ),
            )

    def test_a_non_mapping_metadata_is_a_named_error(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a mapping"):
            self._load(
                tmp_path,
                lambda r: r["packs"][0]["kernels"][0].__setitem__("metadata", "oops"),
            )

    def test_a_mapping_valued_kind_is_a_named_error(self, tmp_path):
        """An over-indented `kind:` block makes the kind a mapping, and the per-kind
        vocabulary lookup then raises an unhashable-key `TypeError` generate.py does not
        catch."""
        with pytest.raises(ConfigError, match="is not a recognized kernel_source kind"):
            self._load(
                tmp_path,
                lambda r: r["packs"][0]["kernels"][0]["kernel_source"].__setitem__(
                    "kind", {"rocke": None}
                ),
            )

    @pytest.mark.parametrize("key", ["packs", "kmd_fields"])
    def test_a_present_but_valueless_list_key_is_a_named_error(self, tmp_path, key):
        """`packs:` with nothing under it is null: `raw.get(key) or []` hides it from
        the shape walk, and `load_config` iterates the null itself."""
        with pytest.raises(ConfigError, match=f"'{key}' must be a list of entries"):
            self._load(tmp_path, lambda r: r.__setitem__(key, None))

    def test_a_present_but_valueless_kernels_key_is_a_named_error(self, tmp_path):
        """The same hole one level down: `list(None)` in the per-pack loop."""
        with pytest.raises(
            ConfigError, match=r"'pack 'p' kernels' must be a list of entries"
        ):
            self._load(tmp_path, lambda r: r["packs"][0].__setitem__("kernels", None))

    def test_a_well_formed_config_still_loads(self, tmp_path):
        assert self._load(tmp_path, lambda r: None) is not None


class TestSpecializationDeclaration:
    """The ``specialization`` block is what a later check has instead of the compiler: a
    shipped bundle is verified where the rocKE that built it is not installed. A merely
    plausible declaration is worse than none -- the check runs and verifies nothing."""

    @staticmethod
    def _rocke_config(**declaration):
        """A packaged rocKE config whose spec carries both KMD fields."""
        kernel = make_kernel(
            kernel_source=KernelSource(
                kind="rocke",
                source="kernels/gfx942/example.py",
                builder="build_example",
                spec={"block_size": 64, "dtype": "bf16"},
            ),
        )
        return make_minimal_config(
            dialect="packaged",
            kernel_source_kind="rocke",
            packs=[make_pack(kernels=[kernel], arch=["gfx942"])],
            specialization=declaration,
        )

    @staticmethod
    def _complete_rocke_declaration(**overrides):
        declaration = {
            "metadata_fields": ["block_size", "dtype"],
            "matcher_only_fields": [],
            "bindings": {
                "block_size": {"field": "block_size"},
                "dtype": {"field": "dtype"},
            },
            "vocabulary": {"dtype": {"bf16": "BF16"}},
        }
        declaration.update(overrides)
        return declaration

    def test_a_complete_declaration_is_accepted(self):
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(**self._complete_rocke_declaration())
        _check_specialization_declaration(config)  # does not raise

    def test_a_partition_that_misses_a_field_is_rejected(self):
        """An unlisted field reads as one nobody specialized on, so a value that decided
        the binary is passed over unchecked."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(
                metadata_fields=["block_size"],
                bindings={"block_size": {"field": "block_size"}},
                vocabulary={},
            )
        )
        with pytest.raises(ConfigError, match="partition"):
            _check_specialization_declaration(config)

    def test_a_partition_that_overlaps_is_rejected(self):
        """A field claiming to be both checked and matcher-only leaves a checker unable
        to decide whether to demand a binding for it."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(matcher_only_fields=["dtype"])
        )
        with pytest.raises(ConfigError, match="BOTH"):
            _check_specialization_declaration(config)

    def test_binding_keys_must_equal_metadata_fields(self):
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(
                bindings={"block_size": {"field": "block_size"}}
            )
        )
        with pytest.raises(ConfigError, match="bindings"):
            _check_specialization_declaration(config)

    def test_a_binding_naming_both_a_field_and_a_method_is_rejected(self):
        """Two readings, no rule for which is authoritative."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(
                bindings={
                    "block_size": {"field": "block_size", "method": "effective_block"},
                    "dtype": {"field": "dtype"},
                }
            )
        )
        with pytest.raises(ConfigError, match="exactly one"):
            _check_specialization_declaration(config)

    def test_a_method_binding_is_accepted(self):
        """An effective accessor is the only truthful reading for a knob the kernel's
        policy resolves."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(
                bindings={
                    "block_size": {"method": "effective_block_size"},
                    "dtype": {"field": "dtype"},
                }
            )
        )
        _check_specialization_declaration(config)  # does not raise

    def test_a_spec_carried_field_may_not_be_called_matcher_only(self):
        """A key in ``kernel_source.spec`` is hydrated into the dataclass the builder is
        called with, so relabelling it matcher-only drops it from the agreement check
        while it keeps deciding the binary."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            metadata_fields=["dtype"],
            matcher_only_fields=["block_size"],
            bindings={"dtype": {"field": "dtype"}},
            vocabulary={},
        )
        with pytest.raises(ConfigError, match="matcher-only"):
            _check_specialization_declaration(config)

    def test_a_direct_load_config_may_not_claim_metadata_fields(self):
        """There is no builder object on that path, so a binding names a read nothing
        performs."""
        from codegen.config_loader import _check_specialization_declaration

        config = make_minimal_config(
            specialization={
                "metadata_fields": ["block_size"],
                "matcher_only_fields": ["dtype"],
                "bindings": {"block_size": {"field": "block_size"}},
                "vocabulary": {},
            }
        )
        with pytest.raises(ConfigError, match="compiled specialization"):
            _check_specialization_declaration(config)

    def test_a_direct_load_matcher_only_declaration_is_accepted(self):
        """The shape every non-compiled bundle must state explicitly."""
        from codegen.config_loader import _check_specialization_declaration

        config = make_minimal_config(
            specialization={
                "metadata_fields": [],
                "matcher_only_fields": ["block_size", "dtype"],
                "bindings": {},
                "vocabulary": {},
            }
        )
        _check_specialization_declaration(config)  # does not raise

    def test_an_authored_consumer_identity_is_rejected(self):
        """Ids are minted, never authored. One config declares one engine and one KMD,
        so an authored id would be a second source of truth pointing at whatever engine
        shared a name."""
        from codegen.config_loader import _check_specialization_declaration

        config = make_minimal_config(
            specialization={
                "metadata_fields": [],
                "matcher_only_fields": ["block_size", "dtype"],
                "bindings": {},
                "vocabulary": {},
                "engine_id": "00000000-0000-0000-0000-000000000000",
            }
        )
        with pytest.raises(ConfigError, match="minted"):
            _check_specialization_declaration(config)

    def test_a_vocabulary_entry_for_an_unchecked_field_is_rejected(self):
        """A translation with no effect leaves the builder's spelling in metadata, which
        loads cleanly and matches nothing."""
        from codegen.config_loader import _check_specialization_declaration

        config = self._rocke_config(
            **self._complete_rocke_declaration(
                vocabulary={"nonexistent": {"a": "B"}},
            )
        )
        with pytest.raises(ConfigError, match="vocabulary"):
            _check_specialization_declaration(config)

    def test_every_shipped_example_config_carries_a_valid_declaration(
        self, load_test_config
    ):
        """The examples are what an author copies, so a bundle they produce must be
        checkable rather than merely loadable."""
        for name in (
            "scale_add.yaml",
            "binary_ops.yaml",
            "axes_example.yaml",
            "variants_example.yaml",
            "gfx950_attention_dense.yaml",
        ):
            config = load_test_config(name)
            declared = {f.name for f in config.kmd_fields}
            declaration = config.specialization
            assert declaration, f"{name} declares no specialization"
            assert (
                set(declaration["metadata_fields"])
                | set(declaration["matcher_only_fields"])
                == declared
            ), name


class TestRuntimeContractRejections:
    """Values this loader accepted and ``DescriptorLoader.hpp`` rejects: each entry
    exits 0 here and fails the provider at load. The positive column is the other half
    -- the runtime's rule is the ceiling, and refusing what it accepts breaks a
    legitimate config."""

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def _raw(self):
        return {
            "authored_subpath": "unit",
            "engine": {"name": "hipkernel:Test", "knobs": ["block_size"]},
            "kmd_fields": [{"name": "block_size", "type": "int", "default_value": 64}],
            "packs": [
                {
                    "name": "p",
                    "kernels": [
                        {
                            "name": "k",
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "K.cpp",
                                "entry_point": "K",
                            },
                            "metadata": {"block_size": 64},
                        }
                    ],
                }
            ],
        }

    @pytest.mark.parametrize(
        "mutate, names",
        [
            # T1(a) -- coerceToDeclaredType, DescriptorLoader.hpp:604-610.
            pytest.param(
                lambda r: r["kmd_fields"][0].__setitem__("default_value", "sixty-four"),
                r"default_value 'sixty-four', which contradicts its declared type 'int'",
                id="kmd-default-value-contradicts-int",
            ),
            pytest.param(
                lambda r: r["kmd_fields"][0].__setitem__("default_value", True),
                r"default_value True, which contradicts its declared type 'int'",
                id="kmd-default-value-bool-for-int",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "dtype", "type": "string", "default_value": 1}
                ),
                r"entry 'dtype' declares default_value 1, which contradicts its declared type 'string'",
                id="kmd-default-value-int-for-string",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "tiles", "type": "int_list", "default_value": [1, "two"]}
                ),
                r"default_value \[1, 'two'\], which contradicts its declared type 'int_list'",
                id="kmd-default-value-mixed-int-list",
            ),
            # T1(b) -- requireNoDuplicates, DescriptorLoader.hpp:634-645, 750, 753.
            pytest.param(
                lambda r: r["engine"].__setitem__(
                    "knobs", ["block_size", "block_size"]
                ),
                r"engine\.knobs lists \['block_size'\] more than once",
                id="duplicate-knob",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__(
                    "behavior_notes", ["runtime_compilation", "runtime_compilation"]
                ),
                r"engine\.behavior_notes lists \['runtime_compilation'\] more than once",
                id="duplicate-behavior-note",
            ),
            # T1(c) -- requireString + Version, DescriptorLoader.hpp:770-781.
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", 1.0),
                r"engine\.sdk_version must be a string; got float \(1\.0\)",
                id="sdk-version-yaml-float",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", "1.0"),
                r"engine\.sdk_version '1\.0' is not a version the loader can parse",
                id="sdk-version-two-components",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", "v1.0.0"),
                r"engine\.sdk_version 'v1\.0\.0' is not a version the loader can parse",
                id="sdk-version-leading-v",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", ""),
                r"engine\.sdk_version '' is not a version the loader can parse",
                id="sdk-version-empty",
            ),
            # T1(d) -- is_number_integer + requireInt64, DescriptorLoader.hpp:977-984.
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", 1.5),
                r"declares priority 1\.5, which must be an integer",
                id="priority-float",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", True),
                r"declares priority True, which must be an integer",
                id="priority-bool",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", "1"),
                r"declares priority '1', which must be an integer",
                id="priority-string",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", 2**63),
                r"declares priority 9223372036854775808, which does not fit a signed 64-bit integer",
                id="priority-past-int64",
            ),
            # T2 -- the per-kind kernel_source vocabulary.
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0]["kernel_source"].__setitem__(
                    "buid", {"defines": {"BLOCK_SIZE": 64}}
                ),
                "buid",
                id="kernel-source-typo",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0]["kernel_source"].__setitem__(
                    "spec", {"block_size": 64}
                ),
                r"kernel_source declares \['spec'\], which kind 'embedded_source' does not read",
                id="kernel-source-key-of-another-kind",
            ),
            # T3 -- container shapes.
            pytest.param(
                lambda r: r["packs"][0].__setitem__("kernel_defaults", "oops"),
                r"pack 'p' kernel_defaults must be a mapping; got str \('oops'\)",
                id="kernel-defaults-scalar",
            ),
            pytest.param(
                lambda r: r.__setitem__("graph_match", "shared_shape"),
                r"graph_match must be a mapping; got str \('shared_shape'\)",
                id="graph-match-scalar",
            ),
            pytest.param(
                lambda r: r.__setitem__("specialization", "oops"),
                r"'specialization' must be a mapping; got str \('oops'\)",
                id="specialization-scalar",
            ),
            pytest.param(
                lambda r: r.__setitem__("kmd_fields", ["block_size"]),
                r"kmd_fields\[0\] must be a mapping; got str \('block_size'\)",
                id="kmd-fields-entry-scalar",
            ),
            pytest.param(
                lambda r: r["packs"][0].__setitem__("kernels", ["k"]),
                r"pack 'p' kernels\[0\] must be a mapping; got str \('k'\)",
                id="pack-kernels-entry-scalar",
            ),
            # A scalar whose text does NOT contain "name" is already caught by the
            # missing-key diagnostic; "namey" slips that substring test and reaches
            # `.get` on a str.
            pytest.param(
                lambda r: r["packs"].__setitem__(0, "namey"),
                r"packs\[0\] must be a mapping; got str \('namey'\)",
                id="pack-entry-scalar-containing-name",
            ),
            pytest.param(
                lambda r: r.__setitem__("engine", "namey"),
                r"engine must be a mapping; got str \('namey'\)",
                id="engine-scalar-containing-name",
            ),
            # One key below the pack-level mapping guarded above: `dict("oops")`
            # names neither the pack nor the key.
            pytest.param(
                lambda r: r["packs"][0].__setitem__(
                    "kernel_defaults", {"spec": "oops"}
                ),
                r"kernel_defaults\.spec",
                id="kernel-defaults-spec-scalar",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("knobs", 5),
                r"engine\.knobs must be a list of field names; got int \(5\)",
                id="engine-knobs-scalar",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("behavior_notes", 5),
                r"engine\.behavior_notes must be a list of field names; got int \(5\)",
                id="engine-behavior-notes-scalar",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("arch", 5),
                r"pack 'p' kernel 'k' arch must be a list of arch ids; got int \(5\)",
                id="kernel-arch-scalar",
            ),
            pytest.param(
                lambda r: r["packs"][0].__setitem__("arch", 5),
                r"pack 'p' arch must be a list of arch ids; got int \(5\)",
                id="pack-arch-scalar",
            ),
            # The "namey" trap one level down: this scalar's own text contains the
            # key the expander tests for membership.
            pytest.param(
                lambda r: (
                    r["packs"][0].__setitem__("axes", {"block_size": [64, 32]}),
                    r["packs"][0].__setitem__("kernel_template", "kernel_source"),
                ),
                r"pack 'p' kernel_template must be a mapping; got str \('kernel_source'\)",
                id="kernel-template-scalar-containing-kernel-source",
            ),
            # T4 -- requireInt64 over metadata, DescriptorLoader.hpp:511-519, 531, 550.
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0]["metadata"].__setitem__(
                    "block_size", 2**63
                ),
                r"metadata 'block_size' carries",
                id="metadata-int-past-int64",
            ),
            pytest.param(
                lambda r: (
                    r["kmd_fields"].append(
                        {"name": "tiles", "type": "int_list", "default_value": [1]}
                    ),
                    r["packs"][0]["kernels"][0]["metadata"].__setitem__(
                        "tiles", [1, 2**63]
                    ),
                ),
                r"metadata 'tiles' carries",
                id="metadata-int-list-element-past-int64",
            ),
            pytest.param(
                lambda r: r["kmd_fields"][0].__setitem__("default_value", 2**63),
                r"kmd_fields entry 'block_size' declares a default_value carrying",
                id="kmd-default-value-past-int64",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {
                        "name": "tiles",
                        "type": "int_list",
                        "default_value": [1, 2**63],
                    }
                ),
                r"kmd_fields entry 'tiles' declares a default_value carrying",
                id="kmd-int-list-default-past-int64",
            ),
        ],
    )
    def test_a_value_the_runtime_refuses_is_a_named_config_error(
        self, tmp_path, mutate, names
    ):
        raw = self._raw()
        mutate(raw)
        with pytest.raises(ConfigError, match=names):
            self._load(tmp_path, raw)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda r: None, id="the-control"),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "scale", "type": "float", "default_value": 2}
                ),
                id="int-default-widens-to-a-float-field",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "scale", "type": "float", "default_value": 0.5}
                ),
                id="float-default-for-a-float-field",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "dtype", "type": "string", "default_value": "FLOAT"}
                ),
                id="string-default-for-a-string-field",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "causal", "type": "bool", "default_value": False}
                ),
                id="false-default-for-a-bool-field",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "tiles", "type": "int_list", "default_value": [1, 2]}
                ),
                id="int-list-default",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {"name": "dtype", "type": "string", "default_value": "FLOAT"}
                ),
                id="a-second-optional-field",
            ),
            pytest.param(
                lambda r: (
                    r["kmd_fields"].append(
                        {"name": "waves", "type": "int", "default_value": 2}
                    ),
                    r["engine"].__setitem__("knobs", ["block_size", "waves"]),
                ),
                id="two-distinct-knobs",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__(
                    "behavior_notes", ["runtime_compilation"]
                ),
                id="one-behavior-note",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", "1.0.0"),
                id="sdk-version-three-components",
            ),
            pytest.param(
                lambda r: r["engine"].__setitem__("sdk_version", "10.20.30"),
                id="sdk-version-multi-digit-components",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", 0),
                id="priority-zero",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__("priority", -5),
                id="priority-negative",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].__setitem__(
                    "priority", 2**63 - 1
                ),
                id="priority-at-the-int64-ceiling",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0].pop("priority", None),
                id="priority-absent",
            ),
            pytest.param(
                lambda r: r["packs"][0].__setitem__(
                    "kernel_defaults",
                    {"kind": "embedded_source", "source_file": "K.cpp"},
                ),
                id="kernel-defaults-supplying-this-kinds-keys",
            ),
            pytest.param(
                lambda r: r.__setitem__(
                    "graph_match", {"shape": "shared_shape", "discriminator": "none"}
                ),
                id="graph-match-mapping",
            ),
            # The int64 bound's own endpoints: requireInt64 accepts the whole
            # signed range.
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0]["metadata"].__setitem__(
                    "block_size", 2**63 - 1
                ),
                id="metadata-int-at-the-int64-ceiling",
            ),
            pytest.param(
                lambda r: r["packs"][0]["kernels"][0]["metadata"].__setitem__(
                    "block_size", -(2**63)
                ),
                id="metadata-int-at-the-int64-floor",
            ),
            pytest.param(
                lambda r: r["kmd_fields"][0].__setitem__("default_value", 2**63 - 1),
                id="kmd-default-value-at-the-int64-ceiling",
            ),
            pytest.param(
                lambda r: r["kmd_fields"].append(
                    {
                        "name": "tiles",
                        "type": "int_list",
                        "default_value": [-(2**63), 2**63 - 1],
                    }
                ),
                id="kmd-int-list-default-at-the-int64-endpoints",
            ),
        ],
    )
    def test_a_value_the_runtime_accepts_still_loads(self, tmp_path, mutate):
        raw = self._raw()
        mutate(raw)
        assert self._load(tmp_path, raw) is not None


class TestExpandedKernelsAreKeyChecked:
    """``_reject_unknown_keys`` walks ``packs[].kernels[]`` only, so an envelope key
    misspelled in a template, or a control key misspelled in an arm, is dropped by the
    expansion and reported by nobody."""

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def _axes_raw(self):
        return {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/t",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [
                {"name": "block_n", "type": "int", "default_value": 64},
                {"name": "dtype", "type": "string", "default_value": "BF16"},
            ],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernel_defaults": {
                        "kind": "rocke",
                        "source": "kernels/x.py",
                        "builder": "build_x",
                        "spec": {"dtype": "bf16"},
                    },
                    "axes": {"block_n": [64, 32]},
                    "kernel_template": {
                        "name": "t",
                        "kernel_source": {"spec": {}},
                        "metadata": {"dtype": "BF16"},
                    },
                }
            ],
        }

    def _variants_raw(self):
        return {
            "dialect": "packaged",
            "kernel_source_kind": "rocke",
            "authored_subpath": "rocKE/t",
            "engine": {"name": "hipkernel:Test"},
            "kmd_fields": [
                {"name": "dtype", "type": "string"},
                {"name": "block_m", "type": "int", "default_value": 256},
            ],
            "packs": [
                {
                    "name": "p",
                    "arch": ["gfx942"],
                    "kernel_defaults": {
                        "kind": "rocke",
                        "source": "kernels/x.py",
                        "builder": "build_x",
                    },
                    "variants": [
                        {
                            "name": "dense.{dtype}_bm{block_m}_{tag}",
                            "metadata": ["dtype", "block_m"],
                            "vocabulary": {"dtype": {"bf16": "BF16"}},
                            "spec_order": ["dtype", "block_m"],
                            "knob_sets": {"pinned": [{"block_m": 256, "tag": "a"}]},
                            "shapes": [{"dtype": "bf16", "knobs": "pinned"}],
                        }
                    ],
                }
            ],
        }

    def test_both_bases_load(self, tmp_path):
        """The control for every rejection below."""
        assert self._load(tmp_path, self._axes_raw()) is not None
        assert self._load(tmp_path, self._variants_raw()) is not None

    def test_a_misspelled_envelope_key_in_a_kernel_template_is_refused(self, tmp_path):
        raw = self._axes_raw()
        raw["packs"][0]["kernel_template"]["metadat"] = {"dtype": "BF16"}
        with pytest.raises(
            ConfigError, match=r"pack 'p' kernel_template declares \['metadat'\]"
        ):
            self._load(tmp_path, raw)

    def test_a_kernel_template_may_carry_every_envelope_key(self, tmp_path):
        raw = self._axes_raw()
        raw["packs"][0]["kernel_template"]["priority"] = 3
        raw["packs"][0]["kernel_template"]["arch"] = ["gfx942"]
        assert self._load(tmp_path, raw) is not None

    def test_a_misspelled_control_key_in_an_arm_is_refused(self, tmp_path):
        raw = self._variants_raw()
        raw["packs"][0]["variants"][0]["knob_sets"]["pinned"][0]["tg"] = "a"
        with pytest.raises(ConfigError, match=r"a knob_set arm declares \['tg'\]"):
            self._load(tmp_path, raw)

    def test_an_arm_may_carry_its_control_keys_and_its_spec_fields(self, tmp_path):
        raw = self._variants_raw()
        raw["packs"][0]["variants"][0]["knob_sets"]["pinned"][0].update(
            {"dtype": "bf16", "ordinal_offset": 1, "metadata": {"block_m": 256}}
        )
        assert self._load(tmp_path, raw) is not None

    def test_a_foreign_kind_key_in_kernel_defaults_is_refused(self, tmp_path):
        raw = self._axes_raw()
        raw["packs"][0]["kernel_defaults"]["source_file"] = "K.cpp"
        with pytest.raises(
            ConfigError, match=r"kernel_defaults .* declares \['source_file'\]"
        ):
            self._load(tmp_path, raw)

    def test_an_expanded_kernel_source_still_carries_its_own_kinds_keys(self, tmp_path):
        raw = self._axes_raw()
        raw["packs"][0]["kernel_template"]["kernel_source"]["spec"] = {"seqlen_q": 256}
        assert self._load(tmp_path, raw) is not None
