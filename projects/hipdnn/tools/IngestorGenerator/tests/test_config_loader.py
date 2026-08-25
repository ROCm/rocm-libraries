# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/config_loader.py.

Covers the happy path over both worked-example configs, the five pre-mint
loader-mirroring checks (in order), deprecated-key rejection, and the
kernel_source_kind rejection for hsaco_file/kpack/rocke_builder.
"""

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
    """Each rejection must name the DIALECT, not merely 'unsupported'.

    The common authoring mistake is a real kind written under the wrong
    dialect, whose fix is a one-line ``dialect:`` change. A bare 'unsupported'
    sends the author looking for a missing feature instead.
    """

    def test_hsaco_file_rejected_naming_prerequisite(self):
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="hsaco_file")
        with pytest.raises(ConfigError, match="supportsSourceKind"):
            _check_kernel_source_kind_implemented(config)

    def test_kpack_rejected_as_produced_not_authored(self):
        """kpack is what hkp_pack WRITES; authoring it is a second source of
        truth for library/toc_key/symbol/sha256 that can silently disagree
        with the archive those four are supposed to describe."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="kpack")
        with pytest.raises(ConfigError, match="PRODUCED kind, never an authored one"):
            _check_kernel_source_kind_implemented(config)

    def test_rocke_builder_rejected_pointing_at_the_packaged_spelling(self):
        """The runtime enum spelling parses and nothing dispatches it. A rocKE
        kernel reaches the loader already lowered to kpack, so the authored
        spelling is 'rocke' under the packaged dialect."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="rocke_builder")
        with pytest.raises(ConfigError, match="never reaches the runtime as rocKE"):
            _check_kernel_source_kind_implemented(config)

    def test_rocke_under_direct_load_names_the_right_dialect(self):
        """The wrong-dialect case: 'rocke' is real, just not in direct_load."""
        from codegen.config_loader import _check_kernel_source_kind_implemented

        config = make_minimal_config(kernel_source_kind="rocke")
        with pytest.raises(ConfigError, match="belongs to dialect 'packaged'"):
            _check_kernel_source_kind_implemented(config)

    def test_embedded_source_under_packaged_names_the_right_dialect(self):
        """And the converse direction."""
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


class TestDeprecatedKeys:
    def _base_raw(self):
        return {
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

    def test_kmd_field_optional_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["kmd_fields"][0]["optional"] = True
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="optional"):
            load_config(path)

    def test_kmd_field_optional_key_rejected_even_when_false(self, tmp_path):
        """Detection is presence, not value-truthy."""
        raw = self._base_raw()
        raw["kmd_fields"][0]["optional"] = False
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="optional"):
            load_config(path)

    def test_kmd_field_default_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["kmd_fields"][0]["default"] = 1
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="default_value"):
            load_config(path)

    def test_top_level_schema_key_rejected(self, tmp_path):
        raw = self._base_raw()
        raw["schema"] = "hipdnn.ued/v1"
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        with pytest.raises(ConfigError, match="schema"):
            load_config(path)


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
