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

    def test_duplicate_pack_names_rejected(self):
        """A pack name keys its descriptor id AND its output filename.

        Two packs sharing a name collided twice over: the same pack id, and the
        second `<slug>_<pack>.kdp.json` overwriting the first, so an entire pack's
        kernels vanished with no error. Only `discriminator` was checked before.
        """
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


class TestPackKernelDefaults:
    """A pack may hoist what every kernel repeats; a kernel overrides by restating.

    Generated variant sets restate `kind`, `source`, `builder` and every spec field
    the sweep does not vary, once per kernel. On the shipped gfx942 dense sets that
    was five spec fields and all three kernel_source keys identical across 2107
    kernels -- about half the file, and it buries the fields that actually differ.
    """

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
        # Hoisted spec fields reach every kernel; per-kernel fields survive.
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
        """The default is a convenience, not a way to omit a required key."""
        raw = self._raw()
        with pytest.raises(ConfigError, match="kind"):
            self._load(tmp_path, raw)


class TestGzippedConfig:
    """A `.gz` config loads identically to its plain-text twin.

    A generated variant set belongs in the repo as plain text, so `.gz` is a
    retained capability rather than the way a config is expected to ship. It still
    has to work: a config that arrives compressed must load identically, not
    almost-identically.
    """

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
        """Compression is transport, not an escape from the pre-mint checks."""
        import gzip as _gzip

        raw = self._raw()
        raw["packs"][0]["kernels"][0]["kernel_source"].pop("kind")
        packed = tmp_path / "c.yaml.gz"
        with _gzip.open(packed, "wt") as f:
            f.write(yaml.dump(raw))
        with pytest.raises(ConfigError, match="kind"):
            load_config(packed)


class TestAxisExpansion:
    """Pack-level `axes` cross-products a `kernel_template` into ordinary
    enumerated kernels at load time (finding H14).

    An enumerated variant set is fine at roughly a hundred kernels; it stops
    being fine the moment the variant set is driven by tuning axes instead of
    hand-picked shapes -- five two-valued knobs over a few hundred shapes is a
    line count no build step reads and no reviewer reads either, when the
    actual information content is the axes plus the shape source, about 30
    lines. `axes` lets a pack author declare that instead of the six-figure
    enumeration it stands for.
    """

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
            # A non-axis template field (spec.seqlen_q) survives untouched.
            assert kernel.kernel_source.spec["seqlen_q"] == 256

    def test_expanded_names_are_distinct_by_construction_not_luck(self, tmp_path):
        """There is precedent for a naming helper shipping a collision: a prior
        `_kernel_name` hardcoded a subset of one op's own field names and, on
        any other op, found none of them -- every variant collapsed onto one
        string. Encoding every axis value into the name, always, in a fixed
        order, must not repeat that: this asserts distinctness directly rather
        than trusting that the axes chosen happen to vary the name.
        """
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
        """Enumeration wearing a costume: a lone value contributes nothing to
        the cross-product and usually means a typo (a second value never
        added)."""
        raw = self._raw({"block_n": [64], "waves_per_eu": [2, 4]})
        with pytest.warns(UserWarning, match="single value"):
            config = self._load(tmp_path, raw)
        assert len(config.packs[0].kernels) == 2

    def test_axes_compose_with_kernel_defaults(self, tmp_path):
        """kernel_defaults hoists what every kernel repeats; axes expands one
        template into many. The two must stack: an axis-expanded kernel is
        just another entry in the same per-kernel loop that already merges
        kernel_defaults underneath it.
        """
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
        """The template may pin an axis field itself (e.g. a fixed default
        that one combination should not disturb); the axis only fills in what
        the template left unstated."""
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


class TestUnknownKeysAreRefused:
    """A key this loader does not read must not generate cleanly.

    Every unrecognised key was previously dropped by `raw.get(key, default)`: exit 0,
    a cheerful success banner, and a bundle silently missing whatever the author
    thought they had configured. `engine.knobbs` for `engine.knobs` emits a UED with
    no knobs at all.

    That is the worst failure this loader can have, because the author is not
    debugging -- they believe it took effect. The loader already refused three
    specific deprecated keys on exactly this reasoning; these tests generalise it.
    """

    def _load(self, tmp_path, raw):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(raw))
        return load_config(path)

    def _valid(self):
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
        """The control. Every rejection below is worthless without it."""
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
        """Naming the offender is half the fix; naming the alternatives is the rest."""
        raw = self._valid()
        raw["engine"]["knobbs"] = []
        with pytest.raises(ConfigError, match="Known keys"):
            self._load(tmp_path, raw)


class TestMappingShapedKeysAreGuarded:
    """A key the loader merges as a dict must actually be one.

    `dict("oops")` raises `ValueError: dictionary update sequence element #0 has
    length 1; 2 is required` from inside the merge -- naming neither the kernel nor
    the key, and generate.py catches only ConfigError, so the raw traceback reaches
    the author. The loader HAS a "must be a mapping" diagnostic; it was simply
    unreachable because the crash came first. A check that cannot fire is not a check.
    """

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

    def test_a_well_formed_config_still_loads(self, tmp_path):
        assert self._load(tmp_path, lambda r: None) is not None
