# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The compact form and the enumeration it stands for must generate the SAME
descriptors."""

import pytest
import yaml

from codegen.config_loader import ConfigError, load_config
from codegen.models import KernelSource

#: The tri-state's own KMD field. `default_value: -1` because 0 is a value the policy
#: returns, so a 0 default would alias "policy decided off" onto "explicitly off".
EXP2_FIELD = {"name": "use_exp2_fast", "type": "int", "default_value": -1}


def _config(**pack_overrides) -> dict:
    pack = {
        "name": "attn",
        "arch": ["gfx942"],
        "kernel_defaults": {
            "kind": "rocke",
            "source": "kernels/gfx942/attention_dense.py",
            "builder": "build_attention_dense",
        },
    }
    pack.update(pack_overrides)
    return {
        "dialect": "packaged",
        "kernel_source_kind": "rocke",
        "engine": {"name": "hipkernel:Attn", "knobs": ["block_m"]},
        "kmd_fields": [
            {"name": "dtype", "type": "string"},
            {"name": "seqlen_q", "type": "int", "default_value": 256},
            {"name": "block_m", "type": "int", "default_value": 256},
            EXP2_FIELD,
        ],
        "packs": [pack],
    }


def _group(**overrides) -> dict:
    group = {
        "name": "attn.{dtype}_sq{seqlen_q}_bm{block_m}_e{md_use_exp2_fast}",
        "metadata": ["dtype", "seqlen_q", "block_m", "use_exp2_fast"],
        "vocabulary": {"dtype": {"bf16": "BF16"}},
        "policy_knobs": ["use_exp2_fast"],
        "knob_sets": {
            "pair": [{"block_m": 128}, {"block_m": 256}],
        },
        "shapes": [
            {
                "dtype": "bf16",
                "seqlen_q": 512,
                "knobs": "pair",
                "resolved": {"use_exp2_fast": 1},
            }
        ],
    }
    group.update(overrides)
    return group


def _load(tmp_path, raw: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def _kernels(tmp_path, **group_overrides):
    return (
        _load(tmp_path, _config(variants=[_group(**group_overrides)])).packs[0].kernels
    )


class TestExpansion:
    def test_shape_crosses_its_own_knob_set(self, tmp_path):
        kernels = _kernels(tmp_path)
        assert [k.kernel_source.spec["block_m"] for k in kernels] == [128, 256]

    def test_each_shape_selects_its_own_knob_set(self, tmp_path):
        """The shipped sets are not a full grid, so a global cross-product would invent
        variants for some shapes and drop them for others."""
        kernels = _kernels(
            tmp_path,
            knob_sets={
                "one": [{"block_m": 64}],
                "pair": [{"block_m": 128}, {"block_m": 256}],
            },
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "knobs": "one",
                    "resolved": {"use_exp2_fast": 1},
                },
                {
                    "dtype": "bf16",
                    "seqlen_q": 1024,
                    "knobs": "pair",
                    "resolved": {"use_exp2_fast": 1},
                },
            ],
        )
        assert [
            (k.kernel_source.spec["seqlen_q"], k.kernel_source.spec["block_m"])
            for k in kernels
        ] == [
            (512, 64),
            (1024, 128),
            (1024, 256),
        ]

    def test_pack_kernel_defaults_supply_the_kernel_source(self, tmp_path):
        kernel = _kernels(tmp_path)[0]
        assert kernel.kernel_source.kind == "rocke"
        assert kernel.kernel_source.builder == "build_attention_dense"

    def test_group_spec_defaults_reach_every_kernel(self, tmp_path):
        kernels = _kernels(tmp_path, spec_defaults={"block_n": 64})
        assert all(k.kernel_source.spec["block_n"] == 64 for k in kernels)

    def test_a_shape_overrides_a_group_spec_default(self, tmp_path):
        kernels = _kernels(
            tmp_path,
            spec_defaults={"block_n": 64},
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "block_n": 32,
                    "knobs": "pair",
                    "resolved": {"use_exp2_fast": 1},
                }
            ],
        )
        assert all(k.kernel_source.spec["block_n"] == 32 for k in kernels)

    def test_spec_order_fixes_the_emitted_key_order(self, tmp_path):
        kernels = _kernels(tmp_path, spec_order=["block_m", "seqlen_q", "dtype"])
        assert list(kernels[0].kernel_source.spec) == ["block_m", "seqlen_q", "dtype"]

    def test_metadata_uses_the_matcher_vocabulary(self, tmp_path):
        """Copying one vocabulary over the other declines every graph while the engine
        still loads and every count reconciles."""
        kernel = _kernels(tmp_path)[0]
        assert kernel.kernel_source.spec["dtype"] == "bf16"
        assert kernel.metadata["dtype"] == "BF16"

    def test_a_value_outside_the_declared_vocabulary_is_refused(self, tmp_path):
        shape = {
            "dtype": "bf61",
            "seqlen_q": 512,
            "knobs": "pair",
            "resolved": {"use_exp2_fast": 1},
        }
        with pytest.raises(ConfigError, match=r"dtype.*bf61|bf61.*dtype"):
            _kernels(tmp_path, shapes=[shape])

    def test_an_arm_metadata_value_may_use_the_matcher_spelling(self, tmp_path):
        """An arm's `metadata` states the matcher-visible value directly, so it is
        written in the vocabulary's values, not its builder keys."""
        kernels = _kernels(
            tmp_path,
            knob_sets={"pair": [{"block_m": 128, "metadata": {"dtype": "BF16"}}]},
        )
        assert [k.metadata["dtype"] for k in kernels] == ["BF16"]

    def test_a_resolved_value_may_use_the_matcher_spelling(self, tmp_path):
        shape = {
            "seqlen_q": 512,
            "knobs": "pair",
            "resolved": {"dtype": "BF16", "use_exp2_fast": 1},
        }
        kernels = _kernels(
            tmp_path,
            name="attn.sq{seqlen_q}_bm{block_m}_e{md_use_exp2_fast}",
            shapes=[shape],
        )
        assert [k.metadata["dtype"] for k in kernels] == ["BF16", "BF16"]

    def test_an_arm_metadata_value_outside_the_vocabulary_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match=r"dtype.*bf61|bf61.*dtype"):
            _kernels(
                tmp_path,
                knob_sets={"pair": [{"block_m": 128, "metadata": {"dtype": "bf61"}}]},
            )

    def test_a_group_with_no_variants_key_expands_to_nothing(self, tmp_path):
        raw = _config(
            kernels=[
                {
                    "name": "hand.authored",
                    "kernel_source": {"spec": {"dtype": "bf16", "block_m": 256}},
                    "metadata": {
                        "dtype": "BF16",
                        "block_m": 256,
                        "seqlen_q": 1,
                        "use_exp2_fast": 0,
                    },
                }
            ]
        )
        assert len(_load(tmp_path, raw).packs[0].kernels) == 1


class TestArchAndTarget:
    """A group-level arch restricts what it expands to; a pack-level target reaches
    every kernel's kernel_source."""

    def _two_arch_config(self, **group_overrides) -> dict:
        raw = _config(
            arch=["gfx1151", "gfx1201"],
            variants=[_group(**group_overrides)],
        )
        raw["packs"][0]["kernel_defaults"]["target"] = "generic"
        return raw

    def test_group_arch_is_each_kernels_arch(self, tmp_path):
        kernels = (
            _load(tmp_path, self._two_arch_config(arch=["gfx1151"])).packs[0].kernels
        )
        assert kernels and all(k.arch == ["gfx1151"] for k in kernels)

    def test_a_group_without_arch_inherits_the_pack(self, tmp_path):
        kernels = _load(tmp_path, self._two_arch_config()).packs[0].kernels
        assert all(k.arch == [] for k in kernels)

    def test_group_arch_must_stay_within_the_pack(self, tmp_path):
        with pytest.raises(ConfigError, match="reaches past the pack's arch"):
            _load(tmp_path, self._two_arch_config(arch=["gfx1100"]))

    def test_group_arch_must_be_a_list(self, tmp_path):
        with pytest.raises(ConfigError):
            _load(tmp_path, self._two_arch_config(arch="gfx1151"))

    def test_pack_target_reaches_every_kernel(self, tmp_path):
        kernels = _load(tmp_path, self._two_arch_config()).packs[0].kernels
        assert {k.kernel_source.target for k in kernels} == {"generic"}
        assert all(
            k.kernel_source.as_document()["target"] == "generic" for k in kernels
        )

    def test_shipped_wmma_catalog_compiles_only_generic_objects(
        self, gfx1151_wmma_attention_config, gfx1151_wmma_attention_decode_config
    ):
        """Gfx1151WmmaAttentionNative's scoreKernel ranks the 32-key single-wave
        transposed geometry above 64-key two-wave unconditionally. That holds for
        generic objects only; #12710 measured the wide geometry faster for
        processor-specific gfx1151 objects at long sequences, and the matcher
        cannot see an object's compile target. A native kernel here needs that
        score keyed on the target first."""
        for config in (
            gfx1151_wmma_attention_config,
            gfx1151_wmma_attention_decode_config,
        ):
            kernels = [k for p in config.packs for k in p.kernels]
            assert kernels
            assert {k.kernel_source.target for k in kernels} == {"generic"}

    def test_a_kernel_without_target_emits_none(self, tmp_path):
        kernels = _kernels(tmp_path)
        assert all("target" not in k.kernel_source.as_document() for k in kernels)

    @pytest.mark.parametrize("target", ["Generic", "gfx11-generic", "", None, 1])
    def test_an_unknown_target_is_rejected(self, tmp_path, target):
        raw = self._two_arch_config()
        raw["packs"][0]["kernel_defaults"]["target"] = target
        with pytest.raises(ConfigError, match="kernel_source.target"):
            _load(tmp_path, raw)

    def test_target_on_a_non_rocke_kind_is_rejected(self, tmp_path):
        raw = _config(
            kernel_defaults={},
            kernels=[
                {
                    "name": "hip.kernel",
                    "kernel_source": {
                        "kind": "hip",
                        "source": "k.hip",
                        "entry": "k",
                        "target": "generic",
                    },
                    "metadata": {
                        "dtype": "BF16",
                        "block_m": 256,
                        "seqlen_q": 1,
                        "use_exp2_fast": 0,
                    },
                }
            ],
        )
        with pytest.raises(ConfigError, match=r"\['target'\], which kind 'hip'"):
            _load(tmp_path, raw)


class TestShapeAxes:
    """`shape_axes` crosses every shape with a grid stated once, first axis
    outermost, so a list every shape shares is not copied per shape."""

    def _axes_kernels(self, tmp_path, shape_axes, shapes=None, **overrides):
        return _kernels(
            tmp_path,
            shape_axes=shape_axes,
            shapes=shapes or [{"knobs": "pair", "resolved": {"use_exp2_fast": 1}}],
            **overrides,
        )

    def test_axes_cross_every_shape_first_axis_outermost(self, tmp_path):
        kernels = self._axes_kernels(
            tmp_path,
            {"dtype": ["bf16", "fp16"], "seqlen_q": [512, 1024]},
            vocabulary={"dtype": {"bf16": "BF16", "fp16": "FP16"}},
        )
        assert [
            (
                k.kernel_source.spec["dtype"],
                k.kernel_source.spec["seqlen_q"],
                k.kernel_source.spec["block_m"],
            )
            for k in kernels
        ] == [
            ("bf16", 512, 128),
            ("bf16", 512, 256),
            ("bf16", 1024, 128),
            ("bf16", 1024, 256),
            ("fp16", 512, 128),
            ("fp16", 512, 256),
            ("fp16", 1024, 128),
            ("fp16", 1024, 256),
        ]

    def test_axes_combine_with_shape_fields(self, tmp_path):
        kernels = self._axes_kernels(
            tmp_path,
            {"seqlen_q": [512, 1024]},
            shapes=[
                {"dtype": "bf16", "knobs": "pair", "resolved": {"use_exp2_fast": 1}}
            ],
        )
        assert [k.metadata["seqlen_q"] for k in kernels] == [512, 512, 1024, 1024]
        assert {k.metadata["dtype"] for k in kernels} == {"BF16"}

    def test_a_shape_restating_an_axis_field_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="already supply"):
            self._axes_kernels(
                tmp_path,
                {"seqlen_q": [512]},
                shapes=[
                    {
                        "dtype": "bf16",
                        "seqlen_q": 512,
                        "knobs": "pair",
                        "resolved": {"use_exp2_fast": 1},
                    }
                ],
            )

    @pytest.mark.parametrize("values", [[], 512, None])
    def test_an_axis_must_be_a_non_empty_list(self, tmp_path, values):
        axes = {"dtype": ["bf16"], "seqlen_q": values}
        with pytest.raises(ConfigError, match="non-empty list"):
            self._axes_kernels(tmp_path, axes)

    @pytest.mark.parametrize("axis", ["knobs", "seqlenq"])
    def test_an_axis_must_name_a_spec_field(self, tmp_path, axis):
        with pytest.raises(ConfigError, match="shape_axes names"):
            self._axes_kernels(
                tmp_path,
                {"dtype": ["bf16"], axis: [512]},
                spec_order=["dtype", "seqlen_q", "block_m"],
            )

    def test_shape_axes_must_be_a_mapping(self, tmp_path):
        with pytest.raises(ConfigError):
            self._axes_kernels(tmp_path, [512])


class TestTriState:
    """The spec decides the binary (ABSENT means the kernel's policy resolves it at
    build time), the metadata is what the matcher compares, and the KMD `default_value`
    is substituted for anything absent at load."""

    @pytest.mark.parametrize(
        "arm,resolved,expect_in_spec,expect_metadata",
        [
            ({"block_m": 128}, {"use_exp2_fast": 1}, False, 1),
            ({"block_m": 128}, {"use_exp2_fast": 0}, False, 0),
            ({"block_m": 128, "use_exp2_fast": False}, {}, True, 0),
            ({"block_m": 128, "use_exp2_fast": True}, {}, True, 1),
        ],
        ids=["policy-on", "policy-off", "pinned-off", "pinned-on"],
    )
    def test_all_four_combinations_survive(
        self, tmp_path, arm, resolved, expect_in_spec, expect_metadata
    ):
        shape = {"dtype": "bf16", "seqlen_q": 512, "knobs": "one"}
        if resolved:
            shape["resolved"] = resolved
        kernel = _kernels(tmp_path, knob_sets={"one": [arm]}, shapes=[shape])[0]
        assert ("use_exp2_fast" in kernel.kernel_source.spec) is expect_in_spec
        assert kernel.metadata["use_exp2_fast"] == expect_metadata

    def test_absent_and_pinned_false_are_different_kernels(self, tmp_path):
        """Both reach the matcher as 0, so the name must separate them: the shipped
        grammar spells policy-decided-off `ed` and pinned-off `e0`."""
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_sq{seqlen_q}_bm{block_m}_{tag}",
            knob_sets={
                "both": [
                    {"block_m": 128, "tag": "ed"},
                    {"block_m": 128, "use_exp2_fast": False, "tag": "e0"},
                ]
            },
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "knobs": "both",
                    "resolved": {"use_exp2_fast": 0},
                }
            ],
        )
        assert [k.metadata["use_exp2_fast"] for k in kernels] == [0, 0]
        assert "use_exp2_fast" not in kernels[0].kernel_source.spec
        assert kernels[1].kernel_source.spec["use_exp2_fast"] is False
        assert len({k.name for k in kernels}) == 2

    def test_arm_metadata_wins_over_the_same_field_in_the_spec(self, tmp_path):
        """The arm's metadata is what the MATCHER compares and the spec is what the
        binary was built from, so the loader must not overwrite the stated value with
        the spec's."""
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_sq{seqlen_q}_bm{block_m}_e{md_use_exp2_fast}",
            knob_sets={
                "one": [
                    {
                        "block_m": 128,
                        "use_exp2_fast": False,
                        "metadata": {"use_exp2_fast": 1},
                    }
                ]
            },
            shapes=[{"dtype": "bf16", "seqlen_q": 512, "knobs": "one"}],
        )
        assert kernels[0].metadata["use_exp2_fast"] == 1
        assert kernels[0].kernel_source.spec["use_exp2_fast"] is False
        assert kernels[0].name.endswith("_e1")

    def test_a_policy_knob_with_no_resolved_value_is_rejected(self, tmp_path):
        """Silence means the loader substitutes the KMD default as the catalog key while
        the binary was built from the policy's answer."""
        with pytest.raises(ConfigError, match="policy"):
            _kernels(
                tmp_path,
                knob_sets={"one": [{"block_m": 128}]},
                shapes=[{"dtype": "bf16", "seqlen_q": 512, "knobs": "one"}],
            )

    def test_an_arm_can_pin_metadata_the_spec_does_not_carry(self, tmp_path):
        """Two catalog entries over ONE binary: sweeping an arch-private knob pins what
        the matcher compares without changing what is compiled."""
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_sq{seqlen_q}_bm{block_m}_e{md_use_exp2_fast}",
            knob_sets={
                "swept": [
                    {"block_m": 128, "metadata": {"use_exp2_fast": 0}},
                    {"block_m": 128, "metadata": {"use_exp2_fast": 1}},
                ]
            },
            shapes=[{"dtype": "bf16", "seqlen_q": 512, "knobs": "swept"}],
        )
        assert [k.metadata["use_exp2_fast"] for k in kernels] == [0, 1]
        assert all("use_exp2_fast" not in k.kernel_source.spec for k in kernels)
        assert len({k.name for k in kernels}) == 2

    def test_a_boolean_metadata_knob_is_emitted_as_one_type(self, tmp_path):
        """Shipped metadata spells this knob four ways: 0, 1, False, True."""
        kernels = _kernels(
            tmp_path,
            knob_sets={
                "mixed": [
                    {"block_m": 128, "use_exp2_fast": True},
                    {"block_m": 256, "use_exp2_fast": False},
                    {"block_m": 64},
                ]
            },
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "knobs": "mixed",
                    "resolved": {"use_exp2_fast": 1},
                }
            ],
        )
        emitted = [k.metadata["use_exp2_fast"] for k in kernels]
        assert emitted == [1, 0, 1]
        assert {type(v) for v in emitted} == {int}


class TestNameInjectivity:
    """Nothing downstream catches a name collision: the dedup pass keys on metadata, so
    two identically named descriptors are indistinguishable in any log."""

    @pytest.mark.parametrize("field,other", [("dtype", "fp16"), ("seqlen_q", 1024)])
    def test_shapes_differing_in_one_field_get_different_names(
        self, tmp_path, field, other
    ):
        base = {
            "dtype": "bf16",
            "seqlen_q": 512,
            "knobs": "one",
            "resolved": {"use_exp2_fast": 1},
        }
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_sq{seqlen_q}_bm{block_m}",
            vocabulary={"dtype": {"bf16": "BF16", "fp16": "FP16"}},
            knob_sets={"one": [{"block_m": 128}]},
            shapes=[base, {**base, field: other}],
        )
        assert len({k.name for k in kernels}) == 2

    def test_arms_differing_in_one_knob_get_different_names(self, tmp_path):
        kernels = _kernels(tmp_path)
        assert len({k.name for k in kernels}) == len(kernels)

    def test_arms_differing_only_in_the_tri_state_get_different_names(self, tmp_path):
        """Pinned-on and policy-on share a metadata value, so the name comes from the
        tag."""
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_sq{seqlen_q}_bm{block_m}_{tag}",
            knob_sets={
                "both": [
                    {"block_m": 128, "tag": "e{md_use_exp2_fast}"},
                    {"block_m": 128, "use_exp2_fast": True, "tag": "pinned"},
                ]
            },
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "knobs": "both",
                    "resolved": {"use_exp2_fast": 1},
                }
            ],
        )
        assert len({k.name for k in kernels}) == 2

    def test_a_name_template_naming_an_absent_field_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="name template"):
            _kernels(tmp_path, name="attn.{no_such_field}")

    def test_the_ordinal_distinguishes_otherwise_identical_names(self, tmp_path):
        kernels = _kernels(
            tmp_path,
            name="attn.{ordinal:05d}_{dtype}",
            knob_sets={
                "pair": [
                    {"block_m": 128, "ordinal_offset": 0},
                    {"block_m": 256, "ordinal_offset": 1},
                ]
            },
            shapes=[
                {
                    "dtype": "bf16",
                    "seqlen_q": 512,
                    "knobs": "pair",
                    "ordinal": 20,
                    "resolved": {"use_exp2_fast": 1},
                }
            ],
        )
        assert [k.name for k in kernels] == ["attn.00020_bf16", "attn.00021_bf16"]


class TestRejections:
    """Every closed vocabulary exists because an unread key generates cleanly."""

    def test_an_unknown_group_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="knob_setz"):
            _kernels(tmp_path, knob_setz={})

    def test_a_shape_naming_an_undeclared_knob_set_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="knob_set"):
            _kernels(
                tmp_path,
                shapes=[{"dtype": "bf16", "seqlen_q": 512, "knobs": "nope"}],
            )

    def test_an_empty_knob_set_is_rejected(self, tmp_path):
        """An empty cross-product expands the shape to ZERO kernels."""
        with pytest.raises(ConfigError, match="non-empty"):
            _kernels(tmp_path, knob_sets={"pair": []})

    def test_a_metadata_field_no_kmd_field_declares_is_rejected(self, tmp_path):
        """An undeclared metadata field drops the WHOLE pack at
        resolveDescriptorSets()."""
        with pytest.raises(ConfigError, match="kmd_fields"):
            _kernels(tmp_path, metadata=["dtype", "block_m", "use_exp2_fast", "nope"])

    def test_two_shapes_rendering_one_name_are_rejected(self, tmp_path):
        """The template omits seqlen_q, the only field the two shapes differ in."""
        with pytest.raises(ConfigError, match="duplicated kernel name"):
            _kernels(
                tmp_path,
                name="attn.{dtype}_bm{block_m}",
                knob_sets={"one": [{"block_m": 128}]},
                shapes=[
                    {
                        "dtype": "bf16",
                        "seqlen_q": 512,
                        "knobs": "one",
                        "resolved": {"use_exp2_fast": 1},
                    },
                    {
                        "dtype": "bf16",
                        "seqlen_q": 1024,
                        "knobs": "one",
                        "resolved": {"use_exp2_fast": 1},
                    },
                ],
            )

    def test_an_empty_tag_does_not_corrupt_a_value_containing_an_underscore(
        self, tmp_path
    ):
        """Squeezing the rendered name cannot tell its own separator from one inside a
        value, so `a__b` and `a_b` would land on one name."""
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_bm{block_m}_{tag}",
            vocabulary={"dtype": {"a__b": "A", "a_b": "B"}},
            knob_sets={"one": [{"block_m": 128, "tag": ""}]},
            shapes=[
                {
                    "dtype": "a__b",
                    "seqlen_q": 512,
                    "knobs": "one",
                    "resolved": {"use_exp2_fast": 1},
                },
                {
                    "dtype": "a_b",
                    "seqlen_q": 512,
                    "knobs": "one",
                    "resolved": {"use_exp2_fast": 1},
                },
            ],
        )
        assert [k.name for k in kernels] == ["attn.a__b_bm128", "attn.a_b_bm128"]

    def test_a_misspelled_control_key_is_rejected(self, tmp_path):
        """`ordinl` for `ordinal` would otherwise become a SPEC field, changing the
        binary while the config still loads."""
        with pytest.raises(ConfigError, match="ordinl"):
            _kernels(
                tmp_path,
                spec_order=["dtype", "seqlen_q", "block_m"],
                shapes=[
                    {
                        "dtype": "bf16",
                        "seqlen_q": 512,
                        "knobs": "pair",
                        "ordinl": 7,
                        "resolved": {"use_exp2_fast": 1},
                    }
                ],
            )

    @pytest.mark.parametrize(
        "override,match",
        [
            ({"policy_knobs": "use_exp2_fast"}, "list of field names"),
            ({"spec_order": "dtype"}, "list of field names"),
            ({"spec_defaults": ["dtype"]}, "must be a mapping"),
            ({"vocabulary": {"dtype": "BF16"}}, "must be a mapping"),
        ],
        ids=["policy_knobs-str", "spec_order-str", "defaults-list", "vocab-scalar"],
    )
    def test_a_mistyped_group_key_is_rejected(self, tmp_path, override, match):
        """A bare string iterates as CHARACTERS, so the key silently does nothing."""
        with pytest.raises(ConfigError, match=match):
            _kernels(tmp_path, **override)

    def test_a_name_template_that_cannot_render_is_rejected(self, tmp_path):
        """An unmatched brace raises ValueError from str.format, not KeyError."""
        with pytest.raises(ConfigError, match="could not be rendered"):
            _kernels(tmp_path, name="attn.{dtype")

    def test_a_metadata_field_nothing_decides_is_rejected(self, tmp_path):
        """`seqlen_q` IS a declared KMD field, so the loader would substitute its
        default_value as the catalog key for a binary built from something else."""
        with pytest.raises(ConfigError, match="neither"):
            _kernels(
                tmp_path,
                knob_sets={"pair": [{"block_m": 128}]},
                shapes=[
                    {
                        "dtype": "bf16",
                        "knobs": "pair",
                        "resolved": {"use_exp2_fast": 1},
                    }
                ],
            )


#: Every KMD field the defaults probe below declares, and the value its pack-level
#: `kernel_defaults.spec` supplies when that row is the one under test.
_PROBE_FIELDS = [
    {"name": "dtype", "type": "string"},
    {"name": "seqlen_q", "type": "int", "default_value": 256},
    {"name": "block_m", "type": "int", "default_value": 256},
    {"name": "waves_per_eu", "type": "int", "default_value": 2},
    EXP2_FIELD,
]


def _probe_config(pack_default_spec: dict, arm: dict, resolved: dict) -> dict:
    """A one-shape, one-arm set whose only source for some spec keys is the pack-level
    ``kernel_defaults.spec``."""
    return {
        "dialect": "packaged",
        "kernel_source_kind": "rocke",
        "engine": {"name": "hipkernel:Probe", "knobs": ["block_m"]},
        "kmd_fields": _PROBE_FIELDS,
        "specialization": {
            "metadata_fields": [f["name"] for f in _PROBE_FIELDS],
            "matcher_only_fields": [],
            "bindings": {
                "dtype": {"field": "dtype"},
                "seqlen_q": {"field": "seqlen_q"},
                "block_m": {"field": "block_m"},
                "waves_per_eu": {"field": "waves_per_eu"},
                "use_exp2_fast": {"method": "effective_use_exp2_fast"},
            },
            "vocabulary": {"dtype": {"bf16": "BF16"}},
        },
        "packs": [
            {
                "name": "probe",
                "arch": ["gfx942"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "kernels/gfx942/probe.py",
                    "builder": "build_probe",
                    "spec": pack_default_spec,
                },
                "variants": [
                    {
                        "name": "probe.{ordinal}",
                        "metadata": [f["name"] for f in _PROBE_FIELDS],
                        "vocabulary": {"dtype": {"bf16": "BF16"}},
                        "policy_knobs": ["use_exp2_fast"],
                        "knob_sets": {"one": [dict(arm)]},
                        "shapes": [{"knobs": "one", "resolved": dict(resolved)}],
                    }
                ],
            }
        ],
    }


class TestPackDefaultsComposeBeforeMetadataProjection:
    """A pack-level ``kernel_defaults.spec`` must reach the metadata projection, and an
    OMITTED key must stay omitted: omission tells the builder its own policy decides,
    which is not the same as pinning the knob to ``false``/``0``."""

    @staticmethod
    def _emitted(tmp_path, config_raw):
        from codegen.generator import build_kdp, mint_ids

        config = _load(tmp_path, config_raw)
        kdp = build_kdp(config, config.packs[0], mint_ids(config))
        assert len(kdp["kernelDescriptors"]) == 1
        return kdp["kernelDescriptors"][0]

    #: (field supplied ONLY by the pack default, spec value, emitted metadata).
    #: `dtype` also crosses the vocabulary; `use_exp2_fast` also crosses the
    #: bool-to-int projection its declared type requires.
    @pytest.mark.parametrize(
        "field,spec_value,expected",
        [
            ("dtype", "bf16", "BF16"),
            ("seqlen_q", 1024, 1024),
            ("block_m", 128, 128),
            ("waves_per_eu", 4, 4),
            ("use_exp2_fast", True, 1),
        ],
    )
    def test_a_key_only_the_pack_default_supplies_reaches_the_metadata(
        self, tmp_path, field, spec_value, expected
    ):
        pack_default = {
            "dtype": "bf16",
            "seqlen_q": 256,
            "block_m": 256,
            "waves_per_eu": 2,
        }
        pack_default[field] = spec_value
        # The arm and the shape state NOTHING: the pack default is the only source.
        emitted = self._emitted(
            tmp_path,
            _probe_config(pack_default, {}, {"use_exp2_fast": 1}),
        )
        assert emitted["metadata"][field] == expected
        assert emitted["kernel_source"]["spec"][field] == spec_value

    def test_a_key_omitted_at_both_levels_stays_absent_from_the_spec(self, tmp_path):
        """``use_exp2_fast`` is absent from both levels, so the builder's policy decides
        and the shape's ``resolved`` block states the answer it gave."""
        emitted = self._emitted(
            tmp_path,
            _probe_config(
                {"dtype": "bf16", "seqlen_q": 256, "block_m": 256, "waves_per_eu": 2},
                {},
                {"use_exp2_fast": 1},
            ),
        )
        assert "use_exp2_fast" not in emitted["kernel_source"]["spec"], (
            "an omitted policy knob was materialized into the spec; that pins a "
            "value the author left to the kernel and changes the compiled binary"
        )
        assert emitted["metadata"]["use_exp2_fast"] == 1


class TestLaunches:
    """A rocke ``kernel_source.launches`` maps launch names to override mappings,
    each merged onto the base source and written as the UKD's ``sources`` map;
    anything else is refused."""

    BASE = {
        "kind": "rocke",
        "source": "k.py",
        "builder": "build_k",
        "spec": {"block_m": 256, "head_size": 64},
    }

    @staticmethod
    def _emitted(tmp_path, raw):
        from codegen.generator import build_kdp, mint_ids

        raw["specialization"] = {
            "metadata_fields": ["dtype", "seqlen_q", "block_m", "use_exp2_fast"],
            "matcher_only_fields": [],
            "bindings": {
                "dtype": {"field": "dtype"},
                "seqlen_q": {"field": "seqlen_q"},
                "block_m": {"field": "block_m"},
                "use_exp2_fast": {"method": "effective_use_exp2_fast"},
            },
            "vocabulary": {"dtype": {"bf16": "BF16"}},
        }
        config = _load(tmp_path, raw)
        kdp = build_kdp(config, config.packs[0], mint_ids(config))
        return kdp["kernelDescriptors"]

    @staticmethod
    def _kernel(name: str, source: dict) -> dict:
        return {
            "name": name,
            "kernel_source": source,
            "metadata": {
                "dtype": "BF16",
                "block_m": 256,
                "seqlen_q": 1,
                "use_exp2_fast": 0,
            },
        }

    def _hand_authored(self, source: dict, name: str = "k") -> dict:
        return _config(kernel_defaults={}, kernels=[self._kernel(name, source)])

    def _with_launches(self, launches, **extra) -> dict:
        return self._hand_authored({**self.BASE, "launches": launches, **extra})

    def test_a_config_without_launches_is_unchanged(self, tmp_path):
        (entry,) = self._emitted(tmp_path, self._hand_authored(self.BASE))
        assert list(entry) == [
            "version",
            "id",
            "name",
            "kernel_source",
            "metadata",
            "priority",
        ]
        assert entry["version"] == "1.0"
        assert entry["kernel_source"] == self.BASE

    def test_two_launches_emit_sources_with_the_overrides_merged(self, tmp_path):
        raw = self._with_launches({"decode": {}, "merge": {"role": "merge"}})
        (entry,) = self._emitted(tmp_path, raw)
        assert "kernel_source" not in entry
        assert entry["version"] == "1.1"
        assert entry["sources"] == {
            "decode": self.BASE,
            "merge": {**self.BASE, "role": "merge"},
        }
        # The key order of an entry is part of the descriptor bytes.
        assert list(entry["sources"]["merge"]) == [
            "kind",
            "source",
            "builder",
            "spec",
            "role",
        ]

    def test_the_version_is_raised_for_that_descriptor_only(self, tmp_path):
        """One engine names one set of launches, so a launch UKD and a plain one are
        generated from two configs; each is stamped by its own source."""
        multi_dir, single_dir = tmp_path / "multi", tmp_path / "single"
        multi_dir.mkdir()
        single_dir.mkdir()
        (multi,) = self._emitted(
            multi_dir,
            self._with_launches({"a": {}, "b": {"role": "b"}}),
        )
        (single,) = self._emitted(single_dir, self._hand_authored(self.BASE))
        assert (multi["version"], single["version"]) == ("1.1", "1.0")
        assert "sources" in multi and "kernel_source" in single
        assert "sources" not in single and "kernel_source" not in multi

    def test_a_launch_may_override_the_builder_and_the_spec(self, tmp_path):
        raw = self._with_launches(
            {
                "segment": {},
                "reduce": {
                    "builder": "build_reduce",
                    "target": "generic",
                    "spec": {"num_splits": 4},
                },
            }
        )
        (entry,) = self._emitted(tmp_path, raw)
        segment, reduce = entry["sources"]["segment"], entry["sources"]["reduce"]
        assert segment == self.BASE
        # The override REPLACES the spec: a different builder's dataclass must not
        # be handed the base builder's fields.
        assert reduce == {
            "kind": "rocke",
            "source": "k.py",
            "builder": "build_reduce",
            "target": "generic",
            "spec": {"num_splits": 4},
        }

    def test_launches_from_kernel_defaults_follow_each_variant(self, tmp_path):
        """A launch with no `spec` override takes each variant's own expanded spec, in
        the spec_order the variant emits, so a variant set carries launches without
        the launches knowing which axes vary. A `spec` override does not follow the
        variants: it replaces the spec and expansion never varies it (see the
        static-spec tests below)."""
        raw = _config(variants=[_group(spec_order=["block_m", "seqlen_q", "dtype"])])
        raw["packs"][0]["kernel_defaults"]["launches"] = {
            "segment": {},
            "reduce": {"role": "reduce"},
        }
        entries = self._emitted(tmp_path, raw)
        assert [e["sources"]["segment"]["spec"]["block_m"] for e in entries] == [
            128,
            256,
        ]
        for entry in entries:
            assert list(entry["sources"]["segment"]["spec"]) == [
                "block_m",
                "seqlen_q",
                "dtype",
            ]
            assert (
                entry["sources"]["reduce"]["spec"]
                == entry["sources"]["segment"]["spec"]
            )
            assert entry["version"] == "1.1"

    def test_a_launch_spec_stating_a_varied_kmd_field_is_rejected(self, tmp_path):
        """A launch `spec` override is static, so a kmd field it states is one value
        for every variant. The variants differing in block_m disagree with it: the
        error names the first variant that does."""
        raw = _config(variants=[_group()])
        raw["packs"][0]["kernel_defaults"]["launches"] = {
            "segment": {},
            "reduce": {"spec": {"block_m": 128, "head_size": 64}},
        }
        with pytest.raises(
            ConfigError,
            match=r"kernel 'attn\.bf16_sq512_bm256_e1'.*launches\['reduce'\]\.spec "
            r"states \['block_m'\]",
        ):
            _load(tmp_path, raw)

    def test_a_launch_spec_stating_a_shape_axis_value_is_rejected(self, tmp_path):
        """The same through `shape_axes`: the axis varies seqlen_q across kernels, so
        a launch stating 256 disagrees with the 512 kernel."""
        group = _group(
            shape_axes={"seqlen_q": [256, 512]},
            knob_sets={"one": [{"block_m": 128}]},
            shapes=[
                {"dtype": "bf16", "knobs": "one", "resolved": {"use_exp2_fast": 1}}
            ],
        )
        raw = _config(variants=[group])
        raw["packs"][0]["kernel_defaults"]["launches"] = {
            "segment": {},
            "reduce": {"spec": {"seqlen_q": 256, "head_size": 64}},
        }
        with pytest.raises(
            ConfigError,
            match=r"kernel 'attn\.bf16_sq512_bm128_e1'.*launches\['reduce'\]\.spec "
            r"states \['seqlen_q'\]",
        ):
            _load(tmp_path, raw)

    def test_a_launch_spec_stating_no_varied_field_is_the_same_on_every_variant(
        self, tmp_path
    ):
        raw = _config(variants=[_group()])
        raw["packs"][0]["kernel_defaults"]["launches"] = {
            "segment": {},
            "reduce": {"spec": {"num_splits": 4}},
        }
        kernels = _load(tmp_path, raw).packs[0].kernels
        assert len(kernels) == 2
        assert [k.kernel_source.launches["reduce"].spec for k in kernels] == [
            {"num_splits": 4},
            {"num_splits": 4},
        ]
        # The control: the launch with no spec override does follow the variants.
        assert [
            k.kernel_source.launches["segment"].spec["block_m"] for k in kernels
        ] == [
            128,
            256,
        ]

    @pytest.mark.parametrize(
        "sets",
        [
            ({"a": {}, "b": {"role": "b"}}, None),
            (None, {"a": {}, "b": {"role": "b"}}),
            ({"a": {}, "b": {"role": "b"}}, {"a": {}, "c": {"role": "c"}}),
        ],
        ids=["multi_then_single", "single_then_multi", "different_names"],
    )
    def test_kernels_must_name_one_set_of_launches(self, tmp_path, sets):
        """One engine has one dispatch handler, which declares one set of launch
        names; a kernel whose set differs would be dropped when the catalog loads."""
        sources = []
        for launches in sets:
            source = dict(self.BASE)
            if launches is not None:
                source["launches"] = launches
            sources.append(source)
        kernels = [self._kernel("one", sources[0]), self._kernel("two", sources[1])]
        kernels[1]["metadata"]["seqlen_q"] = 2
        with pytest.raises(ConfigError, match="names launches"):
            _load(tmp_path, _config(kernel_defaults={}, kernels=kernels))

    def test_kernels_naming_one_set_of_launches_load(self, tmp_path):
        launches = {"a": {}, "b": {"role": "b"}}
        kernels = [
            self._kernel("one", {**self.BASE, "launches": launches}),
            self._kernel("two", {**self.BASE, "launches": launches}),
        ]
        kernels[1]["metadata"]["seqlen_q"] = 2
        config = _load(tmp_path, _config(kernel_defaults={}, kernels=kernels))
        assert config.launch_names == ("a", "b")

    @staticmethod
    def _launch_config(config, names=("decode", "merge")):
        """`config` with every kernel declaring a launch per name."""
        import dataclasses

        for pack in config.packs:
            for kernel in pack.kernels:
                source = kernel.kernel_source
                kernel.kernel_source.launches = {
                    name: dataclasses.replace(source, role=name) for name in names
                }
        return config

    def test_the_native_stub_declares_the_launch_names(
        self, generator, gfx950_attention_dense_config
    ):
        config = self._launch_config(gfx950_attention_dense_config)
        rendered = generator._render_template("native.cpp.j2", config)
        assert "#include <set>" in rendered
        assert "std::set<std::string> launchNames() const override" in rendered
        assert 'return {"decode", "merge"};' in rendered
        # The guidance names the per-launch builder, not the multi-launch refusal.
        assert 'launchKernelDefinition(kernel, "<name>")' in rendered
        assert "buildIngestorKernelCode(kernel) refuses it" in rendered

    def test_the_native_stub_declares_nothing_without_launches(
        self, generator, gfx950_attention_dense_config
    ):
        """The control: no launches, so the handler stays a single-launch one and
        the stub is the one it always was."""
        rendered = generator._render_template(
            "native.cpp.j2", gfx950_attention_dense_config
        )
        for absent in ("launchNames", "#include <set>", "launchKernelDefinition"):
            assert absent not in rendered, absent
        assert "buildIngestorKernelCode overload that takes no compiler" in rendered

    def test_the_census_checks_each_launchs_kind_for_a_multi_launch_kernel(
        self, generator, gfx950_attention_dense_config
    ):
        config = self._launch_config(gfx950_attention_dense_config)
        rendered = generator._render_template("test_packs.cpp.j2", config)
        assert "if(!kernel.sources.empty())" in rendered
        assert "EXPECT_EQ(source.kind, EXPECTED_SOURCE_KIND)" in rendered
        assert "for(const auto& [launch, source] : kernel.sources)" in rendered
        # A single-launch kernel of the same bundle is still checked on `source`.
        assert "EXPECT_EQ(kernel.source.kind, EXPECTED_SOURCE_KIND)" in rendered

    def test_the_census_reads_only_source_without_launches(
        self, generator, gfx950_attention_dense_config
    ):
        rendered = generator._render_template(
            "test_packs.cpp.j2", gfx950_attention_dense_config
        )
        assert "kernel.sources" not in rendered
        assert "EXPECT_EQ(kernel.source.kind, EXPECTED_SOURCE_KIND)" in rendered

    def test_a_kernels_own_launches_replace_the_defaults(self, tmp_path):
        raw = self._hand_authored(
            {**self.BASE, "launches": {"x": {}, "y": {"role": "y"}}}
        )
        raw["packs"][0]["kernel_defaults"] = {
            "launches": {"p": {}, "q": {"role": "q"}, "r": {"role": "r"}}
        }
        (entry,) = self._emitted(tmp_path, raw)
        assert sorted(entry["sources"]) == ["x", "y"]

    def test_the_authored_document_loads_back_to_the_same_launches(self, tmp_path):
        raw = self._with_launches(
            {"a": {}, "b": {"role": "b", "builder": "build_b", "spec": {"n": 1}}}
        )
        source = _load(tmp_path, raw).packs[0].kernels[0].kernel_source
        again = _load(tmp_path, self._hand_authored(source.authored_document()))
        assert again.packs[0].kernels[0].kernel_source.launches == source.launches

    @pytest.mark.parametrize(
        "name",
        ["", "Merge", "2merge", "merge-2", "merge\n", "a b", "_a", 1, None],
        ids=repr,
    )
    def test_a_launch_name_outside_the_identifier_rule_is_rejected(
        self, tmp_path, name
    ):
        raw = self._with_launches({name: {}, "ok": {"role": "ok"}})
        with pytest.raises(ConfigError, match="not a lowercase identifier"):
            _load(tmp_path, raw)

    @pytest.mark.parametrize(
        "launches",
        [{}, {"only": {}}, {"only": {"role": "x"}}],
        ids=["empty", "one", "one_with_role"],
    )
    def test_fewer_than_two_launches_are_rejected(self, tmp_path, launches):
        with pytest.raises(ConfigError, match="at least 2"):
            _load(tmp_path, self._with_launches(launches))

    @pytest.mark.parametrize("launches", [None, [], ["a", "b"], "ab", 3])
    def test_launches_that_are_not_a_mapping_are_rejected(self, tmp_path, launches):
        with pytest.raises(ConfigError, match=r"launches must be a mapping"):
            _load(tmp_path, self._with_launches(launches))

    @pytest.mark.parametrize(
        "source",
        [
            {"kind": "hip", "source": "k.hip", "entry": "k"},
            {"kind": "hsaco", "file": "k.co", "symbol": "k"},
            {"kind": "embedded_source", "source_file": "k.cpp", "entry_point": "k"},
        ],
        ids=["hip", "hsaco", "embedded_source"],
    )
    def test_launches_are_rejected_on_every_other_kind(self, tmp_path, source):
        raw = self._hand_authored({**source, "launches": {"a": {}, "b": {"role": "b"}}})
        with pytest.raises(ConfigError, match="only valid on a 'rocke'"):
            _load(tmp_path, raw)

    def test_launches_in_kernel_defaults_are_rejected_for_a_non_rocke_kernel(
        self, tmp_path
    ):
        raw = self._hand_authored({"kind": "hip", "source": "k.hip", "entry": "k"})
        raw["packs"][0]["kernel_defaults"] = {"launches": {"a": {}, "b": {"role": "b"}}}
        with pytest.raises(ConfigError, match="only valid on a 'rocke'"):
            _load(tmp_path, raw)

    @pytest.mark.parametrize(
        "override, match",
        [
            ({"kind": "rocke"}, r"declares \['kind'\]"),
            ({"launches": {"a": {}}}, r"declares \['launches'\]"),
            ({"buidler": "x"}, r"declares \['buidler'\]"),
            ({"builder": 3}, r"builder must be a string"),
            ({"builder": ""}, r"supplies no 'builder'"),
            ({"role": 3}, r"role must be a string"),
            ({"role": "Merge"}, r"role is 'Merge'"),
            ({"role": "merge\n"}, r"role is 'merge\\n'"),
            ({"spec": "oops"}, r"spec must be a mapping"),
            ({"spec": {}}, r"supplies no 'spec'"),
            ({"target": "gfx942"}, r"target is 'gfx942'"),
            ("oops", r"must be a mapping"),
            (None, r"must be a mapping"),
        ],
    )
    def test_a_bad_launch_override_is_rejected(self, tmp_path, override, match):
        raw = self._with_launches({"a": override, "b": {"role": "b"}})
        with pytest.raises(ConfigError, match=match):
            _load(tmp_path, raw)

    def test_a_launch_override_cannot_change_a_matcher_visible_field(self, tmp_path):
        """One UKD is one selection: its metadata is stated once."""
        raw = self._with_launches(
            {"a": {}, "b": {"spec": {"block_m": 128, "head_size": 64}}}
        )
        with pytest.raises(ConfigError, match=r"states \['block_m'\]"):
            _load(tmp_path, raw)

    def test_a_launch_may_restate_a_matcher_visible_field_at_its_value(self, tmp_path):
        raw = self._with_launches({"a": {}, "b": {"spec": {"block_m": 256, "n": 1}}})
        (entry,) = self._emitted(tmp_path, raw)
        assert entry["sources"]["b"]["spec"] == {"block_m": 256, "n": 1}

    def test_two_launch_names_may_name_one_kernel(self, tmp_path):
        """A pipeline stage may run one kernel twice with different arguments, so
        equal launches are valid config and emit two entries naming one kernel."""
        (entry,) = self._emitted(tmp_path, self._with_launches({"a": {}, "b": {}}))
        assert sorted(entry["sources"]) == ["a", "b"]
        first, second = entry["sources"]["a"], entry["sources"]["b"]
        assert first == second
        assert first["builder"] == "build_k"

    def test_a_role_alone_tells_two_launches_apart(self, tmp_path):
        (entry,) = self._emitted(
            tmp_path, self._with_launches({"a": {"role": "a"}, "b": {"role": "b"}})
        )
        assert [s["role"] for s in entry["sources"].values()] == ["a", "b"]

    def test_role_is_not_a_key_of_a_plain_kernel_source(self, tmp_path):
        raw = self._hand_authored({**self.BASE, "role": "merge"})
        with pytest.raises(ConfigError, match=r"declares \['role'\]"):
            _load(tmp_path, raw)

    def test_a_multi_launch_source_has_no_single_document(self):
        ks = KernelSource(
            kind="rocke",
            source="k.py",
            builder="b",
            spec={"a": 1},
            launches={
                "x": KernelSource(
                    kind="rocke", source="k.py", builder="b", spec={"a": 1}
                ),
                "y": KernelSource(
                    kind="rocke", source="k.py", builder="b", spec={"a": 1}, role="y"
                ),
            },
        )
        with pytest.raises(ValueError, match="no single kernel_source object"):
            ks.as_document()
        assert sorted(ks.descriptor_fields()["sources"]) == ["x", "y"]
        assert ks.ukd_version == "1.1"

    @staticmethod
    def _identity_kernel(launches):
        from tests.helpers import make_kernel

        def source(role=""):
            return KernelSource(
                kind="rocke", source="m.py", builder="build_x", spec={"a": 1}, role=role
            )

        return make_kernel(
            kernel_source=KernelSource(
                kind="rocke",
                source="m.py",
                builder="build_x",
                spec={"a": 1},
                launches={name: source(role) for name, role in launches.items()},
            )
        )

    def test_launches_are_part_of_a_candidate_identity(self):
        from codegen.generator import _candidate_identity

        identity = lambda launches: _candidate_identity(self._identity_kernel(launches))
        base = {"a": "", "b": "b"}
        assert identity(base) == identity(dict(base))
        assert identity(base) != identity({"a": "", "b": "c"})
        assert identity(base) != identity({"a": "", "c": "b"})
        assert identity(base) != identity({"a": "", "b": "b", "c": "c"})
        assert identity(base) != identity({})

    def test_kernels_differing_only_in_launches_are_not_one_candidate(self, tmp_path):
        """Same metadata tuple and base source, so only the launches tell the two
        binaries apart: the dedup pass must refuse them rather than coalesce."""
        raw = _config(
            kernel_defaults={},
            kernels=[
                self._kernel(
                    "one", {**self.BASE, "launches": {"a": {}, "b": {"role": "b"}}}
                ),
                self._kernel(
                    "two", {**self.BASE, "launches": {"a": {}, "b": {"role": "c"}}}
                ),
            ],
        )
        with pytest.raises(ValueError, match="not the same candidate"):
            self._emitted(tmp_path, raw)

    def test_kernels_with_equal_launches_are_one_candidate(self, tmp_path):
        launches = {"a": {}, "b": {"role": "b"}}
        raw = _config(
            kernel_defaults={},
            kernels=[
                self._kernel("one", {**self.BASE, "launches": launches}),
                self._kernel("two", {**self.BASE, "launches": launches}),
            ],
        )
        entries = self._emitted(tmp_path, raw)
        assert [e["name"] for e in entries] == ["one"]
