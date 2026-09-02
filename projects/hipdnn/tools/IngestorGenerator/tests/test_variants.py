# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the compact `variants` config form.

The property under test throughout is that the compact form and the enumeration it
stands for generate the SAME descriptors. Everything below defends one way that can
silently stop being true.
"""

import pytest
import yaml

from codegen.config_loader import ConfigError, load_config

#: The tri-state's own KMD field. `default_value: -1` because no real kernel takes
#: it: 0 is a value the policy genuinely returns, so a 0 default would alias
#: "policy decided off" onto "explicitly off" and collapse two catalog entries.
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
        """Not a global cross-product.

        The shipped sets are not a full grid -- most shapes carry four arms and a
        minority carry six -- so a format that crossed one knob set over every shape
        would silently invent variants for some and drop them for others.
        """
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
        """Key order reaches the descriptor bytes, so it is part of the contract."""
        kernels = _kernels(tmp_path, spec_order=["block_m", "seqlen_q", "dtype"])
        assert list(kernels[0].kernel_source.spec) == ["block_m", "seqlen_q", "dtype"]

    def test_metadata_uses_the_matcher_vocabulary(self, tmp_path):
        """The spec carries the builder's spelling, metadata the matcher's.

        Copying one over the other declines every graph while the engine still loads
        and every count reconciles -- the failure with no symptom.
        """
        kernel = _kernels(tmp_path)[0]
        assert kernel.kernel_source.spec["dtype"] == "bf16"
        assert kernel.metadata["dtype"] == "BF16"

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


class TestTriState:
    """`use_exp2_fast` lives in three layers that must agree.

    The spec decides the compiled binary -- ABSENT means the kernel's own policy
    resolves it at build time. The metadata is what the matcher compares. The KMD
    `default_value` is substituted for anything absent at load. All four
    spec/metadata combinations occur in shipped data, and collapsing "absent" onto
    "explicitly false" throws away the policy.
    """

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
        """Same metadata, DIFFERENT binary. The distinction is the whole point.

        Both reach the matcher as 0, so the metadata mirror cannot tell them apart
        and the name has to: each arm carries its own tag. That is not a quirk of
        this test -- it is why the shipped grammar spells policy-decided-off `ed` and
        pinned-off `e0`. Only the spec says which binary was built, and an arm that
        omits the knob must keep omitting it.
        """
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
        """The discriminating case for the precedence order.

        When a field is in BOTH the arm's `metadata` and its spec, only one order is
        right: the arm's stated metadata is what the MATCHER compares, and the spec
        is what the binary was built from. They are allowed to differ -- that is how
        a knob gets swept in the catalog over a spec the dispatcher fixed -- so the
        loader must not overwrite the stated value with the spec's.

        Without this the two branches are indistinguishable: every other test has the
        field in exactly one of the two places, so swapping the branch order changes
        nothing and no test notices.
        """
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
        """Silence here means the loader substitutes the KMD default as the catalog
        key while the binary was built from the policy's answer."""
        with pytest.raises(ConfigError, match="policy"):
            _kernels(
                tmp_path,
                knob_sets={"one": [{"block_m": 128}]},
                shapes=[{"dtype": "bf16", "seqlen_q": 512, "knobs": "one"}],
            )

    def test_an_arm_can_pin_metadata_the_spec_does_not_carry(self, tmp_path):
        """Two catalog entries over ONE binary.

        The dispatcher returns the shared spec and leaves arch-private knobs to the
        kernel's policy, so sweeping such a knob pins what the MATCHER compares
        without changing what is compiled. Both arms are real descriptors and the
        format has to be able to say that -- writing the pin into the spec instead
        would claim a binary that was never built.
        """
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
        """Shipped metadata spelled this knob four ways -- 0, 1, False, True."""
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
    """Nothing downstream catches a kernel-name collision.

    The loader rejects a pack whose expansion produces duplicate names, because
    nothing after it would: the dedup pass keys on metadata rather than name, so a
    collision that got through would ship as two descriptors impossible to tell apart
    in a log, a winner record or a failure message. A
    previous version hardcoded a subset of attention's field names and gave two
    distinct conv variants the same name.
    """

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
        """Pinned-on and policy-on share a metadata value, so the name must come
        from somewhere else -- here the arm's own tag."""
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
        """Its cross-product is empty, which expands the shape to ZERO kernels."""
        with pytest.raises(ConfigError, match="non-empty"):
            _kernels(tmp_path, knob_sets={"pair": []})

    def test_a_metadata_field_no_kmd_field_declares_is_rejected(self, tmp_path):
        """An undeclared metadata field drops the WHOLE pack at
        resolveDescriptorSets(), so expansion must not manufacture one."""
        with pytest.raises(ConfigError, match="kmd_fields"):
            _kernels(tmp_path, metadata=["dtype", "block_m", "use_exp2_fast", "nope"])

    def test_two_shapes_rendering_one_name_are_rejected(self, tmp_path):
        """The guarantee the whole naming discipline rests on.

        Nothing after the loader catches this: the dedup pass keys on metadata, so a
        collision that got through ships as two descriptors nothing can tell apart.
        Here the template omits seqlen_q, the only field the two shapes differ in.
        """
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
        """Eliding the empty `{tag}` is a TEMPLATE operation, not a text fixup.

        Squeezing the rendered name cannot tell its own separator from one inside a
        value, so it collapses `a__b` to `a_b` -- and two shapes whose dtypes are
        `a__b` and `a_b` then land on one name.
        """
        kernels = _kernels(
            tmp_path,
            name="attn.{dtype}_bm{block_m}_{tag}",
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
        binary the descriptor names while the config still loads."""
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
        """An unmatched brace raises ValueError from str.format, not KeyError, so it
        escaped the diagnostic and reached the author as a bare traceback."""
        with pytest.raises(ConfigError, match="could not be rendered"):
            _kernels(tmp_path, name="attn.{dtype")

    def test_a_metadata_field_nothing_decides_is_rejected(self, tmp_path):
        """`seqlen_q` IS a declared KMD field, so the loader would otherwise
        substitute its default_value as the catalog key for a binary built from
        something else entirely."""
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
