# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for tools/factorise_config.py.

The tool's contract is one sentence: the compact config it writes expands to the
enumeration it was given, kernel-for-kernel and key-for-key. Everything here either
checks that contract directly or checks one thing that would silently break it.
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import factorise_config  # noqa: E402


def _enumerated(kernels: list) -> dict:
    return {
        "dialect": "packaged",
        "kernel_source_kind": "rocke",
        "engine": {"name": "hipkernel:Attn", "knobs": ["block_m"]},
        "kmd_fields": [
            {"name": "dtype", "type": "string"},
            {"name": "seqlen_q", "type": "int", "default_value": 256},
            {"name": "block_m", "type": "int", "default_value": 256},
            {"name": "use_exp2_fast", "type": "int", "default_value": -1},
        ],
        "packs": [{"name": "attn", "arch": ["gfx942"], "kernels": kernels}],
    }


def _kernel(name: str, spec: dict, metadata: dict) -> dict:
    return {
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "source": "kernels/gfx942/attention_dense.py",
            "builder": "build_attention_dense",
            "spec": spec,
        },
        "metadata": metadata,
    }


#: Two shapes x two block_m arms, with the tri-state appearing in three of its four
#: states -- pinned on, pinned off, and left to the kernel's own policy.
SAMPLE = _enumerated(
    [
        _kernel(
            "attn.bf16_sq512_bm128_e1",
            {"dtype": "bf16", "seqlen_q": 512, "block_m": 128, "use_exp2_fast": True},
            {"dtype": "BF16", "seqlen_q": 512, "block_m": 128, "use_exp2_fast": 1},
        ),
        _kernel(
            "attn.bf16_sq512_bm256_e0",
            {"dtype": "bf16", "seqlen_q": 512, "block_m": 256, "use_exp2_fast": False},
            {"dtype": "BF16", "seqlen_q": 512, "block_m": 256, "use_exp2_fast": 0},
        ),
        _kernel(
            "attn.bf16_sq1024_bm128_e1",
            {"dtype": "bf16", "seqlen_q": 1024, "block_m": 128},
            {"dtype": "BF16", "seqlen_q": 1024, "block_m": 128, "use_exp2_fast": 1},
        ),
        _kernel(
            "attn.bf16_sq1024_bm256_e1",
            {"dtype": "bf16", "seqlen_q": 1024, "block_m": 256},
            {"dtype": "BF16", "seqlen_q": 1024, "block_m": 256, "use_exp2_fast": 1},
        ),
    ]
)

KNOBS = ["block_m", "use_exp2_fast"]
VOCABULARY = {"dtype": {"bf16": "BF16", "fp16": "FP16"}}


def _compact(config: dict = SAMPLE, knobs: list = KNOBS) -> dict:
    return factorise_config.factorise(config, knobs, VOCABULARY)


def _expand(tmp_path, compact: dict):
    path = tmp_path / "compact.yaml"
    path.write_text(factorise_config.dump(compact))
    return load_config(path).packs[0].kernels


def _enumerate(tmp_path, config: dict):
    path = tmp_path / "long.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return load_config(path).packs[0].kernels


class TestRoundTrip:
    """The contract. Everything else in this file is a way of losing it."""

    def test_compact_form_expands_to_the_same_kernels(self, tmp_path):
        got = _expand(tmp_path, _compact())
        want = _enumerate(tmp_path, SAMPLE)
        assert [k.name for k in got] == [k.name for k in want]
        assert [k.metadata for k in got] == [k.metadata for k in want]
        assert [k.kernel_source.spec for k in got] == [
            k.kernel_source.spec for k in want
        ]

    def test_spec_key_order_survives(self, tmp_path):
        """Key order reaches the emitted descriptor, so it is part of the bytes."""
        got = _expand(tmp_path, _compact())
        want = _enumerate(tmp_path, SAMPLE)
        assert [list(k.kernel_source.spec) for k in got] == [
            list(k.kernel_source.spec) for k in want
        ]

    def test_kernel_order_survives(self, tmp_path):
        """Descriptor ids are assigned by position, so a reordering is a different
        descriptor set rather than a cosmetic difference."""
        got = _expand(tmp_path, _compact())
        assert [k.name for k in got] == [
            k["name"] for k in SAMPLE["packs"][0]["kernels"]
        ]

    def test_the_tool_refuses_a_compaction_that_does_not_round_trip(self):
        """The check is not advisory: an unrepresentable set must fail, never ship."""
        broken = _compact()
        broken["packs"][0]["variants"][0]["shapes"][0]["seqlen_q"] = 99
        with pytest.raises(factorise_config.FactoriseError, match="round trip"):
            factorise_config._round_trip(SAMPLE, broken)


class TestHoisting:
    def test_constant_kernel_source_keys_move_to_kernel_defaults(self):
        """`kind`, `source` and `builder` are byte-identical in every entry."""
        pack = _compact()["packs"][0]
        assert pack["kernel_defaults"] == {
            "kind": "rocke",
            "source": "kernels/gfx942/attention_dense.py",
            "builder": "build_attention_dense",
        }
        assert "kernels" not in pack

    def test_a_kernels_own_key_still_wins_over_a_default(self, tmp_path):
        config = _enumerated(
            [
                _kernel(
                    "attn.bf16_sq512_bm128_e1",
                    {"dtype": "bf16", "seqlen_q": 512, "block_m": 128},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 512,
                        "block_m": 128,
                        "use_exp2_fast": 1,
                    },
                )
            ]
        )
        config["packs"][0]["kernel_defaults"] = {"spec": {"block_m": 999}}
        kernels = _enumerate(tmp_path, config)
        assert kernels[0].kernel_source.spec["block_m"] == 128


class TestKnobSets:
    def test_shapes_sharing_an_arm_list_share_one_knob_set(self):
        """The whole point: 655 shapes, a handful of distinct arm lists."""
        group = _compact()["packs"][0]["variants"][0]
        assert len(group["shapes"]) == 2
        assert len(group["knob_sets"]) == 2

    def test_shapes_with_different_arm_counts_get_different_sets(self):
        config = _enumerated(
            SAMPLE["packs"][0]["kernels"]
            + [
                _kernel(
                    "attn.bf16_sq2048_bm128_e1",
                    {"dtype": "bf16", "seqlen_q": 2048, "block_m": 128},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 2048,
                        "block_m": 128,
                        "use_exp2_fast": 1,
                    },
                )
            ]
        )
        sets = _compact(config)["packs"][0]["variants"][0]["knob_sets"]
        assert sorted(len(arms) for arms in sets.values()) == [1, 2, 2]


class TestTriState:
    def test_a_policy_decided_knob_stays_absent_from_the_arm(self):
        """Absent means the kernel's policy decides at build time. Writing the
        resolved value into the spec instead would pin a different binary."""
        group = _compact()["packs"][0]["variants"][0]
        policy_arms = [
            arm
            for arms in group["knob_sets"].values()
            for arm in arms
            if "use_exp2_fast" not in arm
        ]
        assert policy_arms, "the sample carries policy-decided arms"

    def test_a_policy_decided_shape_records_what_the_policy_chose(self):
        group = _compact()["packs"][0]["variants"][0]
        policy_shapes = [s for s in group["shapes"] if "resolved" in s]
        assert policy_shapes
        assert policy_shapes[0]["resolved"] == {"use_exp2_fast": 1}

    def test_policy_knobs_names_the_tri_state(self):
        group = _compact()["packs"][0]["variants"][0]
        assert group["policy_knobs"] == ["use_exp2_fast"]

    def test_pinned_false_and_absent_stay_distinguishable(self, tmp_path):
        got = _expand(tmp_path, _compact())
        by_name = {k.name: k for k in got}
        assert (
            by_name["attn.bf16_sq512_bm256_e0"].kernel_source.spec["use_exp2_fast"]
            is False
        )
        assert (
            "use_exp2_fast"
            not in by_name["attn.bf16_sq1024_bm256_e1"].kernel_source.spec
        )


class TestNameTemplates:
    def test_a_field_binds_only_when_it_matches_every_entry(self):
        """A coincidence is not a binding.

        `block_m` and `seqlen_q` could each explain a single entry's token; only the
        field that explains ALL of them may claim the slot, because a template
        inferred from a subset renders the wrong name on the next input.
        """
        template = _compact()["packs"][0]["variants"][0]["name"]
        assert "{seqlen_q}" in template
        assert "{block_m}" in template

    def test_a_field_that_coincides_on_only_the_first_entry_does_not_bind(self):
        """The DISCRIMINATING case for "matches every entry".

        Entry 0's `bm128` agrees with `block_m`. Entry 1's `bm999` agrees with
        nothing -- its block_m is 256. A binder that checked only the first entry
        would bind `bm{block_m}` and render entry 1 as `bm256`: a name that is not
        its own, and a silent rename of a shipped kernel.

        Sound behaviour is to refuse the binding and let the token fall into the
        per-arm `tag`, which is what the assertions below pin.
        """
        config = _enumerated(
            [
                _kernel(
                    "attn.bf16_bm128_e1",
                    {"dtype": "bf16", "seqlen_q": 512, "block_m": 128},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 512,
                        "block_m": 128,
                        "use_exp2_fast": 1,
                    },
                ),
                _kernel(
                    "attn.bf16_bm999_e1",
                    {"dtype": "bf16", "seqlen_q": 512, "block_m": 256},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 512,
                        "block_m": 256,
                        "use_exp2_fast": 1,
                    },
                ),
            ]
        )
        compact = _compact(config)
        template = compact["packs"][0]["variants"][0]["name"]
        assert "bm{block_m}" not in template
        assert "{tag}" in template
        # And the conversion is genuinely lossless, not merely differently wrong.
        factorise_config._round_trip(config, compact)

    def test_the_template_prefers_the_field_the_token_is_named_after(self):
        """`sq512` means seqlen_q, even when another field happens to be 512 too."""
        config = _enumerated(
            [
                _kernel(
                    "attn.bf16_sq512_bm512_e1",
                    {"dtype": "bf16", "seqlen_q": 512, "block_m": 512},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 512,
                        "block_m": 512,
                        "use_exp2_fast": 1,
                    },
                ),
                _kernel(
                    "attn.bf16_sq256_bm256_e1",
                    {"dtype": "bf16", "seqlen_q": 256, "block_m": 256},
                    {
                        "dtype": "BF16",
                        "seqlen_q": 256,
                        "block_m": 256,
                        "use_exp2_fast": 1,
                    },
                ),
            ]
        )
        template = _compact(config)["packs"][0]["variants"][0]["name"]
        assert "sq{seqlen_q}" in template
        assert "bm{block_m}" in template

    def test_every_expanded_name_is_unique(self, tmp_path):
        got = _expand(tmp_path, _compact())
        assert len({k.name for k in got}) == len(got)


class TestRefusals:
    def test_a_non_default_priority_is_refused(self):
        """A variants group emits no priority key, so compacting one would reset it."""
        config = copy.deepcopy(_enumerated(SAMPLE["packs"][0]["kernels"]))
        config["packs"][0]["kernels"][0]["priority"] = 5
        with pytest.raises(factorise_config.FactoriseError, match="priority"):
            _compact(config)

    def test_a_multi_pack_config_is_refused(self):
        config = _enumerated(list(SAMPLE["packs"][0]["kernels"]))
        config["packs"].append(dict(config["packs"][0], name="other"))
        with pytest.raises(factorise_config.FactoriseError, match="single-pack"):
            _compact(config)

    def test_an_empty_pack_is_refused(self):
        with pytest.raises(factorise_config.FactoriseError, match="no kernels"):
            _compact(_enumerated([]))


class TestRendering:
    def test_a_shape_is_one_line(self):
        text = factorise_config.dump(_compact())
        shapes = [line for line in text.splitlines() if line.lstrip().startswith("- {")]
        assert shapes
        assert all(line.rstrip().endswith("}") for line in shapes)

    def test_no_yaml_anchors_are_emitted(self):
        """An anchor saves four lines and costs the reader a cross-reference."""
        text = factorise_config.dump(_compact())
        assert "&id" not in text
        assert "*id" not in text

    def test_the_compact_form_is_much_shorter(self):
        long_form = yaml.safe_dump(SAMPLE, sort_keys=False)
        compact = factorise_config.dump(_compact())
        assert len(compact.splitlines()) < len(long_form.splitlines())


class TestCommandLine:
    """The CLI is what every real user goes through, so it needs its own coverage.

    Calling `factorise`/`_round_trip` directly proves those functions work; it does
    not prove `main` WIRES them together. Deleting the round-trip guard from `main`
    is invisible to a test that calls the guard itself.
    """

    def _write(self, tmp_path, config):
        path = tmp_path / "long.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def _argv(self, src, out):
        return [
            "--config",
            str(src),
            "--out",
            str(out),
            "--knobs",
            ",".join(KNOBS),
            "--vocabulary",
            "dtype:bf16=BF16,fp16=FP16",
        ]

    def test_main_writes_a_config_that_expands_to_the_input(self, tmp_path):
        src = self._write(tmp_path, SAMPLE)
        out = tmp_path / "compact.yaml"
        assert factorise_config.main(self._argv(src, out)) == 0
        got = load_config(out).packs[0].kernels
        want = _enumerate(tmp_path, SAMPLE)
        assert [k.name for k in got] == [k.name for k in want]
        assert [k.metadata for k in got] == [k.metadata for k in want]

    def test_main_refuses_to_write_when_the_round_trip_fails(self, tmp_path):
        """The guard must be WIRED INTO the CLI, not merely exist.

        Omitting `--vocabulary` when the set needs it is the realistic way to get a
        lossy conversion: the metadata keeps the builder's spelling (`bf16`) where
        the matcher expects hipDNN's (`BF16`), which loads cleanly and matches
        nothing. `main` must exit non-zero and leave no file behind rather than
        write a config that generates a different descriptor set.
        """
        src = self._write(tmp_path, SAMPLE)
        out = tmp_path / "compact.yaml"
        argv = self._argv(src, out)
        del argv[argv.index("--vocabulary") : argv.index("--vocabulary") + 2]
        assert factorise_config.main(argv) != 0
        assert not out.exists()

    def test_main_parses_the_vocabulary_flag(self, tmp_path):
        src = self._write(tmp_path, SAMPLE)
        out = tmp_path / "compact.yaml"
        assert factorise_config.main(self._argv(src, out)) == 0
        group = yaml.safe_load(out.read_text())["packs"][0]["variants"][0]
        assert group["vocabulary"]["dtype"]["bf16"] == "BF16"
