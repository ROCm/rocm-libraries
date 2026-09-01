# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The converse of the desk check: can any graph SELECT this variant?

The desk check (`hkp_desk_check.py`) and the variant-set gate
(`verify_variant_sets.py`) both ask, in different ways, "does a shipped variant
match this graph?" and "is the set internally consistent?". Neither ever asks the
question backwards, and backwards is where dead weight hides: a real integration
shipped 48 variants of which 24 could not be selected by ANY graph the author
could write. Every shipped shape happened to have a sequence length divisible by
the wider of two tiles, so both tiles were always APPLICABLE and the scorer --
which ranks the wider tile higher -- picked it every single time. The suite was
green throughout; nothing had ever asked "for the narrow tile, is there a shape
where it wins?"

`TestGfx942AttentionDenseScore.RanksTheWiderKvTileHigher` and
`TestGfx942AttentionDenseKernelMatch.AcceptsEitherShippedTileForA256KeyGraph`
(TestGfx942AttentionDenseMatchers.cpp) are the real engine's own version of this
exact shape: applicability is `seqlen_kv % block_n == 0`, not equality, and both
shipped tiles are simultaneously legal at the shapes the corpus actually has.
`TestHistoricalCase` below reproduces that structure without a build or a device.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "variant_reachability.py"

# One KMD, shared by every test: a `dtype` field compared by equality and a
# `block_n` tile compared by divisibility (declared via --divides), mirroring the
# real engine's own shape-valued metadata split.
_KMD_FIELDS = [
    {"name": "dtype", "type": "string", "default_value": "bf16"},
    {"name": "block_n", "type": "int", "default_value": 64},
]


def _variant(name: str, block_n: int, dtype: str = "bf16") -> dict:
    return {"name": name, "metadata": {"dtype": dtype, "block_n": block_n}}


@pytest.fixture
def env(tmp_path):
    """Write a *.kdp.json/*.kmd.json bundle and a shape corpus; run the tool."""

    def write_bundle(variants: list[dict], fields=None) -> Path:
        kdp = tmp_path / "engine.kdp.json"
        kdp.write_text(json.dumps({"kernelDescriptors": variants}))
        kmd = tmp_path / "engine.kmd.json"
        kmd.write_text(json.dumps({"fields": fields or _KMD_FIELDS}))
        return kdp

    def write_shapes(shapes: list[dict]) -> Path:
        path = tmp_path / "shapes.json"
        path.write_text(json.dumps(shapes))
        return path

    def run(kdp: Path, shapes: Path, *extra) -> subprocess.CompletedProcess:
        argv = [
            sys.executable,
            str(_TOOL),
            "--kdp",
            str(kdp),
            "--shapes",
            str(shapes),
            *extra,
        ]
        return subprocess.run(argv, capture_output=True, text=True)

    return type(
        "Env",
        (),
        {
            "write_bundle": staticmethod(write_bundle),
            "write_shapes": staticmethod(write_shapes),
            "run": staticmethod(run),
            "tmp": tmp_path,
        },
    )


# Every corpus shape here is divisible by 64, so both tiles are always
# applicable -- exactly the property that hid the historical defect.
_DIVISIBLE_SHAPES = [
    {"dtype": "bf16", "seqlen_kv": 256},
    {"dtype": "bf16", "seqlen_kv": 512},
]

_RANKING = (
    "--divides",
    "block_n=seqlen_kv",
    "--score-field",
    "block_n",
    "--score-prefer",
    "max",
)


class TestControlPasses:
    """A bundle where every variant wins somewhere must pass. Every failure
    assertion below is worthless without this."""

    def test_single_variant_always_wins_by_itself(self, env):
        kdp = env.write_bundle([_variant("only", block_n=64)])
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASSED" in result.stdout
        assert "SELECTED                    1" in result.stdout

    def test_the_two_dtype_vocabularies_are_the_same_value(self, env):
        """Metadata says `BF16`; a request corpus says `bf16`. Same value.

        Comparing them raw made EVERY variant unreachable -- observed as 91 of 91
        on a set generated from the very corpus it was checked against. A false
        alarm that total is worse than no check: it teaches an author to pass
        --allow-unreachable and stop reading the output.

        The rest of this pipeline translates between the two spellings on purpose
        (the gate's `vocabulary:` block exists for exactly this), so a reachability
        check that does not is the one component still comparing apples to oranges.
        """
        kdp = env.write_bundle([_variant("upper", block_n=64, dtype="BF16")])
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)  # corpus carries "bf16"
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SELECTED                    1" in result.stdout, (
            "a variant differing from the corpus only in dtype SPELLING must still "
            "be reachable"
        )

    def test_a_genuinely_different_dtype_is_still_unreachable(self, env):
        """The converse, so the case above is not just 'compare nothing'."""
        kdp = env.write_bundle([_variant("wrongtype", block_n=64, dtype="FP8")])
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode != 0
        assert "UNREACHABLE" in result.stdout


class TestUnreachableVariant:
    """Applicable to no corpus shape at all -- either the corpus is missing a
    shape family or the variant should never have been built."""

    def test_a_tile_dividing_nothing_is_unreachable(self, env):
        # block_n=48 divides neither 256 nor 512.
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("orphan", block_n=48)]
        )
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 1
        assert "UNREACHABLE                 1" in result.stdout
        assert "orphan" in result.stdout


class TestHistoricalCase:
    """Two tiles, every corpus shape divisible by the wider one, scorer prefers
    the wider one -- the headline case this tool exists to catch."""

    def test_narrow_tile_is_applicable_but_never_wins(self, env):
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("narrow", block_n=32)]
        )
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 1
        assert "APPLICABLE-BUT-NEVER-WINS   1" in result.stdout
        assert (
            "narrow: applicable to 2 shape(s), always beaten by: wide" in result.stdout
        )
        # The diagnostic must say what actually fixes it: an illegal rival, not
        # more coverage of a shape both tiles already accept.
        assert "rival below is ILLEGAL" in result.stdout

    def test_adding_a_shape_where_the_wider_tile_is_illegal_flips_it_to_selected(
        self, env
    ):
        # 96 % 64 != 0 (wide is inapplicable there); 96 % 32 == 0 (narrow wins).
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("narrow", block_n=32)]
        )
        shapes = env.write_shapes(
            _DIVISIBLE_SHAPES + [{"dtype": "bf16", "seqlen_kv": 96}]
        )
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SELECTED                    2" in result.stdout
        assert "APPLICABLE-BUT-NEVER-WINS   0" in result.stdout


class TestNoRankingDeclared:
    """Without a declared ranking, applicable IS reachable by construction, and
    the output must say the ranking was never asked for -- a gate that silently
    stops checking a property is worse than one that admits it never checked."""

    def test_every_applicable_variant_is_reachable_and_it_says_so(self, env):
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("narrow", block_n=32)]
        )
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        # No --score-field, no --divides at all: block_n then compares by
        # equality and is simply never applicable to a differently-valued rival,
        # so both are "applicable to itself" trivially -- the point of this
        # test is the declared-ranking message, not the bucket counts.
        result = env.run(kdp, shapes)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "NO RANKING DECLARED" in result.stdout
        assert "did NOT verify which one the native scorer would actually pick" in (
            result.stdout
        )


class TestAllowUnreachableFlag:
    def test_flag_suppresses_the_exit_code_but_keeps_the_report(self, env):
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("narrow", block_n=32)]
        )
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING, "--allow-unreachable")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "APPLICABLE-BUT-NEVER-WINS   1" in result.stdout
        assert "narrow" in result.stdout


class TestBundleDiscovery:
    def test_missing_kmd_sibling_is_a_clean_failure(self, env, tmp_path):
        kdp = tmp_path / "engine.kdp.json"
        kdp.write_text(json.dumps({"kernelDescriptors": [_variant("only", 64)]}))
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes)
        assert result.returncode == 2
        assert "no sibling" in result.stdout + result.stderr
