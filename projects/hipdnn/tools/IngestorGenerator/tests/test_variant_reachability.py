# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The converse of the desk check: can any graph SELECT this variant?

The desk check and the variant-set gate both ask whether a shipped variant matches a
graph and whether the set is internally consistent. Backwards is where dead weight
hides: when every shipped shape is divisible by the wider of two tiles, both are always
APPLICABLE, the scorer picks the wider one, and half the set can be selected by no
graph while the suite stays green. Applicability in the real engine is
`seqlen_kv % block_n == 0`, not equality, so both shipped tiles are legal at once.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "variant_reachability.py"

sys.path.insert(0, str(_TOOLS))

import variant_reachability  # noqa: E402

# One KMD, shared by every test: a `dtype` field compared by equality and a `block_n`
# tile compared by divisibility (via --divides), mirroring the real engine's split.
_KMD_FIELDS = [
    {"name": "dtype", "type": "string", "default_value": "bf16"},
    {"name": "block_n", "type": "int", "default_value": 64},
]

_KMD_ID = "88888888-8888-8888-8888-888888888888"
_UED_ID = "99999999-9999-9999-9999-999999999999"


def _variant(name: str, block_n: int, dtype: str = "bf16") -> dict:
    return {"name": name, "metadata": {"dtype": dtype, "block_n": block_n}}


@pytest.fixture
def env(tmp_path):
    """Write an id-wired descriptor bundle and a shape corpus; run the tool. The schema
    is reached through `KDP.engine -> UED.metadata -> KMD`, so the fixture carries the
    UED that links them."""

    def write_bundle(variants: list[dict], fields=None) -> Path:
        kdp = tmp_path / "engine.kdp.json"
        kdp.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "engine": _UED_ID,
                    "kernelDescriptors": variants,
                }
            )
        )
        (tmp_path / "engine.ued.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "id": _UED_ID,
                    "name": "test:Engine",
                    "metadata": _KMD_ID,
                }
            )
        )
        kmd = tmp_path / "engine.kmd.json"
        kmd.write_text(json.dumps({"id": _KMD_ID, "fields": fields or _KMD_FIELDS}))
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


# Every corpus shape here is divisible by 64, so both tiles are always applicable.
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
    """A bundle where every variant wins somewhere must pass; every failure assertion
    below is worthless without it."""

    def test_single_variant_always_wins_by_itself(self, env):
        kdp = env.write_bundle([_variant("only", block_n=64)])
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, *_RANKING)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASSED" in result.stdout
        assert "SELECTED                    1" in result.stdout

    def test_the_two_dtype_vocabularies_are_the_same_value(self, env):
        """Metadata says `BF16`, the corpus says `bf16`, and the pipeline translates
        between them on purpose (the gate's `vocabulary:` block), so a raw comparison
        reports EVERY variant unreachable."""
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
    """Applicable to no corpus shape at all: either the corpus is missing a shape family
    or the variant should never have been built."""

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
    """Two tiles, every corpus shape divisible by the wider one, scorer prefers the
    wider: the headline case this tool exists to catch."""

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
    """Without a declared ranking, applicable IS reachable by construction, so the
    output must say the ranking was never asked for."""

    def test_every_applicable_variant_is_reachable_and_it_says_so(self, env):
        kdp = env.write_bundle(
            [_variant("wide", block_n=64), _variant("narrow", block_n=32)]
        )
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        # With no --divides, block_n compares by equality, so each variant is only
        # "applicable to itself" -- this test is about the declared-ranking message,
        # not the bucket counts.
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


class TestTheSchemaIsReachedByReference:
    """`default_value` decides what an absent metadata key resolves to, and that decides
    applicability, so reaching the schema by filename suffix scores a directory holding
    two bundles against the wrong defaults."""

    def test_a_correctly_wired_bundle_resolves(self, env):
        kdp = env.write_bundle([_variant("only", 64)])
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes, "--divides", "block_n=seqlen_kv")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_dangling_engine_reference_is_a_clean_failure(self, env, tmp_path):
        kdp = env.write_bundle([_variant("only", 64)])
        doc = json.loads(kdp.read_text())
        doc["engine"] = "no-such-ued"
        kdp.write_text(json.dumps(doc))
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes)
        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert "engine" in combined
        assert "no-such-ued" in combined

    def test_a_same_stem_kmd_that_nothing_references_is_not_accepted(
        self, env, tmp_path
    ):
        kdp = env.write_bundle([_variant("only", 64)])
        (tmp_path / "engine.ued.json").unlink()
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes)
        assert result.returncode == 2
        assert "engine" in result.stdout + result.stderr
