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

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "variant_reachability.py"

sys.path.insert(0, str(_TOOLS))

import variant_reachability  # noqa: E402
from launch_surface import find_repo_root  # noqa: E402

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


class TestGfx950RealBundle:
    """The real 84-variant gfx950 bundle against the real 93-shape corpus --
    the case the module docstring and `TestHistoricalCase` above only model in
    miniature. Nothing here needs a device or a build: `.kdp.json`/`.kmd.json`
    are committed descriptor JSON and the corpus is a committed shape list.

    A second-pass review found that `gfx950_attention_dense.profile.yaml` had
    no `score:` block, so this exact bundle ran NARROWED -- every applicable
    variant reported reachable without ever checking which one the native
    scorer would pick. The review also established, by actually computing
    `applicable()` over all 84 descriptors x 93 shapes, that the gap was inert:
    no shape has more than one applicable variant, so no ranking, tie, or
    fall-through is ever evaluated by this bundle today. The tests below pin
    both halves of that finding so a second `block_n` (or any change that
    makes ranking start to matter) cannot silently go unranked again: the
    first test fails the day the corpus and bundle stop being that narrow,
    and the second fails if the declared ranking ever disagrees with the
    narrowed (unranked) verdict while the bundle is still that narrow.
    """

    _REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
    _KDP = (
        _REPO_ROOT
        / "dnn-providers/hip-kernel-provider/descriptor-packaging/examples"
        / "descriptors/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json"
    )
    _SHAPES = (
        Path(__file__).resolve().parents[1]
        / "configs/gfx950_attention_dense.shapes.json"
    )
    _PROFILE = (
        Path(__file__).resolve().parents[1]
        / "configs/gfx950_attention_dense.profile.yaml"
    )
    _FIELD_MAP_AND_DIVIDES = (
        "--field-map",
        "nhead_q=num_query_heads",
        "--field-map",
        "nhead_k=num_kv_heads",
        "--field-map",
        "seqlen_k=seqlen_kv",
        "--field-map",
        "hdim_q=head_size",
        "--divides",
        "block_n=seqlen_kv",
    )

    @classmethod
    def _require_assets(cls):
        """The bundle, the corpus and the profile are gfx950 deliverables that
        exist only on a branch carrying that pack. On a checkout without them --
        the tooling branch these shared tests also run on -- there is nothing to
        assert about, so skip rather than fail: an absent asset is a branch fact,
        not a regression. `TestHistoricalCase` above models the same property in
        miniature and runs everywhere, so skipping here does not leave the
        behaviour unguarded."""
        for label, path in (
            ("gfx950_attention_dense.kdp.json", cls._KDP),
            ("gfx950_attention_dense.shapes.json", cls._SHAPES),
            ("gfx950_attention_dense.profile.yaml", cls._PROFILE),
        ):
            if not path.exists():
                pytest.skip(f"{label} not present in this checkout")

    def test_no_shape_has_more_than_one_applicable_variant_today(self):
        """The inertness claim, checked directly rather than inferred from the
        tool's own report: with 84 descriptors sharing one `block_n` value, if
        any corpus shape ever admits two applicable variants this assertion is
        the first thing to fail, which is exactly when a declared ranking
        starts doing real work instead of being a no-op."""
        self._require_assets()
        defaults, descriptors = variant_reachability.load_bundle(str(self._KDP))
        shapes = json.loads(self._SHAPES.read_text())
        field_map = {
            "nhead_q": "num_query_heads",
            "nhead_k": "num_kv_heads",
            "seqlen_k": "seqlen_kv",
            "hdim_q": "head_size",
        }
        divides = {"block_n": "seqlen_kv"}
        metas = {
            d["name"]: variant_reachability._resolved_metadata(d, defaults)
            for d in descriptors
        }
        remapped = [variant_reachability._remap(s, field_map) for s in shapes]
        counts = [
            sum(
                1
                for meta in metas.values()
                if variant_reachability.applicable(meta, s, divides)
            )
            for s in remapped
        ]
        assert max(counts) <= 1, (
            "a corpus shape now admits more than one applicable variant -- the "
            "declared `score:` block in gfx950_attention_dense.profile.yaml is "
            "no longer a no-op and its correctness needs to be re-verified, "
            "not assumed"
        )

    def test_declared_ranking_matches_the_narrowed_verdict(self, tmp_path):
        """Runs the real tool twice against the real bundle: once with no
        ranking declared (narrowed), once with the profile's `score:` block.
        Both must land on the identical 82 SELECTED / 0 APPLICABLE-BUT-NEVER-
        WINS / 2 UNREACHABLE verdict established by the review -- proof that
        declaring the ranking changed nothing observable today, which is the
        claim the profile comment makes."""
        self._require_assets()
        narrowed = subprocess.run(
            [
                sys.executable,
                str(_TOOL),
                "--kdp",
                str(self._KDP),
                "--shapes",
                str(self._SHAPES),
                *self._FIELD_MAP_AND_DIVIDES,
            ],
            capture_output=True,
            text=True,
        )
        declared = subprocess.run(
            [
                sys.executable,
                str(_TOOL),
                "--kdp",
                str(self._KDP),
                "--shapes",
                str(self._SHAPES),
                "--profile",
                str(self._PROFILE),
                *self._FIELD_MAP_AND_DIVIDES,
            ],
            capture_output=True,
            text=True,
        )
        assert narrowed.returncode == declared.returncode
        assert "NO RANKING DECLARED" in narrowed.stdout
        assert "NO RANKING DECLARED" not in declared.stdout
        assert "ranking declared  block_n (max wins)" in declared.stdout
        for out in (narrowed.stdout, declared.stdout):
            assert "SELECTED                    82" in out
            assert "APPLICABLE-BUT-NEVER-WINS   0" in out
            assert "UNREACHABLE                 2" in out
