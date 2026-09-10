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
import re
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

_KMD_ID = "88888888-8888-8888-8888-888888888888"
_UED_ID = "99999999-9999-9999-9999-999999999999"


def _variant(name: str, block_n: int, dtype: str = "bf16") -> dict:
    return {"name": name, "metadata": {"dtype": dtype, "block_n": block_n}}


@pytest.fixture
def env(tmp_path):
    """Write an id-wired descriptor bundle and a shape corpus; run the tool.

    The bundle's schema is reached through `KDP.engine -> UED.metadata -> KMD`,
    the same chain the loader and the variant-set gate walk, so the fixture has to
    carry the UED that links them. A KDP and a KMD merely sharing a filename stem
    are two unrelated documents here, which is the point: scoring a bundle against
    whatever schema sits beside it decides every default and every applicability
    verdict with a coincidence of naming.
    """

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


class TestTheSchemaIsReachedByReference:
    """The bundle's KMD is found by walking the ids the documents declare.

    Everything below depends on the KMD: `default_value` decides what an absent
    metadata key resolves to, and that resolution decides applicability. Reaching
    the schema by swapping a filename suffix on the same stem picks whichever
    document happens to sit beside the KDP, so a tree with two unrelated bundles
    in one directory -- or one whose real schema lives above it -- is scored
    against the wrong defaults and every verdict here is about a bundle nobody
    ships.
    """

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
        """The defect this replaces: a KDP and a KMD sharing a stem, with nothing
        in either document connecting them."""
        kdp = env.write_bundle([_variant("only", 64)])
        (tmp_path / "engine.ued.json").unlink()
        shapes = env.write_shapes(_DIVISIBLE_SHAPES)
        result = env.run(kdp, shapes)
        assert result.returncode == 2
        assert "engine" in result.stdout + result.stderr


class TestGfx950RealBundle:
    """The real gfx950 bundle against the real 93-shape corpus -- the case the
    module docstring and `TestHistoricalCase` above only model in miniature.
    Nothing here needs a device or a build: `.kdp.json`/`.kmd.json` are
    committed descriptor JSON and the corpus is a committed shape list.

    WHAT THESE USED TO ASSERT, AND WHY THAT EXPIRED. When the pack shipped 84
    variants, a second-pass review established that no corpus shape had more
    than ONE applicable variant, so the profile's missing `score:` block was
    inert -- no ranking, tie, or fall-through was ever evaluated. These tests
    pinned that with literal counts (84 variants, `SELECTED 82`, `UNREACHABLE
    2`) and an `applicable() <= 1` assertion.

    `329865eb878` ("full coverage + block_m variants, 1612 kernels") ENDED that
    inertness ON PURPOSE: it ships both `block_m` geometries per shape, so every
    covered shape now admits exactly the twin. `<= 1` therefore had to fail, and
    the literal counts moved with every resize (84 -> 1612 -> 1678 -> 1694).
    Those literals were fixture, not invariant, and they were asserting the
    absence of the very feature that commit added.

    WHAT ACTUALLY HAS TO HOLD, and what these now check instead -- derived from
    the bundle rather than hard-coded, so a resize cannot make them stale again:

      1. Every candidate set a shape presents is TIED ON `block_n`. That is the
         real precondition for the native `scoreKernel`, which ranks on
         `block_n` ALONE (Gfx950AttentionDenseNative.cpp): while all candidates
         for a shape share one `block_n`, the declared ranking cannot reorder
         them and the choice falls to tuning. The day a second `block_n` reaches
         one shape's candidate set, `score` starts doing real work and its
         correctness needs verifying rather than assuming -- which is the alarm
         the old `<= 1` was reaching for, stated in terms of the thing that
         actually matters.
      2. Declaring the ranking still changes NOTHING observable: the narrowed
         and declared runs must agree line for line. That is the profile
         comment's claim, and it survives the resize untouched.

    Consequence worth stating, because it is the measurement trap: tied
    candidates mean the shipped set is only as good as its TIE-BREAK unless
    something measures them. That is why every sweep over this pack must export
    `HIPDNN_FORCE_BENCHMARKING=1` (Knowledge/hipdnn/gpu-perf-comparison-
    methodology.md trap 3 -- unforced, the identical set measured 1.195 -> 0.999).
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

    def test_every_candidate_set_is_tied_on_block_n(self):
        """The precondition for the native `scoreKernel`, checked directly
        rather than inferred from the tool's own report.

        `scoreKernel` ranks on `block_n` ALONE. So long as every candidate set a
        shape presents holds ONE distinct `block_n`, the declared ranking is a
        tie it cannot break, and which variant runs is decided by tuning rather
        than by `score`. That is the property the pack's own comment says to
        re-check whenever the variant set changes, and it is what makes
        `HIPDNN_FORCE_BENCHMARKING=1` mandatory for any perf claim about it.

        This deliberately does NOT assert `len(candidates) <= 1`. Shipping both
        `block_m` geometries per shape (329865eb878) made multi-candidate sets
        the intended state; the question is no longer "is there a choice?" but
        "can `score` see a difference?", and the answer must stay no.
        """
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
        offenders = {}
        covered = 0
        for shape in shapes:
            remapped = variant_reachability._remap(shape, field_map)
            candidates = [
                meta
                for meta in metas.values()
                if variant_reachability.applicable(meta, remapped, divides)
            ]
            if not candidates:
                continue
            covered += 1
            block_ns = {meta["block_n"] for meta in candidates}
            if len(block_ns) > 1:
                offenders[str(sorted(remapped.items()))] = sorted(block_ns)
        # A query that matches nothing is a failed query, not evidence of
        # tidiness -- the assertion below would pass vacuously on an empty
        # bundle or a mis-keyed field_map.
        assert covered, (
            "no corpus shape matched ANY variant; the field_map or the bundle "
            "is wrong, so this test proved nothing"
        )
        assert not offenders, (
            f"{len(offenders)} shape(s) now present candidates differing in "
            f"`block_n`: {offenders}. The native scoreKernel ranks on block_n "
            f"alone, so it is no longer a tie -- the ranking now picks a "
            f"winner and its correctness needs verifying, not assuming."
        )

    def test_declared_ranking_matches_the_narrowed_verdict(self, tmp_path):
        """Runs the real tool twice against the real bundle: once with no
        ranking declared (narrowed), once with the profile's `score:` block.

        The two must agree on EVERY tally. That is the profile comment's claim
        -- declaring the ranking changes nothing observable -- and it is the
        thing worth pinning. The tallies are compared to EACH OTHER rather than
        to literals: the old form hard-coded `SELECTED 82` from an 84-variant
        bundle and went stale three resizes running (84 -> 1612 -> 1678 -> 1694)
        while the property it meant to protect never changed.
        """

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

        def tallies(text):
            found = dict(
                re.findall(
                    r"^\s+(SELECTED|APPLICABLE-BUT-NEVER-WINS|UNREACHABLE)\s+(\d+)\s*$",
                    text,
                    re.M,
                )
            )
            # An empty parse would make the equality below trivially true, so
            # require all three verdicts to have actually been read.
            assert set(found) == {
                "SELECTED",
                "APPLICABLE-BUT-NEVER-WINS",
                "UNREACHABLE",
            }, f"could not parse the verdict tallies from:\n{text}"
            return found

        narrowed_tallies = tallies(narrowed.stdout)
        assert narrowed_tallies == tallies(declared.stdout), (
            "declaring the ranking changed the verdict; the profile's claim "
            "that `score:` is observationally inert on this bundle no longer "
            "holds and needs re-verifying"
        )
        # APPLICABLE-BUT-NEVER-WINS is the one tally with an absolute meaning:
        # a variant applicable to some shape yet always outranked is dead
        # weight that every other gate reports green.
        assert narrowed_tallies["APPLICABLE-BUT-NEVER-WINS"] == "0"
        assert int(narrowed_tallies["SELECTED"]) > 0
