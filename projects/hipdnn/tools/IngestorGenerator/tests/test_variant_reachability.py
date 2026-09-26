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
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "variant_reachability.py"

sys.path.insert(0, str(_TOOLS))

import variant_reachability  # noqa: E402
from launch_surface import find_repo_root  # noqa: E402

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


class TestGfx950RealBundle:
    """The shipped gfx950 bundle against the shipped shape corpus. Needs no device and no
    build, and every expectation is derived from the bundle rather than written down, so
    resizing the catalog cannot make it stale.

    This asserts tile REACHABILITY, not cold-selection wins. `scoreKernel` ranks an
    applicable (256, 64) first and the engine exposes `block_m`/`block_n` as knobs, so the
    other tiles are auto-tune inventory and are *expected* never to win the cold path.
    `APPLICABLE-BUT-NEVER-WINS == 0` described a single-tile engine and is not the property
    to hold here. A tile applicable to NO corpus shape is still dead weight, and that is
    what this catches.

    How much this discriminates depends on the corpus, and on the present one it is loose
    on six tiles and tight on the seventh. Every sequence length here is a power of two,
    so each tile up to (256, 128) is applicable to the same broad majority of shapes and
    any power-of-two tile that size would pass; against those six the gate catches only a
    tile dividing NOTHING, not one that is merely rare. Sharpening that half needs a
    corpus carrying a non-power-of-two sequence length rather than a different control
    value -- 384, 192 and 96 each divide zero shapes here, exactly like the prime the
    control below uses.

    (256, 256) is the exception and the reason this is load-bearing. It ships at
    head_size 64 alone, and one cohort reaches it: bf16, 64 query heads, 8 KV heads. Drop
    that single model family from the corpus and the descriptors carrying that tile are
    provably selectable by nothing, and this fails. The gate is tightest exactly where the
    catalog is thinnest.
    """

    _REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
    _KDP = (
        _REPO_ROOT
        / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
        / "descriptors/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json"
    )

    #: The request corpus is an author's input that this repository does not ship: the
    #: workflow mines it to a path of the operator's choosing, so there is no in-tree
    #: location for it and a hard-coded one would be a private convention. Name it here
    #: and this class runs; leave it unset and it skips, saying which variable to set.
    _SHAPES_VAR = "HIPDNN_INGESTOR_SHAPES"

    #: Corpus vocabulary -> matcher vocabulary. `seqlen_q`/`seqlen_k` are deliberately
    #: renamed to names no metadata field carries: the KDP records a canonical
    #: `seqlen_q`/`seqlen_kv` that `kernelMatches` never reads (the shape is a runtime
    #: kernarg), so leaving them under their own names would compare them by equality and
    #: report the whole catalog unreachable.
    _FIELD_MAP = {
        "nhead_q": "num_query_heads",
        "nhead_k": "num_kv_heads",
        "hdim_q": "head_size",
        "seqlen_q": "sq",
        "seqlen_k": "skv",
    }
    #: A tile is legal for a shape when it divides it -- `Sq % block_m` and
    #: `Skv % block_n`, per Gfx950AttentionDenseNative.cpp.
    _DIVIDES = {"block_m": "sq", "block_n": "skv"}

    #: mask_type 2 is windowed. No windowed variant ships, so those shapes are out of
    #: scope rather than uncovered, and counting them would understate coverage.
    _WINDOWED = 2

    @classmethod
    def _shapes_path(cls) -> Path:
        """The corpus named by `_SHAPES_VAR`. Skips when unset, and FAILS when set to
        something that is not a file: a typo'd path is an operator error, and reporting
        it as a skip would read as "this class is opt-in and you opted out"."""
        raw = os.environ.get(cls._SHAPES_VAR)
        if not raw:
            pytest.skip(
                f"{cls._SHAPES_VAR} is unset, so there is no request corpus to check "
                f"the shipped bundle against. Mine one with tools/mine_shapes.py and "
                f"point this variable at it to run this class."
            )
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(
                f"{cls._SHAPES_VAR} is set to {raw!r}, which is not an existing file. "
                f"Unset it to skip this class, or point it at a mined corpus."
            )
        return path

    @classmethod
    def _require_assets(cls):
        if not cls._KDP.exists():
            pytest.skip(f"gfx950 bundle not present in this checkout: {cls._KDP}")

    @classmethod
    def _corpus(cls, head_sizes):
        """In-scope corpus shapes in matcher vocabulary.

        Two families are excluded because the catalog ships nothing that could serve
        them, and counting them would make the denominator describe the corpus rather
        than the engine's scope: windowed shapes, since no windowed variant ships, and
        head sizes the catalog does not carry. `batch` is dropped because the metadata
        carries a canonical batch the matcher never compares; left in, it would
        equality-match and reject every multi-batch graph the engine actually serves."""
        shapes = []
        for raw in json.loads(cls._shapes_path().read_text()):
            if raw.get("mask_type") == cls._WINDOWED:
                continue
            if raw.get("hdim_q") not in head_sizes:
                continue
            shape = variant_reachability._remap(raw, cls._FIELD_MAP)
            shape["causal"] = 1 if shape.pop("mask_type") == 1 else 0
            for vestigial in ("batch", "hdim_v", "_provenance"):
                shape.pop(vestigial, None)
            shapes.append(shape)
        return shapes

    @classmethod
    def _metas(cls):
        defaults, descriptors = variant_reachability.load_bundle(str(cls._KDP))
        return [
            variant_reachability._resolved_metadata(d, defaults) for d in descriptors
        ]

    @staticmethod
    def _tile(meta):
        return (meta["block_m"], meta["block_n"])

    def test_every_shipped_tile_is_reachable_by_some_corpus_shape(self):
        """A tile no corpus shape admits cannot be cold-selected OR auto-tuned onto, so
        it is dead weight however the ranking is spelled."""
        self._require_assets()
        metas = self._metas()
        corpus = self._corpus({m["head_size"] for m in metas})
        assert metas and corpus, "empty bundle or corpus proves nothing"

        reachable, matched_shapes = set(), 0
        for shape in corpus:
            hit = False
            for meta in metas:
                if variant_reachability.applicable(meta, shape, self._DIVIDES):
                    reachable.add(self._tile(meta))
                    hit = True
            matched_shapes += hit

        # Without this the assertion below passes vacuously on a broken field map:
        # zero applicable pairs means zero shipped tiles AND zero unreachable ones.
        assert matched_shapes, (
            "no corpus shape matched ANY variant -- the field map or the bundle is "
            "wrong, so this test proved nothing"
        )
        shipped = {self._tile(m) for m in metas}
        assert shipped - reachable == set(), (
            f"{len(shipped - reachable)} shipped tile(s) are applicable to no corpus "
            f"shape at all: {sorted(shipped - reachable)}. Either the corpus is missing "
            f"a shape family or these tiles should not be built."
        )

    def test_a_tile_the_corpus_cannot_admit_is_reported_unreachable(self):
        """The control for the case above. A tile that divides no corpus sequence length
        must be caught -- without this, 'every shipped tile is reachable' could be true
        because the check never rejects anything."""
        self._require_assets()
        metas = self._metas()
        corpus = self._corpus({m["head_size"] for m in metas})
        impossible = dict(metas[0])
        # 2^31-1 is prime, so it divides no sequence length any corpus carries.
        impossible["block_m"] = 2147483647
        assert not any(
            variant_reachability.applicable(impossible, shape, self._DIVIDES)
            for shape in corpus
        ), "an undividable tile was reported applicable; the divides rule is not firing"
