# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one shape a corpus graph is, and the regime it belongs to.

Every source -- packed kernel geometry, a real model, the declaration sweep -- lands
here, which is what makes deduplication across them possible at all: three vocabularies
(`num_query_heads`/`nhead_q`/`heads`, `BF16`/`bfloat16`/`bf16`, `causal:0`/`mask_type:1`)
describing the same problem must collapse to one tuple or the same shape is emitted
three times under three names.
"""
from __future__ import annotations

import dataclasses

#: The only operation this corpus builds. Carried in the dedup key rather than assumed,
#: because the key is the corpus's identity and an `sdpa_fwd` shape tuple is not a
#: `conv_fwd` one even when the numbers coincide.
OP = "sdpa_fwd"

#: Canonical dtype spellings: the `sdpa_fwd.opmeta.json` `dtype` enum, which is also the
#: spelling `IngestorGenerator/tools/mine_shapes.py` normalises every source to. Two
#: tools agreeing on one vocabulary is the point; a third spelling here would undo it.
DTYPES = ("bf16", "fp16")


#: The two anchors a causal diagonal can take, spelled as the hipDNN graph's
#: `diagonal_alignment` enum spells them, lowercased. Top-left is the default the
#: shipped bundles carry; bottom-right is what a generation step actually asks for,
#: since its new queries sit at the end of the KV cache.
TOP_LEFT = "top_left"
BOTTOM_RIGHT = "bottom_right"
ALIGNMENTS = (TOP_LEFT, BOTTOM_RIGHT)

#: Above this KV length a problem is `long` rather than `short`. This is the middle
#: bucket of `sdpa_fwd.opmeta.json`'s declared `seqlen_k` regimes
#: ([128, 512, 2048, 8192, 32768]) -- the declaration's own opinion about where the
#: population splits, rather than a threshold invented here.
LONG_CONTEXT = 2048


@dataclasses.dataclass(frozen=True, order=True)
class Shape:
    """One SDPA forward problem, spelled once.

    Frozen and ordered: it is used as a dict key for deduplication and sorted for
    deterministic output, and a mutable shape would let a corpus contain two entries
    that were equal when they were inserted.
    """

    dtype: str
    batch: int
    heads_q: int
    heads_kv: int
    seqlen_q: int
    seqlen_kv: int
    head_dim: int
    causal: bool
    #: Where a causal mask's diagonal is anchored. Two engines can implement one and
    #: not the other -- AITER's gfx942 forward table carries bottom-right causal
    #: kernels and no top-left ones, so a corpus that spells causality one way makes
    #: it decline every causal graph -- which is what makes this part of the shape
    #: and not a detail of how the graph is written. Meaningless without `causal`,
    #: and pinned to `top_left` there so a non-causal problem has one spelling.
    alignment: str = TOP_LEFT
    op: str = OP

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}")
        for field in ("batch", "heads_q", "heads_kv", "seqlen_q", "seqlen_kv", "head_dim"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive int, got {value!r}")
        if self.heads_q % self.heads_kv and self.heads_kv % self.heads_q:
            raise ValueError(
                f"head counts must divide: {self.heads_q} query heads against "
                f"{self.heads_kv} KV heads is not a grouping any kernel implements"
            )
        if self.alignment not in ALIGNMENTS:
            raise ValueError(
                f"alignment must be one of {list(ALIGNMENTS)}, got {self.alignment!r}")
        if not self.causal and self.alignment != TOP_LEFT:
            raise ValueError(
                f"a non-causal shape has no diagonal to anchor, so it carries the "
                f"canonical {TOP_LEFT!r}; got {self.alignment!r}")
        # No `seqlen_q <= seqlen_kv` check. That relation is a constraint on the
        # DECLARED space (`sdpa_fwd.opmeta.json` `constraints`), which the sweep reads
        # and obeys -- it is not a fact about attention. `gfx942_attention_dense`
        # carries 56 compiled geometries at 512 queries against 256 keys: cross
        # attention, where the query is the image sequence and the keys are the
        # prompt. Refusing those here would delete real, admissible, packed problems
        # from the corpus on the strength of a sampler's rule.

    @property
    def key(self) -> tuple:
        """The full shape tuple two sources are deduplicated on.

        The alignment is part of it: a top-left and a bottom-right causal problem of
        the same geometry are two problems, served by different kernels and -- when
        `seqlen_q != seqlen_kv` -- computing different outputs. Collapsing them would
        drop whichever arrived second as a duplicate of a problem it is not.
        """
        return (self.op, self.dtype, self.batch, self.heads_q, self.heads_kv,
                self.seqlen_q, self.seqlen_kv, self.head_dim, bool(self.causal),
                self.alignment)

    @property
    def phase(self) -> str:
        """Where in a generation this problem occurs.

        Decode (one query token), append (a chunk of new queries against a longer
        cache), prefill (the whole prompt at once) and cross (queries from one
        sequence, keys from another) are the same operation and entirely different
        problems -- decode is memory bound on the KV cache, prefill is compute bound
        on the score matrix -- which is why `sdpa_fwd.opmeta.json` anchors them as
        separate archetypes rather than as one range over `seqlen_q`.
        """
        if self.seqlen_q == 1:
            return "decode"
        if self.seqlen_q > self.seqlen_kv:
            # More queries than keys is cross attention, not a long prefill: the two
            # sequences come from different tensors and vary independently, which is
            # why the catalog tracks `S_kv=77` cross-attention as its own decline.
            return "cross"
        return "prefill" if self.seqlen_q == self.seqlen_kv else "append"

    @property
    def context(self) -> str:
        return "short" if self.seqlen_kv <= LONG_CONTEXT else "long"

    @property
    def grouping(self) -> str:
        if self.heads_q == self.heads_kv:
            return "mha"
        return "mqa" if self.heads_kv == 1 else "gqa"

    @property
    def regime(self) -> str:
        """§5.2's regime label, composed of the three facets that separate populations.

        RFC 0019.13 §11.2 wants the per-regime table as the PRIMARY form of the regret
        report, because an aggregate hides a model that is excellent on the dense middle
        and useless on decode-shaped problems. `uhd_gen evaluate` reports it as
        UNAVAILABLE when no corpus column carries a regime (`evaluate.REGIME_COLUMN_CANDIDATES`),
        which is what this label exists to supply.
        """
        return f"{self.phase}_{self.context}_{self.grouping}"

    @property
    def name(self) -> str:
        """The graph's name, which carries the regime and the whole shape tuple.

        The regime is in the name and not only in the manifest so that a graph
        separated from its manifest -- copied into a bench invocation, quoted in a
        result row, named in a log -- still says which population it belongs to. The
        rest of the tuple follows so the name is unique exactly when the shape is:
        two graphs with one name would be one graph on disk.
        """
        mask = ("causal_br" if self.alignment == BOTTOM_RIGHT else "causal_tl"
                ) if self.causal else "nomask"
        return (f"{self.op}_{self.regime}_{self.dtype}_b{self.batch}"
                f"_hq{self.heads_q}_hkv{self.heads_kv}"
                f"_sq{self.seqlen_q}_skv{self.seqlen_kv}_d{self.head_dim}_{mask}")

    def geometry(self) -> dict:
        """This shape in `make_sdpa_bundles`' KDP vocabulary.

        That module owns the graph document (`bundle_for`) and the byte footprint
        (`footprint_bytes`), and both read a KDP metadata block: rocKE dtype spelling,
        `head_size`, `num_query_heads`. Converting here keeps that reuse honest instead
        of forking a second graph writer that can drift from the shipped bundles.
        """
        return {"dtype": self.dtype.upper(), "head_size": self.head_dim,
                "num_query_heads": self.heads_q, "num_kv_heads": self.heads_kv,
                "seqlen_q": self.seqlen_q, "seqlen_kv": self.seqlen_kv,
                "batch": self.batch, "causal": bool(self.causal),
                "alignment": self.alignment}


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A shape plus where it came from, which the manifest records verbatim.

    Provenance is carried rather than recomputed: once three sources have been
    deduplicated into one corpus, nothing in the shape itself says whether it was
    measured from a packed kernel, read off a model, or sampled -- and a corpus whose
    population mix cannot be audited cannot be reproduced either.
    """

    shape: Shape
    source: str
    origin: str


@dataclasses.dataclass(frozen=True)
class Filter:
    """Which shapes a corpus is allowed to carry, as facet whitelists.

    A corpus exists to make engines compete, and an engine that cannot serve a shape
    contributes nothing to it but a decline. The gate is usually narrow and always
    published: AITER's gfx942 forward table (`asm_kernels/gfx942/fmha_v3_fwd/fmha_fwd.csv`)
    holds four kernels, all `bf16`, all `hdim_v=128` -- so a corpus of `fp16`/`d64`
    problems is one that engine declines in full, which is exactly what run 67928437
    reported for all 24 graphs.

    Empty means unrestricted, per facet: a filter nobody asked for admits everything,
    and `--dtype bf16` alone must not silently also pin the head dimension.
    """

    dtypes: tuple[str, ...] = ()
    head_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for dtype in self.dtypes:
            if dtype not in DTYPES:
                raise ValueError(f"dtype must be one of {list(DTYPES)}, got {dtype!r}")
        for head_dim in self.head_dims:
            if not isinstance(head_dim, int) or head_dim < 1:
                raise ValueError(f"head dim must be a positive int, got {head_dim!r}")

    def __bool__(self) -> bool:
        return bool(self.dtypes or self.head_dims)

    def admits(self, shape: Shape) -> bool:
        if self.dtypes and shape.dtype not in self.dtypes:
            return False
        return not (self.head_dims and shape.head_dim not in self.head_dims)

    def describe(self) -> dict:
        """The filter as the manifest records it, so a corpus says what it excluded."""
        return {"dtypes": list(self.dtypes), "head_dims": list(self.head_dims)}
