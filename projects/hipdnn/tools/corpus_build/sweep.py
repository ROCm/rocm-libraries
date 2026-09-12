# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Source 3: the declared parameter space, sampled to fill what the other two miss.

`corpus_gen` derives an engine's problems from `*.opmeta.json` rather than from
whichever kernels happen to be packed, which is the more general question and the
reason this source exists at all. The packed geometries are whatever rocKE compiled
and the model shapes are whatever anyone wrote down; neither is a sample of the space
a heuristic is asked to predict over. This is: same declaration, same three-way
mixture (`archetypes` / `neighbourhood` / `exploration`), same declared constraint.

Nothing here knows what attention is. Every number is read out of the declaration --
the archetype anchors, the neighbourhood factors, the regime buckets, the dtype enum
and the `seqlen_q <= seqlen_k` constraint. One exception is marked as such: the KV
head count, which the declaration does not carry yet and this tool's graph builder
does. See `_gqa_ratios`.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

from . import graphs
from .shapes import Candidate, Shape

#: The in-tree declaration, relative to the repository root.
DEFAULT_DECLARATION = Path("projects/hipdnn/tools/corpus_gen/operations/sdpa_fwd.opmeta.json")

#: Query-heads-to-KV-heads ratios the sampler draws from.
#:
#: `sdpa_fwd.opmeta.json` declares no KV head count -- its `llama3_70b_gqa` archetype
#: says so in as many words: "grouped-query attention is declared once the builder
#: exposes the KV head count", and `corpus_gen`'s `sdpaForward` passes one `heads` to
#: both operands. This tool's builder (`make_sdpa_bundles.bundle_for`) does expose it,
#: and grouping is not a detail: it decides the KV traffic that makes decode memory
#: bound, so a corpus without it cannot train the regime it exists to get right.
#: 1 is repeated because MHA is not a rare case; a ratio is kept only when it divides
#: the drawn head count.
GQA_RATIOS = (1, 1, 2, 4, 8)


def load(path: Path) -> dict:
    declaration = json.loads(path.read_text(encoding="utf-8"))
    if declaration.get("operation") != "sdpa_fwd":
        raise ValueError(f"{path} declares {declaration.get('operation')!r}, not sdpa_fwd")
    return declaration


def _buckets(declaration: dict, name: str) -> list[int]:
    return list(declaration["regimes"][name]["buckets"])


def _interpolated(buckets: list[int], granularity: int = 1) -> list[int]:
    """The declared buckets plus the geometric midpoint between each adjacent pair.

    The buckets are where the declaration says the population splits; the midpoints
    are the values between them that neither the packs nor the models happen to
    cover, which is this source's whole job. Sequence-length midpoints are snapped to
    a multiple of 16, the granularity every attention kernel in the tree tiles at --
    an interpolated seqlen of 181 measures a padding path rather than the shape it
    names. Declared buckets are never snapped: `seqlen_q: 1` is decode, and rounding
    it up to 16 would delete the regime.
    """
    values = set(buckets)
    for low, high in zip(buckets, buckets[1:]):
        middle = math.sqrt(low * high)
        values.add(max(granularity, int(round(middle / granularity)) * granularity))
    return sorted(values)


def _constraint(declaration: dict):
    """The declared constraints as a predicate over a candidate point.

    Read rather than restated: the declaration says `{"<=": ["$q.seqlen_q",
    "$q.seqlen_k"]}` and a sampler that hardcodes that relation is a second place
    where the operation's rules live -- which is exactly what `corpus_gen` refuses to
    do for the seven operations it already samples.
    """
    comparisons = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
                   ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
                   "==": lambda a, b: a == b}

    def resolve(term, point):
        if isinstance(term, str) and term.startswith("$q."):
            return point[term[3:]]
        return term

    def satisfied(point: dict) -> bool:
        for clause in declaration.get("constraints", []):
            for operator, operands in clause.items():
                compare = comparisons.get(operator)
                if compare is None:
                    raise ValueError(
                        f"unsupported constraint operator {operator!r}; the sampler "
                        f"must understand every relation the declaration states, "
                        f"because silently ignoring one emits problems the operation "
                        f"says do not exist")
                left, right = (resolve(term, point) for term in operands)
                if not compare(left, right):
                    return False
        return True

    return satisfied


def _archetype_points(declaration: dict) -> list[tuple[str, dict]]:
    """Every declared archetype, expanded. `"$q.seqlen_q"` mirrors the drawn value."""
    points = []
    for archetype in declaration.get("archetypes", []):
        values = archetype["values"]
        axes = sorted(values)
        for combination in itertools.product(*(values[axis] for axis in axes)):
            point = dict(zip(axes, combination))
            for axis, value in list(point.items()):
                if isinstance(value, str) and value.startswith("$q."):
                    point[axis] = point[value[3:]]
            points.append((archetype["name"], point))
    return points


def _gqa_ratios(heads: int) -> list[int]:
    return [ratio for ratio in GQA_RATIOS if heads % ratio == 0] or [1]


def _neighbour(rng: random.Random, declaration: dict, point: dict) -> dict:
    """One declared neighbourhood step away from an archetype point."""
    neighbourhood = declaration["neighbourhood"]
    moved = dict(point)
    batch = neighbourhood.get("batch")
    if batch and batch["kind"] == "scale":
        moved["batch"] = max(1, int(round(point["batch"] * rng.choice(batch["factors"]))))
    heads = neighbourhood.get("heads")
    if heads and heads["kind"] == "values":
        moved["heads"] = rng.choice(heads["values"])
    seqlen_q = neighbourhood.get("seqlen_q")
    if seqlen_q and seqlen_q["kind"] == "scale":
        scaled = point["seqlen_q"] * rng.choice(seqlen_q["factors"])
        # A decode point stays a decode point: scaling one query token by 0.5 and
        # rounding is how a decode archetype quietly becomes a 1-token prefill under a
        # different name. Everything longer is snapped to the kernels' 16 granularity.
        moved["seqlen_q"] = (1 if point["seqlen_q"] == 1
                             else max(16, int(scaled) - int(scaled) % 16))
    seqlen_k = neighbourhood.get("seqlen_k")
    if seqlen_k and seqlen_k["kind"] == "mirror":
        base = moved[seqlen_k["of"]]
        ratio = rng.choice(seqlen_k["ratios"])
        moved["seqlen_k"] = max(base, int(base * ratio) if base > 1 else
                                int(point["seqlen_k"] * ratio))
    return moved


def _exploration(rng: random.Random, declaration: dict) -> dict:
    """A point drawn from the declared buckets and the gaps between them.

    `head_dim` is drawn from its buckets and never interpolated, on the declaration's
    own instruction: "an interpolated 96 or 112 is not a configuration anyone ships".
    """
    seqlen_q = rng.choice(_interpolated(_buckets(declaration, "seqlen_q"), 16))
    ratio = rng.choice(declaration["neighbourhood"]["seqlen_k"]["ratios"])
    seqlen_k = max(seqlen_q, seqlen_q * ratio if seqlen_q > 1
                   else rng.choice(_interpolated(_buckets(declaration, "seqlen_k"), 16)))
    return {"batch": rng.choice(_interpolated(_buckets(declaration, "batch"))),
            "heads": rng.choice(declaration["neighbourhood"]["heads"]["values"]),
            "seqlen_q": seqlen_q, "seqlen_k": seqlen_k,
            "head_dim": rng.choice(_buckets(declaration, "head_dim")),
            "is_causal": rng.choice([True, False])}


def _shape(rng: random.Random, declaration: dict, point: dict) -> Shape | None:
    heads = int(point["heads"])
    ratio = rng.choice(_gqa_ratios(heads))
    dtype = rng.choice(declaration["parameters"]["dtype"]["values"])
    try:
        return Shape(dtype=dtype, batch=int(point["batch"]), heads_q=heads,
                     heads_kv=max(1, heads // ratio), seqlen_q=int(point["seqlen_q"]),
                     seqlen_kv=int(point["seqlen_k"]), head_dim=int(point["head_dim"]),
                     causal=bool(point["is_causal"]))
    except ValueError:
        return None


def sample(declaration: dict, wanted: int, seed: int, max_bytes: int,
           exclude=()) -> tuple[list[Candidate], dict]:
    """`wanted` distinct shapes the other two sources do not already carry.

    Drawn one at a time with the source of each draw chosen from the declaration's
    own `mixture` weights, so any prefix of the result holds the declared proportions
    -- the pool is truncated to a budget by the caller, and a pool generated
    archetypes-first would lose its exploration tail to that cut.
    """
    rng = random.Random(seed)
    satisfied = _constraint(declaration)
    archetypes = _archetype_points(declaration)
    mixture = declaration.get("mixture") or {"archetypes": 0.2, "neighbourhood": 0.6,
                                             "exploration": 0.2}
    kinds = sorted(mixture)
    weights = [mixture[kind] for kind in kinds]

    seen = set(exclude)
    candidates: list[Candidate] = []
    stats = {"draws": 0, "duplicate": 0, "constraint": 0, "over_byte_budget": 0,
             "invalid": 0, "by_kind": {kind: 0 for kind in kinds}}
    # Bounded: a declaration whose space is smaller than `wanted` must end the run
    # rather than spin. The multiplier is slack for the duplicate and constraint
    # rejections above, which are ordinary rather than exceptional.
    attempts = max(64, wanted * 64)
    while len(candidates) < wanted and stats["draws"] < attempts:
        stats["draws"] += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        if kind == "exploration" or not archetypes:
            name, point = "exploration", _exploration(rng, declaration)
        else:
            name, point = archetypes[rng.randrange(len(archetypes))]
            if kind == "neighbourhood":
                point = _neighbour(rng, declaration, point)
        if not satisfied({"seqlen_q": point["seqlen_q"], "seqlen_k": point["seqlen_k"]}):
            stats["constraint"] += 1
            continue
        shape = _shape(rng, declaration, point)
        if shape is None:
            stats["invalid"] += 1
            continue
        if shape.key in seen:
            stats["duplicate"] += 1
            continue
        if graphs.footprint_bytes(shape) > max_bytes:
            stats["over_byte_budget"] += 1
            continue
        seen.add(shape.key)
        stats["by_kind"][kind] += 1
        candidates.append(Candidate(shape=shape, source="sweep",
                                    origin=f"{kind}:{name}"))
    stats["shapes"] = len(candidates)
    return candidates, stats
