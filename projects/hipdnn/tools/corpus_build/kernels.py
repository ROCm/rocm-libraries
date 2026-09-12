# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Source 1: the geometries an attention KDP actually carries.

rocKE kernels bake their geometry in, so the kernel matcher pins dtype, head counts,
sequence lengths, head size, batch and causality before ranking starts. That makes the
pack the authority on two things nothing else can answer: which problems a descriptor
engine will *admit* at all, and which of those have enough competing kernels for L2 to
have a choice worth learning. `make_sdpa_bundles` established both rules and this
module reuses them rather than restating them.

Packs are read structurally, not by file name: a descriptor contributes a geometry if
its metadata carries the whole geometry block. That is what makes the gfx950 dense and
tiled packs pick themselves up when they land, and it is also why the shipped
`tiled_attention` pack contributes nothing -- its kernels declare `block_size` and
`dtype` and no shape, so there is no geometry in it to emit.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from . import graphs
from .shapes import Candidate, Shape

#: Where the in-tree packs live, relative to the repository root. Deliberately the
#: whole example-descriptor tree rather than `rocKE/`: a pack is selected by carrying
#: attention geometry, not by which directory it sits in, so a gfx950 pack landing
#: anywhere under here is picked up, and `pointwise_add` is read and contributes
#: nothing -- which the manifest records rather than hides.
DEFAULT_KDP_ROOT = Path("dnn-providers/hip-kernel-provider/descriptor-packaging/"
                        "examples/descriptors")

#: The KDP dtype spellings this corpus can build a graph for. `make_sdpa_bundles`
#: owns the table; a pack dtype outside it (fp8, say) is skipped and counted rather
#: than guessed into `bf16`.
_DTYPES = graphs.bundles.DTYPE_TO_BUNDLE


def discover(roots: list[Path]) -> list[Path]:
    """Every `*.kdp.json` under the given roots, in a stable order."""
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob("*.kdp.json")))
    return sorted(dict.fromkeys(path.resolve() for path in found))


def from_pack(path: Path, min_candidates: int, max_bytes: int) -> tuple[list[Candidate], dict]:
    """The geometries one pack carries, with the ones not worth ranking dropped.

    `min_candidates` is `make_sdpa_bundles`' rule: a geometry served by one kernel has
    nothing to rank, so it teaches a ranking model nothing. `max_bytes` is the other
    one: a geometry whose tensors do not fit cannot be measured, and a corpus entry
    that cannot be measured is a hole in the training set rather than a member of it.
    """
    pack = json.loads(path.read_text(encoding="utf-8"))
    descriptors = pack.get("kernelDescriptors") or []
    counts: collections.Counter = collections.Counter()
    for descriptor in descriptors:
        metadata = descriptor.get("metadata") or {}
        counts[tuple(metadata.get(name) for name in graphs.GEOMETRY)] += 1

    stats = {"pack": str(path), "kernels": len(descriptors), "geometries": len(counts),
             "no_geometry": 0, "unsupported_dtype": 0, "too_few_candidates": 0,
             "over_byte_budget": 0, "eligible": 0}
    candidates: list[Candidate] = []
    for tup, kernels in sorted(counts.items(), key=lambda item: repr(item[0])):
        geometry = dict(zip(graphs.GEOMETRY, tup))
        if any(value is None for value in tup):
            stats["no_geometry"] += 1
            continue
        if geometry["dtype"] not in _DTYPES:
            stats["unsupported_dtype"] += 1
            continue
        if kernels < min_candidates:
            stats["too_few_candidates"] += 1
            continue
        try:
            shape = Shape(dtype=geometry["dtype"].lower(), batch=int(geometry["batch"]),
                          heads_q=int(geometry["num_query_heads"]),
                          heads_kv=int(geometry["num_kv_heads"]),
                          seqlen_q=int(geometry["seqlen_q"]),
                          seqlen_kv=int(geometry["seqlen_kv"]),
                          head_dim=int(geometry["head_size"]),
                          causal=bool(geometry["causal"]))
        except ValueError:
            # A pack may carry a geometry no corpus can label -- causal cross attention,
            # whose declared FLOP count is non-positive. The kernel exists and the engine
            # will run it; there is simply no measurement a training row can be made of.
            stats["unlabellable"] = stats.get("unlabellable", 0) + 1
            continue
        if graphs.footprint_bytes(shape) > max_bytes:
            stats["over_byte_budget"] += 1
            continue
        stats["eligible"] += 1
        candidates.append(Candidate(shape=shape, source="kernel",
                                    origin=f"{path.name}:{kernels} kernels"))
    return candidates, stats


def collect(paths: list[Path], min_candidates: int, max_bytes: int
            ) -> tuple[list[Candidate], list[dict]]:
    """Every pack's eligible geometries, ordered so any prefix stays spread out.

    `make_sdpa_bundles.stratified` does the ordering: it cycles dtype x head_size x
    causal x GQA buckets smallest-first, so truncating this pool to a budget still
    leaves every corner represented. A pool ordered by pack file order and then cut
    would be a corpus of whatever the first pack happened to list.
    """
    candidates: list[Candidate] = []
    reports: list[dict] = []
    for path in paths:
        found, stats = from_pack(path, min_candidates, max_bytes)
        candidates.extend(found)
        reports.append(stats)

    # `stratified` reorders the very dict objects it is handed, so identity maps each
    # one back to the candidate it was derived from -- no second key, and no chance of
    # two equal geometries being matched to the wrong provenance.
    geometries = [candidate.shape.geometry() for candidate in candidates]
    index = {id(geometry): candidate
             for geometry, candidate in zip(geometries, candidates)}
    ordered = graphs.bundles.stratified(geometries, len(geometries))
    return [index[id(geometry)] for geometry in ordered], reports
