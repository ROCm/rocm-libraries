# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The pipeline: read three sources, deduplicate, allocate, write a corpus.

Offline and GPU-free by construction. Every input is a file in this repository or a
directory someone points at, and the output is graph documents plus a manifest --
nothing here loads a plugin, enumerates a device or times anything. That is what makes
it runnable on a laptop before the cluster time it is preparing for.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from . import assemble, kernels, model_shapes, sweep
from .shapes import BOTTOM_RIGHT, Filter

#: The repository root, found from this file rather than from the working directory:
#: the tool is run from `projects/hipdnn/tools` (like `uhd_gen`) but its inputs are
#: addressed from the root, and a default that depends on where you stood is not a
#: default a reproducible corpus can have.
REPO = Path(__file__).resolve().parents[4]

DEFAULT_COUNT = 1000

#: Skip a geometry whose four tensors exceed this. `make_sdpa_bundles`' budget: a
#: problem that does not fit cannot be measured, and an unmeasurable corpus entry is
#: a hole in the training set rather than a member of it.
DEFAULT_MAX_BYTES = 2 * 1024 ** 3

#: `make_sdpa_bundles`' rule for the packs: a geometry served by fewer kernels than
#: this has nothing to rank, so it teaches a ranking model nothing.
DEFAULT_MIN_CANDIDATES = 3


def both_anchors(pool: list) -> list:
    """`pool` with a bottom-right twin beside every causal candidate it carries.

    A top-left and a bottom-right causal problem are two problems, not two spellings
    of one: where `seqlen_q != seqlen_kv` they compute different outputs, and engines
    implement them separately. AITER's gfx942 forward table has bottom-right causal
    kernels and no top-left ones, so the single-anchor corpus of run 67928906 had it
    serving none of its 15 causal graphs. The twin keeps the provenance of the shape
    it came from, because it is the same measurement seen under the other convention.

    Beside, not appended: `assemble.select` takes a prefix of each pool, so twins
    collected at the end would be cut off by every allocation short of the whole pool
    -- which is a corpus that carries the second anchor only when nothing needed it.
    """
    twinned = []
    for candidate in pool:
        twinned.append(candidate)
        if candidate.shape.causal:
            twinned.append(dataclasses.replace(candidate, shape=dataclasses.replace(
                candidate.shape, alignment=BOTTOM_RIGHT)))
    return twinned


def build(out: Path, *, count: int = DEFAULT_COUNT, seed: int = 0,
          kdp_roots=None, kdps=(), catalog: Path | None = None, shape_dirs=(),
          arch: str | None = None, batches=model_shapes.DEFAULT_BATCHES,
          declaration: Path | None = None,
          min_candidates: int = DEFAULT_MIN_CANDIDATES,
          max_bytes: int = DEFAULT_MAX_BYTES, shares=None,
          keep: Filter | None = None) -> dict:
    """Assemble a corpus into `out` and return its manifest."""
    catalog = REPO / model_shapes.DEFAULT_CATALOG if catalog is None else Path(catalog)
    declaration = (REPO / sweep.DEFAULT_DECLARATION if declaration is None
                   else Path(declaration))
    roots = [Path(root) for root in
             (kdp_roots if kdp_roots is not None else [REPO / kernels.DEFAULT_KDP_ROOT])]
    packs = kernels.discover(roots + [Path(path) for path in kdps])
    shape_dirs = [Path(directory) for directory in shape_dirs]
    for directory in shape_dirs:
        if not directory.is_dir():
            raise SystemExit(f"FAIL: --model-shapes {directory} is not a directory")
    shares = dict(assemble.DEFAULT_SHARES if shares is None else shares)

    model_pool, model_reports = model_shapes.collect(
        catalog if catalog.is_file() else None, shape_dirs, batches, arch, max_bytes)
    kernel_pool, kernel_reports = kernels.collect(packs, min_candidates, max_bytes)
    # The filter lands on the measured sources before anything is deduplicated or
    # allocated, and is handed to the sweep so its draws obey it too. Applied once
    # here rather than per source: a corpus admitting a shape from a pack that it
    # would reject from the declaration would be filtered by provenance, not shape.
    keep = keep if keep is not None else Filter()
    excluded = {}
    if keep:
        for source, pool in (("model", model_pool), ("kernel", kernel_pool)):
            admitted = [candidate for candidate in pool if keep.admits(candidate.shape)]
            excluded[source] = len(pool) - len(admitted)
            pool[:] = admitted

    model_pool, kernel_pool = both_anchors(model_pool), both_anchors(kernel_pool)

    # The sweep is asked only for shapes the first two do not already have, so its
    # allocation is never spent on a duplicate that is then dropped. `count` is its
    # capacity: no corpus can need more sampled shapes than it has graphs.
    taken = {candidate.shape.key for candidate in model_pool + kernel_pool}
    declared = sweep.load(declaration)
    sweep_pool, sweep_report = sweep.sample(declared, count, seed, max_bytes, taken, keep)
    sweep_pool = both_anchors(sweep_pool)

    pools, duplicates = assemble.deduplicate(
        {"model": model_pool, "kernel": kernel_pool, "sweep": sweep_pool})
    selected, allocation = assemble.select(pools, count, shares)

    inputs = [path for path in packs]
    if catalog.is_file():
        inputs.append(catalog)
    inputs.append(declaration)
    return assemble.write(
        Path(out), selected, seed=seed, count=count, inputs=inputs,
        allocation=allocation, duplicates=duplicates,
        reports={"model": model_reports, "kernel": kernel_reports,
                 "sweep": sweep_report,
                 "pool_sizes": {source: len(pool) for source, pool in pools.items()},
                 "filter": keep.describe(), "filtered_out": excluded,
                 "shape_dirs": [str(directory) for directory in shape_dirs],
                 "shares": shares})
