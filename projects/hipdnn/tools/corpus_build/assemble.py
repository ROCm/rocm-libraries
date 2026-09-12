# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Three source pools into one corpus: deduplicate, allocate, write, record.

The corpus is engine-agnostic on purpose. RFC 0019 §4.1's UHD carries no engine, no
role and no arch, and an engine binds a model by naming it from its own provider code
-- so the graphs a heuristic is trained on are the operation's problems, not any one
engine's. The same directory trains the descriptor-backed engines and the ones with no
UED, which is the only way their L1 estimates end up comparable at all.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

from . import graphs
from .shapes import Candidate

#: Order of precedence when two sources describe the same shape, and the order the
#: manifest reports. A recorded model shape beats a packed geometry beats a sample:
#: the duplicate is the same problem either way, so what is being chosen is which
#: provenance the manifest records, and "this is what llama runs" is worth more to a
#: later audit than "the sampler also drew it".
SOURCES = ("model", "kernel", "sweep")

#: Default share of the corpus each source is allocated. Kernel geometries take the
#: largest share because they are the only problems guaranteed to have several
#: competing kernels -- L2 has nothing to rank without them and the descriptor
#: engines decline everything else. A source that cannot fill its share hands the
#: remainder back (see `allocate`), so these are floors and preferences, not quotas.
DEFAULT_SHARES = {"model": 0.15, "kernel": 0.60, "sweep": 0.25}


def allocate(count: int, capacity: dict, shares: dict) -> dict:
    """How many graphs each source contributes, given what each source has.

    Shares first, then whatever a short pool could not use is redistributed one at a
    time over the pools that still have room. Round-robin redistribution rather than
    a priority order, because handing an entire shortfall to one source is how a
    corpus that asked for a mix gets 90% of one population -- exactly the failure the
    per-regime table in RFC 0019.13 §11.2 exists to expose.
    """
    total = sum(shares.get(source, 0.0) for source in capacity) or 1.0
    allocation = {source: min(capacity[source],
                              int(count * shares.get(source, 0.0) / total))
                  for source in capacity}
    remaining = count - sum(allocation.values())
    while remaining > 0:
        open_pools = [source for source in SOURCES
                      if source in capacity and allocation[source] < capacity[source]]
        if not open_pools:
            break
        for source in open_pools:
            if remaining == 0:
                break
            allocation[source] += 1
            remaining -= 1
    return allocation


def deduplicate(pools: dict) -> tuple[dict, dict]:
    """One candidate per shape tuple, earlier sources winning.

    Deduplication is on the full tuple -- op, dtype, batch, both head counts, both
    sequence lengths, head dim and causality -- because every one of those changes
    which kernel is fastest. Two entries differing only in provenance are one problem
    measured twice: the same graph benchmarked twice under two names, which inflates
    a corpus and biases whichever regime it lands in.
    """
    seen: set = set()
    unique: dict = {}
    dropped: dict = {}
    for source in SOURCES:
        kept = []
        duplicates = 0
        for candidate in pools.get(source, []):
            if candidate.shape.key in seen:
                duplicates += 1
                continue
            seen.add(candidate.shape.key)
            kept.append(candidate)
        unique[source] = kept
        dropped[source] = duplicates
    return unique, dropped


def select(pools: dict, count: int, shares: dict) -> tuple[list, dict]:
    """The corpus: each pool's allocation, taken from the front of the pool.

    The front, not a sample: every pool is already ordered so that a prefix stays
    spread across the space it covers (`make_sdpa_bundles.stratified` for the packs,
    the declared mixture weights for the sweep). Re-sampling here would undo that.
    """
    capacity = {source: len(pools.get(source, [])) for source in SOURCES}
    allocation = allocate(count, capacity, shares)
    selected: list[Candidate] = []
    for source in SOURCES:
        selected.extend(pools.get(source, [])[:allocation[source]])
    return selected, allocation


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(out: Path, selected: list, *, seed: int, count: int, inputs: list,
          reports: dict, allocation: dict, duplicates: dict) -> dict:
    """Write every graph, then the manifest that makes the set reproducible.

    Two manifests, one content. `manifest.json` is the audit record: what was asked
    for, which files answered, what each graph is and where it came from.
    `manifest.csv` is the same rows in the form `uhd_gen evaluate --regime-column`
    can consume -- keyed by `benchmark`, which is the identity `uhd_gen generate`
    gives an ID-less JSON graph, so a collected corpus joins to it directly. Without
    that column §11.2's per-regime table reports UNAVAILABLE, which is the state this
    tool was written to end.
    """
    out.mkdir(parents=True, exist_ok=True)
    graph_dir = out / "graphs"
    graph_dir.mkdir(exist_ok=True)

    records = []
    for candidate in selected:
        shape = candidate.shape
        file_name, identity = graphs.write(graph_dir, shape)
        records.append({
            "benchmark": identity,
            "file": f"graphs/{file_name}",
            "name": shape.name,
            "source": candidate.source,
            "origin": candidate.origin,
            "regime": shape.regime,
            "phase": shape.phase,
            "context": shape.context,
            "grouping": shape.grouping,
            "op": shape.op,
            "dtype": shape.dtype,
            "batch": shape.batch,
            "heads_q": shape.heads_q,
            "heads_kv": shape.heads_kv,
            "seqlen_q": shape.seqlen_q,
            "seqlen_kv": shape.seqlen_kv,
            "head_dim": shape.head_dim,
            "causal": shape.causal,
            "alignment": shape.alignment,
            "bytes": graphs.footprint_bytes(shape),
        })

    manifest = {
        "tool": "corpus_build",
        "operation": records[0]["op"] if records else None,
        "seed": seed,
        "requested": count,
        "emitted": len(records),
        "mix": dict(collections.Counter(record["source"] for record in records)),
        "allocation": allocation,
        "duplicates_dropped": duplicates,
        "regimes": dict(sorted(collections.Counter(
            record["regime"] for record in records).items())),
        "inputs": [{"path": str(path), "sha256": _digest(path)} for path in inputs],
        "reports": reports,
        "graphs": records,
        "note": (
            "`benchmark` is the UUID5 `uhd_gen generate` assigns an ID-less JSON graph "
            "(uuid5(NAMESPACE_URL, 'hipdnn:graph:' + canonical json)), which is the "
            "`benchmark` column of the corpus it collects -- join manifest.csv on it to "
            "give `uhd_gen evaluate --regime-column regime` the column RFC 0019.13 "
            "§11.2 requires. An L2 collection mints its own graph ids; join on `name` "
            "there, which the graph document carries."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")

    columns = ["benchmark", "name", "regime", "phase", "context", "grouping", "source",
               "origin", "op", "dtype", "batch", "heads_q", "heads_kv", "seqlen_q",
               "seqlen_kv", "head_dim", "causal", "alignment", "bytes", "file"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return manifest
