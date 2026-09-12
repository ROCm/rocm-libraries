# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Writing a shape out as the hipDNN graph JSON both readers take.

`uhd_gen generate --graphs` and `hipdnn_bench --graph` read the same document, and
`make_sdpa_bundles.bundle_for` already writes it correctly -- including the two
mistakes that cost a whole sweep each: `half` rather than `float16`, and the
present-but-null attention window without which `json::to<Graph>` rejects every
nomask graph. So this module converts and delegates; it does not build a graph.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

from .shapes import Shape

#: `projects/hipdnn/tools`, where `make_sdpa_bundles.py` lives.
_TOOLS_ROOT = Path(__file__).resolve().parent.parent


def _load_make_sdpa_bundles():
    """The KDP reader and bundle writer, imported however it is reachable.

    It is a script at the tools root rather than an installed package, so `python -m
    corpus_build` run from that root imports it by name and a run from anywhere else
    does not. Loading it by path in that case is what its own test does
    (`tools/tests/test_make_sdpa_bundles.py`); the alternative -- copying `bundle_for`
    -- is a second graph writer that drifts from the shipped bundles.
    """
    if "make_sdpa_bundles" in sys.modules:
        return sys.modules["make_sdpa_bundles"]
    try:
        import make_sdpa_bundles  # noqa: PLC0415  (deliberate: see docstring)

        return make_sdpa_bundles
    except ImportError:
        pass
    path = _TOOLS_ROOT / "make_sdpa_bundles.py"
    spec = importlib.util.spec_from_file_location("make_sdpa_bundles", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the KDP reader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_sdpa_bundles"] = module
    spec.loader.exec_module(module)
    return module


bundles = _load_make_sdpa_bundles()

#: The KDP metadata keys that spell one geometry, in `make_sdpa_bundles`' order.
GEOMETRY = bundles.GEOMETRY


def footprint_bytes(shape: Shape) -> int:
    """Q, K, V and O in bytes -- the allocation a benchmark of this graph needs."""
    return bundles.footprint_bytes(shape.geometry())


def document(shape: Shape) -> dict:
    """The graph document for one shape, named with its regime.

    `Graph.name` is `cache_ignore` in `flatbuffers_sdk/schemas/graph.fbs`, so naming a
    graph cannot perturb the engine cache key the way an attribute would: the name is
    free to carry the regime, and RFC 0019.13 §11.2's per-regime table is the reason
    to spend it.
    """
    graph = bundles.bundle_for(shape.geometry())
    graph["name"] = shape.name
    return graph


def canonical(graph: dict) -> str:
    """The exact byte-for-byte form `uhd_gen generate` hashes a graph's identity from."""
    return json.dumps(graph, sort_keys=True, separators=(",", ":"), allow_nan=False)


def graph_id(graph: dict) -> str:
    """The `benchmark` identity an L1 collection will record for this graph.

    `uhd_gen.generate` mints a UUID5 of the canonical document for any ID-less JSON
    graph and the corpus row's `benchmark` column is that id -- so computing the same
    id here is what lets the manifest's regime column be joined to the collected
    corpus, which is the whole point of emitting a regime at all. No `id` is written
    into the graph itself: `Graph.id` is a `Uuid` table in the schema, not a string,
    and the shipped bundles carry none.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "hipdnn:graph:" + canonical(graph)))


def write(directory: Path, shape: Shape) -> tuple[str, str]:
    """Write one graph. Returns its file name and the id an L1 collection will use."""
    graph = document(shape)
    file_name = shape.name + ".json"
    (directory / file_name).write_text(json.dumps(graph, indent=4) + "\n", encoding="utf-8")
    return file_name, graph_id(graph)
