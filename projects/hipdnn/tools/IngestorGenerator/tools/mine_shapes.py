"""Build the shape corpus stage 4a resolves, from the sources that actually decide.

Three sources answer three different questions, and no one of them is sufficient:

  * the kernel team's PUBLISHED RESULTS CSV -- what they measure, tune and will
    escalate a regression on. It is the shape list ALREADY RESOLVED, with priority
    and ticket group attached, and it carries shapes a benchmark's nested loops do
    not enumerate. Ask for it before mining anything.
  * dnn-benchmarking's graph corpus -- what real callers ask for.
  * the kernel's own `supports_*` predicate -- what is legal to build.

The first two are the ones an integration keeps skipping. Every source the mining
guidance originally named was kernel-side, so it answered "what is LEGAL?" and
nothing answered "what will anyone ASK for?". Following it exactly produced a legal,
validated, well-tested engine that served zero real workloads -- three times, each
caught only by counting against an external corpus rather than from inside the
integration.

PROVENANCE SURVIVES INTO THE NAME. Every emitted shape carries where it came from,
because the moment a result can be split by source it stops being one number: the
same measured win was large on one synthetic microbenchmark suite and close to
parity on real model traces, and only the provenance split made that visible rather
than suspected. It costs nothing here and cannot be recovered later.

A `microbench/` path is a PROVENANCE LABEL, not a synthetic-data warning. One suite
was discarded on the strength of its directory name and its own manifest said the
opposite -- every shape rendered from a real source, none invented. That mistake cost
72 shapes.

    mine_shapes.py --published <csv> --arch gfx942 --out shapes.json

Emits the request-field mappings `dispatch_parity.py --shapes` consumes. It does NOT
filter by what the kernel can serve: that is the dispatcher's job at stage 4a, which
reports declines with reasons. Filtering here would hide the gap this corpus exists
to measure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: CSV mask spellings -> the request's mask_type. `swin` is a sliding window, which
#: is a different mask kind rather than a causal variant; folding it onto causal
#: collapsed seven distinct shape keys in an earlier join. It is carried through with
#: its own value so the dispatcher declines it explicitly instead of it silently
#: becoming a causal duplicate.
_MASK_TYPE = {"full": 0, "none": 0, "causal": 1, "swin": 2}


def from_published_csv(path: Path, arch: str, include_windowed: bool) -> list[dict]:
    """Shapes from the kernel team's results CSV.

    Why this beats reading the benchmark source: it is the shape list already
    resolved, it names which kernel each published number refers to, and it carries
    `priority`/`ticket_group` -- a shipping-priority signal available nowhere else.
    """
    shapes = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("arch") != arch:
                continue
            mask = (row.get("mask") or "").strip().lower()
            if mask == "swin" and not include_windowed:
                continue
            mask_type = _MASK_TYPE.get(mask)
            if mask_type is None:
                raise SystemExit(
                    f"FAIL: unknown mask spelling {mask!r} in {path}. Add it to "
                    f"_MASK_TYPE rather than defaulting -- guessing a mask is how a "
                    f"windowed graph gets served as plain causal."
                )
            head_dim = int(row["head_dim"])
            shapes.append(
                {
                    "batch": int(row["batch"]),
                    "nhead_q": int(row["heads_q"]),
                    "nhead_k": int(row["heads_kv"]),
                    "seqlen_q": int(row["seq_q"]),
                    "seqlen_k": int(row["seq_kv"]),
                    "hdim_q": head_dim,
                    "hdim_v": head_dim,
                    "dtype": (row.get("dtype") or "bf16").strip().lower(),
                    "mask_type": mask_type,
                    # Provenance, carried not computed. `_provenance` is stripped
                    # before the request is constructed and kept for reporting.
                    "_provenance": {
                        "source": "published",
                        "model": row.get("model") or "",
                        "category": row.get("category") or "",
                        "priority": row.get("priority") or "",
                        "ticket_group": row.get("ticket_group") or "",
                        "shape_idx": row.get("shape_idx") or "",
                    },
                }
            )
    return shapes


def _mask_type_from_graph(graph: dict, path: Path) -> int:
    """Causality from the graph's OWN attributes, never from its filename.

    The first version read `"causal" in path.stem.lower()`. Against this repo's real
    bundle tree that is wrong for every causal graph there is: 25 of them carry
    `causal` in a PARENT DIRECTORY (`.../hd128_causal_batch/Small/Small.json`) and
    none carry it in the leaf name, so the miner reported a corpus with zero causal
    graphs. `causal` is not cosmetic -- the dispatcher does `causal=(mask_type != 0)`,
    so it selects which branch resolves and which kernels get built. A corpus that
    reports no causal graphs sizes a variant set that cannot serve them.

    Reading the attributes is also the only correct derivation, independent of naming.
    hipDNN has NO `causal` boolean: the deprecated `causal_mask` /
    `causal_mask_bottom_right` pair takes precedence WHEN SET, and otherwise causality
    comes from (`left_bound`, `right_bound`, `diagonal_alignment`). Every shipped
    causal bundle in this tree leaves both booleans false and expresses causality as
    `left_bound=-1, right_bound=0` -- so a reader that trusts only the booleans
    computes "not causal" for all of them. That derivation is the single
    highest-value paragraph in this skill's own graph contract, and the filename
    heuristic bypassed it entirely.

    A windowed graph is NOT causal-with-a-tweak: a finite `left_bound` is a sliding
    window, a different mask kind, and folding it onto causal is how one gets served
    as plain causal -- a wrong answer rather than a decline.
    """
    for node in graph.get("nodes") or []:
        attrs = node.get("attributes") or {}
        if not any(
            k in attrs
            for k in ("causal_mask", "causal_mask_bottom_right", "left_bound")
        ):
            continue
        if attrs.get("causal_mask") or attrs.get("causal_mask_bottom_right"):
            return _MASK_TYPE["causal"]
        left = attrs.get("left_bound")
        right = attrs.get("right_bound")
        if left is None and right is None:
            return _MASK_TYPE["full"]
        # left_bound < 0 means "all history": causal. A finite left_bound is a
        # sliding window, which is its own mask kind.
        if isinstance(left, (int, float)) and left >= 0:
            return _MASK_TYPE["swin"]
        return _MASK_TYPE["causal"]
    # No mask attributes at all: the graph does not describe one. Say so by falling
    # back to the path, and only then -- a directory name is a hint, not a contract.
    return _MASK_TYPE["causal"] if "causal" in str(path).lower() else _MASK_TYPE["full"]


def from_graph_corpus(root: Path) -> list[dict]:
    """Shapes from a dnn-benchmarking graph tree, one JSON per graph.

    The suite name is kept because it is the axis a result must be split along. Real
    model traces and parameter sweeps do not behave alike, and a single geomean over
    both reports the synthetic population's win as though it were everyone's.
    """
    shapes = []
    for path in sorted(root.rglob("*.json")):
        try:
            graph = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tensors = {
            str(t.get("name", "")).lower(): t for t in graph.get("tensors", []) or []
        }
        # A backward graph cannot be served by a prefill kernel, and one of them takes
        # the device down through a third-party backward FMHA. Gradient tensors are the
        # structural marker; the filename is not authoritative.
        if {"d_query", "d_key", "d_value", "d_output"} & set(tensors):
            continue
        query = tensors.get("query") or tensors.get("q")
        key = tensors.get("key") or tensors.get("k")
        if not query or not key:
            continue
        qdims = query.get("dims") or []
        kdims = key.get("dims") or []
        if len(qdims) != 4 or len(kdims) != 4:
            continue
        shapes.append(
            {
                "batch": int(qdims[0]),
                "nhead_q": int(qdims[1]),
                "nhead_k": int(kdims[1]),
                "seqlen_q": int(qdims[2]),
                "seqlen_k": int(kdims[2]),
                "hdim_q": int(qdims[3]),
                "hdim_v": int(qdims[3]),
                "dtype": str(query.get("data_type", "bf16")).lower(),
                "mask_type": _mask_type_from_graph(graph, path),
                "_provenance": {
                    "source": "graphs",
                    "suite": str(path.parent.name),
                    "graph": path.stem,
                },
            }
        )
    return shapes


def _shape_key(shape: dict) -> tuple:
    return tuple(
        shape[k]
        for k in (
            "batch",
            "nhead_q",
            "nhead_k",
            "seqlen_q",
            "seqlen_k",
            "hdim_q",
            "hdim_v",
            "dtype",
            "mask_type",
        )
    )


def deduplicate(shapes: list[dict]) -> tuple[list[dict], int]:
    """One entry per distinct shape, keeping the first provenance and counting the rest.

    A corpus is a set of shapes, not a set of rows. Two suites asking for the same
    shape is one variant to compile -- but it is two votes for that shape mattering,
    so the duplicate count is reported rather than discarded.
    """
    seen: dict = {}
    duplicates = 0
    for shape in shapes:
        key = _shape_key(shape)
        if key in seen:
            duplicates += 1
            seen[key]["_provenance"].setdefault("also", []).append(
                shape["_provenance"].get("suite")
                or shape["_provenance"].get("model")
                or shape["_provenance"].get("source")
            )
            continue
        seen[key] = shape
    return list(seen.values()), duplicates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine a shape corpus from the sources that decide what to ship.",
    )
    parser.add_argument("--published", help="The kernel team's results CSV.")
    parser.add_argument("--graphs", help="A dnn-benchmarking graph tree.")
    parser.add_argument("--arch", default="gfx942", help="Filter the CSV to one arch.")
    parser.add_argument(
        "--include-windowed",
        action="store_true",
        help="Keep sliding-window rows. Off by default: they are a different mask "
        "kind, and a kernel that clamps top-left only will decline them anyway -- "
        "but they are excluded LOUDLY here rather than folded onto causal.",
    )
    parser.add_argument("--out", required=True, help="Write the shape corpus here.")
    args = parser.parse_args(argv)

    if not args.published and not args.graphs:
        parser.error(
            "give at least one source. Neither corpus alone is sufficient: the CSV is "
            "what the kernel team measures, the graph tree is what callers send, and "
            "an integration sized from only one of them has missed real shapes twice."
        )

    shapes: list[dict] = []
    if args.published:
        found = from_published_csv(
            Path(args.published), args.arch, args.include_windowed
        )
        print(f"  published CSV : {len(found):5d} rows for {args.arch}")
        shapes += found
    if args.graphs:
        found = from_graph_corpus(Path(args.graphs))
        print(f"  graph corpus  : {len(found):5d} forward graphs")
        shapes += found

    unique, duplicates = deduplicate(shapes)
    print(
        f"  distinct      : {len(unique):5d}  ({duplicates} duplicate shape(s) merged)"
    )

    if not unique:
        print(
            "\nFAIL: no shapes mined; nothing downstream can use this.", file=sys.stderr
        )
        return 1

    by_source: dict = {}
    for shape in unique:
        by_source.setdefault(shape["_provenance"]["source"], 0)
        by_source[shape["_provenance"]["source"]] += 1
    print(f"  by source     : {by_source}")

    Path(args.out).write_text(json.dumps(unique, indent=2))
    print(f"\n  wrote {args.out}")
    print(
        "  Provenance is carried on every shape. Split every reported result by it: a "
        "geomean over a mixed corpus reports the synthetic population's win as if it "
        "were everyone's."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
