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

#: Tensor names that mark a graph as backward rather than forward, in BOTH
#: gradient spellings a corpus uses. `d_query`-style names alone let
#: `sample_sdpa_backward` (whose gradients are `dq`/`dk`/`dv`/`do`) through this
#: filter, where it was then caught only incidentally by its `float` dtype -- a
#: backward graph using a servable dtype would have been mined as a forward
#: shape. Module-level so a consumer outside this file (e.g. a config's own
#: EXCLUDE_TENSORS list) can be checked against the same set rather than a
#: second literal that can silently drift from this one.
BACKWARD_GRADIENT_TENSOR_NAMES = {
    "d_query",
    "d_key",
    "d_value",
    "d_output",
    "dq",
    "dk",
    "dv",
    "do",
}


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
                    "dtype": _normalise_dtype(row.get("dtype"), path, "bf16"),
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
        if left is not None and not isinstance(left, (int, float)):
            raise SystemExit(
                f"FAIL: non-numeric left_bound {left!r} in {path}. Refusing rather "
                f"than defaulting -- an unresolvable bound falling through to "
                f"'causal' is exactly the wrong-answer-not-a-decline failure this "
                f"reader exists to refuse."
            )
        # left_bound < 0 means "all history": causal. A finite left_bound is a
        # sliding window, which is its own mask kind.
        if isinstance(left, (int, float)) and left >= 0:
            return _MASK_TYPE["swin"]
        return _MASK_TYPE["causal"]
    # No mask attributes at all: the graph does not describe one. Say so by falling
    # back to the path, and only then -- a directory name is a hint, not a contract.
    return _MASK_TYPE["causal"] if "causal" in str(path).lower() else _MASK_TYPE["full"]


#: Every spelling a source uses for a dtype -> the spelling the rocKE spec takes.
#: Three vocabularies meet here and none of them agree: hipDNN graphs say
#: `bfloat16`, torch traces say `torch.bfloat16`, the spec says `bf16`. A source
#: dtype that reaches the dispatcher unmapped is REJECTED at spec construction
#: ("dtype must be one of ['bf16', 'fp16']"), which reads like the kernel declining
#: a shape when it is really the miner mis-spelling one -- and the whole graph
#: corpus disappears from the servable count that way.
_DTYPE_SPELLINGS = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
}


def _normalise_dtype(raw, path: Path, fallback: str) -> str:
    """One spelling for a dtype, or a refusal naming the source.

    Refuses rather than defaults, for the same reason the mask derivation does: a
    guessed dtype builds a different binary and still validates, so the failure is
    silent and numeric. An ABSENT dtype falls back (the source simply did not say);
    an UNRECOGNISED one is a mapping this table owes, not a value to paper over.
    """
    if raw is None or str(raw).strip() == "":
        return fallback
    spelling = str(raw).strip().lower()
    resolved = _DTYPE_SPELLINGS.get(spelling)
    if resolved is None:
        raise SystemExit(
            f"FAIL: unknown dtype spelling {raw!r} in {path}. Add it to "
            f"_DTYPE_SPELLINGS rather than defaulting -- a guessed dtype builds the "
            f"wrong binary and still validates."
        )
    return resolved


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
        # the device down through a third-party backward FMHA. The filename is not
        # authoritative, so the marker is structural -- but it has to cover BOTH
        # gradient spellings (see BACKWARD_GRADIENT_TENSOR_NAMES). `d_query`-style
        # names alone let `sample_sdpa_backward` (whose gradients are
        # `dq`/`dk`/`dv`/`do`) through the filter, where it was then caught only
        # incidentally by its `float` dtype. A backward graph that happened to use a
        # servable dtype would have been mined as a forward shape.
        #
        # The node's own op type is the primary marker, since that is what the graph
        # DECLARES it is; the tensor-name sets are the belt-and-braces fallback for a
        # graph whose node type is absent or spelled differently.
        node_types = {str(n.get("type", "")).lower() for n in graph.get("nodes") or []}
        if any("backward" in t or "bwd" in t for t in node_types):
            continue
        if BACKWARD_GRADIENT_TENSOR_NAMES & set(tensors):
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
                "dtype": _normalise_dtype(query.get("data_type"), path, "bf16"),
                "mask_type": _mask_type_from_graph(graph, path),
                "_provenance": {
                    "source": "graphs",
                    "suite": str(path.parent.name),
                    "graph": path.stem,
                },
            }
        )
    return shapes


def _bench_graph_name(path: Path, record: dict) -> str:
    """A stable, human-readable name for one rocKE benchmark trace record.

    Exists because `graph` is the key a `--declines` file is written against, and the
    only alternative the reconciler accepts is the corpus INDEX. An index is a
    position, not an identity: re-mine with a different flag, or land a new trace
    upstream, and every key after the insertion point now marks a DIFFERENT shape.
    The reconciler hard-fails a key matching nothing -- which is right, and does not
    help here, because a shifted index still matches something.

    So the name is built from what the record says about itself rather than where it
    sits: the trace file it came from, its own `variant` label when the suite records
    one, and `call_idx` as the tiebreak for suites that do not. Prefixed with the
    source so it can never collide with a dnn-benchmarking graph stem, which shares
    this field.
    """
    parts = [path.stem]
    variant = str(record.get("variant") or "").strip()
    if variant:
        parts.append(variant)
    # ALWAYS append the shape, even when a variant label exists. A name that does not
    # identify exactly one shape is not usable as a declines key: the `aiter` suite
    # records no `variant` at all, so a name built from the trace stem alone collapsed
    # 82 records onto one key. `call_idx` is deliberately NOT used -- it is a position
    # in a capture, which is the very instability this function exists to avoid.
    # Two records that agree on every one of these fields ARE the same shape and are
    # merged by deduplicate() anyway, so collisions here are correct rather than lossy.
    parts.append(
        f"b{record.get('num_seqs')}_hq{record.get('num_query_heads')}"
        f"_kv{record.get('num_kv_heads')}_d{record.get('head_size')}"
        f"_sq{record.get('max_seqlen_q')}_sk{record.get('max_seqlen_k')}"
    )
    return "rocke_bench__" + "__".join(parts)


def from_rocke_bench(root: Path, dtype_default: str) -> list[dict]:
    """Shapes from rocKE's OWN benchmark tree -- the third source, and for an arch
    with no published CSV it is the only one that says what the kernel team measures.

    Two formats live side by side under `benchmarks/<arch>/attention/`, and they are
    not interchangeable:

      * `*_shapes.json` / `*_bench.json` -- JSONL, ONE RECORD PER LINE (not a JSON
        document; `json.load` raises "Extra data" on all three of them). These are
        captured launch traces: real shapes, with `window_size` and `has_sinks` as
        genuine recorded attributes.
      * `benchmark_*_live.py` -- the sweep that GENERATES shapes, whose `_configs()`
        enumerates (seqlens, Hq, Hkv, W, persistent) per mode.

    CAUSALITY IS NOT IN THE TRACES. No record in any of the three JSONL files carries
    a causal/mask key -- verified by set-union over every key present. The dispatcher
    does `causal = (mask_type != 0)`, so guessing it picks which branch resolves and
    which kernels get built, and a prefill trace defaulted to non-causal sizes a
    variant set that cannot serve the causal traffic it was mined from. So this
    refuses rather than defaults, exactly as the op-shaped-miner contract requires of
    an unrecognised categorical: a trace states causality through `window_size`, or
    it is skipped and counted.

    `window_size` is `[left, right]` in the kernel's own convention, matching the
    graph side's (`left_bound`, `right_bound`) pair:
      * `[-1, -1]` -- unbounded both ways. Prefill attention with no window is
        CAUSAL by construction here (these are prefill suites; `ALL_DECODE` is false
        on every record), and the paired `benchmark_dense_prefill_live.py` labels the
        W=0 arm "full-causal" rather than "no mask".
      * `[W, 0]` with W >= 0 -- a banded causal window: right bound 0 is the causal
        clamp, finite left bound is the window. A DIFFERENT mask kind, never folded
        onto plain causal.
    """
    shapes: list[dict] = []
    skipped_unknown_mask = 0
    for path in sorted(root.rglob("*.json")):
        text = path.read_text().strip()
        if not text:
            continue
        records = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                records = []
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records = []
                break
        for record in records:
            if record.get("ALL_DECODE"):
                continue
            window = record.get("window_size")
            if not (isinstance(window, list) and len(window) == 2):
                # No recorded causality and no way to derive it. Counted, not
                # defaulted -- see the docstring.
                skipped_unknown_mask += 1
                continue
            left, right = window
            if left is None or right is None:
                skipped_unknown_mask += 1
                continue
            # The WIDTH is carried, not just the kind. A windowed shape whose width
            # is dropped reaches the dispatcher as sliding_window=0, which resolves
            # to plain causal -- the kernel then computes a full causal triangle for
            # a banded request and returns a WRONG ANSWER rather than declining. The
            # mask kind alone does not encode the window; both must travel.
            sliding_window = 0
            if int(left) < 0 and int(right) < 0:
                mask_type = _MASK_TYPE["causal"]
            elif int(left) >= 0:
                mask_type = _MASK_TYPE["swin"]
                # `[W, 0]` is a banded causal window of left-context W. The spec
                # counts the window in TOKENS including the current one, matching
                # the kernel's `q-W+1 <= k <= q` band, so a recorded left bound of
                # 127 is a 128-token window.
                sliding_window = int(left) + 1
            else:
                mask_type = _MASK_TYPE["causal"]
            head_size = record.get("head_size")
            seqlen_q = record.get("max_seqlen_q")
            seqlen_k = record.get("max_seqlen_k")
            heads_q = record.get("num_query_heads")
            heads_kv = record.get("num_kv_heads")
            if None in (head_size, seqlen_q, seqlen_k, heads_q, heads_kv):
                continue
            # `q_dtype` is a torch spelling ("torch.bfloat16"), normalised through
            # the same table the graph corpus uses -- one vocabulary, one place to
            # add a spelling, rather than two that can disagree.
            dtype = _normalise_dtype(record.get("q_dtype"), path, dtype_default)
            shapes.append(
                {
                    "batch": int(record.get("num_seqs") or 1),
                    "nhead_q": int(heads_q),
                    "nhead_k": int(heads_kv),
                    "seqlen_q": int(seqlen_q),
                    "seqlen_k": int(seqlen_k),
                    "hdim_q": int(head_size),
                    "hdim_v": int(head_size),
                    "dtype": dtype,
                    "mask_type": mask_type,
                    "sliding_window": sliding_window,
                    # A recorded request attribute, not a tuning choice. Carried so
                    # the dispatcher resolves the shape the trace actually asked for;
                    # whether THIS integration ships a sink variant is a scope
                    # decision made downstream, and filtering here would hide the
                    # shape from the step-9 reconciler entirely.
                    "use_sinks": bool(record.get("has_sinks")),
                    "_provenance": {
                        "source": "rocke_bench",
                        "suite": str(path.parent.name),
                        "trace": path.stem,
                        # A STABLE NAME for this shape, because `graph` is the key a
                        # --declines file is written against and the alternative is a
                        # corpus INDEX. An index shifts the moment the corpus is
                        # re-mined with different flags or a new trace lands, and the
                        # same declines file then marks a DIFFERENT shape -- silently,
                        # since a key that matches nothing is only a hard error, not a
                        # correction. Derived from the trace and the record's own
                        # variant/call_idx so it survives re-mining, and prefixed with
                        # the source so it cannot collide with a dnn-benchmarking
                        # graph stem.
                        "graph": _bench_graph_name(path, record),
                        "model": str(record.get("model") or ""),
                        "variant": str(record.get("variant") or ""),
                        # Recorded, and load-bearing for scope: a sink trace is a
                        # shape this integration declines on purpose, and the step-9
                        # reconciler needs to see it rather than have it filtered out
                        # here.
                        "has_sinks": bool(record.get("has_sinks")),
                    },
                }
            )
    if skipped_unknown_mask:
        print(
            f"  NOTE: {skipped_unknown_mask} rocKE trace record(s) skipped -- no "
            f"recorded causality to derive a mask from. Not defaulted: a prefill "
            f"trace read as non-causal sizes a set that cannot serve it."
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
    parser.add_argument(
        "--rocke-bench",
        help="rocKE's own benchmarks/<arch>/attention tree. The third source, and "
        "the only one that says what the kernel team measures on an arch with no "
        "published results CSV.",
    )
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

    if not args.published and not args.graphs and not args.rocke_bench:
        parser.error(
            "give at least one source. No corpus alone is sufficient: the CSV is "
            "what the kernel team measures, the graph tree is what callers send, "
            "rocKE's bench tree is what the kernel's own authors sweep, and an "
            "integration sized from only one of them has missed real shapes twice."
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
    if args.rocke_bench:
        found = from_rocke_bench(Path(args.rocke_bench), "bf16")
        print(f"  rocKE bench   : {len(found):5d} trace records")
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
