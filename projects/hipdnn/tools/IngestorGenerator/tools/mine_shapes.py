"""Build the shape corpus the dispatcher resolves (RUNBOOK §2's corpus).

Three sources answer three different questions, and none is sufficient alone:

  * the kernel team's published results CSV -- shapes already resolved, with
    priority and ticket group attached. Ask for it before mining anything.
  * dnn-benchmarking's graph corpus -- what real callers ask for.
  * the kernel's own `supports_*` predicate -- what is legal to build.

Kernel-side sources answer what is legal; the first two answer what anyone
asks for. Every emitted shape carries its provenance so a result can be split
by source; a `microbench/` path is a provenance label, not a synthetic-data
warning.

    mine_shapes.py --published <csv> --arch gfx942 --out shapes.json

Emits the request-field mappings `dispatch_parity.py --shapes` consumes, and
does not filter by what the kernel can serve: the dispatcher reports declines
with reasons, and filtering here would hide the gap this corpus measures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

#: CSV mask spellings -> the request's mask_type. `swin` is a sliding window, a
#: different mask kind rather than a causal variant, carried with its own value
#: so the dispatcher declines it explicitly.
_MASK_TYPE = {"full": 0, "none": 0, "no_mask": 0, "causal": 1, "top_left": 1,
              "swin": 2}

#: Tensor names marking a graph as backward, in both gradient spellings a
#: corpus uses: `d_query`-style names alone would let `dq`/`dk`/`dv`/`do`
#: through. Module-level so a consumer outside this file (e.g. a config's
#: EXCLUDE_TENSORS list) checks against the same set.
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
    """Shapes from the kernel team's results CSV: the shape list already
    resolved, naming which kernel each published number refers to and carrying
    `priority`/`ticket_group`, a signal available nowhere else."""
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
            window = 0
            if mask_type == _MASK_TYPE["swin"]:
                raw_window = (row.get("window_size") or "").strip()
                if not raw_window.isdigit() or int(raw_window) <= 0:
                    raise SystemExit(
                        f"FAIL: {path}: windowed CSV row requires a positive window_size width"
                    )
                window = int(raw_window)
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
                    "sliding_window": window,
                    "use_sinks": False,
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


def _mask_from_attributes(
    attrs: dict, path: Path, seqlen_q: int, seqlen_k: int
) -> dict:
    """Normalize the graph dialect to a mask kind, a window, and an anchor.

    The anchor is REPORTED rather than normalized away. Folding bottom-right onto
    top-left is safe only where Sq == Sk, and a UHD corpus deliberately contains the
    case where it is not: `corpus_build.both_anchors` twins every causal shape so an
    engine whose table is bottom-right-only is exercised (AITER's gfx942 forward table
    has no top-left causal kernel, and a single-anchor corpus had it serving none of
    its 15 causal graphs). Refusing those made this reader unable to read the corpus
    the UHD trains on.

    Consumers that only ever see Sq == Sk keep reading `mask_type` and `sliding_window`
    and are unaffected; `alignment` is additive.
    """
    alignment = attrs.get("diagonal_alignment", "TOP_LEFT")
    if alignment not in ("TOP_LEFT", "BOTTOM_RIGHT"):
        raise SystemExit(f"FAIL: {path}: unsupported diagonal_alignment {alignment!r}")
    for flag in ("causal_mask", "causal_mask_bottom_right"):
        if flag in attrs and type(attrs[flag]) is not bool:
            raise SystemExit(f"FAIL: {path}: {flag} must be boolean")
    left, right = attrs.get("left_bound"), attrs.get("right_bound")
    for name, value in (("left_bound", left), ("right_bound", right)):
        if value is not None and (type(value) is not int or value < -1):
            raise SystemExit(
                f"FAIL: {path}: invalid {name} {value!r}; expected null or integer >= -1"
            )
    if attrs.get("causal_mask"):
        left, right, alignment = -1, 0, "TOP_LEFT"
    elif attrs.get("causal_mask_bottom_right"):
        left, right, alignment = -1, 0, "BOTTOM_RIGHT"
    left = -1 if left is None else left
    right = -1 if right is None else right
    if left == -1 and right == -1:
        return {"mask_type": 0, "sliding_window": 0, "alignment": "top_left"}
    if right != 0:
        # A right bound other than 0 is not a causal mask at all, and there is no
        # anchor that makes it one. Still refused.
        raise SystemExit(
            f"FAIL: {path}: unsupported translation of bounds ({left}, {right}), "
            f"alignment {alignment}, Sq={seqlen_q}, Sk={seqlen_k} to AttentionRequest"
        )
    return {
        "mask_type": 1 if left == -1 else 2,
        "sliding_window": 0 if left == -1 else left + 1,
        "alignment": "bottom_right" if alignment == "BOTTOM_RIGHT" else "top_left",
    }


#: Every spelling a source uses for a dtype -> the spelling the rocKE spec
#: takes. hipDNN graphs say `bfloat16`, torch traces `torch.bfloat16`, the spec
#: `bf16`. An unmapped dtype is rejected at spec construction, which reads like
#: the kernel declining a shape when the miner mis-spelled one.
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

    An absent dtype falls back; an unrecognised one is a mapping this table
    owes, since a guessed dtype builds a different binary and still validates.
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
    """Shapes from a dnn-benchmarking graph tree, one JSON per graph. The suite
    name is kept because it is the axis a result must be split along: a single
    geomean over model traces and parameter sweeps reports the synthetic
    population's win as everyone's."""
    shapes = []
    for path in sorted(root.rglob("*.json")):
        try:
            graph = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # A document that is not a graph object is skipped exactly like an unparseable
        # one: this reader mines graphs and says nothing about anything else. Load-bearing
        # for corpus_build, which points it at a published shape directory holding CSV,
        # record-style JSON arrays and key=value files beside the graphs -- all of which
        # parse as JSON and none of which carry tensors.
        if not isinstance(graph, dict):
            continue
        tensors = {
            str(t.get("name", "")).lower(): t
            for t in graph.get("tensors", []) or []
            if isinstance(t, dict)
        }
        # A backward graph cannot be served by a prefill kernel. The filename
        # is not authoritative, so the marker is structural: the node's own op
        # type is primary, with BACKWARD_GRADIENT_TENSOR_NAMES as the fallback
        # for a graph whose node type is absent or spelled differently.
        node_types = {str(n.get("type", "")).lower() for n in graph.get("nodes") or []}
        if any("backward" in t or "bwd" in t for t in node_types):
            continue
        if BACKWARD_GRADIENT_TENSOR_NAMES & set(tensors):
            continue
        sdpa = [
            n
            for n in graph.get("nodes", [])
            if n.get("type") == "SdpaAttributes"
            or "q_tensor_uid" in (n.get("attributes") or {})
        ]
        if len(sdpa) > 1:
            raise SystemExit(
                f"FAIL: {path}: multiple SDPA nodes cannot form one request"
            )
        attrs = (
            (sdpa[0].get("attributes") or {})
            if sdpa
            else next(
                (
                    n["attributes"]
                    for n in graph.get("nodes", [])
                    if n.get("attributes")
                ),
                {},
            )
        )
        by_uid = {t["uid"]: t for t in graph.get("tensors", []) if "uid" in t}
        selected = []
        for short, long in (("q", "query"), ("k", "key"), ("v", "value")):
            uid_key = f"{short}_tensor_uid"
            tensor = (
                by_uid.get(attrs[uid_key])
                if uid_key in attrs
                else (tensors.get(long) or tensors.get(short))
            )
            selected.append(tensor)
        query, key, value = selected
        if not query and not key and not sdpa:
            continue
        if any(t is None for t in selected):
            raise SystemExit(
                f"FAIL: {path}: SDPA requires independent Q, K and V tensors"
            )
        dimensions = [t.get("dims") or [] for t in selected]
        if any(
            len(d) != 4 or any(type(x) is not int or x <= 0 for x in d)
            for d in dimensions
        ):
            raise SystemExit(
                f"FAIL: {path}: SDPA requires positive logical BHSD dimensions"
            )
        qdims, kdims, vdims = dimensions
        if qdims[0] != kdims[0] or kdims[:3] != vdims[:3] or qdims[3] != kdims[3]:
            raise SystemExit(
                f"FAIL: {path}: incompatible independent Q/K/V dimensions {dimensions}"
            )
        dtypes = [_normalise_dtype(t.get("data_type"), path, "bf16") for t in selected]
        if len(set(dtypes)) != 1:
            raise SystemExit(
                f"FAIL: {path}: mixed Q/K/V dtypes cannot form one request"
            )
        mask = _mask_from_attributes(attrs, path, qdims[2], kdims[2])
        sink_uid = attrs.get("sink_token_tensor_uid")
        if sink_uid is not None and sink_uid not in by_uid:
            raise SystemExit(
                f"FAIL: {path}: sink_token_tensor_uid names a missing tensor"
            )
        shapes.append(
            {
                "batch": int(qdims[0]),
                "nhead_q": int(qdims[1]),
                "nhead_k": int(kdims[1]),
                "seqlen_q": int(qdims[2]),
                "seqlen_k": int(kdims[2]),
                "hdim_q": int(qdims[3]),
                "hdim_v": int(vdims[3]),
                "dtype": dtypes[0],
                **mask,
                "use_sinks": sink_uid is not None,
                "_provenance": {
                    "source": "graphs",
                    "suite": str(path.parent.name),
                    "graph": graph.get("name", path.stem),
                    "path": str(path),
                    "mask": {
                        k: attrs.get(k)
                        for k in (
                            "left_bound",
                            "right_bound",
                            "diagonal_alignment",
                            "causal_mask",
                            "causal_mask_bottom_right",
                        )
                    },
                },
            }
        )
    return shapes


def _bench_graph_name(shape: dict) -> str:
    """A stable runtime identity from the complete request, never capture position."""
    return "rocke_bench__" + hashlib.sha256(_shape_key(shape).encode()).hexdigest()


def from_rocke_bench(root: Path, dtype_default: str) -> list[dict]:
    """Shapes from rocKE's own benchmark tree; for an arch with no published
    CSV it is the only source saying what the kernel team measures.

    `*_shapes.json` / `*_bench.json` under `benchmarks/<arch>/attention/` are
    JSONL, one record per line (`json.load` raises "Extra data"): captured
    launch traces carrying `window_size` and `has_sinks`. The paired
    `benchmark_*_live.py` generates shapes instead.

    Causality is not recorded and the dispatcher does
    `causal = (mask_type != 0)`, so a trace states causality through
    `window_size` or it is skipped and counted. `window_size` is
    `[left, right]`: `[-1, -1]` is unbounded both ways, causal for these
    prefill suites, and `[W, 0]` with W >= 0 is a banded causal window, never
    folded onto plain causal.
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
            # The width is carried, not just the kind: a windowed shape whose
            # width is dropped reaches the dispatcher as sliding_window=0,
            # resolves to plain causal, and the kernel computes a full causal
            # triangle for a banded request -- a wrong answer, not a decline.
            sliding_window = 0
            if int(left) < 0 and int(right) < 0:
                mask_type = _MASK_TYPE["causal"]
            elif int(left) >= 0 and int(right) == 0:
                mask_type = _MASK_TYPE["swin"]
                # The spec counts the window in TOKENS including the current one,
                # matching the kernel's `q-W+1 <= k <= q` band, so a recorded left
                # bound of 127 is a 128-token window.
                sliding_window = int(left) + 1
            elif int(left) == -1 and int(right) == 0:
                mask_type = _MASK_TYPE["causal"]
            else:
                raise SystemExit(f"FAIL: {path}: unsupported trace window {window!r}")
            head_size = record.get("head_size")
            seqlen_q = record.get("max_seqlen_q")
            seqlen_k = record.get("max_seqlen_k")
            heads_q = record.get("num_query_heads")
            heads_kv = record.get("num_kv_heads")
            if None in (head_size, seqlen_q, seqlen_k, heads_q, heads_kv):
                continue
            # `q_dtype` is a torch spelling ("torch.bfloat16"), normalised
            # through the same table the graph corpus uses.
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
                    # A recorded request attribute, not a tuning choice: the
                    # dispatcher resolves the shape the trace asked for.
                    # Whether this integration ships a sink variant is decided
                    # downstream, and filtering here would hide the shape from
                    # runtime reconciliation.
                    "use_sinks": bool(record.get("has_sinks")),
                    "_provenance": {
                        "source": "rocke_bench",
                        "suite": str(path.parent.name),
                        "trace": path.stem,
                        # The runtime key binds all request semantics, independently
                        # of trace labels and capture position.
                        "graph": "",
                        "model": str(record.get("model") or ""),
                        "variant": str(record.get("variant") or ""),
                        "has_sinks": bool(record.get("has_sinks")),
                    },
                }
            )
            shapes[-1]["_provenance"]["graph"] = _bench_graph_name(shapes[-1])
    if skipped_unknown_mask:
        print(
            f"  NOTE: {skipped_unknown_mask} rocKE trace record(s) skipped -- no "
            f"recorded causality to derive a mask from. Not defaulted: a prefill "
            f"trace read as non-causal sizes a set that cannot serve it."
        )
    return shapes


def _shape_key(shape: dict) -> str:
    """All request semantics participate; provenance never does."""
    fields = {"sliding_window": 0, "use_sinks": False}
    fields.update(
        {key: value for key, value in shape.items() if not key.startswith("_")}
    )
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False)


def deduplicate(shapes: list[dict]) -> tuple[list[dict], int]:
    """One entry per distinct shape, keeping the first provenance and counting
    the rest: two suites asking for the same shape is one variant to compile
    but two votes for it mattering."""
    seen: dict = {}
    duplicates = 0
    for shape in shapes:
        key = _shape_key(shape)
        if key in seen:
            duplicates += 1
            seen[key]["_provenance_occurrences"].extend(
                shape.get("_provenance_occurrences", [shape.get("_provenance", {})])
            )
            continue
        seen[key] = {
            **shape,
            "_provenance_occurrences": list(
                shape.get("_provenance_occurrences", [shape.get("_provenance", {})])
            ),
        }
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
