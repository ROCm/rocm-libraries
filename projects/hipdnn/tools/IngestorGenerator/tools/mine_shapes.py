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
    mine_shapes.py --catalog ../hipdnn_torch/MODEL_CATALOG.md \
        --shape-dir ~/model-shapes --out-query-csv model-shapes.csv

Emits the request-field mappings `dispatch_parity.py --shapes` consumes, and -- with
`--out-query-csv` -- the `q.<parameter>` columns `hipdnn_corpus_gen --model-shapes`
reads as its model pool. Neither output filters by what a kernel can serve: that is
the dispatcher's job at stage 4a and the corpus tool's oracle at admission, both of
which report declines with reasons. Filtering here would hide the gap this corpus
exists to measure. The CSV does narrow, but only to what one operation declaration can
EXPRESS, and it reports every row it lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

#: CSV mask spellings -> the request's mask_type. `swin` is a sliding window, which
#: is a different mask kind rather than a causal variant; folding it onto causal
#: collapsed seven distinct shape keys in an earlier join. It is carried through with
#: its own value so the dispatcher declines it explicitly instead of it silently
#: becoming a causal duplicate.
#: Public for the same reason BACKWARD_GRADIENT_TENSOR_NAMES is: `write_query_csv`
#: below, and any consumer reading shape files through this module, has to agree with
#: it about what a mask value means, and a second literal is one that can silently
#: drift from this one.
#: `no_mask` and `top_left` are aiter/CK's spellings of the two it already had (see
#: `composablekernel/tile_engine/ops/fmha`, whose published model shapes use them).
#: `bottom_right` is deliberately ABSENT: causal aligned to the bottom right is a
#: different mask from causal aligned to the top left whenever seqlen_q != seqlen_k,
#: so it is refused by name rather than mapped onto `causal`.
MASK_TYPE = {"full": 0, "none": 0, "no_mask": 0, "causal": 1, "top_left": 1, "swin": 2}

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
            mask_type = MASK_TYPE.get(mask)
            if mask_type is None:
                raise SystemExit(
                    f"FAIL: unknown mask spelling {mask!r} in {path}. Add it to "
                    f"MASK_TYPE rather than defaulting -- guessing a mask is how a "
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
                    "dtype": normalise_dtype(row.get("dtype"), path, "bf16"),
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
            return MASK_TYPE["causal"]
        left = attrs.get("left_bound")
        right = attrs.get("right_bound")
        if left is None and right is None:
            return MASK_TYPE["full"]
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
            return MASK_TYPE["swin"]
        return MASK_TYPE["causal"]
    # No mask attributes at all: the graph does not describe one. Say so by falling
    # back to the path, and only then -- a directory name is a hint, not a contract.
    return MASK_TYPE["causal"] if "causal" in str(path).lower() else MASK_TYPE["full"]


#: Every spelling a source uses for a dtype -> the spelling the rocKE spec takes.
#: Three vocabularies meet here and none of them agree: hipDNN graphs say
#: `bfloat16`, torch traces say `torch.bfloat16`, the spec says `bf16`. A source
#: dtype that reaches the dispatcher unmapped is REJECTED at spec construction
#: ("dtype must be one of ['bf16', 'fp16']"), which reads like the kernel declining
#: a shape when it is really the miner mis-spelling one -- and the whole graph
#: corpus disappears from the servable count that way.
#: Public, with `normalise_dtype`, because every source read here normalises the same
#: three vocabularies; a second table elsewhere would be a second opinion about what
#: `half` means.
DTYPE_SPELLINGS = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
}


def normalise_dtype(raw, path: Path, fallback: str) -> str:
    """One spelling for a dtype, or a refusal naming the source.

    Refuses rather than defaults, for the same reason the mask derivation does: a
    guessed dtype builds a different binary and still validates, so the failure is
    silent and numeric. An ABSENT dtype falls back (the source simply did not say);
    an UNRECOGNISED one is a mapping this table owes, not a value to paper over.
    """
    if raw is None or str(raw).strip() == "":
        return fallback
    spelling = str(raw).strip().lower()
    resolved = DTYPE_SPELLINGS.get(spelling)
    if resolved is None:
        raise SystemExit(
            f"FAIL: unknown dtype spelling {raw!r} in {path}. Add it to "
            f"DTYPE_SPELLINGS rather than defaulting -- a guessed dtype builds the "
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
        # A shape directory holds more than graphs -- a published `model_shapes.json`
        # is a LIST of records, and `.get` on it raises rather than mining nothing.
        # A tree that mixes the two is the ordinary case for `~/model-shapes`, so a
        # document that is not a graph object is skipped exactly like an unparseable
        # one: this reader mines graphs, and says nothing about anything else.
        if not isinstance(graph, dict):
            continue
        tensors = {
            str(t.get("name", "")).lower(): t
            for t in graph.get("tensors", []) or []
            if isinstance(t, dict)
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
                "dtype": normalise_dtype(query.get("data_type"), path, "bf16"),
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
                mask_type = MASK_TYPE["causal"]
            elif int(left) >= 0:
                mask_type = MASK_TYPE["swin"]
                # `[W, 0]` is a banded causal window of left-context W. The spec
                # counts the window in TOKENS including the current one, matching
                # the kernel's `q-W+1 <= k <= q` band, so a recorded left bound of
                # 127 is a 128-token window.
                sliding_window = int(left) + 1
            else:
                mask_type = MASK_TYPE["causal"]
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
            dtype = normalise_dtype(record.get("q_dtype"), path, dtype_default)
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


#: Column spellings a published shape file uses -> this tool's field. Three publishers
#: disagree (the kernel team's results CSV says `heads_q`/`seq_q`, aiter's model_shapes
#: say `nhead_q`/`seqlen_q`, a hand-written file says `h`/`hq`), and a shape file that
#: silently mines as zero rows is indistinguishable from one nobody pointed at. Adding a
#: spelling here is the whole maintenance cost of accepting a new publisher.
SHAPE_COLUMNS = {
    "batch": "batch", "batch_size": "batch", "b": "batch", "num_seqs": "batch",
    "heads_q": "heads_q", "nhead_q": "heads_q", "num_query_heads": "heads_q",
    "hq": "heads_q", "h": "heads_q", "heads": "heads_q",
    "heads_kv": "heads_kv", "nhead_k": "heads_kv", "nhead_kv": "heads_kv",
    "num_kv_heads": "heads_kv", "hkv": "heads_kv", "h_kv": "heads_kv",
    "seqlen_q": "seqlen_q", "seq_q": "seqlen_q", "sq": "seqlen_q", "s_q": "seqlen_q",
    "seqlen_k": "seqlen_kv", "seqlen_kv": "seqlen_kv", "seq_kv": "seqlen_kv",
    "seq_k": "seqlen_kv", "skv": "seqlen_kv", "s_kv": "seqlen_kv",
    "head_dim": "head_dim", "hdim_q": "head_dim", "hdim": "head_dim",
    "head_size": "head_dim", "d": "head_dim",
    "dtype": "dtype", "data_type": "dtype", "q_dtype": "dtype",
    "mask": "mask", "mask_type": "mask", "causal": "mask", "is_causal": "mask",
    "model": "model", "name": "model",
    "arch": "arch",
}


def _shape_rows_from_json(path: Path) -> list[dict]:
    """Records from a JSON shape file, which is a list of them or a map of them."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(content, dict):
        if "tensors" in content and "nodes" in content:
            return []  # a graph; `from_graph_corpus` reads those
        content = [
            dict(record, model=record.get("model", name))
            for name, record in content.items()
            if isinstance(record, dict)
        ]
    if not isinstance(content, list):
        return []
    return [record for record in content if isinstance(record, dict)]


def _shape_rows_from_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _shape_rows_from_text(path: Path) -> list[dict]:
    """`key=value key=value` per line, one shape per line.

    The file convention is `dnn-convert-shapes`': one invocation per line, blank lines
    and `#` comments skipped, so a hand-maintained shape list reads the same whichever
    converter is pointed at it. None of that tool's code is reused -- it converts MIOpen
    driver lines for convolution and batchnorm, and MIOpenDriver has no attention
    operation to spell a line for.
    """
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pairs = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if pairs:
            rows.append(pairs)
    return rows


def _shape_row(row: dict, path: Path) -> dict | None:
    """A published row as a record, or None if it is not a shape row at all.

    Every value that decides a kernel goes through this module's own vocabulary rather
    than through a guess: an unrecognised dtype or mask spelling is refused loudly,
    because a guessed one mines a corpus entry that measures a different problem than
    the one the row named.
    """
    fields: dict = {}
    for key, value in row.items():
        if key is None:
            continue
        field = SHAPE_COLUMNS.get(str(key).strip().lower())
        if field is not None and value not in (None, ""):
            fields.setdefault(field, value)
    required = ("batch", "heads_q", "seqlen_q", "seqlen_kv", "head_dim")
    if any(field not in fields for field in required):
        return None

    # A boolean `is_causal` column, or a named mask `MASK_TYPE` already knows. Every
    # other spelling is refused rather than mapped here: `bottom_right`, for one, is a
    # genuinely different mask from top-left causal, and accepting it would put a shape
    # in the corpus that is not the one the row named.
    raw_mask = str(fields.get("mask", "")).strip().lower()
    if raw_mask in ("true", "1", "yes"):
        mask_type = MASK_TYPE["causal"]
    elif raw_mask in ("", "false", "0", "no"):
        mask_type = MASK_TYPE["full"]
    elif raw_mask in MASK_TYPE:
        mask_type = MASK_TYPE[raw_mask]
    else:
        raise SystemExit(
            f"FAIL: unknown mask spelling {raw_mask!r} in {path}. Add it to MASK_TYPE "
            f"rather than defaulting -- guessing a mask puts a differently-masked "
            f"problem in the corpus under the row's name."
        )

    heads_q = int(fields["heads_q"])
    return {
        "batch": int(fields["batch"]),
        "nhead_q": heads_q,
        "nhead_k": int(fields.get("heads_kv", heads_q)),
        "seqlen_q": int(fields["seqlen_q"]),
        "seqlen_k": int(fields["seqlen_kv"]),
        "hdim_q": int(fields["head_dim"]),
        "hdim_v": int(fields["head_dim"]),
        "dtype": normalise_dtype(fields.get("dtype"), path, "bf16"),
        "mask_type": mask_type,
        "_provenance": {
            "source": "shape_dir",
            "file": path.name,
            "model": str(fields.get("model", path.stem)),
            "arch": fields.get("arch"),
        },
    }


def from_shape_dir(root: Path, arch: str | None = None) -> list[dict]:
    """Shapes from a published directory -- the cluster's `~/model-shapes`.

    Both forms are read: hipDNN graph JSON (through `from_graph_corpus`, the same reader
    a `dnn-benchmarking` tree gets) and tabular files -- `.json` records, `.csv`, and
    `key=value` lines. Whichever form the kernel team publishes, the directory is the
    pointer and nothing here has to be told which; a tree that mixes both is the
    ordinary case, which is why the graph reader already skips non-graph documents.

    `arch`, when given, drops rows that name a DIFFERENT arch. A row that names none is
    kept: most publishers record a shape, not a target, and refusing those would mine
    nothing from the common case.
    """
    shapes = list(from_graph_corpus(root))
    wrong_arch = 0
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = _shape_rows_from_json(path)
        elif suffix == ".csv":
            rows = _shape_rows_from_csv(path)
        elif suffix in (".txt", ".shapes"):
            rows = _shape_rows_from_text(path)
        else:
            continue
        for row in rows:
            record = _shape_row(row, path)
            if record is None:
                continue
            if arch is not None and record["_provenance"]["arch"] not in (None, arch):
                wrong_arch += 1
                continue
            shapes.append(record)
    if wrong_arch:
        print(f"  NOTE: {wrong_arch} published row(s) skipped -- named another arch.")
    return shapes


#: Batches to expand each catalog geometry over. The catalog's A/B runs are all
#: single-batch (`output (1,256,1536)`), which is a measurement convention rather than
#: a serving one: `corpus_gen/operations/sdpa_fwd.opmeta.json`'s own archetypes put the
#: same models at batch 1..256. Emitting only batch 1 would mine a corpus that has never
#: seen a served batch of the shapes it exists to get right.
CATALOG_BATCHES = (1, 8, 32)


def _catalog_sections(text: str) -> dict:
    """The catalog's per-model entries, keyed by harness file stem."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^###\s+`([A-Za-z0-9_]+)\.py`", line)
        if heading:
            current = heading.group(1)
            sections[current] = []
            continue
        if line.startswith("### ") or line.startswith("## "):
            current = None
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


def _catalog_table(text: str) -> list[dict]:
    """The `model | D | q/kv heads | causal` drill-down rows.

    The catalog says the same things in prose several times over; this table is the
    one place it says them as columns, so it is the row source and the prose is only
    consulted for the sequence length it does not carry.
    """
    rows, header_seen = [], False
    for line in text.splitlines():
        if not line.startswith("|"):
            header_seen = False
            continue
        cells = [cell.strip().replace("**", "").replace("`", "")
                 for cell in line.strip("|").split("|")]
        if cells[:4] == ["model", "D", "q/kv heads", "causal"]:
            header_seen = True
            continue
        if not header_seen or set(cells[0]) <= set("-: "):
            continue
        rows.append({"model": cells[0], "head_dims": cells[1],
                     "heads": cells[2], "causal": cells[3]})
    return rows


def _catalog_numbers(field: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", field)]


def _catalog_lengths(body: str) -> list[int]:
    """Sequence lengths the entry records, from the two spellings it uses.

    `Sq=Skv=512` in the validated SDPA report and `seq512` in the run's geometry
    line. Nothing is inferred from a model's name: an entry that records no sequence
    length yields no shape and is reported as skipped, because a guessed prompt
    length is a shape no one measured.
    """
    found = set(int(value) for value in re.findall(r"Sq=Skv=(\d+)", body))
    found |= set(int(value) for value in re.findall(r"\bseq(\d+)\b", body))
    return sorted(found)


def from_model_catalog(path: Path, batches=CATALOG_BATCHES) -> list[dict]:
    """Every attention geometry `MODEL_CATALOG.md` records, expanded over batch.

    The fourth source, and the only in-tree one: this is the single place in the
    repository where `bert = 12 heads, D=64, non-causal, Sq=Skv=512` is written down as
    an OBSERVATION rather than as a plausible number. The other three sources all come
    from outside the tree and are absent on a machine nobody staged them to.

    A causal entry additionally yields its decode shape -- one query token against the
    context it just filled. Decode is not an extra flavour of the prefill row: a
    `seqlen_q` of 1 against a long cache is memory bound where the square prefill is
    compute bound, and a corpus that omits it says nothing about the half of serving
    that is decode. Encoder entries (bert, whisper, flux) get no decode shape: they
    have no autoregressive phase to decode in.
    """
    text = path.read_text(encoding="utf-8")
    sections = _catalog_sections(text)
    shapes: list[dict] = []
    skipped: list[str] = []

    for row in _catalog_table(text):
        model = row["model"]
        head_dims = _catalog_numbers(row["head_dims"])
        heads = _catalog_numbers(row["heads"])
        body = next((section for name, section in sorted(sections.items())
                     if name.startswith(model)), "")
        lengths = _catalog_lengths(body)
        # The dtypes the entry was actually validated in, not every dtype the model
        # could run in: an entry validated only in bf16 says nothing about fp16, and a
        # declaration sweep is where unrecorded combinations belong. `\bf16\b` cannot
        # match inside `bf16` -- `b` and `f` are both word characters -- so the two
        # spellings do not collide.
        dtypes = set()
        if re.search(r"\bbf16\b", body):
            dtypes.add("bf16")
        if re.search(r"\bfp16\b|\bf16\b", body):
            dtypes.add("fp16")
        dtypes = sorted(dtypes) or ["bf16"]
        causal = row["causal"].lower().startswith("y")

        if not heads or not head_dims or not lengths:
            skipped.append(model + " (catalog records no " + ", ".join(
                label for label, present in (("head count", heads),
                                             ("head dim", head_dims),
                                             ("sequence length", lengths))
                if not present) + ")")
            continue

        # `12/12` is query/KV; `5/10/20` is three MHA stages of one UNet, not a
        # grouping -- a single value repeats as its own KV count.
        pairs = ([(heads[0], heads[1])] if len(heads) == 2
                 else [(head, head) for head in heads])
        for head_dim in head_dims:
            for heads_q, heads_kv in pairs:
                for dtype in dtypes:
                    for length in lengths:
                        for batch in batches:
                            common = {
                                "batch": batch, "nhead_q": heads_q,
                                "nhead_k": heads_kv, "hdim_q": head_dim,
                                "hdim_v": head_dim, "dtype": dtype,
                                "mask_type": MASK_TYPE["causal" if causal else "full"],
                            }
                            shapes.append({
                                **common, "seqlen_q": length, "seqlen_k": length,
                                "_provenance": {"source": "catalog", "model": model,
                                                "phase": "prefill",
                                                "catalog": path.name},
                            })
                            if causal:
                                shapes.append({
                                    **common, "seqlen_q": 1, "seqlen_k": length,
                                    "_provenance": {"source": "catalog",
                                                    "model": model, "phase": "decode",
                                                    "catalog": path.name},
                                })
    if skipped:
        print(f"  NOTE: {len(skipped)} catalog entr(ies) yielded no shape: "
              + "; ".join(skipped))
    return shapes


def _shape_name(shape: dict, index: int) -> str:
    """A human-readable name for one shape, or a positional one if it has no name.

    The model pool's whole value over a sweep is that its points are NAMED: a regime a
    later audit can argue about is `llama3-70b decode`, never a row of numbers. The
    positional fallback exists so a nameless source still joins, not so it is the norm.
    """
    origin = shape.get("_provenance") or {}
    model = str(origin.get("model") or "")
    phase = str(origin.get("phase") or "")
    if model and phase:
        return f"{model} {phase}"
    for key in ("graph", "trace", "model", "suite"):
        named = str(origin.get(key) or "")
        if named:
            return named
    return f"{origin.get('source') or 'shape'}-{index}"


def write_query_csv(shapes: list[dict], path: Path) -> dict:
    """The mined corpus as the `q.<parameter>` columns `corpus_gen --model-shapes` reads.

    This is a NARROWING, and it reports what it narrowed. The mined record is the union
    of what four sources record; the CSV is what one operation declaration can express,
    and the difference is not empty -- a sliding window, an asymmetric head dim and a
    sink trace are all real recorded shapes that `sdpa_fwd` has no parameter for. Each
    is dropped by name and counted, because a model pool that silently shrinks is
    indistinguishable from a miner nobody pointed at anything.

    `q.alignment` is always `top_left`: `MASK_TYPE` deliberately carries no
    `bottom_right` spelling, so no source here can say bottom-right, and writing the
    column at all keeps the row buildable (the declaration's argument resolution is
    strict about a parameter it reads being present). `q.generate_stats` is always
    `false` for the same reason: every source here records inference forwards, and
    none says whether a shape also ran as a training forward.
    """
    columns = ["name", "op", "q.batch", "q.heads", "q.heads_kv", "q.seqlen_q",
               "q.seqlen_k", "q.head_dim", "q.is_causal", "q.alignment",
               "q.generate_stats", "q.dtype"]
    dropped: dict[str, int] = {}
    written = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, shape in enumerate(shapes):
            reason = None
            if shape["mask_type"] not in (MASK_TYPE["full"], MASK_TYPE["causal"]):
                reason = "windowed mask, which sdpa_fwd declares no parameter for"
            elif shape["hdim_q"] != shape["hdim_v"]:
                reason = "asymmetric head dims (MLA), one head_dim declared"
            elif shape.get("use_sinks"):
                reason = "attention sinks, which sdpa_fwd declares no parameter for"
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            writer.writerow([
                _shape_name(shape, index), "sdpa_fwd",
                shape["batch"], shape["nhead_q"], shape["nhead_k"],
                shape["seqlen_q"], shape["seqlen_k"], shape["hdim_q"],
                "true" if shape["mask_type"] == MASK_TYPE["causal"] else "false",
                "top_left", "false", shape["dtype"],
            ])
            written += 1
    return {"written": written, "dropped": dropped}


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
    parser.add_argument(
        "--catalog",
        help="hipdnn_torch/MODEL_CATALOG.md. The fourth source, and the only in-tree "
        "one: the geometries models were observed running at, rather than plausible "
        "numbers. Expanded over batch and, for causal entries, over decode.",
    )
    parser.add_argument(
        "--shape-dir",
        action="append",
        default=[],
        dest="shape_dirs",
        metavar="DIR",
        help="A published shape directory -- the cluster's `~/model-shapes`. Reads "
        "graph JSON and tabular files alike (.json records, .csv, `key=value` lines) "
        "through SHAPE_COLUMNS, so a new publisher's spelling costs one table entry "
        "rather than a reader. Repeatable.",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="Filter the published CSV to one arch, and drop shape-directory rows "
        "that name a different one.",
    )
    parser.add_argument(
        "--include-windowed",
        action="store_true",
        help="Keep sliding-window rows. Off by default: they are a different mask "
        "kind, and a kernel that clamps top-left only will decline them anyway -- "
        "but they are excluded LOUDLY here rather than folded onto causal.",
    )
    parser.add_argument(
        "--out",
        help="Write the shape corpus here, as the request-field JSON "
        "`dispatch_parity.py --shapes` consumes.",
    )
    parser.add_argument(
        "--out-query-csv",
        help="Also write the corpus as `q.<parameter>` columns, which "
        "`hipdnn_corpus_gen --model-shapes` reads as its model pool. A narrowing to "
        "what one operation declaration can express; what it drops is reported.",
    )
    args = parser.parse_args(argv)

    if not args.out and not args.out_query_csv:
        parser.error("give --out, --out-query-csv, or both; otherwise nothing is kept.")

    if not args.published and not args.graphs and not args.rocke_bench \
            and not args.catalog and not args.shape_dirs:
        parser.error(
            "give at least one source. No corpus alone is sufficient: the CSV is "
            "what the kernel team measures, the graph tree is what callers send, "
            "rocKE's bench tree is what the kernel's own authors sweep, the catalog "
            "is what models were observed running, a shape directory is whatever a "
            "publisher handed over, and an integration sized from only one of them "
            "has missed real shapes twice."
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
    if args.catalog:
        found = from_model_catalog(Path(args.catalog))
        print(f"  model catalog : {len(found):5d} recorded geometries")
        shapes += found
    for directory in args.shape_dirs:
        found = from_shape_dir(Path(directory), args.arch)
        print(f"  shape dir     : {len(found):5d} published shapes in {directory}")
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

    if args.out:
        Path(args.out).write_text(json.dumps(unique, indent=2))
        print(f"\n  wrote {args.out}")
    if args.out_query_csv:
        stats = write_query_csv(unique, Path(args.out_query_csv))
        print(f"\n  wrote {args.out_query_csv}: {stats['written']} row(s)")
        for reason, count in sorted(stats["dropped"].items()):
            print(f"    dropped {count:5d}: {reason}")
        if stats["written"] == 0:
            print(
                "\nFAIL: every mined shape was dropped on the way to the query CSV; "
                "the model pool would be empty and the corpus would report itself as "
                "having one.",
                file=sys.stderr,
            )
            return 1
    print(
        "  Provenance is carried on every shape. Split every reported result by it: a "
        "geomean over a mixed corpus reports the synthetic population's win as if it "
        "were everyone's."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
