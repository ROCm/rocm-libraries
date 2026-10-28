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
    mine_shapes.py --catalog ../hipdnn_torch/MODEL_CATALOG.md \
        --shape-dir <shape-dir> --out-query-csv model-shapes.csv

Emits the request-field mappings `registry_enumerator.py --shapes` reads and, with
`--out-query-csv`, the `q.<parameter>` columns `hipdnn_corpus_gen --model-shapes`
reads. Neither output filters by kernel support -- that would hide the gap this
corpus measures. The CSV narrows only to what one operation declaration can express,
and reports every row it drops.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

#: rocKE's `AttentionMaskType` ordinals, which a mined shape's `mask_type` carries
#: verbatim. The request states the causal anchor through the mask kind and has no
#: other field for it, so a shape never carries a separate alignment key.
NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL, SLIDING_WINDOW = 0, 1, 2, 3

#: CSV mask spellings -> the request's mask_type. `swin` (sliding window) keeps its own
#: value so the dispatcher declines it instead of serving it as causal. `bottom_right`
#: is deliberately absent: it differs from top-left causal when seqlen_q != seqlen_k.
MASK_TYPE = {
    "full": NO_MASK,
    "none": NO_MASK,
    "no_mask": NO_MASK,
    "causal": TOP_LEFT_CAUSAL,
    "top_left": TOP_LEFT_CAUSAL,
    "swin": SLIDING_WINDOW,
}

#: The right bound a mined window carries. Every window source here records a causal
#: band (right bound 0). A request's `right_bound` defaults to -1 (unbounded), which
#: turns the same `sliding_window` width into a non-causal left-only band that the
#: kernels classify as a different feature.
WINDOW_RIGHT_BOUND = 0


def _mask_fields(mask_type: int, window: int) -> dict:
    """The request's mask fields. A sliding window also states its right bound, which
    the request otherwise reads as unbounded."""
    fields = {"mask_type": mask_type, "sliding_window": window}
    if mask_type == SLIDING_WINDOW:
        fields["right_bound"] = WINDOW_RIGHT_BOUND
    return fields


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


def from_published_csv(
    path: Path, arch: str, include_windowed: bool, excluded: list | None = None
) -> list[dict]:
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
            mask_type = MASK_TYPE.get(mask)
            if mask_type is None:
                raise SystemExit(
                    f"FAIL: unknown mask spelling {mask!r} in {path}. Add it to "
                    f"MASK_TYPE rather than defaulting -- guessing a mask is how a "
                    f"windowed graph gets served as plain causal."
                )
            head_dim = int(row["head_dim"])
            window = 0
            if mask_type == MASK_TYPE["swin"]:
                raw_window = (row.get("window_size") or "").strip()
                if not raw_window.isdigit() or int(raw_window) <= 0:
                    raise SystemExit(
                        f"FAIL: {path}: windowed CSV row requires a positive window_size width"
                    )
                window = int(raw_window)
            try:
                dtype = normalise_dtype(row.get("dtype"), path, "bf16")
            except OutOfFamilyDtype as exc:
                _exclude(excluded, path, str(exc), shape_idx=row.get("shape_idx") or "")
                continue
            shapes.append(
                {
                    "batch": int(row["batch"]),
                    "nhead_q": int(row["heads_q"]),
                    "nhead_k": int(row["heads_kv"]),
                    "seqlen_q": int(row["seq_q"]),
                    "seqlen_k": int(row["seq_kv"]),
                    "hdim_q": head_dim,
                    "hdim_v": head_dim,
                    "dtype": dtype,
                    **_mask_fields(mask_type, window),
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


def _causal_mask_type(bottom_right: bool, seqlen_q: int, seqlen_k: int) -> int:
    """The causal mask kind for an anchor. Equal lengths put both anchors on one
    diagonal, so only a cross-length bottom-right request is BOTTOM_RIGHT_CAUSAL."""
    if bottom_right and seqlen_q != seqlen_k:
        return BOTTOM_RIGHT_CAUSAL
    return TOP_LEFT_CAUSAL


def _mask_from_attributes(
    attrs: dict, path: Path, seqlen_q: int, seqlen_k: int
) -> dict:
    """Normalize the graph dialect to the request's mask kind and window.

    The anchor is carried by the mask kind, not folded onto top-left: the two differ
    when Sq != Sk, and the UHD corpus deliberately includes that case. A window kind
    has no anchor, so a bottom-right window across unequal lengths is refused. The
    deprecated booleans merge into the stored bounds and alignment the way the ASM
    SDPA engine's `plan_utils::resolveDiagonalBand` does.
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
    causal = attrs.get("causal_mask")
    causal_bottom_right = attrs.get("causal_mask_bottom_right")
    if causal and causal_bottom_right:
        raise SystemExit(
            f"FAIL: {path}: causal_mask and causal_mask_bottom_right are mutually exclusive"
        )
    left = -1 if left is None else left
    right = -1 if right is None else right
    if causal or causal_bottom_right:
        right = max(right, 0)
    if causal_bottom_right:
        alignment = "BOTTOM_RIGHT"
    if left == -1 and right == -1:
        return {"mask_type": NO_MASK, "sliding_window": 0}
    bottom_right = alignment == "BOTTOM_RIGHT"
    # A right bound other than 0 is not causal under any anchor, and a shifted
    # window has no request spelling.
    if right != 0 or (left != -1 and bottom_right and seqlen_q != seqlen_k):
        raise SystemExit(
            f"FAIL: {path}: unsupported translation of bounds ({left}, {right}), "
            f"alignment {alignment}, Sq={seqlen_q}, Sk={seqlen_k} to AttentionRequest"
        )
    if left == -1:
        return {
            "mask_type": _causal_mask_type(bottom_right, seqlen_q, seqlen_k),
            "sliding_window": 0,
        }
    return _mask_fields(SLIDING_WINDOW, left + 1)


#: Every source spelling of a dtype -> the rocKE spec's spelling. An unmapped dtype
#: is rejected at spec construction, which looks like a kernel declining the shape.
DTYPE_SPELLINGS = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
}

#: Recognised spellings of dtypes outside the bf16/fp16 family this miner
#: emits requests for. A graph in one of them is a real request nobody here can
#: serve, so it is recorded as excluded with its reason and counted rather than
#: aborting the source; an unrecognised spelling is still refused.
_OUT_OF_FAMILY_DTYPES = {
    "float": "fp32",
    "float32": "fp32",
    "fp32": "fp32",
    "torch.float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "fp64": "fp64",
    "torch.float64": "fp64",
}


class OutOfFamilyDtype(Exception):
    """A recognised dtype this miner's request family does not cover."""

    def __init__(self, dtype: str):
        super().__init__(f"dtype {dtype} is outside the bf16/fp16 request family")
        self.dtype = dtype


def normalise_dtype(raw, path: Path, fallback: str) -> str:
    """One spelling for a dtype, or a refusal naming the source.

    An absent dtype falls back; a recognised out-of-family one raises
    `OutOfFamilyDtype` for the reader to record as an exclusion; an
    unrecognised one is a mapping this table owes, since a guessed dtype builds
    a different binary and still validates.
    """
    if raw is None or str(raw).strip() == "":
        return fallback
    spelling = str(raw).strip().lower()
    resolved = DTYPE_SPELLINGS.get(spelling)
    if resolved is None:
        if spelling in _OUT_OF_FAMILY_DTYPES:
            raise OutOfFamilyDtype(_OUT_OF_FAMILY_DTYPES[spelling])
        raise SystemExit(
            f"FAIL: unknown dtype spelling {raw!r} in {path}. Add it to "
            f"DTYPE_SPELLINGS (or _OUT_OF_FAMILY_DTYPES) rather than defaulting "
            f"-- a guessed dtype builds the wrong binary and still validates."
        )
    return resolved


def _exclude(excluded: list | None, path: Path, reason: str, **extra) -> None:
    """Record one excluded input with its reason; the caller skips it."""
    if excluded is not None:
        excluded.append({"path": str(path), "reason": reason, **extra})


def from_graph_corpus(root: Path, excluded: list | None = None) -> list[dict]:
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
        # Non-graph JSON (e.g. records beside graphs in a shape dir) is skipped.
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
            or "q_tensor_uid" in (n.get("inputs") or {})
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
        # hipDNN graphs bind operands under the node's `inputs`; an older
        # spelling put the UIDs among the attributes. Read both, so a sink bound
        # in `inputs` is not mined as use_sinks=False. A null input slot means
        # unbound there, so it never hides an attribute binding.
        node = sdpa[0] if sdpa else {}
        bindings = {
            **attrs,
            **{k: v for k, v in (node.get("inputs") or {}).items() if v is not None},
        }
        by_uid = {t["uid"]: t for t in graph.get("tensors", []) if "uid" in t}
        selected = []
        for short, long in (("q", "query"), ("k", "key"), ("v", "value")):
            uid_key = f"{short}_tensor_uid"
            tensor = (
                by_uid.get(bindings[uid_key])
                if bindings.get(uid_key) is not None
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
        try:
            dtypes = [
                normalise_dtype(t.get("data_type"), path, "bf16") for t in selected
            ]
        except OutOfFamilyDtype as exc:
            _exclude(excluded, path, str(exc), graph=graph.get("name", path.stem))
            continue
        if len(set(dtypes)) != 1:
            raise SystemExit(
                f"FAIL: {path}: mixed Q/K/V dtypes cannot form one request"
            )
        mask = _mask_from_attributes(attrs, path, qdims[2], kdims[2])
        sink_uid = bindings.get("sink_token_tensor_uid")
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


def from_rocke_bench(
    root: Path, dtype_default: str, excluded: list | None = None
) -> list[dict]:
    """Shapes from rocKE's own benchmark tree; for an arch with no published
    CSV it is the only source saying what the kernel team measures.

    `*_shapes.json` / `*_bench.json` under `benchmarks/<arch>/attention/` are
    JSONL, one record per line (`json.load` raises "Extra data"): captured
    launch traces carrying `window_size` and `has_sinks`. The paired
    `benchmark_*_live.py` generates shapes instead; the dense prefill ones
    write theirs in this schema with `--emit-shapes`.

    Captured traces do not record causality and the dispatcher does
    `causal = (mask_type != 0)`, so a trace states causality through
    `window_size` or it is skipped and counted. `window_size` is
    `[left, right]`: `[-1, -1]` is unbounded both ways, causal for these
    prefill suites, and `[W, 0]` with W >= 0 is a banded causal window, never
    folded onto plain causal. A record that carries an explicit boolean
    `causal` (the emitted benchmark lists do) is read by it instead: `false`
    is a full, unmasked request, whose window must be `[-1, -1]`.

    A record with `varlen: true` (a packed ragged batch) is skipped and
    counted: the request has no varlen field, and mining it as a dense batch
    of its longest sequence would ask the dispatcher a different question.
    """
    shapes: list[dict] = []
    skipped_unknown_mask = 0
    skipped_varlen = 0
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
            if record.get("varlen"):
                skipped_varlen += 1
                continue
            causal = record.get("causal")
            if causal is not None and type(causal) is not bool:
                raise SystemExit(
                    f"FAIL: {path}: causal must be boolean, got {causal!r}"
                )
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
            if causal is False:
                if [int(left), int(right)] != [-1, -1]:
                    raise SystemExit(
                        f"FAIL: {path}: a non-causal record cannot carry window "
                        f"{window!r}"
                    )
                mask_type = MASK_TYPE["none"]
            elif int(left) < 0 and int(right) < 0:
                mask_type = MASK_TYPE["causal"]
            elif int(left) >= 0 and int(right) == 0:
                mask_type = MASK_TYPE["swin"]
                # The spec counts the window in TOKENS including the current one,
                # matching the kernel's `q-W+1 <= k <= q` band, so a recorded left
                # bound of 127 is a 128-token window.
                sliding_window = int(left) + 1
            elif int(left) == -1 and int(right) == 0:
                mask_type = MASK_TYPE["causal"]
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
            try:
                dtype = normalise_dtype(record.get("q_dtype"), path, dtype_default)
            except OutOfFamilyDtype as exc:
                _exclude(excluded, path, str(exc), trace=path.stem)
                continue
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
                    **_mask_fields(mask_type, sliding_window),
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
    if skipped_varlen:
        print(
            f"  NOTE: {skipped_varlen} rocKE varlen record(s) skipped -- the request "
            f"has no varlen field, so a packed ragged batch is not one of its shapes."
        )
    return shapes


#: Column spellings a published shape file uses -> this tool's field. Publishers
#: disagree (`heads_q`, `nhead_q`, `h`, ...); a new one costs one entry here.
SHAPE_COLUMNS = {
    "batch": "batch",
    "batch_size": "batch",
    "b": "batch",
    "num_seqs": "batch",
    "heads_q": "heads_q",
    "nhead_q": "heads_q",
    "num_query_heads": "heads_q",
    "hq": "heads_q",
    "h": "heads_q",
    "heads": "heads_q",
    "heads_kv": "heads_kv",
    "nhead_k": "heads_kv",
    "nhead_kv": "heads_kv",
    "num_kv_heads": "heads_kv",
    "hkv": "heads_kv",
    "h_kv": "heads_kv",
    "seqlen_q": "seqlen_q",
    "seq_q": "seqlen_q",
    "sq": "seqlen_q",
    "s_q": "seqlen_q",
    "seqlen_k": "seqlen_kv",
    "seqlen_kv": "seqlen_kv",
    "seq_kv": "seqlen_kv",
    "seq_k": "seqlen_kv",
    "skv": "seqlen_kv",
    "s_kv": "seqlen_kv",
    "head_dim": "head_dim",
    "hdim_q": "head_dim",
    "hdim": "head_dim",
    "head_size": "head_dim",
    "d": "head_dim",
    # Multi-head latent attention publishes V narrower than Q/K (192/128).
    "hdim_v": "head_dim_v",
    "head_dim_v": "head_dim_v",
    "v_head_dim": "head_dim_v",
    "d_v": "head_dim_v",
    "window_size": "window",
    "sliding_window": "window",
    "window": "window",
    "use_sinks": "sinks",
    "sinks": "sinks",
    "dtype": "dtype",
    "data_type": "dtype",
    "q_dtype": "dtype",
    "mask": "mask",
    "mask_type": "mask",
    "causal": "mask",
    "is_causal": "mask",
    "model": "model",
    "name": "model",
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

    Follows `dnn-convert-shapes`' convention: blank lines and `#` comments skipped.
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


class WindowedRowExcluded(Exception):
    """A sliding-window row read without `--include-windowed`."""


def _shape_row(row: dict, path: Path, include_windowed: bool) -> dict | None:
    """A published row as a record, or None if it is not a shape row at all.

    An unrecognised dtype or mask spelling is refused, never guessed; a recognised
    dtype outside the request family raises `OutOfFamilyDtype`, and a windowed row
    without `include_windowed` raises `WindowedRowExcluded`, for the caller to record.
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

    # A boolean `is_causal` column or a `MASK_TYPE` name; anything else (e.g.
    # `bottom_right`, a different mask) is refused.
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

    if mask_type == SLIDING_WINDOW and not include_windowed:
        raise WindowedRowExcluded()

    # As from_published_csv: a windowed row names its width, never a guessed one.
    window = 0
    if mask_type == MASK_TYPE["swin"]:
        raw_window = str(fields.get("window", "")).strip()
        if not raw_window.isdigit() or int(raw_window) <= 0:
            raise SystemExit(
                f"FAIL: {path}: windowed row requires a positive window_size width"
            )
        window = int(raw_window)
    raw_sinks = str(fields.get("sinks", "")).strip().lower()
    if raw_sinks not in ("", "true", "1", "yes", "false", "0", "no"):
        raise SystemExit(f"FAIL: {path}: unknown use_sinks value {raw_sinks!r}")

    heads_q = int(fields["heads_q"])
    head_dim = int(fields["head_dim"])
    return {
        "batch": int(fields["batch"]),
        "nhead_q": heads_q,
        "nhead_k": int(fields.get("heads_kv", heads_q)),
        "seqlen_q": int(fields["seqlen_q"]),
        "seqlen_k": int(fields["seqlen_kv"]),
        "hdim_q": head_dim,
        "hdim_v": int(fields.get("head_dim_v", head_dim)),
        "dtype": normalise_dtype(fields.get("dtype"), path, "bf16"),
        **_mask_fields(mask_type, window),
        "use_sinks": raw_sinks in ("true", "1", "yes"),
        "_provenance": {
            "source": "shape_dir",
            "file": path.name,
            "model": str(fields.get("model", path.stem)),
            "arch": fields.get("arch"),
        },
    }


def from_shape_dir(
    root: Path,
    arch: str | None = None,
    excluded: list | None = None,
    include_windowed: bool = False,
) -> list[dict]:
    """Shapes from a published shape directory.

    Reads hipDNN graph JSON (via `from_graph_corpus`) and tabular files (`.json`
    records, `.csv`, `key=value` lines); a directory may mix both.

    `arch`, when given, drops rows naming a different arch; rows naming none are kept.
    A row or graph in a recognised dtype outside the request family is recorded in
    `excluded` and skipped, as is a tabular sliding-window row unless
    `include_windowed` is set.
    """
    shapes = list(from_graph_corpus(root, excluded))
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
        for index, row in enumerate(rows):
            try:
                record = _shape_row(row, path, include_windowed)
            except OutOfFamilyDtype as exc:
                _exclude(excluded, path, str(exc), row=index)
                continue
            except WindowedRowExcluded:
                _exclude(
                    excluded,
                    path,
                    "sliding-window row (pass --include-windowed to keep it)",
                    row=index,
                )
                continue
            if record is None:
                continue
            if arch is not None and record["_provenance"]["arch"] not in (None, arch):
                wrong_arch += 1
                continue
            shapes.append(record)
    if wrong_arch:
        print(f"  NOTE: {wrong_arch} published row(s) skipped -- named another arch.")
    return shapes


#: Batches to expand each catalog geometry over. The catalog's runs are all batch 1,
#: a measurement convention; served batches (see `sdpa_fwd.opmeta.json`) are larger.
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

    The table is the row source; prose is consulted only for sequence length.
    """
    rows, header_seen = [], False
    for line in text.splitlines():
        if not line.startswith("|"):
            header_seen = False
            continue
        cells = [
            cell.strip().replace("**", "").replace("`", "")
            for cell in line.strip("|").split("|")
        ]
        if cells[:4] == ["model", "D", "q/kv heads", "causal"]:
            header_seen = True
            continue
        if not header_seen or set(cells[0]) <= set("-: "):
            continue
        rows.append(
            {
                "model": cells[0],
                "head_dims": cells[1],
                "heads": cells[2],
                "causal": cells[3],
            }
        )
    return rows


def _catalog_numbers(field: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", field)]


def _catalog_lengths(body: str) -> list[int]:
    """Sequence lengths the entry records (`Sq=Skv=512` or `seq512`).

    Nothing is inferred from a model's name; an entry with no length yields no shape.
    """
    found = set(int(value) for value in re.findall(r"Sq=Skv=(\d+)", body))
    found |= set(int(value) for value in re.findall(r"\bseq(\d+)\b", body))
    return sorted(found)


def from_model_catalog(path: Path, batches=CATALOG_BATCHES) -> list[dict]:
    """Every attention geometry `MODEL_CATALOG.md` records, expanded over batch.

    A causal entry also yields its decode shape: one query token against the full
    context, which is memory bound where prefill is compute bound. The token is the
    newest position, so its diagonal is anchored bottom-right; anchored top-left it
    would attend to key 0 alone.
    """
    text = path.read_text(encoding="utf-8")
    sections = _catalog_sections(text)
    shapes: list[dict] = []
    skipped: list[str] = []

    for row in _catalog_table(text):
        model = row["model"]
        head_dims = _catalog_numbers(row["head_dims"])
        heads = _catalog_numbers(row["heads"])
        body = next(
            (
                section
                for name, section in sorted(sections.items())
                if name.startswith(model)
            ),
            "",
        )
        lengths = _catalog_lengths(body)
        # Only the dtypes the entry was validated in. `\bf16\b` cannot match inside
        # `bf16` (`b` and `f` are both word characters).
        dtypes = set()
        if re.search(r"\bbf16\b", body):
            dtypes.add("bf16")
        if re.search(r"\bfp16\b|\bf16\b", body):
            dtypes.add("fp16")
        dtypes = sorted(dtypes) or ["bf16"]
        causal = row["causal"].lower().startswith("y")

        if not heads or not head_dims or not lengths:
            skipped.append(
                model
                + " (catalog records no "
                + ", ".join(
                    label
                    for label, present in (
                        ("head count", heads),
                        ("head dim", head_dims),
                        ("sequence length", lengths),
                    )
                    if not present
                )
                + ")"
            )
            continue

        # `12/12` is query/KV; `5/10/20` is three MHA stages of one UNet, not a
        # grouping -- a single value repeats as its own KV count.
        pairs = (
            [(heads[0], heads[1])]
            if len(heads) == 2
            else [(head, head) for head in heads]
        )
        for head_dim in head_dims:
            for heads_q, heads_kv in pairs:
                for dtype in dtypes:
                    for length in lengths:
                        for batch in batches:
                            common = {
                                "batch": batch,
                                "nhead_q": heads_q,
                                "nhead_k": heads_kv,
                                "hdim_q": head_dim,
                                "hdim_v": head_dim,
                                "dtype": dtype,
                                "mask_type": MASK_TYPE["causal" if causal else "full"],
                            }
                            shapes.append(
                                {
                                    **common,
                                    "seqlen_q": length,
                                    "seqlen_k": length,
                                    "_provenance": {
                                        "source": "catalog",
                                        "model": model,
                                        "phase": "prefill",
                                        "catalog": path.name,
                                    },
                                }
                            )
                            if causal:
                                shapes.append(
                                    {
                                        **common,
                                        "seqlen_q": 1,
                                        "mask_type": _causal_mask_type(True, 1, length),
                                        "seqlen_k": length,
                                        "_provenance": {
                                            "source": "catalog",
                                            "model": model,
                                            "phase": "decode",
                                            "catalog": path.name,
                                        },
                                    }
                                )
    if skipped:
        print(
            f"  NOTE: {len(skipped)} catalog entr(ies) yielded no shape: "
            + "; ".join(skipped)
        )
    return shapes


def _shape_name(shape: dict, index: int) -> str:
    """A human-readable name for one shape, or a positional one if it has no name."""
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

    Narrows to what `sdpa_fwd` can express (no sliding window, asymmetric head dim,
    or sinks) and counts each drop by reason. `q.alignment` is `bottom_right` for a
    bottom-right causal shape and `top_left` otherwise; it and `q.generate_stats`
    (always `false`: sources record inference forwards) are always written because
    argument resolution requires every parameter it reads.
    """
    columns = [
        "name",
        "op",
        "q.batch",
        "q.heads",
        "q.heads_kv",
        "q.seqlen_q",
        "q.seqlen_k",
        "q.head_dim",
        "q.is_causal",
        "q.alignment",
        "q.generate_stats",
        "q.dtype",
    ]
    dropped: dict[str, int] = {}
    written = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, shape in enumerate(shapes):
            reason = None
            if shape["mask_type"] == SLIDING_WINDOW:
                reason = "windowed mask, which sdpa_fwd declares no parameter for"
            elif shape["hdim_q"] != shape["hdim_v"]:
                reason = "asymmetric head dims (MLA), one head_dim declared"
            elif shape.get("use_sinks"):
                reason = "attention sinks, which sdpa_fwd declares no parameter for"
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            causal = shape["mask_type"] != NO_MASK
            writer.writerow(
                [
                    _shape_name(shape, index),
                    "sdpa_fwd",
                    shape["batch"],
                    shape["nhead_q"],
                    shape["nhead_k"],
                    shape["seqlen_q"],
                    shape["seqlen_k"],
                    shape["hdim_q"],
                    "true" if causal else "false",
                    (
                        "bottom_right"
                        if shape["mask_type"] == BOTTOM_RIGHT_CAUSAL
                        else "top_left"
                    ),
                    "false",
                    shape["dtype"],
                ]
            )
            written += 1
    return {"written": written, "dropped": dropped}


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
        help="A published shape directory. Reads "
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
        help="Keep sliding-window rows of the published CSV and of --shape-dir "
        "tabular files. Off by default: they are a different mask kind, and a kernel "
        "that clamps top-left only will decline them anyway -- but they are excluded "
        "LOUDLY here rather than folded onto causal. A kept window is a causal band "
        "(right_bound 0).",
    )
    parser.add_argument(
        "--out",
        help="Write the shape corpus here, as the request-field JSON "
        "`registry_enumerator.py --shapes` reads.",
    )
    parser.add_argument(
        "--out-query-csv",
        help="Also write the corpus as `q.<parameter>` columns, which "
        "`hipdnn_corpus_gen --model-shapes` reads as its model pool. A narrowing to "
        "what one operation declaration can express; what it drops is reported.",
    )
    parser.add_argument(
        "--excluded-out",
        help="Write the excluded inputs (out-of-family dtypes, sliding-window shape-dir "
        "rows), one record each with its path and reason. Default: <stem>.excluded.json "
        "next to --out, or next to --out-query-csv when --out is not given.",
    )
    args = parser.parse_args(argv)

    if not args.out and not args.out_query_csv:
        parser.error("give --out, --out-query-csv, or both; otherwise nothing is kept.")

    if (
        not args.published
        and not args.graphs
        and not args.rocke_bench
        and not args.catalog
        and not args.shape_dirs
    ):
        parser.error(
            "give at least one source. No corpus alone is sufficient: the CSV is "
            "what the kernel team measures, the graph tree is what callers send, "
            "rocKE's bench tree is what the kernel's own authors sweep, the catalog "
            "is what models were observed running, a shape directory is whatever a "
            "publisher handed over, and an integration sized from only one of them "
            "has missed real shapes twice."
        )

    shapes: list[dict] = []
    excluded: list[dict] = []
    if args.published:
        found = from_published_csv(
            Path(args.published), args.arch, args.include_windowed, excluded
        )
        print(f"  published CSV : {len(found):5d} rows for {args.arch}")
        shapes += found
    if args.graphs:
        found = from_graph_corpus(Path(args.graphs), excluded)
        print(f"  graph corpus  : {len(found):5d} forward graphs")
        shapes += found
    if args.rocke_bench:
        found = from_rocke_bench(Path(args.rocke_bench), "bf16", excluded)
        print(f"  rocKE bench   : {len(found):5d} trace records")
        shapes += found
    if args.catalog:
        found = from_model_catalog(Path(args.catalog))
        print(f"  model catalog : {len(found):5d} recorded geometries")
        shapes += found
    for directory in args.shape_dirs:
        found = from_shape_dir(
            Path(directory), args.arch, excluded, args.include_windowed
        )
        print(f"  shape dir     : {len(found):5d} published shapes in {directory}")
        shapes += found

    primary = Path(args.out or args.out_query_csv)
    excluded_out = (
        Path(args.excluded_out)
        if args.excluded_out
        else primary.with_name(f"{primary.stem}.excluded.json")
    )
    reasons: dict = {}
    for record in excluded:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    print(f"  excluded      : {len(excluded):5d}  {reasons or ''}")
    excluded_out.write_text(json.dumps(excluded, indent=2))

    unique, duplicates = deduplicate(shapes)
    print(
        f"  distinct      : {len(unique):5d}  ({duplicates} duplicate shape(s) merged)"
    )

    if not unique and excluded:
        # Everything was excluded with a reason: an accounted-for empty source,
        # which downstream tools take as such, not a mining failure.
        if args.out:
            Path(args.out).write_text("[]")
        if args.out_query_csv:
            write_query_csv([], Path(args.out_query_csv))
        out = args.out or args.out_query_csv
        print(
            f"\n  WARNING: every input was excluded; wrote an empty {out} and "
            f"{excluded_out}",
            file=sys.stderr,
        )
        return 0
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
        print(f"\n  wrote {args.out} (exclusions: {excluded_out})")
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
