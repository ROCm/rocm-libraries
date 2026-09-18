# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Source 2: the attention shapes real models actually run.

Two inputs, one output. `MODEL_CATALOG.md` is the in-tree record of which models were
driven through hipDNN and at what geometry -- the only place in this repository where
`bert = 12 heads, D=64, non-causal, Sq=Skv=512` is written down as an observation
rather than as a plausible number. A shape directory is the same thing from the
cluster: `--model-shapes ~/model-shapes` points at whatever the kernel team publishes
there, in any of the forms they publish it in.

These are the shapes an L1 heuristic must get right. A model that is excellent on the
sampled middle of the space and wrong on `llama` decode has failed at the only job it
has, and RFC 0019.13 §11.2's per-regime table exists to make that visible rather than
average it away -- so these arrive tagged with their model, not folded anonymously in.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

from . import graphs
from .shapes import Candidate, Shape

#: The in-tree catalog, relative to the repository root.
DEFAULT_CATALOG = Path("projects/hipdnn/tools/hipdnn_torch/MODEL_CATALOG.md")

#: Batches to expand each recorded geometry over. The catalog's A/B runs are all
#: single-batch (`output (1,256,1536)`), which is a measurement convention rather than
#: a serving one: `sdpa_fwd.opmeta.json`'s own archetypes put the same models at
#: batch 1..256. Emitting only batch 1 would train a heuristic that has never seen a
#: served batch of the shapes it exists to get right.
DEFAULT_BATCHES = (1, 8, 32)

#: Column spellings a shape file uses -> this tool's field. Three publishers disagree
#: (the kernel team's results CSV says `heads_q`/`seq_q`, aiter's model_shapes say
#: `nhead_q`/`seqlen_q`, a hand-written file says `h`/`hq`), and a shape file that
#: silently mines as zero rows is indistinguishable from one nobody pointed at.
_COLUMNS = {
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


def _load_mine_shapes():
    """`IngestorGenerator/tools/mine_shapes.py`, which owns the readers reused here.

    Loaded by path because it is a script, not a package -- the same reason
    `graphs._load_make_sdpa_bundles` does it. What is being reused is not
    convenience: `from_graph_corpus` derives causality from the graph's own
    attributes rather than from its file name (a `left_bound` of -1 with
    `causal_mask` false is how every shipped causal bundle spells it), and a second
    reader written here would get that wrong in exactly the way that one documents.
    """
    if "mine_shapes" in sys.modules:
        return sys.modules["mine_shapes"]
    path = (Path(__file__).resolve().parent.parent
            / "IngestorGenerator" / "tools" / "mine_shapes.py")
    spec = importlib.util.spec_from_file_location("mine_shapes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shape readers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mine_shapes"] = module
    spec.loader.exec_module(module)
    return module


mine = _load_mine_shapes()


def _causal_from_mask(mask_type: int) -> bool | None:
    """A mask value as this corpus can express it, or None for one it cannot.

    A sliding window is not causal-with-a-tweak: `mine_shapes` keeps it as its own
    value precisely so it does not get served as plain causal, and the graph this
    tool writes has no way to say `window` -- `make_sdpa_bundles` spells causality as
    the full `left_bound=-1` history. So a windowed shape is dropped and counted, not
    flattened onto causal.
    """
    if mask_type == mine.MASK_TYPE["causal"]:
        return True
    if mask_type == mine.MASK_TYPE["full"]:
        return False
    return None


def _shape_from_mined(record: dict) -> Shape | None:
    """One `mine_shapes` record as a `Shape`, or None if this corpus cannot build it."""
    causal = _causal_from_mask(record["mask_type"])
    if causal is None:
        return None
    if record.get("hdim_v") not in (None, record["hdim_q"]):
        # Asymmetric head dims (MLA's 192x128) need a V tensor of a different width,
        # which `make_sdpa_bundles.bundle_for` does not write. Dropping it is honest;
        # emitting it as a square graph would put a shape in the corpus that is not
        # the shape the row described.
        return None
    # A published row with more queries than keys is cross attention and is kept:
    # the declaration's `seqlen_q <= seqlen_k` governs the sampler, not what a model
    # was measured doing.
    try:
        return Shape(dtype=record["dtype"], batch=int(record["batch"]),
                     heads_q=int(record["nhead_q"]), heads_kv=int(record["nhead_k"]),
                     seqlen_q=int(record["seqlen_q"]), seqlen_kv=int(record["seqlen_k"]),
                     head_dim=int(record["hdim_q"]), causal=causal)
    except ValueError:
        # A head grouping no kernel implements, or a non-positive dimension. The row
        # is reported as unbuildable rather than repaired into a different shape.
        return None


# --------------------------------------------------------------------------- catalog


def _sections(text: str) -> dict:
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


def _sdpa_table(text: str) -> list[dict]:
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


def _numbers(field: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", field)]


def _sequence_lengths(body: str) -> list[int]:
    """Sequence lengths the entry records, from the two spellings it uses.

    `Sq=Skv=512` in the validated SDPA report and `seq512` in the run's geometry
    line. Nothing is inferred from a model's name: an entry that records no sequence
    length yields no shape and is reported as skipped, because a guessed prompt
    length is a shape no one measured.
    """
    found = set(int(value) for value in re.findall(r"Sq=Skv=(\d+)", body))
    found |= set(int(value) for value in re.findall(r"\bseq(\d+)\b", body))
    return sorted(found)


def from_catalog(path: Path, batches=DEFAULT_BATCHES) -> tuple[list[Candidate], dict]:
    """Every attention geometry `MODEL_CATALOG.md` records, expanded over batch.

    A causal entry additionally yields its decode shape -- one query token against
    the context it just filled. Decode is not an extra flavour of the prefill row:
    `sdpa_fwd.opmeta.json` anchors it as its own archetype because a `seqlen_q` of 1
    against a long cache is memory bound where the square prefill is compute bound,
    and a corpus that omits it teaches nothing about the half of serving that is
    decode. Encoder entries (bert, whisper, flux) get no decode shape: they have no
    autoregressive phase to decode in.
    """
    text = path.read_text(encoding="utf-8")
    sections = _sections(text)
    candidates: list[Candidate] = []
    skipped: list[dict] = []

    for row in _sdpa_table(text):
        model = row["model"]
        head_dims = _numbers(row["head_dims"])
        heads = _numbers(row["heads"])
        body = next((section for name, section in sorted(sections.items())
                     if name.startswith(model)), "")
        lengths = _sequence_lengths(body)
        # The dtypes the entry was actually validated in, not every dtype the model
        # could run in: an entry validated only in bf16 says nothing about fp16, and
        # the declaration sweep is where unrecorded combinations belong. `\bf16\b`
        # cannot match inside `bf16` -- `b` and `f` are both word characters -- so the
        # two spellings do not collide.
        dtypes = set()
        if re.search(r"\bbf16\b", body):
            dtypes.add("bf16")
        if re.search(r"\bfp16\b|\bf16\b", body):
            dtypes.add("fp16")
        dtypes = sorted(dtypes) or ["bf16"]
        causal = row["causal"].lower().startswith("y")

        if not heads or not head_dims or not lengths:
            skipped.append({"model": model, "reason": (
                "catalog records no " + ", ".join(
                    label for label, present in (("head count", heads),
                                                 ("head dim", head_dims),
                                                 ("sequence length", lengths))
                    if not present))})
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
                            common = dict(dtype=dtype, batch=batch, heads_q=heads_q,
                                          heads_kv=heads_kv, head_dim=head_dim,
                                          causal=causal)
                            candidates.append(Candidate(
                                shape=Shape(seqlen_q=length, seqlen_kv=length, **common),
                                source="model",
                                origin=f"{path.name}:{model}:prefill"))
                            if causal:
                                candidates.append(Candidate(
                                    shape=Shape(seqlen_q=1, seqlen_kv=length, **common),
                                    source="model",
                                    origin=f"{path.name}:{model}:decode"))
    return candidates, {"catalog": str(path), "rows": len(_sdpa_table(text)),
                        "shapes": len(candidates), "skipped": skipped}


# ------------------------------------------------------------------- shape directory


def _rows_from_json(path: Path) -> list[dict]:
    """Records from a JSON shape file, which is a list of them or a map of them."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(content, dict):
        if "tensors" in content and "nodes" in content:
            return []  # a graph; `mine.from_graph_corpus` reads those
        content = [dict(record, model=record.get("model", name))
                   for name, record in content.items() if isinstance(record, dict)]
    if not isinstance(content, list):
        return []
    return [record for record in content if isinstance(record, dict)]


def _rows_from_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_from_text(path: Path) -> list[dict]:
    """`key=value key=value` per line, one shape per line.

    The file convention is `dnn-convert-shapes`': one invocation per line, blank
    lines and `#` comments skipped, so a hand-maintained shape list reads the same
    whichever converter is pointed at it. None of that tool's code is reused --
    it converts MIOpen driver lines for convolution and batchnorm, and MIOpenDriver
    has no attention operation to spell a line for.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        pairs = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if pairs:
            rows.append(pairs)
    return rows


def _normalise_row(row: dict, path: Path) -> dict | None:
    """A published row in this tool's field names, or None if it is not a shape row.

    Every value that decides a kernel goes through `mine_shapes`' vocabulary rather
    than through a guess: an unrecognised dtype or mask spelling is refused there,
    loudly, because a guessed one produces a corpus entry that measures a different
    problem than the one the row named.
    """
    fields: dict = {}
    for key, value in row.items():
        if key is None:
            continue
        field = _COLUMNS.get(str(key).strip().lower())
        if field is not None and value not in (None, ""):
            fields.setdefault(field, value)
    required = ("batch", "heads_q", "seqlen_q", "seqlen_kv", "head_dim")
    if any(field not in fields for field in required):
        return None

    # A boolean `is_causal` column, or a named mask that `mine_shapes.MASK_TYPE`
    # already knows. Every other spelling is refused there rather than mapped here:
    # `bottom_right`, for one, is a genuinely different mask from the top-left causal
    # `make_sdpa_bundles` writes, and accepting it would put a shape in the corpus
    # that is not the one the row named.
    raw_mask = str(fields.get("mask", "")).strip().lower()
    if raw_mask in ("true", "1", "yes"):
        mask_type = mine.MASK_TYPE["causal"]
    elif raw_mask in ("", "false", "0", "no"):
        mask_type = mine.MASK_TYPE["full"]
    elif raw_mask in mine.MASK_TYPE:
        mask_type = mine.MASK_TYPE[raw_mask]
    else:
        raise SystemExit(
            f"FAIL: unknown mask spelling {raw_mask!r} in {path}. Add it to "
            f"mine_shapes.MASK_TYPE rather than defaulting -- guessing a mask puts a "
            f"differently-masked problem in the corpus under the row's name.")

    heads_q = int(fields["heads_q"])
    return {"batch": int(fields["batch"]), "nhead_q": heads_q,
            "nhead_k": int(fields.get("heads_kv", heads_q)),
            "seqlen_q": int(fields["seqlen_q"]), "seqlen_k": int(fields["seqlen_kv"]),
            "hdim_q": int(fields["head_dim"]), "hdim_v": int(fields["head_dim"]),
            "dtype": mine.normalise_dtype(fields.get("dtype"), path, "bf16"),
            "mask_type": mask_type, "arch": fields.get("arch"),
            "model": str(fields.get("model", path.stem))}


def from_shape_dir(root: Path, arch: str | None = None) -> tuple[list[Candidate], dict]:
    """Shapes from a published directory -- the cluster's `~/model-shapes`.

    Both forms are read: hipDNN graph JSON (through `mine_shapes.from_graph_corpus`,
    which is also what reads a `dnn-benchmarking` tree) and tabular files -- `.json`
    records, `.csv`, and `key=value` lines. Whichever the kernel team publishes, the
    directory is the pointer and nothing here has to be told which.
    """
    stats = {"root": str(root), "graphs": 0, "rows": 0, "unbuildable": 0,
             "wrong_arch": 0, "shapes": 0}
    candidates: list[Candidate] = []

    for record in mine.from_graph_corpus(root):
        stats["graphs"] += 1
        shape = _shape_from_mined(record)
        if shape is None:
            stats["unbuildable"] += 1
            continue
        origin = record.get("_provenance", {}).get("graph", "graph")
        candidates.append(Candidate(shape=shape, source="model",
                                    origin=f"{root.name}:{origin}"))

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".json":
            rows = _rows_from_json(path)
        elif path.suffix.lower() == ".csv":
            rows = _rows_from_csv(path)
        elif path.suffix.lower() in (".txt", ".shapes"):
            rows = _rows_from_text(path)
        else:
            continue
        for row in rows:
            record = _normalise_row(row, path)
            if record is None:
                continue
            stats["rows"] += 1
            if arch is not None and record["arch"] not in (None, arch):
                stats["wrong_arch"] += 1
                continue
            shape = _shape_from_mined(record)
            if shape is None:
                stats["unbuildable"] += 1
                continue
            candidates.append(Candidate(shape=shape, source="model",
                                        origin=f"{path.name}:{record['model']}"))

    stats["shapes"] = len(candidates)
    return candidates, stats


def collect(catalog: Path | None, shape_dirs: list[Path], batches, arch: str | None,
            max_bytes: int) -> tuple[list[Candidate], list[dict]]:
    """Every recorded shape, catalog first, within the measurable byte budget."""
    candidates: list[Candidate] = []
    reports: list[dict] = []
    if catalog is not None:
        found, stats = from_catalog(catalog, batches)
        candidates.extend(found)
        reports.append(stats)
    for directory in shape_dirs:
        found, stats = from_shape_dir(directory, arch)
        candidates.extend(found)
        reports.append(stats)

    kept, over_budget = [], 0
    for candidate in candidates:
        if graphs.footprint_bytes(candidate.shape) > max_bytes:
            over_budget += 1
            continue
        kept.append(candidate)
    if over_budget:
        reports.append({"over_byte_budget": over_budget})
    return kept, reports
