# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The Python model of the SDPA backward kernel matchers, and the probe-graph table it writes.

    python tools/sdpa_bwd_model.py --write     # rewrite tests/data/sdpa_bwd_probe_graphs.csv
    python tools/sdpa_bwd_model.py --check     # compare it with what the model derives from the tables

``serves()`` is a **model of the C++ kernel matchers**, not their implementation, and must move with them:

* split: ``flydslSdpaBwdSplitKernelMatches<SLOT>`` with ``servesHeadDims``, ``floorOf`` and ``rungBelow`` in
  ``FlydslSdpaBwdSplitNative.cpp``;
* merged: ``flydslSdpaBwdMergedKernelMatches`` with ``servesHeadDims`` and ``rungBelow`` in ``FlydslSdpaBwdMergedNative.cpp``.

A row is read as its KMD spells it (``gen_descriptors.kmd_knobs``): ``bias_f32`` (0/1), ``u2_role`` (``none``, ``dk``, ``dq``) and,
on a dQ row, ``bias_stage_resolved`` (``lds``, ``direct``, ``direct_only``, ``na``). The rules of the value-head split and the
FLOAT bias are those of the plan's contract C4:

* every dK/dV and dQ row needs ``row.bias_f32 == graph.bias_f32``;
* a ``none`` row serves a graph iff ``Hk == Hv``; a ``dk`` or ``dq`` row iff ``Hk != Hv``;
* a bias dQ row: an ``lds`` row serves iff the bias is staged (``stagesBias`` of the graph, with the bias width) and 16-bit, a ``direct``
  row iff it is not (an aligned bias never lands on a direct row that has a staged alternative), a ``direct_only`` row any bias;
* the merged engine declines a FLOAT bias and ``Hk != Hv``.

**The probe-graph table** (``tests/data/sdpa_bwd_probe_graphs.csv``, contract C11): per arch, kernel (``prep``, ``dkdv``, ``dq``),
dtype and graph, the Spec hashes of the rows the model serves, ranked by priority, higher first (equal priorities are listed by
hash; the engine orders them by descriptor id). A graph is spelled ``hq=<> hk=<> hv=<> dqk=<> dv=<> bias=none|same|f32 pitch=none|aligned|unaligned
mask=none|causal dropout=0|1 dbias=0|1``: ``bias`` ``same`` is the dtype of Q, ``f32`` a FLOAT bias beside 16-bit Q; ``pitch``
``aligned`` is a bias whose row stride is a multiple of 8 elements and at least ``ceil8(Skv)`` with a 16-byte pointer (it stages), an
``unaligned`` one is not. The graphs have ``Hq == Hk`` (a grouped graph would make a ``gqa_pack`` row serve, whose resolved tile
lives in the built manifest, not the table), except the value-head graphs, whose ``Hk != Hv`` declines every packed row. The rows
are those of the instance tables; a dQ row's ``lds`` or ``direct`` resolution is the table's (``table_stage``) and is held to the
built manifests by ``tests/test_sdpa_bwd_enumeration.py``.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))

import gen_descriptors as gd  # noqa: E402
from generators._instances import (  # noqa: E402
    SDPA_BWD_GFX11_TARGET,
    SDPA_BWD_TARGET,
    instances_for,
)
from generators._sdpa_bwd_spec import rung_below, to_spec  # noqa: E402

PROBE_CSV = FLYDSL / "tests" / "data" / "sdpa_bwd_probe_graphs.csv"
PROBE_COLUMNS = ("arch", "kernel", "dtype", "graph", "served_rows")
ARCHES = (SDPA_BWD_TARGET, SDPA_BWD_GFX11_TARGET)
KERNELS = ("prep", "dkdv", "dq")

PACK_MAX_BLOCK_Q = gd.pack_max_block_q()
# The largest head-dim tile whose gfx950 16-bit dQ bias rows stage the bias in LDS under the policy (the S1 catalogue: tiles 32-224
# resolve ``lds``, tiles 256 and 512 ``direct``).
LDS_BIAS_MAX_TILE = 224


def floor_of(knobs: dict, dtype: str) -> int:
    floor = knobs["floor"]
    if floor == "g1":
        return 0
    if floor == "tail":
        return rung_below(knobs["head_dim"], dtype)
    return int(floor)


def serves_dims(knobs: dict, dtype: str, qk: int, v: int, prep: bool) -> bool:
    low, high = (v, v) if prep else (min(qk, v), max(qk, v))
    if knobs["padded"] == 0:
        return low == knobs["head_dim"] == high
    return low > floor_of(knobs, dtype) and high <= knobs["head_dim"]


def serves(op: str, row: dict, graph: dict) -> bool:
    """Whether the row of @p op is applicable to @p graph: the C++ kernel_match, modelled.

    A graph carries ``dtype``, ``qk``, ``v``, ``cls`` (``plain``, ``causal``, ``bias``), ``dbias``, ``reduces``, ``dropout``, ``gqa``,
    ``staging`` (the bias is staged: ``stagesBias`` with the bias width), and optionally ``bias_f32`` and ``value_heads`` (``Hk != Hv``).
    """
    knobs = row["knobs"]
    if row["dtype"] != graph["dtype"]:
        return False
    slot = op.removeprefix("sdpa_bwd_")
    bias_f32 = graph.get("bias_f32", False)
    value_heads = graph.get("value_heads", False)
    if op == "sdpa_bwd_merged":
        # Merged keeps its declines: a FLOAT bias and distinct V heads.
        if bias_f32 or value_heads:
            return False
        if not serves_dims(knobs, graph["dtype"], graph["qk"], graph["v"], False):
            return False
        if (knobs["has_dropout"] != 0) != graph["dropout"]:
            return False
    else:
        if not serves_dims(
            knobs, graph["dtype"], graph["qk"], graph["v"], slot == "prep"
        ):
            return False
        if slot == "prep":
            return knobs.get("sink", -1) != 1
        if not (
            knobs["dform"] == "rt" or (knobs["has_dropout"] != 0) == graph["dropout"]
        ):
            return False
        if knobs["bias_f32"] != int(bias_f32):
            return False
        if (knobs["u2_role"] != "none") != value_heads:
            return False
    if knobs["causal"] != int(graph["cls"] == "causal"):
        return False
    if knobs["has_bias"] != int(graph["cls"] == "bias"):
        return False
    if slot in ("dkdv", "merged") and knobs.get("gqa_pack", -1) == 1:
        if not graph["gqa"] or not 0 < knobs.get("block_q", -1) <= PACK_MAX_BLOCK_Q:
            return False
    if slot == "dq" and knobs["has_bias"]:
        stage = knobs["bias_stage_resolved"]
        if stage == "lds" and (not graph["staging"] or bias_f32):
            return False
        if stage == "direct" and graph["staging"]:
            return False
    if slot == "merged" and (
        knobs["has_bias"]
        and knobs.get("bias_stage", "auto") != "direct"
        and not graph["staging"]
    ):
        return False
    if slot in ("dq", "merged"):
        if graph["dbias"] and knobs.get("store_db", -1) == 0:
            return False
        if graph["reduces"] and not knobs.get("reduce_symbol"):
            return False
    return True


# ---------------------------------------------------------------------------
# Rows of the instance tables
# ---------------------------------------------------------------------------


def table_stage(knobs: dict, dtype: str, arch: str) -> str:
    """The build's resolution of a dQ row's bias stage, from the table: ``na`` without a bias, ``lds`` where the policy stages it, else ``direct``.

    The policy stages a 16-bit bias on gfx950 when the row spells no ``bias_stage``, resolves the 32-row MFMA family and ``block_n`` 64,
    and the tile is at most ``LDS_BIAS_MAX_TILE``; a gfx11 row never stages, nor an f32-bias or fp32 row.
    """
    if not knobs["has_bias"]:
        return "na"
    if (
        arch == SDPA_BWD_TARGET
        and dtype != "fp32"
        and knobs.get("bias_f32", -1) != 1
        and knobs.get("bias_stage", "auto") == "auto"
        and knobs.get("mfma_rows", -1) in (-1, 32)
        and knobs.get("block_n", -1) in (-1, 64)
        and knobs["head_dim"] <= LDS_BIAS_MAX_TILE
    ):
        return "lds"
    return "direct"


def table_rows(kind: str, arch: str) -> list[dict]:
    """The rows of the ``sdpa_bwd_<kind>`` table of @p arch as the model reads them (KMD spelling), without the dV-only rows.

    Each row: ``name``, ``dtype``, ``knobs``, ``priority`` and the Spec ``hash``.
    """
    op = f"sdpa_bwd_{kind}"
    instances = instances_for(op, arch)
    if arch == SDPA_BWD_TARGET:
        specs = [to_spec(i) for i in instances]
    else:
        from generators import _sdpa_bwd_rdna_build as rdna  # noqa: PLC0415

        specs = rdna.specs_for(kind)
    entries = []
    for instance, spec in zip(instances, specs, strict=True):
        knobs = dict(instance.knobs)
        if kind == "dq":
            knobs["bias_stage_resolved"] = table_stage(knobs, instance.dtype, arch)
        record = {"name": instance.name, "dtype": instance.dtype, "knobs": knobs}
        entries.append(
            (gd.KernelEntry(op, Path(), b"", record), instance.priority, spec.hash())
        )
    limits = gd._lds_tile_limits([entry for entry, _, _ in entries])
    return [
        {
            "name": entry.record["name"],
            "dtype": entry.record["dtype"],
            "knobs": gd.kmd_knobs(entry, limits),
            "priority": priority,
            "hash": digest,
        }
        for entry, priority, digest in entries
        if entry.record["knobs"].get("u2") != "dv"
    ]


# ---------------------------------------------------------------------------
# The probe-graph table
# ---------------------------------------------------------------------------

DIMS = ((64, 64), (100, 100), (256, 256))
# (Hq, Hk, Hv): equal heads, and a nested value-head split.
HEADS = ((8, 8, 8), (8, 4, 2))


def probe_graphs() -> list[tuple[str, dict]]:
    """(dtype, graph) of the probe-graph table: tokens of the CSV spelling as a dict (see the module docstring)."""
    graphs = []
    for dtype in ("f16", "bf16", "fp32"):
        for dqk, dv in DIMS:
            for dropout in (0, 1):
                base = dict(dqk=dqk, dv=dv, dropout=dropout)
                for mask in ("none", "causal"):
                    for hq, hk, hv in HEADS:
                        if dtype == "fp32" and hk != hv:
                            continue
                        graphs.append(
                            (
                                dtype,
                                dict(
                                    base,
                                    hq=hq,
                                    hk=hk,
                                    hv=hv,
                                    bias="none",
                                    pitch="none",
                                    mask=mask,
                                    dbias=0,
                                ),
                            )
                        )
                for bias in ("same", "f32") if dtype != "fp32" else ("same",):
                    for pitch in ("aligned", "unaligned"):
                        for dbias in (0, 1):
                            graphs.append(
                                (
                                    dtype,
                                    dict(
                                        base,
                                        hq=8,
                                        hk=8,
                                        hv=8,
                                        bias=bias,
                                        pitch=pitch,
                                        mask="none",
                                        dbias=dbias,
                                    ),
                                )
                            )
    return graphs


GRAPH_TOKENS = (
    "hq",
    "hk",
    "hv",
    "dqk",
    "dv",
    "bias",
    "pitch",
    "mask",
    "dropout",
    "dbias",
)


def graph_text(graph: dict) -> str:
    return " ".join(f"{name}={graph[name]}" for name in GRAPH_TOKENS)


def parse_graph(text: str) -> dict:
    """The tokens of a probe-graph cell (ints where the token is a number)."""
    tokens = dict(item.split("=") for item in text.split())
    if tuple(tokens) != GRAPH_TOKENS:
        raise ValueError(f"graph {text!r}: tokens are {GRAPH_TOKENS}")
    return {k: int(v) if v.lstrip("-").isdigit() else v for k, v in tokens.items()}


def model_graph(dtype: str, graph: dict) -> dict:
    """The model's graph of a probe graph (``serves``)."""
    has_bias = graph["bias"] != "none"
    return dict(
        dtype=dtype,
        qk=graph["dqk"],
        v=graph["dv"],
        cls=(
            "bias"
            if has_bias
            else graph["mask"] if graph["mask"] != "none" else "plain"
        ),
        dbias=bool(graph["dbias"]),
        reduces=False,
        dropout=bool(graph["dropout"]),
        gqa=graph["hq"] != graph["hk"],
        staging=graph["pitch"] == "aligned",
        bias_f32=graph["bias"] == "f32",
        value_heads=graph["hk"] != graph["hv"],
    )


def graph_declined(graph: dict) -> bool:
    """Whether the split graph match declines the probe graph before any row is matched.

    BI-9: a dBias whose row stride is not a whole 16-byte vector of the bias dtype is declined (an ``unaligned`` pitch with
    ``dbias=1``; the dBias has the bias layout), so no row of any slot serves the graph.
    """
    return graph["pitch"] == "unaligned" and bool(graph["dbias"])


def ranked(rows: list[dict]) -> list[str]:
    """The Spec hashes of @p rows, ranked: priority, higher first, then hash."""
    return [r["hash"] for r in sorted(rows, key=lambda r: (-r["priority"], r["hash"]))]


def probe_rows() -> list[dict]:
    """The rows of the probe-graph table the model derives from the instance tables."""
    rows = []
    for arch in ARCHES:
        tables = {kind: table_rows(kind, arch) for kind in KERNELS}
        for dtype, graph in probe_graphs():
            modelled = model_graph(dtype, graph)
            for kind in KERNELS:
                served = (
                    []
                    if graph_declined(graph)
                    else [
                        r
                        for r in tables[kind]
                        if serves(f"sdpa_bwd_{kind}", r, modelled)
                    ]
                )
                rows.append(
                    {
                        "arch": arch,
                        "kernel": kind,
                        "dtype": dtype,
                        "graph": graph_text(graph),
                        "served_rows": ";".join(ranked(served)),
                    }
                )
    return rows


def render() -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=PROBE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(probe_rows())
    return out.getvalue()


def read_probe_csv(path: Path = PROBE_CSV) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.write:
        PROBE_CSV.write_text(text, encoding="utf-8")
        print(f"wrote {PROBE_CSV} ({text.count(chr(10)) - 1} rows)")
        return 0
    if PROBE_CSV.read_text(encoding="utf-8") != text:
        print(f"{PROBE_CSV} differs from the model; run with --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
