# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The probe-graph table (``tests/data/sdpa_bwd_probe_graphs.csv``, contract C11): the model's served rows over the instance tables.

The C++ replay of this table (``TestFlydslSdpaBwdParityGraphs``) holds the native matchers to the Python model; these tests hold the
table to the model and to the instance tables it is derived from, so a table or a rule that moves without the other fails here.
Pure Python: no content tree and no FlyDSL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
sys.path.insert(0, str(FLYDSL))
import sdpa_bwd_model as model  # noqa: E402


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return model.read_probe_csv()


def test_the_checked_in_table_is_what_the_model_derives(rows) -> None:
    assert model.PROBE_CSV.read_text(encoding="utf-8") == model.render()


def test_every_graph_round_trips_through_its_spelling(rows) -> None:
    for row in rows:
        graph = model.parse_graph(row["graph"])
        assert model.graph_text(graph) == row["graph"]


def test_every_served_row_is_a_row_of_its_table_and_the_list_is_ranked(rows) -> None:
    tables = {
        (arch, kind): {r["hash"]: r for r in model.table_rows(kind, arch)}
        for arch in model.ARCHES
        for kind in model.KERNELS
    }
    served = 0
    for row in rows:
        table = tables[row["arch"], row["kernel"]]
        hashes = [h for h in row["served_rows"].split(";") if h]
        assert set(hashes) <= table.keys(), row
        keys = [(-table[h]["priority"], h) for h in hashes]
        assert keys == sorted(keys), row
        served += len(hashes)
    assert served


def _dq_rows(rows, arch, graph_filter):
    tables = {r["hash"]: r for r in model.table_rows("dq", arch)}
    for row in rows:
        if row["arch"] == arch and row["kernel"] == "dq" and graph_filter(row):
            yield row, [tables[h] for h in row["served_rows"].split(";") if h]


def test_a_bias_that_stages_never_lists_a_direct_row_and_one_that_does_not_never_lists_a_staged_row(
    rows,
) -> None:
    """B-1 over the table: ``direct_only`` rows serve either bias, ``lds`` rows only a staged one, ``direct`` rows only an unstaged one."""
    seen = {"aligned": 0, "unaligned": 0}
    for pitch, forbidden in (("aligned", "direct"), ("unaligned", "lds")):
        for row, served in _dq_rows(
            rows,
            model.SDPA_BWD_TARGET,
            lambda r, p=pitch: f"pitch={p}" in r["graph"] and "bias=same" in r["graph"],
        ):
            stages = {r["knobs"]["bias_stage_resolved"] for r in served}
            assert forbidden not in stages, row
            seen[pitch] += bool(served)
    assert all(seen.values()), seen


def test_a_gfx11_bias_row_serves_every_bias_graph_it_covers(rows) -> None:
    """gfx11 never stages the bias: its rows are ``direct_only``, so the pitch does not change what serves.

    Graphs with a dBias are left out: BI-9 declines an unaligned dBias stride before any row is matched.
    """
    by_graph = {}
    for row, served in _dq_rows(
        rows,
        model.SDPA_BWD_GFX11_TARGET,
        lambda r: "bias=same" in r["graph"] and "dbias=0" in r["graph"],
    ):
        assert {r["knobs"]["bias_stage_resolved"] for r in served} <= {"direct_only"}
        key = (
            row["graph"]
            .replace("pitch=aligned", "pitch=*")
            .replace("pitch=unaligned", "pitch=*")
        )
        by_graph.setdefault((row["dtype"], key), set()).add(row["served_rows"])
    assert by_graph
    assert all(len(variants) == 1 for variants in by_graph.values())
