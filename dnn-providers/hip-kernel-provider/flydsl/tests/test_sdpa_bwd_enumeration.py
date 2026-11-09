# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The kernels that serve one graph have distinct knob tuples, and every promoted knob is real.

The runtime enumerates a slot's (or an unslotted engine's) candidates for a graph and refuses
an enumeration in which two kernels have the same tuple of UED knob values
(``GenericPlanBuilder::enumerateSlotted``, "Ambiguous enrolled knob tuple"). So the kernels
that match one graph must differ in a knob, and a knob is a performance alternative: some
graph must be served by kernels that differ in it, else it is a feature field that
``gen_descriptors.ALTERNATIVE_KNOBS`` must not list.

``tools/sdpa_bwd_model.py`` holds ``serves()``, a **model of the C++ kernel matchers**, not their implementation (it
must move with them; its docstring names them and states the rules of the FLOAT bias, the value-head split and the
bias stage, plan contract C4). This module probes it over synthetic and shipped catalogues.

The model covers the packed-tile guard (a ``gqa_pack`` row serves a graph only when its ``block_q`` is in
``(0, PACK_MAX_BLOCK_Q]``) and the launch-name sets of the rows launched through ``sources`` (``launch_names``).

Launch-grid limits are not modelled. Over the shipped manifests (skipped when the tree has
no content) the probe graphs are the classes in ``probe_graphs()``; a synthetic catalogue
that overlaps on purpose always runs.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402
import sdpa_bwd_model as model  # noqa: E402
from generators._flydsl_env import CONTENT_DIR  # noqa: E402
from sdpa_bwd_model import PACK_MAX_BLOCK_Q, serves  # noqa: E402

ARCH = "gfx950"
SLOTS = ("prep", "dkdv", "dq")


def launch_names(op: str, row: dict) -> tuple[str, ...]:
    """The launch names the handlers of @p op expect for @p row, in launch order (``()``: the row has a single ``kernel_source``)."""
    knobs = row["knobs"]
    if op == "sdpa_bwd_dkdv":
        # A dK-only row launches its dV-only partner beside it; every other dK/dV row has a single ``kernel_source``.
        return (gd.LAUNCH_NAME[op], gd.DVONLY_ROLE) if knobs["u2_role"] == "dk" else ()
    if op not in gd.MAIN_LAUNCH:
        return ()
    # A standalone dBias reduce is a companion of a gfx11 row that stores dB; every other row reduces inside its own object.
    standalone = row.get("standalone_reduce", False) and bool(
        knobs.get("reduce_symbol")
    )
    return (
        gd.MAIN_LAUNCH[op],
        *((gd.REDUCE_ROLE,) if standalone else ()),
        gd.TAILZERO_ROLE,
    )


def wrong_launch_sets(op: str, rows: list[dict]) -> list[str]:
    """The names of the rows whose ``sources`` keys are not their launch-name set."""
    return [
        row["name"]
        for row in rows
        if tuple(row.get("sources", ())) != launch_names(op, row)
    ]


def probe_graphs() -> list[dict]:
    """Graph classes: dtype, exact and padded dims (equal and unequal), mask, bias (staged or not, FLOAT or not), dropout, GQA, value heads."""
    graphs = []
    for dtype in ("bf16", "f16", "fp32"):
        for qk, v in (
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (224, 224),
            (100, 100),
            (200, 200),
            (300, 300),
            (512, 512),
            (100, 64),
            (192, 128),
        ):
            for cls, dbias, reduces in (
                ("plain", False, False),
                ("causal", False, False),
                ("bias", False, False),
                ("bias", True, False),
                ("bias", True, True),
            ):
                # (staging, FLOAT bias, Hk != Hv): the bias axes only on a bias graph, the value heads only without one; neither on fp32.
                if cls == "bias":
                    shapes = [
                        (staging, f32, False)
                        for staging in (True, False)
                        for f32 in ((False, True) if dtype != "fp32" else (False,))
                    ]
                else:
                    shapes = [(True, False, False)] + (
                        [(True, False, True)] if dtype != "fp32" else []
                    )
                for (staging, f32, value_heads), dropout, gqa in itertools.product(
                    shapes, (False, True), (False, True)
                ):
                    graphs.append(
                        dict(
                            dtype=dtype,
                            qk=qk,
                            v=v,
                            cls=cls,
                            dbias=dbias,
                            reduces=reduces,
                            dropout=dropout,
                            gqa=gqa,
                            staging=staging,
                            bias_f32=f32,
                            value_heads=value_heads,
                        )
                    )
    return graphs


def knob_tuple(op: str, row: dict) -> tuple:
    names = gd.knob_fields(op, row["knobs"])
    return tuple(row["knobs"][n] for n in names)


def co_applicable(op: str, rows: list[dict]) -> list[list[dict]]:
    """For each probe graph, the rows of @p op that apply to it (none: skipped)."""
    out = []
    for graph in probe_graphs():
        applicable = [r for r in rows if serves(op, r, graph)]
        if applicable:
            out.append(applicable)
    return out


def duplicate_tuples(op: str, groups: list[list[dict]]) -> list[tuple[str, str]]:
    found = []
    for rows in groups:
        seen: dict[tuple, str] = {}
        for row in rows:
            key = knob_tuple(op, row)
            if key in seen:
                found.append((seen[key], row["name"]))
            seen[key] = row["name"]
    return found


def unexercised_alternatives(op: str, groups: list[list[dict]]) -> set[str]:
    """Promoted fields of @p op that no probe graph has two applicable rows differing in."""
    missing = set()
    for field in gd.ALTERNATIVE_KNOBS.get(op, ()):
        if not any(len({r["knobs"].get(field) for r in rows}) > 1 for rows in groups):
            missing.add(field)
    return missing


# ---------------------------------------------------------------------------
# Synthetic catalogue: always runs
# ---------------------------------------------------------------------------


def _row(op: str, name: str, standalone_reduce: bool = False, **knobs) -> dict:
    base = dict(
        head_dim=64,
        padded=0,
        floor="tail",
        dform="bk",
        has_dropout=0,
        causal=0,
        has_bias=0,
        hints=1,
    )
    if op == "sdpa_bwd_prep":
        base.update(sink=-1)
    if op == "sdpa_bwd_dkdv":
        base.update(gqa_pack=-1, block_q=-1, bias_f32=0, u2_role="none")
    if op == "sdpa_bwd_dq":
        base.update(
            store_db=-1,
            bias_stage="auto",
            reduce_symbol="",
            bias_f32=0,
            u2_role="none",
            bias_stage_resolved="na",
        )
    if op == "sdpa_bwd_merged":
        base.update(gqa_pack=-1, block_q=-1, store_db=-1, reduce_symbol="")
    base.update(knobs)
    # A bias dQ row stages the bias unless it says otherwise.
    if base.get("bias_stage_resolved") == "na" and base["has_bias"]:
        base["bias_stage_resolved"] = "lds"
    row = {"name": name, "dtype": "bf16", "knobs": base}
    if op in gd.MAIN_LAUNCH:
        row["standalone_reduce"] = standalone_reduce
    if launches := launch_names(op, row):
        row["sources"] = dict.fromkeys(launches, {})
    return row


def synthetic() -> dict[str, list[dict]]:
    """Per op, rows that overlap on purpose in exactly the fields the op promotes."""
    catalogue = {}
    for op in ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"):
        rows = [_row(op, "exact64"), _row(op, "padded64", padded=1)]
        if op in ("sdpa_bwd_dkdv", "sdpa_bwd_dq"):
            rows += [
                _row(op, "padded128", head_dim=128, padded=1),
                _row(op, "g1_128", head_dim=128, padded=1, floor="g1"),
                _row(op, "rt64", dform="rt", has_dropout=1),
                _row(op, "bk64d", has_dropout=1),
            ]
        if op in ("sdpa_bwd_dkdv", "sdpa_bwd_merged"):
            rows.append(_row(op, "packed64", gqa_pack=1, block_q=64))
            # Over the bound, or without a resolved tile: never applicable, so never a duplicate of packed64.
            rows.append(_row(op, "packed128", gqa_pack=1, block_q=128))
            rows.append(_row(op, "packed_unresolved", gqa_pack=1, block_q=-1))
        if op == "sdpa_bwd_dkdv":
            rows.append(_row(op, "f32bias64", has_bias=1, bias_f32=1))
            rows.append(_row(op, "u2dk64", u2_role="dk"))
        if op in ("sdpa_bwd_dq", "sdpa_bwd_merged"):
            rows.append(_row(op, "nostore", has_bias=1, store_db=0))
            rows.append(_row(op, "store", has_bias=1, reduce_symbol="r"))
        if op == "sdpa_bwd_dq":
            rows += [
                _row(
                    op,
                    "direct",
                    standalone_reduce=True,
                    has_bias=1,
                    bias_stage="direct",
                    bias_stage_resolved="direct",
                    reduce_symbol="r",
                ),
                # A padded tile above the largest staged tile: no staged row serves its graphs, so it serves staged and unstaged bias alike.
                _row(
                    op,
                    "padded256",
                    head_dim=256,
                    padded=1,
                    floor="g1",
                    has_bias=1,
                    reduce_symbol="r",
                    bias_stage_resolved="direct_only",
                ),
                _row(
                    op,
                    "f32bias64",
                    has_bias=1,
                    bias_f32=1,
                    bias_stage="direct",
                    reduce_symbol="rf",
                    bias_stage_resolved="direct_only",
                ),
                _row(op, "u2dq64", u2_role="dq"),
            ]
        catalogue[op] = rows
    return catalogue


@pytest.mark.parametrize("op", sorted(synthetic()))
def test_the_promoted_knobs_keep_overlapping_kernels_distinct(op: str) -> None:
    groups = co_applicable(op, synthetic()[op])
    assert groups
    assert duplicate_tuples(op, groups) == []


@pytest.mark.parametrize("op", sorted(gd.ALTERNATIVE_KNOBS))
def test_every_promoted_knob_is_an_alternative_somewhere(op: str) -> None:
    groups = co_applicable(op, synthetic()[op])
    assert unexercised_alternatives(op, groups) == set()


@pytest.mark.parametrize("op", sorted(gd.MAIN_LAUNCH))
def test_the_synthetic_rows_launch_their_own_object_then_tailzero(op: str) -> None:
    rows = synthetic()[op]
    assert wrong_launch_sets(op, rows) == []
    assert {tuple(r["sources"]) for r in rows} <= {
        (gd.MAIN_LAUNCH[op], gd.TAILZERO_ROLE),
        (gd.MAIN_LAUNCH[op], gd.REDUCE_ROLE, gd.TAILZERO_ROLE),
    }
    broken = {**rows[0], "sources": {gd.MAIN_LAUNCH[op]: {}}}
    assert wrong_launch_sets(op, [broken]) == [broken["name"]]


@pytest.mark.parametrize("op", ["sdpa_bwd_dkdv", "sdpa_bwd_merged"])
def test_a_packed_row_without_a_tile_in_range_serves_no_graph(op: str) -> None:
    rows = {r["name"]: r for r in synthetic()[op]}
    gqa = dict(probe_graphs()[0], gqa=True)
    assert serves(op, rows["packed64"], {**gqa, "qk": 64, "v": 64})
    for name in ("packed128", "packed_unresolved"):
        assert not any(serves(op, rows[name], g) for g in probe_graphs())


def test_without_the_promotion_the_synthetic_catalogue_collides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gd, "ALTERNATIVE_KNOBS", {})
    for op in ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"):
        groups = co_applicable(op, synthetic()[op])
        assert duplicate_tuples(op, groups), op


def test_a_feature_field_is_not_an_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promoting `has_dropout` would list a field no two applicable rows differ in."""
    monkeypatch.setattr(
        gd,
        "ALTERNATIVE_KNOBS",
        {**gd.ALTERNATIVE_KNOBS, "sdpa_bwd_dkdv": frozenset({"has_bias"})},
    )
    groups = co_applicable("sdpa_bwd_dkdv", synthetic()["sdpa_bwd_dkdv"])
    assert unexercised_alternatives("sdpa_bwd_dkdv", groups) == {"has_bias"}


# ---------------------------------------------------------------------------
# The shipped catalogue
# ---------------------------------------------------------------------------


def _manifest(op: str) -> list[dict]:
    """The kernel rows of @p op's manifest in the KMD's spelling (``gen_descriptors.kmd_knobs``); a dV-only row is no kernel."""
    path = CONTENT_DIR / op / ARCH / "manifest.json"
    if not path.is_file():
        pytest.skip(f"no {op} content under {CONTENT_DIR}")
    records = [
        r
        for r in json.loads(path.read_text(encoding="utf-8"))["instances"]
        if r["knobs"].get("u2") != "dv"
    ]
    entries = [gd.KernelEntry(op, Path(), b"", record) for record in records]
    limits = gd._lds_tile_limits(entries)
    return [
        {
            "name": record["name"],
            "dtype": record["dtype"],
            "knobs": gd.kmd_knobs(entry, limits),
        }
        for record, entry in zip(records, entries)
    ]


@pytest.mark.parametrize(
    "op", ["sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"]
)
def test_the_shipped_kernels_that_serve_a_graph_have_distinct_knob_tuples(
    op: str,
) -> None:
    groups = co_applicable(op, _manifest(op))
    assert groups, f"no probe graph is served by {op}"
    collisions = duplicate_tuples(op, groups)
    assert not collisions, f"{len(collisions)} collisions, first {collisions[:3]}"


@pytest.mark.parametrize("op", sorted(gd.ALTERNATIVE_KNOBS))
def test_every_promoted_knob_of_the_shipped_kernels_is_a_real_alternative(
    op: str,
) -> None:
    groups = co_applicable(op, _manifest(op))
    assert unexercised_alternatives(op, groups) == set(), op


def _kernel_rows(op: str) -> list[dict]:
    """The rows of @p op's KDPs as the model reads them: name, dtype, metadata as knobs and the ``sources`` of a multi-launch row."""
    folder = CONTENT_DIR / op / ARCH
    paths = sorted(folder.glob("*.kdp.json"))
    if not paths:
        pytest.skip(f"no {op} descriptors under {CONTENT_DIR}")
    rows = []
    for path in paths:
        for kernel in json.loads(path.read_text(encoding="utf-8"))["kernelDescriptors"]:
            metadata = dict(kernel["metadata"])
            row = {
                "name": kernel["name"],
                "dtype": metadata.pop("dtype"),
                "knobs": metadata,
                "standalone_reduce": gd.REDUCE_ROLE in kernel.get("sources", {}),
            }
            if "sources" in kernel:
                row["sources"] = kernel["sources"]
            rows.append(row)
    return rows


@pytest.mark.parametrize(
    "op", ["sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"]
)
def test_every_shipped_row_launches_its_launch_name_set(op: str) -> None:
    rows = _kernel_rows(op)
    assert wrong_launch_sets(op, rows) == []
    if op not in gd.MAIN_LAUNCH:
        # A dK-only row of the value-head split is the one multi-launch row of dK/dV.
        assert not any(
            "sources" in r and r["knobs"].get("u2_role") != "dk" for r in rows
        )


@pytest.mark.parametrize("op", ["sdpa_bwd_dkdv", "sdpa_bwd_merged"])
def test_every_shipped_packed_row_has_a_tile_the_engine_accepts(op: str) -> None:
    packed = [r for r in _kernel_rows(op) if r["knobs"].get("gqa_pack") == 1]
    assert packed
    bad = {
        r["name"]: r["knobs"].get("block_q")
        for r in packed
        if not 0 < r["knobs"].get("block_q", -1) <= PACK_MAX_BLOCK_Q
    }
    assert not bad, f"{len(bad)} packed rows, first {list(bad.items())[:3]}"


def test_the_tables_stage_rule_is_the_builds_resolution() -> None:
    """``sdpa_bwd_model.table_stage`` (what the probe-graph table assumes of a dQ row) equals the stage the shipped build resolved."""
    from generators._instances import instances_for  # noqa: PLC0415

    path = CONTENT_DIR / "sdpa_bwd_dq" / ARCH / "manifest.json"
    if not path.is_file():
        pytest.skip(f"no sdpa_bwd_dq content under {CONTENT_DIR}")
    by_name = {i.name: i for i in instances_for("sdpa_bwd_dq", ARCH)}
    records = json.loads(path.read_text(encoding="utf-8"))["instances"]
    assert records
    for record in records:
        assert (
            "bias_stage_resolved" in record["knobs"]
        ), f"{record['name']}: the manifest predates the resolved stage; regenerate"
        instance = by_name[record["name"]]
        assert model.table_stage(instance.knobs, instance.dtype, ARCH) == (
            record["knobs"]["bias_stage_resolved"]
        ), record["name"]


# ---------------------------------------------------------------------------
# The FLOAT bias, the value-head split and the bias stage (contract C4)
# ---------------------------------------------------------------------------


def _graph(**fields) -> dict:
    base = dict(
        dtype="bf16",
        qk=64,
        v=64,
        cls="plain",
        dbias=False,
        reduces=False,
        dropout=False,
        gqa=False,
        staging=True,
        bias_f32=False,
        value_heads=False,
    )
    return {**base, **fields}


def _serving(op: str, **graph) -> set[str]:
    return {r["name"] for r in synthetic()[op] if serves(op, r, _graph(**graph))}


def test_an_aligned_bias_never_lands_on_a_direct_row_that_has_a_staged_alternative() -> (
    None
):
    """B-1 (``staged when available``): an ``lds`` row serves a staged bias, a ``direct`` row only a bias that does not stage."""
    staged = _serving("sdpa_bwd_dq", cls="bias", staging=True)
    assert {"store", "nostore"} <= staged and "direct" not in staged
    unstaged = _serving("sdpa_bwd_dq", cls="bias", staging=False)
    assert "direct" in unstaged and not {"store", "nostore"} & unstaged


def test_a_direct_only_row_serves_a_staged_and_an_unstaged_bias() -> None:
    """Where no staged row serves the graph (head dim above the largest staged tile) an aligned bias goes direct."""
    for staging in (True, False):
        assert "padded256" in _serving(
            "sdpa_bwd_dq", cls="bias", qk=200, v=200, staging=staging
        )
        assert "padded256" not in _serving("sdpa_bwd_dq", cls="plain", qk=200, v=200)


def test_a_float_bias_is_served_by_float_bias_rows_only() -> None:
    for op, name in (("sdpa_bwd_dkdv", "f32bias64"), ("sdpa_bwd_dq", "f32bias64")):
        assert name in _serving(op, cls="bias", bias_f32=True, staging=False)
        assert name in _serving(op, cls="bias", bias_f32=True, staging=True)
        assert name not in _serving(op, cls="bias", bias_f32=False)
    f32 = _serving("sdpa_bwd_dq", cls="bias", bias_f32=True)
    assert f32 == {"f32bias64"}, f32
    assert not _serving("sdpa_bwd_merged", cls="bias", bias_f32=True)


@pytest.mark.parametrize(
    ("op", "name"), [("sdpa_bwd_dkdv", "u2dk64"), ("sdpa_bwd_dq", "u2dq64")]
)
def test_a_value_head_row_serves_only_graphs_with_distinct_head_counts(
    op: str, name: str
) -> None:
    assert name in _serving(op, value_heads=True)
    assert name not in _serving(op, value_heads=False)
    assert name not in _serving(op, value_heads=True, cls="bias")
    # The Hk == Hv rows decline a graph with distinct V heads, and the merged rows decline it too.
    assert "exact64" not in _serving("sdpa_bwd_dkdv", value_heads=True)
    assert not _serving("sdpa_bwd_merged", value_heads=True)


def test_prep_is_blind_to_the_value_head_split() -> None:
    assert _serving("sdpa_bwd_prep", value_heads=True) == _serving("sdpa_bwd_prep")


def test_a_dk_only_row_launches_its_dv_only_partner_beside_it() -> None:
    rows = {r["name"]: r for r in synthetic()["sdpa_bwd_dkdv"]}
    assert tuple(rows["u2dk64"]["sources"]) == ("dkdv", "dvonly")
    assert "sources" not in rows["exact64"]
    assert wrong_launch_sets("sdpa_bwd_dkdv", list(rows.values())) == []
    assert wrong_launch_sets("sdpa_bwd_dkdv", [{**rows["u2dk64"], "sources": {}}]) == [
        "u2dk64"
    ]


# ---------------------------------------------------------------------------
# The installed instance tables (new rows of contract C4; no built content needed)
# ---------------------------------------------------------------------------


def _table_rows(op: str, arch: str) -> list[dict]:
    return model.table_rows(op.removeprefix("sdpa_bwd_"), arch)


@pytest.mark.parametrize("arch", model.ARCHES)
@pytest.mark.parametrize("op", [f"sdpa_bwd_{kind}" for kind in model.KERNELS])
def test_the_table_rows_that_serve_a_graph_have_distinct_knob_tuples(
    op: str, arch: str
) -> None:
    groups = co_applicable(op, _table_rows(op, arch))
    assert groups, f"no probe graph is served by the {arch} {op} table"
    collisions = duplicate_tuples(op, groups)
    assert not collisions, f"{len(collisions)} collisions, first {collisions[:3]}"


@pytest.mark.parametrize("op", sorted(gd.ALTERNATIVE_KNOBS))
def test_every_promoted_knob_is_a_real_alternative_in_the_gfx950_tables(
    op: str,
) -> None:
    """The padding, floor and stage alternatives exist on gfx950 only (a gfx11 row is one ``g1`` tile row).

    ``gqa_pack`` is left out: a packed row's resolved ``block_q`` lives in the built manifest, so no table row serves a grouped graph.
    """
    arch = ARCH
    kind = op.removeprefix("sdpa_bwd_")
    if kind not in model.KERNELS:
        pytest.skip(f"the probe table has no {kind} rows")
    groups = co_applicable(op, _table_rows(op, arch))
    assert unexercised_alternatives(op, groups) - {"gqa_pack"} == set(), (op, arch)


D7_CELLS = [
    (dtype, head_dim, dropout)
    for dtype in ("f16", "bf16")
    for head_dim in (64, 128)
    for dropout in (False, True)
]


def _bias_graph(dtype: str, head_dim: int, dropout: bool, **fields) -> dict:
    return _graph(
        dtype=dtype,
        qk=head_dim,
        v=head_dim,
        cls="bias",
        dropout=dropout,
        **fields,
    )


def _ranked_dq(arch: str, graph: dict) -> list[dict]:
    rows = [
        r for r in _table_rows("sdpa_bwd_dq", arch) if serves("sdpa_bwd_dq", r, graph)
    ]
    return sorted(rows, key=lambda r: (-r["priority"], r["hash"]))


@pytest.mark.parametrize(("dtype", "head_dim", "dropout"), D7_CELLS)
def test_an_unaligned_bias_ranks_the_exact_tile_direct_row_first_on_gfx950(
    dtype: str, head_dim: int, dropout: bool
) -> None:
    """M1: where the bias does not stage, the exact-tile ``store_db=0;bias_stage=direct`` row outranks every padded and every other row."""
    ranked = _ranked_dq("gfx950", _bias_graph(dtype, head_dim, dropout, staging=False))
    top = ranked[0]["knobs"]
    assert (
        top["padded"],
        top["head_dim"],
        top["store_db"],
        top["bias_stage_resolved"],
        top["bias_f32"],
        top["u2_role"],
    ) == (0, head_dim, 0, "direct", 0, "none"), ranked[0]["name"]
    assert ranked[0]["priority"] > ranked[1]["priority"]


@pytest.mark.parametrize(("dtype", "head_dim", "dropout"), D7_CELLS)
def test_an_unaligned_bias_ranks_the_exact_tile_row_first_on_gfx11(
    dtype: str, head_dim: int, dropout: bool
) -> None:
    """gfx11 never stages a bias: its tile row for the head dim outranks the larger tiles' rows (no new 16-bit direct row)."""
    ranked = _ranked_dq(
        "gfx11-generic", _bias_graph(dtype, head_dim, dropout, staging=False)
    )
    top = ranked[0]["knobs"]
    assert (top["head_dim"], top["bias_f32"], top["u2_role"]) == (head_dim, 0, "none")
    assert ranked[0]["priority"] > ranked[1]["priority"]


@pytest.mark.parametrize("dbias", [False, True])
@pytest.mark.parametrize(("dtype", "head_dim", "dropout"), D7_CELLS)
def test_an_aligned_16_bit_bias_graph_is_never_served_by_a_direct_row_while_an_lds_row_serves_it(
    dtype: str, head_dim: int, dropout: bool, dbias: bool
) -> None:
    """B-1 over the real gfx950 tables: with a staged row serving the graph, no ``direct`` row serves it, so none ranks above it."""
    ranked = _ranked_dq(
        "gfx950", _bias_graph(dtype, head_dim, dropout, staging=True, dbias=dbias)
    )
    stages = [r["knobs"]["bias_stage_resolved"] for r in ranked]
    assert "lds" in stages
    assert "direct" not in stages
    assert ranked[0]["knobs"]["bias_stage_resolved"] == "lds"


def test_an_aligned_16_bit_bias_above_the_largest_staged_tile_goes_direct_only() -> (
    None
):
    """B-1's exception: tiles 256 and 512 have no staged row, so an aligned bias there is served by ``direct_only`` rows alone."""
    for dtype in ("f16", "bf16"):
        for head_dim in (256, 512):
            ranked = _ranked_dq(
                "gfx950", _bias_graph(dtype, head_dim, False, staging=True)
            )
            stages = {r["knobs"]["bias_stage_resolved"] for r in ranked}
            assert stages == {"direct_only"}, (dtype, head_dim, stages)


@pytest.mark.parametrize("kind", ["dkdv", "dq"])
def test_no_two_default_rows_of_the_gfx11_table_serving_one_graph_tie(
    kind: str,
) -> None:
    """The rows without a feature knob (no FLOAT bias, no value-head role) of one tile family rank strictly on gfx11."""
    op = f"sdpa_bwd_{kind}"
    rows = [
        r
        for r in _table_rows(op, "gfx11-generic")
        if r["knobs"]["bias_f32"] == 0 and r["knobs"]["u2_role"] == "none"
    ]
    checked = 0
    for graph in probe_graphs():
        priorities = [r["priority"] for r in rows if serves(op, r, graph)]
        assert len(priorities) == len(set(priorities)), graph
        checked += bool(priorities)
    assert checked


# A bias dQ row that stores no dB (``store_db=0``) outranks its dB-storing twin: the criterion sits below the geometry (padded,
# tile, floor, dropout form), so it never lifts a row over a tighter-geometry row.

STORE_DB_DIMS = sorted(
    {16, 32, 48, 64, 80, 96, 100, 128, 160, 192, 200, 224, 256, 384, 512}
    | {d - 1 for d in (32, 64, 96, 128, 160, 192, 224, 256, 512)}
)


def _geometry(row: dict) -> tuple:
    knobs = row["knobs"]
    return (
        knobs["head_dim"],
        knobs["padded"],
        knobs["floor"],
        knobs["dform"],
    )


def _bias_dq_cells(arch: str):
    """(graph, serving rows ranked high first) over the bias dQ graphs of every dtype, dropout, staging and bias dtype."""
    rows = _table_rows("sdpa_bwd_dq", arch)
    for dtype in ("f16", "bf16", "fp32"):
        for head_dim in STORE_DB_DIMS:
            for dropout in (False, True):
                for staging in (False, True):
                    for bias_f32 in (False, True):
                        if dtype == "fp32" and bias_f32:
                            continue
                        for dbias in (False, True):
                            graph = _bias_graph(
                                dtype,
                                head_dim,
                                dropout,
                                staging=staging,
                                bias_f32=bias_f32,
                                dbias=dbias,
                            )
                            ranked = sorted(
                                (r for r in rows if serves("sdpa_bwd_dq", r, graph)),
                                key=lambda r: -r["priority"],
                            )
                            if ranked:
                                yield graph, ranked


@pytest.mark.parametrize("arch", model.ARCHES)
def test_a_bias_graph_without_dbias_picks_a_store_db_0_row_wherever_one_ties_the_top_geometry(
    arch: str,
) -> None:
    """The pick is a no-dB row when one serves at the geometry of the top row; a dBias graph is never served by one."""
    picked = 0
    for graph, ranked in _bias_dq_cells(arch):
        if graph["dbias"]:
            assert all(r["knobs"]["store_db"] != 0 for r in ranked), graph
            continue
        top = _geometry(ranked[0])
        if any(r["knobs"]["store_db"] == 0 for r in ranked if _geometry(r) == top):
            assert ranked[0]["knobs"]["store_db"] == 0, (graph, ranked[0]["name"])
            picked += 1
    assert picked


@pytest.mark.parametrize("arch", model.ARCHES)
def test_every_f32_bias_store_db_0_row_is_the_pick_of_some_graph(arch: str) -> None:
    """The f32-bias no-dB rows that no graph ranked first before the criterion (a lower kernel id broke the tie) are reachable."""
    reachable = {
        ranked[0]["name"]
        for graph, ranked in _bias_dq_cells(arch)
        if not graph["dbias"]
    }
    exact = [
        r["name"]
        for r in _table_rows("sdpa_bwd_dq", arch)
        if r["knobs"]["store_db"] == 0 and r["knobs"]["bias_f32"] == 1
    ]
    assert exact
    assert not set(exact) - reachable


@pytest.mark.parametrize("arch", model.ARCHES)
def test_no_two_16_bit_bias_dq_rows_serving_one_graph_tie(arch: str) -> None:
    """Every row serving a 16-bit bias dQ graph ranks strictly; the fp32 graphs' only ties are the ``block_m`` twins."""
    checked = 0
    for graph, ranked in _bias_dq_cells(arch):
        for first, second in zip(ranked, ranked[1:]):
            if first["priority"] != second["priority"]:
                continue
            assert graph["dtype"] == "fp32", (graph, first["name"], second["name"])
            assert {first["knobs"]["block_m"], second["knobs"]["block_m"]} == {16, 32}
            assert {**first["knobs"], "block_m": 0} == {
                **second["knobs"],
                "block_m": 0,
            }
        checked += len(ranked)
    assert checked
