# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An external tuning workflow's evaluation sets never train, tune or
validate tilewright (lib/external_sets.py): every loader of a shape list
and preflight refuse one, recognised by its fingerprint or its CSV
columns. The sets themselves are never read here: a synthetic list stands
in for one by adding its fingerprint."""

import hashlib
import json

import pytest

import run_orchestrator as ro
from lib import external_sets as xs
from lib import held_out as hol
from lib.bench_yaml import parse_shapes_yaml

GEMMS = [(64, 128, 256, 1), (4096, 4096, 4096, 1), (1, 7, 9, 3)]


def _yaml(path, gemms, flow=True):
    if flow:
        lines = [
            f"- {{rotating_buffer: 512, transA: T, transB: N, M: {m}, N: {n}, K: {k}, "
            f"batch_count: {b}, a_type: bf16_r}}"
            for m, n, k, b in gemms
        ]
    else:
        lines = [
            f"- M: {m}\n  N: {n}\n  K: {k}\n  batch_count: {b}" for m, n, k, b in gemms
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _csv(path, header, rows):
    path.write_text("\n".join([",".join(header), *(",".join(r) for r in rows)]) + "\n")
    return path


@pytest.fixture
def external(monkeypatch):
    """GEMMS stand for an external tuning workflow's evaluation set."""
    monkeypatch.setitem(xs.FINGERPRINTS, xs.fingerprint(GEMMS), "a test set")


def test_the_fingerprint_is_the_sha256_of_the_sorted_distinct_lines():
    want = hashlib.sha256(b"1,7,9,3\n64,128,256,1\n4096,4096,4096,1").hexdigest()
    shuffled = [GEMMS[2], GEMMS[0], GEMMS[1], GEMMS[0]]
    assert xs.fingerprint(GEMMS) == want == xs.fingerprint(shuffled)
    assert xs.fingerprint([(*g, "bf16_r") for g in GEMMS]) == want
    assert xs.fingerprint(GEMMS[:2]) != want
    assert xs.fingerprint([]) == hashlib.sha256(b"").hexdigest()
    # the three sets of the external workflow
    assert set(xs.FINGERPRINTS) == {
        "3432ad47d68c59d76c99707ac25bebcaa31bbc39cd911e1b75c4c47b83f5957d",
        "48a0bb491ef815ea602abb56794e9281fd72cafa9a08090b09f4687e156fc8e3",
        "ffe49717a4b2da1978161912228708a9d51e731b2df9cc13a5e722e6ab2481a8",
    }
    assert all("external tuning workflow" in v for v in xs.FINGERPRINTS.values())
    assert xs.refusal(GEMMS) is None


def test_the_loaders_refuse_a_list_of_its_fingerprint(tmp_path, external):
    flow = _yaml(tmp_path / "flow.yaml", GEMMS + GEMMS[:1])
    rows = tmp_path / "rows.yaml"
    rows.write_text("".join(f"- [{m}, {n}, {k}, {b}]\n" for m, n, k, b in GEMMS))
    table = _csv(
        tmp_path / "list.csv",
        ["m", "n", "k", "batch_count"],
        [[str(v) for v in g] for g in GEMMS],
    )
    (tmp_path / "d" / "stage01").mkdir(parents=True)
    _yaml(tmp_path / "d" / "stage01" / "shapes.yaml", GEMMS)
    for entry in (flow, rows, table, tmp_path / "d"):
        with pytest.raises(
            xs.ExternalSet, match="its GEMMs are a test set, which never"
        ):
            hol.read(entry)
        lists, errors = hol.read_all([entry])
        assert lists == [] and len(errors) == 1
    with pytest.raises(xs.ExternalSet, match="flow.yaml: its GEMMs are a test set"):
        parse_shapes_yaml(flow)
    # a part of the set, or more than it, is another list
    part = _yaml(tmp_path / "part.yaml", GEMMS[1:])
    assert hol.read(part).gemms == set(GEMMS[1:])
    assert len(parse_shapes_yaml(_yaml(tmp_path / "more.yaml", GEMMS + [(2, 2, 2, 1)])))


def test_a_list_with_the_columns_of_its_sets_is_refused(tmp_path):
    rows = [[str(v) for v in g] for g in GEMMS]
    head = ["m", "n", "k", "batch_count"]
    for extra in ("legacy_category", "Refined_Category"):
        bad = _csv(tmp_path / f"{extra}.csv", [*head, extra], [[*r, "3"] for r in rows])
        with pytest.raises(xs.ExternalSet, match=extra.lower()):
            hol.read(bad)
    for value in ("log_sample", "model"):
        mixed = [[*r, "mine"] for r in rows[:-1]] + [[*rows[-1], value]]
        bad = _csv(tmp_path / f"{value}.csv", [*head, "source"], mixed)
        with pytest.raises(xs.ExternalSet, match=f"its source column holds '{value}'"):
            hol.read(bad)
    ours = _csv(tmp_path / "ours.csv", [*head, "source"], [[*r, "audit"] for r in rows])
    assert hol.read(ours).gemms == set(GEMMS)
    assert xs.column_refusal(head) is None
    assert xs.column_refusal(["M", "source"], [{"source": "model"}]) is not None


def test_stage07_and_stage08_refuse_a_bench_yaml_of_its_fingerprint(
    tmp_path, external, monkeypatch
):
    import argparse

    pipeline = str(ro.THIS_DIR / "stages")
    monkeypatch.syspath_prepend(pipeline)
    import stage07_optional_validate_selection_time as s07
    import stage08_optional_validate_selection_efficiency as s08

    flow = _yaml(tmp_path / "flow.yaml", GEMMS)
    block = _yaml(tmp_path / "block.yaml", GEMMS, flow=False)
    for path in (flow, block):
        with pytest.raises(xs.ExternalSet, match="its GEMMs are a test set"):
            s07.load_bench_entries(path)
    assert len(s07.load_bench_entries(_yaml(tmp_path / "part.yaml", GEMMS[:2]))) == 2
    bench = tmp_path / "hipblaslt-bench"
    bench.write_text("")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps({"hipblaslt": {"transA": "T", "transB": "N"}}))
    args = argparse.Namespace(bench_binary=bench, config_yaml=cfg, quiet=True)
    logs, why = s08._bench_yaml("ds", block, tmp_path / "out", args)
    assert logs is None and why.startswith("its GEMMs are a test set, which never")
    assert not (tmp_path / "out" / "bench_input").exists()


def test_preflight_names_the_configs_lists_that_are_its_sets(tmp_path, external):
    flow = _yaml(tmp_path / "flow.yaml", GEMMS)
    chunks = tmp_path / "enriched"
    chunks.mkdir()
    _csv(chunks / "chunk_0000.csv", ["m", "n", "k", "refined_category"], [])
    fine = _yaml(tmp_path / "fine.yaml", GEMMS[:1])
    block = _yaml(tmp_path / "block.yaml", GEMMS, flow=False)
    cfg = {
        "audit": {"held_out": [str(flow)]},
        "stage07": {"bench_yamls": [str(fine), str(flow), str(block)]},
        "stage08": {
            "datasets": [
                {"name": "ds", "path": str(flow)},
                {"name": "enriched", "path": str(chunks)},
                {"name": "fine", "path": str(fine)},
                {"name": "missing", "path": str(tmp_path / "gone.yaml")},
            ]
        },
    }
    errors = xs.config_errors(cfg)
    assert errors == [
        f"audit.held_out: {flow}: its GEMMs are a test set, {xs.NEVER}",
        f"stage07.bench_yamls: {flow}: its GEMMs are a test set, {xs.NEVER}",
        f"stage07.bench_yamls: {block}: its GEMMs are a test set, {xs.NEVER}",
        f"stage08.datasets ds: {flow}: its GEMMs are a test set, {xs.NEVER}",
        f"stage08.datasets enriched: {chunks}: chunk_0000.csv: its column(s) "
        f"refined_category are those of an external tuning workflow's "
        f"evaluation sets, {xs.NEVER}",
    ]
    assert (
        xs.config_errors({"stage08": {"datasets": [{"name": "f", "path": str(fine)}]}})
        == []
    )
    import yaml
    from test_run_orchestrator import make_config

    run_cfg = yaml.safe_load(make_config(tmp_path).read_text())
    run_cfg["audit"] = {"held_out": [str(flow)]}
    errors, _ = ro.validate_config(run_cfg)
    assert errors == [f"audit.held_out: {flow}: its GEMMs are a test set, {xs.NEVER}"]


def test_the_run_preflight_refuses_them(tmp_path, external):
    from bench_fakes import solution
    from test_run_orchestrator import _unlisted_config

    flow = _yaml(tmp_path / "flow.yaml", GEMMS)
    cfg, paths = _unlisted_config(tmp_path, [solution(1, 1)])
    want = f"stage08.datasets ds: {flow}: its GEMMs are a test set, {xs.NEVER}"
    assert want not in ro.preflight(cfg, paths, "validate")
    cfg["stage08"]["datasets"] = [{"name": "ds", "path": str(flow)}]
    assert want in ro.preflight(cfg, paths, "validate")
    assert want in ro.preflight(cfg, paths, "full")


def test_stage08_refuses_a_dataset_of_its_fingerprint_once_it_reads_it(
    tmp_path, monkeypatch
):
    """A dataset of enriched rows or bench logs is recognised by the GEMMs
    stage08 reads from it, before anything is evaluated."""
    import argparse

    import stage08_optional_validate_selection_efficiency as s08
    from lib import repeats
    from stage05_train import load_enriched_chunks
    from test_stage05_train import CELLS3, shapes_for, write_round

    d = write_round(tmp_path / "enriched", shapes_for(21, 3, CELLS3[:2]))
    cells, _ = load_enriched_chunks(str(d), True)
    read = [
        (g["m"], g["n"], g["k"], g["batch_count"]) for gs in cells.values() for g in gs
    ]
    assert len(read) == 6
    monkeypatch.setitem(xs.FINGERPRINTS, xs.fingerprint(read), "a test set")
    ctx = {"identity": None, "aggregation": repeats.DEFAULT_AGGREGATION}
    out = tmp_path / "out"
    res = s08._evaluate_dataset(
        "ds", d, out, d, ctx, argparse.Namespace(quiet=True), []
    )
    assert res["rc"] == 1
    assert res["reason"] == f"its GEMMs are a test set, {xs.NEVER}"
    assert not (out / "per_gemm.csv").exists()
