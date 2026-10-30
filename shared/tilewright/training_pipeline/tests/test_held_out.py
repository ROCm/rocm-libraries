# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Held-out lists (lib/held_out.py, `audit.held_out`): what a list reads in
every format and what it refuses, and that no round draws a GEMM of a
registered list: from the config through the orchestrator's stage01 command
to the shapes stage01 writes, in a run of the orchestrator too."""

import json
import sys
from pathlib import Path

import pytest
import yaml

import run_orchestrator as ro
import stage01_generate_OOB_shapes as s01
from lib import audit as al
from lib import held_out as hol
from lib import pinning
from lib.bench_yaml import parse_shapes_yaml
from lib.grid import all_cell_labels, cell_key
from lib.shapes import ShapeFilter, device_memory_budget
from test_run_orchestrator import LARGE, TINY, env, make_config, run, stubs

GEMMS = [(64, 64, 48, 2), (8, 16, 32, 1), (1536, 4096, 7168, 1)]
STAGE01 = Path(s01.__file__).resolve()
REAL_STAGE01 = """
import runpy, sys
sys.argv[0] = __REAL__
runpy.run_path(__REAL__, run_name="__main__")
"""


def _bench_yaml(path, gemms):
    lines = [
        f"- {{transA: T, transB: N, M: {m}, N: {n}, K: {k}, batch_count: {b}, "
        f"a_type: bf16_r, iters: 10}}\n"
        for m, n, k, b in gemms
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return path


def _rows(path, gemms):
    path.write_text(
        "# OOD list\n---\n"
        + "".join(f"- [{m}, {n}, {k}, {b}]\n" for m, n, k, b in gemms)
    )
    return path


def _csv(path, gemms, batch="batch"):
    path.write_text(
        f"library,stratum,m,n,k,{batch}\n"
        + "".join(f"x,s,{m},{n},{k},{b}\n" for m, n, k, b in gemms)
    )
    return path


def _lists(tmp_path, gemms):
    """`gemms` as a list of every format: an audit directory, a bench yaml,
    plain rows, and CSVs with batch and with batch_count."""
    d = tmp_path / "lists"
    _bench_yaml(d / "fid" / "stage01" / "shapes.yaml", gemms)
    return [
        d / "fid",
        _bench_yaml(d / "bench.yaml", gemms),
        _rows(d / "rows.yaml", gemms),
        _csv(d / "plain.csv", gemms),
        _csv(d / "chunk.csv", gemms, batch="batch_count"),
    ]


def values(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


# ── reading a list ───────────────────────────────────────────────────────────


def test_every_format_reads_the_same_gemms(tmp_path):
    lists = [hol.read(p) for p in _lists(tmp_path, GEMMS)]
    assert all(lst.gemms == set(GEMMS) for lst in lists)
    assert len({lst.gemms_sha256 for lst in lists}) == 1
    assert len({lst.sha256 for lst in lists}) == 4
    fid = lists[0]
    assert (
        fid.file == (tmp_path / "lists" / "fid" / "stage01" / "shapes.yaml").resolve()
    )
    rec = fid.record(n_dropped=2)
    assert rec["n_gemms"] == 3 and rec["n_dropped"] == 2
    assert rec["entry"] == str(tmp_path / "lists" / "fid")
    assert rec["gemms_sha256"] == hol.gemms_digest(reversed(GEMMS))


def test_a_bench_entry_without_a_batch_count_is_batch_one(tmp_path):
    p = tmp_path / "l.yaml"
    p.write_text("- {M: 7, N: 9, K: 11, a_type: bf16_r}\n")
    assert hol.read(p).gemms == {(7, 9, 11, 1)}
    marked = tmp_path / "bom.csv"
    marked.write_bytes(b"\xef\xbb\xbfM,N,K,Batch\r\n7,9,11,1\r\n")
    assert hol.read(marked).gemms == {(7, 9, 11, 1)}


@pytest.mark.parametrize(
    "name, text, why",
    [
        ("three.yaml", "- [64, 64, 48, 2]\n- [1, 2, 3]\n", "three.yaml:2: the entry"),
        ("nok.yaml", "- {M: 64, N: 64, batch_count: 2}\n", "names no GEMM"),
        ("block.yaml", "- M: 64\n  N: 64\n  K: 48\n", "names no GEMM"),
        ("zero.yaml", "- [64, 0, 48, 1]\n", "'0' is not a positive integer"),
        ("empty.yaml", "# nothing\n", "holds no GEMM"),
        ("flow.yaml", "[[64, 64, 48, 2]]\n", "holds no GEMM"),
        ("cols.csv", "m,n,k\n1,2,3\n", "needs the columns m, n, k and batch"),
        ("bad.csv", "m,n,k,batch\n1,2,x,1\n", "bad.csv:2: 'x' is not an integer"),
        ("short.csv", "m,n,k,batch\n1,2,3\n", "is not an integer"),
        ("both.csv", "m,n,k,batch,batch_count\n1,2,3,1,2\n", "differ"),
        ("none.csv", "m,n,k,batch\n", "holds no GEMM"),
    ],
)
def test_a_list_read_short_is_refused(tmp_path, name, text, why):
    p = tmp_path / name
    p.write_text(text)
    with pytest.raises(ValueError, match=why.replace("[", r"\[").replace("(", r"\(")):
        hol.read(p)


def test_a_missing_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        hol.read(tmp_path / "gone.yaml")
    (tmp_path / "d").mkdir()
    with pytest.raises(ValueError, match="a directory without stage01/shapes.yaml"):
        hol.read(tmp_path / "d")
    lists, errors = hol.read_all([tmp_path / "gone.yaml", _rows(tmp_path / "r", GEMMS)])
    assert [lst.gemms for lst in lists] == [set(GEMMS)] and len(errors) == 1
    digests = hol.digests([tmp_path / "gone.yaml"])
    assert digests[str(tmp_path / "gone.yaml")].startswith("unreadable (")


# ── stage01 ──────────────────────────────────────────────────────────────────

CONFIG = {
    "arch": "gfx950",
    "hipblaslt": {
        "transA": "T",
        "transB": "N",
        "a_type": "bf16_r",
        "b_type": "bf16_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "scale_type": "f32_r",
        "compute_type": "c_f32_r",
    },
    "bench": {"device_memory_gib": 64},
}


def _stage01(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["stage01", *map(str, argv)])
    return s01.main()


def _initial(monkeypatch, out, config, *extra):
    args = ["--mode", "initial", "--out-dir", out, "--config-yaml", config]
    return _stage01(monkeypatch, *args, "--target-per-cell", 3, "--quiet", *extra)


def test_stage01_never_draws_a_held_out_gemm_in_any_format(tmp_path, monkeypatch):
    """Every format holds out what the round would draw first: the round
    draws none of it and takes the next draws of the same cells instead."""
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    assert _initial(monkeypatch, tmp_path / "free", config) == 0
    free = parse_shapes_yaml(tmp_path / "free" / "shapes.yaml")
    free_cat = json.loads((tmp_path / "free" / "categorization.json").read_text())
    gemms = sorted(set(free))[::2]
    for i, entry in enumerate(_lists(tmp_path, gemms)):
        out = tmp_path / f"held_{i}"
        assert _initial(monkeypatch, out, config, "--exclude-shapes-yaml", entry) == 0
        drawn = parse_shapes_yaml(out / "shapes.yaml")
        assert not set(drawn) & set(gemms)
        cat = json.loads((out / "categorization.json").read_text())
        assert cat["actual_per_cell"] == free_cat["actual_per_cell"]
        (src,) = cat["exclusion_sources"]
        assert src["entry"] == str(entry)
        assert src["n_gemms"] == len(gemms) == src["n_dropped"]
        assert src["n_in_history"] == 0
        assert src["gemms_sha256"] == hol.gemms_digest(gemms)
        assert cat["n_excluded_shapes"] == len(gemms)


def test_stage01_refuses_a_list_it_cannot_read_whole(tmp_path, monkeypatch, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    bad = tmp_path / "ood.yaml"
    bad.write_text("- [64, 64, 48, 2]\n- {M: 1, N: 2}\n")
    out = tmp_path / "o"
    assert _initial(monkeypatch, out, config, "--exclude-shapes-yaml", bad) == 1
    assert "ood.yaml:2" in capsys.readouterr().err
    assert not (out / "shapes.yaml").exists()
    missing = tmp_path / "gone.yaml"
    assert _initial(monkeypatch, out, config, "--exclude-shapes-yaml", missing) == 1
    assert not (out / "shapes.yaml").exists()


def test_stage01_counts_held_out_gemms_the_history_holds(tmp_path, monkeypatch, capsys):
    """GEMMs of a list that earlier rounds drew are training data already:
    stage01 says how many, per list, and still draws none of them."""
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    assert _initial(monkeypatch, tmp_path / "r0", config) == 0
    r0 = sorted(set(parse_shapes_yaml(tmp_path / "r0" / "shapes.yaml")))
    late = _rows(tmp_path / "late.yaml", r0[:5] + GEMMS)
    out = tmp_path / "r1"
    extra = ["--prior-shapes-yaml", tmp_path / "r0" / "shapes.yaml"]
    extra += ["--round-index", 1, "--exclude-shapes-yaml", late]
    capsys.readouterr()
    assert _initial(monkeypatch, out, config, *extra) == 0
    (src,) = json.loads((out / "categorization.json").read_text())["exclusion_sources"]
    assert src["n_in_history"] == 5 and src["n_gemms"] == 5 + len(GEMMS)
    assert "5 of its 8 GEMMs are in the history" in capsys.readouterr().err
    assert not set(parse_shapes_yaml(out / "shapes.yaml")) & set(r0 + GEMMS)


# ── the config and the orchestrator ──────────────────────────────────────────


def _config(tmp_path, held_out=None, **overrides):
    keys = {
        "bench.device_memory_gib": 64,
        "full.total_rounds": 2,
        "validate.max_split_depth": 0,
        **overrides,
    }
    if held_out is not None:
        keys["audit.held_out"] = [str(p) for p in held_out]
    return make_config(tmp_path, **keys)


def _pipeline(cfg_path, run_dir, audit=False):
    cfg, _ = ro.load_config(cfg_path)
    return ro.Pipeline(
        cfg=al.bench_config(cfg) if audit else cfg,
        snapshot=cfg_path,
        run_dir=run_dir,
        paths=ro.derive_paths(cfg),
        stages_dir=STAGE01.parent,
        timing=ro.Timing(),
        force_deploy=False,
    )


def test_the_config_checks_every_held_out_list(tmp_path):
    lists = _lists(tmp_path, GEMMS)
    cfg = yaml.safe_load(_config(tmp_path, lists).read_text())
    assert ro.validate_config(cfg) == ([], [])
    assert al.held_out_entries(cfg) == [str(p) for p in lists]

    def errors(held_out, **audit):
        bad = dict(cfg, audit=dict(cfg.get("audit") or {}, held_out=held_out, **audit))
        return ro.validate_config(bad)[0]

    bad = _rows(tmp_path / "bad.yaml", GEMMS)
    bad.write_text(bad.read_text() + "- [1, 2]\n")
    (e,) = errors([str(bad)])
    assert e.startswith(f"audit.held_out: {bad}:6: the entry '- [1, 2]'")
    (e,) = errors([str(tmp_path / "gone")])
    assert e == f"audit.held_out: {tmp_path / 'gone'} does not exist"
    dup = [str(lists[0]), str(lists[0] / "stage01" / "shapes.yaml")]
    (e,) = errors(dup)
    assert e.endswith(f"{dup[1]} and {dup[0]} name the same list")
    audit = tmp_path / "audit"
    for own in (audit, audit / "stage01" / "shapes.yaml"):
        (e,) = errors([str(own)], dir=str(audit))
        assert e.endswith("is audit.dir's list, which every round leaves out already")
    assert errors("x.yaml") == ["audit.held_out: expected a list, got str"]
    assert errors([""])[0].startswith("audit.held_out[0]: expected a non-empty")
    del cfg["audit"]
    assert ro.cfg_get(cfg, "audit.held_out") == [] == al.held_out_entries(cfg)


def test_every_round_excludes_the_audit_set_and_every_held_out_list(tmp_path):
    audit = tmp_path / "audit"
    _bench_yaml(al.shapes_yaml(audit), GEMMS[:1])
    lists = _lists(tmp_path, GEMMS)
    p = _pipeline(_config(tmp_path, lists, **{"audit.dir": str(audit)}), tmp_path)
    rd = tmp_path / "run" / "round_1"
    prior = [tmp_path / "run" / "round_0"]
    steps = [
        ro.RoundStep(0, "initial", [], None, False),
        ro.RoundStep(1, "active", prior, None, True),
        ro.RoundStep(1, "active", prior, None, True, targets=[TINY]),
    ]
    for step in steps:
        excluded = values(p.args_stage01(rd, step), "--exclude-shapes-yaml")
        assert excluded == [str(al.shapes_yaml(audit))] + [str(e) for e in lists]


def test_an_audit_set_leaves_out_the_held_out_lists_but_its_own(tmp_path, monkeypatch):
    audit = tmp_path / "audit"
    inside = _rows(tmp_path / "inside.yaml", GEMMS)
    audit.mkdir()
    inside = inside.rename(audit / "earlier.yaml")
    outside = _rows(tmp_path / "ood.yaml", GEMMS)
    cfg_path = _config(tmp_path, [inside, outside])
    cfg, _ = ro.load_config(cfg_path)
    build = ro.AuditRun(_pipeline(cfg_path, audit, audit=True), cfg, [])
    args = build.args_stage01()
    assert values(args, "--exclude-shapes-yaml") == [str(outside)]
    assert _stage01(monkeypatch, *args, "--quiet") == 0
    cat = json.loads((audit / "stage01" / "categorization.json").read_text())
    assert [s["entry"] for s in cat["exclusion_sources"]] == [str(outside)]


def test_held_out_lists_are_pinned_only_when_set(tmp_path):
    """A run pinned before audit.held_out existed keeps its pin; a run that
    sets lists pins them and the digests of their GEMMs, so that a list that
    changes what the rounds may not draw is a change the next invocation
    names. An audit directory pins neither: its stage01 records the lists it
    left out."""
    lists = _lists(tmp_path, GEMMS)
    plain = yaml.safe_load(_config(tmp_path).read_text())
    held = yaml.safe_load(_config(tmp_path, lists[:2]).read_text())
    for audit in (False, True):
        before = ro.pinned_config(plain, audit)
        assert not [k for k in before if k.startswith("audit.held_out")]
        empty = dict(plain, audit={"held_out": []})
        assert ro.pinned_config(empty, audit) == before
        assert ro.pinned_config(held, audit) == (
            before
            if audit
            else {
                **before,
                "audit.held_out": [str(p) for p in lists[:2]],
                "audit.held_out_sha256": {
                    str(p): hol.gemms_digest(GEMMS) for p in lists[:2]
                },
            }
        )

    def pin(cfg):
        return {"config": pinning.config_record(ro.pinned_config(cfg))}

    def changes(old, new):
        return [d.split(":")[0] for d in pinning.differences(old, new, ("config",))]

    before = pin(held)
    _rows(lists[1], GEMMS[::-1])
    assert changes(before, pin(held)) == []
    _rows(lists[1], GEMMS[:2])
    assert changes(before, pin(held)) == ["config audit.held_out_sha256"]
    assert changes(pin(plain), pin(held)) == [
        "config audit.held_out",
        "config audit.held_out_sha256",
    ]


def _raw_draws(cfg_path, per_cell, round_index):
    """What stage01 draws for `per_cell` in round `round_index` of the
    config's seed with no history and nothing held out."""
    cfg, _ = ro.load_config(cfg_path)
    filt = ShapeFilter.from_config(cfg, budget=device_memory_budget(cfg))
    seed = s01.derive_seed(ro.cfg_get(cfg, "seed.seed"), round_index)
    return s01.sample_shapes(per_cell, seed, set(), filt).shapes


def _drawn(rd):
    return set(parse_shapes_yaml(rd / "stage01" / "shapes.yaml"))


def test_a_run_never_draws_a_registered_list(tmp_path, stubs, env, monkeypatch, capsys):
    """The lists hold what rounds 0 and 1 draw without them: a run that
    registers them (an audit directory, plain rows and a CSV) draws none of
    them, with every cell as full as without; registering a list that holds
    GEMMs of earlier rounds is a pin change, and the run names the rounds
    that drew them."""
    (stubs / ro.STAGE_DEFS["stage01"].script).write_text(
        REAL_STAGE01.replace("__REAL__", repr(str(STAGE01)))
    )
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    free_dir, held_dir = tmp_path / "free", tmp_path / "held"
    cfg = _config(tmp_path)
    target = ro.cfg_get(ro.load_config(cfg)[0], "seed.target_per_cell")
    raw0 = _raw_draws(cfg, {c: target for c in all_cell_labels()}, 0)
    raw1 = _raw_draws(cfg, {TINY: target, LARGE: target}, 1)
    assert run(stubs, cfg, free_dir, "--mode", "full") == 0
    assert _drawn(free_dir / "round_0") == set(raw0)
    assert set(raw1) & _drawn(free_dir / "round_1")

    d = tmp_path / "lists"
    fid = _bench_yaml(d / "fid" / "stage01" / "shapes.yaml", raw0[0::3]).parent.parent
    rows = _rows(d / "ood.yaml", raw0[1::3] + raw1[0::2])
    table = _csv(d / "ood.csv", raw0[2::3] + raw1[1::2])
    held_out = [fid, rows, table]
    cfg = _config(tmp_path, held_out)
    assert run(stubs, cfg, held_dir, "--mode", "full") == 0
    registered = set(raw0) | set(raw1)
    for r in (0, 1):
        rd = held_dir / f"round_{r}"
        assert _drawn(rd) and not _drawn(rd) & registered
        cmd = json.loads((rd / "stage01" / ro.STAGE_MARKER).read_text())["cmd"]
        assert values(cmd, "--exclude-shapes-yaml") == [str(p) for p in held_out]
        cat = json.loads((rd / "stage01" / "categorization.json").read_text())
        free = json.loads(
            (free_dir / f"round_{r}" / "stage01" / "categorization.json").read_text()
        )
        assert cat["actual_per_cell"] == free["actual_per_cell"]
        assert cat["n_excluded_shapes"] == len(registered)
    cats = [
        json.loads(
            (held_dir / f"round_{r}" / "stage01" / "categorization.json").read_text()
        )
        for r in (0, 1)
    ]
    fid_n, rows_n, table_n = (s["n_dropped"] for s in cats[0]["exclusion_sources"])
    assert fid_n == len(raw0[0::3])
    assert rows_n >= len(raw0[1::3]) and table_n >= len(raw0[2::3])
    assert {cell_key(*s) for s in raw1} == {TINY, LARGE}
    assert sum(s["n_dropped"] for s in cats[1]["exclusion_sources"]) >= len(raw1)
    pin = json.loads((held_dir / "pin.json").read_text())["pin"]
    assert pin["config"]["values"]["audit.held_out"] == [str(p) for p in held_out]

    late = _rows(d / "late.yaml", sorted(_drawn(held_dir / "round_0"))[:4])
    cfg = _config(tmp_path, held_out + [late])
    capsys.readouterr()
    assert run(stubs, cfg, held_dir, "--mode", "full") == 2
    assert "config audit.held_out" in capsys.readouterr().err
    assert run(stubs, cfg, held_dir, "--mode", "full", "--allow-code-change") == 0
    out = capsys.readouterr()
    assert (
        f"audit.held_out {late}: rounds drew its GEMMs before it held them out "
        f"(4 in round_0)" in out.out + out.err
    )


def test_the_guard_key_needs_the_static_check_and_is_pinned_only_when_on(tmp_path):
    cfg = yaml.safe_load(_config(tmp_path).read_text())
    cfg["correctness"] = {"enabled": True, "lists": []}
    base = ro.pinned_config(cfg)
    assert "correctness.guard_unguarded_custom" not in base
    assert ro.PINNED_UNLESS_NEUTRAL["correctness.guard_unguarded_custom"] is False
    on = dict(cfg, correctness=dict(cfg["correctness"], guard_unguarded_custom=True))
    assert ro.pinned_config(on)["correctness.guard_unguarded_custom"] is True
    errors = ro.validate_config(on)[0]
    assert not [e for e in errors if "guard_unguarded_custom" in e]
    off = dict(on, correctness=dict(on["correctness"], static_check=False))
    assert [e for e in ro.validate_config(off)[0] if "guard_unguarded_custom" in e] == [
        "correctness.guard_unguarded_custom: the guards come from the static check; "
        "turn correctness.static_check on"
    ]
