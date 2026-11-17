# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The search engine wired into dispatch: a tuned toy family searched
through its own KnobSpace, the winner recorded in the tuning DB and served
back through the normal pin path; transfer warm starts; the Autotuner and
sweep hooks; the CLI; and the torch-free import."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
from rocke.dispatch.core import Capability
from rocke.dispatch.tuning import (
    KnobSpace,
    gated,
    make_tuned_candidate,
    normalize_knobs,
    values,
)
from rocke.dispatch.tuning.search import (
    BOOL,
    ORDINAL,
    Budget,
    CompileOutcome,
    KnobSearchSpace,
    SpecBackend,
    TunedRecord,
    TuningDB,
    WarmStart,
    problem_key,
    search_configs,
    tune,
    tune_candidate,
)
from rocke.dispatch.tuning.search.__main__ import main as cli_main
from rocke.dispatch.tuning.search.synthetic import gemm_like
from rocke.helpers.autotune import AutotuneConfig, Autotuner

ABI = "rocke-searchtoy/v1"


@dataclass(frozen=True)
class Req:
    m: int
    arch: str = "gfx950"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    tuning_id: str = "auto"
    tuning_knobs: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))

    def normalized(self):
        return asdict(self)

    def dims(self):
        return {"m": self.m}

    def features(self):
        return frozenset()


@dataclass(frozen=True)
class Kern:
    m: int
    tile: int
    unroll: int = 1
    prefetch: bool = False
    prefetch_depth: int = 1
    waves_per_eu: int = 2


@dataclass(frozen=True)
class Tuned:
    tuning_id: str
    variant_id: str
    config_key: str
    knobs: tuple
    kernel: Kern

    def identity(self):
        return asdict(self.kernel)

    def launch_block(self):
        return (256, 1, 1)


@dataclass(frozen=True)
class Space(KnobSpace):
    outer_knob = "waves_per_eu"

    def axes(self, base):
        return (
            values("unroll", 1, (1, 2, 4, 8)),
            gated("prefetch", {"prefetch_depth": (1, 2, 3)}),
        )

    def build(self, base, knobs):
        return replace(base, **knobs)

    def validate(self, base, kernel):
        if kernel.unroll * kernel.tile > 512:
            return False, "register budget"
        return True, "ok"

    def defaults(self, base, kernel):
        return {k: v for k, v in asdict(base).items() if k != "m"}

    def outer_default(self, base):
        return base.waves_per_eu

    def outer_values(self, base, level):
        return (2, 1, 3, 4)

    def wrap(self, base, kernel, knobs, key, tid):
        return Tuned(tid, self.variant_id, key, knobs, kernel)


def _candidate():
    name = "searchtoy_gfx950_t128"
    return make_tuned_candidate(
        name=name,
        family="searchtoy",
        algorithm="toy_tuned",
        spec_id="gfx950_t128",
        abi_version=ABI,
        priority=10,
        capability=Capability(arches=("gfx950",), dtypes=("bf16",)),
        space=Space(
            abi=ABI, arch="gfx950", path="toy", variant_id="t128", candidate_name=name
        ),
        base=lambda req: Kern(m=int(req.m), tile=128),
        request_errors=lambda req: [],
        signature=lambda spec: (),
        build=lambda spec, arch: spec.kernel,
        bind_torch=lambda *a, **k: None,
        grid=lambda spec, req: (1, 1, 1),
    )


def _truth(k: Kern) -> float:
    t = 10.0 / k.unroll**0.5
    if k.prefetch:
        t *= {1: 0.97, 2: 0.9, 3: 0.95}[k.prefetch_depth]
    t *= {1: 1.1, 2: 1.0, 3: 0.96, 4: 1.05}[k.waves_per_eu]
    return t


def _spec_backend(space):
    return SpecBackend(
        space,
        compile=lambda spec: CompileOutcome(True, artifact=spec.kernel),
        check=lambda spec, art: (True, "ok"),
        measure=lambda spec, art, fid: [_truth(art)] * fid.reps,
    )


def _best_truth():
    best = math.inf
    for u in (1, 2, 4):
        for pf in (None, 1, 2, 3):
            for w in (1, 2, 3, 4):
                k = Kern(
                    m=0,
                    tile=128,
                    unroll=u,
                    prefetch=pf is not None,
                    prefetch_depth=pf or 1,
                    waves_per_eu=w,
                )
                best = min(best, _truth(k))
    return best


def test_knob_search_space_mirrors_the_candidate_space():
    cand = _candidate()
    ks, base = cand.search_space(Req(m=1024))
    space = KnobSearchSpace(ks, base)
    by = {p.name: p for p in space.params}
    assert by["unroll"].kind == ORDINAL and by["unroll"].levels[0] == ()
    assert by["prefetch"].kind == BOOL and by["prefetch_depth"].gate == "prefetch"
    assert by["waves_per_eu"].kind == ORDINAL
    # unroll 8 x tile 128 breaks the register budget: refused, with the reason
    g = space.encode({"unroll": 8})
    assert space.key(g) is None and "register" in space.why(g)
    # every valid key is a config_key the candidate serves
    seen = set()
    for g in space.enumerate():
        k = space.key(g)
        if k is None:
            continue
        seen.add(k)
        spec = space.spec(k)
        assert spec.config_key == k and spec.tuning_id.endswith("@" + k)
    assert len(seen) == 3 * 4 * 4


def test_search_space_refuses_unadmitted_requests():
    cand = _candidate()
    with pytest.raises(ValueError, match="does not admit"):
        cand.search_space(Req(m=1024, arch="gfx942"))


def test_tune_candidate_records_and_dispatch_serves_the_pick(tmp_path):
    cand = _candidate()
    db = TuningDB(tmp_path / "db.json")
    req = Req(m=1024)
    res, rec = tune_candidate(
        cand,
        req,
        _spec_backend,
        db=db,
        features={"m": 1024},
        budget=Budget(evaluations=64),
        strategy="portfolio",
        trace=tmp_path / "t.jsonl",
    )
    assert rec is not None and res.best_value == pytest.approx(_best_truth())
    assert res.observations[0].trial.origin == "warm:dispatcher"
    # confirmed against the dispatcher default: paired speedup and p-value
    c = res.confirmed
    default = c["baseline"]
    assert default in c["values"] and c["winner"] != default
    assert rec.speedup == pytest.approx(c["values"][default] / c["values"][c["winner"]])
    assert rec.speedup > 1.0 and rec.p_value < 0.05 and rec.rounds == 5
    assert rec.samples == 5 * 20 and rec.baseline_key == default
    assert rec.toolchain and rec.machine
    again = TuningDB(tmp_path / "db.json")
    stored = again.get("searchtoy", "gfx950", problem_key(req), cand.name)
    assert stored == rec
    served, served_rec = again.serve(
        req, "searchtoy", "gfx950", problem_key(req), cand.name
    )
    assert served_rec == rec
    pinned = again.pin(req, stored)
    assert served == pinned
    spec = cand.select_spec(pinned)
    assert spec.tuning_id == rec.tuning_id
    assert dict(spec.knobs) == dict(rec.knobs)
    # a stale record (knobs edited) is refused, never silently replaced
    stale = replace(pinned, tuning_knobs=normalize_knobs({"unroll": 2}))
    if dict(stale.tuning_knobs) != dict(rec.knobs):
        with pytest.raises(ValueError, match="stale"):
            cand.select_spec(stale)


def test_transfer_warm_starts_from_neighbors_and_other_archs(tmp_path):
    db = TuningDB(tmp_path / "db.json")

    def rec(problem, arch, m, unroll, value):
        return TunedRecord(
            family="f",
            candidate="c",
            arch=arch,
            problem=problem,
            tuning_id=f"v@k{problem}{arch}",
            config_key=f"k{problem}{arch}",
            knobs={"unroll": unroll},
            value=value,
            features={"m": m},
        )

    db.put(rec("p1", "gfx950", 1024, 2, 1.0))
    db.put(rec("p2", "gfx950", 8192, 4, 1.0))
    db.put(rec("p3", "gfx942", 2048, 8, 1.0))
    near = db.neighbors("f", "gfx950", {"m": 2048}, k=2)
    assert [r.problem for _d, r in near] == ["p1", "p2"]
    warm = db.warm_starts("f", "gfx950", "p3", {"m": 2048}, k=1)
    assert [w.source for w in warm] == ["neighbor:p1", "arch:gfx942"]
    # a slower record does not replace a faster one; the same config refreshes
    slower = replace(
        rec("p1", "gfx950", 1024, 1, 2.0), config_key="kz", tuning_id="v@kz"
    )
    assert not db.put(slower)
    assert db.put(rec("p1", "gfx950", 1024, 2, 1.5))
    assert db.get("f", "gfx950", "p1", "c").value == 1.5


def test_tuned_record_validation():
    with pytest.raises(ValueError, match="end in"):
        TunedRecord(
            family="f",
            candidate="c",
            arch="a",
            problem="p",
            tuning_id="x@1",
            config_key="2",
            knobs={},
            value=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        TunedRecord(
            family="f",
            candidate="c",
            arch="a",
            problem="p",
            tuning_id="x@1",
            config_key="1",
            knobs={},
            value=math.inf,
        )


def test_problem_key_ignores_pins():
    a = Req(m=64)
    b = replace(a, algorithm="x", spec_id="y", tuning_id="z@k", tuning_knobs={"u": 1})
    assert problem_key(a) == problem_key(b) != problem_key(Req(m=65))


# ------------------------------------------------------------ hooks
def test_autotuner_strategy_mode_measures_a_subset(tmp_path):
    @dataclass(frozen=True)
    class Spec:
        tile_m: int
        tile_n: int
        pipe: str

    configs = [
        AutotuneConfig(spec=Spec(tm, tn, p), name=f"{tm}x{tn}{p}")
        for tm in (32, 64, 128, 256)
        for tn in (32, 64, 128, 256)
        for p in ("a", "b", "c")
    ]

    def ms(cfg):
        s = cfg.spec
        return (
            abs(s.tile_m - 128) / 64
            + abs(s.tile_n - 64) / 64
            + {"a": 0.3, "b": 0, "c": 0.1}[s.pipe]
            + 1
        )

    calls = []

    def bench(cfg, **kw):
        calls.append(cfg.name)
        return ms(cfg)

    tuner = Autotuner(
        configs,
        key_fn=lambda **kw: ("k",),
        bench_fn=bench,
        launch_fn=lambda cfg, **kw: None,
        strategy="portfolio",
        budget=24,
        check_fn=lambda cfg, **kw: True,
        cache_path=tmp_path / "c.json",
    )
    win = tuner.select()
    assert len(calls) <= 24 < len(configs)
    assert win.name == "128x64b"
    calls.clear()
    assert tuner.select().name == "128x64b" and not calls  # cached
    # the search's winner is cached apart from a full sweep's: a sweeping
    # Autotuner on the same cache file does not serve the partial pick
    entries = json.loads((tmp_path / "c.json").read_text())
    (entry,) = entries.values()
    assert entry["mode"] == "search:portfolio" and entry["partial"] is True
    sweeper = Autotuner(
        configs,
        key_fn=lambda **kw: ("k",),
        bench_fn=bench,
        launch_fn=lambda cfg, **kw: None,
        cache_path=tmp_path / "c.json",
    )
    assert sweeper.select().name == "128x64b" and len(calls) == len(configs)
    with pytest.raises(ValueError, match="check_fn"):
        Autotuner(
            configs,
            key_fn=lambda **kw: (),
            bench_fn=bench,
            launch_fn=lambda c, **k: None,
            strategy="de",
        )


def test_autotuner_strategy_mode_skips_numeric_failures():
    configs = [AutotuneConfig(spec={"t": t}, name=str(t)) for t in range(12)]
    tuner = Autotuner(
        configs,
        key_fn=lambda **kw: (),
        bench_fn=lambda c, **k: 1.0 + c.spec["t"],
        launch_fn=lambda c, **k: None,
        strategy="exhaustive",
        budget=12,
        check_fn=lambda c, **k: (c.spec["t"] >= 3, "bad output"),
    )
    assert tuner.select().name == "3"


def test_search_configs_and_sweep_hook():
    space, f = gemm_like()
    gs = [g for g in space.enumerate() if space.valid(g)][:300]
    configs = [space.decode(g) for g in gs]
    idx, res = search_configs(
        configs,
        measure=lambda c, fid: f(c),
        check=lambda c: True,
        strategy="de",
        budget=Budget(evaluations=40),
        confirm=0,
    )
    assert idx is not None and res.evaluations <= 40
    assert f(configs[idx]) == res.best_value


def test_sweep_search_instances_hook():
    pytest.importorskip("rocke.sweep")
    from rocke.sweep import search_instances

    specs = [{"a": a, "b": b} for a in range(6) for b in range(6)]
    best, _res = search_instances(
        specs,
        measure=lambda s, fid: 1 + (s["a"] - 4) ** 2 + (s["b"] - 1) ** 2,
        check=lambda s: True,
        strategy="hill",
        budget=30,
        confirm=0,
    )
    assert best == {"a": 4, "b": 1}


# --------------------------------------------------------------- CLI
def test_cli_demo_replay_and_show(tmp_path, capsys):
    assert cli_main(["strategies"]) == 0
    assert "portfolio" in capsys.readouterr().out
    assert (
        cli_main(
            [
                "demo",
                "--objective",
                "sphere",
                "--strategy",
                "de",
                "--budget",
                "16",
                "--seeds",
                "1",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["runs"][0]["evaluations"] <= 16
    csv_path = tmp_path / "sweep.csv"
    rows = ["M,tile,pipe,tflops"]
    for m in (128, 256):
        for tile in (32, 64, 128, 256):
            for pipe in ("a", "b"):
                rows.append(f"{m},{tile},{pipe},{tile / (1 + (pipe == 'b')) + m / 100}")
    csv_path.write_text("\n".join(rows) + "\n")
    out_json = tmp_path / "r.json"
    assert (
        cli_main(
            [
                "replay",
                "--csv",
                str(csv_path),
                "--config-cols",
                "tile,pipe",
                "--value-col",
                "tflops",
                "--higher-is-better",
                "--problem-cols",
                "M",
                "--strategies",
                "de,random,exhaustive",
                "--budgets",
                "8",
                "--seeds",
                "1",
                "--min-configs",
                "4",
                "--out",
                str(out_json),
            ]
        )
        == 0
    )
    rep = json.loads(capsys.readouterr().out)
    assert rep["problems"] == ["128", "256"]
    assert rep["summary"]["8"]["exhaustive"]["worst"] == 1.0
    runs = json.loads(out_json.read_text())["runs"]
    assert all(0 < r["best_seen"] <= 1.0 for r in runs)
    # show
    db = TuningDB(tmp_path / "db.json")
    db.put(
        TunedRecord(
            family="f",
            candidate="c",
            arch="a",
            problem="p",
            tuning_id="v@k",
            config_key="k",
            knobs={"u": 1},
            value=1.0,
        )
    )
    space, f = gemm_like()
    from rocke.dispatch.tuning.search import SyntheticBackend

    tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=4),
        trace=tmp_path / "t.jsonl",
        confirm=0,
    )
    assert (
        cli_main(
            [
                "show",
                "--db",
                str(tmp_path / "db.json"),
                "--trace",
                str(tmp_path / "t.jsonl"),
            ]
        )
        == 0
    )
    shown = capsys.readouterr().out
    assert '"tuning_id": "v@k"' in shown and '"observations": 4' in shown


def test_cli_tune_with_factory(tmp_path, capsys, monkeypatch):
    mod = tmp_path / "searchtoy_factory.py"
    mod.write_text(
        "from rocke.dispatch.tuning.search import SyntheticBackend\n"
        "from rocke.dispatch.tuning.search.synthetic import sphere\n"
        "def make(args):\n"
        "    assert args.factory_args == {'shape': '64x64', 'dtype': 'f16'}\n"
        "    space, f = sphere(levels=5, dims=2)\n"
        "    def record(res):\n"
        "        return dict(family='f', candidate='c', arch='gfx950', problem='p',\n"
        "                    tuning_id='v@' + res.best_key[:8], config_key=res.best_key[:8],\n"
        "                    knobs=res.best_config)\n"
        "    return {'space': space, 'backend': SyntheticBackend(f), 'record': record}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    db = tmp_path / "db.json"
    assert (
        cli_main(
            [
                "tune",
                "--factory",
                "searchtoy_factory:make",
                "--strategy",
                "de",
                "--budget",
                "12",
                "--db",
                str(db),
                "--confirm",
                "0",
                "--arg",
                "shape=64x64",
                "dtype=f16",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len(TuningDB(db)) == 1


def test_warm_start_json_round_trip():
    w = WarmStart({"a": 1, "b": "x"}, "selector")
    assert WarmStart.from_json(json.loads(json.dumps(w.to_json()))) == w


def test_engine_imports_without_torch():
    code = (
        "import sys; import rocke.dispatch.tuning.search as s; "
        "assert 'torch' not in sys.modules, 'torch imported'; print('ok')"
    )
    root = Path(__file__).resolve().parents[4] / "python"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root)},
    )
    assert out.returncode == 0, out.stderr
