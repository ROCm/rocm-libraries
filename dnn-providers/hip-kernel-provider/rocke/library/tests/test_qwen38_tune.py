# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The monokernel tuner (``kernels.gfx950.qwen38_mono.tune``), CPU only.

Catalog contract (every listed level builds alone or is refused by a named
owner rule), canonical identity (base-equal and gated-off knobs drop out,
defaults are in the key), legality pruning, numerics classes, the
signed-lookback strategy (finds a planted optimum, skips consistent losers,
state round-trips), the mono portfolio, the DESIGN 9.1 / e2e win rules and
round mechanics, objectives, the tuning DB layer (records, served
defaults, gain ledger, selector), replay loaders and backends, and the batch
job evaluator with a fake transport (numeric failures never win).
"""

from __future__ import annotations

import dataclasses
import json
import math
import random

import pytest

from rocke.dispatch.tuning.search.driver import run_search
from rocke.dispatch.tuning.search.measure import Evaluator, SyntheticBackend
from rocke.dispatch.tuning.search.params import Param, SearchSpace
from rocke.dispatch.tuning.search.trial import Budget, Fidelity, Status

from kernels.gfx950.qwen38_mono.tune import catalog as C
from kernels.gfx950.qwen38_mono.tune import evaluate as E
from kernels.gfx950.qwen38_mono.tune import experiments as EX
from kernels.gfx950.qwen38_mono.tune import gate as G
from kernels.gfx950.qwen38_mono.tune import objective as O
from kernels.gfx950.qwen38_mono.tune import replay as RP
from kernels.gfx950.qwen38_mono.tune import tdb as D
from kernels.gfx950.qwen38_mono.tune.knobs import BITWISE, TOLERANCE, DataclassKnobs, Knob
from kernels.gfx950.qwen38_mono.tune.strategies import GainPrior, SignedLookback, mono_portfolio

# levels refused when set alone, by an owner rule (they need a companion knob)
ALONE_REFUSED = {
    ("moe_port", "m4_interleave", True): "needs m4_guard=none",
    ("moe_port", "m4_apipe", 1): "needs m4_interleave",
    ("moe_port", "slots", "arith"): "needs dedupe off and m5_ranks=pre",
    ("mx_gemv", "waves", 1): "fused needs ctas <= threads",
    ("mx_gemv", "waves", 2): "fused needs ctas <= threads",
    ("mx_gemv", "ksplit", 8): "waves must be a multiple of ksplit",
    ("mx_gemv", "fsync", "counter"): "power-of-two ctas",
    ("ccl", "self_local", True): "lamport one-shot",
    ("fused_moe", "gu_interleave", False): "gu_apipe needs gu_interleave",
    ("fused_moe", "route", "wave"): "route=wave needs slots arith|wave",  # probe P15
}


@pytest.fixture(scope="module")
def adapters():
    return C.all_adapters_cpu()


# ------------------------------------------------------------------ catalog
def test_catalog_has_every_owner(adapters):
    names = {a.name for a in adapters}
    assert {"framework", "toy_chain", "b3", "moe_port", "stage_fn", "mx_gemv", "ccl", "sched", "direct",
            "fused_moe"} <= names  # fmt: skip


def test_every_level_builds_alone_or_is_refused_by_a_named_rule(adapters):
    refused = {}
    for a in adapters:
        base = a.base_values()
        assert a.key(base) is not None, f"{a.stem}: the shipped build must be legal"
        for k in a.knobs:
            for v in k.values:
                if v == base.get(k.path):
                    continue
                cfg = {k.path: v}
                if k.gate:
                    g = a.knob(k.gate)
                    cfg[k.gate] = next(x for x in g.values if x != base.get(k.gate))
                ok, why = a.legal(cfg)
                if not ok:
                    refused[(a.name, k.path, v)] = why
    assert set(refused) == set(ALONE_REFUSED), sorted(set(refused) ^ set(ALONE_REFUSED))
    for key, why in refused.items():
        assert ALONE_REFUSED[key] in why, (key, why)


def test_level_zero_is_the_shipped_default(adapters):
    for a in adapters:
        base = a.base_values()
        for k in a.knobs:
            if k.kind != "ordinal":
                assert k.values[0] == base[k.path], (a.stem, k.path)
            else:
                assert base[k.path] in k.values, (a.stem, k.path)


def test_excluded_knobs_are_reasoned(adapters):
    for a in adapters:
        for k, why in a.excluded.items():
            assert why and isinstance(why, str), (a.stem, k)
    fm = C.fused_moe(1)
    assert "mutate" in fm.excluded and "w_null" in fm.excluded
    assert all(k.path not in ("mutate", "w_null", "phase_stamps") for k in fm.knobs)


# ----------------------------------------------------------------- identity
def test_keys_are_canonical_and_stable():
    a1, a2 = C.toy_chain(8, 8), C.toy_chain(8, 8)
    base = a1.base_values()
    k0 = a1.key({})
    assert k0 == a2.key({}) == a1.key(base)
    # a knob set to its default is the default build
    assert a1.key({"reduce": "swizzle"}) == k0
    k_dpp = a1.key({"reduce": "dpp"})
    assert k_dpp != k0 and k_dpp == a2.key({"reduce": "dpp"})
    # a gated-off knob drops out: the flip argument is read only with NUMA copies
    assert a1.key({"knobs.numa_flip_arg": "s_down"}) == k0
    on = {"knobs.mailbox.numa_copies": 2}
    assert a1.key({**on, "knobs.numa_flip_arg": "s_down"}) != a1.key(on)
    tid = a1.tuning_id({"reduce": "dpp"})
    assert tid == f"toy_chain.d8w8l1@{k_dpp}"
    # another variant (problem) is another key for the same knobs
    assert C.toy_chain(8, 4).key({"reduce": "dpp"}) != k_dpp


def test_defaults_are_in_the_key():
    a = C.framework()
    b = DataclassKnobs("framework", dataclasses.replace(a.base, head_sleep=64),
                       [dataclasses.replace(k) for k in a.knobs], variant=a.variant)  # fmt: skip
    assert a.key({"not_a_knob": 1}) == a.key({})  # unknown names are not knobs
    assert a.key({}) != b.key({})


def test_legality_pruning():
    toy = C.toy_chain(4, 0)
    assert toy.key({"knobs.fresh_ids": False}) is None
    assert "miscompiles" in toy.why_illegal({"knobs.fresh_ids": False})
    assert C.toy_chain(2, 0).key({"knobs.fresh_ids": False}) is not None
    g = C.mx_gemv(248320, 2560)
    assert g.key({"korder": "lane", "wload": "x1"}) is None
    sp = toy.space()
    rng = random.Random(1)
    for _ in range(50):
        r = sp.repair(sp.random_genome(rng), rng)
        assert r is None or sp.key(r) is not None


def test_numerics_classes():
    assert C.b3().numerics({"exp": "hw"}) == TOLERANCE
    assert C.b3().numerics({"wait": "barrier"}) == BITWISE
    assert C.mx_gemv(248320, 2560).numerics({"act": "mxfp8"}) == TOLERANCE
    assert C.ccl().numerics({"ar_dtype": "bf16"}) == TOLERANCE
    assert C.toy_chain(8, 8).structural({"hop": "reduce_task"}) == ["hop"]


def test_lever_strings():
    s = C.sched("delta", "r1")
    cfg = {"numa": 1, "hold.b1": 32, "hold.m4": 64, "hold.m3": 32}
    assert s.label(cfg) == "hold.b1=32,hold.m3=32,hold.m4=64,numa=1"
    assert s.key({"head_sleep": 64}) == s.key({})  # 64 is ATOM's default
    d = C.direct()
    assert C.direct_env(d.canonical_knobs({"dppmirror": True})) == {}  # gated off under comgr
    env = C.direct_env(d.canonical_knobs({"backend": "direct", "dppmirror": True, "ldsquads": True}))
    assert env == {"ROCKE_CODEOBJ_BACKEND": "direct", "ROCKE_DIRECT_LEVERS": "dppmirror+ldsquads"}


def test_n_levers_join_when_their_field_appears():
    @dataclasses.dataclass(frozen=True)
    class FakeB3:
        wait: str = "pre_poll"
        chunkwise: bool = False

    ad = DataclassKnobs("b3", FakeB3(), [Knob("wait", ("pre_poll", "barrier"))])
    ad2 = C.attach_n_levers(ad)
    names = [k.path for k in ad2.knobs]
    assert "chunkwise" in names and ad2.knob("chunkwise").numerics == TOLERANCE
    assert ad2.numerics({"chunkwise": True}) == TOLERANCE
    assert C.attach_n_levers(C.stage_fn()) is not None
    assert {lv.id for lv in C.N_LEVERS} >= {"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N10", "N11"}


def test_variant_features():
    assert C.variant_features("r4s") == {"rows": 4.0, "seq": 1.0, "group": 1.0, "numa": 1.0}
    assert C.variant_features("r4g2")["group"] == 2.0
    assert C.variant_features("d8w8l1") == {"depth": 8.0, "work": 9.0, "layers": 1.0}
    assert C.variant_features("delta.r1")["rows"] == 1.0


# --------------------------------------------------------------- strategies
def _synthetic_space():
    params = [Param.categorical(f"k{i}", ["off", "a", "b"]) for i in range(6)]
    return SearchSpace(params, name="synthetic")


def _objective(c):
    # k0=a and k1=b help alone; k2=a only helps together with k3=a; k5=b loses badly
    t = 1.0
    t *= 0.97 if c["k0"] == "a" else 1.0
    t *= 0.98 if c["k1"] == "b" else 1.0
    t *= 0.95 if (c["k2"] == "a" and c["k3"] == "a") else (1.004 if c["k2"] == "a" or c["k3"] == "a" else 1.0)
    t *= 1.05 if c["k5"] == "b" else 1.0
    return t


def test_signed_lookback_finds_the_planted_optimum():
    sp = _synthetic_space()
    be = SyntheticBackend(_objective, noise=0.001, seed=3)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    st = SignedLookback(sp, 0, start=(0,) * 6, lookback_batch=6)
    res = run_search(sp, st, ev, Budget(evaluations=60), batch=6, confirm=0)
    best = sp.decode(res.best_genome)
    assert best["k0"] == "a" and best["k1"] == "b" and best["k2"] == "a" and best["k3"] == "a", best
    assert best["k5"] != "b"
    rep = st.report()
    assert any(x.startswith("k0=") for x in rep["applied"])
    assert rep["lookback_screen"] > 0
    assert rep["lookback_pool"]["pruned"] == 0
    # state round-trips
    st2 = SignedLookback(sp, 0, start=(0,) * 6)
    st2.load_state_dict(json.loads(json.dumps(st.state_dict())))
    assert st2.report()["applied"] == rep["applied"]


def test_prior_orders_levers_but_never_prunes():
    """A prior solo loser still enters the lookback: the covering screen
    measures combinations containing it (PLAN decision 13)."""
    sp = _synthetic_space()
    prior = {("k5", "'b'"): [-0.05, -0.04], ("k0", "'a'"): [0.03]}
    st = SignedLookback(sp, 0, start=(0,) * 6, prior_gains=prior, lookback_batch=6)
    st._ensure_start()
    assert st.queue_levers[0] == sp.index("k0")  # most promising first
    assert sp.index("k5") in st.queue_levers  # the loser is still climbed
    be = SyntheticBackend(_objective, noise=0.001, seed=4)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    run_search(sp, st, ev, Budget(evaluations=60), batch=6, confirm=0)
    k5 = sp.index("k5")
    with_loser = [g for g in st.genomes.values() if g[k5] == 2 and sum(x != 0 for x in g) >= 2]
    assert with_loser, "the lookback never measured a combination containing the prior loser"


# ------------------------------------------------- decision 13: losers combine
def _all_lose_alone(c):
    """Every lever loses alone (1 % each); k1 + k2 together and k3 + k4 + k5
    together are much faster; k6 / k7 only ever lose."""
    on = {k for k, v in c.items() if v}
    t = 1.01 ** len(on)
    if {"k1", "k2"} <= on:
        t *= 0.85
    if {"k3", "k4", "k5"} <= on:
        t *= 0.88
    return t


def _bool_space():
    return SearchSpace([Param.boolean(f"k{i}") for i in range(1, 8)], name="losers")


@pytest.mark.parametrize("strategy", ["lookback", "phased"])
def test_combination_of_solo_losers_is_found(strategy):
    from kernels.gfx950.qwen38_mono.tune.strategies import phased

    sp = _bool_space()
    be = SyntheticBackend(_all_lose_alone, noise=0.001, seed=7)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    if strategy == "lookback":
        st = SignedLookback(sp, 1, start=(0,) * 7, lookback_batch=6, budget=70)
    else:
        st = phased(sp, 1, start=(0,) * 7, lookback_options={"budget": 70})
    res = run_search(sp, st, ev, Budget(evaluations=70), batch=6, confirm=0)
    best = sp.decode(res.best_genome)
    on = {k for k, v in best.items() if v}
    assert {"k1", "k2", "k3", "k4", "k5"} <= on and not on & {"k6", "k7"}, on
    lb = st if isinstance(st, SignedLookback) else st.members[0]
    rep = lb.report()
    assert rep["applied"] == []  # no lever wins alone: the climb applies nothing
    assert len(rep["combos"]["solo_losers"]) == 7  # all of them lost alone ...
    w, wo = rep["combos"]["best_with_solo_loser"], rep["combos"]["best_solo_winners_only"]
    assert w["value"] < 0.85 * wo["value"]  # ... and their combination wins by more than 15 %
    # the ledger's interaction terms carry the synergy (negative log-time terms)
    inter = rep["interactions"]
    assert inter and min(inter.values()) < -0.05
    assert any("k1" in k and "k2" in k for k, v in inter.items() if v < -0.05)


def test_ledger_keeps_pairs_and_combos(tmp_path):
    led = D.GainLedger(tmp_path / "g.json")
    led.add("x", {"k1=True": [-0.01]})
    led.add_pairs("x", {"k1=True|k2=True": -0.16})
    led.add_combos("x", {"k1=True,k2=True": 0.87})
    led.add_verdicts("x", [{"lever": "B", "level": "True", "gain": -0.03, "base_changes": [], "verdict": "rejected"},
                           {"lever": "B", "level": "True", "gain": 0.12, "base_changes": ["A=True"], "verdict": "applied"}])
    led.save()
    vb = D.GainLedger(tmp_path / "g.json").verdicts("x")["B=True"]
    assert [(v["base"], v["verdict"]) for v in vb] == [("", "rejected"), ("A=True", "applied")]
    led2 = D.GainLedger(tmp_path / "g.json")
    assert led2.prior("x") == {("k1", "True"): [-0.01]}
    assert led2.prior_pairs("x") == {"k1=True|k2=True": [-0.16]}
    # a later lookback warm-starts its GP prior mean from the pair term
    sp = _bool_space()
    st = SignedLookback(sp, 0, start=(0,) * 7, prior_gains=led2.prior("x"), prior_pairs=led2.prior_pairs("x"))
    st._ensure_start()
    st.center = list(st.best_g)
    f = frozenset({(sp.index("k1"), 1), (sp.index("k2"), 1)})
    assert st._prior_mean(f) == pytest.approx(math.log(1.01) - 0.16)


def test_mono_portfolio_runs_and_shares():
    sp = _synthetic_space()
    be = SyntheticBackend(_objective, noise=0.001, seed=5)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    pf = mono_portfolio(sp, 1)
    res = run_search(sp, pf, ev, Budget(evaluations=48), batch=4, confirm=0)
    assert res.best_value < 0.93
    assert sum(pf.asks) > 0 and len(pf.members) == 4


def test_phased_lookback_then_tpe():
    from kernels.gfx950.qwen38_mono.tune.strategies import phased

    sp = _synthetic_space()
    be = SyntheticBackend(_objective, noise=0.001, seed=9)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    st = phased(sp, 2, start=(0,) * 6, lookback_options={"order_restarts": 0, "revisit": False})
    res = run_search(sp, st, ev, Budget(evaluations=70), batch=4, confirm=0)
    assert res.best_value < 0.93
    assert st.asked[0] > 0 and st.members[0].done  # the climber finished, then TPE asked
    assert st.asked[1] > 0
    st2 = phased(sp, 2, start=(0,) * 6)
    st2.load_state_dict(json.loads(json.dumps(st.state_dict())))
    assert st2.cur == st.cur and st2.asked == st.asked


def test_gain_prior():
    p = GainPrior({("k0", "'a'"): [0.03, 0.01]}, base={"k0": "off"})
    a, b = p.predict([{"k0": "a"}, {"k0": "off"}])
    assert a == pytest.approx(0.98) and b == 1.0


# --------------------------------------------------------------------- gate
def test_toy_chain_gate_blind_spot_and_last_launch_mutants():
    """The toy chain's int32 recurrence multiplies r_prev by P (P + 1) / 2 =
    12880 = 2^4 x 805 per level (P = 160): after 8 levels a fault confined to
    an earlier launch vanishes mod 2^32 from that launch's output, so a gate
    mutant must sit in the last launch (live_toy._mutate)."""
    import numpy as np

    from kernels.gfx950.qwen38_mono import toy_chain as T

    rng = np.random.default_rng(3)
    for depth, blind in ((8, True), (4, False)):
        spec = T.ToyChainSpec(depth=depth, work=8)
        x = rng.integers(-1000, 1000, size=(spec.row_words,), dtype=np.int32)
        w = rng.integers(-1000, 1000, size=(1, depth, spec.producers, spec.row_words), dtype=np.int32)
        a = T.toy_reference(spec, 12345, x, w, [322])
        b = T.toy_reference(spec, 12346, x, w, [322])
        assert (a["out"][0] == b["out"][0]) is blind
        w2 = w.copy()
        j = int(np.flatnonzero(x)[0])
        w2[0, 0, 0, j] ^= 0x10
        c = T.toy_reference(spec, 12345, x, w2, [322])
        assert (a["out"][0] == c["out"][0]) is blind
        # the same fault is always visible in that launch's own level-0 mailbox
        assert not np.array_equal(a["y"][0][0], c["y"][0][0])


def test_win_rule():
    ctl = [[0.999, 1.001, 1.0, 0.998], [1.002, 0.999, 1.001, 1.0]]
    assert G.win_rule([0.98, 0.985, 0.99, 0.982], ctl).verdict == "WIN"
    assert G.win_rule([1.02, 1.03, 1.01, 1.02], ctl).verdict == "LOSS"
    assert G.win_rule([0.997, 1.0, 0.996, 1.001], ctl).verdict == "NOISE"
    # median clears the controls but only half the rounds are below 1
    assert G.win_rule([0.98, 1.01, 0.98, 1.01], ctl).verdict == "NOISE"
    # a throttled round is dropped
    v = G.win_rule([0.98, 0.985, 1.2, 0.982], ctl, drop=[2])
    assert v.verdict == "WIN" and v.rounds == 3


def test_e2e_rule():
    assert G.e2e_rule([0.95, 0.96, 0.94, 0.97, 0.95, 1.01], [0.995, 1.003]).verdict == "WIN"
    assert G.e2e_rule([0.95, 1.02, 0.94, 1.01, 0.95, 1.01], [0.995, 1.003]).verdict == "NOISE"


def test_rounds_are_shuffled_and_seeded():
    seen = []
    rows = G.shuffled_rounds(["base", "a", "b", "c"], 6, lambda a, r: seen.append((r, a)) or 1.0, seed=7)
    orders = [tuple(a for rr, a in seen if rr == r) for r in range(6)]
    assert len(set(orders)) > 1 and all(sorted(o) == ["a", "b", "base", "c"] for o in orders)
    seen2 = []
    G.shuffled_rounds(["base", "a", "b", "c"], 6, lambda a, r: seen2.append((r, a)) or 1.0, seed=7)
    assert seen == seen2 and len(rows) == 6
    ratios = G.per_round_ratios([{"base": 2.0, "x": 1.0}, {"base": 4.0, "x": 3.0}])
    assert ratios == {"x": [0.5, 0.75]}


def test_amd_smi_parse_and_throttle():
    j = [{"gpu": 0, "clock": {"gfx_0": {"clk": {"value": 2100, "unit": "MHz"}}},
          "power": {"socket_power": {"value": 812, "unit": "W"}}}]  # fmt: skip
    assert G.parse_amd_smi_metric(j) == (2100.0, 812.0)
    assert G.parse_amd_smi_metric({"gpu_data": [{"clock": {"gfx_0": {"clk": "1900 MHz"}}, "power": {"socket_power": "700 W"}}]}) == (1900.0, 700.0)
    assert G.throttled_rounds({0: 2100, 1: 2090, 2: 1800, 3: 2100}) == [2]


# ---------------------------------------------------------------- objective
def test_objective_gap_share():
    stage = O.Objective("stage", "K3C1", share=0.1)
    assert stage.round_ratio(0.8) == pytest.approx(0.98)
    assert stage.gap_closed(0.8) == pytest.approx(0.02 / 0.17)
    rnd = O.Objective("round", "K0C1")
    assert rnd.gap_closed(0.78) == pytest.approx(1.0)
    assert O.Objective.time_of(500.0, tokens_per_s=True) == pytest.approx(0.002)
    ranked = O.rank_by_gap(
        [{"name": "a", "objective": "s", "ratio": 0.8}, {"name": "b", "objective": "r", "ratio": 0.95}],
        objectives={"s": stage, "r": O.Objective("round")},
    )
    assert [r["name"] for r in ranked] == ["b", "a"]
    with pytest.raises(ValueError):
        O.Objective("round", share=0.5)


# ----------------------------------------------------------------------- db
def test_db_records_defaults_and_selector(tmp_path):
    db = D.TuningDB(tmp_path / "db.json", min_speedup=1.0)
    s1, s4 = C.sched("delta", "r1"), C.sched("delta", "r4s")
    r1 = D.make_record(s1, {"numa": 1, "hold.b1": 32}, ratio=0.86, speedup=1 / 0.86, p_value=1e-4, rounds=12,
                       strategy="mono", evaluations=40, problem="delta.r1")  # fmt: skip
    r4 = D.make_record(s4, {"numa": 1, "hold.m3": 128}, ratio=0.999, speedup=1.001, p_value=0.4, rounds=12,
                       strategy="mono", evaluations=40, problem="delta.r4s")  # fmt: skip
    assert db.put(r1) and db.put(r4)
    db.save()
    out = D.export_defaults(D.TuningDB(tmp_path / "db.json", min_speedup=1.0))
    assert out["sched"] == {"delta.r1": {"hold.b1": 32, "numa": 1}}  # the insignificant pick is not served
    assert out["sched_string"] == "delta.r1:hold.b1=32,numa=1"
    sel = D.Selector(db)
    ws = sel.top_k("sched", "delta.r2", k=2)
    assert ws and ws[0].source.startswith("neighbor:")
    led = D.GainLedger(tmp_path / "g.json")
    led.add("sched.delta.r1", {"hold.b1=32": [0.018]})
    led.save()
    led2 = D.GainLedger(tmp_path / "g.json")
    led2.add("sched.delta.r1", {"hold.b1=32": [0.018], "hold.m4=64": [0.009]})
    led2.save()
    # the second climb's measurement is a new entry; re-saving a loaded ledger doubles nothing
    want = {("hold.b1", "32"): [0.018, 0.018], ("hold.m4", "64"): [0.009]}
    assert D.GainLedger(tmp_path / "g.json").prior("sched.delta.r1") == want
    led3 = D.GainLedger(tmp_path / "g.json")
    led3.save()
    assert D.GainLedger(tmp_path / "g.json").prior("sched.delta.r1") == want


# ------------------------------------------------------------------- replay
def test_arm_parsers():
    assert RP.parse_sched_arm("mono") == {}
    assert RP.parse_sched_arm("mono2") is None
    assert RP.parse_sched_arm("N") == {"numa": 1}
    assert RP.parse_sched_arm("N_hb132_hm464_hd128") == {"numa": 1, "hold.b1": 32, "hold.m4": 64, "head_sleep": 128}
    assert RP.parse_sched_arm("N_remote") == {"numa": 3}
    assert RP.parse_sched_arm("hb1_32") == {"hold.b1": 32}
    assert RP.parse_sched_arm("hd_none") == {"head_sleep": 0}
    assert RP.parse_sched_arm("pause_4") is None
    assert RP.parse_b3_arm("rk:split=2,state_nt=True,mb_impl=ir") == {"split": 2, "state_nt": True, "mailbox.impl": "ir"}


def _tiny_table():
    t = RP.RoundTable()
    rng = random.Random(0)
    truth = {(): 1.0, (("numa", 1),): 0.9, (("hold.b1", 32),): 0.985, (("hold.b1", 32), ("numa", 1)): 0.87,
             (("hold.m4", 64), ("numa", 1)): 0.895, (("hold.b1", 32), ("hold.m4", 64), ("numa", 1)): 0.86}  # fmt: skip
    for k, v in truth.items():
        t.add(dict(k), [v * (1 + rng.gauss(0, 0.002)) for _ in range(12)], "synthetic")
    t.controls["mono2"] = [1 + rng.gauss(0, 0.002) for _ in range(12)]
    t.controls["mono_rc"] = [1 + rng.gauss(0, 0.002) for _ in range(12)]
    return t


def test_listed_space_and_round_replay():
    t = _tiny_table()
    defaults = C.sched("delta", "r1").base_values()
    p = EX.sched_problem(t, defaults, variant="r1")
    assert p.space.key(tuple(0 for _ in p.space.params)) is not None  # level 0 = shipped = recorded mono
    assert all(par.levels[0][0][1] == defaults[par.name] for par in p.space.params)
    be = p.make_backend(0)
    keys = list(p.truth)[:3]
    got = be.measure_batch([(None, k, be.key_of[k]) for k in keys], Fidelity(reps=1))
    assert len(got) == 3
    r = EX.run_strategy(p, "lookback", 6, 0)
    assert r["hit"], r
    r = EX.run_strategy(p, "mono", 6, 0)
    assert r["frac_gain"] > 0.8


# ----------------------------------------------------------------- evaluate
class _FakeTransport:
    """Pretends to be a batch job: k0 'b' fails numerics (it is the fastest
    level, so a gate failure must never win)."""

    def __init__(self):
        self.batches = []

    def run(self, batch):
        self.batches.append(batch)
        items = []
        for it in batch["items"]:
            c = it["config"]
            if c["k0"] == "b":
                items.append({"key": it["key"], "status": Status.NUMERIC_FAIL, "detail": "mismatch"})
                continue
            v = _objective(c)
            items.append({"key": it["key"], "status": Status.OK, "samples": [v, v * 1.001, v * 0.999]})
        return {"items": items, "meta": {"gate_selftest": {"caught": True}}}


def test_job_evaluator_with_fake_transport():
    sp = _synthetic_space()
    tr = _FakeTransport()
    ev = E.JobEvaluator(sp, {"kind": "synthetic"}, tr, ladder=(Fidelity(reps=3, warmup=0),))
    st = SignedLookback(sp, 0, start=(0,) * 6)
    res = run_search(sp, st, ev, Budget(evaluations=40), batch=6, confirm=0)
    assert tr.batches and all(len(b["items"]) <= 6 for b in tr.batches)
    assert sp.decode(res.best_genome)["k0"] != "b"
    fails = [o for o in res.observations if o.status == Status.NUMERIC_FAIL]
    assert fails and all(not math.isfinite(o.value) for o in fails)
    tok = E.encode_batch({"items": [{"key": "x", "config": {"a": 1}}]})
    assert E.decode_batch(tok) == {"items": [{"key": "x", "config": {"a": 1}}]}


# ------------------------------------- decision 15: order-dependent combinations
def _order_space():
    return SearchSpace([Param.boolean(n) for n in ("A", "B", "C", "D")], name="order")


def _b_needs_a(c):
    """A wins alone; B LOSES alone (+3 %) but wins big once A is applied; C
    loses a little, D does nothing."""
    t = 0.96 if c["A"] else 1.0
    if c["B"]:
        t *= 0.88 if c["A"] else 1.03
    if c["C"]:
        t *= 1.005
    return t


def _run_order(budget=6, **kw):
    sp = _order_space()
    be = SyntheticBackend(_b_needs_a, noise=0.0005, seed=11)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    st = SignedLookback(sp, 0, start=(0,) * 4, order=["B", "C", "A", "D"], **kw)
    res = run_search(sp, st, ev, Budget(evaluations=budget), batch=1, confirm=0)
    return sp, st, res


def test_rejected_lever_is_retested_after_an_accepted_move():
    """B is tested FIRST (on the start, where it loses) and rejected; once A is
    applied B is re-queued and wins on the new base: the climb finds A-then-B
    with the lookback off (8 evaluations: the start, B, C, A, then the re-tests)."""
    sp, st, res = _run_order(budget=8, revisit=False, order_restarts=0, lookback=False)
    best = sp.decode(res.best_genome)
    assert best["A"] and best["B"], best
    rep = st.report()
    assert rep["applied"] == ["A=True", "B=True"]
    b = [v for v in rep["verdicts"] if v["lever"] == "B" and v["level"] == "True"]
    assert b[0]["verdict"] == "rejected" and b[0]["base_changes"] == []  # lost on the start ...
    assert b[-1]["verdict"] == "applied" and b[-1]["base_changes"] == ["A=True"]  # ... won on the A base
    assert [m["move"] for m in rep["moves"]][:2] == ["climb A=True", "climb B=True"]
    assert rep["retests"] >= 1


def test_negative_control_without_retest_misses_the_order_dependence():
    sp, st, res = _run_order(budget=24, retest=False, revisit=False, order_restarts=0, lookback=False)
    assert not sp.decode(res.best_genome)["B"]  # the old one-pass climb gives B up for good
    assert st.done and res.evaluations < 24


def test_full_default_strategy_finds_the_order_dependence():
    sp, st, res = _run_order(budget=40)
    assert sp.decode(res.best_genome)["A"] and sp.decode(res.best_genome)["B"]


def test_revisit_pass_alone_recovers_the_reject():
    sp, st, res = _run_order(budget=16, retest=False, revisit=True, order_restarts=0, lookback=False)
    assert sp.decode(res.best_genome)["B"]
    assert any(m["move"] == "revisit rejects" for m in st.report()["moves"])


def test_order_restart_recovers_the_reject():
    sp, st, res = _run_order(budget=24, retest=False, revisit=False, order_restarts=1, lookback=False)
    assert sp.decode(res.best_genome)["B"]
    assert any(m["move"].startswith("order restart") for m in st.report()["moves"])


def test_external_new_base_requeues_rejects():
    # B, C measured and rejected
    sp, st, res = _run_order(budget=3, retest=True, revisit=False, order_restarts=0, lookback=False)
    assert any(k.startswith(f"{sp.index('B')}:") for k in st.rejected)
    st.set_base((1, 0, 0, 0), reason="restructure")  # e.g. another track's restructure lands A
    assert sp.index("B") in st.queue_levers and sp.index("C") in st.queue_levers  # re-queued, not dropped
    mv = st.report()["moves"][-1]
    assert mv["move"] == "base:restructure"
    assert "B" in mv["retest_front"] + mv["retest_back"]


def test_order_as_a_knob():
    from kernels.gfx950.qwen38_mono.tune.knobs import seq_decode

    @dataclasses.dataclass(frozen=True)
    class Pipeline:
        passes: tuple = ("x", "y", "z", "w")
        unroll: int = 1

    k = Knob.sequence("passes", ("x", "y", "z", "w"))
    assert len(k.values) == 24 and k.values[0] == "x>y>z>w" and k.seq
    ad = DataclassKnobs("pipe", Pipeline(), [k, Knob("unroll", (1, 2, 4), "ordinal")])
    assert ad.build({"passes": "w>x>y>z"}).passes == ("w", "x", "y", "z")
    assert ad.key({"passes": "x>y>z>w"}) == ad.key({})  # the shipped order is the base
    big = Knob.sequence("p8", [str(i) for i in range(8)], max_levels=200)
    assert len(big.values) == 200 and big.values[0] == "0>1>2>3>4>5>6>7"
    target = ("z", "w", "x", "y")

    def obj(c):  # faster the more item pairs are in the target's relative order
        o = seq_decode(c["passes"])
        pos = {x: i for i, x in enumerate(o)}
        good = sum(pos[a] < pos[b] for i, a in enumerate(target) for b in target[i + 1:])
        return 1.0 - 0.02 * good + (0.01 if c["unroll"] == 4 else 0.0)

    sp = ad.space()
    be = SyntheticBackend(obj, noise=0.0005, seed=2)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    st = SignedLookback(sp, 0, start=sp.encode(ad.base_values()))
    res = run_search(sp, st, ev, Budget(evaluations=60), batch=4, confirm=0)
    assert seq_decode(sp.decode(res.best_genome)["passes"]) == target
    # the climb's arms for an order knob are single moves, not all 24 orderings
    first = [e for e in st.report()["ledger"] if e["lever"] == "passes" and e["pass"] == 1]
    assert 0 < len(first) <= 9


def _wide_lever(c):
    """L (8 levels): level 3 wins ~3.5 % alone, every other level a little
    less; M wins ~3 % independently; Z (tested last) wins ~10 %."""
    t = {0: 1.0, 1: 0.97, 2: 0.969, 3: 0.965, 4: 0.97, 5: 0.971, 6: 0.97, 7: 0.972}[c["L"]]
    if c["M"]:
        t *= 0.97
    if c["Z"]:
        t *= 0.90
    return t


def _run_wide(budget, predicted=None, **kw):
    """``predicted``: {level: gain} the (mocked) interaction model claims for
    re-tests of L on the new base."""
    sp = SearchSpace([Param.categorical("L", list(range(8))), Param.boolean("M"), Param.boolean("Z")], name="wide")
    be = SyntheticBackend(_wide_lever, noise=0.0005, seed=3)
    ev = Evaluator(sp, be, ladder=(Fidelity(reps=3, warmup=0),))
    st = SignedLookback(sp, 0, start=(0, 0, 0), order=["L", "M", "Z"], budget=budget, **kw)
    if predicted is not None:

        def fake_rank(levers, scores=None, level_scores=None):
            base = math.log(st._value(st.best_key))
            ls = {lv: base - math.log1p(g) for lv, g in predicted.items()}
            for i in levers:
                if scores is not None:
                    scores[i] = min(ls.values())
                if level_scores is not None:
                    level_scores[i] = dict(ls)
            return list(levers)

        st._rank_rejects = fake_rank
    res = run_search(sp, st, ev, Budget(evaluations=budget), batch=1, confirm=0)
    return sp, st, res


def test_retests_do_not_starve_untested_levers_and_are_level_granular():
    """After M is accepted, L's seven rejected levels are stale. The model sees
    no reason to expect them to beat the new base by what an untested lever is
    expected to bring, so they queue BEHIND Z: a 10-evaluation budget (start,
    L x7, M, Z) still reaches Z. With budget left they are re-tested on the
    new base, and only the rejected levels (never the applied one)."""
    sp, st, res = _run_wide(10, revisit=False, order_restarts=0, lookback=False)
    assert sp.decode(res.best_genome)["Z"]
    sp, st, res = _run_wide(24, revisit=False, order_restarts=0, lookback=False)
    rep = st.report()
    assert sp.decode(res.best_genome)["Z"] and sp.decode(res.best_genome)["M"]
    re_l = [v for v in rep["verdicts"] if v["lever"] == "L" and "M=True" in v["base_changes"]]
    assert re_l, rep["moves"]  # re-tested on a new base: nothing permanently rejected
    assert all(v["level"] != "3" or v["verdict"] == "applied" for v in re_l)


def test_retest_front_gate_weighs_untested_levers():
    """The model predicts L's rejects beat the new base by more than the gain
    floor but by less than an untested lever is expected to bring (the mean
    first-test gain): they queue behind Z, which is reached in 10 evaluations
    (start, L x7, M, Z); a gate on the floor alone spends them on L."""
    sp, st, res = _run_wide(10, predicted={lv: 0.015 for lv in (0, 1, 2, 4, 5, 6, 7)},
                            revisit=False, order_restarts=0, lookback=False)  # fmt: skip
    assert st.gain_threshold(st.min_gain, 1) < 0.015 < st._expected_first_gain()
    assert sp.decode(res.best_genome)["Z"], st.report()["moves"]


def test_retest_is_level_granular():
    """Only the levels the model finds promising are re-tested ahead of the
    queue (one L arm, not seven): Z is reached in 11 evaluations; the other
    rejected levels of L stay open (re-queued at the next accepted move)."""
    sp, st, res = _run_wide(11, predicted={0: -0.02, 1: -0.02, 2: -0.02, 4: -0.02, 5: 0.10, 6: -0.02, 7: -0.02},
                            revisit=False, order_restarts=0, lookback=False)  # fmt: skip
    rep = st.report()
    assert sp.decode(res.best_genome)["Z"], rep["moves"]
    re_l = [v for v in rep["verdicts"] if v["lever"] == "L" and "M=True" in v["base_changes"]]
    assert [v["level"] for v in re_l] == ["5"]
    assert any(k.startswith("0:") and b != st.best_key for k, b in st.rejected.items())


def test_with_lookback_rejects_go_to_the_model_then_the_revisit_pass():
    """Lookback on, budget known: the re-tests the model does not put in front
    are handed to the lookback (its expected-improvement pool) instead of
    being queued as climb arms ahead of it; nothing is dropped: they stay open
    and the revisit pass re-tests them on the final base."""
    sp, st, res = _run_wide(48, order_restarts=0)
    rep = st.report()
    m = next(m for m in rep["moves"] if m["move"] == "climb M=True")
    assert m["retest_screen"] == ["L"] and m["retest_back"] == []
    assert any(m["move"] == "revisit rejects" for m in rep["moves"]), rep["moves"]
    final = sp.decode(res.best_genome)
    assert final["M"] and final["Z"]
    re_l = {v["level"] for v in rep["verdicts"] if v["lever"] == "L" and "M=True" in v["base_changes"]}
    assert re_l >= {"0", "1", "2", "4", "5", "6", "7"} - {str(final["L"])}, re_l


def test_lookback_measures_the_solo_winners_combination():
    """With ``solo_combo_screen`` (a retained, default-off knob) the
    lookback's first screen point is the main-effects pick (the start plus
    every lever at its best first-pass level) when the climb's path did not
    measure it (live 7.5: a search pick lost to that combination once)."""
    sp = SearchSpace([Param.categorical("L", list(range(8))), Param.boolean("M"), Param.boolean("Z")], name="wide")
    st = SignedLookback(sp, 0, start=(0, 0, 0), order=["L", "M", "Z"], budget=24, order_restarts=0,
                        solo_combo_screen=True)  # fmt: skip
    st.solo = {}
    sw = st._solo_winners_combo
    target = [5, 1, 0]
    st._solo_winners_combo = lambda: list(target)
    ev = Evaluator(sp, SyntheticBackend(_wide_lever, noise=0.0005, seed=3), ladder=(Fidelity(reps=3, warmup=0),))
    run_search(sp, st, ev, Budget(evaluations=24), batch=1, confirm=0)
    assert sp.key(tuple(target)) in st.history
    st._solo_winners_combo = sw
    st.solo = {"L=5": [0.02, 0.01], "L=3": [0.01], "M=True": [-0.01], "Z=True": [0.05]}
    assert sp.decode(st._solo_winners_combo()) == {"L": 5, "M": False, "Z": True}
