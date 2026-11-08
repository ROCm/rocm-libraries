# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the gfx950 MXFP6 GEMM search driver (``builders.gfx950.mxfp6_gemm``).

Space derivation from ``Mxfp6GemmSpec`` / ``KNOBS`` (new knobs picked up),
the legality pre-filter, the compile gate, K-loop ISA metrics and the ISA
digest, aiter's packing and launch ABI (the in-harness competitor arm), the
statistics (win rule, paired bootstrap), the lever ledger, and the supervisor
end to end on the synthetic backend (``MXFP6_TUNE_FAKE``: no GPU, no comgr),
including a hung and a crashing candidate and a checkpoint resume.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from builders.gfx950.mxfp6_gemm import space as S
from builders.gfx950.mxfp6_gemm import tune as T
from kernels.gfx950.mxfp6_gemm import gemm as G

# ----------------------------------------------------------------- space


def test_space_is_read_from_the_knobs_table():
    sp = S.Mxfp6Space(G)
    table = getattr(G, "KNOBS")
    fields = {f.name for f in dataclasses.fields(G.Mxfp6GemmSpec)}
    for name, levels in table.items():
        if name in fields and len(set(levels)) > 1:
            assert name in sp.knobs, name
            assert set(sp.knobs[name]) == set(levels)
    for name in S.NOT_KNOBS:
        assert name not in sp.knobs
    # unlisted fields are held at the base and reported
    for name in fields - set(table) - set(S.NOT_KNOBS):
        assert name in sp.fixed
    # the base genome is the defaults (no overrides) and its spec is the default spec
    assert sp.space.decode(sp.base_genome) == {}
    assert sp.spec_of(sp.base_genome) == G.Mxfp6GemmSpec()
    assert sp.key(sp.base_genome) is not None


def test_space_picks_up_new_knobs(monkeypatch):
    """A field the kernel adds to KNOBS, and (with auto_unlisted) a new field without a table entry."""

    @dataclasses.dataclass(frozen=True)
    class Ext(G.Mxfp6GemmSpec):
        brand_new: bool = False

    monkeypatch.setattr(G, "Mxfp6GemmSpec", Ext)
    monkeypatch.setattr(G, "KNOBS", {**G.KNOBS, "store_nt": (False, True)})
    sp = S.Mxfp6Space(G)
    assert "store_nt" in sp.knobs and "store_nt" not in sp.fixed
    assert "brand_new" not in sp.knobs and "brand_new" in sp.fixed
    sp2 = S.Mxfp6Space(G, auto_unlisted=True)
    assert sp2.knobs["brand_new"] == (False, True)
    g = sp2.genome_of({"brand_new": True})
    assert sp2.spec_of(g).brand_new is True


def test_space_warm_base_and_overrides():
    sp = S.Mxfp6Space(G, base={"setprio": 1})
    assert sp.spec_of(sp.base_genome).setprio == 1
    assert sp.overrides(sp.base_genome) == {"setprio": 1}
    g = sp.genome_of({"setprio": 0, "unroll": 2})
    assert sp.overrides(g) == {"unroll": 2}
    assert sp.diff_vs_base(g) == {"setprio": 0, "unroll": 2}
    with pytest.raises(ValueError):
        S.Mxfp6Space(G, base={"no_such_field": 1})


def test_legality_prefilter():
    base = G.Mxfp6GemmSpec()
    assert S.legal(G, base) == (True, "")
    big = replace(base, stages=4, tile_m=256, tile_n=256)
    if G.lds_bytes(big) > S.LDS_LIMIT:
        ok, why = S.legal(G, big)
        assert not ok and ("LDS" in why or "160" in why)
    # every config the space admits is valid for the kernel's own validator
    import random

    sp = S.Mxfp6Space(G)
    rng = random.Random(3)
    for _ in range(200):
        sp.space.valid(sp.space.random_genome(rng))
    assert sp.rejects  # the pre-filter refused uniform draws and recorded why
    seeds = sp.space.stratified(rng, 24)  # repaired space-filling draws: all admitted
    assert len(seeds) >= 12  # the structural sampler keeps space-filling draws legal
    for g in seeds:
        ok, why = G.validate_spec(sp.spec_of(g))
        assert ok, why
        assert G.lds_bytes(sp.spec_of(g)) <= S.LDS_LIMIT


def test_register_estimate():
    base = G.Mxfp6GemmSpec()
    jit = S.reg_estimate(G, replace(base, frag_mode="jit"))
    pre = S.reg_estimate(G, replace(base, frag_mode="prefetch"))
    assert pre["arch"] > jit["arch"]
    assert pre["ok"] and jit["ok"]
    two = S.reg_estimate(G, replace(base, waves_m=4, waves_n=2))
    assert two["waves_per_simd"] == 2 and two["budget"] == 256


# ----------------------------------------------------------------- compile gate + ISA


def _meta(spills=0, scratch=0, loop=None, ok=True):
    return {"ok": ok, "error": None if ok else "boom", "notes": {"vgpr_spill_count": spills, "sgpr_spill_count": 0,
            "private_segment_fixed_size": scratch}, "loop": loop if loop is not None else
            {"mfma": 192, "vmcnt0": 0, "scratch": 0}}


def test_compile_gate():
    spec = G.Mxfp6GemmSpec()
    per = G.mfmas_per_tile(spec)
    assert T.compile_gate(_meta(), spec, G, "strict")[0] == T.STATUS_OK
    assert T.compile_gate(_meta(ok=False), spec, G, "strict")[0] == T.STATUS_COMPILE
    assert T.compile_gate(_meta(spills=4, scratch=16), spec, G, "strict")[0] == T.STATUS_SPILL
    assert T.compile_gate(_meta(spills=4, scratch=16), spec, G, "loop")[0] == T.STATUS_OK
    loop_scr = {"mfma": 192, "vmcnt0": 0, "scratch": 3}
    assert T.compile_gate(_meta(loop=loop_scr), spec, G, "loop")[0] == T.STATUS_SPILL
    assert T.compile_gate(_meta(loop={"mfma": 0, "vmcnt0": 0, "scratch": 0}), spec, G, "strict")[0] == T.STATUS_COMPILE
    # one vmcnt(0) per tile is the acquire; more are LLVM drains
    tiles = 3
    ok_loop = {"mfma": per * tiles, "vmcnt0": tiles, "scratch": 0}
    bad_loop = {"mfma": per * tiles, "vmcnt0": tiles + 1, "scratch": 0}
    assert T.compile_gate(_meta(loop=ok_loop), spec, G, "strict")[0] == T.STATUS_OK
    assert T.compile_gate(_meta(loop=bad_loop), spec, G, "strict")[0] == T.STATUS_SPILL
    assert T.compile_gate(_meta(loop=bad_loop), spec, G, "off")[0] == T.STATUS_OK
    # no objdump: the ISA part is skipped (recorded), the resource part still applies
    assert T.compile_gate(_meta(loop={"error": "no llvm-objdump"}), spec, G, "strict") == (T.STATUS_OK,
                                                                                          "no ISA metrics (no llvm-objdump)")


DIS = """
0000000000001000 <KNAME>:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0                        // 000000001000: C0060002 00000000
\tv_mov_b32 v1, 0                                           // 000000001008: 7E020280
\ts_waitcnt vmcnt(0) lgkmcnt(0)                             // 00000000100C: BF8C0070
\ts_barrier                                                 // 000000001010: BF8A0000
\tv_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:7], v[8:13], a[0:3] // 000000001014: D3AD0000 04021102
\tbuffer_load_dwordx4 v1, s[8:11], 0 offen lds             // 00000000101C: E05C1000 80020001
\tds_read_b128 v[2:5], v20                                  // 000000001024: D9FE0000 02000014
\tscratch_load_dword v9, off, s0                            // 00000000102C: DC500000 097D0000
\ts_cbranch_scc1 65530                                      // 000000001034: BF85FFF6 <KNAME+0xc>
\ts_endpgm                                                  // 000000001038: BF810000
"""


def test_loop_stats_and_isa_digest():
    st = T.loop_stats(DIS)
    assert st["mfma"] == 1 and st["vmcnt0"] == 1 and st["s_barrier"] == 1 and st["dtl"] == 1
    assert st["ds_read"] == 1 and st["scratch"] == 1 and st["kernel_scratch"] == 1
    assert st["instr"] == 7  # s_waitcnt .. s_cbranch (the loop starts at +0xc)
    notes = {"vgpr_count": 128}
    d0 = T.isa_digest(DIS, notes, {"grid": [256, 1, 1]}, "KNAME")
    moved = DIS.replace("00000000", "00000ABC").replace("KNAME", "OTHER_NAME_kdeadbeef")
    assert T.isa_digest(moved, notes, {"grid": [256, 1, 1]}, "OTHER_NAME_kdeadbeef") == d0
    assert T.isa_digest(DIS.replace("v_mov_b32 v1, 0", "v_mov_b32 v1, 1"), notes, {"grid": [256, 1, 1]}, "KNAME") != d0
    assert T.isa_digest(DIS, {"vgpr_count": 129}, {"grid": [256, 1, 1]}, "KNAME") != d0
    assert T.isa_digest(DIS, notes, {"grid": [128, 1, 1]}, "KNAME") != d0
    assert "error" in T.loop_stats("no loop here\n")


# ----------------------------------------------------------------- aiter arm


def _aiter_pack_ref(codes, scales, rows, k):
    """Literal per-slot port of aiter's pack_big_torch / pack_scale_torch (padK = 2)."""
    nt, nk = rows // 256, k // 128
    op = np.zeros((nt, nk + 2, 24576), np.uint8)
    sc = np.full((nt, nk + 2, 1024), 127, np.uint8)
    for t in range(nt):
        for s in range(nk):
            for o in range(1024):
                rb, lane = o // 64, o % 64
                r16, kg = lane % 16, lane // 16
                row, col = 256 * t + 16 * rb + r16, 128 * s + 32 * kg
                b = codes[row, col * 3 // 4 : col * 3 // 4 + 24]
                op[t, s, 16 * o : 16 * o + 16] = b[:16]
                op[t, s, 16384 + 8 * o : 16384 + 8 * o + 8] = b[16:]
                su, kq, rr, sub = o // 512, (o % 512) // 128, (o % 128) // 8, o % 8
                sc[t, s, o] = scales[256 * t + 128 * su + 16 * sub + rr, 4 * s + kq]
    return op.reshape(-1), sc.reshape(-1)


def test_aiter_packing_matches_the_literal_port():
    rng = np.random.default_rng(0)
    rows, k = 512, 384
    codes = rng.integers(0, 256, size=(rows, k * 3 // 4), dtype=np.uint8)
    scales = rng.integers(100, 140, size=(rows, k // 32), dtype=np.uint8)
    ref_op, ref_sc = _aiter_pack_ref(codes, scales, rows, k)
    op = T.aiter_pack_operand(codes, rows, k)
    sc = T.aiter_pack_scales(scales, rows, k)
    assert np.array_equal(op, ref_op)
    assert np.array_equal(sc, ref_sc)
    nk = k // 128
    tiles = op.reshape(rows // 256, nk + 2, 24576)
    assert not tiles[:, nk:].any()  # zero guard tiles
    assert (sc.reshape(rows // 256, nk + 2, 1024)[:, nk:] == 127).all()


def test_aiter_kernargs_abi():
    kb = T.aiter_kernargs(0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 4096, 2048, 65536)
    assert len(kb) == 0x180
    import struct

    q = lambda off: struct.unpack_from("<Q", kb, off)[0]  # noqa: E731
    i = lambda off: struct.unpack_from("<I", kb, off)[0]  # noqa: E731
    assert (q(0x0), q(0x10), q(0x20), q(0x30)) == (0x1000, 0, 0x2000, 0x3000)
    assert struct.unpack_from("<f", kb, 0x40)[0] == 1.0 and struct.unpack_from("<f", kb, 0x50)[0] == 0.0
    assert (i(0x60), i(0xE0), i(0xF0), i(0x100)) == (2048, 4096, 2048, 65536)
    assert (q(0x110), q(0x120), i(0x130), i(0x150), i(0x170)) == (0x4000, 0x5000, 2048, 2048, 0)


# ----------------------------------------------------------------- numeric tightness


def _bf16_rne_exact(x: float) -> int:
    """Reference: the bf16 bits nearest to ``x`` (ties to even), by exact rational comparison."""
    import struct
    from fractions import Fraction

    fx = Fraction(x)
    base = (int(np.array(np.float32(x)).view(np.uint32)) >> 16) & 0xFFFF
    best = None
    for b in range(max(0, base - 2), min(0xFFFF, base + 3) + 1):
        if (b & 0x7F80) == 0x7F80:
            continue
        v = Fraction(struct.unpack("<f", struct.pack("<I", b << 16))[0])
        if v != 0 and (v < 0) != (fx < 0):
            continue
        key = (abs(v - fx), b & 1)  # nearest, then even
        if best is None or key < best[0]:
            best = (key, b)
    return best[1]


def test_f64_to_bf16_rne_is_one_step_rounding():
    rng = np.random.default_rng(5)
    xs = list(rng.normal(0, 1e3, 3000)) + list(rng.normal(0, 1e-3, 1000))
    # exact ties of the 8-bit significand and values just above them (where fp64 -> fp32 -> bf16 double rounding
    # goes wrong)
    for m in range(256, 512, 7):
        for e in (-3, 0, 5):
            xs.append(float(np.ldexp(m + 0.5, e - 8)))
            xs.append(float(np.ldexp(m + 0.5 + 2.0**-30, e - 8)))
            xs.append(-float(np.ldexp(m + 0.5, e - 8)))
    got = T.f64_to_bf16_rne(np.array(xs))
    want = np.array([_bf16_rne_exact(x) for x in xs], dtype=np.uint16)
    assert np.array_equal(got, want)


def test_ulp_stats_and_bitwise_compare():
    ref = np.array([1.0, 2.0, -3.0, 1000.0, 0.0])
    exact = T.f64_to_bf16_rne(ref)
    out = exact.copy()
    out[1] += 1  # one ulp
    out[2] += 3  # three ulps (negative: magnitude grows)
    st = T.ulp_stats(out, ref)
    assert st["n"] == 5 and st["max_ulp"] == 3
    assert st["exact_frac"] == pytest.approx(3 / 5) and st["le1_frac"] == pytest.approx(4 / 5)
    assert st["hist"] == {"0": 3, "1": 1, "2": 0, "3-7": 1, ">=8": 0}
    assert T.ulp_stats(np.array([0x8000], np.uint16), np.array([0.0]))["max_ulp"] == 0  # -0 == +0
    c = T.bitwise_compare(out, exact)
    assert c["differ"] == 2 and not c["identical"] and c["max_ulp"] == 3
    assert T.bitwise_compare(exact, exact)["identical"]


# ----------------------------------------------------------------- statistics + ledger


def test_window_median_attributes_power_to_an_arm():
    trace = [(0.000, 2100.0, 300.0), (0.005, 2200.0, 900.0), (0.010, 2300.0, 1000.0), (0.015, None, 1100.0),
             (0.020, 2400.0, 400.0)]
    w = T.window_median(trace, 0.004, 0.016)
    assert w == {"sclk": 2250.0, "watts": 1000.0, "n": 3}
    assert T.window_median(trace, 1.0, 2.0)["watts"] is None


def test_clock_probe_arithmetic():
    # 2 CTAs: 2400 shader cycles over 100 realtime ticks (1 us) -> 2400 MHz; 1500 over 100 -> 1500 MHz
    rec = np.array([[0, 2400, 0, 100, 0], [10, 1510, 5, 105, 0]], dtype=np.int64)
    c = T.clock_mhz(rec, 2)
    assert c["mhz_med"] == pytest.approx(1950.0) and c["mhz_min"] == pytest.approx(1500.0) and c["n"] == 2
    assert c["us_med"] == pytest.approx(1.0)


def test_win_rule_and_bootstrap():
    ctl = [[1.000, 1.003, 0.998, 1.001, 0.999, 1.002]]
    assert T.win_rule([0.98, 0.985, 0.979, 0.99, 0.981, 0.983], ctl) == "WIN"
    assert T.win_rule([1.02, 1.03, 1.025, 1.01, 1.02, 1.04], ctl) == "LOSS"
    assert T.win_rule([0.999, 1.001, 0.997, 1.0, 1.002, 0.998], ctl) == "TIE"
    num = [1.10, 1.12, 1.09, 1.11, 1.10, 1.13, 1.08, 1.10]
    den = [1.0] * 8
    ci = T.bootstrap_ci(num, den, seed=1)
    assert ci["lo"] <= ci["median"] <= ci["hi"] and ci["n"] == 8
    assert 1.08 <= ci["lo"] and ci["hi"] <= 1.13
    assert T.bootstrap_ci(num, den, seed=1) == ci  # deterministic
    assert T.bootstrap_ci([], [])["median"] is None
    assert T.noise_sigma([0.0] * 10) == 0.0


def test_ledger():
    d = {"a": 0, "b": 0, "c": 0}
    recs = [
        {"type": "eval", "status": "ok", "over": {"a": 1}, "ref_over": {}, "value": 0.97, "vs_best": 0.97, "best_over": {}},
        {"type": "eval", "status": "ok", "over": {"b": 1}, "ref_over": {}, "value": 1.02, "vs_best": 1.02, "best_over": {}},
        {"type": "eval", "status": "ok", "over": {"c": 1}, "ref_over": {}, "value": 1.001, "vs_best": 1.001, "best_over": {}},
        {"type": "eval", "status": "ok", "over": {"a": 1, "b": 1}, "ref_over": {}, "value": 0.94, "vs_best": 0.94,
         "best_over": {}},
        {"type": "eval", "status": "spill", "over": {"c": 2}, "ref_over": {}, "value": None},
        {"type": "eval", "status": "ok", "over": {"b": 2}, "ref_over": {}, "value": 1.0, "isa_dup": "R", "ref_key": "R"},
        {"type": "batch"},
    ]
    led = T.build_ledger(recs, d, 0.004)
    v = {r["lever"]: r["verdict"] for r in led["solo_vs_ref"]}
    assert v == {"a=1": "win", "b=1": "loss", "c=1": "neutral"}
    (pair,) = led["pairs"]
    assert pair["pair"] == "a=1|b=1"
    assert pair["interaction"] == pytest.approx(0.06 - 0.03 - (-0.02))
    assert led["failures"] == {"c=2": {"spill": 1}}
    assert led["noop"] == {"b=2": "same ISA as the warm start"}  # ISA-identical: a no-op, not a 0-gain measurement
    txt = T.ledger_text(led)
    assert "solo vs ref" in txt and "no-op levers" in txt


# ----------------------------------------------------------------- supervisor (synthetic backend)


def _env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[T.FAKE_ENV] = "1"
    env.update(extra)
    return env


def _run(args, env, timeout=600):
    p = subprocess.run([sys.executable, "-m", T.MODULE, *args], capture_output=True, text=True, env=env, timeout=timeout)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


def test_supervisor_end_to_end_with_faults_and_resume(tmp_path):
    env = _env(MXFP6_TUNE_FAKE_HANG="setprio=3", MXFP6_TUNE_FAKE_CRASH="tile_lgkm0=True",
               MXFP6_TUNE_FAKE_HIPERR="peel_first=True")
    common = ["--cache", str(tmp_path / "cache"), "--jobs", "2", "--batch", "6", "--rounds", "6", "--hang-s", "3",
              "--members", "hill,gp,de,hyperband", "--reserve-s", "5", "--final-rounds", "6", "--sustained-rounds", "1",
              "--aiter-co", "fake.co", "--max-restarts", "200"]
    # the faulting configs enter as warm starts, so every injected fault is exercised deterministically
    warm = tmp_path / "warm.json"
    warm.write_text(json.dumps({"top": [{"over": {"setprio": 3}}, {"over": {"tile_lgkm0": True}},
                                        {"over": {"peel_first": True}}, {"over": {"setprio": 1}}]}))
    out1 = tmp_path / "r1"
    log = _run(["run", "--out", str(out1), "--budget", "18", "--minutes", "6", "--warm-from", str(warm), *common], env)
    summ = json.loads((out1 / "summary.json").read_text())
    assert summ["n_gpu"] >= 18 and summ["stop"] == "budget"
    assert summ["gate_mode"] == "strict"
    assert "scale_slot" not in summ["space"]["knobs"]  # the no-op prepass held the synthetic no-op knob
    assert "serpentine" in summ["space"]["knobs"]  # ... but kept the gated one (live under m_outer=False)
    recs = [json.loads(x) for x in (out1 / "evals.jsonl").read_text().splitlines()]
    pre = [r for r in recs if r["type"] == "noop_prepass"]
    assert pre and "serpentine" in pre[0]["gated"]
    sess = [r for r in recs if r["type"] == "session"]
    assert sess and all(v == "caught" for v in sess[0]["mutants"].values())
    evals = [r for r in recs if r["type"] == "eval"]
    assert evals and all("status" in r for r in evals)
    statuses = {r["status"] for r in evals}
    assert "ok" in statuses
    # injected faults are blocklisted, never timed
    for r in evals:
        if r["over"].get("setprio") == 3:
            assert r["status"] in ("timeout", "run_error", "spill"), r  # never timed (gate may come first)
        if r["over"].get("tile_lgkm0") is True:
            assert r["status"] in ("timeout", "run_error", "spill"), r
    assert {"timeout", "run_error"} <= statuses
    # a sticky HIP fault is pinned on its culprit only: every run_error / timeout matches an injected rule
    for r in evals:
        if r["status"] in ("run_error", "timeout"):
            o = r["over"]
            assert o.get("setprio") == 3 or o.get("tile_lgkm0") is True or o.get("peel_first") is True, r
    # stages=4 spills in the synthetic compiler: gated before any GPU time
    for r in evals:
        if r["over"].get("stages") == 4:
            assert r["status"] in ("spill", "invalid")
    fin = summ["finalists"]
    assert fin["rows"] and any(r["arm"] == "ref" for r in fin["rows"])
    assert "aiter" in fin and fin["aiter"]["best_vs_aiter_peak"]["median"] > 0
    assert (out1 / "ledger.txt").exists() and summ["top"]
    # resume: same kernel + space -> the state continues (evaluations are not repeated)
    out2 = tmp_path / "r2"
    _run(["run", "--out", str(out2), "--budget", "26", "--minutes", "6", "--resume", str(out1 / "ckpt.json"), *common], env)
    s2 = json.loads((out2 / "summary.json").read_text())
    assert s2["n_gpu"] >= 26
    keys1 = {r["key"] for r in evals if r["status"] == "ok" and not r.get("cached")}
    e2 = [json.loads(x) for x in (out2 / "evals.jsonl").read_text().splitlines()]
    timed2 = {r["key"] for r in e2 if r.get("type") == "eval" and r.get("timing")}
    assert not (keys1 & timed2), "a resumed run re-measured configs of the checkpoint"


def test_backend_axis_and_rust_retests(tmp_path):
    """``--backends llvm,rust``: the compiler is a search axis (pseudo knob, never a spec field) and the knobs that
    lost only to LLVM register allocation are re-tested under the direct compiler first (warm starts)."""
    sp = S.Mxfp6Space(G, pseudo={"backend": ("llvm", "rust")})
    assert sp.knobs["backend"] == ("llvm", "rust")
    g = sp.genome_of({"backend": "rust", "setprio": 1})
    assert sp.overrides(g) == {"backend": "rust", "setprio": 1}
    assert sp.spec_of(g) == replace(G.Mxfp6GemmSpec(), setprio=1)  # the pseudo knob never reaches the spec
    env = _env()
    out = tmp_path / "rb"
    _run(["run", "--out", str(out), "--cache", str(tmp_path / "cache"), "--jobs", "2", "--batch", "6", "--rounds", "4",
          "--members", "hill,gp", "--budget", "10", "--minutes", "4", "--reserve-s", "5", "--final-rounds", "4",
          "--sustained-rounds", "1", "--backends", "llvm,rust"], env)
    recs = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines()]
    ev = [r for r in recs if r["type"] == "eval"]
    rust = [r for r in ev if r["over"].get("backend") == "rust"]
    assert rust and any(str(r.get("origin", "")).startswith("warm:rust-retest") for r in rust)
    assert all(r["compile"].get("ok") is not False for r in rust if r["status"] == "ok")
    summ = json.loads((out / "summary.json").read_text())
    assert "backend" in summ["space"]["knobs"]


def test_per_lever_axes_and_preset(tmp_path):
    # switches apply on top of the base set; levers that are not axes keep their base state (regression: rounds 14-19)
    assert T.levers_of({"rl_m0cse": False, "rl_dmafree": True}, "scalesep,dsrebase,m0cse") == "scalesep,dsrebase,dmafree"
    assert T.levers_of({"rl_scalesep": True}, "") == "scalesep"
    assert T.levers_of({"rust_levers": "a+b"}, "x") == "a,b" and T.levers_of({}, "a,b") == "a,b"
    assert T.spec_over({"backend": "rust", "rl_m0cse": True, "unroll": 2}) == {"unroll": 2}
    out = tmp_path / "rl"
    preset = "best" if "best" in getattr(G, "PRESETS", {}) else ""
    _run(["run", "--out", str(out), "--cache", str(tmp_path / "cache"), "--jobs", "2", "--batch", "6", "--rounds", "4",
          "--members", "hill,gp", "--budget", "8", "--minutes", "4", "--reserve-s", "5", "--final-rounds", "4",
          "--sustained-rounds", "1", "--backends", "rust", "--rust-levers", "scalesep+dsrebase+m0cse",
          "--rust-lever-axes", "m0cse+dmafree", "--noop-prepass", "0"] + (["--preset", preset] if preset else []), _env())
    summ = json.loads((out / "summary.json").read_text())
    knobs = summ["space"]["knobs"]
    assert knobs["rl_m0cse"] == [True, False] and knobs["rl_dmafree"] == [False, True]
    if preset:
        got = replace(G.Mxfp6GemmSpec(), **T.spec_over(summ["ref_over"]))
        assert all(getattr(got, k) == v for k, v in G.PRESETS[preset].items())  # presets may restate defaults
    assert summ["ref_over"]["backend"] == "rust"


def test_unknown_levers_are_refused(tmp_path):
    exe = tmp_path / "target_compile"
    exe.write_bytes(bytes([127]) + b"ELF..." + b"|".join([b"scalesep", b"dsrebase", b"m0fill"]))
    assert T.unknown_levers(str(exe), "scalesep,dsrebase") == []
    assert T.unknown_levers(str(exe), "scalesep+m0share,dmamanualringw") == ["m0share", "dmamanualringw"]
    # exact list (--list-levers) wins over the substring fallback: "sep" is a substring of "scalesep" but no lever
    T._EXE_LEVERS[str(exe)] = frozenset({"scalesep", "dsrebase"})
    assert T.unknown_levers(str(exe), "scalesep,sep") == ["sep"]


def test_confirm_reports_paired_cis(tmp_path):
    env = _env()
    summ = {"ref_over": {"backend": "llvm"}, "top": [{"over": {"setprio": 1, "backend": "rust",
                                                                "rust_levers": "scalesep+dsrebase"}}, {"over": {"unroll": 2}}]}
    sp = tmp_path / "summary.json"
    sp.write_text(json.dumps(summ))
    out = tmp_path / "c"
    _run(["confirm", "--summary", str(sp), "--out", str(out), "--cache", str(tmp_path / "cache"), "--jobs", "2",
          "--blocks", "2", "--rounds", "5", "--sustained-rounds", "2", "--aiter-co", "fake.co", "--gpu-tag", "g0"], env)
    rep = json.loads((out / "confirm.json").read_text())
    arms = {r["arm"] for r in rep["rows"]}
    assert {"ref", "best", "best_tw", "aiter", "aiter_tw"} <= arms
    assert any(a.startswith("probe_") for a in arms) == ("probe" in S.default_config(G))
    for k in ("best_vs_aiter", "best_vs_ref", "noise_best_twin", "noise_aiter_twin"):
        c = rep["peak_pairs"][k]
        assert c["n"] == 10 and c["lo"] <= c["median"] <= c["hi"]
    assert any(k.startswith("best|aiter|") for k in rep["compare"])  # rocKE vs aiter outputs compared per dist
    assert {k.split("|")[2] for k in rep["compare"]} == {"int", "gauss", "uniform"}
    txt = _run(["report", str(out / "confirm.json")], env)
    assert "best_vs_aiter" in txt and "bitwise best vs aiter" in txt
    cs = rep["coldstart"]["summary"]
    assert {"best", "aiter"} <= set(cs) and len(cs["best"]["launch_ratio"]) == 10
    assert cs["best"]["launch_ratio"][0] > cs["best"]["launch_ratio"][-1]  # the synthetic first launches run slow
    assert rep["rows"][0]["warm_tf"] == rep["rows"][0]["sustained_tf"]  # new names alias the old keys


LOOKBACK = Path(__file__).resolve().parents[6] / "mxfp6-gemm-roofline" / "search" / "vendor" / "signed_lookback.py"


@pytest.mark.skipif(not LOOKBACK.exists(), reason="vendored signed-lookback module not present (workstation only)")
def test_vendored_lookback_member_builds():
    import argparse

    sp = S.Mxfp6Space(G)
    args = argparse.Namespace(members="lookback,hill", lookback_module=str(LOOKBACK), seed=1, min_gain=0.003, batch=4,
                              budget=50)
    strat = T.build_strategy(sp, args, [({}, "kernel-agent")])
    assert [m.name for m in strat.members] == ["lookback", "hill"]
    first = strat.ask(1)
    assert first and first[0].origin == "warm:kernel-agent"
