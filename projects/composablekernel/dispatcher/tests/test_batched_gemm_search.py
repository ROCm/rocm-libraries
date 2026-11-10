#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Adaptive kernel search of the batched GEMM sweep driver
(batched_gemm_full_benchmark.py --search-config).

The batched driver binds the GEMM adapter (gemm_search.GemmSearchAdapter) to
its own build function, worker and problem fields (batch_count, M, N, K), so
these tests repeat the GEMM search tests' checks on batched problems:

- the driver: a run without the search options is unchanged; invalid
  combinations exit 2; a search measures only verified single-kernel calls of
  the pairs a full run measures (the --list-pairs listing, and the pairs a
  plain run measures when every kernel builds), writes BxMxNxK winners the
  driver's --pairs accepts, honors --pairs, keeps each problem on one device
  for its confirmation and reads a BxMxNxK ranked prior. The test_gemm_search_cli
  stubs run here on the batched driver (``cli.drv``);
- the adapter: requests on two problems that differ only in batch_count are
  measured and verified by the real worker in-process, each with its own
  host cache entry; a row of another batch count is not the request; the host
  memory and time estimates scale with the batch;
- the worker: a batched search call gives the same outcome without the host
  cache, on its first call and on a call that reads the entry, whose key holds
  the batch count and the packed strides; a kernel wrong in its last batch
  only fails verification on every path;
- the shipped batched search settings are valid and each search config covers
  its arch's default batched config.

No GPU or compiler needed. Run: python3 -m pytest -q tests/test_batched_gemm_search.py
"""

import contextlib
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_search as AS  # noqa: E402
import batched_gemm_full_benchmark as bdrv  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
import gemm_search as GS  # noqa: E402
import host_cache as HC  # noqa: E402
import run_one_gemm_kernel as worker  # noqa: E402
import test_gemm_driver_verify as V  # noqa: E402
import test_gemm_search_adapter as ad  # noqa: E402
import test_gemm_search_cli as cli  # noqa: E402
from batched_gemm_utils import GpuBatchedGemmRunner, packed_strides  # noqa: E402

DIMS = bdrv.DIMS
# The first two differ only in their batch count: two problems, two keys.
PROBLEMS = [
    {"batch_count": 2, "M": 512, "N": 512, "K": 512},
    {"batch_count": 4, "M": 512, "N": 512, "K": 512},
    {"batch_count": 3, "M": 1024, "N": 512, "K": 256},
]
MISALIGNED = {"batch_count": 3, "M": 1000, "N": 520, "K": 250}
KEYS = [drv.problem_key(p, DIMS) for p in PROBLEMS]
CSV = ["kernel", "problem_idx", "batch_count", "M", "N", "K", "device", "latency_ms",
       "tflops", "non_zero", "max_rel", "verified"]
CONFIGS = GEMM_TE_DIR / "batched_gemm" / "configs"  # sweeps only
SETTINGS = GEMM_TE_DIR / "batched_gemm" / "search"


@pytest.fixture(autouse=True)
def _batched(tmp_path, monkeypatch):
    """The test_gemm_search_cli stubs drive the batched driver; each test's
    search runs take the host lock in its own temp dir, see 1 TiB of host
    memory, and its in-process workers leave the process's oom_score_adj."""
    monkeypatch.setattr(cli, "drv", bdrv)
    monkeypatch.setattr(AS, "HOST_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(AS, "host_memory", lambda: 1 << 40)
    monkeypatch.setattr(worker, "OOM_SCORE_ADJ", str(tmp_path / "oom_score_adj"))


def _key(prob):
    return drv.problem_key(prob, DIMS)


def _main(monkeypatch, capsys, tmp_path, *argv, problems=PROBLEMS, **kw):
    return cli._main(monkeypatch, capsys, tmp_path, *argv, problems=problems, **kw)


def _search(monkeypatch, capsys, tmp_path, *argv, problems=PROBLEMS, **kw):
    return cli._search(monkeypatch, capsys, tmp_path, *argv, problems=problems, **kw)


def _assert_worker(calls):
    """Every call ran the batched worker with the driver's GEMM_PYPATH."""
    want = (GEMM_TE_DIR / "run_one_batched_gemm_kernel.py", os.pathsep.join(
        [str(DISPATCHER_DIR / "python"), str(GEMM_TE_DIR)]))
    assert calls and all(c["worker"] == want for c in calls)


# ------------------------------------------------------------------ driver


def test_run_without_search_is_unchanged_and_loads_no_search_module(
    monkeypatch, capsys, tmp_path
):
    for name in list(sys.modules):
        if name.split(".")[0] in cli.SEARCH_MODULES:
            monkeypatch.delitem(sys.modules, name)
    rc, cap, built, calls = _main(monkeypatch, capsys, tmp_path, "--verify")
    assert rc == 0 and built, cap.out
    # The workers' fp8/bf8 host format follows the arch the kernels were built for.
    assert all(c["extra"] == {"arch": "gfx950"} and c["batch_size"] == 20 for c in calls)
    assert {tuple(c["prob"]) for c in calls} == {DIMS}
    _assert_worker(calls)
    assert not [n for n in sys.modules if n.split(".")[0] in cli.SEARCH_MODULES]
    with open(tmp_path / "out.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == CSV


@pytest.mark.parametrize("argv, message", [
    (["--search-config", "{S}"], "go together"),
    (["--search-output-dir", "{O}"], "go together"),
    (["--search-config", "{S}", "--search-output-dir", "{O}", "--list-pairs", "x.json"],
     "does not take --list-pairs"),
    (["--search-config", "{BAD}", "--search-output-dir", "{O}"], "unknown key"),
    (["--pairs", "p.json", "--list-pairs", "x.json"], "mutually exclusive"),
    (["--pairs", "{CSV}"], "--csv and --pairs name the same file"),
])
def test_invalid_search_options_exit_2(monkeypatch, capsys, tmp_path, argv, message):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "strategy": "tpe",
                               "trials_per_problem": 2, "new_builds_per_run": 4,
                               "bogus": 1}))
    subs = {"{S}": cli._settings(tmp_path), "{O}": str(tmp_path / "new"),
            "{BAD}": str(bad), "{CSV}": str(tmp_path / "out.csv")}
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, capsys, tmp_path, *[subs.get(a, a) for a in argv])
    assert e.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "new").exists() and not any((tmp_path / "libs").iterdir())



def test_help_shows_the_batched_key_form(monkeypatch, capsys):
    # The shared pair and search options, with the batched key form and
    # without the GEMM driver's gemm_universal note.
    monkeypatch.setattr(sys, "argv", ["batched_gemm_full_benchmark.py", "--help"])
    with pytest.raises(SystemExit) as e:
        bdrv.main()
    out = " ".join(capsys.readouterr().out.split())
    assert e.value.code == 0
    for opt in ("--list-pairs", "--pairs", "--search-config", "--search-output-dir"):
        assert opt in out
    assert "'BxMxNxK'" in out and "'MxNxK'" not in out and "gemm_universal" not in out

def test_search_writes_winners_the_driver_accepts(monkeypatch, capsys, tmp_path):
    eligible = cli._listing(monkeypatch, capsys, tmp_path, problems=PROBLEMS)
    assert set(eligible) == set(KEYS)
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path)
    assert rc == 0, cap.out
    cache = calls[0]["extra"]["host_cache"]
    assert calls and all(
        len(c["names"]) == 1 and c["verify"] and c["batch_size"] == 1
        and c["extra"] == {"check_identity": True, "search_host_data": True,
                           "host_cache": cache} for c in calls)
    assert not Path(cache["dir"]).exists()
    _assert_worker(calls)
    assert len(set(built)) == len(built) <= 12
    winners = drv.load_pair_list(out / "winners.json")
    assert set(winners) == set(KEYS)
    for key, prob in zip(KEYS, PROBLEMS):
        (name,) = winners[key]
        assert name in eligible[key]
        tried = {c["names"][0] for c in calls if c["prob"] == prob}
        assert len(tried) <= 5 and cli._ms(name) == min(map(cli._ms, tried))
    summary, run = cli._read(out, "summary.json"), cli._read(out, "run.json")
    assert (summary["status"], summary["exit_code"]) == ("complete", 0)
    assert summary["counts"]["confirm_calls"] == 2 * 2 * len(KEYS)
    assert (run["context"]["arch"], run["context"]["variant"]) == ("gfx950", "batched")
    assert set(run["problems"]) == set(KEYS)
    # The host costs of a batched problem scale with its batch count.
    measurement = run["context"]["measurement"]
    assert "2*batch_count*M*N*K" in measurement["timeout_s"]
    assert "batch_count*(M+N)*K" in measurement["host_memory"]["computing_bytes"]
    assert "SEARCH COMPLETE: 3/3 problems" in cap.out
    # The driver re-checks the winners from the file as a --pairs list.
    rc, _, rebuilt, recalls = _main(monkeypatch, capsys, tmp_path, "--pairs",
                                    str(out / "winners.json"), "--verify")
    assert rc == 0 and set(rebuilt) == {n for v in winners.values() for n in v}
    assert {(_key(c["prob"]), n) for c in recalls for n in c["names"]} == {
        (k, winners[k][0]) for k in KEYS}


def test_search_candidates_are_the_pairs_a_full_run_measures(
    monkeypatch, capsys, tmp_path
):
    # The --list-pairs listing is what a plain run measures when every kernel
    # builds, and it is the search's domain, per problem and kernel.
    problems = PROBLEMS[:2] + [MISALIGNED]
    eligible = cli._listing(monkeypatch, capsys, tmp_path, problems=problems)
    rc, _, _, full = _main(monkeypatch, capsys, tmp_path, problems=problems)
    assert rc == 0
    measured = {}
    for c in full:
        measured.setdefault(_key(c["prob"]), []).extend(c["names"])
    assert measured == eligible
    sizes = {k: len(v) for k, v in eligible.items()}
    assert sizes[_key(MISALIGNED)] > sizes[KEYS[0]] == sizes[KEYS[1]]
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         problems=problems, strategy="random",
                                         new_builds_per_run=24)
    assert rc == 0, cap.out
    run = cli._read(out, "run.json")
    assert {k: p["eligible"] for k, p in run["problems"].items()} == sizes
    assert run["context"]["domain_sha256"] == cli._domain_sha256(eligible)
    assert all(c["names"][0] in eligible[_key(c["prob"])] for c in calls)


def test_pairs_restricts_the_search_domain(monkeypatch, capsys, tmp_path):
    eligible = cli._listing(monkeypatch, capsys, tmp_path, problems=PROBLEMS)
    picks = {k: names[:2] for k, names in eligible.items() if k != KEYS[2]}
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(picks))
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path,
                                         "--pairs", str(path))
    assert rc == 0, cap.out
    assert "the search builds only the kernels it proposes" in cap.out
    assert set(built) <= {n for v in picks.values() for n in v}
    run = cli._read(out, "run.json")
    assert list(run["problems"]) == KEYS[:2]
    assert run["context"]["domain_sha256"] == cli._domain_sha256(picks)
    assert {_key(c["prob"]) for c in calls} == set(KEYS[:2])


def test_confirmation_keeps_each_problem_on_one_device(monkeypatch, capsys, tmp_path):
    rc, cap, _, _, out = _search(monkeypatch, capsys, tmp_path, devices=("0", "1"))
    assert rc == 0, cap.out
    devices = {}
    for e in map(json.loads, (out / "trials.jsonl").read_text().splitlines()):
        if e["event"] == "confirm":
            devices.setdefault(e["problem"], set()).add(e["device"])
    assert devices == {k: {"01"[i % 2]} for i, k in enumerate(KEYS)}


def test_ranked_csv_is_the_search_prior(monkeypatch, capsys, tmp_path):
    eligible = cli._listing(monkeypatch, capsys, tmp_path, problems=PROBLEMS)
    rows = [f"{KEYS[1]},{i + 1},{name},{10.0 - i}"
            for i, name in enumerate(eligible[KEYS[1]][:3])]
    (tmp_path / "ranked.csv").write_text(
        "problem,rank,kernel,predicted_tflops\n" + "\n".join(rows) + "\n")
    rc, cap, _, _, out = _search(
        monkeypatch, capsys, tmp_path,
        warm_start={"ranked_csv": "ranked.csv", "use_prior": True})
    assert rc == 0, cap.out
    entries = {k: p["prior_entries"]
               for k, p in cli._read(out, "run.json")["problems"].items()}
    assert entries == {KEYS[0]: 0, KEYS[1]: 3, KEYS[2]: 0}


def test_real_workers_prepare_each_problem_before_its_first_call(
        monkeypatch, capsys, tmp_path):
    # The batched worker script, started by the search as a subprocess with
    # the driver's environment, stores each problem's host data once: three
    # entries, two of them for the same M, N and K.
    rc, cap, _, _, out = _search(monkeypatch, capsys, tmp_path, prepare=True,
                                 devices=("0", "1"))
    assert rc == 0, cap.out
    assert "Host cache: 3 entries" in cap.out
    prepared = [r for p in sorted((out / "rounds").glob("*.json"))
                if not p.name.endswith("-build.json")
                for r in json.loads(p.read_text())["prepared"]]
    assert sorted(r["problem"] for r in prepared) == sorted(KEYS)
    assert all(r["ok"] and r["stored"] and not r["error"] for r in prepared)


def test_search_without_a_winner_exits_1(monkeypatch, capsys, tmp_path):
    rc, cap, built, calls, out = _search(monkeypatch, capsys, tmp_path, bad="*",
                                         strategy="random")
    assert rc == 1 and built and calls
    assert cli._read(out, "winners.json") == {}
    summary = cli._read(out, "summary.json")
    assert summary["counts"]["failed_by_status"]["numeric_fail"] == len(calls)


# ----------------------------------------------------------------- adapter


BP1, BP2 = ({"batch_count": b, **ad.P1} for b in (2, 3))
BK1, BK2 = _key(BP1), _key(BP2)


def _adapter(tmp_path, tc, cache=None):
    hooks = replace(drv.search_hooks(tc.build, DIMS), build_dir=None)
    return GS.GemmSearchAdapter(
        ad._args(), hooks, [BP1, BP2], ad.CFGS, [[ad.A, ad.B, ad.C, ad.D]] * 2,
        ["0"], (Path("worker.py"), {}), tmp_path / "out", {"arch": "gfx950"},
        cache=cache)


@pytest.mark.parametrize("cached", [False, True], ids=["no-cache", "cache"])
def test_batched_requests_are_measured_and_verified_per_problem(
        tmp_path, monkeypatch, cached):
    # The real worker runs each request in-process on the stand-in runner:
    # its rows carry the batch count, the kernel wrong on M 64 fails on both
    # problems, and with the cache each problem has its own entry.
    tc = ad.Toolchain(tmp_path, monkeypatch, lat={ad.A: 2.0, ad.B: 1.5},
                      wrong={(ad.C, 64)})
    monkeypatch.setattr(worker, "GpuBatchedGemmRunner", tc.runner)
    with HC.HostCache(tmp_path) if cached else contextlib.nullcontext() as cache:
        adapter = _adapter(tmp_path, tc, cache)
        assert list(adapter.problems) == [BK1, BK2]
        assert adapter.build_missing(list(adapter.cands.values()))
        reqs = [ad._req(k, n, i) for i, (k, n) in enumerate(
            [(BK1, ad.A), (BK2, ad.A), (BK1, ad.B), (BK2, ad.C)])]
        got = [(o.status, o.latency_ms) for o in adapter.measure_pairs(reqs)]
        if cached:
            entries = [json.loads((p / "meta.json").read_text())["fields"]
                       for p in Path(cache.dir).iterdir() if p.is_dir()]
    assert got == [("ok", 2.0), ("ok", 2.0), ("ok", 1.5), ("numeric_fail", None)]
    assert [it["problem"] for p in tc.payloads for it in p["items"]] == [
        BP1, BP2, BP1, BP2]
    assert all(o["verified"] is (o["kernel"] != ad.C) for o in tc.outputs)
    if cached:
        assert sorted(e["problem"]["batch_count"] for e in entries) == [2, 3]
        assert all(e["op"] == "batched_gemm" and e["strides"] == packed_strides(
            "rcr", 64, 32, 16) for e in entries)


def test_a_row_of_another_batch_count_is_not_the_request():
    req = ad._req(BK1, ad.A, 0)
    row = {"kernel": ad.A, **BP1, "latency_ms": 1.0, "verified": True}
    assert GS.reconcile(req, BP1, [row], []) == ("ok", 1.0, "")
    status, _, diag = GS.reconcile(req, BP1, [{**row, "batch_count": 3}], [])
    assert status == "run_error" and "is not the requested pair" in diag


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_host_costs_scale_with_the_batch(dtype):
    one = {"M": 256, "N": 768, "K": 4096}
    for b in (1, 3, 1024):
        many = {"batch_count": b, **one}
        assert GS.host_entry_bytes(many, dtype) == b * GS.host_entry_bytes(one, dtype)
        for f in (GS.host_read_bytes, lambda p: GS.host_compute_bytes(p, dtype)):
            assert f(many) - GS.HOST_FIXED_BYTES == b * (f(one) - GS.HOST_FIXED_BYTES)
        m, n, k = one["M"], one["N"], one["K"]
        assert GS.verify_allowance_s(many, dtype) == AS.host_allowance_s(
            2 * b * m * n * k, b * (m + n) * k, GS.host_input_rate(dtype))
    assert GS.verify_allowance_s({"batch_count": 1, **one}, dtype) == (
        GS.verify_allowance_s(one, dtype))


def test_ranked_csv_of_batched_problems(tmp_path):
    path = tmp_path / "ranked.csv"
    path.write_text("problem,rank,kernel,predicted_tflops\n4x64x32x16,1,kA,2.0\n")
    assert GS.read_ranked_csv(path, DIMS) == {
        "4x64x32x16": {"kA": 2 * 4 * 64 * 32 * 16 / 2e9}}
    path.write_text("problem,rank,kernel,predicted_tflops\n64x32x16,1,kA,2.0\n")
    with pytest.raises(ValueError, match="expected a BxMxNxK problem"):
        GS.read_ranked_csv(path, DIMS)


# ------------------------------------------------------------------ worker


@pytest.fixture
def batched_worker(monkeypatch, capsys):
    """run(runner, problem, **kw): the worker's JSON line for a batched
    ``runner`` on ``problem`` (fresh in-process caches)."""
    def run(runner, problem=V.BATCHED_PROBLEM, **kw):
        monkeypatch.setattr(worker, "GpuBatchedGemmRunner", lambda **k: runner)
        V.gemm_utils._SWEEP_CACHE.clear()
        worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True, **kw)
        return capsys.readouterr().out.strip()

    return run


@pytest.mark.parametrize("layout", ["rcr", "crr", "rrr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", V.CASES)
def test_host_cache_gives_the_uncached_outcome(batched_worker, tmp_path, dtype,
                                               use_ocp, layout):
    # A batched search call without the cache, its first call with it and a
    # call that reads the entry give the same line; the legacy inputs verify too.
    runner = V._runner(dtype, use_ocp, layout, GpuBatchedGemmRunner)
    legacy = json.loads(batched_worker(runner))
    plain = batched_worker(runner, search_host_data=True)
    with HC.HostCache(tmp_path) as cache:
        miss = batched_worker(runner, host_cache=cache.spec)
        hit = batched_worker(runner, host_cache=cache.spec)
        assert HC.usage(cache.dir)[0] == 1
    assert plain == miss == hit, (plain, hit)
    for out in (legacy, json.loads(hit)):
        assert out["verified"] and out["max_rel"] < V._c_eps(dtype), out



class _LastBatchWrong(V._ExactKernelLib):
    """An exact batched kernel, except that its last batch repeats batch 0's C."""

    def run(self, A_h, B_h, C_h, *dims, **kw):
        out = super().run(A_h, B_h, C_h, *dims, **kw)
        C_h[-1] = C_h[0]
        return out


@pytest.mark.parametrize("block", [None, 1], ids=["one-block", "block-per-batch"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_a_wrong_last_batch_fails_verification(batched_worker, tmp_path, monkeypatch,
                                               dtype, block):
    # Every batch is compared, not only the first: with the legacy inputs, the
    # search's inputs, and a cache miss and hit, compared in one block of rows
    # or one batch at a time.
    if block:
        monkeypatch.setattr(worker, "BLOCK", block)
    runner = V._runner(dtype, None, "rcr", GpuBatchedGemmRunner)
    runner.lib = _LastBatchWrong(dtype, "rcr", None)
    lines = [batched_worker(runner), batched_worker(runner, search_host_data=True)]
    with HC.HostCache(tmp_path) as cache:
        lines += [batched_worker(runner, host_cache=cache.spec) for _ in range(2)]
    for out in map(json.loads, lines):
        assert out["verified"] is False and out["max_rel"] > 0.1, out

def test_host_cache_keys_the_batch_count_and_strides(batched_worker, tmp_path,
                                                     monkeypatch, capsys):
    # One entry per batch count, apart from the single GEMM of the same M, N
    # and K; a batched entry's key also lists its packed strides. A
    # preparation stores the entry a call then reads.
    runner = V._runner("bf16", None, "crr", GpuBatchedGemmRunner)
    problems = [{**V.BATCHED_PROBLEM, "batch_count": b} for b in (3, 2)]
    with HC.HostCache(tmp_path) as cache:
        worker._prepare(problems[0], runner.kernel_name, cache.spec)
        assert json.loads(capsys.readouterr().out) == {"ok": True, "stored": True}
        lines = [batched_worker(runner, p, host_cache=cache.spec) for p in problems]
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **k: V._runner("bf16", None,
                                                                          "crr"))
        worker._run_one(0, "fake.so", V.PROBLEM, runner.kernel_name, verify=True,
                        host_cache=cache.spec)
        assert json.loads(capsys.readouterr().out)["verified"]
        fields = sorted((json.loads((p / "meta.json").read_text())["fields"]
                         for p in cache.dir.iterdir() if p.is_dir()),
                        key=lambda f: f["problem"].get("batch_count", 0))
    assert all(json.loads(line)["verified"] for line in lines)
    m, n, k = (V.PROBLEM[d] for d in "MNK")
    assert [(f["op"], f["problem"].get("batch_count"), f.get("strides"))
            for f in fields] == [("gemm", None, None)] + [
        ("batched_gemm", b, packed_strides("crr", m, n, k)) for b in (2, 3)]


def test_packed_strides_follow_the_layout():
    assert packed_strides("crr", 64, 96, 256) == {
        "stride_A": 64, "batch_stride_A": 64 * 256,
        "stride_B": 96, "batch_stride_B": 256 * 96,
        "stride_C": 96, "batch_stride_C": 64 * 96}


# ---------------------------------------------------------------- settings


def _values(spec):
    """A config field's values (a list, or min/max/step)."""
    if "values" in spec:
        return set(spec["values"])
    return set(range(spec["min"], spec["max"] + 1, spec["step"]))


@pytest.mark.parametrize("arch, default", [
    ("gfx950", "default_config.json"), ("gfx1250", "default_config_gfx1250.json")])
def test_shipped_search_settings_and_configs(arch, default):
    # The settings load as a search run reads them; the search config holds
    # every value of the arch's default batched config (gfx1250: its TDM
    # pipelines too), plus smaller tiles and padding for misaligned problems.
    settings = AS.SearchSettings.from_json(SETTINGS / f"search_settings_{arch}.json")
    assert (settings.strategy, settings.seed_policy) == ("tpe", "shared")
    search = json.loads((CONFIGS / f"search_config_{arch}.json").read_text())
    base = json.loads((CONFIGS / default).read_text())
    for part in ("tile_config", "trait_config"):
        assert search[part].keys() == base[part].keys()
        for name, spec in base[part].items():
            assert _values(spec) <= _values(search[part][name]), (part, name)
    traits = search["trait_config"]
    assert {32} < _values(search["tile_config"]["tile_m"])
    assert all(_values(traits[f"pad_{d}"]) == {False, True} for d in "mnk")
    if arch == "gfx1250":
        assert {"comp_tdm", "comp_tdm_v2"} <= _values(traits["pipeline"])
        assert "tdm" in _values(traits["epilogue"])
