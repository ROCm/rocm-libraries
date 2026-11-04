#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
The GEMM adapter of the adaptive kernel search (tile_engine/ops/gemm/gemm_search.py).

A fake toolchain stands in for the compiler and the GPU: a built library is a
text file holding the kernel name it reports, and the real worker
(run_one_gemm_kernel.main) runs in-process on a runner that reads that name
back. Measurements go through the driver's real benchmark_on_devices and
_run_batch_on_device, so rows, failure records, compact indices, the identity
check and the per-call output files are all exercised as in a real run.

No GPU or compiler needed. Run: python3 -m pytest -q tests/test_gemm_search_adapter.py
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))

import adaptive_search as AS  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
import gemm_search as GS  # noqa: E402
import run_one_gemm_kernel as worker  # noqa: E402
from gemm_utils import GemmKernelConfig  # noqa: E402

BASE = GemmKernelConfig(gfx_arch="gfx950")
CFGS = [
    replace(BASE, tile_m=m, tile_n=n)
    for m, n in ((64, 64), (128, 64), (128, 128), (256, 128))
]
A, B, C, D = (c.name for c in CFGS)
P1, P2 = {"M": 64, "N": 32, "K": 16}, {"M": 32, "N": 64, "K": 16}
K1, K2 = "64x32x16", "32x64x16"


class _Runner:
    """GpuGemmRunner stand-in: an exact kernel, one whose output is wrong for
    the problems ``wrong(problem)`` names, or one returning ``status``."""

    def __init__(self, name, ms, wrong=lambda problem: False, status=0):
        self.kernel_name, self._ms, self._wrong, self._use_ocp = name, ms, wrong, None
        self._status = status

    def run(self, A, B, problem):
        if self._status:
            return SimpleNamespace(success=False, status=self._status)
        out = A @ B + (1.0 if self._wrong(problem) else 0.0)
        return SimpleNamespace(success=True, status=0, time_ms=self._ms, tflops=1.0,
                               output=out)

    def reference(self, A, B):
        return A @ B


class _InProcess:
    """subprocess.Popen stand-in running the real worker main() in-process."""

    returncode = 0

    def __init__(self, tc):
        self.tc = tc

    def communicate(self, input=None, timeout=None):
        self.tc.payloads.append(json.loads(input))
        out, stdin = io.StringIO(), sys.stdin
        sys.stdin = io.TextIOWrapper(io.BytesIO(input))
        try:
            with contextlib.redirect_stdout(out):
                worker.main()
        finally:
            sys.stdin = stdin
        self.tc.outputs.extend(json.loads(ln) for ln in out.getvalue().splitlines())
        return out.getvalue().encode(), b""

    def kill(self):
        pass

    def poll(self):
        return 0


class Toolchain:
    """Fake build + GPU. ``fail``: names whose build fails; ``mislabel``: name
    -> the name its library reports; ``lat``: name -> ms; ``wrong``: names, or
    (name, M) pairs, whose output is wrong (for every problem, or for M)."""

    def __init__(self, tmp_path, monkeypatch, fail=(), mislabel=None, lat=None,
                 wrong=()):
        self.lib_dir = tmp_path / "examples"
        self.lib_dir.mkdir()
        self.fail, self.mislabel = set(fail), dict(mislabel or {})
        self.lat, self.wrong = dict(lat or {}), set(wrong)
        self.builds, self.payloads, self.outputs, self.loaded = [], [], [], []
        monkeypatch.setattr(worker, "GpuGemmRunner", self.runner)
        monkeypatch.setattr(drv, "subprocess", SimpleNamespace(
            Popen=lambda *a, **kw: _InProcess(self), PIPE=subprocess.PIPE,
            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
        for cache in ("_ab_cache", "_ref_cache"):
            if hasattr(worker._run_one, cache):
                monkeypatch.delattr(worker._run_one, cache)

    def build(self, cfgs, verbose=True, max_workers=None, *, errors=None,
              build_timeout=None):
        self.builds.append({"names": [c.name for c in cfgs], "verbose": verbose,
                            "max_workers": max_workers, "build_timeout": build_timeout})
        out = []
        for c in cfgs:
            if c.name in self.fail:
                errors[c.name] = "Compile failed: k.cpp:1:2: error: boom\nmore"
                out.append(None)
                continue
            lib = self.lib_dir / f"lib{c.name}.so"
            lib.write_text(self.mislabel.get(c.name, c.name))
            out.append(lib)
        return out

    def runner(self, lib_path):
        self.loaded.append(Path(lib_path))
        name = Path(lib_path).read_text()
        return _Runner(name, self.lat.get(name, 1.0),
                       lambda prob: bool({name, (name, prob.M)} & self.wrong))


def _args(**kw):
    a = dict(verify=False, verify_tol=0.02, batch_size=20, kernel_timeout=30,
             workers=3, build_timeout=None, dtype="fp16", problems=None, pairs=None,
             max_kernels=0, no_vector_fallback=False, tune_c_vector_width=False,
             search_output_dir=None)
    a.update(kw)
    return argparse.Namespace(**a)


def _hooks(build, measure=drv.benchmark_on_devices):
    return GS.DriverHooks(build=build, measure=measure, record=drv._kernel_record,
                          problem_key=drv.problem_key, load_pairs=drv.load_pair_list,
                          build_reason=drv.build_failure_summary)


def _adapter(tmp_path, build, measure=drv.benchmark_on_devices, problems=(P1, P2),
             eligible=None, devices=("0",), timeout=None, **args):
    eligible = eligible or [[A, B, C, D]] * len(problems)
    return GS.GemmSearchAdapter(
        _args(**args), _hooks(build, measure), list(problems), CFGS, eligible,
        list(devices), (Path("worker.py"), {}), tmp_path / "out", {"arch": "gfx950"},
        **({} if timeout is None else {"build_timeout": timeout}),
    )


def _req(problem, name, n):
    return AS.PairRequest(problem, name, name, f"{problem}:{n}", f"d{n}")


class _Recorder:
    """benchmark_on_devices wrapper recording each call's compact inputs."""

    def __init__(self):
        self.calls = []

    def __call__(self, problems, pairs, built, devices, args, *rest, **kw):
        self.calls.append({"problems": problems, "pairs": pairs,
                           "built": [(cfg.name, Path(p)) for cfg, p in built],
                           "verify": args.verify, "batch_size": args.batch_size,
                           "extra": kw.get("extra")})
        return drv.benchmark_on_devices(problems, pairs, built, devices, args, *rest, **kw)


# ------------------------------------------------------------ reconcile


@pytest.mark.parametrize("value, want", [
    (True, True), ("True", True), (False, False), ("False", False),
    ("", None), (None, None), ("true", None), (1, None), ("1", None),
])
def test_parse_verified_accepts_only_true_and_its_csv_form(value, want):
    assert GS.parse_verified(value) is want


def _row(**kw):
    r = {"kernel": A, "problem_idx": 0, **P1, "device": "0", "latency_ms": 1.5,
         "tflops": 1.0, "non_zero": 1, "max_rel": 0.0, "verified": True}
    r.update(kw)
    return r


def _fail(kind, error="x", kernel=A):
    return {"kernel": kernel, "problem_idx": 0, **P1, "device": "0", "kind": kind,
            "error": error}


@pytest.mark.parametrize("rows, fails, status", [
    ([_row()], [], "ok"),
    ([_row(verified="True")], [], "ok"),
    ([_row(verified="False")], [], "numeric_fail"),
    ([_row(verified=False)], [_fail("mismatch")], "numeric_fail"),
    ([_row(verified="")], [], "invalid_measurement"),
    ([_row(latency_ms=math.nan)], [], "invalid_measurement"),
    ([_row(latency_ms=0.0)], [], "invalid_measurement"),
    ([_row(latency_ms=-1.0)], [], "invalid_measurement"),
    ([_row(latency_ms=True)], [], "invalid_measurement"),
    ([_row(), _row()], [], "invalid_measurement"),
    ([_row(kernel=B)], [], "run_error"),
    ([_row(M=128)], [], "run_error"),
    ([], [_fail("timeout")], "timeout"),
    ([], [_fail("missing")], "run_error"),
    ([], [_fail("unsupported")], "run_error"),
    ([], [_fail("batch-error")], "run_error"),
    ([_row()], [_fail("worker-exit", kernel="")], "run_error"),
    ([_row()], [_fail("bad-output", kernel="")], "invalid_measurement"),
    ([_row(verified=False)], [_fail("mismatch"), _fail("timeout")], "timeout"),
    ([], [_fail("something-new")], "run_error"),
    ([], [], "unattempted"),
])
def test_reconcile_normalizes_rows_and_failures(rows, fails, status):
    got = GS.reconcile(_req(K1, A, 1), P1, rows, fails)
    assert got[0] == status
    assert (got[1] == 1.5) if status == "ok" else got[1] is None


@pytest.mark.parametrize("code, why", [("-2", "no kernel selected"),
                                       ("-3", "does not support the problem"),
                                       ("-1", None)])
def test_reconcile_names_the_dispatcher_status(code, why):
    fail = _fail("error", f"kernel returned status {code}")
    status, _, diag = GS.reconcile(_req(K1, A, 1), P1, [], [fail])
    assert status == "run_error" and diag.startswith(f"error: kernel returned status {code}")
    assert (why in diag) if why else not diag.endswith(")")


def _ranked(tmp_path, text, mode="w"):
    path = tmp_path / "ranked.csv"
    (path.write_text if mode == "w" else path.write_bytes)(text)
    return path


def test_read_ranked_csv_is_a_latency_prior(tmp_path):
    # select_kernels.py --out_ranked: predicted TFLOPS -> predicted ms.
    path = _ranked(tmp_path, "problem,rank,kernel,predicted_tflops,candidates\n"
                   "64x32x16,1,kB,4.0,3\n64x32x16,2,kA,2.0,3\n64x32x16,3,kC,0.0,3\n"
                   "8x8x8,1,kA,nan,1\n")
    flops = 2 * 64 * 32 * 16
    assert GS.read_ranked_csv(path) == {
        "64x32x16": {"kB": pytest.approx(flops / 4e9), "kA": pytest.approx(flops / 2e9)}}


@pytest.mark.parametrize("text, match", [
    ("problem,kernel\n64x32x16,kA\n", "expected columns"),
    ("problem,rank,kernel,predicted_tflops\n64x32x16,1,kA\n", "line 2"),  # short row
    ("problem,rank,kernel,predicted_tflops\n64x32,1,kA,2.0\n", "line 2"),
    ("problem,rank,kernel,predicted_tflops\n64x32x16,1,kA,fast\n", "line 2"),
    (b"problem,rank,kernel,predicted_tflops\n64x32x16,1,k\xff,2.0\n", "utf-8"),
])
def test_unusable_ranked_csv_is_a_settings_error(tmp_path, text, match):
    path = _ranked(tmp_path, text, "w" if isinstance(text, str) else "b")
    with pytest.raises(ValueError, match=match):
        GS.read_ranked_csv(path)
    # Through the coordinator it is invalid settings (the driver exits 2).
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "tpe", "trials_per_problem": 2,
        "new_builds_per_run": 2, "warm_start": {"ranked_csv": str(path),
                                                "use_prior": True}})
    with pytest.raises(AS.SettingsError, match="warm_start.ranked_csv"):
        AS.SearchCoordinator(None, settings, tmp_path / "out",
                             prior_loader=GS.read_ranked_csv)


# -------------------------------------------------------------- adapter


def test_candidates_are_the_eligible_kernels_with_scalar_records(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, problems=(P1, P2, P1),
                  eligible=[[A, B, A], [B, C], [D]])
    got = ad.enumerate_eligible()
    assert {k: [c.kernel_name for c in v] for k, v in got.items()} == {
        K1: [A, B], K2: [B, C]}
    b1, b2 = got[K1][1], got[K2][0]
    assert b1 is b2 and b1.canonical_key == b1.kernel_name == B
    assert b1.original_config is CFGS[1]
    assert set(b1.record) == set(drv.KERNEL_RECORD_FIELDS)
    assert not any(isinstance(v, str) and v.startswith("gemm_") for v in b1.record.values())
    ctx = ad.context()
    assert ctx["arch"] == "gfx950" and ctx["measurement"]["verify"] is True
    assert ctx["measurement"]["batch_size"] == 1
    assert any("searched once" in n for n in ctx["domain_notes"])
    json.dumps(ctx, allow_nan=False)


@pytest.mark.parametrize("timeout, used", [(None, 180), (900, 900)])
def test_build_gets_the_driver_options_and_reports_one_line(
    tmp_path, monkeypatch, timeout, used
):
    # The search's build timeout, not the driver's --build-timeout (rejected
    # with --search-config): its own default, or build_timeout_s.
    tc = Toolchain(tmp_path, monkeypatch, fail={B})
    ad = _adapter(tmp_path, tc.build, timeout=timeout, workers=5, build_timeout=7)
    res = ad.build_missing([ad.cands[n] for n in (A, B)])
    assert tc.builds == [{"names": [A, B], "verbose": True, "max_workers": 5,
                          "build_timeout": used}]
    assert ad.context()["build"] == {"workers": 5, "build_timeout": used}
    assert res[B] == AS.BuildResult(None, "Compile failed: k.cpp:1:2: error: boom")
    # The one-line reason goes to the events; the full error is kept on disk.
    log = json.loads((tmp_path / "out" / "rounds" / "0001-build.json").read_text())
    assert log == {"kernels": [A, B],
                   "errors": {B: "Compile failed: k.cpp:1:2: error: boom\nmore"}}
    art = tmp_path / "out" / "artifacts" / f"{A}.so"
    assert res[A] == AS.BuildResult(str(art), digest=hashlib.sha256(
        A.encode()).hexdigest())


def test_build_returning_the_wrong_number_of_paths_is_refused(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, lambda cfgs, **kw: tc.build(cfgs, **kw)[:-1])
    with pytest.raises(RuntimeError, match="build returned 1 paths for 2 kernels"):
        ad.build_missing([ad.cands[n] for n in (A, B)])


def test_failed_build_compacts_indices_and_stale_files_are_never_loaded(
    tmp_path, monkeypatch
):
    # Fixture 3: A, B, C requested; B fails. The compact list is A, C, so C is
    # index 1. A stale B library (an earlier build) and stale copies in the
    # artifacts directory are never loaded.
    tc = Toolchain(tmp_path, monkeypatch, fail={B}, lat={A: 2.0, C: 1.0})
    art = tmp_path / "out" / "artifacts"
    art.mkdir(parents=True)
    (tc.lib_dir / f"lib{B}.so").write_text(B)
    (art / f"{B}.so").write_text(B)
    (art / f"{A}.so").write_text(D)  # stale, overwritten by the fresh copy
    rec = _Recorder()
    ad = _adapter(tmp_path, tc.build, rec)
    ad.build_missing([ad.cands[n] for n in (A, B, C)])
    out = ad.measure_pairs([_req(K1, A, 1), _req(K1, B, 2), _req(K1, C, 3)])
    assert [(o.request.kernel_name, o.status, o.latency_ms) for o in out] == [
        (A, "ok", 2.0), (B, "compile_error", None), (C, "ok", 1.0)]
    (call,) = rec.calls
    assert call["built"] == [(A, art / f"{A}.so"), (C, art / f"{C}.so")]
    assert call["pairs"] == [[0], [1]] and call["problems"] == [P1, P1]
    assert (call["verify"], call["batch_size"]) == (True, 1)
    assert call["extra"] == {"check_identity": True}
    assert tc.loaded == [art / f"{A}.so", art / f"{C}.so"]
    assert all(B not in it["so_path"] for p in tc.payloads for it in p["items"])
    # The round's CSV holds both rows; its JSON maps problem_idx to the request.
    rounds = tmp_path / "out" / "rounds"
    with open(rounds / "0002-measure.csv", newline="") as f:
        assert [(r["kernel"], r["problem_idx"]) for r in csv.DictReader(f)] == [
            (A, "0"), (C, "1")]
    side = json.loads((rounds / "0002-measure.json").read_text())
    assert [(r["problem_idx"], r["call_id"]) for r in side["requests"]] == [
        (0, "d1"), (1, "d3")]


def test_identity_mismatch_is_a_run_error(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch, mislabel={A: C})
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[n] for n in (A, C)])
    a, c = ad.measure_pairs([_req(K1, A, 1), _req(K2, C, 2)])
    assert a.status == "run_error" and "identity mismatch" in a.diag, a
    assert c.status == "ok"
    assert all(p["check_identity"] is True for p in tc.payloads)


def test_artifact_changed_after_its_build_is_refused(tmp_path, monkeypatch):
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[A]])
    (tmp_path / "out" / "artifacts" / f"{A}.so").write_text(A + " ")
    (o,) = ad.measure_pairs([_req(K1, A, 1)])
    assert o.status == "run_error" and "changed" in o.diag
    assert not tc.loaded and not tc.payloads


def test_rows_and_failures_belong_to_their_request(tmp_path, monkeypatch):
    # Two kernels on two problems, B wrong on P2 only: every row and failure
    # record goes to its own request, and the confirmation writes its own files.
    tc = Toolchain(tmp_path, monkeypatch, wrong={(B, P2["M"])})
    ad = _adapter(tmp_path, tc.build)
    ad.build_missing([ad.cands[n] for n in (A, B)])
    out = ad.measure_pairs([_req(K1, A, 1), _req(K2, A, 2), _req(K1, B, 3),
                            _req(K2, B, 4)])
    assert [o.status for o in out] == ["ok", "ok", "ok", "numeric_fail"]
    (o,) = ad.confirm([AS.PairRequest(K2, A, A, f"{K2}:2", "c5")])
    assert o.status == "ok"
    names = sorted(p.name for p in (tmp_path / "out" / "rounds").iterdir())
    assert names == ["0001-build.json", "0002-measure.csv", "0002-measure.json",
                     "0003-confirm.csv", "0003-confirm.json"]


def test_attribution_holds_across_devices(tmp_path, monkeypatch):
    # Two device threads take the single-kernel units in any order; each
    # outcome still carries its own latency or its own failure.
    def ms(name, prob):
        return prob["M"] + 1.0 + [A, B, C].index(name)

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None,
                  failures=None):
        prob_idx, prob, batch = unit
        rows, n_fail = [], 0
        for _, cfg, _ in batch:
            base = {"kernel": cfg.name, "problem_idx": prob_idx, **prob,
                    "device": device_id}
            if (cfg.name, prob["M"]) == (B, P2["M"]):
                n_fail += 1
                failures.append({**base, "kind": "timeout", "error": "over 30 s"})
                continue
            rows.append({**base, "latency_ms": ms(cfg.name, prob), "tflops": 1.0,
                         "non_zero": 1, "max_rel": 0.0, "verified": True})
        return rows, [], n_fail, 0

    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, devices=("0", "1"))
    ad.build_missing([ad.cands[n] for n in (A, B, C)])
    reqs = [_req(k, n, i) for i, (n, k) in enumerate(
        ((n, k) for n in (A, B, C) for k in (K1, K2)), 1)]
    got = {(o.request.problem_key, o.request.kernel_name): (o.status, o.latency_ms)
           for o in ad.measure_pairs(reqs)}
    prob = {K1: P1, K2: P2}
    assert got == {(k, n): ("timeout", None) if (n, k) == (B, K2)
                   else ("ok", ms(n, prob[k])) for n in (A, B, C) for k in (K1, K2)}


def test_verified_calls_get_time_for_the_host_reference(tmp_path, monkeypatch):
    # --kernel-timeout covers the kernel; a verified one-kernel call also builds
    # the inputs and the fp32 reference on the host: 8192^3 gets 22 s more.
    big = {"M": 8192, "N": 8192, "K": 8192}
    seen = []

    def measure(problems, pairs, built, devices, args, *rest, **kw):
        seen.append(args.kernel_timeout)

    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, measure, problems=(P1, big), kernel_timeout=30)
    ad.build_missing([ad.cands[A]])
    ad.measure_pairs([_req(K1, A, 1)])
    ad.measure_pairs([_req(K1, A, 2), _req("8192x8192x8192", A, 3)])
    assert seen == [31, 52] and ad.args.kernel_timeout == 30
    assert [GS.verify_allowance_s(p) for p in (P1, big)] == [1, 22]


def test_confirmation_runs_each_problem_on_one_device(tmp_path, monkeypatch):
    # Two devices, two problems with two finalists each: every confirmation
    # unit of a problem runs on that problem's device (K1 on 0, K2 on 1), and
    # each outcome says where it ran.
    ran = []

    def run_batch(device_id, unit, args, worker_path, base_env, extra=None,
                  failures=None):
        prob_idx, prob, batch = unit
        ran.append((device_id, prob["M"]))
        return [{"kernel": cfg.name, "problem_idx": prob_idx, **prob,
                 "device": device_id, "latency_ms": 1.0, "tflops": 1.0, "non_zero": 1,
                 "max_rel": 0.0, "verified": True} for _, cfg, _ in batch], [], 0, 0

    monkeypatch.setattr(drv, "_run_batch_on_device", run_batch)
    tc = Toolchain(tmp_path, monkeypatch)
    ad = _adapter(tmp_path, tc.build, devices=("0", "1"))
    ad.build_missing([ad.cands[n] for n in (A, B)])
    reqs = [AS.PairRequest(k, n, n, f"{k}:{n}", f"c{i}")
            for i, (k, n) in enumerate((k, n) for k in (K1, K2) for n in (A, B))]
    for _ in range(4):
        assert [o.device for o in ad.confirm(reqs)] == ["0", "0", "1", "1"]
    assert sorted(set(ran)) == [("0", P1["M"]), ("1", P2["M"])] and len(ran) == 16


@pytest.mark.parametrize("confirm_ok", [False, True])
def test_misleading_rows_never_become_winners(tmp_path, monkeypatch, confirm_ok):
    # Fixture 2: a fast row with verified "False", one with a blank verified and
    # a slower valid one. Only the valid one is a finalist; when its fresh calls
    # fail there is no winner, never a fallback to the invalid rows.
    fields = {A: {"latency_ms": 0.1, "verified": "False"},
              B: {"latency_ms": 0.2, "verified": ""},
              C: {"latency_ms": 0.9, "verified": True}}
    calls = []

    def measure(problems, pairs, built, devices, args, wp, env, csv_path,
                extra=None, *, record_failures=None, measured=None, **kw):
        confirm = "confirm" in Path(csv_path).name
        for i, (prob, idx) in enumerate(zip(problems, pairs)):
            name = built[idx[0]][0].name
            calls.append((confirm, name))
            base = {"kernel": name, "problem_idx": i, **prob, "device": "0"}
            if confirm and not confirm_ok:
                record_failures([{**base, "kind": "error",
                                  "error": "kernel returned status -1"}])
            else:
                measured.append({**base, "tflops": 1.0, "non_zero": 1, "max_rel": "",
                                 **fields[name]})

    tc = Toolchain(tmp_path, monkeypatch)
    out = tmp_path / "out"
    settings = AS.SearchSettings.from_json({
        "schema_version": 1, "strategy": "exhaustive", "trials_per_problem": 3,
        "new_builds_per_run": 3, "confirm": {"finalists": 2, "fresh_calls": 2}})
    rc = GS.run_gemm_search(
        _args(search_output_dir=str(out)), settings, _hooks(tc.build, measure),
        problems=[P1], configs=CFGS, eligible=[[A, B, C]], devices=["0"],
        worker=(Path("w.py"), {}), context={"arch": "gfx950"},
    )
    assert sorted(n for confirm, n in calls if not confirm) == sorted([A, B, C])
    assert {n for confirm, n in calls if confirm} == {C}
    results = {e["kernel"]: e["status"] for e in map(
        json.loads, (out / "trials.jsonl").read_text().splitlines())
        if e["event"] == "result"}
    assert results == {A: "numeric_fail", B: "invalid_measurement", C: "ok"}
    winners = drv.load_pair_list(out / "winners.json")
    summary = json.loads((out / "summary.json").read_text())
    if confirm_ok:
        assert (rc, winners, summary["status"]) == (0, {K1: [C]}, "complete")
    else:
        assert (rc, winners, summary["status"]) == (1, {}, "no_winner")
        assert "no finalist passed confirmation" in summary["unresolved"][K1]


def test_worker_checks_the_library_identity_only_when_asked(monkeypatch, capsys):
    for cache in ("_ab_cache", "_ref_cache"):
        if hasattr(worker._run_one, cache):
            monkeypatch.delattr(worker._run_one, cache)
    monkeypatch.setattr(worker, "GpuGemmRunner", lambda lib_path: _Runner(C, 1.0))
    worker._run_one(0, "a.so", P1, A, verify=True)
    worker._run_one(1, "a.so", P1, A, verify=True, check_identity=True)
    monkeypatch.setattr(worker, "GpuGemmRunner",
                        lambda lib_path: _Runner(A, 1.0, status=-3))
    worker._run_one(2, "a.so", P1, A, check_identity=True)
    legacy, checked, status = (
        json.loads(ln) for ln in capsys.readouterr().out.splitlines())
    assert legacy["ok"] and legacy["verified"] and legacy["kernel"] == A
    assert set(legacy) == {"idx", "ok", "ms", "tflops", "non_zero", "kernel",
                           "max_rel", "verified"}
    assert not checked["ok"] and "identity mismatch" in checked["error"]
    assert repr(C) in checked["error"] and repr(A) in checked["error"]
    assert status == {"idx": 2, "ok": False, "error": "kernel returned status -3",
                      "kernel": A}


def test_legacy_worker_payload_and_rows_are_unchanged(tmp_path, monkeypatch):
    # A run without the search sends the worker what it always did (no
    # check_identity) and gets back the keys it always got.
    tc = Toolchain(tmp_path, monkeypatch)
    (lib,) = tc.build([CFGS[0]])
    rows = []
    drv.benchmark_on_devices([P1], [[0]], [(CFGS[0], lib)], ["0"], _args(verify=True),
                             *drv.worker_setup(), tmp_path / "legacy.csv", measured=rows)
    (payload,) = tc.payloads
    assert set(payload) == {"items", "verify", "verify_tol"}
    assert set(tc.outputs[0]) == {"idx", "ok", "ms", "tflops", "non_zero", "kernel",
                                  "max_rel", "verified"}
    (row,) = rows
    assert row["verified"] is True
    with open(tmp_path / "legacy.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == [
            "kernel", "problem_idx", "M", "N", "K", "device", "latency_ms", "tflops",
            "non_zero", "max_rel", "verified"]


def test_worker_checks_the_identity_of_a_multi_abd_library(monkeypatch, capsys):
    ran = []

    class MultiABD:
        kernel_name = "gemm_fp16_rcr_other_multiabd_x"

        def __init__(self, lib_path, **kw):
            pass

        def run(self, problem, verify=False, verify_tol=None):
            ran.append(problem)

    monkeypatch.setattr(worker, "GpuMultiABDRunner", MultiABD)
    worker._run_one(0, "m.so", P1, "gemm_fp16_rcr_multiabd_x", check_identity=True)
    (out,) = (json.loads(ln) for ln in capsys.readouterr().out.splitlines())
    assert not out["ok"] and "identity mismatch" in out["error"] and not ran
