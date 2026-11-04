# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the tile_engine GEMM sweep driver's --verify and exit code.

* ``GpuGemmRunner.reference`` builds the fp32 reference from the values the
  kernel reads (the host encoding run() uses), so an exact kernel verifies for
  every dtype. Against unquantized inputs fp8/bf8 input rounding alone exceeds
  the default tolerance and every kernel was reported as MISMATCH.
* ``GpuBatchedGemmRunner`` shares it, and the GEMM, stream-K and batched
  workers use it.
* A sweep with failed measurements exits 1.
* A search's host data (``_search_host_data``) is made in the kernel's
  memory order and a block at a time: the same outcome with or without the
  search's host cache, whose entries hold it once per problem, dtype and
  layout, byte for byte the same whether a preparation or a call stored them,
  in about 5-6 bytes per input element.

The ctypes lib is replaced by a host model of an exact kernel.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(GEMM_TE_DIR.parent / "common"))

import gemm_full_benchmark as drv  # noqa: E402
import gemm_search as GS  # noqa: E402
import host_cache as HC  # noqa: E402
import run_one_batched_gemm_kernel  # noqa: E402
import run_one_gemm_kernel  # noqa: E402
import run_one_streamk_gemm_kernel  # noqa: E402
from batched_gemm_utils import BatchedGemmProblem, GpuBatchedGemmRunner  # noqa: E402
from gemm_utils import (  # noqa: E402
    GemmProblem,
    GpuGemmRunner,
    _bf8_u8_to_fp32,
    _bf16_u16_to_fp32,
    _fp8_u8_to_fp32,
    _fp32_to_bf16_u16,
)

DEFAULT_TOL = 2e-2
PROBLEM = {"M": 64, "N": 96, "K": 256}
BATCHED_PROBLEM = {"batch_count": 3, **PROBLEM}


def _native(h, ocp):
    return h.astype(np.float32)


# How the device reads each host input buffer, per kernel dtype.
DEVICE_DECODE = {
    "fp16": _native,
    "fp32": _native,
    "bf16": lambda h, ocp: _bf16_u16_to_fp32(h),
    "fp8": _fp8_u8_to_fp32,
    "bf8": _bf8_u8_to_fp32,
}
# Unit roundoff of the C the kernel stores (fp8/bf8 store fp16): with exact
# compute, only this rounding is left between C and the reference (fp32: plus
# the summation order of the host matmul).
C_EPS = {"bf16": 2.0**-8, "fp32": 2.0**-20}


def _c_eps(dtype):
    return C_EPS.get(dtype, 2.0**-11)


# fp8/bf8 in both the FNUZ (gfx942) and OCP (gfx950, gfx12) formats.
CASES = [("fp16", None), ("bf16", None), ("fp32", None)] + [
    (d, o) for d in ("fp8", "bf8") for o in (False, True)
]


class _ExactKernelLib:
    """Stands in for the plain or batched ctypes lib: computes C from the host
    buffers it is given in the kernel's layout, as an exact kernel would."""

    def __init__(self, dtype, layout, use_ocp, bias=0.0):
        self.dtype, self.layout, self.use_ocp, self.bias = dtype, layout, use_ocp, bias

    def run(self, A_h, B_h, C_h, *dims, **kw):
        def mat(X, lay):
            return X if lay == "r" else np.swapaxes(X, -1, -2)

        dec = DEVICE_DECODE[self.dtype]
        A = mat(dec(A_h, self.use_ocp), self.layout[0])
        B = mat(dec(B_h, self.use_ocp), self.layout[1])
        C = mat(A @ B + self.bias if self.bias else A @ B, self.layout[2])
        # bf16 C is a uint16 bit pattern; fp8/bf8 store fp16.
        C_h[...] = _fp32_to_bf16_u16(C) if self.dtype == "bf16" else C
        return 0, 1.0


def _runner(dtype, use_ocp, layout="rcr", cls=GpuGemmRunner, bias=0.0):
    """A runner of an exact kernel, or of one off by ``bias`` everywhere."""
    r = object.__new__(cls)
    r._kernel_name = f"gemm_{dtype}_{layout}_compv3_cshuffle_intrawave_128x128x64"
    r._use_ocp = use_ocp
    r.lib = _ExactKernelLib(dtype, layout, use_ocp, bias)
    return r


def _inputs(problem=PROBLEM):
    rng = np.random.RandomState(42)
    lead = (problem["batch_count"],) if "batch_count" in problem else ()
    M, N, K = problem["M"], problem["N"], problem["K"]
    A = (rng.randn(*lead, M, K) * 0.1).astype(np.float32)
    B = (rng.randn(*lead, K, N) * 0.1).astype(np.float32)
    return A, B


def _max_rel(got, ref):
    return float(np.max(np.abs(got.astype(np.float32) - ref)) / np.max(np.abs(ref)))


RUNNERS = {
    "gemm": (GpuGemmRunner, GemmProblem, PROBLEM),
    "batched": (GpuBatchedGemmRunner, BatchedGemmProblem, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", RUNNERS)
@pytest.mark.parametrize("layout", ["rcr", "rrr", "crr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_reference_matches_an_exact_kernel(dtype, use_ocp, layout, kind):
    cls, problem_cls, problem = RUNNERS[kind]
    runner = _runner(dtype, use_ocp, layout, cls)
    A, B = _inputs(problem)
    out = runner.run(A, B, problem_cls.from_dict(problem)).output
    assert _max_rel(out, runner.reference(A, B)) < _c_eps(dtype)


@pytest.mark.parametrize("dtype,use_ocp", [c for c in CASES if c[0] in ("fp8", "bf8")])
def test_unquantized_reference_mismatches_fp8_bf8(dtype, use_ocp):
    # The bug the reference fixes: input rounding alone exceeds the tolerance.
    runner = _runner(dtype, use_ocp)
    A, B = _inputs()
    out = runner.run(A, B, GemmProblem.from_dict(PROBLEM)).output
    assert _max_rel(out, A @ B) > DEFAULT_TOL


def test_reference_follows_the_fp8_format():
    A, B = _inputs()
    fnuz = _runner("fp8", False).reference(A, B)
    ocp = _runner("fp8", True).reference(A, B)
    # OCP E4M3 has one more exponent step than FNUZ, so the values differ.
    assert not np.array_equal(fnuz, ocp)


# ------------------------------------------------------------------ workers


WORKERS = {
    "gemm": (run_one_gemm_kernel, GpuGemmRunner, PROBLEM),
    "streamk": (run_one_streamk_gemm_kernel, GpuGemmRunner, PROBLEM),
    "batched": (run_one_batched_gemm_kernel, GpuBatchedGemmRunner, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", WORKERS)
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_worker_verifies_an_exact_kernel(monkeypatch, capsys, kind, dtype, use_ocp):
    worker, cls, problem = WORKERS[kind]
    runner = _runner(dtype, use_ocp, cls=cls)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    for cache in ("_ab_cache", "_ref_cache"):
        if hasattr(worker._run_one, cache):
            monkeypatch.delattr(worker._run_one, cache)
    worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["verified"], out
    assert out["max_rel"] < _c_eps(dtype)


def test_gemm_worker_reuses_the_reference_per_shape_and_encoding(monkeypatch, capsys):
    # The host matmul depends only on the shape's inputs and their encoding
    # (dtype, fp8 format), never on the kernel, so a sweep computes it once per
    # shape. Kernels with another layout reuse it; the batch below also mixes
    # encodings, which a real sweep never does, to show those are kept apart.
    worker = run_one_gemm_kernel
    for cache in ("_ab_cache", "_ref_cache"):
        if hasattr(worker._run_one, cache):
            monkeypatch.delattr(worker._run_one, cache)
    calls = []
    reference = GpuGemmRunner.reference
    monkeypatch.setattr(
        GpuGemmRunner,
        "reference",
        lambda self, A, B: calls.append(self.kernel_name) or reference(self, A, B),
    )
    other = {"M": 32, "N": 64, "K": 128}
    items = [
        (_runner("bf16", None), PROBLEM),
        (_runner("bf16", None, layout="crr"), PROBLEM),
        (_runner("fp8", False), PROBLEM),
        (_runner("fp8", True), PROBLEM),
        (_runner("bf16", None), other),
        (_runner("bf16", None, layout="ccr"), PROBLEM),
    ]
    for idx, (runner, problem) in enumerate(items):
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **kw: runner)
        worker._run_one(idx, "fake.so", problem, runner.kernel_name, verify=True)
    outs = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(outs) == len(items)
    for out, (runner, _) in zip(outs, items):
        assert out["ok"] and out["verified"], out
        assert out["max_rel"] < _c_eps(_dtype_of(runner)), out
    # bf16 on PROBLEM, fp8 FNUZ, fp8 OCP, bf16 on the other shape
    assert len(calls) == 4, calls


def _dtype_of(runner):
    return runner.kernel_name.split("_")[1]


# ------------------------------------------------------------- host cache


@pytest.fixture
def gemm_worker(monkeypatch, capsys):
    """run(runner, problem, **kw): the GEMM worker's JSON line for ``runner``
    (fresh in-process caches); ``counts`` of the legacy inputs and references
    and of the search's host data made."""
    worker = run_one_gemm_kernel
    counts = {"inputs": 0, "reference": 0, "search": 0}
    inputs, reference = worker._inputs, GpuGemmRunner.reference
    search = worker._search_host_data

    def count(what, fn):
        def wrapped(*a):
            counts[what] += 1
            return fn(*a)
        return wrapped

    monkeypatch.setattr(worker, "_inputs", count("inputs", inputs))
    monkeypatch.setattr(GpuGemmRunner, "reference", count("reference", reference))
    monkeypatch.setattr(worker, "_search_host_data", count("search", search))

    def run(runner, problem=PROBLEM, **kw):
        monkeypatch.setattr(worker, "GpuGemmRunner", lambda **k: runner)
        for cache in ("_ab_cache", "_ref_cache"):
            if hasattr(worker._run_one, cache):
                monkeypatch.delattr(worker._run_one, cache)
        worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True, **kw)
        return capsys.readouterr().out.strip()

    run.counts = counts
    return run


@pytest.mark.parametrize("layout", ["rcr", "crr", "rrr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_host_cache_gives_the_uncached_outcome(gemm_worker, tmp_path, dtype, use_ocp,
                                               layout):
    # A search's call without the cache, its first call with it (a miss) and
    # a call that reads the entry give the same line; the legacy inputs are
    # other values, verified as before.
    runner = _runner(dtype, use_ocp, layout)
    legacy = json.loads(gemm_worker(runner))
    plain = gemm_worker(runner, search_host_data=True)
    with HC.HostCache(tmp_path) as cache:
        miss = gemm_worker(runner, host_cache=cache.spec)
        hit = gemm_worker(runner, host_cache=cache.spec)
        assert HC.usage(cache.dir)[0] == 1
    assert plain == miss == hit and json.loads(hit)["verified"], (plain, hit)
    for out in (legacy, json.loads(hit)):
        assert out["verified"] and out["max_rel"] < _c_eps(dtype), out
    # The hit read everything: the search's host data made by the first two.
    assert gemm_worker.counts == {"inputs": 1, "reference": 1, "search": 2}


@pytest.mark.parametrize("layout", ["rcr", "crr", "rrr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_prepared_entry_gives_the_uncached_outcome(gemm_worker, tmp_path, monkeypatch,
                                                   capsys, dtype, use_ocp, layout):
    # The search's preparation loads no library: it encodes with
    # GpuGemmRunner's own codec and resolves the fp8/bf8 format as the measure
    # worker's runner does (no arch given: the local GPU's), so it stores the
    # entry a measure call would, and that call then computes nothing.
    import gemm_utils

    monkeypatch.setattr(gemm_utils, "_default_use_ocp", lambda: bool(use_ocp))
    runner = _runner(dtype, None, layout)
    plain = gemm_worker(runner, search_host_data=True)

    def no_library(*a, **kw):
        raise AssertionError("a preparation loaded a library")

    monkeypatch.setattr(gemm_utils, "GemmDispatcherLib", no_library)
    with HC.HostCache(tmp_path) as cache:
        run_one_gemm_kernel._prepare(PROBLEM, runner.kernel_name, cache.spec)
        assert json.loads(capsys.readouterr().out) == {"ok": True, "stored": True}
        hit = gemm_worker(runner, host_cache=cache.spec)
        assert HC.usage(cache.dir)[0] == 1
    assert hit == plain and json.loads(hit)["verified"]
    assert gemm_worker.counts == {"inputs": 0, "reference": 0, "search": 2}


def test_preparation_reports_what_it_could_not_store(capsys, tmp_path):
    name = _runner("bf16", None).kernel_name
    for args, want in (((PROBLEM, name, None), {"ok": True, "stored": False}),
                       ((None, name, {"dir": str(tmp_path)}), {"ok": False})):
        run_one_gemm_kernel._prepare(*args)
        out = json.loads(capsys.readouterr().out)
        assert {k: out[k] for k in want} == want
    assert out["error"].startswith("TypeError: ")  # the type, not only its text


def test_host_cache_keeps_dtypes_layouts_and_problems_apart(gemm_worker, tmp_path):
    # One entry per (problem, dtype, layout, fp8 format): a crr kernel handed
    # an rcr entry's buffers (or bf16 an fp16 one's) would fail verification.
    # So does a change of K alone, which leaves C's shape: another K's A and B
    # have the wrong length for the kernel.
    kernels = [(_runner("bf16", None, lay), PROBLEM) for lay in ("rcr", "crr", "ccr")]
    kernels += [(_runner("fp16", None), PROBLEM), (_runner("fp8", False), PROBLEM),
                (_runner("fp8", True), PROBLEM)]
    kernels += [(_runner("bf16", None), {**PROBLEM, **o})
                for o in ({"M": 96, "N": 64}, {"K": 128})]
    with HC.HostCache(tmp_path) as cache:
        first = [gemm_worker(r, p, host_cache=cache.spec) for r, p in kernels]
        again = [gemm_worker(r, p, host_cache=cache.spec) for r, p in kernels]
        assert HC.usage(cache.dir)[0] == len(kernels)
    assert first == again
    for line, (runner, _) in zip(first, kernels):
        out = json.loads(line)
        assert out["verified"] and out["max_rel"] < _c_eps(_dtype_of(runner)), out
    assert gemm_worker.counts["search"] == len(kernels)


@pytest.mark.parametrize("layout", ["rcr", "crr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_preparation_and_call_store_the_same_entry_bytes(gemm_worker, tmp_path,
                                                         monkeypatch, dtype, use_ocp,
                                                         layout):
    # A preparation and a call that misses compute the entry with the same
    # function: every file of the two entries is the same, byte for byte.
    import gemm_utils

    monkeypatch.setattr(gemm_utils, "_default_use_ocp", lambda: bool(use_ocp))
    runner = _runner(dtype, None, layout)
    entries = []
    for name in ("prepared", "called"):
        (tmp_path / name).mkdir()
        with HC.HostCache(tmp_path / name) as cache:
            if name == "prepared":
                run_one_gemm_kernel._prepare(PROBLEM, runner.kernel_name, cache.spec)
            else:
                gemm_worker(runner, host_cache=cache.spec)
            (entry,) = [p for p in cache.dir.iterdir() if p.is_dir()]
            entries.append({f.name: f.read_bytes() for f in entry.iterdir()})
    assert set(entries[0]) == {"a.npy", "b.npy", "ref.npy", "denom.npy", "meta.json"}
    assert entries[0] == entries[1]


def test_search_inputs_do_not_depend_on_the_block_size(monkeypatch):
    runner = _runner("fp8", False, "crr")
    problem = GemmProblem.from_dict(PROBLEM)
    whole = run_one_gemm_kernel._search_host_data(runner, problem)
    monkeypatch.setattr(run_one_gemm_kernel, "BLOCK", 7)
    blocks = run_one_gemm_kernel._search_host_data(runner, problem)
    assert {k: v.tobytes() for k, v in whole.items()} == {
        k: v.tobytes() for k, v in blocks.items()}
    assert whole["a"].shape == (PROBLEM["K"], PROBLEM["M"])  # a "c" A, as stored


@pytest.mark.parametrize("rows", [1, 5, 64, 1000])
def test_comparison_in_blocks_of_rows_gives_the_same_values(rows):
    rng = np.random.RandomState(1)
    ref = rng.randn(64, 96).astype(np.float32)
    out = (ref + 1e-3 * rng.randn(64, 96)).astype(np.float16)
    nan = out.copy()
    nan[40, 3] = np.nan
    for output in (out, np.asfortranarray(out), nan):
        whole, blocks = {}, {}
        run_one_gemm_kernel._check(whole, output, ref, 2.5, 2e-2)
        run_one_gemm_kernel._check(blocks, output, ref, 2.5, 2e-2, rows)
        assert json.dumps(whole) == json.dumps(blocks)


def test_search_calls_compare_a_block_of_rows_at_a_time(gemm_worker, monkeypatch):
    # C and the reference are compared BLOCK elements at a time in a search,
    # all at once otherwise (the legacy path is unchanged).
    seen, check = [], run_one_gemm_kernel._check
    monkeypatch.setattr(run_one_gemm_kernel, "_check",
                        lambda *a: seen.append(a[5:]) or check(*a))
    gemm_worker(_runner("bf16", None), search_host_data=True)
    gemm_worker(_runner("bf16", None))
    assert seen == [(run_one_gemm_kernel.BLOCK // PROBLEM["N"],), ()]


@pytest.mark.parametrize("bias", [-0.5, 0.5])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_search_calls_refuse_a_kernel_off_in_either_direction(gemm_worker, dtype,
                                                              use_ocp, bias):
    # The metric is max|C - ref| / max|ref|: a kernel whose every output is
    # too small fails as one whose outputs are too large.
    out = json.loads(gemm_worker(_runner(dtype, use_ocp, bias=bias), search_host_data=True))
    assert out["verified"] is False and out["max_rel"] > 0.1, out


@pytest.mark.parametrize("sign", [1, -1])
def test_search_denominator_is_the_largest_magnitude_of_the_reference(monkeypatch,
                                                                      sign):
    # B negated negates the reference: in one of the two runs its largest
    # magnitude is a negative entry, and the denominator is still that magnitude.
    operand, made = run_one_gemm_kernel._search_operand, []

    def signed(*a):
        enc, vals = operand(*a)
        made.append(vals)
        if len(made) == 2:  # B: the reference reads its fp32 values
            vals *= sign
        return enc, vals

    monkeypatch.setattr(run_one_gemm_kernel, "_search_operand", signed)
    host = run_one_gemm_kernel._search_host_data(_runner("fp16", None),
                                                 GemmProblem.from_dict(PROBLEM))
    ref = host["ref"]
    assert float(np.max(ref)) != -float(np.min(ref))
    assert float(host["denom"]) == float(np.max(np.abs(ref)))


def test_legacy_inputs_are_unchanged():
    # A run without the search keys measures with these inputs, as before.
    got = run_one_gemm_kernel._inputs(GemmProblem.from_dict(PROBLEM))
    for a, b in zip(got, _inputs()):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_search_workers_ask_to_be_ended_first_when_memory_runs_out(monkeypatch,
                                                                   tmp_path):
    # A search worker (a preparation, or a call with search_host_data) sets its
    # oom_score_adj to 1000; a plain run's worker leaves it alone.
    import io

    adj = tmp_path / "oom_score_adj"
    monkeypatch.setattr(run_one_gemm_kernel, "OOM_SCORE_ADJ", str(adj))
    monkeypatch.setattr(run_one_gemm_kernel, "_run_one", lambda *a, **kw: None)
    monkeypatch.setattr(run_one_gemm_kernel, "_prepare", lambda *a: None)
    plain = {"so_path": "x.so", "problem": PROBLEM, "kernel_name": "k", "verify": True}
    for payload, first in ((plain, False), ({**plain, "check_identity": True}, False),
                           ({**plain, "search_host_data": True}, True),
                           ({"prepare": True, "problem": PROBLEM}, True)):
        if adj.exists():
            adj.unlink()
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(
            json.dumps(payload).encode())))
        run_one_gemm_kernel.main()
        assert (adj.read_text() if adj.exists() else None) == ("1000" if first else None)
    monkeypatch.setattr(run_one_gemm_kernel, "OOM_SCORE_ADJ", str(tmp_path / "no" / "f"))
    run_one_gemm_kernel._first_to_go()  # cannot be written: nothing happens


_RSS_JOB = r"""
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
import gemm_utils
import run_one_gemm_kernel as w

name, mode = sys.argv[2], sys.argv[4]


class Lib:  # a kernel that writes every element of C
    def run(self, A, B, C, *dims):
        C[...] = 1
        return 0, 1.0


runner = object.__new__(gemm_utils.GpuGemmRunner)
runner._kernel_name, runner._use_ocp, runner.lib = name, False, Lib()
w.GpuGemmRunner = lambda lib_path: runner


def run(problem):  # _search_host_data alone, or a whole search call without an entry
    if mode == "data":
        return w._search_host_data(runner, gemm_utils.GemmProblem.from_dict(problem))
    w._run_one(0, "x.so", problem, name, verify=True, search_host_data=True)


def status(field):
    with open("/proc/self/status") as f:
        return next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith(field + ":"))


run({"M": 64, "N": 64, "K": 4096})
try:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")  # the peak (VmHWM) restarts at the current RSS
except OSError:
    print("skip")
    sys.exit()
base = status("VmRSS")
host = run(json.loads(sys.argv[3]))
print(status("VmHWM") - base)
"""


def _peak_rss(dtype, problem, mode):
    """Peak RSS (bytes, imports excluded) of ``mode``: "data" for
    ``_search_host_data``, "call" for a whole search call without an entry
    (a kernel that writes C, then the comparison). Its bound is the guard's
    estimate without the fixed part (the GPU runtime, the interpreter) plus
    128 MiB (numpy's and the fp8 encoder's temporaries of one block)."""
    paths = [str(DISPATCHER_DIR / "python"), str(GEMM_TE_DIR), str(GEMM_TE_DIR.parent /
                                                                   "common")]
    got = subprocess.run([sys.executable, "-c", _RSS_JOB, json.dumps(paths),
                          f"gemm_{dtype}_rcr_x", json.dumps(problem), mode],
                         capture_output=True, text=True, check=True).stdout.split()
    if got == ["skip"]:
        pytest.skip("/proc/self/clear_refs is not writable")
    bound = GS.host_compute_bytes(problem, dtype) - GS.HOST_FIXED_BYTES + (128 << 20)
    assert int(got[-1]) <= bound, (problem, int(got[-1]), bound)
    return int(got[-1])


@pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="Linux only")
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_search_host_data_needs_its_encoded_inputs_plus_4_bytes_each(dtype):
    # Peak RSS of _search_host_data at two K (256x768): each more input
    # element costs its encoded bytes plus its fp32 value, 5 B in fp8 and
    # 6 B in bf16, against about 58 and 16.5 B for the inputs, encoding and
    # reference a search made before (float64 normals, the fp8 encoder on
    # whole arrays, transposed copies, the inputs encoded twice). The guard's
    # estimate, less its fixed part, covers the peak.
    per = (_peak_rss(dtype, {"M": 256, "N": 768, "K": 36864}, "data")
           - _peak_rss(dtype, {"M": 256, "N": 768, "K": 4096}, "data")) / (
        (256 + 768) * (36864 - 4096))
    assert 0 < per <= GS.HOST_INPUT_BYTES[dtype] + 4 + 0.25, per


@pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="Linux only")
@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_search_call_needs_at_most_16_bytes_per_output_element(dtype):
    # A whole search call without an entry on output-heavy shapes: each more
    # output element costs the fp32 reference, C (bf16: also decoded to fp32)
    # and its comparison, within HOST_OUTPUT_BYTES.
    small, large = ({"M": n, "N": n, "K": 64} for n in (1024, 3072))
    inputs = (2 * 3072 - 2 * 1024) * 64 * (GS.HOST_INPUT_BYTES[dtype] + 4)
    per = (_peak_rss(dtype, large, "call") - _peak_rss(dtype, small, "call")
           - inputs) / (3072 ** 2 - 1024 ** 2)
    assert 4 < per <= GS.HOST_OUTPUT_BYTES, per


def test_host_cache_over_its_cap_computes_as_without_it(gemm_worker, tmp_path):
    runner = _runner("bf16", None, "crr")
    plain = gemm_worker(runner, search_host_data=True)
    with HC.HostCache(tmp_path, max_entry_bytes=1) as cache:
        lines = [gemm_worker(runner, host_cache=cache.spec) for _ in range(2)]
        assert HC.usage(cache.dir) == (0, 0)
    assert lines == [plain, plain]
    assert gemm_worker.counts == {"inputs": 0, "reference": 0, "search": 3}


# ------------------------------------------------------------- exit code


def _main(monkeypatch, capsys, tmp_path, n_fail):
    lib = tmp_path / "libfake.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemm_full_benchmark.py",
            "--arch",
            "gfx950",
            "--max-kernels",
            "1",
            "--csv",
            str(tmp_path / "out.csv"),
        ],
    )
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: ["0"])
    monkeypatch.setattr(
        drv, "setup_multiple_gemm_dispatchers", lambda cfgs, **kw: [lib] * len(cfgs)
    )
    monkeypatch.setattr(
        drv, "_run_batch_on_device", lambda *a, **kw: ([], [], n_fail, 0)
    )
    rc = drv.main()
    return rc, capsys.readouterr().out


@pytest.mark.parametrize("n_fail,rc", [(0, 0), (1, 1)])
def test_sweep_exit_code_reflects_failures(monkeypatch, capsys, tmp_path, n_fail, rc):
    got, out = _main(monkeypatch, capsys, tmp_path, n_fail)
    assert got == rc
    assert "BENCHMARK COMPLETE" in out


DRIVERS = [
    "gemm_full_benchmark.py",
    "batched_gemm_full_benchmark.py",
    "gemm_multi_d_full_benchmark.py",
    "streamk_gemm_full_benchmark.py",
    "grouped_gemm_full_benchmark.py",
    "block_scale_gemm/gemm_bquant/gemm_bquant_full_benchmark.py",
    "mx_gemm_full_benchmark.py",
]


@pytest.mark.parametrize("path", DRIVERS)
def test_every_gemm_sweep_driver_returns_its_failures(path):
    # The other drivers share the summary tail; their last return must depend
    # on the failure count instead of being a constant 0.
    tree = ast.parse((GEMM_TE_DIR / path).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    last = main.body[-1]
    assert isinstance(last, ast.Return)
    assert not isinstance(last.value, ast.Constant), ast.unparse(last)
    assert "fail" in ast.unparse(last.value)
