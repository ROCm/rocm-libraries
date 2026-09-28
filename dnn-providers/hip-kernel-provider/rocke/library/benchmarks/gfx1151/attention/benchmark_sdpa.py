# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Frozen inference SDPA coverage and on-device AOTriton comparison.

Run as a module with the platform and library on PYTHONPATH. Inputs, precision,
mask semantics, source identity, and the comparator are fixed before timing.
Every case remains in the report, including unsupported operations. Results go
only to stdout; this program never writes measured results into the source tree.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from rocke.runtime import hip_module
from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_num_cus

from .aotriton import Aotriton, TensorView
from .candidate import RockeKernels, UnsupportedCase
from .cases import CASES, make_inputs, reference, suite_hash


def _host_bytes(array):
    if not array.flags.c_contiguous:
        raise ValueError("GPU input storage must be contiguous")
    # ml_dtypes scalars do not expose a ctypes-compatible buffer format.
    # A byte view preserves the exact BF16/FP8 storage without conversion.
    return (ctypes.c_ubyte * array.nbytes).from_buffer(array.reshape(-1).view(np.uint8))


class DeviceBuffers:
    def __init__(self, rt, inputs):
        self.rt = rt
        self.arrays = {}
        self.ptrs = {}
        try:
            for field in dataclasses.fields(inputs):
                value = getattr(inputs, field.name)
                if value is not None:
                    self.add(field.name, np.ascontiguousarray(value))
            self.add("rocke_out", np.full_like(inputs.q, np.nan))
            self.add("aot_out", np.full_like(inputs.q, np.nan))
        except BaseException:
            self.close()
            raise

    def add(self, name, array):
        ptr = self.rt.alloc(array.nbytes)
        self.ptrs[name] = ptr
        self.arrays[name] = array
        self.rt.memcpy_h2d(ptr, _host_bytes(array), array.nbytes)

    def read_output(self, name):
        array = self.arrays[name]
        self.rt.memcpy_d2h(_host_bytes(array), self.ptrs[name], array.nbytes)
        return array.astype(np.float32)

    def tensor(self, name, *, attention=False, rank=None):
        if name not in self.ptrs:
            return None
        array = self.arrays[name]
        if attention:
            if array.ndim == 4:
                array = array.transpose(0, 2, 1, 3)
            elif array.ndim == 3:
                array = array.transpose(1, 0, 2)[None, ...]
            else:
                raise ValueError(f"Invalid attention tensor rank: {array.ndim}")
        if rank is not None:
            while array.ndim < rank:
                array = array[None, ...]
        names = {
            "float16": "fp16", "bfloat16": "bf16", "float32": "fp32",
            "int32": "i32", "uint64": "u64",
        }
        dtype = names.get(str(array.dtype), str(array.dtype))
        return TensorView(
            ptr=self.ptrs[name], shape=tuple(array.shape),
            strides=tuple(s // array.itemsize for s in array.strides), dtype=dtype,
        )

    def close(self):
        self.rt.sync()
        for ptr in self.ptrs.values():
            self.rt.free(ptr)
        self.ptrs.clear()


class HipGraphs:
    """Same-stream graph timing avoids comparing Python enqueue overhead.

    Captured rocKE argument buffers stay retained by Runtime until every graph
    is destroyed. Both libraries use the exact HIP runtime resolved by Runtime.
    """

    def __init__(self, rt):
        self.rt = rt
        lib = hip_module._resolve_hip()
        signatures = {
            "hipStreamCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
            "hipStreamDestroy": [ctypes.c_void_p],
            "hipStreamBeginCapture": [ctypes.c_void_p, ctypes.c_int],
            "hipStreamEndCapture": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            "hipGraphInstantiateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong],
            "hipGraphLaunch": [ctypes.c_void_p, ctypes.c_void_p],
            "hipGraphExecDestroy": [ctypes.c_void_p],
            "hipGraphDestroy": [ctypes.c_void_p],
        }
        self.functions = {}
        for name, args in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
            self.functions[name] = fn
        stream = ctypes.c_void_p()
        self.call("hipStreamCreateWithFlags", ctypes.byref(stream), 1)
        self.stream = int(stream.value)
        self.graphs = []

    def call(self, name, *args):
        status = self.functions[name](*args)
        if status:
            raise RuntimeError(f"{name}: HIP error {status}")

    def capture(self, launch, count):
        for _ in range(5):
            launch(self.stream)
        self.rt.stream_sync(self.stream)
        graph = ctypes.c_void_p()
        self.call("hipStreamBeginCapture", self.stream, 0)
        try:
            for _ in range(count):
                launch(self.stream)
        except BaseException:
            self.functions["hipStreamEndCapture"](self.stream, ctypes.byref(graph))
            if graph.value:
                self.functions["hipGraphDestroy"](graph)
            raise
        self.call("hipStreamEndCapture", self.stream, ctypes.byref(graph))
        executable = ctypes.c_void_p()
        try:
            self.call("hipGraphInstantiateWithFlags", ctypes.byref(executable), graph, 0)
        except BaseException:
            self.functions["hipGraphDestroy"](graph)
            raise
        self.graphs.append((graph, executable))
        for _ in range(3):
            self.call("hipGraphLaunch", executable, self.stream)
        self.rt.stream_sync(self.stream)
        return executable

    def measure(self, graph, count):
        start, end = self.rt.event(), self.rt.event()
        try:
            start.record(self.stream)
            self.call("hipGraphLaunch", graph, self.stream)
            end.record(self.stream)
            end.synchronize()
            us = start.elapsed_to(end) * 1000.0 / count
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError(f"Invalid HIP event duration: {us}")
            return us
        finally:
            start.destroy()
            end.destroy()

    def close(self):
        self.rt.stream_sync(self.stream)
        for graph, executable in reversed(self.graphs):
            self.call("hipGraphExecDestroy", executable)
            self.call("hipGraphDestroy", graph)
        self.graphs.clear()
        self.rt.wait_stream(self.stream)
        self.call("hipStreamDestroy", self.stream)


def _prepare_aot(aot, case, buffers, stack):
    if case.layout == "paged":
        raise UnsupportedCase("AOTriton v3 has no paged-KV input API; no free densification")
    if case.kv_dtype not in ("", case.dtype):
        raise UnsupportedCase("AOTriton v3 forward has no FP8 KV input")
    if case.softcap:
        raise UnsupportedCase("AOTriton v3 forward has no score-softcap parameter")
    if case.sinks:
        raise UnsupportedCase("AOTriton v3 forward has no attention-sink parameter")
    if case.qq_bias and (case.layout != "dense" or case.seqlen_q != case.seqlen_k):
        raise UnsupportedCase("AOTriton additive bias is not packed context-relative QQ bias")
    if case.qq_bias and buffers.arrays["qq_bias"].dtype != buffers.arrays["q"].dtype:
        raise UnsupportedCase("AOTriton requires bias dtype == Q; rocKE QQ-bias is float32")
    if case.layout == "ragged" and case.window and case.mask == "causal_bottomright":
        raise UnsupportedCase("AOTriton finite window needs per-sequence left diagonals for this packed layout")
    if case.alibi:
        raise UnsupportedCase("AOTriton v3 has no native ALiBi parameter")
    prepared = aot.prepare(
        q=buffers.tensor("q", attention=True),
        k=buffers.tensor("k", attention=True),
        v=buffers.tensor("v", attention=True),
        out=buffers.tensor("aot_out", attention=True),
        scale=case.scale, mask=case.mask, window=case.window,
        bias=buffers.tensor("qq_bias", rank=4),
        sinks=buffers.tensor("sinks", rank=2),
        cu_seqlens_q=buffers.tensor("cu_seqlens_q") if case.layout == "ragged" else None,
        cu_seqlens_k=buffers.tensor("cu_seqlens_k") if case.layout == "ragged" else None,
        seqused_k=None,
        max_seqlen_q=max(case.q_lengths, default=case.seqlen_q),
        max_seqlen_k=max(case.k_lengths, default=case.seqlen_k),
        batch=case.batch,
    )
    stack.callback(prepared.close)
    return lambda stream: prepared.launch(stream=stream, backend=-1)


def _check_output(buffers, name, expected, atol):
    actual = buffers.read_output(name)
    if actual.shape != expected.shape:
        raise ValueError(f"Output shape {actual.shape} != reference {expected.shape}")
    if not np.isfinite(actual).all():
        return {"status": "incorrect", "reason": "nonfinite or unwritten output"}
    difference = np.abs(actual - expected)
    maximum = float(difference.max())
    rms = float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))
    return {
        "status": "passed" if maximum <= atol else "incorrect",
        "max_abs_error": maximum, "rms_error": rms, "atol": atol,
    }


def _case_result(case, rt, kernels, aot, graph_count, repeats):
    inputs = make_inputs(case)
    expected = reference(case, inputs)
    if not np.isfinite(expected).all():
        raise ValueError(f"Independent reference is nonfinite for {case.name}")
    row = {"case": dataclasses.asdict(case), "rocke": {}, "aotriton": {}}
    with ExitStack() as stack:
        buffers = DeviceBuffers(rt, inputs)
        stack.callback(buffers.close)
        graphs = HipGraphs(rt)
        stack.callback(graphs.close)
        launches = {}
        for arm in ("rocke", "aotriton"):
            try:
                if arm == "rocke":
                    launch, kernel_name = kernels.prepare(case, buffers)
                    row[arm]["kernel"] = kernel_name
                else:
                    launch = _prepare_aot(aot, case, buffers, stack)
                    row[arm]["backend"] = "AOTriton 0.14.2b automatic dispatch"
                launch(graphs.stream)
                rt.stream_sync(graphs.stream)
                out_name = "rocke_out" if arm == "rocke" else "aot_out"
                row[arm].update(_check_output(buffers, out_name, expected, case.atol))
                if row[arm]["status"] == "passed":
                    launches[arm] = graphs.capture(launch, graph_count)
            except UnsupportedCase as error:
                row[arm].update(status="unsupported", reason=str(error))
            except Exception as error:
                row[arm].update(status="error", reason=f"{type(error).__name__}: {error}")
        timings = {arm: [] for arm in launches}
        for repeat in range(repeats):
            order = list(launches)
            if repeat % 2:
                order.reverse()
            for arm in order:
                timings[arm].append(graphs.measure(launches[arm], graph_count))
        for arm, samples in timings.items():
            out_name = "rocke_out" if arm == "rocke" else "aot_out"
            row[arm].update(_check_output(buffers, out_name, expected, case.atol))
            if row[arm]["status"] == "passed":
                row[arm]["us"] = statistics.median(samples)
                row[arm]["samples_us"] = samples
                row[arm]["spread_pct"] = 100.0 * (max(samples) - min(samples)) / row[arm]["us"]
        if all(row[arm].get("status") == "passed" for arm in ("rocke", "aotriton")):
            row["speedup"] = row["aotriton"]["us"] / row["rocke"]["us"]
    return row


def _geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def _report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"]["group"]].append(row)
    print("\n| Operation / features | rocKE pass/required | AOT pass/required | Paired rocKE us | Paired AOT us | Speedup | Gaps |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for group, members in groups.items():
        pairs = [r for r in members if "speedup" in r]
        passed = sum(r["rocke"]["status"] == "passed" for r in members)
        aot_passed = sum(r["aotriton"]["status"] == "passed" for r in members)
        gaps = sorted({r[arm]["status"] for r in members for arm in ("rocke", "aotriton") if r[arm]["status"] != "passed"})
        numbers = [_geomean([r[arm]["us"] for r in pairs]) for arm in ("rocke", "aotriton")]
        speedup = _geomean([r["speedup"] for r in pairs])
        values = [f"{n:.3f}" if n is not None else "N/A" for n in (*numbers, speedup)]
        print(f"| {group} | {passed}/{len(members)} | {aot_passed}/{len(members)} | {' | '.join(values)} | {', '.join(gaps) or 'none'} |")
    pairs = [r for r in rows if "speedup" in r]
    ratios = [r["speedup"] for r in pairs]
    metrics = {
        "coverage_passed": sum(r["rocke"]["status"] == "passed" for r in rows),
        "coverage_total": len(rows),
        "correctness_failures": sum(r[arm]["status"] == "incorrect" for r in rows for arm in ("rocke", "aotriton")),
        "execution_errors": sum(r[arm]["status"] == "error" for r in rows for arm in ("rocke", "aotriton")),
        "unsupported_cases": sum(r["rocke"]["status"] == "unsupported" for r in rows),
        "aotriton_passed": sum(r["aotriton"]["status"] == "passed" for r in rows),
        "comparable_cases": len(pairs),
    }
    if ratios:
        metrics.update(geomean_speedup=_geomean(ratios), worst_speedup=min(ratios))
    print("\nSUMMARY " + json.dumps(metrics, sort_keys=True, allow_nan=False))
    for name, value in metrics.items():
        print(f"METRIC {name}={value}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aotriton-shim", required=True, type=Path)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--graph-count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if args.graph_count < 1 or args.repeats < 3:
        parser.error("graph-count must be positive and repeats must be >=3")
    if get_device_arch() != "gfx1151":
        raise RuntimeError(f"Expected gfx1151, got {get_device_arch()!r}")
    import ml_dtypes

    print("PROVENANCE " + json.dumps({
        "source_sha256": args.source_hash, "suite_sha256": suite_hash(),
        "arch": "gfx1151", "compute_units": get_device_num_cus(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "ml_dtypes": ml_dtypes.__version__, "llvm_flavor": os.environ.get("ROCKE_LLVM_FLAVOR"),
        "aotriton": "0.14.2b", "measurement": "HIP graph replay, alternating arms, median batches",
        "graph_count": args.graph_count, "repeats": args.repeats,
        "scope": "inference; direct existing WMMA builder until gfx1151 dispatch is implemented",
    }, sort_keys=True), flush=True)
    rt = Runtime()
    kernels = RockeKernels(rt)
    aot = Aotriton(args.aotriton_shim)
    print("AOTRITON_BUILD " + json.dumps({
        "release": "0.14.2b", "runtime_version": aot.version,
        "runtime_git_sha1": aot.git_sha1, "namespace_suffix": aot.name_suffix,
        "forward_params_version": aot.params_version,
    }, sort_keys=True), flush=True)
    rows = []
    try:
        for case in CASES:
            row = _case_result(case, rt, kernels, aot, args.graph_count, args.repeats)
            rows.append(row)
            print("CASE " + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    finally:
        kernels.close()
    if [r["case"]["name"] for r in rows] != [case.name for case in CASES]:
        raise RuntimeError("Frozen case coverage was altered or incomplete")
    metrics = _report(rows)
    if metrics["correctness_failures"] or metrics["execution_errors"]:
        return 1
    if not metrics["aotriton_passed"] or not metrics["coverage_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
