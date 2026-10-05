# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixed inference SDPA coverage and timing for public gfx1151 dispatch.

Run as a module with the platform and library on PYTHONPATH. Inputs, precision
and mask semantics are fixed before timing. Every case remains in the report,
including unsupported operations. Results go only to stdout.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import math
import statistics
import sys
from collections import defaultdict
from contextlib import ExitStack

import numpy as np

from rocke.runtime import hip_module
from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_num_cus

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
        except BaseException:
            self.close()
            raise

    def add(self, name, array):
        ptr = self.rt.alloc(array.nbytes)
        self.ptrs[name] = ptr
        self.arrays[name] = array
        if array.nbytes:  # newer HIP rejects zero-byte copies to a null allocation
            self.rt.memcpy_h2d(ptr, _host_bytes(array), array.nbytes)

    def read_output(self, name):
        array = self.arrays[name]
        if array.nbytes:
            self.rt.memcpy_d2h(_host_bytes(array), self.ptrs[name], array.nbytes)
        return array.astype(np.float32)

    def close(self):
        self.rt.sync()
        for ptr in self.ptrs.values():
            self.rt.free(ptr)
        self.ptrs.clear()


class HipGraphs:
    """Same-stream graph timing avoids comparing Python enqueue overhead.

    Captured rocKE argument buffers stay retained by Runtime until every graph
    is destroyed. Graphs use the exact HIP runtime resolved by Runtime.
    """

    def __init__(self, rt):
        self.rt = rt
        lib = hip_module._resolve_hip()
        signatures = {
            "hipStreamCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "hipStreamDestroy": [ctypes.c_void_p],
            "hipStreamBeginCapture": [ctypes.c_void_p, ctypes.c_int],
            "hipStreamEndCapture": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            "hipGraphInstantiateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ],
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
            self.call(
                "hipGraphInstantiateWithFlags", ctypes.byref(executable), graph, 0
            )
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
        "max_abs_error": maximum,
        "rms_error": rms,
        "atol": atol,
    }


def _case_result(case, rt, kernels, graph_count, repeats):
    inputs = make_inputs(case)
    expected = reference(case, inputs)
    if not np.isfinite(expected).all():
        raise ValueError(f"Independent reference is nonfinite for {case.name}")
    row = {"case": dataclasses.asdict(case), "rocke": {}}
    result = row["rocke"]
    with ExitStack() as stack:
        buffers = DeviceBuffers(rt, inputs)
        stack.callback(buffers.close)
        graphs = HipGraphs(rt)
        stack.callback(kernels.release, graphs.stream)
        stack.callback(graphs.close)
        try:
            launch, result["kernel"] = kernels.prepare(case, buffers)
            launch(graphs.stream)
            rt.stream_sync(graphs.stream)
            result.update(_check_output(buffers, "rocke_out", expected, case.atol))
            if result["status"] != "passed":
                return row
            graph = graphs.capture(launch, graph_count)
            samples = [graphs.measure(graph, graph_count) for _ in range(repeats)]
            result.update(_check_output(buffers, "rocke_out", expected, case.atol))
            if result["status"] == "passed":
                result["us"] = statistics.median(samples)
                result["samples_us"] = samples
                result["spread_pct"] = (
                    100.0 * (max(samples) - min(samples)) / result["us"]
                )
        except UnsupportedCase as error:
            result.update(status="unsupported", reason=str(error))
        except Exception as error:
            result.update(status="error", reason=f"{type(error).__name__}: {error}")
    return row


def _geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def _report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"]["group"]].append(row)
    print("\n| Operation / features | pass/required | Geomean us | Gaps |")
    print("|---|---:|---:|---|")
    for group, members in groups.items():
        results = [r["rocke"] for r in members]
        passed = [r for r in results if r["status"] == "passed"]
        gaps = sorted({r["status"] for r in results if r["status"] != "passed"})
        mean = _geomean([r["us"] for r in passed])
        value = f"{mean:.3f}" if mean is not None else "N/A"
        print(
            f"| {group} | {len(passed)}/{len(members)} | {value} | {', '.join(gaps) or 'none'} |"
        )
    statuses = [r["rocke"]["status"] for r in rows]
    metrics = {
        "coverage_passed": statuses.count("passed"),
        "coverage_total": len(rows),
        "correctness_failures": statuses.count("incorrect"),
        "execution_errors": statuses.count("error"),
        "unsupported_cases": statuses.count("unsupported"),
    }
    print("\nSUMMARY " + json.dumps(metrics, sort_keys=True, allow_nan=False))
    for name, value in metrics.items():
        print(f"METRIC {name}={value}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if args.graph_count < 1 or args.repeats < 3:
        parser.error("graph-count must be positive and repeats must be >=3")
    if get_device_arch() != "gfx1151":
        raise RuntimeError(f"Expected gfx1151, got {get_device_arch()!r}")
    import ml_dtypes

    print(
        "PROVENANCE "
        + json.dumps(
            {
                "suite_sha256": suite_hash(),
                "arch": "gfx1151",
                "compute_units": get_device_num_cus(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "ml_dtypes": ml_dtypes.__version__,
                "measurement": "HIP graph replay, median batches",
                "graph_count": args.graph_count,
                "repeats": args.repeats,
                "scope": "inference; public rocKE attention dispatch and tensor binding",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    rt = Runtime()
    kernels = RockeKernels(rt)
    rows = []
    try:
        for case in CASES:
            row = _case_result(case, rt, kernels, args.graph_count, args.repeats)
            rows.append(row)
            print(
                "CASE " + json.dumps(row, sort_keys=True, allow_nan=False), flush=True
            )
    finally:
        kernels.close()
    if [r["case"]["name"] for r in rows] != [case.name for case in CASES]:
        raise RuntimeError("Frozen case coverage was altered or incomplete")
    metrics = _report(rows)
    if metrics["correctness_failures"] or metrics["execution_errors"]:
        return 1
    return 0 if metrics["coverage_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
