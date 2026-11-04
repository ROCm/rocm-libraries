#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running GEMM kernels in an isolated subprocess.

Mirrors grouped_conv's run_one_grouped_conv_kernel.py:
- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":..}, "kernel_name": "..."}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "..."}, ...]}

Optional top-level keys ``verify`` (bool) and ``verify_tol`` (float) enable an
fp32 numpy reference check; when set, each OK result also carries ``verified``
and ``max_rel``. With ``check_identity`` (bool) a library whose own kernel name
is not the requested ``kernel_name`` is refused (the adaptive search sets it).
With ``host_cache`` (a ``host_cache.HostCache.spec``, also set by the search
only) a standard kernel's host inputs and fp32 reference come from the search
run's cache when the problem has an entry there, else they are computed and
stored: the same values, computed once per problem instead of once per call.
With ``prepare`` (the search, before a problem's first measure call) the
worker loads no library: it stores the host data of ``problem`` for the dtype
and layout of ``kernel_name`` in ``host_cache`` and prints
``{"ok": true, "stored": <whether the cache holds it now>}``.

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7, "non_zero": 1, "kernel": "..."}
    {"idx": 0, "ok": true, ..., "verified": true, "max_rel": 3.1e-4}   # with --verify
    {"idx": 1, "ok": false, "error": "...", "kernel": "..."}

``kernel`` echoes the request. The library's own name, which
``check_identity`` compares with it, is ``GpuGemmRunner.kernel_name`` (the
codegen ``KERNEL_NAME`` that ``GemmKernelConfig.name`` reproduces); a mismatch
is an error naming both.
"""

import json
import os
import sys

# Add dispatcher python paths from environment (os.pathsep-separated).
gemm_pypath = os.environ.get("GEMM_PYPATH", "")
if gemm_pypath:
    for p in gemm_pypath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)

from gemm_utils import (  # noqa: E402
    GemmProblem,
    GpuGemmRunner,
    GpuMultiABDRunner,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
    _resolve_use_ocp,
)
import numpy as np  # noqa: E402

INPUT_SEED, INPUT_SCALE = 42, 0.1  # the random host inputs: randn * scale
_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                           "common")


def _is_multi_abd(kernel_name):
    """A multi_abd kernel name carries the ``_multiabd_`` suffix (see codegen)."""
    return "_multiabd_" in (kernel_name or "")


def _check_identity(runner, kernel_name):
    """Refuse a library that reports another kernel than the requested one, so
    a stale or mislabeled .so cannot produce a result for the wrong kernel."""
    if runner.kernel_name != kernel_name:
        raise RuntimeError(
            f"identity mismatch: the library reports {runner.kernel_name!r}, "
            f"requested {kernel_name!r}"
        )


def _inputs(problem):
    """The fp32 host A (MxK) and B (KxN) every GEMM measurement uses."""
    rng = np.random.RandomState(INPUT_SEED)
    return (
        (rng.randn(problem.M, problem.K) * INPUT_SCALE).astype(np.float32),
        (rng.randn(problem.K, problem.N) * INPUT_SCALE).astype(np.float32),
    )


class _HostCodec(GpuGemmRunner):
    """``GpuGemmRunner``'s host encoding and reference for ``kernel_name``,
    without its library or a GPU: what preparing a problem needs. The fp8/bf8
    format is resolved as for the measure worker's runner (no arch given)."""

    def __init__(self, kernel_name):  # the base __init__ loads the library
        self._kernel_name, self._use_ocp = kernel_name, None


def _host_data(runner, problem, spec, prepare=False):
    """A and B as ``runner``'s kernel reads them, the fp32 reference and its
    max |ref| (``a``, ``b``, ``ref``, ``denom``), from the host cache ``spec``
    (``prepare``: whether the cache holds them once stored there). The
    entry's key holds everything they depend on: the problem, the kernel's
    dtype and layout (its packed strides follow from the layout and the
    extents), the fp8/bf8 format and the input generator."""
    if _COMMON_DIR not in sys.path:
        sys.path.insert(0, _COMMON_DIR)
    import host_cache

    dtype = _dtype_from_kernel_name(runner.kernel_name)
    fields = {
        "op": "gemm",
        "problem": problem.to_dict(),
        "dtype": dtype,
        "layout": _layout_from_kernel_name(runner.kernel_name),
        "fp8_ocp": _resolve_use_ocp(runner._use_ocp) if dtype in ("fp8", "bf8") else None,
        "inputs": [INPUT_SEED, INPUT_SCALE],
    }

    def compute():
        A, B = _inputs(problem)
        a, b = runner.encode_inputs(A, B)
        ref = runner.reference(A, B)
        return {"a": a, "b": b, "ref": ref,
                "denom": np.float64(float(np.max(np.abs(ref))) or 1.0)}

    return (host_cache.prepare if prepare else host_cache.cached)(spec, fields, compute)


def _prepare(prob_dict, kernel_name, spec):
    """Store a problem's host data for ``kernel_name``'s dtype and layout in
    the host cache ``spec`` and print whether the cache holds it."""
    try:
        problem = GemmProblem.from_dict(prob_dict)
        out = {"ok": True, "stored": _host_data(_HostCodec(kernel_name), problem, spec,
                                                prepare=True)}
    except Exception as e:
        out = {"ok": False, "error": str(e)}
    print(json.dumps(out), flush=True)


def _check(out, output, ref, denom, verify_tol):
    """Add ``max_rel`` of ``output`` against ``ref`` and ``verified`` to ``out``."""
    got = output.astype(np.float32)
    max_rel = float(np.max(np.abs(got - ref)) / denom)
    out["max_rel"] = max_rel
    out["verified"] = bool(max_rel <= verify_tol)


def _run_one(
    idx,
    so_path,
    prob_dict,
    kernel_name,
    verify=False,
    verify_tol=2e-2,
    layout4=None,
    a_op=None,
    b_op=None,
    cde_op=None,
    check_identity=False,
    host_cache=None,
):
    """Run a single kernel and emit its result as one JSON line.

    When ``verify`` is set, the kernel output is checked against an fp32 numpy
    reference using the global relative metric ``max|out - ref| / max|ref|``;
    the emitted ``verified`` field then reflects correctness, not just liveness
    (``non_zero``).

    Standard path: reference is ``A @ B`` on the kernel's quantized inputs
    (``GpuGemmRunner.reference``). Multi-ABD path (B1): the reference is
    the full ``E = CDE( AB(As) @ BB(Bs), {Ds} )`` computed inside
    GpuMultiABDRunner (it owns the operands); the runner returns ``max_rel`` on
    the result. ``layout4`` and the per-group element-wise ops come from the
    config object (N2) so the runner never guesses the layout from the name.
    """
    try:
        problem = GemmProblem.from_dict(prob_dict)

        # Multi-ABD has a divergent (array-pointer) ABI and its own runner, which
        # generates its A/B/D operands internally (no shared A,B). Route by name.
        if _is_multi_abd(kernel_name):
            # CRITICAL: load the library ONLY inside this subprocess.
            runner = GpuMultiABDRunner(
                lib_path=so_path,
                layout4=layout4,
                a_elementwise_op=a_op,
                b_elementwise_op=b_op,
                cde_elementwise_op=cde_op,
            )
            if check_identity:
                _check_identity(runner, kernel_name)
            # The multi_abd runner computes its own numpy reference (it owns the
            # operands), so verification flows through the runner, not the A@B
            # check below.
            result = runner.run(problem, verify=verify, verify_tol=verify_tol)
            A = B = host = None
        else:
            A = B = host = None
            if host_cache is None:
                # Cache host matrices per shape so batch mode doesn't regenerate
                # huge inputs per kernel.
                cache = getattr(_run_one, "_ab_cache", {})
                key = (problem.M, problem.N, problem.K)
                if key not in cache:
                    cache[key] = _inputs(problem)
                    _run_one._ab_cache = cache
                A, B = cache[key]

            # CRITICAL: load the library ONLY inside this subprocess.
            runner = GpuGemmRunner(lib_path=so_path)
            if check_identity:
                _check_identity(runner, kernel_name)
            if host_cache is None:
                result = runner.run(A, B, problem)
            else:
                host = _host_data(runner, problem, host_cache)
                result = runner.run_encoded(host["a"], host["b"], problem)

        if result.success:
            non_zero = (
                int(np.count_nonzero(result.output))
                if result.output is not None
                else 0
            )
            out = {
                "idx": idx,
                "ok": True,
                "ms": result.time_ms,
                "tflops": result.tflops,
                "non_zero": non_zero,
                "kernel": kernel_name,
            }
            if verify and _is_multi_abd(kernel_name):
                # Multi-ABD verifies against its own full multi-tensor reference
                # (E = CDE(AB(As) @ BB(Bs), {Ds})) computed inside the runner,
                # exercising the BRIDGE launch path (dispatcher_run_multi_abd ->
                # SelectedKernel::launch), not just the Old-TE --verify path.
                if result.max_rel is not None:
                    out["max_rel"] = result.max_rel
                    out["verified"] = bool(result.max_rel <= verify_tol)
            elif verify and host is not None:
                _check(out, result.output, host["ref"], float(host["denom"]), verify_tol)
            elif verify and A is not None and B is not None:
                # Reference uses the SAME quantized inputs the device sees, so the
                # metric isolates compute error from input quantization (fp8/bf8
                # rounding alone exceeds the default tolerance). It depends only on
                # the cached inputs and their host encoding, so batch mode computes
                # the host matmul once per shape and encoding, not once per kernel.
                refs = getattr(_run_one, "_ref_cache", {})
                ref_key = (
                    key,
                    _dtype_from_kernel_name(runner.kernel_name),
                    runner._use_ocp,
                )
                if ref_key not in refs:
                    ref = runner.reference(A, B)
                    refs[ref_key] = (ref, float(np.max(np.abs(ref))) or 1.0)
                    _run_one._ref_cache = refs
                _check(out, result.output, *refs[ref_key], verify_tol)
            print(json.dumps(out), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": False,
                        "error": f"kernel returned status {result.status}",
                        "kernel": kernel_name,
                    }
                ),
                flush=True,
            )

    except Exception as e:
        print(
            json.dumps(
                {"idx": idx, "ok": False, "error": str(e), "kernel": kernel_name}
            ),
            flush=True,
        )


def main():
    """Read JSON from stdin, run kernel(s), output results."""
    try:
        d = json.loads(sys.stdin.buffer.read())
    except Exception as e:
        print(
            json.dumps({"idx": 0, "ok": False, "error": f"JSON parse error: {e}"}),
            flush=True,
        )
        sys.exit(1)

    if d.get("prepare"):
        _prepare(d.get("problem"), d.get("kernel_name"), d.get("host_cache"))
        return

    verify = bool(d.get("verify", False))
    verify_tol = float(d.get("verify_tol", 2e-2))
    check_identity = bool(d.get("check_identity", False))
    host_cache = d.get("host_cache")

    if "items" in d:
        for i, item in enumerate(d["items"]):
            _run_one(
                i,
                item["so_path"],
                item["problem"],
                item.get("kernel_name", "unknown"),
                verify=verify,
                verify_tol=verify_tol,
                layout4=item.get("layout4"),
                a_op=item.get("a_elementwise_op"),
                b_op=item.get("b_elementwise_op"),
                cde_op=item.get("cde_elementwise_op"),
                check_identity=check_identity,
                host_cache=host_cache,
            )
    else:
        _run_one(
            0,
            d["so_path"],
            d["problem"],
            d.get("kernel_name", "unknown"),
            verify=verify,
            verify_tol=verify_tol,
            layout4=d.get("layout4"),
            a_op=d.get("a_elementwise_op"),
            b_op=d.get("b_elementwise_op"),
            cde_op=d.get("cde_elementwise_op"),
            check_identity=check_identity,
            host_cache=host_cache,
        )


if __name__ == "__main__":
    main()
