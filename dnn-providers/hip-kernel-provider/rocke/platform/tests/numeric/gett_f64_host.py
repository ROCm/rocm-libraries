# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host side of the f64 GETT tests: buffers, launch, and exact references.

The reference for an output is the correctly rounded exact sum
``sum_k a_k * b_k``: every product is split into two doubles without error
(Veltkamp / Dekker), and ``math.fsum`` adds all of them exactly. The error
bound the device must meet is the f64 MFMA (sequential RN fma chain) bound
plus the single-rounding epilogue, per the precision contract:

``|d - t| <= u*|t| + (K + 2)*u*(|alpha| * sum|a||b| + |beta|*|c|)``

(complex: ``2K + 4`` roundings per component).
"""

from __future__ import annotations

import ctypes
import math
import struct
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rocke.instances.common.gett_f64 import (
    GettF64Problem,
    GettF64Spec,
    REDUCE_THREADS,
    build_gett_f64,
    build_gett_f64_reduce,
    digits_span,
    gett_f64_grid,
    gett_f64_reduce_grid,
    gett_f64_reduce_signature,
    gett_f64_runtime_args,
    gett_f64_signature,
    gett_f64_workspace_bytes,
)

U = 2.0**-53


def offsets(digits, n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    off = np.zeros(n, dtype=np.int64)
    for e, s in digits:
        off += (idx % e) * s
        idx //= e
    return off


def gather(problem: GettF64Problem, which: str, flat: np.ndarray) -> np.ndarray:
    """Logical [L, rows, cols] view of an operand (complex: complex128)."""
    p = problem
    if which == "A":
        r, c, l, R, Cn = p.a_m, p.a_k, p.a_l, p.M, p.K
    elif which == "B":
        r, c, l, R, Cn = p.b_n, p.b_k, p.b_l, p.N, p.K
    elif which == "C":
        cm, cn, cl = p.c_digits()
        r, c, l, R, Cn = cm, cn, cl, p.M, p.N
    else:
        r, c, l, R, Cn = p.d_m, p.d_n, p.d_l, p.M, p.N
    off = (
        offsets(l, p.L)[:, None, None]
        + offsets(r, R)[None, :, None]
        + offsets(c, Cn)[None, None, :]
    )
    if p.complex_:
        z = flat.view(np.complex128)
        return z[off]
    return flat[off]


def span_elems(problem: GettF64Problem, which: str) -> int:
    return problem.span_bytes(which) // problem.elem_bytes


def _split(x):
    c = 134217729.0 * x
    hi = c - (c - x)
    return hi, x - hi


def two_prod(a: np.ndarray, b: np.ndarray):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def exact_dot(a: np.ndarray, b: np.ndarray) -> float:
    p, e = two_prod(a, b)
    return math.fsum(np.concatenate([p, e]).tolist())


def reference(
    problem, A, B, C, alpha, beta, samples
) -> List[Tuple[Tuple[int, int, int], complex, float]]:
    """(index, correctly rounded alpha*t + beta*c, bound) for each sample."""
    Am, Bm = gather(problem, "A", A), gather(problem, "B", B)
    Cm = gather(problem, "C", C) if C is not None else None
    K = problem.K
    out = []
    for l, m, n in samples:
        a, b = Am[l, m], Bm[l, n]
        if problem.complex_:
            ar, ai = a.real.copy(), a.imag.copy()
            br, bi = b.real.copy(), b.imag.copy()
            if problem.conj_a:
                ai = -ai
            if problem.conj_b:
                bi = -bi
            pr1, er1 = two_prod(ar, br)
            pr2, er2 = two_prod(-ai, bi)
            pi1, ei1 = two_prod(ar, bi)
            pi2, ei2 = two_prod(ai, br)
            tr = math.fsum(np.concatenate([pr1, er1, pr2, er2]).tolist())
            ti = math.fsum(np.concatenate([pi1, ei1, pi2, ei2]).tolist())
            s = float(np.sum(np.abs(a) * np.abs(b)))
            c = complex(Cm[l, m, n]) if Cm is not None else 0j
            al, be = complex(alpha), complex(beta) if Cm is not None else 0j
            F = Fraction
            vr = (
                F(al.real) * F(tr)
                - F(al.imag) * F(ti)
                + F(be.real) * F(c.real)
                - F(be.imag) * F(c.imag)
            )
            vi = (
                F(al.real) * F(ti)
                + F(al.imag) * F(tr)
                + F(be.real) * F(c.imag)
                + F(be.imag) * F(c.real)
            )
            val = complex(float(vr), float(vi))
            bound = U * abs(val) + (2 * K + 4) * U * (abs(al) * s + abs(be) * abs(c))
        else:
            t = exact_dot(a, b)
            s = float(np.sum(np.abs(a) * np.abs(b)))
            c = float(Cm[l, m, n]) if Cm is not None else 0.0
            be = float(beta) if Cm is not None else 0.0
            val = float(
                Fraction(float(alpha)) * Fraction(t) + Fraction(be) * Fraction(c)
            )
            bound = U * abs(val) + (K + 2) * U * (abs(alpha) * s + abs(be) * abs(c))
        out.append(((l, m, n), val, bound))
    return out


def _scalars(spec, alpha, beta):
    cx = spec.problem.complex_
    out = {}
    if cx:
        out["alpha_re"], out["alpha_im"] = complex(alpha).real, complex(alpha).imag
        if spec.bilinear:
            out["beta_re"], out["beta_im"] = complex(beta).real, complex(beta).imag
    else:
        out["alpha"] = float(alpha)
        if spec.bilinear:
            out["beta"] = float(beta)
    return out


def compile_kernel(spec: GettF64Spec, arch: str, *, engine="python", reduce=False):
    """(kernel name, HSACO) of ``spec`` lowered through ``engine`` (``reduce``:
    the second kernel of ``split_k``)."""
    from rocke.core.ir_serialize import serialize
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    if reduce:
        kernel = build_gett_f64_reduce(spec, arch=arch)
    else:
        kernel = build_gett_f64(spec, arch=arch)
    if engine == "python":
        ll = _lower_kernel_to_llvm_python(kernel, arch=arch)
    else:
        import rocke_engine as native

        ll = native.lower_serialized_ir(serialize(kernel), arch=arch)
    blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    return kernel.name, blob


def run(
    spec: GettF64Spec,
    arch: str,
    A,
    B,
    C,
    D,
    alpha,
    beta,
    *,
    engine="python",
    problem: Optional[GettF64Problem] = None,
    compiled=None,
    launches: int = 1,
):
    """Compile (unless ``compiled`` is given) and launch one GETT kernel on flat
    host buffers; returns D. A ``runtime_sizes`` kernel needs ``problem``.
    ``split_k``: the partial-sum kernel into a workspace, then the reduce
    kernel (``compiled`` then holds both); with ``split_serial`` one kernel
    with the workspace and the arrival counters (zeroed once), launched
    ``launches`` times (every launch must leave the counters at zero)."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    serial = getattr(spec, "split_serial", 0)
    split = spec.split_k > 1 and not serial
    if compiled is None:
        compiled = compile_kernel(spec, arch, engine=engine)
        if split:
            compiled = (
                compiled,
                compile_kernel(spec, arch, engine=engine, reduce=True),
            )
    if split:
        (name, blob), (rname, rblob) = compiled
    else:
        name, blob = compiled
    rt = Runtime()
    module = rt.load_module(blob)
    rmodule = rt.load_module(rblob) if split else None
    wptr = None
    names = ["A", "B"] + (["C"] if spec.bilinear else []) + ["D"]
    bufs = [A, B] + ([C] if spec.bilinear else []) + [D]
    ptrs = []
    try:
        for arr in bufs:
            arr = np.ascontiguousarray(arr)
            p = rt.alloc(max(arr.nbytes, 16))
            ptrs.append((p, arr))
            rt.memcpy_h2d(p, (ctypes.c_ubyte * arr.nbytes).from_buffer(arr), arr.nbytes)
        vals = {n: p for n, (p, _) in zip(names, ptrs)}
        vals.update(_scalars(spec, alpha, beta))
        if spec.runtime_sizes:
            vals.update(gett_f64_runtime_args(spec, problem))
        if split or serial:
            nbytes = gett_f64_workspace_bytes(spec, problem)
            wptr = rt.alloc(max(nbytes, 16))
            vals["W"] = wptr
        if serial:
            from rocke.instances.common.gett_f64 import gett_f64_counter_count

            zeros = np.zeros(max(gett_f64_counter_count(spec, problem), 4), np.int32)
            cptr = rt.alloc(zeros.nbytes)
            ptrs.insert(0, (cptr, zeros))  # freed with the buffers; D stays last
            rt.memcpy_h2d(
                cptr, (ctypes.c_ubyte * zeros.nbytes).from_buffer(zeros), zeros.nbytes
            )
            vals["Cnt"] = cptr
        args = pack_args(gett_f64_signature(spec), vals)
        for _ in range(launches if serial else 1):
            rt.launch_blocking(
                module.get_function(name),
                gett_f64_grid(spec, problem),
                (spec.threads, 1, 1),
                args,
            )
        if serial:
            back = np.empty_like(zeros)
            rt.memcpy_d2h(
                (ctypes.c_ubyte * back.nbytes).from_buffer(back), cptr, back.nbytes
            )
            assert not back.any(), "split_serial left a counter non-zero"
        if split:
            rt.launch_blocking(
                rmodule.get_function(rname),
                gett_f64_reduce_grid(spec, problem),
                (REDUCE_THREADS, 1, 1),
                pack_args(gett_f64_reduce_signature(spec), vals),
            )
        dp, darr = ptrs[-1]
        out = np.empty_like(darr)
        rt.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), dp, out.nbytes)
        return out
    finally:
        for p, _ in ptrs:
            rt.free(p)
        if wptr is not None:
            rt.free(wptr)
        module.unload()
        if rmodule is not None:
            rmodule.unload()


def run_plan(plan, arch, A, B, C, D, alpha, beta, *, bilinear=True, **knobs):
    """Launch every planned launch (pointer offsets per launch); returns D."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
    from rocke.instances.common.gett_f64 import arch_spec
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    GA, GB = (B, A) if plan.swapped else (A, B)
    bufs = [GA, GB] + ([C] if bilinear else []) + [D]
    ptrs = []
    modules = []
    try:
        for arr in bufs:
            arr = np.ascontiguousarray(arr)
            p = rt.alloc(max(arr.nbytes, 16))
            ptrs.append((p, arr))
            rt.memcpy_h2d(p, (ctypes.c_ubyte * arr.nbytes).from_buffer(arr), arr.nbytes)
        for ln in plan.launches:
            spec = arch_spec(ln.problem, arch, bilinear=bilinear, **knobs)
            kernel = build_gett_f64(spec, arch=arch)
            ll = _lower_kernel_to_llvm_python(kernel, arch=arch)
            blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
            module = rt.load_module(blob)
            modules.append(module)
            es = ln.problem.elem_bytes
            offs = (
                [ln.a_offset, ln.b_offset]
                + ([ln.c_offset] if bilinear else [])
                + [ln.d_offset]
            )
            args = struct.pack(
                f"{len(ptrs)}Q", *[p + o * es for (p, _), o in zip(ptrs, offs)]
            )
            cx = ln.problem.complex_
            al = [alpha.real, alpha.imag] if cx else [float(alpha)]
            be = ([beta.real, beta.imag] if cx else [float(beta)]) if bilinear else []
            args += struct.pack(f"{len(al) + len(be)}d", *al, *be)
            rt.launch_blocking(
                module.get_function(kernel.name),
                gett_f64_grid(spec),
                (spec.threads, 1, 1),
                args,
            )
        dp, darr = ptrs[-1]
        out = np.empty_like(darr)
        rt.memcpy_d2h((ctypes.c_ubyte * out.nbytes).from_buffer(out), dp, out.nbytes)
        return out
    finally:
        for p, _ in ptrs:
            rt.free(p)
        for m in modules:
            m.unload()


def random_buffers(problem: GettF64Problem, rng, *, integer=False, bilinear=True):
    def buf(which):
        n = span_elems(problem, which) * (2 if problem.complex_ else 1)
        if integer:
            return rng.integers(-8, 9, n).astype(np.float64)
        return rng.uniform(-1.0, 1.0, n)

    A, B = buf("A"), buf("B")
    C = buf("C") if bilinear else None
    D = np.full(span_elems(problem, "D") * (2 if problem.complex_ else 1), np.nan)
    return A, B, C, D


def check(problem, A, B, C, D_out, D_in, alpha, beta, samples):
    """Max error/bound over the samples, and that unwritten D slots stayed put."""
    Dm = gather(problem, "D", D_out)
    worst = 0.0
    for (l, m, n), val, bound in reference(problem, A, B, C, alpha, beta, samples):
        got = complex(Dm[l, m, n]) if problem.complex_ else float(Dm[l, m, n])
        err = abs(got - val)
        if not (err <= bound):
            return math.inf, ((l, m, n), got, val, bound)
        worst = max(worst, err / bound if bound else 0.0)
    return worst, None
