#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
r"""BF16 baseline side: M-sweep for AITER's warp-per-scalar wvSplitK* family.

Mirrors test/ck_tile/gemm_decode/bench_msweep.cpp (gemm_decode) and
flydsl_msweep.py (the MFMA M>=5 ceiling) on the *VALU* small-M side so all three
CSVs can be joined on (impl, M, N, K) by r1_compare.py. This closes the gap the
crossover work left open: gemm_decode claims to subsume the wvSplitK* VALU
kernels (design doc section 16), but that M<=4 peer comparison was never run.

Kernels benched (all warp-per-scalar, BF16/FP16, K%8==0):
  - wvSpltK                    (the fast skinny GEMM; kernel enforces M<=4)
  - wv_splitk_small_fp16_bf16  (covers M=1..16, but ~4x slower at M<=4)
  - LLMM1                      (M=1 only; --extra-kernels, convention differs)

For each M we also emit a synthesized "best of family" row (impl=aiter_wvsplitk)
that r1_compare.py joins against gemm_decode_best: wvSpltK wins M<=4, and
wv_splitk_small extends the curve to M=5..16 where wvSpltK is unsupported.

Convention (matches flydsl_msweep.py): A=(M,K) activation, B=(N,K) weight,
C = A @ B.T = (M,N). The AITER call is weight-first and passes the activation
row count as the 4th arg:
    wvSpltK(B, A, out, M, cu_count)

The kernels live in AITER's `module_custom` pybind extension. We load the
prebuilt .so directly (importlib) so this harness does NOT depend on the
top-level `aiter` package importing cleanly (the MX-quant eager import is broken
on some branches, but module_custom is self-contained). A `--so` override and an
`import aiter` fallback are provided.

  /opt/venv/bin/python3 wvsplitk_msweep.py \
    --N 8192 --K 7168 --mmax 16 --warmup 10 --repeat 100 \
    --csv-out /tmp/wvsplitk_msweep_8192x7168.csv

Emits CSV columns: impl,M,N,K,time_us,tflops,gbytes_s,config
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback

DEFAULT_SO = "/home/AMD/samremes/dev/aiter/aiter/jit/module_custom.so"

# AITER custom-kernel entry points, in family order. Each entry is
# (kernel_attr, impl_label, supports_M_predicate). The predicate is an upper
# bound on what we *attempt*; the kernel itself rejects unsupported (N, M) cells
# at runtime ("Unsupported N value: ..."), which we catch and skip.
_FAMILY = [
    ("wvSpltK", "aiter_wvspltk", lambda m: 1 <= m <= 16),
    ("wv_splitk_small_fp16_bf16", "aiter_wvsplitk_small", lambda m: 1 <= m <= 16),
    ("LLMM1", "aiter_llmm1", lambda m: m == 1),
]

# impl label of the synthesized per-M family best (what r1_compare.py joins on).
BEST_IMPL = "aiter_wvsplitk"


def _load_custom(so_path: str):
    """Load AITER's module_custom pybind extension.

    Primary path: load the prebuilt .so directly (no `aiter` package import, so
    an unrelated broken eager import in aiter/__init__.py cannot block us).
    Fallback: import the aiter package and pull the ops off it.
    """
    import torch  # noqa: F401 - ensures libtorch symbols are resolved first

    if so_path and os.path.exists(so_path):
        spec = importlib.util.spec_from_file_location("module_custom", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "wv_splitk_small_fp16_bf16"):
            print(f"# loaded module_custom.so directly: {so_path}", file=sys.stderr)
            return mod
        print(f"# {so_path} missing wv_splitk_small_fp16_bf16; trying aiter pkg",
              file=sys.stderr)

    import aiter  # noqa: E402 - fallback only
    print(f"# using aiter package: {aiter.__file__}", file=sys.stderr)
    return aiter


def _kernel_callable(mod, attr):
    """Return a uniform `call(mod, B, A, out, m, cu)` for kernel `attr`, or None.

    All wvSplitK-family entry points are weight-first. LLMM1 takes a
    rows_per_block arg instead of (N_in, cu) and only supports M==1.
    """
    fn = getattr(mod, attr, None)
    if fn is None:
        return None
    if attr == "LLMM1":
        return lambda _mod, B, A, out, m, cu: fn(B, A, out, 4)
    return lambda _mod, B, A, out, m, cu: fn(B, A, out, m, cu)


def _import_aiter(aiter_dir: str):
    """Import the top-level `aiter` package for the FP8 path.

    wvSplitKQ / gemm_a8w8_CK / per_tensor_quant are exposed only via the package
    (not module_custom alone). On some branches aiter.utility.mx_types eager-imports
    MxScaleRoundMode/MxDtype from a stale module_aiter_core.so and raises; those
    enums are irrelevant to per-tensor FP8, so we splice in the int-mirror fallbacks
    (MxScaleRoundModeInt/MxDtypeInt) via a one-module meta-path shim. This avoids
    forcing an AITER_REBUILD just to bench the FP8 competitors.
    """
    import importlib.abc

    if aiter_dir and aiter_dir not in sys.path:
        sys.path.insert(0, aiter_dir)

    class _Wrap(importlib.abc.Loader):
        def __init__(self, real):
            self.real = real

        def create_module(self, spec):
            return self.real.create_module(spec)

        def exec_module(self, module):
            self.real.exec_module(module)
            for name in ("MxScaleRoundMode", "MxDtype"):
                if not hasattr(module, name) and hasattr(module, name + "Int"):
                    setattr(module, name, getattr(module, name + "Int"))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "aiter.utility.mx_types":
                return None
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    spec.loader = _Wrap(spec.loader)
                    return spec
            return None

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    import aiter  # noqa: E402
    print(f"# imported aiter: {aiter.__file__}", file=sys.stderr)
    return aiter


def run_fp8(args) -> int:
    """FP8 per-tensor M-sweep for the two AITER competitors.

    Same convention as the BF16 path and gemm_decode's FP8 sweep
    (A=(M,K) act, B=(N,K) weight, C = A @ B.T = (M,N)):
      - wvSplitKQ      VALU warp-per-scalar; the FP8 peer of wvSpltK (kernel M<=4).
                       AITER call is weight-first with (w_scale, x_scale):
                           wvSplitKQ(Bq, Aq, out, w_scale, x_scale, cu)
      - gemm_a8w8_CK   classic-CK MFMA per-tensor fallback (the MFMA ceiling that
                       isn't M=16-locked like CKTile gemm_quant). First call JIT-
                       builds module_gemm_a8w8 (slow, one-time).
    Per M we also emit aiter_fp8 = the faster of the two -- the headline the 4-way
    FP8 compare joins gemm_decode_fp8_best against.

    Bytes model matches bench_msweep_fp8.cpp: fp8 A,B = 1 B/elem, bf16 C = 2 B/elem.
    """
    import torch

    aiter = _import_aiter(args.aiter_dir)
    from aiter import dtypes

    dev = torch.device("cuda")
    fp8 = dtypes.fp8
    cu = args.cu or torch.cuda.get_device_properties(0).multi_processor_count
    N, K, Mmax = args.N, args.K, args.mmax
    print(f"# AITER FP8 per-tensor M-sweep: fp8={fp8} cu={cu} N={N} K={K} "
          f"Mmax={Mmax} warmup={args.warmup} repeat={args.repeat}", file=sys.stderr)
    if K % 8 != 0:
        print(f"# WARNING: K={K} not %8==0; wvSplitKQ requires it", file=sys.stderr)

    Abf = torch.randn((Mmax, K), dtype=torch.bfloat16, device=dev)
    Bbf = torch.randn((N, K), dtype=torch.bfloat16, device=dev) * 0.1
    Aq, x_scale = aiter.per_tensor_quant(Abf, quant_dtype=fp8)
    Bq, w_scale = aiter.per_tensor_quant(Bbf, quant_dtype=fp8)
    # Reference = dequant(Aq) @ dequant(Bq).T -- the value both kernels approximate
    # (NOT Abf @ Bbf.T: the fp8 round-trip is what each kernel actually computes).
    ref_full = (Aq.float() * x_scale.float()) @ (Bq.float() * w_scale.float()).t()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def _relerr(out, m):
        ref = ref_full[:m]
        denom = ref.abs().mean().clamp_min(1e-6)
        return ((out.float() - ref).abs().mean() / denom).item()

    # wvSplitKQ: preallocated bf16 out, weight-first, scales = (w_scale, x_scale).
    def _wvsplitkq_run(m):
        out = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
        aiter.wvSplitKQ(Bq, Aq[:m].contiguous(), out, w_scale, x_scale, cu)
        return out

    # gemm_a8w8_CK: returns a fresh (M,N) tensor. The accepted per-tensor scale
    # *shape* differs across AITER branches; resolve it once against the reference.
    _ck_forms = [
        ("pertensor[1]", lambda m: (x_scale, w_scale)),
        ("bcast[m,1]/[1,N]",
         lambda m: (x_scale.view(1, 1).expand(m, 1).contiguous(),
                    w_scale.view(1, 1).expand(1, N).contiguous())),
    ]
    _ck_state = {"form": None}

    def _gemm_a8w8_ck_run(m):
        Am = Aq[:m].contiguous()
        if _ck_state["form"] is not None:
            sa, sb = _ck_forms[_ck_state["form"]][1](m)
            return aiter.gemm_a8w8_CK(Am, Bq, sa, sb, None, torch.bfloat16)
        last_exc = None
        for i, (tag, mk) in enumerate(_ck_forms):
            try:
                sa, sb = mk(m)
                y = aiter.gemm_a8w8_CK(Am, Bq, sa, sb, None, torch.bfloat16)
                torch.cuda.synchronize()
                if _relerr(y, m) <= max(args.rtol, 1e-1):
                    _ck_state["form"] = i
                    print(f"#   gemm_a8w8_CK scale form = {tag}", file=sys.stderr)
                    return y
            except Exception as e:  # noqa: BLE001 - try the next scale form
                last_exc = e
                print(f"#   gemm_a8w8_CK form {tag} raised "
                      f"{type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("gemm_a8w8_CK: no scale form verified")

    competitors = [
        ("aiter_wvsplitkq", "wvSplitKQ", _wvsplitkq_run),
        ("aiter_gemm_a8w8_ck", "gemm_a8w8_CK", _gemm_a8w8_ck_run),
    ]

    rows = []
    best_per_m = {}  # m -> (t_us, tflops, gbps, cfg)
    for impl, label, run_m in competitors:
        print(f"# --- {impl} ({label}) ---", file=sys.stderr)
        for m in range(1, Mmax + 1):
            if not args.no_verify:
                try:
                    out = run_m(m)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001 - unsupported (kernel,M) cell
                    print(f"#   M={m}: run raised {type(e).__name__}: "
                          f"{str(e)[:160]}", file=sys.stderr)
                    continue
                rel = _relerr(out, m)
                if rel > args.rtol:
                    print(f"#   M={m}: rel_err {rel:.3e} > {args.rtol}; skip",
                          file=sys.stderr)
                    continue
            else:
                rel = float("nan")
            cfg = f"{label}/{_ck_forms[_ck_state['form']][0]}" \
                if (label == "gemm_a8w8_CK" and _ck_state["form"] is not None) \
                else f"{label}/cu{cu}"
            try:
                for _ in range(args.warmup):
                    run_m(m)
                torch.cuda.synchronize()
                start_evt.record()
                for _ in range(args.repeat):
                    run_m(m)
                end_evt.record()
                torch.cuda.synchronize()
                t_us = start_evt.elapsed_time(end_evt) * 1000.0 / args.repeat
            except Exception as e:  # noqa: BLE001
                print(f"#   M={m}: timing raised {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            tflops = 2.0 * m * N * K / (t_us * 1e-6) / 1e12
            gbps = (m * K + N * K + m * N * 2) / (t_us * 1e-6) / 1e9
            rows.append((impl, m, N, K, t_us, tflops, gbps, cfg))
            if m not in best_per_m or t_us < best_per_m[m][0]:
                best_per_m[m] = (t_us, tflops, gbps, cfg)
            print(f"#   M={m:2d}  {t_us:8.2f}us  {tflops:6.2f} TF/s  {gbps:7.1f} GB/s"
                  f"  rel={rel:.2e}  [{cfg}]", file=sys.stderr)

    if not rows:
        print("# no FP8 competitor cells succeeded; aborting", file=sys.stderr)
        return 1

    print("# --- aiter_fp8 (per-M best of {wvSplitKQ, gemm_a8w8_CK}) ---",
          file=sys.stderr)
    for m in sorted(best_per_m):
        t_us, tflops, gbps, cfg = best_per_m[m]
        rows.append(("aiter_fp8", m, N, K, t_us, tflops, gbps, cfg))
        print(f"#   M={m:2d}  {t_us:8.2f}us  {tflops:6.2f} TF/s  {gbps:7.1f} GB/s"
              f"  [{cfg}]", file=sys.stderr)

    with open(args.csv_out, "w") as f:
        f.write("impl,M,N,K,time_us,tflops,gbytes_s,config\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.2f},{r[7]}\n")
    print(f"# wrote {len(rows)} rows -> {args.csv_out}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--mmax", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--csv-out", default="/tmp/wvsplitk_msweep.csv")
    ap.add_argument("--so", default=DEFAULT_SO,
                    help="Path to AITER module_custom.so (direct load).")
    ap.add_argument("--cu", type=int, default=0,
                    help="CuCount override; 0 = use device multi_processor_count.")
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--fp8", action="store_true",
                    help="FP8 per-tensor mode: bench AITER wvSplitKQ (VALU peer) and "
                         "gemm_a8w8_CK (classic-CK MFMA); emit aiter_wvsplitkq / "
                         "aiter_gemm_a8w8_ck / aiter_fp8 (per-M best of the family).")
    ap.add_argument("--aiter-dir", default="/home/AMD/samremes/dev/aiter",
                    help="aiter checkout to import for --fp8 (prepended to sys.path).")
    ap.add_argument("--extra-kernels", action="store_true",
                    help="Also bench wvSpltK (all M) and LLMM1 (M=1).")
    ap.add_argument("--rtol", type=float, default=5e-2,
                    help="Max mean relative error to accept a (kernel,M) cell.")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.fp8:
        return run_fp8(args)

    import torch

    mod = _load_custom(args.so)
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cu = args.cu or torch.cuda.get_device_properties(0).multi_processor_count
    N, K, Mmax = args.N, args.K, args.mmax
    print(f"# AITER wvSplitK* M-sweep: dtype={args.dtype} cu={cu} N={N} K={K} "
          f"Mmax={Mmax} warmup={args.warmup} repeat={args.repeat}", file=sys.stderr)
    if K % 8 != 0:
        print(f"# WARNING: K={K} not %8==0; wvSplitK kernels require it", file=sys.stderr)

    # Persistent buffers sized for the largest M; smaller M use leading rows.
    A = torch.randn((Mmax, K), dtype=dt, device=dev)
    B = (torch.randn((N, K), dtype=dt, device=dev) * 0.1)
    ref_full = A.float() @ B.float().t()  # (Mmax, N); rows reused per M

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    kernels = ["wvSpltK", "wv_splitk_small_fp16_bf16"]
    if args.extra_kernels:
        kernels += ["LLMM1"]

    def verify(call, m):
        Am = A[:m].contiguous()
        out = torch.empty((m, N), dtype=dt, device=dev)
        call(mod, B, Am, out, m, cu)
        torch.cuda.synchronize()
        ref = ref_full[:m]
        denom = ref.abs().mean().clamp_min(1e-6)
        return ((out.float() - ref).abs().mean() / denom).item()

    def time_cell(call, m):
        Am = A[:m].contiguous()
        out = torch.empty((m, N), dtype=dt, device=dev)
        for _ in range(args.warmup):
            call(mod, B, Am, out, m, cu)
        torch.cuda.synchronize()
        start_evt.record()
        for _ in range(args.repeat):
            call(mod, B, Am, out, m, cu)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) * 1000.0 / args.repeat  # us

    rows = []
    best_per_m = {}  # m -> (t_us, tflops, gbps, cfg)
    for attr in kernels:
        impl = next(lbl for (a, lbl, _p) in _FAMILY if a == attr)
        pred = next(p for (a, _lbl, p) in _FAMILY if a == attr)
        call = _kernel_callable(mod, attr)
        if call is None:
            print(f"# kernel {attr} not present in module; skipping", file=sys.stderr)
            continue
        cfg = f"{attr}/cu{cu}"
        print(f"# --- {impl} ({attr}) ---", file=sys.stderr)
        for m in range(1, Mmax + 1):
            if not pred(m):
                continue
            if not args.no_verify:
                try:
                    rel = verify(call, m)
                except Exception as e:  # noqa: BLE001 - unsupported (kernel,M) cell
                    print(f"#   M={m}: verify raised {type(e).__name__}: {e}",
                          file=sys.stderr)
                    continue
                if rel > args.rtol:
                    print(f"#   M={m}: rel_err {rel:.3e} > {args.rtol}; skip",
                          file=sys.stderr)
                    continue
            else:
                rel = float("nan")
            try:
                t_us = time_cell(call, m)
            except Exception as e:  # noqa: BLE001
                print(f"#   M={m}: run raised {type(e).__name__}: {e}", file=sys.stderr)
                continue
            tflops = 2.0 * m * N * K / (t_us * 1e-6) / 1e12
            gbps = (m * K + N * K + m * N) * 2 / (t_us * 1e-6) / 1e9
            rows.append((impl, m, N, K, t_us, tflops, gbps, cfg))
            if m not in best_per_m or t_us < best_per_m[m][0]:
                best_per_m[m] = (t_us, tflops, gbps, cfg)
            print(f"#   M={m:2d}  {t_us:8.2f}us  {tflops:6.2f} TF/s  {gbps:7.1f} GB/s"
                  f"  rel={rel:.2e}  [{cfg}]", file=sys.stderr)

    if not rows:
        print("# no wvSplitK cells succeeded; aborting", file=sys.stderr)
        return 1

    # Synthesized per-M family best (the headline VALU baseline for r1_compare).
    print(f"# --- {BEST_IMPL} (per-M best of family) ---", file=sys.stderr)
    for m in sorted(best_per_m):
        t_us, tflops, gbps, cfg = best_per_m[m]
        rows.append((BEST_IMPL, m, N, K, t_us, tflops, gbps, cfg))
        print(f"#   M={m:2d}  {t_us:8.2f}us  {tflops:6.2f} TF/s  {gbps:7.1f} GB/s"
              f"  [{cfg}]", file=sys.stderr)

    with open(args.csv_out, "w") as f:
        f.write("impl,M,N,K,time_us,tflops,gbytes_s,config\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},"
                    f"{r[6]:.2f},{r[7]}\n")
    print(f"# wrote {len(rows)} rows -> {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
