# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric verification for the gated-delta-rule state-scan study kernel.

Builds, compiles, launches, and compares against the fp32 torch reference.

    PYTHONPATH=python ROCKE_BACKEND=python \
        python3 -m rocke.examples.gfx942.gdn_state_scan.verify

Correctness only — no timing. The expected residual is ~1e-3 relative: the
kernel stages the state through bf16 in LDS so GEMM1 consumes a rounded ``h``,
while the reference keeps ``h`` in fp32 end to end. That gap is inherent to the
algorithm, not a defect, and the tolerance below is the one the FlyDSL harness
uses for the same comparison.
"""

from __future__ import annotations

import argparse
import sys

import torch

from rocke.helpers import compile_kernel
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

from .builder import build_k5
from .reference import ref_chunk_gated_delta_rule_fwd_h
from .spec import GdnStateScanSpec

ATOL = RTOL = 5e-2

_SIG = [
    {"name": "Kt", "type": "ptr<bf16,global>"},
    {"name": "Wt", "type": "ptr<bf16,global>"},
    {"name": "Ut", "type": "ptr<bf16,global>"},
    {"name": "Gate", "type": "ptr<f32,global>"},
    {"name": "H0", "type": "ptr<f32,global>"},
    {"name": "Vnew", "type": "ptr<f32,global>"},
    {"name": "Hout", "type": "ptr<f32,global>"},
    {"name": "Ht", "type": "ptr<f32,global>"},
    {"name": "T_val", "type": "i32"},
    {"name": "NT_val", "type": "i32"},
]

#: (label, T, H, BV, NR_SPLIT, gate). ``T`` not a multiple of BT exercises the
#: tail-chunk row mask (numerics item N2), the most failure-prone part.
CONFIGS = [
    ("gk-1chunk",     64,  4, 32, 1, "gk"),
    ("gk-3chunks",    192, 4, 32, 1, "gk"),
    ("gk-tail",       160, 4, 32, 1, "gk"),   # 160 = 2*64 + 32
    ("gk-tail-short", 96,  4, 32, 1, "gk"),   # 96  = 1*64 + 32
    ("gk-bv16",       192, 4, 16, 1, "gk"),
    ("gk-bv64",       192, 4, 64, 1, "gk"),
    ("gk-bv64-spl2",  192, 4, 64, 2, "gk"),
    ("gk-bv32-spl2",  192, 4, 32, 2, "gk"),
    ("gk-heads12",    128, 12, 32, 1, "gk"),
    ("g-1chunk",      64,  4, 32, 1, "g"),
    ("g-3chunks",     192, 4, 32, 1, "g"),
    ("g-tail",        160, 4, 32, 1, "g"),
    ("g-bv64-spl2",   192, 4, 64, 2, "g"),
    ("g-gqa2",        192, 8, 32, 1, "g"),    # H=8, Hg=4 -> GQA ratio 2
]


def run_one(label, T, H, BV, NR, gate="gk", *, K=128, V=128, arch="gfx942",
            seed=0, verbose=False, vgpr_form=False):
    torch.manual_seed(seed)
    BT = 64
    use_gk = gate == "gk"
    # exercise GQA on one scalar-gate case
    Hg = H // 2 if (label == "g-gqa2") else H
    spec = GdnStateScanSpec(K=K, V=V, BV=BV, H=H, Hg=Hg, NR_SPLIT=NR,
                            USE_G=not use_gk, USE_GK=use_gk, arch=arch,
                            MFMA_VGPR_FORM=vgpr_form)
    kern = build_k5(spec)
    art = compile_kernel(kern, isa=f"amdgcn-amd-amdhsa--{arch}")

    dev = "cuda"
    k = (torch.randn(1, T, Hg, K, device=dev) * 0.1).bfloat16()
    w = (torch.randn(1, T, H, K, device=dev) * 0.1).bfloat16()
    u = (torch.randn(1, T, H, V, device=dev) * 0.1).bfloat16()
    h0 = torch.randn(1, H, V, K, device=dev) * 0.01
    NT = -(-T // BT)

    if use_gk:
        gate_t = (torch.randn(T, H, K, device=dev).abs() * -0.1).cumsum(0).contiguous()
        gk_arg, g_arg = gate_t, None
    else:
        # head-major [H, T], non-positive, cumulative along T
        gate_t = (torch.randn(H, T, device=dev).abs() * -0.5).cumsum(1).contiguous()
        gk_arg, g_arg = None, gate_t

    h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g_arg, gk=gk_arg, initial_state=h0,
        output_final_state=True, chunk_size=BT)

    Vn = torch.zeros(1, T, H, V, dtype=torch.float32, device=dev)
    Ho = torch.zeros(1, NT, H, V, K, dtype=torch.float32, device=dev)
    Ht = torch.zeros(1, H, V, K, dtype=torch.float32, device=dev)

    ln = KernelLauncher(hsaco=art.hsaco, kernel_name=kern.name, signature=_SIG)
    ln(values={"Kt": k, "Wt": w, "Ut": u, "Gate": gate_t, "H0": h0,
               "Vnew": Vn, "Hout": Ho, "Ht": Ht, "T_val": T, "NT_val": NT},
       config=LaunchConfig(grid=spec.grid(H), block=(spec.block_threads, 1, 1),
                           stream=int(torch.cuda.current_stream().cuda_stream),
                           fence=True))
    torch.cuda.synchronize()

    worst, ok = 0.0, True
    parts = []
    for nm, got, ref in (("h", Ho, h_ref), ("vn", Vn, vn_ref), ("fs", Ht, fs_ref)):
        g, r = got.float(), ref.float()
        err = (g - r).abs().max().item()
        rel = err / max(r.abs().max().item(), 1e-9)
        worst = max(worst, rel)
        good = torch.allclose(g, r, atol=ATOL, rtol=RTOL)
        ok &= good
        parts.append(f"{nm}={rel:.1e}{'' if good else '!'}")
    print(f"  {label:14s} T={T:4d} H={H:2d} BV={BV:3d} NR={NR} thr={spec.block_threads:4d}"
          f"  rel[{' '.join(parts)}]  {'PASS' if ok else 'FAIL'}")
    if verbose and not ok:
        print(f"     kernel={kern.name}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--only", default=None, help="run a single config by label")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--vgpr-form", action="store_true",
                    help="force VGPR-form MFMA (PERF_PLAN P0)")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("no GPU; skipping")
        return 0

    configs = [c for c in CONFIGS if args.only in (None, c[0])]
    if not configs:
        print(f"no config named {args.only!r}; have: {[c[0] for c in CONFIGS]}")
        return 2

    print(f"gdn_state_scan verify — arch={args.arch}, tol atol={ATOL} rtol={RTOL}")
    results = [run_one(*c, arch=args.arch, verbose=args.verbose,
                       vgpr_form=args.vgpr_form) for c in configs]
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
