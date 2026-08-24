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

#: Target magnitude of the cumulative gate over the whole sequence. Keeps
#: exp(gate) in a range where the state decay is actually observable.
GATE_TOTAL = 2.5

LOG2E = 1.4426950408889634


def _sig(spec):
    """Kernel signature for `spec` — varlen adds cu_seqlens/chunk_offsets/T_flat,
    and the SSM-state dtype tracks STATE_DTYPE_BF16."""
    st = "bf16" if spec.STATE_DTYPE_BF16 else "f32"
    out = "bf16" if spec.OUTPUT_DTYPE_BF16 else "f32"
    sig = [
        {"name": "Kt", "type": "ptr<bf16,global>"},
        {"name": "Wt", "type": "ptr<bf16,global>"},
        {"name": "Ut", "type": "ptr<bf16,global>"},
        {"name": "Gate", "type": "ptr<f32,global>"},
        {"name": "H0", "type": f"ptr<{st},global>"},
        {"name": "Vnew", "type": f"ptr<{out},global>"},
        {"name": "Hout", "type": f"ptr<{out},global>"},
        {"name": "Ht", "type": f"ptr<{st},global>"},
        {"name": "T_val", "type": "i32"},
        {"name": "NT_val", "type": "i32"},
        {"name": "N_val", "type": "i32"},
    ]
    if spec.IS_VARLEN:
        sig += [
            {"name": "cu_seqlens", "type": "ptr<i32,global>"},
            {"name": "chunk_offsets", "type": "ptr<i32,global>"},
            {"name": "T_flat", "type": "i32"},
        ]
    return sig


#: (label, seqlens, H, BV, NR, gate, extra-spec-kwargs). ``seqlens`` is the list
#: of per-sequence token counts; a single-element list is the non-varlen case
#: unless the config sets IS_VARLEN. Non-multiple-of-BT lengths exercise the
#: tail-chunk row mask (numerics item N2).
CONFIGS = [
    # -- base coverage: non-varlen, token-major, natural-log, f32 state --
    ("gk-1chunk",     [64],  4, 32, 1, "gk", {}),
    ("gk-3chunks",    [192], 4, 32, 1, "gk", {}),
    ("gk-tail",       [160], 4, 32, 1, "gk", {}),
    ("gk-tail-short", [96],  4, 32, 1, "gk", {}),
    ("gk-bv16",       [192], 4, 16, 1, "gk", {}),
    ("gk-bv64",       [192], 4, 64, 1, "gk", {}),
    ("gk-bv64-spl2",  [192], 4, 64, 2, "gk", {}),
    ("gk-bv32-spl2",  [192], 4, 32, 2, "gk", {}),
    ("gk-heads12",    [128], 12, 32, 1, "gk", {}),
    ("g-1chunk",      [64],  4, 32, 1, "g", {}),
    ("g-3chunks",     [192], 4, 32, 1, "g", {}),
    ("g-tail",        [160], 4, 32, 1, "g", {}),
    ("g-bv64-spl2",   [192], 4, 64, 2, "g", {}),
    ("g-gqa2",        [192], 8, 32, 1, "g", {}),
    # -- new features --
    ("gk-varlen",     [256, 128, 192], 4, 32, 1, "gk", dict(IS_VARLEN=True)),
    ("g-varlen",      [256, 128, 192], 4, 32, 1, "g",  dict(IS_VARLEN=True)),
    ("gk-varlen-tail",[160, 96, 224],  4, 32, 1, "gk", dict(IS_VARLEN=True)),
    ("gk-wu-contig",  [192], 4, 32, 1, "gk", dict(WU_CONTIGUOUS=True)),
    ("g-wu-contig",   [192], 4, 32, 1, "g",  dict(WU_CONTIGUOUS=True)),
    ("gk-log2",       [192], 4, 32, 1, "gk", dict(G_IS_LOG2_SCALED=True)),
    ("g-log2",        [192], 4, 32, 1, "g",  dict(G_IS_LOG2_SCALED=True)),
    ("gk-state-bf16", [192], 4, 32, 1, "gk", dict(STATE_DTYPE_BF16=True)),
    ("gk-varlen-wu",  [256, 192], 4, 32, 1, "gk",
     dict(IS_VARLEN=True, WU_CONTIGUOUS=True)),
]


class Case:
    """Everything needed to launch and check one shape: compiled kernel,
    packed input dict, launch config, and per-sequence references. Shared by
    :func:`run_one` (correctness) and the benchmark driver."""

    def __init__(self, spec, kern, art, ln, vals, cfg, refs, dims):
        self.spec = spec
        self.kern = kern
        self.art = art
        self.ln = ln
        self.vals = vals
        self.cfg = cfg
        self.refs = refs          # (h_ref_l, vn_ref_l, fs_ref_l)
        self.dims = dims          # dict of N, T_flat, NT_list, NT_total, H, V, K

    def launch(self):
        self.ln(values=self.vals, config=self.cfg)


def build_case(label, seqlens, H, BV, NR, gate="gk", extra=None, *,
               K=128, V=128, arch="gfx942", seed=0, Hg=None,
               vgpr_form=False, swizzle=True, prefetch=False,
               prefetch_w_early=False,
               prefetch_k_early=False,
               prefetch_k_interleave=False,
               buffer_desc=False, xcd_remap=False, fence=True,
               want_refs=True):
    """Build + compile the kernel and pack inputs for one shape.

    ``Hg`` defaults to ``H`` (no GQA); pass it explicitly for GQA shapes. The
    legacy ``g-gqa2`` config keeps its H//2 shorthand.
    """
    torch.manual_seed(seed)
    BT = 64
    extra = dict(extra or {})
    use_gk = gate == "gk"
    if Hg is None:
        Hg = H // 2 if (label in ("g-gqa2",)) else H
    varlen = extra.get("IS_VARLEN", False)

    # base harness layout, unless the config's `extra` overrides it
    layout = dict(IS_VARLEN=False, WU_CONTIGUOUS=False,
                  G_IS_LOG2_SCALED=False, STATE_DTYPE_BF16=False,
                  OUTPUT_DTYPE_BF16=False)
    layout.update(extra)
    spec = GdnStateScanSpec(
        K=K, V=V, BV=BV, H=H, Hg=Hg, NR_SPLIT=NR,
        USE_G=not use_gk, USE_GK=use_gk, arch=arch,
        MFMA_VGPR_FORM=vgpr_form, LDS_SWIZZLE=swizzle, PREFETCH=prefetch,
        PREFETCH_W_EARLY=prefetch_w_early,
        PREFETCH_K_EARLY=prefetch_k_early,
        PREFETCH_K_INTERLEAVE=prefetch_k_interleave,
        BUFFER_DESC=buffer_desc, XCD_REMAP=xcd_remap,
        **layout)

    kern = build_k5(spec)
    art = compile_kernel(kern, isa=f"amdgcn-amd-amdhsa--{arch}")
    dev = "cuda"

    N = len(seqlens)
    T_flat = sum(seqlens)
    NT_list = [-(-s // BT) for s in seqlens]
    NT_total = sum(NT_list)
    # For the non-varlen path the kernel treats grid.y as N sequences of T_val
    # each; the harness only uses N=1 there.
    if not varlen:
        assert N == 1, f"{label}: non-varlen config needs a single sequence"

    # -- per-sequence inputs, then pack into the layout the spec asks for -----
    # Reference runs per sequence on unpacked [1, s, ...] tensors; the kernel
    # reads packed tensors. We build both from the same random draw.
    seg_k, seg_w, seg_u, seg_g, seg_gk, seg_h0 = [], [], [], [], [], []
    for s in seqlens:
        step = GATE_TOTAL / max(s, 1)
        seg_k.append((torch.randn(1, s, Hg, K, device=dev) * 0.1).bfloat16())
        seg_w.append((torch.randn(1, s, H, K, device=dev) * 0.1).bfloat16())
        seg_u.append((torch.randn(1, s, H, V, device=dev) * 0.1).bfloat16())
        seg_h0.append(torch.randn(1, H, V, K, device=dev) * 0.01)
        if use_gk:
            seg_gk.append((torch.randn(s, H, K, device=dev).abs()
                           * -step).cumsum(0).contiguous())
        else:
            seg_g.append((torch.randn(H, s, device=dev).abs()
                          * -step).cumsum(1).contiguous())

    # -- reference (fp32), per sequence, concatenated ------------------------
    h_ref_l, vn_ref_l, fs_ref_l = [], [], []
    if want_refs:
        for i, s in enumerate(seqlens):
            hr, vr, fr = ref_chunk_gated_delta_rule_fwd_h(
                seg_k[i], seg_w[i], seg_u[i],
                g=(None if use_gk else seg_g[i]),
                gk=(seg_gk[i] if use_gk else None),
                initial_state=seg_h0[i], output_final_state=True, chunk_size=BT)
            h_ref_l.append(hr); vn_ref_l.append(vr); fs_ref_l.append(fr)

    # -- pack kernel inputs --------------------------------------------------
    st_dt = torch.bfloat16 if spec.STATE_DTYPE_BF16 else torch.float32
    out_dt = torch.bfloat16 if spec.OUTPUT_DTYPE_BF16 else torch.float32
    # k is always token-major packed [1, T_flat, Hg, K]
    Kt = torch.cat([x.reshape(1, sl, Hg, K) for x, sl in zip(seg_k, seqlens)], dim=1)
    if spec.WU_CONTIGUOUS:
        # head-major: varlen [i_h, T_flat, *] packed per head; non-varlen [1,H,T,*]
        Wt = torch.cat([x.permute(0, 2, 1, 3).reshape(H, sl, K)
                        for x, sl in zip(seg_w, seqlens)], dim=1).reshape(-1).contiguous()
        Ut = torch.cat([x.permute(0, 2, 1, 3).reshape(H, sl, V)
                        for x, sl in zip(seg_u, seqlens)], dim=1).reshape(-1).contiguous()
    else:
        Wt = torch.cat([x.reshape(1, sl, H, K) for x, sl in zip(seg_w, seqlens)],
                       dim=1).reshape(-1).contiguous()
        Ut = torch.cat([x.reshape(1, sl, H, V) for x, sl in zip(seg_u, seqlens)],
                       dim=1).reshape(-1).contiguous()
    if use_gk:
        Gate = torch.cat(seg_gk, dim=0).contiguous()          # [T_flat, H, K]
    else:
        Gate = torch.cat(seg_g, dim=1).contiguous()           # [H, T_flat]
    if spec.G_IS_LOG2_SCALED:
        Gate = (Gate * LOG2E).contiguous()
    H0 = torch.cat([x for x in seg_h0], dim=0).to(st_dt)      # [N, H, V, K]

    Vn = torch.zeros(T_flat * H * V, dtype=out_dt, device=dev)
    Ho = torch.zeros(NT_total * H * V * K, dtype=out_dt, device=dev)
    Ht = torch.zeros(N * H * V * K, dtype=st_dt, device=dev)

    vals = {"Kt": Kt.reshape(-1).contiguous(), "Wt": Wt, "Ut": Ut,
            "Gate": Gate.reshape(-1).contiguous(), "H0": H0.reshape(-1).contiguous(),
            "Vnew": Vn, "Hout": Ho, "Ht": Ht,
            "T_val": (seqlens[0] if not varlen else T_flat),
            "NT_val": NT_list[0], "N_val": N}
    if varlen:
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
                          dtype=torch.int32, device=dev)
        co = torch.tensor([0] + list(torch.cumsum(torch.tensor(NT_list[:-1]), 0)),
                          dtype=torch.int32, device=dev)
        vals["cu_seqlens"] = cu
        vals["chunk_offsets"] = co
        vals["T_flat"] = T_flat

    ln = KernelLauncher(hsaco=art.hsaco, kernel_name=kern.name, signature=_sig(spec))
    cfg = LaunchConfig(grid=spec.grid(N * H), block=(spec.block_threads, 1, 1),
                       stream=int(torch.cuda.current_stream().cuda_stream),
                       fence=fence)
    dims = dict(N=N, T_flat=T_flat, NT_list=NT_list, NT_total=NT_total,
                H=H, V=V, K=K, Vn=Vn, Ho=Ho, Ht=Ht)
    return Case(spec, kern, art, ln, vals, cfg,
                (h_ref_l, vn_ref_l, fs_ref_l), dims)


def run_one(label, seqlens, H, BV, NR, gate="gk", extra=None, *,
            K=128, V=128, arch="gfx942", seed=0, verbose=False,
            vgpr_form=False, swizzle=True, prefetch=False,
            prefetch_w_early=False,
            prefetch_k_early=False,
            prefetch_k_interleave=False,
            buffer_desc=False, xcd_remap=False):
    case = build_case(label, seqlens, H, BV, NR, gate, extra, K=K, V=V,
                      arch=arch, seed=seed, vgpr_form=vgpr_form, swizzle=swizzle,
                      prefetch=prefetch, prefetch_w_early=prefetch_w_early,
                      prefetch_k_early=prefetch_k_early,
                      prefetch_k_interleave=prefetch_k_interleave,
                      buffer_desc=buffer_desc,
                      xcd_remap=xcd_remap, fence=True, want_refs=True)
    spec = case.spec
    case.launch()
    torch.cuda.synchronize()

    d = case.dims
    N, T_flat, NT_list, NT_total = d["N"], d["T_flat"], d["NT_list"], d["NT_total"]
    H, V, K = d["H"], d["V"], d["K"]
    Vn, Ho, Ht = d["Vn"], d["Ho"], d["Ht"]
    h_ref_l, vn_ref_l, fs_ref_l = case.refs
    spec = case.spec
    BV, NR = spec.BV, spec.NR_SPLIT

    # -- compare, per sequence ----------------------------------------------
    ok = True
    parts = []
    Ho_v = Ho.reshape(NT_total, H, V, K)
    Ht_v = Ht.float().reshape(N, H, V, K)
    # v_new is emitted in the SAME layout as u: token-major [T_flat, H, V] by
    # default, head-major [H, T_flat, V] under WU_CONTIGUOUS (matching the
    # parent, whose vn_base tracks the u layout).
    if spec.WU_CONTIGUOUS:
        Vn_hm = Vn.reshape(H, T_flat, V)          # [H, T_flat, V]
    else:
        Vn_v = Vn.reshape(T_flat, H, V)           # [T_flat, H, V]
    chunk0, tok0 = 0, 0
    worst = {"h": 0.0, "vn": 0.0, "fs": 0.0}
    for i, s in enumerate(seqlens):
        nt = NT_list[i]
        got_h = Ho_v[chunk0:chunk0 + nt].unsqueeze(0)
        if spec.WU_CONTIGUOUS:
            got_vn = Vn_hm[:, tok0:tok0 + s].permute(1, 0, 2).unsqueeze(0)
        else:
            got_vn = Vn_v[tok0:tok0 + s].unsqueeze(0)
        got_fs = Ht_v[i:i + 1]
        for nm, g, r in (("h", got_h, h_ref_l[i]),
                         ("vn", got_vn, vn_ref_l[i]),
                         ("fs", got_fs, fs_ref_l[i])):
            rel = (g.float() - r.float()).abs().max().item() / max(
                r.float().abs().max().item(), 1e-9)
            worst[nm] = max(worst[nm], rel)
            ok &= torch.allclose(g.float(), r.float(), atol=ATOL, rtol=RTOL)
        chunk0 += nt
        tok0 += s
    parts = [f"{nm}={worst[nm]:.1e}{'' if worst[nm] < ATOL else '!'}"
             for nm in ("h", "vn", "fs")]
    print(f"  {label:16s} seqs={seqlens} H={H} BV={BV} NR={NR} thr={spec.block_threads:4d}"
          f"  rel[{' '.join(parts)}]  {'PASS' if ok else 'FAIL'}")
    if verbose and not ok:
        print(f"     kernel={case.kern.name}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--only", default=None, help="run a single config by label")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--prefetch", action="store_true",
                    help="enable loop-carried prefetch (PERF_PLAN P5)")
    ap.add_argument(
        "--prefetch-w-early",
        action="store_true",
        help="issue next-chunk w reads before GEMM1 (requires --prefetch)",
    )
    ap.add_argument(
        "--prefetch-k-early",
        action="store_true",
        help="load current-chunk K rows before the first barrier",
    )
    ap.add_argument(
        "--prefetch-k-interleave",
        action="store_true",
        help="interleave current-chunk K row loads across GEMM1",
    )
    ap.add_argument("--buffer-desc", action="store_true",
                    help="global loads via bounds-checked descriptors (P6)")
    ap.add_argument("--xcd-remap", action="store_true",
                    help="chiplet remap of the flat block id (P7)")
    ap.add_argument("--no-swizzle", action="store_true",
                    help="disable the LDS XOR swizzle (padded fallback)")
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
                       vgpr_form=args.vgpr_form,
                       swizzle=not args.no_swizzle,
                       prefetch=args.prefetch,
                       prefetch_w_early=args.prefetch_w_early,
                       prefetch_k_early=args.prefetch_k_early,
                       prefetch_k_interleave=args.prefetch_k_interleave,
                       buffer_desc=args.buffer_desc,
                       xcd_remap=args.xcd_remap) for c in configs]
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
