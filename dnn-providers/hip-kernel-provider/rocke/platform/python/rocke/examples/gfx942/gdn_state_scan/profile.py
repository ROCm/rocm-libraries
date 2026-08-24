# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch one K5 state-scan variant for rocprofv3 / WaveScope capture.

Unlike :mod:`benchmark`, this entry point builds no torch reference and compiles
no variants other than the requested one.  That keeps a profiler capture focused
on one kernel dispatch and avoids collecting unrelated torch code objects.

Example::

    ROCKE_DEBUG_LOC=1 python3 -m \
      rocke.examples.gfx942.gdn_state_scan.profile --shape 1 --launches 2
"""

from __future__ import annotations

import argparse

import torch

from .benchmark import SHAPES, _split
from .verify import build_case


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", type=int, choices=range(1, 5), required=True)
    ap.add_argument("--bv", type=int, choices=(16, 32, 64), default=None)
    ap.add_argument("--nr-split", type=int, choices=(1, 2), default=None)
    ap.add_argument("--launches", type=int, default=2)
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--prefetch-w-early", action="store_true")
    ap.add_argument("--prefetch-k-early", action="store_true")
    ap.add_argument("--prefetch-k-interleave", action="store_true")
    ap.add_argument("--no-buffer-desc", action="store_true")
    ap.add_argument("--no-xcd-remap", action="store_true")
    ap.add_argument("--no-swizzle", action="store_true")
    ap.add_argument("--wu-contiguous", action="store_true")
    ap.add_argument("--output-bf16", action="store_true")
    args = ap.parse_args(argv)

    rec = {s[0]: s for s in SHAPES}[args.shape]
    _, name, h, hg, total_t, n_seq, gate, kind = rec
    seqlens = _split(total_t, n_seq, kind)
    extra = {
        "IS_VARLEN": n_seq > 1,
        "WU_CONTIGUOUS": args.wu_contiguous,
        "OUTPUT_DTYPE_BF16": args.output_bf16,
    }

    # The defaults are the stable full-stack winners for the two launch
    # geometries represented by shapes 1-4.  Explicit CLI values always win.
    bv = args.bv if args.bv is not None else (16 if n_seq == 1 else 32)
    nr_split = (
        args.nr_split if args.nr_split is not None else (1 if n_seq == 1 else 2)
    )
    case = build_case(
        name,
        seqlens,
        h,
        bv,
        nr_split,
        gate,
        extra,
        Hg=hg,
        arch="gfx942",
        swizzle=not args.no_swizzle,
        prefetch=not args.no_prefetch,
        prefetch_w_early=args.prefetch_w_early,
        prefetch_k_early=args.prefetch_k_early,
        prefetch_k_interleave=args.prefetch_k_interleave,
        buffer_desc=not args.no_buffer_desc,
        xcd_remap=not args.no_xcd_remap,
        fence=False,
        want_refs=False,
    )
    print(case.kern.name, flush=True)
    for _ in range(args.launches):
        case.launch()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
