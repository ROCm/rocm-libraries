# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) FMHA-bwd stats-prep build + numeric verify on CDNA.

Builds the FMHA-backward softmax-statistics prep kernel
(:func:`ck_dsl.instances.common.sdpa_lse_prep.build_sdpa_lse_prep`),
compiles it through the HIP-C++ -> hipcc backend
(``compile_kernel_via_hipcc``), launches it via the HIP runtime, and
compares the produced ``M_out`` / ``L_out`` against a torch-free numpy
reference (computed in fp32).

The kernel converts the hipDNN forward's single head-major natural-log
LSE ``stats`` tensor (``[B, Hq, Sq]``) into the two per-batch q-major
inputs the CK FMHA-backward kernel reads::

    M_out[(b*Sq + q)*Hq + h] = stats[(b*Hq + h)*Sq + q] * log2(e)
    L_out[(b*Sq + q)*Hq + h] = 1.0

The body is a plain f32 load / multiply / store with one ``q < Sq``
guard, so the emitted IR is identical on gfx942 and gfx950; parity with
the numpy reference is exact (single fp32 multiply by ``log2(e)``), so
the default tolerance is tight (``1e-6``).

Must run on a device matching ``--arch`` (e.g. a gfx942 box).

    PYTHONPATH=python python3 -m ck_dsl.examples.common.sdpa_lse_prep_verify_hip \
        --arch gfx942 --B 2 --heads 4 --seqlen 64
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers.compile import compile_kernel_via_hipcc
from ck_dsl.instances.common.sdpa_lse_prep import (
    SdpaLsePrepSpec,
    build_sdpa_lse_prep,
    is_valid_spec,
    sdpa_lse_prep_grid,
)
from ck_dsl.runtime.hip_module import Runtime


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--B", type=int, default=2, help="batch")
    p.add_argument("--heads", "--Hq", dest="heads", type=int, default=4, help="Hq")
    p.add_argument(
        "--seqlen", "--Sq", dest="seqlen", type=int, default=64, help="Sq"
    )
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0xA11E)
    args = p.parse_args()

    import numpy as np

    B = args.B
    Hq = args.heads
    Sq = args.seqlen

    spec = SdpaLsePrepSpec(B=B, Hq=Hq, Sq=Sq)

    ok, why = is_valid_spec(spec, args.arch)
    print(
        f"[{args.arch}] HIP-path sdpa_lse_prep "
        f"B{B} HQ{Hq} Q{Sq} validate -> {ok} ({why})"
    )
    if not ok:
        return 2

    # ArchTarget resolves here too (is_valid already checked it); kept for
    # parity with the other verify harnesses' wiring.
    ArchTarget.from_gfx(args.arch)

    art = compile_kernel_via_hipcc(
        build_sdpa_lse_prep(spec, arch=args.arch), arch=args.arch
    )
    print(
        f"[{args.arch}] HIP-path built {art.kernel_name} "
        f"({art.hsaco_bytes} B, isa={art.isa}) "
        f"hipcc={art.timings.get('hipcc', 0):.0f}ms "
        f"total={art.timings.get('total', 0):.0f}ms"
    )

    rng = np.random.default_rng(args.seed)

    # Head-major source stats [B, Hq, Sq] (natural-log LSE domain).
    stats = rng.standard_normal((B, Hq, Sq)).astype(np.float32)

    # Per-batch q-major flat outputs, length B*Sq*Hq.
    n_out = B * Sq * Hq
    M_out = np.zeros((n_out,), dtype=np.float32)
    L_out = np.zeros((n_out,), dtype=np.float32)

    grid = sdpa_lse_prep_grid(spec)  # (ceil(Sq/64), Hq, B)
    block = (64, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    stats_c = np.ascontiguousarray(stats)
    statsd = rt.alloc(stats_c.nbytes)
    md = rt.alloc(M_out.nbytes)
    ld = rt.alloc(L_out.nbytes)

    rt.memcpy_h2d(statsd, u8(stats_c), stats_c.nbytes)
    # Zero the destinations so partial-tile gaps (Sq not a multiple of 64)
    # stay observable as zeros rather than stale device memory.
    rt.memset(md, 0, M_out.nbytes)
    rt.memset(ld, 0, L_out.nbytes)

    # Kernarg pack (6 args): 3 ptrs + 3 i32 == 3*8 + 3*4 = 36 B, naturally
    # aligned, no pad. Order mirrors sdpa_lse_prep._declare_params exactly:
    # stats, M_out, L_out, B, Hq, Sq.
    packed = struct.pack("<3Q3i", statsd, md, ld, B, Hq, Sq)
    rt.launch(fn, grid, block, packed)
    rt.sync()

    rt.memcpy_d2h(u8(M_out), md, M_out.nbytes)
    rt.memcpy_d2h(u8(L_out), ld, L_out.nbytes)

    for ptr in (statsd, md, ld):
        rt.free(ptr)
    module.unload()

    # numpy reference (fp32).
    log2e = np.float32(math.log2(math.e))
    M_ref = np.zeros((n_out,), dtype=np.float32)
    L_ref = np.ones((n_out,), dtype=np.float32)
    for b in range(B):
        for h in range(Hq):
            for q in range(Sq):
                out_off = (b * Sq + q) * Hq + h
                M_ref[out_off] = stats[b, h, q] * log2e

    m_diff = float(np.abs(M_out - M_ref).max())
    l_diff = float(np.abs(L_out - L_ref).max())
    max_abs = max(m_diff, l_diff)
    all_ok = max_abs <= args.tol

    tag = "PASS" if all_ok else "FAIL"
    print(
        f"[{args.arch}] HIP-path sdpa_lse_prep B={B} Hq={Hq} Sq={Sq}: "
        f"M={m_diff:.3e} L={l_diff:.3e} "
        f"tol={args.tol:.0e} -> {tag}"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
