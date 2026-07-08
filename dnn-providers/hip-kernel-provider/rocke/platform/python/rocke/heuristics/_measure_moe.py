#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch-free GPU measurement pass for the moe heuristics smoke test.

Reads the build-only parquet emitted by ``gen_sweep_data --op moe`` (where
``measured_tflops`` is 0.0), actually *runs* each variant's gather / silu_mul /
topk_reduce kernels on the GPU via the ctypes HIP runtime (``rocke.runtime.
hip_module``), times them with HIP events, and writes back real
``latency_ms`` / ``bandwidth_gb_s`` / ``measured_tflops`` (a byte-throughput
proxy, since the streaming trio has no GEMM FLOP) plus ``is_valid``.

No torch dependency: device buffers come from ``Runtime.alloc`` (hipMalloc),
args are packed with ``runtime.torch_module.pack_args`` (which accepts integer
device pointers directly).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import pandas as pd

from rocke.instances import FusedMoeSpec
from rocke.instances.common import fused_moe as fm
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.runtime.comgr import build_hsaco_from_llvm_ir
from rocke.runtime.hip_module import Runtime
from rocke.runtime.torch_module import pack_args

_DTYPE_BYTES = {"f16": 2, "fp16": 2, "bf16": 2}


def _phase_defs(spec: FusedMoeSpec):
    return {
        "gather": (
            fm.build_moe_gather(spec),
            fm.moe_gather_signature(spec),
            fm.moe_gather_grid(spec),
        ),
        "silu_mul": (
            fm.build_moe_silu_mul(spec),
            fm.moe_silu_mul_signature(spec),
            fm.moe_silu_mul_grid(spec),
        ),
        "topk_reduce": (
            fm.build_moe_topk_weighted_reduce(spec),
            fm.moe_topk_weighted_reduce_signature(spec),
            fm.moe_topk_weighted_reduce_grid(spec),
        ),
    }


def _alloc_values(rt: Runtime, spec: FusedMoeSpec):
    """Allocate device buffers for all three phases; return (values_per_phase, keep)."""
    bpe = _DTYPE_BYTES.get(spec.dtype, 2)
    T, K, H, IM = spec.tokens, spec.topk, spec.hidden, spec.intermediate
    tp = T * K

    def dev(nbytes):
        nbytes = max(int(nbytes), 16)
        p = rt.alloc(nbytes)
        rt.memset(p, 0, nbytes)
        return p

    X = dev(T * H * bpe)
    GroupedInput = dev(tp * H * bpe)
    GateOut = dev(tp * IM * bpe)
    UpOut = dev(tp * IM * bpe)
    Hidden = dev(tp * IM * bpe)
    DownOut = dev(tp * H * bpe)
    Y = dev(T * H * 4)  # f32 accumulator
    SortedWeights = dev(tp * 4)

    # SortedTokenIds: valid indices in [0, T); upload a host buffer.
    import ctypes

    sids_host = struct.pack("<%di" % tp, *[i % T for i in range(tp)])
    SortedTokenIds = rt.alloc(max(len(sids_host), 16))
    cbuf = (ctypes.c_ubyte * len(sids_host)).from_buffer_copy(sids_host)
    rt.memcpy_h2d(SortedTokenIds, cbuf, len(sids_host))

    values = {
        "gather": {
            "X": X,
            "SortedTokenIds": SortedTokenIds,
            "GroupedInput": GroupedInput,
            "tokens": T,
            "hidden": H,
        },
        "silu_mul": {
            "GateOut": GateOut,
            "UpOut": UpOut,
            "Hidden": Hidden,
            "total_pairs": tp,
            "intermediate": IM,
        },
        "topk_reduce": {
            "DownOut": DownOut,
            "SortedTokenIds": SortedTokenIds,
            "SortedWeights": SortedWeights,
            "Y": Y,
            "total_pairs": tp,
            "hidden": H,
            "tokens": T,
        },
    }
    keep = [
        X,
        GroupedInput,
        GateOut,
        UpOut,
        Hidden,
        DownOut,
        Y,
        SortedWeights,
        SortedTokenIds,
    ]
    return values, keep


def _measure_spec(
    rt: Runtime, spec: FusedMoeSpec, arch: str, warmup: int, repeat: int
) -> dict:
    isa = f"amdgcn-amd-amdhsa--{arch}"
    block = (spec.block_size, 1, 1)
    defs = _phase_defs(spec)
    values, keep = _alloc_values(rt, spec)

    fns = {}
    mods = []
    for phase, (kdef, sig, grid) in defs.items():
        ll = lower_kernel_to_llvm(kdef)
        hsaco, _ = build_hsaco_from_llvm_ir(ll, isa=isa)
        mod = rt.load_module(hsaco)
        mods.append(mod)
        name = kdef.name if hasattr(kdef, "name") else spec.kernel_name(phase)
        fns[phase] = (mod.get_function(name), sig, grid)

    def run_chain():
        for phase in ("gather", "silu_mul", "topk_reduce"):
            fn, sig, grid = fns[phase]
            packed = pack_args(sig, values[phase])
            rt.launch(fn, grid, block, packed, stream=0)

    # Warmup.
    for _ in range(warmup):
        run_chain()
    rt.wait_stream(0)

    # Timed.
    start = rt.event()
    end = rt.event()
    start.record(stream=0)
    for _ in range(repeat):
        run_chain()
    end.record(stream=0)
    end.synchronize()
    total_ms = start.elapsed_to(end)
    per_iter_ms = total_ms / repeat
    rt.wait_stream(0)
    start.destroy()
    end.destroy()
    for m in mods:
        m.unload()
    for p in keep:
        rt.free(p)

    # Bytes moved across the trio (read+write of the streamed buffers).
    bpe = _DTYPE_BYTES.get(spec.dtype, 2)
    T, K, H, IM = spec.tokens, spec.topk, spec.hidden, spec.intermediate
    tp = T * K
    bytes_moved = (
        # gather: read X-rows + write GroupedInput
        2 * tp * H * bpe
        # silu_mul: read Gate+Up, write Hidden
        + 3 * tp * IM * bpe
        # topk_reduce: read DownOut + atomic into Y
        + tp * H * bpe
        + tp * H * 4
    )
    sec = per_iter_ms / 1e3
    gb_s = (bytes_moved / sec) / 1e9 if sec > 0 else 0.0
    # measured_tflops proxy: effective element-ops/s (bytes/2 ~ elems) in T-ops/s.
    tflops = (bytes_moved / 2.0 / sec) / 1e12 if sec > 0 else 0.0
    return {
        "latency_ms": float(per_iter_ms),
        "bandwidth_gb_s": float(gb_s),
        "measured_tflops": float(tflops),
        "is_valid": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.inp)
    rt = Runtime()
    out_rows = []
    for i, row in df.iterrows():
        spec = FusedMoeSpec(
            tokens=int(row["tokens"]),
            experts=int(row["experts"]),
            topk=int(row["topk"]),
            hidden=int(row["hidden"]),
            intermediate=int(row["intermediate"]),
            dtype=str(row["dtype"]),
            block_size=int(row["block_size"]),
            vec=int(row["vec"]),
        )
        rec = dict(row)
        try:
            m = _measure_spec(rt, spec, args.arch, args.warmup, args.repeat)
            rec.update(m)
        except Exception as e:  # noqa: BLE001
            rec["is_valid"] = False
            rec["measured_tflops"] = 0.0
            rec["latency_ms"] = 0.0
            rec["build_error"] = f"{type(e).__name__}: {e}"
            print(f"[measure] FAIL row {i}: {e}", file=sys.stderr)
        out_rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"[measure] {i + 1}/{len(df)} done", file=sys.stderr, flush=True)

    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False, engine="pyarrow")
    nz = int((out_df["measured_tflops"] > 0).sum())
    print(
        f"[measure] wrote {len(out_df)} rows -> {args.out}; "
        f"nonzero measured_tflops={nz}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
