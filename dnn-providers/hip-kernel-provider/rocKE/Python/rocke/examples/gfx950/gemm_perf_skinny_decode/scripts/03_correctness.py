#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 3: standalone bf16 correctness check.

Runbook anchors:
- §2.1 Correctness Baselines (verify before quoting speed)
- §14.1 "If correctness fails, do not report speed as a win"

`rocke.run_manifest --verify` allocates fp16 buffers regardless of
manifest dtype, so it cannot validate bf16 (kernel reads fp16 bytes as
bf16 → garbage; `max_abs_diff` is then meaningless). We re-implement
the verify path with actual bf16 (`ml_dtypes.bfloat16`).
"""

from __future__ import annotations
import ctypes
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5])
)  # .../rocke/.. = python root

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rocke.runtime.hip_module import Runtime  # noqa: E402


def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def bf16_bytes_from_fp32(arr_fp32: np.ndarray) -> np.ndarray:
    """Round fp32 → bf16 → return raw u16 view (numpy-only bf16 path)."""
    t = torch.from_numpy(arr_fp32).to(torch.bfloat16)
    u16 = t.view(torch.int16).contiguous().numpy().view(np.uint16)
    return u16


def fp32_from_bf16_bytes(buf: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()
    t = torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16)
    return t.to(torch.float32).numpy()


sweep = json.loads((ROOT / "data" / "02_sweep_bench.json").read_text())
ok = [r for r in sweep["results"] if "ms_best" in r]
winner = min(ok, key=lambda r: r["ms_best"])
label = winner["label"]
M, N, K = sweep["shape"]["M"], sweep["shape"]["N"], sweep["shape"]["K"]

salted = f"{label}__m{M}n{N}k{K}"
sub = ROOT / "build" / salted
manifest = json.loads((sub / "manifest.json").read_text())
hsaco_path = sub / manifest["hsaco"]

print(f"Verifying winner: {label}")
print(f"  shape M={M} N={N} K={K}, dtype bf16, layout RCR")
print(f"  hsaco: {hsaco_path}")

rng = np.random.default_rng(0xC0FFEE)
A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)
A_u16 = bf16_bytes_from_fp32(A_f32)
B_u16 = bf16_bytes_from_fp32(B_f32)
# Reference: bf16-rounded A and B multiplied in fp32, accumulator in fp32
A_bf32 = fp32_from_bf16_bytes(A_u16.tobytes(), A_f32.shape)
B_bf32 = fp32_from_bf16_bytes(B_u16.tobytes(), B_f32.shape)
ref_f32 = A_bf32 @ B_bf32.T  # fp32 reference (matches MFMA fp32 acc)
C_nbytes = M * N * 2

rt = Runtime()
mod = rt.load_module(hsaco_path.read_bytes())
fn = mod.get_function(manifest["kernel_name"])

A_dev = rt.alloc(A_u16.nbytes)
B_dev = rt.alloc(B_u16.nbytes)
C_dev = rt.alloc(C_nbytes)
rt.memcpy_h2d(A_dev, _as_u8(A_u16), A_u16.nbytes)
rt.memcpy_h2d(B_dev, _as_u8(B_u16), B_u16.nbytes)
rt.memset(C_dev, 0, C_nbytes)

args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
bn = int(manifest["block_n"])
bm = int(manifest["block_m"])
grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
block = (int(manifest["threads_per_block"]), 1, 1)

rt.launch(fn, grid, block, args, stream=0)
rt.stream_sync(0)

out_buf = (ctypes.c_uint8 * C_nbytes)()
rt.memcpy_d2h(out_buf, C_dev, C_nbytes)
C_out_f32 = fp32_from_bf16_bytes(bytes(out_buf), (M, N))

# Reference also rounded to bf16 storage (kernel emits bf16)
ref_bf_f32 = fp32_from_bf16_bytes(bf16_bytes_from_fp32(ref_f32).tobytes(), (M, N))
diff = np.abs(C_out_f32 - ref_bf_f32)
max_abs = float(diff.max())
bad = int((diff > 0).sum())
tot = int(diff.size)
ref_max = float(np.abs(ref_bf_f32).max())

print(f"\n  max|out-ref|={max_abs:.4f}   bad={bad}/{tot}   ref_max={ref_max:.0f}")
status = "PASS" if max_abs < 1.0 else "FAIL"
print(f"  → {status}")

out = ROOT / "data" / "03_correctness.json"
with out.open("w") as f:
    json.dump(
        {
            "label": label,
            "max_abs_diff": max_abs,
            "bad": bad,
            "total": tot,
            "ref_max": ref_max,
            "status": status,
        },
        f,
        indent=2,
    )
print(f"\nWrote {out}")

rt.free(A_dev)
rt.free(B_dev)
rt.free(C_dev)
