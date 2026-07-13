# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Minimal bf16 GEMM example using the hipblaslt Python interface.

Generates random bf16 A (m×k) and B (k×n), runs D = A @ B on the GPU via
hipBLASLt, then compares against a numpy f32 reference (upcast for accuracy).

Usage:
    cd projects/hipblaslt/python
    conda activate pydev313
    python examples/bf16_matmul.py
"""
import numpy as np
import ml_dtypes
from hipblaslt import _core as c

# ── problem size ──────────────────────────────────────────────────────────────
M, K, N = 512, 256, 128

# ── generate random bf16 inputs ───────────────────────────────────────────────
rng = np.random.default_rng(42)
A_bf16 = rng.standard_normal((M, K)).astype(np.float32).astype(ml_dtypes.bfloat16)
B_bf16 = rng.standard_normal((K, N)).astype(np.float32).astype(ml_dtypes.bfloat16)

# CPU reference: upcast bf16 → f32 to match the GPU's f32 accumulator
ref = A_bf16.astype(np.float32) @ B_bf16.astype(np.float32)

print(f"A: {A_bf16.shape}  dtype={A_bf16.dtype}")
print(f"B: {B_bf16.shape}  dtype={B_bf16.dtype}")
print(f"D: ({M}, {N})  expected dtype=bf16\n")

# ── upload to device ──────────────────────────────────────────────────────────
# hipBLASLt uses column-major order internally. We store A.T / B.T so each
# column is contiguous in memory (leading dimension == number of rows).
# nanobind only accepts standard numpy scalar types, so ml_dtypes arrays are
# viewed as uint16 (same 2-byte width) before the H2D copy.
def _to_device(arr_ml, dtype_tag):
    arr = np.ascontiguousarray(arr_ml).view(np.uint16)
    return c.DeviceArray.from_numpy(arr, dtype_tag)

dA = _to_device(A_bf16.T, c.DataType.R_16BF)
dB = _to_device(B_bf16.T, c.DataType.R_16BF)
dC = _to_device(np.zeros((N, M), ml_dtypes.bfloat16), c.DataType.R_16BF)
dD = _to_device(np.zeros((N, M), ml_dtypes.bfloat16), c.DataType.R_16BF)

# ── matrix layout descriptors ─────────────────────────────────────────────────
la = c.MatrixLayout(c.DataType.R_16BF, M, K, M)   # A: M rows, K cols, ld=M
lb = c.MatrixLayout(c.DataType.R_16BF, K, N, K)   # B: K rows, N cols, ld=K
lc = c.MatrixLayout(c.DataType.R_16BF, M, N, M)   # C: M rows, N cols, ld=M
ld_layout = c.MatrixLayout(c.DataType.R_16BF, M, N, M)   # D: same

# ── heuristic: enumerate available algorithms ─────────────────────────────────
h = c.Handle()
try:
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)

    results = c.heuristic(h, desc, la, lb, lc, ld_layout, pref, max_results=16)
    if not results:
        raise RuntimeError("No algorithm found for this problem")

    print(f"Heuristic returned {len(results)} algorithm(s)")
    r0 = results[0]
    print(f"Using algo #{r0.algo.index}  "
          f"workspace={r0.workspace_size} bytes  "
          f"waves={r0.waves_count:.2f}\n")

    ws = c.DeviceArray.from_numpy(
        np.zeros(max(1, r0.workspace_size), np.uint8), c.DataType.R_8I
    )

    # ── run the GEMM ──────────────────────────────────────────────────────────
    c.matmul(h, desc,
             1.0, dA, la, dB, lb,
             0.0, dC, lc, dD, ld_layout,
             r0.algo, ws)
finally:
    h.close()

# ── retrieve result ───────────────────────────────────────────────────────────
# dD is column-major (N×M stored as rows×cols after the transpose trick),
# so we reshape to (N, M) then transpose back to (M, N) row-major.
# copy_to_host only accepts standard numpy dtypes; use uint16 wire buffer
# then re-view as bfloat16.
out_wire = np.empty((N, M), dtype=np.uint16)
dD.copy_to_host(out_wire)
D_bf16 = out_wire.T.copy().view(ml_dtypes.bfloat16)   # (M, N), bfloat16

# ── compare against CPU reference ────────────────────────────────────────────
D_f32 = D_bf16.astype(np.float32)
max_err = float(np.max(np.abs(D_f32 - ref)))
rel_err = max_err / (float(np.max(np.abs(ref))) + 1e-6)
match = bool(np.allclose(D_f32, ref, rtol=0.02, atol=0.02))

print(f"Max absolute error : {max_err:.6f}")
print(f"Max relative error : {rel_err:.4%}")
print(f"allclose(rtol=2%)  : {match}")

print("\nSpot-check (GPU vs CPU reference):")
for i in range(3):
    print(f"  D[{i},{i}] = {D_f32[i, i]:.4f}  ref={ref[i, i]:.4f}")

if not match:
    raise RuntimeError("GEMM result does not match CPU reference within tolerance")

print("\nPASS")
