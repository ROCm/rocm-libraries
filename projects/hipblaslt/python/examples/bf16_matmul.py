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
import hipblaslt
from hipblaslt import _core as c

# ── problem size ──────────────────────────────────────────────────────────────
M, K, N = 512, 256, 128

# ── generate random bf16 inputs ───────────────────────────────────────────────
rng = np.random.default_rng(42)
# numpy has no native bf16, so generate in f32 then cast to bf16
A_f32 = rng.standard_normal((M, K)).astype(np.float32)
B_f32 = rng.standard_normal((K, N)).astype(np.float32)

A_bf16 = A_f32.astype(ml_dtypes.bfloat16)
B_bf16 = B_f32.astype(ml_dtypes.bfloat16)

# CPU reference: upcast bf16 → f32 before multiplying to match GPU accumulation
ref = A_bf16.astype(np.float32) @ B_bf16.astype(np.float32)

print(f"A: {A_bf16.shape}  dtype={A_bf16.dtype}")
print(f"B: {B_bf16.shape}  dtype={B_bf16.dtype}")
print(f"D: ({M}, {N})  expected dtype=bf16\n")

# ── upload to device ──────────────────────────────────────────────────────────
# hipBLASLt uses column-major (Fortran) order internally.
# Store A.T and B.T as contiguous arrays so each column is contiguous.
#
# nanobind's ndarray binding only recognises standard numpy scalar types, so
# ml_dtypes.bfloat16 arrays must be viewed as uint16 (same 2-byte layout)
# before passing to from_numpy. The DataType.R_16BF tag tells hipBLASLt how
# to interpret the raw bytes.
def _bf16_to_device(arr_bf16, dtype_tag):
    return c.DeviceArray.from_numpy(
        np.ascontiguousarray(arr_bf16).view(np.uint16), dtype_tag
    )

dA = _bf16_to_device(A_bf16.T, c.DataType.R_16BF)
dB = _bf16_to_device(B_bf16.T, c.DataType.R_16BF)
dC = _bf16_to_device(np.zeros((N, M), ml_dtypes.bfloat16), c.DataType.R_16BF)
dD = _bf16_to_device(np.zeros((N, M), ml_dtypes.bfloat16), c.DataType.R_16BF)

# ── descriptor setup ──────────────────────────────────────────────────────────
# Leading dimensions equal the number of rows in the column-major layout.
# For column-major A (stored as A.T): shape is (K, M), so ld = K.
la = c.MatrixLayout(c.DataType.R_16BF, M, K, M)  # A: M rows, K cols, ld=M
lb = c.MatrixLayout(c.DataType.R_16BF, K, N, K)  # B: K rows, N cols, ld=K
lc = c.MatrixLayout(c.DataType.R_16BF, M, N, M)  # C/D: M rows, N cols, ld=M
ld = c.MatrixLayout(c.DataType.R_16BF, M, N, M)

# ── run ───────────────────────────────────────────────────────────────────────
# Use explicit close() — nanobind 2.13 has an incompatibility with the
# context-manager __exit__ signature for C++ extension types.
h = c.Handle()
try:
    desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
    pref = c.Preference()
    pref.set_max_workspace(64 * 1024 * 1024)  # 64 MB

    results = c.heuristic(h, desc, la, lb, lc, ld, pref, max_results=16)
    if not results:
        raise RuntimeError("No algorithm found for this problem")

    print(f"Heuristic returned {len(results)} algorithm(s)")
    print(f"Using algo #{results[0].algo.index}  "
          f"workspace={results[0].workspace_size} bytes  "
          f"waves={results[0].waves_count:.2f}\n")

    ws = c.DeviceArray.from_numpy(
        np.zeros(max(1, results[0].workspace_size), np.uint8), c.DataType.R_8I
    )

    c.matmul(h, desc,
             1.0, dA, la, dB, lb,
             0.0, dC, lc, dD, ld,
             results[0].algo, ws)
finally:
    h.close()

# ── retrieve result ───────────────────────────────────────────────────────────
# dD was stored column-major (N×M), so transpose back to row-major (M×N).
# to_numpy() returns uint16 (that's what we uploaded); re-view as bfloat16.
D_bf16 = dD.to_numpy().reshape(N, M).T.view(ml_dtypes.bfloat16)

# ── compare ───────────────────────────────────────────────────────────────────
D_f32 = D_bf16.astype(np.float32)
max_err = np.max(np.abs(D_f32 - ref))
rel_err = max_err / (np.max(np.abs(ref)) + 1e-6)
match = np.allclose(D_f32, ref, rtol=0.02, atol=0.02)

print(f"Max absolute error : {max_err:.6f}")
print(f"Max relative error : {rel_err:.4%}")
print(f"allclose(rtol=2%)  : {match}")

# Spot-check a few elements
print("\nSpot-check (GPU vs CPU reference):")
for i in range(3):
    print(f"  D[{i},{i}] = {D_f32[i, i]:.4f}  ref={ref[i, i]:.4f}")
