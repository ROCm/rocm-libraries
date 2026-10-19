# MLA decode (weight-absorbed, bf16) — correctness record, gfx950

**Kernels:** [`build_mla_decode_gfx950`](../../../kernels/mla/mla_decode_gfx950.py) (latent) and
[`build_mla_decode_combine`](../../../kernels/mla/mla_decode_combine.py) (split-KV combine)
**Arch:** gfx950. The design, contract, gate and reproduction steps are shared
with gfx942 and recorded once, in
[`../../gfx942/mla_decode/README.md`](../../gfx942/mla_decode/README.md); this
page records what differs and the gfx950 results.
**Scope:** functional bring-up and its correctness evidence; no performance claim.

---

## Table of Contents

- [What differs from gfx942](#what-differs-from-gfx942)
- [Results](#results)
- [Static resources](#static-resources)

---

## What differs from gfx942

The same emitter builds both arches; the spec class
(`MlaDecodeGfx950Spec`) fixes two operand choices:

| | gfx942 | gfx950 |
|---|---|---|
| Score MFMA | `16x16x16` bf16, 36 k-steps over K = 576, 8 B fragments | `16x16x32` bf16, 18 k-steps, 16 B fragments |
| PV latent operand | gathered from the natural `kv_lds` tile with a row permutation the epilogue undoes | `ds_read_tr16_b64` from the natural tile, latent tiles in natural order |
| `kv_lds` / `qr_lds` row pad | 4 bf16 | 8 bf16 (16 B-aligned rows for the 16 B reads) |

The PV keeps the `16x16x16` atom: a key tile is one 16-row page. Everything
else -- 64 heads per workgroup, the online softmax, split-KV and the combine,
the input contract and its refusals -- is the gfx942 kernel's. The compiled
code object was checked to contain the `v_mfma_f32_16x16x32_bf16` and
`ds_read_b64_tr_b16` instructions.

The prefill `absorbed_query` kernel used as an independent oracle on gfx942 has
no gfx950 build, so on gfx950 the gate is the fp64 reference alone (the
reference itself is cross-checked on CPU against the expanded form and the
prefill oracle).

## Results

Run on a gfx950 device (ROCm 7.2 torch, `ROCKE_LLVM_FLAVOR=llvm22`), seed 0, at the [correctness gate](../../gfx942/mla_decode/README.md#the-correctness-gate) of the gfx942 record. Every row also passed its NaN, `S_k = 0` and LSE checks. `num_seg` marked `*` is the count dispatch chose for this device; the others are forced. "worst max_abs / peak" is over the whole batch and reported for scale; "worst per-sequence max_abs / peak" is the gated bar (≤ 2e-2), and the LSE column is gated at 1.7e-2. The numeric CI lane ([`tests/test_mla_decode_numeric.py`](../../../tests/test_mla_decode_numeric.py), three seeds for the whole op) passed on the same device: 19 passed, 1 skipped (the prefill oracle, gfx942-only).

**Latent kernel vs fp64 reference** (10/10 scenarios pass)

| scenario | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst per-sequence max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|---|
| `long_h128` | 64*, 1, 2, 7, 16, 64 | 0.74 | 2.28e-03 | 3.07e-03 | 3.07e-03 | 2.80e-06 | pass |
| `long_h64` | 64*, 1, 2, 7, 16, 64 | 0.55 | 2.37e-03 | 4.28e-03 | 4.28e-03 | 2.35e-06 | pass |
| `no_query_h128` | 4*, 1, 2, 7, 16, 64 | 3.63 | 8.53e-03 | 2.35e-03 | 4.73e-03 | 1.57e-06 | pass |
| `no_query_h64` | 4*, 1, 2, 7, 16, 64 | 2.84 | 1.24e-02 | 4.38e-03 | 4.38e-03 | 2.07e-06 | pass |
| `ragged_h128` | 8*, 1, 2, 7, 16, 64 | 4.02 | 1.29e-02 | 3.21e-03 | 4.68e-03 | 1.84e-06 | pass |
| `ragged_h64` | 8*, 1, 2, 7, 16, 64 | 3.25 | 8.27e-03 | 2.55e-03 | 5.09e-03 | 2.53e-06 | pass |
| `spike_h128` | 8*, 1, 2, 7, 16, 64 | 3.58 | 1.12e-02 | 3.12e-03 | 3.55e-03 | 1.33e-05 | pass |
| `spike_h64` | 8*, 1, 2, 7, 16, 64 | 3.58 | 8.07e-03 | 2.26e-03 | 2.57e-03 | 1.01e-05 | pass |
| `zero_keys_h128` | 1*, 1, 2, 7, 16, 64 | 2.93 | 1.12e-02 | 3.82e-03 | 3.82e-03 | 1.90e-06 | pass |
| `zero_keys_h64` | 1*, 1, 2, 7, 16, 64 | 3.37 | 7.71e-03 | 2.29e-03 | 4.38e-03 | 2.35e-06 | pass |

**Whole op ([A] bmm, kernel, [D] bmm) vs fp64 reference** (10/10 scenarios pass)

| scenario | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst per-sequence max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|---|
| `long_h128` | 64*, 1, 7 | 1.55 | 8.11e-03 | 5.22e-03 | 5.22e-03 | 1.71e-03 | pass |
| `long_h64` | 64*, 1, 7 | 1.17 | 5.16e-03 | 4.42e-03 | 4.88e-03 | 1.63e-03 | pass |
| `no_query_h128` | 4*, 1, 7 | 6.32 | 3.71e-02 | 5.87e-03 | 6.73e-03 | 4.66e-03 | pass |
| `no_query_h64` | 4*, 1, 7 | 5.77 | 2.85e-02 | 4.94e-03 | 5.41e-03 | 5.09e-03 | pass |
| `ragged_h128` | 8*, 1, 7 | 9.16 | 3.02e-02 | 3.29e-03 | 8.78e-03 | 1.12e-02 | pass |
| `ragged_h64` | 8*, 1, 7 | 10.20 | 2.64e-02 | 2.59e-03 | 6.45e-03 | 7.91e-03 | pass |
| `spike_h128` | 8*, 1, 7 | 9.10 | 2.80e-02 | 3.08e-03 | 3.08e-03 | 1.03e-02 | pass |
| `spike_h64` | 8*, 1, 7 | 9.69 | 3.05e-02 | 3.14e-03 | 3.44e-03 | 9.58e-03 | pass |
| `zero_keys_h128` | 1*, 1, 7 | 6.90 | 2.27e-02 | 3.29e-03 | 5.49e-03 | 5.06e-03 | pass |
| `zero_keys_h64` | 1*, 1, 7 | 7.02 | 2.23e-02 | 3.18e-03 | 5.47e-03 | 5.86e-03 | pass |

**Whole op over the bf16 decode shape file** (26/26 shapes pass)

| shape | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst per-sequence max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|---|
| `DeepSeek-V3-R1/ds_v3_b1_kv512/b1_q1k512_h128_bf16` | 1* | 3.37 | 2.08e-02 | 6.16e-03 | 6.16e-03 | 5.88e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv1024/b1_q1k1024_h128_bf16` | 2* | 4.93 | 1.03e-02 | 2.09e-03 | 2.09e-03 | 1.58e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv2048/b1_q1k2048_h128_bf16` | 4* | 1.99 | 1.13e-02 | 5.68e-03 | 5.68e-03 | 2.00e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv4096/b1_q1k4096_h128_bf16` | 8* | 1.82 | 1.27e-02 | 6.95e-03 | 6.95e-03 | 1.63e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv8192/b1_q1k8192_h128_bf16` | 16* | 3.81 | 1.27e-02 | 3.35e-03 | 3.35e-03 | 2.39e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv16384/b1_q1k16384_h128_bf16` | 32* | 0.90 | 4.37e-03 | 4.84e-03 | 4.84e-03 | 1.29e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv32768/b1_q1k32768_h128_bf16` | 64* | 0.90 | 3.71e-03 | 4.11e-03 | 4.11e-03 | 1.49e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b4_kv2048/b4_q1k2048_h128_bf16` | 4* | 2.71 | 1.88e-02 | 6.96e-03 | 6.96e-03 | 2.33e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b8_kv2048/b8_q1k2048_h128_bf16` | 4* | 4.18 | 1.75e-02 | 4.18e-03 | 6.18e-03 | 2.37e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b4_kv8192/b4_q1k8192_h128_bf16` | 16* | 2.76 | 8.81e-03 | 3.19e-03 | 5.81e-03 | 1.77e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b8_kv8192/b8_q1k8192_h128_bf16` | 16* | 2.86 | 1.83e-02 | 6.40e-03 | 6.40e-03 | 2.25e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b32_kv2048/b32_q1k2048_h128_bf16` | 4* | 4.32 | 1.83e-02 | 4.23e-03 | 7.58e-03 | 3.66e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b32_kv8192/b32_q1k8192_h128_bf16` | 8* | 2.86 | 1.61e-02 | 5.63e-03 | 7.92e-03 | 2.42e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv512/b1_q1k512_h64_bf16` | 1* | 4.06 | 1.30e-02 | 3.21e-03 | 3.21e-03 | 2.58e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv1024/b1_q1k1024_h64_bf16` | 2* | 2.81 | 1.32e-02 | 4.70e-03 | 4.70e-03 | 1.95e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv2048/b1_q1k2048_h64_bf16` | 4* | 2.13 | 8.79e-03 | 4.12e-03 | 4.12e-03 | 1.36e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv4096/b1_q1k4096_h64_bf16` | 8* | 1.51 | 9.42e-03 | 6.25e-03 | 6.25e-03 | 1.48e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv8192/b1_q1k8192_h64_bf16` | 16* | 1.11 | 4.64e-03 | 4.17e-03 | 4.17e-03 | 1.24e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv16384/b1_q1k16384_h64_bf16` | 32* | 1.13 | 6.83e-03 | 6.05e-03 | 6.05e-03 | 1.28e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv32768/b1_q1k32768_h64_bf16` | 64* | 0.90 | 3.40e-03 | 3.80e-03 | 3.80e-03 | 1.19e-03 | pass |
| `Kimi-K2/kimi_k2_b4_kv2048/b4_q1k2048_h64_bf16` | 4* | 2.44 | 1.93e-02 | 7.91e-03 | 7.91e-03 | 2.31e-03 | pass |
| `Kimi-K2/kimi_k2_b8_kv2048/b8_q1k2048_h64_bf16` | 4* | 4.26 | 1.75e-02 | 4.12e-03 | 6.60e-03 | 4.98e-03 | pass |
| `Kimi-K2/kimi_k2_b4_kv8192/b4_q1k8192_h64_bf16` | 16* | 2.00 | 9.66e-03 | 4.82e-03 | 8.04e-03 | 1.73e-03 | pass |
| `Kimi-K2/kimi_k2_b8_kv8192/b8_q1k8192_h64_bf16` | 16* | 2.56 | 2.38e-02 | 9.27e-03 | 9.27e-03 | 2.38e-03 | pass |
| `Kimi-K2/kimi_k2_b32_kv2048/b32_q1k2048_h64_bf16` | 4* | 5.35 | 2.84e-02 | 5.31e-03 | 7.19e-03 | 3.61e-03 | pass |
| `Kimi-K2/kimi_k2_b32_kv8192/b32_q1k8192_h64_bf16` | 16* | 2.95 | 1.19e-02 | 4.04e-03 | 6.04e-03 | 1.96e-03 | pass |


Every latent-kernel row matches the gfx942 record to the printed digits:
both arches round the same bf16 inputs to the same outputs, and the wide-K
score atom only regroups exact bf16 products. The whole-op rows match too, up
to the last LSE digit of `no_query_h64` (the `torch.bmm` projections are each
device's own library GEMM), and `ds_v3_b32_kv8192` differs because dispatch
chose a different split count on each device.

## Static resources

Compiled through comgr (llvm22) and read from the code object. Zero scratch, and at most 256 VGPRs and 32 KiB of LDS: two four-wave workgroups per CU. `TestResources` in [`tests/test_mla_decode.py`](../../../tests/test_mla_decode.py) checks this in CI.

| kernel | VGPR | SGPR | LDS (B) | scratch (B) |
|---|---|---|---|---|
| `mla_decode_gfx950_h64_rkv512r64_k16_hw16_w4_bf16` | 254 | 48 | 27904 | 0 |
| `mla_decode_gfx950_h64_rkv512r64_k16_hw16_w4_split_bf16` | 254 | 52 | 27904 | 0 |
| `mla_decode_combine_h64_rkv512_w4_bf16` | 26 | 30 | 0 | 0 |
| `mla_decode_gfx950_h128_rkv512r64_k16_hw16_w4_bf16` | 254 | 48 | 27904 | 0 |
| `mla_decode_gfx950_h128_rkv512r64_k16_hw16_w4_split_bf16` | 254 | 52 | 27904 | 0 |
| `mla_decode_combine_h128_rkv512_w4_bf16` | 26 | 30 | 0 | 0 |
