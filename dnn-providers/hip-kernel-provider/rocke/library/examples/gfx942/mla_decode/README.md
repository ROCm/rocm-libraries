# MLA decode (weight-absorbed, bf16) — correctness record, gfx942

**Kernels:** [`build_mla_decode`](../../../kernels/mla/mla_decode.py) (latent) and
[`build_mla_decode_combine`](../../../kernels/mla/mla_decode_combine.py) (split-KV combine)
**Dispatch:** [`dispatch.mla`](../../../dispatch/mla/decode.py), selected for `max_seqlen_q == 1`
**Arch:** gfx942. The gfx950 record is [`../../gfx950/mla_decode/README.md`](../../gfx950/mla_decode/README.md).
**Scope:** functional bring-up and its correctness evidence. Nothing here is a
performance claim; tuning is a separate piece of work.

---

## Table of Contents

- [What this records](#what-this-records)
- [The kernel as built](#the-kernel-as-built)
- [Contract](#contract)
- [The correctness gate](#the-correctness-gate)
- [Reproducing](#reproducing)
- [Results](#results)
- [Static resources](#static-resources)
- [Limits](#limits)

---

## What this records

That the decode kernel family computes the weight-absorbed MLA decode op
correctly on gfx942, at the tolerance below, for both supported head counts
(64, a Kimi-K2-shaped layer, and 128, DeepSeek-V3-shaped), across ragged key
lengths, split-KV segment counts, and the benchmark shape file. It also records
the design that was built, which refines
[`builders/mla/DESIGN.md`](../../../builders/mla/DESIGN.md) §4–§5 in the ways
listed below.

## The kernel as built

Decode has one query per sequence, so the MFMA axis that prefill gives to
queries is given to **heads**: every head of an absorbed MLA layer attends to
the same 576-wide key row `[c_kv | k_rope]`, so 64 heads of one sequence share
every staged key tile.

| Choice | As built | Why |
|---|---|---|
| Absorb | two-stage, outside the kernel: `Qa = [Q_nope @ W_UK_nopeᵀ \| Q_rope]` | DESIGN §4's materialized `W_abs` reads 192 MiB per step; the two-stage form 16 MiB of `W_UK` here |
| `W_UV` | after the combine, outside the kernel | heads on the MFMA axis make `W_UV` vary along that axis, so it cannot be an in-wave operand; it then runs once per head row, not once per segment |
| Head layout | 16 heads per wave on the MFMA n axis, 4 waves, 64 heads per workgroup | the latent is the score's K axis and the PV's N axis; splitting it across waves would turn the score into a split-K with a combine per tile |
| Score | `Sᵀ = [C \| K_rope] · Qaᵀ`, `mfma_f32_16x16x16_bf16`, K = 576 | the prefill one-wave-per-head pipeline with its n axis relabelled from query to head |
| PV | `accᵀ += Cᵀ · Pᵀ`, latent operand gathered from the natural `kv_lds` tile | gfx942 has no transpose LDS read |
| Mask | the last, partial page only | at `S_q = 1` the bottom-right causal mask admits every key |
| Split-KV | `num_seg` segments of whole key tiles, fp32 partials `(acc, m, l)`, arch-neutral combine; an empty segment writes only `(m, l)` and the combine skips its `acc` row | fills the device at small batch; at least 32 tiles per segment keeps the partials' write and read under half the key traffic they split (an untuned floor) |

The one-launch path (`num_seg == 1`) writes the normalized latent and the LSE
directly; `num_seg > 1` writes partials and launches the combine. `num_seg` is a
kernel argument, so one compiled split kernel serves every segment count.

## Contract

Decode serves `max_seqlen_q == 1` with the **absorbed** input contract: the
caller absorbs `W_UK_nope` into the query and applies `W_UV` to the latent.
The prefill kernel serves `max_seqlen_q >= 2` with a different contract (the
projected query plus `W_UK` in, the attention output out), so it is never
substituted for decode: a one-query request that decode cannot serve -- a
head count that is not a multiple of 64 -- is refused.

| Tensor | Shape | |
|---|---|---|
| `qa` | `[total_q, H, 576]` bf16 | the absorbed query |
| `c_kv`, `k_rope` | `[pages, 16, 512]`, `[pages, 16, 64]` bf16 | paged latent cache, head-shared RoPE keys |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B + 1]` i32 | a sequence may have no query (`total_q <= B`) |
| `block_table` | `[B, max_pages]` i32 | row pitch is its stride |
| `scale` | f32 | the host's softmax scale, never derived in-kernel |
| `out`, `lse` | `[total_q, H, 512]` bf16, `[total_q, H]` f32 | normalized latent; natural-log LSE |

A sequence with `S_k = 0` gets a zero latent and `lse = -inf`. No sequence may
carry more than one query: the kernel computes each sequence's first token
only. The single-pass argument list is the prefill `absorbed_query` kernel's,
argument for argument, which is what lets that kernel serve as an independent
GPU oracle on the same buffers. A split-KV selection is two launches;
[`mla_decode_launch`](../../../dispatch/mla/decode.py) describes the combine
and the fp32 workspace between them.
[`MlaDecodeRunner`](../../../builders/mla/mla_decode_runner.py) composes the
whole op (`torch.bmm` absorb, latent, combine, `torch.bmm` `W_UV`) and checks the
caller's tensors against the dispatched request before launching.

## The correctness gate

- **Reference:** [`ref_mla_decode`](../../../builders/mla/ref_mla_decode.py), fp64,
  fed the same bf16 inputs as the device. Its own tests tie it to the expanded
  (per-head `K_nope`/`V`) form, to the prefill oracle at one query per sequence,
  and to a single-pass softmax for the split merge.
- **Tolerance:** elementwise `|got − ref| ≤ 4e-2 + 2e-2·|ref|` on the bf16
  output (DESIGN §8.1's absolute bound plus the MLA prefill harness's relative
  one), `|got − ref| ≤ 4e-2` on the fp32 LSE. The relative term is needed:
  outputs of the whole op peak at 6–10 here, where one bf16 ulp alone is 3–6e-2,
  and exact math with only the required bf16 roundings already used about 78%
  of an absolute-only budget.
- **Traps the gate cannot miss:** output buffers and the split workspace are
  NaN-seeded, and every cache row no sequence owns (including each sequence's
  partial last page) is NaN, so an unwritten slot or a key read past `S_k` fails.
  Block-table slots past a sequence's last page are `-1`. `S_k = 0` rows are
  checked exactly.
- **Independent GPU oracle:** the prefill `absorbed_query` kernel run on the same
  device buffers at one query per sequence: different code, the same math.

## Reproducing

From `rocke/library`, on a gfx942 node, with ROCm torch:

```bash
python -m builders.mla.verify_mla_decode --num-seg auto 1 2 7 16 64     # latent, every scenario
python -m builders.mla.verify_mla_decode --mode full --num-seg auto 1 7  # whole op
python -m builders.mla.verify_mla_decode --mode oracle                  # vs the prefill kernel
python -m benchmarks.common.benchmark_mla_decode_live --mode verify      # the shape file, whole op
python -m pytest -m gpu tests/test_mla_decode_numeric.py                 # the CI lane
```

The scenarios (in [`verify_mla_decode.py`](../../../builders/mla/verify_mla_decode.py)):
`ragged` (one key, 15, 16, 17, …, 4097 keys in one batch), `no_query` (a
sequence with no query), `zero_keys` (`S_k = 0`), `long` (16384 and 32763
keys), `spike` (key 300 lifts half of every wave's heads past the lazy-rescale
threshold, so the rescale runs on a live accumulator for some lanes only),
each at 64 and 128 heads.
`--num-seg 64` exceeds most of their tile counts, so trailing segments are
empty and the combine must weigh them to zero.

## Results

Run on a gfx942 device (ROCm 7.2 torch, `ROCKE_LLVM_FLAVOR=llvm22`), seed 0, at the gate above. Every row also passed its NaN, `S_k = 0` and LSE checks. `num_seg` marked `*` is the count dispatch chose for this device; the others are forced. "max_abs / peak" is reported for scale, not gated. The numeric CI lane ([`tests/test_mla_decode_numeric.py`](../../../tests/test_mla_decode_numeric.py), three seeds for the whole op) passed on the same device.

**Latent kernel vs fp64 reference** (10/10 scenarios pass)

| scenario | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|
| `long_h128` | 64*, 1, 2, 7, 16, 64 | 0.74 | 2.28e-03 | 3.07e-03 | 2.80e-06 | pass |
| `long_h64` | 64*, 1, 2, 7, 16, 64 | 0.55 | 2.37e-03 | 4.28e-03 | 2.35e-06 | pass |
| `no_query_h128` | 4*, 1, 2, 7, 16, 64 | 3.63 | 8.53e-03 | 2.35e-03 | 1.57e-06 | pass |
| `no_query_h64` | 4*, 1, 2, 7, 16, 64 | 2.84 | 1.24e-02 | 4.38e-03 | 2.07e-06 | pass |
| `ragged_h128` | 8*, 1, 2, 7, 16, 64 | 4.02 | 1.29e-02 | 3.21e-03 | 1.84e-06 | pass |
| `ragged_h64` | 8*, 1, 2, 7, 16, 64 | 3.25 | 8.27e-03 | 2.55e-03 | 2.53e-06 | pass |
| `spike_h128` | 8*, 1, 2, 7, 16, 64 | 3.58 | 1.12e-02 | 3.12e-03 | 1.33e-05 | pass |
| `spike_h64` | 8*, 1, 2, 7, 16, 64 | 3.58 | 8.07e-03 | 2.26e-03 | 1.01e-05 | pass |
| `zero_keys_h128` | 1*, 1, 2, 7, 16, 64 | 2.93 | 1.12e-02 | 3.82e-03 | 1.90e-06 | pass |
| `zero_keys_h64` | 1*, 1, 2, 7, 16, 64 | 3.37 | 7.71e-03 | 2.29e-03 | 2.35e-06 | pass |

**Whole op ([A] bmm, kernel, [D] bmm) vs fp64 reference** (10/10 scenarios pass)

| scenario | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|
| `long_h128` | 64*, 1, 7 | 1.55 | 8.11e-03 | 5.22e-03 | 1.71e-03 | pass |
| `long_h64` | 64*, 1, 7 | 1.17 | 5.16e-03 | 4.42e-03 | 1.63e-03 | pass |
| `no_query_h128` | 4*, 1, 7 | 6.32 | 3.71e-02 | 5.87e-03 | 4.66e-03 | pass |
| `no_query_h64` | 4*, 1, 7 | 5.77 | 2.85e-02 | 4.94e-03 | 5.08e-03 | pass |
| `ragged_h128` | 8*, 1, 7 | 9.16 | 3.02e-02 | 3.29e-03 | 1.12e-02 | pass |
| `ragged_h64` | 8*, 1, 7 | 10.20 | 2.64e-02 | 2.59e-03 | 7.91e-03 | pass |
| `spike_h128` | 8*, 1, 7 | 9.10 | 2.80e-02 | 3.08e-03 | 1.03e-02 | pass |
| `spike_h64` | 8*, 1, 7 | 9.69 | 3.05e-02 | 3.14e-03 | 9.58e-03 | pass |
| `zero_keys_h128` | 1*, 1, 7 | 6.90 | 2.27e-02 | 3.29e-03 | 5.06e-03 | pass |
| `zero_keys_h64` | 1*, 1, 7 | 7.02 | 2.23e-02 | 3.18e-03 | 5.86e-03 | pass |

**Whole op over the bf16 decode shape file** (26/26 shapes pass)

| shape | `num_seg` run (* = dispatched) | peak \|ref\| | worst max_abs | worst max_abs / peak | worst LSE max_abs | gate |
|---|---|---|---|---|---|---|
| `DeepSeek-V3-R1/ds_v3_b1_kv512/b1_q1k512_h128_bf16` | 1* | 3.37 | 2.08e-02 | 6.16e-03 | 5.88e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv1024/b1_q1k1024_h128_bf16` | 2* | 4.93 | 1.03e-02 | 2.09e-03 | 1.58e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv2048/b1_q1k2048_h128_bf16` | 4* | 1.99 | 1.13e-02 | 5.68e-03 | 2.00e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv4096/b1_q1k4096_h128_bf16` | 8* | 1.82 | 1.27e-02 | 6.95e-03 | 1.63e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv8192/b1_q1k8192_h128_bf16` | 16* | 3.81 | 1.27e-02 | 3.35e-03 | 2.39e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv16384/b1_q1k16384_h128_bf16` | 32* | 0.90 | 4.37e-03 | 4.84e-03 | 1.29e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b1_kv32768/b1_q1k32768_h128_bf16` | 64* | 0.90 | 3.71e-03 | 4.11e-03 | 1.49e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b4_kv2048/b4_q1k2048_h128_bf16` | 4* | 2.71 | 1.88e-02 | 6.96e-03 | 2.33e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b8_kv2048/b8_q1k2048_h128_bf16` | 4* | 4.18 | 1.75e-02 | 4.18e-03 | 2.37e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b4_kv8192/b4_q1k8192_h128_bf16` | 16* | 2.76 | 8.81e-03 | 3.19e-03 | 1.77e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b8_kv8192/b8_q1k8192_h128_bf16` | 16* | 2.86 | 1.83e-02 | 6.40e-03 | 2.25e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b32_kv2048/b32_q1k2048_h128_bf16` | 4* | 4.32 | 1.83e-02 | 4.23e-03 | 3.66e-03 | pass |
| `DeepSeek-V3-R1/ds_v3_b32_kv8192/b32_q1k8192_h128_bf16` | 10* | 2.86 | 2.11e-02 | 7.40e-03 | 2.42e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv512/b1_q1k512_h64_bf16` | 1* | 4.06 | 1.30e-02 | 3.21e-03 | 2.58e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv1024/b1_q1k1024_h64_bf16` | 2* | 2.81 | 1.32e-02 | 4.70e-03 | 1.95e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv2048/b1_q1k2048_h64_bf16` | 4* | 2.13 | 8.79e-03 | 4.12e-03 | 1.36e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv4096/b1_q1k4096_h64_bf16` | 8* | 1.51 | 9.42e-03 | 6.25e-03 | 1.48e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv8192/b1_q1k8192_h64_bf16` | 16* | 1.11 | 4.64e-03 | 4.17e-03 | 1.24e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv16384/b1_q1k16384_h64_bf16` | 32* | 1.13 | 6.83e-03 | 6.05e-03 | 1.28e-03 | pass |
| `Kimi-K2/kimi_k2_b1_kv32768/b1_q1k32768_h64_bf16` | 64* | 0.90 | 3.40e-03 | 3.80e-03 | 1.19e-03 | pass |
| `Kimi-K2/kimi_k2_b4_kv2048/b4_q1k2048_h64_bf16` | 4* | 2.44 | 1.93e-02 | 7.91e-03 | 2.31e-03 | pass |
| `Kimi-K2/kimi_k2_b8_kv2048/b8_q1k2048_h64_bf16` | 4* | 4.26 | 1.75e-02 | 4.12e-03 | 4.98e-03 | pass |
| `Kimi-K2/kimi_k2_b4_kv8192/b4_q1k8192_h64_bf16` | 16* | 2.00 | 9.66e-03 | 4.82e-03 | 1.73e-03 | pass |
| `Kimi-K2/kimi_k2_b8_kv8192/b8_q1k8192_h64_bf16` | 16* | 2.56 | 2.38e-02 | 9.27e-03 | 2.38e-03 | pass |
| `Kimi-K2/kimi_k2_b32_kv2048/b32_q1k2048_h64_bf16` | 4* | 5.35 | 2.84e-02 | 5.31e-03 | 3.61e-03 | pass |
| `Kimi-K2/kimi_k2_b32_kv8192/b32_q1k8192_h64_bf16` | 16* | 2.95 | 1.19e-02 | 4.04e-03 | 1.96e-03 | pass |


**vs the prefill `absorbed_query` kernel on the same device buffers** (10/10 scenarios,
clean cache -- see below): the single-pass decode output is **bit-identical** to the
prefill kernel's in every scenario (max_abs 0, LSE max_abs 0). With the dispatched
split count the latent differs by at most 1.56e-2, one bf16 ulp at the scenario's
peak, from the different summation order of the merge.

The oracle runs on a clean cache because the prefill kernel does not survive NaN in
the unused rows of a sequence's partial last page: it masks those keys in the score
but contracts over them in the PV, where `0 * NaN = NaN` reaches every head
(`verify_mla_decode --mode oracle --poison-unused` reproduces it). The decode kernel
zeroes those rows before its PV and is gated on poisoned caches throughout.


## Static resources

Compiled through comgr (llvm22) and read from the code object. Zero scratch, and at most 256 VGPRs and 32 KiB of LDS: two four-wave workgroups per CU. `TestResources` in [`tests/test_mla_decode.py`](../../../tests/test_mla_decode.py) checks this in CI.

| kernel | VGPR | SGPR | LDS (B) | scratch (B) |
|---|---|---|---|---|
| `mla_decode_gfx942_h64_rkv512r64_k16_hw16_w4_bf16` | 256 | 49 | 27264 | 0 |
| `mla_decode_gfx942_h64_rkv512r64_k16_hw16_w4_split_bf16` | 256 | 53 | 27264 | 0 |
| `mla_decode_combine_h64_rkv512_w4_bf16` | 26 | 30 | 0 | 0 |
| `mla_decode_gfx942_h128_rkv512r64_k16_hw16_w4_bf16` | 256 | 49 | 27264 | 0 |
| `mla_decode_gfx942_h128_rkv512r64_k16_hw16_w4_split_bf16` | 256 | 53 | 27264 | 0 |
| `mla_decode_combine_h128_rkv512_w4_bf16` | 26 | 30 | 0 | 0 |

## Limits

- bf16 only; `page_block_size == 16`; `d_rope = 64`, `r_kv = 512`; head counts
  that are multiples of 64.
- `qa`, `out` and the split workspace use 32-bit element offsets: dispatch
  refuses batches that would overflow them (about 29K tokens at 128 heads) and
  caps the segment count to fit the workspace. The key cache is 64-bit
  addressed per page and has no such limit.
- The design note [`builders/mla/DESIGN.md`](../../../builders/mla/DESIGN.md)
  predates this kernel; where they differ, this page records what was built:
  - §7.1 names two ops, `mla_prefill_fwd` and `mla_decode_absorb_fwd`; the
    family serves one, `mla_fwd`, and `max_seqlen_q` picks the regime.
  - §4 puts query heads on the MFMA M dimension; they are on the n axis.
  - §4 takes `c_q` and a materialized `W_abs` and applies `W_UV` per segment;
    the kernel takes the absorbed `qa`, and `W_UV` runs once, after the
    combine, outside it.
  - §8.1's reference `ref_mla_decode_absorb(c_q, …, W_abs, W_rope_proj, W_UV,
    …)` is [`ref_mla_decode`](../../../builders/mla/ref_mla_decode.py), over
    the projected query and the fused per-head `W_UK`.
- LLVM 22 or later: llvm20 ignores the kernel's `agpr_alloc`, the kernel then
  spills, and dispatch refuses it.
