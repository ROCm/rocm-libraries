<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE GPU attention coverage map

**Scope: only rocKE tests that exercise the GPU**, focused on attention and the
related KDA/GDN operators. CPU/IR/dispatch/compile-only checks and provider
C++/hipDNN integration tests are excluded. GPU-harness negative controls are
labelled separately and do not count as additional kernel or shape coverage.

Source inventory: commit `270fab13c2c`, reviewed 2026-10-02. The latest supplied
gfx942 CI log executed on 2026-10-01: all eight reference cases and three negative
checks passed; the wider library GPU suite reported 84 passes and 134 skips.
Published artifact packaging was separately verified at run `36897764926`.
A source-declared shape is not necessarily selected, executed, or passed in CI.

## Reading this map

- [Coverage at a glance](#coverage-at-a-glance) and [the eight required cases](#the-eight-required-cases)
- [GPU test files](#gpu-test-files)
- [Full attention shape matrix](#full-attention-shape-matrix)
- [Full recurrent shape matrix](#full-recurrent-shape-matrix)

The matrices contain 111 attention and 109 recurrent configuration/shape rows.
They describe source declarations, including cases that skip in the observed CI.

Rows are not unique shapes or pytest pass counts. Binary-reuse tests have two
shape rows; the eight new references intentionally duplicate existing cohort
configurations. The recurrent table includes a separately identified GPU oracle
negative control (`test_the_tolerance_gate_can_actually_fail`); it is not a
rocKE kernel numerical execution. The SDPA harness's three negative tests are
listed separately in the inventory, not added to the eight workload rows.

## Coverage at a glance

| GPU test area | Existing shape/configuration rows | Covered by required references? | gfx942 CI without Torch |
|---|---:|---|---|
| gfx942 base dense SDPA | 8 | Yes: equivalent configurations, different inputs | Required suite runs; original suite skips |
| gfx942 dense sliding-window | 6 | No | Skipped |
| gfx942 binary reuse across shapes | 2 shapes in 1 test | No reuse assertion or larger shape | Skipped |
| gfx942 non-default tile | 1 | No | Skipped |
| gfx942 forced exp2 A/B | 2 | No A/B assertion | Skipped |
| gfx942 paged D128 ring regression | 12 | No: different path | Skipped |
| gfx942 paged GQA head-fold | 4 | No: different path | Skipped |
| gfx950 dense attention | 51 shapes in 50 test configurations | No | Torch/device-gated |
| gfx1250 WMMA attention | 2 | No | Device-gated; verifier uses NumPy/HIP |
| Extended FMHA/Sage/sparse harness | 15 named cases | No | Outer test skips; per-case hardware guards also apply |
| KDA/GDN prefill and decode | See separate matrix | No: recurrent operators | Torch/device-gated; KDA decode file is not selected |

## The eight required cases

All use gfx942, B=1, Sq=Sk=512, Hq=16, one fixed NumPy input recipe, and
current Python-emitted kernels compared to qualified precompiled kernels.
Q/output are `[B,Sq,Hq,D]`; K/V are `[B,Sk,Hkv,D]`.

| # | Dtype | D | Hkv | Mask | Implementation |
|---|---|---:|---:|---|---|
| 1 | FP16 | 128 | 4 | Causal | Default |
| 2 | FP16 | 128 | 4 | Causal | Persistent |
| 3 | BF16 | 128 | 4 | Causal | Persistent |
| 4 | BF16 | 128 | 4 | Causal | Default |
| 5 | FP16 | 64 | 16 | Causal | Default |
| 6 | BF16 | 64 | 4 | Causal | Persistent |
| 7 | FP16 | 128 | 16 | Causal | Default |
| 8 | FP16 | 128 | 4 | Full | Default |

There are four distinct Q/K/V shape combinations, not eight unique shapes.
The additional three GPU tests are harness rejection checks (perturbed current
output, perturbed baseline output, missing launch), not more workload coverage.
The 54 host tests similarly validate the harness, not 54 GPU shapes.

## Concrete missing gfx942 coverage

| Gap | Existing shape domain | Why the eight references do not cover it |
|---|---|---|
| Sliding windows | B1, S512, Hq16/Hkv4: BF16 D128 default/persistent W128; FP16 D128 default/persistent W256; BF16 D64 default/persistent W128 | No reference case enables a sliding window |
| Runtime shape/binary reuse | FP16 D128 Hq16/Hkv4: `(B,S)=(1,512),(4,1024)` | The reference has only the first shape and no same-binary assertion |
| Non-default tile | FP16 D128 Hq16/Hkv4 B1/S512, block_n=32 | Production dispatch in the reference chooses its normal tile |
| Forced exp2 variants | BF16 D128 Hq16/Hkv4 B1/S512, default/persistent | The reference does not force fast and plain paths and compare them bitwise |
| Paged D128 ring regression | FP16/BF16 × Hq/Hkv=(32,8)/(16,16) × S=512/1024/4096, B1 | Unified/paged builder and cache layout differ from dense SDPA |
| Paged head-fold | BF16 D128, Hq32/Hkv8, `(S,W)=(2048,512),(8192,4096)` × page size16/32, B1 | No head-fold kernel, paging, long sequence, or clipped window in reference cohort |
| Seeds/distributions | One fixed normally distributed recipe per configuration in required suite | Multiple runs test repeatability, not additional input coverage |
| Additional dense combinations | No BF16 full-mask case; no D64 full-mask case; not every dtype × grid × head-ratio combination | Eight selected rows are not a Cartesian-product sweep |
| Unequal query/KV lengths, bottom-right masks | Existing detailed numeric cohort is gfx950: see 11 rows in shape table | All required Sq=Sk=512; no bottom-right or ragged reference |
| Decode / varlen / append / sparse / quantized / backward | Exact 15 extended-harness rows in shape table | Different operations, ABI layouts, and sometimes architectures |

The paged ring test also checks for an external parity harness and can skip if
it is absent. Installing Torch alone is not proof that every older test will
execute in an artifact. The inventory preserves these dependency guards.

## Recurrent KDA/GDN is separate coverage

These operators update recurrent state; a correct SDPA output does not validate
them. Chunkwise tests predominantly use B2/H4/T256/DK128/DV128 with gate ranges
-0.1/-0.5/-2/-5, plus nonzero state, packed/raw forms, chunk16/32, value partitioning,
and exact split/fused or prefetch comparisons. gfx942 also varies DK64/128 and
uses a physical DV64 kernel partition for a logical DV128 problem.

Decode tests use one token and cover batch sizes including 1,3,4,8,16,32,64,256,
Hk16/Hv32 defaults (some Hk=Hv=32), DK=DV128 (plus DK64), state/IO dtype variants,
padding and untouched-state behavior, tuning bands, and large-pool addressing.
See the explicit rows: not every combination is tested. Prep-tile checks are
listed as tile shapes rather than inventing an equivalent batch/sequence shape.

## Selection and interpretation

- `test_kda_decode_gfx950_numeric.py` contains GPU numerical tests but is **not
  selected by the installed library CTest entries**.
- The paged ring test can also skip if its parity helper is absent; supplying
  Torch alone does not guarantee execution.
- Other-architecture rows require their target GPU. Their skip on gfx942 is
  not itself a test failure.
- The gfx1250 file also contains host checks and GEMM probes; those are excluded
  from this attention GPU map.
- The platform `test_rocke_numeric.py::test_extended_parity` has stale attention
  wording, but its current helper moved FMHA/Sage/sparse attention cases to the
  library harness. It is not counted as additional attention execution here.
- Module-level Torch skips hide whole parameterized cohorts. Raw skip counts
  cannot be interpreted as numbers of missing GPU shapes.

**Main gaps:** sliding windows; long and runtime-varying shapes; separate
paged/ring/head-fold paths; additional architectures; and recurrent operators.
The eight references are required base-cohort coverage, not a drop-in replacement for
all these GPU tests.

## Input equivalence and limitations

“Equivalent configuration” means matching the original base case's dtype, shape,
mask, and default/persistent choice. It does not mean matching input values.
The reference suite uses NumPy PCG64 seed 0, standard-normal float32 samples,
then fp16/bf16 rounding and exact digest checks. Original Torch tests generate
random samples on the GPU through Torch. The samples differ even when the seed
number is equal. There is one qualified input recipe per row, not a seed sweep.
See [the SDPA guide](sdpa-test-reference.md) for the oracle and accuracy budget.

## GPU test files

| File | GPU test functions/methods | Installed selection |
|---|---:|---|
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) | 5 | Selected by library CTest file list |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) | 6 | Selected by library CTest file list |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) | 1 | Selected by library CTest file list |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) | 1 | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) | 13 | Selected by library CTest file list |
| [test_gfx1250_attention.py](../library/tests/test_gfx1250_attention.py) | 2 | Selected by library CTest file list |
| [test_gfx942_gqa_head_fold_numeric.py](../library/tests/test_gfx942_gqa_head_fold_numeric.py) | 1 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) | 6 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) | 14 | Selected by library CTest file list |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) | 8 | not selected by installed CTest |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) | 2 | Selected by library CTest file list |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) | 3 | selected: required SDPA GPU |

The inventory includes 12 files and 62 GPU-related test functions/methods,
including separately identified harness controls. These are declaration counts,
not executed GPU-kernel counts. All filenames below are relative to
`library/tests/`. Function names identify the source declarations to inspect.

## Full attention shape matrix

| File / test | Arch | Dtype | B | Sq | Sk | Hq | Hkv | D | Variant | Required-reference coverage |
|---|---|---|---:|---|---|---:|---:|---:|---|---|
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 16 | 64 | persistent=False; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 64 | persistent=True; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | persistent=False; causal=True | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=False | Equivalent configuration (different inputs) |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_one_binary_serves_every_shape | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | same binary for both shapes; default causal | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_one_binary_serves_every_shape | gfx942 | fp16 | 4 | 1024 | 1024 | 16 | 4 | 128 | same binary for both shapes; default causal | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | causal; window=128; persistent=False | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | causal; window=128; persistent=True | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | causal; window=256; persistent=False | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | causal; window=256; persistent=True | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 64 | causal; window=128; persistent=False | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_swa_numeric_vs_fp32_sdpa | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 64 | causal; window=128; persistent=True | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_dense_fp16_d128_numeric_correct_at_non_shipped_tile_width | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | block_n=32; default causal | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_exp2_fast_matches_plain_exp2 | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | forced fast/plain exp2 exact equality; persistent=False | Missing |
| [test_attention_dense_gfx942_numeric.py](../library/tests/test_attention_dense_gfx942_numeric.py) :: test_exp2_fast_matches_plain_exp2 | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | forced fast/plain exp2 exact equality; persistent=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | fp16 | 1 | 512 | 512 | 16 | 16 | 64 | persistent=False; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | persistent=True; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_dense_numeric_vs_fp32_sdpa | gfx950 | bf16 | 1 | 512 | 512 | 16 | 4 | 64 | persistent=True; causal=True | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_one_binary_serves_every_shape | gfx950 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | same binary for both shapes; default causal | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_one_binary_serves_every_shape | gfx950 | bf16 | 4 | 1024 | 1024 | 16 | 4 | 128 | same binary for both shapes; default causal | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_gqa_pair_variant_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'_name': 'plain', 'sliding_window': 0, 'use_sinks': False, 'interleave': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_gqa_pair_variant_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'_name': 'interleave', 'sliding_window': 0, 'use_sinks': False, 'interleave': True} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_gqa_pair_variant_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'_name': 'sliding_window', 'sliding_window': 128, 'use_sinks': False, 'interleave': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_gqa_pair_variant_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'_name': 'sinks', 'sliding_window': 0, 'use_sinks': True, 'interleave': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_gqa_pair_variant_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'_name': 'sinks_sliding_window', 'sliding_window': 128, 'use_sinks': True, 'interleave': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_wide_dma_mha_numeric | gfx950 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | persistent; wide DMA; block_m=256 | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_wide_dma_mha_numeric | gfx950 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | persistent; wide DMA; block_m=128 | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 64 | {'dtype': 'bf16', 'd': 64, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 64 | {'dtype': 'bf16', 'd': 64, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 64 | {'dtype': 'bf16', 'd': 64, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 64 | {'dtype': 'bf16', 'd': 64, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 128, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 128, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 128, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 128, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 256, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 256, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 256, 'causal': True, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'fp16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 256, 'causal': True, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': False, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': False, 'sw': 0, 'causal': False, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': False, 'sink_magnitude': 'above_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_sinks_numeric | gfx950 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | {'dtype': 'bf16', 'd': 128, 'hq': 32, 'hkv': 8, 'persistent': True, 'sw': 0, 'causal': False, 'sink_magnitude': 'below_qk_max'} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 197 | 400 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 197, 'skv': 400, 'batch': 1, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 300 | 1234 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 300, 'skv': 1234, 'batch': 1, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 512 | 4097 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 512, 'skv': 4097, 'batch': 1, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 100 | 8000 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 100, 'skv': 8000, 'batch': 1, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 2 | 300 | 1000 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 300, 'skv': 1000, 'batch': 2, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 256 | 512 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 256, 'skv': 512, 'batch': 1, 'ragged': False, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 512 | 1024 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 512, 'skv': 1024, 'batch': 1, 'ragged': False, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 512 | 1024 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 512, 'skv': 1024, 'batch': 1, 'ragged': False, 'use_sinks': True} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 300 | 1234 | 4 | 1 | 128 | {'geometry': 'default', 'sq': 300, 'skv': 1234, 'batch': 1, 'ragged': True, 'use_sinks': True} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 512 | 1024 | 4 | 1 | 128 | {'geometry': 'bm128', 'sq': 512, 'skv': 1024, 'batch': 1, 'ragged': False, 'use_sinks': False} | Missing |
| [test_attention_dense_gfx950_numeric.py](../library/tests/test_attention_dense_gfx950_numeric.py) :: test_jit_matches_shifted_diagonal_not_top_left | gfx950 | bf16 | 1 | 197 | 400 | 4 | 1 | 128 | {'geometry': 'bm128', 'sq': 197, 'skv': 400, 'batch': 1, 'ragged': True, 'use_sinks': False} | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 512 | 512 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 1024 | 1024 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 4096 | 4096 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 512 | 512 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 1024 | 1024 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | bf16 | 1 | 4096 | 4096 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 512 | 512 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 1024 | 1024 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 4096 | 4096 | 32 | 8 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 1024 | 1024 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_attn_bf16_d128_ring.py](../library/tests/test_attn_bf16_d128_ring.py) :: test_d128_numeric_vs_fp32_oracle_at_magnitude | gfx942 | fp16 | 1 | 4096 | 4096 | 16 | 16 | 128 | paged/unified causal; unit-variance input; ring regression | Missing |
| [test_gfx942_gqa_head_fold_numeric.py](../library/tests/test_gfx942_gqa_head_fold_numeric.py) :: test_fold_numeric_vs_fp32_windowed_oracle | gfx942 | bf16 | 1 | 2048 | 2048 | 32 | 8 | 128 | paged head-fold; window=512; page size=16 | Missing |
| [test_gfx942_gqa_head_fold_numeric.py](../library/tests/test_gfx942_gqa_head_fold_numeric.py) :: test_fold_numeric_vs_fp32_windowed_oracle | gfx942 | bf16 | 1 | 8192 | 8192 | 32 | 8 | 128 | paged head-fold; window=4096; page size=16 | Missing |
| [test_gfx942_gqa_head_fold_numeric.py](../library/tests/test_gfx942_gqa_head_fold_numeric.py) :: test_fold_numeric_vs_fp32_windowed_oracle | gfx942 | bf16 | 1 | 2048 | 2048 | 32 | 8 | 128 | paged head-fold; window=512; page size=32 | Missing |
| [test_gfx942_gqa_head_fold_numeric.py](../library/tests/test_gfx942_gqa_head_fold_numeric.py) :: test_fold_numeric_vs_fp32_windowed_oracle | gfx942 | bf16 | 1 | 8192 | 8192 | 32 | 8 | 128 | paged head-fold; window=4096; page size=32 | Missing |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=True; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | fp16 | 1 | 512 | 512 | 16 | 16 | 64 | persistent=False; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | bf16 | 1 | 512 | 512 | 16 | 4 | 64 | persistent=True; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | fp16 | 1 | 512 | 512 | 16 | 16 | 128 | persistent=False; causal=True | Required GPU |
| [test_sdpa_pinned_reference.py](../library/tests/test_sdpa_pinned_reference.py) :: test_sdpa_correctness_against_qualified_rocke | gfx942 | fp16 | 1 | 512 | 512 | 16 | 4 | 128 | persistent=False; causal=False | Required GPU |
| [test_gfx1250_attention.py](../library/tests/test_gfx1250_attention.py) :: TestGfx1250Gpu::test_attention_fwd_verify | gfx1250 | fp16 | 2 | 64 | 64 | 4 | 4 | 64 | causal=False; direct LLVM path | Missing |
| [test_gfx1250_attention.py](../library/tests/test_gfx1250_attention.py) :: TestGfx1250Gpu::test_attention_fwd_causal | gfx1250 | fp16 | 2 | 64 | 64 | 4 | 4 | 64 | causal=True; HIP/clang path | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_appendkv_norope | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 3 new | 7 initial; capacity 16 | 4 | 4 | 64 | KV append; no attention output | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_appendkv_rotary | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 3 new | 4 initial; capacity 32 | 2 | 2 | 64 | KV append + rotary | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_varlen_causal | runtime gfx942/gfx950; per-case capability gates | f16 | 2 | [16,32] | [16,32] | 2 | 2 | 64 | packed 48 queries; causal | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_head_grouping_gqa | runtime gfx942/gfx950; per-case capability gates | f16 | 2 | 16 | 16 | 4 | 2 | 64 | causal | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_fwd_fp8 | runtime gfx942/gfx950; per-case capability gates | f16 Q / FP8 KV | 1 | 16 | 16 | 2 | 2 | 64 | noncausal; per-tensor scales; architecture guard | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_fwd_splitkv_decode | runtime gfx942/gfx950; per-case capability gates | f16 | 2 | 1 | [16,12] | 2 | 2 | 64 | 4 split-KV segments | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_fwd_paged_prefill | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 4 | 48 | 2 | 2 | 64 | page16; noncontiguous physical blocks [3,1,5] | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_bwd | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 16 | 16 | 2 | 2 | 64 | dQ/dK/dV; atomic accumulation | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_fmha_fwd_mfma | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 16 | 16 | 2 | 2 | 64 | MFMA forward | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_sage_attention_fp16 | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 8 | 32 | 2 | 2 | 64 | scales=1 | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_sage_attention_fp8 | runtime gfx942/gfx950; per-case capability gates | FP8 | 1 | 8 | 32 | 2 | 2 | 64 | per-block scales | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_sage_attention_i8 | runtime gfx942/gfx950; per-case capability gates | i8/fp8 | 1 | 8 | 32 | 2 | 2 | 64 | integer QK quantization | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_sage_attention_i4 | runtime gfx942/gfx950; per-case capability gates | i4/fp8 | 1 | 8 | 32 | 2 | 2 | 128 | packed integer QK quantization | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_jenga_sparse | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 16 | 64 | 2 | 2 | 64 | block_k=32; second KV block selected | Missing |
| [test_extended_parity_attention.py](../library/tests/test_extended_parity_attention.py) :: case_vsa_sparse | runtime gfx942/gfx950; per-case capability gates | f16 | 1 | 16 | 128 | 2 | 2 | 64 | block_k=32; LUT [0,3] | Missing |

## Full recurrent shape matrix

KDA/GDN are recurrent operators, not softmax SDPA. T=1 denotes one decode step.
Logical shapes are listed; packed layouts, physical partitions, and variant
parameters remain in the notes. No row is covered by the SDPA reference cohort.
`test_the_tolerance_gate_can_actually_fail` is a GPU-oracle negative control,
not a rocKE kernel execution; it remains here to expose that distinction.

| File / test | Arch | B | T | Hk | Hv | DK | DV | Dtype | Parameters / notes | Installed selection |
|---|---|---|---|---|---|---|---|---|---|---|
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_the_tolerance_gate_can_actually_fail | gfx950 | 4 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_matches_fp32_reference | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 1}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_matches_fp32_reference | gfx950 | 3 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 3}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_matches_fp32_reference | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_matches_fp32_reference | gfx950 | 64 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 64}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_simple_reference_path_matches | gfx950 | 4 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_every_dispatched_tile_is_correct | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 1}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_every_dispatched_tile_is_correct | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_every_dispatched_tile_is_correct | gfx950 | 64 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 64}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_every_dispatched_tile_is_correct | gfx950 | 256 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 256}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_padding_lanes_are_skipped_and_leave_state_untouched | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_mismatched_skip_index_leaves_write_page_untouched | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_large_pool_crosses_the_i32_offset_boundary | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; pool slots=floor(2^31/(Hv*DV*DK))+1; memory-gated; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_results_are_deterministic | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_state_dtype_variant_is_correct | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; f16 state; bf16 IO; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_f16_io_variant_is_correct | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | f16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_use_qk_l2norm_off_matches_reference | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_fallback_head_dim_geometry_is_numerically_correct | gfx950 | 1 | 1 | 16 | 32 | 64 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_gdn_decode_gfx950_numeric.py](../library/tests/test_gdn_decode_gfx950_numeric.py) :: test_end_to_end_through_the_dispatch_result | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx942 | tile_count=128 | chunk=16 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -0.1}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx942 | tile_count=128 | chunk=16 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -0.5}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx942 | tile_count=128 | chunk=16 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -2.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx942 | tile_count=128 | chunk=16 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -5.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_split_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_split_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_split_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_split_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_path_matches_token_serial | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_kt_generation_is_stable | gfx942 | 2 | 256 | 4 | 4 | 64 | 128 | bf16 | {"head_k": 64}; logical DV=128; spec tile head_v=64; chunk16; four repeated checks per DK | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_kt_generation_is_stable | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"head_k": 128}; logical DV=128; spec tile head_v=64; chunk16; four repeated checks per DK | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_split_and_fused_agree_bitwise | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx942_numeric.py](../library/tests/test_kda_chunkwise_gfx942_numeric.py) :: test_fused_input_prefetch_is_bitwise_identical | gfx942 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {}; logical DV=128; spec tile head_v=64; chunk16 | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx950 | tile_count=128 | chunk=32 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx950 | tile_count=128 | chunk=32 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx950 | tile_count=128 | chunk=32 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_prep_tiles_match_float64_oracle | gfx950 | tile_count=128 | chunk=32 | packed tiles | packed tiles | 128 | 128 | bf16 | {"gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_split_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_split_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_split_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_split_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "split", "gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "split", "gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "split", "gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "split", "gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "fused", "gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "fused", "gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "fused", "gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_nonzero_initial_state_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"path": "fused", "gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_subtiled_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_subtiled_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_subtiled_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_subtiled_fused_path_matches_token_serial | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_subtiled_and_chunk_wide_fused_agree_to_one_ulp | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_input_prefetch_is_bitwise_identical | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"tile_kw": {}};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_input_prefetch_is_bitwise_identical | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"tile_kw": {"block_size": 512, "scan_atom_m": 16}};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_fused_lds_overlay_is_bitwise_identical | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_c32_tile_phase_16x16_panels_agree_to_one_ulp | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_c16_fused_matches_token_serial_oracle | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_split_and_fused_agree_bitwise | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_raw_split_matches_aligned_oracle | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.1};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_raw_split_matches_aligned_oracle | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -0.5};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_raw_split_matches_aligned_oracle | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -2.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_raw_split_matches_aligned_oracle | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"gate_low": -5.0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_value_splits_agree_with_vs1 | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"value_splits": 1, "block": 256, "scan_atom_m": 0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_value_splits_agree_with_vs1 | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"value_splits": 2, "block": 128, "scan_atom_m": 0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_value_splits_agree_with_vs1 | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"value_splits": 4, "block": 64, "scan_atom_m": 0};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_value_splits_agree_with_vs1 | gfx950 | 1 | 256 | 4 | 4 | 128 | 128 | bf16 | {"value_splits": 8, "block": 64, "scan_atom_m": 16};  | Selected by library CTest file list |
| [test_kda_chunkwise_gfx950_numeric.py](../library/tests/test_kda_chunkwise_gfx950_numeric.py) :: test_dispatched_specs_match_the_token_serial_oracle | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {};  | Selected by library CTest file list |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_simple_path_matches_reference | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 1}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_simple_path_matches_reference | gfx950 | 3 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 3}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_simple_path_matches_reference | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_simple_path_matches_reference | gfx950 | 64 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 64}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_gdn_simple_path_still_matches_reference | gfx950 | 4 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_warp_tiled_matches_reference | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 1}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_warp_tiled_matches_reference | gfx950 | 3 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 3}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_warp_tiled_matches_reference | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_warp_tiled_matches_reference | gfx950 | 64 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 64}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_every_kda_tuned_tile_is_correct | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16, "tile": [4, 16, 4]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_every_kda_tuned_tile_is_correct | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16, "tile": [1, 16, 4]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_every_kda_tuned_tile_is_correct | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16, "tile": [2, 16, 1]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_dispatch_band_launches_selected_kernel | gfx950 | 1 | 1 | 32 | 32 | 128 | 128 | bf16 | {"batch": 1, "expected_spec_id": "kda_w128", "expected_tile": [4, 16, 4]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_dispatch_band_launches_selected_kernel | gfx950 | 8 | 1 | 32 | 32 | 128 | 128 | bf16 | {"batch": 8, "expected_spec_id": "kda_w512", "expected_tile": [1, 16, 4]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_dispatch_band_launches_selected_kernel | gfx950 | 32 | 1 | 32 | 32 | 128 | 128 | bf16 | {"batch": 32, "expected_spec_id": "kda_w_large", "expected_tile": [2, 16, 1]}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_kda_mha_shape_matches_reference | gfx950 | 8 | 1 | 32 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_precomputed_decay_matches_reference | gfx950 | 1 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 1}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_precomputed_decay_matches_reference | gfx950 | 16 | 1 | 16 | 32 | 128 | 128 | bf16 | {"batch": 16}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_decode_gfx950_numeric.py](../library/tests/test_kda_decode_gfx950_numeric.py) :: test_precomputed_decay_agrees_with_fused_gate | gfx950 | 8 | 1 | 16 | 32 | 128 | 128 | bf16 | {}; output and updated state checked; state [pool,Hv,DV,DK] | not selected by installed CTest |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"Hv": 4, "Hk": 4, "gate_low": -0.5, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"Hv": 4, "Hk": 4, "gate_low": -0.5, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"Hv": 4, "Hk": 4, "gate_low": -5.0, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 4 | 128 | 128 | bf16 | {"Hv": 4, "Hk": 4, "gate_low": -5.0, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 8 | 128 | 128 | bf16 | {"Hv": 8, "Hk": 4, "gate_low": -0.5, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 8 | 128 | 128 | bf16 | {"Hv": 8, "Hk": 4, "gate_low": -0.5, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 8 | 128 | 128 | bf16 | {"Hv": 8, "Hk": 4, "gate_low": -5.0, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 4 | 8 | 128 | 128 | bf16 | {"Hv": 8, "Hk": 4, "gate_low": -5.0, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 8 | 32 | 128 | 128 | bf16 | {"Hv": 32, "Hk": 8, "gate_low": -0.5, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 8 | 32 | 128 | 128 | bf16 | {"Hv": 32, "Hk": 8, "gate_low": -0.5, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 8 | 32 | 128 | 128 | bf16 | {"Hv": 32, "Hk": 8, "gate_low": -5.0, "with_h0": false};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_split_parity | gfx950 | 2 | 256 | 8 | 32 | 128 | 128 | bf16 | {"Hv": 32, "Hk": 8, "gate_low": -5.0, "with_h0": true};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_prefill_dispatched_value_splits | gfx950 | 8 | 256 | 4 | 8 | 128 | 128 | bf16 | {"batch": 8, "expected_value_splits": 8};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_prefill_dispatched_value_splits | gfx950 | 16 | 256 | 4 | 8 | 128 | 128 | bf16 | {"batch": 16, "expected_value_splits": 2};  | Selected by library CTest file list |
| [test_kda_gdn_gfx950_numeric.py](../library/tests/test_kda_gdn_gfx950_numeric.py) :: test_gdn_prefill_dispatched_value_splits | gfx950 | 32 | 256 | 4 | 8 | 128 | 128 | bf16 | {"batch": 32, "expected_value_splits": 1};  | Selected by library CTest file list |

## Maintaining the map

When adding or changing a GPU test, update its file/function mapping, explicit
parameter rows, dependency and device guards, installed CTest selection, and
reference-case relationship. Expand source parameterizations and helper defaults;
retain symbolic dimensions when runtime-generated instead of inventing a shape.
Record execution evidence separately, including source revision, target, selected
cases, passes, skips, and skip reasons. Re-run collection or inspect module guards
when a module-level skip hides its full cohort. Keep CPU/IR/dispatch-only checks
and provider C++ integration outside this GPU attention map.

This map does not inventory convolution tests. A convolution reference cohort
needs its own operation-specific coverage map, as described in the
[strategy and extension guide](gpu-ci-pinned-rocke-test-reference-plan.md#add-convolution-or-another-operation).
