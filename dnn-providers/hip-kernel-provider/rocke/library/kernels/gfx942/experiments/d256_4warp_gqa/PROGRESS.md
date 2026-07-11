# D256 gfx942 attention — session progress (2026-07-11)

## Headline
The **4-warp / 128-query-per-CTA structure** is the real AITER lever (NOT the V-swizzle,
which was proven a dead end). A from-scratch 4-warp paged GQA-16/2 causal kernel reaches
**0.87x AITER** on a fair, apples-to-apples, L2-honest comparison (both correct to ~2.4e-4).

## Validated (GPU, gfx942 MI300X via kreb)
- build_e2e_T3.py            : 1-warp transposed register-softmax base (fixed K tile-offset bug -> 7.6e-5)
- build_e2e_T3_4warp.py      : 4-warp/128q, shared K/V (dense)
- build_e2e_T3_paged.py      : + paged block-table (bs=16, scattered) K/V
- build_e2e_T3_paged_cv.py   : + causal + KV-varlen masking
- gqa_kernel.py              : + GQA-16/2 (single query-block)  [importable]
- gqa_kernel_mc.py           : + multi-sequence batched          [importable]
- compare_gqa_mc.py          : head-to-head vs AITER unified_attention (identical inputs)
- build_e2e_T3_sq4096.py     : SQ=4096 ticket-shape scaffold (WIP: cu_seqlens_q + causal loop bound)

## Honest numbers (identical GQA-16/2 paged causal inputs, 64 seqs x 128q/512k, L2-honest)
- OURS  : 593us, correct (~1e-4)
- AITER : 519us, correct (~2e-4)
- ratio : OURS = 0.87x AITER   (shipped CK std-QK paged = 0.55x on SQ=4096)

## Key lessons (bug class that recurred)
Descriptor-sizing seams: buffer_rsrc size / seq offset omitted in string-edits caused
OOB->zero reads that looked like SPEEDUPS. Always validate correctness across ALL
sequences/heads + ground-truth dumps, never formula-staring or single-slice checks.
paged_mc "98.8 TF/s" was invalidated by exactly this (only seq0 was validated).

## Dead ends (proven, do not retry)
- V-staging coalescing (rotating/XOR swizzle, transpose): neutral at ceiling (~-3%); not the bottleneck.
- ds_read2st64_b64 intrinsic: unnecessary — AITER uses NO intrinsic; AMDGPU backend fuses plain LDS loads.

## Open
- cu_seqlens_q packed ragged-Q addressing (needs >=2 seqs of different Q length to validate).
- SQ=4096 single-seq ticket-representative perf number (causal loop bound essential).
- Productionize: wire 4-warp GQA kernel into the dispatcher (like gfx950 routing).
