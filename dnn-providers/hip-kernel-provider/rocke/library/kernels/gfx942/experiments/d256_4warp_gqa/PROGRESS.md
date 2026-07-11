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

## UPDATE: SQ=4096 ticket-shape result (2026-07-11)
build_e2e_T3_sq4096.py: NUM_KV=64 (4096 keys), 32 q-blocks x 16 heads = 512 CTAs,
causal loop bound (kvend = 2*(qblock+1) tiles -> early blocks cheaper), cu_seqlens_q
packed Q/O (qstart = cu_q[0] + qblock*128).
- OURS  SQ=4096: 1367us, correct (vs fp32 ref 4e-3, bf16 accumulation over 4096 keys)
- AITER SQ=4096: 1305us
- ratio: OURS = 1.05x AITER time (~0.95x throughput) -- NEAR-PARITY on the ticket shape.
Arc: shipped CK std-QK 0.55x -> 4-warp SQ=128/512 0.87x -> 4-warp SQ=4096 0.95x AITER.

cu_seqlens_q: packed Q/O wired + single-seq validated; RAGGED (>=2 seqs diff Q len)
needs the CTA->seq schedule (binary search on cu_q) -- remaining varlen-scheduling work.

## UPDATE 2: SQ=8192 + bottleneck profiling (2026-07-11)
Ticket shapes (GQA-16/2 causal paged, vs AITER unified_attention, identical inputs):
- SQ=4096: OURS 1366us / AITER 1305us = 0.95x AITER (near-parity)
- SQ=8192: OURS 4600us / AITER 4685us = 1.02x AITER (WE EDGE AHEAD)
vs shipped CK std-QK 0.55x -> ~2x improvement to parity+.

Bottleneck profiling (rocprofv3, SQ=4096): MFMA & occupancy at PARITY with AITER.
Gap = VALU (ours 4.45e9 vs AITER 3.32e9, 1.34x) + LDS bank conflicts (6.7x).
BUT: V_lds row-stride pad [64,256]->[64,264] cut conflicts 2.3x (1.85e9->7.92e8)
with ZERO perf change (GRBM 1.39e9 unchanged) -> conflicts are HIDDEN behind compute,
NOT critical-path. amd_rotating_shared swizzle would net ~0 (measure-first, confirmed).
Real remaining lever = VALU (address-math: 101 v_lshlrev vs AITER's 6; hoist phys_key
mul/mod/shift, use bfe+add3) -- only matters more at smaller SQ (overhead-bound);
at SQ=8192 we already beat AITER.
