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
- SQ=8192: OURS median 4594us / AITER median 4647us = ~1.01x (consistently ~1-2% ahead;
  3 runs OURS 4598/4590/4594, AITER 4683/4635/4648 - ours never overlaps AITER's range,
  but margin is small: report as 'at parity, edge to us' not a bald 'beats')
vs shipped CK std-QK 0.55x -> ~2x improvement to parity+.

Bottleneck profiling (rocprofv3, SQ=4096): MFMA & occupancy at PARITY with AITER.
Gap = VALU (ours 4.45e9 vs AITER 3.32e9, 1.34x) + LDS bank conflicts (6.7x).
BUT: V_lds row-stride pad [64,256]->[64,264] cut conflicts 2.3x (1.85e9->7.92e8)
with ZERO perf change (GRBM 1.39e9 unchanged) -> conflicts are HIDDEN behind compute,
NOT critical-path. amd_rotating_shared swizzle would net ~0 (measure-first, confirmed).
Real remaining lever = VALU (address-math: 101 v_lshlrev vs AITER's 6; hoist phys_key
mul/mod/shift, use bfe+add3) -- only matters more at smaller SQ (overhead-bound);
at SQ=8192 we already beat AITER.

## UPDATE 3: next bottleneck = scalar strided-key V read (2026-07-11)
Rich rocprof (SQ=4096) + ISA region split (in-loop = per-KV-tile):
  in-loop VALU: OURS 1497 vs AITER 1218 (1.23x/iter). MFMA 128=128 (parity).
  LDS insts: OURS 1.10e9 vs AITER 3.69e8 (2.97x). NOT LDS-wait-bound (0.45x AITER).
  -> VECTOR-ISSUE-BOUND: VALU(4.45e9)+LDS(1.10e9)=5.55e9 ~= busy 5.36e9.
ROOT CAUSE (single): the scalar [key,dim] V read. Excess in-loop ops all trace to it:
  222 v_or (per-read LDS address) + 98 v_lshlrev,1 (bf16 byte offset)
  + 144 v_perm (pack scalars->MFMA operand) + 256 ds_read_u16 (scalar).
  AITER reads same strided keys in 1 ds_read2st64_b64 (strided-paired), no addr/pack.
FIX = strided-paired wide read (ds_read2st64_b64) via rotating-swizzle layout.
  NEW vs prior neutral V-staging tests: those measured conflicts/latency (neutral;
  pad disproved conflicts). This is the ISSUE-COUNT axis (the actual binding resource)
  - never measured before. Potential win is real, but needs the multi-day rotating
  layout (store side measured net -26% in isolation; read-side win now justified).
Cheap separable slice: 98 v_lshlrev,1 (x2 byte offset) - fold into element-addressed
  ds_read if rocKE emits it (small).

## UPDATE 4: issue-count hypothesis DISPROVEN by proxy - rewrite NO-GO (2026-07-11)
Throwaway proxy: hacked V read to 1 wide n=4 ds_read (wrong numerics, ~4x fewer reads),
optimistic upper bound (fully contiguous, NO store penalty). Result:
  LDS insts 1.097e9 -> 1.73e8 (6.3x FEWER), VALU 4.45e9 -> 3.96e9 (-11%)
  BUT SQ_BUSY_CYCLES 5.36e9 -> 6.55e9 (+22%), time 1366 -> 1654us (+21% SLOWER).
=> Cutting LDS 6.3x + VALU 11% made it SLOWER. Vector-issue-bound model was
   CORRELATION not causation. Strided-paired ds_read2st64 rewrite = NO-GO
   (optimistic upper bound already fails; store -26% never even matters).
Three independent measurements now agree LDS/V-read is NOT binding:
  (1) pad cut conflicts 2.3x -> 0% perf; (2) SQ_WAIT_INST_LDS 0.45x (barely wait);
  (3) proxy cut LDS 6.3x -> busy ROSE 22%.
REAL bottleneck: MFMA-dependency/scheduling-bound (~10 cyc/MFMA; accumulator chain
  serializes). AITER's 7% edge @4096 = LLVM-vs-Triton instruction SCHEDULER, a
  compiler-level diff, not a kernel-structural lever. We are at the rocKE ceiling:
  parity+ with AITER (0.95x @4096, ~1.01x @8192). No cheap kernel lever remains.

## UPDATE 5: CORRECTNESS BUG FIXED - Q-offset per q-block (2026-07-11)
Found via tight diagnostic: sq4096 kernel loaded Q[0:128] for EVERY q-block
(qi missing qstart, line 55). Loose 2.85e-3 tolerance MASKED it (softmax over
2000+ keys averages -> wrong-Q output landed "close"). Confirmed: kernel matched
ref(Q[0:128]) at 4.5e-5 (exact) but correct ref(Q[qb*128]) only 3.6e-3.
FIX: qi = (qstart + wq + n_in_atom)*H + qhead (mirror output oi indexing).
After fix: qblock16/31 = 4.6e-5/3.5e-5 (genuinely correct); ours-vs-AITER
1.5e-2 -> 1.9e-3 (bug was the main AITER-discrepancy source). PERF UNCHANGED
(1375us@4096, 4597us@8192) - Q[0:128] vs Q[qb*128] load costs identically.
So all prior perf numbers STAND; correctness for qblock>0 now actually validated.

## UPDATE 6: ragged multi-seq cu_seqlens_q scheduling DONE (2026-07-11)
build_e2e_T3_ragged.py: CTA->seq schedule via host-precomputed SID[gqb]/LQ[gqb]
(seq + local-qblock per global qblock), per-seq q_start=cu_seqlens_q[sid]+lqb*128,
qlen/klen per seq, block-table offset sid*NPHYS, causal loop bound min((lqb+1)*2,
ceil(klen/BN)), variable-length row mask via scf_if store guard.
GENERALIZED for qlen!=klen (context_off=klen-qlen, bottom-right causal = flash/vLLM/AITER):
  causal_t=ceil((context_off+qbase+128)/BN), q_g=context_off+in_seq_pos. Validated 3 ragged:
  seq0 prefill 300/300, seq1 DECODE 1/500, seq2 CHUNKED 100/500 - manual AND independent
  torch-SDPA(explicit bottom-right mask) BOTH agree: 8.4e-4/7.6e-5/9.4e-5.
  (SDPA is_causal=True is TOP-LEFT - wrong for qlen!=klen; kernel bottom-right is correct.)
Inherits scatter-paging from sq4096 (identical phys_key + sid offset).
NOTE: uses host-precomputed schedule (SID/LQ arrays); production ABI would derive
seq via in-kernel binary_search on cu_seqlens (rocKE has binary_search_seq_idx).

## UPDATE 7: productionization port plan (2026-07-11)
STATUS: ragged DONE+validated. Productionize = multi-day compute-core port (NOT config flip).
Finding: build_stdqk_attention_paged already supports num_warps=4 + binary_search_seq_idx
+ paged ABI, BUT its 4-warp config is 1.6-2.0x SLOWER (line 266) - it's the std-QK
SPILLING algo. Our 4-warp GQA (natural-QK + LDS-5stage-swizzle softmax + conflict-free-V)
is a DIFFERENT algo at parity+. So port our core into the wired production shell.

Precise steps (all Python; branch C++ diff vs develop is EMPTY = Python-only, like gfx950):
1. New build_gfx942_4warp_gqa(spec,arch) in gfx942/attention_tiled_2d.py emitting
   _attn_signature (18-arg: output/query/key_cache/value_cache/sink/block_tables/
   seq_lens/alibi/qq_bias/query_start_len/scale/k_scale/v_scale/out_scale/softcap/
   num_seqs/block_table_stride[/qq_bias_stride]).
2. CTA->seq: binary_search_seq_idx on query_start_len_ptr (REPLACES host SID/LQ).
   grid=(total_q_blocks*H,), total_q_blocks from launch meta = sum(ceil(qlen_i/128)).
3. Paged K/V: block_tables_ptr[seq*block_table_stride + logical_block]; key_cache
   [num_blocks,block_size,num_kv_heads,head_dim] (matches our phys_key + bt_stride).
4. Compute core = validated build_e2e_T3_ragged.py algorithm (per-block Q offset FIX,
   context_off=seq_len-qlen bottom-right causal, GQA kvh=qhead//GQAG, 4-warp/128-q).
5. Launch meta _get_2d_launch_meta: grid above, block (256,1,1).
6. Cache key: _d256_gfx942_fast discriminator already present.
7. Wire: attention_unified.py:3391 build_stdqk_attention_paged -> build_gfx942_4warp_gqa
   under _d256_gfx942_fast (gate/selectors/cache/seam ALL already exist).
8. GPU-validate: run_unified_attention_torch SQ=4096 GQA-16/2 causal paged vs AITER +
   fp32 ref (reuse builders/gfx942/attention/prefill/parity_unified_attention.py).
RISK: full-dispatch GPU validation is the gate; est 1-2 days. Not shipped unvalidated.

## UPDATE 8: productionization Port Step 1 DONE (2026-07-11)
build_e2e_T3_ragged_bs.py: replaced host-precomputed SID/LQ with in-kernel
binary_search_seq_idx(CUQ, gqb, num_seqs, block_q=128) - the SAME helper + pattern
as production build_stdqk_attention_paged (attention_tiled_2d.py:5590). Derivation
mirrors production exactly: cu_q_start=CUQ[sid], qlen=CUQ[sid+1]-cu_q_start,
q_block_start=cu_q_start//128+sid, lqb=gqb-q_block_start, klen=KL[sid],
context_off=klen-qlen (== production context_len line 5604). Validated dual-ref:
prefill 300/300, decode 1/500, chunked 100/500 -> 8e-4/7e-5/1e-4 both refs agree.
Remaining: Step2 emit _attn_signature (runtime scalars scale/num_seqs/bt_stride);
Step3 wire seam attention_unified.py:3391 + full-dispatch GPU validation.
NOTE: MARKHAM defq queue ~2min/alloc; switching to miopen/AUSTIN pool (kreb skill +
beegfs /scratch staging since /home not shared) for subsequent runs.

## UPDATE 9: productionization Port Step 2 DONE (2026-07-11)
build_e2e_T3_ragged_abi.py: emits full production _attn_signature (18-arg paged ABI)
+ RUNTIME scalars: scale (folds log2e in-kernel), num_seqs (-> binary_search),
block_table_stride (-> per-seq block-table offset sid*bt_stride). Param order/names
mirror attention_tiled_2d.py:5547-5580 exactly. Validated dual-ref 8e-4/7e-5/1e-4.
CAVEAT (advisory): validates COMPUTE + ABI-plumbing (runtime scalars flow), NOT
layout compat - harness builds K/V/block_tables per ASSUMED layout, kernel+ref share
it. Layout (key_cache [num_blocks,block_size,num_kv_heads,head_dim]; block_table_stride)
is an offset-arithmetic HYPOTHESIS until Step 3 validates vs PRODUCTION-generated
paged inputs via run_unified_attention_torch (feed exact dispatch tensors, don't rebuild).
Step3 remaining: build_gfx942_4warp_gqa(spec) in attention_tiled_2d.py (read HD/H/HKV/
block_size/dtype from spec, not module consts) + launch meta (grid from problem.total_q)
+ swap seam attention_unified.py:3391 + parity harness (final_shapes_check.py) diff.
