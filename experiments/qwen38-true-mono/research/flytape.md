I've read everything you listed. Below is the report.

# FlyTape Qwen4 decode megakernel (gfx950): technical report

Repo root: `C:\Users\vanantha\Documents\dev\flytape-main\flytape-main`. All paths below are relative to it. Byte and grid-sync counts marked **(derived)** are my own arithmetic from the shapes; everything else is quoted from the code or docs.

---

## 1. Model shape and precisions

**Config** (`flytape/models/qwen4/config.py:49-93`, derived properties `:117-221`). This is Qwen3.8-Flash-Next (`qwen4_exp`), text decoder only, no MTP head or vision tower.

| Item | Value |
|---|---|
| vocab / hidden / layers | 248320 / 2560 / 48; `eos_token_id` 248044 |
| Layer pattern | `full_attention_interval=4`: layers with (i+1)%4==0 are QSA, so 36 GDN + 12 QSA |
| Gated residual ("HC") | `hc_count`=4 streams, so `hc_hidden`=10240; `hc_lowrank`=320; `gr_down_rows`=324 (320 down rows + 4 inject rows) |
| GDN | 16 key heads × 128, 48 value heads × 128. key_dim 2048, value_dim 6144. conv_dim 10240, conv kernel 4. `gdn_in_rows` = 10240 + 6144 + 48 + 48 = **16480** (qkv, z, b, a) |
| QSA | 24 q heads, 2 KV heads, head_dim 256, rotary 64, rope_theta 1e7. q_rows = 2·24·256 = 12288 (query and gate interleaved per head). kv_rows 512. Indexer: 4 heads × 128 plus 1 key → 640 rows. `qsa_in_rows` = **13952**. out_dim 6144. Indexer budget 2048 tokens = 512 blocks of 4 (`indexer_compress_ratio`) |
| MoE | 512 experts, top-10, intermediate 640, shared expert 640. Router has 513 rows (512 + shared-expert gate). 11 slots per row (slot 10 is the shared expert) |
| PLE / n-gram | One PLE layer (`ple_layer_ids=(2,)`, so zero-based layer 1). ngram_size 3, 8 heads per n-gram → 16 hash heads × head_dim 160. Each head's table size is the next prime above 20M, padded to a multiple of 128 rows → ~320M rows × 160 FP8 bytes ≈ **51 GB**. `ple_proj` is [12800, 2560]. Conv kernel 4, dilation 3, history 9 |

**Schedule** (`model.py:53-72`): `(gdn, gdn_ple, gdn, qsa)×1 + (gdn, gdn, gdn, qsa)×11`. There are three layer types: `gdn`, `gdn_ple`, `qsa`. The run-length-encoded groups become dynamic loops; each run is traced once (`runtime.py:1539-1590`).

**Precision presets** (`precision.py:35-52`). Every weight not listed here is bf16, and activations are bf16 apart from the MoE under `mxfp4`.

| Preset | Routed experts | Shared expert | n-gram table | MoE activations |
|---|---|---|---|---|
| `bf16` | bf16 | bf16 | bf16 | bf16 |
| `fp8` | FP8 E4M3, 128×128 block, bf16 scales, logical row-major | bf16 | FP8, one scale per tensor | bf16 (W8A16) |
| `fp8_preshuffle` (**model default**) | Same, in AITER 16×16 shuffle (`byte = (n>>4)·16K + (k>>4)·256 + (n&15)·16 + (k&15)`, `expert-preshuffle.md:13-20`) | bf16 | FP8, per tensor | bf16 |
| `mxfp4` | MXFP4, 32-column groups, E8M0 scales, 16×16 shuffle + `e8m0_shuffle` | MXFP4 shuffled | FP8, per tensor | **W4A4**: expert input and SiLU product quantized per 32 values, RCEIL scale |

- The mixers (`gr_down`/`gr_up`) stay bf16 in every preset (`qwen4/README.md:74`).
- Routed gate/up matrices are spaced `EXPERT_GAP`=256 bytes apart so different experts don't land on the same HBM channels. Any gap from 256 B to 2 KiB measured faster; 4 KiB did not (`weights.py:37-40`).
- Weights are random, checkpoint-layout tensors. There is no loader (`weights.py:101-191`).

**TP sharding** (`weights.py:340-677`):
- GDN and QSA split by heads. TP2 gives 8 key / 24 value GDN heads and 12 q / 1 KV QSA heads. TP4 is rejected because 2 KV heads don't divide.
- MoE splits the intermediate dimension, not the experts. MXFP4 at TP2 is an exact 320/320. FP8 at TP2 stores 384 columns per rank with 320 real.
- The indexer, router, mixers, embedding and n-gram table are replicated. The LM head is vocab-parallel.

---

## 2. Per-token phase lists

**Common facts:**
- 256 threads per workgroup (4 waves), one workgroup per CU, `waves_per_eu=1` (`model.py:355`). On 256 CUs that is 1024 waves, one per SIMD, so latency is hidden only by keeping loads in flight.
- Every phase defaults to a grid sync before it (`model.py:14`).
- Prefetch hooks issue a GEMV's first-unit weight loads between `grid_arrive` and `grid_wait` (`runtime.py:1607-1615`, `bodies.py:2549-2566`).
- Phase bodies receive `site.ordinal` (index within the layer type) and `site.position` (absolute layer, used for router, mixers and experts).

### 2a. GEMV tape (batch 1–4, `GEMV_MAX_BATCH=4`, `model.py:48`)

**GEMV core** (`blocks/gemv.py`):
- A wave owns whole weight rows. It issues every 16-byte `buffer_load_dwordx4` of its rows before the first FMA.
- Activations come from a bf16 LDS `stage` page via `ds_read_b128`.
- Products use `v_dot2_f32_bf16`. fp8 is decoded with `cvt_scalef32_pk_bf16_fp8`, fp4 with `cvt_scalef32_pk_bf16_fp4`. Block scales are applied by an FMA on each chunk's partial (`gemv.py:115-133, 279-327`).
- Each wave reduces with a DPP row reduce plus `readlane` (`mem.wave_sum`).
- Units are dealt round-robin over the 1024 waves. `pipelined`/`drain` keeps one or two units in flight (`slots`); the ping-pong in `_drain_two` avoids carrying load registers across the loop back edge (`gemv.py:545-609`).
- Batch is unrolled into per-lane accumulators. One weight load serves every row.

Weight bytes are per occurrence at TP1 **(derived)**.

| # | Phase (body) | Computes | Sync | Decomposition | Weight bytes |
|---|---|---|---|---|---|
| P | `embed` (`bodies.py:2330`) | Gathers embedding row into all 4 `r_a` streams; zeroes `s_mlp`. TP: pushes zeros to `moe_y`, then an `embed_sync` grid sync | NONE | Grid-strided 16-byte vectors | 5 KB per row |
| 1 | `attn_mix_down` (`_mix_down`, `bodies.py:239-333`) | Combines `r + s·y` → stage (bf16 B×10240); 4-stream RMS stats (one batched block reduction through `work`); normalizes with `(1+gr_norm)`; writes `r_b` and `rrms`; GEMV over `gr_down[2·pos]` (324 rows × K 10240) → `down` f32 | GRID (TP: WORKGROUP, polls `moe_y.prev` inside the load) | `row_blocks`, 1 row per unit, `spread=True`. Only 324 of 1024 waves have work | 6.6 MB |
| 2 | `attn_mix_up` (`bodies.py:359-416`) | stage = `silu(down/4)`; `s_attn = 2σ(inject/4)`; GEMV over `gr_up` (10240 rows × K 320). Epilogue: `x = Σ_i σ(acc_i)·r_i·rrms_i·(1+g_i)/4` | GRID | Unit = 1 column × 4 stream rows → 2560 units, `slots=2`. `MIX_UP_COLS=1`: 3 columns was 45 µs/step faster but faulted (unresolved) (`:336-340`) | 6.55 MB |
| 3 | `gdn_proj` (`bodies.py:600-675`) | GEMV over `gdn_in`. Lane epilogue does depthwise conv (3-deep state) + SiLU → `gdn_act`, z, β = σ(b), g = −exp(A_log)·softplus(a + dt_bias) | GRID + prefetch | 2 rows per unit → 8240 units, ~8 per wave | 84.4 MB |
| 4 | `gdn_recur` (`bodies.py:685-779`) | Per (row, value head): L2-normalize q (×1/√128) and k; S ← exp(g)·S; kv = Sᵀk; Δ = (v − kv)·β; S += k⊗Δ; o = Sᵀq; RMSNorm(o)·w·σ(z) → `core` | GRID | Workgroup per (row, head) = B·48 units over 256 workers. Fp32 VALU only (no MFMA). Thread owns 4 value columns × 32 keys; 16-byte state accesses | State 64 KiB read + 64 KiB written per (row, head) → 6.3 MB per row |
| 5 | `gdn_out` (`bodies.py:554-571`) | GEMV [2560 × 6144] → `y_attn`. TP: pushes packed bf16 pairs to `attn_y` | GRID + prefetch | 1 row per unit (2 under TP), `slots=2` | 31.5 MB |
| 6 | `mlp_mix_down` | Same as 1, with `gr_down[2·pos+1]` | GRID (TP: WORKGROUP, polls `attn_y`) | — | 6.6 MB |
| 7 | `mlp_mix_up` | Same as 2 → `x`, `s_mlp` | GRID | — | 6.55 MB |
| 8 | `moe_router` (`bodies.py:587`) | GEMV over `router[pos]` (513 × 2560) → f32 logits | GRID + prefetch | 2 rows per unit → 257 units | 2.6 MB |
| 9 | `moe_up` (`bodies.py:1716-1738`) | Every workgroup redundantly runs top-10 (one wave per row, 10 iterative wave argmaxes), renormalized softmax over the top-k, shared gate σ → LDS `ids`/`wts`. Shared-expert units first (Plan over I=640, rows n and n+I), then routed. Routed mapping by preset: `fp8_preshuffle` → `_UpTiles` (16-row tile as two 1 KiB atoms, ping-pong, K split over 4 waves, partials in LDS `work`); `fp8`/`mxfp4` → `_Up` octet (8 rows for fp8, 16 for mxfp4, per workgroup unit, K quarters); bf16 → `_RoutedUp`. Writes `act = silu(g)·u` (f32 under W4A4, after fake-quantizing the staged x) | GRID + prefetch (shared expert) | Routed round-robin starts `ceil(I/4)`=160 workgroups in, so shared units sit on workgroups 0–159 | Per (row, slot): fp8 3.28 MB, mxfp4 1.64 MB (+3% scales). Shared bf16 6.55 MB |
| 10 | `moe_down` (`_Down`, `bodies.py:1773-2140`) | Quad unit = 4 output rows × 1 batch row. Wave w reads routed slots w, w+4, w+8; waves 2–3 take the shared expert's K halves instead of a third slot. Router weights fold into the per-piece multiplier. Fixed-order LDS sum → `y_mlp`, or TP push to `moe_y` | GRID | B·640 units. Two units in flight, no live set at the back edge | Per slot: fp8 1.64 MB. Shared bf16 3.28 MB |

**`gdn_ple` layer** adds, in front of these phases:
- `ple_proj` (WORKGROUP sync): hashes bigram/trigram → 16 table rows; gathers FP8 rows into stage; GEMV [12800 × 2560] × table scale → `ple_kv`. 65.5 MB.
- [TP only] `ple_reduce` (NONE).
- `ple_mix` (GRID): per-stream σ(sign·√|g|) gate, dilated conv over `ple_hist`, writes `r_b`.
- `ple_copy` (GRID): `r_b` → `r_a`, zero `s_mlp`.

(`model.py:321-327`, `bodies.py:2153-2309`)

**`qsa` layer** replaces phases 3–5 with:
- `qsa_proj`: GEMV, 2 rows per unit → 6976 units, 71.4 MB.
- `qsa_core`: row-gang, see §3.
- `qsa_out`: as `gdn_out`, 31.5 MB.

**Epilogue `sample`:**
- `final_mix_down`: `gr_down[96]`, 320 rows.
- `final_mix_up` → `h`.
- `lm_head` (`bodies.py:2368`): 2 vocab rows per wave unit → 124160 units; per-workgroup argmax into `partial_val`/`partial_idx`; reads the host cancel word. 1.27 GB.
- `select_token`: workgroup `block_argmax` over every worker's partial; updates tokens, history and done flags; writes `tokens_out`; sets the verdict. Under TP it uses `sample_exchange`.

**Totals (derived):**
- 10 phases per layer type, all grid-synced, plus 2 extra for PLE and 4 in the epilogue → **≈486 grid syncs per step** at TP1.
- Weights read per token at TP1 B=1 fp8 ≈ 11 GB: mixers 1.27, GDN 4.17, QSA 1.23, MoE 2.83, router 0.13, LM head 1.27, PLE 0.07. Against the measured 6.1 ms at B=2, that is ~1.8 TB/s effective.
- The authors say small-shape phases run at about 1.4–1.8 TB/s and are latency-bound (`expert-preshuffle.md:51`).
- `scripts/run_qwen4.py --trace` prints MB and TB/s per GEMV phase (`run_qwen4.py:57-116`).

### 2b. Tiled tape (batch 5–128, `MAX_TILED_BATCH=128`; required above 4, allowed at any batch)

**GEMM core** (`blocks/tiled.py`):
- `v_mfma_f32_16x16x32_bf16` with weights as the A operand, loaded straight from global memory into registers. Batch is the N dimension (ceil(B/16) token tiles).
- Activations go through LDS in 128-wide K chunks, double-buffered, row stride 136 (`XSTRIDE`, padded against bank conflicts).
- `DEPTH`=2 chunks of loads in flight. Loops up to `UNROLL`=4 chunks are unrolled.
- Workgroup unit = 64 weight rows (16 per wave) × a K slice. `SplitK` writes f32 partials to `part`, which are summed in split order: either by a reduce phase or by the group's last arriver through an atomic counter (`fold_count`).

| Phase | What | Decomposition (TP1 unless noted) |
|---|---|---|
| `{attn,mlp,final}_mix_combine` (`bodies_tiled.py:112-198`) | Wave per (row, stream): `r + s·y` → bf16 `dst`, f32 rrms. TP: polls exchange lines; at B=128 also warms the `gr_up` lines the later up worker on the same XCD will read | B·4 wave units over `xcd_worker` (XCD round-robin slot). GRID (TP: WORKGROUP) |
| `*_mix_down` (`:241-287`) | GEMM over `gr_down`. A `TwoStage` source normalizes stream·rrms·(1+g) on the fly | SplitK 324 rows (6 groups) × K 10240 (80 chunks), 40 splits = 240 units. At B=128: two 64-token blocks × 20 splits, still 240 units |
| `*_mix_reduce` (`:305-338`) | Sums splits, /4, silu → `down` bf16 [B, 320]; gate `2σ` | Grid-strided |
| `*_mix_up` (`:341-386`) | 16-column unit, wave = stream, K=320 all in flight; σ·residual·rrms·(1+g); 4-wave sum through LDS `mix_acc` → `x` | 160 units |
| `gdn_proj` (`:706-720`) | GEMM + last-arriver fold + GDN epilogue on row quads. Ragged rest rows run one unit per K chunk and are summed in `gdn_recur` | `GDN_SPLITS=2` (TP2: 128 groups → 256 units) |
| `gdn_recur` | Same fp32 VALU body as the GEMV tape (`bodies.gdn_recur_with`) | (row, head) |
| `gdn_out` / `qsa_out` (`:494-514`) | GEMM K=6144, `OUT_SPLITS=6` → 40×6 = 240 units. TP1: separate `*_out_reduce` phase. TP: last arriver folds and pushes 1 KB lines | — |
| `qsa_proj` (`:560-569`) | `QSA_SPLITS=2`, last-arriver fold | TP2: 114 groups → 228 units |
| `moe_router` (`:572-624`) | SplitK 513 rows × 20 splits = 180 units. The first ceil(B/4) units spin on `router_count`, then run top-k into `route`. Spare units quantize `x` → `x4`/`x4s` (MXFP4) | — |
| `moe_up` / `moe_down` (`bodies_moe.py:387-535`) | Every workgroup builds the same per-expert token lists by an LDS counting sort (`_build_lists`). Wave unit = (active expert, 16-row tile), or a 32-row group for mxfp4; loops over the expert's 16-token blocks. fp8/bf16 are decoded to bf16 for the bf16 MFMA. MXFP4 uses `mfma_scale_f32_16x16x128_f8f6f4` (`blocks/mx.py:140`) with 16-byte scale dwords and `UP_DEPTH=DOWN_DEPTH=3`. Down unit = expert × 4 output tiles (`DOWN_TILES=4`) → f32 `part[B·11, H]`. Shared expert is its own unit kind. Expert weights use `CachePolicy.NT` (`blocks/moe.py:19`) | — |
| `moe_expert` (fused; `blocks/moe.py:355-534`) | Workgroup unit = (expert, ≤16 tokens). x4 rows are gathered into LDS by async DMA; up via scaled fp4 MFMA (5 gate + 5 up tiles per wave at I=320); `silu·up` into an LDS f32 panel; requantized to MXFP4 in LDS; down (each wave 640 rows in blocks of 4 tiles) → `part` via LDS-staged 256-byte stores | Used only for MXFP4 with intermediate = 320 (i.e. TP2) and B ∈ 12–32 or 96–128 (`FUSED_BATCHES`, `bodies_moe.py:92`) |
| `moe_combine` (`:567-615`) | y = Σ_j w_j·part in slot order. TP: one lane covers 8 columns and pushes lines | — |
| `lm_head` / `sample_reduce` / `select_token` (`bodies_head.py`) | GEMM, `HEAD_MTILES=4` (256-row units, ~970 units), argmax epilogue into LDS slots; wave per row merges; thread per row selects | — |
| PLE: `ple_gather` (WORKGROUP), `ple_proj`, `ple_stats`, `ple_apply`, `ple_copy` | — | — |

- Per layer the tiled tape has ~16 phases at TP1 versus 10 on the GEMV tape. That is why it was measured slower at B≤4: a fixed cost of about twice the phases (`qwen4/README.md:64-69`).
- TP1 at B=128 is unsupported because `gdn_state` would be 3.6e9 elements, past the i32 index limit (`model.py:75-80`, `README.md:158-160`).
- **(derived)** Per-rank bytes at TP2 B=128 MXFP4:
  - GDN state 36 × 128 × 24 × 64 KiB ≈ 7.25 GB, read and written = 14.5 GB.
  - About 470 of 512 experts are active per layer, × ~1.3 MB × 48 ≈ 29 GB.
  - Dense weights ≈ 4.8 GB.
  - Total ≈ 48 GB. This is consistent with the authors' "per-rank memory roofline about 10–11 ms" (`README.md:151`).

---

## 3. Runtime mechanics

**Residency ticket** (`runtime.py:1046-1114`):
- Thread 0 of each workgroup reads `HW_REG_XCC_ID` and `HW_REG_HW_ID` and atomic-adds a per-XCD rank (padded 256-byte words, `SYNC_WORD_STRIDE=64` ints).
- It then adds to `START` and spins until `expected_workers` have arrived, with an `s_memrealtime` timeout (100 MHz clock). On timeout it records `RESIDENCY_TIMEOUT`.
- Worker id = prefix over XCDs + `xcd_rank`, so ids are XCD-contiguous. `xcd_worker` is a round-robin slot across XCDs.
- One workgroup per CU is forced by reserving more than half the CU's LDS: `cu_lds_bytes//2 + 16`, as unused tail padding (`spec.py:660-685`).

**Grid sync** (`runtime.py:1117-1183`):
- `grid_arrive`: `s_waitcnt vmcnt(0)` + barrier, then thread 0 atomic-adds (agent scope) one of **8 stripes per XCD** (`stripe = x·8 + xcd_rank%8`).
- `grid_wait`: all 64 lanes of wave 0 each poll one stripe against an epoch-scaled expected count (cached in LDS at startup); a ballot combines them. It sits behind a `readfirstlane` uniform branch so there is no exec-narrowing join. Ends with barrier + workgroup acquire fence.
- Idle cost: 1.42 µs with one counter per XCD, 1.20 µs with 4 stripes, 1.14 µs with 8. Per-worker flags measured 1.73 µs (`runtime.py:86-89`, `design.md:324-326`).
- Removing a serial thread-0 poll loop cut the decode step 8%; the stripes cut it another 6%.

**Coherence** (`runtime.py:540-643`, `mem.py:30-38`):
- DEVICE-scope buffers use agent-scope monotonic `generic_load`/`store` for scalars, and buffer ops with `sc1` for vectors. These write through and bypass the per-XCD L2, so weights stay cached and no L2 writeback/invalidate is needed.
- WORKER scope (GDN state, conv state, PLE history) uses plain accesses.
- Peer exchanges use SYSTEM (`sc0|sc1`).
- `--noncoherent` is the negative control and must mismatch the oracle.
- Not formally correct under the LLVM memory model. The formal alternative, an LLVM agent-scope fence sync, costs 18.1 µs (`README.md:143`).

**Rotating buffers:** `.cur`/`.next` are chosen by the parity of `advances_done`, folded at trace time where possible (`runtime.py:1385-1396, 1591-1627`). In qwen4 the residual ping-pong is explicit through `r_a`/`r_b`.

**LDS:**
- Runtime pages (control words, tokens, done, hist, wave scratch) come first, then one union region shared by all layer types, with in-layer aliases as nested unions (`runtime.py:1296-1329`).
- GEMV `stage` page = B·10240 bf16 (80 KB at B=4). Tiled `xs` page = 2·tiles·16·136 bf16 (~70 KB at B=128).
- QSA: `qsa_big` (aliased with stage) and `qsa_aux`. At 256K context LDS is about 146 of 160 KiB (`qsa-long-context.md:543`).
- Any one tape uses a single `flat_work_group_size`/`waves_per_eu`.

**Request loop** (`runtime.py:1707-1739`): `while RUNNING: emit_step (prologue, schedule, epilogue); publish_step`. In `publish_step`, worker 0 thread 0 stamps `step_ticks[step]=s_memrealtime`, drains, then system-stores the cursor; the status is stored on stop. The verdict comes from `select_token`: CANCELLED > ALL_EOS > MAX_TOKENS.

**Mailbox:** `HostMapped` = `hipHostMalloc` (mapped | coherent) obtained through ctypes from `libamdhip64` found in `/proc/self/maps` (Linux-only). It holds 16 words (cursor, status), plus a separate cancel word. The host polls with plain loads (`Session.follow`, `runtime.py:400-421`). Error codes: −1 residency, −2 grid sync, −3 peer, −4 QSA.

**Phase trace:** a separate compiled kernel variant (`_traced`). Only worker 0 thread 0 stamps, after each phase's sync. A phase's time runs to the next phase's stamp, so it **includes the next sync's wait** (`runtime.py:336-360, 1687-1690`).

**TP exchange** (`runtime.py:678-951`, `docs/multi-gpu.md`):
- One process per GPU. The mailbox is uncached memory from `hipExtMallocWithFlags(hipDeviceMallocUncached)`, exported with IPC handles; the peer table is loaded once before the request loop.
- **Tagged pairs** (GEMV tape and sampling): i64 `(tag<<32 | value)`, one `buffer_store_dwordx2 sc0 sc1` per rank. Readers poll with a monotonic system-scope load until every tag matches, then add in rank order in f32, so every rank gets bit-identical results.
- **Lines** (tiled tape `attn_y`/`moe_y`): 1 KB lines of payload (lane stores 16 bytes per rank); `s_waitcnt vmcnt(0)`, then one lane stores a flag per line. Payload loads add the (zero) flag difference to their index so they can't issue before the poll completes.
- Tag = use + 1, slot = use & 1. Write-after-read safety rests on existing grid syncs; read-after-write needs no sync.
- The pushes sit in the epilogues of `gdn_out`/`qsa_out`/`moe_down`/`moe_combine`. The next `mix_down`/`mix_combine` polls while loading `y`. So the exchange adds **no phases**: two all-reduces per layer.
- Poll deadline is half of `timeout_ticks`.
- Before each launch, `PeerGroup.reset()` runs synchronize, barrier, memset, synchronize, barrier.

**QSA gang** (`bodies_qsa.py`):
- Units are B × gang (row, member); a row's members are consecutive resident workers.
- `gang_size = pow2floor(max(2, min(32, grid//B, ceil(min(max_ctx, 2051)/64), grid & -grid)))` (`:68-75`). Examples: B=2 at ctx 191 → gang 2; ctx 8K with B≤8 → 32; B=128 → 2.
- Within budget (dense path): every member normalizes and RoPEs all heads (duplicate stores agree), member 0 appends indexer keys, each member attends its split of 64-token stages.
- Past budget (sparse path):
  - Member 0 publishes an f32 indexer query to `qsa_iq` and sets the `qsa_ready` flag.
  - All members scan disjoint 256-block chunks of the FP8 compressed keys (relu dot over 4 heads × 128, one power-of-two scale per key).
  - The last arriver runs a 16-bit radix-style threshold select of the top 512 blocks in LDS, then publishes.
  - All attend. The last finisher clears the flags. Waits are bounded at 50 ms.
- Attention per wave: QK on `mfma_f32_16x16x32_bf16` (Sᵀ = K·Qᵀ, 16 query heads in N, 12 real); online exp2 softmax private to the wave; PV on `mfma_f32_16x16x16bf16_1k` with V read transposed by `ds_read_tr16_b64`; next stage's K/V prefetched into registers.
- Merges: 4 waves through LDS, then members in groups of 8 (`MERGE_FAN`) by last arriver, then groups.
- KV layout is `[layer, B, max_ctx, nkv, 2, 256]` bf16 with 64-bit row windows (`bodies.py:793-806`).

**MoE scheme summary:**
- Route: GEMV tape does it redundantly in every workgroup; tiled tape does it in the router fold.
- Up/down/combine are as tabulated above.
- Combine order is fixed by slot, so results are deterministic regardless of token-list order.

---

## 4. Benchmark harness

**`scripts/bench_qwen4_full.py`** (`:224-241`):

| Flag | Default |
|---|---|
| `--tp` | **2** |
| `--batch` | 2 |
| `--precision` | **`fp8`** (not the model default `fp8_preshuffle`) |
| `--prompt-len` | 128 |
| `--max-new` | 64 |
| `--layers` | 48 |
| `--trace` | off |
| `--qsa-splits` | auto (choices 2–64) |
| `--expert-gap` | 256 |
| `--moe-fused` / `--no-moe-fused` | auto (`FUSED_BATCHES`) |
| `--configs` | `BxP,...` (e.g. `8x128,16x8192`) |
| `--reps` | 1 |

How it runs:
- TP>1 spawns one process per rank with a gloo file rendezvous; rank r uses visible device r (`:211-261`).
- `FLYTAPE_ROOT` selects which checkout to import.
- **Weights:** `random_weights` at the rank-local shape (seed 0), with `ngram_vocab_size_base=1000`. Then the full ~51 GB FP8 n-gram table is drawn on the GPU in 2^26-element chunks (seed 1).
  - It monkey-patches `quant.quantize` to chunk large per-tensor FP8 tensors. The stock path materializes fp32 (191 GiB) or bf16 (~95 GB) (`:44-63, 112-133`).
  - The LM head is vocab-sharded.
  - One mailbox is sized for the largest config.
- **State:** a zero-initialized `Reference` at position `prompt_len − 1`. No prefill. Prompts are seed 7 (`:142-159`). The docstring argues decode work doesn't depend on state values.
- **Step time:** gaps between consecutive `step_ticks` (s_memrealtime ÷ 100 = µs), excluding the startup stamp. Reports median/min/max.
- **JSON line per sample (rank 0 only):** `rank, tp, precision, expert_gap, moe_fused, rep, batch, prompt_len, free_gib, status, steps, launch_s, qsa_splits, median_decode_step_us, min_us, max_us, wall_s`, plus `traced{…}` when `--trace` is on.
- **`--trace`:** a second launch of the traced kernel. Prints per-(layer type, phase) medians over steps of the per-step sums across tape positions.
- Reps rotate the config order. Each sample does `gc` + `empty_cache` and a new `Session` allocation.

**`scripts/bench_ab.py`** (`:242-256`):

| Flag | Default |
|---|---|
| `--arm NAME=REF_OR_PATH` | repeatable; first arm is the baseline; refs are `git archive`d into `--work-dir` (`~/.cache/flytape-bench`, env `FLYTAPE_BENCH_DIR`) |
| `--devices` | repeatable groups, e.g. `0,1` |
| `--rounds` | 4 |
| `--out` | required |
| `--bench-from` | `self` (or `arm`) |
| `--timeout` | 3600 s |
| `--metric` | `median_decode_step_us` |
| `--group` | `batch,prompt_len` |
| `--where` | `K=V` filter |
| `--skip-reps` | 0 |
| `--summarize OUT`, `--base` | — |

- The bench command goes after `--`.
- Rounds alternate ABBA by round and by device group; groups run in parallel threads.
- Each arm gets its own `FLYDSL_RUNTIME_CACHE_DIR`. Core dumps are disabled with `RLIMIT_CORE=0` and `HSA_COREDUMP_PATTERN`. `OMP_NUM_THREADS=4`. On timeout the whole process group is killed (`killpg`; Linux only).
- Outputs: `meta.json`, `runs.jsonl`, logs. The summary shows the mean of process means, delta, %, paired % ± s.e. over rounds, process sd, and in-process sd.

**`scripts/run_qwen4.py`**: shrunk by default (4 layers, 16 experts, n-gram base 1000), runs the oracle comparison, and `--trace` gives a phase table with TB/s.

**Environment:**
- gfx950 only (`detect_device`).
- FlyDSL **0.3.4.1** exactly. The multi-GPU numbers were taken on 0.4.0. `prepare()` relies on private attributes present in both.
- Never put a FlyDSL source tree on `PYTHONPATH`.
- torch with ROCm. Linux.

**Documented numbers**, all gfx950 with 256 CUs:

*Runtime and toy (`README.md:132-169`, `multi-gpu.md:475-527`):*
- Grid sync 1.44 µs (historical; 1.14 µs with 8 stripes). Graph kernel boundary 1.75 µs. Fence-based sync 18.1 µs.
- Toy: 9–11 µs per layer, 23 µs fixed per token.
- Mapped-word read from device 168 ns; device-memory read 15.8 ns; host read of mapped word 0.75 µs; host copy of one word ~17 µs.
- Toy TP, `dense*32`: B1 508.8 µs single GPU / 515.8 tp1 / 489.1 TP2 / 402.9 TP4. B4 871.9 / 865.4 / 854.2 / 740.0. Reduce phase 2.7–3.6 µs.
- `PeerGroup.reset`: 0.4–2.5 ms warm, 40–75 ms on first call.

*Qwen4 full shape:*

| Config | Step | Source |
|---|---|---|
| L4/E16 slice, `fp8`, B1 / B3, prompt 4 | 642 / 866 µs | `qwen4/README.md:25-27` |
| Full, `fp8` logical, B2, p128, 64 new, 1 GPU | **6.08 ms** median (range 6.06–6.13) vs vLLM ITL 10.22 ms (TPOT 10.23) | `:37-49` |
| Full B2, crossing the 2051-token budget | 7.22 → 7.39 ms; 7.41 at 8K (older kernel) | `:55-57` |
| Context curve, full B2 | 64: 7166 µs; 1024: 7205; 2048: 7219; 2060: 7387; 4K: 7399; 8K: 7408. Same table at L4/E16 B1/B3 out to 32K | `qsa-long-context.md` |
| Full B2 p128, `fp8` / `fp8_preshuffle` at the 16-row-atom commit | 6120 / 6119 µs. `moe_up` 876 / 898; `moe_down` 565 / 599; `mix_down` ~1186; `mix_up` ~923 µs/step | `expert-preshuffle.md:931-990` |
| TP2 MXFP4, p128, 64 new | B2 4.25, B16 8.54, B32 10.70, B64 14.09, B128 19.74 ms. B128 at 8K prompt 22.84 ms. Families at B128: MoE 8.38, mixers 5.13, GDN 4.84, QSA 1.19 ms. QSA attention 3.24 ms/step at 8K vs 0.46 at p128 | `README.md:145-163` |
| B128 TP2 MXFP4, later matched A/Bs (zero state, 32 steps) | Token blocking 17.80 → 17.53 ms at 8K, 18.66 → 18.40 at 32K. Further: −0.93% / −0.56% (XCD slot, bf16 up operand, full-depth up); −0.68–0.75% / −1.00% (up-weight warming) | `README.md:91-96` |
| Row-gang QSA vs the old two-phase path | Step 17.83 vs 18.22 ms at 8K; 17.01 vs 17.25 at p128 | `:128-138` |
| Lines vs pair exchange, TP2 B128 p128 | Exchange 3.15 → 1.9–2.1 ms; mix_combine wait 20.27 → 19.2–19.6 ms | `multi-gpu.md:159-161` |
| Tuning-constant probes | DEPTH 2/3/4 at 48 layers: 17.41/17.12, 17.97/17.95, 18.66/18.30 ms. `UNROLL`: `mix_up` 725 → 635 µs, `gdn_out` 430 → 365. `HEAD_MTILES` 1/2/4: 250/178/160 µs (8 layers). DOWN_TILES 4 vs 8: 386 vs 436 µs. UP_DEPTH 3 vs 6: 670 vs 728 µs | `tiled.py:75-86`, `bodies_head.py:56-59`, `bodies_moe.py:45-51` |

*Benchmark noise (`README.md:123-130`):* within one process 4–10 µs; process means vary ~0.6%; GPU pairs differ by up to 1%. Use ≥8 rounds to resolve a 0.5% effect. Full-shape run-to-run spread can reach 2.4%.

**Setup cost:** about 6 s per process (imports, rank spawn, random weights); a cached sample takes under 1 s.

---

## 5. Levers and bottlenecks the authors list

- **B=128 remaining levers** (`README.md:154-157`):
  - MFMA-based QSA at long context.
  - Accelerated replay recurrence for GDN. A 16-token checkpoint-and-replay was exact but took 2.7 → 6.8 ms.
  - The mixer/exchange synchronization floor. An in-phase mixer combine repeated GEMM source work and changed rounding, so it was rejected.
- **Roofline:** about 10–11 ms per rank at B=128 TP2. GEMV-tape phases run at about 1.4–1.8 TB/s and are latency-bound.
- **Not implemented** (`README.md:171-193`): cross-layer weight prefetch, a graph-launched comparison, `sc0` XCD atomics, PLE host offload (plan in `ple-offload.md`), paged KV ("plan D6"), host-side long prefill (OOM past ~8–16K).
- **Register pressure:** `.vgpr_count` 512; full-shape builds spill 146–159 VGPRs and ~1700–1960 SGPRs, with 584–628 B scratch. An earlier note records 476 VGPR + 220 AGPR. SGPR spill regression is unresolved (`expert-preshuffle.md:880-900`).
- **MoE:**
  - The shuffled down projection still reads half lines.
  - A lane order specific to the shuffled layout could close about half of the `moe_up` gap.
  - A whole-line shuffled down projection was not attempted.
  - `moe_expert` loses when its rounds are mostly idle (B≤10, B40–80).
  - Fixing the GEMV `drain` latch (load sets live across the back edge, 50–80 `v_mov` copies behind `vmcnt(0)`) was the main win: `moe_up` 1230 → 877 µs.
- **QSA:** crossing the budget costs ~11–14 µs per layer, and the select is latency-bound. A short chain at one split; the redundant select reads 32 MiB of MALL per layer at 256K.
- **TP:** TP8 untested (poll traffic, SGPR spills growing with TP); cached vs uncached mailboxes; two-shot all-reduce past 16 rows; fusing the gated residual into the reduce.
- **Other:** `MIX_UP_COLS=3` is faster but faults. `mix_down` at 1 row per unit uses only 324 of 1024 waves (derived).

---

## 6. Cluster benchmarking pitfalls

1. **Exclusive, idle GPUs.** The kernel holds one workgroup on every CU for the whole launch. Another compute process can hang it or cause `RESIDENCY_TIMEOUT`. The authors gate on an idle check (`gpu_idle.sh`). Set `HIP_VISIBLE_DEVICES` to idle ordinals and disable core dumps (`docs/testing.md:161-164`).
2. **TP means multiple processes.** FlyDSL binds a module to one device, so each rank is its own process. Ranks need xGMI peer IPC (`hipIpcGetMemHandle` on uncached memory) and a gloo file rendezvous in a temp dir. A rank that doesn't launch makes its peers end with `PEER_TIMEOUT`. Pick GPU pairs with `--devices`.
3. **Compile cost and caching.** Every distinct (batch, max_ctx = prompt − 1 + max_new, precision, tp, tile, trace, qsa_splits, moe_fused, expert_gap) is a separate kernel. The launch call compiles cold, and the host can't poll until it returns. Keep `FLYDSL_RUNTIME_CACHE_DIR` on node-local storage (bench_ab keeps one cache per arm). Compile time grows with `max_ctx`. No full-shape compile time is documented; the toy compiles in ~0.7–0.9 s.
4. **Memory.**
   - The n-gram table alone is ~51 GB of VRAM per rank (replicated).
   - At TP1 fp8 the experts are ~120 GB, ≈182 GB total (derived). The stock generator needs 95–191 GB transient memory, which the bench script avoids by chunking.
   - TP2 MXFP4 is ≈90 GB per rank plus state: `gdn_state` 7.25 GB at B=128, KV ~24 KiB per token per row.
   - TP1 cannot run B=128 (i32 index limit).
5. **Measurement semantics.**
   - Step time is device-side and excludes the first step.
   - Phase traces need a second, differently compiled launch, and each phase time includes the following sync wait.
   - State is zero-initialized and the router random, so the expert spread is near-uniform rather than checkpoint-realistic.
6. **Miscompile risk.** Lint every benchmarked build: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=…` then `scripts/lint_isa.py …/22_final_isa.s`. Exec-join spills have silently corrupted results before (`design.md:284-319`).
7. **Linux/HIP specifics.** `HostMapped` reads `/proc/self/maps`, and `bench_ab` uses `resource`/`killpg`. The grid must be divisible by the QSA gang (a power of two). The default timeouts are 20 s in `generate` and 60 s in the bench, which can be too short on a cold compile in TP if one rank lags before `prepare()`.
## Measured on the rocke partition (2026-10-29; ratios, absolute in private-perf)
- FlyTape TP1 "mxfp4" device step (random weights, prompt 128): B1 about 1.70x the plain mono's end-to-end K=0
  per-token time; B4 about 1.6x the mono's whole K=3 round. TP2 is still slower than the TP1 mono at B1.
- B4 phase shares: HC mixers about 0.34 (mix_down latency bound: 324 rows over 1024 waves), MoE about 0.31,
  GDN about 0.20, QSA about 0.10, lm_head about 0.03.
- Caveat: FlyTape's "mxfp4" preset quantizes the experts only; dense projections, mixers, router and head stay
  bf16, unlike ATOM's MXFP4 checkpoint (MXFP4 dense GEMVs). Much of the gap is bytes, not only syncs.
