# Ragged MoE Grouped GEMM Case Study (gfx950 / MI355X)

Fused gather + grouped bf16 GEMM + weighted scatter for sparse Mixture-of-Experts inference on CDNA4.

**Author:** yraparti
**Date:** 2026-07-02
**Kernel:** `rocke.instances.gfx950.ragged_moe.build_ragged_moe`
**Harness:** `rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py`

---

## Problem Statement

An MoE layer routes each token to its **top_k experts** (e.g., top_k=2 of E=64), multiplying the token's activation `A[token]` by each expert's weight matrix `B[expert]`, then combining the `k` weighted partials:

```
final[token] = Σ_k  w_k · (A[token] @ B[expert_k])
```

Naïve batched approach: scatter tokens into per-expert buffers, run `E` separate GEMMs, gather results. **Too slow** — small, irregular per-expert shapes prevent efficient MFMA utilization.

**Expert-sorted grouped GEMM:** Sort the `num_expanded = num_tokens × top_k` rows by expert, pad to tile boundaries, run as one large grouped GEMM where each block knows its expert via `expert_ids[m_tile]`. Dense MFMA efficiency on ragged workloads. But: requires **gather** (A) and **scatter** (C) index indirection to map sorted rows ↔ original tokens.

This kernel **fuses gather + GEMM + weighted scatter** into a single pass, eliminating intermediate buffers. Host pre-sorts token indices; kernel reads `A[token_indices[sorted_row]]`, computes, and writes `C[sorted_row] = result × routing_weight`.

---

## Architecture

### Inputs (host-precomputed via `_moe_align`)

- `A[num_tokens, K]` — bf16 activations (one copy, shared across all experts)
- `B[E, N, K]` or `[E, K, N]` — bf16 expert weights (RCR/NT or RRR/NN layout)
- `sorted_token_ids[num_m_tiles × TM]` — expanded token indices grouped by expert, padded to TM (sentinel `= num_expanded`)
- `expert_ids[num_m_tiles]` — expert id per m-tile
- `routing_weights[num_expanded]` — per-row routing weight
- `num_expanded` — scalar = `num_tokens × top_k`

### Grid / Tile Structure

- Grid: `(ceil(N/TN), num_m_tiles, 1)`
- Each block computes one `(m_tile, n_tile)` — exactly TM×TN output elements
- `expert = expert_ids[pid_m]` → this tile's B-slice is `B[expert, :, :]`
- No block-id `z` expert loop (unlike dense grouped GEMM); num_m_tiles is **dynamic** (depends on token distribution)

### Fused Gather (A)

Each tile cooperatively loads TM rows of `A[sorted_token_ids[m_base : m_base+TM], kt×TK : (kt+1)×TK]` into LDS via **DTL** (direct-to-LDS) async buffer-load. The token index is read from LDS-staged `sorted_token_ids`, the activation row is `token // top_k`, and padded rows (token ≥ num_expanded) clamp to row 0 (their output is discarded to a sink row).

### Grouped GEMM

Standard dense-GEMM inner loop (double-buffered DTL A/B loads, 16x16x32 bf16 MFMA, inline-asm ds_read operand fetches, s_setprio pipeline). Same proven schedule as `grouped_gemm_hip.py`.

### Weighted Scatter (C)

Each lane's accumulator fragment (4 f32) is scaled by `routing_weights[sorted_token_ids[row]]` and stored to `C[sorted_token_ids[row], n_base + col]`. Padded rows write to a **sink row** `C[num_expanded]` (discarded). Output is `[num_expanded, N]` — the **expanded** buffer where each token's `top_k` partials are separate rows. Host post-pass (`view(num_tokens, top_k, N).sum(1)`) combines them.

---

## Optimizations (CDNA4-specific)

### 1. Prologue Overlap (PLOG, default ON)

**Problem:** Token staging (`sorted_token_ids` → LDS) and B tile-0 DTL load are **serial** — the B load can only start after the token barrier. Two exposed latencies.

**Fix:** B's tile-0 load is **token-independent** (indexed by `expert` and `n_base`, not per-row tokens), so issue it **before** the token-staging barrier:

```python
if plog:
    if hoist: precompute_b()
    load_tile_b(0, 0)          # B tile-0 DMA in flight
    stage_tokens()             # token loads (overlaps B DMA)
    b.sync()                   # drains both token loads AND B DMA
    if hoist: precompute_a_gather()
    dtl_load_a_gather(As[0], 0, 0)  # A tile-0 (needs staged tokens)
```

The B global-to-LDS DMA now **overlaps** the token loads + barrier instead of running after. Collapses two latencies into one.

**Measured impact:** +5–15 TF (743 TF → 746 TF typical). Durable win, bit-identical correctness.

---

### 2. Offset Hoisting (HOIST, default ON)

**Problem:** The A-gather DTL load computes `A_offset = (token // top_k) × K + kt×TK + swizzle` **every K-tile**. The token is constant across K-tiles (only `kt` changes), so the `token // top_k` division + `row×K` multiply is redundant.

**Fix:** Precompute the **loop-invariant part** once per tile, hold it in a register:

```python
def precompute_a_gather():
    nonlocal a_gather_base
    vals = []
    for p in range(passes_a):
        row = ...
        tok = stok_lds(row)
        a_row = b.select(b.cmp_lt(tok, NEXP_v), b.div(tok, cTOPK), 0)
        vals.append(b.add(b.mul(a_row, cK), swizzle(col, row)))
    a_gather_base = vals  # held across K-tiles

def dtl_load_a_gather(buf, kt, coh):
    for p in range(passes_a):
        if hoist:
            off = b.add(a_gather_base[p], b.const_i32(kt * TK))  # +compile-time const
```

The K-loop now only adds a **compile-time constant** (`kt×TK`). Removes TM token LDS re-reads + TM div/mul per K-tile. Same technique applied to B-load offsets.

**Measured impact:** Embedded in the PLOG win (+5–15 TF total). LDS `ds_read_b32` token reads drop from 256 → 4 per tile.

---

### 3. Epilogue Fusion (EPIFUSE, default ON)

**Problem:** Default epilogue scatters all TM rows serially **after** the K-loop — an exposed tail of ~TM scattered short stores (address VALU + 4 stores/row). Long-tailed on the critical path.

**Fix:** Emit each m-row's scatter **immediately after** that row's final K-tile MFMA (inside the `_mma_block` loop). The store-issue + address arithmetic then **overlaps** the later m-rows' MFMAs:

```python
def _mma_block(a_fr, b_fr, acc, store_row_cb=None):
    for mi in range(mm):
        for nj in range(nn):
            acc[idx] = atom.emit(b, a_fr[mi], b_fr[nj], acc[idx])
        if store_row_cb is not None:
            store_row_cb(acc, mi)  # scatter row mi (overlaps next mi's MFMAs)
```

The scatter is now **pipelined** instead of exposed. Requires `RWLDS=1` (routing weights in LDS) because a mid-MFMA-pipeline global scatter-load would race the hand-managed `lgkmcnt` waits.

**Measured impact:** Default epilogue. Baseline is EPIFUSE=ON; turning it off exposes the tail. The win is baked into the 743 TF number.

---

### 4. Routing Weights in LDS (RWLDS, default ON)

**Problem:** Each output row is scaled by `routing_weights[token]`. Reading it from global is a scattered `global_load_dword` per row (TM scattered loads per tile). Mid-pipeline (EPIFUSE) this races lgkmcnt waits.

**Fix:** Stage routing weights into LDS alongside the tokens (one f32 per TM rows):

```python
def stage_tokens():
    for p in range((TM + BS - 1) // BS):
        val = b.global_load(STOK, m_base + sidx, I32)
        b.smem_store_vN(STOKL, [sidx, 0], val, 1)
        if rwlds:
            wv = b.global_load(RW, val, F32)
            b.smem_store_vN(RWL, [sidx, 0], wv, 1)

def rw_lds(row):
    return b.vec_extract(b.smem_load_vN(RWL, row, 0, dtype=F32, n=1), 0)
```

The scatter reads the weight from LDS (`ds_read_b32`) instead of a scattered global load. Drops `global_load_dword` from 33 → 2 per tile. Required by EPIFUSE.

**Measured impact:** Embedded in EPIFUSE. The LDS co-staging cost is negligible (same token-load barrier).

---

### 5. Inline-ASM ds_read with Manual lgkmcnt (ASM, default ON)

**Problem:** Compiler-generated `ds_read_b128` + fine-grained `s_waitcnt lgkmcnt(N)` are correct but slightly conservative. Measured at large K (dense kernel).

**Fix:** Emit operand reads as inline-asm volatile with **coarse `lgkmcnt(0)` drains** before each MFMA block:

```python
def ds_read_imm(buf, base_row, col_swz, fi, frag_ty):
    base_off = b.mul(b.add(b.mul(base_row, cTK), col_swz), b.const_i32(2))
    addr = b.add(b.trunc(_addr(buf), I32), base_off)
    imm = fi * AM * TK * 2
    raw = b.inline_asm(
        f"ds_read_b{ds_w} $0, $1 offset:{imm}",
        "=v,v", [addr], result_type=raw_ty, sideeffect=True
    )
    return b.vec_bitcast(raw, frag_ty)

def compute(ai, bi, acc):
    a_fr, b_fr = read_chunk(ai, bi, 0)
    for kk in range(kchunks):
        if asm_reads:
            b.s_waitcnt(lgkmcnt=0)  # coarse drain before MFMAs
        _mma_block(a_fr, b_fr, acc)
        if kk + 1 < kchunks:
            a_fr, b_fr = read_chunk(ai, bi, kk + 1)
```

The `sideeffect=True` + `"memory"` clobber makes the asm volatile (in-order wrt other ds_reads), but the backend can still schedule them freely before the `lgkmcnt(0)` fence.

**Measured impact:** Negligible at ragged-MoE K=512 (within noise). The compiler's fine-grained waits are equally efficient here. Measured +10 TF at dense K=2048. Kept as a proven lever from the dense kernel.

---

### 6. Chiplet-Aware Super-Tile Remap (CHIP, default ON)

**Problem:** Sequential m-tile assignment scatters consecutive m-tiles across XCDs (chiplets), thrashing the per-XCD L2 cache. Expert weight `B[expert]` is re-fetched from HBM for every m-tile of that expert.

**Fix:** Group `wgm` consecutive m-tiles into a **super-tile** and assign super-tiles round-robin across `num_xcds` XCDs:

```python
sw = chiplet_aware_super_tile_dynamic(
    b, wgid, num_pid_m=NMT_v, num_pid_n=c_npn,
    wgm=8, num_xcds=8, chunk_size=64
)
mt, nt = sw.row, sw.col
```

Consecutive m-tiles for the same expert now run on the **same XCD**, keeping `B[expert]` L2-resident. Reuses the dense kernel's proven super-tile remap.

**Measured impact:** +~40 TF in the dense kernel (805 TF → 745 TF when turned off). Applies equally to ragged-MoE. Durable win.

---

### 7. Int64 C-Scatter Addressing (bug fix)

**Problem:** Scatter address `orow × N + col` computed in i32 **overflows** when `num_expanded × N > 2^31` (e.g., `num_expanded=1048576, N=4096` → 4.29e9). GPU memory fault at large shapes.

**Fix:** Compute the **row term in i64** before adding the (small) column offset:

```python
cN64 = b.const_i64(N)

def c_rowbase(orow, col_i32):
    # i64 address = orow*N + col, with orow*N widened to 64-bit
    return b.add(b.mul(b.zext(orow, I64), cN64), b.zext(col_i32, I64))
```

All four scatter sites (default, OPSW, COMBINE, CSHUF) use `c_rowbase()` + i64 pointer math. The column term `n_base + warp_n_off + col_in` stays `< N` (< 2^31), safe to fold in as i32→i64 zext.

**Measured impact:** Correctness fix. Passes at `NTOK=524288 N=4096` (previously faulted). No perf change.

---

### 8. LDS Token Staging (default ON, not gated)

**Problem:** The default dense kernel loads per-row metadata (here: token index + routing weight) from global memory **per-element** in the epilogue. For ragged-MoE that's TM × c_per_lane = 1024 scattered `global_load` per tile, and the token index is also re-read in every K-tile's A-gather DTL load.

**Fix:** Stage `sorted_token_ids[m_base : m_base+TM]` into LDS **once per tile** (before the K-loop):

```python
STOKL = b.smem_alloc(I32, [TM, 1], name_hint="stok")

def stage_tokens():
    for p in range((TM + BS - 1) // BS):
        idx = b.add(tid, b.const_i32(p * BS))
        sidx = b.select(b.cmp_lt(idx, cTM), idx, cTM - 1)
        val = b.global_load(STOK, b.add(m_base, sidx), I32)
        b.smem_store_vN(STOKL, [sidx, 0], val, 1)

def stok_lds(row):
    return b.vec_extract(b.smem_load_vN(STOKL, row, 0, dtype=I32, n=1), 0)
```

The A-gather DTL and scatter epilogue then read the token via **cheap LDS reads** instead of redundant global loads. The token-staging barrier is **overlapped** by the B tile-0 DTL load (PLOG), so the latency is hidden.

**Measured impact:** Embedded in PLOG+HOIST. Removes thousands of scattered global token reads per frame.

---

## Performance

**Reference shape:** `num_tokens=524288, N=1024, K=512, E=64, top_k=2`
→ `num_expanded=1048576` rows, grid=`(4, 4126, 1)`, ~1.07 trillion FLOPS

### NT Layout (B=[E,N,K], default)

| Config | Median TF | Peak TF | Notes |
|--------|-----------|---------|-------|
| **Full optimizations** (PLOG+HOIST+RWLDS+EPIFUSE+ASM+CHIP) | **743 TF** | 746 TF | Production config (NT) |
| Dense grouped GEMM (no gather/scatter) | 805 TF | 808 TF | Ceiling (see dense case study) |
| PLOG=0 (no prologue overlap) | ~730 TF | — | Token-load + B-DMA serial (-13 TF) |
| EPIFUSE=0 (exposed scatter tail) | ~710 TF | — | Scatter tail un-pipelined (-33 TF) |
| CHIP=0 (no super-tile remap) | ~700 TF | — | L2 thrash on B weights (-43 TF) |

### NN Layout (B=[E,K,N], BRRR=1)

| Config | Median TF | Peak TF | Notes |
|--------|-----------|---------|-------|
| **Full optimizations** (BRRR=1, all opts) | **728 TF** | 731 TF | NN production config |
| Dense grouped GEMM NN (no gather/scatter) | 728 TF | 730 TF | NN ceiling (dense case study) |

**NN vs NT:** NN is ~15 TF slower (-2%) due to transpose-read path (2× `ds_read_tr16_b64` per B fragment vs 1× `ds_read_b128`). NT (RCR) is the recommended default. NN exists for compatibility with NN-native weight layouts and as the required path for COMBINE (which needs OPSW contiguous-N layout).

**Breakdown (estimated from STORESINK diagnostic + isolations):**

| Phase | Time (ms) | % Wall | Bound |
|-------|-----------|--------|-------|
| Compute (MFMAs) | 0.979 | 65.5% | Instruction throughput |
| C-scatter stores (+ address VALU) | 0.514 | 34.4% | Store bandwidth (2.15 GB @ 4.2 TB/s) |
| *Total kernel* | *1.493* | *100%* | — |
| Host top_k combine (`view.sum(1)`) | 0.542 | — | Separate pass (3.2 GB read) |

The scatter is **34% of wall time** despite EPIFUSE overlap. It's the next bottleneck. Attempts to eliminate it (COMBINE fusion via atomics, CSHUF wide stores, OPSW packed stores) all measured slower — see §Fusion Analysis.

---

## Attempted Optimizations (measured NOT wins)

### COMBINE — Fused Atomic Top-K Reduction

**Idea:** Atomically `+=` each expert's partial directly into `C[token, n]` (final output) instead of writing the expanded buffer. Eliminates the host combine pass (0.54 ms) and halves C-store volume (2.15 GB → 1.07 GB).

**Implementation:** `global_atomic_add_pk_bf16` (packed 2×bf16 per atomic) on the OPSW contiguous-N layout. Requires `BRRR=1` (NN weights). Fixed a **latent compiler bug**: the original lowering emitted `@llvm.amdgcn.global.atomic.fadd.v2bf16`, an intrinsic that **doesn't exist** in ROCm 7.2 LLVM. Rewrote to use generic `atomicrmw fadd <2 x bfloat>` + AMDGPU metadata, which correctly lowers to the `global_atomic_pk_add_bf16` HW instruction.

**Result:** **3.7× slower end-to-end** (2.03 ms baseline → 7.60 ms). Kernel drops from 743 TF → 144 TF.

**Why it loses:** Atomic RMW throughput is **~1.3 TB/s** (measured uncontended), vs ~4.6 TB/s for plain stores — a **3.5× per-op penalty** intrinsic to the read-modify-write pipeline. The fusion saves ~5 GB of bandwidth but adds ~5.6 ms of atomic serialization. Atomics throttle the entire GEMM to 144 TF because the epilogue is inline. Physics: streaming stores + BW-bound combine (2.0 ms total) beats atomic serialization (7.6 ms) decisively.

**Status:** Compilable and correct (after the lowering fix), but documented dead-end. The lowering fix is a **durable bug fix** (makes the LLVM path work for any packed-bf16 atomic) and is kept in `lower_llvm.py:1813`.

---

### OPSW — Operand Swap for Packed Stores

**Idea:** Swap MFMA operands so each lane's 4 accumulators are **contiguous N** for a single token (instead of 4 different tokens). Pack them into one b64 store `C[token, n..n+3]` (4× fewer, 8-byte-aligned, same-token contiguous).

**Result:** **Slower** at all shapes. 740 TF (compute-bound) → 611 TF; 544 TF (store-heavy) → 411 TF.

**Why it loses:** The default fragment already coalesces each token's N across **16 adjacent lanes** in one 32 B transaction. OPSW instead maps adjacent lanes to **different token rows** → scattered 8 B stores (cache-line thrash). Plus the NN transpose-read penalty (2× `ds_read_tr16_b64` vs 1× `ds_read_b128`). Trading wave-coalesced 2 B stores for lane-packed 8 B scattered stores is a losing bargain.

**Status:** Gated off (OPSW=0 default). Kept as experimental lever only.

---

### CSHUF — C-Shuffle Wide-Store Epilogue

**Idea:** Reuse the dead-after-K-loop A/B LDS pool as a `[TM,TN]` C-stage. Scatter weighted accumulators into LDS (the "C-lane transpose"), then cooperatively emit **wide b128** coalesced stores per row. Widens stores 16× (128 `global_store_short` → 0, 16 `global_store_dwordx4`) with zero extra LDS.

**Result:** **Slightly slower** than EPIFUSE at all shapes (within ±30 TF noise). Slower when store-bound (N=4096/K=256).

**Why it loses:** Two extra workgroup barriers (`sync` before C-stage write, `sync` before read) + the serial LDS round-trip outweigh the store-width gain. The per-element scatter is already **wave-coalesced** (16 lanes/token → 32 B transaction), so widening to b128 only saves ~0.1 ms. EPIFUSE (**overlap** the narrow stores with MFMAs) is cheaper than CSHUF (**serialize** wide stores after a barrier). The LDS round-trip is not free.

**Status:** Gated off (CSHUF=0 default). Kept as documented negative result.

---

## Fusion Analysis: Why Combine Can't Be Fused

The top_k combine (`final[token] = C[token×top_k : (token+1)×top_k].sum()`) costs 0.54 ms (host pass, BW-bound at 5.9 TB/s). Eliminating it via in-kernel fusion is the obvious next lever. Three approaches, all measured:

### 1. Atomics (COMBINE lever) — 3.7× slower

Cross-block accumulation requires either atomics or co-scheduling. Atomics are 3.5× slower per-op than plain stores (1.3 vs 4.6 TB/s, **uncontended**). The slowness is **per-op cost**, not contention (top_k=2 → only 2 atomics/element). A tree reduction can't help — it removes contention we don't have. Physics wall: atomic RMW pipeline is intrinsically slower than fire-and-forget stores.

### 2. Co-scheduling a token's top_k experts — kills GEMM reuse

To accumulate online, one block must compute all `k` partials for the same token. Two ways:

**Option A:** Group by token, loop over `k` inside the block. Within one `TM=256` tile of tokens, **each token has different experts**, so for a given `k` the TM rows need **TM different B** weight matrices. Dense MFMA requires all rows share the **same B** (that's operand reuse). If every row needs its own B, it's no longer a GEMM — it's batched gemv with zero B-reuse. Destroys ~740 TF compute to save a 0.54 ms combine.

**Option B:** Make one block compute two experts' tiles. Expert e0's tile and expert e1's tile contain **different, mostly-disjoint token sets**. The overlap (tokens routed to both e0 and e1) is small and data-dependent. Computing 2×TM rows to fuse the handful that overlap is massive redundant compute.

### 3. Online-softmax analogy — doesn't transfer

Online-softmax works because **one block owns the entire reduction axis** (K/context) — it sees every input to the running max/sum in sequence. MoE-combine's reduction axis is **top_k**, and a token's `k` partials are produced by `k` **different blocks** computing different experts. No single block owns the axis. The enabling precondition doesn't hold.

### Conclusion

Online reduction needs the reduction axis to be **sequential within one owner**. MoE's expert-sorted layout scatters a token's partials across blocks **by construction** (required for dense-GEMM efficiency). You can have online combine *or* dense-GEMM operand reuse — not both. The combine stays a separate BW-bound pass at 5.9 TB/s ceiling. The only optimization there is **free**: use `C[:,0]+C[:,1]` (0.54 ms) instead of `view(num_tokens, top_k, N).sum(1)` (0.74 ms) — 0.18 ms saved with zero kernel risk.

---

## Comparison to HipKittens

HipKittens' grouped GEMM uses **half-block-barrier phase-desync** (producer/consumer wave specialization) for pipelining. Measured at same shape (256×256×64 tile, K-starved regime):

| Implementation | TF | Notes |
|----------------|-----|-------|
| rocKE ragged-MoE | 743 TF | DTL + inline-asm + EPIFUSE |
| HipKittens grouped | 139–665 TF | Half-block barrier, per-MMA barriers, s_setprio toggle |

**Why phase-desync doesn't apply here:**

Phase-desync splits the block into **producer** (loads next tile) and **consumer** (compute current tile) waves via a half-block barrier (`if (warp_row == 1) s_barrier()`). Only works when the producer can issue loads **independently** from global. Our ragged-MoE uses **cooperative DTL** (async_buffer_load_lds_addr) — all waves participate in the load, so they can't be split. The architectural precondition doesn't hold.

HipKittens' own numbers (139–665 TF range) confirm **no win at K-starved shapes** (our regime). Phase-desync targets deep-K CPU-like out-of-order benefit, not GPU MFMA throughput.

---

## Lessons Learned

1. **PLOG prologue overlap** (+5–15 TF) — token-independent work (B-load, offset precompute) issued before the token barrier. Collapsed two serial latencies into one. Durable win.

2. **EPIFUSE epilogue fusion** — scatter pipelined into the final K-tile's MFMAs instead of exposed tail. Required RWLDS (routing weights in LDS) to avoid mid-pipeline global scatter-load hazards. Default epilogue.

3. **Int64 addressing** — correctness fix for large `num_expanded × N`. Compute row term in i64 before adding column offset. No perf cost, passes at overflow shapes.

4. **Atomic fusion loses on per-op cost, not contention** — uncontended `global_atomic_pk_add_bf16` is 3.5× slower than plain stores (1.3 vs 4.6 TB/s). Tree reduction can't help. Measured 3.7× slower end-to-end.

5. **Fixed latent LLVM bug** — `@llvm.amdgcn.global.atomic.fadd.v2bf16` intrinsic doesn't exist. Rewrote to generic `atomicrmw fadd <2 x bfloat>` with AMDGPU metadata → correct `global_atomic_pk_add_bf16` lowering. Durable compiler fix.

6. **Expert-sorted layout prevents online combine** — a token's partials are scattered across blocks by construction (required for dense-GEMM reuse). Can't fuse without atomics or killing GEMM efficiency. Combine stays a separate BW-bound pass.

7. **Wave-coalesced beats lane-packed** — default fragment coalesces 16 lanes/token → 32 B transactions. OPSW packed stores scatter to different cache lines → slower. Don't optimize the wrong dimension.

8. **Inline-asm ds_read at ragged-MoE K=512 is neutral** — compiler fine-grained waits perform identically to hand-managed coarse `lgkmcnt(0)` (within noise). Measured +10 TF only at dense K=2048. Kept as proven dense-kernel lever.

---

## Files

- **Kernel builder:** `rocke/instances/gfx950/ragged_moe.py` — `build_ragged_moe()` pure IR construction
- **Harness:** `rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py` — workflow (test, time, validate)
- **Dense parent:** `rocke/examples/gfx950/grouped_gemm/grouped_gemm_hip.py` — 805 TF dense grouped GEMM (no gather/scatter)
- **Atomic lowering fix:** `rocke/core/lower_llvm.py:1813` — generic `atomicrmw` for packed-bf16

---

## Run

```bash
export PYTHONPATH=Python PATH=/opt/rocm/llvm/bin:$PATH
python3 Python/rocke/examples/gfx950/grouped_gemm/ragged_moe_hip.py

# Env levers:
PLOG=1 HOIST=1 RWLDS=1 EPIFUSE=1 CHIP=1 ASM=1 SWZ=1  # defaults (full opts)
BRRR=1  # NN weights (transpose-read path, needed for COMBINE)
COMBINE=1  # fused atomic combine (slow, documented dead-end)
CSHUF=1  # wide-store epilogue (neutral, off by default)
NTOK=524288 N=1024 K=512 E=64 TOPK=2  # shape knobs
```

**Output:**
```
[rmoe] N=1024 K=512 E=64 top_k=2 num_tokens=524288 num_expanded=1048576 num_m_tiles=4126
[rmoe] grid=(4, 4126, 1) max_abs=0.2500 rel=0.0097 PASS
[rmoe] med=743.5 TF  peak=745.2 TF
```

---

**End of case study.**
