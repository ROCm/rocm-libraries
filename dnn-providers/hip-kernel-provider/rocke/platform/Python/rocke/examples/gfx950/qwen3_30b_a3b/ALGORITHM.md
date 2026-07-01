# The A3B decode step, layer by layer — strategy and the dispatch model

This is the *why* behind the seven benchmark scripts in this folder: what one
decode step of **Qwen3-30B-A3B** actually computes, why each layer is shaped the
way it is on gfx950 (MI355X), and — the recurring theme of the whole example —
**why for kernels this small the win is almost never "a faster kernel" but
"a cheaper way to dispatch it."**

> If you just want to *run* the scripts and read the measured ledger, see
> [`README.md`](README.md). This file explains the model, the per-layer kernel
> strategy, and the two cost models (GPU compute vs. host dispatch) that decide
> every number.

---

## 0. Notation

All symbols are the `_common.py` model constants for the canonical A3B decode
shape (batch = 2).

| symbol | value | meaning |
|---|---|---|
| `T` | 2 | tokens in the batch (decode = 1 new token per active sequence) |
| `H` | 2048 | hidden (model) dimension |
| `I` | 768 | per-expert MoE intermediate dimension |
| `E` | 128 | number of experts |
| `K` | 8 | top-K experts chosen per token |
| `nhead_q` | 32 | query heads |
| `nhead_k` | 4 | KV heads (GQA group ratio = 8) |
| `head_dim` | 64 | per-head dimension |
| `block_size` | 16 | paged-KV cache block size |
| dtype | bf16 | weights and activations (f32 accumulate) |

The shape that defines the whole example is **`T = 2`**: a decode step processes
a *handful* of tokens. Every GEMM is therefore "skinny" (`M ≪ K, N`), every
norm/router kernel is tiny, and the MoE routing is sparse (`T·K = 16` active
(token, expert) pairs out of `T·E = 256` possible). This is why the dominant
cost is not arithmetic — it is getting the work onto the GPU.

---

## 1. What one decode step computes (the specification)

For each new token, an A3B transformer block runs, in order:

```
x1            = rmsnorm(x + residual)                 # input norm
q, k, v       = x1 · W_qkv^T                          # QKV projection  (skinny GEMM)
attn          = paged_decode_attention(q, k_cache, v_cache)   # GQA, head_dim 64
x2            = x + attn · W_o^T                       # O projection + residual
x3            = rmsnorm(x2)                            # post-attention norm
                                                       #   (pre-MoE norm is the same op again)
logits        = x3 · W_router^T                        # router
w, idx        = topk_softmax(logits)                  # K experts per token + weights
y             = Σ_{e ∈ topk}  w_e · down_e( silu(gate_e(x3)) ⊙ up_e(x3) )   # fused MoE
x_out         = x2 + y
```

The seven scripts each isolate one of these pieces, benchmark the `rocke`
kernel against the production AITER/ATOM kernel, and verify correctness:

| layer | script | kernel family |
|---|---|---|
| RMSNorm (+ residual) | `02` | `add_rmsnorm2d_bf16` |
| QKV / O projection | `01` | `gemm_universal` (skinny BF16) |
| paged decode attention | `03` | unified 3D split-KV |
| router top-K | `04` | `topk_softmax` |
| token sort (fallback path) | `05` | `MoeSortingLauncher` (3-kernel) |
| fused MoE forward | `06` | `FusedMoeForward` |
| whole step (Amdahl roll-up) | `07` | all of the above |

`RMSNorm` appears **three times** per block (input, post-attention, pre-MoE), so
`07` counts it three times when weighting the end-to-end roll-up.

---

## 2. The two cost models (read this before any number)

Every kernel here is timed with the **batched-event** timer in `_common.ms`: it
brackets `ITERS = 200` launches under a single `torch.cuda.Event` pair and
divides, repeating `REPEATS = 5` times and taking the median. The reason is
arithmetic: a `torch.cuda.Event` pair costs ~2–5 µs of host overhead per pair on
gfx950/ROCm 7. For a 3 µs kernel, one pair *per launch* would inflate the reading
by 67–167 %. Batching 200 launches under one pair amortizes the event overhead to
~0.01–0.025 µs/iter.

But the timer still measures whatever the *launch path* costs, and that splits
the kernels into two regimes:

- **GPU-compute-bound** (the GEMMs, attention, the fused MoE). The kernel runs
  long enough (tens of µs) that the ~5–8 µs host dispatch is a minority of the
  reading. Here a faster kernel is a faster number, and the wins are *real
  compute* wins.
- **Dispatch-bound** (RMSNorm ~3 µs, router top-K ~2 µs). The GPU work is shorter
  than the ~5–8 µs the HIP runtime spends building and submitting the launch
  packet. The reading is dominated by *calling* the kernel, not running it.

For the dispatch-bound layers the lever is **CUDA-graph capture**
(`_common.capture_graph`): record the launch packet once, then `graph.replay()`
submits the whole recorded stream as a single ~0.45 µs packet, independent of how
many kernels it contains. **The GPU does exactly the same work** — the speedup is
purely the removed host dispatch. The README is explicit that the 13–30× figures
on RMSNorm / top-K are *dispatch-path* wins, not algorithmic kernel wins; this
document is where the distinction comes from. The ~0.45 µs replay cost is itself
a floor: any kernel faster than ~0.45 µs on the GPU still reads ~0.45 µs through
the graph.

CUDA graphs require **stable tensor pointers** between capture and replay, so
every script pre-allocates its output buffers and reuses the same pointers.

---

## 3. Skinny GEMM — QKV and O projections (`01`)

### The shape problem
At `T = 2` the projection GEMMs are `M = 2`, `N ∈ {2560, 2048}`, `K = 2048`. The
general (square-tile) GEMM path fetches large `B` (weight) tiles whose cost cannot
be amortized across only 2 output rows, so it leaves most of HBM bandwidth on the
table. The whole game on a `M = 2` GEMM is **stream the weights once, as cheaply
as possible** — it is bandwidth-bound, not FLOP-bound.

### The kernel: three levers, all in `_common.build_gemm_kernel`
The winning `UniversalGemmSpec` (verified in the source) is a `tile_m = 16,
tile_n = 16` tile with `warp_tile = (16, 16, 32)`, `pipeline="mem"`,
`epilogue="cshuffle"`, and three structural choices:

1. **Direct-to-LDS for A (`direct_to_lds=True`).** The activation tile is tiny
   (`2 × K` bf16 ≤ 8 KB). Streaming it HBM → LDS in one instruction, bypassing
   the L2 round-trip and the VGPR staging hop, removes a memory hop on the one
   operand small enough to make that free.
2. **Deep `tile_k` (512 or 1024).** A wide K-accumulation tile unrolls the K-loop
   deeply enough to hide HBM fetch latency and amortizes the fixed per-tile
   overhead (LDS alloc, barrier, store). `01` uses `tile_k = 512` for the
   `N = 2560` QKV shape and `tile_k = 1024` for the square `N = K = 2048` O-proj.
3. **Chiplet swizzle (`chiplet_swizzle=True`).** MI355X has 8 XCDs, each with its
   own L2 slice. The default CTA linearization piles work onto one XCD's L2; the
   `chiplet_wgm` / `chiplet_chunk_size` mapping spreads CTAs across all 8 XCDs so
   each L2 sees a different, non-overlapping `B`-tile shard. `01` uses
   `chiplet_wgm = 4` for QKV and `chiplet_wgm = 8` for O-proj.

This is the family the README and the script docstrings identify as the winner
for `M = 2` decode GEMMs. It is a **genuine compute win** (the GEMM is the same
length either way; the kernel is faster), and `01` checks the BF16 result against
an f32 reference (`max|err|` gate).

### Why bf16 *correctness* is load-bearing here
gfx950 now ships a full bf16 MFMA family — `(16,16,16)`, `(16,16,32)`,
`(32,32,8)`, and `(32,32,16)` — wired into the atom catalog
(`helpers/atoms.py:MFMA_BF16_ATOMS`), the layout map
(`core/arch/target.py`), and the LLVM lowerer (`core/lower_llvm.py` emits
`llvm.amdgcn.mfma.f32.32x32x16.bf16`). The inner-GEMM atom selector
(`helpers/mfma_gemm_inner.py`) returns the matching `bf16_*` atom for the
requested `(m, n)` and dtype, so a correctly-tagged bf16 GEMM lowers to a real
bf16 MFMA on every tile shape.

The silent-correctness trap is therefore **not** a missing atom — it is a
**dtype-tag mismatch**. `BatchedGemmSpec.to_universal_spec()` previously left
`DataSpec` at its `dtype_a=dtype_b=dtype_c="fp16"` default; reading bf16 bits
through an *f16* MFMA produces *finite* garbage (~1e36, not NaN — so a naive
`isfinite` check passes). `06` documents this as its "FP16/BF16 dtype mismatch"
fix, and every GEMM spec class now threads an explicit `dtype` field through
`DataSpec` (`build_gemm_kernel` pins `dtype_a/b/c="bf16"` directly). The MoE
spec's bf16 default tile pins `warp_tile=(16,16,32)` (`_default_bf16_gemm_tile`)
not because the 32×32 bf16 atom is missing but to satisfy a `load_vec ≥ 2`
constraint on that path. The lesson recurs across the example: a wrong *dtype
tag* — feeding bf16 bits through an f16 atom — is a *silent* correctness bug on
gfx950, not a crash.

---

## 4. Paged decode attention — 3D split-KV (`03`)

### What it computes
Decode attention is the `S_q = 1` case of flash attention: each of the `T`
sequences has exactly **one** query token attending to its whole KV cache (here
`kv_len ∈ {512 … 4096}`), with the KV cache stored in **paged** `block_size = 16`
blocks indexed through a `block_table`. With GQA, `nhead_k = 4` KV heads are
shared across `nhead_q = 32` query heads (group ratio 8).

### The 3D split-KV strategy
The classic flash-attention parallelization (one CTA per `(q-tile, head, batch)`)
under-fills the GPU at decode: with `S_q = 1` there is no q-tile axis to spread
over, so only `nhead × batch` CTAs launch — far fewer than the GPU's CU count.
The fix is to **split the KV-sequence axis** as a third grid dimension: each CTA
processes a *chunk* of `kv_len` keys/values, computes a *partial* online-softmax
accumulator plus its log-sum-exp, and writes them to a scratch buffer; a second
reduction pass merges the partials across chunks (the standard
max-rescale-and-combine that makes online softmax associative). This exposes
parallelism along the sequence dimension that decode otherwise lacks.

### The one lever: `num_sms`
`num_sms` is how many CTAs participate in the split — the chunk count. It is a
direct parallelism-vs-overhead tradeoff: too few CTAs strand compute on idle CUs;
too many make the merge pass dominate. `03` sweeps
`num_sms ∈ {30, 60, 80, 120, 152, 304}` and keeps the fastest (`07` sweeps a
smaller `{30, 60, 80, 120}` for speed).

### Why this layer is *parity*, not a win
`head_dim = 64` is half the typical 128 the 3D kernel's tiles are sized for. The
MFMA tiles assume a 128-wide dot product; at 64-wide the matrix-unit utilization
is lower and the kernel goes bandwidth-bound sooner. Both the `rocke` 3D kernel
and the AITER Triton `unified_attention` land within ~95 % of each other across
all tested `kv_len`. The README lists this as an **open gap**: near-parity is
accepted because decode dominates the A3B workload and a head-dim-64-tuned kernel
is future work.

**A correctness trap baked into the harness:** for decode you must build
`cu_seqlens_q = [0, 1, …, T]` (one query token per sequence). A plain `[0, T]`
declares a *single* length-`T` sequence, which mismatches the `T`-row
`block_table` / `seqused_k` and reads `cu_q[T]` out of bounds inside the kernel —
an intermittent GPU memory fault. Both `03` and `07` construct `cu_q` correctly
and comment the trap.

---

## 5. Router top-K — fused softmax + selection (`04`)

### What it computes
The MoE router scores each token against all `E = 128` experts, applies a softmax,
and selects the `K = 8` highest-weighted experts (returning both the weights and
the indices). The `rocke` `topk_softmax` kernel does the whole thing in **one
pass per token** (one CTA per row of `T`), with no separate sort.

### Why this is the purest dispatch-bound case
The GPU kernel runs in ~2 µs. AITER's `moe_fused_gate` reading is ~13 µs — the
extra ~11 µs is the host call chain (pybind11 arg unpacking, torch tensor
dispatch, `hipModuleLaunchKernel`). So this layer is the textbook §2
dispatch-bound case: the kernel is tiny, the dispatch dwarfs it, and the lever is
CUDA-graph capture. Captured, the DSL reads ~0.45 µs (the replay floor) — but the
~2 µs of GPU work is *unchanged*; only the host chain is removed. `04` reports
both numbers and is explicit that the headline ratio is a dispatch-path win.

`04` verifies correctness by checking the DSL top-`K` index *set* matches
`torch.topk`. (Order can differ on ties; the *set* is the contract.)

---

## 6. MoE token sorting — and why decode skips it (`05`)

### What sorting is for
A batched per-expert GEMM wants all tokens routed to expert `e` laid out
contiguously, so each expert's GEMM reads one slab. Producing that layout from the
router's scattered `(token, expert)` pairs is the **sort**: histogram the experts,
exclusive-scan the counts into offsets, then scatter each pair into its sorted
slot. The `rocke` `MoeSortingLauncher` implements exactly this as a **3-kernel
chain** (`moe_histogram` → `moe_scan` → `moe_scatter`), each launch a separate
HBM round-trip.

### Why the DSL chain "loses" here on purpose
AITER's `moe_sorting_opus_fwd` fuses histogram + scan + scatter into **one**
on-chip kernel (~6 µs). The DSL 3-kernel chain (~28 µs) cannot match a 1-kernel
fused sort: three launches, three HBM round-trips. The README records this as a
~0.22× ratio and labels it a **deliberate design choice, documenting the fallback
cost** — because in decode the sort is **not on the hot path at all** (see §7).

**Hard A3B-specific constraint, verified in the source:** `MoeSortingSpec`'s
support gate rejects `experts > block_size` (the LDS scan holds one expert count
per lane). `FusedMoeForwardSpec.sort_block_size` defaults to **64** and is passed
straight through as the sort spec's `block_size`; A3B has `E = 128 > 64`, so `05`
(and `06`, and `07`) must set `sort_block_size = 128` or the sort spec fails its
validity check at build time. The histogram and scatter phases **atomic-add** into their
`Hist` / `Counter` workspaces, so `05` zeroes both before every chain — the
launcher does not own that lifetime.

---

## 7. Fused MoE forward — the dominant layer (`06`)

This is ~49 % of the decode step (per `07`'s Amdahl roll-up) and where the
structural decode optimizations live.

### What it computes (the spec)
```
Hidden_{t,e} = silu( x · W_gate_e^T ) ⊙ ( x · W_up_e^T )      # SwiGLU, [I]
FFN_{t,e}    = Hidden_{t,e} · W_down_e^T                       # [H]
y_t          = Σ_{e ∈ topk(t)}  w_{t,e} · FFN_{t,e}           # [H]
```
`silu(x) = x · σ(x)`. The `rocke` pipeline is: `topk_softmax` → (sort, **skipped**)
→ gather X into expert-grouped buffers → an **interleaved gate+up** GEMM with fused
SiLU → down GEMM → weighted top-K reduce into `Y`.

### The decode-specific levers (the real wins)
At `T = 2` the routing is *sparse*: only `T·K = 16` (token, expert) pairs are
active out of `T·E = 256` slots. The pipeline exploits that with two structural
choices, both verified in `06`'s `FusedMoeForwardSpec` / runtime flags:

1. **Static-offset mode (skip the sort).** Setting `fwd._use_static_offsets =
   True` with `fwd._static_slot_size = 1` pre-computes fixed offsets
   `[0, slot_size, 2·slot_size, …]` so the histogram+scan+scatter sort (the ~28 µs
   of §6) is **never launched**. `slot_size = 1` keeps the padding waste minimal
   for sparse routing. This is why §6's sort cost is a documented fallback, not a
   decode cost.
2. **Active-tile skip (`active_tile_skip_gemms=True`).** With 16 active pairs out
   of 128 expert slots, ~87.5 % of the per-expert GEMM tiles are empty. The kernel
   uses a `SortedTokenIds == -1` sentinel and a single `scf.if` at CTA entry: a CTA
   whose `(expert, m-tile)` slot has no valid token id short-circuits — skipping all
   MFMAs, LDS reads, and HBM stores — instead of running the dense GEMM. (The CTAs
   are still launched; the empty ones just exit immediately.) So it does work
   proportional to the *real* routing, not the worst case.

Together these two are "do only the work the sparse routing needs," the same
structural idea (size the launched work to real tokens) that recurs throughout the
example. On top of them, **CUDA-graph capture** collapses the multi-kernel
pipeline (topk → gather → 2× GEMM → reduce, ~15 µs of cumulative host dispatch
across 5+ launches) into one ~0.5 µs replay packet.

### Why graph capture is *required*, not optional, for the MoE win
The README is precise about this and the code bears it out: **without graph
capture the DSL pipeline is *slower* than AITER**, because AITER's 2-stage CK
pipeline is a single fused call with low dispatch cost, whereas the DSL pipeline
pays host dispatch on every one of its 5+ launches. The underlying GPU compute is
what produces the win; the graph is what makes it *visible* by removing the
multi-launch dispatch floor. `06` therefore captures the whole pipeline into one
HIP graph and times `replay`.

The bf16 correctness fix from §3 (the threaded explicit `dtype`, so the GEMM
lowers through a real bf16 MFMA atom instead of an f16 one) is what makes this
kernel *correct* on gfx950 bf16 in the first place; `06`
documents them as the first two of its six steps.

---

## 8. The whole step — Amdahl, and why kernel speedup ≠ serving speedup

### `07`: rolling the layers up
`07` runs every layer once, weights RMSNorm ×3, and prints an Amdahl table:
end-to-end speedup is `1 / (Σ_i f_i / S_i)` where `f_i` is layer `i`'s fraction of
the baseline step and `S_i` its per-layer speedup. The table is computed **live
from the measured rows** (not hard-coded), and the README header reports the
result as **≈1.28× end-to-end (~46 µs saved per decode step)** at the GPU-kernel
level. The Amdahl insight the README draws out: a 29× win on the router top-K
contributes only ~6 µs of the ~46 µs saved because top-K is ~6 % of the step —
the dominant MoE forward (~49 % of the step) and the near-parity attention layer
(~25 %, the second largest) set the ceiling, while the GEMM layers (QKV + O ≈
11 % combined) add a smaller, genuine compute win.

> The per-script docstrings and the README's per-layer tables carry slightly
> different absolute µs (re-measured at different times; the box thermally
> drifts). Treat the **end-to-end ≈1.28×** and
> the **per-layer *ratios*** as the durable claims, and any single absolute µs as
> a point-in-time reading. `07` always recomputes the live number on the machine
> it runs on.

### Why the GPU win does not appear at the serving layer
`bench_atom.py` / `bench_atom_sweep.py` measure the *whole ATOM decode step*
end-to-end (real engine, KV cache, sampler), one backend config per subprocess,
interleaved round-robin so all configs see the same thermal state. The measured
result: **all three configs (`baseline`, `dsl_gemm`, `dsl_all`) are statistically
indistinguishable** — the ~120–140 µs run-to-run stdev from GPU power-state
variation is *larger* than the ~50–100 µs of GPU time the DSL kernels save.

The reason is Amdahl again, one level up: the ATOM engine loop (Python IPC between
the main process and the ModelRunner subprocess, scheduler, sampler) contributes
~4200 µs per decode step on top of ~200–300 µs of actual GPU kernel time. Saving
~50–100 µs of GPU time is <1 % of a ~4500 µs step and below the noise floor. The
true GPU-level 1.28× is real (and confirmed by `07`); exposing it at the serving
layer requires cutting the engine-loop overhead from ~4200 µs to under ~500 µs —
out of scope for these kernels.

---

## 9. The recurring lesson

Every layer in this example resolves to one of two cost models, and the right
optimization depends on which:

- **GPU-compute-bound layers (GEMMs, attention, the MoE GEMMs)** are won with
  *kernel* levers — the skinny-GEMM tile family (DTLA + deep `tile_k` + chiplet
  swizzle), the 3D split-KV parallelization, the sparse active-tile skip. These
  are genuine compute wins and survive at the GPU level.
- **Dispatch-bound layers (RMSNorm, router top-K, and the multi-launch MoE
  pipeline)** are won with *host* levers — CUDA-graph capture, static-offset sort
  elimination. The GPU does the same work; what is removed is the cost of *asking*
  it to. These wins are real and matter (dispatch is paid every step) but must be
  named honestly: a 29× top-K "speedup" is a dispatch-path win, not a faster
  kernel.

And one structural theme cuts across both: **size the launched work to the real
tokens.** At `T = 2` the static-offset sort skip, the active-tile skip, and the
skinny tile all express the same principle — do work proportional to 2 tokens and
16 routed pairs, not to the model's worst case. The single largest end-to-end
risk is the opposite mistake (a wrong *dtype tag* — feeding bf16 bits through an
f16 MFMA atom) producing *silent* finite garbage; the correctness gates in
`01`/`02`/`04`/`06` exist to catch it.
