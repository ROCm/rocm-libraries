# Attention Backward (dQ, dK, dV): design of the rocKE SDPA backward kernels

This document explains how the rocKE SDPA backward kernels work and why they are
built the way they are: the math, the kernel decomposition, the per-step tile
program, the memory and register plans, masking and sequence handling, the
per-arch differences, the tuning model, the numerics, and the invariants the
tests hold. The host contract (statistics format, kernarg ABI, workspace,
predicate, declaration) is at the end (section 12).

Source of truth is the code under `library/` (paths below are relative to the
rocke root). Anything described as "not built" exists only as a registered
knob value, a refused value, or a recorded design; no shipped kernel emits it.

## 0. Overview

### 0.1 What exists

* **One family is built: the layout-general backward.** It serves fp16 / bf16,
  head sizes 32 / 64 / 128, any non-negative B / H / S element strides (D
  stride 1), fixed, padded and THD (ragged) lengths, MHA / GQA / MQA and
  `h_k != h_v`, no mask / causal top-left / causal bottom-right / two-sided
  sliding windows, and `s_q = 1`, on gfx942, gfx950, gfx1151 and gfx1201.
* Three kernels per backward on one stream: **prep** (statistics and workspace
  initialisation), **main** (KV-parallel, all five GEMMs), **convert** (fp32
  workspace to the output dtype), plus two more converts when dK / dV are
  accumulated with atomics.
* dQ is accumulated with fp32 atomics into a workspace (the "atomic sink").
  A separate Q-parallel dQ kernel ("split dQ") is designed and its ABI is
  reserved, but it is **not built**.
* The **dense** family (shape-specialised, per-arch) and the **unified**
  family (THD queries over a paged KV cache) are designed and **not built**.
  The plan has a dense routing hook whose pack is empty, and the shared aux
  module already contains the two paged-workspace roles the unified family
  would use (section 8).
* Every tile, residency, path, schedule and codegen decision is a defaulted
  spec field or a plan value, registered as a tuning axis
  (`dispatch/attention_bwd/tuning_common.py`). Values whose emission is not
  built are refused before any IR is emitted (`attention_bwd._not_built`).

### 0.2 File map

| File | Content |
|---|---|
| `library/kernels/common/attention_bwd.py` | `AttnBwdSpec`, per-arch default geometry, legality (`validate_attn_bwd_spec`), register model, LDS footprint, kernel names, ABI lists (`attn_bwd_params`), `build_attn_bwd_main` (CTA decode, addressing, q loops, dK / dV epilogue) |
| `library/kernels/common/_attention_bwd_body.py` | `BwdTileGeometry`, `BwdTileStep` (K / V staging, G0..G4, softmax gradient, P / dS hand-off, dQ sink call, epilogue readers), `StatsView`, `BwdLds`, the Q / dO staging copies |
| `library/kernels/common/_attention_bwd_gemm.py` | `BlockGemm` (multi-wave block GEMM over a warp grid), `LdsOperand`, the C-to-A relabel entry points, the inline-asm MFMA clusters (`mma_agpr_asm`, `mma_agpr_asm_steps`) |
| `library/kernels/common/_attention_bwd_frag.py` | lane-map evaluation on an integer builder, atom facts, the checked C-to-A relabel plan, K-permuted B coordinates, the LDS fragment loaders including `ds_read_tr16_b64` |
| `library/kernels/common/_attention_bwd_lds.py` | the LDS phase planner (`plan_bwd_lds`), XT and DMA-slab layouts (layouts only; their users are not built), conflict prediction |
| `library/kernels/common/_attention_bwd_caps.py` | per-arch capability checks, atom selection, the asm MFMA mnemonics and hazard wait states |
| `library/kernels/common/_attention_bwd_band.py` | inverse band (`q_tile_range`), tile classifier, interior q range, mask policies, dK / dV store rule |
| `library/kernels/common/_attention_bwd_addr.py` | sequence decode (batched, THD), strided tensor and workspace address generators, packed rows, paged cache addressing |
| `library/kernels/common/_attention_bwd_sinks.py` | dQ sinks (atomic, packed atomic; null, register and split-slice sinks recorded) |
| `library/kernels/common/_attention_bwd_worklist.py` | CTA decode, XCD remap, load-balance orders, head ranges |
| `library/kernels/common/_attention_bwd_sched.py` | stage table (instruction count model and sched-group emission hooks) |
| `library/kernels/common/attention_bwd_aux.py` | `AttnBwdAuxSpec`, prep, convert and the paged roles |
| `library/kernels/common/attention_bwd_plan.py` | `AttnBwdRequest`, predicate, workspace, plan, routing, performance classes, declaration, `AttnBwdPolicy` |
| `library/kernels/common/attention_bwd_run.py` | launch glue: compile, pack kernargs, launch on one stream |
| `library/dispatch/attention_bwd/tuning_common.py` | the knob registry (`KnobAxis`) and `backward_spec_legality` |
| `library/benchmarks/common/attention_bwd_sweep.py` | the funnel sweep driver; `attention_bwd_compile.py` (compile cache, crash isolation), `attention_bwd_bench.py` (paired timing harness), `attention_bwd_census.json` (shape census, no measurements) |

All backward modules are Python-only family glue (no C++ twin); compiled
artifacts are built with the Python lowering backend.

## 1. Math

### 1.1 Forward recap

For one (batch, query head) with `Q [S_q x D]`, `K, V [S_kv x D]` and scale
`s` (default `1 / sqrt(D)`):

```
S   = s * Q K^T                       (masked cells: -inf)
P   = softmax_rows(S)
O   = P V
LSE = log(sum_k exp(S[q, k]))         (natural log, fp32, one per row)
```

The backward never recomputes the row statistics: it takes `O` and `LSE` from
the forward (section 12.1). Any forward that stores `O` and a natural-log fp32
LSE is consumable.

### 1.2 Backward equations

Given `dO`, with `S_raw = Q K^T` (unscaled) recomputed tile by tile:

```
P   = exp(s * S_raw - LSE)            (recomputed; masked and dead cells 0)
dP  = dO V^T
D   = rowsum(dO * O)                  (= rowsum(P * dP); "Dsum")
dS  = P * (dP - D)
dV  = P^T dO
dK  = s * dS^T Q
dQ  = s * dS K
```

`D` is the per-row correction of the softmax Jacobian; computing it from `dO`
and `O` (one dot product per row) is cheaper than `rowsum(P * dP)` over all
keys, and it is computed once per row by prep instead of once per key tile.

The kernel names the five matrix products of one tile step:

| GEMM | Product | Shape (M x N over K) | Accumulator |
|---|---|---|---|
| G0 | `S = Q K^T` | `kM0 x kN0` over `D` | fresh per step |
| G2 | `dP = dO V^T` | `kM0 x kN0` over `D` | fresh per step |
| G1 | `dV += P^T dO` | `kN0 x D` over `kM0` | carried across steps |
| G3 | `dK += dS^T Q` | `kN0 x D` over `kM0` | carried across steps |
| G4 | `dQ_part = dS K` | `kM0 x D` over `kN0` | fresh per step, sent to the dQ sink |

### 1.3 The log2 domain

The kernels use `exp2`. Prep converts the statistic once:
`lse2 = LSE * log2(e)`; the plan passes `scale_log2 = s * log2(e)`. Then

```
P = exp2(S_raw * scale_log2 - lse2) = exp(s * S_raw - LSE)
```

so the exponent costs one multiply and one subtract per cell
(`BwdTileStep._softmax_grad`), with no per-cell `log2(e)` multiply. A log2 LSE
fed as input is wrong data (it gives wrong gradients); the negative-control
tests feed one on purpose.

### 1.4 Where the scale is applied

The scale `s` must reach dK and dQ but not dV. Three placements are built; the
plan picks one (`scale_placement`, default `fold_ds`) and sets four fp32
multipliers (`attention_bwd_plan._scale_mults`):

| Placement | `ds_mult` (main, on dS) | `dk_mult` (main epilogue) | `dq_mult` (main, before each dQ atomic) | convert `mult` (dQ / dK) |
|---|---|---|---|---|
| `fold_ds` (default) | `s` | 1 | 1 | 1 / 1 |
| `convert` | 1 | `s` in direct mode, 1 in atomic mode | 1 | `s` / `s` in atomic mode |
| `dq_before_atomic` | 1 | as `convert` | `s` | 1 / `s` in atomic mode |

`fold_ds` scales each dS element once, before it is rounded to the I/O dtype,
so dK and dQ come out scaled with no extra multiply per output element. The
other placements keep dS unscaled and multiply later; they round at a
different point and are an order change (section 10).

The main kernel is compiled for its placement (`AttnBwdSpec.static_scale`,
plan-owned): the multipliers that are exactly 1.0 under it are not emitted
(`attention_bwd.static_unit_mults`). `x * 1.0 == x` for every fp32 `x`
(denormals are not flushed), so this is bitwise; `static_scale = "runtime"`
applies all three and serves every placement from one binary. The kernargs
stay in the ABI either way.

### 1.5 Precision and rounding points

* Every accumulation is fp32: S, dP, dK, dV and the dQ partial are MMA
  accumulators; Dsum is an fp32 dot product in prep; dQ (and, in atomic mode,
  dK / dV) are summed with fp32 atomics.
* P and dS are rounded to the I/O dtype (fp16 / bf16) only because they are
  MMA operands: P and dS as the A operands of G1 / G3, dS as the A operand of
  G4 (read back from LDS). Their fp32 values are never stored.
* Before the rounding, each `ds_mult * dS` value passes through an empty
  inline asm (`_rounding_point`). That materialises the fp32 value, so the
  backend cannot fuse the multiply and the conversion into one mixed-precision
  operation in one step variant and not in another; edge and interior steps
  (and every other variant of the step) therefore round dS identically.
* Each output is rounded once: dK / dV in the epilogue (after `dk_mult`), dQ
  (and atomic-mode dK / dV) in the convert kernel (after `mult`).
* `exp2` is the IR's guarded `exp2` (`llvm.exp2.f32`). The single-instruction
  `exp2_fast` exists in the IR and is used by the forward, but the backward
  does not use it (section 13).

### 1.6 Dead rows and "select, never multiply"

A row is **dead** when its LSE is not finite (`!(lse > -inf && lse < +inf)`,
so NaN is dead too) or when it lies outside the sequence. The forward marks
fully masked rows with `-inf`. Prep gates both statistics on one predicate
`row_live = q < len_q && lse > -inf && lse < +inf`:
`lse2 = row_live ? lse * log2(e) : +inf`, `Dsum = row_live ? rowsum(dO * O) : 0`.

The main kernel then forms

```
P  = select(keep,                    exp2(S * scale_log2 - lse2), 0)
dS = select(keep && lse2 < +inf,     P * (dP - Dsum),             0)
```

Selects, never multiplies by a mask: a dead row (`lse2 = +inf`) or a masked
cell contributes an exact 0 whatever `Q`, `dO` or `O` hold there (a NaN in
`O` of a dead row cannot leak through `Dsum`). On a dead row `P` is `+0`
anyway (`exp2(x - inf)`), so G1 adds nothing; the dS select is what blocks
`0 * NaN`. The dead-row term costs one compare per row, folded into the
per-row key limits on edge steps (section 6.3). A test-only build without the
term (`_emit_main(..., sentinel=False)`) is a negative control.

## 2. Kernel decomposition

### 2.1 Launch sequence

```
            per (query head, row)               per (kv tile, head unit, batch)        per (head, row)
 O, dO,  +--------------------+   WS_LSE2   +------------------------------+  WS_DQ   +-----------------+
 LSE --> |  prep              |-- WS_DSUM ->|  main (KV-parallel)          |--------->|  convert(dQ)    |--> dQ
         |  lse2, Dsum,       |   WS_DQ=0   |  G0..G4 per q step           |          +-----------------+
         |  zero WS_DQ        |  (WS_DK=0,  |  dK / dV: on chip (direct)   |  WS_DK   +-----------------+
         |  (zero WS_DK/DV)   |   WS_DV=0)  |   or fp32 atomics (atomic)   |--------->|  convert(dK)    |--> dK  (atomic mode)
         +--------------------+             |  dQ: fp32 atomics            |  WS_DV   +-----------------+
                                            +------------------------------+--------->|  convert(dV)    |--> dV  (atomic mode)
                                                     | direct mode                    +-----------------+
                                                     +--> dK, dV stored once
```

All launches are on one stream in plan order (`attention_bwd_run.run_attn_bwd`).
The caller owns every buffer; nothing is allocated and no memset is issued,
because prep initialises every workspace byte a later kernel reads.

### 2.2 Prep (`attention_bwd_aux.build_attn_bwd_prep`)

Grid `(ceil(max(S_q, S_kv) / R), h_q [+ h_k + h_v], B)`, `R` rows per CTA
(default `max(wave, 64) / (D / 8)`, so one row is `D / 8` lanes of 8
elements). Grid y below `h_q` is the q role; above it, in atomic dK / dV
mode, the kv-zero role.

* q role: each lane loads 8 elements of `O` and `dO`, accumulates their
  products in fp32, and the `D / 8` lanes of a row reduce with an XOR-shuffle
  butterfly; the row's first lane stores `lse2` and `Dsum` (gated as in 1.6).
  User tensors are read only in CTAs with a live row; in batched mode the
  other CTAs write the neutral values (`+inf`, `0`) to every row below
  `S_q_max`. It also zeroes the dQ workspace rows (`zero_dq`).
* kv-zero role (`zero_kv`): zeroes the `WS_DK` / `WS_DV` rows.
* Batched mode initialises every row below `S_max`; THD initialises the rows
  inside each segment only (no other row is ever read).

### 2.3 Main: why KV-parallel

One CTA owns one key tile (`kN0` keys) of one head unit and batch, keeps K and
V of that tile on chip, and walks every q tile that attends to it. This is
the FlashAttention-2 backward structure:

* dK and dV of the key tile accumulate in registers (or LDS) across all q
  steps and leave the CTA once, so in direct mode they are deterministic and
  need no workspace.
* K and V are read from global memory once per CTA; Q, dO and the row
  statistics are streamed per q step.
* The price is dQ: every key tile contributes a partial `dS K` to the same dQ
  rows, so dQ needs a cross-CTA reduction. The built design uses fp32 atomics
  into a zeroed workspace (2.4).

In direct mode a CTA owns a kv head and loops over all `G = h_q / h_k` query
heads that share it, summing their dK / dV contributions on chip (no
expand-then-reduce of dK / dV per query head).

### 2.4 dQ strategies

| Strategy | Status | Mechanism |
|---|---|---|
| Atomic sink (`dq_mode = "atomic"`) | built, the only mode | `_attention_bwd_sinks.AtomicWorkspaceSink`: each q step's fp32 partial is added with `global_atomic_add_f32` (agent scope, native no-return) into `WS_DQ`; the convert kernel scales and rounds once |
| Packed atomic sink | built | `PackedAtomicWorkspaceSink`: the same for head-packed steps (per-row head rebase, dead packed rows add an exact 0) |
| Split dQ (`dq_mode = "split"`) | **not built** (`SPLIT_DQ_AVAILABLE = False`; the spec raises) | a separate Q-parallel kernel recomputes S and dP for its q tile, forms dS and accumulates `dQ = dS K` in registers over the band, then stores dQ once in the output dtype (no atomics, deterministic, no dQ workspace, no zeroing, no dQ convert). The main kernel would take `NullSink` (no G4, no dS LDS round trip, no atomics). Its kernarg list is already in the v3 ABI (`attn_bwd_params("dq")`), the planner already drops the dS buffer for it, and the stage-table counts handle it |
| Register / split-slice sinks | recorded designs (`RegisterAccumulateSink` exists for Q-parallel use; `SplitSliceSink` raises) | dQ in registers across kv tiles; per-split slices with a fixed-order reduce |

The atomic sink is branch-free: a slot row is clamped to `max(len_q - 1, 0)`
(or not, on interior steps), a row at or past `ws_rows_q` adds an exact 0 at a
clamped address, and the workspace pointer is rebased in i64 once per head
(`begin_head`, called where it dominates every step variant) so the in-slab
index stays a small i32. In the token-major workspace layout a row is
`H * D` elements, so the sink rebases per step to the step's first target row.

G4 issue granularity (`dq_issue`): `"block"` computes the whole `kM0 x D`
partial, then issues its atomics; `"fragment"` computes one 16 x 16 fragment
and issues its atomics before the next (the dS row is re-read from LDS per
fragment), so one fragment is live and its atomics overlap the next
fragment's MMAs; `"fragment_regs"` reads dS into registers once, then works
per fragment. Each fragment keeps the block's K order, so the partials are
bit-identical. LDS-accumulator geometries always work per fragment.

### 2.5 dK / dV modes and head mapping

* **Direct** (`h_k == h_v`, `g_split = 1`): unit = kv head `u`; query heads
  `[u * G, u * G + G)`; dK / dV stored once by the CTA (2.7 epilogue).
* **Atomic** (`h_k != h_v`, or a runtime `g_split > 1`): unit = a group of
  `G = gcd(h_q / h_k, h_q / h_v)` query heads, which share one K head and one V
  head because `G` divides both `gk = h_q / h_k` and `gv = h_q / h_v`. Unit `u`
  of split `p` owns the `G / g_split` heads from `u * G + p * G / g_split`,
  with `hk = u * G / gk`, `hv = u * G / gv` (`_attention_bwd_worklist.head_range`).
  The CTA adds its fp32 dK / dV into `WS_DK` / `WS_DV` with atomics; two
  converts follow. `g_split` spreads a large group over more CTAs on
  under-filled grids at the price of dK / dV atomics.

Example: `h_q = 8, h_k = 2, h_v = 4` gives `gk = 4, gv = 2, G = 2`, four units;
unit 3 owns query heads 6 and 7, K head 1, V head 3.

### 2.6 Convert (`attention_bwd_aux.build_attn_bwd_convert`)

`DST = cast(mult * WS)` at arbitrary output strides. Grid
`(ceil(S_max / R), heads, B)`; `convert_vec` elements per lane (8, 4 or 16).
Batched rows `[len, S_max)` are written as exact zeros; THD rows outside the
segment are not touched.

### 2.7 Grid and CTA decode

Main grid `(gx, units * g_split, B)` with `gx = n_kv_tiles = ceil(S_kv_max / kN0)`,
`units = h_k` (direct) or `h_q / G` (atomic); block `W * wave_size` threads.

```
 (x, y, z) = block id
   |
   |  [xcd_remap built and xcd_n > 0]   per batch plane z:
   |     P  = x + gx * y                       plane id in [0, gx * gy)
   |     P' = chiplet_transform_chunked(P, num_wgs = gx * gy, n = xcd_n, chunk = xcd_chunk)
   |     x  = P' % gx,  y = P' / gx
   v
 t     = x                                     kv-tile index (fastest)
 unit  = y / g_split,  split = y % g_split     (direct mode: unit = y, split = 0)
 batch = z
   |
   v  load-balance order on t, inside one (unit, batch):
 kt = t                          lb_order natural (0)
 kt = n_kv_tiles - 1 - t         lb_order reverse (1)
 kt = t/2 or n_kv_tiles-1-t/2    lb_order interleave (3; even / odd t), decode built with lb_interleave
      (mirror pairing, two kv tiles per CTA: not built, the plan refuses lb_order = mirror)
```

* **Why kv tile fastest:** consecutive CTAs of one (unit, batch) share Q, dO,
  the statistics and the dQ workspace rows of that head group, so they reuse
  them through L2.
* **XCD remap** (`XcdRemap`, `xcd_remap_id`): consecutive block ids are
  dispatched round-robin over the chiplets; the chunked transform gives each
  chiplet `xcd_chunk` consecutive logical ids, so with a chunk that is a
  multiple of `gx` all kv tiles of one (unit, batch) land on one chiplet and
  its L2. The transform is applied per batch plane because the main ABI
  carries no batch count; it is a bijection of each plane and equals the
  whole-grid transform when `xcd_n * xcd_chunk` divides the plane. The
  transform is monotone inside each chiplet, so a heaviest-first order stays
  heaviest-first per chiplet. A kernel built without `xcd_remap` emits no
  remap IR; the plan sets the field when the request's `num_xcds` hint and a
  non-zero `xcd_chunk` give `xcd_n > 0`.
* **Load-balance orders:** under a causal or window mask the number of q tiles
  per key tile varies with `kt`. `reverse` dispatches the heaviest tiles first
  (bottom-right causal); `interleave` alternates heavy and light tiles. All
  orders visit each kv tile of a (unit, batch) exactly once. Mirror pairing
  (one CTA processes `t` and `n_kv_tiles - 1 - t`) is decoded by
  `decode_cta(pair=...)` but the main kernel does not build the two-tile CTA
  (a CTA-uniform loop over two tiles keeps the staging and epilogue values
  live across the q loops and spills), so the plan refuses it.

## 3. The tile step

### 3.1 Geometry, warp grids and atoms

`BwdTileGeometry.from_spec` turns a resolved spec into three `BlockGemm`s
(`_attention_bwd_gemm.BlockGemm`): a block GEMM `C[M x N] += A[M x K] B[K x N]`
computed by the `W` waves of the CTA on a warp grid `(rows, cols)`; wave `w`
owns the C sub-block at row `(w / cols) * M / rows`, column
`(w % cols) * N / cols`, as a static `frags_m x frags_n` array of 16 x 16
atom tiles, and K advances in `k_steps` atom-K steps. Every fragment index is
a Python int (no dynamic register indexing, so no scratch).

| GEMM | M x N, K | Default warp grid | Waves split |
|---|---|---|---|
| G0 / G2 | `kM0 x kN0`, `D` | `(1, W)` (`warp_grid_g02`) | the keys |
| G1 / G3 | `kN0 x D`, `kM0` | `(W, 1)` (`warp_grid_g13`) | the keys (as rows) |
| G4 | `kM0 x D`, `kN0` | `(1, W)` or `(2, W / 2)` (`warp_grid_g4`) | D (and q rows) |

With the default grids, wave `w` owns the same keys in G0 / G2 (its columns)
as in G1 / G3 (its rows) and every q row of the step. That is the condition
for the register relabel of 3.3. `(2, W / 2)` grids for G0 / G2 or G1 / G3
are legal only with `pt_route = "lds"`.

MMA atoms (`_attention_bwd_caps.bwd_atoms`, 16 x 16 only):

| Target | Atoms | Notes |
|---|---|---|
| gfx942 | `16x16x16` (fp16; bf16 through the `_1k` form) | a K = 32 step is two 16x16x16 instructions |
| gfx950 | `16x16x16`, native `16x16x32` | defaults: G0 / G2 / G4 on `16x16x32`, G1 / G3 on `16x16x16` |
| gfx1151, gfx1201 | `wmma16x16x16` (wave32) | |

32 x 32 atoms are excluded: `kM0 = 16` cannot fill them and the relabel is
proven only for 16 x 16 atoms. `atom_g13 = "16x16x32"` needs `kM0 >= 32`.

MFMA instructions per wave per q step (`_attention_bwd_sched.StageTable.mfma`):

```
G0 = G2 = (kM0/16) * (kN0/(16 W)) * (D/k_g02)
G1 = G3 = (kN0/(16 W)) * (D/16) * (kM0/k_g13)
G4      = (kM0/(16 g4r)) * (D/(16 g4c)) * (kN0/k_g4)
dQ atomics per lane = kM0 * D / (W * wave_size)
```

For the gfx942 d128 default (16 x 128, W = 8, G4 grid (1, 8), K = 16 atoms)
that is 8 instructions per GEMM per wave and 4 atomics per lane per step.

Build-time proofs (`BwdTileGeometry.check`): every atom's A / B / C lane maps
are evaluated on an integer builder for every lane and slot
(`_attention_bwd_frag.atom_facts`) and must be K-contiguous and affine; the
relabel is proven per atom pair; the transpose read is proven against the
catalog operand map. An illegal geometry raises `ValueError` before any IR.

### 3.2 Softmax gradient

`BwdTileStep._softmax_grad` runs per accumulator slot of the G0 / G2
fragments. The row of a slot depends only on `(fragment row, slot)`, so each
lane loads the `(lse2, Dsum)` of its own C rows once per step straight from
the workspace (`StatsView`: per-wave redundant scalar loads; rows at or past
`ws_rows` read `+inf` / `0`; never a multi-dword load across the workspace
end). There is no LDS copy of the statistics. `stats_load = "step_top"` issues
these loads at the top of the step, ahead of the staging and G0 / G2, so
their latency overlaps them; `"at_use"` (default) issues them just before
the softmax gradient.

Per slot: one multiply and one subtract feed `exp2`; on edge steps one
compare against the row's key limits gives `keep`; then the `P` select, `dP -
Dsum`, the multiply by P, the dS select, the `ds_mult` multiply, the rounding
point. The two fp32 fragments P and dS are then handed off (3.3).

### 3.3 The P^T / dS^T hand-off: relabel or LDS

G1 needs `P^T` and G3 needs `dS^T` as A operands (rows = keys, K = q rows).
They exist only as G0 / G2 C fragments (rows = q, cols = keys).

* **`pt_route = "relabel"`** (CDNA default). For the 16 x 16 MFMA atoms the C
  layout of one tile coincides with the A layout of the transposed product:
  the four rows a lane holds at one column are exactly the four K values its
  A fragment needs at one row. `_attention_bwd_frag.plan_c_to_a` evaluates
  both maps on every lane and slot and returns a `"direct"` plan; the A
  fragment is then just the C registers rounded to the I/O dtype and repacked
  (`emit_c_to_a`). No data moves between lanes and nothing goes through LDS.
  `check_relabel_grids` additionally proves the warp-grid condition of 3.1.
  For a consumer atom with K = 32 (`16x16x32` on gfx950) two C tiles are
  concatenated and the 32 K values arrive in a permuted order; the plan
  records the permutation and the B operand of G1 / G3 is then loaded with
  the same permutation (`kperm_b_coords`), so the K sum is unchanged. That
  permuted map is not what the transpose read delivers, so relabel +
  `16x16x32` G1 / G3 + `tr_read` is not built; it is built with `lds_plain`.
  A `lane ^ 16` exchange plan (`permlanex16` + `perm_b32`, proven by replaying
  the program on coordinates) exists for wave32 WMMA, but the family refuses
  the relabel route on WMMA targets.
* **`pt_route = "lds"`** (RDNA default, CDNA option). P and dS are rounded and
  written to the `pt` / `ds` LDS buffers in C layout (`_write_c_tiles`), a
  barrier publishes them, and G1 / G3 read them back as K-outer A operands
  through the transpose source. It frees the warp grids (`(2, W / 2)` becomes
  legal) at the cost of an LDS round trip and a barrier.

dS is always written to LDS once per step (`ds` buffer) because G4 needs it as
an A operand with a different warp split (G4 splits D, not keys).

### 3.4 Operand sources

| Operand | Storage | Read as | Source |
|---|---|---|---|
| G0 A = Q, G2 A = dO | `q`, `do` LDS tiles `[q][d]`, per step | K-inner (vector reads along D) | staged from global every step |
| G0 B = K^T, G2 B = V^T | K, V `[key][d]` | K-inner | registers loaded once per CTA (`kv_residency = "reg"`), or the resident LDS copy every step (`"lds"`) |
| G1 B = dO, G3 B = Q | `do`, `q` tiles `[q][d]`, K = q | K-outer | transpose source |
| G4 A = dS | `ds` tile `[q][key]` | K-inner | written by the step |
| G4 B = K | K `[key][d]`, K = key | K-outer | registers loaded once (`kt_source = "reg"`) or LDS every step (`"lds"`), through the transpose source |
| G1 A = P^T, G3 A = dS^T | registers (relabel) or `pt` / `ds` tiles (LDS route) | K-outer from LDS | 3.3 |

"K^T" in the code names the G4 B operand: K stored key-major is K-outer for
G4, so it needs a transposing read. K-outer reads go through the transpose
source (`transpose_source`):

* `lds_plain`: one 16-bit element per read (gfx942 default, and every
  narrow-access tile);
* `tr_read`: `ds_read_tr16_b64` on gfx950 (default there): one 64-bit read
  delivers 4 K-consecutive values per lane; a K = 16 fragment takes one read,
  K = 32 two. Each lane supplies its own address (`tr16_lane_address`), and
  the delivered coordinates (`tr16_model_coords`) are proven equal to the
  catalog operand map at build time; the per-lane model was validated on the
  device by the transpose probe. The 128-bit form is never used.
* `wmma_lds`: the plain element reader on RDNA;
* `xt_lds`: an XOR-swizzled transposed LDS image written by the staging
  copy (gfx942 design): layout and conflict model exist (`XtLayout`,
  `XtWriter`), the emission is **not built**.

`kt_source = "reg"` requires `kv_residency = "reg"`. Register residency costs
`kN0 * D / (2 * wave * W)` registers per lane per operand (K, V, K^T) and saves
the per-step LDS reads; LDS residency is required on WMMA targets.

### 3.5 The step pipeline and its barriers

`BwdTileStep.run_q_step`, default knobs:

```
 q step (qt, head h), CTA-wide; "|B|" = workgroup barrier
 ----------------------------------------------------------------------------------
 [stats_load = step_top: lse2 / Dsum loads]
 stage Q, dO rows [q0, q0 + kM0) -> LDS q, do       (qdo_stage: split | loads_first | joint)
 |B1|  publish Q, dO
 G0   S  = Q K^T          A: q (LDS)   B: K (regs | LDS)
 G2   dP = dO V^T         A: do (LDS)  B: V (regs | LDS)
 softmax gradient -> P, ds_mult * dS (fp32, rounding point)
 hand-off   relabel: P^T, dS^T in registers; dS -> LDS ds
            lds:     P -> LDS pt, dS -> LDS ds; |B2| publish P, dS
 G1   dV += P^T dO        B: do via transpose source        \
 G3   dK += dS^T Q        B: q  via transpose source         >  stage_order (default g1g3g4)
 G4   dQ_part = dS K      A: ds (LDS)  B: K (regs | LDS)    /   relabel: |B2| sits right before G4
      -> dQ sink (fp32 atomics; block or per fragment)
 |B3|  retire the reads of q, do, ds, pt before the next step overwrites them
 [agpr_fence route: carried dK / dV through an "=a" asm at the back-edge]
```

What each barrier protects:

| Barrier | Protects | Notes |
|---|---|---|
| B1 | RAW: the Q / dO `ds_write`s of all waves before any wave's G0 / G2 / G1 / G3 reads | the global loads that feed the writes are waited by data dependency inside the issuing wave |
| B2 | RAW: dS (and P on the LDS route) written by every wave before G4 (and G1 / G3 on the LDS route) reads it | under relabel it is placed right before G4 in every stage order, so the `ds_write` latency overlaps G1 / G3 |
| B3 | WAR: every read of `q`, `do`, `ds`, `pt` of this step before the next step's staging writes | |

Three barriers per step on both routes. `loop_sync` chooses their flavour:
`"full"` (default) is `s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier`, so B3 also
waits for this step's dQ atomics; `"lds_only"` is `s_waitcnt lgkmcnt(0)` +
`s_barrier`. `lds_only` is correct because no VMEM result crosses an in-loop
barrier from one wave to another: the staging loads are consumed by their own
wave, the statistics are used by the wave that loaded them, and the dQ
atomics are never read inside the kernel. It is proven for the register
staging path only (a DMA path writes LDS from VMEM), hence the coupling to
`global_path = "vgpr"`. The staging and epilogue barriers are always full.

`stage_order` permutes the issue order of G1, G3 and G4 (six orders; for
example `g4g1g3` issues G4 and its atomics right after dS is published so they
overlap the G1 / G3 matrix ops). Each accumulator's MMA chain is unchanged by
the order, so dK, dV and every dQ partial are bitwise those of the default.

Q / dO staging (`copy_row_sets_to_lds`, `qdo_stage`): `"split"` copies each
tensor in turn with each chunk's LDS store right after its global load;
`"loads_first"` issues every global load of both tiles before the first LDS
store (tail chunks load a clamped chunk and only the store is predicated);
`"joint"` gives the Q and dO tiles disjoint whole waves of one pass (legal
only when `kM0 * D / 8` is below the thread count and a multiple of the wave
size). All three move the same bytes to the same cells. Rows are clamped to
`len_q - 1` so MMA operands always come from valid rows; out-of-range rows are
masked by `keep`.

### 3.6 One-time K / V staging

`BwdTileStep.stage_kv`, before the q loops. Global reads happen only inside
`scf_if(live)` with `live = k0 < len_kv` (CTA-uniform), so a tile past the
sequence reads nothing.

```
 kv_residency = "reg"                               kv_residency = "lds"
 ------------------------------------------         -------------------------------
 [live] K rows -> LDS (k_stage or k_res)            [live] K -> k_res, V -> v_res
 |B| full                                           |B| full
 K B-fragments for G0 -> registers                  (operands read every step)
 [kt_source = reg] K^T fragments -> registers
 |B| full   K reads done before V overwrites the stage-0 region
 [live] V rows -> LDS v_stage
 |B| full
 V B-fragments for G2 -> registers
 |B| full   stage-0 reads done before the loop phase is written
```

The stage-0 K and V buffers and the q-loop buffers alias one LDS region
(section 4.1); these barriers are what make that aliasing safe. The dK / dV
accumulators are zeroed after the q-loop trip counts are known, so the
empty-band case (trip count 0) is decided before any accumulator is live.

### 3.7 dK / dV epilogue

After the q loops (`attention_bwd._direct_epilogue`, `_atomic_epilogue`):

* **Direct mode.** Rows `k < len_kv` get the values (`dk_mult` on dK); batched
  rows `len_kv <= k < S_kv_max` get exact zeros (also for an empty band or a
  tile past the sequence, where the accumulators stayed zero); rows at or past
  `S_kv_max`, and THD rows outside the segment, are not touched
  (`_attention_bwd_band.dkv_store_rule`). `dkv_epilogue` picks the path; every
  path stores the same values to the same elements:
  * `"lds_staged"` (default): round once to the I/O dtype, write the C layout
    to the `dkv_out` LDS buffer, barrier, then store row segments of
    `dkv_store_vec` elements (dK, then dV through the same buffer);
  * `"lds_staged_pair"`: dK and dV in two buffers behind one barrier;
  * `"direct"`: each lane stores its accumulator cells element by element
    (no LDS, no barrier).
  With LDS accumulators the epilogue reads the fp32 images directly.
* **Atomic mode.** Inside `scf_if(live)`, every accumulator element is one
  `global_atomic_add_f32` into `WS_DK` (times `dk_mult`) or `WS_DV`, rebased in
  i64 per head and tile row. `dkv_atomic_guard = "branch"` puts one branch per
  key row around its atomics; `"select"` issues them all and adds `+0.0` for a
  row outside the sequence or the workspace (an fp32 sum that starts at `+0.0`
  never becomes `-0.0`, so this is bitwise).

THD direct mode can form the dK / dV output slabs before the q loops and pin
each 64-bit pointer with an opaque SGPR copy (`dkv_slab_hoist`), so two
pointers instead of the token base, both tensor pointers and their strides
stay live across the loops (an SGPR-pressure fix; same values).

### 3.8 Designed, not built

| Feature | Knob value | What it would do |
|---|---|---|
| Prefetch ring | `ring_depth = 2 / 3` | load the next step's Q / dO (and stats) while this step computes (register ring with `global_path = "vgpr"`, LDS ring on gfx950); the register model already counts the prefetch registers and the planner the ring slots |
| LDS DMA | `global_path = "dma"` (gfx950) | fill Q / dO with `buffer_load ... lds` (16-byte lanes) into the 128-byte slab layout with a source-side XOR (`DmaSlabLayout`) |
| Barrier merge | none registered | with a two-slot ring, publish the next step's tiles and retire this step's reads in one barrier, and issue the next step's loads before the dQ atomics so a counted `vmcnt` wait leaves the atomics in flight |
| Stage tables / iglp | `sched != "none"`, `setprio = True` | interleave each stage's memory work between single-MFMA `sched_group_barrier` groups (A: G0; B: G1 + G2; C: G3; D: G4) with `sched_barrier(0)` fences; `StageTable` holds the per-wave count model and the emission hooks, which emit nothing with `sched = "none"` |
| G4 dS key slicing | `block_k4` | read dS in `kK4` key slices double-buffered in registers; today G4 is emitted one atom-K step at a time over the whole `kN0` tile, so `block_k4` changes only the kernel name (`NOT_YET_EFFECTIVE_KNOBS`) |
| XT transposes (gfx942) | `transpose_source = "xt_lds"`, `lds_swizzle = ...=xor` | write Q^T / dO^T / K^T images with an XOR swizzle so a K-outer B read is one `ds_read_b64` |
| Split dQ | `dq_mode = "split"` | 2.4 |
| Mirror pairing | `lb_order = "mirror"` | 2.7 |
| Work list | `use_worklist = True` | a prep role that writes a heaviest-first list of (sequence, kv tile) items for skewed THD batches |

## 4. Memory layout

### 4.1 LDS phase planner

`_attention_bwd_lds.plan_bwd_lds` places every LDS buffer of a CTA inside one
flat `smem_alloc` (`BwdLds`), and the body reads every buffer through an
`LdsView` at the planner's byte offset, so the planner's footprint is the
kernel's group segment and its aliasing is exactly what the IR does.

Buffers (names are stable):

| Buffer | Phase | Size | Present when |
|---|---|---|---|
| `k_res`, `v_res` | resident | `kN0 x D` | `kv_residency = "lds"` (`k_res` also when K^T is read from LDS with register K / V) |
| `dk_acc`, `dv_acc` | resident | `kN0 x D` fp32 | `acc_in_lds` |
| `kt_res` | resident | XT image `D x kN0` | `xt_lds` with LDS K^T (not built) |
| `k_stage` | stage-0 K | `kN0 x D` | register K / V without a resident `k_res` |
| `kt_stage` | stage-0 K | XT image | `xt_lds` with register K^T (not built) |
| `v_stage` | stage-0 V | `kN0 x D` | register K / V |
| `q`, `do` | loop, per ring slot | `kM0 x D` | always |
| `qt`, `dot` | loop, per ring slot | XT images | `xt_lds` (not built) |
| `ds` | loop | `kM0 x kN0` | atomic dQ or LDS route |
| `pt` | loop | `kM0 x kN0` | `pt_route = "lds"` |
| `dkv_out` | epilogue | `kN0 x D` (one or two slots) | direct register accumulators with an LDS-staged epilogue |

Footprint = resident + max(stage-0 K, stage-0 V, loop, epilogue), every buffer
aligned to a 128-byte line so a per-buffer swizzle sees the same address
classes at any offset. `lds_order` puts the resident region below
(`resident_low`, default) or above (`phase_low`) the aliased region; sizes are
equal. `LdsPlan.check` proves that no two buffers live in one phase overlap.

```
 gfx950 d128 default (16 x 128, K / V in registers, K^T from LDS), fp16 / bf16
 offset 0 KiB          32 KiB                                 64 KiB
        +--------------+---------------------------------------+
        |   k_res      |  stage-0 V:  v_stage (32 KiB)         |   one of these
        |  (resident:  |  loop:       q 4 | do 4 | ds 4 KiB    |   at a time,
        |   G4 K^T)    |  epilogue:   dkv_out (32 KiB)         |   barrier-separated
        +--------------+---------------------------------------+
 gfx942 d128 default (16 x 128, K, K^T, V in registers): no resident buffer;
 k_stage, v_stage, the loop tiles and dkv_out share one 32 KiB region.
```

Row layouts: every buffer the body reads is row-major with an optional row
pad (`lds_swizzle`: `none`, `pad8`, `pad16`, `pad32` bytes per row, per buffer,
canonical string `"buf=value,..."`). `LdsView` never claims more vector
alignment than the origin and pitch allow (a padded pitch splits wide
accesses). The XOR layouts (XT image, DMA slab) belong to the unbuilt paths.

Per-arch capacity is the catalog's `lds_capacity_bytes`: 64 KiB on gfx942,
gfx1151 and gfx1201, 160 KiB on gfx950. The validator refuses a plan above it.

### 4.2 Register plan

**Model.** `attention_bwd.register_estimates` (per lane, 4-byte registers, two
16-bit operands per register):

| Term | Registers |
|---|---|
| resident K, V (`kv_residency = "reg"`), K^T (`kt_source = "reg"`) | `kN0 * D / (2 * wave * W)` each |
| Q / dO step operands | `2 * kM0 * D / (2 * wave)` |
| register prefetch (ring >= 2; not built) | `kM0 * D / (wave * W)` |
| P / dS operands | `kM0 * kN0 / (wave * W)` |
| G4 dS operands | `2 * kM0 * k_g4 / (2 * wave)` (one atom-K step live) |
| dK, dV accumulators | `kN0 * D / (wave * W)` each (none with `acc_in_lds`) |
| S, dP accumulators | `2 * kM0 * kN0 / (wave * W)` |
| dQ partial | `kM0 * D / (wave * W)` |
| allowance (addresses, loop state, stats, masks) | 24 |

Each accumulator term is counted in the arch-VGPR or the AGPR budget according
to where the emitted code keeps it (`_agpr_placement`, below).

**Budgets.** CDNA has 256 arch VGPRs and 256 AGPRs per lane, one 512-entry
unified file per SIMD. A CTA of `W` waves puts `ceil(W / 4)` waves on each
SIMD, which share the file: an eight-wave CTA gives each wave 256 registers in
total, whatever the AGPR attribute says. The validator applies: arch-VGPR
estimate within 256 (plus a static prune allowance of 32; the compiled
resource notes then decide), AGPR estimate within the reservation, arch +
AGPR within the per-wave share for `W > 4` and for `waves_per_eu` 2 / 3, and
an explicit reservation `(n, n)` must leave at least 24 arch VGPRs of the
share (`(256, 256)` on eight waves is refused). RDNA has 256 VGPRs per lane
and no AGPRs.

**Why AGPR placement matters.** The VALU reads S and dP (softmax gradient)
and the dQ partial (the atomics) every step. When an MFMA writes its result in
AGPR form, each such read is a `v_accvgpr_read` inside the q loop, and an
allocator short of registers adds AGPR spill copies (`v_accvgpr_write` /
`mov`) as well. The backend picks the form per kernel, not per GEMM: with an
`amdgpu-agpr-alloc` reservation (or no attribute and a file that does not fit)
it selects the AGPR form for every MFMA, G0 / G2 / G4 included. Only dK / dV
benefit from AGPRs: they are written by G1 / G3 every step and read by the
VALU only in the epilogue.

**The plans** (`agpr_alloc`, `agpr_route`):

| Plan | Spelling | Arch VGPRs hold | AGPRs hold | Mechanism |
|---|---|---|---|---|
| VGPR form | `agpr_alloc = (0, 0)` | everything | nothing | attribute `0,0` plus the comgr option `-amdgpu-mfma-vgpr-form` |
| attr route ("CK plan") | `(n, n)`, `agpr_route = None` | operands, P / dS | dK, dV, S, dP, dQ partial | the reservation attribute only; the backend selects the AGPR form for every MFMA |
| split, `vgpr_form` | `(n, n)` + `"vgpr_form"` | everything but dK / dV | dK, dV (if the allocator puts them there) | reservation plus the VGPR-form comgr option (`mfma_vgpr_form` kernel attribute) |
| split, `agpr_fence` | `(n, n)` + `"agpr_fence"` | everything but dK / dV | dK, dV | as `vgpr_form`, plus each carried dK / dV fragment passes through an empty asm with an `=a` output at the back-edge (`_agpr_fence`) and once through `=v` after the loop (`finish`) |
| split, `asm_g13` | `(n, n)` + `"asm_g13"` | everything but dK / dV | dK, dV | G1 / G3 issued as AGPR-form inline-asm MFMA clusters (section 5); G0 / G2 / G4 stay VGPR-form intrinsics |
| `asm_g13` + `s_in_agpr` | | | dK, dV, S, dP | G0 / G2 as asm too: **not built** |

`agpr_alloc = None` is the policy: `(0, 0)` when the whole estimate fits 256
arch VGPRs (always with LDS accumulators), else no attribute (backend choice);
`"backend"` forces no attribute. The shipped CDNA defaults are eight-wave
tiles precisely because of this: on four waves the per-wave state (resident K,
V, K^T and the dK / dV accumulators) fits only with accumulators in AGPRs,
which drags S / dP / dQ into AGPRs too; eight waves halve the per-wave state
and the VGPR form holds all of it.

**Copy budget.** `attention_bwd.agpr_copy_budget` is the one model of the
hot-loop rule: reads allowed per q step are `2 * kM0 * kN0 / (wave * W)` (S, dP)
plus `kM0 * D / (wave * W)` (dQ partial) under the attr route with
`s_in_agpr = True`, the S / dP term only under `asm_g13` with `s_in_agpr`, and 0
everywhere else; writes and moves are always 0 (they can only be spill
traffic). The resource gate counts `v_accvgpr_read / write / mov` in every loop
that issues matrix instructions (edge tiles emit three q loops) against it.

**`s_in_agpr`** declares S / dP in AGPRs. Under the attr route it changes no
code (the backend already puts them there); it selects the register estimate
and the copy budget, so the honest declaration of the attr route is
`s_in_agpr = True`, and the `False` twin always fails the copy rule
(`DECLARATION_ONLY`). It is legal only with the attr route or `asm_g13`.

Other register levers: `waves_per_eu` (kernel attribute; 3 only on gfx942 with
at most two waves), `scheduler_strategy` (the backend's machine scheduler; the
one-wave d128 tiles, the CDNA narrow d128 tile and the RDNA d128 tiles resolve
to `iterative-minreg`, because the default scheduler fills the 256-VGPR file
there and some instances spill), `acc_in_lds` (dK / dV as fp32 LDS images:
each 16 x 16 tile is read before its MMA chain and written back after it; each
lane reads back exactly the cells it writes, so no barrier is needed; implies
the VGPR form), and `dkv_slab_hoist` / `interior_clamp` (address and clamp
placement, which shift the allocator's choices).

## 5. Inline-asm MFMA (`asm_g13`)

`_attention_bwd_gemm.mma_agpr_asm_steps` issues the MFMAs of one K step
(`asm_cluster = "k_step"`, default) or of a whole GEMM call (`"gemm"`) as one
`asm sideeffect convergent` block:

* every accumulator `acc[fm][fn]` is an `=a` output tied to its input, so dK /
  dV live in AGPRs across steps with no copy;
* the A and B fragments are `v` inputs, each passed once (bitcast to an i32
  vector of the same width), read exactly as on the intrinsic route;
* MFMAs are issued K outer, fragment inner (the intrinsic route's
  accumulation order, so the result is bitwise the intrinsic one); with a
  single fragment per wave the K steps of a `"gemm"` cluster are separated by
  `s_nop`;
* the block ends with `s_nop asm_tail_nop`.

**Hazards.** An asm MFMA is opaque to the backend's hazard recognizer, so the
cluster must cover its own dependencies (`_attention_bwd_caps`):

* a dependent read of an XDL MFMA result that is not the next MFMA's
  full-overlap srcC (a VALU / VMEM / LDS read, `v_accvgpr_read`, srcA / srcB,
  partial srcC) needs `passes + 3` wait states on gfx942 and `passes + 4` on
  gfx950; every 16 x 16 f16 / bf16 atom here is 4-pass, so 7 and 8
  (`mfma_result_wait_states`);
* a full-overlap srcC chain (the next K step on the same accumulator) needs
  none;
* a VALU write of a VGPR an MFMA reads needs 2 wait states before the MFMA
  (`ASM_MFMA_VALU_LEAD_WAIT_STATES`).

`s_nop n` gives `n + 1` wait states and the instruction right after the
cluster may read the last MFMA's result, so the floor is the result wait
states minus one: **6 on gfx942, 7 on gfx950** (`asm_tail_nop_min`), whatever
the cluster size. The validator refuses a smaller `asm_tail_nop`
(`check_asm_tail_nop`); the shipped count is 15 (`AGPR_ASM_TAIL_NOP`), and a
resolved spec spells 15 as the default `None`.

**Evidence** (`tests/test_attention_bwd_agpr_probe.py`). A block GEMM cannot
show the hazard (a full-overlap srcC chain needs no wait state), so the probe
uses a negative control that provably reads early: one MFMA and a
`v_accvgpr_read` of its result in one asm block with nothing in between (the
disassembly is checked). On gfx942 and gfx950 that control gives wrong values,
while every count from the floor up is bitwise equal to the intrinsic, and a
chained srcC sequence with nothing in between is bitwise. An ISA audit of
whole `asm_g13` kernels built at the floor checks that every asm MFMA result
is read no earlier than the table allows and that no VALU write of an asm MFMA
source lands inside the lead window.

**Costs and couplings.** Each asm block is a scheduling fence (side effect), so
`asm_g13` excludes stage tables and `iglp` (the scheduler cannot see the MFMAs
inside), and `asm_cluster = "gemm"` reduces the number of fences.

## 6. Masking and iteration

### 6.1 Keep rule

The frozen forward band (`rocke.helpers.attention_band`) plus the backward's
row term:

```
keep(q, k) = q < len_q and k < len_kv
             and (right < 0 or k <= q + off + right)
             and (left  < 0 or k >= q + off - left)
off = 0 (top-left) or len_kv - len_q (bottom-right), per sequence
```

Causal is `left = -1, right = 0`; a sliding window sets either or both bounds
(either alignment); no mask uses `NoMaskKTail` (only the q tail and the k tail
are masked). Bounds and lengths stay below `2^30` (i32-safe arithmetic).

### 6.2 Inverse band (`_attention_bwd_band.q_tile_range`)

A key tile `[k0, k0 + kN0)` is attended by rows `q_lo <= q <= q_hi`:
`q_lo = max(0, k0 - off - right)` (right bounded), `q_hi = min(len_q - 1,
k_last - off + left)` (left bounded), `k_last = min(k0 + kN0 - 1, len_kv - 1)`.
The row range is exact, so the only slack is tile quantisation at both ends.
A tile past `len_kv`, `len_q = 0`, or a band that misses every row gives
`n_qt = 0`. Both results go through `readfirstlane`, so the q-loop trip counts
live in SGPRs. The classifier and the inverse band are checked exhaustively
against a brute-force keep on the host (`tests/test_attention_bwd_band.py`).

### 6.3 Per-row key limits

On edge steps each row gets limits `[lo, hi]` once
(`RuntimeBand._row_lo_hi`: `lo = max(q + lo_off, 0)`, `hi = min(q + hi_off,
len_kv - 1)` with CTA-uniform `hi_off = right < 0 ? 2^30 : off + right` and
`lo_off = left < 0 ? -2^30 : off - left`), with `hi = -1` for a row that keeps
nothing (outside `len_q`, empty band row, dead row, dead packed row). The cell
test is one compare: `min(max(k, lo), hi) == k`.

### 6.4 Edge / interior / edge loops

A (q tile, key tile) pair is **interior** when every row is `< len_q`, every
key `< len_kv` and every cell inside the band (checked at the two corners
that bound the band). The interior tiles of a key tile form one contiguous
sub-range `[i0, i1)` of the visited range (`interior_q_range`), so with
`edge_tiles = True` the q loop is emitted as three loops over
`[qt_start, i0)`, `[i0, i1)`, `[i1, qt_end)`, in the single loop's order:

* interior steps skip the per-cell keep selects and keep only the dead-row
  select of dS (P of a dead row is `+0` anyway), and drop the row clamps of
  the Q / dO staging and the dQ sink (`interior_clamp = True` keeps them, which
  changes only register allocation);
* the accumulation order is unchanged, so dK / dV are bitwise those of
  `edge_tiles = False` in direct mode (a device test checks it).

A head-packed tile is always an edge tile. The default is `edge_tiles = False`
(one loop, selects on every step).

### 6.5 Step order, GQA groups and head packing

The CTA's work is a flattened sequence of (q tile, head group) steps with
`dK / dV` carried through it:

* `step_order = "qtile_outer"` (default): q tiles outer, head groups inner, so
  each q-tile range (edge / interior / edge) is one contiguous run of steps;
* `"head_outer"`: q tiles inner; with several q-tile ranges a loop over the
  head groups runs the range loops of one group per trip. dK / dV are summed in
  a different order when a CTA has more than one head group and more than one
  q tile.
* `q_rotate`: each q-tile range starts at `qt0 + kt mod len` and wraps, so the
  kv-tile CTAs of one head walk its q tiles (and dQ rows) out of lockstep (an
  order change for dK / dV).

**Head packing** (`head_pack`, `PackedRows`, `PackedStep`). A GQA group whose
query span is short (`s_q = 1` decode-like cells) would fill one of the `kM0`
rows of each step. The plan sets `head_pack` when `G > 1` and
`G * S_q <= kM0`, and passes `pack_heads = G`: M row `r` of the single q tile
then holds row `r % rows_per_head` of query head `head0 + r / rows_per_head`
(`rows_per_head = max(len_q, 1)` at runtime). Staging, statistics, the keep and
the dQ sink use each row's own head (per-row i64 head rebase); rows of a head
index past the group are dead in every term. `pack_heads = 0` runs the
unpacked loop in the same binary.

### 6.6 Sequences and addressing

**Decode** (`_attention_bwd_addr.decode_batched`, `decode_thd`):

* batched: `len = has_len ? clamp(SEQ[b * len_stride], 0, S_max) : S_max`; the
  SEQ pointer is read inside a 0/1-trip loop, so a null pointer is never
  dereferenced when `has_len = 0`; workspace rows start at `b * S_max`;
* THD: `o0 = OFF[b]`, `o1 = OFF[b + 1]` (int32, or int64 read as two words, by
  the runtime `off64`; an int32 table is never read past `B + 1` entries);
  `tok0 = o0 * mult / div`, `len = clamp((o1 - o0) * mult / div, 0, S_max)` in
  64-bit arithmetic (`div` is the tensor's token stride for element offsets),
  `min`-ed with the SEQ count under `has_len`. Workspace rows start at `tok0`
  (`ws_seg = 0`, rows by storage token) or at `b * S_max` (`ws_seg != 0`, rows
  by sequence slot); the selected base is `readfirstlane`'d so the backend
  cannot sink it past the loops.

**User tensors** (`StridedTensorAddr`): per-tensor `(batch, head, token)`
element strides, unit D stride. `slab(h, row0)` rebases the pointer once with a
64-bit byte offset (`global_ptr_add`) to (batch or sequence, head, first row of
the tile); the remaining per-element index is a small i32 (`< rows * stride_t +
D`, guaranteed by the predicate's `INDEX_RANGE`). In THD mode the sequence base
`tok0 * stride_t` is folded into the base pointer when the address generator is
built, so one 64-bit pointer (not a pointer and a 64-bit offset) stays live
across the q loops; the gfx950 THD kernels sit at the SGPR ceiling without it.
Rows are clamped to `max(len - 1, 0)` for operand reads (`clamp=True`) and
addressed raw for predicated stores (`clamp=False`). `stage_vec` sets the global
access width of the user tensors only: 8 issues 16-byte vectors (every used
B / H / S stride a multiple of 8 elements, bases 16-byte aligned), 1 issues
2-byte accesses (any element stride, any element-aligned base). LDS, MMA,
workspace and ABI are the same for both.

**Workspace** (`WorkspaceAddr`): fp32 `[h][rows][X]` (`head_major`) or
`[rows][h][X]` (`token_major`), `X = D` or 1. The (head, base row) offset is an
i64 pointer rebase; rows at or past `ws_rows` are clamped to a valid row and
their values replaced (reads), zeroed (atomics) or skipped (epilogue atomics),
so no access leaves the workspace even when a caller's THD bound is too small.

### 6.7 Zero-length spans

* `len_q = 0`: `n_qt = 0` for every key tile, so dK / dV of that batch are
  exact zeros; prep writes neutral statistics; in batched mode the convert
  writes the dQ rows below `S_q_max` as exact zeros, in THD mode it writes
  nothing.
* `len_kv = 0`: every key tile has `k0 >= len_kv`: no global read, no q step;
  in batched direct mode the epilogue still stores exact zeros to dK / dV rows
  below `S_kv_max`; dQ stays exactly 0 (zeroed workspace, no contribution).
* THD zero-length sequences, including the last one: the offset table is read
  at `b` and `b + 1` only; no row of the segment exists, nothing is written.
* An empty band behaves the same (0 trips, zero accumulators, zero rows).

## 7. Arch specifics

| | gfx942 | gfx950 | gfx1151, gfx1201 |
|---|---|---|---|
| Matrix path | MFMA, wave64 | MFMA, wave64 | WMMA, wave32 |
| Atoms | 16x16x16 | 16x16x16, 16x16x32 | wmma 16x16x16 |
| LDS per workgroup | 64 KiB | 160 KiB | 64 KiB |
| Registers per lane | 256 arch VGPR + 256 AGPR (unified 512 per SIMD) | same | 256 VGPR, no AGPR |
| Transpose read | none (`lds_plain`) | `ds_read_tr16_b64` (`tr_read`) | none (`wmma_lds`) |
| LDS DMA | one dword per lane (not used) | 1, 3 or 4 dwords (`dma` path not built) | none |
| Legal waves | 1, 2, 4, 8 | 1, 2, 4, 8 | 1, 2, 4 |
| Legal ring depth (registered) | 1, 2 | 1, 2, 3 | 1 |
| `waves_per_eu` | 1, 2, 3 (3 with W <= 2) | 1, 2 | 1, 2 |
| asm tail `s_nop` floor | 6 | 7 | n/a |
| Native fp32 no-return atomic add | yes | yes | yes |

Default start points (`attn_bwd_default_geometry`; a tuned table overrides
them per class):

| Target, d | kM0 x kN0, W | G4 grid | K / V, K^T | Transpose | Atoms G02 / G13 / G4 |
|---|---|---|---|---|---|
| gfx942 d32 | 32 x 128, 4 | (2, 2) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx942 d64 | 32 x 128, 8 | (2, 4) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx942 d128 | 16 x 128, 8 | (1, 8) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx950 d32 | 32 x 128, 4 | (2, 2) | reg, reg | tr_read | 32 / 16 / 32 |
| gfx950 d64 | 32 x 256, 8 | (2, 4) | reg, lds | tr_read | 32 / 16 / 32 |
| gfx950 d128 | 16 x 128, 8 | (1, 8) | reg, lds | tr_read | 32 / 16 / 32 |
| CDNA narrow (`stage_vec = 1`) | 16 x 64, 4 (d32: 32 x 64) | (1, 4) / (2, 2) | reg, reg | lds_plain | gfx950 wide where legal |
| RDNA | 16 x 16, 1 | (1, 1) | lds, lds | wmma_lds | wmma |

Every default uses `global_path = "vgpr"`, ring depth 1, no stage table,
`pt_route = "relabel"` on CDNA and `"lds"` on RDNA, register accumulators on
CDNA and LDS accumulators on RDNA, `waves_per_eu = 1` on CDNA. The gfx950 d128
tile is 16 x 128 on eight waves because 192 keys do not split over eight waves.
The narrow tiles exist because scalar global access cannot feed a large tile.

RDNA specifics: one wave runs the whole 16 x 16 tile; K / V and the transposed
operands come from LDS; the relabel route is refused (the WMMA C layout is not
the A layout of the transposed product without a cross-lane exchange); dK / dV
are fp32 LDS images, so G4 always feeds the sink one fragment at a time, which
bounds the live registers. RDNA targets carry no performance claim
(`degraded` in the declaration) and are never declined on performance.

## 8. Families

### 8.1 Layout-general (built)

Everything above. One builder (`build_attn_bwd_main(spec, *, arch)`), one
aux spec shared with prep / convert (`AttnBwdSpec.aux_spec`), one plan. The
compile-time problem class is `(dtype, d, seq_mode, dkv_mode, mask_class,
stage_vec)`; everything else is a knob.

### 8.2 Dense (designed, not built)

A per-arch family for contiguous BSHD / BHSD fixed-length problems with the
shapes baked at build time: compile-time strides and tile counts, buffer
resources with exact `num_records` (out-of-bounds loads return zero instead of
clamping), compile-time mask classification (mask-free interior loop plus a
masked diagonal tail), a separate kernel for lengths that are not tile
multiples, the batch as a runtime value. It would only accelerate cells the
general family already declares.

What exists: `dense_bwd_eligibility` (gfx942 / gfx950, every tensor dense
BSHD or BHSD with the same layout on the q and kv sides, base alignment >= 16
bytes, no padding and no SEQ_LEN tensors, `h_k == h_v`, d 64 / 128; never a
decline, only a routing tag) and `attn_bwd_route`, which takes the dense route
only when the exact dense key is in `AttnBwdPolicy.dense_pack`. The shipped
pack is empty, so every request takes the general route; `attn_bwd_plan`
plans only the general route.

### 8.3 Unified (designed, not built)

THD queries over a paged KV cache (`[num_blocks, page, h_kv, D]` with a block
table), with dK / dV scattered into a paged gradient cache: direct scatter
when the caller asserts that the block table is injective over live slots,
else an fp32 page workspace with atomics. Reachable from rocKE dispatch only
(hipDNN keeps declining `PAGED`).

What exists: `AttnBwdAuxSpec(kv_target = "paged")` with
`build_attn_bwd_page_zero` and `build_attn_bwd_page_convert` (they walk
`j < ceil(len_kv / page)` blocks of every sequence, read the block-table entry
once per CTA and zero / convert that physical page's slots; shared pages are
visited once per referencing sequence with identical writes), the
`PagedCacheAddr` / `block_table_entry` address generators, the
`rocke.attn_bwd_unified.v1` aux ABI tag, and the request field
`block_table_injective` (ignored by the general predicate). No unified main
kernel, plan or predicate exists.

### 8.4 Routing

```
request -> attn_bwd_support (general predicate) -> decline (reason code) | OK
OK      -> attn_bwd_route: dense-eligible and key in dense_pack -> "dense" (no family built)
                           otherwise                           -> "general" -> attn_bwd_plan
```

Routing uses request fields only (no device query) and never adds or removes
a declared cell. There is no backward dispatcher (request / candidate /
ranker); `dispatch/attention_bwd/` holds the knob registry only.

The scalar `fmha_bwd.py` kernel and the `mfma_attention_bwd.py` helpers are a
separate, older backward (CK Tile parity); this family neither uses nor
extends them.

## 9. Tuning model

### 9.1 Knob registry

`dispatch/attention_bwd/tuning_common.py` registers every tuning decision as a
`KnobAxis`: every `AttnBwdSpec` field except the problem class, every plan
runtime value (`RUNTIME_KNOBS`: `g_split`, `lb_order`, `xcd_chunk`,
`scale_placement`, `use_worklist`) and every prep / convert value (`AUX_KNOBS`).
`_check_registry` fails the import if a spec field has no axis. Per axis:

* `choices[0]` is the default (the field's declared default, or the plan's
  value), so leaving an axis alone emits today's code;
* `family`: one of eight lever families in release order (geometry,
  residency / registers, transpose / global path, atoms / operand routes,
  schedule, codegen / minor / swizzle, runtime, split dQ);
* `stage`: the funnel stage that owns it (geometry, structure, schedule,
  codegen, swizzle, runtime);
* `kind`: `compile` (a spec field: recompiles, salted into the kernel name) or
  `runtime` (a kernarg, or an aux-kernel recompile);
* `level`: `production` or `full`; `full_only` values are enumerated only by the
  full (per-shape) sweep;
* `numerics`: `bitwise`, `reorders_fp` (same algorithm, another fp32 order or
  rounding point: within tolerance, not bit-equal), `tile` (another tile,
  atom, route or placement: its own kernel, held to the tolerance) or `n/a`;
* `not_built`: values the emission check or the plan refuses (recorded by the
  sweep, never compiled); `gated`: values a device probe has not cleared (none
  today);
* `plan_owned`: fields the plan derives from a runtime value (`static_scale`,
  `lb_interleave`, `xcd_remap`, `head_pack`), never enumerated on their own;
* `requires`: a cheap coupling filter; the kernel validator still decides.

Legality has one source: `backward_spec_legality` = `validate_attn_bwd_spec`
then `_not_built`.

**Lever retention.** A knob that loses everywhere stays registered and off by
default; nothing is deleted for losing. Knob values that change nothing under a
given spec are listed in `INERT_KNOBS` (predicate per spec), `s_in_agpr` under
the attr route in `DECLARATION_ONLY`, and `block_k4` in
`NOT_YET_EFFECTIVE_KNOBS`; the sweep compiles one representative per set of
configurations that emit the same code.

### 9.2 Funnel sweep

`benchmarks/common/attention_bwd_sweep.py` sweeps primary slices `(arch, d)`
(bf16, band mask, batched, direct, `stage_vec = 8`) and secondary slices
(fp16, no mask, THD, atomic, `stage_vec = 1`, head packing), stage by stage:

```
geometry (keep 8) -> structure (top 4 geometries; keep 8) -> schedule (4) -> codegen (4)
  -> swizzle (4) -> alternate (re-run structure on the top 16; 4) -> runtime (2) -> confirm (top 2 on the cohort)
```

Each stage: build candidates from the previous stage's merged records, drop
illegal and duplicate candidates (resolved specs equal apart from inert
knobs), compile (on-disk cache keyed by spec, arch, LLVM flavor, comgr and
source hash; knobs that can crash the compiler compile in child processes),
apply the resource gate (section 11) as a static prune, then time the
survivors with the paired harness: rocKE configurations and the competitor
in the same fresh-process attempts, every configuration correctness-gated in
the timing process. A lever family that is not released contributes only its
default. Records are JSONL outside the source tree; stdout carries counts and
verdict tokens only.

### 9.3 Tuned tables and performance classes

* **Performance class** (`attn_bwd_perf_class`, request fields only):
  `(arch, route, d, dtype, mask kind {none, causal_tl, causal_br, window}, head
  mode {mha, gqa, hk_ne_hv}, length relation {square, unequal, s_q_1}, length
  mode {fixed, padded, thd}, stage_vec)`; the dense route's class is its exact
  key. The device hints `num_cus` / `num_xcds` never enter it.
* **Tuning key** (`attn_bwd_tuning_key`): the class plus a kv-tile bucket
  (`ceil(s_kv / kN0)` at the class's default kN0 against 1, 4, 16, 64) and a
  grid-fill bucket (default grid against `num_cus`: under, full, over,
  unknown). `AttnBwdPolicy.tuned_table` is looked up most-specific first (exact,
  then either bucket as `*`, then both), then the `tuning` hook overrides it.
* **Policy** (`shipped_policy`): reads `kernels/common/attention_bwd_tuning.py`
  when the tree has it. This tree has none, so no class is measured: gfx942 /
  gfx950 requests are declined `PERF_UNVERIFIED` unless a policy waives
  unverified classes, and RDNA requests are served as `degraded`.

### 9.4 Knob catalog

Default first. Numerics: **bitwise** (same fp32 operations in the same order),
**order** (another summation order or rounding point; within tolerance),
**tile** (own kernel; tolerance against the oracle, no bitwise claim). No
registered knob has relaxed numerics; a NaN-guard-free bf16 cast and the fast
`exp2` are designed and not registered.

Geometry, residency, paths, atoms:

| Knob | Values | What changes in the generated code | Numerics | Couplings / status |
|---|---|---|---|---|
| `waves` | policy; 1, 2, 4, 8 (WMMA: 1, 2, 4) | CTA size; every warp grid | tile | `kN0 % 16W == 0`; W = 1 only for 16 x 16 |
| `block_m` (kM0) | policy; 16, 32, 64 | q rows per step | tile | multiple of `16 * g4r`; >= 32 for `atom_g13 = 16x16x32` |
| `block_n` (kN0) | policy; 16, 32, 64, 128, 192, 256 | keys per CTA, grid x | tile | multiple of `16W` and of `block_k4` |
| `block_k4` (kK4) | policy; 16, 32, 64 | nothing but the kernel name | bitwise | not yet effective |
| `warp_grid_g4` | policy; `(1, W)`, `(2, W/2)` | G4 wave split | tile | must divide D and kM0 |
| `warp_grid_g02` | `(1, W)`; `(2, W/2)` | G0 / G2 wave split | tile | `(2, W/2)` needs `pt_route = "lds"` |
| `warp_grid_g13` | `(W, 1)`; `(2, W/2)` | G1 / G3 wave split | tile | as above |
| `kv_residency` | policy; reg, lds | K / V B-fragments in registers once vs LDS reads per step | tile | reg illegal on WMMA |
| `kt_source` | policy; reg, lds | G4 K fragments in registers once vs transposed LDS reads per step | tile | reg needs register K / V |
| `transpose_source` | policy; tr_read (gfx950), lds_plain, wmma_lds (WMMA), xt_lds | K-outer LDS reader | tile | `xt_lds` not built |
| `global_path` | policy; vgpr, dma (gfx950) | global-to-LDS path | tile | `dma` not built |
| `ring_depth` | policy; 1, 2, 3 | prefetch depth | tile | 2, 3 not built |
| `atom_g02`, `atom_g4` | policy; 16x16x16, 16x16x32 (gfx950), wmma16x16x16 | MFMA instruction of G0 / G2, G4 | tile | arch atom set |
| `atom_g13` | policy; same | MFMA of G1 / G3; K-permuted B under relabel | tile | x32 needs kM0 >= 32; relabel + x32 + tr_read not built |
| `pt_route` | policy; relabel, lds | P^T / dS^T hand-off (3.3) | tile | relabel illegal on WMMA |
| `acc_in_lds` | policy; False, True | dK / dV as fp32 LDS images | order | implies `(0, 0)`; required on WMMA |

Registers, schedule, issue order:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `agpr_alloc` | policy; backend, (0,0), (128,128), (192,192), (256,256) | `amdgpu-agpr-alloc` attribute and comgr option | bitwise | reservation fits the estimate and leaves >= 24 arch VGPRs |
| `agpr_route` | attr; vgpr_form, agpr_fence, asm_g13 | register plan (4.2) | bitwise | needs `(n, n)`, n > 0; asm_g13 excludes stage tables and iglp |
| `s_in_agpr` | policy; False, True | attr route: estimate and copy budget only | bitwise | not with `(0, 0)`; with asm_g13 not built |
| `asm_tail_nop` | 15; floor to 14 (gfx942 6, gfx950 7) | the `s_nop` ending each asm cluster | bitwise | asm_g13 only; below the floor refused |
| `asm_cluster` | k_step; gemm | one asm cluster per K step or per GEMM | bitwise | asm_g13 only |
| `waves_per_eu` | policy; 1, 2, 3 | occupancy attribute | bitwise | 3 on gfx942 with W <= 2; halved register budget |
| `scheduler_strategy` | backend default; max-ilp, max-memory-clause, iterative-ilp, iterative-minreg, iterative-maxocc | machine scheduler | bitwise | compiled crash-isolated by the sweep |
| `sched` | none; iglp0, iglp1, fences_only, stage_table, stage_table_atomics | sched groups / fences | bitwise | not built |
| `setprio` | False; True | `s_setprio` around stages | bitwise | not built; stage tables only |
| `stage_order` | g1g3g4; g3g1g4, g4g1g3, g1g4g3, g3g4g1, g4g3g1 | issue order of G1, G3, G4 + sink | bitwise | not with stage tables |
| `loop_sync` | full; lds_only | in-loop barrier waits | bitwise | lds_only needs `global_path = "vgpr"` |
| `qdo_stage` | split; loads_first, joint | Q / dO staging issue order / thread split | bitwise | ring 1 and vgpr path; joint below one pass |
| `stats_load` | at_use; step_top | where lse2 / Dsum loads issue | bitwise | ring 1 and vgpr path |
| `dq_issue` | block; fragment, fragment_regs | G4 then atomics per block or per fragment | bitwise | inert with acc_in_lds, and fragment over one fragment per wave |

Codegen, epilogue, layout:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `edge_tiles` | False; True | edge / interior / edge q loops | bitwise | |
| `interior_clamp` | False; True | row clamps kept on interior steps | bitwise | edge_tiles only |
| `step_order` | qtile_outer; head_outer | step loop nesting | order | |
| `q_rotate` | False; True | rotated start of each q-tile range | order | |
| `head_pack` | plan; False, True | (head, row) packing on M rows | order | plan-owned; `G * s_q <= kM0` |
| `ws_layout` | head_major; token_major | workspace addressing (main, prep, convert) | bitwise | |
| `dkv_epilogue` | lds_staged; lds_staged_pair, direct | direct-mode dK / dV store path | bitwise | register accumulators for the non-default values |
| `dkv_store_vec` | 8; 4, 2, 1 | row-store width of the staged epilogue | bitwise | direct mode, staged epilogue |
| `dkv_atomic_guard` | branch; select | row guard of the atomic epilogue | bitwise | atomic mode |
| `dkv_slab_hoist` | False; True | dK / dV pointers formed before the loops | bitwise | THD direct only |
| `ptr_noalias` | False; workspace, all | `noalias` on workspace / all main pointers | bitwise | all needs atomic mode |
| `lds_swizzle` | none per buffer; pad8, pad16, pad32 (xor) | LDS row pitch | bitwise | xor not built |
| `lds_order` | resident_low; phase_low | LDS region placement | bitwise | inert without resident buffers |
| `static_scale` | fold_ds; dq_before_atomic, convert, runtime | which multipliers are emitted | bitwise | plan-owned (follows `scale_placement`) |
| `lb_interleave` | False; True | decode serves `lb_order = 3` | bitwise | plan-owned |
| `xcd_remap` | False; True | decode reads `xcd_n` / `xcd_chunk` | bitwise | plan-owned |
| `dq_mode` | atomic; split | dQ strategy | n/a | split not built |

Runtime and aux values:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `g_split` | 1; divisors of G | atomic dK / dV, more CTAs per group | order | `> 1` recompiles main in atomic mode |
| `lb_order` | natural; reverse, mirror, interleave | kv-tile order of the decode | bitwise | mirror not built |
| `xcd_chunk` | 0; gx, 2gx, 4gx, 64 | XCD remap chunk | bitwise | needs `num_xcds` |
| `scale_placement` | fold_ds; dq_before_atomic, convert | where `s` is applied (1.4) | order | |
| `use_worklist` | False; True | work-list role | n/a | not built |
| `prep_rows_per_cta` | 1x; 2x, 4x | prep CTA size | bitwise | prep recompile |
| `convert_rows_per_cta` | 1x; 2x, 4x | convert CTA size | bitwise | convert recompile |
| `convert_vec` | 8; 4, 16 | convert elements per lane | bitwise | convert recompile |

## 10. Numerics and determinism

* **Oracle and tolerance** (`tests/sdpa/bwd_cases.py`). Gradients are compared
  with a float64 reference evaluated on exactly the stored inputs. The bound is
  norm-relative per gradient tensor over valid rows,
  `max|got - ref| <= atol + rtol * g * max|ref|` with
  `g = max(1, sqrt(kv_len_max / 1024))`: fp16 `rtol 4e-3, atol 1e-4`, bf16
  `rtol 2.5e-2, atol 1e-3`, plus a per-row check floored at 5 % of the tensor
  maximum (catches errors confined to one row, such as a dropped tail key).
  A per-element relative bound is meaningless here: each gradient is a sum of
  many products that can cancel to near zero, while the error of a correct
  fp32-accumulate kernel scales with the tensor's magnitude. The bounds were
  calibrated against a low-precision simulation of the backward and leave
  headroom for other summation orders, `exp` implementations and P / dS
  rounding points.
* **Nondeterminism.** dQ is summed with fp32 atomics from many CTAs in a
  run-dependent order: nondeterministic in the last bits. In atomic mode dK /
  dV are too. In direct mode dK / dV are deterministic: each is summed on chip
  in a fixed step order and written once. A deterministic backward is declined.
* **Bitwise knobs.** Every knob marked bitwise gives direct-mode dK / dV bit for
  bit equal to the default (device tests: runtime variants, stage orders, edge
  tiles, narrow twin, token-major workspace, slab hoist, static scale). Order
  knobs stay within the tolerance.
* **Relaxed numerics.** None is built. A designed relaxed knob (a bf16 cast
  without the NaN guard, the fast `exp2`, an FMA of the exponent argument)
  would be registered default-off and gated by a recorded numerics decision.
* **Contracted rounding.** `_rounding_point` (1.5) is what keeps every step
  variant's dS rounding identical; removing it can make edge and interior steps
  disagree in the last bit.

## 11. Invariants enforced by the tests

* **Resource gate** (`tests/_attention_bwd_resources.evaluate_resource_gate`,
  run on every IR-suite instance by
  `test_attention_bwd_ir.test_zero_scratch_no_spills_and_lds_equal_to_the_plan`):
  zero scratch, zero VGPR and SGPR spills, arch VGPRs and AGPRs each within 256
  (read from the code object's metadata and the kernel descriptor's
  `accum_offset`), occupancy at least 1, LDS within the per-arch capacity and
  not above the planner's total, AGPR copies in every matrix loop within
  `agpr_copy_budget`, and an SGPR watch level whose crossing is treated as a
  failure. A result counts only on the `llvm22` flavor with the production
  comgr vintage (`GATE_FLAVOR`, `GATE_COMGR_ROCM`); other toolchains skip.
* **IR identity of defaults.** Spelled-out defaults emit the default IR; every
  non-default value of a built knob changes the IR beyond the kernel name;
  not-yet-effective and declaration-only values emit the default IR apart
  from the name; asm knobs that emit the default code resolve to the default
  spelling (`_canonical_asm_knobs`). Kernel names are salted by every
  non-default compile-time field (`AttnBwdSpec.kernel_name`).
* **Structure** (`test_attention_bwd_ir.py`): kernarg order equals the ABI
  lists; LDS barriers separate cross-wave accesses; MMA and atomic counts per
  step equal the stage-table model; dQ atomics are agent-scope native adds;
  workspace rebases are 64-bit; global access widths follow `stage_vec`; no
  128-bit transpose read; the q-loop trip count is `readfirstlane`'d, the
  empty-band exit precedes live accumulators, and no matrix instruction sits in
  a divergent region.
* **Proofs on the host:** lane-map facts and relabel plans per atom pair,
  transpose-read coordinates, the inverse band and tile classifier against a
  brute force, LDS plan non-overlap, register model against worked tables,
  validator / planner agreement, predicate reachability of every reason code.
* **Device:** the case table against the oracle on every target, edge /
  interior bitwise equality, dead rows (`-inf` / NaN LSE, poisoned O), runtime
  variants, zero-length and ragged storage forms, understated THD bounds
  (memory-safe, exact zeros past the bound), the asm hazard probe with its
  negative control, the transpose-read lane-map probe, the forward-to-backward
  chain (with a frozen fixture), and negative controls that must fail (log2
  LSE, shifted LSE, flipped bottom-right, missing dead-row term, missing group
  sum, skipped prep, double scale, skipped converts, ignored SEQ_LEN, wrong
  offset width).
* **Registry:** every spec field has exactly one axis; marked values are
  non-default choices; plan-derived fields mirror the plan.

## 12. Host contract

### 12.1 Statistics (LSE)

* Input: **natural-log** log-sum-exp, **fp32**, one value per query row and
  query head. Accepted forms (`lse_dims` / `lse_strides`; `()` dims are the
  layout's canonical dims, `()` strides their packed strides):
  * dense (batched, padded): `[B, H_q, S_q, 1]` with rank-4 `(b, h, t, 1)`
    strides; packed `(H_q * S_q, S_q, 1, 1)`;
  * THD: `[T_q, H_q, 1]` (any `T_q`) with rank-3 `(t, h, 1)` strides; packed
    `t = H_q`, `h = 1`;
  * THD: the batched shape `[B, H_q, S_q, 1]` with explicit rank-4 strides
    whose B stride is unused (tokens come from the ragged offsets); no packed
    default.
  Every stride is a non-negative integer; the stride of a trailing extent-1
  dim never moves an address. Any other form is declined `LSE_FORMAT` (the
  detail names why); no form is replaced by a default.
* Dead rows: `!(lse > -inf && lse < +inf)` or outside the sequence; on a dead
  row `dQ = 0` and the row contributes nothing to dK / dV, whatever `O` holds.
* Only this contract couples the backward to a forward
  (`tests/test_attention_bwd_fwd_chain.py` checks it against the forward's LSE
  epilogues and a frozen fixture, `tests/sdpa/bwd_fixture_v1.py`).

### 12.2 Coverage

| Axis | Values |
|---|---|
| dtype | fp16, bf16 for every tensor; fp32 accumulation |
| head size | 32, 64, 128 (`d_qk == d_v`); 256 declined `HEAD_DIM_256_BWD` |
| layout | any non-negative B / H / S element strides with D stride 1 (BSHD, BHSD, packed QKV views), per tensor |
| lengths | fixed; padded (`SEQ_LEN_Q/KV` with `padding_mask`); THD (`[B + 1]` int32 / int64 offsets, multiplier) |
| heads | MHA, GQA, MQA; `h_k != h_v` (atomic dK / dV) |
| masks | none, causal top-left / bottom-right, sliding window (left / right, either alignment), deprecated causal booleans |
| `s_q = 1` | general kernel with head packing |
| scale | host f32 (default `1 / sqrt(d)`); pass-by-value scale tensors resolved on the host |

Strides that are not multiples of 8 elements, or a base alignment in
`[2, 16)` bytes, select the narrow-access instance (`stage_vec = 1`).

### 12.3 Kernarg ABI `rocke.attn_bwd.v3`

Pointers are global; inputs `readonly`; by default no pointer is `noalias`
(`ptr_noalias` marks the workspace pointers, or every main pointer in atomic
mode; the ABI is unchanged); unused pointers are 0 and never dereferenced.
Order is exact (`attn_bwd_params`): pointers, then i64, f32 and i32 scalars (the
i64 block is 8-byte aligned without padding):

```
main   ptr : Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dk_b dk_h  dv_b dv_h
       f32 : scale_log2 ds_mult dk_mult dq_mult
       i32 : q_t k_t v_t do_t dk_t dv_t
             h_q h_k h_v G gk gv  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_rows_kv ws_seg
             g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist
prep   ptr : O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST
       i64 : o_b o_h  do_b do_h  l_b l_h
       i32 : o_t do_t l_t  h_q h_k h_v  S_q_max S_kv_max
             has_len len_stride  off64 q_mult q_div kv_mult kv_div  ws_rows_q ws_rows_kv ws_seg
             zero_kv zero_dq use_worklist n_batch wl_kn0
convert ptr : SRC DST SEQ OFF
       i64 : d_b d_h
       f32 : mult
       i32 : H  d_t  S_max  has_len len_stride  off64 tok_mult tok_div  ws_rows ws_seg
dq     ptr : Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV            (split dQ kernel, not built)
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dq_b dq_h
       f32 : scale_log2 ds_mult dq_mult
       i32 : q_t k_t v_t do_t dq_t
             h_q h_k h_v G  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_seg  xcd_n xcd_chunk
```

Batch and head strides are i64 (a tensor may span `2^31` elements or more
between batches or heads); token strides are i32 (the predicate bounds
`S_max * stride_t`). Strides are in elements; THD tensors ignore the `*_b`
strides. `G = h_q / h_k` in direct mode, `gcd(h_q / h_k, h_q / h_v)` in atomic
mode. Every v2 scalar keeps its name and meaning in v3. Every request the
predicate admits packs without narrowing (`pack_attn_bwd_args` raises, naming
the kernarg, otherwise). Launch sequence: `prep -> main -> convert(dQ)`, plus
`convert(dK)` and `convert(dV)` in atomic mode.

### 12.4 Workspace

`A(x) = round_up(x, 256)`; every array is fp32; sub-buffers are packed in this
order at 256-byte aligned offsets (`attn_bwd_workspace_layout`):

```
rows_q, rows_kv  : fixed / padded: B * S (the tensor dims)
                   THD with both max_total_q and max_total_kv: those bounds (rows by storage token, ws_seg = 0)
                   THD otherwise: B * S_q_max, B * S_kv_max (rows by sequence slot b * S_max + i, ws_seg = 1)
WS_DQ    = A(4 * rows_q * h_q * D)                 dq_mode == "atomic"
WS_LSE2  = A(4 * rows_q * h_q)
WS_DSUM  = A(4 * rows_q * h_q)
WS_DK    = A(4 * rows_kv * h_k * D)                h_k != h_v or g_split > 1
WS_DV    = A(4 * rows_kv * h_v * D)                same condition
WORKLIST = A(8 * (B + ceil(rows_kv / kN0)))        work list only (not built)
workspace_bytes = sum of the above
```

Example: B = 2, S_q = S_kv = 128, h_q = h_k = h_v = 8, D = 128, fixed lengths,
`g_split = 1`: `1,048,576 + 8,192 + 8,192 = 1,064,960` bytes; with
`g_split = 2` add `2,097,152`.

The size is an upper bound for every admitted request without a device query.
Without both caller bounds (hipDNN has no max-total API) the rows of THD
sequence `b` are its slot `[b * S_max, b * S_max + len_b)`, which no storage
layout can push past `B * S_max`. With both bounds the rows are the storage
tokens and the caller asserts that each bound covers the end token of every
segment; a single bound is not used. Caller contract: a 256-byte aligned base;
an understated bound gives wrong gradients for the excess tokens but no
out-of-bounds access (every workspace row is clamped); a supplied
`max_total_q` / `max_total_kv` must be at least 1 for a non-empty problem.

### 12.5 Predicate and reason codes

`attn_bwd_support(req, arch, *, policy=None) -> Verdict(ok, code, detail)` is a
pure function of the request and the in-repo policy tables; the first failing
check wins:

| # | Code | Decline when |
|---|---|---|
| 1 | `ARCH_UNSUPPORTED` | arch not in {gfx942, gfx950, gfx1151, gfx1201} |
| 2 | `ARCH_NOT_VALIDATED` | the arch is in the policy's `not_validated_archs` (empty in the shipped policy) |
| 3 | `DROPOUT`, `BIAS_IN_BACKWARD`, `DBIAS`, `ALIBI`, `PAGED`, `FP8`, `DETERMINISTIC`, `UNSUPPORTED_FEATURE` | the feature is requested (in this order) |
| 4 | `SCALE_DEVICE_TENSOR` | scale is a device-buffer tensor |
| 5 | `DTYPE` | Q, K, V, O, dO not all fp16 or all bf16 |
| 6 | `OUT_DTYPE` | dQ, dK, dV differ from the input dtype |
| 7 | `LSE_FORMAT` | stats not fp32 or not in a form of 12.1 |
| 8 | `DQK_NE_DV` | `d_qk != d_v` |
| 9 | `HEAD_DIM_256_BWD` | d == 256 (out of scope for the backward in the requirements; the detail cites the scope) |
| 10 | `HEAD_DIM` | d not in {32, 64, 128} |
| 11 | `GQA_RATIO` | `h_q % h_k != 0` or `h_q % h_v != 0` |
| 12 | `BAND_BOUND` | a bound `< -1` or `>= 2^30`; both causal booleans; a causal boolean with explicit bounds |
| 13 | `STRIDE_FORMAT` | a stride tuple without exactly four integer entries `(B, H, S, D)` |
| 14 | `STRIDE_D` | a D stride != 1 |
| 15 | `STRIDE_NEGATIVE` | a negative stride |
| 16 | `OUTPUT_OVERLAP` | two distinct indices of dQ, dK or dV share an element (a sorted-extent test proves most layouts disjoint, the rest are decided by enumerating index differences; beyond the enumeration budget the layout is declined conservatively) |
| 17 | `STRIDE_ALIGN` | base alignment below 2 bytes |
| 18 | `INDEX_RANGE` | `S_max * stride_s + D >= 2^31` for a tensor; `S_q * stride_t + 1 >= 2^31` for the LSE; an i64 stride or byte offset `>= 2^63`; `rows_q >= 2^31`; `rows_kv >= 2^31` (atomic mode); `S_max * D >= 2^31`; `S_max >= 2^30` |
| 19 | `GRID_LIMIT` | `B > 65535`; main units `> 65535`; prep y `> 65535`; prep / convert x `>= 2^31` |
| 20 | `SEQ_LEN_FORMAT` | SEQ_LEN tensors without `padding_mask`; padding without both tensors; not int32 `(B, 1, 1, 1)` |
| 21 | `RAGGED_FORMAT` | THD offsets not int32 / int64 `[B + 1]`; q-side or kv-side tensors not on one table; multiplier or token stride outside `[1, 2^31)`; a supplied `max_total_*` below 1 for a non-empty problem |
| 22 | `EMPTY` | B, a head count, a sequence dim or D is 0 |
| 23 | `INSTANCE_UNAVAILABLE` | the serving instance failed the resource gate |
| 24 | `PERF_BELOW_BASELINE` | the request's performance class on its route is `below_baseline` |
| 25 | `PERF_UNVERIFIED` | the class is `unverified` and the policy does not waive unverified classes |
| 26 | `NOT_YET_DECLARED` | the request needs a functional feature group whose correctness gate has not passed |
| - | `OK` | otherwise |

Functional feature groups, in gate order (all declared in the shipped
policy): `base`, `masks_and_stats` (any mask, explicit scale, d32, `s_q = 1`),
`grouped_heads`, `split_kv_heads`, `padding`, `ragged`, `narrow_access`.
`attn_bwd_feature_groups(req)` lists the groups a request needs. The plan also
refuses runtime values that are not built (mirror pairing, the work list) and
a `g_split` that does not divide the head group.

### 12.6 Declaration

`attn_bwd_declaration(arch, *, policy=None)` returns a JSON-serialisable dict:
`family`, `version` (3), `abi` (`rocke.attn_bwd.v3`), `arch`, `status`
(`declared`, `not_validated`, `unsupported`), `rdna_status`, accepted value sets
per axis (`head_dim_declined` carries the code and the reason), the ordered
`reason_codes`, `feature_groups` and `declared_feature_groups`, the `lse`
contract, the `workspace` formula, the `determinism` statement, the
`perf_policy`, `perf` (one entry per (route, class) with its status), the
`deferral_register` and `notes` (gfx1201: declared on its hardware run; its
forward-chain tests skip because the forward probe shell has no gfx12 body).

Deferral register (`DEFERRAL_REGISTER`):

| Requirement | Code | Consequence for hipDNN |
|---|---|---|
| item 9 (d = 256) / item 15 | `HEAD_DIM_256_BWD` | d = 256 models are servable for inference but not trainable through this provider; their training attention stays on CK / AOTriton |
| item 14 (bias, forward only) / item 15 | `BIAS_IN_BACKWARD` | `F.sdpa(attn_mask=...)` under autograd falls back to CK / AOTriton on every target |
| item 15 (no dBias) | `DBIAS` | graphs that need dBias train on CK / AOTriton |
| deterministic backward | `DETERMINISTIC` | a deterministic backward request falls back to CK / AOTriton |

### 12.7 Out of scope (declined)

dBias (`DBIAS`), bias or attention mask in the backward (`BIAS_IN_BACKWARD`),
dropout (`DROPOUT`), deterministic backward (`DETERMINISTIC`), d = 256 backward
(`HEAD_DIM_256_BWD`; the forward serves d = 256), fp8 (`FP8`), ALiBi (`ALIBI`),
paged KV through this entry (`PAGED`; paged problems belong to the unified
family, not built), block masks, sink tokens and extra statistics outputs
(`UNSUPPORTED_FEATURE`).

## 13. Known limitations and performance characteristics

Qualitative only; measurements live outside this repository.

* **The main kernel dominates.** Prep and convert are a small share of the
  backward on all but the smallest problems, and launch gaps are negligible,
  so fusing the auxiliary kernels is not a lever; the per-step efficiency of
  the main kernel is.
* **dQ atomic drain.** With `loop_sync = "full"` the end-of-step barrier waits
  for the step's dQ atomics (`vmcnt` counts no-return atomics). `lds_only`
  removes that wait from the barrier, but the next step's staging loads are
  issued after the atomics and `vmcnt` retires in order, so the wait tends to
  move to the next staging `s_waitcnt` rather than disappear. Hiding it needs
  the next step's loads issued before the atomics (the prefetch ring and a
  counted wait), or no atomics (split dQ); neither is built.
* **Staging latency.** With ring depth 1 every step loads Q / dO
  synchronously, and B1 exposes the global latency once per step unless
  other resident workgroups hide it.
* **Softmax-gradient VALU.** Per cell: the exponent multiply-subtract, the
  guarded `exp2` (a range-reduction sequence), compares and selects, the dS
  arithmetic, and two fp32-to-16-bit casts. On gfx942 the bf16 cast is a
  multi-instruction NaN-preserving round-to-nearest-even sequence (gfx950 has a
  packed convert instruction). Designed cheaper forms (fast `exp2`, an FMA for
  the exponent argument, a guard-free bf16 cast, packed fp32 math) are not
  built.
* **AGPR plans.** Four-wave tiles with large key tiles need AGPR accumulators;
  the backend's all-or-nothing AGPR form puts S / dP / dQ there too and adds
  hot-loop copies. The split routes (`vgpr_form`, `agpr_fence`, `asm_g13`) keep
  them out; `asm_g13` costs scheduling freedom (each cluster is a fence).
* **Causal imbalance.** Key tiles near the diagonal carry fewer q steps;
  reverse and interleave orders are built, two-tile mirror pairing is not.
* **gfx942 transposes.** Without a transpose read, K-outer operands on gfx942
  are read one element at a time (`lds_plain`); the XOR-swizzled transposed
  images that would make them vector reads are not built.
* **Competitor structure** (stated neutrally). The production backward
  kernels this family is compared against on ROCm use one or more of: a
  separate Q-parallel dQ kernel that recomputes S and dP and writes dQ once
  without atomics; small tiles at occupancy above 1 without in-kernel
  pipelining; or hand-scheduled kernels that own a large key tile per
  workgroup at occupancy 1 with DMA streaming, issue the next tile's loads
  before the step's dQ atomics and wait with partial `vmcnt` counts (no full
  drain in the loop), and pair causal key tiles per workgroup. Some resolve
  GQA by expanding dK / dV per query head and reducing afterwards, where this
  family sums the group on chip.
