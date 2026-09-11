# Gated DeltaNet (GDN) — algorithm and design

> **Scope.** The GDN operator family on gfx950: a dedicated single-token **decode** kernel and a
> chunkwise **prefill** forward that ships as a mode of the existing KDA chunkwise kernel.
> This document specifies *what* the kernels compute and *why they are shaped the way they are*.
> It is a specification, not a tuning history, and it carries **no measurements** — latency is
> recorded in the internal perf repository per repository compliance.
>
> **This file is the entry point for the family.** Its sibling `README.md` covers the host-side
> tools — driver, benchmark, retuning, exit codes — and nothing about what the kernels compute.

---

## Contents

- [0. Notation](#0-notation)
- [1. What GDN is](#1-what-gdn-is)
  - [1.1 Why a fixed-size state can stand in for the past](#11-why-a-fixed-size-state-can-stand-in-for-the-past)
  - [1.2 The gated delta rule](#12-the-gated-delta-rule)
  - [1.3 Operator dimensions](#13-operator-dimensions)
  - [1.4 Target geometries](#14-target-geometries)
- [2. GDN is a special case of KDA](#2-gdn-is-a-special-case-of-kda)
  - [2.1 The single operator difference](#21-the-single-operator-difference)
  - [2.2 Why that makes the chunkwise machinery reusable](#22-why-that-makes-the-chunkwise-machinery-reusable)
  - [2.3 What GDN mode still has to add](#23-what-gdn-mode-still-has-to-add)
- [3. Two kernels for one operator](#3-two-kernels-for-one-operator)
  - [3.1 The two regimes](#31-the-two-regimes)
  - [3.2 Why decode is not prefill at C = 1](#32-why-decode-is-not-prefill-at-c--1)
  - [3.3 Why decode also needs serving infrastructure](#33-why-decode-also-needs-serving-infrastructure)
- [4. Decode kernel](#4-decode-kernel)
  - [4.1 Tensor contract](#41-tensor-contract)
  - [4.2 Parallel decomposition](#42-parallel-decomposition)
  - [4.3 Dataflow and pipeline](#43-dataflow-and-pipeline)
  - [4.4 Cross-lane reduction](#44-cross-lane-reduction)
  - [4.5 State pool addressing](#45-state-pool-addressing)
  - [4.6 Tile selection by batch](#46-tile-selection-by-batch)
  - [4.7 The reference path](#47-the-reference-path)
  - [4.8 Spec validation](#48-spec-validation)
- [5. Prefill kernel](#5-prefill-kernel)
  - [5.1 Chunkwise factorization](#51-chunkwise-factorization)
  - [5.2 The triangular solve](#52-the-triangular-solve)
  - [5.3 The state scan](#53-the-state-scan)
  - [5.4 Split and fused schedules](#54-split-and-fused-schedules)
  - [5.5 value_splits](#55-value_splits)
  - [5.6 GDN-mode deltas, all in prep](#56-gdn-mode-deltas-all-in-prep)
- [6. Reuse in the other direction: KDA on GDN](#6-reuse-in-the-other-direction-kda-on-gdn)
- [7. Validation strategy](#7-validation-strategy)
- [8. Known limits and follow-ups](#8-known-limits-and-follow-ups)

---

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| `C` | chunk length in tokens (`KdaTileSpec.chunk`, default 32, legal `(16, 32)`) |
| `DK`, `DV` | `head_k_dim`, `head_v_dim` |
| `Hk`, `Hv` | `num_k_heads`, `num_v_heads` |
| `BH` | `batch × num_v_heads` — the number of independent recurrences |
| `NC` | chunks per sequence, `seqlen / C` |
| `S` | the recurrent state of one head, `DV × DK` |
| `q̂`, `k̂` | L2-normalised query/key |
| `Γ_i` | cumulative in-chunk decay up to row `i`; `γ_C` is the whole-chunk decay |
| `EV` | per-band value extent, `DV / value_splits` — the scan's working V rows (§5.5) |

Ownership vocabulary: a **workgroup** is one thread block; a **wave** is 64 lanes; a **lane** is
one thread. `WTK = warp_threads_k`, `WTV = wave_size / WTK`, `NW = num_warps`,
`BPV = blocks_per_v_dim`, `VPT = STATE_VEC = 8` (the 16-byte bf16 vector width).

Both kernels are Python **emitters**, not GPU code: `build_gdn_decode(spec, arch)` and the KDA
chunkwise builders use rocKE's `IRBuilder` to construct a target-neutral `KernelDef`. Python loops
in an emitter run at build time and unroll into IR; they are not runtime loops unless the emitter
creates explicit control flow. The path from a request to a running kernel is

```
request → dispatch picks a spec → builder emits KernelDef → rocKE lowers to LLVM IR
        → comgr compiles a gfx950 code object → launcher packs kernargs and launches
```

so a spec is the unit that dispatch selects, that the golden test pins, and that the validators
in §4.8 accept or reject.

---

## 1. What GDN is

GDN is a **linear-attention** operator. Instead of re-reading every past token, it carries a
fixed-size recurrent state `S` per value head and updates it once per token. Cost per token is
constant in sequence length, and the memory footprint does not grow.

### 1.1 Why a fixed-size state can stand in for the past

Removing the softmax lets the read collapse:

```
o = Σᵢ (q · kᵢ) vᵢ  =  (Σᵢ vᵢ kᵢᵀ) q  =  S q ,   S = Σᵢ vᵢ kᵢᵀ
```

The sequence index is the contracted axis, so it cancels: `S` is `DV × DK` regardless of token
count. This regrouping is only valid because the read is linear in `q` — the softmax's `exp` and
normalisation sit between `q` and `vᵢ` and block it. The same linearity is what makes the
chunkwise prefill formulation in §5 possible at all.

The trade is recall: `S` is a lossy summary. Contributions decay geometrically and superimpose,
so exact retrieval of one distant token is not recoverable. Production models accept this by
interleaving a minority of full-attention layers.

### 1.2 The gated delta rule

Per token, per value head:

```
q̂     = l2norm(q) * head_k_dim**-0.5
k̂     = l2norm(k)
decay = exp(-exp(A_log[vh]) * softplus(a[vh] + dt_bias[vh]))     # (0, 1)
β     = sigmoid(b[vh])                                            # (0, 1)

S     = decay * S                        # 1. gated forget
v_new = (v - S @ k̂) * β                  # 2. error-correcting delta
out   = S @ q̂ + v_new * dot(k̂, q̂)        # 3. readout
S     = S + outer(v_new, k̂)              # 4. rank-1 write
```

The `q̂`/`k̂` L2-normalisation shown here is the default (`use_qk_l2norm`); the decode spec can
disable it to consume pre-normalised inputs, a variant pinned by its own golden IR case.

Three properties worth stating explicitly, because each one drives kernel structure:

1. **`decay` is a keep factor, not a forget factor.** `decay = 1` retains everything. Smaller
   values forget faster, compounding across steps.
2. **The write is error-correcting.** `S @ k̂` is what the state already predicts for this key;
   only the residual `v - S @ k̂` is stored. Writing the same key twice does not double-count,
   and unrelated associations are left intact. This is what distinguishes the *delta* rule from
   a plain outer-product accumulation `S += v kᵀ`.
3. **The readout does not depend on the rewritten state.** Step 3 uses the identity
   `S_after @ q̂ = S_faded @ q̂ + v_new * (k̂ · q̂)`, so the output and the state write are
   independent given the faded state. The decode kernel exploits this directly (§4.3).

### 1.3 Operator dimensions

GDN uses a grouped layout in which **value heads outnumber key heads**, because the recurrent
state is per *value* head:

| Quantity | Role |
| --- | --- |
| `num_k_heads` | heads for `q` and `k` |
| `num_v_heads` | heads for `v`, `out`, and **one `S` each** |
| `kv_group = num_v_heads / num_k_heads` | value heads sharing one key head |
| `head_k_dim`, `head_v_dim` | state is `head_v_dim × head_k_dim` per value head |

Value head `h` reads key/query head `h // kv_group`. `kv_group = 1` is the MHA case and is the
KDA configuration.

### 1.4 Target geometries

Every constant in this document is sized for these deployments. Values are from each model's
published `config.json`.

| Model | `Hv` | `Hk` | `DK` | `DV` | `kv_group` | gate |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Next-80B-A3B (36 of 48 layers) | 32 | 16 | 128 | 128 | 2 | scalar — GDN |
| Kimi-Linear-48B-A3B (20 of 27 layers) | 32 | 32 | 128 | 128 | 1 | per-channel — KDA |

Qwen3-Next reads `linear_num_value_heads`, `linear_num_key_heads` and
`linear_key_head_dim = linear_value_head_dim`; Kimi Linear reads `linear_attn_config.num_heads`
and `linear_attn_config.head_dim`, and KDA has no head grouping, so `Hk = Hv`.

> **Out of scope: Qwen3-Next's other 12 layers.** The same model interleaves gated *full*
> attention every fourth layer (`full_attention_interval = 4`), and those layers are
> `head_dim = 256` with `num_attention_heads = 16` / `num_key_value_heads = 2` — no recurrent
> state, so no value of `kv_group` makes them servable here. They belong to the attention
> family, not this one.

Three things the table pins down that the rest of the document assumes:

- **`DK == DV` on every supported row.** The tile table and the `DV × DK` state shape both rely
  on it. A target with `DK ≠ DV` needs the tile geometry re-derived.
- **`kv_group = 2` is the only shipping GDN grouping.** The `(Hv, Hk) = (32, 8)` case in §7 —
  `kv_group = 4` — is a validation stress point, not a deployment.
- **Where these models land in the tuned tables.** Both have `Hv = 32`, so `BH = 32 × batch`.
  Prefill's `value_splits` bands on `BH` (`≤64 → 8`, `≤128 → 2`, else `1`), which means batch 1-2
  gets 8 splits, batch 3-4 gets 2, and batch 5 and up runs unsplit; batches 2 and 4 sit exactly
  on band edges. Decode's tile table bands on *batch* directly (`≤4`, `≤32`, `≤128`, larger), so a
  serving batch crosses all four.

This also fixes the scope of §2.3's reuse-over-fork argument. A target that keeps the gated delta
rule but changes the gate's *formula* stays a `gate_kind`, not a fork. A target that changes `C`,
forces a different tile set, or makes the decay non-separable across `Γ_i / Γ_j` is a fork.

---

## 2. GDN is a special case of KDA

### 2.1 The single operator difference

KDA (Kimi Delta Attention) and GDN are the same gated delta rule. The operator difference is
**gate granularity**:

| | forget gate | shape |
| --- | --- | --- |
| KDA | per key channel | a `DK`-wide vector — each state column fades by its own amount |
| GDN | per (token, head) | one scalar, broadcast across `DK` |

Their functional forms differ too. Both are stated in **log space** — the domain of the gate
value itself, related to §1.2's multiplier by `decay = exp(gate)`, so `0` means no forgetting and
more negative means faster forgetting. There, KDA's gate is
`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, bounded in `(lower_bound, 0)`; GDN's is
`-exp(A_log) * softplus(a + dt_bias)`, which is `≤ 0` but **unbounded below** (`g` and `a` are the
raw per-token inputs, not the gate). §8 records the consequence.

### 2.2 Why that makes the chunkwise machinery reusable

A scalar decay is a *legal value* of a per-channel decay vector: set all `DK` entries equal.
Everything downstream of the gate — the six per-chunk tiles, the triangular solve, the serial
state scan — is indifferent to whether the decay vector happens to be constant. The general
machine therefore computes the special case exactly, with no change to its structure.

This is why GDN prefill is implemented as `gate_kind="gdn"` inside `kda_chunkwise.py` rather than
as a second engine. The direction matters and is not symmetric — see §6.

### 2.3 What GDN mode still has to add

Reuse is not free. GDN mode contributes, **entirely within the prep kernel**:

- its own gate evaluation, `-exp(A_log) * softplus(a + dt_bias)`, with a `softplus` shortcut above
  a threshold: past it `softplus(x) ≈ x` is used instead, so the overflowing `exp2` — still computed,
  then selected away — is never propagated;
- a **GQA gather** (`kv_group > 1`), so a value head reads the correct key head;
- a **scalar gate load** — one `f32` per row, broadcast across the channel group, where KDA reads a
  per-channel vector. The gate pointer is consequently typed `f32` in GDN mode; a mismatched
  pointer type would index at the wrong stride in the C++/HIP backend, where opaque pointers hide
  the error;
- `a` uses the token-major `beta` layout `[B, T, H]`, not the per-channel `[B, T, H, D]` layout;
- fused `q`/`k` L2-normalisation and fused `β = sigmoid(...)`, so the kernel consumes raw inputs.

The spec validator enforces that `gate_kind="gdn"` implies the raw-input path plus its three
input-fusion flags (q/k L2-norm, gate, β = sigmoid), and that
`kv_group > 1` is only valid in GDN mode. `gate_kind` and `kv_group` both participate in the kernel
name, so the cache key stays faithful to the emitted code.

**Compatibility guarantee.** GDN sits behind default-off spec flags: existing KDA specs emit
byte-identical IR, which the golden-hash test enforces.

---

## 3. Two kernels for one operator

### 3.1 The two regimes

| | Prefill | Decode |
| --- | --- | --- |
| Work per launch | the whole prompt | one token per sequence |
| Parallelism available | `BH × NC` — a sequence axis | `BH` only — no sequence axis |
| Dominant cost | matrix throughput | launch overhead and occupancy |
| Right tool | chunk the sequence, use MFMA | manufacture parallelism, minimise launch cost |

### 3.2 Why decode is not prefill at C = 1

This is the load-bearing reason for a second kernel, and it is structural rather than a tuning
preference. Prefill's efficiency *is* the chunk: a `C × C` key-key product, a `C × C` triangular
solve, six tiles amortised over `C` tokens. At `C = 1` every one of those degenerates to a scalar
— a `1 × 1` "solve", MFMA issued for a single element, tiles that reconstruct nothing — while the
full staging and setup cost remains. Decode is a *different algorithm*: a direct
fade → probe → correct → write sequence with no chunk, no solve, and no matrix unit.

### 3.3 Why decode also needs serving infrastructure

A second, independent reason. Decode serves many concurrent sequences across many steps, each with
its own persistent `S` retrieved by identity, so the state lives in a **pool** addressed by
`read_indices` / `write_indices`, with a negative sentinel marking idle continuous-batching lanes.
Prefill is one-shot: it ingests a prompt and *returns* a final state, so it has no notion of a pool.

The first reason forces a separate kernel; the second explains why that kernel also looks like
serving infrastructure.

---

## 4. Decode kernel

### 4.1 Tensor contract

All tensors are contiguous row-major.

| Tensor | Shape | Type | Direction |
| --- | --- | --- | --- |
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]` | `dtype` | in |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]` | `dtype` | in / out |
| `a`, `b` | `[B, 1, num_v_heads]` | `dtype` | in |
| `dt_bias` | `[num_v_heads]` | `dtype` | in |
| `A_log` | `[num_v_heads]` | `f32` | in |
| `read_indices`, `write_indices` | `[B]` | `i32` | in |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]` | `state_dtype` | in-place |

The launch also passes a trailing `batch_size` `i32` scalar (not a tensor).

### 4.2 Parallel decomposition

The grid is one-dimensional:

```
grid  = batch × num_v_heads × blocks_per_v_dim
bidx  = ((sequence × num_v_heads) + value_head) × BPV + v_sub_block
```

so the V sub-block varies fastest. Within a workgroup of `NW × 64` threads the state tile is cut
three ways:

| Cut | Knob | Effect |
| --- | --- | --- |
| across workgroups | `BPV` | each owns `TILE_V = DV / BPV` value rows |
| across waves and v-lanes | `NW`, `WTV` | `WGROUP_V = NW × WTV` rows in flight; each lane walks `WTV_ITERS = TILE_V / WGROUP_V` rows |
| across k-lanes | `WTK` | `WTK` lanes cover a row, `VPT = 8` contiguous channels each, repeated `WTK_ITERS = DK / (WTK × VPT)` times |

Live state per lane is `WTV_ITERS × WTK_ITERS × VPT` values, held **in registers**. The design is
deliberately register-resident: the kernel allocates **no LDS and issues no barriers**.

`BPV` is a parallelism-manufacturing knob, not a work-reducing one — each of the `BPV` workgroups
re-loads `q` and `k` and re-runs the normalisation reductions. It buys occupancy at small batch and
is retired at large batch, where the grid is already ample.

### 4.3 Dataflow and pipeline

One workgroup, one decode step, in emission order:

1. decode `bidx` and `tid` into `(sequence, value head, v-sub-block)` and `(wave, k-lane, v-lane)`;
2. load `read_indices` / `write_indices` and form the `active` predicate — **outside** the guard, so
   a padded lane costs two `i32` loads and exits;
3. under `scf_if(active)`: evaluate `decay` and `β`;
4. load the lane's `q` and `k` slices as 16-byte vectors, promoted to `f32`;
5. reduce the two L2 norms (two cross-lane reductions, §4.4);
6. reduce `dot(k̂, q̂)` (one more);
7. form the state read pointer, load the whole lane tile, applying `decay` as it lands;
8. per owned V row: reduce `s·k̂` and `s·q̂`, form `v_new = β (v − s·k̂)`, then emit the output and
   the rank-1 state update.

Two consequences of the identity in §1.2 item 3: the output store and the state write in step 8 are
**independent** — neither reads the other's result — and the output is broadcast across the k-lane
group, so only lane 0 of each group stores it.

Precision: every load is promoted to `f32` and all arithmetic — gates, norms, dot products, the
rank-1 update — is `f32`. Only the final `out` and the state write pack back to the storage type.
Transcendentals are synthesised from the hardware base-2 primitives rather than called, and the
normalisations use `NORM_EPS = 1e-6` to keep a zero-norm row finite.

### 4.4 Cross-lane reduction

A state row is spread across `WTK` lanes, so every dot product needs a reduction across that lane
group. It is done in two stages: a local balanced fold over the lane's own products, then an
**XOR butterfly** across the group — lane `l` exchanges with lane `l ^ off` for
`off = 1, 2, 4, … < WTK`, doubling the folded span each step.

XOR is chosen over a shift-down tree deliberately: the pattern is symmetric, so **every lane ends
holding the full sum**. That is what each lane needs — it must scale its own channels — so no
broadcast step is required afterwards. Offsets 1 and 2 lower to `quad_perm`, a lane-read modifier
on the arithmetic instruction itself; wider offsets use `ds_swizzle`. Neither allocates shared
memory, which is why the kernel has no LDS and no `lgkmcnt` barrier stalls on the narrow steps.

### 4.5 State pool addressing

One pool slot is `num_v_heads × head_v_dim × head_k_dim` elements. Indexing a deep pool as
`slot × slot_stride` in 32-bit signed arithmetic overflows once the pool is large enough, so the
base pointer is advanced by a **sign-extended 64-bit byte offset**; all indices within a slot remain
32-bit, since they are bounded by the slot stride.

The kernel bounds-checks nothing on device. The host `prepare()` therefore validates the state
shape and the index range (allowing the `-1` skip sentinel) against pool depth — a default-on,
hot-path-disableable check. This guard lives on the driver/`prepare()` path; as with the decay
guard in §8, dispatch selects a *spec*, not tensors, so a caller that launches the selected spec
without going through `prepare()` gets neither this host validation nor a device bounds-check. A
production launch path must call `prepare()`, or replicate its shape and index-range checks,
before launch.

### 4.6 Tile selection by batch

`(num_warps, warp_threads_k, blocks_per_v_dim)` is chosen from a batch-banded table (four bands:
`b4`, `b32`, `b128`, `b_large`). The trend it encodes: **as batch grows, `BPV` is spent down —
`8` at the smallest band to `1` by the `b128` band — because a larger natural grid needs less
manufactured parallelism; only at the largest band (`b_large`), where `BPV` is already `1`, is the
workgroup widened (to `num_warps = 8`) for throughput.** `num_warps` is therefore not monotone in
batch — the `b128` band is the narrowest workgroup.

The bands are deliberately coarse. Adjacent legal configurations sit within run-to-run variation of
each other, so a finer table would encode noise rather than signal. The table was produced by an
exhaustive sweep of the legal tile space, correctness-gated at every point, with the band edges
interpolated between measured anchors.

### 4.7 The reference path

A second, simpler emitter exists in which one thread owns an entire state row, making every dot
product thread-local and requiring no cross-lane traffic at all. It is register-heavy by
construction and is **not reachable through dispatch** — it is the correctness baseline for the
warp-tiled path, selected only by naming the spec directly.

### 4.8 Spec validation

`is_valid_spec(spec, arch)` rejects a configuration before any IR is built. It refuses an
unsupported activation or state dtype, `num_v_heads` not divisible by `num_k_heads`, a head
dimension that is not a multiple of `VPT = 8`, a workgroup over the target's thread limit, a
`wave_size` not divisible by `WTK`, a `DK` that is not a multiple of the warp's key tile
(`WTK × VPT`), a `BPV` that does not divide `DV`, and a resulting value tile that does not divide
across the workgroup's value lanes.

The dispatcher's support check ends by calling this same validator, so "the spec the kernel can
emit" and "the spec dispatch may select" are one rule rather than two copies that can drift.

---

## 5. Prefill kernel

### 5.1 Chunkwise factorization

GDN prefill is the KDA chunkwise kernel in `gate_kind="gdn"` mode, so the factorization is KDA's,
unchanged: see `../kda/ALGORITHM.md` for the six state-independent tiles (`A`, `GK`, `GQ`, `Aqk`,
`Kt`, `dec`), the chunk-parallel / state-serial split, and the midpoint-factored decay that keeps
`Γ_i / Γ_j` inside the `f32 exp2` range. Two conventions differ there: that doc states the
recurrence transposed (`S_kda = Sᵀ`, §5.3), and it carries the cumulative gate in the **log**
domain — its `Gc` / `Gref` are `log Γ` at the current and midpoint rows, which is why
`exp(Gc − Gref)` there reconstructs the ratio `Γ_i / Γ_j` here. (The `log2`/`exp2` the emitter
actually issues is a hardware detail: the cumulative sum is pre-scaled by `log₂(e)`.)

What GDN changes is only the gate's shape. The scalar per-`(token, head)` gate is broadcast across
all `DK` channels, so `Γ` is channel-constant and `dec` is a `DK` vector whose entries are equal.
Every tile above is indifferent to that — §2.2. The gate evaluation itself, the GQA gather and the
raw-input fusions are prep-side additions, listed in §2.3.

### 5.2 The triangular solve

`A` is produced by **blocked forward substitution**, not by forming an inverse. The right-hand side
`Diag(β)` is seeded into the output tile, so the substitution reads its starting value in place.
Each block step is two halves:

1. a **rank update** on MFMA, folding the already-solved columns into the remaining ones;
2. an **in-block substitution** that is genuinely serial, on the vector ALU, one lane per output
   column.

The per-block serial work scales as the *square* of the block size, but there are `C / solve_block`
blocks, so the total serial substitution work is **linear** in `solve_block` (`≈ C · solve_block / 2`):
a smaller block moves more of the cubic work onto the matrix unit at the cost of one more block
step; `solve_block` is the knob.

Two scheduling details follow from the solve being **wave-0 only**: the LDS hand-offs inside it need
no workgroup barrier, only explicit `lgkmcnt` waits, which is cheaper; and on the split/prep path
(`overlap_solve`, always set for GDN) the *other*, idle waves are given the `Kt` tile to build — it
depends only on `k` and the decay, none of the solve's live tiles. (The 256-thread fused path runs
the solve without this overlap.)

The solved block is written back transposed, in the operand order the next block's rank update
wants.

### 5.3 The state scan

`S` keeps the `DV × DK` orientation of §0 throughout — the device layout both kernels allocate
(`[pool, HV, DV, DK]` for decode, `[BH, DV, DK]` for the prefill scan). `kda/ALGORITHM.md` states
the same recurrence transposed (`S_kda = Sᵀ`, `DK × DV`); read across the two docs with that
mapping in mind. Per chunk, one value band of the state:

```
Z   = S GKᵀ                     EV × C
Rᵀ  = Vᵀ − Z                    EV × C   (in register)
Ṽᵀ  = Rᵀ Aᵀ                     EV × C
O   = GQ Sᵀ + Aqk Ṽ             C × EV
S  ← S Diag(dec) + Ṽᵀ Ktᵀ       EV × DK
```

Here `EV = DV / value_splits` is the band's value extent (§5.5), so a band of `S` is `EV × DK`.
`dec` is `DK`-wide, so `Diag(dec)` is `DK × DK` and multiplies the state from the **right**.

The remaining `ᵀ` superscripts mark operand orientation, not a re-layout: they keep every product
in `A Bᵀ` form with the contraction on the fastest axis, so no operand ever needs an LDS transpose.

Parallel structure inside a chunk: each wave owns one atom-sized band of `S` and the matching band
of value channels, so all five products are wave-local; the only cross-wave rendezvous are LDS
visibility barriers at phase boundaries. Across chunks the loop is serial, and **`S` is carried in
MFMA accumulator registers for the entire walk** — sequence length costs no additional registers.

The residual `Rᵀ` never needs an `f32` staging tile, and the output `O` is stored straight to HBM
because a slot's column index is already the lane's position in the atom's N extent.

### 5.4 Split and fused schedules

The same math, two packagings:

| | Split | Fused |
| --- | --- | --- |
| Kernels | two: prep, then scan | one |
| Prep grid | `BH × NC` — one workgroup per chunk | — |
| Scan grid | `BH × value_splits` | `BH` — one per `(batch, head)` |
| Tiles | written to HBM, read back by the scan | built and consumed in LDS, never reach HBM |

Fusing removes the tile round trip, which sounds strictly better and is not. Holding the tile
builder's staging *and* the scan's operands live at once puts the workgroup over half the LDS
budget, so only one fits per CU. The scan is a **latency-bound chain of small matmuls**; with one
workgroup per CU there is no second workgroup to cover its stalls. Paying tile traffic to keep two
resident is the cheaper side of that trade.

This is enforced, not merely asserted: the split scan's LDS request is validated against
`LDS_LIMIT / min_occupancy` — *half* the budget — and a spec that exceeds it is rejected with that
reasoning in the message.

The fused kernel additionally aliases its tiles onto the tile-phase staging buffers (which is why
`GK`/`GQ` are rebuilt after the `C × C` products have consumed their inputs — recomputing the
exponential is cheaper than the LDS the tiles would need), and can prefetch the next chunk's inputs
during the current chunk's scan. Those two are mutually exclusive: the prefetch writes the same
addresses the overlaid scan tiles occupy, and the validator rejects the combination.

**GDN currently runs the split path only.** The fused kernel is packed-input-only and cannot emit
the in-kernel GDN gate, so the dispatcher pins `chunk_prep` then `chunk_scan`. A fused GDN kernel is
a follow-up (§8).

### 5.5 value_splits

The scan's natural grid is `BH` workgroups, which starves at small `BH`. The value extent of `S` is
independent across rows, so it is banded into `value_splits` slices, each its own workgroup:

- grid becomes `BH × value_splits`; each band owns `ev_slice = DV / value_splits` rows;
- the state base and the `V`/`O` addresses are offset by the band, so bands do not overlap and need
  **no reduction** afterwards;
- the scan's LDS request shrinks with the split, which is what keeps a high split inside the
  occupancy budget;
- the cost is redundancy: the tiles are addressed by chunk only, so **every band re-reads the full
  tile set for every chunk**. `V` and `O` are banded and are not duplicated.

Legality is checked against the partitioning rule that one wave owns one atom-sized row band, so the
band must tile the waves exactly. `value_splits` is selected from a `BH`-banded table, and each
split fixes the scan tile geometry it requires. This is the prefill analogue of decode's `BPV`:
both manufacture workgroups when the natural grid is too small.

### 5.6 GDN-mode deltas, all in prep

Every GDN-specific branch listed in §2.3 lives in the prep kernel. **The scan body contains no GDN
branches at all** — it consumes tiles, and tiles from GDN mode are shaped exactly like tiles from
KDA mode. That containment is the reason the reuse is cheap and the reason existing KDA IR is
unaffected.

---

## 6. Reuse in the other direction: KDA on GDN

The piggy-back is **not symmetric**, and the asymmetry is worth stating precisely because it is easy
to assume otherwise.

**GDN prefill on KDA — free, and shipped.** General subsumes special: KDA's kernel has a slot for a
`DK`-wide decay, and a broadcast scalar is a legal occupant of that slot (§2.2).

**KDA decode on GDN — a real kernel change.** A GDN-only decode kernel has *one scalar slot* per
head; there is nowhere to put `DK` distinct decays. The general case cannot be expressed as an input
to the special machine.

What does transfer, and it is most of the kernel: KDA has **no decode kernel at all**, and the GDN
decode kernel is a ready chassis for one — the single-token structure, the paged state pool with
read/write indices and the negative-index skip, the batch-banded tile table, and the `BPV`
parallelism split. The change required is confined to the fade: `s = decay * S[row]` becomes an
element-wise multiply by a per-channel vector instead of a broadcast scalar. The probe, the delta,
the readout and the rank-1 write are untouched, as is the reduction structure.

On the prefill side, some machinery now benefits both families and some was inherited rather than
added for GDN. The chunkwise **raw-prep fused path** and the `value_splits` **knob**
(`KdaChunkScanSpec.value_splits`, `_RAW_VALUE_SPLITS = (1, 2, 4, 8)`) are pre-existing KDA work that
GDN mode reuses. What GDN added and now genuinely shares is the **hoisted dispatch core**
(`rocke.dispatch.core`, re-exported by the KDA dispatch). The **GQA gather** and the `value_splits`
**selection table** are GDN-only today — the gather is gated to `gate_kind="gdn"`, and KDA dispatch
builds its scan spec without a tuned `value_splits` table. Unlocking either for KDA is a scope
decision, not a rewrite.

---

## 7. Validation strategy

Both kernels are validated against **independent oracles**, not against each other:

- **Decode** — a token-serial `f32` reference, checking *both* the output and the written state
  pages, across the decode batch range, mixed state dtype, determinism, negative-index padding, and
  a deep-pool case that crosses the 32-bit offset boundary on device.
- **Prefill** — an `f64` oracle, checking output and final state across MHA and GQA head shapes, the
  gate range, and with and without an initial state.
- **IR stability** — GDN golden IR entries are pinned and SHA-stable across the supported lowerer
  flavours, while **all pre-existing KDA golden hashes remain unchanged**, which is what makes the
  "byte-identical" claim in §2.3 testable rather than asserted.
- **Dispatch** — CPU wiring tests assert the correct kernel and spec are selected for both
  operators, with GPU parity confirmed through the dispatcher.

---

## 8. Known limits and follow-ups

**Prefill decay range (documented, accepted).** The chunkwise stabilisation of §5.1 is sized for the
reference gate lower bound. GDN's gate is unbounded (§2.1), so a head whose per-token decay is
steeper than that bound exceeds the clamped `exp2` range, the midpoint factoring no longer
reconstructs the ratio, and that single steepest-decay head's output degrades. Trained GDN keeps the
product of `exp(A_log)` and the timestep small and stays well inside the envelope. This is an
accepted input contract: a helper flags an out-of-range workload on validation and driver paths, but
it is **not** a runtime production guard, because dispatch selects specs and not gate tensors.
Widening the range needs nested chunking or per-token rescaling.

**Follow-ups.**

1. A fused-path GDN prefill kernel (§5.4).
2. gfx942 support; the tuned tables are arch-specific and need re-sweeping.
3. Extending the supported decay range.
4. Scan-side parallelism beyond the current `value_splits` cap, or a shorter serial chain — the scan
   is the critical path at small `BH` (§5.4).
5. KDA decode on the GDN chassis, or a single decode kernel generalised over both gates (§6).
6. Host-struct consolidation of the GDN and KDA request lineage.
7. Machine-checked byte-identity for the cross-engine surfaces this family touches — currently
   reasoned and Python-verified.
