# GDN/KDA gated-delta decode and prefill — algorithm and design

> **Scope.** The shared gated-delta family on gfx950: a single-token **decode**
> emitter that serves GDN and KDA, plus chunkwise **prefill** kernels where GDN
> ships as a mode of the existing KDA implementation.
> This document specifies *what* the kernels compute and *why they are shaped the way they are*.
> It is a specification, not a tuning history, and it carries **no measurements** — latency is
> recorded in the internal perf repository per repository compliance.
>
> **This file is the entry point for the family.** Its sibling `README.md` covers the host-side
> tools — driver, benchmark, retuning, exit codes — and nothing about what the kernels compute;
> `CASE_STUDY.md` keeps the compiler-dependent per-tile register counts of the decode load order.

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
  - [4.6 Registry and tile selection](#46-registry-and-tile-selection)
  - [4.7 The reference path](#47-the-reference-path)
  - [4.8 Spec validation](#48-spec-validation)
  - [4.9 Optional fusions: conv1d and gated RMSNorm](#49-optional-fusions-conv1d-and-gated-rmsnorm)
  - [4.10 Performance knobs](#410-performance-knobs)
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
| `NW` | `num_warps`, waves per warp-tiled workgroup |
| `WTK` | `warp_threads_k`, lanes per warp assigned to the key reduction |
| `BPV` | `blocks_per_v_dim`, workgroups splitting one value head |
| `q̂`, `k̂` | L2-normalised query/key |
| `Γ_i` | cumulative in-chunk decay up to row `i`; `γ_C` is the whole-chunk decay |
| `EV` | per-band value extent, `DV / value_splits` — the scan's working V rows (§5.5) |
| `B` | decode batch: sequences in one launch |
| `pool` | depth of a decode state pool: slots addressed by `read_indices` / `write_indices` |
| `C_conv` | channels of the packed `[q \| k \| v]` row the fused conv reads, `2 × Hk × DK + Hv × DV` (`GdnDecodeSpec.conv_dim`, §4.9) |
| `W` | causal conv1d width of the fused conv, `CONV_WIDTH = 4`; a channel keeps `W − 1` history taps |
| `NORM_EPS` | compile-time `1e-6` of the `q`/`k` L2 norms; the fused output norm's eps is the separate runtime `norm_eps` |
| `x · y` | dot product of two vectors (a scalar); integer extents and other scalar products use `×` |
| `x ⊙ y` | elementwise product; a scalar operand broadcasts (written `*` inside code blocks) |
| `r` | the fused output norm's scale of one head, `rsqrt(mean(o ⊙ o) + norm_eps)` over its `DV` outputs `o` (§4.9) |
| `OUT_W` | rows one `out_lds` storing lane normalizes and stores with one packed store: 2, 4 or 8 (§4.10) |

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

The layer neighbours that §4.9 can fuse are pinned the same way. Widths and eps come from
`config.json`; the conv layout and the gate activation come from each model's published modeling
code (`modeling_qwen3_next.py` in Hugging Face Transformers; `modeling_kimi.py` in the
Kimi-Linear checkpoint repository):

| Model | conv width `W` | conv state | output-norm gate | output-norm eps |
| --- | --- | --- | --- | --- |
| Qwen3-Next | 4 (`linear_conv_kernel_dim`) | ONE depthwise conv over the packed `[q \| k \| v]` row, `C_conv = 2 × Hk × DK + Hv × DV` channels | `SiLU` (`Qwen3NextRMSNormGated`) | `1e-6` (`rms_norm_eps`) |
| Kimi Linear | 4 (`short_conv_kernel_size`) | THREE separate convs (`q_conv1d`, `k_conv1d`, `v_conv1d`) with three states | `sigmoid` (`FusedRMSNormGated(activation='sigmoid')`) | `1e-5` (`rms_norm_eps`) |

So `fuse_out_norm` derives its gate activation from `gate_kind` (`OUT_GATE_ACTIVATION`: GDN →
SiLU, KDA → sigmoid), and the eps is a runtime argument because the two models differ. Kimi
Linear's three conv states are one packed `[q | k | v]` state to `fuse_conv`: a host that serves
Kimi Linear through it lays its three states (and weights) out as one `[pool, C_conv, W − 1]`
tensor. That packing is an assumption of this kernel, not of the model.

> **Out of scope: Qwen3-Next's other 12 layers.** The same model interleaves gated *full*
> attention every fourth layer (`full_attention_interval = 4`), and those layers are
> `head_dim = 256` with `num_attention_heads = 16` / `num_key_value_heads = 2` — no recurrent
> state, so no value of `kv_group` makes them servable here. They belong to the attention
> family, not this one.

Three things the table pins down that the rest of the document assumes:

- **`DK == DV` on every supported row.** The tile geometry and the `DV × DK` state shape both rely
  on it. A target with `DK ≠ DV` needs the tile geometry re-derived.
- **`kv_group = 2` is the only shipping GDN grouping.** The `(Hv, Hk) = (32, 8)` case in §7 —
  `kv_group = 4` — is a validation stress point, not a deployment.
- **Where these models land in the selection policies.** Both have `Hv = 32`.
  Prefill's `value_splits` bands on `BH = 32 × batch` (`≤64 → 8`, `≤128 → 2`,
  else `1`), so batches 1-2 get 8 splits, batches 3-4 get 2, and larger
  batches run unsplit. GDN decode instead enumerates validator-approved
  registry tiles and uses the documented static `(2, 16, 8)` priority for the
  supported D128 deployment; batch changes the grid, not its tile.

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

## 4. Decode kernel shared by GDN and KDA

### 4.1 Tensor contract

Every tensor below is contiguous row-major (the fused-mode rows of §4.9 are the one exception:
they may be strided row slices). `gate_kind` changes the extent and type of the gate inputs, not
the recurrent-state layout or the rest of the ABI:

| Tensor | GDN shape/type | KDA shape/type | Direction |
| --- | --- | --- | --- |
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]`, `dtype` | same | in |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]`, `dtype` | same | in / out |
| `a` | `[B, 1, num_v_heads]`, `dtype` | `[B, 1, num_v_heads, head_k_dim]`, `dtype` | in |
| `b` | `[B, 1, num_v_heads]`, `dtype` | same | in |
| `dt_bias` | `[num_v_heads]`, `dtype` | `[num_v_heads, head_k_dim]`, `f32` | in |
| `A_log` | `[num_v_heads]`, `f32` | same | in |
| `read_indices`, `write_indices` | `[B]`, `i32` | same | in |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]`, `state_dtype` | same | in-place |

`state_dtype` is `bf16`, `f16` or `f32`; the I/O `dtype` is `bf16` or `f16`. The kernel
updates the state in f32 registers either way, so `f32` only widens the state loads and
stores (two 16-byte accesses per 8 elements instead of one). With a 2-byte state the
emitted code is exactly what it was before `f32` was admitted.

The launch also passes a trailing `batch_size` `i32` scalar (not a tensor).
`prepare()` validates the gate-kind-dependent shapes, dtypes, devices and
contiguity before launch, in addition to the state-pool checks in §4.5. It also
checks every pointer's base address against the alignment the emitted vector
accesses assume (`gdn_decode_pointer_alignment`: 16 B for the vector-loaded
tensors, 32 B for KDA's f32 `dt_bias`), because nothing on the device does.

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
| across k-lanes | `WTK` | `WTK` lanes cover a row, `VPT = 8` channels each (knob-free contiguous; `interleave_cols` splits them into 16 B runs, §4.10), repeated `WTK_ITERS = DK / (WTK × VPT)` times |

Live state per lane is `WTV_ITERS × WTK_ITERS × VPT` values, held **in registers**. The design is
deliberately register-resident: in the unfused mode the kernel allocates **no LDS and issues no
barriers**. The fused-norm mode of §4.9 adds, when `NW > 1`, one `NW`-float LDS reduction and its
barrier; fused conv alone adds one barrier when `NW > 1`. Knob-free, with `NW = 1` neither adds
either; `conv_once` and `out_lds` (§4.10) add LDS blocks and, even at `NW = 1`, an LDS barrier.

`BPV` is a parallelism-manufacturing knob, not a work-reducing one — each of the `BPV` workgroups
re-loads `q` and `k` and re-runs the normalisation reductions. It buys occupancy at small batch and
is retired at large batch, where the grid is already ample.

### 4.3 Dataflow and pipeline

One workgroup, one decode step, in emission order (warp-tiled path):

1. decode `bidx` and `tid` into `(sequence, value head, v-sub-block)` and `(wave, k-lane, v-lane)`;
2. load `read_indices` / `write_indices` and form the `active` predicate — **outside** the guard, so
   a padded lane costs two `i32` loads and exits;
3. under `scf_if(active)`, issue **every** global load before any math: the gate inputs (GDN: the
   `a`, `b`, `dt_bias` and `A_log` scalars; KDA: `b` and `A_log` plus the lane's per-channel `a`
   slice and `dt_bias` vector), the lane's `q` and `k` slices as 16-byte vectors, the raw lane tile of
   the state, and one `v` per owned V row — all promoted to `f32`;
4. evaluate `decay` and `β`;
5. reduce the two L2 norms (two cross-lane reductions, §4.4);
6. reduce `dot(k̂, q̂)` (one more);
7. apply `decay` to the raw state tile;
8. per owned V row: reduce `s·k̂` and `s·q̂`, form `v_new = β (v − s·k̂)`, then emit the output and
   the rank-1 state update.

Why loads first: the AMDGPU scheduler does not hoist a load above earlier math or across the
`k_lane == 0` output-store branch, so emission order bounds how many separate load batches (each
ended by a `vmcnt` wait) a wave exposes. In the gfx950 code object, loads first brings the GDN
default tile and the KDA tuned tiles down to one or two batches. Every tile with `WTK ≤ 2`, most
with `WTK = 4`, and the fused tiles still split their loads into several batches: under register
pressure the compiler re-interleaves some loads with the math. These are static counts from the
code object, not hardware counters.

What moved: only loads, plus (with `fuse_conv`) the `v` conv + SiLU, which now runs next to the
`q`/`k` conv before the L2 norms instead of inside the step-8 row loop. Each value still goes
through the same floating-point operations in the same order; no operation was added, removed or
re-associated. The A/B against the interleaved order found output, state and conv state
bit-identical in every configuration it ran.

One exception: with `fuse_conv` and a single wave, one lane owns many V rows (32 of the 128 at the
`(1,16,1)` tile). Hoisting their `v` conv inputs (value, `W − 1` taps, `W` weights), the state
tile and the norm inputs on top of the `q`/`k` conv inputs would need more than the 512 VGPRs a
lane can have at `(1,16,1)`, so it would spill; at `WTK = 8` only bf16 state without the norm
would fit. The exception covers every single-wave `fuse_conv` tile except where `conv_once`
narrows it (below). Those tiles load these inputs where the parent order did: each row's `v`
inputs and its conv in the step-8 row loop, each state chunk at step 7, and `out_gate` /
`norm_weight` at the norm. Their `q`, `k` and gate inputs still load first. `late_row_inputs` in
the emitter holds the rule.

With `conv_once` (§4.10) the exception is narrower. The `v` channels then come from LDS, so only
the state chunks and the per-lane norm inputs would stay late. Late, each state chunk's load is
emitted behind the preceding math, so its wait cannot overlap the other chunks' loads. At
`DK = DV = 128` with `WTK` 4 or 8 and no `waves_per_eu` floor, hoisting them spills no more than
the late order in the gfx950 code object, so those tiles load first like every other tile.
`test_one_wave_conv_once_loads_first_without_a_new_spill` pins that for both gate kinds, bf16 and
`f32` state, conv alone and with the norm (per-lane, and with `out_lds` and `norm_gate_once`), each
with every other knob the tile accepts. Every other single-wave `fuse_conv` tile keeps the late
order: at `WTK` 1, 2 and 16, at `DK = DV = 256`, under a `waves_per_eu` floor and at the other head
dims, no test shows hoisting spill-free.

Multi-wave `fuse_conv` tiles hoist every load. Their per-lane cap is `512 / ceil(NW / 4)` VGPRs, so
tiles with a small `WTK` (wide `q`/`k` slices with their taps and weights) or many waves can still
spill; `CASE_STUDY.md` lists the ones that do, per compiler.

Cost: holding the raw state, `q`, `k` and `v` (and, fused, their taps and weights) live at once
changes the register count, up for some tiles and down for others. Where that crosses a
register-allocation granule a tile gains or loses waves per SIMD, or starts, stops or changes a
spill. Which tiles move, and in which direction, depends on the compiler: the same tile can gain
waves under one comgr and lose them under another. No dispatcher candidate starts to spill; some
multi-wave `fuse_conv` tiles do, and `test_lowers_and_compiles_without_scratch` holds at 0 B the
fused cases that every measured compiler keeps there. `CASE_STUDY.md` has the per-tile counts for
each measured compiler and the command that regenerates them; re-run it when the compiler or the
lowering changes.

Tile defaults (§4.6) were measured on the interleaved order and are re-measured when the defaults
are re-picked.

The simple path (`spec.simple`) keeps the interleaved order: q/k, norms, gates, dot, then the state
row.

Two consequences of the identity in §1.2 item 3: the output store and the state write in step 8 are
**independent** — neither reads the other's result — and the output is broadcast across the k-lane
group, so only lane 0 of each group stores it (except with `out_lds`, §4.10: there the first
`DV / OUT_W` lanes store packed row groups).

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
on the arithmetic instruction itself; wider offsets use `ds_swizzle` by default. With the
`dpp_reduce` knob (§4.10), offsets 4 and 8 use `tile.dpp_xor` with masks 7 and 15 instead, which
lower to the DPP (data-parallel primitives: a modifier that lets a VALU instruction read another
lane's register; `quad_perm` is one) `row_half_mirror` / `row_mirror` modifiers on gfx9 CDNA, so a
group of up to 16 lanes reduces without `ds_swizzle`; offsets 16 and 32 keep it. Those masks pair
lane `i` with `i ^ 7` (`i ^ 15`), not `i ^ 4` (`i ^ 8`), but the stages run in ascending order:
when one runs, every lane of a quad (half-row) already holds that quad's (half-row's) sum, so both
pairings add the same two values. None of these allocates shared memory, which is why the unfused
kernel has no LDS and no `lgkmcnt` barrier stalls on the narrow steps.

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

Slot aliasing has one more rule that the host does not check: a sequence may write the slot it
read, but not a slot that another active sequence reads in the same launch. Those are different
workgroups, and nothing orders one's write after the other's read; the same holds for the fused
conv-state pool of §4.9, which uses the same indices.

### 4.6 Registry and tile selection

GDN exposes the Cartesian product of:

- `num_warps ∈ {1, 2, 4, 8, 16}`;
- `warp_threads_k ∈ {1, 2, 4, 8, 16, 32}`;
- `blocks_per_v_dim ∈ {1, 2, 4, 8, 16, 32}`.

This produces 180 stable identities. `is_valid_spec()` is the only legality
authority and admits 54 GDN candidates for the default D128 shape. Production
`auto` deterministically prefers `(2, 16, 8)` whenever legal; batch changes
grid size, not GDN tile selection. A caller may pin an exact candidate with
`nw<num_warps>_wtk<warp_threads_k>_bpv<blocks_per_v_dim>`.


KDA remains keyed on `work = batch × num_v_heads`. Tensor-parallel sharding
changes `num_v_heads` per rank, so two launches with the same batch can expose
different amounts of GPU work:

| Band | Work | `(num_warps, warp_threads_k, blocks_per_v_dim)` |
| --- | --- | --- |
| `kda_w128` | `≤ 128` | `(4, 16, 4)` |
| `kda_w512` | `≤ 512` | `(1, 16, 4)` |
| `kda_w_large` | larger | `(2, 16, 1)` |

`BPV` manufactures workgroups when the natural grid is too small. KDA's table
comes from exhaustive legal-tile sweeps with every candidate correctness-gated
before timing. Its band edges interpolate measured anchors; exact measurements
live in the protected performance record. Both this table and the GDN default
were measured on the interleaved load order, before §4.3's loads-first order,
so they are a snapshot until the sweep is re-run.

The request also carries the state width and the fused modes (§4.9). An `f32` state
(`state_dtype` `f32`, for either gate kind) reaches the spec and the kernel name but not the tile
choice: `auto` picks what it picks for the same request with a 16-bit state. Fused requests
(`fuse_conv` and/or `fuse_out_norm`) admit only `BPV = 1` tiles. Without a fused default, `auto`
would serve a fused GDN request with the first legal tile in product order, `(1, 1, 1)`, because
GDN's default `(2, 16, 8)` is not `BPV = 1`, and every fused KDA request with its largest band,
`(2, 16, 1)`. `auto` therefore prefers one interim static fused tile, `(4, 16, 1)`, for both gate
kinds and every state width. It is the tile the fused numeric suite is built around, not a
measured winner; per-mode fused defaults are a separate measurement. It needs `DK` a multiple of
`WTK · VPT = 128` and `DV` a multiple of 16, which both §1.4 deployments meet. If it is illegal,
`auto` falls back to the first legal candidate; for example, fused GDN at `DK = 64` falls back to
`(1, 1, 1)`, and at `DV = 8` to `(1, 8, 1)`. KDA registers the tile as
one extra candidate (`kda_nw4_wtk16_bpv1`) outside its work table, at the lowest priority. The
fused flags must be real bools and the dims real ints in the request, and `fuse_conv` with
`num_k_heads ≠ num_v_heads` is a request error (§4.9).

### 4.7 The reference path

A second, simpler emitter exists in which one thread owns an entire state row, making every dot
product thread-local and requiring no cross-lane traffic at all. It is register-heavy by
construction and is **not reachable through dispatch** — it is the correctness baseline for the
warp-tiled path, selected only by naming the spec directly.

### 4.8 Spec validation

`is_valid_spec(spec, arch)` rejects a configuration before any IR is built. It first refuses a
field of the wrong type — a flag that is not a real `bool` (`fuse_conv="false"` is truthy and
would select the fused ABI), or a size that is not an integer — so no rule below can coerce a
malformed spec into a different valid one. It then refuses an
unsupported activation or state dtype, `num_v_heads` not divisible by `num_k_heads`, a head
dimension that is not a multiple of `VPT = 8`, a workgroup over the target's thread limit, a
`wave_size` not divisible by `WTK`, a `DK` that is not a multiple of the warp's key tile
(`WTK × VPT`), a `BPV` that does not divide `DV`, and a resulting value tile that does not divide
across the workgroup's value lanes. The three §4.9 fusion rules are emitter limits, not hardware
ones, and carry the family's `NOT_YET_IMPLEMENTED:` marker.

It also checks the performance knobs of §4.10: each flag must be a real `bool`, each state hint
one of `default` / `streaming`, `waves_per_eu` an `int` in `[0, 8]`, and a fused-only knob needs
its fusion (`conv_once` needs `fuse_conv`; `norm_gate_once` and `out_lds` need `fuse_out_norm`).
`norm_gate_once` without `conv_once` is an emitter limit, rejected as `NOT_YET_IMPLEMENTED:`: its
products live in `conv_once`'s LDS block (§8, follow-up 8).

The dispatcher's support check ends by calling this same validator, so "the spec the kernel can
emit" and "the spec dispatch may select" are one rule rather than two copies that can drift.

### 4.9 Optional fusions: conv1d and gated RMSNorm

A hybrid model runs two neighbours around this decode step: a causal conv1d (width `W = 4`) +
SiLU on `q`, `k` and `v` before it, and a gated RMSNorm on its output after it (§1.4 pins both
per model). Two spec flags fuse them into the warp-tiled kernel, independently:

| Flag | Computes, per channel `c` / per head | Extra arguments |
| --- | --- | --- |
| `fuse_conv` | `x'_c = silu(h0_c w0_c + h1_c w1_c + h2_c w2_c + x_c w3_c)` (scalars; `h0` oldest); the taps shift in place to `(h1, h2, x)` | `mixed_qkv [B, ≥ C_conv]`, rows `qkv_stride` elements apart (replaces `query`/`key`/`value`); `conv_state [pool, C_conv, W − 1]` (I/O dtype, same read/write slots as the recurrent state); `conv_weight [C_conv, W]` f32 |
| `fuse_out_norm` | `y = o ⊙ rsqrt(mean(o ⊙ o) + norm_eps) ⊙ norm_weight ⊙ act(out_gate)` over the head's `DV` outputs, `act` = SiLU for GDN, sigmoid for KDA | `out_gate [B, ≥ Hv × DV]`, rows `og_stride` elements apart; `norm_weight [DV]` f32, shared by every head; runtime `norm_eps` |

Each row of `mixed_qkv` / `out_gate` must be contiguous; the rows may be slices of a wider
buffer, so only the first `C_conv` / `Hv × DV` elements of a row are read. `qkv_stride` must keep
every row on a 16-byte boundary, since `q`/`k` chunks are vector loads.

Both flags off emit exactly the unfused kernel: the name gains `_cv` / `_rn` only when a flag is
on, and every pre-existing golden IR hash is unchanged. The flags compose with either gate kind,
either I/O dtype and any state dtype.

**One workgroup per head (not yet implemented beyond).** Three rules, all limits of this emitter's
in-place, one-workgroup-per-head design, rejected as `NOT_YET_IMPLEMENTED` (§8, follow-up 7): both
flags require the warp-tiled path and `blocks_per_v_dim == 1`, and `fuse_conv` also requires
`Hk == Hv`. The norm needs all `DV` outputs of a head in one workgroup, and with `BPV > 1` several
workgroups would read the q/k conv taps while one shifts them in place. With `Hv > Hk` the `Hv/Hk`
workgroups of one k-head share the q/k conv channels, and shifting them in place would race across
workgroups — a workgroup barrier cannot order that. For `fuse_conv`, an out-of-place conv state
(read one slot, write another) removes both races and is the planned way to lift its `BPV` and
`Hk == Hv` rules. `fuse_out_norm` with `BPV > 1` needs a cross-workgroup norm instead: a reduction
of `Σ o²` across the head's workgroups. Until then a GQA model (Qwen3-Next) fuses only the norm and
keeps a separate conv kernel; a model with `Hk == Hv` (Kimi Linear, with its three conv states
packed as §1.4 states) can fuse both.

**Dataflow.** The fused loads join step 3 of §4.3; the unfused math steps keep their order. With
`fuse_conv`, step 3 loads each `q`/`k` chunk from the packed row together with its `W − 1` taps
and `W` weights, and each owned V row's `v` with its taps and weights; conv + SiLU then run on the
raw `q`, `k` and `v` after the gates and before the L2 norms. The raw (pre-conv) values are kept,
since they become the newest tap. With the norm on, step 3 also loads each owned row's `out_gate`
and `norm_weight`, and step 8 holds each row's output instead of storing it; afterwards each wave
reduces its `Σ o²` (k-lane 0 only, since every k-lane holds the row's output). When `NW > 1` each
wave writes one float to LDS and a workgroup barrier separates those writes from the reads that
finish the sum; the gated store follows. That barrier comes after every wave's tap reads (taps →
outputs → partial sums), so it also orders the tap shift; with conv on and the norm off, one
barrier is emitted for that alone when `NW > 1`. Knob-free, with `NW = 1` there is no LDS and no
barrier: one wave reads its taps before it writes them. Knob-free, each conv channel is written
once: q/k channels by wave 0 v-lane 0, a V row's channel by its k-lane-0 owner (with `conv_once`,
§4.10, the one thread that computes a channel writes its taps). The norm runs on the fp32 output (a
reference implementation that rounds `o` to bf16 first differs by up to one bf16 step). The knobs
change two of these loads (§4.10): `conv_once` replaces the per-lane conv loads with its
prologue's per-slot loads, and `out_lds` / `norm_gate_once` change which threads load the norm's
`out_gate` and `norm_weight`.

**Cost.** `fuse_conv` with a single wave owns all `DV` rows plus their taps and weights, so it is a
register-heavy (legal) tile, like the unfused `(1,1,1)`; it keeps part of the interleaved order to
limit spills (§4.3), while multi-wave conv tiles hoist every load; some of them start to spill or
run fewer waves per SIMD than with the interleaved order, and which ones depends on the compiler
(`CASE_STUDY.md`). Without `conv_once`, the fused
modes are an enablement path and recompute shared work instead of sharing it. A q/k conv channel
depends only on its k-lane, so every `(wave, v-lane)` pair of the workgroup recomputes it:
`NW × WTV = NW × 64 / WTK` times per head, each time re-loading its taps and weights (16 times at
the `(4, 16, 1)` tile). A V channel's conv, and (without `out_lds` or `norm_gate_once`) an output's
gate activation, are computed by all `WTK` k-lanes of the row and stored by one: `WTK`-fold. This
trades occupancy and redundant loads for fewer launches. With `conv_once` a channel is computed
once per workgroup (one thread instead of `NW × WTV`); `norm_gate_once` does the same for the gate
product. Tile selection for the fused modes is dispatch's (§4.6), not this emitter's.

### 4.10 Performance knobs

Ten spec fields change how the warp-tiled kernel computes, not what it computes (the two state
hints and `waves_per_eu` also apply on the reference path). Each one's default is the code the
kernel emitted before the knob existed, so with every knob at its default the kernel name and the
emitted IR are byte-identical to the knob-free emitter (the golden IR test pins
this). Dispatch does not set the knobs yet: the registry search space and per-mode defaults are
separate changes, so today a knob is reached only by building a spec directly.

| Knob | Values (default first) | Changes code where | Effect |
| --- | --- | --- | --- |
| `state_load_hint`, `state_store_hint` | `default`, `streaming` | always, both emitters | cache policy of the state loads / stores; `streaming` marks them LLVM `!nontemporal` (the state is read once and written once per call) |
| `dpp_reduce` | off, on | warp-tiled, `WTK ≥ 8` (the sum has an xor-4 stage) | the xor-4 / xor-8 stages of the in-group sum (§4.4) run on `tile.dpp_xor` with mask 7 / 15 (DPP `row_half_mirror` / `row_mirror` on gfx9 CDNA) instead of `ds_swizzle` |
| `xcd_remap` | off, on | warp-tiled | workgroups are renumbered so each XCD (accelerator complex die, one of the 8 XCDs of gfx950; the XCDs receive workgroup `pid` round-robin, `pid % 8`) runs one contiguous range of (sequence, head, v-block) work instead of every 8th workgroup; a bijection for any grid size. It computes the grid size from the `batch_size` argument (otherwise unread), so the launch grid must be `gdn_decode_grid(batch_size, spec)`, as `prepare()` builds it |
| `stream_rows` | off, on | warp-tiled, more than one state row per lane | each owned row is decayed, reduced, updated and stored in its own iteration, so the first math waits only for the rows loaded before it; off, every row is decayed up front |
| `interleave_cols` | off, on | warp-tiled, `f32` state, `WTK > 1` | K columns map to lanes in 16 B runs interleaved across the WTK-lane group (run `p` of lane `l` at chunk column `p × WTK × 4 + l × 4`) instead of 8 consecutive columns per lane, so one 16 B state access across the group covers a contiguous `WTK × 16` B span; q, k, the KDA decays and the conv taps / weights follow the same map. A 2-byte state already moves 16 B per lane |
| `conv_once` | off, on | `fuse_conv` (required) | each conv1d + SiLU channel, and each KDA decay channel, is computed once per workgroup by one thread and shared through LDS, instead of every lane recomputing the channels it reads; that thread also shifts the channel's taps, so no barrier guards the in-place shift |
| `norm_gate_once` | off, on | `fuse_out_norm` (required); `conv_once` (not yet implemented without it) | the norm's `act(out_gate) ⊙ norm_weight` (§4.9) is computed once per output row into `conv_once`'s LDS block; off, every lane loads its rows' `out_gate` and `norm_weight` |
| `out_lds` | off, on | `fuse_out_norm` (required) | each row's pre-norm f32 output goes to LDS ahead of the norm's cross-wave barrier (a one-wave tile adds an LDS barrier); after it the first `DV / OUT_W` lanes each normalize `OUT_W` consecutive rows (`OUT_W` = 2, 4 or 8 as `DV` exceeds 2 and 4 wave widths; in more than one round when `DV / OUT_W` exceeds the workgroup size) and store them with one packed store, instead of one 2 B store per row from its k-lane-0 owner |
| `waves_per_eu` | `0`, `1`–`8` | always, both emitters | `amdgpu-waves-per-eu` = `"N,8"`: the compiler must fit at least `N` waves per SIMD, which caps registers per lane; `0` emits no attribute. The ceiling 8 is the hardware limit: a gfx950 SIMD holds at most 8 waves |

**Inert knobs and names.** The kernel name is the compile and launcher cache key (the comment in
`GdnDecodeSpec.kernel_name` states the rule), so a knob adds its name token only where it changes
code. For the four tile-dependent knobs that is what `dpp_reduce_applies`, `xcd_remap_applies`,
`stream_rows_applies` and `interleave_cols_applies` decide: on a tile where the predicate is False
the knob is legal, emits the knob-off kernel, and keeps the knob-off name. The three fused-only
knobs need no predicate, because the validator rejects them wherever they would be inert. The
tokens are `lhnt` / `shnt` (streaming hints), `wpe<N>`, `dppr`, `xcd`, `sr`, `ic`, `co`, `ngo` and
`olds`.

**Loads first.** The knobs keep §4.3's order. `conv_once` issues its channel loads (with indices
clamped into range, so no load sits behind a branch) ahead of the state loads, and `out_lds`
issues its storing lanes' gate and weight loads with the other loads; the structural load-order
test covers knob-on specs too. The single-wave `fuse_conv` tile keeps §4.3's exception except where
`late_row_inputs` lets it load first (`conv_once` at `DK = DV = 128`, `WTK` 4 or 8, no
`waves_per_eu` floor); elsewhere its state chunks (and, for the per-lane norm, `out_gate` /
`norm_weight`) still load where they are first used.

**Numerics.** `dpp_reduce`, `xcd_remap`, `stream_rows`, `out_lds`, the hints and `waves_per_eu`
run the same floating-point operations on the same values (for `dpp_reduce`, the xor-7 / xor-15
partner holds the same quad or half-row sum as the xor-4 / xor-8 partner). `conv_once` computes
each channel with the same operations as the per-lane path. Two knobs change a rounding order:
`interleave_cols` gives each lane different K columns, so its partial dot products sum in a
different order, and `norm_gate_once` computes `(o ⊙ r) ⊙ (act(out_gate) ⊙ norm_weight)` instead of
`(o ⊙ r ⊙ norm_weight) ⊙ act(out_gate)`, where `r = rsqrt(mean(o ⊙ o) + norm_eps)` (§4.9). On
device every knob is checked against the fp32 reference, and with a bf16 or `f32` state every
knob except these two is also compared with the knob-off kernel's output, state and conv state
(§7 lists the tiles). With bf16 output that comparison is bit for bit. With f16 output it is bit
for bit except `out_lds`'s output, which may differ by one f16 ulp (unit in the last place): the
IR rounds the same way, but the backend folds the per-lane path's last multiply and its f16
conversion into one `v_fma_mixlo_f16` (one rounding), while `out_lds` packs with
`v_cvt_pk_f16_f32` after an f32 multiply (two roundings). This was observed in the code object
with ROCm 7.13 comgr and is not pinned by a test; `kernels/common/_lse_store.py` documents the
same fold.

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
blocks, so the total serial substitution work is **linear** in `solve_block` (`≈ C × solve_block / 2`):
a smaller block moves more of the cubic work onto the matrix unit at the cost of one more block
step; `solve_block` is the knob.

Two scheduling details follow from the solve being **wave-0 only**: the LDS hand-offs inside it need
no workgroup barrier, only explicit `lgkmcnt` waits, which is cheaper; and on the split/prep path
(`overlap_solve`, always set for GDN) the *other*, idle waves are given the `Kt` tile to build — it
depends only on `k` and the decay, none of the solve's live tiles. (The 256-thread fused path runs
the solve without this overlap.)

This is enforced in `kernels/gfx950/kda_chunkwise.py`: the block loop sits inside a `tid < 64`
`scf_if`, the barrier/`lgkmcnt` rationale is in the comment immediately above it, and the
idle-wave `Kt` work is in `_emit_idle_during_solve`. Follow-up 4 in §8 — a shorter serial
chain — would have to change this constraint.

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

- grid becomes `BH × value_splits`; each band owns `EV = DV / value_splits` rows (§0);
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

## 6. Reuse in both directions

**GDN prefill on KDA — shared and shipped.** General subsumes special: KDA's
prefill kernel has a slot for a `DK`-wide decay, and a broadcast scalar is a
legal occupant of that slot (§2.2).

**KDA decode on GDN — shared and shipped.** This direction required extending
the decode emitter: its original GDN gate produced one scalar decay per head,
while KDA needs `DK` distinct per-channel decays. `GdnDecodeSpec.gate_kind`
selects the gate at build time:

```text
gdn: log_decay[h]   = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
kda: log_decay[h,d] = lower_bound * sigmoid(exp(A_log[h]) * (a[h,d] + dt_bias[h,d]))
decay               = exp(log_decay)
```

Only gate production and the fade differ. The probe, delta, readout, rank-1
write, cross-lane reduction, paged state pool, negative-index skip and
`blocks_per_v_dim` split are the same emitter code. In the warp-tiled path a
lane loads the decay for its K-channel slice once and reuses it for every state
row it owns.

`fuse_gate=True` is the dispatched production path. `fuse_gate=False` accepts
precomputed natural-log decay and is validated but not dispatched; it exposes
recurrence-only cost for controlled measurement.

On the prefill side, some machinery benefits both families and some was
inherited rather than added for GDN. The chunkwise **raw-prep fused path** and
the `value_splits` **knob** (`KdaChunkScanSpec.value_splits`,
`_RAW_VALUE_SPLITS = (1, 2, 4, 8)`) are pre-existing KDA work that GDN mode
reuses. What GDN added and now genuinely shares is the **hoisted dispatch
core** (`rocke.dispatch.core`, re-exported by the KDA dispatch). The **GQA
gather** and the `value_splits` **selection table** remain GDN-prefill-only.

---

## 7. Validation strategy

Both kernels are validated against **independent oracles**, not against each other:

- **Decode** — a token-serial `f32` reference, checking *both* the output and the written state
  pages, across the decode batch range, mixed state dtype, determinism, negative-index padding, and
  a deep-pool case that crosses the 32-bit offset boundary on device.
- **Fused decode (§4.9)** — the same reference with the conv1d + SiLU applied to the packed row
  first and the gated RMSNorm (SiLU for GDN, sigmoid for KDA, written from the models rather than
  read from the kernel) applied to the `f32` output last. The written conv-state slots are checked
  against the shifted taps, and every untouched slot of both pools against its pre-launch
  contents. Under `fuse_out_norm` the output is RMS-normalised to `|o|` of a few units, so one bf16
  rounding step of an exact answer exceeds the absolute tolerance; the output error is then taken
  relative to `max(1, |ref|)`, which equals the absolute error wherever `|ref| ≤ 1`. CPU tests show
  that metric still rejects a dropped gate and a norm over the wrong length. The barrier that orders
  the in-place tap shift is checked twice: on silicon by a read-slot = write-slot case with 16 waves
  (with the barrier removed, 2-8 waves stayed correct but 16 waves corrupted the taps on every run),
  and in the IR, where a barrier must sit between the last tap read and the first tap write.
- **Performance knobs (§4.10)** — each knob alone, against the same reference, on gfx950: at the
  default heads (`DK = DV = 128`) on every registered tile and the reference path, and in every
  fused mode it accepts on every `BPV = 1` tile, per gate kind and state dtype (bf16, `f32`); at
  `DK = DV = 64` and `256` on one warp-tiled and one fused tile; and `out_lds` at `DV = 256` and
  `512` and with two store rounds. Those use bf16 I/O. With f16 I/O and a bf16 or `f32` state,
  each knob runs on the default tile, the reference path and the fused `(4,16,1)` tile in every
  fusion mode it accepts; an f16 state runs the same specs. The knobs that keep the
  floating-point order are also compared with the knob-off kernel: bit for bit with bf16 output,
  and with f16 output bit for bit except `out_lds`'s output, which may differ by one f16 ulp
  because the backend folds the per-lane path's last multiply and f16 conversion into
  `v_fma_mixlo_f16` (§4.10, Numerics; observed with ROCm 7.13 comgr, not pinned by a test). An
  f16 state is checked against the reference only and not compared bit for bit: whether the
  knob-on and knob-off kernels round an f16 state the same way has not been established. bf16
  I/O with an f16 state and the knobs on is not separately covered. Each knob's flag-on IR is
  golden-pinned on every path where it changes code: gate kind × state dtype × reference /
  warp-tiled / fused (both fusions, and every mode a fused-only knob accepts), and at the other
  head dims.
- **IR stability** — GDN golden IR entries are pinned and SHA-stable across the supported lowerer
  flavours, while **all pre-existing KDA golden hashes remain unchanged**, which is what makes the
  "byte-identical" claim in §2.3 testable rather than asserted. The golden lowers through the
  Python engine only; `test_gdn_decode_ir_cpp_parity.py` lowers the same cases through the C++
  engine and requires byte-identical IR.
- **Prefill** — an `f64` oracle, checking output and final state across head shapes
  `(Hv, Hk) = (4, 4)` (MHA), `(8, 4)` (`kv_group = 2`, the shipping grouping) and `(32, 8)`
  (`kv_group = 4`, a stress point above any deployed config), the gate range, and with and
  without an initial state.
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
2. gfx942 support; KDA work bands are arch-specific and need re-sweeping.
3. Extending the supported decay range.
4. Scan-side parallelism beyond the current `value_splits` cap, or a shorter serial chain — the scan
   is the critical path at small `BH` (§5.4).
5. Host-struct consolidation of the GDN and KDA request lineage.
6. Machine-checked byte-identity for the prefill-side cross-engine surfaces this family touches —
   currently reasoned and Python-verified. (Decode is checked: the golden cases lower identically
   through both engines.)
7. A fast fused decode path (§4.9): lifting the three `NOT_YET_IMPLEMENTED` rules of §4.9 —
   `fuse_conv` with `Hk ≠ Hv` (GQA, e.g. Qwen3-Next) and `BPV > 1` for small-batch parallelism,
   through an out-of-place conv state, with `fuse_out_norm` at `BPV > 1` also needing a
   cross-workgroup norm; and the fusions on the `simple` reference path. Computing each conv
   channel and gate once and sharing it through LDS is now the `conv_once` / `norm_gate_once` knobs
   (§4.10); choosing when to use them is still open.
8. `norm_gate_once` without `conv_once`: its `act(out_gate) ⊙ norm_weight` products need their own
   LDS block and barrier, so a norm-only spec can use it; rejected as `NOT_YET_IMPLEMENTED` until
   then (§4.8).
