# mining.md — gfx942 `attention_dense`, what the kernel can answer

Produced at RUNBOOK step 2b, per `rocke-mining.md`. Written after the kernel module
and its spec, before any third source, then extended.

**Source budget (cap = 5 beyond the kernel module), used:**

1. `kernels/gfx950/attention_dense.py` — `AttentionDenseSpec.__post_init__` (the base
   class; gfx942 has no `__post_init__` of its own).
2. `rocke/library/dispatch/attention/gfx942.py` — the dispatcher for THIS builder
   (variant-set authority per RUNBOOK § Sizing the variant set).
3. `integration-tests/gpu-ref/kernels/sdpa/GpuRefSdpaFwd.cpp` — the mask predicate
   (also used at 2a).
4. `src/engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp` — incumbent mask derivation
   (also used at 2a).
5. `dnn-providers/integration-tests/README.md` — reference-executor limits (2a/step 1b).

At the cap. Remaining unknowns are marked `UNSURE` and become step-3 questions.

---

## 1. The constraint table

Buckets per `rocke-mining.md`'s six-question sort: **SCOPE** (feature declined
wholesale), **GRAPH** (graph fact vs fixed constant), **GRAPH/BAKED** (graph fact vs a
value a variant bakes in), **KNOB-SEL** (graph fact vs a knob we pick — per-variant),
**SPEC** (both sides knobs — spec construction only), **UNREP** (real capability with no
hipDNN attribute).

Rules enumerated mechanically with the AST walk from `rocke-mining.md`. It printed
**32 guarded rules across 6 functions**; the doc's warning was accurate — this module has
**no `__post_init__`**, and 25 of the rules live in `supports_attention_dense`. A walk
pinned to `__post_init__` would have printed nothing.

### 1a. `supports_attention_dense` (gfx942, 25 guarded rules)

| # | Line | Rule | Bucket | Verdict / where enforced |
|---|---|---|---|---|
| 1 | 862 | `arch != "gfx942"` | GRAPH | Pack `arch: [gfx942]` — the pack arch-prunes before the matcher. No hook code needed. |
| 2 | 867 | `not isinstance(spec, AttentionDenseSpec)` | SPEC | Python-side type guard; not reachable from a descriptor. |
| 3 | 870 | `dtype not in ("bf16","fp16")` | GRAPH/BAKED | `kernel_match`: all four tensor dtypes == `$kernel.dtype`. |
| 4 | 875 | `head_size not in (64,128)` | GRAPH/BAKED | `kernel_match`: `D_qk == D_v == $kernel.head_size`. |
| 5 | 908 | every extent `> 0` | GRAPH | `graph_match`: all dims positive. (The comment notes `%` is sign-following, so the divisibility rules do **not** imply positivity.) |
| 6 | 918 | `spec.varlen` | SCOPE | Decline `seq_len_q/kv_tensor_uid` in `graph_match`. Never ship `varlen=true`. |
| 7 | 920 | `spec.ragged` | SCOPE | Decline `ragged_offset_tensor_uid` (a **tensor** attribute, not a node one). |
| 8 | 922 | `spec.sliding_window` | SCOPE | Decline any mask that derives to `SLIDING_WINDOW`. |
| 9 | 924 | `spec.use_sinks` | SCOPE | Decline `sink_token_tensor_uid`. |
| 10 | 939 | `block_m > 0 && block_m % 32 == 0` | SPEC | We ship `block_m = 256`. Not graph-visible. |
| 11 | 946 | `block_m//32*64 <= 1024` | SPEC | Same. |
| 12 | 955 | **`seqlen_q % block_m != 0`** | **KNOB-SEL** | `kernel_match`: `Sq % $kernel.block_m == 0`. At the shipped `block_m=256` this means **Sq must be a multiple of 256**. Comment: Q/O are addressed with no bounds check, so a violation reads and writes OOB. |
| 13 | 971 | pads `>= 0` and `% 4 == 0` | SPEC | Shipped defaults only. |
| 14 | 978 | `waves_per_eu > 0` | SPEC | From `_tuned_waves_per_eu`. |
| 15 | 986 | `use_cfvst` forced ON off-policy | SPEC | We follow `_use_cfvst` policy exactly. |
| 16 | 996 | `use_v_swizzle` ON without cfvst | SPEC | Same. |
| 17 | 1005/1007 | swizzle needs pow2 `V_LDROW >= 64` | SPEC | Same. |
| 18 | 1018 | `block_m % block_n != 0` | SPEC | Both knobs. |
| 19 | 1035 | `block_n % waves` and `(block_n//waves) % rows_per_instr` | SPEC | Both knobs, but *coupled to `head_size`* via `rows_per_instr`. Constrains the legal `(head_size, block_n)` cells of the variant set. |
| 20 | 1045/1048 | cfvst V-block count divisible by thread count | SPEC | Same. |
| 21 | 1060 | **LDS budget `_lds_bytes(spec) <= 65536`** | SPEC (variant-set bound) | Rejects `block_n=128/256` at D128 and `block_n=256` at D64 — *verified empirically*, see §6. Bounds which `block_n` values can be shipped. |
| 22 | 1073 | **`B·Skv·Hkv·D·2 < 2^31`** | GRAPH | `graph_match`: 32-bit K/V extent bound. Comment: the offsets are `add nsw`/`mul nsw` i32 — signed overflow is UB, so this is a *silent* failure, not a wrap. |
| 23 | 1079 | **`B·Sq·Hq·D < 2^31` (elements)** | GRAPH | Same, note the units differ from #22 (elements vs bytes). |

### 1b. Base `AttentionDenseSpec.__post_init__` (in the **gfx950** module)

Re-run by `supports_attention_dense` at line 890 over the *base* fields only.

| Rule | Bucket | Verdict |
|---|---|---|
| `dtype in ("bf16","fp16")` | GRAPH/BAKED | dup of #3 |
| `head_size in (64,128)` | GRAPH/BAKED | dup of #4 |
| `lds_k_group_pad >= 0 && % 8 == 0` | SPEC | shipped default 8 |
| `block_n % 32 == 0`, `> 0` | SPEC | |
| **non-ragged: `seqlen_q % _BLOCK_M(256) != 0`** | KNOB-SEL | dup of #12 at the shared constant |
| **non-ragged: `seqlen_kv % block_n != 0`** | **KNOB-SEL** | `kernel_match`: `Skv % $kernel.block_n == 0`. This is the exact worked example in `rocke-mining.md` § *knob-selection*. At the shipped `block_n=64`, **Skv must be a multiple of 64**. |
| `num_kv_heads != 0 && Hq % Hkv == 0` | GRAPH | `graph_match`: GQA divisibility. Kernel emits `sdiv i32 %hq, gqa`. |
| `persistent ⇒ num_persistent > 0` | SPEC | |
| `persist_decode in (qb_major, hkv_major, auto)` | SPEC | |
| `sliding_window >= 0`; `>0 ⇒ causal`; `>0 ⇒ % block_n == 0` | SCOPE | declined wholesale (#8) |
| `varlen ⇒ !persistent`, `varlen ⇒ causal` | SCOPE | declined (#6) |
| `waves_per_eu in [1,8]` | SPEC | |
| **`ragged ⇒ Sq == Skv`, `!varlen`, `!sliding_window`** | SCOPE | declined (#7) |
| **`paged ⇒` block_size pow2 > 0, `block_n % block_size == 0`, `num_kv_blocks > 0`, `batch == 1`, `!varlen`, `!persistent`, `head_size == 128`, dtype fp16/bf16, `sliding_window > 0`** | SCOPE | declined — see §1c |

### 1c. The `paged` finding — the runbook's step-1b rule gives the wrong answer here

The runbook's 1b command greps for `if spec.paged` in the gfx942 module. **It matches
nothing**, and the runbook's stated reading of "no branch" is *"the feature is
unconditional, there is no dense subset."* That reading is wrong here, and so is the
opposite ("the feature is absent").

The truth is a third case the runbook does not enumerate:

- `paged` is a field of the shared spec and is **never referenced by
  `supports_attention_dense`**;
- it *is* validated by the base `__post_init__`, which `supports_attention_dense`
  re-runs — so the paged rules **do** fire, from a different file;
- and `rocke-mining.md`'s own "not yet implemented guards read like capabilities" trap
  fires exactly as written: paged requires `sliding_window > 0`, but
  `supports_attention_dense` then rejects `sliding_window` **unconditionally**.

Verified empirically:

```
paged b=1, no sw : SPEC-REJECT: paged not yet implemented for plain-causal (sliding_window>0 only)
paged b=1, sw=64 : False  gfx942 attention_dense: sliding_window not yet supported
paged b>1        : SPEC-REJECT: paged multi-sequence (batch>1) not yet implemented
```

**Verdict: `paged` is unreachable on gfx942 — every paged spec is rejected by one gate
or the other.** SCOPE. Ship `paged: false`; decline `page_table_k/v_tensor_uid` in
`graph_match`. The reference executor declines paged anyway (§ step 1b), so nothing is
lost.

The `batch == 1` rule that `rocke-mining.md` §*Worked example* holds up as the shipped
defect is **paged-only** on this kernel — it lives inside `if self.paged:`. Under
question 1 of the sort, that makes it SCOPE, not a matcher obligation. **But `batch` is
still baked** — see §5, where it is a matcher obligation for a completely different
reason.

### 1d. Builder-only rules (`_build_attention_dense_single_buffer`)

Three `raise ValueError`s at 1253, 1278/1292 — all mirrored in
`supports_attention_dense` (#19, #20), by the module's stated `support ⇒ build`
contract. Nothing extra to enforce.

### 1e. `run_attention_dense_torch` launch-time guards

The runbook's `awk` command found two (it works on this module):

- 1878: re-runs `supports_attention_dense`, raises `NotImplementedError`.
- 1881: `cu_seqlens_q/kv is not None` → `ValueError`.

Neither is reachable through the descriptor path (`hkp_pack` calls the builder, not the
torch wrapper). No matcher obligation. Recorded so the ABI's absence of `cu_seqlens` is
explicit.

---

## 2. Layout — the arithmetic, per operand

From `_build_attention_dense_single_buffer`, lines 1155-1156, 1394-1409, 1708-1713:

```python
stride_q_tok = Hq * D          # Q and O
stride_k_tok = Hkv * D         # K and V
q_base = bt*Sq*stride_q_tok + hq*D
k_base = bt*Skv*stride_k_tok + hkv*D
addr_q = q_base + q_tok*stride_q_tok + col        # global_load_vN(q, …)
o_base = bt*Sq*stride_q_tok + hq*D                # IDENTICAL form to q_base
addr_o = o_base + qtok*stride_q_tok + …           # global_store_vN(o, …)
```

`((b·S + s)·H + h)·D + d` — row-major `[B, S, H, D]`, **BSHD**, head varying faster
than sequence.

Per operand, as `rocke-mining.md` instructs:

| Operand | Base | Stride | Requirement |
|---|---|---|---|
| Q | `q_base` | `Hq·D` | BSHD with `Hq` heads |
| K | `k_base` | `Hkv·D` | BSHD with `Hkv` heads |
| V | **reuses `k_base`/`stride_k_tok`** | `Hkv·D` | must share K's layout **exactly** |
| O | **reuses `q_base` form** | `Hq·D` | must match Q **exactly** |

**No stride arguments exist in the ABI** (§4). The strides are computed from `H` and
`D` and baked. A permuted, sliced or padded tensor cannot be accepted at all.

**Consequence for the matcher (Tier 1, silent):** a BHSD graph is read as if BSHD —
in-bounds, wrong elements, **no fault**. K/V go through `buffer_rsrc` (line 1264-1265),
so OOB there is zero-filled, also silent; only an undersized Q on the unguarded
`global_load_vN` path actually faults.

**Every shipped `SdpaFwd` bundle on this tree is BHSD** (strides `[131072,32768,128,1]`
on `[2,4,256,128]`), so:

- `graph_match` **must** check strides per tensor: `stride_d==1`, `stride_s==H·D`,
  `stride_h==D`, `stride_b==S·H·D`.
- Stage 8 needs **new BSHD bundles**. Running the shipped bhsd ones proves nothing —
  the engine correctly declines them.

---

## 3. Grid / block, constants resolved, and the KMD fields they read

```python
def attention_dense_grid(spec):                       # line 1791
    if spec.persistent:  return (spec.num_persistent, 1, 1)
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    return (nqb, spec.num_query_heads, spec.batch)

def attention_dense_block(spec):                      # line 1810
    return (spec.block_m // 32 * 64, 1, 1)
```

Resolved at the shipped `block_m = 256`, `persistent = False`:

- **grid** = `(Sq/256, Hq, B)` — exact division, because `seqlen_q % block_m == 0` is
  enforced (#12).
- **block** = `(512, 1, 1)` — 8 wave64s.

Note the branch order: the `persistent` arm returns **before** `_as_gfx942_spec`
promotion, so it never reads `block_m`.

**KMD fields the geometry reads, therefore KMD fields regardless of matching**
(`rocke-mining.md` § *KMD fields are not only matcher inputs*):
`seqlen_q`, `num_query_heads`, `batch`, `block_m`, `persistent`, `num_persistent`.

---

## 4. ABI — per-argument conditionality

`attention_dense_signature` (line 1819), via `SignatureBuilder`:

| # | Slot | Type | Conditional? |
|---|---|---|---|
| 0 | `q_ptr` | ptr(`spec.dtype`) | **always present** |
| 1 | `k_ptr` | ptr(`spec.dtype`) | **always present** |
| 2 | `v_ptr` | ptr(`spec.dtype`) | **always present** |
| 3 | `o_ptr` | ptr(`spec.dtype`) | **always present** |
| 4 | `scale` | f32 scalar | **always present** |

**Unconditional, five slots, no branches.** Verified two ways: the `.ptr(`/`.scalar(`
grep returns exactly these five lines (1833-1837) and none sits inside an `if`; and the
docstring states it — *"No `cu_seqlens` pair: varlen is rejected … Those pointers land
when varlen does."* This is the *second* of the two ABI conventions
`rocke-mining.md` describes (fixed count), not the first (conditional append).

`scale` is a **scalar kernarg**, so `scale_tensor_uid` in the graph has no slot and must
be declined. The kernel multiplies by `LOG2E` internally (line 1184) — the graph's
`attn_scale_value` passes through unmodified.

---

## 5. Every field a variant pins — the exhaustive check

`rocke-mining.md`'s enumeration script needs `$KDP`, which does not exist until step 4.
Enumerating from the spec instead, using what `kernel_name()` bakes as the ground truth
for "the compiled binary differs on this axis":

```
b2  d128 hq8 kv8 bn64 bf16 sq256 sk256 causal lazyrs gfx942 b2 wpe2
b1  …                                                  gfx942 b1 wpe2      <- batch in the name
D64 …_d64_…_kpad8_…
```

| Spec field | Pinned to (proposed set) | Enforced by | Hook |
|---|---|---|---|
| `dtype` | bf16, fp16 | all 4 tensor dtypes == `$kernel.dtype` | `kernel_match` |
| `head_size` | 64, 128 | `D_qk == D_v == $kernel.head_size` | `kernel_match` |
| `num_query_heads` | per variant | `Q.dims[1] == $kernel.num_query_heads` | `kernel_match` |
| `num_kv_heads` | per variant | `K.dims[1] == $kernel.num_kv_heads` | `kernel_match` |
| `seqlen_q` | per variant | `Q.dims[2] == $kernel.seqlen_q` | `kernel_match` |
| `seqlen_kv` | per variant | `K.dims[2] == $kernel.seqlen_kv` | `kernel_match` |
| **`batch`** | **per variant** | **`Q.dims[0] == $kernel.batch`** | **`kernel_match`** |
| `causal` | true, false | derived mask type == `$kernel.causal` | `kernel_match` |
| `block_n` | 64 (+32 if shipped) | `Skv % $kernel.block_n == 0` | `kernel_match` (KNOB-SEL) |
| `block_m` | 256 | `Sq % $kernel.block_m == 0` | `kernel_match` (KNOB-SEL) |
| `paged`, `varlen`, `ragged`, `sliding_window`, `use_sinks` | all off | declined wholesale | `graph_match` |
| `block_size`, `num_kv_blocks` | 0 | paged-only, unreachable | — |
| `waves_per_eu` | `_tuned_waves_per_eu(D,dtype)` | not graph-visible | — |
| `lds_k_group_pad`, `lds_row_pad`, `v_row_pad`, `use_cfvst`, `use_v_swizzle`, `use_exp2_fast`, `iglp`, `lazy_rescale`, `interleave` | shipped defaults | not graph-visible | — |
| `persistent`, `num_persistent`, `persist_decode` | false / 304 / auto | not graph-visible | — |
| — (no spec field) | **tensor strides** | BSHD pattern per tensor | **`graph_match`** |

**`batch` is a matcher obligation on this kernel, and not for the reason the worked
example gives.** The paged `batch == 1` rule is unreachable here (§1c). But the *body*
bakes `B` into the K/V buffer-resource extent:

```python
k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))     # line 1264
v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))     # line 1265
```

A graph with a larger `B` than the variant was compiled for reads **zero-fill past the
bound** — silently wrong, no fault. The kernel module's own docstring confirms the
consequence: *"two specs differing only in those MUST NOT share a cached binary, or a
B>1 launch is served the B=1 kernel and reads out of bounds."*

This is precisely the defect class `rocke-mining.md` describes, arrived at by a
different route — and **the doc's own discovery command for it fails.** The command
printed in § *Rules can also be baked WITHOUT appearing in `__post_init__`* is

```bash
grep -n "buffer_rsrc\|num_records" $M   # returns NOTHING, exit 1
```

the BRE `\|` form that the same document warns about three sections earlier. Only the
`-E` form finds lines 1264-1265. See the friction table.

**Every `seqlen_q`/`seqlen_kv`/`Hq`/`Hkv`/`batch` value is baked into the binary** — this
kernel is fully shape-specialized. There is no "compiled for a capacity, serves anything
below" bound-checked case here; all shape fields are equality obligations.

---

## 6. Variant-set inputs (feeds step 3)

Authority order per RUNBOOK: the dispatcher first.

**`dispatch/attention/gfx942.py`** — the module the builder actually reaches:

- `_DENSE_BLOCK_N = 64` — the single production `block_n`.
- `_GFX942_NUM_PERSISTENT = 304` (gfx942 CU count), vs the shared default 256.
- `waves_per_eu` from `_tuned_waves_per_eu(head_size, dtype)`.
- Every gfx942-private knob is **left at its shipped default** — the docstring is
  explicit: *"those knobs are sweep-visible and dispatch-invisible. Wiring one of them
  into this factory would make it a production path and would need its own measured
  verdict first."*
- **The candidate is OPT-IN ONLY**: `priority=3`, and `support()` returns False unless
  the request names `algorithm="attention_dense"` / `spec_id="gfx942_attention_dense"`.
  Per RUNBOOK § *Sizing the variant set*: *"If it does not auto-select your kernel …
  say so in the step-9 report."* Recorded — we are exposing something rocKE itself does
  not route to by default.

**LDS budget** bounds `block_n` (empirically probed):

| head_size | block_n=32 | 64 | 128 | 256 |
|---|---|---|---|---|
| 128 | ok | ok | **reject** (69632 B > 65536) | **reject** (139264 B) |
| 64 | ok | ok | ok | **reject** (67584 B) |

So the tuning axis is `block_n ∈ {32, 64}` at D128 and `{32, 64, 128}` at D64. The
dispatcher ships only 64.

**Axes are not fully orthogonal** (RUNBOOK § *When the axes are not orthogonal*):
`block_n` legality depends on `head_size` (LDS budget, and rule #19's
`rows_per_instr` coupling), and the cfvst/swizzle policy is a function of
`(head_size, dtype)`.

**Shape fields are baked**, so the variant set is a cross-product over
`(dtype, head_size, causal, Hq, Hkv, B, Sq, Skv)` — this explodes immediately.
The honest first integration is narrow and explicit (step 3).

---

## 7. Rejection checklist — ordered by failure severity

The `graph_match` implementation order.

### Tier 1 — silent wrong answers (no fault, no error)

1. **Layout not BSHD**, any of Q/K/V/O. Reads wrong elements in bounds. *The shipped
   bundles are all BHSD.*
2. **`batch` mismatch** vs the variant. Baked into the buffer-resource extent → zero-fill
   past the bound.
3. **`seqlen_q` / `seqlen_kv` / `Hq` / `Hkv` / `head_size` mismatch** vs the variant.
   All baked; the KV loop trip count `n_ktiles = Skv // BN` is a compile-time constant,
   so a longer graph is silently truncated to a prefix.
4. **Mask-type mismatch.** A causal graph served by `causal=false` (or vice versa)
   computes a valid-looking wrong answer. Includes `BOTTOM_RIGHT_CAUSAL` at
   `Sq != Skv` — the kernel's clamp is top-left only.
5. **`attn_mask_tensor_uid` set.** No bias input; the mask would be ignored entirely.
6. **`generate_stats` / `stats_tensor_uid`.** Output never written; caller reads garbage.
7. **32-bit extent overflow** (`B·Skv·Hkv·D·2 >= 2^31` bytes, or `B·Sq·Hq·D >= 2^31`
   elements). `add nsw`/`mul nsw` ⇒ UB, LLVM may poison the address chain.
8. **`D_qk != D_v`.** hipDNN allows it; the kernel assumes one `D`.
9. **Mixed tensor dtypes.** One `spec.dtype` covers all four operands.

### Tier 2 — faults / hard failures

10. **Non-positive extents.** `Hq == 0` emits `sdiv i32 %hq, 0`.
11. **`Hq % Hkv != 0`.** GQA group is an integer divide.
12. **`Sq % block_m != 0`.** Q/O addressed without a bounds check → OOB read *and write*.
13. **`Skv % block_n != 0`.** KV loop drops the tail.

### Tier 3 — declined features (loud, correct)

14. `seq_len_q/kv_tensor_uid` (varlen) · `ragged_offset_tensor_uid` ·
    `page_table_k/v_tensor_uid` · `block_mask_tensor_uid` · `sink_token_tensor_uid` ·
    `alibi_mask` · `padding_mask` · dropout (`dropout_probability`, `seed`, `offset`,
    `dropout_mask`, `dropout_scale`, `rng_dump`) · FP8 (`descale_*`, `scale_s`,
    `scale_o`, `amax_*`) · `max_seq_len_kv` · `max_tensor_uid` / `sum_exp_tensor_uid` ·
    `scale_tensor_uid` · `mma_core_mode` other than UNSET/float ·
    `implementation` other than AUTO · sliding-window mask types.

### Tier 4 — missed opportunity

15. No variant matches the graph's shape tuple. Correct decline; the signal to widen
    the set.

---

## 8. `UNSURE` rows → step 3 questions

| # | Question |
|---|---|
| U1 | `attn_scale_value` absent but `scale_tensor_uid` also absent — is there a defined default (1/√D), or must we decline? The schema default is `null`. |
| U2 | Do we accept `BOTTOM_RIGHT` alignment at `Sq == Skv`? (Required to serve any shipped causal bundle; mathematically identical there. Proposing yes.) |
| U3 | Which shape tuples to ship, given that every shape field is baked (§6)? |
| U4 | `implementation != AUTO` — decline, or ignore? The runbook notes the shipped `AttentionDenseNative.cpp` leaves it `UNCHECKED`. |

---

## GATE

`mining.md` exists. Every constraint-table row carries a bucket verdict. The layout
statement names the arithmetic and covers all four operands. The ABI list states per
argument that its slot is unconditional. Rejection checklist ordered silent-first.
