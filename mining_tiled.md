# mining.md — gfx950 2D tiled attention (`hipkernel:Gfx950AttentionTiled`)

**RUNBOOK step 2b.** What the kernel can actually answer, classified against what §2a
established the graph can ask. Filed as `mining_tiled.md` because `mining.md` at this
worktree root is the DENSE sibling's artifact and is still live — two files, two kernels,
neither overwriting the other.

**Base:** `users/bharriso/rocke-gfx950-tiled-attention` @ `37970632e15`.
**Module:** `kernels/gfx950/attention_tiled_2d.py`.
**Builder:** `build_unified_attention_2d_tiled` (`:1028`) — the only builder in the module
(`grep -nE '^def build_'` → one hit). Enumerated, not guessed.
**Spec:** `UnifiedAttention2DTiledSpec`, defined **in the same module** (`:151-780`,
`__post_init__` at `:477-780`) — not in a separate spec file. gfx942 and gfx1250 have their
own copies; this file's is authoritative for gfx950.
**RUNBOOK 1a re-run this session:** `signature_error` empty, 46 fields, 8 required, spec class
as above. GATE PASS — the kernel has not moved under us.

## Sources used (budget: kernel module + spec free, then 5)

| # | Source | Rows it resolved |
|---|---|---|
| — | `kernels/gfx950/attention_tiled_2d.py` | free — the kernel module itself |
| — | `UnifiedAttention2DTiledSpec` + `__post_init__` | free — the spec |
| 1 | `kernels/common/attention_unified.py` | `select_path`, the tiled predicate, `_get_2d_launch_meta`, `_tiled_spec_from_problem` |
| 2 | `rocke/platform/python/rocke/helpers/attention.py` | `binary_search_seq_idx`, the grid-slack inverse |
| 3 | `rocke/platform/python/rocke/helpers/transforms.py` | `indirect()`, `TensorDescriptor.naive` — the page-table lowering |
| 4 | `builders/common/attention_spec_builder.py` | the D256 override fold (§6) |
| 5 | live execution against the real dataclasses, arch memo pinned | the 2D/3D measurements in §3 |

**4 of 5 spent.** Everything below is source-derived or measured; `UNSURE` rows are marked.

---

## 1. The constraint table — three independent surfaces

`Kind` is one of:

- **GRAPH-DERIVABLE** — a hipDNN `SdpaAttributes` graph can present a violating value, so
  `graph_match` MUST check it.
- **SPEC-INTERNAL** — it constrains only flag combinations chosen at descriptor-generation
  time. A shipped descriptor either satisfies it or fails to build; the matcher need not
  check it.
- **BOTH.**

### 1a. `supports_tiled_2d` (`:895-1020`) — 14 decline rows, all `return False, reason`

Read directly from source, not via a summary. **No raises** on this surface — it is a
verdict-pair predicate throughout.

| # | Line | Condition | Kind |
|---|---|---|---|
| 1 | `:930` | `validate_tiled_attention_arch(arch)` — the real arch gate (`arches: []` from introspection means *unknown*, not unsupported) | SPEC-INTERNAL (arch is pack-pruned) |
| 2 | `:932` | `dtype ∈ {fp16, bf16}` | **GRAPH-DERIVABLE** — Q/K/V `data_type` |
| 3 | `:934` | `head_size ∈ {64, 128, 256}` | **GRAPH-DERIVABLE** — `Q.dim[3]` |
| 4 | `:940` | `head_size % 32 == 0` | **GRAPH-DERIVABLE** (subsumed by row 3, but check both — row 3 is a set and this is an invariant) |
| 5 | `:946` | `block_size ∈ {16, 32, 64}` | **GRAPH-DERIVABLE** — `K.dim[2]`, §2a G2 |
| 6 | `:952` | `1 <= num_queries_per_kv <= 16` | **GRAPH-DERIVABLE** — `Q.dim[1] / K.dim[1]` |
| 7 | `:958` | `num_warps ∈ {1, 2, 4, 8}` | SPEC-INTERNAL |
| 8 | `:964` | `block_m_per_warp ∈ {16, 32}` | SPEC-INTERNAL |
| 9 | `:970` | `block_m_per_warp == 32 ⇒ num_warps ∈ {1,2,4}` (1024-thread CTA cap) | SPEC-INTERNAL |
| 10 | `:979` | `kv_storage_dtype ∈ {None, "fp8e4m3"}` | BOTH — fp8 declined in v1 |
| 11 | `:985` | `use_fp8 ⇒ kv_storage_dtype == "fp8e4m3"` | BOTH — fp8 declined in v1 |
| 12 | `:990` | `q_dtype ∈ {None, fp16, bf16}` | **GRAPH-DERIVABLE** |
| 13 | `:993` | `tile_size > 0 and tile_size % block_size == 0` | SPEC-INTERNAL |
| 14 | `:1000` | async-DMA payload floor: `tile_size * head_size >= num_warps * 64 * 8` | SPEC-INTERNAL |
| 15 | `:1015` | wave-uniform block-table invariant: `(512 // head_size) <= block_size` | **BOTH** — see below |

**Row 15 is the sharp one and it is genuinely GRAPH-DERIVABLE**, despite reading like a
tuning rule. `per_wave_tokens = (64*8)//head_size` must not exceed `block_size`, and BOTH
terms come from the graph (`head_size = Q.dim[3]`, `block_size = K.dim[2]`). It is only
evaluated when `tile_size is not None`, so a descriptor that pins `tile_size` inherits it.
The kernel's own comment says a violation makes the per-lane block-table lookup
lane-divergent and the async DMA under-fills the LDS slab — i.e. **wrong numbers, not a
fault**. Concretely `head_size=256` needs `block_size >= 2` (always true) but
`head_size=64` needs `block_size >= 8` — satisfied by the whole legal set, so on the
shipped scope it never fires. Recorded because it is one graph field away from firing and
nothing else would catch it.

### 1b. `__post_init__` (`:477-780`) — 62 raises

Counted mechanically over the line range, not estimated. **Exception types matter:**

| Type | Count |
|---|---|
| `ValueError` | 61 |
| **`NotImplementedError`** | **1** |

**TRAP T4 confirmed by count.** A matcher or harness written as `except ValueError` misses
exactly one rule — `kv_ring_depth ∈ {2,3}` — and misses it as an uncaught exception rather
than a decline. Any code catching spec-construction failures must catch `Exception`, which
is what `dispatch_parity.py:180` already does.

**Classification: overwhelmingly SPEC-INTERNAL.** These are multi-way experimental
flag-combination ANDs — the `use_transposed_*` family all require `use_mfma_32x32`;
`use_register_pv` is *incompatible* with `use_mfma_32x32`. The shipped scope turns none of
these on beyond what the dispatcher resolves, and the dispatcher resolves them
self-consistently by construction (§6). They fail at *descriptor build* time, loudly, which
is the correct place.

The GRAPH-DERIVABLE subset of `__post_init__` is the same shape/dtype set already covered by
1a rows 2-6 — `supports_tiled_2d` runs first and on raw ints, before a spec object exists,
so it is the surface `graph_match` mirrors.

### 1c. Builder-body raises — the rules NEITHER official surface catches

**Four raises, not three.** The plan claimed three at `:1432`, `:2798`, `:4264`; all three
are real but each cited the `if` guard line, with the `raise` on the next. Verified by
reading each site:

| Line (raise) | Plan said | Type | Rule |
|---|---|---|---|
| `:1433` | `:1432` ✓ | `ValueError` | `use_softmax_mfma_interleave` and `use_sched_barrier` are mutually exclusive (they steer the post-RA scheduler in opposite directions) |
| **`:1868`** | **not in plan** | `AssertionError` | fp8-mfma K loader: `T*HD` cannot be covered by any supported async-DMA payload |
| `:2799` | `:2798` ✓ | `NotImplementedError` | `use_fp8_mfma_pv` + `use_mfma_32x32` unsupported — 32x32x16 PV needs bf16 V in LDS |
| `:4265` | `:4264` ✓ | `NotImplementedError` | native fp8 PV requires `PV_K_STEP=32` |

Plus 7 bare `assert` statements (`:1298, :1843, :1989, :4439, :4514, :4526, :4534`) —
internal geometry invariants.

**All four are SPEC-INTERNAL, and three are fp8-only.** A matcher validating against
`supports_*` + `__post_init__` alone still hits these at *emission* — but emission happens
at `hkp_pack` time, on our own descriptors, not at match time on someone's graph. So they
are **step-4/5 build failures, not Tier-1 `graph_match` rows.**

> **This corrects the plan.** Plan §3.1 calls these "Tier-1 rejection-checklist rows for
> `graph_match`". They are not: no graph field reaches them. `use_softmax_mfma_interleave`,
> `use_sched_barrier`, `use_fp8_mfma_pv` and the fp8 K-loader are all descriptor-side
> choices. Putting them in `graph_match` would be dead code that reports green. They belong
> in the step-5 gate, where a descriptor that trips one fails to pack — loudly, which is
> correct. Recorded as a deviation from the plan with its reasoning, per the skill.

---

## 2. The declined class, and why the predicate does not enforce it — MEASURED

**Decision D4's scope is `select_path() == "3d"`.** `select_path` (`attention_unified.py:158-173`)
delegates to `rocke.helpers.attention.use_2d_kernel`:

```python
return (sliding_window > 0) or (max_seqlen_k <= 512) or (num_2d_prgms > target_num_prgms)
```

Route to **3D** iff: no sliding window, `max_seqlen_k > 512`, **and** the 2D grid
under-fills the device. Long-context, small-batch, few-KV-head decode.

**The finding: `supports_native_unified_attention_tiled` does NOT enforce the 2D/3D split,
and taking it at face value would ship descriptors for a path this engine does not have.**

Measured on the real dataclasses, arch memo pinned to gfx950, over a decode grid
(`max_seqlen_q=1`, `head_size=128`, `bf16`, `hq32/kv8`):

| num_seqs | max_seqlen_k | `select_path()` | predicate |
|---|---|---|---|
| 1, 4, 32 | 1024 | **3d** | `(True, 'supported')` |
| 1, 4, 32 | 4096 | **3d** | `(True, 'supported')` |
| 1, 4, 32 | 8192 | **3d** | `(True, 'supported')` |
| 1, 4, 32 | 32768 | **3d** | `(True, 'supported')` |

**12 of 12 3D-routed shapes pass the tiled predicate.** It answers for the tiled *family*,
not for the 2D *path*. The scope gate is therefore mechanical work this integration owes,
not something inherited — implemented in `tools/tiled_parity_adapter.py`
(`supports_tiled_2d_for_spec`), which applies rocKE's predicate **first** (so a genuinely
unbuildable shape reports the kernel's own reason) and the `select_path()=="2d"` gate
second.

Spot-checks confirming the predicate does still catch the hard bounds:

| probe | `select_path()` | predicate |
|---|---|---|
| `head_size=192` | 3d | `(False, 'head_size in {64,128,256} (got 192)')` |
| `block_size=8` | 3d | `(False, 'block_size in {16,32,64} (got 8)')` |
| `dtype=fp32` | 3d | `(False, 'supports fp16/bf16')` |
| D256 bf16 prefill sq4096 | **2d** | `(True, 'supported')` — the D5 cohort routes 2D ✓ |

---

## 3. Two calling-convention faults the profile cannot express — MEASURED

Both are prerequisites for step 4a and both are silent-or-misleading, not loud.

**Fault 1 — `UnifiedAttentionProblem` cannot be the profile's `request.class`.**
`dispatch_parity.py:171-172` injects `fields["arch"]` whenever the profile declares `arch:`,
and the problem dataclass has no such field:

```
TypeError: UnifiedAttentionProblem.__init__() got an unexpected keyword argument 'arch'
```

Every shape would land in `rejected` with a TypeError reading like a corpus defect. Dropping
`arch:` is not a fix — five other tools read it.

> **This corrects plan §6/P2**, which recommends `request.class = UnifiedAttentionProblem`
> directly. It cannot be, for this mechanical reason.

**Fault 2 — the tiled predicate takes a PROBLEM; the tool passes a SPEC.**
`dispatch_parity.py:188` calls `predicate(spec, arch=arch)`. The dense predicate takes a
spec; `supports_native_unified_attention_tiled` takes a problem
(`inspect.signature` → `(problem: 'UnifiedAttentionProblem')`). Measured:

```
predicate(problem) -> (True, 'supported')
predicate(spec)    -> AttributeError: 'UnifiedAttention2DTiledSpec'
                                       object has no attribute 'use_fp8'
```

**Why the adapter parks the problem on the spec rather than recomputing it.**
`_tiled_spec_from_problem` is **lossy** — a 22-field problem becomes a 46-field spec that
drops `num_cus`, `target_ctas`, `max_seqlen_k` and more. Reconstructing a problem from a
spec would be a second, hand-written implementation of the mapping: precisely the
restatement the skill says ships wrong. The adapter stashes the originating problem via
`object.__setattr__` under `_parity_problem` (the spec is frozen). **Verified invisible:**
`dataclasses.fields(spec)` reports 46 names before and after, and `_parity_problem` is not
among them — so `knob_partition`, `build_config` and the metadata writer, which all
enumerate declared fields, cannot see it.

---

## 4. The ABI — 18 kernargs, all unconditional

**Two RUNBOOK greps return zero on this module and BOTH are failed queries**, with positive
controls run:

- `grep -nE '\.ptr\(|\.scalar\('` → 0. The tiled ABI is declared with **`b.param(...)`**.
  Control: the pattern matches in `attention_dense_prefill.py:67-70`.
- `awk '/^def run_/'` → 0. There is **no** `run_`-prefixed wrapper in any tiled module.
  Control: `run_unified_attention_torch` exists at `attention_unified.py:4137`.

Declaration order (`:1191-1230`) — the C++ kernarg mirror is a **fixed 18-slot pack**:

| # | Kernarg | Type | Always present? |
|---|---|---|---|
| 1 | `output_ptr` | ptr(dtype) | yes |
| 2 | `query_ptr` | ptr(dtype) | yes |
| 3 | `key_cache_ptr` | ptr(kv_io_dtype) | yes |
| 4 | `value_cache_ptr` | ptr(kv_io_dtype) | yes |
| 5 | `sink_ptr` | ptr(dtype) | yes — even when `use_sinks=False` |
| 6 | `block_tables_ptr` | ptr(i32) | yes |
| 7 | `seq_lens_ptr` | ptr(i32) | yes |
| 8 | `alibi_slopes_ptr` | ptr(f32) | yes — even when `use_alibi=False` |
| 9 | `qq_bias_ptr` | ptr(f32) | yes — even when `use_qq_bias=False` |
| 10 | `query_start_len_ptr` | ptr(i32) | yes |
| 11 | `scale` | f32 | yes |
| 12 | `k_scale` | f32 | yes |
| 13 | `v_scale` | f32 | yes |
| 14 | `out_scale` | f32 | yes — **bound but never read in-kernel**; still a real slot |
| 15 | `softcap` | f32 | yes |
| 16 | `num_seqs` | i32 | yes |
| 17 | `block_table_stride` | i32 | yes |
| 18 | `qq_bias_stride_0` | i32 | yes |

**Seven are conditionally *consumed* by baked flags but always *present*.** There is no
reduced-arity signature — structurally simpler than the dense kernel, which conditionally
*extends* its signature and whose pack therefore has to reason about argument counts.

**Consequence for `graph_match`, and it cuts the opposite way to dense.** The dense pack's
declines are load-bearing *because a sixth kernarg would be read uninitialised*. Here they
are not: slots 5/8/9 always exist. A graph asking for sinks or alibi is declined for
**variant-coverage** reasons (no shipped descriptor has the flag baked), never to protect
the ABI. That is a genuinely weaker safety argument and it is recorded as such.

---

## 5. Launch geometry — the `+ num_seqs` slack, and its inverse

Production path: **`_get_2d_launch_meta`** (`attention_unified.py:4058-4103`), branch B:

```
block_m = num_warps * block_m_per_warp
block_q = block_m // num_queries_per_kv   if num_queries_per_kv <= block_m else 1
total_num_q_blocks = total_q // block_q + num_seqs        # <-- varlen slack
grid  = (num_kv_heads, total_num_q_blocks, 1)
block = (64 * num_warps, 1, 1)                            # wave_size 64 on gfx950
```

The `+ num_seqs` term reserves one padding q-block per sequence so ragged batches need no
exact-division block count. The kernel compensates (`attention_tiled_2d.py:1247-1266`) with
a binary search over `cu_q` and an early `b.ret()`:

```python
seq_idx           = binary_search_seq_idx(b, cu_q, q_block_global_idx, num_seqs_p, ...)
q_block_start_idx = cu_q_start // BLOCK_Q + seq_idx        # :1260 -- the per-seq +1
q_block_local_idx = q_block_global_idx - q_block_start_idx  # :1261
if qb_start_pos >= cur_batch_q_len: b.ret()                 # :1265-1266
```

`binary_search_seq_idx`'s loop invariant (`helpers/attention.py:685-733`) is
`cu_q[i] // BLOCK_Q + i <= target` — **exactly inverting** the grid construction.

**Both halves must be restated in C++.** Omit the `+ num_seqs` and the kernel
under-launches, dropping tail blocks — output rows never written, which the harness's
NaN sentinel catches as `allClose=false` with **zero finite mismatches** (T: never a
tolerance problem). This is launch-surface entry #1 and the kind of arithmetic the skill
says ships wrong silently. It gets a pure-function geometry header + per-shape test,
mirroring `Gfx950AttentionDenseGeometry.hpp`.

---

## 6. The D256 composition — established, not re-derived

Carried from the plan, which traced it to source and confirmed by execution:

`_spec_gfx950_generic` **already folds** `_d256_gfx950_spec_overrides()` in via a tail
`replace()` for exactly the D256 cohort (`attention_spec_builder.py:313-317`). Idempotent
on-cohort. Applying them **off-cohort silently builds a different binary** — 7 fields
differ, different kernel name, no error.

**Therefore:** bake the resolver's output unmodified. Never hand-transcribe the override
dict; never `--knobs` one of those seven fields (`use_kq_lds_pad`, `kq_lds_pad_halves`,
`use_mfma32_skip_legacy_qreg`, `use_k_single_buffer`, `use_q_direct_reg`,
`softmax_interleave_mode`, `use_mask_phase_split`).

Cohort predicate (`attention_unified.py:699-716`): `head_size==256 and dtype=="bf16" and
not use_fp8 and sliding_window==0 and softcap==0 and not use_sinks and not use_alibi and
not use_qq_bias and max_seqlen_q>1`. Confirmed reachable: the D256 bf16 prefill probe in §2
routes 2D and passes the predicate.

Per D5, the profile's `dispatch:` targets the **production wrapper**
`kernels.common.attention_unified._tiled_spec_from_problem` (`:2756-2767`), which adds
`_resolve_lds_budget` — not the builders-layer function.

---

## 7. `kernel_name()` omissions — a symbol-uniqueness hazard

`kernel_name()` (`:837-874`) interpolates most spec fields but **omits four the kernel still
bakes**:

| Omitted field | What it changes in the binary |
|---|---|
| `num_seqs` | drives `binary_search_iters` — a compile-time loop trip count |
| `waves_per_eu` | an LLVM occupancy attribute (`:1185-1186`) |
| `use_kq_lds_pad`, `kq_lds_pad_halves` | K_lds allocation shape and addressing |
| `use_i64_kv_addr` | 64- vs 32-bit buffer-offset addressing |

Two specs differing only in these collide on `kernel_name()` while emitting **different
machine code**. This is legal and the skill says so — uniqueness is `(toc_key, symbol)`,
never the symbol string alone. Recorded because the desk check's invariant 4 exists for
exactly this, and because a variant set that varies `waves_per_eu` (a live sweep candidate)
would hit it. **Mitigation:** the descriptor template must encode these four, since the
loader rejects a pack whose expansion produces two kernels with the same name.

---

## 8. Traps carried into steps 5-9

| # | Trap | Evidence |
|---|---|---|
| T1 | `_resolve_attention_arch()` queries the LOCAL DEVICE, memoizes to a module global, falls back to `'gfx950'`. **No env var.** Host tooling works today only because the fallback happens to be our arch. Pin `attention_unified._RESOLVED_ATTENTION_ARCH` in every harness. | `:237-262`; pinned in all §2/§3 measurements above |
| T2 | D256 overrides off-cohort build a different kernel, silently | §6 |
| T4 | `kv_ring_depth` raises **`NotImplementedError`**, not `ValueError` — 1 of 62 | §1b, counted |
| T5 | Four builder-body raises no applicability surface catches — but they are build-time, not match-time | §1c |
| T6 | RUNBOOK's `.ptr(`/`.scalar(` and `^def run_` greps both return zero here — failed queries, positive controls run | §4 |
| **T14** | **The tiled predicate accepts 3D-routed shapes (12/12 measured).** Scope must be enforced by us. | §2 |
| **T15** | **`UnifiedAttentionProblem` cannot be `request.class`** — the tool injects `arch=` | §3 |
| **T16** | **The tiled predicate takes a problem, the tool passes a spec** | §3 |
| T17 | `kernel_name()` omits 4 baked fields — symbol collision is legal, key on `(toc_key, symbol)` | §7 |
| T18 | The 18-slot ABI is fixed, so declines protect **coverage**, not the ABI — a weaker argument than dense's | §4 |

---

## GATE

`ls mining_tiled.md` succeeds, and **every constraint row above carries a verdict**: 15
rows in 1a each classified GRAPH-DERIVABLE / SPEC-INTERNAL / BOTH, 62 `__post_init__` raises
counted and classified by exception type, and 4 builder-body raises located at confirmed
current lines with their tier corrected relative to the plan.
