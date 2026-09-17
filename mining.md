# mining.md — gfx950 `attention_dense`, re-mined against the #11627 kernel

**Base:** `users/bharriso/rocke-gfx942-dense-C-extended` @ `ef0b7cd24fb` + `pr11627` @
`9a3c3b3273b`, merged clean (shared base `f2af4e5e360`).
**Module:** `kernels/gfx950/attention_dense.py`
**Builder:** `build_attention_dense` — the ONLY builder in the module
(`grep -nE '^def build_'` → one hit, line 612). Enumerated, not guessed.
**Spec class:** `AttentionDenseSpec`, **26 fields**, 6 required, `signature_error` empty.

**This supersedes the pre-#11627 mining that the profile's header encodes.** Three of
that profile's claims are falsified; each is marked **FALSIFIED** below.

## Sources used (budget: kernel module + spec free, then 5)

| # | Source | Rows it resolved |
|---|---|---|
| — | `kernels/gfx950/attention_dense.py` (spec, `__post_init__`, properties, `kernel_name`, `supports_*`, grid/block/signature, `run_*`) | free — the kernel module itself |
| 1 | `dispatch/attention/gfx950.py` — `_dense_spec` | every `derived` row; the `wide_lds_dma` rule |
| 2 | `git log ef0b7cd24fb..pr11627 -- <module>` | knob verdicts (bodies are empty; verdicts live in source comments) |

3 sources of the 5-source budget remain unspent. Rows still open are marked `UNSURE` and
go to step 3.

---

## 1. What #11627 changed, against the profile's recorded claims

### 1.1 `block_m` — FALSIFIED (S1, silent)

**Profile line 21-24 and `Gfx950AttentionDenseGeometry.hpp:27,52` both assert
`block_m` is a module constant, not a field. That is now false.**

```python
DENSE_TILE_GEOMETRIES = {"default": {"block_m": 256, "block_n": 64, "lds_v_row_pad": 32},
                         "bm128":   {"block_m": 128, "block_n": 64, "lds_v_row_pad": 32}}
block_m: int = _DEFAULT_TILE_GEOMETRY["block_m"]        # spec field, line 168
num_waves = block_m // 32                                # property, line 483-484
```

`_BLOCK_M` no longer exists anywhere in the module. Consequences, all of which the C++
currently gets wrong without failing to build:

- `attention_dense_grid` (line 2307) ceils on `spec.block_m`, not 256.
- `attention_dense_block` (line 2315) is `num_waves * 64` = `block_m // 32 * 64` =
  **`block_m * 2`** threads. 512 only at `block_m=256`.
- `__post_init__` line 307 requires `seqlen_q % block_m == 0` on the aligned path —
  a per-variant divisibility, i.e. a **KNOB-SEL** row, not a constant predicate.
- `kernel_name()` line 536 appends `bm{block_m}` when non-default, so the two geometries
  are distinct binaries with distinct symbols.
- `supports_attention_dense` adds three NEW rules keyed on it (lines 591-608):
  `block_m ∈ {128, 256}`, `block_m % block_n == 0`, `block_n % num_waves == 0`.

**Action:** `block_m` becomes a `kmd_fields` entry and a `metadata_fields` entry;
`GFX950_ATTENTION_DENSE_BLOCK_M` stops being `inline constexpr` and becomes a
descriptor-read value; `kernelMatches` compares it like any other baked shape field.

### 1.2 `wide_lds_dma` — new field, and it is a RULE (S1, silent)

`wide_lds_dma: bool = False` (line 262). The dispatcher COMPUTES it (gfx950.py:84-93):

```python
wide_lds_dma = (persistent and hdim_q == 128 and hdim_v == 128
                and dtype in ("fp16","bf16") and mask_type != 0
                and sw == 0 and not use_sinks and not ragged)
```

This is the 4a transcription trap verbatim. **Hand-writing `False` mislabels the binary**
for every shipped shape that satisfies the conjunction — which is most of them, since the
shipped set is hdim=128, bf16/fp16, causal, non-ragged, persistent. Per
`verify_variant_sets.py` invariant 4, `absent` and `explicitly false` are different
kernels whenever the policy would have answered true.

`__post_init__` lines 431-446 adds its own preconditions, which are *narrower* than the
dispatcher's rule and must hold for any variant carrying it: `persistent`,
`head_size == 128 and block_n == 64`, `not (ragged or varlen or paged)`,
`lds_k_group_pad == 8 and lds_v_row_pad == 32`.

`kernel_name()` line 566 appends `wdma`, so it is part of binary identity.

**Action:** add to `kmd_fields` + `metadata_fields`, and let `dispatch_parity.py` resolve
the rule per shape. Never transcribe a value.

### 1.3 `lds_v_row_pad` — new field, dispatcher-set from the geometry table

`lds_v_row_pad: int = 32` (line 175), set by the factory from
`geometry["lds_v_row_pad"]` (gfx950.py:105). Constant at 32 for the `default` geometry
the dispatcher uses; IR-live (line 764, 1440, 1474) and in `kernel_name()` on the D64
path only (line 544). Measured verdict in source: conflicts {0:30, 8:29, 16:11, 32:0},
TFLOPS {906, 901, 944, 953} — **32 is settled, not an axis**.

**Action:** add to `kmd_fields` so the descriptor states it rather than defaulting.
`sweep:` entry is `values: [32]` with the measured verdict.

### 1.4 Launch-surface `block` entry — FALSIFIED

Profile line 395 records the block as *"A CONSTANT 512 threads, unlike gfx942 where it
is derived from spec.block_m"*, and line 402 warns *"a future block_m knob would silently
invalidate it"*. #11627 added exactly that knob. `launch_surface.py --check` cannot catch
this — the surface IS declared; its *content* is now wrong.

**Action:** `kmd_fields: [block_m]` on the `block` surface, guard restated, and the
geometry test extended to cover `block_m=128 → 256 threads`.

### 1.5 New `persist_decode` values

`gqa_pair` and `gqa_pair_2phase` join `qb_major`/`hkv_major`/`auto` (line 329-339).
Each carries its own preconditions (lines 447-480), and `resolved_persist_decode`
(line 495-525) now prefers them when their CTA-count equations hold. Both are keyed on
`num_persistent == NQB*Hkv*B` (one-phase) or `== NQB*Hkv*B*gqa/2` (two-phase) — i.e.
they only fire when the request's `dense_num_persistent` happens to equal that product.

`kernel_name()` appends `gqapair` / `gqapair2` (lines 572-575), so these ARE distinct
binaries — but the field remains a resolved-by-property string, and `prepare()` still
does not branch on it (the grid is `(num_persistent, 1, 1)` for every decode).

**Verdict: still NOT a `kmd_field`, for the same reason as before** — no graph field maps
to it, and the engine does not read it. The profile's existing exclusion note stands, but
its wording must be updated to name the two new values. **UNSURE-1:** whether the
narrowing of `auto` changes which binary a given shipped shape resolves to. That is a
`dispatch_parity.py` question, answered by regenerating, not by reading.

### 1.6 New invariant: `block_n % num_waves == 0`

`supports_attention_dense` line 604. With `num_waves = block_m // 32`:
`block_m=256 → num_waves=8 → block_n ∈ {64,128}` both pass;
`block_m=128 → num_waves=4 → block_n ∈ {64,128}` both pass. Not currently binding for
the geometries on offer, but it is a **KNOB-SEL** row: it constrains the `(block_m,
block_n)` pairs a variant set may ship, and it did not exist before.

---

## 2. Constraint table — every rule, with a verdict

Buckets per `rocke-mining.md`: **SCOPE** (feature not shipped), **GRAPH** (graph fact vs
fixed constant), **GRAPH/BAKED** (graph fact vs baked metadata), **KNOB-SEL** (graph fact
vs a knob we choose), **SPEC** (both sides ours).

| # | Rule (source line) | Bucket | Enforced by |
|---|---|---|---|
| 1 | `dtype ∈ {bf16, fp16}` (265) | GRAPH/BAKED | `kernelMatches` dtype equality, via the `vocabulary:` mapping |
| 2 | `block_m > 0` (269) | SPEC | spec construction |
| 3 | `lds_v_row_pad >= 0 and % 8 == 0` (271) | SPEC | spec construction |
| 4 | `head_size ∈ {64, 128}` (279) | GRAPH/BAKED | `kernelMatches` |
| 5 | `lds_k_group_pad >= 0 and % 8 == 0` (286) | SPEC | spec construction |
| 6 | `block_n % 32 == 0` (292, 321) | SPEC | spec construction |
| 7 | ragged ⇒ `seqlen_q == seqlen_kv` (297) | GRAPH/BAKED | `kernelMatches` (ragged is a KMD field) |
| 8 | ragged ⇒ `not varlen` (302) | SCOPE | varlen declined wholesale |
| 9 | ragged ⇒ `sliding_window == 0` (304) | GRAPH/BAKED | `kernelMatches` |
| 10 | **aligned ⇒ `seqlen_q % block_m == 0` (307)** | **KNOB-SEL** | **`kernelMatches`: `Sq % $kernel.block_m == 0`. WAS a constant-256 test. NEW.** |
| 11 | aligned ⇒ `seqlen_kv % block_n == 0` (312) | KNOB-SEL | `kernelMatches`: `Skv % $kernel.block_n == 0` |
| 12 | `num_kv_heads > 0 and num_query_heads % num_kv_heads == 0` (316) | GRAPH | `graph_match` |
| 13 | persistent ⇒ `num_persistent > 0` (325) | SPEC | spec construction |
| 14 | `persist_decode ∈ {qb_major, hkv_major, gqa_pair, gqa_pair_2phase, auto}` (329) | SPEC | spec construction. **Two values are new.** |
| 15 | `sliding_window >= 0` (340) | GRAPH | `graph_match` |
| 16 | `sliding_window > 0 ⇒ causal` (343) | GRAPH | `graph_match` |
| 17 | `sliding_window % block_n == 0` (345) | KNOB-SEL | `kernelMatches` against `$kernel.block_n` |
| 18 | varlen ⇒ `not persistent`, `causal` (350-354) | SCOPE | varlen declined |
| 19 | `waves_per_eu ∈ [1, 8]` (355) | SPEC | spec construction |
| 20-29 | the whole `paged` block (360-426) incl. `batch != 1` refusal (406) | SCOPE | paged declined at `graph_match` |
| 30 | `use_sinks and paged` (427) / `use_sinks and varlen` (429) | SCOPE | both declined |
| 31 | **`wide_lds_dma ⇒ persistent`** (432) | GRAPH/BAKED | `kernelMatches` — new KMD field |
| 32 | **`wide_lds_dma ⇒ head_size==128 and block_n==64`** (434) | KNOB-SEL | `kernelMatches` |
| 33 | **`wide_lds_dma ⇒ not (ragged or varlen or paged)`** (436) | GRAPH/BAKED | `kernelMatches` |
| 34 | **`wide_lds_dma ⇒ lds_k_group_pad==8 and lds_v_row_pad==32`** (440) | SPEC | spec construction |
| 35 | **`gqa_pair ⇒ persistent and causal`** (451) | GRAPH/BAKED | implied by the descriptor's own resolved decode |
| 36 | **`gqa_pair ⇒ not (ragged or varlen or paged)`** (453) | GRAPH/BAKED | as above |
| 37 | **`gqa_pair ⇒ NQB even and GQA even`** (457) | GRAPH/BAKED | as above; `NQB = ceil(Sq/block_m)` |
| 38 | **`gqa_pair ⇒ num_persistent == NQB*Hkv*B`** (459) | GRAPH/BAKED | as above |
| 39 | **`gqa_pair_2phase ⇒ … GQA >= 2, num_persistent == NQB*Hkv*B*gqa/2`** (464-480) | GRAPH/BAKED | as above |
| 40 | **`block_m ∈ {128, 256}`** (594) | SPEC | spec construction |
| 41 | **`block_m % block_n == 0`** (599) | KNOB-SEL | pair constraint on the shipped set |
| 42 | **`block_n % num_waves == 0`**, `num_waves = block_m//32` (604) | KNOB-SEL | pair constraint on the shipped set |
| 43 | arch `!= gfx950` (585) | — | pack arch-prune, before the matcher |

Rows 31-42 are new with #11627. Rows 35-39 are enforced *transitively*: the descriptor
carries a concrete `persist_decode`-resolved binary, so a graph selecting it already
matches the shape fields those equations are written in terms of. **UNSURE-2:** whether
that transitivity is airtight for `gqa_pair`, whose equation involves `num_persistent`,
which IS a KMD field the matcher compares — likely yes, to be confirmed when the shipped
set actually contains a `gqa_pair` binary. Step-3 question.

---

## 3. Layout — unchanged by #11627

BSHD, derived from the address arithmetic in `build_attention_dense`: strides computed
from `Hq*D` / `Hkv*D`, no stride kernarg. V reuses K's base/stride; O reuses Q's. A graph
in another layout is read in-bounds and wrong, with no fault; K/V OOB is zero-filled by
the bounds-checked buffer loads (`buffer_rsrc` at 832-833, 1501-1502), which is also
silent. Only an undersized Q on the unguarded path faults.

`hasBshdStrides` in the pack already enforces this. **No change required.**

Buffer bounds ARE spec-derived (`B * Skv * Hkv * D * 2`), so they are a
`kernelMatches` equality obligation on `batch`, `seqlen_kv`, `num_kv_heads`,
`head_size` — all already KMD fields.

## 4. Grid / block / KMD fields they read

| Surface | Formula | KMD fields |
|---|---|---|
| grid, persistent | `(num_persistent, 1, 1)` | `persistent`, `num_persistent` |
| grid, default | `(ceil(seqlen_q / block_m), num_query_heads, batch)` | `seqlen_q`, **`block_m`**, `num_query_heads`, `batch` |
| block | `(num_waves * 64, 1, 1)` = **`(block_m * 2, 1, 1)`** | **`block_m`** |

`block_m` is new on both. Constants are no longer resolvable at C++ compile time.

## 5. ABI — unchanged by #11627

`attention_dense_signature` (2318-2342) declares slots **conditionally**:
base 5 = `(q_ptr, k_ptr, v_ptr, o_ptr, scale:f32)`;
`+sink_ptr` iff `use_sinks`; `+cu_seqlens_q, cu_seqlens_kv` iff `varlen`;
`+block_tables, kv_lens, block_table_stride` iff `paged`.

Every pointer maps to a graph tensor; **no pointer is synthesised**, so there is no
undeclared assumption to enforce in `graph_match` on this axis. The shipped set is
5-slot because sinks/varlen/paged are all declined.

## 6. Rejection checklist — implementation order for `graph_match`

**Tier 1 — silent wrong answers (these go first):**
1. non-BSHD strides on Q/K/V/O → `hasBshdStrides`
2. `sliding_window` present → the window convention is `W = L + 1` against the
   reference's `leftBound`; an off-by-one here is one wrong key on every masked graph
3. **`block_m` mismatch — NEW.** A graph whose `Sq % block_m != 0` selecting an aligned
   binary built at a different `block_m` gets a grid that does not cover it
4. **`wide_lds_dma` mismatch — NEW.** A descriptor mislabelled `False` names a binary
   that was compiled `True` (or vice versa); the LDS layout and PV traversal differ
5. `ragged` vs aligned mismatch — the aligned grid misses the partial last query block
6. batch/head-count/seqlen inequality — the baked buffer bounds are sized from them

**Tier 2 — faults / declines:**
7. `head_size ∉ {64, 128}`
8. GQA non-divisibility
9. `Skv % block_n != 0` on the aligned path with no matching variant

**Tier 3 — scoped out, each with a named reason:**
10. `use_sinks` — the GPU reference executor declines sinks; the CPU one computes them.
    Named gap, not a silent cut.
11. `varlen` — no reference executor
12. `paged` — no reference executor; additionally the spec requires `sliding_window > 0`
    and rejects `paged+persistent` and `batch != 1`
13. backward graphs — structurally excluded by their gradient tensors

## 7. Open rows for step 3

- **UNSURE-1** — does `auto`'s new pair-mapping preference change which binary a shipped
  shape resolves to? Answered by regenerating with `dispatch_parity.py`, not by reading.
- **UNSURE-2** — is `gqa_pair` applicability transitively enforced by the existing shape
  equalities, or does it need its own `kernelMatches` row?
- **Q for the PR author** (carried from the impact doc, still open): is `wide_lds_dma`
  intended as policy-resolved (as written) or as an exposed knob? Policy ⇒ descriptors
  carry the resolved value per shape. Knob ⇒ it is a fourth tuning axis and changes the
  sizing conversation.
- **`block_m` as a tuning axis.** `bm128` is described in-source as *"an explicit 4-wave
  experiment geometry"*, and the dispatcher only ever resolves `default` (bm256):
  `_DENSE_GEOMETRY = "default"`, hardcoded at gfx950.py:30. So `block_m` is CONSTANT
  across every dispatch decision — per RUNBOOK 4a, that makes it "a value the library
  ships", not an axis. It is nonetheless a **matcher field**, because a variant could be
  built at 128 and the matcher must not evaluate its tile predicate against 256.
  Those are two different questions and only the second is settled here.
