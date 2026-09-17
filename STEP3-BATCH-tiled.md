# RUNBOOK step 3 — the batch. Proposals, not questions.

**This is sent, and the run PROCEEDS on the stated defaults.** Every item carries a default
derived and defended from stages 1-2. Correct any of them cheaply; nothing here blocks.
Each is marked as an assumption and carried to step 10.

D1-D6 are settled and are not re-opened. Three findings below **contradict the plan** (not
D1-D6) and are flagged loudly per the "stop and say so rather than quietly deviating" rule —
items F1-F3.

---

## Settled, restated for the record

| Item | Value |
|---|---|
| Engine name | `hipkernel:Gfx950AttentionTiled` — scoped, and byte-identical to the UED `name` |
| Arch | `gfx950` only, never wider |
| Dialect | `packaged`, `kind: rocke`. Splice points 1 and 2 **never** apply |
| Workspace | `none` — the 2D kernel allocates no scratch and the 18-slot ABI has no workspace slot |
| Builders shipped | `build_unified_attention_2d_tiled` only (D3) |
| Declined | `select_path()=="3d"` (D4), fastkv cohort, softcap, qq_bias, block-sparse, backward, fp8, dropout, stats |
| Stage-8 oracle | rocKE `ref_paged_attn` (D1) — **confirmed pure torch**, 79 scenarios |
| CI target | registered at both sites, correctly pinned, **inert until D1's repair lands** (D2) |

---

## F1 — the builder-body raises are NOT `graph_match` rows *(corrects plan §3.1)*

The plan calls the three builder-body raises "Tier-1 rejection-checklist rows for
`graph_match`". They are not, and putting them there would be dead code that reports green.

Verified: there are **four**, not three (`:1433` ValueError, `:1868` AssertionError, `:2799`
and `:4265` NotImplementedError). Each plan line number cited the `if` guard; the `raise` is
the next line. All four are gated on **descriptor-side** flags — `use_softmax_mfma_interleave`,
`use_sched_barrier`, `use_fp8_mfma_pv`, the fp8 K-loader payload. **No graph field reaches
them.** They fire at `hkp_pack` emission time, on our own descriptors.

**Default: they become step-5 build gates, not matcher rows.** A descriptor that trips one
fails to pack, loudly, which is the correct place.

## F2 — the tiled predicate does not enforce the 2D/3D split *(new; scope work we owe)*

`supports_native_unified_attention_tiled` answers for the tiled **family**, not the 2D
**path**. Measured on the real dataclasses, arch memo pinned, over a 12-shape decode grid:
**12 of 12 shapes that `select_path()` routes to `3d` returned `(True, 'supported')`.**

Taking the predicate at face value would size a 2D-only variant set from a corpus containing
every 3D shape — descriptors built for a path this engine does not ship.

**Default: enforce `select_path()=="2d"` ourselves**, as a first-class decline with a named
reason, in `tools/tiled_parity_adapter.py`. rocKE's own predicate runs **first** so a
genuinely unbuildable shape reports the kernel's reason rather than being masked by our
scope gate. D4's scope becomes mechanical instead of assumed.

## F3 — `UnifiedAttentionProblem` cannot be `request.class` *(corrects plan §6/P2)*

Plan P2 recommends pointing the profile's `request.class` at `UnifiedAttentionProblem`. It
cannot be, for a mechanical reason: `dispatch_parity.py:171-172` injects `fields["arch"]`
whenever the profile declares `arch:`, and the dataclass has no `arch` field —

```
TypeError: UnifiedAttentionProblem.__init__() got an unexpected keyword argument 'arch'
```

Every shape lands in `rejected` with a TypeError that reads like a corpus defect. Dropping
`arch:` is not a fix; five other tools read it.

Compounding it: the tiled predicate takes a **problem**, while `dispatch_parity.py:188`
passes a **spec** (`AttributeError: ... has no attribute 'use_fp8'`).

**Default: a thin adapter** (`tools/tiled_parity_adapter.py`) supplying a
`TiledAttentionRequest` that accepts-and-drops `arch`, and a predicate wrapper in the
`(spec, arch=)` shape the tool calls. It restates **no** rocKE logic: the spec still comes
from the production `_tiled_spec_from_problem` (D5), and applicability still comes from
rocKE's predicate. The originating problem rides on the spec under a private attribute —
verified invisible to `dataclasses.fields()` (46 names before and after), so no metadata
consumer can see it. Recomputing the problem from the spec was rejected: the mapping is
lossy (22 fields → 46, dropping `num_cus`/`target_ctas`/`max_seqlen_k`), so it would be a
second hand-written implementation of exactly the kind the skill says ships wrong.

---

## The variant set — and how it lands against real workload shapes

**Sizing sources, in priority order.** The dense corpus is **not reusable**: verified
`hdim_q ∈ {128, 64}` with **no 256 at all**, no `block_size`, no `num_seqs`, and
`block_size` is absent even from `_shape_key`'s dedup tuple. It cannot exercise the D256
cohort that D5's composition question is about.

1. **`ref_paged_attn`'s 79 scenarios** — the kernel team's own validated shapes, already
   carrying `block_size` and real varlen `seq_lens`. Confirmed distribution:

   | axis | distribution |
   |---|---|
   | `head_size` | 128:58, 256:**17**, 64:4 |
   | `block_size` | 16:58, 64:16, 32:5 |
   | `dtype` | fp16:48, bf16:31 |
   | `(Hq,Hkv)` | 13 distinct pairs, dominated by (16,2)=39 |
   | features | sliding_window 6, softcap 6, alibi 9, qq_bias 6, sinks 1 |

   These are the shapes stage 8 verifies against, so shipping variants for them means the
   numeric evidence and the shipped set describe the same kernels.
2. **`dnn-benchmarking` `Workloads/`** — enumerate every suite **before** pulling any, and
   report `served/declined/could-build` out of the **enumerated total** (8e denominator rule).
3. **A published tiled results CSV** — **UNVERIFIED that one exists.** The dense
   `compare_to_rocke_csv.py` hardcodes a 7-tuple key with no paging dimension, so it needs
   work either way. Not on the critical path; noted.

**Budget: under ~100 kernels for v1.** Packing cost scales with the compiled *shape*, not
just the count (dense measured ~8× per kernel at production lengths), and this kernel bakes
its loop trip counts the same way.

**Method: dispatcher parity FIRST** (`dispatch_parity.py` through the adapter), then only
deviations I can name and attach a measurement to. Every shape's spec comes from the
production resolver, so the D256 override fold is applied by rocKE for exactly its cohort
and never hand-transcribed.

## Exposed knobs, and which values ship AOT — two separate decisions

**Only `int`-typed fields can be a real knob**; `getCustomKnobs` silently drops non-int knobs
at plan-build time, discovered only against a real device. Of the spec's 46 fields, **34
bools and 2 strs are structurally ineligible.**

Eligible bare ints: `head_size, block_size, num_query_heads, num_kv_heads, sliding_window,
num_seqs, num_warps, kv_ring_depth, sched_barrier_mask, block_m_per_warp, kq_lds_pad_halves,
softmax_interleave_mode, softmax_interleave_groups`. `Optional[int]`: `waves_per_eu,
tile_size`.

**Default: expose the int-typed shape/geometry fields; ship AOT values from
dispatcher parity only.** No `--knobs` cross-product in v1 — a knob in the shipping set with
no isolation arm behind it is, in the runbook's words, a guess wearing evidence's clothes.
`knob_sweep.py --plan` runs before any deviation is proposed, and **never carries the dense
engine's knob set across** (that would import its exclusions in both directions).

**Hard constraint, non-negotiable:** never `--knobs` one of the seven D256 override fields
(`use_kq_lds_pad`, `kq_lds_pad_halves`, `use_mfma32_skip_legacy_qreg`, `use_k_single_buffer`,
`use_q_direct_reg`, `softmax_interleave_mode`, `use_mask_phase_split`). Pinning one
off-cohort silently builds a different binary — 7 fields, different kernel name, no error.
Note two of them (`kq_lds_pad_halves`, `softmax_interleave_mode`) are int-typed and would
otherwise look like perfectly good knobs. That is the trap.

## UMD vs `graph_match`

**`graph_match`**, per the default — one pack, one builder, no graph fact on which two packs
need to differ.

## The rejection checklist → the negative tests owed at 8b

Each gets a C++ applicability negative asserting the **reason**, not just the decline:
3D-routed shapes (F2), fastkv cohort, softcap (no schema field — G1), qq_bias (no schema
field — G3), block-sparse, dropout, fp8 descale, stats/max/sum_exp, additive `attn_mask`
bias, `implementation == COMPOSITE`, non-BSHD Q/O strides, `block_size ∉ {16,32,64}`,
`head_size ∉ {64,128,256}`, mismatched or missing page tables.

---

## Two things I am NOT asking, because the tree answered them

- **The page-table → `block_size` derivation.** Resolved at 2a from three concurring sources
  (`graph_contract.md` §5 G2): `block_size = K.dim[2]` — the K/V tensor IS the paged
  container, and the page table resolves the block *index* only, never the block *size*.
- **Which entry point `dispatch:` names.** D5 settled it; §6 of the mining doc records the
  evidence that the production wrapper adds `_resolve_lds_budget`.

## The one thing genuinely worth a human's eye

**`reconcile_applicability.py`'s stage-9 gate is APPROXIMATE for this kernel, in a way it was
not for dense.** It scopes by `algorithm`, and **no registered rocKE candidate corresponds to
the generic tiled path**. The registry holds `unified_2d`/`unified_3d` (generic, priority 10),
`d256_decode`, `attention_dense` (opt-in), `d256_gfx950` (priority 5). Closest usable oracle
is `algorithm: unified_2d`; matching on `family` selects everything, since every rocKE
attention candidate shares `attention_unified`.

**Default: run it scoped to `unified_2d`, and say in the report that the gate is approximate
rather than reporting a clean gate I did not earn.** Flagged now because it is the one place
where a green number at stage 9 would mean less than it appears to.
