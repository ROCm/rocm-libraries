# RFC 0017 Worked Example: SDPA as a UKD

This document is the long-form companion to
[RFC 0017: Universal Kernel Descriptors](../0017_UniversalKernelDescriptor.md). It carries the
full worked example the main RFC summarises: the complete matcher for a real SDPA forward
kernel, the mask-mode classifier encoded as criteria data, one accept and two declines traced
end to end, the dispatch geometry for both performance cohorts, and the engine, metadata
schema, and two kernel packs that bind them.

Descriptor semantics, the criteria operator vocabulary, and the expression language are defined
in the main RFC; this document only uses them.

## Table of Contents

1. [One Matcher, Evaluated Per Candidate Kernel](#1-one-matcher-evaluated-per-candidate-kernel)
2. [The Matcher](#2-the-matcher)
3. [Encoding the Mask Classifier](#3-encoding-the-mask-classifier)
4. [One Accept, Two Declines](#4-one-accept-two-declines)
5. [Dispatch Geometry from `$kernel.*`](#5-dispatch-geometry-from-kernel)
6. [The Engine, Metadata, and Two Kernel Packs](#6-the-engine-metadata-and-two-kernel-packs)
7. [What an Author Actually Writes](#7-what-an-author-actually-writes)

---


The dense flash-attention prefill kernel productized in
[PR #9480](https://github.com/ROCm/rocm-libraries/pull/9480) collapses into a matcher, a dispatch
descriptor, an engine, and a small kernel vector. It lives in the HIP kernel provider's rocKE tree
as `kernels/gfx950/attention_dense.py`, and it is gfx950-only, bf16/fp16, causal or full, with no
paging. Three symbols carry its contract: `AttentionDenseSpec`, the frozen compile-time spec;
`build_attention_dense`, the builder that emits the kernel from one; and
`supports_attention_dense`, which validates a spec. This document shows what ingesting it
looks like.

The kernel is real but not yet hipDNN-reachable, which is what makes it a useful example. Today it
is reached only through rocKE's own Python `CandidateRegistry`, which takes a normalized
`AttentionRequest` dataclass instead of a hipDNN op graph, so bringing it into hipDNN by hand would
mean writing another `SdpaGraphAdapter`-style C++ class to enforce its constraints. The
descriptors below express those same constraints as data, and adoption becomes a handful of
descriptor files.

Every gate below traces to a real condition, either in `AttentionDenseSpec.__post_init__`, which
validates the spec itself, or in the dispatch candidate's `support` function, which decides
whether a request reaches this kernel at all. Showing *decline* correctly matters more than
showing *accept*, so this document walks through one accept and two distinct declines.

## 1. One Matcher, Evaluated Per Candidate Kernel

This family's real dispatch is one function: it checks the request's arch, dtype, and feature
flags, then calls `supports_attention_dense`, which validates the full spec. There is no separate
catalog-matching stage. Section 5's `conv.tile_fit` already shows the mechanism: a matcher's
criteria can reference `$kernel.*` fields directly
(`divisible($y.n*$y.ho*$y.wo, $kernel.MPerBlock)`), so the same matcher, evaluated once per UKD in
a KDP's kernel vector, does the work a hand-written per-kernel selection function would otherwise
do. A graph either satisfies some UKD's instantiation of the matcher or it does not; there is no
second phase.

## 2. The Matcher

Grounded in `AttentionDenseSpec.__post_init__` and the dispatch candidate's `support` function,
this pack targets the aligned (non-ragged, non-varlen) dense causal path. Ragged and varlen
inputs are real, separately gated modes of the same kernel file, called out as an extension point
in §7.

```jsonc
{
  "schema": "hipdnn.umd/v1",
  "id":   "9c2a9e2e-8a2a-4a52-9d1a-9d9e6e5d9f11",
  "name": "SDPA forward (attention_dense family, gfx950) match",
  "nodes": [
    {"kind": "op", "id": "sdpa_fwd", "op": "sdpa_fwd",
     "operands": {
       // Required operands, named as `sdpa_attributes.fbs` names them, minus the `_tensor_uid`
       // suffix: a pattern binds the tensor, not its uid.
       "q": "$q", "k": "$k", "v": "$v",

       // Every optional tensor the schema declares, bound here and declined below. The set is
       // generic hipDNN SDPA vocabulary; none of it appears in AttentionDenseSpec.
       "attn_mask":     "$attn_mask?",
       "scale":         "$scale?",           // the scale-tensor form; accepted, see the scale gate
       "seq_len_q":     "$seq_len_q?",       // varlen: a real AttentionDenseSpec.varlen mode, not
       "seq_len_kv":    "$seq_len_kv?",      // wired into this candidate; see Section 7.
       "seed":          "$seed?",
       "offset":        "$offset?",
       "dropout_mask":  "$dropout_mask?",
       "dropout_scale": "$dropout_scale?",
       "page_table_k":  "$page_table_k?",
       "page_table_v":  "$page_table_v?",
       "block_mask":    "$block_mask?",
       "sink_token":    "$sink_token?",
       "descale_q":     "$descale_q?",
       "descale_k":     "$descale_k?",
       "descale_v":     "$descale_v?",
       "descale_s":     "$descale_s?",
       "scale_s":       "$scale_s?",
       "scale_o":       "$scale_o?",
       "stats":         "$stats?",
       "max":           "$max?",
       "sum_exp":       "$sum_exp?",
       "rng_dump":      "$rng_dump?",
       "amax_s":        "$amax_s?",
       "amax_o":        "$amax_o?"
     },
     "results": {"o": "$o"}}
  ],
  "criteria": {"and": [
    // --- graph-level: a prebuilt kernel serves one fixed compile-time shape ---
    {"!": ["$graph.is_override_shape_enabled"]},
    {"==": ["$graph.node_count", 1]},

    // --- 23 of the 24 optional tensors are refused outright. The 24th, the scale tensor, is
    //     served, and its gate is further down. ---
    {"not_present": ["$attn_mask", "$seq_len_q", "$seq_len_kv", "$seed", "$offset",
                     "$dropout_mask", "$dropout_scale", "$page_table_k", "$page_table_v",
                     "$block_mask", "$sink_token", "$descale_q", "$descale_k", "$descale_v",
                     "$descale_s", "$scale_s", "$scale_o", "$stats", "$max", "$sum_exp",
                     "$rng_dump", "$amax_s", "$amax_o"]},

    // --- rank 4, and cross-tensor dim equality by capture reuse: a name repeated across two
    //     shapes must bind the same value. ---
    {"shape": ["$q", ["batch", "num_heads",    "seqlen_q",  "head_size"]]},
    {"shape": ["$k", ["batch", "num_kv_heads", "seqlen_kv", "head_size"]]},
    {"shape": ["$v", ["batch", "num_kv_heads", "seqlen_kv", "head_size"]]},
    {"shape": ["$o", ["batch", "num_heads",    "seqlen_q",  "head_size"]]},

    // --- dtype and head_size AttentionDenseSpec accepts at all ---
    {"in": ["$q.dtype", ["HALF", "BFLOAT16"]]},
    {"==": ["$k.dtype", "$q.dtype"]}, {"==": ["$v.dtype", "$q.dtype"]}, {"==": ["$o.dtype", "$q.dtype"]},
    {"in": ["$q.head_size", [64, 128]]},

    // --- kernel-level pins. This family bakes shape and dtype into the binary, so the gates
    //     above are not sufficient: each candidate must also agree with the graph on every
    //     quantity it baked, or a d64 fp16 graph launches a d128 bf16 code object. `batch` is
    //     pinned like the rest because it sizes the K/V buffer bounds and, when persistent, the
    //     grid-stride trip count. Every name here is a KMD field. ---
    {"==": ["$q.dtype",     "$kernel.dtype"]},
    {"==": ["$q.head_size", "$kernel.head_size"]},
    {"==": ["$q.batch",     "$kernel.batch"]},
    {"==": ["$q.num_heads", "$kernel.num_heads"]},
    {"==": ["$k.num_kv_heads", "$kernel.num_kv_heads"]},
    {"==": ["$q.seqlen_q",  "$kernel.seqlen_q"]},
    {"==": ["$k.seqlen_kv", "$kernel.seqlen_kv"]},

    // --- GQA: num_heads must be a positive multiple of num_kv_heads ---
    {"divisible": ["$q.num_heads", "$k.num_kv_heads"]},

    // --- layout. The kernel bakes BSHD-contiguous strides at build time (stride_q_tok = Hq * D
    //     is a Python int, never read from an argument), so exactly one stride_order is legal:
    //     [0,2,1,3] over the axis order this pattern names. A family accepting either BHSD or
    //     BSHD would anchor `{"in": [..., [[0,1,2,3],[0,2,1,3]]]}`; anchor to the literal set
    //     the kernel accepts, whatever its size. ---
    "$q.packed", "$k.packed", "$v.packed", "$o.packed",
    {"==": ["$q.stride_order", [0, 2, 1, 3]]},
    {"==": ["$k.stride_order", "$q.stride_order"]},
    {"==": ["$v.stride_order", "$q.stride_order"]},
    {"==": ["$o.stride_order", "$q.stride_order"]},

    // --- compute precision, mma_core_mode, implementation: no per-family policy exists to check
    //     against, so these three are a proposed convention, not a verified constraint. ---
    {"==": ["$sdpa_fwd.compute_data_type", "FLOAT"]},
    {"==": ["$sdpa_fwd.mma_core_mode", "UNSET"]},
    {"in": ["$sdpa_fwd.implementation", ["AUTO", "UNIFIED"]]},

    // --- no ALiBi, no padding mask, no dropout. alibi_mask and padding_mask are plain bools
    //     defaulting false, so a bare negation suffices. dropout_probability is an optional
    //     float where unset and an explicit 0.0 both accept. ---
    {"!": ["$sdpa_fwd.alibi_mask"]}, {"!": ["$sdpa_fwd.padding_mask"]},
    {"or": [{"not_present": ["$sdpa_fwd.dropout_probability"]},
            {"==": ["$sdpa_fwd.dropout_probability", 0.0]}]},

    // --- stats and paged-KV attributes this family does not implement. generate_stats is
    //     tri-state: unset and false both accept, so it is defaulted then negated. ---
    {"!": [{"value_or_default": ["$sdpa_fwd.generate_stats", false]}]},
    {"not_present": ["$sdpa_fwd.max_seq_len_kv"]},

    // --- scale. hipDNN carries this three ways and this pack takes all three: the
    //     `attn_scale_value` attribute, a scale tensor holding a compile-time value, or a scale
    //     tensor marked `is_runtime_pass_by_value` whose value arrives per execution (RFC 0016).
    //     The UDD's `scalar` source reads whichever one the graph supplied, so the matcher
    //     constrains how many were supplied, not which: exactly one of the attribute and the
    //     tensor, since a graph carrying both has said two different things. ---
    {"or": [
      {"and": [{"present":     ["$sdpa_fwd.attn_scale_value"]},
               {"not_present": ["$scale"]}]},
      {"and": [{"not_present": ["$sdpa_fwd.attn_scale_value"]},
               {"present":     ["$scale"]}]}
    ]},

    // --- the mask-mode classifier. Section 3 derives it; it is spliced in here whole, the
    //     contradiction check and the mode disjunction together, as one element of this outer
    //     `and`. Splicing only the inner `or` would make the contradiction check a sibling
    //     disjunct, letting a graph with both deprecated causal booleans set pass by satisfying
    //     another arm. ---
    {"and": [
      {"!": [{"and": ["$sdpa_fwd.causal_mask", "$sdpa_fwd.causal_mask_bottom_right"]}]},

      {"or": [
        {"and": [{"==": ["$kernel.mask_mode", "causal_top_left"]},
                 "$sdpa_fwd.causal_mask"]},

        {"and": [{"==": ["$kernel.mask_mode", "causal_bottom_right"]},
                 "$sdpa_fwd.causal_mask_bottom_right"]},

        {"and": [{"==": ["$kernel.mask_mode", "none"]},
                 {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]}, -1]}]},

        {"and": [{"==": ["$kernel.mask_mode", "causal_bottom_right"]},
                 {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]},
                 {"==": ["$sdpa_fwd.diagonal_alignment", "BOTTOM_RIGHT"]}]},
        {"and": [{"==": ["$kernel.mask_mode", "causal_top_left"]},
                 {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                 {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]},
                 {"!": [{"==": ["$sdpa_fwd.diagonal_alignment", "BOTTOM_RIGHT"]}]}]},

        {"and": [{"==": ["$kernel.mask_mode", "sliding_window"]},
                 {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
                 {"!": [{"and": [{"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                                 {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]}, -1]}]}]},
                 {"!": [{"and": [{"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                                 {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]}]}]}]}
      ]}
    ]}
  ]}
}
```

Every gate above traces to a real, current constraint in `AttentionDenseSpec.__post_init__` or the
dispatch candidate's `support` function, except the three marked as illustrative convention
(compute precision, `mma_core_mode`, `implementation`), which have no real per-family policy to
check against because no hipDNN adapter for this family exists yet.

## 3. Encoding the Mask Classifier

`attention_dense`'s real mask surface is narrower than a generic SDPA classifier: the kernel has
exactly two structurally distinct modes it can be built for, `causal: bool` (top-left causal only;
there is no bottom-right-causal variant anywhere in the file) and the `causal=False` "full" case.
A third real spec field, `sliding_window: int`, requires `causal=True` and
`sliding_window % block_n == 0`, but the dispatch candidate declines any request with
`sliding_window` nonzero today, so this pack's kernel vector does not (yet) ship a sliding-window
instance.

The classifier this maps onto is a 5-input precedence machine (`causal_mask`,
`causal_mask_bottom_right`, `left_bound`, `right_bound`, `diagonal_alignment`; first match wins;
both deprecated booleans set is a contradiction). Inverting it into a boolean `or` over
`$kernel.mask_mode` is equivalent by exhaustive case analysis over the five inputs, and the
inversion holds for any kernel family: the classifier reasons purely about the graph's mask
attributes, not about which kernel serves the result. What changes per family is which modes its
kernel vector covers: only `none` and `causal_top_left` have real, buildable `AttentionDenseSpec`
instances today (`causal=False` and `causal=True` respectively). The classifier below keeps all
four legal `mask_mode` values structurally, so the KMD field stays open for a future
`causal_bottom_right` or `sliding_window` UKD with no matcher change, but this pack's own §6
vector populates only one of them, `causal_top_left`; §4's Case C shows what happens to a graph
that resolves to an unpopulated mode.

`left_bound` and `right_bound` are optional (`long = null`), and the C++ they mirror treats an
absent bound as unbounded, i.e. `-1`. Written out, each arm below would need
`{"or": [{"not_present": ["$sdpa_fwd.left_bound"]}, {"==": ["$sdpa_fwd.left_bound", -1]}]}`
wherever it means "left unbounded", correct but unreadable six times over. The arms below instead
use `value_or_default` to normalize an absent bound to `-1` first and then compare. The two
spellings are equivalent; this one is legible.

```jsonc
// The contradiction check and the classifier, derived below and spliced into §2's matcher as
// one conjunct. Pasting only the inner `or` would make the contradiction check a disjunct
// instead, so a graph with both deprecated causal booleans set could pass by satisfying
// another arm.
{"and": [
  {"!": [{"and": ["$sdpa_fwd.causal_mask", "$sdpa_fwd.causal_mask_bottom_right"]}]},

  {"or": [
    {"and": [{"==": ["$kernel.mask_mode", "causal_top_left"]},
             "$sdpa_fwd.causal_mask"]},

    {"and": [{"==": ["$kernel.mask_mode", "causal_bottom_right"]},
             "$sdpa_fwd.causal_mask_bottom_right"]},

    {"and": [{"==": ["$kernel.mask_mode", "none"]},
             {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
             {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
             {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]}, -1]}]},

    {"and": [{"==": ["$kernel.mask_mode", "causal_bottom_right"]},
             {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
             {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
             {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]},
             {"==": ["$sdpa_fwd.diagonal_alignment", "BOTTOM_RIGHT"]}]},
    {"and": [{"==": ["$kernel.mask_mode", "causal_top_left"]},
             {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
             {"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
             {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]},
             {"!": [{"==": ["$sdpa_fwd.diagonal_alignment", "BOTTOM_RIGHT"]}]}]},

    {"and": [{"==": ["$kernel.mask_mode", "sliding_window"]},
             {"!": ["$sdpa_fwd.causal_mask"]}, {"!": ["$sdpa_fwd.causal_mask_bottom_right"]},
             {"!": [{"and": [{"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                             {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]}, -1]}]}]},
             {"!": [{"and": [{"==": [{"value_or_default": ["$sdpa_fwd.left_bound",  -1]}, -1]},
                             {"==": [{"value_or_default": ["$sdpa_fwd.right_bound", -1]},  0]}]}]}]}
  ]}
]}
```

**Verdict.** No custom operation, no escape hatch, no new operator. The apparent need for one came
from copying the C++ shape, compute a mode then compare it, into a language that never needs to
name the value: the kernel's own metadata supplies the right-hand side, so the comparison
collapses into the predicate. That inversion is the general recipe for porting a classifier.

## 4. One Accept, Two Declines

All three cases share one graph: a single `sdpa_fwd` node, Q/K/V/O rank-4, bf16, BSHD-contiguous,
`batch=1`, `num_heads=16`, `num_kv_heads=2` (GQA ratio 8), `head_size=128`, `seqlen_q=seqlen_kv=2048`,
`compute_data_type=FLOAT`, `mma_core_mode=UNSET`, `implementation=AUTO`, no optional tensors, no
alibi/padding/dropout, `attn_scale_value=0.08838834764831845` (`1/sqrt(128)`) and no scale tensor.
They differ only in the field named.

**Case A: accept.** `causal_mask=true`. Every §2 gate passes, `mask_mode` resolves to
`causal_top_left`, and §6's non-persistent UKD declares exactly that alongside `head_size=128`,
`batch=1`, and `dtype="BFLOAT16"`, so every `$kernel.*` pin agrees with the graph.
`nqb = ceil(2048/256) = 8`; `work = nqb * num_heads * batch = 8*16*1 = 128`, below
`num_persistent`'s default of 256, so the real host-side rule in `_dense_spec` (`persistent = work
>= num_persistent` in `"auto"` mode) would itself pick the non-persistent cohort for this exact
shape. Applicable.

**Case B: matcher decline.** Same graph plus an additive attention bias, so `$attn_mask` is
bound. The `not_present` list fails the moment that operand is bound, before mask mode, dtype, or
layout are considered. Any of the other 22 refused-outright operands declines the same way, and
so does `$graph.is_override_shape_enabled`, which rejects before any tensor is bound because this
kernel bakes its shape at compile time and cannot serve a runtime-overridden one.

**Case C: catalog decline.** Same graph, but `causal_mask=false` and both bounds unbounded
(`left_bound=-1, right_bound=-1`), so `mask_mode` resolves to `none`. Every other gate still
passes, but neither of §6's two UKDs declares `mask_mode="none"`; both are built with
`causal=True`. No kernel in this pack's vector covers a full (non-causal) graph, so the engine is
not applicable to it, per §1's matcher-plus-`$kernel.*` mechanism. The gap is easy to fix: the
kernel builds a `causal=False` kernel today (`AttentionDenseSpec(causal=False, ...)` is a valid,
buildable spec), so adding a `mask_mode="none"` UKD to this pack needs one more
`kernelDescriptors` entry and no matcher change.

## 5. Dispatch Geometry from `$kernel.*`

`attention_dense`'s two real performance cohorts, `persistent=False` (one CTA per query block)
and `persistent=True` (a fixed-size grid-stride loop over all query blocks,
`attention_dense_grid` and `attention_dense_block`), use structurally different grid formulas:
the default case is a 3-D grid sized from graph dimensions, the persistent case a fixed 1-D grid
whose size is a per-kernel constant with no graph dimension at all. That is the second trigger
for a separate pack: same argument slots, different formula shape. This worked example uses
**two KDPs** sharing one matcher and one engine (§6), each with its own UDD.

The persistent UDD, the measured ~940-970 TFLOPS path (PR #9480):

```jsonc
{
  "schema": "hipdnn.udd/v1",
  "id":   "6a0f2d0e-2b6b-4a2b-8c9d-8b8b6f6e9a10",
  "name": "SDPA forward (attention_dense, persistent) dispatch",
  // fixed 1-D grid-stride launch; num_persistent is per-kernel metadata
  "grid":  {"x": "$kernel.num_persistent", "y": 1, "z": 1},
  // baked: num_waves = _BLOCK_M / 32 = 8 wave64s, a module constant this family never varies
  "block": {"x": 512, "y": 1, "z": 1},
  // LDS is sized statically inside the kernel body, so there is no dynamic shared-memory launch
  // argument. Whether the UDD schema should distinguish dynamic LDS from baked-in static LDS is
  // unsettled; 0 is the honest value here.
  "shared_mem_bytes": 0,
  "workspace_bytes": 0,
  "args_signature": [
    {"name": "q_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$q"}},
    {"name": "k_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$k"}},
    {"name": "v_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$v"}},
    {"name": "o_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$o"}},
    {"name": "scale", "kind": "scalar", "type": "f32",
      "source": {"from": "expr", "expr": {"value_or_default": ["$sdpa_fwd.attn_scale_value",
                                                               "$scale"]}}}
  ]
}
```

This is the real, complete 5-argument ABI (`attention_dense_signature`), with no stride or seqlen
scalars, because every shape quantity is a Python-level compile-time constant baked into the IR
rather than a runtime argument. This is the tradeoff "a prebuilt match is exact" argues for
([RFC 0017 §5](../0017_UniversalKernelDescriptor.md#5-matching-and-the-umd)): the kernel author's own docstring for
`AttentionDenseSpec` confirms it in the same words, that the functional fields "are baked into the
kernel as constants, this is a dense, statically-sized ABI".

The non-persistent UDD, referenced from a second KDP, differs only in the `grid` field (a formula
over graph dims instead of a `$kernel.*` constant) and shares the identical `args_signature`:

```jsonc
{
  "schema": "hipdnn.udd/v1",
  "id":   "d5e6c9a4-1f2a-4e3a-9a3b-2f7d0f6c4b21",
  "name": "SDPA forward (attention_dense, default grid) dispatch",
  // nqb = ceil(seqlen_q / 256). 256 is _BLOCK_M, a module constant this family never varies (the
  // kernel faults at other values), so it is a literal here rather than $kernel.block_m.
  "grid":  {"x": {"ceil_div": ["$q.seqlen_q", 256]}, "y": "$q.num_heads", "z": "$q.batch"},
  "block": {"x": 512, "y": 1, "z": 1},
  "shared_mem_bytes": 0,
  "workspace_bytes": 0,
  "args_signature": [
    {"name": "q_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$q"}},
    {"name": "k_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$k"}},
    {"name": "v_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$v"}},
    {"name": "o_ptr", "kind": "pointer", "source": {"from": "tensor", "ref": "$o"}},
    {"name": "scale", "kind": "scalar", "type": "f32",
      "source": {"from": "expr", "expr": {"value_or_default": ["$sdpa_fwd.attn_scale_value",
                                                               "$scale"]}}}
  ]
}
```

**What the scale binding shows.** Both UDDs resolve `scale` from whichever form the graph
supplied, with no branch in the dispatch: `value_or_default` reads the node attribute and falls
back to the scale operand, and the matcher has already guaranteed exactly one of them is there.
Whether that operand carries a baked value or a runtime one supplied through the variant pack is
the launcher's business, not the descriptor's, so a runtime scale costs this pack nothing: the
same kernarg is filled from a different place. This is the field-reference fallback form of
`value_or_default`, distinct from the literal one §3 uses.

The pack could accept more. The SDPA convention's implicit default (`1/sqrt(head_size)`, matching
what both `asm_sdpa_engine`'s `SdpaFwdPlanBuilder::buildPlan` and `attention_unified`'s dispatch
code compute) is expressible, since `rsqrt` is a defined operator. Requiring the scale to be
stated is a choice this pack makes, not a language limit.

## 6. The Engine, Metadata, and Two Kernel Packs

One matcher (§2), one engine, one KMD, shared across two KDPs because the two cohorts need
different UDDs (§5). The KMD carries `persistent` as a field so the two UKDs' metadata values
are distinct, not just their `id`s, satisfying the RFC's own KMD-uniqueness rule (§4: "every
kernel in the engine must produce a distinct key"). Each UKD's `kernel_source` is a rocKE adapter
invocation: the builder, plus the exact build values for that instance.

```jsonc
// --- KMD: the engine-wide metadata schema, shared by both KDPs below ---
{
  "schema": "hipdnn.kmd/v1",
  "id":     "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40",
  "name":   "attention_dense variant fields",
  "fields": [
    // every graph quantity the compiled binary bakes, so the matcher can pin the graph to it
    {"name": "head_size",      "type": "int",     "optional": false},
    {"name": "dtype",          "type": "string",  "optional": false},
    {"name": "batch",          "type": "int",     "optional": false},
    {"name": "num_heads",      "type": "int",     "optional": false},
    {"name": "num_kv_heads",   "type": "int",     "optional": false},
    {"name": "seqlen_q",       "type": "int",     "optional": false},
    {"name": "seqlen_kv",      "type": "int",     "optional": false},
    {"name": "mask_mode",      "type": "string",  "optional": false},
    // and the tuning axes: baked too, but not derivable from the graph, so they are catalog-key
    // material and heuristic features rather than things a `$kernel.*` criterion pins. Defaults
    // are the spec's own, so a UKD that omits one gets the binary the builder would produce.
    {"name": "persistent",     "type": "bool",    "optional": true, "default": false},
    {"name": "num_persistent", "type": "int",     "optional": true, "default": 256},
    {"name": "persist_decode", "type": "string",  "optional": true, "default": "qb_major"},  // resolved, never "auto"
    {"name": "interleave",     "type": "bool",    "optional": true, "default": false},
    {"name": "block_n",        "type": "int",     "optional": true, "default": 64},
    {"name": "waves_per_eu",   "type": "int",     "optional": true, "default": 2},
    {"name": "lazy_rescale",   "type": "bool",    "optional": true, "default": true}
  ]
}

// --- UHD: the engine's selection model, ranking the whole catalog. The two cohorts are one axis
//     of that catalog: `persistent` and `num_persistent` are ordinary features alongside the
//     shape, so the model choosing between cohorts is the model that will later choose among
//     block_n and decode variants. It trains on the sweep that produced the cohort split; the
//     host-side `work >= num_persistent` threshold is one decision boundary in that data. ---
{
  "schema": "hipdnn.uhd/v1",
  "id":     "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",
  "name":   "attention_dense forward selector",
  "kind":   "model",
  "model":  {"framework": "lightgbm", "artifact": "attention_dense/gfx950_fwd.bin"},
  "features_signature": [
    "$device.cu_count",
    "$kernel.persistent",
    "$kernel.num_persistent",
    "$kernel.block_n",
    "$q.seqlen_q",
    "$q.num_heads",
    "$q.batch",
    // the work term the host rule thresholded on, as an ordinary derived feature
    {"*": [{"ceil_div": ["$q.seqlen_q", 256]}, {"*": ["$q.num_heads", "$q.batch"]}]}
  ],
  "objective": "max"
}

// --- UED: the engine, referenced by both KDPs below ---
{
  "schema":    "hipdnn.ued/v1",
  "id":        "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",
  "name":      "attention_dense forward engine",
  "heuristic": "2b7a4e1c-6f3d-4a8e-9c2b-5d1f0a7e8b93",
  "metadata":  "9c53b6b0-9a1e-4b1d-8b5c-7e2d9a6f3c40"
}

// --- KDP 1: default grid ---
{
  "schema": "hipdnn.kdp/v1",
  "id":        "e3a1b7c4-5d92-4f06-8b3e-6c0d2a9f14b8",
  "name":      "attention_dense fwd d128 bf16 (default grid, gfx950)",
  "arch":      ["gfx950"],
  "matchers":  ["9c2a9e2e-8a2a-4a52-9d1a-9d9e6e5d9f11"],   // §2's matcher
  "engine":    "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",
  "dispatch":  "d5e6c9a4-1f2a-4e3a-9a3b-2f7d0f6c4b21",     // §5's default-grid UDD
  "kernelDescriptors": [
    {
      "schema": "hipdnn.ukd/v1",
      "id":   "3f8a6c1d-2e5b-4a9c-8d7e-1b6f4a3c9e02",
      "name": "attention_dense d128 bf16 causal (default grid, gfx950)",
      // The rocKE source kind: the builder plus this instance's build values. The adapter calls
      // build_attention_dense with exactly these, runs the AOT compile, and packs the code
      // object. Build keys are AttentionDenseSpec's own field names and dtype spelling; the
      // metadata below restates the same instance in the KMD's vocabulary.
      "kernel_source": {
        "kind":   "rocke",
        "source": "kernels/gfx950/attention_dense.py",
        "entry":  "build_attention_dense",
        "build":  {"batch": 1, "seqlen_q": 2048, "seqlen_kv": 2048,
                   "num_query_heads": 16, "num_kv_heads": 2, "head_size": 128,
                   "causal": true, "dtype": "bf16", "block_n": 64, "persistent": false}
      },
      "metadata": {"head_size": 128, "dtype": "BFLOAT16", "batch": 1, "num_heads": 16,
                   "num_kv_heads": 2, "seqlen_q": 2048, "seqlen_kv": 2048,
                   "mask_mode": "causal_top_left", "persistent": false, "block_n": 64},
      "priority":  0
    }
  ]
}

// --- KDP 2: persistent grid-stride ---
{
  "schema": "hipdnn.kdp/v1",
  "id":        "f70c8d25-4b16-4a3d-9e82-1a5b6f0c7d93",
  "name":      "attention_dense fwd d128 bf16 (persistent grid-stride, gfx950)",
  "arch":      ["gfx950"],
  "matchers":  ["9c2a9e2e-8a2a-4a52-9d1a-9d9e6e5d9f11"],   // the SAME matcher as KDP 1
  "engine":    "7d4c2a9e-3b6f-4e1a-8c5d-9a2f7b0e6c14",     // the SAME engine as KDP 1
  "dispatch":  "6a0f2d0e-2b6b-4a2b-8c9d-8b8b6f6e9a10",     // §5's persistent UDD
  "kernelDescriptors": [
    {
      "schema": "hipdnn.ukd/v1",
      "id":   "b1e7d4a0-9c3f-4e6b-8a1d-2f5c9b7e0a44",
      "name": "attention_dense d128 bf16 causal (persistent grid-stride, gfx950)",
      "kernel_source": {
        "kind":   "rocke",
        "source": "kernels/gfx950/attention_dense.py",
        "entry":  "build_attention_dense",
        // persist_decode carries the RESOLVED decode, never "auto". "auto" is a request to
        // compute one, not a third configuration. The KMD is both the catalog key and the
        // heuristic's feature space, so it records what the binary does: two UKDs both saying
        // "auto" but resolving differently would collide on the key and rank on a string rather
        // than the decode they were built with. Here (gqa=8, nqb=8, batch=1) "auto" resolves to
        // qb_major, since 8*8*1 < 2*256.
        "build":  {"batch": 1, "seqlen_q": 2048, "seqlen_kv": 2048,
                   "num_query_heads": 16, "num_kv_heads": 2, "head_size": 128,
                   "causal": true, "dtype": "bf16", "block_n": 64,
                   "persistent": true, "num_persistent": 256, "persist_decode": "qb_major"}
      },
      "metadata": {"head_size": 128, "dtype": "BFLOAT16", "batch": 1, "num_heads": 16,
                   "num_kv_heads": 2, "seqlen_q": 2048, "seqlen_kv": 2048,
                   "mask_mode": "causal_top_left", "persistent": true, "num_persistent": 256,
                   "persist_decode": "qb_major", "block_n": 64},
      // Measured preference for the persistent cohort (512 -> 853 TFLOPS, +70%), so priority sits
      // above the default cohort's wherever the UHD's score is not decisive, keeping the choice
      // off the meaningless id-byte tie-break.
      "priority":  10
    }
  ]
}
```

Neither UKD names an artifact file, because neither author chooses one: the adapter builds the
code object from the `build` values and owns its name. The two `kernelDescriptors` vectors above
*are* the AOT build list for this pack, so what gets compiled matches what is catalogued.

The generic launcher runs either KDP's kernel with no SDPA-specific code, and decline is handled
the same way whether it lands on the matcher's graph-only clauses or its `$kernel.*`-referencing
clauses.

## 7. What an Author Actually Writes

The example above is the whole system; this is the slice a kernel author touches. Adding one
kernel to an **existing** engine, the common case, is a single UKD:

1. Pick the build values for the instance. For a rocKE family that is one `AttentionDenseSpec`
   worth of fields: shape and dtype baked in, plus the tuning knobs. Nothing is compiled by
   hand; the build-only rocKE adapter runs the AOT build for every UKD in the pack and produces
   the code object the runtime loads.
2. Write one UKD: an `id`, a `name`, a `kernel_source` carrying the builder and those build
   values, and a value for each field the engine's KMD declares, distinct from every existing
   kernel's. The build values and the metadata describe the same instance in two vocabularies,
   the builder's and the engine's; the loader checks the metadata against the KMD.
3. Add it to a KDP's `kernelDescriptors`, or ship it as a drop-in pack. **This step alone does
   not make the kernel the default choice.** Under an unchanged KMD, the new UKD is loaded,
   catalogued, and immediately measurable through the engine's self-measure lever (the single
   autotune knob modelled on MIOpen provider's exhaustive-search flag), which benchmarks the
   catalog and caches the winner. The UHD's ordinary ranking picks it only once the heuristic is
   retrained to expose it, or if it is the only kernel matching the graph, in which case it runs
   because the engine already claimed the graph. This dormancy is intentional: a new kernel is
   testable the moment it is catalogued, and becomes the ranked default once a retrain (itself
   meant to be self-serve) picks it up.

The two shipped modes this family supports today that are not in this pack, `ragged=True`
(on-chip padding for non-256-multiple sequence lengths) and `varlen=True` (packed
variable-length batches via `cu_seqlens_q`/`cu_seqlens_kv`, a real, already-built 7-argument ABI
variant of the same kernel), each need a new UMD (different graph shape or different
optional-operand binding) plus a new UDD (a different `args_signature`), not a change to this
pack. `sliding_window`, gated off by the dispatch candidate today even though
`AttentionDenseSpec` itself supports it, needs only a KMD value and a UKD once that gate is
lifted, with no schema change. Both cases inherit the same dormancy rule as any other new UKD:
cataloguing is not selection.
