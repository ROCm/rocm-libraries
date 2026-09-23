[← MLA design doc index](../DESIGN.md)

## 7. hipDNN exposure plan

> **Note:** The hipDNN heuristics/dispatch layer is still being stood up, and rocKE's own
> dispatch surface moved during this spike: `library/api/` (the C++ `SdpaProblem` /
> `AotCatalog` / `SelectionConstraints` path this section was originally scoped
> against) was **deleted**, and selection is now Python under
> `library/dispatch/attention/`. This section is written against that path. Field names
> and the eventual packaging format still need confirming with the hipDNN team.

### 7.1 Op identifiers

MLA prefill and decode-absorb are distinct ops. `AttentionRequest.op` currently defaults
to `"attention"`, and the only comparison against it is in `_request_errors`
(`library/dispatch/attention/common.py`), which rejects anything else. That one function
is called by all seven unified candidate predicates (`generic`, `gfx942`, `gfx950`,
`gfx1250`) and by `attention_sweep_space`, so it is the gate that keeps a new op string
from reaching any candidate — but for exactly that reason it must **not** be widened in
place. Widening it would drop the op guard from every existing candidate at once, and the
only thing left rejecting an MLA request would be the `hdim_q != hdim_v` check that §7.2
also proposes to relax. MLA candidates should instead get their own
`_mla_request_errors` accepting only the MLA op strings, leaving `_request_errors` pinned
to `op == "attention"`; that keeps the two sets mutually exclusive by construction. A
second, non-gating hardcode needs fixing at the same time: `_kernel_id`
(`library/dispatch/attention/__init__.py`) emits `op="attention"` unconditionally, so
without a change there every MLA selection would be logged, hashed into
`KernelId.selection_key`, and recorded in tuning data under the SDPA op name.

| Op string | Kernel |
|---|---|
| `mla_prefill_fwd` | Prefill (compressed-KV + decoupled RoPE), pre-kernel + flash loop (§3) |
| `mla_decode_absorb_fwd` | Decode-absorb (weight-absorbed, q=1) |

### 7.2 AttentionRequest extensions

`AttentionRequest` (`library/dispatch/attention/common.py`) is the normalized request.
MLA geometry is additive; zero-defaults keep every existing caller on the standard path:

```python
kv_lora_rank : int = 0       # r_KV; 0 = standard SDPA, not MLA
q_lora_rank  : int = 0       # r_Q
qk_nope_dim  : int = 0       # d_nope
qk_rope_dim  : int = 0       # d_rope
mla_mode     : str = "none"  # "none" | "prefill" | "decode_absorb"
```

**On `hdim_q` / `hdim_v`.** These are already *separate fields* on `AttentionRequest`, so
MLA's asymmetry needs no new field — but five things downstream still collapse or reject
it, and all five are in scope for the implementation stories:

1. `_request_errors` rejects `hdim_q != hdim_v` outright ("only hdim_q == hdim_v is
   supported").
2. `_problem()` discards the distinction when it builds the kernel-side problem
   (`head_size=int(req.hdim_q)`), and `UnifiedAttentionProblem` carries a single
   `head_size`.
3. `AttentionSpec` (`library/dispatch/attention/common.py`) also carries a single
   `head_size` and composes it into `kernel_name()` as `hd{head_size}` — so an
   asymmetric MLA spec needs its own spec type or a second field, or two distinct MLA
   shapes would hash and name identically.
4. `UNIFIED_HEAD_SIZES = (64, 128, 256)` (`library/kernels/common/attention_unified.py`)
   excludes 192 by set membership, both in `supports_native_unified_attention` and via
   the `_UNIFIED_CAPABILITY` `ShapeRange`.
5. The per-arch admission gates `supports_tiled_2d` and `supports_tiled_3d`
   (`library/kernels/{gfx942,gfx950}/attention_tiled_{2d,3d}.py`) independently reject
   `head_size not in (64, 128, 256)` against a hardcoded literal, *not* against
   `UNIFIED_HEAD_SIZES`. Widening the constant does not widen these; they are four
   separate edits, and they are the layer an MLA kernel would have to re-implement
   rather than widen.

**Why 192 is the right value to admit**, stated on its merit rather than on compatibility
with any prior catalog: 192 is `d_nope + d_rope`, the width of the *score-side* contraction
(§2.2). It is the head dimension the QK product actually runs at, and it is unrelated to
the output width `d_V = 128` or to the decode kernel's `r_KV + d_rope = 576` memory
layout. Admitting it is widening a gate to a value the math requires, not a compatibility
carve-out.

### 7.3 Capability gating and candidate registration

Selection is capability-driven, in two stages: `KernelCandidate.admits()` runs the
declarative `Capability.check()` prefilter (arch, dtype, `ShapeRange`s over
`request.dims()`, features) and only then the residual predicate passed as `_supports`,
which returns `(bool, reason)`. `Capability` is contractually a *superset* of the
predicate — a constraint it cannot express stays in the predicate. Registration is
explicit, not an import side effect: an MLA module exposes `register(registry)` and is
added to the module tuple in `library/dispatch/attention/__init__.py`.

Two registration constraints are load-bearing. `CandidateRegistry.register` rejects a
candidate whose `family` differs from the registry's, and `ATTENTION_REGISTRY` is built
with `FAMILY == "attention_unified"` — MLA is not that family, so it wants its own
registry and dim vocabulary. `register` also rejects a capability constraining a dim
outside the registry's `dim_vocabulary`, and `Capability.check` reads values from
`request.dims()`; so the §7.2 fields are inert as selection keys until added to **both**
`AttentionRequest.dims()` and that vocabulary. A `ShapeRange("kv_lora_rank", ...)` added
without this raises at import ("constrains unknown dims"); added to the vocabulary but
not to `dims()`, it rejects every request ("dim not provided").

An MLA candidate's `Capability` declares arches, dtypes,
`ShapeRange`s and features. The existing unified capability is `_UNIFIED_CAPABILITY`
(`library/dispatch/attention/generic.py:49-57`):

```python
Capability(
    arches=known_arches(),
    dtypes=UNIFIED_DTYPES,
    shapes=(ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
            ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES)),
    supports_features=ATTENTION_FEATURES,
)
```

MLA needs its **own** capability rather than a widened unified one, for the same reason
the two ops are distinct: an `hdim_q` of 192 with `hdim_v` of 128 must not become
selectable for standard SDPA requests as a side effect. A separate capability is
necessary but **not sufficient**: `Capability` has no `op` field, so the op never
participates in the declarative prefilter. Both directions of the exclusion are carried
by the predicates — an MLA request is rejected by every unified candidate only because
`_request_errors` rejects a non-`"attention"` op, and a standard request is rejected by
MLA candidates only because the MLA predicate rejects `op == "attention"`. That is why
§7.1 keeps `_request_errors` pinned and gives MLA its own request-errors function rather
than widening the shared one. The shared *matching* logic (`Capability.check`, `admits`,
`CandidateRegistry.select`) is genuinely untouched; the shared *predicate helper* is not,
and that is the piece to keep separate. `ATTENTION_FEATURES` is
`{"causal", "sliding_window", "sinks"}` — MLA prefill needs `causal`; the decoupled-RoPE
and latent-KV behaviour is intrinsic to the op, not a feature flag.

### 7.4 Open questions regarding hipDNN integration

- [ ] Confirm preferred op string naming convention (`mla_prefill_fwd` vs
      `sdpa_fwd_mla_prefill` vs other).
- [ ] **Absorbed weights as graph inputs (blocking):** `W_abs`, `W_rope_proj`, and
      `W_UV` are **model-load-time constant GPU tensors** (computed once at startup,
      not per-request). They must be represented as persistent graph inputs in the
      hipDNN graph, not as AOT compilation constants — their values are known only
      after the model is loaded, not at kernel compilation time. Confirm how
      the graph adapter will expose them: as additional weight-type `IGraph` inputs,
      as opaque constant handles, or another mechanism. This is a blocking question
      for the prefill and decode-absorb implementations.
- [ ] Confirm whether the two-kernel decode structure (pre-step GEMM + flash loop)
      is expressed as a single fused op in the graph or as two separate ops with an
      intermediate tensor. The pre-step (`c_q · W_abs` — no transpose, §0) is a batched
      GEMM that produces `q_abs[B, H_q, r_KV]` — its lifetime is one decode step.
- [ ] Confirm whether the prefill crossover dispatch (§2.5: in-loop vs materialize,
      threshold ~200 tokens) is handled inside the kernel op or by the framework
      choosing between two ops.
- [ ] Confirm whether chunked-prefill (`softmax_lse` output) is in scope for the
      initial integration target.
- [ ] Confirm the AOT packaging timeline relative to implementation start, and what
      replaces the deleted `AotCatalog` path for shipping prebuilt MLA instances.

---

