[← MLA design doc index](../DESIGN.md)

## 8. Test and bench plan

### 8.1 Correctness reference

A Python reference in `library/builders/mla/ref_mla_attn.py` implementing the expanded-form attention.

> **Layout note.** `library/builders/` is a Python package and every existing
> subdirectory (`common/`, `gfx942/`, `gfx950/`, `gfx1151/`, `gfx1250/`) carries an
> `__init__.py`; `mla/` currently holds only this document, so the first code change
> under it must add one or `builders.mla.ref_mla_attn` will not import. `mla/` is also
> the first *family*-scoped rather than arch-scoped builders directory — deliberate,
> since the reference is arch-neutral — but the arch-specific parity and bench entry
> points still belong under `builders/gfx942/attention/` and
> `builders/gfx950/attention/` next to their siblings, not under `mla/`.

Weights are stored `[in, out]` (§0), so every up-projection below is a plain `@`
with no `.T`; the only transposes are the score-forming contractions, which are
written as `einsum` to keep the `H_q` axis explicit.

`ref_mla_prefill` takes `q_latent` and `W_UQ` because it models the **whole op** —
pre-kernel plus flash loop (§3) — not the flash kernel alone. It is a numerical
reference, so it does not reproduce the kernel split; the parity gate compares the op's
output. Where the reference *does* mirror kernel structure is the decode accumulation
order, for the reason given below. Both functions take `scale` as an explicit argument,
supplied by the caller from the same source the kernel gets it from — see the note in §0
and the tolerance discussion below.

```python
# `scale` is a REQUIRED argument to both references, never a module default: the gate
# below is only meaningful if the reference and the kernel receive the *same*
# host-supplied value. A geometry-derived default would silently agree with a kernel
# that derived it the same wrong way. (1/sqrt(192) is the value for the models in §1 —
# NOT 1/sqrt(576), and not the right value at all under YaRN rope scaling. See §0.)

def ref_mla_prefill(q_latent, c_kv, k_rope, W_UQ, W_UK, cu_seqlens, positions, scale,
                    causal=True):
    # q_latent [total_q, r_Q] (no head axis); W_UQ [H_q, r_Q, d_nope + d_rope]
    q = torch.einsum("tr,hro->tho", q_latent, W_UQ)       # [total_q, H_q, d_nope+d_rope]
    q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)    # [t,h,128], [t,h,64]
    q_rope = apply_rope(q_rope, positions)                # k_rope is stored post-rotation (§2.1)
    # expand all KV positions — W_UK is PER HEAD [H_q, r_KV, d_nope + d_V] (§0),
    # so the shared latent c_kv expands to a different K_nope/V for every head
    K_nope = torch.einsum("sr,hro->sho", c_kv, W_UK[:, :, :d_nope])   # [S, H_q, d_nope]
    V      = torch.einsum("sr,hro->sho", c_kv, W_UK[:, :, d_nope:])   # [S, H_q, d_V]
    scores = scale * (torch.einsum("thd,shd->ths", q_nope, K_nope)
                      + torch.einsum("thd,sd->ths", q_rope, k_rope))  # k_rope is head-shared
    p = softmax(causal_mask(scores, cu_seqlens) if causal else scores, dim=-1)
    return torch.einsum("ths,shv->thv", p, V)             # [total_q, H_q, d_V]

def ref_mla_decode_absorb(c_q, c_kv, k_rope, W_abs, W_rope_proj, W_UV, positions, scale):
    # c_q [B, r_Q] (no head axis); W_abs [H_q, r_Q, r_KV]; W_rope_proj [H_q, r_Q, d_rope];
    # W_UV [H_q, r_KV, d_V] — per head (§0)
    q_abs  = torch.einsum("br,hrk->bhk", c_q, W_abs)          # [B, H_q, r_KV]
    q_rope = torch.einsum("br,hrd->bhd", c_q, W_rope_proj)    # [B, H_q, d_rope]
    q_rope = apply_rope(q_rope, positions)                    # rotate at the current position
    scores = scale * (torch.einsum("bhk,sk->bhs", q_abs, c_kv)
                      + torch.einsum("bhd,sd->bhs", q_rope, k_rope))
    p   = softmax(scores, dim=-1)                             # [B, H_q, S]
    acc = torch.einsum("bhs,sk->bhk", p, c_kv)                # latent-space accum [B, H_q, r_KV]
    return torch.einsum("bhk,hkv->bhv", acc, W_UV)            # [B, H_q, d_V]
```

The decode reference accumulates in latent space and applies `W_UV` last,
**mirroring the kernel's structure** (§4) rather than the mathematically equivalent
per-token expansion. Keeping the reference's contraction order aligned with the
kernel's is what makes the tolerance below meaningful: a parity gate that reduces in
a different order absorbs part of the error budget it is supposed to be measuring.

Tolerance gate: `max_abs ≤ 4e-2` bf16 (matching the existing unified attention gate).
The sweep should span the full `kv_len` range of §8.2 (512 … 32768), for two reasons
that are properties of this design rather than of the tolerance: long `kv_len` is what
exercises the 3D split-KV reduce with more than one segment per query row, and it is
where the fp32 latent accumulator (§4) has the deepest reduction chain. This gate does
**not** discriminate a wrong softmax `scale` (§0) by `kv_len` — a $1/\sqrt{576}$ vs
$1/\sqrt{192}$ error is a 1.73× temperature change that should fail at every length;
the defence against it is that reference and kernel take the same host-supplied
`scale`, not the sweep range.

### 8.2 Benchmark shapes

Four shape files: `mla_shapes.json` under
`library/benchmarks/{gfx942,gfx950}/attention/decode/` and `mla_prefill_shapes.json`
under the matching `attention/prefill/` directories. Decode sweep (`seqlen_q=1`):
kv_len in {512, 1024, 2048, 4096, 8192, 16384, 32768} at `batch = 1` for DeepSeek
V3/R1 and Kimi-K2, plus `batch ∈ {4, 8, 32}` at `seqlen_k ∈ {2048, 8192}` to bracket the
pre-step/flash-loop crossovers of §4 (`B ≈ 2.3` two-stage, `B ≈ 7` materialized).
`B = 4` sits **between** the two crossovers — past the two-stage form's but not the
materialized form's — which is where the two pre-step forms make their most divergent
predictions and so the most informative single point. `B = 8` is just above both, and
`B = 32` well above, together fixing the batch-dominated asymptote. `B = 1` brackets
from below but is the weakest point of the four: §4 flags it as launch-bound rather than
bandwidth-bound, so it does not test the model the other three test.
Two `seqlen_k` values suffice on the length axis — the flash loop is linear in `kv_len`
and the pre-step is invariant, so two points fix the line; but see §4 on the
`B = 32, kv_len = 8192` KV working set exceeding MALL.

Prefill sweep, **two families**:

1. **Full-prompt** — `seqlen_q = seqlen_k ∈ {512, 1024, 2048, 4096, 8192}` at
   `batch = 1`, plus `batch = 4` at the two short lengths.
2. **Chunked** — `seqlen_q ∈ {128, 256, 512}` × `seqlen_k ∈ {8192, 32768}` at
   `batch = 1`, plus `seqlen_q = 192` at `seqlen_k = 8192` (the one point inside the
   cited 171–228 band), plus `batch = 4` at `(seqlen_q, seqlen_k) = (256, 8192)`.

> **Family 1 alone cannot measure this design, and an earlier revision of this section
> shipped only family 1.** Every family-1 shape has $S_q = S_k \ge 512$, so every one of
> them sits on the **materialize** side of §2.5's threshold — the §3 flash loop, which is
> the kernel §9 scopes, is never measured in the regime it exists for. Family 2 fixes
> that and does three things family 1 cannot:
>
> - **It brackets the §2.5 threshold.** `seqlen_q = 128` is below the cited 171–228 band,
>   `192` inside it, `256` and `512` above — so the sweep confirms locally that a
>   crossover exists and which side wins at each end. It does **not** localize the
>   crossover better than the citation: the straddling points bound it to (128, 192),
>   width 64, against the citation's width 57. Keep ~200 as the dispatch default until a
>   denser sweep runs (§9 row 4).
> - **It is the only place $S_q \neq S_k$.** §3's input table carries `cu_seqlens_k`
>   separately from `cu_seqlens_q` precisely because chunked prefill decouples them, and
>   no family-1 shape exercises that. The causal mask against a $S_k \gg S_q$ context is
>   a distinct code path from the square-mask case, not a parameterisation of it.
> - **It is where the materialize path is supposed to lose.** §2.5's $1/S_q$ argument
>   and §3.1's footprint — **2.5 GiB** at this `seqlen_k`, 4× the 8192 figure — predict a
>   large gap at `(128, 32768)`. 2.5 GiB of per-layer scratch to serve 128 query tokens is
>   not a gap, it is infeasible, which is why §2.5 gates the path on a scratch budget.
>   If the two strategies land within noise **at the extremes** — `(128, 32768)` and
>   `(512, 8192)` — the §2.5 threshold is not doing work in this regime and the design
>   should be amended; three $S_q$ points at two $S_k$ values is enough for that endpoint
>   claim, not for a "within noise across the whole family" one. That is the cheapest
>   falsification available and it should be run before either kernel is tuned.
>
> `seqlen_k = 32768` matches the decode sweep's upper bound, so both kernels are
> exercised against the same maximum context.

**The prefill batch axis is narrower than
decode's on purpose, and it measures something else.** Decode needs batch because the
pre-step is batch-invariant while the flash loop is not, so the crossovers above only
appear across `B`. Prefill has no *batch* crossover (its $S_q$ crossover is §2.5's, and
that is swept on the `seqlen_q` axis above, not this one): its pre-kernel amortizes over
`total_q = batch × seqlen_q` (§3), so batch and `seqlen_q` are interchangeable for it and
a uniform-length batch sweep re-measures points the length sweep already covers, at
multiplied cost. What batch buys at prefill is coverage of the **packed-varlen path** —
`cu_seqlens_q` / `cu_seqlens_k` prefix sums and the per-sequence causal mask (§3) —
which `B = 1` exercises only trivially. Hence `B = 4` where that coverage is cheapest.
The case that most needs covering is *mixed* sequence lengths within a batch, which the
shape-file schema cannot express; that belongs with the runner work (§9).

All four files use `block_size: 16` (the repo default for paged KV); the `Bk` values in
§5 are kernel tile sizes covering 1–2 blocks each, not block sizes (see the note in §5.2).

**Dtype coverage and harness limits.** The two gfx942 files are **bf16 only** —
Phase 1 (§6). The two gfx950 files additionally carry `dtype: "fp8_e4m3"` entries for
Phase 2; they are **specification, not runnable input for Phase 1**:

- Each file carries `_dtype_note` and `_harness_note` fields recording, per file,
  which harness silently mis-types a shape and which silently parses zero shapes. Those
  notes are normative and travel with the data; they are not duplicated here. Net effect:
  **none of the four files is loadable by a current harness**, and two of the four
  failure modes are silent. Wiring MLA shapes into a runner is part of the kernel
  deliverable (§9), not a prerequisite of this doc.

Consequence: a Phase-1 bf16 run must select the `bf16` entries explicitly. Do not run
the fp8 entries against a bf16/fp16 harness and read the result as an fp8 number. Both
constraints are repeated as `_dtype_note` / `_harness_note` inside the JSON files so
they travel with the data.

### 8.3 Parity baselines

The parity harness lives under `builders/gfx942/attention/` and
`builders/gfx950/attention/` next to its siblings, not under `mla/` (§8.1 layout note —
only the arch-neutral reference goes there). It follows the same three-table
methodology as `library/builders/gfx950/attention/README.md`: three apples-to-apples
lanes — `auto` vs `auto`, `2d` vs `2d`, `3d` vs `3d` — each comparing the reference
backend against the rocKE kernel *within* that lane, so a selector difference is never
read as a kernel difference. Two external baselines should be included for meaningful
comparison:

| Baseline | Source | What to compare |
|---|---|---|
| **AITER Triton MLA** (`ROCM_AITER_TRITON_MLA`) | `aiter.ops.mla` | Primary comparison; AITER reports this as best on gfx942 |
| **TileLang MLA** | `github.com/tile-ai/tilelang` | Open-source; upstream reports 95% of AITER ASM on gfx942 in ~80 lines (§10.3); transparent tiling strategy |

AITER's own assembly decode kernel (`ROCM_AITER_MLA`) is the performance ceiling;
TileLang is the most transparent public reference for understanding tiling choices
that drive gfx942/gfx950 MLA decode performance.

---

