[← MLA design doc index](../DESIGN.md)

## 9. Implementation scoping

| # | Deliverable | Arch | Dtype | Spec |
|---|---|---|---|---|
| 1 | MLA prefill — chunked (in-loop expansion) | gfx942 | bf16 | §3, §5.1 prefill tiling |
| 2 | MLA prefill — chunked (in-loop expansion) | gfx950 | bf16 | §3, §5.2 prefill tiling |
| 3 | MLA prefill — full-prompt (materialize) | arch-independent | bf16 | **§3.1** + §7.2 spec-layer edits |
| 4 | Prefill strategy dispatch (§2.5 footprint bound + $S_q$ threshold) | arch-independent | — | **§2.5**, `library/dispatch/attention/` |
| 5 | MLA decode-absorb | gfx942 | bf16 | §4, §5.1 decode tiling |
| 6 | MLA decode-absorb | gfx950 | bf16 | §4, §5.2 decode tiling |
| 7 | MLA prefill | gfx950+ | fp8 e4m3 | §6 fp8 plan, §3 struct |
| 8 | MLA decode-absorb | gfx950+ | fp8 e4m3 | §6 fp8 plan, §4 struct |

Each bf16 kernel delivers: kernel impl + parity gate + bench run against
`mla_shapes.json`. The fp8 kernels additionally deliver the fp8 KV cache dequant path.

> **Rows 3 and 4 are new in this revision, and their absence was a defect.** §2.5 has
> always specified two prefill regimes and required a dispatch heuristic between them,
> but an earlier revision of this table scoped only the in-loop kernel. That left the
> design mandating a two-branch dispatch with nothing to dispatch *to* on the
> full-prompt side, while §8.2's bench plan measured only that side — three sections
> implying three different scopes. Rows 3 and 4 close it.
>
> **Row 3 is not a kernel row.** It writes no tiling: it is the per-head expansion GEMM
> plus §7.2's five spec-layer edits, after which the *existing* unified attention runs
> the shape at `(hdim_q, hdim_v) = (192, 128)`. It is listed arch-independent for that
> reason — the arch-specific tiling it lands on is already shipped. It is also the
> cheapest of the eight and the one that unblocks a number on the §8.2 family-1 shapes
> soonest; schedule it accordingly, but see the padding warning in §3.1 before starting.
>
> **Row 4 is two conditions, not one threshold.** The heuristic takes
> $(S_q, S_k, H_q, \texttt{dtype})$ and a scratch budget, not a scalar $S_q$ cut. The
> footprint bound of §2.5 — $S_k \cdot H_q \cdot (d_{\text{nope}} + d_{\text{rope}} +
> d_V) \cdot \texttt{sizeof(dtype)}$ against the budget — is a **hard admissibility
> gate** on the §3.1 branch; the $S_q$ threshold is a tunable that only arbitrates below
> it. Keying on $S_q$ alone routes `(512, 32768)` to a 2.5 GiB materialization.
>
> **Row 4 also depends on rows 1–3 and on a measurement.** The threshold it implements is
> a Hopper citation until §8.2's family-2 sweep gives it a local value (§2.5). Ship rows
> 1–3 with a provisional constant, then set it from the sweep; do not treat ~200 as
> settled. If family 2 shows the two branches within measurement noise at both extremes,
> row 4 collapses to the footprint gate alone and one of rows 1/3 should be dropped from
> the design — record that outcome rather than shipping a dispatch that chooses between
> equivalents.

#### Known implementation traps

Four errors that an implementation draft has already made, all of which pass a naive
parity gate because they are mirrored into the reference. Each is a spec violation
stated elsewhere in this doc; they are collected here because §9 is what an implementer
reads.

1. **`W_UK` / `W_UV` are per head — dropping the head axis is not a simplification, it
   is a different operator.** DeepSeek-V2 §2.1.2 defines
   $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$, so
   $k^C_t = W^{UK} c^{KV}_t$ has width $d_h \cdot n_h = 16384$, not $d_h = 128$ (§0,
   §2.3, §3 steps 6–7, §3.1, §8.1). A head-shared `[r_KV, d_nope]` up-projection makes
   the expanded K/V shareable across heads and every downstream cost model wrong. If the
   *reference* also drops the axis, the parity gate cannot detect it — check the
   reference's shapes against §8.1 before trusting a green gate.
2. **$K_{\text{rope}}$ *is* head-shared — and it is the only part of K that is.** It is
   $\mathbb{R}^{d^R_h}$ per token, broadcast across heads (§2.1). Getting item 1 right
   by giving `k_rope` a head axis is the opposite error.
3. **RoPE is not optional on the query side.** $K_{\text{rope}}$ is stored
   *post*-rotation (§2.1), so `positions` is a required input and the query rotation
   must happen (§3 step 3, §4 pre-step). A reference that omits it agrees with a kernel
   that omits it.
4. **`scale` is host-supplied, and the kernel ABI wants `scale · log₂(e)`, not
   `log₂(scale)`.** The value is $1/\sqrt{d_{\text{nope}} + d_{\text{rope}}} =
   1/\sqrt{192}$ (§0; DeepSeek-V2 Eq. 18 divides by $\sqrt{d_h + d^R_h}$), never
   $1/\sqrt{576}$. The `scale_log2` parameter every rocKE attention kernel takes is the
   *base-2-exponent form* used by the `exp2` softmax — see
   `library/builders/common/parity_fmha_extended.py` (`math.log2(math.e) /
   math.sqrt(head_size)`) and `library/tests/differential/numeric_attention.py`.
   `log2(scale)` is a different number ($-3.79$ against $0.104$ at $\sqrt{192}$) and
   fails every shape, which makes it a useful smoke test: if a first correctness run is
   uniformly wrong rather than marginally wrong, check this before the kernel.

**The pre-kernels are deliverables too.** Both ops now normatively span two device
kernels (§3 step 2, §4 pre-step), and neither pre-kernel appears in the table above.
Before implementation starts, resolve for each: does it reuse an existing rocKE GEMM
instance — name it, and it costs no new row — or is it a new instance, in which case it
is a ninth and tenth row here under the DoD below? The batched-over-`H_q`
`[total_q, r_Q] × [r_Q, 192]` and `[B, r_Q] × [r_Q, 512]` shapes are not obviously
covered by an existing universal-GEMM candidate; assume they are not until checked.

**Definition of done — Python only, no C++ mirror.** MLA follows the dense attention
kernels: it is authored in the Python engine alone. There is **no** `platform/cpp/`
mirror for any row above, and no `check_byte_identity.py` obligation — the dual-engine
byte-identity gate does not apply to this family. (§5's and §11's references to
`platform/cpp/instances/gfx950/…` are citations of existing shipped code for the layout
and `ds_read_tr` techniques they demonstrate, not work items.)

What each row still owes, in the same change:

1. the Python builder under `library/kernels/<arch>/`;
2. wiring into the **MLA registry** (§7.3 — a separate `CandidateRegistry` with its own
   `family` and dim vocabulary, *not* `ATTENTION_REGISTRY`, which
   `CandidateRegistry.register` would reject on the family mismatch);
3. dispatch entry under `library/dispatch/attention/` (row 4 is the strategy heuristic
   itself; rows 1–3 and 5–8 each need to be *reachable* from it);
4. a **golden `.ll`** case under `library/tests/golden/`, blessed in the same change and
   re-blessed with the diff reviewed on any intentional IR change;
5. parity emit cases under `library/tests/parity/`, plus the numeric parity run of §8.3;
6. the bench-shape wiring of §8.2 and the support-matrix / doc updates for the new
   family.

Any new IR op or spec trait required by the latent-space accumulator, the decoupled
`hdim_kv` / `hdim_out` descriptor, the decode `BLOCK_H` head remap (§4), the *separate*
prefill `BLOCK_H` head-block grid dim0 (§3 — an independently-sized second spec knob,
not the §4 one), or the XCD-aware grid mapping (§4) lands in
`platform/python/rocke/core/lower_llvm.py` only. It must not regress the goldens of any
*other* family that shares those paths — run `library/tests/golden/` and
`platform/tests/` before and after. This docs-only spike carries no such obligation; the
implementation PRs do.

---

