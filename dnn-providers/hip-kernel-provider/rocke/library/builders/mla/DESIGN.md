# MLA kernel family — design doc

> **Status:** Design spike (no kernel code). DoD = this doc approved. No implementation
> in scope, and **nothing here is measured**: the µs figures in §4 and the WG/CU figures
> in §5 are analytical estimates, labelled as such at each use.
>
> **Revision 2 (AICK-1502) — prefill scope correction.** Revision 1 specified two prefill
> regimes in §2.5 but scoped only one of them in §9, and gave §8.2 a bench plan whose
> every shape belonged to the *unscoped* regime. This revision adds §3.1 (the
> full-prompt materialize path), rows 3–4 of §9's table (that path plus the §2.5
> dispatch), and §8.2's chunked shape family — the only shapes that exercise the §3
> kernel in the regime it exists for, and the sweep that turns §2.5's ~200 threshold
> from a citation into a measurement. It also adds a trap list to §9 and paper citations
> to §11. **No equation or geometry value changed**; §0–§2 and §4–§5 are re-verified
> against the primary sources now listed in §11 and stand as written.

Covers DeepSeek V2/V3/V3.1/R1 and Kimi-K2. The kernel family splits into
two distinct kernels (prefill and decode-absorb) and a separate fp8 phase for
gfx950+. All sections below are specifications; nothing is a tuning history.

---

## Contents

Each numbered section is a separate file under [`design/`](design/). Section
numbers are stable — prose and benchmark configs cite them as `§N`.

- [0. Notation](design/00-notation.md#0-notation)
- [1. MLA geometry and model variants](design/01-geometry.md#1-mla-geometry-and-model-variants)
- [2. Math and data layout](design/02-math-and-layout.md#2-math-and-data-layout)
  - [2.1 KV cache layout](design/02-math-and-layout.md#21-kv-cache-layout)
  - [2.2 Full attention score for one query head $h$, position $i$](design/02-math-and-layout.md#22-full-attention-score-for-one-query-head-h-position-i)
  - [2.3 Prefill: latent expansion path](design/02-math-and-layout.md#23-prefill-latent-expansion-path)
  - [2.4 Decode: weight absorption path](design/02-math-and-layout.md#24-decode-weight-absorption-path)
  - [2.5 Prefill strategy: absorption vs materialize crossover](design/02-math-and-layout.md#25-prefill-strategy-absorption-vs-materialize-crossover)
  - [2.6 Online softmax](design/02-math-and-layout.md#26-online-softmax)
- [3. Prefill kernel specification](design/03-prefill-kernel.md#3-prefill-kernel-specification)
  - [3.1 Full-prompt prefill: the materialize path](design/03-prefill-kernel.md#31-full-prompt-prefill-the-materialize-path)
- [4. Decode-absorb kernel specification](design/04-decode-absorb-kernel.md#4-decode-absorb-kernel-specification)
- [5. Per-arch tiling and LDS budget](design/05-tiling-and-lds-budget.md#5-per-arch-tiling-and-lds-budget)
  - [5.1 gfx942](design/05-tiling-and-lds-budget.md#51-gfx942)
  - [5.2 gfx950](design/05-tiling-and-lds-budget.md#52-gfx950)
- [6. Dtype plan](design/06-dtype-plan.md#6-dtype-plan)
- [7. hipDNN exposure plan](design/07-hipdnn-exposure.md#7-hipdnn-exposure-plan)
  - [7.1 Op identifiers](design/07-hipdnn-exposure.md#71-op-identifiers)
  - [7.2 AttentionRequest extensions](design/07-hipdnn-exposure.md#72-attentionrequest-extensions)
  - [7.3 Capability gating and candidate registration](design/07-hipdnn-exposure.md#73-capability-gating-and-candidate-registration)
  - [7.4 Open questions regarding hipDNN integration](design/07-hipdnn-exposure.md#74-open-questions-regarding-hipdnn-integration)
- [8. Test and bench plan](design/08-test-and-bench.md#8-test-and-bench-plan)
  - [8.1 Correctness reference](design/08-test-and-bench.md#81-correctness-reference)
  - [8.2 Benchmark shapes](design/08-test-and-bench.md#82-benchmark-shapes)
  - [8.3 Parity baselines](design/08-test-and-bench.md#83-parity-baselines)
- [9. Implementation scoping](design/09-implementation-scoping.md#9-implementation-scoping)
  - [Known implementation traps](design/09-implementation-scoping.md#known-implementation-traps)
- [10. State of the art — public MLA kernel implementations](design/10-state-of-the-art.md#10-state-of-the-art--public-mla-kernel-implementations)
  - [10.1 AITER (AMD Inference Toolkit)](design/10-state-of-the-art.md#101-aiter-amd-inference-toolkit)
  - [10.2 FlashMLA (DeepSeek)](design/10-state-of-the-art.md#102-flashmla-deepseek)
  - [10.3 TileLang MLA](design/10-state-of-the-art.md#103-tilelang-mla)
  - [10.4 CK (Composable Kernels)](design/10-state-of-the-art.md#104-ck-composable-kernels)
  - [10.5 FlashInfer ROCm MLA](design/10-state-of-the-art.md#105-flashinfer-rocm-mla)
  - [10.6 SGLang weight absorption](design/10-state-of-the-art.md#106-sglang-weight-absorption)
- [11. References](design/11-references.md#11-references)

---

