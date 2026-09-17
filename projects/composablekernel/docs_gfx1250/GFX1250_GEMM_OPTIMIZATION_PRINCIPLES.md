# gfx1250 GEMM Optimization Principles (AMD Internal Training) — Delta for CK Convolution

Source: `/home/sgundabo/MISA/docs/gfx1250_gemm_optimization_guide.md` — MISA's
own distillation of an **AMD internal presentation and training session**
("Practical GEMMs, their Performance, and how MI400 Addresses them",
slide deck `UnderstandingSystolicMMAs.pptx` by Hashem Hashemi, plus a
transcribed "MI400 Training 004" meeting). Unlike the four project
investigations already written into this folder (MISA's own empirical
bring-up, rocKE, hipConv, FlyDSL), this document is **pedagogical and
architectural**, not one project's empirical findings — it is AMD's own
explanation of *why* the gfx1250 GEMM techniques other projects independently
discovered exist, plus several techniques not yet captured anywhere in this
investigation series.

**Scope of this document**: only what is genuinely **new or additive**
relative to `GFX950_VS_GFX1250_COVERAGE.md`, `ROCKE_GFX1250_PORTING_NOTES.md`,
`MISA_GFX1250_CONV_LEARNINGS.md`, `HIPCONV_GFX1250_CONV_LEARNINGS.md`, and
`FLYDSL_GFX1250_CONV_LEARNINGS.md`. Content already covered by those (async
direct-to-LDS, the 5 bifurcated wait counters, TDM descriptor basics, split
barriers with `sched_barrier(0)`, the wave32 WMMA register-layout porting
rule) is **not** repeated here except where this source adds a genuinely new
detail or nuance.

## 1. Cache Scope and Temporal Hints — an entirely new tuning axis, not covered by any prior investigation

The guide states VMEM operations on gfx1250/MI450 carry two independent
hint fields not discussed anywhere in the four project investigations:

- **SCOPE** (coherence domain): `WGP`, `SE`, `DEV`, `SYS` — a finer-grained
  hierarchy than the CU/DEVICE/SYSTEM scope bits already found in CK's own
  `AmdBufferCoherenceEnum` (§9 of `ROCKE_GFX1250_PORTING_NOTES.md`) and
  MISA's atomic-scope findings — this confirms `WGP`/`SE` intermediate
  scopes exist as real hint levels, not just the CU/DEVICE/SYSTEM triple
  those investigations focused on for *atomics* specifically.
- **TH** (temporal behavior): `RT` (regular temporal), `NT` (non-temporal,
  low/no reuse expected), `HT` (high-priority temporal), `LU` (last-use —
  cache may treat as non-temporal after this read; architecture-specific
  dirty-data behavior noted), plus near/far-cache combinations (e.g.
  `RT_NT` = temporal in near cache, non-temporal in far/L2).

**Suggested GEMM usage from the source material** (directly transferable to
CK's WMMA epilogue/main-loop hint annotations, if CK exposes any such
mechanism for gfx1250):

| Traffic | Suggested hint | Rationale |
|---|---|---|
| Cooperative macro-tile A/B loads | `RT_NT` | Temporal in near cache during the cooperative load (other lanes/waves will consume it shortly), non-temporal in L2 if not re-read after staging |
| C output stores | `NT` | No reuse expected once written |
| Last-use loads | `LU` where semantically valid | Signals the cache line won't be touched again |

**Explicit warning, worth preserving verbatim for CK engineers**: "Do not
apply hints mechanically. Classify reuse at each cache level and benchmark.
Streaming traffic with poor reuse should not evict data with meaningful
reuse." This is a real correctness-adjacent performance risk (a wrong hint
can evict useful data), not a free win to blindly apply.

**Actionable for CK**: audit whether CK's gfx1250 WMMA GEMM/conv loads and
stores use any coherence/temporal hint mechanism at all today. If CK's
`amd_buffer_addressing.hpp`-style buffer descriptors only ever specify a
*scope* (per the DEVICE-scope atomic fix already found) and never a
*temporal* hint, this is an entirely unexplored optimization axis — distinct
from and additive to the scope-only work already audited.

## 2. Claused Stores — a third output-store strategy not previously named

Prior investigations covered two output-store strategies for gfx1250 WMMA's
lane-swizzled accumulator: **LDS-staged reshuffle** (CK's default epilogue,
per `MISA_GFX1250_CONV_LEARNINGS.md` §1.3) and **direct per-lane store**
(the technique independently confirmed by MISA/FlyDSL/hipConv, per §4.1 of
`MISA_GFX1250_CONV_LEARNINGS.md`). This guide names a **third, distinct**
strategy not previously captured:

> **Claused stores**: compute all store addresses first, then issue the
> stores as a consecutive instruction clause. The write-combining buffer can
> observe adjacent partial writes, merge them, and avoid unnecessary
> read-modify-write traffic.

This is conceptually different from direct-per-lane-store (which relies on
gfx1250's WMMA lane geometry already being coalesced) — claused stores are
useful precisely when addresses are **not** already coalesced but *are*
adjacent enough for the hardware write-combiner to merge partial-cache-line
writes if issued back-to-back rather than interleaved with other work. The
guide explicitly frames this as one of three strategies to **benchmark
against each other** (claused stores / LDS staging / async LDS-to-global),
not a strict improvement over the others — "the best path depends on K,
epilogue complexity, output layout, tile shape, and available LDS."

**Actionable for CK**: if CK's WMMA epilogue output addresses aren't fully
coalesced for a given tile/precision combination (making direct-per-lane
store inapplicable) but are close enough for write-combining, evaluate
issuing the scattered stores as an instruction clause rather than defaulting
straight to LDS-reshuffle staging.

## 3. Speculative vs. Non-Speculative L2 Prefetch — a safety distinction MISA's own finding didn't fully cover

`MISA_GFX1250_CONV_LEARNINGS.md` §2.4 already covers the `SCOPE_DEV` vs
`SCOPE_CU` correctness/perf trap MISA found for `wmma_l2_prefetch`. This
guide adds a **distinct, additional safety dimension** for the same
`GLOBAL_PREFETCH_B8` mechanism:

> - **Speculative prefetch**: quietly drops a bad address — safe to use near
>   tile boundaries where the target address may be out of bounds.
> - **Non-speculative prefetch**: can walk page tables — the programmer
>   must guarantee the address is valid, or it can fault/behave incorrectly.

This is a real, previously-uncaptured correctness hazard class: **using the
non-speculative prefetch variant near a tensor boundary (e.g. a conv
tile's edge, or an M/N tail) could fault or misbehave**, where the
speculative variant would silently and safely do nothing. **Actionable for
CK**: if CK ever adopts `GLOBAL_PREFETCH_B8`-style L2 prefetch for gfx1250
conv/GEMM main loops (as recommended cautiously in
`MISA_GFX1250_CONV_LEARNINGS.md`'s action list), it must use the
**speculative** variant for any prefetch address near a tile/tensor
boundary, and may only use non-speculative prefetch where the address is
provably always in-bounds.

## 4. Cluster-Barrier Drift/Timeout Mechanics — quantifies and *softens* the caution raised in the FlyDSL investigation

`FLYDSL_GFX1250_CONV_LEARNINGS.md` §9 flagged cluster-launch/TDM-multicast
as "not yet a safe pattern to copy," citing FlyDSL's own unresolved compiler
hang. This guide provides the **underlying hardware mechanics** AMD
documents for cluster multicast timeout/fallback, which meaningfully
refines (and partially softens) that caution:

- If one cluster peer workgroup drifts **more than ~1024 cycles** ahead of
  others, a pending multicast TDM transfer can **time out**. The meeting
  discussion cites a default timeout of "approximately 1000 clocks" with an
  implementation detail involving "a register value of 512 with a
  multiplier."
- **Critically: "The multicast mechanism waits for matching requests from
  peers. If requests do not align within the timeout window, it falls back
  to separate transfers rather than causing a functional failure."** This
  is an explicit, AMD-documented statement that **multicast timeout is a
  performance-only fallback, not a correctness hazard** — the hardware
  degrades gracefully to redundant per-workgroup loads rather than hanging
  or producing wrong data.
- Cluster barriers exist specifically to **bound this drift for performance
  reasons** (not correctness): `s_barrier_signal(-3)`/`s_barrier_wait(-3)`
  is the cluster-scope barrier (distinct from `-1`, the ordinary
  intra-workgroup barrier); **exactly one wave per workgroup** must signal
  `-3` (all waves signaling would send duplicate arrivals to the cluster);
  all waves must wait. The guide states explicitly: "In this GEMM usage,
  the cluster barrier is primarily a performance mechanism to preserve
  multicast alignment, not a correctness requirement for the mathematical
  result." Barrier frequency is a tunable tradeoff — too frequent adds
  overhead, too infrequent allows drift/timeouts; the source gives only a
  rough starting point ("every low number of thousands of clocks") and says
  frequency is workload-dependent.

**Nuance for CK**: this means FlyDSL's own cluster+multicast+WMMA-GEMM
compiler hang (the JIT-compilation hang noted in
`FLYDSL_GFX1250_CONV_LEARNINGS.md` §9) is very likely a **compiler/toolchain
bug in FlyDSL's specific lowering path**, not evidence that the underlying
hardware mechanism is unsafe — the hardware itself is documented by AMD to
degrade gracefully on timeout. **CK should not avoid cluster
multicast on gfx1250 out of a belief that it's inherently
correctness-risky; the actual risk is toolchain/implementation immaturity
(as FlyDSL's own unresolved bug demonstrates), and the underlying mechanism
has a documented, correctness-preserving fallback path.** If CK ever
implements cluster-launch multicast for large grouped-conv weight/activation
reuse across neighboring workgroups, budget for cluster-barrier tuning
(signal-frequency vs. drift/timeout tradeoff) as a genuine, separate tuning
dimension — this is real engineering work, but not evidence the feature is
unsafe to build against.

## 5. Section 21 ("Translating These Principles to Convolution") — the single most directly relevant section, reproduced in full

This is the guide's own explicit convolution-adaptation section — the most
directly on-topic material in the entire document for CK's purposes, and
not something any of the four project investigations produced (none of
MISA/rocKE/hipConv/FlyDSL wrote a general "how to think about conv on
gfx1250" principles list; they only showed specific kernel implementations).
Reproduced with light annotation for CK-specific relevance:

1. **Define the logical M, N, and K for the convolution direction and
   layout.** *(CK already does this per direction — fwd/bwd-data/bwd-weight
   — via its grouped-conv-to-GEMM transformer; worth re-verifying the M/N/K
   assignment is optimal for gfx1250 specifically, not just inherited from
   the gfx950/XDL mapping.)*
2. **Determine whether activation or weight addresses are naturally
   coalesced across lanes** *(for the specific WMMA lane geometry — 16
   lanes per half-wave on gfx1250 wave32 — not assumed from a gfx950 wave64
   MFMA analysis).*
3. **Avoid materializing `im2col` unless its conversion cost is amortized;
   prefer an implicit address mapping when practical.** *(CK's implicit-GEMM
   design already follows this; confirms CK's overall architectural choice
   is aligned with AMD's own guidance, not a finding requiring action.)*
4. **Stage activation and weight tiles into LDS in a layout directly
   consumable by WMMA.**
5. **Reuse weights across output positions and activations across output
   channels.**
6. **Treat stride, dilation, padding, and boundary predicates as potential
   divergence and address-generation costs** — explicitly flagged as a
   distinct cost category from the dense-GEMM-only guidance above.
7. **Evaluate TDM descriptors for regular multidimensional regions.
   Irregular boundary regions may require a separate path.** *(Directly
   relevant to CK's own wrw/grouped-conv boundary-handling gaps —
   corroborates the FlyDSL-derived recommendation in
   `FLYDSL_GFX1250_CONV_LEARNINGS.md` §7 to use TDM's native per-dimension
   OOB-clip field for the *regular* interior, while explicitly expecting a
   separate, non-TDM path may be needed for genuinely irregular boundary
   regions rather than forcing TDM to cover every case.)*
8. **Use cluster multicast when neighboring workgroups consume the same
   weight or activation tile** — explicitly named as a convolution-specific
   opportunity (neighboring output-tile workgroups in a conv very commonly
   share overlapping input-activation footprints or the same weight tile,
   arguably an even stronger multicast opportunity than dense GEMM's simple
   row/column sharing).
9. **Keep boundary or tail work from disrupting the hot interior path.**
10. **Optimize the output transform and fused epilogue, since lane-swizzled
    accumulator stores can become expensive.**

**A final, important guardrail from this section, worth quoting verbatim**:

> When an LLM is optimizing convolution code, it should not blindly
> reinterpret every convolution as a dense contiguous GEMM. It must preserve
> convolution indexing and verify that the proposed data movement remains
> valid for stride, dilation, groups, padding, and tensor layout.

This is a direct, explicit caution against over-applying dense-GEMM
optimization patterns (including several of this document's own techniques)
to convolution without re-verifying validity against conv-specific
indexing — relevant given how much of this investigation series' findings
(including this document's own) are GEMM-first and only secondarily mapped
to conv.

## 6. Reusable Process Artifacts (not architecture facts, but directly adoptable engineering practice)

Three sections of this guide are process/methodology content that CK could
adopt directly as engineering practice for its own gfx1250 conv-optimization
work, independent of any specific technique:

### 6.1 A pre-merge correctness/safety checklist for gfx1250 WMMA/TDM changes
Section 24 lists a checklist worth adapting verbatim into CK's own PR
review discipline for gfx1250-touching changes:
- Verify transpose flags and physical strides.
- Verify M/N/K tails.
- Verify vector alignment and load validity.
- Verify lane-to-fragment mapping for wave32 gfx1250 WMMA (not inherited
  from a wave64 assumption).
- Verify output index mapping for all eight FP32 values per lane.
- Verify LDS padding and bank behavior.
- Verify producer/consumer ordering around async/TDM transfers.
- Verify the exact counter being waited on (not a generic/broad wait).
- Verify `sched_barrier(0)` placement where required.
- Verify only one wave signals a cluster barrier (if cluster launch is ever
  used).
- Verify multicast timeout fallback is only a performance effect, never
  relied upon for synchronization (see §4 above).
- Verify speculative vs. non-speculative prefetch behavior at boundaries
  (see §3 above).
- Verify cache hints do not change required coherence semantics (see §1
  above).

### 6.2 A diagnostic decision tree for triaging a slow gfx1250 kernel
Section 23 gives a practical, symptom-first triage flow (e.g. "Global loads
are scattered → check addresses across lanes → try layout transpose/swizzle
if reusable → otherwise stage cooperatively through LDS → for skinny shapes,
evaluate intra-wave Split-K") that could be adapted into an internal CK
runbook for engineers profiling a specific slow gfx1250 conv shape, rather
than each engineer re-deriving the same diagnostic sequence independently.

### 6.3 An explicit tuning-parameter and measurement checklist
Section 22 lists the full candidate tuning-parameter space (`BLOCK_M/N/K`,
waves in M/N, per-wave output-tile count, LDS padding/swizzle, vector width,
pipeline stages, prefetch distance, split-K factor, cluster dimensions,
multicast masks, cluster-barrier interval, output-store strategy, temporal
hints, epilogue fusion) alongside the required measurement checklist for
each candidate (correctness, median/tail latency, throughput, VGPR/LDS use,
occupancy, load/store efficiency, LDS bank-conflict indicators, WMMA issue
density, wait/sync stalls, tail-tile behavior) — a comprehensive, reusable
template for structuring CK's own gfx1250 tuning-sweep infrastructure if one
doesn't already enumerate this full parameter/metric set.

**Explicit warning this section shares with the tuning-methodology lessons
already found in `MISA_GFX1250_CONV_LEARNINGS.md`** (e.g. the
measurement-noise/re-benchmark-on-uncontended-GPU lesson): *"non-linear
effects, especially leftover tiles and scheduling rounds, are difficult to
capture completely in an analytical model... must not select configurations
from theory alone"* — i.e. even AMD's own internal training material
explicitly defers to empirical tuning over analytical prediction for
gfx1250, corroborating the same lesson independently learned by MISA's own
bring-up.

## Summary: What This Document Adds Beyond the Four Prior Investigations

| # | New content | Status relative to prior docs |
|---|---|---|
| 1 | Cache SCOPE (WGP/SE/DEV/SYS) + temporal hints (RT/NT/HT/LU) for VMEM ops | **Entirely new axis** — not covered by any prior investigation |
| 2 | Claused stores as a third output-store strategy | **New**, complements LDS-staging and direct-per-lane-store already found |
| 3 | Speculative vs. non-speculative L2 prefetch safety distinction | **New nuance** on MISA's own `SCOPE_DEV`/`SCOPE_CU` prefetch finding |
| 4 | Cluster-barrier drift/timeout mechanics (~1024 cycle drift, graceful fallback) | **New detail that refines/softens** the FlyDSL-derived caution about cluster multicast |
| 5 | Explicit 10-step conv-translation framework | **New** — the only document in this series with conv-specific general principles, not just GEMM kernel code |
| 6 | Pre-merge checklist, diagnostic decision tree, tuning-parameter template | **New process artifacts**, directly adoptable regardless of any specific technique |

## Recommendations

1. **(New optimization axis) Investigate whether CK's gfx1250 WMMA memory
   operations use any coherence/temporal hints today; if not, this is
   unexplored territory** worth a dedicated experiment (§1) — distinct from
   and additive to the already-audited DEVICE-scope atomic fix.
2. **(Low-risk addition) Add "claused stores" to the set of output-store
   strategies CK benchmarks for its gfx1250 WMMA epilogue**, alongside
   LDS-staging and direct-per-lane-store (§2).
3. **(Safety) If CK adopts L2 prefetch for gfx1250, always use the
   speculative variant near any tile/tensor boundary** (§3).
4. **(Re-calibrate risk assessment) Do not treat cluster-launch/TDM-multicast
   as inherently unsafe for gfx1250 based on FlyDSL's compiler bug alone —
   the underlying hardware mechanism has a documented, correctness-preserving
   timeout fallback.** Budget cluster-barrier tuning as real but
   non-blocking engineering work if CK pursues this (§4).
5. **(Process) Adopt the 10-step conv-translation framework (§5) as an
   explicit checklist when porting any dense-GEMM gfx1250 optimization
   (from this document or the other four in this series) into CK's
   grouped-conv code** — particularly the guardrail against blindly
   reinterpreting convolution as dense GEMM without re-verifying
   stride/dilation/groups/padding/layout validity.
6. **(Process) Adapt the correctness checklist (§6.1) into CK's own PR
   template or review checklist for gfx1250-touching WMMA/TDM changes.**
