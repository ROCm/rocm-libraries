# SDPA Backward Test Plan

> **Status: draft for review (ALMIOPEN-2894).** This plan has not been reviewed yet. It
> follows RFC 0022 (#13937), which has not merged. The [methodology](methodology.md) and the
> [test-plans index](README.md) this plan links to belong to that RFC's first phase and do not
> exist yet. The bundles also depend on #14312, which derives O and Stats from the forward and
> has not merged either. The generator is the source of truth for counts and
> budgets: `generate_sdpa_bwd_bundles.py --dry-run --report` declares 580 cases, 500 written
> (quick 57, PR 151, nightly 222, full 70) and 80 gated. The engine defects DEF-01 to DEF-08
> (§4.2) are tracked under ALMIOPEN-3035: DEF-01 is ALMIOPEN-3036, DEF-02 ALMIOPEN-3037,
> DEF-03 ALMIOPEN-3046, DEF-04 ALMIOPEN-3039, DEF-05 and DEF-08 ALMIOPEN-3038, DEF-06
> ALMIOPEN-3043 and DEF-07 ALMIOPEN-3044.

This plan lists every feature of SDPA backward (`SdpaBackwardAttributes`) that a complete test
sweep must exercise, and how the sweep is spread across levels. It does not depend on any engine:
the sweep describes the problem space, and each engine's support claims decide which cases it runs.
The forward op is out of scope; it gets its own plan in this folder (`sdpa-fwd.md`).

The plan follows RFC 0022 (Per-Op Test Plans, #13937) and the [methodology](methodology.md). See
[the test-plans index](README.md) for the level definitions.

The bundles are **generated** from this plan. Nobody edits them by hand:

```bash
python dnn-providers/integration-tests/reference-data-scripts/generate_sdpa_bwd_bundles.py \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles --report
```

| Plan item | Where it lives |
|---|---|
| Axes, legality layers, shape catalog, ceilings, cost model, value domains, tolerance classes, every case record (shape, band, layout, dtype, class, cost, purpose, gate), golden list | `reference-data-scripts/sdpa_bwd_plan.py` |
| Level rules, topology templates, sweep writer, all-pairs generator and checker, purpose tags, TOML class rows, the `--report` checks (counts, budgets, trace) | `reference-data-scripts/generate_sdpa_bwd_bundles.py` |
| Bundles | `integration-test-bundles/{quick,standard,comprehensive,full}/SdpaBackward/<Topology>` (§4) |
| Tolerance rows | one `[[tolerance_overrides]]` row per dtype and class in each claiming engine TOML (§6.1) |
| Golden data | `golden/<CaseId>/tensors.dvc` on the `golden-data` remote, after gate GT1 (§5) |

Every sweep case records the plan items it implements (`metadata.plan_items`), its `purpose`
(§4.2) and its size class: traceability in both directions. `--report` prints the plan items of each
feature of §2 and fails when a feature has neither a case nor a disposition, or a case names no
feature. The "Earliest level" column of §2 comes from the same function.

The features come from `flatbuffers_sdk/schemas/sdpa_backward_attributes.fbs` (`fbs`),
`tensor_attributes.fbs` and `graph.fbs`; the frontend rules in `SdpaBwdNode.hpp` and
`SdpaBackwardAttributes.hpp`; and the semantics contract: cudnn-frontend `Attention.md` and
`scaled_dot_product_flash_attention.h` (`sdpa_node.h`) at `7d24006e6d7`, and the hipDNN SDPA
semantics register (`SdpaSemanticsRegister.md`, entries SEM-nnn, added by #14358).

**Baseline.** develop `61c26758398` (checked 2026-10-26 22:40 UTC) plus #14312, which derives O and
Stats from the forward in the fill path; every file and line cited here is unchanged since the first
audit at `b4fb42b18d4`. The CPU reference is the oracle on this baseline. Later PRs open gates
(§4.1): #14358 (CPU and GPU references for the whole API, the SdpaBackward golden lane), #14287 or
#14358 (SEM-001, the `left_bound` meaning), #14346 (`expected_rejection`), and the companion engine
PRs that turn open defects into typed declines (§4.2). **Levels:** quick (`quick/`), PR
(`standard/`), nightly (`comprehensive/`), full (`full/`); they cascade. PR CI runs `standard` on
gfx94X in TheRock's `hipkernelprovider` job and nightly runs `comprehensive`; full runs only in
TheRock prerelease builds, and no weekly lane exists (`projects/hipdnn/docs/TESTING.md:105-110`).

**Ids.** Features T, A, G, DT, GO, AW, V, E (§2); frontend rules R0–R14, rejection-list rules IV,
proposed rules IVP (§2.9); shapes S, L, M, H, X (§3); core cells C, all-pairs AP, targeted products
TP, seed replicas SD, scale-tensor claims CLM, probes PRB, escaped-bug reproducers REG and open engine
defects DEF (a valid graph that an engine accepts and computes wrongly today) (§4); golden cases GL
and generator requirements GEN (§5); harness gaps HG (§6.4). "The gfx942 pilot" is the run of 182 +
45 cases on gfx942 (MI300X) with the #14312 CI build on 2026-10-26; it calibrated the cost model and
the tolerance classes and found the DEF classes.

**Deviations from RFC 0022.** Each needs reviewer approval; the plan applies them until then.

| Id | RFC rule | Deviation | Reason | Compensation |
|---|---|---|---|---|
| DV-1 | §7: no default is degenerate | The default shape has g = Hq/Hk = 1 and Dqk = Dv. RFC 0022 §7 wants every ratio that can exceed 1 to do so in the default; these two do not. | With g > 1 or Dqk ≠ Dv in the default, every S and L case would also test the head-group reduction or the Dv indexing. A defect there would fail all of them, and S and L would no longer isolate one feature. Every S and L case would also become a probe (no engine implements those values, P2), so the one-at-a-time set would hold no claim case. | g and Dqk/Dv are their own S, H and L cases (S06–S12, S18, S19, H43–H50, L08–L12), all-pairs axes (`heads`, `hdim`) and the product TP-C (GQA and MQA × every core cell). The default moves to 6/3/3 and 192/128 when claims are harvested for both; the generator then rewrites S and L. |
| DV-2 | §12.3: matched precision for deep ops and low-precision types | Golden data uses FP64 compute rounded once (precision uplift) for bf16 and fp16 | The kernel's P and dS rounding sets the tolerance, not the golden intermediates. FP64 rounded once is reproducible across BLAS builds; matched FP32 changes 1–5 ulps between torch builds (§5). | This answers RFC 0022 open question 2 for SdpaBackward. RFC 0011's strategy table needs the same note. |
| DV-3 | §8.1, §12.2: core cover and a G-A anchor for each fused topology | `FwdBwdChained` has one nightly probe (PRB-X01), no core cells and no G-A anchor | Its oracle is inferred from code, not executed, and the golden generator writes no fused graph. | Register R-T1. Core cells and an anchor are added when PRB-X01 runs green and the generator writes fused graphs. |
| DV-4 | §8.1: S and L shapes isolate one feature | 15 L cases add causal TOP_LEFT to a non-mask feature: L08–L13, L15–L21, L32, L33 | Causal is the training mask of each of these features. L01 runs causal TOP_LEFT alone on the same base and level, so a pass of L01 and a failure of Lnn names the feature or its interaction with causal. The S set isolates each feature at small size. | Register R-L2 lists the 15 unmasked medium-size cases not run. |

**Out of scope.**

| Item | Where it belongs |
|---|---|
| The forward op | `sdpa-fwd.md` (ALMIOPEN-2893) |
| cuDNN backward features with no schema field: sinks and dSink (ALMIOPEN-3020), `rng_dump`, `score_mod`, `max_total_seq_len_q/kv`, `deterministic_algorithm(true)`, the FP8 backward node, paged K/V, block-sparse masks | frontend unit tests of rule R0 (`SdpaBackwardAttributes.hpp:565-629`) |
| Engine knobs (accumulator type, rounding mode); kernel-internal tests | provider unit tests; the kernel project |
| Exhaustive generation and fuzzing; release-train testing | the separate tool (§7); release readiness |
| The provider-local C++ test `IntegrationGpuSdpaBackward.cpp` | retired by the bundle PR that lands this plan |

---

## 1. The operation

SdpaBackward computes the gradients of scaled dot-product attention. Its inputs are the forward
inputs Q, K, V, the forward output O, the upstream gradient dO, and the forward softmax statistics
Stats (natural-log LSE). For each batch b and query head h:

```
S   = scale · Q Kᵀ + Bias (+ ALiBi)          scores; Bias broadcast to [B,Hq,Sq,Skv]
P   = exp(S − Stats)  on live positions        0 on masked positions (§1.2)
P̃   = s · Z ∘ P                               dropout mask Z, scale s; P̃ = P without dropout
D   = rowsum(dO ∘ O) · scale_inv               scale_inv = 1 without dropout
dP  = dO Vᵀ
dS  = s · P ∘ (Z ∘ dP − D)                     without dropout: dS = P ∘ (dP − D)
dV  = P̃ᵀ dO,   dQ = scale · dS K,   dK = scale · dSᵀ Q
dBias = sum of dS over the broadcast dims of Bias   (dS before the scale; SEM-014)
```

With grouped heads (g = Hq/Hk, gv = Hq/Hv, hk(h) = ⌊h/g⌋, hv(h) = ⌊h/gv⌋), dK of key head k sums
over the query heads with hk(h) = k, and dV of value head v over those with hv(h) = v.

`scale` is `attn_scale_value` or the one-element scale tensor; unset means 1.0 (SEM-013). A row with
no live key, or Stats = −inf, has P = 0 and adds 0 to every gradient (SEM-010). When O and Stats come
from the forward of the same inputs and attributes, the outputs are the gradients of ⟨dO, O⟩;
otherwise they are the formula applied to the inputs (V2). The references accumulate in FP32 and
round once (SEM-012); the API sets no accumulation rule. Sources: `SdpaBwdNode.hpp:18-34`,
`Attention.md:540-705`, SEM-010, SEM-011, SEM-014, SEM-015, SEM-021.

### 1.1 Tensors

"WF" is the well-formed form of the cuDNN contract (§2.8). Rules R0–R14 are listed in §2.9.

| Tensor | Field | Direction | WF dims | WF dtype | Frontend rule (develop) |
|---|---|---|---|---|---|
| Q, K, V | `q/k/v_tensor_uid` | in | [B,Hq,Sq,Dqk], [B,Hk,Skv,Dqk], [B,Hv,Skv,Dv] | FP16, BF16 (FP32: extension) | present, rank 4 (R1, R2); R3–R6 |
| O, dO | `o/do_tensor_uid` | in | [B,Hq,Sq,Dv] | as Q | O: present only; dO: rank 4, dims = O dims (R7) |
| Stats | `stats_tensor_uid` | in | [B,Hq,Sq,1] | FLOAT | present only |
| dQ, dK, dV | `dq/dk/dv_tensor_uid` | out | = Q, K, V | as Q | dims and packed strides inferred when unset; explicit values unchecked |
| Bias | `attn_mask_tensor_uid` | in, optional | [{B,1},{Hq,1},Sq,Skv] | FP16, BF16 (FLOAT: extension) | rank ≤ 4; last dim ∈ {Skv,1}; second-last ∈ {Sq,1} (R8) |
| dBias | `dbias_tensor_uid` | out, optional | = Bias | = Bias | output validation only (explicit dims) |
| Scale | `scale_tensor_uid` | in, optional | one element | FLOAT | one element, pass-by-value, not with `attn_scale_value` (R11, R12) |
| seq_len_q, seq_len_kv | `seq_len_*_tensor_uid` | in, optional | [B,1,1,1] | INT32 | checked only with `padding_mask` (R10) |
| Dropout mask | `dropout_mask_tensor_uid` | in, optional | [{B,1},{Hq,1},Sq,Skv] | FP16, BF16 (references: FLOAT) | rank 4, dims (R9) |
| Seed, Offset; dropout scale, scale_inv | `seed`, `offset`, `dropout_scale(_inv)_tensor_uid` | in, optional | one element | INT32 or INT64; FLOAT | none |
| Ragged offsets | `ragged_offset_tensor_uid` on a primary tensor | in, optional | (B+1,1,1,1) | INT32 | tensor rules only |

Every tensor has a declared `alignment` (default 16). `TensorAttributes::validate()` requires a
dtype, dims > 0, as many strides as dims, alignment ≥ 1 and no virtual pass-by-value tensor
(`TensorAttributes.hpp:613-655`); it sets no stride rule, so zero, negative and overlapping strides
pass. cuDNN requires innermost stride 1 on every rank-4 port (`sdpa_node.h:1284-1288`) and lets each
tensor have its own layout (`Attention.md:125`).

**Shape rules.** Rank 4 on Q, K, V (R2) and dO (R7); batch Q = K = V (R3; Bias and the dropout mask
may broadcast to 1); Dqk of Q = Dqk of K (R4); Skv of K = Skv of V (R5); Hk, Hv > 0 dividing Hq,
independently (R6); Dv free. O, Stats, dBias and explicit gradient dims have no frontend rule; WF
requires the dims above. cuDNN also rejects Sq = Skv = 1 (`sdpa_node.h:1339-1341`); hipDNN keeps it.

### 1.2 Masking and live positions

A position (b, h, i, j) is **live** when every active rule keeps it. The diagonal offset is
`off = 0` for TOP_LEFT (TL) and `off = Skv − Sq` for BOTTOM_RIGHT (BR); under padding it is per batch,
`seq_len_kv[b] − seq_len_q[b]` (SEM-008).

| Rule | Keeps key j when | Unset |
|---|---|---|
| Right bound R (`right_bound`) | j ≤ i + off + R | no right limit (unset or −1) |
| Left window W | j ≥ i + off − W + 1 | no left limit |
| Padding | i < seq_len_q[b] and j < seq_len_kv[b] | all rows and keys |
| Bias | a −inf entry removes the position through its value | — |

**Window convention.** The plan states a window as W, the keys kept including the diagonal key (the
model convention: Hugging Face, FlashAttention `window_size_left + 1`, gpt-oss, vLLM). The generator
writes `left_bound = W − 1` on develop, which keeps `left_bound + 1` keys
(`CpuFpReferenceSdpa.hpp:883-898`), and `left_bound = W` after SEM-001 lands (#14287 or #14358),
which keeps `left_bound` keys as cuDNN does (`Attention.md:380`) and makes 0 illegal (R13); W = 1
writes 0 on develop and 1 after. GL-10 pins the meaning. The PR that lands SEM-001 regenerates the 123 plan items (128 cases) that set W;
case ids do not change.

**Deprecated flags (SEM-002).** `causal_mask` sets R_eff = R if R ≥ 0, else 0, and keeps the
alignment; `causal_mask_bottom_right` also sets BOTTOM_RIGHT; a left bound still applies. Both flags
together are undefined: the frontend accepts them and the references throw (`PlanUtils.hpp:46-62`).
The `set_causal_mask*` setters never write these fields, so only JSON, flatbuffer and backend graphs
reach them. **ALiBi (SEM-021)** adds (j − i)·slope_h after the bias, with cuDNN's slopes, and is
defined only with TOP_LEFT causal (R_eff = 0).

**Structural zeros** are exact in the references, the golden data and a correct engine:

| Class | Where | Result |
|---|---|---|
| Dead row | no live key: BR with Sq > Skv + R, a window that misses every key, seq_len_kv[b] = 0, a bias of −inf on every live key | O = 0 and Stats = −inf from the forward; dQ row = 0; no contribution to dK, dV, dBias |
| Single-key row | exactly one live key | P = 1, so dS = dP − D cancels: dQ row ≈ 0 within rounding (not exact); dV receives dO |
| Unattended key | no live query in any head of its group (TL with Sq < Skv; windows) | dK and dV rows = 0; dBias column = 0 |
| Padded position | i ≥ seq_len_q[b] or j ≥ seq_len_kv[b] | the references write 0; the contract does not state it (§1.3) |

### 1.3 Semantics sources and status

hipDNN's contract is cuDNN's semantics. In order of authority: `Attention.md:360-470, 540-705` and
`sdpa_node.h:1183-1641`; the SEM register; the schema and the frontend node; the references.

| Entry | Contract | develop | after #14287 / #14358 |
|---|---|---|---|
| SEM-001 `left_bound` | L keys including the diagonal | L + 1 keys (diverges) | aligned (both PRs) |
| SEM-002–004, 006, 012, 013 | deprecated flags compose (#13065); explicit alignment, window on the same diagonal; bias, then band (extension); FP32 P; unset scale 1.0 (#12926) | aligned | aligned |
| SEM-008 padding flag | extension: seq_len without the flag is padding | CPU reference declines seq_len | modelled |
| SEM-010 dead rows | O = 0, Stats = −inf, no contribution | band dead rows correct; a row dead by −inf bias gives NaN | aligned |
| SEM-011 Stats | natural log, FP32, [B,Hq,Sq,1] | rank 4 only | rank 3 or 4 |
| SEM-014 dBias | sum of unscaled dS over broadcast dims | never written: the reference dBias stays NaN | aligned |
| SEM-015 dropout | explicit mask form; Philox needs `rng_dump` | not modelled | explicit mask modelled; Philox declined |
| SEM-021 ALiBi | cuDNN form, TOP_LEFT causal only | ignored | aligned |

Open points: dQ, dK and dV at padded positions are expected to be 0 until a SEM entry decides (the
references write 0); rank-3 Stats is an extension probe (T6); `dropout_probability = 0` is a probe
until a SEM entry defines it (A10): develop's references treat it as no dropout, #14358 declines it.

---

## 2. Feature inventory

Each feature lists the values a complete sweep must hit. **Oracle** says what can judge it (§6):
`ready` (develop + #14312 with the CPU reference), `#14358` (its references), `I6` (the #14312 O and
Stats derivation must be extended to the feature), `In` or `HGn` (a harness gap), `none`. **Earliest
level** is the lowest level holding a case (`Q` quick, `PR`, `N` nightly, `F` full); "gated → PR"
waits for a gate (§4.1), "entry" for an engine disposition (§4.2), `—` means no case. The column is
generated from the case list; purposes are in §4.2. **Legality** is declared once, in
`sdpa_bwd_plan.legal()` and `layer()` (§2.8); the generator writes an undefined graph only as a
rejection probe.

### 2.1 Tensor

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| T1–T3, T22 | Q, K, V; tensor kind | rank 4 device tensors (§1.1; every tensor but the scale is a device tensor, outputs are stored, virtual tensors appear only in fused graphs); K batch and Dqk = Q's, V Skv = K's, Hq % Hk = Hq % Hv = 0 (R3–R6); Dv free (G9); dtype DT1; layout T18–T20; alignment T23 | ready | Q |
| T4, T5 | O, dO | WF [B,Hq,Sq,Dv]; the frontend checks O's presence only and dO dims = O dims (R7); O comes from the forward of the same inputs (V2); dO layout independent of O (AW12) | ready | Q |
| T6 | Stats rank and dtype | FLOAT [B,Hq,Sq,1], natural-log LSE, −inf on dead rows; rank 3 [B,Hq,Sq] (extension); non-FLOAT is U (DT3) | rank 4 ready; rank 3 #14358 | Q; rank 3 gated → PR |
| T7–T9 | dQ, dK, dV | dims = Q, K, V; dK sums over the g query heads of each key head and over Sq; dV over the gv heads; unwritten elements read as the harness NaN sentinel | ready | Q |
| T10 | Bias | additive after the scale; rank 1–4 right-aligned broadcast (G11); FLOAT, HALF, BF16 (DT4); −inf entries (V3); strides packed, row pitch Skv + 8, transposed (Sq stride 1) | ready; a row of −inf #14358 + I4 | Q |
| T11 | dBias | dims and dtype = Bias; sum of the unscaled dS over the broadcast dims (SEM-014); explicit dims; packed and padded row pitch | #14358 (develop never writes it) | PR (rejection probes); gated → PR |
| T12 | Scale tensor | one FLOAT element, compile-time constant or runtime pass-by-value (RFC 0016; the backend applies the plugin API 1.2.0 floor itself, so bundles keep `min_required_engine_api_version` 1.0.0), value 0.5 (neither 1.0 nor 1/√Dqk); a device-memory scale is illegal (R12); exclusive with `attn_scale_value` (R11) | ready | Q |
| T13 | seq_len_q, seq_len_kv | INT32 [B,1,1,1], per-batch values (G13); both required with `padding_mask` (R10) | #14358 + I6 + HG4 | PR (rejection probe); gated → PR |
| T14 | Seed, Offset | one INT32 or INT64 element (Philox dropout) | none: the backward schema has no `rng_dump` | — |
| T15, T16 | Dropout mask, scale, scale_inv | mask [{B,1},{Hq,1},Sq,Skv] (R9) with values {0,1} (the references need FLOAT); s = 1/(1−p) with scale_inv = 1−p; scale_inv absent; scale_inv without a mask is U | #14358 + I6 + HG4 | PR (rejection probe); gated → PR |
| T17 | Ragged offsets | INT32 (B+1,1,1,1) on each ragged tensor; multiplier ≥ 1; THD packing (G14) | #14358 + I6 + HG20 | gated → N |
| T18 | Layout family (innermost stride 1) | packed BHSD; packed BSHD (every corpus graph); sequence stride D + 8; interleaved QKV (BS3HD); each tensor on its own (B,H,S) permutation; random 0–8 element gaps per axis; B = H = 1 (BHSD equals BSHD) | ready | Q |
| T19 | Innermost stride ≠ 1 | stride 2 on each of the eight IO tensors; Q with its last two axes transposed. cuDNN rejects; hipDNN has no stride rule, so the strides define the meaning (EXT) | ready | PR (entry: DEF-01) |
| T20 | Stats strides | packed; head stride Sq + 64; batch stride padded; sequence stride 2; [B,Sq,Hq] order. All WF | ready | PR (entry: DEF-02) |
| T21 | Degenerate strides | zero stride on an input (broadcast view, EXT); negative or overlapping output strides are U and never generated | HG17: the harness rejects zero strides | — |
| T23 | Declared alignment and base offset | declared 8 with allocator pointers; the base pointer at exactly the declared alignment N (not 2N) with a non-zero offset, crossed with a leading stride that is not a multiple of the vector width; a pointer less aligned than declared expects BAD_PARAM (E4). A case that only declares a small alignment tests nothing: allocator pointers are 256-byte aligned. | declared: ready; placement: I3 | PR; placement gated → PR |
| T24 | 64-bit indexing | > 2³¹ elements; > 2³¹ score positions per head; byte offsets > 4 GiB, including the O and dO span (DEF-07); dims that a 32-bit field truncates | GPU reference and GPU forward (X01, X02, X04); decline-only probe mode (X03, X15); unit test (X14) | gated (out of the levels) |

### 2.2 Op attributes

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| A1 | `attn_scale_value` | unset (1.0, SEM-013); 1/√Dqk; 1.0; model values 0.1352 (DeepSeek-V3, Kimi-K2), 168^−0.5 (Gemma-3-27B), 1.0 (Gemma-4); 0.02; 4.0; 0 and negative (DEF-06); NaN or Inf is U | ready | Q |
| A2 | `diagonal_alignment` | TL (default), BR; they differ only when Sq ≠ Skv or per-batch lengths differ; the packer rejects other values, but the JSON loader maps an unknown name to TL silently | ready | Q |
| A3 | `right_bound` R | unset or −1; 0 (causal); R > 0; R ≥ Skv − 1 (equals unset); R < −1 illegal after #14358 (R14) | ready | Q |
| A4 | Left window W | unset; 1 (diagonal only: dQ = dK = 0, dV = dO); 2; 16, 17, 33 (below or off a tile); 63, 64, 65 (tile boundary); 287 (Sq − 1); W ≥ Skv (equals unset); model W 128 (gpt-oss), 512, 1024 (Gemma-3), 4096 (Mistral-7B-v0.1) | ready (develop rule) | Q |
| A5 | Band shape (W, R) | causal; causal sliding window; symmetric W = R + 1 (ModernBERT W 65, R 64); asymmetric W 33, R 96; right-only; left-only | ready | Q |
| A6, A7, AW7 | `causal_mask`, `causal_mask_bottom_right` | each alone; `causal_mask` with W, with R 5, with BR (composition per SEM-002); both together is U (IVP6) | ready | Q |
| A8 | `alibi_mask` | TL causal only; any other band is U (IVP10) | #14358 + I6; develop ignores the flag | PR (rejection probes); gated → PR |
| A9, AW6 | `padding_mask` | true with both seq_len tensors; false with both is padding (SEM-008 extension, cuDNN rejects); true without seq_len is illegal (R10) | #14358 + I6 + HG4 | gated → PR |
| A10 | `dropout_probability` | unset; 0 (probe until a SEM entry defines it); 0 < p < 1 with seed and offset (Philox); p without RNG state, p ≥ 1, p < 0, NaN, or p with a mask is U (IVP7) | 0: ready; Philox: none | Q |
| A11 | Node `compute_data_type` | FLOAT; HALF, BFLOAT16, DOUBLE (listed: each needs its own tolerance class); unset on node and graph is illegal | ready (FLOAT) | Q (FLOAT) |
| A12, A13 | Node name; cuDNN-parity setters with no schema field | no semantic effect, not swept; rejected by R0 | — | unit tests |

### 2.3 Geometry

Defaults are non-degenerate except for the two ratios of DV-1: the default shape (§3) has B ≥ 2,
Hq ≥ 2, Sq ≠ Skv and lengths that are not multiples of 64.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| G1 | Batch B | 1; 2; 3, 5, 9 (odd); 8; ≥ 32 (M); 65,537 (X); B = H = 1 makes batch and head strides equal | ready | Q |
| G2 | Query heads Hq | 1; 2; 3; 4; 8–128, including 14, 20, 24, 28, 40, 96 (not powers of two) | ready | Q |
| G3 | Group g = Hq/Hk | 1; 2; 3, 5, 7 (odd); 4, 6, 8, 12, 16, 32; MQA (Hk = 1) with g 4, 8, 16, 64. Hazards: the head map ⌊h/g⌋; the dK/dV reduction over g heads | ready | Q |
| G4 | Hv relative to Hk | Hv = Hk; Hv ≠ Hk (R6 checks each; cuDNN allows it); Hk = Hq > Hv (DEF-04) | ready | Q |
| G5, G6 | Sq, Skv | 1; 2; 16, 17; 63, 64, 65; 113, 197, 255, 257, 1023 (not tile multiples); 128–1024; 1500; 2K–8K (M); ≥ 16K (X); residues mod 64 of model lengths; for Skv also 77, 191, 192, 193, 385, 577, 2¹⁶ + 1 | ready | Q |
| G7 | Sq relative to Skv | equal; Sq < Skv (prefix, ring step, cross attention 448 × 1500); Sq > Skv; extremes 1 × 512, 512 × 1, 4096 × 64; q-tail and k-tail twins 255/256, 511/512, 1023/1024 | ready | Q |
| G8 | Head dim Dqk = Dv | 128, 192; 1, 8, 32, 40, 59, 64, 72, 80, 96, 100, 111, 160, 224, 256, 384, 512, 513 | ready (cost grows with D) | Q (192); PR (others) |
| G9 | Dv relative to Dqk | equal; 192/128 (MLA); 64/128, 40/88, 256/108, 120/8 | ready | PR |
| G10 | Output dims and strides | the frontend infers dQ, dK, dV when unset; bundles state every dim and stride; explicit strides that differ from the inputs (AW12); explicit dims that differ from the inputs are U (IVP3) | ready | Q |
| G11 | Bias broadcast class | WF [B,Hq,Sq,Skv], [1,Hq,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv]; extension rank 1–3 and size-1 trailing dims ([..,Sq,1], [..,1,Skv]); leading dims outside {B,1} and {Hq,1} are U (IVP5) | ready | PR |
| G12–G14 | Dropout-mask broadcast; per-batch lengths; ragged packing | the four WF mask classes; lengths = S, < S, 0, 1, mixed (17, 64, 5), and seq_len_q[b] > seq_len_kv[b] under BR (per-batch dead rows); B + 1 offsets with documents of length 1 | #14358 + I6 + HG4 (G14: HG20) | gated → PR (G14: N) |

### 2.4 Data types

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| DT1, DT5 | IO dtype (the eight IO tensors equal); graph dtypes | BF16, FP16; FP32 (extension: the references support it, cuDNN does not); FP8 and integer are U (IVP11); graph io, intermediate and compute types fill unset dtypes, so bundles state every tensor dtype | ready | Q |
| DT2 | Mixed IO dtypes | FP32 dQ, dK, dV with BF16 inputs; dO ≠ O dtype; O ≠ Q dtype: all U (IVP14) | none: the references decline | PR (rejection probes) |
| DT3 | Stats dtype | FLOAT; any other dtype is U (IVP2) | ready | Q |
| DT4, AW13 | Bias dtype | BF16, FP16, FLOAT; bias dtype ≠ IO dtype (FLOAT or HALF bias with BF16 IO); BOOLEAN is U (IVP15) | ready | Q |
| DT6 | Auxiliary dtypes | seq_len INT32; seed and offset INT32 or INT64; dropout mask FLOAT; dropout scales FLOAT; others are U (IVP8, IVP9) | #14358 + I6 + HG4 | gated → PR |

### 2.5 Graph options and execution

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| GO1 | Override shapes (RFC 0008) | per-execute shape and stride overrides | HG16 | — |
| GO2 | Multi-node topologies | forward and backward in one graph (O and Stats virtual); backward outputs consumed by another node (virtual dQ, dK, dV) | ready, inferred from code | N (first value); second listed |
| GO3 | Build gate `HIPDNN_ENABLE_SDPA` | ON in every lane that runs the bundles (escaped bug #10374, #11381) | — | lane configuration |
| E1, E2 | Workspace and plan reuse | an accumulator not cleared between launches; the same plan with new buffers and a new runtime scale value | I5 | — |
| E3, E5, E6 | Serialization; validation; engine knobs | bundles load through `Graph::from_binary` without `Graph::validate()`, so the generator enforces legality; round trips are unit tests; knobs are provider unit tests | — | unit tests |
| E4 | Execute-time alignment check | a pointer less aligned than declared gives BAD_PARAM; runtime pass-by-value tensors are exempt | I3 | gated → PR |

### 2.6 Awkward combinations

Legal graphs with a hazard. Undefined graphs (layer U, §2.8) are rejection probes, not here.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| AW1 | Alignment at Sq = Skv, no padding | BR equals TL: one equivalence case; every other alignment case uses Sq ≠ Skv | ready | Q |
| AW2 | Dead rows | BR causal or band with Sq > Skv; TL window with Sq > Skv. cuDNN rejects both without padding; hipDNN defines them (SEM-010). Hazards: exp(s − (−inf)) = NaN; unwritten rows | ready | Q |
| AW3 | Unattended keys | TL causal or window with Sq < Skv: dK and dV rows exactly 0 | ready | Q |
| AW4 | Single-key rows | W = 1; the first live row under BR with Sq > Skv; Skv = 1 | ready | Q |
| AW5 | Bias with a BR band or a window | extension (SEM-006): bias, then band; cuDNN rejects | ready; dropout + band #14358 + I6 | PR |
| AW8 | Sq = Skv = 1 | P = 1: dQ = dK = 0 within rounding, dV = dO; cuDNN rejects (EXT) | ready | Q |
| AW9 | Skv ≤ 2 with long Sq | dV sums dO over Sq ≥ 512; fp16 error grows with Sq (DEF-03); bf16 is bit-exact | ready | Q (bf16) |
| AW10 | Window or bound covering the sequence | W = Skv − 1, Skv, Skv + 1 (the #14287 covering-window boundary); R ≥ Skv − 1 | ready | Q |
| AW11 | GQA with a window, causal mask, D 64 or D 256 | Mistral-7B 32/8 W 4096; gpt-oss 64/8 D 64 W 128; Gemma-3 16/8 D 256 W 1024 | ready | N |
| AW12 | Independent layouts | dO layout ≠ O; dO batch stride with gaps; dQ, dK, dV layouts ≠ inputs; dO row pitch D + 1 (DEF-05) | ready | Q |
| AW14 | K and V on one buffer | shared K = V (DeepSeek-V4); a bundle has one buffer per uid | HG17 | — |

### 2.7 Values

The fill recipes are `free` (uniform in [lo, hi]) and `fixed`, set per uid in `metadata.inputs`,
with a derived seed per uid (§6.2). Every case sets its fills explicitly.

| Id | Feature | Values to cover | Oracle | Earliest level |
|---|---|---|---|---|
| V1 | Fill domain of Q, K, V, dO | U[−1,1]; V in U[0,1] (D-term regime); Q, K in U[−4,4] (peaked); Q U[±8.66], K and V U[±5.2] (overflow); U[±6]; dO = 0; Q = K = 0. Not expressible yet: zero-excluded, subnormal and structured fills (I4) | ready; dO = 0 needs HG9 | Q |
| V2 | Source of O and Stats | derived from the forward of the same inputs, scale, band and bias (#14312; the golden forward half); random O and Stats never (#14008) | ready | Q |
| V3 | Masking through values | −inf on part of a bias row; −inf on every live key of a row (dead row by bias); seq_len 0; seq_len < S | #14358 + I4 | gated → PR |
| V4, V5 | Non-finite inputs; deep-negative score rows | NaN or Inf in Q, K, V or dO; all-equal 133120 at scale 0.125 (fused-exponent NaN); one key column at −120√D, so scores fall 87 below the row max | HG18; I4 | gated → PR |
| V6 | Accumulation length | dQ over 4096 and 65,536 keys; dK and dV over 4096 and 65,536 rows and over g·Sq rows (MQA g 64); dBias over its broadcast dims | ready; 65,536 rows HG6 | Q |
| V7 | Output magnitude regime | max\|ref\| spans 3e-3 to 110; tolerance classes D, DL, P, S, U, K1, F32 (§6.1) | ready | Q |
| V8 | Determinism | repeat within tolerance; rerun with NaN-refilled outputs and the same workspace; bitwise where an engine claims it | I5 | PR (entry: DEF-02a, repeat 3) |
| V9 | Tiny gradients | dO in U[±1e-4] (fp16 output near subnormal, #11325); fp16 inputs ×4 at scale 0.05 | ready; dO 1e-4 needs HG6 | Q |
| V10 | Seeds | one generator seed per case; per-uid seeds; four extra seeds on core cells (SD-01–SD-04) | ready | Q |

### 2.8 Legality

The bundle harness loads a graph with `Graph::from_binary` and never runs `Graph::validate()` (E5),
so the generator enforces legality and classifies every case into one layer:

**F** is frontend-legal on develop (R0–R12, tensor validate, post-validate); **F_R13** adds R13
(`left_bound` ∈ {−1} or ≥ 1; #14287 or #14358) and **F_R13R14** adds R14 (`right_bound` ≥ −1; #14358);
**F_active** is the frontend of the baseline. **WF** is F_R13R14 plus the cuDNN tensor tables and
support rules (`Attention.md:444-486, 693-705`; `sdpa_node.h:1284-1530`). **EXT** is outside WF with
semantics hipDNN defines (SEM register, stride and broadcast rules). **U** is in F_active, outside WF
and EXT: legal but undefined.

The first failing WF rule names the reason; the third column says when a graph outside WF is EXT.

| WF rule | Reason token | Outside WF, EXT when |
|---|---|---|
| O and dO dims [B,Hq,Sq,Dv]; dQ, dK, dV dims = Q, K, V; output strides do not overlap; dBias only with a matching bias; not both deprecated flags; ALiBi only with TL causal; dropout scales only with a mask | `O/dO dims`, `gradient dims`, `overlapping output strides`, `dBias without matching bias`, `both deprecated flags`, `ALiBi outside TL causal`, `dropout scale without a mask` | never (U) |
| Stats is FLOAT [B,Hq,Sq,1] | `Stats rank/dtype` | FLOAT rank 3 [B,Hq,Sq] |
| IO dtype is HALF or BFLOAT16, and the eight IO tensors share it | `IO dtype`, `mixed IO dtypes` | every IO tensor FLOAT; mixed is never |
| innermost stride 1 on every rank-4 IO tensor | `innermost stride != 1` | always: strides define the meaning |
| bias rank 4, dim 0 ∈ {B,1}, dim 1 ∈ {Hq,1}, HALF or BFLOAT16 | `bias rank/broadcast/dtype` | rank 1–4, each right-aligned dim full or 1, FLOAT/HALF/BFLOAT16 |
| 0 ≤ p < 1 with seed and offset and no dropout mask | `dropout_probability without RNG state` | p = 0 |
| seq_len_q and seq_len_kv both or neither, and only with `padding_mask` | `one seq_len`, `seq_len without padding_mask` | both without the flag: padding (SEM-008) |
| no BR band or window with Sq > Skv without padding | `BR band with Sq > Skv`, `window with Sq > Skv` | always: dead rows add 0 (SEM-010) |
| no BR band with bias, ALiBi or dropout | `BR band with bias/ALiBi/dropout` | bias or dropout, not ALiBi (SEM-006) |
| right_bound ≥ 0 when set | `right_bound < 0 when set` | −1 (unset) |
| not Sq = Skv = 1; Dqk and Dv multiples of 8; dropout mask HALF or BFLOAT16 | `Sq = Skv = 1`, `head dim not a multiple of 8`, `FLOAT dropout mask` | always (hipDNN keeps it legal; engines decide; the references need FLOAT) |

**What the generator writes.** Claim, regression and probe cases are in (WF ∪ EXT) ∩ F_active; a
rejection probe is in U; a graph outside F_active is a frontend unit test, never a case. The case
list holds 447 WF, 113 EXT and 23 U cases under develop semantics, and 40 claim or regression cases
are EXT (mostly dead rows). Under F_R13R14, PRB-M02 and PRB-M02f (`right_bound = −2`) become illegal
and leave the bundles for unit test IV14. Bands are written from (W, R), so every case is legal under
both window meanings. The generator writes what the frontend would infer and the bundle path skips:
gradient dims and strides, every tensor dtype, and `diagonal_alignment` spelled exactly.

### 2.9 Invalid inputs

Each validation rule has one frontend unit test; these are not sweep cases. IV0–IV18 cover R0–R14,
tensor validate, post-validate, backend finalize and the packer enum. Unit tests missing on develop:
R1 (O, dK, dV unset), R2 (V rank), R7 (dO rank ≠ 4; O rank < 4, a possible out-of-bounds read that
needs IVP1), R8 (all three bias checks), R10 (seq_len_kv rank, batch and non-one dims), R12 (node
wiring) and post-validate (dBias without dims). R13 arrives with #14287 and #14358, R14 with #14358
only; the merge must keep #14358's R14 test.

**Proposed rules.** Each closes a U class. Until it lands, its class is a rejection probe (§4.2);
then the probe leaves the bundles and the rule gets a unit test. Tracking: ALMIOPEN-2134 covers part
of IVP3 (explicit gradient dims); the rest is to file.

| Id | Proposed rule | Rejection probe |
|---|---|---|
| IVP1 | O rank 4, dims [B,Hq,Sq,Dv] | PRB-U01 |
| IVP2 | Stats FLOAT [B,Hq,Sq,1] (and [B,Hq,Sq] if rank 3 stays) | PRB-T05 |
| IVP3 | explicit dQ, dK, dV dims equal Q, K, V dims | PRB-U03 |
| IVP4 | dBias needs a bias with its dims and dtype | PRB-B15, PRB-B15n |
| IVP5 | rank-4 bias: dim 0 ∈ {B,1}, dim 1 ∈ {Hq,1} | PRB-U05 |
| IVP6 | not `causal_mask` with `causal_mask_bottom_right` | PRB-M01, PRB-M01n |
| IVP7 | 0 ≤ p < 1; p > 0 needs seed and offset; p not with a mask | PRB-R02, PRB-R02f |
| IVP8 | dropout scale, scale_inv, seed, offset have one element; scale_inv needs a mask | PRB-R05 |
| IVP9 | seq_len tensors both or neither, INT32 [B,1,1,1] whenever present | PRB-U09 |
| IVP10 | `alibi_mask` needs R_eff = 0 and TOP_LEFT | PRB-M05, PRB-M06 |
| IVP11 | IO dtypes ∈ {HALF, BFLOAT16, FLOAT} | PRB-U11 |
| IVP12 | innermost stride 1 on rank-4 IO tensors (open: adopt, or keep the extension) | none: those graphs are EXT (DEF-01) |
| IVP13 | output strides do not overlap | PRB-U13 |
| IVP14 | the eight IO tensors share one dtype | PRB-T02, PRB-T03, PRB-T04, PRB-T04n |
| IVP15 | bias dtype ∈ {FLOAT, HALF, BFLOAT16} | PRB-B10 |

---

## 3. Shape catalog

The sweep draws its shapes from five sets: **S** small shapes that change one feature from the
default; **L** the same features at medium size; **M** real models and the dnn-benchmarking corpus;
**H** hazard breakpoints; **X** stress. A shape is written `B H Sq×Skv D`, with H as `Hq/Hk/Hv` and D
as `Dqk/Dv` when they differ. The CPU reference sets the cost of a case on the baseline, so the
ceilings are wall times and FLOP follows from them.

| Level | RFC start | Wall per case | FLOP (implied) | MiB (fp32-eq.) | Cumulative budget | Sets |
|---|---|---|---|---|---|---|
| quick | ≤ 20 MFLOP, ≤ 1 MB | ≤ 1.5 s | ≤ 100 M | ≤ 4 | 20 s | S; H for regressions and three cheap probes |
| PR | ≤ 200 MFLOP, ≤ 16 MB | ≤ 5 s | ≤ 350 M | ≤ 16 | 120 s | S, H, six M |
| nightly | ≤ 8 GFLOP, ≤ 64 MB | ≤ 75 s | ≤ 5.5 G | ≤ 64 | 480 s | S, L, M, H |
| full | none | ≤ 5 min | ≤ 22 G | none; X states host memory | 1800 s | all, with X |

- **Binding cost.** A case costs max(pilot model, alt model), where the alt model multiplies the
  reference term by 1.10 for fp16. The same cost decides the per-case fit and the cumulative budgets
  (§4.4). The budgets are the SdpaBackward share of the gfx942 lane.
- **Why the RFC values change.** The RFC quick ceiling is below every inherited quick case (37–126
  MFLOP counting live pairs) and the default shape (66 MFLOP). Memory grows as S·D and work as S²·D,
  so memory never binds below full; 4 MiB fp32-equivalent is about 2 MiB in bf16, the golden limit.

**Cost model.** T = B·Hq·Sq·Skv counts every score position and U the live pairs. The binding model
is the gfx942 pilot fit of fill, CPU forward derivation, three engine launches, CPU backward
reference and compare:

    t_pilot = 4.6e-8·Dv·T + 8.7e-8·((3·Dqk + Dv)/4)·U + 0.14 s
    t_alt   = 27.9 ns·f·[U·(3·Dqk + Dv) + T·Dv] + 1.0e-8·((Dqk + Dv)/2)·T + 0.08 s,   f = 1.10 for fp16

On the 62 pilot cases above 1 s, measured/pilot is 0.94–1.08 and measured/alt 0.81–1.22. The CPU
backward reference is 90 % of a case, and it computes dP for every key: a W = 32 window at Skv = 2048
has 1.6 % of the dense FLOP and 36 % of the time. GQA gets no discount. Backward FLOP =
2·(3·Dqk + 2·Dv) per live pair per query head. With the #14358 GPU reference, a case costs the fill,
the #14312 CPU forward (0.4–1.7 µs per pair at D 128), the untimed GPU reference and the engine: at
least 1.0e-8·D·T + 0.08 s. A cold-cache GPU-reference timing run (RFC 0022 I9) then re-derives the
FLOP and MiB ceilings and re-sizes L and M; the wall ceilings and budgets stay. #14358's CPU reference
is about 10 % slower, so lanes that force `--verification-mode=cpu` scale by 1.1.

**Default shape.** B 2, Hq = Hk = Hv 3, 72 × 120, D 128, no mask, `attn_scale_value` 1/√Dqk, packed
BHSD on the eight IO tensors, packed FLOAT [B,Hq,Sq,1] Stats, declared alignment 16, Q, K, V, dO in
U[−1,1], O and Stats from the forward. B ≠ H, so a batch/head swap changes the result; 3 is odd and
not a power of two. Sq ≠ Skv, so TL and BR differ and TL causal leaves 48 keys unattended; 72 and 120
are multiples of 8, not of 64, like 197, 257, 577 and 1500 (tile-exact lengths are S57). D 128 is the
dominant training head dim, and 1/√Dqk is what 103 of 103 corpus graphs pass. Cost: 66.4 MFLOP,
2.25 MiB, 1.07 s. **The default has g = 1 and Dqk = Dv (DV-1).** g ∈ {2, 3, 4, 5, 8} are S variants
(S06–S12), g ∈ {6, 7, 16, 32} and Hk ≠ Hv are H variants (H46–H50), M variants keep their model's g,
TP-C crosses g 2 and MQA with every core cell, and the axis `heads` meets every other axis by
nightly. Dqk ≠ Dv is S18, S19, H43–H45, L12, M09, M10, M42, M46 and the all-pairs value 192/128.

### 3.1 Small (S)

The default with one feature changed; topology features use the default shape in their own
template. Each S case runs one dtype unless stated; the core cells and all-pairs add the other.
S17.q, S29.q, S40.q, S42.q (quick) and S16.c (PR) put an S feature on the core base (B 2, H 3,
40 × 72). "→" names the plan item that carries the shape.

| Id | Features | Change from the default | Level |
|---|---|---|---|
| S00 | G1, G2, G5–G8, DT1, T18 | none (bf16 and fp16) | PR |
| S01 | DT1 | fp32 IO | PR |
| S02–S05, S13, S14, S57 | G1, G2, G5–G7 | B 1; B 5; H 1; H 4; 72 × 72; 120 × 72; 64 × 128 (multiples of 64) | PR; full; PR; nightly; PR; nightly; PR |
| S06–S12 | G3, G4 | MQA 3/1/1; Hk ≠ Hv 3/1/3 and 3/3/1; GQA 6/3/3, 8/2/2; MQA 5/1/1, 8/1/1 | → L08, PRB-H08, DEF-04b, PRB-H03, PRB-H01; nightly (S11); → PRB-H06 |
| S15–S24 | G8, G9 | D 64, 256, 192, 192/128, 64/128, 96, 72, 80, 32, 512 | PR (S15, S21, S23); nightly (S16–S20, S22); full (S24) |
| S25–S32 | A2–A5, AW3 | causal TL, BR; window W 32 TL, BR; band W 17 R 16 TL, BR; left-only W 32; right-only R 16 | PR |
| S33–S35 | A6, A7, AW7 | `causal_mask`; `causal_mask_bottom_right`; `causal_mask` + W 32 | → REG-01, REG-02b |
| S36–S39 | A1, T12 | scale unset; 1.0; 0.1352; runtime pass-by-value 0.5 | → REG-04; nightly; nightly; → CLM-S1 |
| S40–S44 | T18, AW12, G10 | BSHD on all eight; dO BSHD; dQ, dK, dV BSHD; sequence stride D + 8; interleaved QKV | nightly (S40 both dtypes, S41, S44); PR (S42, S43) |
| S45–S47 | T23, E4 | declared alignment 8; base pointer at exactly 16 B with an odd leading stride; pointer less aligned than declared | → PRB-L03 (PR); gated I3 |
| S48–S50 | T10, T11, G11, DT4 | bias [1,1,Sq,Skv] FLOAT; [B,Hq,Sq,Skv] in the IO dtype; [1,Hq,Sq,Skv] with dBias | → PRB-B01, REG-07a, PRB-B12 |
| S51–S56 | A8–A10, T6, T13–T17 | ALiBi with TL causal; seq_len [72,37]/[120,75]; explicit dropout mask p 0.2; Philox p 0.2; ragged; Stats rank 3 | → PRB-M04, PRB-P02, PRB-R04, PRB-R03, PRB-P06, REG-08 (all gated) |

### 3.2 Medium (L)

L base: the default at 288 × 480, which crosses two tiles of 128 on both axes and three KV tiles of
192. One feature per case, except the 15 DV-4 cases, which add causal TL and list A3. The generator
halves B, then Hq (keeping the head map, never below 2), then S, until a case fits the nightly target
of 5 s. Every L case runs in nightly, one dtype each, alternating. L retests at medium size every
feature that has a runnable case (P5), except the size-independent ones of register R-L1.

| Id | Features | Change from the L base |
|---|---|---|
| L00 | G5–G7 | none (B 1, H 2 after reduction) |
| L01–L07 | A2–A5, AW2, AW3, G7 | causal TL; causal BR; window W 128 TL, BR; band W 65 R 64; 480 × 288 causal BR (192 dead rows); 240 × 240 causal |
| L08–L14 | G3, G4, G8, G9 | MQA 3/1/1; GQA 6/3/3; Hk ≠ Hv 4/2/1; D 192; 192/128; D 256 (all causal TL, DV-4); D 64 (no mask) |
| L15–L18 | T18, AW12, G10 | BSHD; dO BSHD; dQ, dK, dV BSHD; sequence stride D + 8 (causal TL, DV-4) |
| L19, L20 | A1, T12 | scale unset; runtime pass-by-value scale 0.5 (causal TL, DV-4) |
| L21–L23 | T10, G11, DT4, AW5, AW13 | bias [1,1,Sq,Skv] FLOAT, causal TL; [B,Hq,Sq,Skv] in the IO dtype; [1,Hq,Sq,Skv] HALF with BF16 IO, causal BR |
| L24–L26 | A4, A6, AW4, AW7, AW10 | `causal_mask` + W 128; W 1; W 287 (one key short of covering) |
| L27–L30 | V1, A4, A5 | V in U[0,1]; Q, K in U[−4,4]; left-only W 128; right-only R 64 |
| L31, L35 | AW11, G3 | GQA 4/2/2 with W 128; MQA 8/1/1 causal BR (144 × 240) |
| L32–L34 | DT1, G1, G2, G5, G7, V6 | fp32 IO, causal TL; B 3, H 5 at 128 × 192, causal TL (DV-4); Sq 1 against 4096 keys |

### 3.3 Model (M)

Sources: Hugging Face `config.json` files fetched 2026-10-26, papers, and dnn-benchmarking
`2c3778ccd034` (`attn.tar.gz`, `cudnn_attention_training.tar.gz`). `sdpa_bwd_plan.py` holds each
model's published layer, its variants and its citation.

- Every variant uses the corpus layout (BSHD, packed Stats: 103 of 103 corpus graphs). Models with a
  corpus graph keep its mask encoding, `causal_mask = true` plus `left_bound` (69 of 103 graphs).
  Scale is 1/√Dqk unless the model sets another value.
- Variants: `.f` the published layer (gated on the GPU reference, except M26.f at B 32 in full);
  `.pr` reduced to about 2.5 s; `.n` to about 6 s; `.fc` a full-level CPU anchor. The suffix names
  the size target, not the level.
- Reduction keeps g as long as possible: B → 1; heads → (2g, 2, 2); halve S keeping Sq:Skv and S
  mod 64, down to 128 (nightly) or 64 (PR), with W = S/4 when S < 2W; Hkv → 1; then halve S, then g.

| Family | Models (heads Hq/Hk, head dim, notes) | Variants in the levels |
|---|---|---|
| Dense GQA and MHA LLMs, causal | M01 Llama-3.1-8B 8K (32/8); M02 Llama-3.3-70B TP8 slice (8/1); M04 Qwen2.5-7B (28/4); M05 Qwen2.5-0.5B (14/2, D 64); M06 Qwen3-235B 32K (64/4); M12 GLM-4.5 (96/8); M21 gpt-oss-120b dense (64/8, D 64); M22 Nemotron-3-Ultra (64/2); M23 Llama-4 Scout chunked (40/8); M24 Phi-3-mini (32, D 96); M41 Llama-2-7B, GPT-3-13B class (32, D 128); M43 cuDNN preset llama3.1 (64/8, no mask) | M01.pr, M41.pr (PR); M02.n, M04.n, M06.n, M21.n, M22.n, M23.n, M41.n (nightly); M01.fc, M12.n, M41.fc, M43.n (full) |
| Long context and context parallel | M03 Llama-3.1-405B CP16 (Q chunk 4096 against a 65,536-key prefix, BR, 16/1); M36, M37 CP ring steps (8192 × 4096, 4096 × 8192, 8/1) | M03.n, M36.n (full) |
| Head dim 256 and 512 | M07 Qwen3.5-397B (32/2, D 256); M08 Qwen3.5-27B (24/4, D 256); M11 GLM-4.7-Flash (D 256); M17 Gemma-3-27B global (32/16, scale 168^−0.5); M18 Gemma-4-31B global (32/4, D 512, scale 1) | M07.pr (PR); M11.n, M17.n, M18.n (nightly) |
| MLA 192/128 | M09 DeepSeek-V3/R1 (scale 0.1352); M10 Kimi-K2 32K; M42 cuDNN preset dsv3 (no mask); M46 corpus MLA as MQA (128/1) | M09.pr (also M10), M42.n, M46.n (nightly) |
| Sliding window and band | M13 Mistral-7B-v0.1 (32/8, W 4096); M14 OLMo-3-7B (W 4096); M15 Gemma-3-12B local (16/8, D 256, W 1024); M16 Gemma-3-1B local (4/1, W 512); M19 Gemma-4-31B sliding (32/16, W 1024, scale 1); M20 gpt-oss-120b banded (64/8, D 64, W 128; sinks not expressible); M25 DeepSeek-V4-Flash (64/1, D 512, W 128); M38 ModernBERT-base (W 65, R 64); M47 DeepSeek-V4-Pro (128/1, D 512, W 128) | M20.pr (PR); M13.n, M14.n, M19.n, M25.n, M38.n (nightly); M15.fc, M20.n (full) |
| Vision | M26 ViT-B/16 (S 197, D 64); M27 CLIP ViT-L/14-336 (S 577); M28 SigLIP so400m (S 729, D 72); M29 Qwen2.5-VL ViT (S 4988, D 80) | M26.n, M28.pr, M29.n (nightly); M26.f at B 32 (full) |
| Speech | M30 Whisper-large-v3 encoder (S 1500); M31 its cross attention (448 × 1500); M39 its decoder (S 448, causal) | M31.pr, M39.n (nightly) |
| Diffusion and video | M32 SD3.5-Large MMDiT (S 4429); M33 FLUX.1-dev (S 4608); M34, M35 Wan2.1-14B self (S 32760) and cross (32760 × 512); M44 SDXL cross (1024 × 77); M45 LTX-2 (S 17556) | M44.pr, M45.pr (PR); M45.n (nightly); M44.n (full) |
| Packed documents | M40 Llama-3.1-8B SFT documents [3000, 1500, 2600, 1092] | gated (#14358 + I6 + HG4) |

M05, M08, M16, M24, M27, M30, M32–M35, M37 and M47 have no variant in the levels: another variant
carries their geometry class (register R-M2).

### 3.4 Hazard (H)

One breakpoint per shape on the base B 1, H 2, 72 × 120, D 128. Sources: RFC P3 classes; the
FlashAttention, PyTorch, cuDNN, composable_kernel, aiter and TransformerEngine suites; FlyDSL #1250;
the gfx942 pilot; kernel dispatch breakpoints stated as shape classes (Sq ≤ 16, S ≡ 0 mod 192, odd
KV-tile counts); escaped bugs (§4.2).

| Id | Hazard | Level |
|---|---|---|
| H01–H06, H94, H95 | odd tails 113 × 203 and 203 × 113; q-tail and k-tail twins 255/256, 511/512, 1023/1024 and their swaps | → GL-01; PR; nightly (H05, H06); full (H94, H95) |
| H07, H21, H22, H32 | S = 257; 63 × 129; 65 × 127; 513² causal | PR; nightly (H32) |
| H08–H16 | Sq 1; Sq 16 (short-Sq path); Sq 17; Skv 1; Skv 2; Sq = Skv = 1; Skv 1 at Sq 512 and 4096; Skv 2 at Sq 512 | quick (H08, H11); PR; → PRB-G01, DEF-03a, DEF-03d |
| H17–H20 | Skv 191, 192, 193 (causal TL), 385 (causal BR): KV-tile tails at 192 | PR |
| H23, H28–H31 | 516 × 253 (off-tile ratio 2); 1 × 512; dK, dV over 4096 rows; dQ over 4096 keys; 512 × 128 (many Q tiles per KV tile) | nightly (H23, H29, H30); PR |
| H24–H27, H72 | BR causal dead rows 192 × 64, 301 × 190 (shared tile), 4096 × 128; TL causal 128 × 768 (640 unattended keys); window W 17 BR at 96 × 64 | PR; nightly (H26) |
| H33–H42 | head dims 59, 111, 100, 40, 8, 1, 160, 224, 384, 513 | PR (H33–H39); nightly |
| H43–H45, H99 | Dqk/Dv 40/88, 256/108, 120/8; D 192 at Skv 320 causal BR | PR; nightly (H99) |
| H46–H50 | MQA g 7, 6, 16; GQA 4/2/1; GQA 32/8 at 16 × 16 (#11543) | PR (H46, H49); nightly (H47); → REG-05c; quick (H50, regression) |
| H51–H57 | innermost stride 2 on Q and dQ; Q last two axes transposed; Stats head stride Sq + 64, sequence stride 2, [B,Sq,H] order, padded batch stride | → DEF-01, DEF-02 |
| H58–H62 | dO sequence-major with batch stride 2·H·D; dO row pitch D + 1; each tensor its own permutation; random 0–8 element gaps; K, V batch stride 0 | PR (H58; H61 after DEF-05's entry gate); → DEF-05; nightly (H60); gated (H62, HG17) |
| H63–H71, H98 | W 1; W 2; W 1 BR at Sq < Skv; W 17, 64, 65, 63 at 256²; W = Skv; 65 × 65 with W 64; band W 33 R 96 | → REG-13c, REG-13a; quick (H65, regression); PR (H66, H68, H69); nightly (H67, H98, H71); → REG-13d |
| H73, H74, H97, H100 | R ≥ Skv; `left_bound = right_bound = −1` explicit; 128² causal TL at B 2 H 3 (the class of 102 of 103 corpus graphs); Sq 16 with BR causal | PR |
| H75–H80, H101, H102 | bias with BR causal; bias rank 1; [1,1,Sq,1]; [1,1,1,Skv]; a bias row all −inf; dBias over B and H; bias row pitch Skv + 8; transposed bias | PR (H75, H76); nightly; → REG-14, PRB-B11 (gated) |
| H81–H85 | scale 0; −1.2; 0.02; 4.0; B 9 | → DEF-06; PR; nightly (H85) |
| H86–H93, H96 | inputs U[±6]; Q = K = V = 133120 at scale 0.125; FlashAttention `bwd_overflow` ranges at 17²; V in U[0,1]; fp16 ×4 at scale 0.05; dO = 0; Q = K = 0; one key column at −120√D; dO in U[±1e-4] | PR; gated (H87 HG18, H91 HG9, H93 I4); quick (H90); gated (H96 HG6) |

Dead-row shapes need #14312: random Stats never reach −inf. They are never golden (their Stats
input holds −inf, §5).

### 3.5 Stress (X)

Limits, placed in full; each shape states what its oracle needs.

| Id | Stress | Shape | Level and oracle |
|---|---|---|---|
| X01, X02, X04 | per-head Sq·Skv > 2³¹ with W 128; 5.7·10⁹ positions per head (wan22 as stored); (b,h) slab offset > 2 GiB with W 256 | B1 1 46341² D128; B1 40 75600² D128; B2 65 65536² D128 | gated: GPU reference and GPU forward |
| X03 | dq_acc batch stride 2³² B | B1 128 65536×64 D128 | gated: decline-only probe mode (HG22) |
| X05, X09 | dK, dV over 65,536 rows; Skv = 2¹⁶ + 1 (16-bit length hazard, composable_kernel #12629) | B1 1 65536×64 D128; B1 2 16×65537 D128 | gated: HG6 and the GPU reference (121 s on the CPU reference, beyond the full reserve) |
| X06 | dQ over 65,536 keys (dO scaled ×128 for its class) | B1 1 64×65536 D128 | full; CPU reference |
| X07 | Skv 1, Sq 65,536 (dV sums 65,536 rows) | B1 1 65536×1 D128 | full, bf16 and fp16 (fp16 is DEF-03) |
| X08 | B·H = 65,537 grid entries | B65537 1 16² D64 | full |
| X10 | Skv = 2¹⁶ + 1 with ALiBi, TL causal | B1 2 16×65537 D128 | gated: #14358 + I6 |
| X11 | long causal accumulation, S 4096 | B1 1 4096² D128 causal | full (192 s) |
| X12 | long causal, S 16384 | B1 1 16384² D128 causal | gated: GPU reference and timing |
| X13 | huge batch, tiny S (Qwen2.5-VL window attention) | B78 16 64² D80 | full |
| X14 | dims narrowed to 32 bits (Dqk = 2³² + 128) | descriptor only | unit test |
| X15 | O and dO byte span ≥ 2³² (the ODO kernel forms b·batch-stride in 32 bits; DEF-07) | B2 1 128² D128, O and dO batch stride 2³¹ + 2¹⁴ elements | gated: decline-only probe mode (HG22) |

---

## 4. Building the sweep

`sdpa_bwd_plan.py` holds one record per case (geometry, per-tensor dims, strides and dtypes, bounds,
scale, attributes, value domains, level, topology, purpose, plan items, class, oracle, gate, cost):
580 cases: 500 written on the baseline and 80 gated (§4.1). **Core axes:** dtype
{BF16, FP16} × Sq:Skv {<, =, >} × mask {none, causal, window} × alignment {TL, BR} where it changes
the mask: 13 cells per dtype. The mask and the Sq:Skv relation choose the kernel's code path (dead
rows, unattended keys, the window anchor, the masked-tile skip); the dtype selects the kernel and the
tolerance. Layout is not a core axis: it changes address arithmetic only (register R-C1). **Feature
axes:** everything else in §2.

| Rule | Meaning |
|---|---|
| Core cover | The 26 core cells are in quick, so every level runs them. The plain op carries them; the optional-tensor topologies are features. `FwdBwdChained` has no core cells (DV-3). |
| Executed cover (P10) | Quick holds a case of each feature that has a claim or regression case, so an engine that claims a feature executes it in quick. |
| All-pairs | By the end of nightly, every value of the 10 axes below meets every value of every other axis. A seeded generator writes the cases and a checker proves the cover. |
| Probes and regressions | Every unsupported feature, awkward combination and hazard has a probe, and every level holds rejection probes. Each escaped bug's reproducer is a case in the lowest level that affords it (§4.2). |
| Oracle gate | A case whose oracle is not ready is gated: level `notrun`, a target level and its gate. The generator does not write it (P7). |
| Entry gate | A DEF case (§4.2), and H61, which shares DEF-05's hazard, enters a lane only after its engine disposition lands. |
| Concrete graphs | Each level runs checked-in cases only. Random sampling belongs to the exhaustive tool (§7). |
| Size and budget | Per-case ceilings (§3); cumulative budgets with a 5 % reserve that includes every gated case whose gate can open before #14358 (§4.4). |

**Core cells.** B 2, H 3, D 128, packed BHSD, V in U[0,1] (§6.2). C01–C05 at 40 × 72: no mask,
causal TL (unattended keys), causal BR, window TL W 16, window BR W 16 (anchored at Skv − Sq).
C06–C08 at 64 × 64 (the class of 102 of 103 corpus graphs): no mask, causal, window. C09–C13 at
72 × 40: no mask, causal TL, causal BR and both windows (dead rows). Both dtypes: 26 cases, 9.4 s.

**All-pairs axes.** One declaration feeds the generator, the checker and the exhaustive tool. A
`probe` value has no implementing engine; it moves to `claim` when claims are harvested for it.

| Axis | Feature | Claim values | Probe values |
|---|---|---|---|
| dtype | DT1 | bf16, fp16 | — |
| seq | G7 | Sq < Skv, =, > | — |
| mask | A3–A5 | none, causal, window (W, R 0), band (W, R > 0), left-only, right-only | — |
| align | A2 | TL, BR | — |
| heads | G2–G4 | MHA 3/3/3 | GQA 4/2/2; GQA 6/2/2; MQA 4/1/1; Hk ≠ Hv 4/2/1 |
| hdim | G8, G9 | 128, 192 | 64, 256, 192/128 |
| layout | T18, AW12, G10 | BHSD, BSHD, dO BSHD, dQ/dK/dV BSHD, sequence stride D + 8 | — |
| scale | A1, T12 | 1/√Dqk, unset, 0.35, runtime pass-by-value 0.5, constant pass-by-value 0.5 | — |
| vdom | V1 | U[−1,1], V in U[0,1], Q and K in U[−4,4] | — |
| bias | T10, G11, DT4 | none | [1,1,Sq,Skv] FLOAT; [B,Hq,Sq,Skv] in the IO dtype |

- A scale tensor never meets a bias: that needs a combined topology (register R-AP1).
- Head dim 192 with a window-type band is a probe combination: no engine implements it.
- BR with no band, or at Sq = Skv, equals TL; the generator writes TL there. A value joins the axes
  when its oracle is ready (dBias, padding, dropout, ALiBi after their gates).

The generator (`seed 20261026`) projects every runnable quick, PR and nightly case onto the axes,
leaving out DEF cases, H61 and rejection probes, and adds cases until every pair is covered, claim
pairs first by claim-only cases: 41 cases, AP001–AP041 in nightly, 20 of them claim-only. With this
placement they cover 669 of 669 required pairs, and claim-only cases cover 385 of 385 claim-claim
pairs. `--report` re-proves this and checks each stored axis value against the case's geometry,
band, layout, scale, values and bias.

**Targeted products.** A complete product runs only where an interaction is plausible.

| Product | Why the interaction is plausible | Cases | Level |
|---|---|---|---|
| TP-A: head dim 192 × the 13 cells | each head dim has its own kernel per mask class; dead rows and unattended keys are handled per tile | 13 | nightly (dtype alternates) |
| TP-B: {BSHD, dQ/dK/dV BSHD} × the 13 cells | zero rows of dK, dV and dQ are written at strided addresses; masked-tile skips use the sequence stride | 26 | PR: BSHD × window BR and × causal BR with dead rows (TP-Bs05, TP-Bs11); the rest nightly |
| TP-C: {4/2/2, 4/1/1} × the 13 cells, bf16 | the dK/dV group reduction meets unattended keys and dead rows (the #11543 class) | 26 | nightly (probes) |
| TP-D: {BHSD, BSHD} × {128, 192} × {bf16, fp16} × the 13 cells, completed | the full layout × head dim × dtype product | 52 | full |
| TP-E: head dim 192 × {no mask fp16, causal TL bf16, causal BR with dead rows fp16} | the head-dim-192 kernel of each mask class ahead of TP-A | 3 | PR |

**Topologies.** A template with a fixed tensor set; the verifier rejects a case whose uids differ.
uids: K 0, O 1, Q 2, dK 3, dQ 4, dV 5, V 6, dO 7, Stats 8, Bias 9, SeqLenQ 10, SeqLenKv 11,
DropoutMask 12, DropoutScale 13, DropoutScaleInv 14, dBias 15 (as #14358's generator, so its golden
blobs attach unchanged), Scale 16, Seed 17, Offset 18, ragged offsets 19–22 (provisional, HG20).

| Topology | Tensors | Status | Runnable / gated cases |
|---|---|---|---|
| `Default` | Q, K, V, O, dO, Stats, dQ, dK, dV | exists; template rewritten, sweeps overwritten, the 16 inherited ids deleted | 440 / 62 |
| `AdditiveMask` | Default + Bias | as #14358's; its literal scale becomes a placeholder | 34 / 1 |
| `AdditiveMaskDBias` | Default + Bias + dBias | new | 0 / 6 |
| `DBiasOnly`, `DropoutScaleInv`, `SeqLenQOnly` | Default + dBias; + DropoutScaleInv; + SeqLenQ | new; rejection probes only | 2, 1, 1 / 0 |
| `ScaleTensor` | Default + Scale (constant 0.5 in the template) | new; `attn_scale_value` null | 23 / 0 |
| `DropoutMask`, `Philox` | Default + DropoutMask + DropoutScale + DropoutScaleInv; Default + Seed + Offset | new; Philox has no oracle | 0 / 2, 0 / 1 |
| `Padding`, `Ragged` | Default + SeqLenQ + SeqLenKv (`padding_mask`); Default + ragged offsets of Q, K, V, O | new | 0 / 8, 0 / 1 |
| `FwdBwdChained` | SdpaFwd + SdpaBackward (O and Stats virtual) | new; nightly probe (DV-3) | 1 / 0 |

**Placeholders in every topology template** at every level, `AdditiveMask` included:
`attn_scale_value` (null = unset), `causal_mask`, `causal_mask_bottom_right`, `alibi_mask`,
`dropout_probability`, `padding_mask` where seq_len exists, per-tensor `alignment`, node
`compute_data_type`, `left_bound`, `right_bound`, `diagonal_alignment`. The generator and the
verifier fail a case whose scale differs from a template literal. A runtime scale case patches uid 16
to `is_runtime_pass_by_value` and sets `metadata.inputs."16"` to `fixed` 0.5 (else the default draw
U[0.1, 1.0] sets the scale).

**Case ids.** `<item>_b<B>h<Hq>k<Hk>v<Hv>_<Sq>x<Skv>_d<Dqk>[v<Dv>]_<dtype>_<band>[_<extras>]_tol<class>`,
lowercase, at most 96 characters (golden ids `<item>_<dtype>_<band>_tol<class>`). The dtype token
keeps `_bfp16_` / `_fp16_`; the class token is last (`_told`, `_toldl`, `_tolp`, `_tols`, `_tolu`,
`_tolk1`, `_tolf32`); the band token states W and R (`tlw16` is a 16-key window whatever
`left_bound` holds). Example: `c05_b2h3k3v3_40x72_d128_bfp16_brw16_v0to1_told`.

**Metadata.** `purpose`, `plan_item`, `plan_items`, `regression_of`, `defect`, `seed`, `inputs` (per
uid, never uid 1 or 8), `repeat`; the harness ignores unknown keys, so they are written now.
`expected_rejection` appears on rejection probes once #14346 is in the generator's baseline;
`no_cpu_reference` is not used; the golden pointer `cases[].golden` appears when its gate opens.

### 4.1 Level contents

Each level answers one question (RFC 0022 §8.2); a case sits in the lowest level that holds it.

| Feature group | quick: gross breakage | PR: every PR | nightly: wider per engine | full: wider sweeps |
|---|---|---|---|---|
| Wall per case | ≤ 1.5 s | ≤ 5 s | ≤ 75 s | ≤ 5 min |
| Core cells (26) | C01–C13 × bf16, fp16 | — | SD-01–SD-04 (extra seeds) | — |
| One at a time (S, L) | S17.q, S29.q, S40.q, S42.q | S00, the cheap S, S16.c, S01 (fp32) | the rest of S; L00–L35 | S03, S24 |
| Masks and bands | H65, REG-01–03, REG-13a/c/d/e | S25–S32, H19, H20, H24, H25, H27, H66, H68, H69, H72–H74, H100 | H26, H32, H67, H71, H98 | TP-D |
| Head grouping | REG-05a–c, H50 | PRB-H01–H08, H46, H49; DEF-04 (entry) | TP-C, S11, H47 | M variants at model ratios |
| Head dims | S17.q | TP-E, S15, S16.c, S21, S23, H33–H39, H43–H45 | TP-A, S16–S20, S22, H40–H42, H99 | TP-D, S24 |
| Layouts and strides | S40.q, S42.q | S42, S43, H58, H61 (entry), PRB-L01, TP-Bs05, TP-Bs11; DEF-01, -02, -05, -08 (entry) | S40, S41, S44, H60, TP-B | TP-D |
| Scale | REG-04, CLM-S1, H90 | CLM-S2, CLM-S3, H83, H84; DEF-06 (entry) | S37, S38 | — |
| Bias | REG-07a–c | H75, H76 | PRB-B01–B07, H77, H78, H101, H102 | — |
| Sequence hazards | H08, H11, PRB-G01 | S13, S57, H02–H04, H07, H09, H10, H12, H17–H22, H28, H31, H97 | S14, H05, H06, H23 | H94, H95 |
| Values and accumulation | REG-18c | H86, H88, H89, H92, PRB-E04t, PRB-E05t, DEF-03 (entry) | H29, H30, PRB-E06a | X06, X07 |
| Model | — | M01, M07, M20, M41, M44, M45 (`.pr`) | 24 variants | M01.fc, M15.fc, M41.fc, M26.f and six `.n` |
| All-pairs, fused, stress | — | — | AP001–AP041; PRB-X01 | X06, X07, X08, X11, X13 |
| Rejection probes | PRB-M01 | 17 | 3 | 2 |
| Golden (§5) | GL-01 | GL-02–GL-08, GL-10–GL-13, GL-17, GL-18 | GL-19–GL-21 | — |

Quick holds the core cells; four core-base S variants; every hipDNN escaped-bug reproducer that has
an oracle; the cheapest claim case of each claimed feature the core cells miss (H08, H11, H90,
REG-18c, PRB-A01, CLM-S1, GL-01); and three probes chosen by cost and hazard: PRB-M01 (a rejection
probe with a verdict before #14346), PRB-R01 (p = 0) and PRB-G01 (Sq = Skv = 1).

Cases generated (own level / cumulative):

| Level | `Default` | Other topologies | Own | Cumulative | Claim | Probe | Regression | Distinct geometries |
|---|---|---|---|---|---|---|---|---|
| quick | 54 | 3 | **57** | 57 | 38 | 3 | 16 | 21 |
| PR | 139 | 12 | **151** | 208 | 69 | 79 | 3 | 84 |
| nightly | 175 | 47 | **222** | 430 | 109 | 113 | 0 | 105 |
| full | 70 | 0 | **70** | 500 | 45 | 25 | 0 | 22 |

Probes include the rejection probes (1, 17, 3, 2). By size class: quick S 31, H 26; PR S 37, M 9,
H 105; nightly S 125, L 36, M 28, H 33; full S 54, M 5, H 5, X 6. 204 distinct geometries in all.
`--report` prints these counts; the measured run (§4.5) fixes them.

**Gated cases** have level `notrun`, a target level and a gate, and are written when it opens.

| Target | Cases | Gate | s, CPU / GPU era |
|---|---|---|---|
| PR | GL-14, PRB-B11–B14, PRB-B16, REG-08 | #14358 | 2.84 / 0.71 |
| PR | REG-14; PRB-M04 | #14358 + I4; #14358 + I6 | 0.68 / 0.20 |
| PR | GL-15, GL-16a, M40.pr, PRB-P01–P04, PRB-R04 | #14358 + I6 + HG4 | 7.44 / 1.56 |
| PR (reserved) | S46, S47; H87; H91; H93 | I3; HG18; HG9; I4 | 3.24 / 0.58 |
| nightly | PRB-P06; M40.n | #14358 + I6 + HG20; #14358 + I6 + HG4 | 42.2 / 6.46 |
| nightly (reserved) | H96 | HG6 | 0.44 / 0.10 |
| full | X10; X05, X09 | #14358 + I6; HG6 + GPU reference | 133 / 11.0 |
| out of the levels | 23 model layers with X01, X02, X04; 20 layers with X12; M40.f; X03, X15 | #14358 + GPU forward; #14358 + timing run; + I6 + HG4; HG22 | — |
| never (listed) | H62; PRB-R03; X14 | HG17; no oracle; unit test | — |

Listed with no case: GO1 override shapes (HG16); GO2 backward outputs consumed by another node
(PRB-X03, after PRB-X01 runs green); AW14 K/V aliasing (HG17); V4 NaN and Inf injection (I4, HG18);
V8, E1, E2 repeat-and-compare (I5); V1 zero-excluded, subnormal and structured fills (I4); A11 HALF,
BFLOAT16, DOUBLE compute types (one tolerance class each); A4 windows 127, 129, 2047 (R-W1).

### 4.2 Probes and regressions

Each case carries a `purpose` in its metadata (RFC 0022 §9):

| Purpose | Expected outcome | Claims |
|---|---|---|
| `claim` | A claiming engine accepts and is correct. | `--write-support-claims` records them. |
| `probe` | A typed rejection, or a correct accept. Accept and fail is a FAIL. | Not recorded. Support matrices exclude probes. |
| `regression` | Same as the original bug case. | Same as the original case. |

A case is a **probe** when it has at least one of these features:

| Probe class | Items |
|---|---|
| Value no engine implements (P2), or an extension | G3 g > 1 and G4 Hk ≠ Hv; G8 head dims other than 128 and 192; G9 Dqk ≠ Dv; DT1 fp32; T10, G11, DT4 bias; head dim 192 with a window-type band; A10 p = 0; GO2 `FwdBwdChained`; after their gates, T11 dBias, A8 ALiBi, A9 and T13 padding, T14–T16 dropout, T17 ragged, T6 rank-3 Stats |
| Awkward combination | T23 declared alignment (PRB-L03); AW5 bias with a BR band; AW8 and AW9 in fp16 (PRB-G01 fp16, H12) |
| Open engine defect | DEF-01 to DEF-08 (below) |
| External escaped bug at a probe value | PRB-E03a, PRB-E03b (BR dead rows at D 64 and D 160) |
| Rejection probe | layer U: one or more per proposed rule (§2.9) and IV14, 23 cases; expected outcome a typed rejection |

Everything else is a `claim`: BF16, FP16, MHA, head dims 128 and 192, every band and alignment, the
layouts of T18, every scale kind and the domains of V1. When an engine adds a probe value, the run
reports `unclaimed_support`, the value moves to `claim` in `sdpa_bwd_plan.py`, and the cases are
regenerated.

**Rejection probes** have a verdict before #14346 only when a wrong accept reaches a failing check:
PRB-M01, PRB-M01n, PRB-M02, PRB-M02f, PRB-B15, PRB-B15n, PRB-T02, PRB-T03. The other 15 report SKIP
whatever the engine does and are not counted as executed cover; a wrong accept of PRB-U05 passes
silently. PRB-T04, PRB-T04n (O dtype ≠ Q dtype) and PRB-R05 (`dropout_scale_inv` without a mask) are
accepted today and hidden by the fill; they get `expected_rejection` only after an engine declines them.

**Regressions** sit in the lowest level that affords them, shrunk only when the defect does not
depend on the shape; a regression whose reference is shared with the engines also has a G-C case.

| Id | Bug | Fix | Reproducer | Level |
|---|---|---|---|---|
| REG-01–03 | deprecated `causal_mask` overrode a left bound, ignored BR and ignored `right_bound` 5; `causal_mask_bottom_right` set no band | #13065 `e0b0ad5f6d0` | B1 1: 128² with W 33; 64×128 BR (two flags); 64×96 with R 5 | quick |
| REG-04 | unset scale meant 1/√D instead of 1.0 | #12926 `bca44464c85` | B1 1 64×96 D128, scale unset | quick |
| REG-05a–c | MQA, GQA and Hk ≠ Hv overran dK/dV (memory fault) | #11543 `1f55a293e9b` | B1 4/1/1, 4/2/2, 4/2/1 64² causal TL; H50 32/8/8 16² | quick |
| REG-07a–c | the reference read the bias as FLOAT whatever its dtype | #13474 `8a4428ed822` | B1 2 64×96: BF16 bias; HALF bias with BF16 IO; HALF bias with FP16 IO | quick |
| REG-08 | the CPU reference threw on rank-3 Stats instead of declining | #12933 `73e2396380e` | B1 1 128² D128, Stats rank 3 | gated (#14358) |
| REG-13a, c, d, e | `left_bound` kept W + 1 keys (SEM-001); W 1; covering-window boundary W = Skv − 1 and Skv | #14287, #14358 (ALMIOPEN-3024) | B1 1: 64×128 W 2; 64² W 1; 65² W 64, W 65; H65 W 1 BR | quick |
| REG-14 | the CPU reference gave NaN for a row that a −inf bias masks | #14358 `6be2472f007` | B1 2 64×96, bias row 7 and column 5 all −inf | gated (#14358 + I4) |

Not sweep cases: #8469 (bf16 accumulation in the CPU reference) is pinned by GL-01; #12912, #13418
(TOML globs on dead names) by HG10; #14008 (random O and Stats) by #14312 and the class guard; #10374,
#11381 (SDPA compiled out) by GO3 and enforced claims; #10990 (code objects lost on relocation) by
#13328's packaging test; the CPU reference ignoring ALiBi, `padding_mask`, `dropout_probability` and
dBias by #14358's unit tests (ALMIOPEN-2695); frontend-rule probes by §2.9.

**Open engine defects.** Each DEF case is a valid graph that an engine accepts and computes wrongly
today (the gfx942 pilot and follow-up diagnoses on develop + #14312). Its purpose is `probe`, its
metadata names the `defect`, and it becomes a `regression` when an engine gets it right. It enters a
lane only after its disposition lands; a typed decline in a companion engine PR is the disposition
for every class an engine can detect from the graph. Widening a tolerance class never is.

| Defect | What an engine gets wrong today | Cases | Disposition (entry gate) |
|---|---|---|---|
| DEF-01 | innermost stride ≠ 1 on any IO tensor is accepted and misread | DEF-01a–h (stride 2 on each tensor), DEF-01i (Q last two axes transposed) | typed decline, companion engine PR 1 |
| DEF-02 | Stats strides other than packed [B,Hq,Sq,1] give wrong, run-to-run different dQ and NaN or Inf in dK, dV | DEF-02a (repeat 3), DEF-02b–d | typed decline, PR 1 |
| DEF-03 | fp16 dV is 4–18 ulps off at Skv ≤ 2 and Sq ≥ 512; bf16 is bit-exact | DEF-03a, DEF-03d, X07 fp16 | engine TOML `[[test_skips]]` on these three ids with a tracking issue |
| DEF-04 | Hk = Hq > Hv is accepted; V and dV are indexed with K's head count, so dV is written past its buffer | DEF-04a, DEF-04b | typed decline, PR 1 |
| DEF-05 | an odd element stride on the sequence axis of Q, K, V, dO, dK or dV, or the head axis of dK or dV: rows start mid-dword and are read one element early | DEF-05 (dO row pitch D + 1); H61 (claim, random gaps) | typed decline, PR 1 |
| DEF-06 | scale 0 gives NaN; a negative scale gives Inf and NaN | DEF-06a–c | typed decline, companion PR 2; a runtime scale needs an execute-time check |
| DEF-07 | the ODO kernel forms the O and dO batch offset in 32 bits: dQ and dK are wrong past a 4 GiB span | X15 | typed decline, a later companion PR; X15 also waits for HG22 |
| DEF-08 | a sequence stride below D, also on a length-1 axis: each (b, h) buffer ends S·stride bytes in, so the end of the last row reads as 0 and writes there are dropped | DEF-08a (Q and K sequence stride D/2), DEF-08b (Sq 1, sequence stride 1 on Q, O, dO, dQ) | typed decline, PR 1 |

DEF-03 is a precision defect in a prebuilt kernel, so no decline rule describes it. Its cases stay
in their levels and the engine TOML skips the three ids: other engines still execute them, the skip
names the defect and its ticket, and REG-18c and X07 bf16 keep executing. This gives less false
coverage than moving them out of the levels, which would hide the class from every engine.

*Non-normative:* on ASM_SDPA_ENGINE (gfx942), the one implementing engine today, the DEF cases and H61
fail until companion PR 1 merges and are typed declines after it; the analysis doc tracks outcomes.

### 4.3 Products deliberately not run

Each product left out is a decision. Rejection probes are never here and are never trimmed.

| Id | Product not run | Cases | Covered instead by | Add it back when |
|---|---|---|---|---|
| R-C1 | Layout as a core axis | 26 per layout | S40.q, S42.q, S40–S43, L15–L18, M variants (BSHD), TP-B, TP-D, axis `layout` | a layout-dependent defect escapes |
| R-C2 | Core cells at 72 × 120 instead of 40 × 72 | 26 | S00, S13, S14, S25–S28 | the GPU reference is the oracle |
| R-T1 | Core cells and a G-A anchor for `FwdBwdChained` (DV-3) | 28 | PRB-X01 | PRB-X01 runs green and the golden generator writes fused graphs |
| R-AP1 | Scale tensor × bias | 1 pair class | `ScaleTensor` and `AdditiveMask` cases separately | a combined template exists |
| R-AP2 | fp32 × every other axis | ≈ 30 pairs | S01, L32 | class F32 is calibrated |
| R-TP1 | TP-C in fp16; TP-C with Hk ≠ Hv | 39 | all-pairs `heads` × `dtype` and `hk_ne_hv` × every axis | claims are harvested for g > 1 |
| R-TP2 | TP-A and TP-B in both dtypes at nightly | 39 | TP-D at full; dtype alternates per cell; TP-E and two TP-B cells in PR | the GPU reference is the oracle |
| R-L1 | L cases of size-independent features | A6, A7, A10, AW1, AW8, AW9, T12, T19, T20, T23, V9 | their S, H and quick cases | a size-dependent defect appears in one of them |
| R-L2 | The 15 DV-4 L cases without a mask (GQA, MQA, Hk ≠ Hv, D 192, 192/128, D 256, four layouts, unset and tensor scale, FLOAT bias, fp32, odd B and H) | 15 | L01 (causal alone on the same base), their S cases, all-pairs mask × every axis | a feature fails in L and passes in S while L01 passes |
| R-M1 | Model lengths (S ≥ 1024) in the M variants; real batch of vision layers (M26 at B 64, M27 and M28 at B 32); W 4096 at S ≥ 8192 (M13, M14) and W 512 binding (M16) | 47 models | head geometry, head dims, masks, scales and S residues at S ≤ 512; M26.f at B 32; W shrunk to S/4, W 1024 binding in M15.fc | GPU reference and timing run |
| R-M2 | M03.pr, M05, M08, M15.n, M16, M24, M27, M30, M31.n, M32–M35, M36.pr, M37 in the levels | 15 | M03.n and M36.n (full); M04.n, M20, M21 (GQA, D 64); M07.pr, H47 (D 256, g 6); M19.n, M15.fc; M06.n; S20, M41.pr; M26.n, H07; M31.pr; M45.n; M44.pr; AP cases with Sq < Skv | the GPU reference is the oracle |
| R-M3 | Model layers other than M26.f | 46 | the reduced variants; M01.fc, M15.fc, M41.fc; X11 replaces M39.f | GPU reference, GPU forward and timing run |
| R-M6 | Nightly variants of M03, M12, M20, M36, M43 (trim rule, §4.4) | 5 | the same shapes in full; M20.pr in PR; M04.n, M06.n, M22.n (GQA in nightly) | the GPU reference is the oracle |
| R-G1 | GQA ratios g ≥ 5 at more than two KV tiles | 7 ratios | M04.n, M06.n, M23.n (128²), M22.n, M46.n (64²), TP-C, H46, H47, H49 | the GPU reference is the oracle |
| R-B1 | B·Hq ≥ 160 at D 128 and 192 (a grid larger than the device) | — | M41.n (32), M41.fc, X08 (65,537 at D 64), X13 (1,248 at D 80) | the GPU reference is the oracle |
| R-W1 | Windows W 127, 129, 2047; tail twins at 1K below full | 5 | W 63, 64, 65 (H98, H67, H68); W 17, 33, 287; W 1024 (M15.fc); twins 255/256 (PR), 511/512 (nightly) | a window defect beyond the first tile escapes |
| R-X1 | Long causal accumulation beyond S 4096 (X12) | 1 | X11, M41.fc, X06 | the GPU reference is the oracle |
| R-X2 | dnn-benchmarking S ladders 2048 → 32768 for one geometry | ≈ 60 | the model layer and the smallest-S corpus graph per family | never (performance sweeps) |
| R-X3 | ALiBi above 65,536 keys | 1 | composable_kernel `TestCkTileFmhaBwdLongSeqlen` (#12629) | a reference models ALiBi beyond TL causal |
| R-X4 | cuDNN `gpt_oss` preset (128/128); DeepSeek-V4-Pro below full | 2 | M20, M21 (the real 64/8); M25 | never |

### 4.4 Cost

| Estimate | Value |
|---|---|
| FLOP per case | quick ≤ 59 M, PR ≤ 101 M, nightly ≤ 354 M, full ≤ 10.7 G (M01.fc) |
| Seconds per case | largest quick 0.96 s (GL-01), PR 2.12 s (DEF-02a, three runs), nightly 6.4 s (M41.n), full 201 s (M01.fc); no case exceeds its ceiling under either model |
| Memory | X08 needs 2 GiB fp32-equivalent; X15 needs two 4 GiB device allocations, so its case sets `minimum_vram_mb` |

Every case is costed as accepted, with the binding model of §3 (a repeated case costs repeat × its
time; golden cases at the live CPU reference). The reserve holds every gated case whose gate can open
before #14358, on the CPU reference: S46, S47, H87, H91, H93 (PR, 3.2 s) and H96 (nightly, 0.4 s).
The build fails when a level crosses 95 % of its budget with the reserve.

| Level | s, own | s, cumulative | + reserved gates | Budget less 5 % | GPU-reference era, cumulative |
|---|---|---|---|---|---|
| quick | 18.7 | 18.7 | 18.7 | 19.0 | 5.7 |
| PR | 91.2 | 110.0 | 113.2 | 114.0 | 24.3 |
| nightly | 341.0 | 451.0 | 454.6 | 456.0 | 74.8 |
| full | 1247.8 | 1698.8 | 1702.4 | 1710.0 | 194.3 |

- Gates that open with #14358 also make the GPU reference the oracle; all gated cases then add 3.1,
  6.6 and 11.0 s to the GPU-era column of PR, nightly and full.
- **Trim rule.** When a level crosses its reserve, move probes that are not rejection probes, DEF
  cases or L cases to the next level, largest first; then shrink M variants by one reduction step.
  This plan applied it once: M09.pr, M28.pr and M31.pr (6.6 s) moved from PR to nightly; M03.n,
  M12.n, M20.n, M36.n and M43.n (25.4 s) moved from nightly to full (register R-M6).
- **Shards.** Each level lane runs two GTest shards in a 30-minute step; cases are ordered by
  descending cost so heavy ones alternate: quick 9.7 / 9.1 s, PR 55.8 / 54.2 s, nightly 227 / 224 s,
  full 896 / 803 s. In TheRock prerelease the full lane shares each step with other suites (HIP_MLOPS
  took 631–641 s per shard on 2026-10-25); the lane owner signs off on that share.

### 4.5 Measured run

The bundle PR merges after one cold-cache run of the generated bundles on gfx942, configured like CI
(device fill, two shards; RFC 0022 §13.1 step 10), recorded in the pull request. It confirms the cost
per level, the tolerance classes and the executed cover per feature (P10), and is repeated whenever
the generator changes the level contents.

---

## 5. Golden reference

Golden cases are chosen by the criteria of [methodology §4.2](methodology.md#42-which-cases-should-be-golden).
PyTorch on the CPU (`torch 2.10.0+cpu`) generates each one; a hipDNN reference never does. Each fits
2 MiB with finite values and is also a live sweep case, so a lane without `dvc pull` loses nothing.

**Ownership.** This plan owns the SdpaBackward bundles and the golden case list. #14358 owns the
references, the generator (`generate_sdpa_bwd_golden.py`, ALMIOPEN-2863) and the golden lane. The
bundle PR generates no golden data and writes no pointer: the data lands after #14358's lane, in a
follow-up that runs that generator on this list. #14358 drops its SdpaBackward bundles, or the bundle
PR deletes them when it rebases; the configurations kept (GL-12, GL-14, GL-15) become sweep cases.

Base case: B1 2/2/2 113 × 203 D128, scale 1/√128, inputs drawn in FP32 in U[−1,1] and rounded to the
IO dtype, seed 42 (640 KiB). Every case uses FP64 compute rounded once (DV-2) and needs gate GT1.

| Id | Case | Criterion | Level | Extra gate |
|---|---|---|---|---|
| GL-01 | plain, BHSD, bf16 | G-A; G-E (#8469) | quick | — |
| GL-02, GL-03 | plain, BHSD fp16; plain, BSHD bf16 | G-A | PR | — |
| GL-04 | BF16 bias [1,1,Sq,Skv] with BR causal, bf16 | G-A; G-C (SEM-006); G-E (#13474) | PR | — |
| GL-05 | HALF bias [1,Hq,Sq,Skv], fp16 | G-A; G-E (#13474) | PR | — |
| GL-06, GL-07 | TL causal (90 unattended keys per head); BR causal | G-C (SEM-003) | PR | — |
| GL-08 | BR with `right_bound` 90 at 203 × 113 (offset −90, no dead row) | G-C | PR | — |
| GL-10 | window W 2, BR (`left_bound` 1 on develop, 2 after SEM-001) | G-C (SEM-001, SEM-004); G-E | PR | — |
| GL-11 | `attn_scale_value` unset | G-C (SEM-013); G-E (#12926) | PR | — |
| GL-12 | GQA B2 4/2/2 64 × 96 D64, BR causal | G-C (head grouping ⌊h/g⌋) | PR | — |
| GL-13 | `causal_mask` with BR (tensors shared with GL-07) | G-E (#13065); G-C (SEM-002) | PR | — |
| GL-14 | dBias of a [1,4,64,96] bias, BR causal, GQA 4/2/2 D64 | G-B (SEM-014) | PR | #14358 |
| GL-15 | explicit dropout mask p 0.2, inputs U[0,1], GL-14's geometry | G-B (SEM-015) | PR | #14358 + I6 + HG4 |
| GL-16a | KV-side padding, seq_len_kv [96,71,64], BR, B3 2/2/2 64 × 96 D64 | G-B (SEM-008) | PR | #14358 + I6 + HG4 |
| GL-17, GL-18 | Q, K in U[−4,4] (peaked); V in U[0,1], BSHD, fp16 (D-term regime) | G-D; GL-18 also G-A | PR | — |
| GL-19 | B1 1/1/1 896 × 64 fp16: dK and dV over 896 rows, the longest within 1 MiB | G-D | nightly | — |
| GL-20 | DeepSeek-V3 MLA, 2/2/2 192² D 192/128, causal, scale 0.1352 | G-F; G-C | nightly | — |
| GL-21 | gpt-oss-120b banded TP8 slice, 8/1/1 192² D64, `causal_mask` + W 128, scale 1/8 | G-F; G-E (#13065); G-C | nightly | — |

Twenty cases (1 quick, 16 PR, 3 nightly), 19.9 MiB with forward halves (18.9 MiB on the remote:
GL-13 shares GL-07's blobs), largest 970 KiB. G-A covers plain bf16 and fp16 in BHSD and BSHD and the
bias topology in both dtypes. Each G-C case separates its reading from the wrong one by 2.5× (GL-21)
to 1708× its seed tolerance; GL-10 pins SEM-001. GQA is G-C: the CPU reference already models it.

| Not selected | Covered instead by | Add it back when |
|---|---|---|
| fp32 G-A anchors | S01, L32 | class F32 is calibrated |
| GL-09, GL-16b: dead rows, Stats −inf | C11–C13 and the other dead-row cases, live; GL-08, GL-16a | the RFC 0022 §12.2 amendment for SEM-010 lands |
| Bias topology × BSHD | GL-03, GL-18 (BSHD); GL-04, GL-05 (bias); all-pairs `layout` × `bias` | a layout-dependent bias defect escapes |
| `ScaleTensor` and `FwdBwdChained` anchors | GL-01, GL-02 and the live CPU reference; PRB-X01 | GEN20 lands; DV-3 closes |
| fp16 for dBias, dropout and padding | the bf16 cases: the reference path does not depend on the IO dtype | a dtype-dependent defect escapes there |
| Tiny output (Sq ≤ 17) | live class-S cases | the golden lane reads per-case tolerances (HG7) |
| Accumulation beyond 900 rows; model layers at real S | the live references | never: they exceed 2 MiB |
| MQA, Hk ≠ Hv, model GQA ratios | GL-12; the live CPU reference | a grouping defect escapes |
| ALiBi | nothing live yet: PRB-M04 and X10 wait for #14358 and I6 | the generator writes backward ALiBi and PRB-M04's gate opens |
| Ragged (THD) | nothing live yet: PRB-P06 waits for #14358, I6 and HG20 | the generator writes ragged backward and PRB-P06's gate opens |
| Philox dropout | nothing: no reference models it | the API gains `rng_dump` |
| #14358's `padneq_window_br`, `padneq_dbias_batch`; a row dead by a −inf bias | GL-16a, GL-14; REG-14 and #14358 unit tests | a padding defect with a window or dBias escapes; never for −inf in a bias |

**Dead rows.** `verify_golden_bundles.py` and RFC 0022 §12.2 reject non-finite values, so a case whose
Stats hold −inf (SEM-010) cannot be golden. The proposed amendment adds to §12.2: "LSE = −inf on rows
the case masks completely (SEM-010) is not a non-finite value for this rule"; the verifier then
accepts −inf, and only −inf, in an SDPA stats tensor on exactly the rows the case's bounds and
lengths kill (GEN9). A finite placeholder LSE is rejected: it breaks the pairing with the forward.

**Generator requirements** on `generate_sdpa_bwd_golden.py` v2.0.0, byte-identical with default
flags. They include FP32-drawn inputs (an IO-dtype draw gives 256 distinct bf16 values), FP64 compute
rounded once, `--attn-scale unset`, `--layout bshd`, `--window W --left-bound-semantics`,
`--deprecated-causal` with an independent SEM-002 mask, per-tensor ranges, sweep-mode GQA, bias,
dBias, dropout and padding, a check that the template's scale literal equals the CLI scale (without
it a wrong-scale case passes the verifier), a 2 MiB payload guard, provenance, batch mode from the
golden list, and the rule that the bundle generator owns every case entry. The analysis doc lists all
22 (GEN1–GEN22). Tracking: ALMIOPEN-2863 covers the base generator; the extensions are to file.

**Precision and provenance (DV-2).** The forward runs in FP64; O is rounded to the IO dtype and LSE
to FP32; the backward runs in FP64 on them, and the gradients are rounded once. The same-precision
kernel model has the same error against matched-FP32 and FP64 data (26 bundles), and FP64 rounded
once gave 0 differing elements between NumPy/OpenBLAS and torch/MKL (57 bundles × 5 tensors).
`meta.json` records the RFC 0011 fields, `reference_strategy: precision_uplift`, `forward_source` and
a `golden` block (criteria, ranges, window semantics, tolerance, confirmation, triggers).

**Tolerance and confirmation.** The seed tolerance is 2 × the worst t = max|a − b|/(1 + |b|) of the
same-precision kernel model, rounded up to 1-2-5 (2e-4 to 2e-2); the confirmed value replaces it.
Engine lanes grade a golden case with its class; the golden lane uses the op default 1e-2 (HG7).
Confirm (1) by NumPy FP64 re-derivation from the bundle alone, with D anchored on the stored O:
max|Δ|/max|ref| ≤ 1.25·2^−p (bf16 4.88e-3, fp16 6.10e-4), LSE within 1e-5, identical −inf positions,
exact structural zeros; (2) `hipdnn_golden_data_tests --reference both` on the forward halves; (3) the
same on the backward halves on a #14358 build; (4) engine lanes in golden mode. No case merges before
steps 1 and 3 pass. Never regenerate from a hipDNN reference; PyTorch and NumPy decide.

**Gates and lifecycle.** GT1: develop's `ReferenceOpCoverage.cpp:34,46` lists `SdpaAttributes` only,
so a SdpaBackward golden pointer fails as `<bundle>_Unvalidated` until #14358 (ALMIOPEN-2588 adds the
coverage gate). GT3: the objects reach the `golden-data` remote before the pointer merges; a human
runs `dvc push` (tracking: to file). GL-14, GL-15 and GL-16a wait for their live counterparts, so
golden data is never a level's only coverage of a feature. Forward halves stay `_fwd_pair` bundles,
outside SdpaBackward counts and claims (HG13). Regenerate when SEM-001 lands (GL-10 and GL-21 rewrite
`left_bound`; tensors unchanged), at generator v2.0.0, when the SEM-010 amendment lands (add GL-09,
GL-16b), when HG7 lands, and when torch, the generator major version or the strategy changes.

---

## 6. Pass criteria and reference prerequisites

| Item | Rule |
|---|---|
| Verification | `auto`: golden → GPU reference → CPU reference (`TestConfig.hpp:39-58`). On develop the GPU reference throws `ReferenceCapabilityError` for SdpaBackward (`GpuReferenceGraphExecutor.hpp:188-199`), so `auto` reaches the CPU reference. The engine runs first; a decline is a SKIP and needs no oracle. A case no oracle can verify fails (ALMIOPEN-2837), so gated cases are not written. After #14358 the GPU reference comes first. |
| O and Stats | #14312 overwrites them with the CPU forward of the filled Q, K, V under the node's scale, band and bias (`SdpaBackwardInputs.cpp`) and reports "unverifiable" for ALiBi, padding, dropout ≠ 0, ragged and a non-scalar scale. The generator never writes `inputs` for uid 1 or 8 (random O and Stats were #14008). Golden cases load them from their forward half. |
| Tolerance | elementwise \|out − ref\| ≤ atol + rtol·\|ref\|, one pair per case from its class (§6.1); NaN fails. |
| Claims | enforced from `support.json` by `--enforce-support-claims` (default on): a claimed case that declines fails with CLAIM_BROKEN. Only `claim` and `regression` cases are harvested. `--fail-on-unsupported` is not used: it would fail every declined probe. A probe's decline reason is not surfaced yet (HG8). |
| #14346 flags | `expected_rejection` on rejection probes once #14346 is in the baseline (ALMIOPEN-3023); `no_cpu_reference` is not used (ALMIOPEN-3022). |

### 6.1 Tolerance classes

A correct kernel differs from an FP32-accumulating reference by 1–2 output ulps at the largest |ref|
of each gradient (≤ 9.7e-3 bf16, 2.4e-3 fp16 relative, over 177 pilot cases, flat in S, B, H, D,
layout, mask and seed). max|ref| spans 3e-3 to 110, so no single atol is safe and non-vacuous: the
shipped bf16 1e-2 is vacuous for 50 of 111 bf16 cases. rtol is 2 ulps at the bottom of a binade.

| Class | Rule | bf16 atol / rtol | fp16 atol / rtol | Evidence (pilot, bf16 / fp16) |
|---|---|---|---|---|
| F32 | fp32 IO | 1e-5 / 1e-5 (harness default) | — | uncalibrated; guard holds on S01, L32 |
| K1 | Skv (or every live seq_len_kv) ≤ 2 | 3e-4 / 1.6e-2 | 3e-4 / 4e-3 | bf16 bit-exact at Skv 1; fp16 is DEF-03 |
| U | logit spread λ ≥ 2 | 2.5e-2 / 1.6e-2 | 3e-3 / 4e-3 | n = 5 / 4 |
| S | Sq ≤ 17 | 3e-4 / 1.6e-2 | 2e-4 / 4e-3 | n = 12 / 8; margin 3.3 / 5.4 |
| P | estimated max\|dV\| ≥ 0.5 | 1.2e-2 / 1.6e-2 | 3e-3 / 4e-3 | n = 50 / 25; margin 3.6 / 3.6 |
| DL | bf16, Skv > 1024, not P | 1.2e-3 / 1.6e-2 | (D) | derived, not measured |
| D | otherwise (diffuse) | 2.5e-3 / 1.6e-2 | 7.5e-4 / 4e-3 | n = 40 / 19; margin 4.6 / 5.1 |

- λ = scale·√Dqk·σ_Q·σ_K (scale 1 when unset). The generator assigns the first class in the order
  F32, K1, U, S, P, DL, D from the shape, mask, scale, domain and its NumPy estimate of max|ref|.
- **Guard.** atol ≤ 0.1 × the smallest max|ref| over the gradients, per case; a zeroed output then
  fails at its largest element (the negative control). **Margin.** The same-precision kernel model
  (FP32 accumulation, P and dS rounded to the IO dtype) passes the class with margin ≥ 2.5, or the
  generator moves the case to the next class whose guard holds, or scales the dO range by 2^k (exact),
  or stops. X05, X09 and H96 fit no class and wait for HG6.
- **Never loosen a class to make an engine pass.** A failing engine has a defect (DEF) or a probe
  without a claim; fp16 cases at Skv ≤ 2 are probes until DEF-03 is fixed.
- **Rows.** The generator owns 12 anchored `[[tolerance_overrides]]` rows,
  `*_SdpaBackward_*.*_<dtype>_*_<token>`, in each claiming engine TOML: each id matches one row,
  forward halves none, and `_fp16_` never matches `_bfp16_` (lint HG10). They replace the two
  dtype-only SdpaBackward rows of today's engine TOML. The measured run confirms the classes with
  device fill (HG14). Long term, a per-tensor validator `|out − ref| ≤ c·max|ref|`
  (c = 2.5e-2 bf16, 6e-3 fp16) replaces tokens and rows (HG6).

### 6.2 Value domains

A bundle sets per-uid `free` (uniform lo..hi) or `fixed` fills and an optional per-uid seed; without
one, each uid's seed derives from the case seed in uid order (`FillInputs.cpp:558-572`). Inputs of at
least 2¹⁴ elements are filled on the device, so CI values differ from host-fill values.

| uid | Tensor | Domains | How |
|---|---|---|---|
| 2, 0, 6, 7 | Q, K, V, dO | U[−1,1]; V in U[0,1]; Q, K in U[−4,4]; Q U[±8.66], K, V U[±5.2]; dO scaled by 2^k | `free` |
| 9 | Bias | U[−1,1]; U[−4,0] in the bias-dtype regressions | `free` |
| 16 | Scale | 0.5 | runtime: `fixed`; constant: the template value |
| 12–14 | Dropout mask, scale, scale_inv | Bernoulli(1 − p); 1/(1 − p); 1 − p | mask: HG4 (ALMIOPEN-2861); scales `fixed` |
| 10, 11 | seq_len_q, seq_len_kv | per-batch lengths | HG4 (ALMIOPEN-2862) |

O and Stats (uids 1, 8) are derived, never filled. **D-term domain:** each core cell has V in
U[0,1]: with zero-mean V, D = rowsum(dO∘O) is small next to dP, and dropping D moves dQ by only
t = 1.1e-2 (inside bf16 class D) instead of 1.4e-1. Dropout cases use U[0,1] on all inputs. Peaked
Q and K run once per dtype × mask class from PR on (class U). An overflow probe passes when every
output is finite and its class holds.

### 6.3 Output checks and determinism

- **NaN sentinel.** Every output is prefilled with a quiet NaN (`VariantPackBuilder.cpp:25-100`), so
  an unwritten element fails; the gradients are finite by contract (SEM-010).
- **Exact zeros** of dead rows, unattended keys and padded positions need HG9: allclose proves only
  |x| ≤ atol. **Output slack:** a write into the gaps of a strided output passes until HG19.
- **Determinism** is not promised (R0 rejects `deterministic_algorithm(true)`); every run must pass
  its class, and repeat-and-compare (I5) fails a run-to-run difference above the class atol. Until
  I5, DEF-02a runs three times in its own level-YAML category (`extra_args: ['--gtest_repeat=3']`).

### 6.4 Reference and harness prerequisites

The generator emits these features' cases as gated or listed items; they run when the work lands.

| Gap | Effect | Needed | Tracking |
|---|---|---|---|
| develop CPU reference ignores dBias, `alibi_mask`, `dropout_probability` without RNG state, `padding_mask` alone; gives NaN on a row a −inf bias masks | false PASS or FAIL; those cases are gated | #14358 | ALMIOPEN-2695, 2697–2700; 2103 (part) |
| no GPU backward reference; no SdpaBackward golden lane | CPU reference only; M and X gated; no golden data | #14358 | ALMIOPEN-2103, 2694; 2863, 2588 |
| the #14312 derivation declines ALiBi, padding, dropout and ragged (I6); its CPU forward is too slow for M and X | those cases stay gated after #14358 | extend it; derive with the GPU forward | ALMIOPEN-2855 (GPU forward); extension to file |
| #14312 calls the forward overload that #14358 deletes | the second PR to land does not build | move to `SdpaFwdOptions` | PR review |
| `left_bound` keeps L + 1 keys (SEM-001) | window cases pass while wrong against cuDNN | #14287 or #14358 | ALMIOPEN-3024 |
| rank-3 Stats throw; non-FLOAT Stats read as float; #14358 throws for a device-memory scale and declines p = 0 | false FAIL; silent mis-model; PRB-R01 loses its oracle | references decline; rule IVP2; a SEM entry for p = 0 | to file; PR review |
| CPU reference cost: single-threaded, dP for every key | 90 % of lane time | parallelise over (b, KV head); skip masked keys | to file |
| E5: bundles load without `Graph::validate()` | frontend rules never apply to a bundle | the generator enforces legality; validate after load | to file |
| HG1, HG2: no bundle generator; no `purpose` | plan and bundles drift; probes counted as claims | this plan's generator; harness and `feature_matrix.py` skip probes | ALMIOPEN-2948 (design); to file |
| HG3: no exact-alignment placement (I3) | alignment cases prove nothing | base pointer at the declared alignment with an offset | to file |
| HG4: `free` and `fixed` fills only (I4) | no dropout masks, per-batch lengths, structured −inf, NaN, Inf, subnormals | a recipe for each | ALMIOPEN-2861, 2862; injection to file |
| HG5: no repeat-and-compare (I5) | non-deterministic wrong output passes one run | `metadata.repeat: N` | to file |
| HG6: one atol and rtol per case | class tokens; X05, X09, H96 wait | per-tensor magnitude-relative validator | ALMIOPEN-338, 2704 (part); to file |
| HG8: decline reasons not surfaced | a probe cannot assert why it declined | the reason in the SKIP and the summary | to file (ALMIOPEN-2480 related) |
| HG9, HG11: no exact-zero check; max\|ref\| not printed | structural zeros pass within atol; the guard cannot be audited | structural-zero check; print max\|ref\| | to file |
| HG10: no lint on name-keyed TOML | `bf16` vs `bfp16` tokens fall back silently | one-row-per-id lint | to file |
| HG7, HG12, HG13: golden lane uses the op default tolerance; verifier rejects −inf Stats; forward halves register as tests | vacuous checks for class S; no dead-row golden cases; inflated counts | read `meta.golden.tolerance`; GEN9 and the §12.2 sentence; exclude forward halves | to file |
| HG14, HG15: host-fill calibration; no cold-cache timing per shape (I9) | margins and counts unconfirmed on CI | the measured run (§4.5) | to file |
| HG16, HG17: no override-shape path; one buffer per uid, no zero strides | GO1, AW14, H62 not run | overrides from metadata; uid aliasing; zero input strides | to file |
| HG18, HG19: no finite-only validator; no output-slack check | H87 cannot run; gapped outputs prove logical elements only | a finiteness-only validator; non-logical elements keep the sentinel | to file |
| HG20: no ragged fill | PRB-P06 and document packing gated | offset-table recipe, dense-rewrite fill | ALMIOPEN-2125 |
| HG21: a case that fails to load is dropped with an ERROR line | silent case loss | load test of registered against sweep counts | to file |
| HG22: no decline-only probe mode | X03, X15 cannot run without 4–16 GiB allocations | check the decline before allocating | to file |

Engine-specific tolerances and skips go in the engine TOML and key on the case ids of §4; a generator
change that renames ids updates the TOML rows and any FFM pin in the same pull request.

---

## 7. Exhaustive mode

SdpaBackward has an exhaustive mode. It is a separate tool (RFC 0022 §14); this plan only declares
what the tool reads, in `sdpa_bwd_plan.py`, the module the bundle generator reads.

| Declaration | Where | Content |
|---|---|---|
| Axes | `AXES` (the all-pairs axes of §4), widened | B 1–65,537; Hq 1–128 with Hk and Hv divisors of Hq chosen independently; Sq, Skv 1–65,537 with residues mod 8, 16, 64, 128, 192 and 2^k ± 1; Dqk, Dv 1–513, independent; W ∈ {unset, 1 … Skv + 1}, R ∈ {unset, −1, 0 … Skv}, TL or BR, deprecated flags; per-tensor (B,H,S) permutations, gaps, row pitch D + k, innermost stride ≠ 1; Stats strides and rank 3; every scale kind; bias rank 1–4 in every broadcast class and dtype, with dBias; dropout masks; per-batch lengths and THD offsets; BF16, FP16, FP32; declared alignment 1–256 |
| Legality | `legal()`, `layer()` | F, F_R13, F_R13R14, WF, EXT, U (§2.8); WF and EXT expect a typed rejection or a correct accept, U a typed rejection, and an illegal graph is a frontend unit test |
| Oracle readiness | `ORACLE` | a graph whose oracle is not ready (§6.4) is recorded as not run |
| Size budget | `COST` | per graph, the nightly wall ceiling (75 s on the CPU reference) and 64 MiB host memory; with the GPU reference, the timing run sets it |
| Value domains | `VALUE_DOMAINS` | the domains of §6.2; structured −inf rows and columns, NaN, Inf and subnormals once I4 lands |
| Pass criteria | `TOLERANCE_CLASSES` | the classes, guard and dO scaling of §6.1; a graph with no class waits for HG6 |
| Return path | — | minimized reproducers become `regression` cases written by the generator; oracle disagreements become G-B golden candidates |
