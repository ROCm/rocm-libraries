# rocKE Data-Type Support Reference

## Contents

- [The canonical dtype set](#1-the-canonical-dtype-set)
- [Coverage matrix](#2-coverage-matrix--support-status-by-layer)
- [Conversions](#3-conversion-coverage-helpers-layer)
- [Hardware atom support](#4-hardware-atom--dtype-support-the-hardware-ceiling)
- [Kernel-family support](#5-per-family-support-snapshot)

**What this is.** A source-backed record of which data types rocKE supports, at
which layer, and what each gap consists of. It covers the **compiler** (the two
byte-identical LLVM-IR engines), the **helpers** (conversions / I/O / quantization),
the **kernel families**, and the **hardware ceiling** the families sit under.

**Scope boundary — facts only.** This document records *what is supported and what
is not*. It deliberately contains **no** priority ranking, demand weighting, effort
estimate, shipping order, or acceptance checklist. Those are planning artifacts;
they are tracked outside the repository and are intentionally not mirrored here, so
there is only ever one copy to keep current.

| | |
|---|---|
| **Basis** | Initial source review: [c90f135be20b456c3449fca3a3d5021acf092a14](https://github.com/ROCm/rocm-libraries/commit/c90f135be20b456c3449fca3a3d5021acf092a14), 2026-09-17. Review corrections checked against [7917434a5120f32400d7ef716529ece2b0a27ef1](https://github.com/ROCm/rocm-libraries/commit/7917434a5120f32400d7ef716529ece2b0a27ef1) and public ISA XML on 2026-09-23. |
| **Method** | Initial review: source exploration of both engines. Scores in §2.4 use the criteria in §2.2 and source anchors in §2.6. Review corrections: public ISA XML and pinned LLVM source comparison, plus a Python host-side block-scale GEMM validator, signature, and IR-parameter probe. No GPU execution or numerical validation. Silicon claims come from [`matrix_instructions_summary.md`](./matrix_instructions_summary.md); declared atoms come from [arch_specs.json](../../python/rocke/core/arch/data/arch_specs.json). |
| **Out of scope** | Runtime/dispatch dtype validation, the provider's C-api surface, and any measurement. |
| **Keeping it current** | When a dtype's support changes at any layer, update the affected row(s) and the §2.6 anchors in the same change. The criteria (§2.2) and rubric (§2.3) are the stable part; the scores are not. |

> Compliance note: this document records *capability facts and methodology* only.
> It contains no measured, software-achieved performance numbers (per
> `platform/AGENTS.md` §Compliance).

---

## 1. The canonical dtype set

rocKE's compiler knows exactly **10 scalar IR types**, defined identically in
both engines:

- Python: `platform/python/rocke/core/ir.py:44-53`
- C++: `platform/cpp/include/rocke/ir.h:93-108`,
  `platform/cpp/core/ir/core_types.cpp:47-56`

| IR name | LLVM-IR type | Bits | Family |
|---|---|---|---|
| `i1` | `i1` | 1 | int / predicate |
| `i8` | `i8` | 8 | int |
| `i16` | `i16` | 16 | int |
| `i32` | `i32` | 32 | int |
| `i64` | `i64` | 64 | int |
| `f16` | `half` | 16 | float |
| `bf16` | `bfloat` | 16 | float |
| `f32` | `float` | 32 | float |
| `fp8e4m3` | `i8` (storage) | 8 | float (OCP E4M3) |
| `bf8e5m2` | `i8` (storage) | 8 | float (OCP E5M2) |

LLVM mapping: `lower_llvm.py:1024-1060` (Python) ↔
`cpp/core/lower_llvm/core.cpp:938-994` (C++). FP8/BF8 are carried as `i8`
storage and materialized through `llvm.amdgcn.cvt.*` intrinsics — there is no
native `<8 x fp8>` LLVM type in the emitter.

**Sub-byte / block formats are NOT scalar IR types.** `i4`, `fp4`, `fp6`, and
the `e8m0` MX scale exist only as *packed encodings* consumed by dedicated
helpers/atoms, never as first-class `Type` objects (see §4).

**Reduced-precision compute (the "tf32 request") is a mode, not a storage type.**
It never appears as a tensor dtype: storage stays `f32` and the *precision* is
reduced before the MMA. What is missing is the **mode itself**: no spec knob selects
reduced-precision compute, so a caller's TF32 request cannot be expressed at all.

Two facts must be kept apart, because they are easy to run together:

1. **Native XF32 is available on gfx942, a supported rocKE target.** It is
   absent on gfx950 and gfx1250 (§4D), the focus of the reduced-precision proposal.
   That focus does not define rocKE's supported target set. The catalog has no
   XF32 atom even on gfx942.
2. **A BF16 compute mode is one possible reduced-precision mode.**
   Converting `f32` operands to `bf16` can use existing bf16 MFMA/WMMA atoms.
   Native XF32 absence does not rule out software implementations of a different
   precision contract. BF16 does not implement TF32 semantics: the reference
   TF32 form keeps **10** mantissa
   bits (`truncateToTf32()` zeroes the bottom 13 bits of an `f32`,
   `GpuRefTypes.h:92-103`, gated by the `USE_TF32` compile define), while `bf16`
   keeps **7**. A value such as `1.0009765625` survives the 10-bit truncation and
   collapses to `1.0` in `bf16`.

Accordingly this document names the capability a **BF16 compute mode over `f32`
storage**; no API selector or TF32-request mapping currently exists. Rows and scores
below are labelled `bf16-mode` rather than `tf32`. No choice of tolerance makes the
two numerically equivalent; see §2.5.

**Types that are entirely absent from the vocabulary:** `f64`, unsigned
`u8/u16/u32/u64`, and a distinct `bool` (predicates reuse `i1`). No engine
references them.

Engine parity: **10/10 identical** — no dtype exists in one engine but not the
other. This is the byte-identity invariant working as intended.

---

## 2. Coverage matrix — support status by layer

### 2.1 Why explicit criteria

A per-layer verdict of Full / Partial / Absent is not enough on its own: it never
says *what makes a layer Full*, so two "Partial" types cannot be compared against
each other. §2.2 therefore names six **support criteria**, each with the external
requirement it is derived from; §2.4 scores every dtype against them.

The criteria are chosen so that each one is (a) independently observable in the
source, and (b) traceable to a requirement from one of three ecosystems:
**Composable Kernel** (the reference for what a mature AMD kernel library supports
per dtype), **hipDNN** (the descriptor/graph interface rocKE must honor — its
`data_objects::DataType` enum is the contract), and **MIOpen** (the incumbent
provider whose accepted dtype set defines the must-not-regress floor).

### 2.2 The six support criteria (C1–C6)

| # | Criterion | What it asks | Rationale — where the requirement comes from |
|---|---|---|---|
| **C1** | **IR representability** | Is the dtype a first-class `Type` in *both* engines, byte-identically? | hipDNN attaches a `DataType` to every tensor descriptor; if rocKE cannot *name* the type, no descriptor carrying it can ever be honored. CK makes the same choice — every supported type is a real C++ type (`half_t`, `bhalf_t`, `f8_t`, …), not an untyped byte blob. This is the gate all other criteria sit behind. |
| **C2** | **Compute + MMA lowering** | Can the emitter lower arithmetic on it, and does a matrix atom (MFMA/WMMA) accept it on some target arch? | This is the difference between "storable" and "fast." CK dispatches its xdlops/wmma pipelines per dtype; MIOpen's conv solvers are keyed by dtype for the same reason. A dtype with no atom can only ever be a storage/epilogue type, which caps every downstream family. |
| **C3** | **Conversion completeness** | Are casts to **and** from the `f32` hub present, plus the direct sibling casts that avoid a double bounce, with correct rounding/saturation? | Every hipDNN plan in this repo asserts `computeDataType() == DataType::FLOAT` (SDPA, RMSNorm, batchnorm, resample), so *every* tensor dtype must round-trip through f32 or it cannot participate. CK's `type_convert<Dst,Src>` is deliberately **total** over its type set; partial conversion coverage is the single most common way a dtype is "supported" on paper but unusable in a fusion. |
| **C4** | **Memory I/O plumbing** | Scalar load/store, vectorised (`n∈{2,4,8}`) load/store, and pack/unpack at the helper surface — not just raw IR primitives. | A tensor dtype is only usable if a kernel can move it. CK exposes `buffer_load`/`buffer_store` at every vector width for each supported type; rocKE's equivalent is `helpers/io.py`. If `io.py` raises `ValueError` on a dtype, every small-op family (norm, reduce, elementwise, transpose) is closed to it regardless of what the compiler can express. |
| **C5** | **Kernel-family reach** | How many shipping families actually accept it — GEMM, MFMA-GEMM, conv, attention, norm/quant epilogues? | This is the criterion the *user* feels. The reference set is the hipDNN op surface this provider implements: conv fwd/bwd/wgrad, SDPA fwd/bwd, batchnorm, RMSNorm, resample, matmul. A dtype that passes C1–C4 but ships in zero instances delivers nothing. |
| **C6** | **Numerical / dialect hygiene** | Is rounding + saturation defined, and is the *encoding dialect* unambiguous at the arch boundary? | Purely a correctness criterion, and the one with the worst failure mode: silence. gfx942 fp8 is **FNUZ** and gfx950 fp8 is **OCP** — same bytes, results off by a power of two. CK models this explicitly with distinct `f8_fnuz_t` / `f8_ocp_t` types; rocKE currently carries it as an arch-dependent decode dialect, so this criterion tracks real risk, not pedantry. MX types add a second axis (E8M0 scales are power-of-two only). |

Not every criterion applies to every dtype. For `i1`, `i16`, and `i64`, this
matrix assesses predicate/index/address storage and use: `C2` (matrix arithmetic),
`C3` (float conversions), and `C6` (rounding/dialect) are **N/A**. Their integer
operations do not imply that an MMA atom accepts those types. `i32` is also a
matrix accumulator/output type, so its `C2` and `C3` are assessed; `C6` remains
N/A for exact integer arithmetic. `i8` is assessed as quantized matrix input,
including conversion rounding and saturation, so all six criteria apply.

### 2.3 Scoring rubric

Each criterion is scored **0–3**. `N/A` criteria are dropped from both the
numerator and the denominator. `Support% = 100 × sum(scores) / (3 × number of
applicable criteria)`, rounded to the nearest integer. It describes only the
applicable criteria; percentages for indexing types and matrix inputs do not
measure the same capabilities.

| Score | Meaning |
|---|---|
| **3 — Full** | No constraint a kernel author would hit. |
| **2 — Mostly** | Works, with a documented constraint or an extra step (e.g. an obligatory `f32` bounce, a generic rather than dedicated primitive). |
| **1 — Partial** | Exists in one direction / one surface / one arch only; a kernel author must route around it. |
| **0 — Absent** | Not available at this layer at all. |

**What the scores are not.** `Support%` measures *implementation coverage in this
tree*, nothing else. It carries no judgement about how much a gap matters, how
much effort closing it would take, or when it should be closed — those are
planning questions and deliberately live outside this document (see
"Scope boundary" at the top). A 0% row is not automatically more urgent than a
94% row.

> Sourcing note: the hipDNN and MIOpen statements are cited from files in this
> repo (paths given inline). The **CK** rationale reflects the public
> Composable-Kernel / CK-Tile type vocabulary — there is no CK checkout in this
> tree, so CK is used as a design reference for *what a criterion should
> require*, not as a line-cited source.

### 2.4 Scored matrix

Scores 0–3 per criterion; `—` = N/A (excluded from the denominator). Rows are
grouped by type family, not ranked — see "What the scores are not" in §2.3.

Gap categories: **[C]** conversion gap · **[P]** partial/plumbing gap ·
**[M]** missing entirely.

| dtype | C1 IR | C2 MMA | C3 cvt | C4 I/O | C5 family | C6 numeric | Support% | Gap cat. |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Float — core** | | | | | | | | |
| `f32` | 3 | 3 | 3 | 3 | 3 | 3 | **100%** | — |
| `f16` | 3 | 3 | 2 | 3 | 3 | 3 | **94%** | [C] |
| `bf16` | 3 | 3 | 2 | 3 | 3 | 3 | **94%** | [C] |
| `f64` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | [M] |
| **Float — 8-bit** | | | | | | | | |
| `fp8e4m3` (OCP) | 3 | 3 | 1 | 0 | 1 | 1 | **50%** | [C][P] |
| `bf8e5m2` (OCP) | 3 | 3 | 1 | 0 | 1 | 1 | **50%** | [C][P] |
| `fp8 fnuz` (gfx942 only) | 1 | 3 | 1 | 0 | 1 | 1 | **39%** | [P] |
| **Sub-byte / block formats** | | | | | | | | |
| `i4` (packed) | 0 | 1 | 3 | 2 | 2 | 2 | **56%** | [P] |
| `fp4` (E2M1, MX) | 0 | 2 | 1 | 1 | 0 | 2 | **33%** | [P][M] |
| `fp6` (E3M2/E2M3, MX) | 0 | 2 | 1 | 1 | 0 | 2 | **33%** | [P][M] |
| `e8m0` (MX scale) | 0 | 3 | 3 | 2 | 3 | 2 | **72%** | — |
| **Integer** | | | | | | | | |
| `i32` | 3 | 3 | 3 | 3 | 3 | — | **100%** | — |
| `i1` | 3 | — | — | 1 | 3 | — | **78%** | [P] |
| `i8` | 3 | 2 | 2 | 2 | 1 | 3 | **72%** | [P] |
| `i64` | 3 | — | — | 1 | 1 | — | **56%** | [P] |
| `i16` | 3 | — | — | 2 | 1 | — | **67%** | [P] |
| `u8` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | [M] |
| `u16/u32/u64` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | [M] |
| **Compute mode (not a storage type)** | | | | | | | | |
| `bf16-mode` (reduced-precision compute over `f32`) | 0 | 3 | 2 | 3 | 0 | 1 | **50%** | [P][M] |

`bf16-mode` is scored as a **compute mode over `f32` storage**, which is why its
profile is unlike every other row: the storage and I/O columns are already Full
(C4=3, it *is* `f32`), the compute column is Full (C2=3, bf16 MFMA/WMMA atoms
exist), and the entire gap sits in C1/C5 — nothing can *select* the mode, so no
family offers it. Its C6=1 is not a plumbing gap but a definitional one: the mode
is bf16-accurate and must be documented as such rather than as TF32 (§1).

### 2.5 Score justifications (why each non-3 was assigned)

- **`f16` / `bf16` C3=2** — the only conversion missing is the *direct*
  `f16 ↔ bf16` cast; today it costs two casts through `f32`. Everything else
  round-trips. `bf16` reaches the same score as `f16` despite using the generic
  `global_load_vN(ptr, idx, BF16, n)` where `f16` has a dedicated
  `global_load_vN_f16` — same capability, different opcode spelling, so C4=3
  for both.
- **`fp8e4m3` / `bf8e5m2` C3=1** — `↔f32` only. No `→f16`, `→bf16`, or `→i8`,
  which is exactly the cast a mixed-precision attention inner loop wants.
- **`fp8e4m3` / `bf8e5m2` C4=0** — `helpers/io.py:56-62` raises `ValueError` for
  anything outside f16/fp16/bf16, so the entire I/O helper surface is closed;
  callers must drop to IR primitives or `quant.py`.
- **`fp8e4m3` / `bf8e5m2` C5=1** — present in block-scale GEMM A/B, MoE-fp8, and
  attention K/V, but **absent from universal GEMM** (`gemm_universal.py:659`
  raises `NotImplementedError` for any dtype that is not `F16`/`BF16`) and from
  MFMA GEMM (`mfma_gemm.py:62-65`: "`f16` is the only shipped option").
- **`fp8*` C6=1** — the FNUZ/OCP split is guarded only in the attention path;
  nothing generalizes it, so the same bytes can silently mis-decode elsewhere.
- **`fp8 fnuz` C1=1** — not a distinct IR type at all; it shares `i8` storage
  with OCP fp8 and is distinguished only by an arch-dependent decode dialect.
  CK's separate `f8_fnuz_t` / `f8_ocp_t` types are the contrast that makes this
  a 1 rather than a 3. FNUZ is a gfx942-only encoding; gfx950 and gfx1250 both
  use OCP fp8 (§4E).
- **`i8` C2=2** — full integer arithmetic and an `iu8` WMMA atom exists on
  gfx1151, but no CDNA integer MFMA path is wired *in rocKE*. Note the ceiling is
  software, not silicon: `V_MFMA_I32_*_I8` ships on every CDNA generation
  including gfx942 and gfx950 (§4A), and gfx950 adds K64/K32 wide forms.
  **C3=2** — both directions through f32 exist in
  [quant.py](../../python/rocke/helpers/quant.py): `quantize_scalar_f32` uses
  RNE with a symmetric ±127 clamp, and `dequantize_scalar_to_f32` handles
  i8→i32→f32 internally. Use `scale=b.const_f32(1.0)` for an unscaled conversion; callers
  do not need to write the extension themselves. Direct sibling conversions
  still route through f32, so this remains Mostly rather than Full.
  **C5=1** — quant epilogues and the gfx1151 integer WMMA GEMM instances in §5
  exist. Matrix-family reach remains limited to targets with the `iu8` WMMA
  atom and packed-input builders; universal/MFMA GEMM and attention do not
  expose i8. The score remains Partial, but is not an assertion that integer
  GEMM is absent.
- **`i4` C1=0 / C3=3** — the inverse of the fp8 profile: not a `Type` at all,
  yet its conversion coverage is the *best* of any quantized type
  (`i4_dequant.py` targets i32, f32, f16, fp8, bf8). C2=1: `iu4` WMMA on RDNA
  only. No native INT4 matrix instruction is present on the MFMA targets in §4A;
  software dequantization remains possible on gfx950. No gfx1250 `iu4` opcode
  was located in the public XML (§4A †), which is why C2 stays at 1.
- **`fp4` / `fp6` C5=0** — the widest silicon-to-software gap in the matrix. The
  gfx950 atoms exist (`mfma_f32_16x16x128_fp4`, `mfma_f32_16x16x96_fp6`) and the
  hipBLASLt provider already maps `FP4_E2M1`→`HIP_R_4F_E2M1` and
  `FP6_E3M2`→`HIP_R_6F_E3M2`, so the interface and the silicon are both ready —
  but **no rocKE kernel instance ships**. C3=1: codebook `unpack_fp4_*` /
  `unpack_fp6_*` to f32 only, no scalar `cvt`.
- **`e8m0` C1=0 / C6=2** — a scale operand rather than a type, and
  `cvt_scalef32_pk_*` restricts scales to powers of two; arbitrary scaling needs
  an unscaled cvt plus an explicit `fmul`.
- **`i1` C4=1** — predicates work, but there is no zero-vector primitive
  (guarded), so vector-width predicate init must be open-coded.
- **`i64` C4=1** — 8-byte LDS works and addressing is fine, but there is no
  `global_store_vN` and no zero-vec. It is an addressing type with no matrix atom.
- **`u8` — 0 at every layer.** `UINT8` appears in the hipDNN enum
  (`GpuReferenceValidationFactory.cpp:31`), but only in a reference-validation
  switch; no rocKE surface accepts it. Note that the silicon is not the
  constraint: the RDNA/GFX12 WMMA integer atom is `IU8`, whose per-operand
  signedness selects make it an *unsigned* int8 matmul as well as a signed one
  (§4A).
- **`bf16-mode` C1=0 / C5=0** — storage is plain `f32` (C4=3), and the lowering is
  **truncate to `bf16`, then run the existing bf16 matrix path** (C2=3, C3=2 —
  `cast_f32_to(v, BF16)` is already the f32→bf16 cast). What is missing is only
  the *selector*: there is no spec knob, dtype string, or compile define by which
  a caller asks for reduced-precision compute, so no family exposes it (C5=0) and
  the mode cannot be named in either engine (C1=0).
  **C6=1 — a naming obligation, not an unresolved design question.** The gpu-ref
  `USE_TF32` reference keeps 10 mantissa bits; `bf16` keeps 7. A bf16 route is
  therefore 3 mantissa bits coarser and **does not implement TF32 semantics**.
  XF32 is absent on gfx950/gfx1250 but present on gfx942 (§4D); its absence
  does not make BF16 the only possible software implementation. The
  obligation is to *describe* the mode accurately: it is a BF16 compute mode, its
  accuracy contract is bf16's, and `USE_TF32` gpu-ref output is not a valid
  bit-accuracy oracle for it (it remains a useful coarse reference for "reduced
  precision, not full f32"). Widening a tolerance around the TF32 reference would
  not make the two equivalent.
- **`f64` / `u16-64` — 0 at every layer.** Absent from the hipDNN enum, rejected
  by the MIOpen provider, and not requested by any family. For `f64` specifically
  `V_MFMA_F64_*` is present on gfx90a/gfx942/gfx950, where f64 is unwired
  software. gfx1250 has no F64 matrix instruction: LLVM gates
  `V_WMMA_F64_16X16X4_F64` on `gfx1251-gemm-insts` (§4A).

### 2.6 Source anchors for the scores

Every score above is traceable to: compiler `ir.py:44-53`,
`lower_llvm.py:1024-1060`,
FP-op guards `lower_llvm.py:1992-2057` (FMA/FABS/FMAX3 accept only f32/f16/bf16);
helpers `ir.py:849-1139` (casts), `helpers/io.py:48-211`,
`helpers/quant.py:58-357`, `helpers/i4_dequant.py`, `helpers/mx_scale.py`;
families `instances/common/gemm_universal.py:43,643-659`,
`instances/common/mfma_gemm.py:62-65,192-195`,
`instances/common/block_scale_gemm.py`, `instances/common/_matmul_nbits_common.py`,
[integer WMMA GEMM](../../python/rocke/instances/gfx1151/wmma_gemm_iu8.py),
[integer WMMA GEMM with dequantization](../../python/rocke/instances/gfx1151/wmma_gemm_iu8_dequant.py),
`library/kernels/common/attention_unified.py` (`UNIFIED_DTYPES=("fp16","bf16")`),
`library/kernels/common/fmha_fwd_fp8.py`; atoms
`core/arch/data/arch_specs.json`, `helpers/atoms.py:302`,
`cpp/core/lower_llvm/mma.cpp:571` (scalar f32 MFMA);
**hardware ceiling**
[`matrix_instructions_summary.md`](./matrix_instructions_summary.md)
(per-arch VOP3P opcode tables transcribed from the official AMD ISA documentation —
the index behind §4A, and behind the XF32, integer-MFMA, `IU8` and f64 findings);
reduced-precision reference
semantics `integration-tests/gpu-ref/kernels/types/GpuRefTypes.h:92-103`,
`integration-tests/gpu-ref/kernels/convolution/GpuRefConvFwd.cpp:54-56`.

---

## 3. Conversion coverage (helpers layer)

`f32` is the universal hub; every quantized/reduced type converts through it.
Documented conversion **gaps** (the "partially supported (missing conversions)"
bucket the ticket calls out):

| Missing / constrained conversion | Current workaround | Category |
|---|---|---|
| `f16 ↔ bf16` (direct) | route via `f32` (2 casts) | [C] |
| `fp8e4m3/bf8e5m2 → f16` and `→ bf16` | dequant to `f32`, then cast | [C] |
| `fp8/bf8 → i8` (and reverse) | via `f32` | [C] |
| `fp8/bf8` in `io.py` `load_scalar`/`store_scalar`/`pack_f32_to` | raises `ValueError`; must use IR primitives or `quant.py` | [P] |
| `sitofp_f32` primitive from `i8/i16` | Primitive accepts i32 only. For signed i8, use [quant.py](../../python/rocke/helpers/quant.py) `dequantize_scalar_to_f32(b, x, scale=b.const_f32(1.0))`; it emits sext + sitofp internally. For signed i16, explicitly sext to i32 first. | [P] |
| MX scaled cvt (`cvt_scalef32_pk_*`) | E8M0 scale only → **power-of-two scales**; arbitrary scale needs unscaled cvt + `fmul` | [P] |
| packed store `store_packed_chunk_local` | `n∈{4,8}` only (no `n=2`); `load_vec`/`store_vec` `n∈{2,4,8}` | [P] |
| scalar `cvt_fp4_to_f32` / `cvt_fp6_to_f32` | No dedicated scalar cvt primitive. [i4_dequant.py](../../python/rocke/helpers/i4_dequant.py) provides `unpack_fp4_byte_to_pair_f32` and `unpack_fp6_bytes_to_quad_f32`, which decode packed codebook values to f32 without MFMA. | [P] |
| stochastic rounding (any quant) | not implemented ("v2 follow-on") | [M] |

Rounding/saturation that **is** present: round-to-nearest-even + saturating
clamp for all `f32→{i8,fp8e4m3,bf8e5m2}` (`quant.py:141-204`, `ir.py:1128-1139`);
clamp magnitudes `i8=127`, `fp8e4m3=448`, `bf8e5m2=57344` (`quant.py:66-70`).

---

## 4. Hardware atom → dtype support (the hardware ceiling)

This section separates three different things that are easy to conflate: what the
**silicon** provides (§4A), what **rocKE declares** it can use (§4B), and the
**deltas** between them (§4C). §4D and §4E then take the two arch-boundary issues
that recur throughout this document — XF32/TF32 and the FP8 encoding dialect — and
state them once in full.

Nothing in §4A was measured. It is transcribed from the per-architecture AMD ISA
documentation via [`matrix_instructions_summary.md`](./matrix_instructions_summary.md);
for any specific instruction, refer to the official AMD ISA document for the
corresponding architecture.

### 4A. Silicon capability (per the per-arch AMD ISA documentation)

| Arch | ISA gen | Engine | f32 | f16 | bf16 | fp8/bf8 | i8 | i4 | fp4/fp6 | MX scale | f64 | XF32 (TF32) | Sparse |
|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| gfx90a | CDNA2 | MFMA | ✅ | ✅ | ✅ (1K) | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| gfx942 | CDNA3 | MFMA + SMFMAC | ✅ | ✅ | ✅ | ✅ **FNUZ** | ✅ | ❌ | ❌ | ❌ | ✅ | **✅ only here** | ✅ |
| gfx950 | CDNA4 | MFMA + SMFMAC | ✅ | ✅ (+K-packed) | ✅ (+K-packed) | ✅ **OCP** | ✅ (+K64) | ❌ | ✅ `F8F6F4` | ✅ | ✅ | ❌ removed | ✅ |
| gfx1151 | RDNA3.5 | WMMA | ❌ | ✅ | ✅ | ❌ | ✅ `iu8` | ✅ `iu4` | ❌ | ❌ | ❌ | ❌ | ❌ |
| gfx1201 | RDNA4 | WMMA + SWMMAC | ❌ | ✅ | ✅ | ✅ | ✅ `iu8` | ✅ `iu4` (+K32) | ❌ | ❌ | ❌ | ❌ | ✅ |
| gfx1250 | CDNA5 | WMMA + SWMMAC | ✅ | ✅ | ✅ | ✅ (K64/K128) | ✅ `iu8` | ❓ † | ✅ `F8F6F4` + `F4` | ✅ (+block-16) | ❌ | ❌ | ✅ |

† The gfx1250 integer matrix entries enumerate `V_WMMA_I32_16X16X64_IU8` /
`V_SWMMAC_I32_16X16X128_IU8` — no `IU4` form was located. Treat i4-on-gfx1250 as
**unconfirmed** and check the official ISA document before relying on it.

The ISA labels and instruction evidence follow the public AMD XML files linked
in [the ISA reference](./matrix_instructions_summary.md#sources-and-reproduction).
The gfx1250 source is `amdgpu_isa_cdna5.xml`; rocKE classifies the target as
`family: "cdna"`, `target_family: "gfx12_cdna"` (wave32, WMMA, no MFMA).
F64 WMMA is absent from that XML; the pinned LLVM target predicates restrict it
to gfx1251, so this is not a missing gfx1250 atom.

### 4B. What rocKE declares — the atom catalog

`core/arch/data/arch_specs.json` is rocKE's architecture SSOT. An atom that is not
declared there is unusable by any kernel regardless of what the silicon provides.
Verbatim at the basis commit:

| Arch | Declared matrix atoms | dtypes reachable |
|---|---|---|
| `gfx90a` | `mfma_f32_{16x16x16,32x32x8}_{f16,bf16}` | f16, bf16 |
| `gfx942` | the gfx90a set + `mfma_f32_{16x16x4,32x32x2}_f32` + `mfma_f32_{16x16x32,32x32x16}_{fp8,bf8}` | f32, f16, bf16, fp8, bf8 |
| `gfx950` | the gfx942 set + K-packed `{16x16x32,32x32x16}_{f16,bf16}` + `mfma_f32_16x16x128_fp4` + `mfma_f32_16x16x96_fp6` | f32, f16, bf16, fp8, bf8, **fp4, fp6** |
| `gfx1151` | `wmma_f32_16x16x16_{f16,bf16}`, `wmma_i32_16x16x16_{iu8,iu4}` | f16, bf16, **iu8, iu4** |
| `gfx1201` | `wmma_gfx12_f32_16x16x16_{f16,bf16}` | f16, bf16 |
| `gfx1250` | `wmma_gfx1250_f32_16x16x4_f32`, `..._16x16x32_{f16,bf16}`, `..._16x16x64_{fp8_fp8,fp8_bf8,bf8_fp8,bf8_bf8}`, `wmma_scale_f32_16x16x128_fp8_fp8`, `wmma_scale16_f32_16x16x128_fp8_fp8` | f32, f16, bf16, fp8, bf8 (+ fp8 block-scaled) |
| `gfx11-generic` | same as `gfx1151` | f16, bf16, iu8, iu4 |

Two points worth stating, because they are the ones most often mis-stated:

- **`gfx1250` is fully present in the catalog.** It carries a complete entry —
  target family, memory model, limits, and nine matrix atoms including the fp8
  `SCALE`/`SCALE16` block-scaled slice. gfx1250 is not missing from the catalog; §4C.1 states precisely which atoms it
  does omit.
- **No integer MFMA atom is declared on any CDNA arch**, even though the silicon
  has shipped `V_MFMA_I32_*_I8` since CDNA1. Integer matrix on CDNA is unwired
  software, not absent hardware.

### 4C. Deltas — silicon capability vs. rocKE's declared atoms

The following gaps compare §4A with §4B. The first four concern atom declarations;
f64 also lacks an IR type. The gfx942 XF32 declaration gap is covered in §4D.

#### 4C.1 gfx1250 — fp4/fp6/bf6 block-scaled atoms are omitted (deliberately)
The part carries `V_WMMA_F32_16X16X128_F8F6F4`, a dedicated
`V_WMMA_F32_32X16X128_F4`, and block-size-16 scaling. The catalog declares only the
**fp8×fp8** `SCALE`/`SCALE16` slice; the entry's own comment records that
"FP4/FP6/BF6 block-scaled forms remain intentionally omitted." The gap is therefore
**the fp4/fp6 block-scaled WMMA atoms specifically**, within an otherwise complete
and validated arch entry — it is not an absence of gfx1250 support.

#### 4C.2 gfx1201 — fp8/bf8 WMMA is in silicon but undeclared
RDNA4 provides `V_WMMA_F32_16X16X16_{FP8,BF8}_{FP8,BF8}` and the matching SWMMAC
sparse forms; the catalog declares f16/bf16 only, so fp8 on this arch is
undeclared rather than unavailable.

#### 4C.3 CDNA — integer MFMA is in silicon but undeclared everywhere
`V_MFMA_I32_*_I8` ships on every CDNA generation in §4A, and gfx950 adds the wide
`V_MFMA_I32_16X16X64_I8` / `V_MFMA_I32_32X32X32_I8` forms. rocKE declares none of
them on any CDNA arch. This is why `i8` scores C2=2 and C5=1 (§2.5).

#### 4C.4 Sparse (SMFMAC / SWMMAC) is entirely undeclared
Structured sparsity is available on gfx942, gfx950, gfx1201 and gfx1250. The
catalog declares no sparse atom on any arch, and no family requests one. No dtype
in §2.4 is blocked on it.

#### 4C.5 f64 matrix is in silicon but undeclared
`V_MFMA_F64_*` is present on gfx90a/gfx942/gfx950. `f64` is not an IR type
at all (§1), so those targets have a vocabulary gap as well as missing atoms.
gfx1250 has no native F64 matrix instruction and is excluded from this gap.

### 4D. XF32 / TF32 across the arch set

| Arch | XF32 matrix instruction |
|---|---|
| gfx90a (CDNA2) | ❌ never present |
| gfx942 (CDNA3) | ✅ `V_MFMA_F32_16X16X8_XF32` (op 62), `V_MFMA_F32_32X32X4_XF32` (op 63) |
| gfx950 (CDNA4) | ❌ absent from the public XML and LLVM target features |
| gfx1151 / gfx1201 (RDNA3.5 / RDNA4) | ❌ never present |
| gfx1250 (CDNA5) | ❌ absent from the public XML and LLVM target features |

**Consequence for rocKE.** gfx942 is a supported target: it has a catalog entry
and block-scale GEMM accepts it for FP8/BF8. It provides native XF32, but rocKE
does not declare XF32 atoms. gfx950/gfx1250 have no native XF32 instruction.
A BF16 compute mode on those targets would need its own explicit precision
contract; it cannot silently satisfy a TF32 request. Software approaches that
meet a TF32 contract are not ruled out by native XF32 absence (§1).

### 4E. The FP8 encoding dialect (FNUZ vs OCP)

Same 8 bits, two incompatible interpretations of the exponent bias:

| Arch | fp8/bf8 dialect |
|---|---|
| gfx942 | **FNUZ** |
| gfx950 | **OCP** |
| gfx1201, gfx1250 | **OCP** |

rocKE carries one `fp8e4m3` / `bf8e5m2` IR type per §1, with the dialect resolved
by target arch at lowering time rather than by the type. CK takes the opposite
approach (`f8_fnuz_t` vs `f8_ocp_t` as distinct types), which is the contrast
behind the `fp8*` C6=1 score (§2.5).

The failure mode is silence: the same byte pattern decodes to values a power of two
apart with no error raised anywhere. Today the FNUZ/OCP selection is guarded only
in the attention path; nothing generalizes it. gfx950/gfx1250 use OCP, while
the supported gfx942 target uses FNUZ, so callers must account for both dialects.

---

## 5. Per-family support snapshot

The "Not supported" column records what a family rejects or lacks, not what it is
scheduled to gain.

| Family | Supported | Not supported |
|---|---|---|
| Universal GEMM (`gemm_universal.py`) | f16, bf16 (A/B/C homogeneous, f32 accum) | **fp8/bf8** — `NotImplementedError` at `:659` for any dtype that is not `F16`/`BF16` |
| MFMA GEMM (`mfma_gemm.py`) | f16, bf16 | **fp8/bf8** — `:62-65` records that lane-decode and byte-wise A/B loads are absent |
| Integer WMMA GEMM ([wmma_gemm_iu8.py](../../python/rocke/instances/gfx1151/wmma_gemm_iu8.py)) | Signed int8 A/B packed four values per i32 → i32 accumulation/output; gfx1151 and gfx11-generic (`iu8` WMMA atom) | Raw byte-pointer inputs; targets without this atom; requires complete 16×16 tiles and K a multiple of 16 |
| Integer WMMA GEMM with dequantization ([wmma_gemm_iu8_dequant.py](../../python/rocke/instances/gfx1151/wmma_gemm_iu8_dequant.py)) | Same packed int8 input and i32 accumulation, runtime per-tensor scales → f16 output | Same target and packing constraints; no arbitrary output dtype selector |
| Block-scale GEMM ([block_scale_gemm.py](../../python/rocke/instances/common/block_scale_gemm.py)) | `abquant`, FP8/BF8 A/B → **f32** out on gfx942/gfx950 | `aquant`/`bquant`; `i4_fp8`/`i4_bf8`; non-f32 output; WMMA targets including gfx1250 |
| MatMul-NBits (`_matmul_nbits_common.py`) | A=f16, B=int4, scale f16/f32 (RDNA only) | B widths other than 4-bit; CDNA arches |
| Convolution (`_conv_implicit_gemm_common.py`) | f16, bf16, f32 | **reduced-precision compute over f32 storage** — no selector exists to request it (gpu-ref carries a `USE_TF32` switch; see §1); **fp8/bf8** |
| Attention unified (`attention_unified.py`) | Q/V f16, bf16; f32 accum | fp8/bf8 K/V is **not uniform across instances**; no int8 Q/V |
| Attention FP8 (`fmha_fwd_fp8.py`) | fp8/bf8 K/V decode (gfx950 preferred) | on gfx942 the FNUZ dialect must be selected explicitly (§4E) |
| Grouped convolution (dispatch) | f16, bf16 | fp8/bf8/int8 |
| Quant epilogues (`add_rmsnorm2d_rdquant`) | out i8, fp8e4m3, bf8e5m2 | — |

---
