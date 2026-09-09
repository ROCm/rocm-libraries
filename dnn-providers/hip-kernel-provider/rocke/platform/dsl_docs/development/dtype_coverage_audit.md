# rocKE Data-Type Coverage Audit

**Status:** point-in-time audit (read-only; no type support was added).
**Scope:** catalog what dtypes rocKE supports today across the **compiler**
(the two byte-identical LLVM-IR engines), the **helpers** (conversions / I/O /
quantization), and the **kernel families**; categorize every gap; and give a
prioritized list of types to add or complete.

> Compliance note: this document records *methodology and levers* only. It
> contains no measured, software-achieved performance numbers (per
> `platform/AGENTS.md` §Compliance). Where a dtype has a perf implication it is
> described as a lever, not a benchmark.

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

**`tf32` is a compute mode, not a storage type.** It never appears as a tensor
dtype: storage stays `f32` and the *precision* is reduced before the MMA. The
gpu-ref convolution models exactly this — `truncateToTf32()` zeroes the bottom 13
mantissa bits of an `f32` (`GpuRefTypes.h:92-103`), gated by a `USE_TF32` compile
define. AMD's native TF32 equivalent, **XF32**, exists in silicon on gfx942 only
and is removed on both of rocKE's target arches (§4D), so the mode must be
emulated. What is missing is the **mode itself**: no spec knob selects
reduced-precision compute, so a hipDNN TF32 request cannot be expressed. See
§2.5 and P1.

**Types that are entirely absent from the vocabulary:** `f64`, unsigned
`u8/u16/u32/u64`, and a distinct `bool` (predicates reuse `i1`). No engine
references them.

Engine parity: **10/10 identical** — no dtype exists in one engine but not the
other. This is the byte-identity invariant working as intended.

---

## 2. Coverage matrix — support status by layer

### 2.1 Why explicit criteria

A per-layer verdict of Full / Partial / Absent is not enough on its own: it never
says *what makes a layer Full*. Without that, two things are impossible —
comparing two "Partial" types against each other, and justifying a priority order
to anyone who didn't write the audit. §2.2 therefore names six **support
criteria** and one **demand weight**, each with the external requirement it is
derived from; §2.4 scores every dtype against them.

The criteria are chosen so that each one is (a) independently observable in the
source, and (b) traceable to a requirement from one of the three ecosystems the
ticket names: **Composable Kernel** (the reference for what a mature AMD kernel
library supports per dtype), **hipDNN** (the descriptor/graph interface rocKE
must honor — its `data_objects::DataType` enum is the contract), and **MIOpen**
(the incumbent provider whose accepted dtype set defines the must-not-regress
floor).

### 2.2 The six support criteria (C1–C6)

| # | Criterion | What it asks | Rationale — where the requirement comes from |
|---|---|---|---|
| **C1** | **IR representability** | Is the dtype a first-class `Type` in *both* engines, byte-identically? | hipDNN attaches a `DataType` to every tensor descriptor; if rocKE cannot *name* the type, no descriptor carrying it can ever be honored. CK makes the same choice — every supported type is a real C++ type (`half_t`, `bhalf_t`, `f8_t`, …), not an untyped byte blob. This is the gate all other criteria sit behind. |
| **C2** | **Compute + MMA lowering** | Can the emitter lower arithmetic on it, and does a matrix atom (MFMA/WMMA) accept it on some target arch? | This is the difference between "storable" and "fast." CK dispatches its xdlops/wmma pipelines per dtype; MIOpen's conv solvers are keyed by dtype for the same reason. A dtype with no atom can only ever be a storage/epilogue type, which caps every downstream family. |
| **C3** | **Conversion completeness** | Are casts to **and** from the `f32` hub present, plus the direct sibling casts that avoid a double bounce, with correct rounding/saturation? | Every hipDNN plan in this repo asserts `computeDataType() == DataType::FLOAT` (SDPA, RMSNorm, batchnorm, resample), so *every* tensor dtype must round-trip through f32 or it cannot participate. CK's `type_convert<Dst,Src>` is deliberately **total** over its type set; partial conversion coverage is the single most common way a dtype is "supported" on paper but unusable in a fusion. |
| **C4** | **Memory I/O plumbing** | Scalar load/store, vectorised (`n∈{2,4,8}`) load/store, and pack/unpack at the helper surface — not just raw IR primitives. | A tensor dtype is only usable if a kernel can move it. CK exposes `buffer_load`/`buffer_store` at every vector width for each supported type; rocKE's equivalent is `helpers/io.py`. If `io.py` raises `ValueError` on a dtype, every small-op family (norm, reduce, elementwise, transpose) is closed to it regardless of what the compiler can express. |
| **C5** | **Kernel-family reach** | How many shipping families actually accept it — GEMM, MFMA-GEMM, conv, attention, norm/quant epilogues? | This is the criterion the *user* feels. The demand set is the hipDNN op surface this provider implements: conv fwd/bwd/wgrad, SDPA fwd/bwd, batchnorm, RMSNorm, resample, matmul. A dtype that passes C1–C4 but ships in zero instances delivers nothing. |
| **C6** | **Numerical / dialect hygiene** | Is rounding + saturation defined, and is the *encoding dialect* unambiguous at the arch boundary? | Purely a correctness criterion, and the one with the worst failure mode: silence. gfx942 fp8 is **FNUZ** and gfx950 fp8 is **OCP** — same bytes, results off by a power of two. CK models this explicitly with distinct `f8_fnuz_t` / `f8_ocp_t` types; rocKE currently carries it as an arch-dependent decode dialect, so this criterion tracks real risk, not pedantry. MX types add a second axis (E8M0 scales are power-of-two only). |

Not every criterion applies to every dtype. `C3` and `C6` are **N/A** for exact
integer types used only for indexing (`i1`, `i16`, `i32`, `i64`) — there is no
rounding to get wrong and no float bridge they are required to have.

### 2.3 Scoring rubric

Each criterion is scored **0–3**. `N/A` criteria are dropped from both the
numerator and the denominator, so `Support%` is comparable across rows with
different applicable-criterion counts.

| Score | Meaning |
|---|---|
| **3 — Full** | No constraint a kernel author would hit. |
| **2 — Mostly** | Works, with a documented constraint or an extra step (e.g. an obligatory `f32` bounce, a generic rather than dedicated primitive). |
| **1 — Partial** | Exists in one direction / one surface / one arch only; a kernel author must route around it. |
| **0 — Absent** | Not available at this layer at all. |

**Demand weight (D 0–3)** — this is *not* a support score; it weights how much a
gap matters. It is read off the three external ecosystems:

| D | Definition | Evidence |
|---|---|---|
| **D3** | Required by a shipping hipDNN or MIOpen op **today**, or named as a committed roadmap requirement | `MiopenUtils.cpp:12-26` accepts exactly `FLOAT`/`HALF`/`BFLOAT16` and throws `BAD_PARAM` on anything else — that trio is the non-negotiable floor. Plus `INT32` (resample index tensors, `ResampleApplicabilityChecks.cpp:64`), and `tf32` as a compute mode (gpu-ref conv already implements `USE_TF32`, `GpuRefTypes.h:92-103`). |
| **D2** | In the hipDNN `DataType` enum **and** has a hardware atom **and** a rocKE family is blocked on it | e.g. `FP8_E4M3` is accepted by asm-SDPA fwd (`SdpaFwdPlanBuilder.cpp:87-88`, Q/K/V fp8 → bf16 out) and mapped by the hipBLASLt provider; CK ships fp8 GEMM broadly. |
| **D1** | Named in the hipDNN enum but with no rocKE family demand yet, **or** demanded only on an arch outside the current focus | `FP6_E2M3`, `UINT8`, `INT64` (index-only); also `fp8 fnuz`, whose only consumer is gfx942 (see the arch-focus note below). |
| **D0** | Not in the hipDNN enum, not used by MIOpen, no CK demand for this provider | `f64`, `u16/u32/u64`. |

**Hardware readiness (H)** — a multiplier, not a score: `H = 1.0` if a matrix
atom or native instruction exists on some target arch (so the work is
*plumbing*), `H = 0.3` if it does not (so the work is *green-field*, and a high
demand-times-gap number would otherwise overstate it).

**Priority index** = `D × (1 − Support%) × H`. It ranks *demand-weighted,
hardware-adjusted gap*. It deliberately does **not** model effort — §6
reconciles the index against effort to produce the shipping order.

**Arch focus (applies to D).** rocKE's current target arches are **gfx950 and
gfx1250**. Demand that exists only on gfx942 is therefore discounted by one D
level. The concrete consequence is `fp8 fnuz`: the FNUZ encoding is a gfx942-only
dialect — gfx950 fp8 is OCP — so despite a large support gap it scores **D1**,
not D2, and drops out of the P0 bucket. It remains listed because the FNUZ/OCP
split is a *silent* mis-decode risk (§4E), not because the encoding itself needs
investment.

> Sourcing note: the hipDNN and MIOpen columns are cited from files in this
> repo (paths above). The **CK** rationale reflects the public
> Composable-Kernel / CK-Tile type vocabulary — there is no CK checkout in this
> tree, so CK is used as a design reference for *what a criterion should
> demand*, not as a line-cited source.

### 2.4 Scored matrix

Scores 0–3 per criterion; `—` = N/A (excluded from the denominator). Rows are
ordered by priority index, descending.

Gap categories: **[C]** conversion gap · **[P]** partial/plumbing gap ·
**[M]** missing entirely.

| dtype | C1 IR | C2 MMA | C3 cvt | C4 I/O | C5 family | C6 numeric | Support% | D | H | **Priority idx** | Gap cat. |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `fp8e4m3` (OCP) | 3 | 3 | 1 | 0 | 1 | 1 | **50%** | 3 | 1.0 | **1.50** | [C][P] |
| `tf32` (compute mode) | 0 | 3 | 2 | 3 | 0 | 1 | **50%** | 3 | 1.0 | **1.50** | [P][M] |
| `fp4` (E2M1, MX) | 0 | 2 | 1 | 1 | 0 | 2 | **33%** | 2 | 1.0 | **1.33** | [P][M] |
| `bf8e5m2` (OCP) | 3 | 3 | 1 | 0 | 1 | 1 | **50%** | 2 | 1.0 | **1.00** | [C][P] |
| `u8` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | 1 | 1.0 | **1.00** | [M] |
| `i4` (packed) | 0 | 1 | 3 | 2 | 2 | 2 | **56%** | 2 | 1.0 | **0.89** | [P] |
| `fp6` (E3M2/E2M3, MX) | 0 | 2 | 1 | 1 | 0 | 2 | **33%** | 1 | 1.0 | **0.67** | [P][M] |
| `fp8 fnuz` (gfx942 only) | 1 | 3 | 1 | 0 | 1 | 1 | **39%** | 1 | 1.0 | **0.61** | [P] |
| `i8` | 3 | 2 | 2 | 2 | 1 | 3 | **72%** | 2 | 1.0 | **0.56** | [P] |
| `e8m0` (MX scale) | 0 | 3 | 3 | 2 | 3 | 2 | **72%** | 2 | 1.0 | **0.56** | — |
| `i64` | 3 | 3 | 1 | 1 | 1 | — | **60%** | 2 | 0.3 | **0.24** | [P] |
| `f16` | 3 | 3 | 2 | 3 | 3 | 3 | **94%** | 3 | 1.0 | **0.17** | [C] |
| `bf16` | 3 | 3 | 2 | 3 | 3 | 3 | **94%** | 3 | 1.0 | **0.17** | [C] |
| `i1` | 3 | 3 | — | 1 | 3 | — | **83%** | 1 | 1.0 | **0.17** | [P] |
| `i16` | 3 | 2 | 1 | 2 | 1 | — | **60%** | 1 | 0.3 | **0.12** | [P] |
| `f32` | 3 | 3 | 3 | 3 | 3 | 3 | **100%** | 3 | 1.0 | **0.00** | — |
| `i32` | 3 | 3 | 3 | 3 | 3 | — | **100%** | 3 | 1.0 | **0.00** | — |
| `f64` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | 0 | 1.0 | **0.00** | [M] |
| `u16/u32/u64` | 0 | 0 | 0 | 0 | 0 | 0 | **0%** | 0 | 0.3 | **0.00** | [M] |

`tf32` is scored as a **compute mode over `f32` storage**, which is why its
profile is unlike every other row: the storage and I/O columns are already Full
(C4=3, it *is* `f32`), the compute column is Full (C2=3, bf16 MFMA atoms exist),
and the entire gap sits in C1/C5 — nothing can *select* the mode, so no family
offers it.

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
- **`fp8 fnuz` C1=1, D1** — not a distinct IR type at all; it shares `i8` storage
  with OCP fp8 and is distinguished only by an arch-dependent decode dialect.
  CK's separate `f8_fnuz_t` / `f8_ocp_t` types are the contrast that makes this
  a 1 rather than a 3. **D is 1, not 2**, because FNUZ is a gfx942-only encoding
  and rocKE's current targets are gfx950 and gfx1250, both of which use OCP fp8.
  The remaining work here is a *guard* — make the dialect choice explicit so a
  gfx942 tensor cannot be silently read as OCP — not an investment in the
  encoding.
- **`i8` C2=2** — full integer arithmetic and an `iu8` WMMA atom exists on
  gfx1151, but no CDNA integer MFMA path is wired *in rocKE*. Note the ceiling is
  software, not silicon: `V_MFMA_I32_*_I8` ships on every CDNA generation
  including gfx942 and gfx950 (§4A), and gfx950 adds K64/K32 wide forms. **C3=2** — `f32→i8` with RNE
  and saturating clamp at ±127 is present (`quant.py:66-70,141-204`), but
  `i8→f32` requires a manual sext to `i32` first because `sitofp_f32` is
  i32-only. **C5=1** — quant epilogues only (`add_rmsnorm2d_rdquant`); no
  integer GEMM or attention instance ships.
- **`i4` C1=0 / C3=3** — the inverse of the fp8 profile: not a `Type` at all,
  yet its conversion coverage is the *best* of any quantized type
  (`i4_dequant.py` targets i32, f32, f16, fp8, bf8). C2=1: `iu4` WMMA on RDNA
  only. The ISA confirms this is a *silicon* limit, not a wiring one — INT4 is
  absent from every CDNA generation through CDNA4 (§4A), so an i4 path on gfx950
  is impossible by construction. gfx1250 is the one target arch that may carry
  `iu4`, but the opcode is unconfirmed (§4A †), which is why C2 stays at 1.
- **`fp4` / `fp6` C5=0** — this is the sharpest finding in the matrix. The
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
  `global_store_vN` and no zero-vec. H=0.3 because it is an addressing type with
  no matrix atom; the gap is real but nothing is blocked on it.
- **`u8` D1/H=1.0** — `UINT8` appears in the hipDNN enum
  (`GpuReferenceValidationFactory.cpp:31`), but only in a reference-validation
  switch, so demand stays at D1. **H is 1.0, not 0.3**: the RDNA WMMA integer
  atom is `IU8`, whose per-operand signedness selects make it an *unsigned* int8
  matmul as well as a signed one (§4A), and gfx1250 is a target arch. With zero
  support at every layer that yields an index of 1.00 — high, and honestly so, but
  it stays unscheduled because D1 reflects real demand rather than potential
  (see §6.0).
- **`tf32` C1=0 / C5=0, D3** — the highest-leverage row in the matrix, because
  almost everything it needs already exists. Storage is plain `f32` (C4=3), and
  the planned lowering is **truncate to `bf16`, then run the existing bf16 MFMA
  path** (C2=3, C3=2 — `cast_f32_to(v, BF16)` is already the f32→bf16 cast). What
  is missing is only the *selector*: there is no spec knob, dtype string, or
  compile define by which a caller asks for reduced-precision compute, so no
  family exposes it (C5=0) and the mode cannot be named in either engine (C1=0).
  D3 because gpu-ref already ships the reference semantics (`USE_TF32`,
  `GpuRefTypes.h:92-103`) — the oracle exists before the kernel does. The ISA
  removes the alternative: native **XF32** exists on gfx942 only and is deleted on
  gfx950 and gfx1250 (§4D), so emulation is the only route on the target arches.
  **C6=1, and this is the open question:** gpu-ref defines TF32 as f32 with the
  bottom 13 mantissa bits zeroed, i.e. **10 mantissa bits**; `bf16` has **7**.
  Lowering tf32 to bf16 is therefore 3 mantissa bits *less* precise than the
  reference it will be validated against, so the bf16 route needs either a
  widened tolerance in the conv parity harness or an explicit statement that
  rocKE's TF32 mode is bf16-accurate by design. Flagging it as the one decision
  to settle before P1 lands; it does not change the priority.
- **`f64` / `u16-64` D0** — absent from the hipDNN enum, rejected by the MIOpen
  provider, and not requested by any family. Zero demand, so zero priority
  regardless of zero support. For `f64` specifically, **H=1.0**: `V_MFMA_F64_*`
  ships on gfx90a/gfx942/gfx950 and `V_WMMA_F64_16X16X4_F64` on gfx1250 (§4A), so
  f64 is unwired software, not missing silicon. The index stays 0.00 because D=0.

### 2.6 Source anchors for the scores

Every score above is traceable to: compiler `ir.py:44-53`,
`lower_llvm.py:1024-1060`,
FP-op guards `lower_llvm.py:1992-2057` (FMA/FABS/FMAX3 accept only f32/f16/bf16);
helpers `ir.py:849-1139` (casts), `helpers/io.py:48-211`,
`helpers/quant.py:58-357`, `helpers/i4_dequant.py`, `helpers/mx_scale.py`;
families `instances/common/gemm_universal.py:43,643-659`,
`instances/common/mfma_gemm.py:62-65,192-195`,
`instances/common/block_scale_gemm.py`, `instances/common/_matmul_nbits_common.py`,
`library/kernels/common/attention_unified.py` (`UNIFIED_DTYPES=("fp16","bf16")`),
`library/kernels/common/fmha_fwd_fp8.py`; atoms
`core/arch/data/arch_specs.json`, `helpers/atoms.py:302`,
`cpp/core/lower_llvm/mma.cpp:571` (scalar f32 MFMA);
**hardware ceiling and the H multiplier**
[`matrix_instructions_summary.md`](matrix_instructions_summary.md) (per-arch VOP3P
opcode tables — the source of truth for §4A, and for the XF32, integer-MFMA, `IU8`
and f64 findings); TF32 reference
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
| `sitofp` from `i8/i16` | sext/zext to `i32` first (`sitofp_f32` is i32-only) | [P] |
| MX scaled cvt (`cvt_scalef32_pk_*`) | E8M0 scale only → **power-of-two scales**; arbitrary scale needs unscaled cvt + `fmul` | [P] |
| packed store `store_packed_chunk_local` | `n∈{4,8}` only (no `n=2`); `load_vec`/`store_vec` `n∈{2,4,8}` | [P] |
| scalar `cvt_fp4_to_f32` / `cvt_fp6_to_f32` | not exposed; fp4/fp6 only via MX-MFMA path | [P] |
| stochastic rounding (any quant) | not implemented ("v2 follow-on") | [M] |

Rounding/saturation that **is** present: round-to-nearest-even + saturating
clamp for all `f32→{i8,fp8e4m3,bf8e5m2}` (`quant.py:141-204`, `ir.py:1128-1139`);
clamp magnitudes `i8=127`, `fp8e4m3=448`, `bf8e5m2=57344` (`quant.py:66-70`).

---

## 4. Hardware atom → dtype support (demand ceiling)

### 4A. Silicon capability (per `matrix_instructions_summary.md`)

| Arch | ISA gen | Engine | f32 | f16 | bf16 | fp8/bf8 | i8 | i4 | fp4/fp6 | MX scale | f64 | XF32 (TF32) | Sparse |
|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| gfx90a | CDNA2 | MFMA | ✅ | ✅ | ✅ (1K) | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| gfx942 | CDNA3 | MFMA + SMFMAC | ✅ | ✅ | ✅ | ✅ **FNUZ** | ✅ | ❌ | ❌ | ❌ | ✅ | **✅ only here** | ✅ |
| gfx950 | CDNA4 | MFMA + SMFMAC | ✅ | ✅ (+K-packed) | ✅ (+K-packed) | ✅ **OCP** | ✅ (+K64) | ❌ | ✅ `F8F6F4` | ✅ | ✅ | ❌ removed | ✅ |
| gfx1151 | RDNA3.5 | WMMA | ❌ | ✅ | ✅ | ❌ | ✅ `iu8` | ✅ `iu4` | ❌ | ❌ | ❌ | ❌ | ❌ |
| gfx1201 | RDNA4 | WMMA + SWMMAC | ❌ | ✅ | ✅ | ✅ | ✅ `iu8` | ✅ `iu4` (+K32) | ❌ | ❌ | ❌ | ❌ | ✅ |
| gfx1250 | RDNA5 | WMMA + SWMMAC | ✅ | ✅ | ✅ | ✅ (K64/K128) | ✅ `iu8` | ✅ `iu4` † | ✅ `F8F6F4` + `F4` | ✅ (+block-16) | ✅ | ❌ removed | ✅ |

† gfx1250 INT4 is ✅ in the cross-architecture capability summary, but the dense
opcode table enumerates only `V_WMMA_I32_16X16X64_IU8` /
`V_SWMMAC_I32_16X16X128_IU8` — no `IU4` form. Treat i4-on-gfx1250 as **unconfirmed**
until the opcode is located.

gfx1250 is absent from `matrix_instructions_summary.md`; its row comes from
`arch_specs.json` alone, so the `?` cells are *unverified*, not known-absent.

---

## 5. Per-family support snapshot

| Family | Supported today | Missing / required |
|---|---|---|
| Universal GEMM (`gemm_universal.py`) | f16, bf16 (A/B/C homogeneous, f32 accum) | **fp8/bf8 emitter path** — `NotImplementedError` at `:659`; "no atom yet, mechanical extension" `:43` |
| MFMA GEMM (`mfma_gemm.py`) | f16, bf16 | **fp8/bf8** — needs lane-decode + byte-wise A/B loads `:62-65` |
| Block-scale GEMM (`block_scale_gemm.py`) | fp8, bf8, i4 A/B → f16 out (CDNA) | non-f16 output; broader arch reach |
| MatMul-NBits (`_matmul_nbits_common.py`) | A=f16, B=int4, scale f16/f32 (RDNA only) | other B widths; CDNA reach |
| Convolution (`_conv_implicit_gemm_common.py`) | f16, bf16, f32 | **tf32 compute mode** — gpu-ref already implements `USE_TF32`, rocKE has no selector; **fp8/bf8** (mechanical) |
| Attention unified (`attention_unified.py`) | Q/V f16, bf16; f32 accum | **fp8/bf8 K/V pipeline not uniform across instances**; no int8 Q/V |
| Attention FP8 (`fmha_fwd_fp8.py`) | fp8/bf8 K/V decode (gfx950 preferred) | gfx942 needs FNUZ flag; VGPR-pressure occupancy lever |
| Grouped convolution (dispatch) | f16, bf16 | fp8/bf8/int8 |
| Quant epilogues (`add_rmsnorm2d_rdquant`) | out i8, fp8e4m3, bf8e5m2 | — |

---

## 6. Prioritized list of types to add / complete

Priority is ranked by (a) hardware readiness — the atom already exists, so it's
plumbing not silicon; (b) breadth of kernel-family demand; (c) correctness risk
if left half-done.

### 6.0 Reconciling the priority index (§2.4) with shipping order

The §2.4 index measures **demand-weighted, hardware-adjusted gap**. It does not
model effort, so it is an input to the shipping order, not the order itself:

| Rank by index | dtype | Index | Effort | Shipping bucket |
|---|---|---|---|---|
| 1= | `fp8e4m3` (OCP) | 1.50 | Low — atom + cvt exist; plumbing only | **P0** |
| 1= | `tf32` (compute mode) | 1.50 | Low — a mode selector + existing bf16 path | **P1** |
| 3 | `fp4` (E2M1, MX) | 1.33 | **High** — block-scale MX plumbing + E8M0 scales, gfx950-only | **P4** |
| 4= | `bf8e5m2` (OCP) | 1.00 | Low — shares fp8's path | **P0** |
| 4= | `u8` | 1.00 | Medium — `IU8` reuse of the int8 path | not scheduled |
| 5 | `i4` | 0.89 | Medium — CDNA atom missing | **P5** |
| 6 | `fp6` | 0.67 | High — same MX plumbing as fp4 | **P4** |
| 7 | `fp8 fnuz` | 0.61 | Low — a dialect flag, not a type | **P6** (gfx942-only; guard, not investment) |
| 8= | `i8` | 0.56 | Medium — one new GEMM instance | **P2** |
| 8= | `e8m0` | 0.56 | — (rides the fp4/fp6 work) | **P4** |
| 10 | `i64`, `i1`, `i16` | ≤0.24 | — | not scheduled |

Four divergences are deliberate and worth stating:

- **`fp4` ranks #3 by index but ships at P4.** Its gap is large and the demand is
  real (the hipBLASLt provider already maps `FP4_E2M1`), but it is the only
  entry whose remaining work is a *subsystem* — MX block-scale plumbing — rather
  than filling in an existing one. High index, high effort.
- **`f16 ↔ bf16` ships at P3 despite a 0.17 index.** The index is low because
  both types are otherwise complete; the work is scheduled early anyway because
  it is a few hours of effort and it is the one gap every mixed-precision fusion
  hits. Low index, near-zero effort.
- **`fp8 fnuz` is deprioritized despite a 39% support score.** FNUZ is a
  gfx942-only encoding; gfx950 and gfx1250 — the current targets — use OCP fp8.
  What survives from it is a small correctness guard folded into P0 (make the
  dialect explicit so gfx942 bytes cannot be read as OCP), not an effort to
  complete the encoding.
- **`u8` ties for #4 by index yet is not scheduled at all.** The hardware is
  genuinely there — `IU8` covers unsigned int8 on RDNA — and rocKE's
  support is zero, so the formula correctly returns 1.00. What the formula cannot
  see is that `UINT8`'s only appearance is a reference-validation switch, not an op
  a caller can request. It is listed at its true index rather than suppressed, so
  that if a real `u8` consumer ever appears the work is already sized; until then
  it rides on P2's integer path rather than being scheduled on its own.

### P0 — complete fp8/bf8 (OCP) end-to-end
**Why:** the OCP atoms already exist on gfx950 (and gfx1250 continues the OCP
lineage), the compiler cvt path is Full, and multiple families are blocked at the
plumbing layer, not the hardware layer. Highest demand-to-effort ratio.
- Add the **fp8/bf8 emitter path in universal GEMM** (`gemm_universal.py:659`)
  and **MFMA GEMM** lane-decode/byte-wise A/B loads (`mfma_gemm.py:62-65`).
- Make the **attention fp8 K/V pipeline uniform across every instance**
  (currently documented-but-partial: `quantization.md:143,154`).
- Add **`io.py` fp8/bf8 load/store/pack** so the I/O helper surface stops
  raising `ValueError` and matches the quant surface.
- Close direct-cast gaps **`fp8/bf8 → f16/bf16`** to avoid the mandatory
  `f32` bounce in mixed-precision inner loops.
- **Correctness gate (small):** make the FNUZ-vs-OCP selection explicit so gfx942
  bytes can't be silently read as OCP (already guarded in attention; generalize
  it). This is the only FNUZ work in scope — see P6.
- **Arch reach (cheap add-on):** gfx1201 (RDNA4) has fp8/bf8 WMMA in silicon but
  rocKE declares only f16/bf16 atoms for it (§4C.2). Declaring the RDNA4 fp8 atoms
  extends this bucket to a third arch for little extra work.

### P1 — tf32 compute mode, lowered to bf16
**Why:** ties `fp8e4m3` for the top priority index (1.50) and is the cheapest
entry on the list, because every piece except the selector already exists.
Storage is plain `f32` (I/O already Full), the reference semantics already ship
in gpu-ref (`USE_TF32`, `GpuRefTypes.h:92-103`), and the compute path is the
existing, fully-supported **bf16** MFMA pipeline. The design decision is settled:
**tf32 tensors are converted to `bf16` before any operation** — rocKE does not
introduce a tf32 IR type. The ISA makes this the only viable route rather than
merely the preferred one: native XF32 is gfx942-only and is removed on gfx950 and
gfx1250 (§4D). The scalar f32 MFMA atoms remain available as a
precision-preserving fallback, but they are full-f32 and narrow-K, not a TF32 path.
- Add a **compute-mode selector** — a spec knob / dtype string that requests
  reduced-precision compute over `f32` storage. This is the whole `[M]` gap;
  today nothing can ask for it.
- Wire the mode to `cast_f32_to(v, BF16)` on the A/B ingest path, then reuse the
  existing bf16 MFMA atoms unchanged. No new atom, no new IR type.
- Extend **convolution** first (`_conv_implicit_gemm_common.py`) since gpu-ref
  already has the matching `USE_TF32` reference, then universal GEMM.
- **Settle the tolerance question before landing:** gpu-ref TF32 keeps 10
  mantissa bits, `bf16` keeps 7 (§2.5). Either widen the conv parity tolerance
  for this mode or document that rocKE's TF32 mode is bf16-accurate by design.
  Whichever is chosen, state it in the parity harness rather than leaving the
  discrepancy to be rediscovered.

### P2 — int8 GEMM/attention path
**Why:** `i8` is Full in compiler+helpers and gfx1151 has native `iu8` WMMA
(accum `i32`), but there is **no integer GEMM/attention instance** — only quant
epilogues. Unblocks INT8 inference on RDNA. The ISA comparison strengthens this
bucket: integer MFMA (`V_MFMA_I32_*_I8`) exists on *every* CDNA generation
including both target arches (§4C.3), so the CDNA side is unwired software rather
than absent silicon, and gfx950 additionally offers K64/K32 wide forms.
- Add an i8×i8→i32 GEMM instance wired to the `wmma_i32_16x16x16_iu8` atom.
- Declare CDNA i8 MFMA atoms in `arch_specs.json` (none exist today) and extend
  the instance to gfx950.
- Add `i8/i16 → f32` direct `sitofp` (drop the i32-extend step).

### P3 — f16 ↔ bf16 direct cast
**Why:** small, self-contained conversion gap `[C]` that every mixed-precision
kernel hits; removing the `f32` bounce simplifies fusions. Low effort.

### P4 — ship fp4/fp6 (OCP MX) kernel instances
**Why:** gfx950 **atoms exist** (`mfma_f32_16x16x128_fp4`,
`mfma_f32_16x16x96_fp6`) and the codebook unpackers exist, but **no kernel
instance ships** and there is no scalar cvt. High future value, larger effort
(block-scale MX plumbing + E8M0 scales). In the ISA these are the single
`F8F6F4` mixed-format instruction plus its `V_MFMA_SCALE_*` microscaling form,
so fp8/fp6/fp4 share one opcode family and the plumbing is shared work. gfx1250
carries the same family plus a dedicated `V_WMMA_F32_32X16X128_F4` and
block-size-16 scales, so this is no longer gfx950-only — but gfx1250 has no
`arch_specs.json` entry yet (§4C.1), which must land first.
- Optionally expose scalar `cvt_fp4_to_f32` / `cvt_fp6_to_f32`.

### P5 — i4 as a broader first-class quantized weight path
**Why:** works today only inside `matmul_nbits` (RDNA) and block-scale mantissa.
Generalizing i4 weight consumption to CDNA GEMM broadens WOQ coverage.

### P6 — stochastic rounding for quant; `fp8 fnuz` completion
**Stochastic rounding —** explicitly deferred ("v2 follow-on", `quant.py:19`).
Needed for some training-time quant; no hardware blocker, moderate effort.

**`fp8 fnuz` —** deliberately parked. FNUZ is gfx942-only and rocKE is targeting
gfx950 and gfx1250, both OCP. Beyond the P0 correctness guard, promoting FNUZ to a
distinct IR type (as CK does with `f8_fnuz_t`) buys nothing on the current
roadmap. Revisit only if gfx942 re-enters scope as a first-class target.

### Not planned (documented as out of vocabulary)
`f64`, unsigned integer types, and a distinct `bool` — none are referenced by any
engine, helper, or family. Adding any is a green-field type, not a gap to
"complete." Track only if a concrete kernel spec requires them. (`tf32` was
previously in this bucket; it is now P1 — it is a compute mode over `f32`, not a
green-field type.)

---

## 7. Acceptance-criteria checklist

- [x] **Evaluation criteria explicitly defined** — six support criteria C1–C6
      with a stated rationale each (§2.2), a 0–3 rubric (§2.3), and a demand
      weight D derived from the CK type vocabulary, the hipDNN `DataType` enum,
      and the MIOpen provider's accepted set (§2.3).
- [x] **Every dtype scored per criterion** — §2.4, with a per-score
      justification for every non-3 in §2.5.
- [x] **Coverage matrix complete** for all relevant dtypes across compiler /
      helpers / kernel families — §2.4 (+ conversion sub-matrix §3, atom
      ceiling §4, per-family snapshot §5).
- [x] **Every gap categorized** — `[C]` conversion, `[P]` partial/plumbing,
      `[M]` missing entirely — in §2.4 and §3.
- [x] **Priority order with rationale tied to kernel-family need** — §6
      (P0 fp8/bf8 → P6), each justified by hardware readiness + family demand
      under the current gfx950 / gfx1250 arch focus, and reconciled against the
      computed priority index in §6.0 with the four divergences called out.
- [x] **Hardware ceiling verified against the ISA source of truth** — §4A/4B
      separate silicon capability (`matrix_instructions_summary.md`) from rocKE's
      declared atoms (`arch_specs.json`), and §4C lists the five deltas between
      them.

*Audit basis: read-only exploration of the compiler engines, `helpers/`,
`instances/`, `library/kernels` + `library/dispatch`, `arch_specs.json`, and the
existing `dsl_docs` precision/quantization notes. No source or type support was
modified.*
