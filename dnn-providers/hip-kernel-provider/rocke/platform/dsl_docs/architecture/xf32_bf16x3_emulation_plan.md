# XF32 via BF16x3 MMA emulation — solution plan

**Goal:** capability-based TF32 emulation — a logical TF32 MMA atom that lowers to three
BF16 MMAs on a target with no native XF32 instruction (gfx950), next to the native atoms
gfx942 already has.
**Reference implementation:** Composable Kernel, public repo `ROCm/composable_kernel`

This document is a design/implementation plan, not a benchmark record. It contains no
measured performance results.

## Contents

- [1. What CK actually does (verified against the source)](#1-what-ck-actually-does-verified-against-the-source)
  - [1.1 The logical dtype: a tag in CK, a typed operand in rocke](#11-the-logical-dtype-a-tag-in-ck-a-typed-operand-in-rocke)
  - [1.2 Native XF32 where the target has it](#12-native-xf32-where-the-target-has-it)
  - [1.3 The BF16x3 emulation on gfx950](#13-the-bf16x3-emulation-on-gfx950)
  - [1.4 Rounding mode — the one real parity hazard](#14-rounding-mode--the-one-real-parity-hazard)
- [2. Current rocke state](#2-current-rocke-state)
  - [2.1 Starting point: native TF32 on gfx942 only](#21-starting-point-native-tf32-on-gfx942-only)
  - [2.2 No result-dtype split is needed](#22-no-result-dtype-split-is-needed)
  - [2.3 The split primitive already exists and already has the right rounding](#23-the-split-primitive-already-exists-and-already-has-the-right-rounding)
  - [2.4 Physical bf16 atoms are already registered with verified layouts](#24-physical-bf16-atoms-are-already-registered-with-verified-layouts)
- [3. Design](#3-design)
  - [3.1 The operand ABI: `tf32`, carried as I32](#31-the-operand-abi-tf32-carried-as-i32)
  - [3.2 Catalog rows](#32-catalog-rows)
  - [3.3 The lowering](#33-the-lowering)
  - [3.4 Known cost, and the phase-2 lever](#34-known-cost-and-the-phase-2-lever)
  - [3.5 Explicitly out of scope](#35-explicitly-out-of-scope)
- [4. Work breakdown](#4-work-breakdown)
- [5. Definition of Done](#5-definition-of-done)
- [6. Decisions and remaining open points](#6-decisions-and-remaining-open-points)

---

## 1. What CK actually does (verified against the source)

Two independent CK implementations agree on the recipe. Both were read directly. Line
numbers below are CK-repo-relative and were accurate when this plan was written; re-confirm
them against upstream before relying on any one of them.

### 1.1 The logical dtype: a tag in CK, a typed operand in rocke

`include/ck/utility/data_type.hpp:31`

```cpp
using tf32_t = _BitInt(19); // 1 sign bit, 8 exponent bits, 10 mantissa bits
```

`tf32_t` never appears in memory or in a register. It exists only to key atom
selection. In ck_tile the atom is registered as
`amdgcn_mma<tf32_t, tf32_t, fp32_t, M, N, K, ...>` — a *logical* `(xf32, xf32, fp32)`
triple — while the `exec()` signature takes and returns **fp32** vectors.

The constraint that matters is that emulation must not be registered as
`(fp32, fp32, fp32)`: ordinary fp32 selection must not move. CK satisfies it with a tag.

rocke needs the same key and one thing more. Its native gfx942 atoms already exist, and
they take a logical `tf32` operand — a real 32-bit IR type carried as I32 — so that raw
payload bits and explicit round-to-nearest-even preparation stay distinguishable. The
emulated atoms take **the same operand type**, which is what lets one kernel body reach
either implementation through a `(tf32, tf32, fp32)` catalog query (§3.1). CK's
fp32-typed `exec()` is the one place the two deliberately differ.

### 1.2 Native XF32 where the target has it

`include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp:1035,1055` register the same
logical `tf32_t` triple on gfx942 against native instructions:

| logical atom | physical instruction | A/B frag | C frag |
|---|---|---|---|
| `(xf32,xf32,fp32)` 16x16x8 | `__builtin_amdgcn_mfma_f32_16x16x8_xf32` | 2 | 4 |
| `(xf32,xf32,fp32)` 32x32x4 | `__builtin_amdgcn_mfma_f32_32x32x4_xf32` | 2 | 16 |

Also at `include/ck/utility/amd_xdlops.hpp:1698-1736`. **Preference by capability, not
by dtype**, falls out of registering native and emulated forms under one logical key.

### 1.3 The BF16x3 emulation on gfx950

`include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp:1640,1673` and the classic-CK twin at
`include/ck/utility/amd_xdlops.hpp:1739-1836`:

| logical atom | physical instruction | A/B frag | C frag |
|---|---|---|---|
| `(xf32,xf32,fp32)` 16x16x32 | `__builtin_amdgcn_mfma_f32_16x16x32_bf16` ×3 | 8 | 4 |
| `(xf32,xf32,fp32)` 32x32x16 | `__builtin_amdgcn_mfma_f32_32x32x16_bf16` ×3 | 8 | 16 |

Split (`include/ck/utility/amd_xdlops.hpp:15-29`):

```
big   = bf16(x)
small = bf16(x - fp32(big))
```

The split as written is not defined for every fp32 payload: `bf16(x)` overflows to
infinity for magnitudes of `0x7f7f8000` or more, after which `small` is the opposite
infinity and a finite product comes out NaN. The lowering therefore holds `big` at the
largest finite bf16 (§3.1, [step 5](xf32_bf16x3_steps/step-05-bf16x3-lowering.md#saturation-the-top-of-the-finite-range));
below that threshold it is this split bit for bit.

Then three MFMAs, accumulator threaded through all three, `small*small` dropped:

```
acc = mfma(a_small, b_big,   C  )
acc = mfma(a_big,   b_small, acc)
acc = mfma(a_big,   b_big,   acc)
```

> **Watch out:** the comment above both CK call sites says "Run 3 times: big\*big,
> small\*big, big\*small" — the *code* issues `small*big`, `big*small`, `big*big`.
> The code is what matters, and it is the order this plan targets. Do not port the comment.

Order is load-bearing: FP32 accumulation is not associative, so reordering changes
output bits and breaks the CK-parity acceptance criterion.

### 1.4 Rounding mode — the one real parity hazard

CK has **two** split paths with **different rounding**:

- Packed fast path (`include/ck_tile/core/numeric/bfloat16.hpp:551-570`) — uses
  `cvt_pk_bf16_f32` → `__builtin_convertvector(fp32x2_t, bf16x2_t)` → `v_cvt_pk_bf16_f32`,
  which is **round-to-nearest-even**. Gated on `__gfx94__ && CK_TILE_TF32_USE_PACKED_CVT`,
  and `CK_TILE_TF32_USE_PACKED_CVT` defaults to 1. **This is the path that ships on gfx950.**
- Scalar fallback (same file, `:572-577`) — uses `float_to_bf16`, whose default is
  `CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE` (`include/ck_tile/core/config.hpp:94`), i.e.
  **truncation**.

So "bit-exact with CK" is ambiguous until you pin the path. **Target the packed/RNE
path** — it is the shipping one on the targets we enable, and it is what rocke's
existing primitive already produces (see §2.3). Record this decision in the test, and
treat the truncate fallback as explicitly out of scope.

---

## 2. Current rocke state

### 2.1 Starting point: native TF32 on gfx942 only

The base carries a logical `tf32` dtype (`xf32` is an alias), carried as I32, and two
native gfx942 atoms, `mfma_f32_16x16x8_xf32` and `mfma_f32_32x32x4_xf32`. gfx950 has no
native XF32 instruction and no TF32 atom, so a TF32 query there is rejected. Nothing in
existing selection has to move. The base states that cross-target emulation is outside
what it adds; this plan is that part.

### 2.2 No result-dtype split is needed

An earlier draft treated independent C and D atom metadata as a prerequisite. It is not:
every shipped atom, native and emulated, accumulates and produces fp32, and the TF32
verifier requires the result of a TF32 atom to be its FP32 accumulator. The field that
was added for it had no user and was removed.

### 2.3 The split primitive already exists and already has the right rounding

`arith.trunc_f32_to_bf16` lowers to `fptrunc float ... to bfloat`
(`python/rocke/core/lower_llvm.py:2414`). LLVM `fptrunc` to `bfloat` is RNE, and the
AMDGPU backend selects `v_cvt_pk_bf16_f32` for the packed form on gfx950 — i.e. it
lands on **CK's shipping packed path** with no new op needed. Emitting the vector form
(`fptrunc <8 x float> to <8 x bfloat>`) should select 4 packed converts per operand.

### 2.4 Physical bf16 atoms are already registered with verified layouts

Queried from the live catalog:

| arch | op_id | shape | a/b frag | c frag | wave | layouts verified |
|---|---|---|---|---|---|---|
| gfx950 | `mfma_f32_16x16x32_bf16` | 16x16x32 | 8 / 8 | 4 | 64 | yes |
| gfx950 | `mfma_f32_32x32x16_bf16` | 32x32x16 | 8 / 8 | 16 | 64 | yes |
| gfx942 | `mfma_f32_16x16x16_bf16` | 16x16x16 | 4 / 4 | 4 | 64 | yes |
| gfx942 | `mfma_f32_32x32x8_bf16` | 32x32x8 | 4 / 4 | 16 | 64 | yes |
| gfx1250 | `wmma_gfx1250_f32_16x16x32_bf16` | 16x16x32 | 16 / 16 | 8 | 32 | yes |

The gfx950 frag lengths (a=8, c=4 and a=8, c=16) match CK's
`amdgcn_mma_base<..., 8,1,1,1,1, 4,1>` / `<..., 8,1,1,1,1, 16,4>` exactly. **gfx950 is
ready to enable on day one.** `MmaOp._require_layout` already raises
`NotImplementedError` for unregistered layouts, which gives the acceptance criterion
"unsupported targets reject the logical XF32 atom explicitly" almost for free.

---

## 3. Design

### 3.1 The operand ABI: `tf32`, carried as I32

The logical dtype is `tf32`; `xf32`, AMD's name for the same 19-bit format, is an accepted
alias spelling. It is defined in [`python/rocke/core/dtypes.py`](../../python/rocke/core/dtypes.py)
and its C mirror [`cpp/core/dtypes.cpp`](../../cpp/core/dtypes.cpp), and it is a real
32-bit IR type, `tf32` / `vec<tf32xN>`, carried as I32 words: `bitcast` wraps an fp32 or
I32 payload without rounding, `cvt_f32_to_tf32` rounds explicitly, and generic TF32
arithmetic is rejected.

An emulated atom has exactly the native atoms' ABI:

- A and B are `vec<tf32xN>`, N being the fragment length the atom borrows from its
  physical atom (8 for both gfx950 atoms). The accumulator and the result are
  `vec<f32xM>`.
- The verifier derives all of it from the catalog — the `emulation` descriptor plus the
  inherited fragment lengths — rather than from a per-atom table, so a new emulated row
  needs no verifier edit. An fp32 vector, or a TF32 vector of the wrong length, is
  rejected with the expected types named.
- **No payload bit is dropped.** A native XF32 instruction ignores the low 13 bits of
  each operand; the emulation reinterprets the whole 32-bit payload as fp32 and splits
  it. It therefore tracks the exact product of the payloads it is given. On raw fp32
  payloads that is more accurate than the native atom, which truncates them first; on
  payloads already prepared to TF32 precision the native atom is exact and the emulation
  is within `2^-16` (relative, per product) of it, the bound on the omitted
  `small*small` term, which a pair of operands halfway between two bf16 values reaches
  99% of (`A = B = 1 + 2^-8`: about `1.514e-5`). The two are not bit-reproducible against
  each other across targets. The device gate pins the first half with operands that
  carry bits below the native mantissa.
- **Every 32-bit payload splits.** The leading bf16 is capped at the largest finite
  value, so the top `2^-9` of the finite range — TF32's largest finite value among it —
  does not round to an infinity and turn a finite product into NaN; and an infinity or
  NaN in an operand makes every output element it feeds non-finite, and no other. The
  one place that is weaker than IEEE: an infinity against a partner of magnitude 512 or
  more may come out as NaN rather than `inf`. See
  [step 11](xf32_bf16x3_steps/step-11-tf32-operand-abi.md#the-edges-of-the-payload-domain).

### 3.2 Catalog rows

Add an `emulation` descriptor to the MMA catalog row schema
(`python/rocke/core/arch/data/arch_specs.json` + the C mirror in
`cpp/core/arch/data.cpp`):

```json
{
  "op_id": "mfma_f32_16x16x32_xf32",
  "family": "mma",
  "a": "tf32", "b": "tf32", "c": "fp32",
  "m": 16, "n": 16, "k": 32,
  "emulation": { "recipe": "bf16x3", "physical_op_id": "mfma_f32_16x16x32_bf16" }
}
```

Rules:

- Frag lengths and layout maps are **inherited verbatim** from `physical_op_id`. The
  logical A fragment occupies the same `(row, k)` coordinates; only the element type
  differs (TF32 in, bf16 to the instruction). Derive rather than duplicate, so the two
  can never drift.
- The recipe fixes the operand dtype (`bf16x3` emulates TF32), and the physical atom
  must be a **native row of the same arch**. Both are checked when the catalog loads: a
  row naming an atom its target cannot issue would claim support the compiler then
  rejects at codegen.
- A row with **no** `emulation` key is native. Selection prefers native over emulated at
  the same shape — one comparator in `MmaCatalog.enumerate`. The comparator keys on the
  full `(m, n, k)`, so it is sound only while no arch lists a native and an emulated atom
  for the same `(a, b, c, m, n)`; a test pins that, and the rule has to be re-keyed on
  `(m, n)` before an arch breaks it.
- Absence of a row = explicit rejection. No architecture-wide fallback.

Enablement: **gfx950 only** (`16x16x32`, `32x32x16`). gfx942 keeps its native atoms and is
offered no emulated ones. A gfx942 emulation could not reuse these two rows: the K-packed
bf16 atoms they expand into do not exist there, so it would expand into gfx942's own
(`16x16x16`, `32x32x8`), and the selection rule above would need re-keying first.

### 3.3 The lowering

One shared routine, mirrored byte-for-byte in both engines
(`python/rocke/core/lower_llvm.py` and `cpp/core/lower_llvm/mma.cpp`). Expand at the
`tile.mma` lowering site, *before* ISA dispatch, so the same expansion serves MFMA and
WMMA:

```
payload(X) -> F:                                          # X: vec<tf32xN>, an I32 carrier
  F        = bitcast <N x i32> X to <N x float>           # reinterpret, never round

split(F) -> (big, small):
  big_bf   = fptrunc <N x float> F    to <N x bfloat>     # RNE, packed
  big_f32  = fpext   <N x bfloat> big_bf to <N x float>
  diff     = fsub    <N x float> F, big_f32
  small_bf = fptrunc <N x float> diff to <N x bfloat>

emit_mma(phys, a_small, b_big, C  ) -> t0
emit_mma(phys, a_big,   b_small, t0) -> t1
emit_mma(phys, a_big,   b_big,   t1) -> D
```

Keep the split **inside** the atom for phase 1 — it matches CK structurally, and it
keeps the byte-identity gate tractable.

### 3.4 Known cost, and the phase-2 lever

Register pressure goes up on two axes: A/B arrive as 8×f32 (8 VGPRs) instead of
8×bf16 (4 VGPRs), and the split materializes `big`+`small` (4 more VGPRs per operand).
In a GEMM inner loop an A fragment is reused across several N-tiles, so splitting
inside the atom redundantly re-splits it.

**Phase 2 (optional, separate change):** expose an `xf32_split` helper so kernel
bodies can hoist the split out of the N-loop and feed pre-split fragments to the atom.
Do not attempt this in the same change as the atom — land correctness and byte-identity
first.

### 3.5 Explicitly out of scope

- **Frontend `allow_tf32` routing.** Accepting XF32 as a frontend dtype, or honouring a
  PyTorch-style `allow_tf32` flag, is provider/dispatch-layer work and needs a selector
  that maps a request onto a catalog query, which no production selector does yet. It is
  also an autotuner policy change, which this plan does not make.
- HIP-source lowering. The expansion exists only in the LLVM lowerer; the HIP lowerers
  reject an emulated atom by name rather than failing as though they had merely missed a
  hardware instruction.
- Implicit C/D conversion.
- The `small*small` term.

---

## 4. Work breakdown

Each completed step leaves a note under
[`xf32_bf16x3_steps/`](xf32_bf16x3_steps/) recording what it actually turned out to
be, the design calls, and how it was verified — including where the step diverged
from this plan. Counts and figures in the older notes were taken on the base as it stood
at the time; [step 10](xf32_bf16x3_steps/step-10-byte-identity.md) and
[step 11](xf32_bf16x3_steps/step-11-tf32-operand-abi.md) record the current gates.

| # | Step | Touches | Blocking | Notes |
|---|---|---|---|---|
| 1 | ~~Add independent `d_dtype` to atom metadata~~ | — | — | **dropped** — no atom needs D ≠ C (§2.2) |
| 2 | ~~Register `xf32` as a selection-tag dtype~~ | — | — | **superseded** by the base's logical `tf32` type (§3.1, [step 11](xf32_bf16x3_steps/step-11-tf32-operand-abi.md)) |
| 3 | Add `emulation` descriptor to catalog schema; derive frag/layout from `physical_op_id`; native-preferred comparator in `MmaCatalog.enumerate` | `core/arch/target.py`, catalog JSON + C mirror | after 1 | ✅ [step 3](xf32_bf16x3_steps/step-03-emulation-descriptor.md) |
| 4 | Add the two gfx950 emulated rows | catalog JSON + C mirror | after 3 | ✅ [step 4](xf32_bf16x3_steps/step-04-gfx950-xf32-rows.md) |
| 5 | BF16x3 expansion in the Python lowering | `core/lower_llvm.py` | after 4 | ✅ [steps 5+6](xf32_bf16x3_steps/step-05-bf16x3-lowering.md) |
| 6 | Same expansion, byte-identical, in the C++ engine | `cpp/core/lower_llvm/mma.cpp` | same change as 5 | ✅ [steps 5+6](xf32_bf16x3_steps/step-05-bf16x3-lowering.md) |
| 7 | FP32-selection-unchanged regression test | `tests/` | after 3 | ✅ [step 7](xf32_bf16x3_steps/step-07-selection-regression.md) |
| 8 | COMGR compile gate for gfx950 | `tests/` | after 6 | ✅ [step 8](xf32_bf16x3_steps/step-08-comgr-compile-gate.md) |
| 9 | Numeric regression: vs FP32 reference **and** vs CK packed-path reference | `tests/core/` + `examples/gfx950/xf32/` | after 8 | ✅ [step 9](xf32_bf16x3_steps/step-09-numeric-regression.md) — host half green; device harness wired into pytest (`test_xf32_bf16x3_device.py`); **gfx950 device run green** — bitwise against the hand-written BF16x3 at both atoms, rel err 1.19e-07 vs K*alpha*beta, and both injected faults rejected |
| 10 | Re-bless byte-identity goldens; run both LLVM flavors | `tools/check_byte_identity.py` | last | ✅ [step 10](xf32_bf16x3_steps/step-10-byte-identity.md) |
| 11 | Move the emulated atoms onto the base's TF32 operand ABI: `vec<tf32xN>` operands, `mfma_f32_<shape>_xf32` names, catalog-driven verifier, explicit HIP rejection | `core/arch/`, `core/tf32.py`, both lowerers, parity pair, tests | after 10 | ✅ [step 11](xf32_bf16x3_steps/step-11-tf32-operand-abi.md) |

---

## 5. Definition of Done

Per `KERNEL_AUTHORING.md` (new-feature path), plus the acceptance criteria this work
started from:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python tools/check_byte_identity.py
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py
```

- [ ] Python/C catalog parity — both engines enumerate identical atom sets including `emulation`.
- [ ] Byte identity GREEN for every family at both LLVM flavors.
- [ ] COMGR compiles for each enabled target.
- [ ] GPU numeric regression per enabled arch family, at a declared tolerance, against an independent reference. Byte-identity and successful compile are **not** correctness gates.
- [ ] Ordinary `(fp32, fp32, fp32)` selection provably unchanged (test 7).
- [ ] Unsupported targets reject `(tf32, tf32, fp32)` explicitly rather than silently falling back.
- [ ] Native XF32 preferred over emulated where both are registered.

---

## 6. Decisions and remaining open points

1. **Tolerance — decided.** BF16x3 minus `small*small` yields roughly 16 bits of effective
   product mantissa, between TF32's 10 and FP32's 24. The numeric gate states its
   tolerance post-accumulation, per output position, against `K*alpha*beta`
   ([step 9](xf32_bf16x3_steps/step-09-numeric-regression.md)); a per-product tolerance
   would be a different, tighter claim and is not made.
2. **Rounding — decided.** The parity target is CK's packed/RNE path (§1.4); the truncating
   scalar fallback is out of scope.
3. **Scope split — decided.** The `allow_tf32` frontend routing is not part of this work
   (§3.5).
4. **Target list — decided.** gfx950 only. gfx942 keeps its native atoms and is offered no
   emulated ones (§3.2); gfx1250 is WMMA/wave32 and would need its own layout
   verification.
5. **Open: emulation on gfx942.** It would be an accuracy option rather than a
   capability, since the native atoms exist there, and it needs the re-keyed selection
   rule (§3.2) and its own device validation.
