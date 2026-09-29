# XF32 via BF16x3 MMA emulation — solution plan

**Ticket:** AICK-2208 (`[rocKE] Add capability-based TF32 emulation using BF16x3 MMA`)
**Related:** AICK-1714 (native XF32 MFMA atoms) · AICK-2169 (merged in as duplicate)
**Reference implementation:** Composable Kernel, public repo `ROCm/composable_kernel`

This document is a design/implementation plan, not a benchmark record. It contains no
measured performance results.

## Contents

- [1. What CK actually does (verified against the source)](#1-what-ck-actually-does-verified-against-the-source)
  - [1.1 The logical dtype is a selection tag, not a storage type](#11-the-logical-dtype-is-a-selection-tag-not-a-storage-type)
  - [1.2 Native XF32 where the target has it](#12-native-xf32-where-the-target-has-it)
  - [1.3 The BF16x3 emulation on gfx950](#13-the-bf16x3-emulation-on-gfx950)
  - [1.4 Rounding mode — the one real parity hazard](#14-rounding-mode--the-one-real-parity-hazard)
- [2. Current rocke state](#2-current-rocke-state)
  - [2.1 Greenfield](#21-greenfield)
  - [2.2 The stated prerequisite is real](#22-the-stated-prerequisite-is-real)
  - [2.3 The split primitive already exists and already has the right rounding](#23-the-split-primitive-already-exists-and-already-has-the-right-rounding)
  - [2.4 Physical bf16 atoms are already registered with verified layouts](#24-physical-bf16-atoms-are-already-registered-with-verified-layouts)
- [3. Design](#3-design)
  - [3.1 `xf32` as a logical dtype](#31-xf32-as-a-logical-dtype)
  - [3.2 Catalog rows](#32-catalog-rows)
  - [3.3 The lowering](#33-the-lowering)
  - [3.4 Known cost, and the phase-2 lever](#34-known-cost-and-the-phase-2-lever)
  - [3.5 Explicitly out of scope](#35-explicitly-out-of-scope)
- [4. Work breakdown](#4-work-breakdown)
- [5. Definition of Done](#5-definition-of-done)
- [6. Open questions to close with the reporter](#6-open-questions-to-close-with-the-reporter)

---

## 1. What CK actually does (verified against the source)

Two independent CK implementations agree on the recipe. Both were read directly. Line
numbers below are CK-repo-relative and were accurate when this plan was written; re-confirm
them against upstream before relying on any one of them.

### 1.1 The logical dtype is a selection tag, not a storage type

`include/ck/utility/data_type.hpp:31`

```cpp
using tf32_t = _BitInt(19); // 1 sign bit, 8 exponent bits, 10 mantissa bits
```

`tf32_t` never appears in memory or in a register. It exists only to key atom
selection. In ck_tile the atom is registered as
`amdgcn_mma<tf32_t, tf32_t, fp32_t, M, N, K, ...>` — a *logical* `(xf32, xf32, fp32)`
triple — while the `exec()` signature takes and returns **fp32** vectors.

This is exactly the constraint AICK-2208 states twice: do not register emulation as
`(fp32, fp32, fp32)`. CK arrived at the same design independently, which is good
validation of the ticket's framing.

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

Then three MFMAs, accumulator threaded through all three, `small*small` dropped:

```
acc = mfma(a_small, b_big,   C  )
acc = mfma(a_big,   b_small, acc)
acc = mfma(a_big,   b_big,   acc)
```

> **Watch out:** the comment above both CK call sites says "Run 3 times: big\*big,
> small\*big, big\*small" — the *code* issues `small*big`, `big*small`, `big*big`.
> The code is what matters and it matches the ticket body. Do not port the comment.

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

### 2.1 Greenfield

`xf32`/`tf32` appear nowhere in `python/`, and only in one unrelated comment in
`cpp/core/lower_llvm/mma.cpp:664`. Nothing to refactor, nothing to preserve.

### 2.2 The stated prerequisite is real

`MmaOp` (`python/rocke/core/arch/target.py:148`) carries `a_dtype`, `b_dtype`,
`c_dtype` and **no `d_dtype`**. `_op_id_c_dtype()` (`:1130`) is built on the assumption
that C is the accumulator and D is implied by it. The ticket's "independent C and D
atom metadata" dependency is genuine and blocking.

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

### 3.1 `xf32` as a logical dtype

Add `xf32` to the dtype normalization table in
[`python/rocke/core/dtypes.py`](../../python/rocke/core/dtypes.py) and its C mirror
[`cpp/core/dtypes.cpp`](../../cpp/core/dtypes.cpp). Note this is the *logical* dtype
registry, **not** the IR scalar type system in `cpp/core/ir/core_types.cpp` — `xf32`
must stay out of the latter, exactly as `fp6e2m3` does. Accept `tf32` as an alias
spelling that canonicalises to `xf32`. Constraints:

- **Selection tag only.** No memref element type, no load/store, no conversion op.
  This falls out of the existing design rather than needing new code: `io_ir_type`
  already raises for anything but f16/bf16, and `rocke_scalar_by_name` already
  returns NULL for unknown names. Pin both with tests so it stays true.
- Physical A/B registers stay `f32`. The atom's `a_frag_len` describes **f32** slots.

### 3.2 Catalog rows

Add an `emulation` descriptor to the MMA catalog row schema
(`python/rocke/core/arch/data/arch_specs.json` + the C mirror in
`cpp/core/arch/data.cpp`):

```json
{
  "op_id": "xf32_bf16x3_f32_16x16x32",
  "family": "mma",
  "a": "xf32", "b": "xf32", "c": "fp32", "d": "fp32",
  "m": 16, "n": 16, "k": 32,
  "emulation": { "recipe": "bf16x3", "physical_op_id": "mfma_f32_16x16x32_bf16" }
}
```

Rules:

- Frag lengths and layout maps are **inherited verbatim** from `physical_op_id`. The
  logical A fragment occupies the same `(row, k)` coordinates; only the element type
  differs (f32 in, bf16 to the instruction). Derive rather than duplicate, so the two
  can never drift.
- A row with **no** `emulation` key is native. Selection prefers native over emulated
  for the same logical triple and shape — one comparator in `MmaCatalog.enumerate`.
- Absence of a row = explicit rejection. No architecture-wide fallback.

Initial enablement: **gfx950 only** (`16x16x32`, `32x32x16`). gfx942 native XF32 rows
are AICK-1714's; this ticket only supplies the preference rule that will select them.

### 3.3 The lowering

One shared routine, mirrored byte-for-byte in both engines
(`python/rocke/core/lower_llvm.py` and `cpp/core/lower_llvm/mma.cpp`). Expand at the
`tile.mma` lowering site, *before* ISA dispatch, so the same expansion serves MFMA and
WMMA:

```
split(X) -> (big, small):
  big_bf   = fptrunc <N x float> X    to <N x bfloat>     # RNE, packed
  big_f32  = fpext   <N x bfloat> big_bf to <N x float>
  diff     = fsub    <N x float> X, big_f32
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

- **Frontend `allow_tf32` routing.** The merged AICK-2169 scope asks the frontend to
  accept XF32 as a dtype or honour a PyTorch-style `allow_tf32` flag. That is
  provider/dispatch-layer work and it contradicts this ticket's own "no autotuner
  policy change" boundary. **Recommend splitting it into a follow-up ticket** rather
  than letting it gate this one.
- XF32 load/store dtypes.
- Implicit C/D conversion.
- The `small*small` term.

---

## 4. Work breakdown

Each completed step leaves a note under
[`xf32_bf16x3_steps/`](xf32_bf16x3_steps/) recording what it actually turned out to
be, the design calls, and how it was verified — including where the step diverged
from this plan.

| # | Step | Touches | Blocking | Notes |
|---|---|---|---|---|
| 1 | Add independent `d_dtype` to atom metadata (`MmaOp`, catalog JSON, C mirror, `_op_id_c_dtype` twin for D) | `core/arch/target.py`, `arch_specs.json`, `cpp/core/arch/data.cpp`, `cpp/.../query.cpp` | **yes — prerequisite** | ✅ [step 1](xf32_bf16x3_steps/step-01-mma-result-dtype.md) |
| 2 | Register `xf32` logical dtype in both dtype registries; test that it is rejected for memory ops | `core/dtypes.py`, `cpp/core/dtypes.cpp` (**not** `cpp/core/ir/core_types.cpp` — see the step note) | | ✅ [step 2](xf32_bf16x3_steps/step-02-xf32-logical-dtype.md) |
| 3 | Add `emulation` descriptor to catalog schema; derive frag/layout from `physical_op_id`; native-preferred comparator in `MmaCatalog.enumerate` | `core/arch/target.py`, catalog JSON + C mirror | after 1 | ✅ [step 3](xf32_bf16x3_steps/step-03-emulation-descriptor.md) |
| 4 | Add the two gfx950 emulated rows | catalog JSON + C mirror | after 3 | ✅ [step 4](xf32_bf16x3_steps/step-04-gfx950-xf32-rows.md) |
| 5 | BF16x3 expansion in the Python lowering | `core/lower_llvm.py` | after 4 | ✅ [steps 5+6](xf32_bf16x3_steps/step-05-bf16x3-lowering.md) |
| 6 | Same expansion, byte-identical, in the C++ engine | `cpp/core/lower_llvm/mma.cpp` | same change as 5 | ✅ [steps 5+6](xf32_bf16x3_steps/step-05-bf16x3-lowering.md) |
| 7 | FP32-selection-unchanged regression test | `tests/` | after 3 | ✅ [step 7](xf32_bf16x3_steps/step-07-selection-regression.md) |
| 8 | COMGR compile gate for gfx950 | `tests/` | after 6 |
| 9 | GPU numeric regression: vs FP32 reference **and** vs CK packed-path reference | `library/builders/**` or `examples/common/*parity*` | after 8 |
| 10 | Re-bless byte-identity goldens; run both LLVM flavors | `tools/check_byte_identity.py` | last |

---

## 5. Definition of Done

Per `KERNEL_AUTHORING.md` (new-feature path), plus the ticket's own criteria:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python tools/check_byte_identity.py
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py
```

- [ ] Python/C catalog parity — both engines enumerate identical atom sets including `d_dtype` and `emulation`.
- [ ] Byte identity GREEN for every family at both LLVM flavors.
- [ ] COMGR compiles for each enabled target.
- [ ] GPU numeric regression per enabled arch family, at a declared tolerance, against an independent reference. Byte-identity and successful compile are **not** correctness gates.
- [ ] Ordinary `(fp32, fp32, fp32)` selection provably unchanged (test 7).
- [ ] Unsupported targets reject `(xf32, xf32, fp32)` explicitly rather than silently falling back.
- [ ] Native XF32 preferred over emulated where both are registered.

---

## 6. Open questions to close with the reporter

1. **Tolerance.** The AICK-2169 comment says "within tolerance (7-bit BF16 mantissa)".
   BF16x3 minus `small*small` yields roughly 16 bits of effective product mantissa —
   between TF32's 10 and FP32's 24. Clarify whether the tolerance is per-product or
   post-accumulation before writing the numeric gate.
2. **Rounding.** Confirm §1.4: parity target is CK's packed/RNE path, and the truncate
   fallback is out of scope.
3. **Scope split.** Confirm the `allow_tf32` frontend routing moves to its own ticket
   (§3.5).
4. **Target list.** gfx950 first. Which, if any, of gfx942 / gfx1250 must be enabled in
   this change versus a follow-up? gfx942 has native XF32 (AICK-1714), so emulation
   there is likely pointless; gfx1250 is WMMA/wave32 and needs its own layout
   verification.
